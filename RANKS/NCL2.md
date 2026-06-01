# NCL2: Multi-GPU Coordinated Capture Issue Analysis

## 1. Current Status

| Metric | Status |
|--------|--------|
| Single-GPU training | 99.46% (consistent) |
| Multi-GPU hang | FIXED |
| Multi-GPU accuracy | DEGRADED to ~99.0% (expected 99.5%) |
| Mean AllReduce unit test | PASS on 8 GPUs |

User observation: "Feels like no communication, only 1/8 samples trained"

## 2. Root Cause Analysis

### 2.1 Issue A: has_nccl_ops() checks ENTIRE ComputationGraph

**Location**: `src/graph/captured_graph.cpp` Phase B2.5

Current code:
```cpp
if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
    is_nccl_key[k] = true;
}
```

`has_nccl_ops()` checks ALL nodes across ALL gid buckets, not just the specific subgraph being captured.

**Consequence**:
- `train_cg_` contains `FIRST_COMM`/`DEEP_COMM` -> `has_nccl_ops()` returns true for ALL train subgraphs
- ALL `train_cg_` subgraphs (`DEEP_FWD_BWD`, `FIRST_LAYER_FWD`, `ZERO_GRAD`, `OPTIMIZER`, etc.) are marked as NCCL graphs
- Similarly for `infer_cg_` due to `VAL_RESULT_COMM`

Out of 28 unique graphs, only ~3-4 truly contain NCCL nodes, but 10+ are marked for coordinated capture.

**Impact**: Excessive `ncclGroupStart`/`ncclGroupEnd` calls may pollute NCCL communicator state, causing actual NCCL allreduce operations to silently misbehave at runtime.

### 2.2 Issue B: cleanup_all_events() called BEFORE cudaStreamEndCapture

**Location**: `capture_nccl_graph_coordinated()` replay loop

Current: `state.cleanup_all_events()` is called inside the per-rank replay loop, BEFORE `ncclGroupEnd()` and `cudaStreamEndCapture()`.

Original `capture_cuda`: cleanup happens AFTER `cudaStreamEndCapture` + `cudaGraphInstantiate`.

If `cudaEventRecord` nodes were captured, destroying the event before `EndCapture` may produce incorrect graph topology.

### 2.3 Issue C: Missing cudaDeviceSynchronize()

Original `capture_cuda` calls `cudaDeviceSynchronize()` before `cudaStreamBeginCapture`. The coordinated capture path lacks this sync.

### 2.4 Most Likely Cause of Accuracy Drop

**Primary hypothesis**: NCCL communicator state pollution (Issue A)
- 10+ consecutive `ncclGroupStart`/`ncclGroupEnd` calls on same communicator
- Only 3-4 graphs actually need them
- Excessive group calls may leave communicator in abnormal state
- At runtime, `ncclAllReduce` in `FIRST_COMM`/`DEEP_COMM` appears to run (no deadlock) but produces wrong results or is silently skipped

**Secondary hypothesis**: Scale kernel not captured properly
- If `launch_tr_scale_fp32_kernel` is not in the captured graph, gradients are summed but not divided by world_size
- 8x gradient amplification = 8x effective LR -> training instability
- Consistent with 99.0% vs 99.5% drop

## 3. Proposed Fixes

### Fix 1: Check specific gid nodes only (fixes Issue A)

Replace whole-graph check with per-gid check:
```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    const auto& key = key_by_idx[k];
    if (!key.cg) continue;
    bool has_nccl = false;
    for (const auto& node : key.cg->nodes(key.gid)) {
        if (node.kind == GraphNode::Kind::RANGE &&
            (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE)) {
            has_nccl = true;
            break;
        }
    }
    is_nccl_key[k] = has_nccl;
}
```

### Fix 2: Move cleanup after Instantiate (fixes Issue B)

Save states during replay, clean up after all `cudaGraphInstantiate` calls complete.

### Fix 3: Add cudaDeviceSynchronize() (fixes Issue C)

Add `cudaDeviceSynchronize()` before each rank begins capture in coordinated capture.

## 4. Validation Plan

1. Implement Fix 1+2+3
2. Windows compile check
3. A100x8 test: `./mnist_best --gpu --activation hardswish`
4. Verify `[B3-NCCL]` log count drops to ~3-4
5. Check accuracy recovery to 99.5%
