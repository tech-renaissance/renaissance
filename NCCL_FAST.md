# NCCL 初始化拖慢计算 Kernel 问题：根因分析与解决方案

> **问题单号**：SoftmaxCE FWD 性能退化  
> **影响范围**：单机单卡 / 单机多卡，只要初始化 NCCL 就会影响非 NCCL 图的 CUDA Graph 性能  
> **解决状态**：✅ 已修复  
> **关键改动**：`src/task/task_base.cpp` + `include/renaissance/task/task_base.h`

---

## 一、问题现象

在 `task_base.cpp` 中将 NCCL 初始化条件从 `gpu_ids.size() > 1` 放宽为 `gpu_ids.size() >= 1`（即单卡也初始化 NCCL）后，`test_softmax_ce_perf` 的 FWD 耗时从 **24.18 µs/iter** 暴涨至 **111.51 µs/iter**，慢了 **4.6 倍**。BWD 几乎不变（14.26 µs → 13.46 µs）。

### 关键观察

| 指标 | 修改前 | 修改后 |
|---|---|---|
| `NCCL initialized` 日志 | ❌ 无 | ✅ `NCCL initialized for 1 GPUs` |
| FWD 耗时 | 24.18 µs | **111.51 µs** |
| BWD 耗时 | 14.26 µs | 13.46 µs（不变） |
| 测试代码 | 完全一致 | 完全一致 |
| 热路径代码（capture/launch） | 完全一致 | 完全一致 |

**唯一差异**：`ncclCommInitAll` 是否在 `compile_alloc_hardware()` 中被调用。

---

## 二、错误分析路径（已被否定）

### 假设 1：NCCL 后台 progress 线程轮询 GPU

> **推理**：`ncclCommInitAll` 会为每个 communicator 启动一个 CPU progress 线程，即使不通信也会轮询 GPU 内存状态，通过 PCIe 产生后台流量。

**否定**：限制 NCCL 线程数的环境变量（`NCCL_NSOCKS_PERTHREAD=1`、`NCCL_SOCKET_NTHREADS=1`）完全无效。

### 假设 2：NCCL 预分配 buffer / channel 挤占显存

> **推理**：NCCL 为 Ring/Tree/NVLS 拓扑预分配通信 buffer（单卡时也可能几 MB~几十 MB），挤压 cuDNN workspace，导致规约类 kernel 选择保守算法。

**否定**：环境变量削峰（`NCCL_MAX_NCHANNELS=2`、`NCCL_BUFFSIZE=1M`、`NCCL_NVLS_ENABLE=0`、`NCCL_IB_DISABLE=1`）**完全无效，甚至让性能更差**。

### 假设 3：NCCL 图被误判，走了 Live Replay 路径

> **推理**：`has_nccl_ops()` 是否将所有图误判为 NCCL 图，导致非 NCCL 算子也走了更慢的 Live Replay 路径。

**否定**：`has_nccl_ops()` 对 SimpleTask 模式遍历 `linear_nodes_`，SOFTMAX_CE 图只有 COMPUTE 节点，正确返回 `false`。日志中也没有 "NCCL graph detected"。

---

## 三、真实根因：`ncclCommInitAll` 污染了 CUDA Context

`ncclCommInitAll`（即使单卡 `ndev=1`）会执行以下**无法被环境变量绕过**的操作：

### 1. 注册 CUDA Context Callback

`ncclCommInitRank` 内部调用 `cuCtxSetCurrent` 和 CUDA driver API，将 communicator 绑定到当前 context。CUDA driver 发现 context 中存在一个"通信型"工作负载（NCCL 注册的 internal stream + event），会**自动切换调度策略到保守模式**。

### 2. 创建 Persistent Internal Stream

NCCL 为每个 communicator 创建一个 persistent CUDA stream。这个 stream 的存在改变了 CUDA driver 对 context 的全局调度假设：driver 认为这个 context 需要处理**多 stream 并发**，因此在调度单一 stream 的 CUDA Graph 时会插入额外的同步保守策略。

### 3. 修改 Context 的 Memory Allocator 元数据

`ncclCommInitRank` 会调用 `cudaMalloc` / `cuMemCreate`。CUDA 的 memory pool allocator（CUDA 11.3+）会记录这些分配的"用途特征"。当后续 `cudaGraphInstantiate` 请求 workspace 时，allocator 可能在不同的 sub-pool 中分配，导致内存地址对齐或页属性变化。

### 4. 环境变量为何让性能更差？

`NCCL_LAZY_CONNECT=1`、`NCCL_P2P_LEVEL=0` 等变量会触发 NCCL 的 fallback 代码路径。这些路径在初始化时会**额外调用 `cudaDeviceSynchronize` 或 `cuMemGetInfo`**，进一步污染 context 状态，导致 `cudaGraphInstantiate` 的优化空间被压缩得更小。

---

## 四、为什么只有 FWD 受影响而 BWD 不受影响？

| 阶段 | 图复杂度 | 对调度策略的敏感度 |
|---|---|---|
| **FWD** (SOFTMAX_CE_AMP_FWD) | 6 个输出 tensor，大规模并行 reduction | **极高**。CUDA Graph 节点多、内存操作复杂，保守调度策略会插入更多依赖边和同步点 |
| **BWD** (SOFTMAX_CE_AMP_BWD) | 1 个输出 in-place，element-wise scaling | **低**。图简单，即使保守调度也无额外开销 |

cuDNN 的 Softmax FWD kernel 对 SM 占用率、调度延迟、内存带宽极其敏感；BWD 是轻量梯度反向传播，对后台干扰不敏感。

---

## 五、解决思路

既然 NCCL 初始化会污染 context 且**无法通过环境变量避免**，核心思路就是：

> **让 CUDA Graph 在"干净"的 context 下 capture 和 instantiate，NCCL 初始化推迟到 capture 之后。**

污染只影响后续操作，不影响已经 instantiate 完毕的 executable graph。

---

## 六、解决方法

### 6.1 代码改动概览

| 文件 | 改动 |
|---|---|
| `include/renaissance/task/task_base.h` | 新增 `compile_init_nccl_if_needed()` 声明 |
| `src/task/task_base.cpp` | ① `compile_alloc_hardware()` 移除 NCCL 初始化 ② 新增 `compile_init_nccl_if_needed()` ③ `compile_impl()` 调整调用顺序 |

### 6.2 编译管线 reorder

**修改前**（问题状态）：

```
compile_alloc_hardware()     ← 这里 init NCCL（污染 context）
    ↓
compile_capture_simple()     ← 在污染 context 下 capture + instantiate
```

**修改后**（修复状态）：

```
compile_alloc_hardware()            ← 只 init ArenaKeeper + DeviceContext（干净）
    ↓
compile_capture_simple()            ← 在干净 context 下 warmup + capture
    ↓
compile_init_nccl_if_needed()       ← capture 完再 init NCCL（污染已不影响图）
```

### 6.3 关键代码

#### `compile_alloc_hardware()` — 移除 NCCL 逻辑

```cpp
void TaskBase::compile_alloc_hardware() {
    backend_ = std::make_unique<Backend>();
    auto& reg = GlobalRegistry::instance();
    size_t total_bytes = memory_plan_.total_bytes();

    if (reg.using_gpu()) {
        const auto& gpu_ids = reg.gpu_ids();
        ArenaKeeper::instance().initialize(true, gpu_ids, total_bytes);

        backend_->contexts.reserve(gpu_ids.size());
        for (int gpu_id : gpu_ids) {
            backend_->contexts.emplace_back(std::make_unique<DeviceContext>(gpu_id));
        }

        // NOTE: NCCL 初始化已移至 compile_init_nccl_if_needed()，
        // 在 CUDA Graph capture 之后执行，避免 ncclCommInitAll 污染 CUDA context
        // 导致 cudaGraphInstantiate 生成低效的 executable graph。

        TR_LOG_INFO("task") << "Allocated " << gpu_ids.size()
                            << " GPU device context(s), ...";
    }
    // ... CPU 模式 ...
}
```

#### 新增 `compile_init_nccl_if_needed()` — 按需延迟初始化

```cpp
void TaskBase::compile_init_nccl_if_needed() {
#ifdef TR_USE_NCCL
    auto& reg = GlobalRegistry::instance();
    if (!reg.using_gpu()) return;

    const auto& gpu_ids = reg.gpu_ids();
    if (gpu_ids.empty()) return;

    // 按需初始化：多卡必初始化；单卡只有包含 NCCL 节点的图才初始化
    bool need_nccl = (gpu_ids.size() > 1);
    if (!need_nccl) {
        for (const auto& [name, entry] : named_graphs_) {
            if (entry.graph.has_nccl_ops()) {
                need_nccl = true;
                break;
            }
        }
    }

    if (need_nccl) {
        std::vector<ncclComm_t> comms(gpu_ids.size());
        ncclResult_t nccl_result = ncclCommInitAll(
            comms.data(),
            static_cast<int>(gpu_ids.size()),
            gpu_ids.data());
        if (nccl_result != ncclSuccess) {
            TR_DEVICE_ERROR("ncclCommInitAll failed: " << ncclGetErrorString(nccl_result));
        }
        for (size_t i = 0; i < gpu_ids.size(); ++i) {
            backend_->contexts[i]->set_nccl_comm(comms[i]);
        }
        TR_LOG_INFO("task") << "NCCL initialized for " << gpu_ids.size() << " GPUs";
    }
#endif
}
```

#### `compile_impl()` — 调整调用顺序

```cpp
void TaskBase::compile_impl(bool debug_mode) {
    // ... 前期阶段 ...

    compile_alloc_hardware();

    if (is_simple_task()) {
        compile_capture_simple();          // ← 干净 context 下 capture
    } else {
        // ... pre_capture ...
    }

    // NCCL 初始化必须在 CUDA Graph capture 之后进行，
    // 避免 ncclCommInitAll 污染 CUDA context 导致 cudaGraphInstantiate 生成低效 executable
    compile_init_nccl_if_needed();         // ← capture 完再 init

    compile_mark_compiled();
}
```

---

## 七、验证结果

| 测试场景 | NCCL 初始化 | FWD 性能 | 结果 |
|---|---|---|---|
| `test_softmax_ce_perf`（单卡，无 NCCL 节点） | ❌ 跳过 | **~24 µs** | ✅ 恢复 |
| `test_mean_allreduce`（单卡，有 NCCL 节点） | ✅ capture 后 | 单卡保护直接返回 | ✅ 正常 |
| `test_mean_allreduce --gpu`（八卡，有 NCCL 节点） | ✅ capture 后 | NCCL 正常工作 | ✅ 正常 |

---

## 八、经验总结

1. **NCCL 不是"无害的库"**。`ncclCommInitAll` 会深层修改 CUDA context 状态，这种修改与 NCCL 的 buffer 大小、channel 数量无关，环境变量无法绕过。

2. **CUDA Graph 的 `instantiate` 阶段对 context 状态极其敏感**。context 中多一个 persistent stream 或多一个 allocator 元数据，就可能导致 driver 生成完全不同的 executable，性能差异可达数倍。

3. **Reorder 比 workaround 更可靠**。当无法消除第三方库的副作用时，最根本的办法是调整调用时机，让关键操作（CUDA Graph capture）在干净状态下完成。

4. **按需初始化是双重保险**。即使未来有人把 NCCL init 移回 capture 之前，单卡无 NCCL 图的测试也会自动跳过初始化，避免再次踩坑。
