# 【问题】

```shell
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# CUDA_VISIBLE_DEVICES=0 ./test_relu_fwd_bwd --amp --no-gen
[2026-05-17 14:47:29.822] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 14:47:29.822] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0]
Device: GPU [0]
[2026-05-17 14:47:29.822] [INFO ] [TR] GlobalRegistry: fixed_using_amp set to true
[2026-05-17 14:47:29.822] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B0477BCAE88208
Skipping reference generation (--no-gen).
[2026-05-17 14:47:30.409] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:47:30.409] [INFO ] [task] Allocated 1 GPU device context(s), 576.001 MB each
=== SimpleTask MemoryPlan ===
Total bytes: 603981056
MemoryPlan Layout (total_bytes=603981056)
===============================================
Region [F_FEATURE_FP16]  ID=54  offset=0  bytes=536871936
-----------------------------------------------
  Tensor[id=0]: offset=0,  bytes=134217984,  dtype=FP16,  shape=[8,1024,1024,8]
  Tensor[id=1]: offset=134217984,  bytes=134217984,  dtype=FP16,  shape=[8,1024,1024,8]
  Tensor[id=3]: offset=268435968,  bytes=134217984,  dtype=FP16,  shape=[8,1024,1024,8]
  Tensor[id=4]: offset=402653952,  bytes=134217984,  dtype=FP16,  shape=[8,1024,1024,8]
-----------------------------------------------
===============================================
Region [S_MASK]  ID=60  offset=536871936  bytes=67109120
-----------------------------------------------
  Tensor[id=2]: offset=536871936,  bytes=67109120,  dtype=INT8,  shape=[8,1024,1024,8]
-----------------------------------------------
===============================================

=== SimpleTask ComputationGraphs ===
Registered graphs: 1

[Graph: "fwd_bwd"]
  StreamKind: 1
  Total nodes: 2
=== ComputationGraph Debug Dump ===
[Linear Nodes] 2 nodes:
  RELU_AMP_FWD inputs=[0] outputs=[1, 2] params: 
  RELU_AMP_BWD inputs=[3, 2] outputs=[4] params: 

  Rank 0 FWD MSE = 0.000000E+00
  Rank 0 BWD MSE = 0.000000E+00

===== ReLU FWD+BWD AMP  [FP16] (1 rank(s)): PASS =====
  Avg:   498.828350 us/iter
  MaxMSE: 0.000000E+00
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# CUDA_VISIBLE_DEVICES=0,1 ./test_relu_fwd_bwd --amp --no-gen
[2026-05-17 14:47:40.756] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 14:47:40.756] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1]
Device: GPU [0, 1]
[2026-05-17 14:47:40.756] [INFO ] [TR] GlobalRegistry: fixed_using_amp set to true
[2026-05-17 14:47:40.756] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B0477E56A72E64
Skipping reference generation (--no-gen).
[2026-05-17 14:47:41.496] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:47:41.503] [INFO ] [backend] DeviceContext 1: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:47:41.904] [INFO ] [task] NCCL initialized for 2 GPUs
[2026-05-17 14:47:41.904] [INFO ] [task] Allocated 2 GPU device context(s), 576.001 MB each

===============================================================================
            RENAISSANCE FRAMEWORK - FATAL ERROR
===============================================================================

Exception Type : DeviceError
Root Message   : cuDNN FE [execute AMP FWD]: s_amp_fwd_cache->graph->execute(handle, vp, nullptr) failed with code: 11

Call Stack (bottom to top):
  -> cuDNN FE [execute AMP FWD]: s_amp_fwd_cache->graph->execute(handle, vp, nullptr) failed with code: 11 (at relu_op.cpp :: launch_relu_amp_fwd_cuda())

Full Description:
  [DeviceError] cuDNN FE [execute AMP FWD]: s_amp_fwd_cache->graph->execute(handle, vp, nullptr) failed with code: 11 (at relu_op.cpp :: launch_relu_amp_fwd_cuda())

===============================================================================
Program will now abort.
===============================================================================

Aborted (core dumped)
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu_fwd_bwd --amp --no-gen
[2026-05-17 14:48:13.500] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 14:48:13.500] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]
[2026-05-17 14:48:13.500] [INFO ] [TR] GlobalRegistry: fixed_using_amp set to true
[2026-05-17 14:48:13.500] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B04785F65496F1
Skipping reference generation (--no-gen).
[2026-05-17 14:48:15.189] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.198] [INFO ] [backend] DeviceContext 1: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.206] [INFO ] [backend] DeviceContext 2: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.215] [INFO ] [backend] DeviceContext 3: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.223] [INFO ] [backend] DeviceContext 4: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.232] [INFO ] [backend] DeviceContext 5: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.240] [INFO ] [backend] DeviceContext 6: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:15.249] [INFO ] [backend] DeviceContext 7: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 14:48:16.452] [INFO ] [task] NCCL initialized for 8 GPUs
[2026-05-17 14:48:16.452] [INFO ] [task] Allocated 8 GPU device context(s), 576.001 MB each

===============================================================================
            RENAISSANCE FRAMEWORK - FATAL ERROR
===============================================================================

Exception Type : DeviceError
Root Message   : cuDNN FE [execute AMP FWD]: s_amp_fwd_cache->graph->execute(handle, vp, nullptr) failed with code: 11

Call Stack (bottom to top):
  -> cuDNN FE [execute AMP FWD]: s_amp_fwd_cache->graph->execute(handle, vp, nullptr) failed with code: 11 (at relu_op.cpp :: launch_relu_amp_fwd_cuda())

Full Description:
  [DeviceError] cuDNN FE [execute AMP FWD]: s_amp_fwd_cache->graph->execute(handle, vp, nullptr) failed with code: 11 (at relu_op.cpp :: launch_relu_amp_fwd_cuda())

===============================================================================
Program will now abort.
===============================================================================

Aborted (core dumped)
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# 

```



# 【小伙伴K】

• 结论：问题出在 捕获（capture）阶段
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  证据链
   线索                                                   分析
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   单卡正常、多卡失败                                     不是算子逻辑错误，而是多设备共享状态冲突
   错误码 11 来自 graph->execute()                        build_amp_fwd_graph 本身没抛异常，是 execute 时炸的
   errors.txt 中没有 === SimpleTask MemoryPlan === 打印   task.compile() 内部就失败了，main() 里 compile() 之后的 print
                                                          _memory_plan() 根本没执行到
   日志停在 Allocated 2 GPU device context(s)             compile_alloc_hardware() 之后、compile_mark_compiled() 之前，
                                                          即 compile_capture_simple() 内部
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  compile_capture_simple() 的精确执行流程
  Phase A: warmup（第270-288行）
      └── 只在 master GPU（rank 0）上串行执行
      └── 调用 launch_relu_amp_fwd_cuda → 创建 s_amp_fwd_cache（绑定 handle_0）
      └── execute 成功，cudaDeviceSynchronize 返回

  Phase B: capture（第305-340行）
      └── 串行循环 for (int rank = 0; rank < num_gpus_; ++rank)
      └── 每个 rank 调用 CapturedGraph::capture_cuda()
          ├── rank 0: s_amp_fwd_cache 匹配 handle_0 → 直接 execute → ✅ 成功
          └── rank 1: s_amp_fwd_cache 不匹配 handle_1 → 触发 rebuild
              ├── build_amp_fwd_graph(handle_1, ...) → 表面成功（没抛异常）
              └── s_amp_fwd_cache->graph->execute(handle_1, vp, nullptr) → ❌ 失败，code 11
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  根因：全局静态 cache 在 capture mode 下触发 rebuild
  relu_op.cpp 第 72-73 行：
  static std::unique_ptr<AmpReluFwdCache> s_amp_fwd_cache;  // ← 全局唯一
  static std::unique_ptr<AmpReluBwdCache> s_amp_bwd_cache;
  致命缺陷：每个 GPU 有自己的 cudnnHandle_t，但算子级 cache 只有一份。
  • warmup 阶段：在 普通 stream 模式 下为 rank 0 创建 cache，handle = handle_0
  • capture 阶段 rank 0：cache 命中，在 capture mode 下 replay execute → 成功（因为 graph 是在普通模式下预热的，内部 wor
    pace 已就绪）
  • capture 阶段 rank 1：发现 handle 不匹配，在 capture mode 下当场 build_amp_fwd_graph + execute
  根据 Warmup.md 的核心铁律：
  ▌ Phase 1（warmup）负责在普通模式下完成所有 build_plans + execute，填满 cuDNN 进程级 cache。Phase 2（capture）只负 …
  ▌ replay，绝不应再触发 build_plans。
  当前代码违反了这条铁律——rank 1 在 capture mode 内触发 rebuild，而 cuDNN FE 的某些 lazy 初始化（workspace 分配、引擎状
  态绑定）在 capture mode 下行为异常，导致 execute 返回 code 11。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  不是预热、实例化或运行的原因
   阶段                   是否可能   排除理由
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   预热 (warmup)          ❌         单卡也走 warmup，且 warmup 阶段 rank 0 的 execute 成功了
   实例化 (instantiate)   ❌         错误栈明确指向 relu_op.cpp :: launch_relu_amp_fwd_cuda()，不是 cudaGraphInstantiat
                                     e
   运行 (runtime)         ❌         错误发生在 task.compile() 期间，run_iter() 根本没执行到
   捕获 (capture)         ✅         rank 1 在 capture mode 内 rebuild cache，execute 炸在 replay 节点时
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  修复方向（按 Warmup.md + P_ULTIMATE.md 设计）
  1. 消灭全局唯一 cache：改为 std::unordered_map<cudnnHandle_t, ...> 或按 device_id 索引
  2. 预热阶段覆盖所有 handle：在 compile_capture_simple() 的 warmup 循环中，为每个 rank 的 handle 预热，而不是只预热 ran
  3. capture 阶段只读不写：launch_relu_amp_fwd_cuda 在 capture mode 下禁止 rebuild，若 cache miss 直接报错（确保设计契约
     强制执行）

# 【小伙伴D】

## 代码审查报告
阅读了 Warmup.md 和 P_ULTIMATE.md 后，对照当前代码，现报告以下问题：

### 问题 1：build_amp_fwd_graph 在 CUDA Graph 捕获窗口内被调用（当前风险）
核心矛盾 ： Warmup.md 明确要求：
 Phase 1 warmup必须在capture之前完成。如果在capture stream的begin/end之间调用build_plans，可能导致cuDNN引擎搜索的临时kernel被错误捕获到CUDA Graph中。
当前流 ： compile_capture_simple 中，预热只在 GPU 0 上执行（ task_base.cpp:L278-L287 ），只为 handle_0 建立了缓存。在捕获阶段，对 rank 1~N-1， launch_relu_amp_fwd_cuda 发现缓存缺失 → 在 cudaStreamBeginCapture 和 cudaStreamEndCapture 之间 调用 build_amp_fwd_graph(handle_r) ，其中包括：

```
g->build_operation_graph(handle)    ← 在捕获窗口内
g->validate()                       ← CPU操作，安全
g->create_execution_plans(...)      ← CPU操作，安全
g->check_support(handle)            ← 在捕获窗口内
g->build_plans(HEURISTICS_CHOICE)   ← 在捕获窗口内
```
虽然 HEURISTICS_CHOICE 理论上不触发benchmark kernel，但 build_operation_graph 和 check_support 仍涉及 cuDNN 内部 API 调用。 在捕获窗口内做非 replay 的 CUDA 库调用是一个结构性问题 。

对于 SimpleTask（串行捕获）， HEURISTICS_CHOICE 实际上不会启动 kernel，所以目前不会触发错误，但违反设计原则。对于 DeepLearningTask 的并行捕获路径，多线程并发 build_plans 会有额外风险（见问题3）。

### 问题 2：核函数调用 execute() 后没有 cudaStreamSynchronize，cuDNN 可能不是真正异步安全的
在 launch_relu_amp_fwd_cuda （ relu_op.cpp:L293 ）：

```
TR_CUDNN_FE_CHECK(
    fwd_cache.graph->execute(handle, vp, nullptr),
    "execute AMP FWD");
cudaEventRecord(state.streams[si].last_done_event, s);
```
cuDNN FE execute() 提交 kernel 到 stream 后 立即返回 （非阻塞）。紧接着 cudaEventRecord 记录事件。这个逻辑是正确的—— cudaEventRecord 会在该 stream 上所有先前操作（包括 execute 提交的 kernel）完成后才被触发。所以跨算子屏障能正确等待。这部分没问题。

但问题在于： warmup 阶段的 execute() 是在 stream 上异步执行的，然后立即 cudnnStreamSynchronize 等待 （ op_registry.cpp:L160 ）。但在捕获阶段（ launch_relu_amp_fwd_cuda ）， execute() 后的 kernel launch 只是被 记录 到 CUDA Graph 中，并不是实时执行。 cudaEventRecord 也被记录为图中的一个节点，运行时才触发。这没问题——这就是 CUDA Graph 的核心机制。

### 问题 3：pre_capture（DeepLearningTask）路径下的并行 build_plans 竞态风险
当前 s_amp_fwd_caches 使用 static std::unordered_map<cudnnHandle_t, unique_ptr<Cache>> 。在 pre_capture 的 Phase B3 并行捕获中（ captured_graph.cpp 中的 capture_all_for_rank ），多个线程 并发 调用 launch_relu_amp_fwd_cuda ，每个线程使用不同的 handle ，所以各自独立触发 build_amp_fwd_graph → build_plans(HEURISTICS_CHOOSE) 。

Warmup.md 特别指出 ：
 cuDNN engine cache 并非 per-handle 隔离，底层是进程级全局数据结构。如果 8 线程并发 build_plans()，会严重串行化（cuDNN 内部有全局锁），且可能产生损坏 cache 条目。
虽然 HEURISTICS_CHOICE 模式下 build_plans 只做查找（不做benchmark），但 cuDNN 内部仍有全局锁保护 engine cache。并发查找本身应该是安全的（读锁），但 build_plans 对每个 handle 第一次调用时可能涉及内部状态的 写入 （为该 handle 注册查询结果），这就有竞态风险。

注意 ：这个问题目前只影响 DeepLearningTask（使用 pre_capture 路径），不影响 SimpleTask（ compile_capture_simple 是串行循环）。

### 问题 4：s_amp_fwd_caches 可能无限增长
当 DeviceContext 被销毁重建时（新的 cudnnHandle_t ），旧 handle 对应的缓存条目仍然存在于 static map 中，持有的 feg::Graph 不会被释放。虽然在实际运行场景中 DeviceContext 通常不会反复创建销毁，但这是一个潜在的资源泄漏。

更隐蔽的是：如果 cuDNN handle 被销毁后重新创建，新 handle 的值可能碰巧等于某个旧 handle 值。此时会 错误命中 旧缓存，使用一个绑定到已销毁 handle 的 feg::Graph ，导致未定义行为。

### 修复建议
针对问题 1（最关键） ：预热时预填充所有 handle 的缓存。

修改 compile_capture_simple 的预热逻辑，不仅对 GPU 0 预热，而是 对所有 rank 预热 （因为它们仍然是串行的，不会触发并行竞态）：

```
// 当前：只对 GPU 0 预热
warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[0]);

// 建议：对所有 GPU 预热（串行，无竞态）
for (int rank = 0; rank < num_gpus_; ++rank) {
    warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[rank]);
}
```
这样在捕获阶段，所有 rank 的 handle 都在缓存中，捕获窗口内只有 execute() （纯 kernel launch replay），完全符合 Warmup.md 的设计原则。

针对问题 3（DeepLearningTask 并行捕获） ：在 pre_capture 的 Phase B2 预热阶段，同样对所有 handle 预热：

```
// Phase B2: 对所有 GPU 串行预热（不是仅 GPU 0）
for (int rank = 0; rank < num_ranks; ++rank) {
    cudaSetDevice(contexts[rank]->device_id());
    // warmup on each GPU's handle
}
```
这样 Phase B3 并行捕获时，所有 handle 的缓存都已命中，不会有任何 build_plans 调用。

针对问题 4 ：可以在 DeviceContext::~DeviceContext() 中清理对应的缓存条目，或者考虑不使用 static 变量而改用 per-task 生命周期管理。

### 总结
问题 严重程度 影响范围 状态 #1 build_plans 在捕获窗口内调用 中 SimpleTask (ranks>0) 待修复 #3 并行 build_plans 竞态 高 DeepLearningTask 待修复 #4 缓存无限增长 低 全部 后续优化 原 singleton unique_ptr 崩溃 高 SimpleTask 多卡 ✅ 已修复

核心理念 ：Warmup.md 的三段式策略的精髓是「 所有 build_plans 必须在 BeginCapture 之前完成 」。当前代码的预热只覆盖 handle_0，导致 rank>0 的捕获在窗口内做 build_plans 。修复方案很简单： 预热循环覆盖所有 rank （预热阶段本身就是串行的，无竞态风险）。

4/4 已完成

阅读 Warmup.md 和 P_ULTIMATE.md（已完成）

阅读 relu_op.cpp / task_base.cpp / op_registry.cpp / capture_cuda.cpp（已完成）

对比设计文档分析代码问题

输出问题清单与修复建议

以上就是完整的分析。问题的本质是： 预热只覆盖了 GPU 0，导致 rank>0 的捕获期间在 BeginCapture / EndCapture 窗口内做了 build_plans 。原来的 singleton unique_ptr 问题已经通过 per-handle map 修复了，但 build_plans 在捕获窗口内的结构性问题仍然存在。

需要我实施上述修复建议吗？

# 【专家DS】

关于多卡CUDA Graph捕获中遇到的问题，你同事的分析是正确的。在cuDNN Frontend（Graph API）的语境下，单GPU预热、多GPU并行捕获的方案**不现实且风险极高**。这个问题确实主要源于cuDNN Frontend架构里“全局静态cache”与“捕获模式”的深度绑定。

### 🤔 为什么"单卡预热"在Frontend行不通？
cuDNN Frontend的性能核心是 **`Execution Plan`（执行计划）**。`build_plans` 是搜索最优执行策略的**启发式搜索（Heuristics Search）**，过程中会生成大量**特定于当前GPU架构的、已编译好的CUDA Kernel**。这恰恰是与"Legacy API"最根本的设计差异。

*   **在 `Legacy API` 时期**：算子与实现高度绑定，预热行为相对简单且可预测。每个设备的Context天然隔离，因此"单卡预热，多卡捕获"策略常能凑效。
*   **在 `cuDNN Frontend` 时期**：Framework引入了**全局静态缓存（Global Static Cache）** 来管理昂贵的`Execution Plan`，特别是已编译的Kernel代码。在CUDA图捕获期间，任何需要新Kernel的操作都会被视为非法。因此，**cuDNN Frontend要求所有Kernel必须在捕获前完成加载，并缓存于该进程中**。

你的场景之所以失败，核心链路如下：
1.  **GPU 0 预热**：为GPU 0生成了最优`Execution Plan`及相关Kernel，存入**全局静态缓存**。
2.  **并行捕获**：Rank 0利用已缓存的Kernel成功完成图捕获。而Rank 1开始并行捕获，并试图复用缓存中的`Execution Plan`。
3.  **触发`build_plans`**：由于GPU架构的微小差异或驱动的上下文差异，缓存中的那个`Execution Plan`对Rank 1的GPU可能不适用或不是最优的。**Frontend启发式算法**判定需要为Rank 1的GPU重新优化或重新编译某些Kernel。
4.  **非法操作 & 静默失败**：在**捕获模式**下，任何Kernel的编译与加载（`build_plans`的核心工作）都被CUDA视为非法操作，因为这会修改GPU上下文。这会导致捕获失败、死锁，或产生一个损坏的、无法重放的CUDA Graph（即静默失败）。

### ✅ 如何彻底规避此问题
解决思路很明确：必须100%确保`BeginCapture`之前，所有GPU所需的所有cuDNN Kernel都已编译并缓存。

#### 🔧 方案：为每个Rank独立、充分地预热
实现一个“**多卡独立预热**”函数，在创建任何CUDA Graph之前调用，它为每个参与训练的GPU单独、充分地完成预热。

**预热流程 (Warmup Procedure)**：
1.  **准备阶段**：为每个GPU（`local_rank`）创建一个独立的`cudnnHandle_t`，并调用`cudaSetDevice(local_rank)`绑定设备。同时，准备一些与真实训练相同的虚拟输入数据（形状、数据类型一致）。
2.  **内核预热**：在每个GPU上，使用你框架中定义的cuDNN Frontend算子执行**至少2-3次**完整的前向+反向的“dry-run”。必须执行反向传播，确保梯度的计算图也被捕获和优化。请务必调用`cudaDeviceSynchronize()`等待该次执行完成，以确保Kernel编译和加载完全完成。
3.  **并行化**：为提高效率，上述步骤可通过C++的`std::thread`或OpenMP为每个GPU启动一个独立的线程来并行执行。
4.  **同步**：在进入CUDA Graph的捕获流程前，必须确保所有参与`beginCapture`的stream和线程都已完成上述预热。

### 💡 延伸建议与排查方向
*   **善用“无设备编译”**：关注 **cuDNN Frontend的Deviceless AOT Compilation** 功能，可将计算图的`Execution Plan`**提前编译成平台无关的中间表示**并缓存，是更彻底的优化方案。
*   **关注版本与前沿特性**：密切关注cuDNN Frontend版本的更新，特别是与CUDA Graphs的兼容性改进。
*   **诊断与确认**：在捕获代码中开启cuDNN Frontend详细日志。通过对比各Rank的日志，可以精确定位`build_plans`被触发的位置。

你的同事的判断是准确的。在cuDNN Frontend下，**必须在所有目标GPU上完成完整的、独立的预热，才能安全地进行并行图捕获**。

# 【专家KM】

小伙伴的判断是对的：在使用 cuDNN Frontend 的多卡 CUDA Graph 场景下，必须每个 rank 都预热，单卡预热无法覆盖多卡。你感觉"以前可以"是因为 Legacy API 和 Frontend API 在 plan/kernel 缓存机制上有本质差异。

---

根本原因：三层 per-device 隔离

隔离层	说明	
Kernel Cache	cuDNN Frontend 的默认 kernel cache 是 per-device（按 device ordinal） 的。文档明确说："Each device has its own separate cache. Cached kernels are not shared across devices." citeweb_search:6#0	
Execution Plan	`ExecutionPlan` 绑定到特定的 `cudnnHandle_t`，而 handle 和设备强绑定。即使序列化 plan，文档也声明 "No ... cross-device compatibility guarantee is offered." citeweb_search:7#2	
RTC 编译	cuDNN Frontend 的 fusion 算子（SDPA、Flash Attention、Conv fusion 等）大量依赖 Runtime Compilation（RTC/JIT）。编译出的 CUBIN 是 per-device 的，必须在目标卡上现场编译。	

这三层隔离决定了：在 GPU 0 上预热产生的所有缓存，对 GPU 1 完全不可见。

---

为什么 Capture 窗口会炸

CUDA Graph Capture 有一个硬性规则：在 `BeginCapture` / `EndCapture` 之间，不允许任何同步、内存分配、kernel 编译 citeweb_search:7#0。

当 rank>0 开始捕获时，如果该设备的 kernel cache 是空的，cuDNN Frontend 在执行算子时会尝试 `build_plans` → 触发 heuristics 搜索 → 触发 RTC kernel 编译。这一系列操作会：

1. 分配临时 workspace（内存分配）；
2. 调用 NVRTC 编译 kernel（同步/编译）；
3. 可能查询 workspace size（设备同步）。

全部落在 Capture 窗口内，直接违反规则，导致捕获失败或产生非法 Graph。

---

为什么 Legacy API "以前可以"

Legacy API（`cudnnConvolutionForward` 等）和 Frontend API 的差异在于：

维度	Legacy API	cuDNN Frontend (Graph API)	
引擎类型	大量预编译 kernel（binary 已存在于 libcudnn.so）	大量 RTC 引擎（JIT 编译）	
Plan 构建	轻量，主要是 engine config 选择	重，包含 kernel 编译、workspace 计算	
缓存位置	主要在 cuDNN 内部全局 heuristics cache	显式的 per-device KernelCache + ExecutionPlanCache	

Legacy API 的 plan 构建开销小，且预编译 kernel 不需要在 capture 窗口内做 RTC，所以即使 cache 未命中，通常也不会触发 capture 禁止的操作。而 Frontend API 引入的 fusion 算子（尤其是你们可能用的 SDPA/Flash Attention）重度依赖 RTC，这是问题的根源。

---

解决方案

方案 A：每个 rank 都预热（最稳妥）

```cpp
void worker_thread(int gpu_id) {
    cudaSetDevice(gpu_id);
    
    // 在自己的卡上完整跑几次，填充该设备的 kernel cache 和 plan cache
    for (int i = 0; i < num_warmup_iters; ++i) {
        run_full_forward_backward();  // 触发 cuDNN Frontend build_plans
    }
    cudaStreamSynchronize(stream);      // 确保所有异步编译完成
    
    // 然后再捕获
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    run_full_forward_backward();       // 此时所有 plan 已缓存，纯执行
    cudaStreamEndCapture(stream, &graph);
}
```

cuDNN 官方文档和 PyTorch 的 CUDA Graph 教程都明确建议："first execute their workloads during a warm up run before attempting graph-capture" citeweb_search:7#0web_search:7#5。

方案 B：用 `populate_cuda_graph` 绕过 Capture 限制（cuDNN Frontend 1.8.0+）

如果你们预热时间实在无法接受，可以考虑 cuDNN Frontend 1.8.0 引入的 Native CUDA Graph API：

```cpp
// 不在 capture 窗口内做，而是在外部直接构建
cudnn_frontend::graph.populate_cuda_graph(empty_cuda_graph, variant_pack);
```

这允许你在 capture 窗口外完成所有 plan 构建和 kernel 编译，然后直接把 cuDNN 的节点"填充"进一个现成的 CUDA Graph。后续通过 `update_cuda_graph` 更新指针即可 replay citeweb_search:7#1web_search:6#8。

注意：这仍然需要 per-device 构建，只是不需要在 `BeginCapture/EndCapture` 之间做。而且只有部分引擎支持（带 `CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API` 标记）。

方案 C：序列化 Plan（不推荐）

虽然 cuDNN 提供了 `getJsonRepresentation` / `loadFromJson` 来序列化 plan，但文档明确说不保证跨设备兼容 citeweb_search:7#2。同型号 GPU 理论上可以，但官方不保证，生产环境不建议赌。

---

总结

问题	答案	
单卡预热能否覆盖多卡捕获？	不能。Kernel cache、plan、RTC 编译结果都是 per-device 的。	
小伙伴说的 "全局静态 cache" 是什么？	可能是指 cuDNN 内部的全局引擎注册表，但实际生效的编译缓存是 per-device 的。GPU 0 的 cache 对 GPU 1 不可见。	
为什么加了 Frontend 算子就不行？	Frontend 的 fusion 算子（SDPA 等）大量依赖 RTC，必须在目标设备上编译 kernel。Legacy API 用预编译 kernel，没有这个问题。	
最佳实践？	每个 rank 都预热，或升级到 cuDNN Frontend 1.8.0+ 使用 `populate_cuda_graph` 绕过 capture 窗口限制。	

你们现在的做法（只预热 GPU 0）在 cuDNN Frontend 下是行不通的，必须改成每卡独立预热。如果预热时间成为瓶颈，优先考虑 `populate_cuda_graph` 方案。

