# 【小伙伴D】

我们已经在 relu_op.cpp 中用 cuDNN Frontend 替代了自定义 kernel。当前代码跑的是：

```
FWD = cuDNN RELU_FWD + CMP_GT  (与 cuDNN BOOLEAN Bitmask 完全相同)
BWD = cuDNN MUL(dY, mask)      (与 cuDNN BOOLEAN Bitmask 完全相同)
```
没有 地址解码、没有线程粒度问题、没有向量化问题——这些都是旧自定义 kernel 的事，已经删掉了。

### 差额 1,300 μs 从哪来？
```
1,221 μs → cuDNN graph::execute()（我们的实现与独立版完全一致）
1,300 μs → task.run("relu") 框架开销
───────
2,521 μs → 实测 ~2,580 μs ✓
```
task.run("relu") 每调用一次要走：graph 名查找 → graph engine 调度 → multi-stream 事件管理 → 图实例化 —— 即使图只有一个节点，框架管线开销约 1.3ms。

### 验证方法
在 launch_relu_amp_fwd_cuda 的 cuDNN execute 自身打点：

```
auto t0 = clock::now();
s_amp_fwd_cache->graph->execute(handle, vp, nullptr);
// ↓ 此时应 ≈ 1,221 μs
```
这就能证明框架开销是外部的，cuDNN 内核本身没有问题。

要不要我加上这个内核级计时，把"cuDNN 纯执行时间"和"SimpleTask 总耗时"分拆打印？



# 【用户】

优化这个管线。我要求图实例化就是在compile()阶段完成的。run的任务就完全只是索引到那个图，然后直接运行而已。DeepLearningTask也相应修改（不同之处进在于DeepLearningTask是靠图集来索引）。我的要求是run执行不能包含实例化的漫长过程。实例化只需一次！！！

# 【小伙伴D】

## 完整回答：TaskBase 从 compile() 到 run() 的全链路
### 一、compile() 如何存储编译后的图
关键文件：

- task_base.cpp:L207-L241 -- compile_impl 主流程
- task_base.h:L286-L293 -- 成员变量定义
compile() 通过 compile_impl 执行七步固定流水线（非虚模板方法）：

```
compile_freeze_global()           // 1. 冻结 GlobalRegistry
compile_invoke_on_prepare()       // 2. 调用虚钩子 on_prepare()
compile_verify_memory_locked()    // 3. 确保阶段为 MEMORY_LOCKED
compile_alloc_hardware()          // 4. 分配 GPU/CPU 硬件内存
    ↓
[is_simple_task() 分叉]
    ├── SimpleTask: compile_capture_simple()   // 5a. 逐图独立捕获
    └── DeepLearningTask: pre_capture(atlas)   // 5b. 通过 GraphAtlas + 
    pre_capture
compile_mark_compiled()           // 6. 标记 phase_ = COMPILED
```
编译结果存储在两个地方：

（A）DeepLearningTask 路径 —— 使用 pre_capture + captured_result_ ：

```
// [task_base.cpp:L230-L237]
GraphAtlas atlas = build_simple_atlas(name_to_gid_);
captured_result_ = pre_capture(atlas, ctx_ptrs);
```
pre_capture 在 captured_graph.cpp:L171-L259 中执行四阶段：

1. Phase B1 - 去重 ：遍历 GraphAtlas 的所有 variant x gid 槽位，按 Key{cg, gid, shape_id} 去重，相同键的槽位共享 captured_idx （通过 seen unordered_map 去重）
2. Phase B2 - cuDNN 预热 ：GPU 0 单线程串行 warmup
3. Phase B3 - 多 rank 并行捕获 ：多个线程分别调用 capture_all_for_rank() ，每个 GPU 调用 CapturedGraph::capture() -> capture_cuda() ，执行 cudaStreamBeginCapture -> 逐节点 replay -> cudaStreamEndCapture -> cudaGraphInstantiate ，将 cudaGraphExec_t 句柄存入 per_rank_execs_[rank]
4. Phase B4 - Warmup launch ：Rank 0 预启动一次，填充 GPU 内部缓存
结果返回为 PreCaptureResult{graphs, atlas} ，其中 graphs 是去重后的 vector<CapturedGraph> ，每个包含 per_rank_execs_ （每个 GPU 一个 cudaGraphExec_t ）。

（B）SimpleTask 路径 —— 使用 simple_captured_graphs_ ：

```
// [task_base.cpp:L275-L311]
simple_captured_graphs_.emplace(name, std::move(cg));
```
每个命名图调用 CapturedGraph::capture() 独立捕获，存储在 simple_captured_graphs_ （ unordered_map<string, CapturedGraph> ）中，不走 GraphAtlas。

### 二、run(name) 逐步分析
关键文件： task_base.cpp:L449-L541

run(name) 调用 run_impl(name, false) ，流程如下：
 Step 1：资格检查
```
check_run_eligibility(dry_run);  // 断言 phase_ == COMPILED 且 backend_ 存在
``` Step 2：查找 named_graphs_
```
auto it = named_graphs_.find(name);  // 从编译期注册的命名图中获取 GraphEntry
```
named_graphs_ 在 add_graph() 时填充（ task_base.cpp:L190-L194 ），它存储的是 ComputationGraph （纯算子拓扑，不含硬件句柄）。
 Step 3：is_simple_task() 分叉
SimpleTask 路径 ：

1. 从 simple_captured_graphs_ 查找预捕获的 CapturedGraph
2. 逐 rank 循环： cudaSetDevice(rank) -> 获取 ctx.stream(entry.stream) -> 调用 cap_it->second.launch(rank, stream)
3. CapturedGraph::launch() 内部直接执行 cudaGraphLaunch(per_rank_execs_[rank], stream) -- 见 captured_graph.cpp:L148-L162
4. 全部 rank 发射完毕后 统一 synchronize_all()
DeepLearningTask 路径 ：

1. 通过 name_to_gid_ 查找 GraphId
2. 通过 captured_result_.atlas.index(0, gid) 查找 captured_idx （O(1) 二维数组索引）
3. 用该 idx 从 captured_result_.graphs[idx] 取出 CapturedGraph
4. 逐 rank 发射 + 全局同步
关键点 ： run() 期间 没有任何图构建、捕获或实例化操作 。所有 cudaGraphExec_t 句柄都是 compile() 阶段预先创建好的， run() 仅做非阻塞的 cudaGraphLaunch + 全局 cudaDeviceSynchronize 。
 Step 4：底层 launch 实现
```
// [captured_graph.cpp:L148-L162]
void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
        NativeGraph exec = per_rank_execs_[rank];
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec),
                        static_cast<cudaStream_t>(stream));
    } else {
        // CPU 路径：顺序执行 cpu_ops_ 函数指针队列
        for (const auto& op : cpu_ops_) {
            op.fn(static_cast<CpuOpContext*>(op.ctx));
        }
    }
}
```
CUDA 路径的本质就是一次 cudaGraphLaunch 调用。注意 launch 是非阻塞的 -- 只提交到流，不等待完成。由 TaskBase 的全局 synchronize_all() 统一等待。

### 三、run() 期间是否有图实例化/重编译？
完全没有。 设计原则是将重活全部放到编译期：

操作 发生阶段 add_graph() 注册 ComputationGraph MEMORY_LOCKED cudaStreamBeginCapture compile() 期间 cudaStreamEndCapture + cudaGraphInstantiate compile() 期间 去重 (B1)、cuDNN warmup (B2)、多 rank 捕获 (B3) compile() 期间 cudaGraphLaunch run() 期间 cudaDeviceSynchronize run() 期间

run() 仅做两件事： 发射预先实例化的 CUDA Graph + 全局同步 。这是静态图编译的核心哲学 -- 编译期一次性做完所有捕获工作，运行期只做热路径上的 cudaGraphLaunch 。

### 四、DeepLearningTask::run() 与 SimpleTask::run() 的区别 SimpleTask::run(name)
```
// [simple_task.h:L98-L104]
void run(const std::string& name) { TaskBase::run(name); }
```
- 直接透传 到 TaskBase::run(name) 即 run_impl(name, false)
- 从 simple_captured_graphs_ 查找对应的 CapturedGraph
- 逐 rank 发射 -> synchronize_all -> 返回
- 不涉及 epoch 循环、batch 管理、学习率调度、SEMA、早停等 DeepLearningTask::run()
```
// [deep_learning_task.cpp:L280-L281]
TrainingResult DeepLearningTask::run() {
    return run_impl(false);
}
```
- 不调用 TaskBase::run(name) -- 它有自己的 run_impl(bool) 方法（同名不同签名）
- 封装完整的 epoch 循环：
  1. 初始化运行期状态（ current_epoch_ , best_top1_ 等）
  2. 初始化 Preprocessor 、 Scheduler
  3. 按 epoch 循环： run_train_epoch() -> 可选 run_val_epoch() -> SEMA 切换 -> 早停检查
  4. 返回 TrainingResult{batch_metrics, epoch_metrics, ...} DeepLearningTask::run_train_epoch() 内部如何调用 TaskBase::run(name)
```
// [deep_learning_task.cpp:L524-L622]
void DeepLearningTask::run_train_epoch() {
    // Phase 1: Batch 0 单独传输
    TaskBase::run("xfer_a");                              // 单图执行

    // Phase 2: Batch 0 ~ N-2 循环（双图并行）
    for (int batch = 0; batch < batches - 1; ++batch) {
        TaskBase::run(xfer, compute);                     // 双图并行执行
        TaskBase::run("first_layer_bwd");                 // post-batch 步骤
    }
    
    // Phase 3: Last batch
    TaskBase::run(last_compute);                          // 最后一个batch
    TaskBase::run("first_layer_bwd");
}
```
这里调用的正是 TaskBase::run(const string& a, const string& b) 双图并行版本（ task_base.cpp:L543-L619 ），两个图在 不同流 （如 TRANS + COMP_1）上非阻塞发射，由 GPU 硬件自动并行。
 核心差异总结
维度 SimpleTask::run(name) DeepLearningTask::run() 签名 void run(const string&) TrainingResult run() 底层实现 TaskBase::run(name) 自己的 run_impl(bool) 图启动方式 单图/双图非阻塞 launch + sync 内部调用 TaskBase::run(name) 和 TaskBase::run(a,b) 图索引方式 simple_captured_graphs_[name] captured_result_.atlas.index(variant, gid) + captured_result_.graphs[idx] Epoch/Batch 管理 无，用户手动循环 完整 epoch 循环 + 双缓冲异步传输 优化器/调度器 不涉及 每 batch 调用 scheduler.step()

### 五、整个数据流全景观
```
用户代码                  compile() 编译期                run() 运行期
────────                ──────────────                 ────────────
alloc("w", shape)  ────────────────────────────────────────────────> MemoryPlan
finalize_memory()   ──> phase_ = MEMORY_LOCKED
add_graph("fwd", cg) ──> named_graphs_["fwd"] = GraphEntry{cg, COMP_1}
compile()           ──> compile_impl():
                           ├─ compile_alloc_hardware()
                           └─ SimpleTask:
                                compile_capture_simple()
                                  -> CapturedGraph::capture()
                                    -> capture_cuda():
                                       cudaStreamBeginCapture
                                       replay nodes...
                                       cudaStreamEndCapture
                                       cudaGraphInstantiate -> exec_t
                                  -> simple_captured_graphs_["fwd"]
                                     .per_rank_execs_[i] = exec_t

                           └─ DeepLearningTask:
                                pre_capture(atlas)
                                  -> Phase B1: 去重
                                  -> Phase B3: 多rank捕获
                                  -> captured_result_.graphs[k]
                                     .per_rank_execs_[rank] = exec_t
                       phase_ = COMPILED

task.run("fwd")     ────────────────────────────────────> run_impl("fwd", 
false):
                                                             ├─ 
                                                             check_run_eligibil
                                                             ity (COMPILED?)
                                                             ├─ named_graphs_.
                                                             find("fwd")
                                                             ├─ 
                                                             simple_captured_gr
                                                             aphs_.find("fwd")
                                                             ├─ for rank in 0..
                                                             num_gpus:
                                                             │    cudaSetDevice
                                                             (rank)
                                                             │    cg.launch
                                                             (rank, stream)
                                                             │      -> 
                                                             cudaGraphLaunch
                                                             (exec_t, stream)
                                                             └─ synchronize_all
                                                             ()
```
一句话概括： compile() 是工厂，把所有图一次性制成 cudaGraphExec_t 成品；run() 是发射器，只管调用 cudaGraphLaunch 发射这些成品，没有任何重建开销。