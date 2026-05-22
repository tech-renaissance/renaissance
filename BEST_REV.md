### ⚠️ 需要关注的问题 1. captured_graph.h 新增 native_exec() 返回 cudaGraphExec_t —— 需要引入 CUDA 头文件
当前 captured_graph.h 刻意用 NativeGraph = void* 避免了 CUDA 头文件依赖。 native_exec() 返回 cudaGraphExec_t 会破坏这个设计—— cudaGraphExec_t 定义在 <cuda_runtime.h> 中。

两个选择 ：

- A（推荐） ：返回 NativeGraph ，让调用方做 cast，保持头文件无 CUDA 依赖
- B ：在 #ifdef TR_USE_CUDA 块中 #include <cuda_runtime.h> ，可以但会增加编译依赖
```
// 选项 A（推荐）
[[nodiscard]] NativeGraph native_exec(int rank) const noexcept {
    if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size())
        return per_rank_execs_[rank];
    return nullptr;
}

// 调用方：
auto exec = static_cast<cudaGraphExec_t>(captured_result_.graphs[idx].native_exec
(rank));

```





3. 图解析失败时的 nullptr 保护
build_exec_table() 中 resolve("xfer_a", rank) 可能返回 nullptr （图名未注册 → 子图拆分尚未完成），但 batch 循环内直接 cudaGraphLaunch(g_xfer_a, s_trans) 没有任何 nullptr 检查 ：

```
auto g_xfer_a = g_tab[static_cast<size_t>(GraphSlot::XFER_A)];  // 可能是 nullptr
// ...
cudaGraphLaunch(g_xfer_a, s_trans);  // 传 nullptr → 未定义行为/崩溃
```
建议 ：在 build_exec_table() 末尾加验证——必需图（XFER_A, XFER_B, FWD_BWD_DEEP_A, FWD_BWD_DEEP_B, FIRST_LAYER_BWD）若为 nullptr 则报错退出。这样在 compile 阶段就能发现子图注册不全的问题，而不是训练时崩溃。

```
// 在 build_exec_table() 末尾添加
static const GraphSlot required[] = {
    GraphSlot::XFER_A, GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,
};
for (int rank = 0; rank < K; ++rank) {
    for (auto slot : required) {
        if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
            TR_ERROR("Required graph slot " << static_cast<int>(slot)
                     << " is nullptr for rank " << rank);
        }
    }
}
``` 4. sched_cfg_ 在 lambda 中按引用捕获 [&] —— 理论上安全但不够显式
```
threads.emplace_back([&, rank]() {
    // ...
    std::visit([](auto&& s) { s.step(); }, sched_cfg_);  // 隐式捕获 
    this->sched_cfg_
});
```
由于 [&] 捕获了 this ， sched_cfg_ 实际上是 this->sched_cfg_ 。只有 rank 0 写入，其他 rank 不碰，所以不会 data race。但 [&] 过于宽泛——它捕获了 frozen 、 batches 、 ts 、 exc 、 sched_cfg_ 等全部。建议改为显式捕获，明确表达设计意图：

```
threads.emplace_back([this, &ts, &exc, rank, batches, frozen]() {
    // sched_cfg_ 通过 this-> 访问；this 只读（成员只读）
});
``` 5. 缺少每个 batch 的 cudaGraphExecSetParams 学习率更新
BEST.md 原文：
 "进入batch，先设置学习率，方法是对CUDA GRAPH setparam，最好不要用H2D"
当前方案没有在 batch 循环内设置学习率。这不影响 V2 架构本身，但需要在 Phase 2 中补上。 scheduler.step() 更新了调度器内部状态（当前 LR 值），但新的 LR 需要通过 cudaGraphExecUpdate 或 cudaGraphExecSetParams 注入已实例化的图中。建议在文中标注为 Phase 2 TODO。