# 【小伙伴D】

## 🔴 根因 1（BWD）：run_iter 每迭代同步 — 框架级串行化

simple_task.h:L172-L176

```
for (int i = 0; i < iterations; ++i) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);  // ← 每轮都同步！
}
```
而 legacy gap_legacy.cpp:L210-L217 ：

```
for (int i = 0; i < iterations; ++i) execute_backward();
CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));  // ← 仅末尾一次
```
影响 ：每次 CUDA Graph replay 被强制拆分为"发射→等完成→发射→等完成"的串行模式。GPU 无法将连续 kernel 流水线化，每次 sync 引入 ~几到十几 μs 的空转间隙（CPU 被阻塞 → GPU 做完 → CPU 唤醒 → 发下一发）。

虽然这在 FC 测试中"几乎不带来开销"(*)，但 FC BWD 的 kernel 时间远长，相对占比小。对于 GAP BWD 这种短 kernel（仅 ~390 μs）， sync 间隙占比放大到 ~40%+ 是可预期的 。

(*) FC 的 test_fc_amp.cpp 可能用的是 FWD+BWD 合并图模式（图内已包含两个 kernel），而不是拆成两个独立 BWD 子图。

## 🔴 根因 2（BWD）：BWD kernel 未预热，单元编译在 CUDA Graph capture 内发生
op_registry.cpp:L63-L64 — require_warmup 列表：

```
case ComputeOp::GAP_FP32_FWD:
case ComputeOp::GAP_AMP_FWD:
    return true;
// ❌ GAP_FP32_BWD / GAP_AMP_BWD 不在列表中！
```
BWD 的 CUDA kernel 从未预热，直接在 cudaStreamBeginCapture 内部的首次 launch 中触发 CUDA 模块的延迟加载（lazy JIT/模块解析）。这部分"冷启动"开销固化为 CUDA Graph 的一部分，replay 时每次产生 不必要的事件处理 / 固定开销 。

## 🟡 根因 3（BWD）：Stride 参数推送与冗余地址计算
当前 kernel gap_op.cu:L30-L56 接收 13 个参数（含 6 个 size_t stride），legacy 仅 7 个 gap_backward.cu:L56-L59 。

当前 kernel 为每个线程计算：

```
size_t dx_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c * 
dx_c_stride;
```
即 4 次 64-bit 乘法 + 3 次 64-bit 加法 （A100 上 64-bit 整数 MP/AD 吞吐为 32-bit 的 1/4）。

而 legacy 直接用 dx[idx] （ idx = blockIdx.x * blockDim.x + threadIdx.x ），编译器充分优化的单一 write。当前 kernel 在紧凑 NHWC 布局下 dx_idx == idx ，但 stride 是运行时参数，编译器无法折叠 ，每线程多执行 7 条额外整数指令，累计对 51M 元素造成可观测开销。

## 🟢 根因 4（FWD）：13% 轻微差距
finalize_cudnn_graph cudnn_utils.h:L102-L107 使用 {HeurMode_t::B, HeurMode_t::FALLBACK} ，legacy 使用 {HeurMode_t::B} 。两者等价（B 优先命中），差异来自 cudaEventRecord 被捕获进了图 （ gap_op.cpp:L403 ），每次 replay 多处理一个 event node。13% 在可接受范围。