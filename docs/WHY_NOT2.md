# 为什么 CPU 模式不实现双图多线程并行

> **结论先行**：CPU 模式保持串行执行。用多线程强行"对齐"GPU 双图并行的语义，收益为负、复杂度陡增、且会造成虚假的平台一致性幻觉。

- **版本**：4.20.1
- **日期**：2026-04-20
- **作者**：Tech-Renaissance Team
- **状态**：设计决策冻结（Design Decision Frozen）

---

## 1. 核心问题：我们在试图对齐什么？

GPU 的双图并行（`task.run("xfer", "compute")`）有明确的物理意义：

```
GPU 时间轴:
[  xfer: PCIe H2D  ]  ← StreamKind::TRANS
                   [  compute: CUDA kernel  ]  ← StreamKind::COMP_1
                   ↑
            真正的硬件级并发，传输延迟被计算掩盖
```

CPU 的双图并行如果也用多线程实现，时间轴长这样：

```
CPU 时间轴（假想的双线程）:
[  xfer: memcpy (内存带宽)  ]  ← Thread A
[  compute: SIMD (内存带宽) ]  ← Thread B
            ↑
            两个线程抢同一根内存总线，加上锁/同步/上下文切换开销
```

**关键差异**：GPU 的 xfer 和 compute 使用**不同的物理资源**（PCIe vs SM），而 CPU 的 xfer 和 compute 使用**同一资源**（系统内存带宽 + 共享 L3）。多线程不会减少总工作量，只会增加调度开销。

---

## 2. 硬件特性对比

| 特性 | GPU | CPU |
|------|-----|-----|
| 内存架构 | 独立显存 + 页锁定主机内存 | 统一的系统内存 |
| 传输成本 | PCIe 传输，延迟 ~10μs | `std::memcpy`，延迟 ~ns |
| 并行单位 | 硬件 Stream，异步执行引擎真并发 | OS 线程，时间片轮转伪并发 |
| 计算单元 | 数千 CUDA Core | 数十核心，宽 SIMD |
| 典型瓶颈 | 显存带宽 / PCIe 带宽 | 计算吞吐量（若 SIMD 优化到位）|

GPU 需要双图并行是因为**传输和计算有独立的硬件管道**，重叠它们能带来 30-50% 的端到端性能提升。

CPU 没有这种管道分离。`memcpy` 和 Eigen3 SIMD 计算都要走 DDR 总线，两个线程同时跑只会互相驱逐缓存、争抢内存控制器，最终总时间 ≥ 串行时间。

---

## 3. 收益估算：为什么是负数

| 场景 | 串行耗时 | 多线程并行耗时 | 说明 |
|------|---------|---------------|------|
| GPU 双图 | `T_xfer + T_compute` | `max(T_xfer, T_compute)` | **显著收益**，硬件级重叠 |
| CPU 双图 | `T_xfer + T_compute` | `T_xfer + T_compute + T_overhead` | **收益为负**，总线争用 + 同步开销 |

其中 `T_overhead` 包括：
- 线程创建/销毁（~μs 级，小图不可忽略）
- `std::mutex` 或 `std::future` 同步
- 缓存行 bouncing（false sharing）
- NUMA 跨节点内存访问（多路服务器场景）

---

## 4. 实现复杂度与维护成本

如果要在 CPU 模式下实现"真正的"双图并行，需要引入：

1. **线程池管理**：避免每次 `run()` 都创建销毁线程
2. **异常传播**：一个图的 kernel 抛异常，另一个图要安全中断
3. **内存模型约束**：`CpuArena` 当前是单线程设计的，需要加锁或改为 thread-local
4. **调度亲和性**：需要考虑 NUMA、超线程、核心绑定，否则性能更差
5. **调试地狱**：Heisenbug、race condition、死锁——而这些 bug 在 GPU 模式下根本不存在

这些成本换来的不是性能提升，而是**更慢、更复杂、更难调试**的代码。

---

## 5. CPU 模式的正确自我定位

TR4 的设计目标是 **A100×8 ResNet-50 极致性能**。CPU 模式在这个蓝图中的角色是：

| 场景 | 是否应该用 CPU 模式 | 期望 |
|------|-------------------|------|
| CI/CD 无 GPU 环境跑单元测试 | ✅ 是 | 结果正确即可 |
| 开发阶段快速验证算法 | ✅ 是 | 结果正确即可 |
| 教学/学习框架原理 | ✅ 是 | 串行逻辑更易理解 |
| 生产环境训练 ResNet-50 | ❌ 否 | 请用 GPU 模式 |
| 追求极致吞吐量 | ❌ 否 | 请用 GPU 模式 |

CPU 模式的核心 KPI 是**与 GPU 模式结果逐 bit 一致**，而不是**与 GPU 模式行为逐周期对齐**。串行执行在这个定位下是最优解：简单、确定、无锁、易于调试。

---

## 6. 如果真的需要 CPU 并行，应该怎么做？

如果未来确实有 CPU 性能需求（边缘场景），正确的优化层级是**算子内部**，而非**图调度层**：

```
❌ 错误方向：图层面多线程
   task.run("xfer", "compute")  // 两个线程各跑一个图

✅ 正确方向：算子层面多线程/OpenMP
   launch_tr_axpy_fp32_kernel_cpu(...)
       ↓
   Eigen3 + OpenMP 内部并行（已在 TR_USE_EIGEN 路径）
       ↓
   单张量计算利用多核 SIMD，图调度保持单线程
```

这与 PyTorch CPU 后端、TensorFlow XLA CPU 的设计哲学一致：**框架层串行，算子层并行**。

---

## 7. 唯一值得重审的例外

多路 CPU 服务器（2-4 socket，每 socket 独立内存控制器）理论上可以通过 NUMA-aware 绑定实现类似 GPU 的"传输-计算"流水线：

```
Socket 0: [memcpy from Node 1 DRAM]  →  [compute on Node 0 CPU]
Socket 1: [memcpy from Node 0 DRAM]  →  [compute on Node 1 CPU]
```

但这要求：
- 显式的 NUMA 内存分配（`numa_alloc_onnode`）
- 显式的线程绑核（`sched_setaffinity`）
- 显式的跨 socket 数据搬运（仍是 memcpy，非 DMA）

实现复杂、适用面极窄、且收益远不如直接上 GPU。在当前 TR4 的范围内，**不值得为此增加框架复杂度**。

---

## 8. 当前实现与注释规范

CPU 模式下 `task.run(a, b)` 的当前行为：

```cpp
#else
    // CPU模式：串行执行
    // 设计决策：不实现多线程并行，详见项目根目录 WHY_NOT2.md
    for (int i = 0; i < num_gpus_; ++i) {
        backend_->run_graph(a, i, stream_a);
        backend_->run_graph(b, i, stream_b);
    }
#endif
```

已在 `example_dual_graph_cpu.cpp` 中添加注释说明：

```cpp
// Note: In CPU mode, StreamKind is only a label; dual graphs execute sequentially
```

---

## 最终结论

> **保持当前设计。CPU 模式串行执行双图。用注释明确告知用户限制，而不是用多线程制造虚假的平台一致性。**

诚实的不一致，优于伪装的对齐。
