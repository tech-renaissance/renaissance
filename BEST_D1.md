# DeepLearningTask CUDA Graph 执行架构 —— 最优设计方案

## 零、设计前提

### 0.1 DeepLearningTask::run() 的本质

`DeepLearningTask::run()` 是一个**统括性的、多 epoch 的、包含复杂功能和输出的任务**。它不是 SimpleTask 那种"重复跑同一个算子测性能"的简单循环，而是一个完整的训练引擎：

```
DeepLearningTask::run()
  └─ for epoch in 0..total_epochs_:
       ├─ 渐进式分辨率调整
       ├─ SEMA 切换
       ├─ run_train_epoch()     ← 核心热路径：每 batch 反复执行同一组图
       ├─ run_val_epoch()       ← 验证阶段
       ├─ 指标收集 & 日志
       ├─ 模型保存 & 早停检查
       └─ epoch 结束
```

它天然拥有自己的**epoch 级多线程边界**（train 展开多线程 → join → val 展开多线程 → join），与 BEST.md 的"每个 rank 一个线程"完全契合。

### 0.2 与 TaskBase::run() 的彻底切割

`TaskBase::run(name)` 是为 SimpleTask 设计的通用接口：每次调用可能跑不同的图，需要字符串查找。DeepLearningTask 的 batch 循环是**反复执行同一组图**，完全不同的使用模式。因此：

> **DeepLearningTask::run_train_epoch() 必须彻底重写，不再调用 TaskBase::run()。直接在循环内操作 cudaGraphLaunch。**

### 0.3 奥卡姆剃刀原则

CPU 不需要捕获。CUDA 需要捕获。不要在它们上面建抽象层。在 `run_train_epoch()` 入口处根据 `GlobalRegistry::using_gpu()` 分支，之后走完全不同的代码路径。不要为了一致性而引入不必要的封装。

### 0.4 compile 阶段已经完备

经过代码审计确认：图实例化（`cudaGraphInstantiate`）已经在 `compile()` 阶段完成，`cudaGraphExec_t` 句柄已存入 `captured_result_.graphs[].per_rank_execs_[rank]`。**run 阶段不需要也不应该做任何实例化。**

---

## 一、当前热路径瓶颈（代码审计）

当前 `run_train_epoch()` 的 batch 循环（[deep_learning_task.cpp:564-598](file:///r:/renaissance/src/task/deep_learning_task.cpp#L564-L598)）：

```cpp
for (int batch = 0; batch < batches - 1; ++batch) {
    // ...
    TaskBase::run(xfer, compute);        // ① 双图并行
    // ...
    TaskBase::run("first_layer_bwd");    // ② 单图
    // scheduler.step();
}
```

每次 `TaskBase::run(name)` 的热路径（[task_base.cpp:512-540](file:///r:/renaissance/src/task/task_base.cpp#L512-L540)）：

| 步骤 | 操作 | 分类 |
|---|---|---|
| 1 | `name_to_gid_.find(name)` | 字符串哈希查找 |
| 2 | `captured_result_.atlas.index(0, gid)` | 图集索引 |
| 3 | `for rank: cudaSetDevice(gpu_id)` | 冗余 CUDA API |
| 4 | `for rank: ctx.stream(stream_kind)` | 流指针获取 |
| 5 | `for rank: cg.launch(rank, stream)` → `cudaGraphLaunch` | **唯一有效工作** |
| 6 | `for ctx: synchronize_all()` → `cudaSetDevice` + `cudaDeviceSynchronize` | 全局同步（错误策略） |

**`cudaDeviceSynchronize` 是最大性能杀手**。它会阻塞该 GPU 上所有流的所有 work，彻底破坏多流并行性。BEST.md 明确要求："除非到了 epoch 结束，否则不要调用 DeviceSync，而是对指定的流进行同步。"

### 量化浪费（4 卡 × 每 batch 3 次 run()）

| 操作 | 每 batch 调用次数 |
|---|---|
| `name_to_gid_.find()` | 3 次 |
| `atlas.index()` | 3 次 |
| `cudaSetDevice` | 24 次（3×4 launch + 3×4 sync） |
| `cudaDeviceSynchronize` | 12 次（3×4） |
| `cudaGraphLaunch` | 12 次 ← 唯一需要保留的 |

---

## 二、三个方案的评估

### 2.1 小伙伴 S 的方案

**核心思路**：在 TaskBase 中添加 `CachedGraphPtr` / `CachedStreamPtrs` 包装结构，提供 `get_cuda_graph(rank, GraphId)` 接口。

**优点**：预解析思想正确，流同步方向正确。

**问题**：
1. 引入了 `CachedGraphPtr` 封装 —— 用户明确反对不必要的抽象
2. 图指针存的是 `cudaGraphExec_t*`（指针的指针），而 BEST.md 建议直接用 `cudaGraphExec_t`
3. `sync_three_compute_streams()` 等方法内部仍调用 `get_stream_ptr()` + per-rank 遍历 —— 多了一层间接
4. 单线程模型（非 per-rank 线程），循环内仍需反复 `cudaSetDevice`

### 2.2 小伙伴 K 的方案

**核心思路**：三阶段路线图。阶段 1 单线程预解析 + 直接发射；阶段 2 清理 DeviceContext；阶段 3 per-rank 多线程。

**优点**：
- 三个阶段分工清晰，可逐步实施
- 阶段 1 的 `RankEpochContext` 结构设计合理
- 明确不修改 TaskBase::run()（两者分离）
- 异常传播模式参考了 `pre_capture` 的成熟实现

**问题**：
1. 阶段 1 仍是单线程，循环内 `cudaSetDevice` 没有消除（单线程必须在 rank 间切换 device）
2. `get_exec` lambda 内部仍有 `name_to_gid_.find()` —— 虽然在循环外执行，但应该用更简洁的 `name_to_graph_index_` 替代
3. 三阶段中阶段 1 和 3 实际是同一个功能的两种实现，应该直接做阶段 3

### 2.3 小伙伴 D 的方案

**核心思路**：四层优化（预解析 → streamSync → per-rank 线程 → 二维数组直存）。

**优点**：层次清晰，与 BEST.md 逐条对齐。

**问题**：
1. `name_to_graph_idx_` 放在 TaskBase 中 —— 这是 DeepLearningTask 特有的需求，不应污染基类
2. 没有充分考虑 CPU 分支的独立处理
3. 没有给出具体的 Compiler 子图注册方案（"xfer_a" 等名字从哪来）

### 2.4 综合评估

| 维度 | S | K | D | 最优选择 |
|---|---|---|---|---|
| 预解析思想 | ✅ | ✅ | ✅ | 三者一致 |
| Per-rank 线程 | ❌ 单线程 | ✅ 阶段 3 | ✅ 第三层 | K + D |
| 不污染 TaskBase | ❌ 加了很多 | ✅ | ❌ 加在 TaskBase | K |
| 直接 CUDA API | ❌ CachedGraphPtr | ✅ | ✅ | K + D |
| CPU/GPU 分支 | ❌ 未考虑 | ❌ 未考虑 | ❌ 未考虑 | 需补充 |
| 实施可行性 | 中 | 高（三阶段） | 高 | K |

---

## 三、用户补充意见（设计约束）

1. **不要受基类影响，彻底重写** —— DeepLearningTask 的 run 自成体系
2. **不要封装 CudaGraph/CpuGraph** —— 直接操作 CUDA API
3. **CPU/GPU 在 run 入口分支** —— 根据 `GlobalRegistry::using_gpu()` 一次性判断，之后走不同代码路径
4. **奥卡姆剃刀** —— 如无必要，勿增实体
5. **CPU 不需要捕获，CUDA 需要** —— 核心是通过 ShapeId + GraphId 去重

---

## 四、最优设计方案

### 4.1 总体架构

```
DeepLearningTask::run()
  │
  ├─ if (GPU 模式) → run_gpu()
  │     └─ for epoch:
  │          ├─ 渐进式分辨率 / SEMA 等前置
  │          ├─ run_train_epoch_gpu()        ← 重写，per-rank 线程
  │          │    ├─ 主线程：数据协调（TransferStation 等待）
  │          │    └─ per-rank 线程：
  │          │         cudaSetDevice(device_id)          ← 一次
  │          │         预解析 cudaGraphExec_t + cudaStream_t 到栈变量
  │          │         for batch:
  │          │           cudaGraphLaunch(...)             ← 直接发射
  │          │           cudaStreamSynchronize(...)       ← 精确流同步
  │          ├─ run_val_epoch_gpu()
  │          └─ 指标收集 / 日志 / 保存
  │
  └─ else (CPU 模式) → run_cpu()
        └─ 直接执行，不需要 CUDA Graph
```

**关键原则**：
- GPU 路径和 CPU 路径在 `run()` 入口就分叉，之后完全独立，不存在反复的 `if (is_gpu())` 判断
- GPU 路径中，每个 rank 一个线程，线程 ID = rank ID
- 线程内部 cudaSetDevice 一次，跑完全部 batch
- 同步使用 `cudaStreamSynchronize`，不使用 `cudaDeviceSynchronize`
- 循环内零查找、零 setdevice、零实例化

### 4.2 数据结构

#### 4.2.1 CapturedGraph 新增方法

在 [captured_graph.h](file:///r:/renaissance/include/renaissance/graph/captured_graph.h) 中添加：

```cpp
/// 获取指定 rank 的 cudaGraphExec_t（用于 DeepLearningTask 直接发射）
[[nodiscard]] cudaGraphExec_t native_exec(int rank) const noexcept {
#ifdef TR_USE_CUDA
    if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size())
        return static_cast<cudaGraphExec_t>(per_rank_execs_[rank]);
#endif
    return nullptr;
}
```

#### 4.2.2 DeepLearningTask 新增成员

在 [deep_learning_task.h](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h) 中添加：

```cpp
// 图索引：字符串图名 → captured_result_.graphs[] 中的索引
// 在 compile() 完成后一次性构建，run 阶段只读不写（线程安全）
std::unordered_map<std::string, int> name_to_graph_index_;

// 构建图索引（在 compile_impl 末尾调用）
void build_graph_index();
```

#### 4.2.3 Per-rank 线程上下文（栈变量，不存为成员）

这是**纯栈变量**，每个线程内部创建，不增加类成员：

```cpp
// GPU 模式 per-rank 线程函数内的局部变量
struct {  // 匿名，仅作文档说明
    int rank;
    int device_id;

    // 从 captured_result_.graphs[] 预解析的 cudaGraphExec_t
    cudaGraphExec_t xfer_a;
    cudaGraphExec_t xfer_b;
    cudaGraphExec_t fwd_bwd_deep_a;
    cudaGraphExec_t fwd_bwd_deep_b;
    cudaGraphExec_t first_layer_bwd;
    // ... 后续 Phase 2 增加的图（zero_grad, deep_comm, optimizer, ema 等）

    // 从 DeviceContext 预解析的 cudaStream_t
    cudaStream_t stream_trans;
    cudaStream_t stream_comp_1;
    cudaStream_t stream_comp_2;
    cudaStream_t stream_comp_3;
    cudaStream_t stream_update;
};
```

不需要结构体定义！直接用独立的栈变量即可：

```cpp
auto exec_xfer_a = static_cast<cudaGraphExec_t>(graph.native_exec(rank));
auto s_trans     = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
```

### 4.3 核心实现：run_train_epoch_gpu()

```cpp
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    const int K = num_gpus_;

    auto& registry = GlobalRegistry::instance();

    // ═══════════════════════════════════════════════════════════
    // 阶段 0：构建 per-rank 预解析上下文（主线程，循环外）
    // ═══════════════════════════════════════════════════════════

    auto resolve_exec = [&](const std::string& name, int rank) -> cudaGraphExec_t {
        auto it = name_to_graph_index_.find(name);
        if (it == name_to_graph_index_.end()) return nullptr;
        return captured_result_.graphs[it->second].native_exec(rank);
    };

    // per-rank 预解析好的图句柄（主线程一次性构建，线程只读访问）
    struct RankGraphs {
        cudaGraphExec_t xfer_a = nullptr;
        cudaGraphExec_t xfer_b = nullptr;
        cudaGraphExec_t fwd_bwd_a = nullptr;
        cudaGraphExec_t fwd_bwd_b = nullptr;
        cudaGraphExec_t first_bwd = nullptr;
    };
    std::vector<RankGraphs> rank_graphs(K);
    for (int r = 0; r < K; ++r) {
        rank_graphs[r].xfer_a    = resolve_exec("xfer_a", r);
        rank_graphs[r].xfer_b    = resolve_exec("xfer_b", r);
        rank_graphs[r].fwd_bwd_a = resolve_exec("fwd_bwd_deep_a", r);
        rank_graphs[r].fwd_bwd_b = resolve_exec("fwd_bwd_deep_b", r);
        rank_graphs[r].first_bwd = resolve_exec("first_layer_bwd", r);
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 1: Batch 0 单独传输（数据未就绪，不能与计算并行）
    // ═══════════════════════════════════════════════════════════

    TransferStation* ts = static_cast<TransferStation*>(
        registry.transfer_station_ptr(0));
    while (!ts->buffer_is_readable(0)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    // 所有 rank 并行发射 batch 0 传输
    std::vector<std::thread> threads(K);
    for (int r = 0; r < K; ++r) {
        threads[r] = std::thread([&, r]() {
            cudaSetDevice(backend_->contexts[r]->device_id());
            auto s_trans = static_cast<cudaStream_t>(
                backend_->contexts[r]->stream(StreamKind::TRANS));
            cudaGraphLaunch(rank_graphs[r].xfer_a, s_trans);
            cudaStreamSynchronize(s_trans);
        });
    }
    for (auto& t : threads) t.join();

    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    // 边界：只有 1 个 batch
    if (batches == 1) {
        for (int r = 0; r < K; ++r) {
            threads[r] = std::thread([&, r]() {
                cudaSetDevice(backend_->contexts[r]->device_id());
                auto& ctx = *backend_->contexts[r];
                auto s_c1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                auto s_c2 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                auto s_c3 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                cudaGraphLaunch(rank_graphs[r].fwd_bwd_a, s_c1);
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_c2);
                cudaStreamSynchronize(s_c3);

                if (!frozen) {
                    cudaGraphLaunch(rank_graphs[r].first_bwd, s_c1);
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                }
            });
        }
        for (auto& t : threads) t.join();
        std::visit([](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (!std::is_same_v<T, std::monostate>) s.step();
        }, sched_cfg_);
        return;
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 2: Batch 0 ~ batches-2 核心循环
    // ═══════════════════════════════════════════════════════════

    for (int batch = 0; batch < batches - 1; ++batch) {
        const bool from_a = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        while (!ts->buffer_is_readable(next_buf)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        // ── 每个 batch 展开 per-rank 线程 ──
        // 注：如果证明线程创建/join 开销显著（通常只在 batch 数很少时），
        // 可将线程创建提升到 batch 循环外，通过 barrier 同步。
        for (int r = 0; r < K; ++r) {
            threads[r] = std::thread([&, r, from_a]() {
                cudaSetDevice(backend_->contexts[r]->device_id());
                auto& ctx = *backend_->contexts[r];
                auto& rg = rank_graphs[r];

                auto s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                auto s_c1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                auto s_c2 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                auto s_c3 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                // 双图并行：传输 + 计算
                auto exec_xfer = from_a ? rg.xfer_b : rg.xfer_a;
                auto exec_comp = from_a ? rg.fwd_bwd_a : rg.fwd_bwd_b;

                cudaGraphLaunch(exec_xfer, s_trans);
                cudaGraphLaunch(exec_comp, s_c1);

                // 精确流同步（替代 cudaDeviceSynchronize）
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_c2);
                cudaStreamSynchronize(s_c3);
                cudaStreamSynchronize(s_trans);

                // first_layer_bwd
                if (!frozen) {
                    cudaGraphLaunch(rg.first_bwd, s_c1);
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                }
            });
        }
        for (auto& t : threads) t.join();

        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        std::visit([](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (!std::is_same_v<T, std::monostate>) s.step();
        }, sched_cfg_);
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 3: Last batch（数据已在上轮传输到位）
    // ═══════════════════════════════════════════════════════════

    const bool last_from_a = ((batches - 1) % 2 == 0);
    for (int r = 0; r < K; ++r) {
        threads[r] = std::thread([&, r]() {
            cudaSetDevice(backend_->contexts[r]->device_id());
            auto& ctx = *backend_->contexts[r];
            auto& rg = rank_graphs[r];
            auto s_c1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
            auto s_c2 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
            auto s_c3 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

            auto exec_last = last_from_a ? rg.fwd_bwd_a : rg.fwd_bwd_b;
            cudaGraphLaunch(exec_last, s_c1);
            cudaStreamSynchronize(s_c1);
            cudaStreamSynchronize(s_c2);
            cudaStreamSynchronize(s_c3);

            if (!frozen) {
                cudaGraphLaunch(rg.first_bwd, s_c1);
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_c2);
                cudaStreamSynchronize(s_c3);
            }
        });
    }
    for (auto& t : threads) t.join();

    std::visit([](auto&& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (!std::is_same_v<T, std::monostate>) s.step();
    }, sched_cfg_);
}
```

### 4.4 线程模型的选择

**选项 A：每个 batch 展开/join 线程（如上代码）**
- 优点：简单直接，与 BEST.md 伪代码一致
- 代价：`std::thread` 创建/join 开销（~10-50 μs）
- 适用：batch 数较多时（≥100），开销可忽略

**选项 B：batch 循环外创建线程，batch 循环内 barrier 同步**
- 优点：消除线程创建开销
- 代价：需要 barrier 或 atomic 同步，代码复杂度增加
- 适用：batch 数较少时

**推荐：先实现选项 A**。对于典型的 ResNet-50 训练（数百到数千 batch/epoch），线程创建开销占比极低。如果后续发现瓶颈，再优化为选项 B。

### 4.5 CPU 路径

```cpp
void DeepLearningTask::run_train_epoch_cpu() {
    // CPU 不需要 CUDA Graph，直接顺序执行即可
    // 如果 CPU 后端已有 graph executor，沿用现有逻辑
    // 否则走编译期生成的 cpu_ops_ 序列
    for (int batch = 0; batch < batches; ++batch) {
        for (int rank = 0; rank < num_gpus_; ++rank) {
            for (const auto& graph_name : {"xfer_a", "fwd_bwd_deep_a", "first_layer_bwd"}) {
                auto idx = name_to_graph_index_[graph_name];
                captured_result_.graphs[idx].launch(rank, nullptr);
            }
        }
    }
}
```

CPU 路径本身冷路径，不需要像 GPU 那样极致优化。

### 4.6 cudaStream_t 传值还是传指针？

BEST.md 提出了这个问题。答案：**传值**。

`cudaStream_t` 在 CUDA 中定义为 `typedef struct CUstream_st* cudaStream_t`，即它本身就是一个指针（8 字节）。传 `cudaStream_t` 值和传 `cudaStream_t*` 再解引用，效果完全一样，传值更简洁。

```cpp
cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
cudaGraphLaunch(exec, s);              // 直接传值
cudaStreamSynchronize(s);              // 直接传值
```

不需要 `cudaStream_t*` 包装。

---

## 五、实施计划

### 5.1 文件改动清单

| 文件 | 改动内容 |
|---|---|
| `include/renaissance/graph/captured_graph.h` | 新增 `native_exec(int rank)` 方法 |
| `include/renaissance/task/deep_learning_task.h` | 新增 `name_to_graph_index_` 成员、`build_graph_index()` 声明、`run_train_epoch_gpu()` / `run_train_epoch_cpu()` 声明 |
| `src/task/deep_learning_task.cpp` | 重写 `run_train_epoch()` 为 GPU/CPU 分支；实现 `run_train_epoch_gpu()`；实现 `run_train_epoch_cpu()`；实现 `build_graph_index()` |
| `src/task/task_base.cpp` | 在 `compile_impl()` 末尾（`compile_mark_compiled()` 之前），对 DeepLearningTask 调用 `build_graph_index()` |
| `src/backend/device_context.cpp` | 移除 `synchronize_all()` 和 `synchronize_stream()` 内部的冗余 `cudaSetDevice`（体系清洁，配合新架构） |

### 5.2 实施顺序

**第一步**（立即，低风险）：
1. `captured_graph.h` 添加 `native_exec()`
2. `deep_learning_task.h` 添加 `name_to_graph_index_` 和 `build_graph_index()`
3. `task_base.cpp` 在 compile 末尾调用 `build_graph_index()`
4. `device_context.cpp` 清理冗余 `cudaSetDevice`

**第二步**（核心，中等风险）：
5. 重写 `run_train_epoch()` → 分叉为 `run_train_epoch_gpu()` / `run_train_epoch_cpu()`
6. 实现 per-rank 线程 + 直接 cudaGraphLaunch + cudaStreamSynchronize

**第三步**（后续，面向完整训练流水线）：
7. 补充 `zero_grad`、`deep_comm`、`optimizer`、`ema_update` 等 Phase 2 图
8. 运行完整训练测试验证

### 5.3 不做的改动

- **不修改 `TaskBase::run()`** —— 它继续为 SimpleTask 和 dry run 服务
- **不添加任何 CudaGraph/CpuGraph 包装类** —— 直接操作 CUDA API
- **不修改 `CapturedGraph::launch()`** —— 它继续为 SimpleTask 路径服务

---

## 六、与 BEST.md 的对齐检查

| BEST.md 原则 | 本方案 | 说明 |
|---|---|---|
| cudaSetDevice 每个 epoch 一次 | ✅ | 每个 rank 线程入口 set 一次 |
| 只用 cudaStreamSynchronize | ✅ | 精确同步指定流 |
| 循环外预解析图指针 | ✅ | `rank_graphs[]` + 栈变量 |
| 循环内零查找 | ✅ | 直接 `cudaGraphLaunch(exec, stream)` |
| per-rank 线程 | ✅ | 每个 rank 独立线程 |
| 实例化在 compile 完成 | ✅ | 已经是，`native_exec()` 只读取 |
| 三计算流同步 | ✅ | `cudaStreamSynchronize(c1); c2; c3` |
| 双图并行 | ✅ | `cudaGraphLaunch(xfer, s_trans)` + `cudaGraphLaunch(comp, s_c1)` 在不同流上 |
| 不封装 CudaGraph | ✅ | 直接操作 `cudaGraphExec_t` |
| CPU/GPU 分支 | ✅ | `run()` 入口根据 `using_gpu()` 分叉 |
| 奥卡姆剃刀 | ✅ | 无多余抽象层 |

---

## 七、预期收益

### 7.1 消除的开销

| 开销 | 优化前（每 batch） | 优化后 |
|---|---|---|
| 字符串哈希查找 | 3 次 | 0 次 |
| 图集索引 | 3 次 | 0 次 |
| `cudaSetDevice` | 24 次（4 卡） | 4 次（每 rank 线程 1 次） |
| `cudaDeviceSynchronize` | 12 次（4 卡） | 0 次 |
| `cudaStreamSynchronize` | 0 次 | 按需精确同步 |

### 7.2 性能提升

- **Host 端 overhead 降低 ~90%**：从每 batch 约 200-300 μs 降至约 20-30 μs（主要是线程创建）
- **GPU 利用率提升**：`cudaStreamSynchronize` 只等待相关流，不阻塞其他流上的并行工作
- **为多流双图并行奠基**：不同流上的图可以真正并行执行（传输流和计算流互不阻塞）

---

## 八、附：子图注册方案

当前的 `on_prepare()` 只注册了 `"train"` 和 `"inference"` 两个 ComputationGraph。但 `run_train_epoch()` 需要 `"xfer_a"`、`"fwd_bwd_deep_a"` 等子图。这需要在 compile 流水线中将 `"train"` ComputationGraph 的各个 GraphId bucket 拆分为独立的 ComputationGraph 并注册。

具体方案（简要）：

```cpp
// 在 compile_impl() 中，pre_capture 之前：
void DeepLearningTask::decompose_train_graph() {
    auto& train_entry = named_graphs_["train"];
    ComputationGraph& full = train_entry.graph;

    struct SubGraphDef {
        const char* name;
        GraphId gid;
        StreamKind stream;
    };

    static const SubGraphDef defs[] = {
        {"xfer_a",          GraphId::TRANSFER_A,   StreamKind::TRANS},
        {"xfer_b",          GraphId::TRANSFER_B,   StreamKind::TRANS},
        {"fwd_bwd_deep_a",  GraphId::DEEP_FWD_BWD, StreamKind::COMP_1},
        {"fwd_bwd_deep_b",  GraphId::DEEP_FWD_BWD, StreamKind::COMP_1},
        {"first_layer_fwd_a",  GraphId::FIRST_FWD_A,  StreamKind::COMP_1},
        {"first_layer_fwd_b",  GraphId::FIRST_FWD_B,  StreamKind::COMP_1},
        {"first_layer_bwd",    GraphId::FIRST_BWD,     StreamKind::COMP_1},
        {"zero_grad",       GraphId::ZERO_GRAD,     StreamKind::UPDATE},
        {"deep_comm",       GraphId::DEEP_COMM,     StreamKind::UPDATE},
        {"cast_and_check",  GraphId::CAST_AND_CHECK,StreamKind::UPDATE},
        {"optimizer",       GraphId::OPTIMIZER,     StreamKind::UPDATE},
        {"ema_update",      GraphId::EMA_UPDATE,    StreamKind::UPDATE},
    };

    for (auto& def : defs) {
        auto sub = full.extract_subgraph(def.gid);
        if (!sub.empty()) {
            add_graph(def.name, std::move(sub), def.stream);
        }
    }

    named_graphs_.erase("train");  // 删除聚合图，避免重复
}
```

此子图拆分逻辑属于 compile 管线的完善，不在本次架构优化的核心范围内，但需要与 `build_graph_index()` 配合实现。

---

## 九、总结

本方案的核心思想只有一条：

> **把 DeepLearningTask 的 batch 循环变成一个尽可能"纯 GPU"的循环。Host 端在循环外把所有准备工作做完（预解析图指针、预解析流指针、set device），循环内只剩下 `cudaGraphLaunch` + `cudaStreamSynchronize`，没有任何查找、没有任何配置、没有任何分支。**

这样，DeepLearningTask 的训练性能就由 GPU 计算能力决定，而非 Host 端框架开销。