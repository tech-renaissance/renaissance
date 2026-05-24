# 仅测量 H2D 传输 —— 完整科学方案

## 1. 需求回顾

为 `DeepLearningTask` 添加两个方法，测试 **Preprocessor 与异步 H2D 传输对接后的每 epoch 耗时**：

| 方法 | 功能 |
|------|------|
| `compile_h2d_only()` | 只编译/捕获 XFER_A 和 XFER_B 两个 H2D 传输 cudaGraph，不捕获 FWD/BWD/OPT/推理图 |
| `run_h2d_only()` | 启动 Preprocessor::train()，交替 launch XFER_A / XFER_B，跑完一个 epoch，wall-clock 计时 |

必须兼容 `test_pw_ultimate.cpp` 的 PREPROCESSOR_SETTING 配置框架，支持 CIFAR-10 / ImageNet。

---

## 2. 架构基础：三个必须理解的事实

### 2.1 compile_impl 的完整流程

从 [task_base.cpp:248-263](file:///r:/renaissance/src/task/task_base.cpp#L248-L263) 可知，DeepLearningTask 的编译分为三个关键阶段（均在 compile_impl 内部顺序调用）：

```
TaskBase::compile()
  └─ TaskBase::compile_impl()
       ├─ compile_freeze_global()               # 锁定 GlobalRegistry
       ├─ compile_invoke_on_prepare()           # → on_prepare() 生成 MemoryPlan + ComputationGraph
       │    └─ Compiler::compile(plan, spec)    # train_cg_ 含 TRANSFER_A/B + DEEP 子图
       ├─ compile_alloc_hardware()              # DeviceContext + NCCL + StagingBufferPool
       └─ dynamic_cast<DeepLearningTask*>(this):
            ├─ build_graph_atlas()              # GraphId → slot 映射 → GraphAtlas
            ├─ pre_capture(atlas, ctxs)         # 捕获 cudaGraph（只对 Atlas 中的 slot）
            └─ build_exec_table()               # captured_result → gpu_exec_[rank][GraphSlot]
```

**关键结论**：`build_graph_atlas` 决定哪些 GraphId 进入 Atlas → `pre_capture` 只捕获 Atlas 中的图 → `build_exec_table` 从 captured_result 解析句柄。三阶段管线联动，唯一的介入点是 `build_graph_atlas`。

### 2.2 多 RANK 的 Staging Buffer 隔离

从测试日志和 [staging_buffer_pool.cpp](file:///r:/renaissance/src/core/staging_buffer_pool.cpp) 可知，每个 rank 拥有独立的 `StagingBufferPool` 块（pinned memory，不同基址、不同 NUMA node）：

```
RANK[0]: GPU=0, NUMA=6,  base=0x73eb54000000, size=588MB
RANK[1]: GPU=1, NUMA=6,  base=0x73eb2c000000, size=588MB
...
RANK[7]: GPU=7, NUMA=16, base=0x73eab4000000, size=588MB
```

每个 rank 的 cudaGraph 经过 capture 时，其 H2D memcpy 的源地址已硬编码为该 rank 对应的 StagingBufferPool 基址。因此 **rank-i 的 XFER 图从 rank-i 的 staging buffer 读取，不会跨 rank 访问**。

### 2.3 TransferStation 同步模型

`run_train_epoch_gpu()` 使用 engine 0 的 TransferStation 作为同步信号：

```cpp
TransferStation* ts = static_cast<TransferStation*>(
    registry.transfer_station_ptr(0));   // 只用 engine 0
```

当 rank 0 调用 `ts->set_buffer_writeable(buf, true)` 后，Preprocessor 的相应 worker 组可开始填充 engine 0 的下一批数据。但 engine 0 的 Preprocessor workers 会立即开始对 engine 0 的 staging buffer 进行写入，而其他 rank（如 rank 7）可能尚未完成当前 batch 的 DMA。

**虽然每个 rank 的 staging buffer 是独立的，且每个 engine 的 Preprocessor workers 也是独立的（engine 0 用 workers 0/8/16/...，engine 7 用 workers 7/15/...），但在 run_h2d_only 的简化流程中，我们无法依赖 ALLREDUCE 等隐式 barrier。因此 rank 间 barrier 是正确性必需的安全措施。**

---

## 3. 三个提案的对比分析

### 3.1 小伙伴 S 的方案：手动构建图 + TaskBase::run()

**做法**：绕过 Compiler，用 `TaskBase::alloc()` 手动分配 I_A_LABEL/I_A_DATA 等 DTensor，用 `TaskBase::add_graph()` 手动注册 h2d_xfer_a/h2d_xfer_b，再用 `TaskBase::run("h2d_xfer_a")` 逐次调用。

| 问题 | 严重性 | 说明 |
|------|--------|------|
| `TaskBase::alloc()` 是 protected | 🔴 致命 | DeepLearningTask 无法从外部调用 |
| 手动 MemoryPlan 与 Compiler 的 Region 布局不一致 | 🔴 致命 | Compiler 内部按特定顺序分配 Region，手动 alloc 可能打乱布局 |
| `TaskBase::run()` 是 SimpleTask 模式 | 🔴 致命 | 不支持多 GPU，不创建 rank 线程 |
| `is_finished()` 作为循环终止条件 | 🟡 风险 | `no_more_samples()` 的触发时机复杂，可能在最后一个 batch 后还有一次 buffer 翻转才触发 is_finished |
| 不经过 compile_impl | 🔴 致命 | 跳过了 compile_alloc_hardware()，缺少 DeviceContext、StagingBufferPool 等关键初始化 |

**结论**：❌ 不可行。方案基于对 TaskBase 内部 API 的错误假设。

### 3.2 小伙伴 K 的方案：h2d_only_ 标志 + 复用 Compiler

**做法**：`compile_h2d_only()` 设置 `h2d_only_ = true` 后调用 `compile()`，在 `build_graph_atlas()` 和 `build_exec_table()` 内部根据标志走简化分支。run 用 rank 线程 + barrier。

| 优点 | 说明 |
|------|------|
| 复用 Compiler | train_cg_ 天然包含 TRANSFER_A/B 子图，无需手写图构建 |
| 复用 compile_impl 完整管线 | DeviceContext、StagingBufferPool、MemoryPlan 全部自动分配 |
| barrier 保障多 rank 安全 | 所有 rank DMA 完成后才释放 buffer |

| 问题 | 严重性 | 说明 |
|------|--------|------|
| `init_all()` 在 h2d_only 下冗余 | 🟢 可接受 | 单 FC 层权重初始化仅微秒级，不影响 compile 耗时；且 `compile()` 结束后不可逆 |
| `compile()` 末尾分配 lr_pinned_ | 🟢 可接受 | 8 个 float 的 cudaMallocHost，微不足道 |

**结论**：✅ 架构正确，可作为主方案。

### 3.3 小伙伴 D 的方案：首 batch 特判 + 无 barrier

**做法**：与 K 的核心思路相同（h2d_only_ 标志），但 run 循环写成 "batch 0 预传输 + for batch=0..batches-2" 的两段式，且不设 rank 间 barrier。

| 问题 | 说明 |
|------|------|
| 首 batch 特判增加复杂度 | `batches == 1` 的 early return + 外层 batch 0 与内层循环分离，逻辑冗余。直接用 `for batch=0..batches-1` 自然覆盖所有情况 |
| 无 barrier | 如 2.3 节分析，多 rank 下缺少同步保护 |

**结论**：⚠️ 架构方向正确，但实现细节有缺陷。

---

## 4. 最终推荐方案（以 K 为基准，补充修正）

### 4.1 总体设计

```
compile_h2d_only():
  1. TR_CHECK phase == PLANNING
  2. h2d_only_ = true
  3. compile()  ──→ compile_impl()
       ├─ on_prepare() → Compiler 生成 train_cg_（含 TRANSFER_A/B 子图）
       ├─ compile_alloc_hardware() → DeviceContext + StagingBufferPool
       ├─ build_graph_atlas() → [h2d_only_ 分支] 只填入 TRANSFER_A, TRANSFER_B
       ├─ pre_capture() → 只捕获这 2 张图
       └─ build_exec_table() → [h2d_only_ 分支] 只解析 XFER_A, XFER_B
  4. 不调用 run()，仅 compile

run_h2d_only():
  1. 启动 prep.train() 线程
  2. 等待 TransferStation 就绪（engine 0）
  3. rank 线程循环：
     for batch = 0 .. batches-1:
       buf = batch % 2
       等待 buffer_is_readable(buf)
       launch XFER_A 或 XFER_B on TRANS stream
       cudaStreamSynchronize
       barrier(K)  ← 所有 rank DMA 安全完成后才释放
       if rank == 0: set_buffer_readable=false, set_buffer_writeable=true
  4. join rank 线程
  5. join prep 线程
  6. return elapsed_seconds
```

### 4.2 DeepLearningTask 代码修改

#### 4.2.1 头文件（deep_learning_task.h）

```cpp
// public:
    void compile_h2d_only();

    /**
     * @return wall-clock耗时（秒）
     */
    double run_h2d_only();

// private:
    bool h2d_only_ = false;
    void build_exec_table_h2d_only();
```

#### 4.2.2 compile_h2d_only()

```cpp
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "compile_h2d_only() must be called in PLANNING phase");
    h2d_only_ = true;
    compile();
    // h2d_only_ 保持 true，供后续 run_h2d_only 使用
}
```

#### 4.2.3 修改 build_graph_atlas()

在函数开头插入 `h2d_only_` 快速路径：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    // === H2D-Only 快速路径 ===
    if (h2d_only_) {
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = StreamKind::TRANS;
                sl.shape_id = kShapeInvariant;
            }
        }
        name_to_gid_.clear();
        return atlas;
    }
    // === 原逻辑不变 === ...
}
```

#### 4.2.4 修改 build_exec_table()

```cpp
void DeepLearningTask::build_exec_table() {
    if (h2d_only_) {
        build_exec_table_h2d_only();
        return;
    }
    // === 原逻辑不变 === ...
}
```

#### 4.2.5 新增 build_exec_table_h2d_only()

```cpp
void DeepLearningTask::build_exec_table_h2d_only() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;
    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    gpu_exec_.graphs.resize(K);

    auto resolve = [&](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
            return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);
        g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank);
        g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank);

        TR_CHECK(g[S(GraphSlot::XFER_A)] && g[S(GraphSlot::XFER_B)],
                 ValueError, "H2D graph slots are nullptr for rank " << rank);
    }
#endif
}
```

#### 4.2.6 run_h2d_only()

```cpp
double DeepLearningTask::run_h2d_only() {
#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;

    if (batches == 0) return 0.0;

    // 启动 Preprocessor
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    // 等待 TransferStation 就绪
    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    TR_CHECK(ts, RuntimeError, "TransferStation not ready within timeout");

    // rank 间 barrier
    std::mutex mtx;
    std::condition_variable cv;
    int barrier_count = 0;

    auto sync_barrier = [&](int total) {
        std::unique_lock<std::mutex> lk(mtx);
        ++barrier_count;
        if (barrier_count == total) {
            barrier_count = 0;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return barrier_count == 0; });
        }
    };

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    auto t0 = std::chrono::steady_clock::now();

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                DeviceContext& ctx = context(rank);
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    ctx.stream(StreamKind::TRANS));

                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    while (!ts->buffer_is_readable(buf_id))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);

                    sync_barrier(K);

                    if (rank == 0) {
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    auto t1 = std::chrono::steady_clock::now();

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    return std::chrono::duration<double>(t1 - t0).count();
#else
    return 0.0;
#endif
}
```

### 4.3 流程图解

```
Preprocessor (128 workers)          GPU Rank 0..7 (H2D-only loop)
        │                                    │
        ├─ fill buf0 (A区) ─────────────────►│ wait buf0 readable → XFER_A → sync
        │  all 128 workers done              │ barrier(K) → set_writeable(buf0)
        │                                    │
        ├─ fill buf1 (B区) ─────────────────►│ wait buf1 readable → XFER_B → sync
        │  all 128 workers done              │ barrier(K) → set_writeable(buf1)
        │                                    │
        ├─ fill buf0 (batch 2) ────────────►│ wait buf0 → XFER_A → sync
        │         ...                        │ barrier → set_writeable(buf0)
        │                                    │         ...
        │                                    │
        ├─ fill buf{N%2} (last) ───────────►│ wait → XFER_* → sync
        │                                    │ barrier → set_writeable
        ▼ prep.train() 自然结束              ▼ join, return elapsed
```

### 4.4 边界情况

| 场景 | 处理方式 |
|------|---------|
| `batches == 0` | `run_h2d_only()` 直接 return 0.0，不启动任何线程 |
| `batches == 1` | `for batch=0` 执行一次：buf=0 → XFER_A → sync → barrier → set_writeable |
| `batches` 为奇数 | last batch 落在 buf0：最后一次循环 `batch=batches-1`，buf_id=0，传 XFER_A |
| `batches` 为偶数 | last batch 落在 buf1：最后一次循环 `batch=batches-1`，buf_id=1，传 XFER_B |
| AMP (FP16) | Compiler 已在 on_prepare 时根据 `using_amp()` 生成 FP16 Region，cudaGraph capture 自动适配，无需额外处理 |
| CPU 模式 | `#ifdef TR_USE_CUDA` 保护，直接返回 0.0 |
| 单 GPU | K=1 时 barrier 无等待开销（立即通过），循环逻辑不变 |

### 4.5 为什么不需要 drain 循环

当前 `run_train_epoch_gpu` 有一个 drain 循环用于在 epoch 结束后消费 Preprocessor 可能额外生产的一批 buffer。但在 `run_h2d_only` 中不需要，原因：

- `for batch = 0 .. batches-1` 消费了恰好 `batches` 个 buffer
- 最后一个 batch 的 buffer 被消费后设为 writeable
- Preprocessor 感知到数据集遍历完毕（`steps_per_epoch` 已达成），`prep.train()` 自然返回
- Preprocessor 不会在 steps_per_epoch 之后继续填充新 batch 到 staging buffer

---

## 5. 测试样例：test_h2d_only_epoch.cpp

### 5.1 设计原则

1. **兼容 test_pw_ultimate.cpp 的命令行格式** — 复用其 PREPROCESSOR_SETTING 配置链
2. **支持 CIFAR-10 和 ImageNet** — 自动根据 dataset type 设定 num_classes 和 BluePrint
3. **最小 BluePrint** — 仅一个 FC 层，输出 = num_classes，不追求准确率
4. **输出精确计时和带宽**

### 5.2 关键参数映射

```
test_pw_ultimate 参数          → 我们的参数
─────────────────────────────────────────────
--dataset cifar10|imagenet     → dataset_type, num_classes
--path <dir>                   → dataset_path
--batch-size N                 → GLOBAL_SETTING.local_batch_size(N)
--resolution N                 → GLOBAL_SETTING.train_resolution(N)
--amp                          → GLOBAL_SETTING.amp(true)
--device GPU --gpu-ids "0..7"  → GLOBAL_SETTING.use_gpu(gpu_ids)
--loaders N                    → PREPROCESSOR_SETTING.load_workers(N)
--preproc N                    → PREPROCESSOR_SETTING.preprocess_workers(N)
--po-train1 <op>               → PREPROCESSOR_SETTING.train_transforms(...)
--po-val1 <op>                 → PREPROCESSOR_SETTING.val_transforms(...)
--seed N                       → GLOBAL_SETTING.auto_seed(N)
```

### 5.3 测试文件骨架

```cpp
#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main(int argc, char* argv[]) {
    // === 1. 命令行参数（与 test_pw_ultimate 风格一致） ===
    std::string dataset = "imagenet", path;
    int bs = 512, res = 224, n_load = 4, n_prep = 8, seed = 42;
    bool use_amp = false;
    std::string gpu_ids = "0";
    // ... 解析 ...

    // === 2. 全局配置 ===
    GLOBAL_SETTING.use_gpu(gpu_ids).local_batch_size(bs)
                  .train_resolution(res).val_resolution(res)
                  .auto_seed(seed);
    if (use_amp) GLOBAL_SETTING.amp(true);

    auto& reg = GlobalRegistry::instance();
    // num_classes 由 PREPROCESSOR_SETTING.commit() 自动设定
    // dataset_type 由 .dataset() 自动设定

    // === 3. Preprocessor 配置（完全复用 test_pw_ultimate 模式） ===
    NormMode norm = (dataset == "cifar10") ? NormMode::CIFAR : NormMode::MLPERF;
    PREPROCESSOR_SETTING
        .dataset(dataset, path)
        .load_workers(n_load)
        .preprocess_workers(n_prep)
        .normalization(norm)
        .train_transforms(/* CLI args → PO chain */)
        .val_transforms(/* CLI args → PO chain */)
        .partial_mode(true)
        .commit();
    // 此时 num_classes 已被 PREPROCESSOR_SETTING 自动设置

    int num_classes = reg.num_classes();

    // === 4. DeepLearningTask + 极简 BluePrint ===
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)))
        .loss(CrossEntropyLoss())
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .num_classes(num_classes)
        .total_epochs(1);

    task.compile_h2d_only();

    const int steps = Preprocessor::instance().steps_per_epoch();
    std::cout << "=== H2D-Only Epoch Test ===\n"
              << "Dataset:   " << dataset << "\n"
              << "BS:        " << bs << "\n"
              << "Res:       " << res << "\n"
              << "AMP:       " << (use_amp ? "on" : "off") << "\n"
              << "Ranks:     " << reg.world_size() << "\n"
              << "Steps:     " << steps << "\n";

    double elapsed = task.run_h2d_only();

    // === 5. 统计输出 ===
    size_t per_zone_bytes = 0;
    const auto& mp = task.memory_plan();
    for (const auto& d : mp.dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }
    size_t total_bytes = per_zone_bytes * steps * reg.world_size();
    double bw = (elapsed > 0.0)
        ? (total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0)
        : 0.0;

    std::cout << std::fixed << std::setprecision(3)
              << "Elapsed:    " << elapsed << " s\n"
              << "Bytes:      " << total_bytes / 1024.0 / 1024.0 << " MB\n"
              << "Bandwidth:  " << bw << " GB/s\n"
              << "=== DONE ===\n";

    return 0;
}
```

---

## 6. 与 test_pw_ultimate 的性能对比预期

| 数据集 | test_pw_ultimate | test_h2d_only_epoch | 差异原因 |
|--------|:---:|:---:|------|
| ImageNet (8 GPU) | ~39s/epoch | 预期 ~0.15-0.5s/epoch | test_pw_ultimate 包含 88 个图节点（FWD/BWD/OPT/ALLREDUCE），本测试仅 2 个 H2D memcpy 图 |
| CIFAR-10 (8 GPU, AMP) | ~0.12s/epoch | 预期 ~0.001-0.005s | CIFAR-10 仅 13 batches，数据量极小 |

本测试的价值在于 **验证 Preprocessor → TransferStation → cudaMemcpy 的异步管线是否畅通**，而非测量绝对 H2D 带宽（裸带宽测试应使用 perf_h2d_copy_a/b.cpp）。

---

## 7. 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `include/renaissance/task/deep_learning_task.h` | 修改 | 新增 `compile_h2d_only()`, `run_h2d_only()`, `h2d_only_`, `build_exec_table_h2d_only()` |
| `src/task/deep_learning_task.cpp` | 修改 | 实现上述方法；修改 `build_graph_atlas()` / `build_exec_table()` |
| `tests/correction/test_h2d_only_epoch.cpp` | **新建** | H2D-only epoch 计时测试 |
| `tests/correction/CMakeLists.txt` | 修改 | 添加 `test_h2d_only_epoch` 编译目标 |

---

## 8. 风险清单

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| `h2d_only_` 为 true 时误调 `run()` | 低 | 文档标注 compile_h2d_only 与 compile 互斥；run() 阶段若 `gpu_exec_` 缺少 deep 图会 TR_CHECK 失败而非静默崩溃 |
| init_all() 开销 | 低 | 单 FC 层约数 KB 权重，初始化 < 1ms，远低于 ImageNet compile 阶段的 Preprocessor 初始化（~4s） |
| barrier 引入微小延迟 | 极低 | 8 rank × 微秒级 mutex，与 DMA 毫秒级相比可忽略 |
| AMP 下 Region 类型正确性 | 极低 | Compiler 在 on_prepare 阶段已根据 using_amp() 生成正确 dtype 的 Region，cudaGraph capture 自动适配 |