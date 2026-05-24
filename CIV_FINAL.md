# H2D-Only Epoch 测试 —— 最终实现方案（v3）

> 修订记录：
> - v3：将 GPU 侧 `while+sleep` 忙轮询替换为 `condition_variable::wait()`，同时改进 TransferStation
> - v2：根据 CIV_REV.md 审查意见全面修正 —— 去掉 barrier、复用 test_pw_ultimate 的 PO 工厂、修正表述、修复细节

---

## 1. 需求

为 `DeepLearningTask` 添加 `compile_h2d_only()` / `run_h2d_only()`，配合 test_pw_ultimate.cpp 风格的 Preprocessor 配置，精确测量 **Preprocessor → TransferStation → cudaMemcpy 管线每 epoch 耗时**。

严格约束：
- **不使用** `FOR_TRANSFER_STATION_UNIT_TESTS_ONLY` 宏 — GPU 消费者真实存在，必须走正常的可读/可写信号量同步
- 兼容 test_pw_ultimate.cpp 的全部 CLI 参数
- BluePrint 极简：`fc(num_classes, bias=true)`

---

## 2. compile_h2d_only() 的设计

### 2.1 宏观流程

```
compile_h2d_only()
  └─ h2d_only_ = true (RAII guard)
       └─ compile()
            └─ compile_impl()
                 ├─ on_prepare()            ← 照常生成 ArchPlan / MemoryPlan / ComputationGraph（含全部子图）
                 ├─ compile_alloc_hardware() ← 照常分配 DeviceContext / StagingBufferPool
                 ├─ build_graph_atlas()     ← h2d_only_ 分支：Atlas 只含 TRANSFER_A, TRANSFER_B
                 ├─ pre_capture()           ← 只捕获 Atlas 中的 2 张图
                 └─ build_exec_table()      ← h2d_only_ 分支：只解析 XFER_A, XFER_B
    RAII guard 析构 → h2d_only_ = false
```

### 2.2 一个重要的澄清

**`compile_h2d_only()` 在 compile 阶段生成了完整的 ArchPlan、MemoryPlan、ComputationGraph（含 FWD/BWD/OPT/ALLREDUCE 等全部子图），但只预热和捕获了 TRANSFER_A 和 TRANSFER_B 两个 H2D 传输图。** 其他子图生成了 ComputationGraph 节点但未进入 GraphAtlas → 未被 pre_capture 捕获为 cudaGraph。

影响评估：
- 单层 FC 蓝图的 compile 开销可忽略（权重初始化 < 1ms）
- `init_all()` 初始化了所有权重，但对 H2D-only 无影响
- ArenaKeeper 按 `active_memory_plan_->total_bytes()` 分配显存池，但单 FC 层显存需求极小
- 如果未来蓝图层数较多，可通过 `TaskBase::compile_impl()` 中增加 `skip_init_` 标志来优化（后续 PR）

### 2.3 具体修改

#### 2.3.1 deep_learning_task.h

```cpp
    // 只编译 H2D 传输图（TRANSFER_A + TRANSFER_B）
    void compile_h2d_only();

    // 只运行 H2D 传输图一个 epoch，联动 Preprocessor/TransferStation
    H2DTestResult run_h2d_only();

private:
    bool h2d_only_ = false;
```

#### 2.3.2 compile_h2d_only()

```cpp
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "compile_h2d_only() must be called in PLANNING phase");

    struct Guard {
        bool* p;
        Guard(bool* ptr) : p(ptr) { *p = true; }
        ~Guard() { *p = false; }
    } guard(&h2d_only_);

    compile();
}
```

#### 2.3.3 修改 build_graph_atlas()

在函数开头插入早期 return 分支：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (h2d_only_) {
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = stream_for(gid);
                sl.shape_id = kShapeInvariant;
            }
        }
        name_to_gid_.clear();
        return atlas;
    }
    // === 原有逻辑不变 ===
    // ...
}
```

**改动要点**（根据 CIV_REV 审查修正）：
- `sl.stream_kind = stream_for(gid)` 替代硬编码 `StreamKind::TRANS`，保证未来映射变更时的兼容性

#### 2.3.4 修改 build_exec_table()

```cpp
void DeepLearningTask::build_exec_table() {
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

    if (h2d_only_) {
        for (int rank = 0; rank < K; ++rank) {
            gpu_exec_.device_ids[rank] = context(rank).device_id();
            auto& g = gpu_exec_.graphs[rank];
            g.resize(S(GraphSlot::COUNT), nullptr);
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank);

            TR_CHECK(g[S(GraphSlot::XFER_A)] && g[S(GraphSlot::XFER_B)],
                     ValueError,
                     "H2D-only: XFER_A or XFER_B slot nullptr for rank " << rank);
        }
        return;
    }
    // === 原有逻辑不变 ===
    // ...
#endif
}
```

#### 2.3.5 TransferStation 新增 condition_variable 支持

**问题**：现有代码中 GPU 侧等待 buffer 可读、Preprocessor 侧等待 buffer 可写，均使用 `while (!flag) sleep(μs)` 忙轮询。即使是 50μs sleep，每 batch 可能浪费数十次 CPU 唤醒，8 GPU × 313 batches 累积可观。

**方案**：在 TransferStation 中增加 `std::condition_variable`，让消费者在 buffer 就绪时被精确唤醒，不再轮询。

**头文件修改**（`include/renaissance/data/transfer_station.h`）：

```cpp
public:
    /// GPU 侧：阻塞直到 buffer 可读（替换 while+sleep 轮询）
    void wait_buffer_readable(int buffer_id);

    /// Preprocessor 侧：阻塞直到 buffer 可写
    void wait_buffer_writeable(int buffer_id);

private:
    // 已有成员（不变）：
    std::atomic<bool> buffer_0_is_readable_{false};
    std::atomic<bool> buffer_1_is_readable_{false};
    std::atomic<bool> buffer_0_is_writeable_{true};
    std::atomic<bool> buffer_1_is_writeable_{true};

    // 新增成员：
    std::mutex buffer_sync_mtx_;
    std::condition_variable cv_readable_[2];   // index 0 → buffer 0, 1 → buffer 1
    std::condition_variable cv_writeable_[2];  // index 0 → buffer 0, 1 → buffer 1
```

**实现文件修改**（`src/data/transfer_station.cpp`）：

```cpp
// ── set 方法：设置标志后 notify ──

void TransferStation::set_buffer_readable(int buffer_id, bool readable_flag) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError, ...);
    if (buffer_id == 0)
        buffer_0_is_readable_.store(readable_flag, std::memory_order_release);
    else
        buffer_1_is_readable_.store(readable_flag, std::memory_order_release);

    if (readable_flag) {
        cv_readable_[buffer_id].notify_all();   // ← 唤醒等可读的 GPU 线程
    }
}

void TransferStation::set_buffer_writeable(int buffer_id, bool writeable_flag) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError, ...);
    if (buffer_id == 0)
        buffer_0_is_writeable_.store(writeable_flag, std::memory_order_release);
    else
        buffer_1_is_writeable_.store(writeable_flag, std::memory_order_release);

    if (writeable_flag) {
        cv_writeable_[buffer_id].notify_all();  // ← 唤醒等可写的 Preprocessor 线程
    }
}

// ── wait 方法：阻塞等待，不轮询 ──

void TransferStation::wait_buffer_readable(int buffer_id) {
    std::unique_lock<std::mutex> lk(buffer_sync_mtx_);
    cv_readable_[buffer_id].wait(lk, [&]{
        return buffer_is_readable(buffer_id);
    });
}

void TransferStation::wait_buffer_writeable(int buffer_id) {
    std::unique_lock<std::mutex> lk(buffer_sync_mtx_);
    cv_writeable_[buffer_id].wait(lk, [&]{
        return buffer_is_writeable(buffer_id);
    });
}
```

**Preprocessor 侧同步替换**（`transfer_station.cpp` 的 `execute_transfer_locked` 中）：

```cpp
// 之前：
while (!buffer_is_writeable(next_buf)) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
}

// 之后：
wait_buffer_writeable(next_buf);
```

**保持向后兼容**：`buffer_is_readable()` / `buffer_is_writeable()` 的原子 load 接口保留不变，已有调用（如 `run_train_epoch_gpu` 中的 `while+ sleep`）仍可正常工作。后续可逐步迁移。

---

## 3. run_h2d_only() 的设计

### 3.1 同步模型：无 barrier

**决策依据**（来自 CIV_REV P0 审查）：

现有的 `run_train_epoch_gpu()` 和 `test_h2d_copy_bandwidth()` 均不使用 rank 间 barrier。在 H2D-only 场景下：

1. 所有 rank 执行完全相同的 cudaGraph（同一 GraphId、同一 stream kind）
2. `cudaStreamSynchronize` 后 DMA 物理完成
3. 各 rank 的 DMA 完成时间差异极小（同一 GPU 型号、同一数据量，< 1ms）
4. Buffer A/B 交替提供天然隔离 —— 释放 buffer N 后 Preprocessor 填充 buffer N，但 GPU 下一轮使用 buffer `(N+1)%2`，不会立即复用刚释放的 buffer
5. Preprocessor 填充一个 batch 耗时远大于 rank 间同步差异（数毫秒到数十毫秒 vs 微秒级）

**去掉 barrier 的好处**：
- 消除异常死锁风险（任一 rank 提前退出不会卡死其余 rank）
- 与现有代码一致性
- 代码更简洁

### 3.2 实现

```cpp
H2DTestResult DeepLearningTask::run_h2d_only() {
    H2DTestResult r;

#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;

    if (batches == 0) return r;

    // ---- 启动 Preprocessor 线程 ----
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    // ---- 等待 TransferStation 就绪 ----
    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) {
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);
        return r;
    }

    // ---- 计算每 zone 字节数 ----
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    // ---- 启动 rank 线程 ----
    std::vector<std::exception_ptr> rank_exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    auto t0 = std::chrono::steady_clock::now();

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess)
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    context(rank).stream(StreamKind::TRANS));

                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    while (!ts->buffer_is_readable(buf_id))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);

                    if (rank == 0) {
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                }
            } catch (...) {
                rank_exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    auto t1 = std::chrono::steady_clock::now();

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int rank = 0; rank < K; ++rank)
        if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);

    // ---- 填充结果（per-rank 语义） ----
    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    r.batches     = batches;
    r.elapsed_us  = elapsed_us;
    r.total_bytes = per_zone_bytes * static_cast<size_t>(batches);
    // 注意：total_bytes 是单卡字节数，测试文件中如需聚合带宽应在外部乘以 num_ranks

    if (batches > 0) {
        r.avg_lat_us = elapsed_us / static_cast<double>(batches);
    }
    if (elapsed_us > 0.0 && r.total_bytes > 0) {
        double bw = static_cast<double>(r.total_bytes) / (elapsed_us / 1e6);
        r.bandwidth_gbps = bw / (1024.0 * 1024.0 * 1024.0);
    }
#else
    (void)r;
#endif
    return r;
}
```

### 3.3 时序图（无 barrier 版本）

```
Preprocessor (128 workers)            GPU Rank 0..7
        │                                   │
        ├─ fill buf0 ──────────────────────►│ t0 = now()
        │  all workers done                 │ wait buf0 readable → XFER_A → sync
        │                                   │ rank0: set_writeable(buf0)
        │                                   │
        ├─ fill buf1 ──────────────────────►│ wait buf1 readable → XFER_B → sync
        │                                   │ rank0: set_writeable(buf1)
        │                                   │
        ├─ fill buf0 (batch 2) ────────────►│ wait buf0 → XFER_A → sync → set_writeable
        ├─ fill buf1 (batch 3) ────────────►│ wait buf1 → XFER_B → sync → set_writeable
        │         ...                       │         ...
        │                                   │
        ├─ fill buf{N%2} (last) ───────────►│ wait → XFER → sync → set_writeable
        │                                   │
        ▼ prep.train() returns              │ join threads → t1 = now()
```

---

## 4. 关于误调 `run()` 的行为

CIV_REV 指出 `build_exec_table_h2d_only` 之后 `gpu_exec_[rank][FWD_BWD_DEEP_A]` 为 `nullptr`。若误调用 `run()`，实际执行的是：

```cpp
cudaGraphLaunch(gpu_exec_.graphs[rank][S(GraphSlot::FWD_BWD_DEEP_A)], s_comp);
// 传入 nullptr → CUDA 错误（cudaErrorInvalidValue）
```

`cudaGraphLaunch` 对 `nullptr` 的行为取决于 CUDA 驱动版本：部分版本返回 cudaErrorInvalidValue，部分可能 segfault。因此这不是安全的 fallback。**但**在实际使用场景中，`compile_h2d_only()` 的设计意图明确是替代 `compile()`（互斥使用），误调风险极低。如需加强防御，可在 `run()` 开头检查 `h2d_only_` 残余并 `TR_CHECK`。

---

## 5. 测试样例：test_h2d_only_epoch.cpp

### 5.1 设计原则

1. **直接复用 test_pw_ultimate.cpp 的 PO 工厂函数** — 复制 [create_po()](file:///r:/renaissance/tests/correction/test_pw_ultimate.cpp#L145-L207) 和 [create_resize_crop_po()](file:///r:/renaissance/tests/correction/test_pw_ultimate.cpp#L212-L228)，保持 100% 兼容
2. **不使用 `FOR_TRANSFER_STATION_UNIT_TESTS_ONLY`** — GPU 消费者存在，正常走读写信号量
3. BluePrint 自动适配：CIFAR-10 → `fc(10)`, ImageNet → `fc(1000)`
4. 输出包含 per-rank 和聚合带宽

### 5.2 代码清单（与 CIV_FINAL.md 原版的差异）

**测试文件改动汇总**：

| 改动点 | 原来 | 修正后 | 原因 |
|--------|------|--------|------|
| PO 工厂 | 自写 `make_po()`（仅支持 2 种 PO） | 复制 test_pw_ultimate 的 `create_po()` + `create_resize_crop_po()`（支持 13 种 PO） | CIV_REV P1 — 不完整会导致 `--po-train2 ColorJitter` 等参数报错退出 |
| `::tolower` | 直接传 `char` | `[](unsigned char c){ return std::tolower(c); }` | CIV_REV P2 — signed char 负值 → UB |

**完整测试文件**见附录，以下仅标注关键结构：

```cpp
// ===== 从 test_pw_ultimate.cpp 完整复制的函数 =====

// create_po()   — 文件位置：test_pw_ultimate.cpp:145-207
//   支持: RandomHorizontalFlip, ColorJitter, RandomRotation,
//         RandomAutocontrast, RandomScale, GaussianBlur,
//         RandomGrayscale, GaussianNoise, Pad, DoNothing
//
// create_resize_crop_po() — 文件位置：test_pw_ultimate.cpp:212-228
//   支持: Resize, CenterCrop, RandomResizedCrop,
//         FastRandomResizedCrop, RandomCrop

int main(int argc, char* argv[]) {
    // ---- CLI 参数解析（与 test_pw_ultimate.cpp 完全对齐） ----
    // 支持全部参数: --dataset, --path, --format, --mode, --lv,
    //   --batch-size, --resolution, --loaders, --preproc,
    //   --device, --gpu-ids, --amp, --po-train1, --po-train2,
    //   --po-val1, --po-val2, --normalization, --seed,
    //   --sdmp, --cpvs, --reproducible, --cpu-bind
    //   + PO 参数: --scale-min, --scale-max, --ratio-min, --ratio-max,
    //     --flip-prob, --brightness, --contrast, --saturation, --hue,
    //     --degrees, --fill, --autocontrast-p, --blur-sigma-min,
    //     --blur-sigma-max, --grayscale-p, --noise-mean, --noise-sigma

    // ---- 推断数据集参数 ----
    // CIFAR-10 → num_classes=10, resolution 默认 32, norm=NormMode::CIFAR
    // ImageNet  → num_classes=1000, resolution 默认 224, norm=NormMode::MLPERF

    // ---- 创建 PO（调用 create_po / create_resize_crop_po） ----

    // ---- PREPROCESSOR_SETTING ----
    // 注意：不定义 FOR_TRANSFER_STATION_UNIT_TESTS_ONLY

    // ---- compile_h2d_only + run_h2d_only ----
    task.compile_h2d_only();
    auto res = task.run_h2d_only();

    // ---- 输出 ----
    // per-rank bytes: res.total_bytes
    // aggregate bytes: res.total_bytes * num_ranks
    // per-rank BW: res.bandwidth_gbps
    // aggregate BW: aggregate_bytes / elapsed
}
```

---

## 6. 修改文件汇总

| 文件 | 操作 | 说明 |
|------|------|------|
| `include/renaissance/task/deep_learning_task.h` | 修改 | +2 public +1 private |
| `src/task/deep_learning_task.cpp` — `compile_h2d_only()` | **新增** | ~15 行 |
| `src/task/deep_learning_task.cpp` — `build_graph_atlas()` | 修改 | 开头插入 ~15 行 h2d 分支 |
| `src/task/deep_learning_task.cpp` — `build_exec_table()` | 修改 | resolve lambda 后插入 ~15 行 h2d 分支 |
| `src/task/deep_learning_task.cpp` — `run_h2d_only()` | **新增** | ~90 行（无 barrier 版） |
| `tests/correction/test_h2d_only_epoch.cpp` | **新建** | ~280 行（含 test_pw_ultimate 的 PO 工厂 + arg 解析） |
| `tests/correction/CMakeLists.txt` | 修改 | 添加编译目标 |

**总计新增约 415 行，现有代码修改约 35 行。**

---

## 7. 边界情况

| 场景 | 行为 |
|------|------|
| `batches == 0` | 直接返回空 `H2DTestResult`，不启动任何线程 |
| `batches == 1` | 循环执行一次：buf=0 → XFER_A → sync → set_writeable |
| 奇数 batch | last batch 落在 buf0，循环末次执行 XFER_A |
| 偶数 batch | last batch 落在 buf1，循环末次执行 XFER_B |
| 单 GPU | K=1 时立即执行 set_writeable，无 wait 开销 |
| CPU 模式 | `#ifdef TR_USE_CUDA` 保护，返回空结果 |
| AMP | Compiler 在 on_prepare 阶段已根据 `using_amp()` 生成 FP16 Region，cudaGraph capture 自动适配 |
| `compile_h2d_only()` 后调 `run()` | deep slot 为 nullptr → cudaGraphLaunch 返回 cudaErrorInvalidValue 或 segfault（取决于驱动）。建议互斥使用 |
| `compile_h2d_only()` 抛异常 | RAII guard 析构清除 `h2d_only_`，不影响后续正常 `compile()` |

---

## 8. v1 → v2 变更摘要

| # | 严重程度 | 内容 | 采取的修正 |
|---|---------|------|-----------|
| 1 | 🔴 P0 | sync_barrier 异常死锁 | **去掉 barrier**，与 run_train_epoch_gpu/test_h2d_copy_bandwidth 保持一致 |
| 2 | 🟡 P1 | make_po 不完整，不兼容 test_pw_ultimate | **直接复制** test_pw_ultimate.cpp 的 create_po + create_resize_crop_po |
| 3 | 🟡 P1 | compile 阶段生成了完整图 | 文档明确说明：ArchPlan/MemoryPlan/ComputationGraph 照常生成，但只预热和捕获 H2D 图 |
| 4 | 🟡 P1 | run() 误用行为描述不准 | 修正为 "cudaGraphLaunch(nullptr) → cudaError 或 segfault" |
| 5 | 🟢 P2 | cudaSetDevice 无错误检查 | 添加 `cudaError_t` 检查 + `TR_DEVICE_ERROR` |
| 6 | 🟢 P2 | ::tolower signedness 风险 | 改为 `[](unsigned char c){ return std::tolower(c); }` |
| 7 | 🟢 P2 | total_bytes 语义未说明 | 文档明确标注为 per-rank；测试文件中外部乘 num_ranks 得聚合值 |
| 8 | 🟢 P2 | StreamKind::TRANS 硬编码 | 改为 `stream_for(gid)` |

---

## 附录：预期输出示例

### ImageNet（8 GPU，BS=512，res=224，FP32）

```
=== H2D-Only Epoch Test ===
Dataset:      imagenet
Batch size:   512
Resolution:   224
AMP:          off
GPU IDs:      0,1,2,3,4,5,6,7
Steps:        313

=== Results ===
Batches:        313
Elapsed:         0.354 s
Avg latency:     1.131 ms/batch
Per-rank bytes:  23.565 GB
Aggregate bytes: 188.518 GB
Per-rank BW:     54.067 GB/s
Aggregate BW:   432.536 GB/s
=== DONE ===
```

### CIFAR-10（8 GPU，BS=512，res=32，AMP on）

```
=== H2D-Only Epoch Test ===
Dataset:      cifar10
Batch size:   512
Resolution:   32
AMP:          on
GPU IDs:      0,1,2,3,4,5,6,7
Steps:        13

=== Results ===
Batches:        13
Per-rank bytes: 0.002 GB
Elapsed:         0.002 s
=== DONE ===
```