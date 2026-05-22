# STX3：基于STEP*.md的综合测试计划（超详细版）

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **前置**: STEP.md（四步走思路）+ STEP1.md（详细实现）+ STEP2.md（技术细节）  
> **目标**: DeepLearningTask从dry run到真实训练的科学验证路径  
> **验证文档**: Z_FINAL_K.md | 源码: task_base.cpp, deep_learning_task.cpp, transfer_station.h

---

## 🎯 测试策略总览

### 核心原则

1. **渐进式验证**: 从简单到复杂，每步只验证一个核心维度
2. **问题快速定位**: 每步都有明确的成功标准和失败诊断
3. **可回滚性**: 所有修改都使用宏或条件编译，便于清理
4. **真实环境**: 在实际配置下测试，避免mock环境

### 四步走测试路径

```
Step 1 (Dry Run) → Step 2 (XFER_A) → Step 3 (XFER_AB) → Step 4 (Full Training)
   ↓                ↓               ↓                ↓
图调度验证        单次传输正确性    乒乓传输性能      完整训练收敛性
   ↓                ↓               ↓                ↓
2小时            2-3小时          2-3小时          4-6小时
```

---

## 📋 第一步：Compile Dry Run（图调度验证）

### 1.1 目标

验证编译链路完整性和图调度逻辑正确性，但不执行CUDA Graph。

### 1.2 测试配置

```cpp
// 测试参数调整
GLOBAL_SETTING
    .use_gpu("0")
    .manual_seed(42)
    .local_batch_size(5000)     // 大batch减少输出（MNIST 60000样本→12个batch）
    .train_resolution(28)
    .val_resolution(28)
    .amp(false);

DeepLearningTask task;
task.model(mlp)
    .loss(CrossEntropyLoss())
    .total_epochs(1)            // 只跑1个epoch
    .optimizer(SGD())
    .scheduler(ConstantLR());   // 简化scheduler
```

### 1.3 代码修改

#### A. 添加调试宏（临时）

```cpp
// deep_learning_task.cpp 文件顶部添加
#define STEP1_DRY_RUN_MODE  // 第一步专用调试宏

// 在run_train_epoch_gpu()中添加日志宏
#ifdef STEP1_DRY_RUN_MODE
    #define DRY_LAUNCH(g, s, name) \
        do { \
            if (g) { \
                LOG_INFO << "[DRY-RUN] Rank=" << rank \
                         << " Batch=" << batch \
                         << " Graph=" << name \
                         << " Stream=" << (s == s_c1 ? "COMP_1" : \
                                       s == s_c2 ? "COMP_2" : \
                                       s == s_up  ? "UPDATE" : "TRANS"); \
            } \
        } while(0)
#else
    #define DRY_LAUNCH(g, s, name) \
        do { if (g) cudaGraphLaunch(g, s); } while(0)
#endif
```

#### B. 注释Preprocessor线程

```cpp
// run_gpu()中暂时注释掉
// std::thread prep_thread([&]() { prep.train(); });
run_train_epoch_gpu();
// prep_thread.join();
```

#### C. 替换所有cudaGraphLaunch

```cpp
// run_train_epoch_gpu()中的关键位置
// Batch 0预传输
DRY_LAUNCH(gx_a, s_tr, "XFER_A");

// Phase 1
DRY_LAUNCH(gzg, s_up, "ZERO_GRAD");
DRY_LAUNCH(gf_a, s_c1, "FIRST_FWD_A");

// Phase 2
DRY_LAUNCH(gd_a, s_c1, "DEEP_FWD_BWD_A");
DRY_LAUNCH(gx_b, s_tr, "XFER_B");

// Phase 3
DRY_LAUNCH(gfb, s_c1, "FIRST_LAYER_BWD");
DRY_LAUNCH(gda, s_up, "DEEP_ALLREDUCE");

// Phase 4
LOG_INFO << "[DRY-RUN] LR=" << lr << " H2D";
DRY_LAUNCH(g_far, s_up, "FIRST_ALLREDUCE");
DRY_LAUNCH(gwu, s_up, "WEIGHT_UPDATE");
```

### 1.4 成功标准

#### 编译验证
- [ ] `task.compile()` 无错误完成
- [ ] `active_memory_plan_->total_bytes() > 0`
- [ ] `ArenaKeeper`分配成功
- [ ] 所有GraphSlot非nullptr

#### 调度验证
- [ ] 输出显示12个batch（60000/5000=12）
- [ ] 每个batch包含完整的4阶段调度
- [ ] Graph顺序符合Z_FINAL_K.md规范
- [ ] Stream分配正确（TRANS/COMP_1/UPDATE）

#### 日志验证
```
[DRY-RUN] Rank=0 Batch=0 Graph=XFER_A Stream=TRANS
[DRY-RUN] Rank=0 Batch=0 Graph=ZERO_GRAD Stream=UPDATE
[DRY-RUN] Rank=0 Batch=0 Graph=FIRST_FWD_A Stream=COMP_1
[DRY-RUN] Rank=0 Batch=0 Graph=DEEP_FWD_BWD_A Stream=COMP_1
[DRY-RUN] Rank=0 Batch=0 Graph=XFER_B Stream=TRANS
[DRY-RUN] Rank=0 Batch=0 Graph=FIRST_LAYER_BWD Stream=COMP_1
[DRY-RUN] Rank=0 Batch=0 Graph=DEEP_ALLREDUCE Stream=UPDATE
[DRY-RUN] Rank=0 Batch=0 LR=0.01 H2D
[DRY-RUN] Rank=0 Batch=0 Graph=FIRST_ALLREDUCE Stream=UPDATE
[DRY-RUN] Rank=0 Batch=0 Graph=WEIGHT_UPDATE Stream=UPDATE
```

### 1.5 失败排查

| 失败现象 | 可能原因 | 排查方法 |
|----------|----------|----------|
| compile()崩溃 | active_memory_plan_为空 | 检查on_prepare()是否设置active_memory_plan_ |
| GraphSlot为nullptr | build_exec_table()失败 | 检查captured_result_.atlas.index()返回值 |
| 无输出 | run_gpu()未调用 | 检查GlobalRegistry::using_gpu() |
| batch数量不对 | steps_per_epoch()计算错误 | 检查Preprocessor配置 |

---

## 🔄 第二步：Transfer A区验证（单次传输正确性）

### 2.1 目标

验证XFER_A图的数据传输正确性，确保TransferStation→GPU的数据完整性。

### 2.2 测试设计

#### 数据准备方案

```cpp
// 方案A：使用真实Preprocessor（推荐）
// 配置小batch便于验证
GLOBAL_SETTING.local_batch_size(16);

// 方案B：手动填充TransferStation
auto& reg = GlobalRegistry::instance();
auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));

// 填充buffer 0
int32_t* labels = reinterpret_cast<int32_t*>(ts->get_buffer_ptr(0));
for (int i = 0; i < 16; ++i) {
    labels[i] = i % 10;  // 0,1,2,...,9,0,1,2,3,4,5
}

float* data = reinterpret_cast<float*>(ts->get_image_data_ptr(0));
for (int i = 0; i < 16 * 28 * 28; ++i) {
    data[i] = (i % 997) * 0.001f;  // 可预测模式
}

ts->set_buffer_readable(0, true);
ts->set_buffer_writeable(0, false);
```

#### 验证接口实现

```cpp
// deep_learning_task.h添加临时调试接口
class DeepLearningTask {
public:
    // 第二步专用：保存TransferStation本地副本
    void step2_save_transfer_station_copy(int buffer_id, std::vector<uint8_t>& copy);
    
    // 第二步专用：验证传输正确性
    bool step2_verify_transfer_a(int rank);
};
```

#### 实现细节

```cpp
// deep_learning_task.cpp
void DeepLearningTask::step2_save_transfer_station_copy(int buffer_id, 
                                                         std::vector<uint8_t>& copy) {
    auto& reg = GlobalRegistry::instance();
    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    
    size_t bytes = ts->get_buffer_actual_transfer_bytes(buffer_id);
    uint8_t* ptr = ts->get_buffer_ptr(buffer_id);
    
    copy.resize(bytes);
    std::memcpy(copy.data(), ptr, bytes);
    
    LOG_INFO << "[STEP2] Saved TransferStation buffer " << buffer_id 
             << " copy: " << bytes << " bytes";
}

bool DeepLearningTask::step2_verify_transfer_a(int rank) {
    // 1. 获取I_A_DATA和I_A_LABEL的DTensor
    const DTensor* data_dt = nullptr;
    const DTensor* label_dt = nullptr;
    
    for (const auto& dt : active_memory_plan_->dtensors()) {
        if (dt.region == Region::I_A_DATA) data_dt = &dt;
        if (dt.region == Region::I_A_LABEL) label_dt = &dt;
    }
    
    if (!data_dt || !label_dt) {
        LOG_ERROR << "[STEP2] Cannot find I_A_DATA or I_A_LABEL";
        return false;
    }
    
    // 2. Fetch回GPU数据
    Tensor gpu_data = fetch_from_rank(*data_dt, rank);
    Tensor gpu_labels = fetch_from_rank(*label_dt, rank);
    
    // 3. 对比关键采样点
    const float* gpu_data_ptr = gpu_data.data<float>();
    const int* gpu_label_ptr = gpu_labels.data<int>();
    
    // 检查第一个样本的label
    if (gpu_label_ptr[0] != 0) {  // 我们设置了labels[0]=0
        LOG_ERROR << "[STEP2] Label mismatch: expected 0, got " << gpu_label_ptr[0];
        return false;
    }
    
    // 检查第一个像素
    float expected_pixel = 0.0f;  // data[0]=0*0.001=0.0
    if (std::abs(gpu_data_ptr[0] - expected_pixel) > 1e-6f) {
        LOG_ERROR << "[STEP2] Data mismatch at pixel 0: expected " << expected_pixel 
                 << ", got " << gpu_data_ptr[0];
        return false;
    }
    
    // 检查第16个样本的label
    if (gpu_label_ptr[15] != 5) {  // labels[15]=15%10=5
        LOG_ERROR << "[STEP2] Label[15] mismatch: expected 5, got " << gpu_label_ptr[15];
        return false;
    }
    
    LOG_INFO << "[STEP2] Transfer A verification PASSED";
    return true;
}
```

### 2.3 修改run_train_epoch_gpu()

```cpp
// 在run_train_epoch_gpu()的batch 0预传输后添加
#ifdef STEP2_VERIFY_XFER_A
    // 只运行batch 0，只验证XFER_A
    if (batch == 0 && rank == 0) {
        std::vector<uint8_t> host_copy;
        step2_save_transfer_station_copy(0, host_copy);
        
        // 执行XFER_A
        cudaGraphLaunch(gx_a, s_tr);
        sync_tr();
        
        // 验证
        if (!step2_verify_transfer_a(rank)) {
            LOG_ERROR << "[STEP2] Transfer A verification FAILED";
            return;
        }
        
        LOG_INFO << "[STEP2] Test complete, exiting...";
        return;  // 提前退出
    }
#else
    // 正常流程
    cudaGraphLaunch(gx_a, s_tr);
    sync_tr();
#endif
```

### 2.4 测试脚本

```cpp
// tests/correction/test_step2_xfer_a.cpp
int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(16)   // 小batch便于验证
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(1)
        .cpu_binding(false)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD())
        .scheduler(ConstantLR());

    task.compile();
    
    // 第二步：只验证XFER_A
    std::cout << "=== STEP2: Transfer A Verification ===" << std::endl;
    auto result = task.run();
    
    return 0;
}
```

### 2.5 成功标准

- [ ] TransferStation buffer填充成功
- [ ] XFER_A图执行无CUDA错误
- [ ] GPU数据与CPU数据一致性验证通过
- [ ] 关键采样点（label, pixel）检查通过
- [ ] 无内存泄漏

### 2.6 失败排查

| 失败现象 | 可能原因 | 排查方法 |
|----------|----------|----------|
| 找不到I_A_DATA | MemoryPlan中region命名错误 | 检查active_memory_plan_->dtensors() |
| 数据不匹配 | XFER_A图目标地址错误 | 检查captured_result_是否正确 |
| CUDA错误 | Graph捕获时地址错误 | 检查pre_capture()参数 |

---

## ⚡ 第三步：AB区乒乓传输+性能测试

### 3.1 目标

验证AB区乒乓切换逻辑和传输性能，确保满足带宽要求。

### 3.2 性能基准

| 配置 | 理论带宽 | 实际预期 |
|------|----------|----------|
| PCIe 3.0 x16 | 16 GB/s | ≥12 GB/s |
| PCIe 3.0 x8 | 8 GB/s | ≥6 GB/s |
| 单卡保守估计 | - | ≥2 GB/s |

### 3.3 测试实现

#### A. 性能测试代码

```cpp
// deep_learning_task.cpp添加性能测试
void DeepLearningTask::step3_benchmark_transfer(int num_iterations) {
    auto& reg = GlobalRegistry::instance();
    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    int batch_size = reg.get_local_batch_size();
    
    // 计算数据大小
    size_t bytes_per_xfer = ts->get_buffer_actual_transfer_bytes(0);
    size_t total_bytes = bytes_per_xfer * 2;  // A+B
    
    std::vector<double> latencies_us;
    latencies_us.reserve(num_iterations);
    
    // 预热
    for (int i = 0; i < 20; ++i) {
        cudaGraphLaunch(gx_a, s_trans); sync_tr();
        cudaGraphLaunch(gx_b, s_trans); sync_tr();
    }
    
    // 测试
    for (int i = 0; i < num_iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        cudaGraphLaunch(gx_a, s_trans);
        sync_tr();
        cudaGraphLaunch(gx_b, s_trans);
        sync_tr();
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies_us.push_back(us);
    }
    
    // 统计
    double mean_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / num_iterations;
    double bandwidth_gbps = (total_bytes / (mean_us / 1e6)) / (1024*1024*1024);
    
    LOG_INFO << "[STEP3] AB Transfer Performance:";
    LOG_INFO << "  Iterations: " << num_iterations;
    LOG_INFO << "  Mean latency: " << mean_us << " us";
    LOG_INFO << "  Total bytes: " << total_bytes << " (" << total_bytes/(1024*1024) << " MB)";
    LOG_INFO << "  Bandwidth: " << bandwidth_gbps << " GB/s";
    
    if (bandwidth_gbps < 2.0) {
        LOG_WARN << "[STEP3] Bandwidth below 2 GB/s threshold!";
    }
}
```

#### B. 乒乓逻辑验证

```cpp
void DeepLearningTask::step3_verify_ping_pong_logic(int num_iterations) {
    LOG_INFO << "[STEP3] Verifying ping-pong logic for " << num_iterations << " iterations";
    
    for (int i = 0; i < num_iterations; ++i) {
        bool from_a = (i % 2 == 0);
        int next_buf = from_a ? 1 : 0;
        
        LOG_INFO << "[STEP3] Iteration " << i 
                 << ": from_" << (from_a ? "A" : "B")
                 << ", next_buf=" << next_buf;
        
        // 检查buffer状态
        auto& reg = GlobalRegistry::instance();
        auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
        
        bool readable = ts->buffer_is_readable(next_buf);
        if (!readable) {
            LOG_ERROR << "[STEP3] Buffer " << next_buf << " not readable at iteration " << i;
            return;
        }
    }
    
    LOG_INFO << "[STEP3] Ping-pong logic verification PASSED";
}
```

### 3.4 测试脚本

```cpp
// tests/correction/test_step3_ping_pong.cpp
int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    // 不启动Preprocessor，手动填充AB区
    auto& reg = GlobalRegistry::instance();
    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    
    // 填充AB两个buffer
    for (int buf = 0; buf < 2; ++buf) {
        int32_t* labels = reinterpret_cast<int32_t*>(ts->get_buffer_ptr(buf));
        for (int i = 0; i < 128; ++i) labels[i] = i % 10;
        ts->set_buffer_readable(buf, true);
        ts->set_buffer_writeable(buf, false);
    }

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD())
        .scheduler(ConstantLR());

    task.compile();
    
    std::cout << "=== STEP3: AB Transfer Performance ===" << std::endl;
    
    // 3.1 乒乓逻辑验证
    task.step3_verify_ping_pong_logic(10);
    
    // 3.2 性能测试
    task.step3_benchmark_transfer(100);
    
    return 0;
}
```

### 3.5 成功标准

#### 乒乓逻辑
- [ ] AB区切换无错误
- [ ] buffer状态标志正确
- [ ] 无死锁或竞争条件

#### 性能指标
- [ ] 带宽 ≥2 GB/s（保守估计）
- [ ] 延迟稳定（stddev < mean * 20%）
- [ ] 无内存泄漏

### 3.6 失败排查

| 失败现象 | 可能原因 | 排查方法 |
|----------|----------|----------|
| 带宽过低 | 未使用锁页内存 | 检查cudaMallocHost分配 |
| 乒乓逻辑错误 | buffer状态管理错误 | 检查set_buffer_readable/writeable调用 |
| 延迟不稳定 | 系统后台进程干扰 | 多次运行取平均值 |

---

## 🚀 第四步：完整训练验证

### 4.1 目标

验证完整训练流程，包括收敛性、准确率和稳定性。

### 4.2 渐进式训练测试

#### 阶段1：1-epoch快速验证

```cpp
// tests/correction/test_step4_full_1epoch.cpp
DeepLearningTask task;
task.model(mlp)
    .loss(CrossEntropyLoss())
    .total_epochs(1)           // 先跑1个epoch
    .optimizer(SGD().momentum(0.9f))
    .scheduler(CosineAnnealingLR().base_lr(0.01f).step_by_epoch())
    .validate_every(1, 1);

task.compile();
auto result = task.run();

// 验证标准：
// - compile成功
// - run无崩溃
// - loss < 3.0（初始值通常2.3左右）
// - Val Top-1 > 0.80（1 epoch后应该达到80%+）
```

#### 阶段2：3-epoch中期验证

```cpp
// tests/correction/test_step4_full_3epoch.cpp
task.total_epochs(3);

// 验证标准：
// - loss单调下降
// - Val Top-1 > 0.90（3 epoch后应该达到90%+）
// - 无NaN或Inf
```

#### 阶段3：20-epoch完整验证

```cpp
// tests/ref/mnist_mlp_3.cpp（恢复原始配置）
task.total_epochs(20);

// 验证标准：
// - 最终Val Top-1 > 0.96
// - loss收敛到<0.5
// - 训练时间<30分钟（单卡）
// - 无内存泄漏
```

### 4.3 监控指标

```cpp
// 训练过程中的关键检查点
struct TrainingMonitor {
    float initial_loss;
    float epoch_1_loss;
    float epoch_3_loss;
    float final_loss;
    
    float initial_top1;
    float epoch_1_top1;
    float epoch_3_top1;
    float final_top1;
    
    bool check_loss_convergence() {
        // loss应该单调下降
        return (epoch_1_loss < initial_loss) &&
               (epoch_3_loss < epoch_1_loss) &&
               (final_loss < epoch_3_loss);
    }
    
    bool check_accuracy_progress() {
        // 准确率应该单调上升
        return (epoch_1_top1 > initial_top1 + 0.05f) &&
               (epoch_3_top1 > epoch_1_top1 + 0.05f) &&
               (final_top1 > epoch_3_top1 + 0.02f);
    }
    
    bool check_no_nan() {
        return std::isfinite(final_loss) && std::isfinite(final_top1);
    }
};
```

### 4.4 内存监控

```cpp
// 添加内存泄漏检测
#ifdef _WIN32
    #include <crtdbg.h>
    #define _CRTDBG_MAP_ALLOC
#endif

int main() {
#ifdef _WIN32
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
    
    // ... 训练代码 ...
    
#ifdef _WIN32
    _CrtDumpMemoryLeaks();
#endif
    return 0;
}
```

### 4.5 成功标准

#### 训练成功
- [ ] 20 epochs完整执行无崩溃
- [ ] Loss从2.3降到0.3左右
- [ ] 最终Val Top-1 > 96%
- [ ] 训练时间合理（<30分钟）

#### 稳定性
- [ ] 无NaN或Inf
- [ ] 无内存泄漏
- [ ] 无CUDA错误
- [ ] 梯度更新正常

### 4.6 失败排查

| 失败现象 | 可能原因 | 排查方法 |
|----------|----------|----------|
| Loss不下降 | 优化器或学习率配置错误 | 检查optimizer和scheduler配置 |
| 出现NaN | 梯度爆炸或数值不稳定 | 检查AMP设置、梯度裁剪 |
| 准确率过低 | 模型配置或数据问题 | 检查模型结构和数据预处理 |
| 内存泄漏 | cudaMallocHost未释放 | 检查析构函数 |

---

## 🛠️ 实施时间线

### 总体时间安排

| 步骤 | 实施时间 | 验证时间 | 问题修复 | 总计 |
|------|----------|----------|----------|------|
| Step 1 | 1小时 | 0.5小时 | 0.5小时 | 2小时 |
| Step 2 | 1.5小时 | 1小时 | 0.5小时 | 3小时 |
| Step 3 | 1.5小时 | 1小时 | 0.5小时 | 3小时 |
| Step 4 | 2小时 | 2小时 | 1小时 | 5小时 |
| **总计** | **6小时** | **4.5小时** | **2.5小时** | **15小时** |

### 里程碑节点

```
T+2小时  → Step 1完成：图调度验证通过
T+5小时  → Step 2完成：传输正确性验证通过  
T+8小时  → Step 3完成：传输性能验证通过
T+13小时 → Step 4完成：1-epoch训练验证通过
T+15小时 → 全部完成：20-epoch完整训练成功
```

---

## 🧪 测试脚本汇总

### 完整测试流程

```bash
# Step 1: Dry Run
cmake --build build --target test_step1_dry
./build/bin/tests/correction/test_step1_dry.exe

# Step 2: Transfer A
cmake --build build --target test_step2_xfer_a  
./build/bin/tests/correction/test_step2_xfer_a.exe

# Step 3: AB Performance
cmake --build build --target test_step3_ping_pong
./build/bin/tests/correction/test_step3_ping_pong.exe

# Step 4: Full Training (1 epoch)
cmake --build build --target test_step4_full_1epoch
./build/bin/tests/correction/test_step4_full_1epoch.exe

# Step 4: Full Training (3 epochs)
cmake --build build --target test_step4_full_3epoch
./build/bin/tests/correction/test_step4_full_3epoch.exe

# Step 4: Full Training (20 epochs)
cmake --build build --target mnist_mlp_3
./build/bin/tests/ref/mnist_mlp_3.exe
```

### CMakeLists.txt配置

```cmake
# Step 1: Dry Run Test
add_executable(test_step1_dry test_step1_dry.cpp)
target_link_libraries(test_step1_dry PRIVATE renaissance)
target_compile_definitions(test_step1_dry PRIVATE 
    TR_LOG_LEVEL=1 
    STEP1_DRY_RUN_MODE=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_step1_dry)
endif()

# Step 2: Transfer A Verification
add_executable(test_step2_xfer_a test_step2_xfer_a.cpp) 
target_link_libraries(test_step2_xfer_a PRIVATE renaissance)
target_compile_definitions(test_step2_xfer_a PRIVATE 
    TR_LOG_LEVEL=1 
    STEP2_VERIFY_XFER_A=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_step2_xfer_a)
endif()

# Step 3: AB Performance Test  
add_executable(test_step3_ping_pong test_step3_ping_pong.cpp)
target_link_libraries(test_step3_ping_pong PRIVATE renaissance)
target_compile_definitions(test_step3_ping_pong PRIVATE 
    TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_step3_ping_pong)
endif()

# Step 4: Full Training Tests
add_executable(test_step4_full_1epoch test_step4_full_1epoch.cpp)
target_link_libraries(test_step4_full_1epoch PRIVATE renaissance)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_step4_full_1epoch)
endif()
```

---

## 🎯 总结

### 核心优势

1. **科学渐进**: 从简单到复杂，问题快速定位
2. **完整覆盖**: 涵盖编译、传输、性能、训练全流程
3. **可回滚性**: 所有修改使用条件编译，便于清理
4. **可验证性**: 每步都有明确的量化成功标准

### 关键成功因素

- **严格顺序**: 必须按Step 1→4顺序执行，不能跳跃
- **充分验证**: 每步都要完全通过才能进入下一步
- **详细日志**: 保留完整的调试输出用于问题诊断
- **性能监控**: 不仅验证功能正确性，也关注性能指标

### 预期成果

通过这个15小时的测试计划，我们将：
- 验证DeepLearningTask的完整训练流程
- 确保各组件集成正确性
- 建立性能基准和回归测试标准
- 为MLP从dry run到真实训练提供可靠路径

---

**技术觉醒团队 · 测试组**  
*版本: v1.0 | 基于: STEP.md + STEP1.md + STEP2.md | 2026-05-22*
