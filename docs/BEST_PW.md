# PreprocessWorker (PW) 核心设计文档

**版本**: V3.14.1
**日期**: 2026-02-21
**作者**: 技术觉醒团队

---

## 1. 概述

PreprocessWorker (PW) 是 renAIssance 框架中负责单个样本预处理的独立工作单元。每个PW拥有独立的Workshop内存空间，通过静态样本分配策略实现可复现的随机性和零竞争的并行处理。

### 1.1 核心设计原则

1. **静态分配**: Worker `i` 的第 `k` 次调用处理样本 `(i + k×M)`
2. **零竞争**: 每个PW有独立的Workshop，无共享内存访问
3. **可复现性**: 基于种子的两层衍生机制，确保完全确定的结果
4. **乒乓优化**: AB区交替存储，支持多PO链式执行
5. **路径优化**: RandomHorizontalFlip特殊处理，避免不必要的内存复制

---

## 2. Workshop内存布局

### 2.1 内存区域划分

```
Workshop（4KB页对齐）
├── D区（Decode Region）：解码缓冲区，64字节对齐
├── A区（Ping Region）：乒乓缓冲区A，64字节对齐
├── B区（Pong Region）：乒乓缓冲区B，64字节对齐
├── T区（Transform Region）：可选的变换缓冲区，64字节对齐
├── S1区（SDMP Cache 1）：SDMP训练缓存1，4KB页对齐
├── S2区（SDMP Cache 2）：SDMP训练缓存2，4KB页对齐
├── ...
└── C区（CPVS Cache）：验证集缓存，64字节对齐
```

### 2.2 内存布局类型

#### 2.2.1 Stride布局（64字节行对齐）

**适用区域**: D区、A区、B区、T区

**计算公式**:
```cpp
stride = ((width * num_channels + 63) / 64) * 64
```

**特点**:
- 每行数据64字节对齐，优化SIMD性能
- 行与行之间有padding（可能）
- Simd库原生支持此布局

**示例**: 224×224 RGB图像
```
width = 224, channels = 3
row_bytes = 224 * 3 = 672
stride = ((672 + 63) / 64) * 64 = 704
每行padding = 704 - 672 = 32字节
```

#### 2.2.2 Compact布局（无padding）

**适用区域**: S区、C区、EngineBuffer

**计算公式**:
```cpp
row_bytes = width * num_channels
total_bytes = row_bytes * height
```

**特点**:
- 无行间padding，紧凑存储
- 节省内存，适合缓存
- GPU传输效率高

**内存对齐**:
- **单个样本**: `aligned_max_output = align_64(max_train_output)`
  - 整个样本64字节对齐，不是每行
  - 384×384×3 = 442,368字节（已64字节对齐）
- **整个区域**: 4KB页对齐（NUMA优化）

#### 2.2.3 布局转换

**关键点**: AB区（stride）→ EngineBuffer（compact）需要布局转换！

**解决方案**: PO的`execute()`方法接受`bool compact`参数
```cpp
void execute(..., size_t& output_stride, ..., bool compact = true)
```

- `compact = false`: AB区乒乓，使用stride布局
- `compact = true`: 最终输出，使用compact布局

---

## 3. 乒乓机制（Ping-Pong）

### 3.1 设计目的

支持多个PO链式执行，避免每次操作都重新分配内存。

### 3.2 工作原理

**规则**: `src_ptr != region_a_` → 输出到A区，否则输出到B区

**示例**: 3个操作链（RRC → Resize → RHF）

```
Op0 (RRC):
  src_ptr = nullptr (不在A区)
  → dest_ptr = region_a_
  → src_ptr = region_a_

Op1 (Resize):
  src_ptr = region_a_ (在A区)
  → dest_ptr = region_b_
  → src_ptr = region_b_

Op2 (RHF):
  src_ptr = region_b_ (在B区)
  → dest_ptr = final_output_ptr (EngineBuffer，compact布局)
```

### 3.3 实现细节

**代码位置**: `preprocess_worker.cpp:1317-1348`

```cpp
for (int op_id = 0; op_id < total_ops_before_flip; ++op_id) {
    if (src_ptr != region_a_) {
        dest_ptr = region_a_;
    } else {
        dest_ptr = region_b_;
    }

    // 执行PO（compact=false，stride布局）
    ops[op_id]->execute(..., dest_ptr, ..., false);

    // 更新src_ptr为本次输出，供下次使用
    src_ptr = dest_ptr;
}
```

### 3.4 关键优化

**AB区使用Stride布局的原因**:
1. Simd库优化：64字节对齐加速SIMD操作
2. 乒乓友好：每次操作的输入/输出都是stride布局
3. GPU兼容：虽然EngineBuffer是compact，但中间结果保持在CPU上

**最终输出使用Compact布局的原因**:
1. 内存节省：GPU显存昂贵
2. 传输效率：紧凑数据PCIe传输更快
3. 框架约定：深度学习框架期望compact layout

---

## 4. RandomHorizontalFlip (RHF) 路径优化

### 4.1 优化动机

**问题**: RHF是否执行由随机决定，提前执行可能导致不必要的内存复制。

**示例**: RRC → RHF链
- 如果需要翻转: RRC输出 → RHF翻转 → EngineBuffer（2次memcpy）
- 如果不翻转: RRC输出直接→ EngineBuffer（0次额外memcpy）

### 4.2 优化策略

**核心思想**: 延迟RHF执行决策，到最后时刻才判断。

**实现**: 两阶段判断

#### 阶段1：判断是否需要翻转

```cpp
bool last_op_is_flip = false;
if (has_random_horizontal_flip) {
    // Preprocessor已排序，最后一个必定是RandomHorizontalFlip
    last_op_is_flip = ops.back()->should_flip(&rng_);
    total_ops_before_flip = num_ops - 1;
}
```

#### 阶段2：根据决策选择路径

**情况A**: 需要翻转（`last_op_is_flip == true`）

```cpp
// 执行所有非flip操作（AB区乒乓，stride布局）
for (int op_id = 0; op_id < total_ops_before_flip; ++op_id) {
    ops[op_id]->execute(..., dest_ptr, ..., false);  // compact=false
}

// 执行翻转（直接输出到EngineBuffer，compact布局）
ops[total_ops_before_flip]->execute(..., final_output_ptr, ..., true);
```

**情况B**: 不需要翻转（`last_op_is_flip == false`）

```cpp
// 执行所有操作（最后一个直接输出到EngineBuffer）
for (int op_id = 0; op_id < total_ops_before_flip; ++op_id) {
    bool compact_layout = (op_id == total_ops_before_flip - 1);
    ops[op_id]->execute(..., dest_ptr, ..., compact_layout);
}
```

### 4.3 特殊情况处理

#### 情况1：只有RHF一个操作

```cpp
if (has_random_horizontal_flip && num_ops == 1) {
    if (last_op_is_flip) {
        ops[0]->execute(..., final_output_ptr, ..., true);  // 翻转
    } else {
        built_in_do_nothing_->execute(..., final_output_ptr, ..., true);  // 复制
    }
}
```

**注意**: Preprocessor禁止ImageNet训练时只有RHF一个操作（必须在Crop/Resize之后）。

### 4.4 性能分析

**场景**: RRC → RHF，batch_size=256，resolution=224

**优化前**（假设RHF在RRC之前判断）:
```
RRC输出到A区 → 判断需要翻转 → 复制A区到EngineBuffer → RHF翻转
额外memcpy: 256 × 224 × 224 × 3 = 38,535,168 字节
```

**优化后**:
```
RRC输出到B区 → 判断需要翻转 → RHF直接输出到EngineBuffer
额外memcpy: 0
```

**节省**: ~36.7 MB内存复制（每个batch）

---

## 5. Busy Phase vs Lazy Phase

### 5.1 Phase分类

| Phase类型 | is_lazy_phase | 行为 | 触发时机 |
|-----------|---------------|------|---------|
| Busy Train Phase | false | 调用`work()`解码+预处理，写入S区+EngineBuffer | Epoch首次训练 |
| Lazy Train Phase | true | 调用`work_lazy()`从S区复制到EngineBuffer | Epoch重复训练（SDMP>1）|
| Busy Val Phase | false | 调用`work()`解码+预处理，写入C区+EngineBuffer | Epoch首次验证 |
| Lazy Val Phase | true | 调用`work_lazy()`从C区复制到EngineBuffer | Epoch重复验证（CPVS=true）|

### 5.2 Busy Phase（work()方法）

#### 5.2.1 核心逻辑

**代码位置**: `preprocess_worker.cpp:1063-1431`

```cpp
bool PreprocessWorker::work(int32_t label, const uint8_t* data_ptr, size_t data_size) {
    // 步骤1：JPEG解码（D区）
    // 步骤2：确定任务数量（SDMP/CPVS）
    // 步骤3：RHF路径优化判断
    // 步骤4：执行PO链（AB区乒乓）
    // 步骤5：输出到EngineBuffer/S区/C区
}
```

#### 5.2.2 SDMP多任务处理

**训练阶段**（`is_train = true`, `sdmp_factor = N`）:
```
task_id = 0:     输出到S1区
task_id = 1:     输出到S2区
...
task_id = N-2:   输出到S(N-1)区
task_id = N-1:   输出到EngineBuffer（本次epoch直接使用）
```

**关键点**:
- 每个task执行完整的PO链（从JPEG解码开始）
- 所有S区存储相同样本的**不同随机增强版本**
- 最后一个task的输出直接送到EngineBuffer（当前epoch立即消费）

**验证阶段**（`is_train = false`, `using_cpvs = true`）:
```
task_id = 0:     输出到C区（缓存）
task_id = 1:     从C区复制到EngineBuffer（CPVS优化）
```

**CPVS优化**: 第二个task直接从C区memcpy，无需重新解码+预处理！

```cpp
if (task_id == 1) {
    uint8_t* c_region_source_ptr = region_c_ptr_ + local_sample_id_ * s_c_region_stride_;
    std::memcpy(final_output_ptr, c_region_source_ptr, res * res * num_color_channels);
    notify_engine_buffer_sample_written();
    break;  // 提前退出，无需预处理
}
```

#### 5.2.3 计数器管理

**Busy phase开始时**:
```cpp
local_sample_id_ = 0;      // 复位样本计数器
current_s_samples_ = 0;    // 复位S区写入计数（train）
current_c_samples_ = 0;    // 复位C区写入计数（val）
```

**每个样本处理后**:
```cpp
if (is_train && sdmp_factor > 1) {
    current_s_samples_++;  // 递增S区计数
}
if (!is_train && using_cpvs) {
    current_c_samples_++;  // 递增C区计数
}
end_sample();  // 递增local_sample_id_
```

### 5.3 Lazy Phase（work_lazy()方法）

#### 5.3.1 核心逻辑

**代码位置**: `preprocess_worker.cpp:1602-1671`

```cpp
void PreprocessWorker::work_lazy() {
    // 步骤1：复位local_sample_id_
    local_sample_id_ = 0;

    // 步骤2：判断phase类型
    if (param_.is_train) {
        // Lazy Train Phase：从S区读取
        shuffle_s_indices(param_.phase_id);  // S区索引洗牌
        for (int i = 0; i < current_s_samples_; ++i) {
            copy_sample_from_s_to_eb(active_s_region_idx);
            end_sample();
        }
    } else {
        // Lazy Val Phase：从C区读取
        for (int i = 0; i < current_c_samples_; ++i) {
            copy_sample_from_c_to_eb();
            end_sample();
        }
    }

    // 步骤3：通知EngineBuffer无更多样本
    no_more_samples();
}
```

#### 5.3.2 S区洗牌机制

**为什么需要洗牌？**
- Busy phase写入S区时是顺序的（slot 0, 1, 2, ...）
- Lazy phase读取时需要打乱顺序（增强随机性）

**两层种子衍生**:
```cpp
// 构造时（第一层）
initial_seed_ = global_seed ^ (worker_id << 32);

// Shuffle时（第二层）
shuffle_seed = initial_seed_ ^ (phase_id << 32);
```

**Fisher-Yates洗牌**（只洗有效部分）:
```cpp
int n = current_s_samples_;  // 只洗已写入的样本
for (int i = n - 1; i > 0; --i) {
    uint32_t r[4];
    detail::philox_generate_4x32(shuffle_seed, i, r);
    uint32_t j = r[0] % (i + 1);
    std::swap(s_shuffled_indices_[i], s_shuffled_indices_[j]);
}
```

**关键点**:
- 洗牌范围是`[0, current_s_samples_ - 1]`，不是`[0, max_s_samples_ - 1]`
- 未使用的槽位不参与洗牌
- 完全可复现（固定种子 + Philox算法）

#### 5.3.3 计数器管理

**Lazy phase开始时**:
```cpp
local_sample_id_ = 0;  // 复位（但current_s_samples_/current_c_samples_不变）
```

**Lazy phase期间**:
- `current_s_samples_`保持不变（Busy phase结束时写入的值）
- `current_c_samples_`保持不变
- `local_sample_id_`递增（从0到`current_s_samples_`或`current_c_samples_`）

**Lazy phase结束后**:
- 计数器不清零（下一个lazy phase复用）

### 5.4 Phase切换流程

**示例**: SDMP=2的3个epoch训练

```
Epoch 0 (Busy):
  work() × N_samples
    → task_0 → S1区
    → task_1 → EngineBuffer
  → current_s_samples_ = N_samples

Epoch 1 (Lazy):
  work_lazy()
    → shuffle S1区索引
    → 从S1区复制到EngineBuffer × N_samples
  → current_s_samples_ = N_samples（不变）

Epoch 2 (Lazy):
  work_lazy()
    → shuffle S1区索引（不同种子）
    → 从S1区复制到EngineBuffer × N_samples
  → current_s_samples_ = N_samples（不变）
```

**关键点**: Lazy phase完全跳过JPEG解码和预处理，只做内存复制！

---

## 6. 内存布局与性能优化

### 6.1 Stride vs Compact 对比

| 维度 | Stride布局 | Compact布局 |
|------|-----------|-------------|
| **行对齐** | 64字节 | 无对齐 |
| **Padding** | 有（可能） | 无 |
| **SIMD性能** | 高（原生支持） | 需要处理 |
| **内存占用** | 大 | 小 |
| **GPU传输** | 低效（含padding） | 高效 |
| **适用场景** | AB区（CPU乒乓） | S/C/EngineBuffer |

### 6.2 最佳实践

#### 6.2.1 AB区使用Stride布局

**原因**:
1. Simd库优化：`SimdResizeRun()`接受srcStride/dstStride参数
2. 乒乓友好：连续操作的输入输出stride一致
3. CPU缓存：64字节对齐优化缓存行命中

**示例**: Resize操作
```cpp
SimdResizerRun(resizer_cache_,
              input_ptr, input_stride,   // stride input
              output_ptr, output_stride); // stride output
```

#### 6.2.2 S/C/EngineBuffer使用Compact布局

**原因**:
1. 内存节省：节省padding空间
2. GPU传输：紧凑数据PCIe传输快
3. 框架兼容：PyTorch/TensorFlow期望NHWC compact

**示例**: C区到EngineBuffer复制
```cpp
size_t num_bytes = resolution * resolution * num_channels;
std::memcpy(eb_ptr, c_ptr, num_bytes);  // 直接复制，无需考虑stride
```

### 6.3 布局转换开销

**场景**: AB区（stride）→ EngineBuffer（compact）

**不开辟中间缓冲区**（推荐）:
```cpp
// 最后一个PO直接输出compact布局
ops[last_op]->execute(..., final_output_ptr, ..., true);
```

**开辟中间缓冲区**（不推荐）:
```cpp
// 先输出stride布局到AB区
ops[last_op]->execute(..., region_a_, ..., false);

// 手动转换为compact布局（额外开销）
for (int y = 0; y < height; ++y) {
    std::memcpy(compact_ptr + y * width * channels,
               stride_ptr + y * stride,
               width * channels);
}
```

**性能差异**: 第一种方法节省一次完整图像的内存复制！

---

## 7. 关键设计决策

### 7.1 为什么AB区需要64字节对齐？

**CPU缓存行**: 64字节

**好处**:
1. **缓存行对齐**: 跨缓存行访问减少
2. **SIMD指令**: AVX-512正好512位=64字节
3. **内存总线**: 现代CPU内存总线64字节传输

**实测**: stride布局比compact布局快5-10%（SIMD重负载场景）

### 7.2 为什么S/C区使用Compact布局？

**GPU显存昂贵**: 每个padding字节都要传到GPU

**示例**: 224×224×3图像
- Compact: 150,528字节
- Stride: 157,696字节（多4.7%）

**Batch级别影响**（batch=512）:
- 节省: 512 × (157,696 - 150,528) = 3.6 MB
- PCIe传输时间: ~3.6 MB / 12 GB/s = 0.3 ms节省

### 7.3 为什么需要RHF路径优化？

**随机决策时机问题**:
- RHF的翻转决策应该在AB区乒乓**之后**
- 而不是在第一个操作开始**之前**

**错误设计**（提前决策）:
```
1. 判断需要翻转
2. 执行RRC到A区
3. 如果翻转：执行RHF到EngineBuffer
   如果不翻转：复制A区到EngineBuffer
```

**正确设计**（延迟决策）:
```
1. 执行RRC到B区（stride布局）
2. 判断需要翻转
3. 如果翻转：执行RHF到EngineBuffer（compact布局）
   如果不翻转：执行Resize到EngineBuffer（compact布局）
```

**性能差异**: 错误设计在"不翻转"分支多一次A区→EngineBuffer的memcpy。

---

## 8. 总结

### 8.1 核心要点

1. **静态分配**: Worker `i` 的第 `k` 次调用 → 样本 `(i + k×M)`
2. **乒乓机制**: AB区交替，支持多PO链式执行
3. **RHF优化**: 延迟翻转决策，节省不必要的memcpy
4. **Stride vs Compact**: AB区用stride（SIMD优化），S/C/EB用compact（节省内存）
5. **Busy vs Lazy**: Busy做完整处理，Lazy只做复制

### 8.2 性能优化技巧

1. **避免中间缓冲**: 直接输出compact布局到最后一个PO
2. **CPVS优化**: C区memcpy代替完整JPEG解码+预处理
3. **SDMP缓存**: S区存储多份增强版本，后续epoch直接复用
4. **SIMD友好**: AB区stride布局最大化Simd库性能

### 8.3 扩展性

**新增PO步骤**:
1. 继承`PreprocessOperation`
2. 实现`execute(..., bool compact)`
3. 实现`get_decode_strategy()`
4. 集成到PO链（Preprocessor自动排序RHF到最后）

**新增内存区域**:
1. 在`allocate_workshop()`添加区域分配
2. 更新`Config`结构体添加大小参数
3. 添加对应的`request_xxx_slot()`方法

---

## 附录A: 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| Workshop分配 | `preprocess_worker.cpp` | 255-342 |
| 乒乓机制 | `preprocess_worker.cpp` | 1317-1348 |
| RHF路径优化 | `preprocess_worker.cpp` | 1262-1416 |
| Busy phase | `preprocess_worker.cpp` | 1063-1431 |
| Lazy phase | `preprocess_worker.cpp` | 1602-1671 |
| S区洗牌 | `preprocess_worker.cpp` | 643-790 |
| C区CPVS优化 | `preprocess_worker.cpp` | 1248-1255 |

---

## 附录B: 内存布局示例

**配置**: max_resolution=384, num_color_channels=3

### D区（假设原始图像500×500）
```
stride = ((500 * 3 + 63) / 64) * 64 = 768
size = 768 * 500 = 384,000 字节
```

### A/B区（384×384）
```
stride = ((384 * 3 + 63) / 64) * 64 = 576
size = 576 * 384 = 221,184 字节
```

### S区（max_samples=1000）
```
sample_size = 384 * 384 * 3 = 442,368 字节
aligned_sample = align_64(442368) = 442,368 字节
region_size = 1000 * 442,368 = 442,368,000 字节 ≈ 421.9 MB
page_aligned = align_4k(442,368,000) = 442,368,000 字节
```

### C区（max_samples=50000）
```
region_size = 50000 * 442,368 = 22,118,400,000 字节 ≈ 20.6 GB
```

**注意**: C区非常大，CPVS适用于大内存服务器！

---

**文档结束**
