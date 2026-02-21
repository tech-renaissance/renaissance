# S区洗牌方案设计文档

## 版本信息
- **版本**: V1.0.0
- **日期**: 2026-02-19
- **作者**: 技术觉醒团队

---

## 一、背景和目标

### 1.1 S区和C区的作用

**S区（SDMP缓存）**：
- 数量：n = sdmp_factor - 1
- 用途：训练阶段（train phase）的样本缓存
- 特点：**支持洗牌**

**C区（CPVS缓存）**：
- 数量：m = 0或1（取决于using_cpvs）
- 用途：验证阶段（val phase）的样本缓存
- 特点：**禁止洗牌**

### 1.2 洗牌需求

每个lazy epoch（train phase）开始时，需要对S区进行洗牌：
- **洗牌范围**：当前S区实际已有的样本数（`current_s_samples_`），不是`max_s_samples_`
- **零拷贝**：不改变数据在S区的物理存储位置，只修改读取顺序
- **标签对应**：洗牌后，标签与S区样本的对应关系必须保持不变
- **可复现**：相同的种子序列必须产生相同的洗牌结果

---

## 二、核心设计思想

### 2.1 索引映射方案

**核心思想**：数据和标签物理位置不变，只修改"索引映射"

```
传统方式（直接访问）：
  读取顺序：data[0], data[1], data[2], ...

索引映射方式：
  读取顺序：data[index[0]], data[index[1]], data[index[2]], ...
  洗牌只修改index，不动data
```

### 2.2 为什么只需要一个索引向量？

**关键洞察**：每个S区的样本编号都是从0到max_s_samples_-1，因此**同一个索引向量可以用于所有S区**

```
假设有3个S区，max_s_samples = 1000：

GlobalRegistry::fixed_s_original_indices_ = [0, 1, 2, ..., 999]

每个PW复制到本地：
s_shuffle_indices_ = [0, 1, 2, ..., 999]

洗牌后（假设current_s_samples_ = 5）：
s_shuffle_indices_ = [3, 1, 4, 0, 2, 5, 6, 7, ..., 999]
                      ^^^^^^^^^^^^^^^
                      洗牌部分（前5个元素）

读取S区0的第i个样本：
  label = s_label_vectors_[0][s_shuffle_indices_[i]]
  data  = region_s_ptrs_[0] + s_shuffle_indices_[i] * sample_bytes

读取S区1的第i个样本：
  label = s_label_vectors_[1][s_shuffle_indices_[i]]
  data  = region_s_ptrs_[1] + s_shuffle_indices_[i] * sample_bytes

读取S区2的第i个样本：
  label = s_label_vectors_[2][s_shuffle_indices_[i]]
  data  = region_s_ptrs_[2] + s_shuffle_indices_[i] * sample_bytes
```

**所有S区共享同一个索引向量，但各自的样本数据是独立的！**

---

## 三、数据结构设计

### 3.1 GlobalRegistry中的全局向量

```cpp
// GlobalRegistry成员变量
std::vector<int> fixed_s_original_indices_;  // 全局共享的原始顺序索引
```

**初始化时机**：在确定`max_s_samples_`之后（通常是GlobalRegistry初始化阶段）

**初始化内容**：
```cpp
fixed_s_original_indices_.resize(max_s_samples);
for (int i = 0; i < max_s_samples; ++i) {
    fixed_s_original_indices_[i] = i;  // [0, 1, 2, ..., max_s_samples-1]
}
```

**特点**：
- 全局可见，所有PW共享
- 长度固定，内容为有序序列
- **只读，永不修改**（作为"原始顺序"的标准定义）

### 3.2 PreprocessWorker中的本地变量

```cpp
// PreprocessWorker成员变量
int current_s_samples_ = 0;                      // 当前S区实际已有的样本数
std::vector<int> s_shuffle_indices_;             // 洗牌索引向量（从GlobalRegistry复制）
```

**数据存储**（已有的成员变量）：
```cpp
std::vector<uint8_t*> region_s_ptrs_;                  // S区数据指针数组
std::vector<std::vector<int32_t>> s_label_vectors_;    // S区标签向量数组
std::vector<int32_t> c_label_vector_;                  // C区标签向量（禁止洗牌）
```

---

## 四、洗牌流程

### 4.1 洗牌触发时机

**每个lazy epoch（train phase）开始时调用**：
```cpp
void PreprocessWorker::shuffle_s_zones(int epoch_id);
```

### 4.2 洗牌步骤

#### 步骤1：复制原始索引
```cpp
// 从GlobalRegistry直接复制（向量赋值，O(1)操作）
s_shuffle_indices_ = GlobalRegistry::instance().fixed_s_original_indices();
```

**关键点**：
- 每次都从原始有序状态开始
- 不需要手动初始化[0, 1, 2, ...]序列
- 避免了反复初始化的开销

#### 步骤2：生成洗牌种子
```cpp
// 生成本次洗牌的种子（保证可复现）
uint64_t global_seed = get_default_generator().seed();
uint64_t shuffle_seed = global_seed
                      ^ (static_cast<uint64_t>(config_.worker_id) << 32)
                      ^ static_cast<uint64_t>(epoch_id);
```

**种子设计原则**：
- 全局种子：基础随机性
- worker_id：不同PW使用不同洗牌
- epoch_id：不同epoch使用不同洗牌
- XOR操作：保证各分量独立影响

#### 步骤3：Fisher-Yates洗牌
```cpp
// 创建临时RNG用于洗牌
Generator shuffle_rng(shuffle_seed);

// 只洗牌前current_s_samples_个元素
int n = current_s_samples_;
for (int i = n - 1; i > 0; --i) {
    // 生成随机数 j ∈ [0, i]
    int j = shuffle_rng.random_int(0, i);

    // 交换索引
    std::swap(s_shuffle_indices_[i], s_shuffle_indices_[j]);
}
```

**关键点**：
- 洗牌范围：`[0, current_s_samples_-1]`
- 后面的元素（`[current_s_samples_, max_s_samples_-1]`）保持不变
- Fisher-Yates算法：从后向前，O(n)时间复杂度

---

## 五、读取方式

### 5.1 S区读取（洗牌后）

```cpp
// 读取第zone个S区的第i个样本
int physical_id = s_shuffle_indices_[i];  // 先查索引
int32_t label = s_label_vectors_[zone][physical_id];
uint8_t* data = region_s_ptrs_[zone] + physical_id * sample_bytes;
```

### 5.2 C区读取（禁止洗牌）

```cpp
// C区不使用索引向量，直接访问
int32_t label = c_label_vector_[i];
uint8_t* data = region_c_ptr_ + i * sample_bytes;
```

---

## 六、为什么标签对应关系不变？

**因为索引向量作用于"样本编号"，数据和标签共享同一个编号：**

```
洗牌前：
  索引:    [0, 1, 2, 3]
  数据:    [A, B, C, D]  （物理位置不变）
  标签:    [1, 2, 3, 4]  （物理位置不变）

  读取index[0] → data[0]=A, label[0]=1  ✅

洗牌后（索引变成[3, 1, 0, 2]）：
  索引:    [3, 1, 0, 2]
  数据:    [A, B, C, D]  （物理位置不变）
  标签:    [1, 2, 3, 4]  （物理位置不变）

  读取index[0] → data[3]=D, label[3]=4  ✅
  读取index[1] → data[1]=B, label[1]=2  ✅
  读取index[2] → data[0]=A, label[0]=1  ✅
  读取index[3] → data[2]=C, label[2]=3  ✅
```

**数据和标签同步变化，因为它们使用同一个物理ID！**

---

## 七、current_s_samples_ 的管理

### 7.1 变量含义

- `max_s_samples_`：S区最大容量（固定，配置时确定）
- `current_s_samples_`：当前S区实际已有的样本数（动态变化）

### 7.2 更新时机

需要在适当的时候更新`current_s_samples_`：

```cpp
void PreprocessWorker::set_current_s_samples(int count) {
    TR_CHECK(count <= config_.max_s_samples, ValueError,
             "current_s_samples exceeds max_s_samples: "
             << count << " > " << config_.max_s_samples);
    current_s_samples_ = count;
}
```

**可能的更新时机**：
- PW完成一批样本的写入后
- 从外部（如Preprocessor）通知当前样本数

---

## 八、方案优势

### 8.1 与多索引向量方案对比

| 特性 | 多索引向量方案（旧） | 单索引向量方案（新） |
|------|---------------------|---------------------|
| **索引向量数量** | num_region_s个 | 1个 |
| **内存占用** | O(num_region_s × max_s_samples) | O(max_s_samples) |
| **初始化开销** | 需要逐个初始化每个向量 | 从GlobalRegistry直接复制 |
| **逻辑复杂度** | 需要管理多个索引向量 | 只需管理一个索引向量 |
| **全局一致性** | 各PW独立初始化 | 共享同一个"原始顺序"定义 |

### 8.2 核心优势

1. **内存高效**
   - 只需1个索引向量，节省(num_region_s - 1) × max_s_samples × sizeof(int)字节

2. **初始化零开销**
   - 直接从GlobalRegistry复制（向量赋值）
   - 不需要手动构造[0, 1, 2, ...]序列

3. **全局一致性**
   - 所有PW共享同一个"原始顺序"标准定义
   - 便于调试和验证

4. **灵活的洗牌策略**
   - 每个epoch独立洗牌
   - 洗牌范围动态调整（current_s_samples_）
   - 种子可定制（worker_id、epoch_id）

5. **可复现性**
   - Fisher-Yates算法保证均匀分布
   - 种子设计保证可复现

---

## 九、C区为什么禁止洗牌？

**C区用于验证阶段（val phase）**：
- 验证数据需要**固定顺序**，不能打乱
- 验证指标（如accuracy、top5）依赖于样本顺序
- 不同epoch的验证结果需要可比性

因此：
- C区不使用索引向量
- 读取时直接按物理顺序访问
- 不调用shuffle方法

---

## 十、实现示例

### 10.1 PreprocessWorker成员变量

```cpp
class PreprocessWorker {
private:
    // S区和C区的内存指针
    std::vector<uint8_t*> region_s_ptrs_;                  // S区数据指针数组
    uint8_t* region_c_ptr_ = nullptr;                      // C区数据指针

    // S区和C区的标签存储
    std::vector<std::vector<int32_t>> s_label_vectors_;    // S区标签向量数组
    std::vector<int32_t> c_label_vector_;                  // C区标签向量

    // 洗牌索引（S区专用）
    int current_s_samples_ = 0;                            // 当前S区实际样本数
    std::vector<int> s_shuffle_indices_;                   // 洗牌索引向量
};
```

### 10.2 洗牌方法实现

```cpp
void PreprocessWorker::shuffle_s_zones(int epoch_id) {
    // 步骤1：从GlobalRegistry复制原始索引
    s_shuffle_indices_ = GlobalRegistry::instance().fixed_s_original_indices();

    // 步骤2：生成洗牌种子
    uint64_t global_seed = get_default_generator().seed();
    uint64_t shuffle_seed = global_seed
                          ^ (static_cast<uint64_t>(config_.worker_id) << 32)
                          ^ static_cast<uint64_t>(epoch_id);

    // 步骤3：创建临时RNG
    Generator shuffle_rng(shuffle_seed);

    // 步骤4：Fisher-Yates洗牌（前current_s_samples_个元素）
    int n = current_s_samples_;
    for (int i = n - 1; i > 0; --i) {
        int j = shuffle_rng.random_int(0, i);
        std::swap(s_shuffle_indices_[i], s_shuffle_indices_[j]);
    }

    LOG_DEBUG << "PW " << config_.worker_id
              << " shuffled S zones: epoch=" << epoch_id
              << ", seed=" << shuffle_seed
              << ", range=[0," << (n-1) << "]";
}
```

### 10.3 读取示例

```cpp
// 从S区读取第i个样本（已洗牌）
void read_from_s_zone(int zone_id, int logical_index, int32_t& label, uint8_t*& data) {
    // 先查索引，获取物理样本ID
    int physical_id = s_shuffle_indices_[logical_index];

    // 根据物理ID读取数据和标签
    label = s_label_vectors_[zone_id][physical_id];
    data = region_s_ptrs_[zone_id] + physical_id * sample_bytes;
}

// 从C区读取第i个样本（禁止洗牌）
void read_from_c_zone(int index, int32_t& label, uint8_t*& data) {
    // 直接访问，无索引映射
    label = c_label_vector_[index];
    data = region_c_ptr_ + index * sample_bytes;
}
```

---

## 十一、总结

本方案通过**单索引向量 + GlobalRegistry共享**的设计，实现了：

1. **高效的内存使用**：只需1个索引向量
2. **零开销初始化**：从GlobalRegistry直接复制原始顺序
3. **灵活的洗牌策略**：每个epoch独立洗牌，洗牌范围动态调整
4. **可复现的随机性**：种子 = f(全局种子, worker_id, epoch_id)
5. **标签自动对应**：索引作用于样本编号，数据和标签同步变化

这是一个兼顾效率、灵活性和正确性的优雅设计！
