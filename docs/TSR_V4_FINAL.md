 **TSR-V4.20 张量存储格式规范（V4.21 修订版）**

---

**版本**：V4.20（V4.21 修订）
**适用框架**：技术觉醒 V4（TR4）
**状态**：正式规范
**日期**：2026-05-06（修订日期）
**作者**：技术觉醒团队

---

### 一、概述

#### 1.1 定位与目标

TSR（Tensor Storage Record）-V4.20 是 TR4 框架的原生张量持久化格式。本版本专为 TR4 的静态图、极致性能目标设计，服务于 A100×8 平台上的 ResNet-50 AMP 训练等高密度负载场景。

本规范定义单文件多张量的序列化协议。

**V4.21 修订说明**：自 TR4 V4.21 起，`Tensor` 类强制采用紧凑布局（无行间 Padding），TSR 协议的保存与加载逻辑相应简化。本文档已全面更新以反映当前代码实现。

#### 1.2 核心设计原则

1. **Tensor 强制紧凑**：自 V4.21 起，`tr::Tensor` 在 CPU 端永远采用紧凑布局——`nbytes = numel × sizeof(dtype)`，数据连续存放，无任何行间或通道间 Padding。通道补齐（C→8/4）仅存在于 DTensor（GPU 侧），与 TSR 存储格式无关。
2. **直接存储**：TSR 保存和加载均直接在紧凑 Tensor 上操作，无需"创建干净缓冲区"或"按行展开"等转换步骤。
3. **CRC32 完整性**：CRC32 对 Tensor 全部有效数据（`nbytes` 字节）计算，由于 Tensor 已是紧凑的，等同于 `numel × sizeof(dtype)` 字节。
4. **首地址 256 字节对齐**：Tensor 基地址由分配器保证 256 字节对齐，TSR RAW 模式数据区起始偏移也为 256 字节对齐。

#### 1.3 与 TSR V3 的关键演进

| 特性 | TSR V3（TR3） | TSR-V4.20（TR4） | V4.21 修订 |
|:---:|:---:|:---:|:---|
| **魔数** | `TSR3` | `TSR4` | 不变 |
| **维度语义** | 0–4D 右对齐 `dims[4]` | **强制 4D NHWC** | 不变 |
| **数据类型** | FP32 / BF16 / INT32 / INT8 | **FP32 / FP16 / INT8 / INT32** | 不变 |
| **行步幅** | 不存储 | **显式存储 `row_stride`** | **V4.21：紧凑值 = W×C×sizeof(dtype)** |
| **总字节数** | `raw_data_size`（逻辑大小） | **显式存储 `nbytes`** | **V4.21：= numel × sizeof(dtype)** |
| **多张量支持** | ❌ 单文件单张量 | ✅ **单文件多张量** | 不变 |
| **Tensor 布局** | 带 Padding | 带 Padding | **V4.21：强制紧凑，无 Padding** |
| **加载重建** | 直接读取 | 按行展开重建对齐 | **V4.21：直接读取，无需重建** |

---

### 二、文件格式规范

#### 2.1 整体结构

一个 TSR-V4.20 文件由以下四段组成，顺序不可变更：

```
┌─ [0x0000] 文件头 (TSR4FileHeader)
│            固定 128 字节
├─ [0x0080] 张量目录区 (Tensor Directory)
│            N × 64 字节，N = tensor_count
├─ [对齐边界] 对齐填充区 (仅 RAW 模式存在)
│            0x00 填充至 256 字节边界
└─ [data0]  数据载荷区 (Payload)
             包含 N 个独立数据块
```

- **文件头**与**目录区**的长度之和恒为 `128 + N × 64`。
- **RAW 模式**：在目录区之后填充 0x00，使首个数据块的起始偏移为 256 的整数倍；后续每个数据块的起始偏移也必须是 256 的整数倍。
- **ZLIB 模式**：目录区之后无填充，数据块紧密排列。

#### 2.2 文件头结构 (TSR4FileHeader)

文件头固定 **128 字节**，采用 `#pragma pack(push, 1)` 严格打包，无编译器填充。

| 偏移 | 字段 | 类型 | 大小 | 说明 |
|:---:|:---|:---:|:---:|:---|
| +0x00 | `magic[4]` | `char[4]` | 4 B | 魔数，固定为 `"TSR4"`（0x54, 0x53, 0x52, 0x34） |
| +0x04 | `version_major` | `uint16_t` | 2 B | 主版本号，固定为 **4** |
| +0x06 | `version_minor` | `uint16_t` | 2 B | 次版本号，固定为 **20** |
| +0x08 | `header_size` | `uint32_t` | 4 B | 本头部大小，固定为 **128** |
| +0x0C | `file_mode` | `uint32_t` | 4 B | 全局模式：**0 = RAW**，**1 = ZLIB** |
| +0x10 | `tensor_count` | `uint32_t` | 4 B | 文件中张量数量 N（N ≥ 1） |
| +0x14 | `entry_size` | `uint32_t` | 4 B | 单个目录项大小，固定为 **64** |
| +0x18 | `dir_offset` | `uint64_t` | 8 B | 目录区起始偏移，固定为 **128** |
| +0x20 | `data_offset` | `uint64_t` | 8 B | 数据区起始偏移。<br>RAW 模式：`align_up(128 + N×64, 256)`；<br>ZLIB 模式：`128 + N×64` |
| +0x28 | `header_crc32` | `uint32_t` | 4 B | 对整个 128 字节头部计算的 CRC32（计算时本字段自身置 0） |
| +0x2C | `reserved_0` | `uint32_t` | 4 B | 保留，必须为 0 |
| +0x30 | `reserved_1[80]` | `uint8_t[80]` | 80 B | 保留扩展区，必须为 0 |

**头部长度验证**：`4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 8 + 4 + 4 + 80 = 128` 字节。

#### 2.3 张量目录项结构 (TSR4TensorEntry)

每个张量对应一个固定 **64 字节** 的目录项，总计 `tensor_count × 64` 字节。

| 偏移 | 字段 | 类型 | 大小 | 说明 |
|:---:|:---|:---:|:---:|:---|
| +0x00 | `dtype` | `uint8_t` | 1 B | 数据类型。<br>**0 = FP32**，**1 = FP16**，**2 = INT8**，**3 = INT32** |
| +0x01 | `reserved_0[3]` | `uint8_t[3]` | 3 B | 对齐填充，必须为 0 |
| +0x04 | `shape[4]` | `int32_t[4]` | 16 B | NHWC 形状，`[N, H, W, C]`，所有值 ≥ 1 |
| +0x14 | `numel` | `int64_t` | 8 B | 逻辑元素总数 `N×H×W×C`，冗余校验字段 |
| +0x1C | `row_stride` | `uint64_t` | 8 B | 行步幅（字节）。V4.21 起 `Tensor` 强制紧凑，**该值恒等于 `W × C × sizeof(dtype)`** |
| +0x24 | `nbytes` | `uint64_t` | 8 B | 总字节数。V4.21 起 `Tensor` 强制紧凑，**该值恒等于 `numel × sizeof(dtype)`** |
| +0x2C | `data_offset` | `uint64_t` | 8 B | 该张量数据块在文件中的绝对偏移 |
| +0x34 | `payload_size` | `uint64_t` | 8 B | 数据块在文件中的实际字节数。<br>**RAW 模式**：等于 `nbytes`（= numel × sizeof(dtype)）；<br>**ZLIB 模式**：等于压缩后大小 |
| +0x3C | `data_crc32` | `uint32_t` | 4 B | 对全部有效数据的 CRC32。由于 `Tensor` 已是紧凑的，该值对 `nbytes` 字节（= `numel × sizeof(dtype)`）计算 |

**目录项长度验证**：`1 + 3 + 16 + 8 + 8 + 8 + 8 + 8 + 4 = 64` 字节（恰好一个 Cache-Line）。

---

### 三、布局、数据区与 CRC 铁律

#### 3.1 Tensor 紧凑布局

自 V4.21 起，`tr::Tensor` 在 CPU 端**永远采用紧凑布局**：

- `nbytes = numel × sizeof(dtype)`——无任何 Padding
- 数据在内存中按 NHWC 维度顺序连续存放
- `row_stride = W × C × sizeof(dtype)`——等于有效行字节数
- 基地址 256 字节对齐（由分配器保证）

此规则适用于所有数据类型（FP32、FP16、INT32、INT8），无一例外。

**设计理由**：

- Tensor 作为二等公民（仅用于 H2D/D2H 搬运），无需维护 GPU 侧的对齐策略。
- 对齐需求（C 通道补齐）仅在 DTensor（GPU 侧）处理，由 `MemoryPlan` 根据 `Region` 和 `DType` 决定。
- 紧凑布局使得 TSR 保存/加载流程极大简化，消除所有中间转换步骤。

#### 3.2 数据区对齐规则

- **RAW 模式**：
  - 首个数据块起始：`data_offset = align_up(128 + N × 64, 256)`
  - 后续数据块起始：`data_offset_i = align_up(data_offset_{i-1} + payload_size_{i-1}, 256)`
  - 块间填充字节必须全部置为 `0x00`。
- **ZLIB 模式**：
  - 数据块紧密排列，无对齐要求，`data_offset_i = data_offset_{i-1} + payload_size_{i-1}`。

#### 3.3 CRC32 计算规范

由于 `Tensor` 已是紧凑的，CRC32 直接对 `Tensor` 的全部有效数据计算。`nbytes` 字节（= `numel × sizeof(dtype)`）在内存中连续存放，无需按行遍历或跳过 Padding。

---

### 四、双模式存储架构

本协议规定**整张文件统一模式**，不支持 MIXED 模式，以降低解析时的分支预测开销。

#### 4.1 RAW 模式（`file_mode = 0`）—— 极致性能

- **存储形式**：
  - 数据块为 `Tensor` 的完整紧凑数据副本，字节级等同于 `tensor.data()` 的内容。
  - `payload_size == nbytes == numel × sizeof(dtype)`。
  - 每个数据块起始偏移为 256 字节对齐，块间填充 0x00。

- **加载方式**：读入构造的紧凑 `Tensor` 后直接按需使用。

- **适用场景**：大模型权重、训练检查点、频繁读取的热点数据、ResNet-50 AMP 权重加载。

#### 4.2 ZLIB 模式（`file_mode = 1`）—— 极致压缩

- **存储形式**：目录区后紧跟各张量有效数据的压缩流，块间无填充。
- **保存流程**：
  1. 对 `Tensor` 全部有效数据（`nbytes = numel × sizeof(dtype)` 字节）计算 CRC32 并写入目录项；
  2. 将全部有效数据压缩，压缩后大小记入 `payload_size`。
- **加载流程**：
  1. 读取 `payload_size` 字节的压缩数据；
  2. 按目标 `shape` 与 `dtype` 构造新的紧凑 `Tensor`；
  3. 解压至 `tensor.data()`（紧凑布局，无需额外操作）；
  4. 对 `tensor.data()` 的全部 `nbytes` 字节计算 CRC32，与目录项比对。
- **适用场景**：模型分发、稀疏张量、网络传输、磁盘空间受限环境。

---

### 五、数据完整性校验体系

协议采用三级校验，任何一级失败均须向调用方报告格式错误或数据损坏。

| 级别 | 校验对象 | 方式 | 失败判定 |
|:---:|:---|:---|:---|
| **L1** | 魔数与版本 | 直接比对 | `magic != "TSR4"` 或版本号不为 `4.20` |
| **L2** | 头部完整性 | CRC-32 (IEEE 802.3) | 对整个 128 字节头部计算 CRC（`header_crc32` 字段置 0），结果与记录值不符 |
| **L3** | 每张量数据 | CRC-32 (IEEE 802.3) | 对 `Tensor` 全部有效数据（`nbytes` 字节）计算 CRC，与 `entry.data_crc32` 不符 |

**CRC32 计算规范**：

- 采用与 zlib 一致的多项式 `0xEDB88320`；
- 初始值：`0xFFFFFFFF`（与zlib标准一致）；
- L2 计算时，头部结构中的 `header_crc32` 字段必须视为 4 字节的 `0x00`。
- L3 计算时，由于 `Tensor` 已是紧凑布局，对 `tensor.data()` 的连续 `nbytes` 字节直接计算即可。

---

### 六、接口规范

以下接口为 `tr::Tensor` 类的静态方法声明，仅规定调用语义，不涉及具体实现。

```cpp
namespace tr {

class Tensor {
public:
    // 保存单个张量（文件内仅含 1 个张量）
    static void save_tensor(const Tensor& tensor,
                            const std::string& filename,
                            bool compress = false);

    // 保存多个张量（指针版本，零拷贝传参）
    static void save_tensors(const std::vector<const Tensor*>& tensors,
                             const std::string& filename,
                             bool compress = false);

    // 保存多个张量（值引用版本）
    static void save_tensors(const std::vector<Tensor>& tensors,
                             const std::string& filename,
                             bool compress = false);

    // 加载文件中的所有张量，按索引顺序返回
    static std::vector<Tensor> load_tensors(const std::string& filename);

    // 加载文件中的首个张量（索引 0），不检查张量总数
    static Tensor load_first_tensor(const std::string& filename);

    // 严格加载单个张量；若文件内张量数量 ≠ 1，须报错
    static Tensor load_tensor(const std::string& filename);
};

} // namespace tr
```

#### 6.1 API 行为语义

| API | 单张量文件（count=1） | 多张量文件（count=N） | 空文件（count=0） |
|:---|:---|:---|:---|
| `save_tensor` | 创建 count=1 的文件 | — | — |
| `save_tensors` | 创建 count=N 的文件 | 创建 count=N 的文件 | 须抛出异常 |
| `load_tensor` | 返回唯一张量 | **须抛出异常** | 须抛出异常 |
| `load_first_tensor` | 返回唯一张量 | 返回索引 0 | 须抛出异常 |
| `load_tensors` | 返回 size=1 的 vector | 返回 size=N 的 vector | 须抛出异常 |

---

### 七、使用示例

以下示例展示序列化与反序列化的典型调用方式，不涉及内部实现细节。

```cpp
#include "renaissance/tensor/tensor.h"

using namespace tr;

int main() {
    // 1. 构建混合精度的 ResNet-50 首层权重
    Tensor conv1_w = Tensor::randn_fp16(Shape(64, 3, 7, 7), DType::FP16, 0.0f, 0.1f);  // FP16权重
    Tensor conv1_b = Tensor::fill(Shape(1, 1, 1, 64), DType::FP32, 0.0f);              // FP32偏置
    Tensor bn1_s   = Tensor::normal(Shape(1, 1, 1, 64), DType::FP32, 1.0f, 0.01f);     // FP32批归一化参数

    // 2. 批量保存为 RAW 模式（用于训练时极速加载）
    std::vector<const Tensor*> stage0 = { &conv1_w, &conv1_b, &bn1_s };
    Tensor::save_tensors(stage0, "resnet50_stage0.tsr", false);

    // 3. 训练启动时加载全部权重
    auto loaded = Tensor::load_tensors("resnet50_stage0.tsr");
    // loaded[0] : conv1_w (FP16), nbytes = 64×3×7×7×2 = 18816
    // loaded[1] : conv1_b (FP32), nbytes = 1×1×1×64×4 = 256
    // loaded[2] : bn1_s   (FP32), nbytes = 1×1×1×64×4 = 256

    // 4. 检查点严格单张量加载
    Tensor ckpt = Tensor::load_tensor("single_fp32_checkpoint.tsr");

    // 5. 验证数据完整性（所有CRC32自动校验）
    // 加载过程中自动进行三级校验，任何失败都会抛出异常
}
```

---

### 八、实现合规性审查清单

任何声称兼容 TSR-V4.20 的实现，必须通过以下全部条款的验证：

1. **魔数与版本**：拒绝所有 `magic != "TSR4"` 或版本号不为 `4.20` 的文件。
2. **头部与目录项长度**：严格校验 `header_size == 128` 且 `entry_size == 64`。
3. **4D 强制性**：加载时须将 `shape[4]` 直接映射为 NHWC，不可进行右对齐推断。
4. **紧凑布局复现**：加载后须验证 `tensor.nbytes() == entry.nbytes == numel × sizeof(dtype)`。自 V4.21 起，`Tensor` 强制紧凑，`nbytes` 和 `row_stride` 均由 shape 唯一确定，无需额外显式校验 `row_stride`。
5. **RAW 对齐**：RAW 模式下，每个 `data_offset` 必须是 256 的倍数，且 `payload_size == nbytes`。
6. **CRC32 闭环**：
   - 头部 CRC 覆盖整个 128 字节头部（计算时 `header_crc32` 置 0）；
   - 数据 CRC 覆盖 `Tensor` 的全部 `nbytes` 字节（= `numel × sizeof(dtype)`）。由于 `Tensor` 是紧凑的，数据连续存放，直接计算即可。
7. **索引区分**：多张量文件内张量不具有名称属性，仅以目录项的物理顺序（0, 1, 2…）作为唯一索引。
8. **原子写入**：建议采用"临时文件 + 原子重命名"策略，防止写入过程中进程崩溃导致文件损坏。
9. **异常安全**：任何校验失败（魔数错误、CRC 不匹配、`tensor_count` 为 0、越界访问等）均须以明确异常终止加载，禁止静默忽略。
10. **numel 冗余校验**：加载时须验证目录项 `numel` 字段等于 `shape[0] × shape[1] × shape[2] × shape[3]`。
11. **nbytes 一致性校验**：加载时须验证 `tensor.nbytes() == entry.nbytes`。
12. **文件大小一致性**：加载完成后须验证文件总大小等于最后一个张量的 `data_offset + payload_size`。
13. **保留字段校验**：头部 `reserved_0`、`reserved_1` 及目录项 `reserved_0` 必须全部为 0。

---

### 九、附录

#### 9.1 数据类型协议映射表

| TR4 DType | 协议 `dtype` 值 | 元素大小 |
|:---:|:---:|:---:|
| `FP32` | 0 | 4 bytes |
| `FP16` | 1 | 2 bytes |
| `INT8` | 2 | 1 byte |
| `INT32` | 3 | 4 bytes |

*注：协议值与 C++ 枚举的底层整数值无关，解析方必须按上表硬编码映射。Tensor 强制紧凑，所有类型的 `row_stride = W × C × sizeof(dtype)`，`nbytes = numel × sizeof(dtype)`。*

#### 9.2 文件格式长度速查表

| 组成部分 | RAW 模式 | ZLIB 模式 |
|:---|:---|:---|
| 文件头 | 128 bytes | 128 bytes |
| 目录区 | N × 64 bytes | N × 64 bytes |
| 对齐填充 | `align_up(128+N×64, 256) - (128+N×64)` | 0 |
| 数据区 | Σ`nbytes_i` + 块间填充（每个数据块 256 字节对齐） | Σ`payload_size_i`（紧密排列） |

**重要说明**：
- **RAW 模式数据区**：每个张量的数据块为 `Tensor` 紧凑数据的完整副本，`payload_size == nbytes == numel × sizeof(dtype)`。
- **ZLIB 模式数据区**：每个张量的数据块为紧凑数据的压缩后大小。

#### 9.3 V4.21 修订摘要

| 修订项 | 旧版（V4.20） | 新版（V4.21） |
|:---|:---|:---|
| Tensor 布局 | 带行间 Padding | **强制紧凑**，无 Padding |
| `row_stride` | 对齐值（128/256 字节对齐） | **紧凑值**：`W×C×sizeof(dtype)` |
| `nbytes` | `N×H×aligned_row_stride` | **紧凑值**：`numel × sizeof(dtype)` |
| 保存流程 | 须先构建"干净缓冲区" | **直接使用 `tensor.data()`** |
| 加载流程 | 须按行展开重建对齐 | **直接读入紧凑 Tensor** |
| CRC32 计算 | 对有效数据逐行计算 | **对整个 `nbytes` 连续数据计算** |
| DTensor兼容性 | 无此概念 | **Tensor↔DTensor自动转换** |

---

**文档版本**：V4.20（V4.21 修订）
**最后更新**：2026-05-06
**状态**：正式规范
**技术觉醒团队 版权所有**
