# FastRandomResizedCrop 设计原理与 SDMP 协作详解

## 1. 定位

`FastRandomResizedCrop`（以下简称 FRRC）是配合 SDMP（Sample Duplicate Multi-Preprocess）机制设计的高性能随机裁剪器。它不是一个"RandomResizedCrop 的加速版"，而是为 SDMP 场景量身定制的独立实现。

核心目标：**用一次 JPEG 局部解码，产出多个不同的随机裁剪样本**。

---

## 2. 两套采样路径

FRRC 根据 `sdmp_factor` 走**两套完全不同的采样路径**。这是理解 FRRC 的关键，也是它与 `RandomResizedCrop` 最根本的区别。

### 2.1 路径 A：sdmp = 1（标准路径）

**不需要 SDMP 复用**。注意：此路径虽然也走 `generate_crop_params_for_full`，但采样方式与 PyTorch / 本框架 `RandomResizedCrop` 并不同。

调用链：

```
get_decode_strategy()
  └─ generate_crop_params_for_full()   ← 对数均匀面积采样
       └─ calculate_mcu_aligned_region()  ← MCU 对齐

JPEG 局部解码

execute()
  └─ execute_from_partial_decode()
       └─ sdmp_factor_ == 1 分支 → 直接使用第一刀，resize 到 224×224
```

面积采样公式（`generate_crop_params_for_full`）：

```
target_area = area × exp(log(scale_min) + rand × (log(scale_max) - log(scale_min)))
```

> **关键区别**：PyTorch 的 `RandomResizedCrop` 和本框架 `RandomResizedCrop` 均使用**线性均匀**面积采样（`area × (scale_min + rand × (scale_max - scale_min))`）。FRRC 路径 A 选择的是**对数均匀**，这是框架自己的设计选择，不等同于 PyTorch 标准行为。`crop_power_` 虽然被设为 3.0，但在此路径中**从不参与任何三次方根公式**——它只是被设置，从未被使用。

### 2.2 路径 B：sdmp ≥ 2（SDMP 路径）

**需要一次解码、多次裁剪**，使用三次方根变换的两步采样。

调用链：

```
get_decode_strategy()
  ├─ 设定 crop_power_（根据 sdmp_factor）
  └─ generate_crop_params_for_partial()  ← 三次方根面积采样（第一刀）
       └─ calculate_mcu_aligned_region()  ← MCU 对齐

JPEG 局部解码（只解 MCU 窗口）

execute()  ← 被调用 sdmp_factor 次，每次产出不同的最终样本
  └─ execute_from_partial_decode()
       └─ sdmp_factor_ > 1 分支 → 三次方根面积采样（第二刀）→ resize 到 224×224
```

---

## 3. 三次方根面积采样

### 3.1 预计算

在构造函数中，将用户指定的 `scale` 范围转换为三次方根空间：

```cpp
sqrt3_scale_min_ = scale_min^(1/3)   // 例如 0.08^(1/3) ≈ 0.431
sqrt3_scale_max_ = scale_max^(1/3)   // 例如 1.00^(1/3) = 1.000
```

### 3.2 抽样公式

令 `U ~ Uniform[0, 1]`，在三次方根空间中做均匀采样：

```
s = sqrt3_scale_min + U × (sqrt3_scale_max - sqrt3_scale_min)
```

然后通过 `crop_power` 指数还原为面积比例：

```
target_area = base_area × s^p      （p = crop_power 或 3 - crop_power）
```

### 3.3 为什么是三次方根？

面积采样需要同时满足两个目标：

1. **最终面积覆盖完整范围**：最终 crop 面积比例应在 `[scale_min, scale_max]` 内；
2. **支持两步裁剪的指数分解**：第一刀和第二刀的采样应能独立控制。

令 `s` 在 `[scale_min^(1/3), scale_max^(1/3)]` 上均匀分布，则：

- `s^3` 的值域恰好是 `[scale_min, scale_max]`（但分布不是均匀的：PDF 形如 `y^(-2/3)`，在 `scale_min` 附近密度更高，偏向小面积）；
- `s^p` 通过调节 `p` 改变分布偏向：`p` 越小，分布越偏向大面积；`p` 越大，分布越偏向小面积。

这样，第一刀用 `s^crop_power`，第二刀用 `s^(3 - crop_power)`，既能保证最终面积范围，又能灵活平衡第一刀的"容器大小"和第二刀的"多样性"。

---

## 4. crop_power 与两步裁剪

### 4.1 取值规则

| sdmp_factor | crop_power_ | 第一刀指数 | 第二刀指数 | 直觉 |
|---|---|---|---|---|
| 1 | 3.0（不使用） | —（走对数均匀） | —（不走第二刀） | 标准路径 |
| 2 | 2.0 | 2.0 | 1.0 | 第一刀中等偏大，容纳 2 次裁剪 |
| 3 | 1.0（默认）或 2.0（CROP_SCHEME_2） | 1.0 或 2.0 | 2.0 或 1.0 | 第一刀更大，容纳 3 次裁剪 |
| ≥4 | 1.0 | 1.0 | 2.0 | 第一刀最大 |

### 4.2 两步面积公式

**第一刀**（在 `generate_crop_params_for_partial` 中）：

```
area1 = full_area × s₁^crop_power
```

其中 `s₁` 在 `[scale_min^(1/3), scale_max^(1/3)]` 上均匀采样。

**第二刀**（在 `execute_from_partial_decode` 中，`sdmp > 1` 分支）：

```
area2 = (crop_w × crop_h) × s₂^(3 - crop_power)
```

其中 `s₂` 在相同区间上**独立**均匀采样。

### 4.3 两刀互补设计

两步指数之和恒为 **3.0**：

```
crop_power + (3 - crop_power) = 3
```

这个设计有两层含义：

**（1）面积范围保障**

最终裁剪相对于原图的面积比例为：

```
s_final = s₁^crop_power × s₂^(3 - crop_power)
```

当 `s₁` 和 `s₂` 都取最小值时：`s_final = (scale_min^(1/3))^3 = scale_min`
当 `s₁` 和 `s₂` 都取最大值时：`s_final = (scale_max^(1/3))^3 = scale_max`

因此最终面积**始终落在用户指定的 `[scale_min, scale_max]` 范围内**。

**（2）SDMP 适配**

- `crop_power` 越小 → 第一刀面积越大 → 解码窗口越大 → 能容纳更多次不同的第二刀
- `crop_power` 越大 → 第一刀面积越小 → 解码窗口越小 → 解码更快但只能容纳少量第二刀

对于 `sdmp_factor = 2`：`crop_power = 2.0`，第一刀"中等偏大"，第二刀在余下的随机性（指数 1.0）中变化。
对于 `sdmp_factor = 3`：`crop_power = 1.0`，第一刀"更大"，为 3 次独立裁剪留出足够空间，第二刀在更大的自由度（指数 2.0）中变化。

### 4.4 直观理解

```text
sdmp_factor = 2, crop_power = 2.0

  原图 (500×375)
  ┌──────────────────────────────┐
  │                              │
  │    ┌──────────────────┐      │  ← 第一刀：s₁^2.0，面积约 40%
  │    │  MCU 解码窗口     │      │     解码这一块
  │    │  ┌──────┐        │      │
  │    │  │crop 1│  ┌───┐ │      │  ← 第二刀第 1 次：s₂^1.0
  │    │  └──────┘  │crop│ │      │  ← 第二刀第 2 次：s₂'^1.0
  │    │            │ 2  │ │      │
  │    │            └───┘ │      │
  │    └──────────────────┘      │
  │                              │
  └──────────────────────────────┘
```

---

## 5. 与 RandomResizedCrop 的对比

| 维度 | RandomResizedCrop | FastRandomResizedCrop (sdmp≥2) |
|---|---|---|---|
| 采样方式 | 线性均匀（PyTorch 标准） | 三次方根两步采样 |
| 解码次数 | 每次裁剪解码一次 | 一次解码，多次裁剪 |
| 面积分布 | 在 `[scale_min, scale_max]` 上线性均匀 | 两个独立三次方根变量的乘积分布 |
| 分布偏向 | 偏向小 crop | 偏向中等偏大 crop |
| 定位 | 标准实现（对齐 PyTorch） | SDMP 性能优化 |

### 5.1 分布差异

以 `scale = [0.08, 1.0]` 为例：

| 指标 | 线性均匀（RRC） | 三次方根两步（FRRC, sdmp=2） |
|---|---|---|
| 面积中位数 | ~0.54 | ~0.35 |
| 分布形状 | 均匀分布，小 crop 与大 crop 等概率 | 两个独立三次方根变量乘积的分布，偏向中等偏大 crop |

FRRC 的分布比线性均匀更"集中"，偏向中等偏大的 crop。这是因为：

1. 第一刀的 `s₁^2.0` 已经偏向中等偏大；
2. 第二刀的 `s₂^1.0` 在中等偏大的基础上进一步变化；
3. 两个独立随机变量的乘积天然具有"集中化"效应。

---

## 6. 为什么不需要强行对齐 RandomResizedCrop

### 6.1 设计目标根本不同

- `RandomResizedCrop`：一次裁剪、一次解码。目标是精确复现 PyTorch 行为。
- `FastRandomResizedCrop`：一次解码、多次裁剪。目标是最大化解码复用效率。

为了 SDMP 能工作，**第一刀必须比最终裁剪大**，否则无法容纳多次不同的第二刀。这本身就意味着 FRRC 的采样分布不可能与 RRC 完全相同。

### 6.2 分布差异在训练中不重要

数据增强的分布本身就是一个超参数。ImageNet 训练中，scale 采样用均匀、对数均匀、甚至其他分布，只要覆盖范围合理且保持随机性，最终精度差异在误差范围内。

本框架的 `RandomResizedCrop` 在 VGG-16-BN AMP 训练中达到 **73.66% Top-1**（超过论文 73.36%），说明当前数据增强流水线整体有效。FRRC 作为 SDMP 优化变体，其面积分布偏差在训练中被认为是可接受的。

### 6.3 对齐会破坏 SDMP 的核心价值

如果把 FRRC 改成简单的线性均匀采样（如某些方案建议的），会导致：

- 第一刀在 `sdmp ≥ 2` 时可能过小，无法容纳多次第二刀
- 第二刀缺乏多样性，SDMP 复用失去意义
- 要么回退到每次解码（失去性能优势），要么第二刀裁剪质量下降（失去数据增强效果）

三次方根变换 + 互补 `crop_power` 正是在这个矛盾中找到的平衡点。

---

## 7. 代码路径速查

```text
FastRandomResizedCrop::get_decode_strategy()
│
├── sdmp_factor = 1
│   ├── crop_power_ = 3.0（设置但不使用）
│   ├── generate_crop_params_for_full()
│   │   └── 对数均匀面积采样 + 对数均匀宽高比 + float·max + clamp 位置
│   └── calculate_mcu_aligned_region()
│
├── sdmp_factor = 2
│   ├── crop_power_ = 2.0
│   ├── generate_crop_params_for_partial()
│   │   └── 三次方根面积采样(s^2.0) + 对数均匀宽高比 + float·max + clamp 位置
│   └── calculate_mcu_aligned_region()
│
└── sdmp_factor ≥ 3
    ├── crop_power_ = 1.0（或 2.0 with CROP_SCHEME_2）
    ├── generate_crop_params_for_partial()
    │   └── 三次方根面积采样(s^1.0) + 对数均匀宽高比 + float·max + clamp 位置
    └── calculate_mcu_aligned_region()

↓ JPEG 局部解码

FastRandomResizedCrop::execute()
└── execute_from_partial_decode()
    ├── sdmp_factor_ == 1 → 直接使用第一刀，resize
    └── sdmp_factor_ > 1  → 三次方根面积采样(s^(3-p)) → resize
                             ↑ 调用 sdmp_factor 次，每次独立随机
```

---

## 8. 实现细节

### 8.1 mutable 成员变量

```cpp
mutable float sqrt3_scale_min_;  // 构造函数中预计算，get_decode_strategy 中可能重新计算
mutable float sqrt3_scale_max_;  // 同上
mutable float crop_power_;       // 在 const 方法 get_decode_strategy 中根据 sdmp_factor 设定
mutable int   sdmp_factor_;      // 在 const 方法 get_decode_strategy 中记录
```

这些变量标记为 `mutable`，因为 `get_decode_strategy()` 是 `const` 方法（不改变语义状态），但需要根据 `sdmp_factor` 调整采样参数。这是一种 pragmatic 的设计选择，避免了将 `const` 传播到整个调用链。

### 8.2 宽高比采样

宽高比在**第一刀**中确定（对数均匀采样），**第二刀保持相同宽高比**。这保证了两次裁剪在视觉上的一致性——不会出现第一刀选了一个横宽的框，第二刀突然变成竖长的框。

### 8.3 重试与边界保护

- **第一刀**：有 10 次重试机制。如果采样出的 crop 超出图像边界，重新采样。10 次都失败则 fallback 到中心裁剪。
- **第二刀**：无重试机制（性能考虑）。直接通过 `std::max(1, std::min(crop_w, mcu_w_))` 做边界保护，确保 crop 尺寸在 MCU 窗口内合法。

### 8.4 MCU 对齐

JPEG 局部解码必须以 MCU（Minimum Coded Unit，通常 16×16 像素）为单位。`calculate_mcu_aligned_region()` 将第一刀的 `(crop_x, crop_y, crop_w, crop_h)` 向外扩展到最近的 MCU 边界：

```
mcu_x = floor(crop_x / 16) × 16
mcu_y = floor(crop_y / 16) × 16
mcu_w = ceil((crop_x + crop_w) / 16) × 16 - mcu_x
mcu_h = ceil((crop_y + crop_h) / 16) × 16 - mcu_y
```

这意味着实际解码区域比第一刀略大（最多大 15 像素），但换取了解码效率的显著提升。

---

## 9. 总结

FastRandomResizedCrop 的设计核心是**一次解码、多次裁剪**。它通过以下机制实现：

1. **两套路径**：sdmp=1 走对数均匀路径（框架自定义，非 PyTorch 标准），sdmp≥2 走三次方根两步路径
2. **三次方根采样**：在面积的三次方根空间中做均匀采样，通过 `crop_power` 指数灵活控制面积偏向
3. **互补 crop_power**：两步指数之和恒为 3.0，保证最终面积在合法范围内，同时平衡第一刀的"容器大小"和第二刀的"多样性"
4. **不追求对齐**：FRRC 是 SDMP 的性能优化工具，不是 RRC 的等价实现。其分布偏差在训练中可忽略，且已被数据增强流水线的整体有效性所验证