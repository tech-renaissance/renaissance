# MemoryPlan 最终设计方案

> 基于 [REGION_FINAL.md](./REGION_FINAL.md) V2.9，综合 S/K/D 三轮讨论，最终裁定。
>
> 设计目标：**最精简、最正确、零冗余。**

---

## 一、核心设计原则

### 从旧到新的质变

| 旧版 (memory_plan_legacy.h) | 新版 (本方案) |
|---|---|
| 13 个 Region，PARAM 区 FP32/FP16 混排 | 65 个 Region，精度×功能完全分离 |
| 两遍布局（预计算→预留→回填） | 一遍线性累加 |
| FP32 = 2×FP16 配对公式 | 删除，Region 独立 |
| `alloc_param(pclass, bucket, model_tag, paired_id)` 6参数 | `alloc(shape, dtype, region)` 3参数 |
| `is_bn`/`is_first_layer` 间接推断分桶 | Region 直接编码功能类型 |
| 5 重校验含配对验证 | 4 重校验，纯连续性+层序对齐 |
| `ParamClass`/`TensorRole`/`ParamBucket`/`model_tag`/`paired_id` | 全部删除 |

### 三条铁律

1. **65 个 Region 锚点**：每个 Region 直接编码功能类型，调用方不再间接推断
2. **finalize 退化为线性累加**：cursor 从 001 走到 065，旧版两遍布局和配对公式全部废除
3. **15 个批量操作范围预计算**：finalize 时查表拼出，CUDA Graph/NCCL 零运行时开销

---

## 二、数据结构

### 2.1 Region 枚举（放在 `types.h`）

```cpp
enum class Region : uint8_t {
    B_PREV_MEAN  = 0,   // 001
    B_PREV_VAR,          // 002
    B_NEXT_MEAN,         // 003
    B_NEXT_VAR,          // 004

    W_EQ_BIAS,           // 005
    W_EQ_SCALE,          // 006
    W_BN_BIAS,           // 007
    W_BN_WEIGHT,         // 008
    W_FC_BIAS,           // 009
    W_FC_WEIGHT,         // 010
    W_FIRST_CONV,        // 011
    W_DEEP_CONV,         // 012

    E_BN_BIAS,           // 013
    E_BN_WEIGHT,         // 014
    E_FC_BIAS,           // 015
    E_FC_WEIGHT,         // 016
    E_FIRST_CONV,        // 017
    E_DEEP_CONV,         // 018
    E_FC_WEIGHT_FP16,    // 019
    E_FIRST_CONV_FP16,   // 020
    E_DEEP_CONV_FP16,    // 021

    A_FC_WEIGHT,         // 022
    A_FIRST_CONV,        // 023
    A_DEEP_CONV,         // 024

    G_BN_BIAS,           // 025  桶2起点
    G_BN_WEIGHT,         // 026  桶2
    G_FC_BIAS,           // 027  桶2
    G_FC_WEIGHT,         // 028  桶2
    G_FIRST_CONV,        // 029  桶2终点
    G_DEEP_CONV,         // 030  桶1
    G_FC_WEIGHT_FP16,    // 031
    G_FIRST_CONV_FP16,   // 032
    G_DEEP_CONV_FP16,    // 033

    M_BN_BIAS,           // 034
    M_BN_WEIGHT,         // 035
    M_FC_BIAS,           // 036
    M_FC_WEIGHT,         // 037
    M_FIRST_CONV,        // 038
    M_DEEP_CONV,         // 039

    V_BN_BIAS,           // 040
    V_BN_WEIGHT,         // 041
    V_FC_BIAS,           // 042
    V_FC_WEIGHT,         // 043
    V_FIRST_CONV,        // 044
    V_DEEP_CONV,         // 045

    N_FC_WEIGHT,         // 046
    N_FIRST_CONV,        // 047
    N_DEEP_CONV,         // 048

    I_A_LABEL,           // 049
    I_A_DATA,            // 050
    I_B_LABEL,           // 051
    I_B_DATA,            // 052

    F_FEATURE_FP32,      // 053
    F_GRAD_SLOT_FP32,    // 054
    F_FEATURE_FP16,      // 055
    F_GRAD_SLOT_FP16,    // 056

    S_SCALAR_FP32,       // 057
    S_SCALAR_FP16,       // 058
    S_SCALAR_INT32,      // 059
    S_SCALAR_INT8,       // 060
    S_MASK,              // 061

    T_TEMP_FP32,         // 062
    T_TEMP_FP16,         // 063
    T_TEMP_INT32,        // 064
    T_TEMP_INT8,         // 065

    NUM_REGIONS = 65
};
```

**同时删除旧 `ParamBucket` 枚举**（types.h 中），新版不再需要通信分桶枚举。

### 2.2 PlanConfig

```cpp
struct PlanConfig {
    bool amp_enabled  = false;
    bool bn_folded    = true;
    bool use_lars     = false;
    bool use_adam     = false;
    bool use_momentum = true;    // Adam 隐含需要，configure 时自动校验
    bool has_ema      = false;
    int  num_models   = 1;      // 预留，当前固定 1
    bool need_mask    = false;
    bool need_temp    = false;
};
```

> **校验规则**：`use_adam = true` 时 `use_momentum` 必须也为 `true`（Adam 同时需要 M_* 和 V_* 两个系列）。

### 2.3 BatchOp 枚举

```cpp
enum class BatchOp : uint8_t {
    BN_STATS_COPY = 0,
    CAST_W32_TO_W16,
    ZERO_GRAD,
    CAST_G16_TO_G32_FC,
    CAST_G16_TO_G32_FIRST,
    CAST_G16_TO_G32_DEEP,
    NAN_CHECK_ALL_G,
    ALLREDUCE_BUCKET1,
    ALLREDUCE_BUCKET2,
    UPDATE_BN_PARAM_AND_FC_BIAS,
    UPDATE_WEIGHT,
    CAST_EMA32_TO_EMA16,
    EMA_PARAM_UPDATE,
    SEMA_SWITCH,
    BN_STATS_ALLREDUCE,
    NUM_BATCH_OPS = 15
};
```

### 2.4 范围与查询结构体

```cpp
struct OpSegment {
    uint64_t start = 0;   // inclusive
    uint64_t end   = 0;   // exclusive
};

struct BatchOpRange {
    std::vector<OpSegment> inputs;
    std::vector<OpSegment> outputs;
};

struct RegionInfo {
    uint64_t base_offset = 0;
    uint64_t total_bytes = 0;
};

struct CommRange {
    uint64_t offset = 0;
    uint64_t size   = 0;
};

struct BNStatsBuffers {
    DTensor prev_mean, prev_var, next_mean, next_var;
};

struct InputBuffers {
    DTensor label_a, data_a, label_b, data_b;
};
```

> **设计决策**：使用 `OpSegment {start, end}` 而非 `pair<uint64_t, uint64_t>`，语义更清晰。`CommRange {offset, size}` 保留独立——它是 NCCL API 的直接映射。

---

## 三、MemoryPlan 类定义

```cpp
// ============================================================================
// memory_plan.h — 新版 MemoryPlan
// ============================================================================

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/tensor/distributed_tensor.h"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>

namespace tr {

class MemoryPlan {
public:
    explicit MemoryPlan(const PlanConfig& config);
    ~MemoryPlan();

    MemoryPlan(const MemoryPlan&) = delete;
    MemoryPlan& operator=(const MemoryPlan&) = delete;
    MemoryPlan(MemoryPlan&&) = delete;
    MemoryPlan& operator=(MemoryPlan&&) = delete;

    // ===================================================================
    // 一、语义化分配接口（内部硬编码 Region，杜绝错区）
    // ===================================================================

    // —— B: BN 统计量（一次性分配 4 个 Region，Shape 必须一致）——
    BNStatsBuffers alloc_bn_stats(const Shape& shape);

    // —— W: 主模型权重 ——
    DTensor alloc_eq_bias(const Shape& shape);
    DTensor alloc_eq_scale(const Shape& shape);
    DTensor alloc_bn_bias(const Shape& shape);
    DTensor alloc_bn_weight(const Shape& shape);
    DTensor alloc_fc_bias(const Shape& shape);
    DTensor alloc_fc_weight(const Shape& shape);
    DTensor alloc_first_conv_weight(const Shape& shape);
    DTensor alloc_deep_conv_weight(const Shape& shape);

    // —— E: EMA 权重 ——
    DTensor alloc_ema_bn_bias(const Shape& shape);
    DTensor alloc_ema_bn_weight(const Shape& shape);
    DTensor alloc_ema_fc_bias(const Shape& shape);
    DTensor alloc_ema_fc_weight(const Shape& shape);
    DTensor alloc_ema_first_conv(const Shape& shape);
    DTensor alloc_ema_deep_conv(const Shape& shape);
    DTensor alloc_ema_fc_weight_fp16(const Shape& shape);
    DTensor alloc_ema_first_conv_fp16(const Shape& shape);
    DTensor alloc_ema_deep_conv_fp16(const Shape& shape);

    // —— A: AMP FP16 权重 ——
    DTensor alloc_amp_fc_weight(const Shape& shape);
    DTensor alloc_amp_first_conv(const Shape& shape);
    DTensor alloc_amp_deep_conv(const Shape& shape);

    // —— G: 梯度 ——
    DTensor alloc_grad_bn_bias(const Shape& shape);
    DTensor alloc_grad_bn_weight(const Shape& shape);
    DTensor alloc_grad_fc_bias(const Shape& shape);
    DTensor alloc_grad_fc_weight(const Shape& shape);
    DTensor alloc_grad_first_conv(const Shape& shape);
    DTensor alloc_grad_deep_conv(const Shape& shape);
    DTensor alloc_grad_fc_weight_fp16(const Shape& shape);
    DTensor alloc_grad_first_conv_fp16(const Shape& shape);
    DTensor alloc_grad_deep_conv_fp16(const Shape& shape);

    // —— M: 一阶动量 ——
    DTensor alloc_momentum_bn_bias(const Shape& shape);
    DTensor alloc_momentum_bn_weight(const Shape& shape);
    DTensor alloc_momentum_fc_bias(const Shape& shape);
    DTensor alloc_momentum_fc_weight(const Shape& shape);
    DTensor alloc_momentum_first_conv(const Shape& shape);
    DTensor alloc_momentum_deep_conv(const Shape& shape);

    // —— V: 二阶动量 ——
    DTensor alloc_velocity_bn_bias(const Shape& shape);
    DTensor alloc_velocity_bn_weight(const Shape& shape);
    DTensor alloc_velocity_fc_bias(const Shape& shape);
    DTensor alloc_velocity_fc_weight(const Shape& shape);
    DTensor alloc_velocity_first_conv(const Shape& shape);
    DTensor alloc_velocity_deep_conv(const Shape& shape);

    // —— N: LARS 范数 ——
    DTensor alloc_norm_fc_weight(const Shape& shape);
    DTensor alloc_norm_first_conv(const Shape& shape);
    DTensor alloc_norm_deep_conv(const Shape& shape);

    // —— I: 输入缓冲区 ——
    InputBuffers alloc_input_buffers();
    InputBuffers alloc_input_buffers(const Shape& label_shape,
                                     const Shape& data_shape,
                                     DType dtype);

    // —— F: 特征图与梯度槽 ——
    DTensor alloc_feature(const Shape& shape, DType dtype);
    DTensor alloc_grad_slot(const Shape& shape, DType dtype, int slot_idx);

    // —— S: 标量与掩码 ——
    DTensor alloc_scalar(DType dtype = DType::FP32);
    DTensor alloc_mask(const Shape& shape);

    // —— T: 临时张量 ——
    DTensor alloc_temp(const Shape& shape, DType dtype);

    // ===================================================================
    // 二、通用分配接口（模型编译场景核心入口，对外开放）
    // ===================================================================
    [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype, Region region);

    // ===================================================================
    // 三、布局锁定
    // ===================================================================
    void finalize();
    [[nodiscard]] bool is_finalized() const noexcept { return finalized_; }
    [[nodiscard]] uint64_t total_bytes() const;

    // ===================================================================
    // 四、运行期查询
    // ===================================================================
    [[nodiscard]] const DTensor& get_dtensor(int32_t id) const;
    [[nodiscard]] bool has_dtensor(int32_t id) const noexcept;
    [[nodiscard]] const std::vector<DTensor>& dtensors() const;

    [[nodiscard]] RegionInfo get_region_info(Region region) const;
    [[nodiscard]] const std::vector<int32_t>& get_ids_by_region(Region region) const;

    [[nodiscard]] const BatchOpRange& get_batch_op_range(BatchOp op) const;

    [[nodiscard]] CommRange get_comm_range_bucket1() const;
    [[nodiscard]] CommRange get_comm_range_bucket2() const;

    // ===================================================================
    // 五、调试与校验
    // ===================================================================
    [[nodiscard]] std::string dump_layout() const;
    void validate() const;

private:
    struct Entry {
        DTensor dt;               // offset=0 pre-finalize, finalize 时赋值
        int8_t  grad_slot_idx = -1;
    };

    PlanConfig config_;
    bool finalized_ = false;
    uint64_t total_bytes_ = 0;
    int32_t next_id_ = 0;

    std::vector<Entry> entries_;
    std::unordered_map<int32_t, size_t> id_to_idx_;

    std::array<std::vector<int32_t>,
               static_cast<size_t>(Region::NUM_REGIONS)> region_dt_ids_{};
    std::array<RegionInfo,
               static_cast<size_t>(Region::NUM_REGIONS)> region_infos_{};
    std::array<BatchOpRange,
               static_cast<size_t>(BatchOp::NUM_BATCH_OPS)> batch_op_ranges_{};

    std::vector<DTensor> dtensor_cache_;
    std::array<int32_t, 4> grad_slot_ids_{-1, -1, -1, -1};

    DTensor alloc_impl(const Shape& shape, DType dtype, Region region);

    void validate_config() const;
    void build_batch_op_ranges();
    void validate_region_order() const;
    void validate_contiguity() const;
    void validate_layer_correspondence() const;
    void validate_alignment() const;

    OpSegment make_region_segment(Region r) const;
    OpSegment make_region_segment(Region start, Region end) const;
    bool is_condition_enabled(Region r) const;

    friend class TaskBase;
    friend class Compiler;
};

} // namespace tr
```

---

## 四、关键算法实现

### 4.1 alloc_impl — 通用内部分配

```cpp
DTensor MemoryPlan::alloc_impl(const Shape& shape, DType dtype, Region region) {
    TR_CHECK(!finalized_, ValueError, "Cannot alloc after finalize");
    TR_CHECK(is_condition_enabled(region), ValueError,
             "Region not enabled under current PlanConfig");

    int32_t id = next_id_++;
    DTensor dt(id, shape, dtype, region, 0);

    entries_.push_back({dt, -1});
    id_to_idx_[id] = entries_.size() - 1;
    region_dt_ids_[static_cast<size_t>(region)].push_back(id);

    return dt;
}
```

> `region_dt_ids_` 与 `entries_` 在同一处更新，原子操作，不存在一致性问题。

### 4.2 alloc_bn_stats

```cpp
BNStatsBuffers MemoryPlan::alloc_bn_stats(const Shape& shape) {
    return {
        alloc_impl(shape, DType::FP32, Region::B_PREV_MEAN),
        alloc_impl(shape, DType::FP32, Region::B_PREV_VAR),
        alloc_impl(shape, DType::FP32, Region::B_NEXT_MEAN),
        alloc_impl(shape, DType::FP32, Region::B_NEXT_VAR),
    };
}
```

> 四个 Region 一次性分配，Shape 完全相同，杜绝错序和漏配。

### 4.3 alloc_feature

```cpp
DTensor MemoryPlan::alloc_feature(const Shape& shape, DType dtype) {
    if (dtype == DType::FP16) {
        TR_CHECK(config_.amp_enabled, ValueError,
                 "FP16 feature requires amp_enabled");
        return alloc_impl(shape, DType::FP16, Region::F_FEATURE_FP16);
    }
    TR_CHECK(!config_.amp_enabled, ValueError,
             "FP32 feature requires !amp_enabled");
    return alloc_impl(shape, DType::FP32, Region::F_FEATURE_FP32);
}
```

### 4.4 alloc_grad_slot

```cpp
DTensor MemoryPlan::alloc_grad_slot(const Shape& shape, DType dtype, int slot_idx) {
    TR_CHECK(slot_idx >= 0 && slot_idx < 4, IndexError,
             "slot_idx=" << slot_idx);

    int32_t existing = grad_slot_ids_[slot_idx];
    if (existing >= 0) {
        TR_CHECK(get_dtensor(existing).dtype == dtype, ValueError,
                 "GradSlot dtype mismatch");
        return get_dtensor(existing);
    }

    Region region = (dtype == DType::FP16)
        ? Region::F_GRAD_SLOT_FP16 : Region::F_GRAD_SLOT_FP32;

    DTensor dt = alloc_impl(shape, dtype, region);
    entries_.back().grad_slot_idx = static_cast<int8_t>(slot_idx);
    grad_slot_ids_[slot_idx] = dt.id;
    return dt;
}
```

### 4.5 validate_config

```cpp
void MemoryPlan::validate_config() const {
    if (config_.use_adam && !config_.use_momentum) {
        TR_THROW(ValueError, "use_adam=true requires use_momentum=true");
    }
}
```

### 4.6 finalize — 一遍线性累加

```cpp
void MemoryPlan::finalize() {
    TR_CHECK(!finalized_, ValueError, "MemoryPlan already finalized");
    validate_config();

    uint64_t cursor = 0;

    for (size_t ri = 0; ri < static_cast<size_t>(Region::NUM_REGIONS); ++ri) {
        auto& info = region_infos_[ri];
        info.base_offset = cursor;

        for (int32_t dt_id : region_dt_ids_[ri]) {
            auto& entry = entries_[id_to_idx_.at(dt_id)];
            entry.dt.offset = cursor;
            cursor += entry.dt.slot_bytes();
        }

        info.total_bytes = cursor - info.base_offset;
        // 空区：base = cursor(不变), total_bytes = 0
    }

    total_bytes_ = cursor;

    dtensor_cache_.clear();
    dtensor_cache_.reserve(entries_.size());
    for (const auto& entry : entries_) {
        dtensor_cache_.push_back(entry.dt);
    }

    build_batch_op_ranges();

    validate_region_order();
    validate_contiguity();
    validate_layer_correspondence();
    validate_alignment();

    finalized_ = true;
}
```

> **为何不需要旧版两遍算法**：旧版 PARAM 区 FP32/FP16 混排，需要"预计算 FP16 → 预留 FP32 → 布局 FP16 → 回填 FP32"。新版 FP32（W_）和 FP16（A_）是不同的 Region，各自独立连续，cursor 线性走过即可。

### 4.7 build_batch_op_ranges

```cpp
void MemoryPlan::build_batch_op_ranges() {
    auto seg  = [this](Region r) { return make_region_segment(r); };
    auto seg2 = [this](Region s, Region e) { return make_region_segment(s, e); };

    // OP-1: BN_STATS_COPY — next → prev（两次独立 memcpy）
    auto& r1 = batch_op_ranges_[0];
    r1.inputs.push_back(seg(Region::B_NEXT_MEAN));
    r1.inputs.push_back(seg(Region::B_NEXT_VAR));
    r1.outputs.push_back(seg(Region::B_PREV_MEAN));
    r1.outputs.push_back(seg(Region::B_PREV_VAR));

    // OP-2: CAST_W32_TO_W16 — 010-012 → 022-024
    auto& r2 = batch_op_ranges_[1];
    r2.inputs.push_back(seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));
    r2.outputs.push_back(seg2(Region::A_FC_WEIGHT, Region::A_DEEP_CONV));

    // OP-3: ZERO_GRAD — 025-033 全体清零
    auto& r3 = batch_op_ranges_[2];
    r3.outputs.push_back(seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16));

    // OP-4: CAST_G16_TO_G32_FC — 031 → 028
    auto& r4 = batch_op_ranges_[3];
    r4.inputs.push_back(seg(Region::G_FC_WEIGHT_FP16));
    r4.outputs.push_back(seg(Region::G_FC_WEIGHT));

    // OP-5: CAST_G16_TO_G32_FIRST — 032 → 029
    auto& r5 = batch_op_ranges_[4];
    r5.inputs.push_back(seg(Region::G_FIRST_CONV_FP16));
    r5.outputs.push_back(seg(Region::G_FIRST_CONV));

    // OP-6: CAST_G16_TO_G32_DEEP — 033 → 030
    auto& r6 = batch_op_ranges_[5];
    r6.inputs.push_back(seg(Region::G_DEEP_CONV_FP16));
    r6.outputs.push_back(seg(Region::G_DEEP_CONV));

    // OP-7: NAN_CHECK_ALL_G — 025-030
    auto& r7 = batch_op_ranges_[6];
    r7.inputs.push_back(seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV));

    // OP-8: ALLREDUCE_BUCKET1 — 030 in-place
    auto& r8 = batch_op_ranges_[7];
    r8.inputs.push_back(seg(Region::G_DEEP_CONV));
    r8.outputs = r8.inputs;

    // OP-9: ALLREDUCE_BUCKET2 — 025-029 in-place
    auto& r9 = batch_op_ranges_[8];
    r9.inputs.push_back(seg2(Region::G_BN_BIAS, Region::G_FIRST_CONV));
    r9.outputs = r9.inputs;

    // OP-10: UPDATE_BN_PARAM_AND_FC_BIAS
    auto& r10 = batch_op_ranges_[9];
    r10.inputs.push_back(seg2(Region::W_BN_BIAS, Region::W_FC_BIAS));
    r10.inputs.push_back(seg2(Region::G_BN_BIAS, Region::G_FC_BIAS));
    r10.inputs.push_back(seg2(Region::M_BN_BIAS, Region::M_FC_BIAS));
    if (config_.use_adam)
        r10.inputs.push_back(seg2(Region::V_BN_BIAS, Region::V_FC_BIAS));
    r10.outputs.push_back(seg2(Region::W_BN_BIAS, Region::W_FC_BIAS));
    r10.outputs.push_back(seg2(Region::M_BN_BIAS, Region::M_FC_BIAS));
    if (config_.use_adam)
        r10.outputs.push_back(seg2(Region::V_BN_BIAS, Region::V_FC_BIAS));

    // OP-11: UPDATE_WEIGHT
    auto& r11 = batch_op_ranges_[10];
    r11.inputs.push_back(seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));
    r11.inputs.push_back(seg2(Region::G_FC_WEIGHT, Region::G_DEEP_CONV));
    r11.inputs.push_back(seg2(Region::M_FC_WEIGHT, Region::M_DEEP_CONV));
    if (config_.use_adam)
        r11.inputs.push_back(seg2(Region::V_FC_WEIGHT, Region::V_DEEP_CONV));
    if (config_.use_lars)
        r11.inputs.push_back(seg2(Region::N_FC_WEIGHT, Region::N_DEEP_CONV));
    r11.outputs.push_back(seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));
    r11.outputs.push_back(seg2(Region::M_FC_WEIGHT, Region::M_DEEP_CONV));
    if (config_.use_adam)
        r11.outputs.push_back(seg2(Region::V_FC_WEIGHT, Region::V_DEEP_CONV));

    // OP-12: CAST_EMA32_TO_EMA16 — 016-018 → 019-021
    auto& r12 = batch_op_ranges_[11];
    r12.inputs.push_back(seg2(Region::E_FC_WEIGHT, Region::E_DEEP_CONV));
    r12.outputs.push_back(seg2(Region::E_FC_WEIGHT_FP16, Region::E_DEEP_CONV_FP16));

    // OP-13: EMA_PARAM_UPDATE — W → E
    auto& r13 = batch_op_ranges_[12];
    r13.inputs.push_back(seg2(Region::W_BN_BIAS, Region::W_DEEP_CONV));
    r13.inputs.push_back(seg2(Region::E_BN_BIAS, Region::E_DEEP_CONV));
    r13.outputs.push_back(seg2(Region::E_BN_BIAS, Region::E_DEEP_CONV));

    // OP-14: SEMA_SWITCH — E → W
    auto& r14 = batch_op_ranges_[13];
    r14.inputs.push_back(seg2(Region::E_BN_BIAS, Region::E_DEEP_CONV));
    r14.outputs.push_back(seg2(Region::W_BN_BIAS, Region::W_DEEP_CONV));

    // OP-15: BN_STATS_ALLREDUCE — 003-004 in-place
    auto& r15 = batch_op_ranges_[14];
    r15.inputs.push_back(seg2(Region::B_NEXT_MEAN, Region::B_NEXT_VAR));
    r15.outputs = r15.inputs;
}
```

### 4.8 辅助方法

```cpp
OpSegment MemoryPlan::make_region_segment(Region r) const {
    TR_CHECK(static_cast<size_t>(r) < static_cast<size_t>(Region::NUM_REGIONS),
             IndexError, "Invalid region: " << static_cast<int>(r));
    auto& info = region_infos_[static_cast<size_t>(r)];
    return {info.base_offset, info.base_offset + info.total_bytes};
}

OpSegment MemoryPlan::make_region_segment(Region start, Region end) const {
    TR_CHECK(static_cast<size_t>(start) <= static_cast<size_t>(end), ValueError,
             "Invalid segment order: " << static_cast<int>(start)
             << " > " << static_cast<int>(end));
    auto& si = region_infos_[static_cast<size_t>(start)];
    auto& ei = region_infos_[static_cast<size_t>(end)];
    return {si.base_offset, ei.base_offset + ei.total_bytes};
}

RegionInfo MemoryPlan::get_region_info(Region region) const {
    TR_CHECK(static_cast<size_t>(region) < static_cast<size_t>(Region::NUM_REGIONS),
             IndexError, "Invalid region: " << static_cast<int>(region));
    return region_infos_[static_cast<size_t>(region)];
}

const std::vector<int32_t>& MemoryPlan::get_ids_by_region(Region region) const {
    TR_CHECK(static_cast<size_t>(region) < static_cast<size_t>(Region::NUM_REGIONS),
             IndexError, "Invalid region: " << static_cast<int>(region));
    return region_dt_ids_[static_cast<size_t>(region)];
}

const BatchOpRange& MemoryPlan::get_batch_op_range(BatchOp op) const {
    TR_CHECK(static_cast<size_t>(op) < static_cast<size_t>(BatchOp::NUM_BATCH_OPS),
             IndexError, "Invalid BatchOp: " << static_cast<int>(op));
    return batch_op_ranges_[static_cast<size_t>(op)];
}
```

### 4.9 通信分桶

```cpp
CommRange MemoryPlan::get_comm_range_bucket1() const {
    auto& r = region_infos_[static_cast<size_t>(Region::G_DEEP_CONV)];
    return {r.base_offset, r.total_bytes};
}

CommRange MemoryPlan::get_comm_range_bucket2() const {
    auto& s = region_infos_[static_cast<size_t>(Region::G_BN_BIAS)];
    auto& e = region_infos_[static_cast<size_t>(Region::G_FIRST_CONV)];
    return {s.base_offset,
            e.base_offset + e.total_bytes - s.base_offset};
}
```

### 4.10 is_condition_enabled

```cpp
bool MemoryPlan::is_condition_enabled(Region r) const {
    using R = Region;

    if (r == R::W_EQ_BIAS || r == R::W_EQ_SCALE)
        return config_.bn_folded;

    if (r >= R::E_BN_BIAS && r <= R::E_DEEP_CONV)
        return config_.has_ema;

    if (r >= R::E_FC_WEIGHT_FP16 && r <= R::E_DEEP_CONV_FP16)
        return config_.has_ema && config_.amp_enabled;

    if (r >= R::A_FC_WEIGHT && r <= R::A_DEEP_CONV)
        return config_.amp_enabled;

    if (r >= R::G_FC_WEIGHT_FP16 && r <= R::G_DEEP_CONV_FP16)
        return config_.amp_enabled;

    if (r >= R::M_BN_BIAS && r <= R::M_DEEP_CONV)
        return config_.use_momentum;

    if (r >= R::V_BN_BIAS && r <= R::V_DEEP_CONV)
        return config_.use_adam;

    if (r >= R::N_FC_WEIGHT && r <= R::N_DEEP_CONV)
        return config_.use_lars;

    if (r == R::F_FEATURE_FP32 || r == R::F_GRAD_SLOT_FP32)
        return !config_.amp_enabled;
    if (r == R::F_FEATURE_FP16 || r == R::F_GRAD_SLOT_FP16)
        return config_.amp_enabled;

    if (r == R::S_SCALAR_FP16) return config_.amp_enabled;
    if (r == R::S_MASK)        return config_.need_mask;

    if (r >= R::T_TEMP_FP32 && r <= R::T_TEMP_INT8)
        return config_.need_temp;

    return true;
}
```

---

## 五、校验

4 重校验，全部在 `finalize()` 末尾自动执行：

| 校验 | 检查内容 |
|------|---------|
| `validate_region_order` | 001→065 base 严格非递减 |
| `validate_contiguity` | 7 条硬约束（桶2连续、桶1连续、所有FP32梯度连续、BN统计量连续等） |
| `validate_layer_correspondence` | 同功能编号 W_/G_/M_/V_ 的 DTensor 数量一致 |
| `validate_alignment` | 每个 DTensor 的 offset 满足 256B 对齐 |

**删除项**：
- `validate_pairing_iron_law` — FP32=2×FP16 公式不存在
- `validate_alias_safety` — 当前无别名需求，有需时再加回

---

## 六、DTensor cuda_alignment 更新

修改 [distributed_tensor.h](file:///r:/renaissance/include/renaissance/tensor/distributed_tensor.h#L73-L87)：

```cpp
uint8_t cuda_alignment() const noexcept {
    if (dtype != DType::FP16) return 1;
    switch (region) {
        case Region::I_A_DATA:
        case Region::I_B_DATA:  return 4;
        case Region::F_FEATURE_FP16:
        case Region::F_GRAD_SLOT_FP16:
#ifdef TR_USE_CUDA
            return 8;
#else
            return 1;
#endif
        default: return 1;
    }
}
```

仅 4 个特殊 Region，其余走 default。

---

## 七、相关文件变更清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `include/renaissance/core/types.h` | 替换 Region 枚举（13→65），删除 ParamBucket | 核心类型定义 |
| `include/renaissance/tensor/distributed_tensor.h` | 更新 cuda_alignment() 使用新 Region 名 | 仅 4 个 Region 特殊 |
| `include/renaissance/memory/memory_plan.h` | **新建** | 完整 MemoryPlan 类 |
| `tests/unit_tests/` 中的调用方 | Region::FEATURE → Region::F_FEATURE_FP32 等 | 适配新枚举 |
| `include/renaissance/task/task_base.h` | 移除 ParamBucket 参数 | 旧版分桶概念废除 |
| `src/` 中的 MemoryPlan 使用方 | 按新 API 重写调用 | alloc_param → alloc |

---

## 八、新旧 API 对照

| 功能 | 旧版 | 新版 |
|------|------|------|
| 分配权重 | `alloc_weight(shape, FP16, is_first, is_bn)` | `alloc_deep_conv_weight(shape)` |
| 分配梯度 | `alloc_param(shape, FP32, FP32_G, DEEP_PLAIN, "main", -1)` | `alloc_grad_deep_conv(shape)` |
| 分配 EMA | `alloc_ema_weight(shape, ..., "ema")` | `alloc_ema_deep_conv(shape)` |
| 通用分配 | 6 参数 | 3 参数 `alloc(shape, dtype, region)` |
| 通信桶1 | `get_comm_range_deep()` | `get_comm_range_bucket1()` |
| 通信桶2 | `get_comm_range_first()` | `get_comm_range_bucket2()` |
| CAST 范围 | `get_weight_cast_ranges()` / `get_grad_cast_ranges()` | `get_batch_op_range(CAST_W32_TO_W16)` |
| 布局算法 | 两遍（预计算→预留→回填） | 一遍线性累加 |
| **删除** | `ParamClass`, `TensorRole`, `ParamBucket`, `model_tag`, `paired_id`, `is_bn`, `is_first_layer`, `fp16_to_fp32_map_`, `fp32_to_fp16_map_`, `param_bucket_anchors_`, `BatchCastRange` | — |

---

## 九、实施优先级

| 优先级 | 任务 | 说明 |
|:------:|------|------|
| **P0** | 更新 `types.h` Region 枚举 + 删除 ParamBucket | 所有下游文件的基础 |
| **P0** | 新建 `memory_plan.h`（类定义 + finalize + build_batch_op_ranges） | 核心实现 |
| **P0** | 更新 `distributed_tensor.h` cuda_alignment | 必须与 Region 同步 |
| **P1** | 修复所有调用方（Region::FEATURE → F_FEATURE_FP32 等） | 保证编译通过 |
| **P1** | 实现全部语义化 alloc_* 接口 | 均为 `alloc_impl` 的一行转发 |
| **P1** | 实现 4 重校验 | 硬约束自动化检查 |
| **P2** | 集成 DeviceContext 的 ptr_table 构建 | offset → GPU 指针 |

---

> **最终规格**：65 个 Region · 12 个板块 · 15 个批量操作 · 7 条硬约束 · 4 重校验 · 0 条冗余设计
>
> Region 设计的复杂性全部体现在枚举定义中，换来的是 MemoryPlan 实现的极度简洁。
