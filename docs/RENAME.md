# TR4 算子重命名对应表

> **依据**：`docs/OPS_NAME.md`（V4.20 Final）  
> **源文件**：`include/renaissance/graph/op_kind.h`  
> **生成日期**：2026-05-05

---

## 变更类型说明

| 标记 | 含义 |
|------|------|
| ✏️ 改名 | 旧名称改为新名称 |
| 🔀 合并 | 两个旧枚举值合并为一个新枚举值 |
| ➕ 新增 | 新方案中新增的枚举值 |
| — 不变 | 名称不变 |

---

## 完整重命名表

### 一、基础算子

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `IDENTITY` | `IDENTITY_FWD` | ✏️ | 补充 `_FWD` 方向后缀 |
| `ADD` | `ADD_FWD` | ✏️ | 补充 `_FWD` 方向后缀 |
| `MUL` | `MUL_FWD` | ✏️ | 补充 `_FWD` 方向后缀 |
| `AXPY` | `AXPY_FWD` | 🔀 | 基础算子补充 `_FWD`，与 `FUSED_AXPY_FP32` 统一 |
| `FUSED_AXPY_FP32` | `AXPY_FWD` | 🔀 | 去掉 `FUSED_` 和 `_FP32`，与 `AXPY` 统一为 `AXPY_FWD`（类型多态） |

### 二、激活

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `RELU_FWD` | `RELU_FWD` | — | 类型多态，不变 |
| `RELU_BWD` | `RELU_BWD` | — | 类型多态，不变 |

### 三、卷积

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `CONV_FWD` | `CONV_AMP_FWD` | ✏️ | 标注 `_AMP` 混合精度管线 |
| `CONV_BWD_DATA` | `CONV_AMP_BWD` | 🔀 | 与 `CONV_BWD_WEIGHT` 合并为单算子，双输出 `[dX, dW]` |
| `CONV_BWD_WEIGHT` | `CONV_AMP_BWD` | 🔀 | 与 `CONV_BWD_DATA` 合并为单算子，双输出 `[dX, dW]` |

### 四、BatchNorm

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `BN_FWD` | `BN_AMP_FWD` | ✏️ | 标注 `_AMP`。AMP 下 running stats 更新策略与 FP32 不同 |
| `BN_BWD` | `BN_AMP_BWD` | ✏️ | 同上 |

### 五、池化

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `MAXPOOL_FWD` | `MAXPOOL_FWD` | — | 类型多态，不变 |
| `MAXPOOL_BWD` | `MAXPOOL_BWD` | — | 类型多态，不变 |
| `GAP_FWD` | `GAP_FWD` | — | 类型多态，不变 |
| `GAP_BWD` | `GAP_BWD` | — | 类型多态，不变 |

### 六、全连接

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `FC_FWD` | `FC_AMP_FWD` | ✏️ | 标注 `_AMP`。GEMM 的 accumulator 精度影响算法选择 |
| `FC_BWD` | `FC_AMP_BWD` | ✏️ | 同上。双输出 `[dX, dW]` |

### 七、形状变换

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `FLATTEN_FWD` | `FLATTEN_FWD` | — | 内存拷贝，类型多态，不变 |
| `FLATTEN_BWD` | `FLATTEN_BWD` | — | 不变 |

### 八、融合算子

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `FUSED_CBR` | `CBR_AMP_FWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP`，补充 `_FWD` |
| `FUSED_CBR_BWD` | `CBR_AMP_BWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP` |
| `FUSED_CBRP` | `CBRP_AMP_FWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP`，补充 `_FWD` |
| `FUSED_CBRP_BWD` | `CBRP_AMP_BWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP` |
| `FUSED_BOTTLENECK_FWD` | `BOTTLENECK_AMP_FWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP` |
| `FUSED_BOTTLENECK_BWD` | `BOTTLENECK_AMP_BWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP` |
| `FUSED_GAP_FC_FWD` | `GAP_FC_AMP_FWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP`，补充 `_FWD` |
| `FUSED_GAP_FC_BWD` | `GAP_FC_AMP_BWD` | ✏️ | 去掉 `FUSED_`，标注 `_AMP` |
| — | `CBR_AMP_INF` | ➕ | 新增：CBR 推理版，允许 BN fold 进 Conv |
| — | `CBRP_AMP_INF` | ➕ | 新增：CBRP 推理版 |
| — | `BOTTLENECK_AMP_INF` | ➕ | 新增：Bottleneck 推理版 |
| — | `GAP_FC_AMP_INF` | ➕ | 新增：GAP_FC 推理版 |

### 九、损失函数

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `CROSS_ENTROPY_LOSS` | `CROSS_ENTROPY_LOSS` | — | 保留 `_LOSS`，避免与 `cross_entropy()` 函数名混淆 |

### 十、通信 / 同步

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `ALLREDUCE_SUM` | `ALLREDUCE_SUM` | — | 保留 `_SUM`，AllReduce 有多种归约方式 |
| `BROADCAST` | `BROADCAST` | — | 不变 |
| `BN_STATS_SYNC` | `BN_STATS_SYNC` | — | 保留完整语义 |

### 十一、类型转换

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `CAST_FP16_TO_FP32` | `CAST_H2F` | ✏️ | H=Half(FP16), F=Float(FP32)。类型信息融入 BASE |
| `CAST_FP32_TO_FP16` | `CAST_F2H` | ✏️ | 同上 |

### 十二、优化器更新

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `LARS_UPDATE` | `LARS_UPDATE` | — | 保留 `_UPDATE`，避免与类名 `class LARS` 混淆 |
| `SGD_UPDATE` | `SGD_UPDATE` | — | 同上 |
| `ADAM_UPDATE` | `ADAM_UPDATE` | — | 同上 |
| `ADAMW_UPDATE` | `ADAMW_UPDATE` | — | 同上 |
| `EMA_UPDATE` | `EMA_UPDATE` | — | 类型多态，不标 FP32 |

### 十三、哨兵值

| 旧名称 | 新名称 | 变更 | 备注 |
|--------|--------|------|------|
| `COUNT` | `COUNT` | — | 不变 |
| `UNKNOWN` | `UNKNOWN` | — | 不变 |

---

## 统计

| 类别 | 数量 |
|------|------|
| ✏️ 改名 | 18（一对一改名，不含合并项） |
| 🔀 合并 | 2 组（涉及 4 个旧值 → 2 个新值：`AXPY`+`FUSED_AXPY_FP32`→`AXPY_FWD`；`CONV_BWD_DATA`+`CONV_BWD_WEIGHT`→`CONV_AMP_BWD`） |
| ➕ 新增 | 4 |
| — 不变 | 17 |
| **旧枚举总数** | **39**（不含 COUNT、UNKNOWN） |
| **新枚举总数** | **41**（不含 COUNT、UNKNOWN） |

---

## 实施要点

1. **合并 `AXPY` + `FUSED_AXPY_FP32`**：两个旧枚举值统一为 `AXPY_FWD`（类型多态，不标精度），确保所有引用点统一替换。
2. **合并 `CONV_BWD_DATA` + `CONV_BWD_WEIGHT`**：合并为 `CONV_AMP_BWD`，双输出 `[dX, dW]`。Compiler 和 execute_compute_node 中的逻辑需同步更新。
3. **去掉所有 `FUSED_` 前缀**：共9处（`FUSED_CBR`、`FUSED_CBR_BWD`、`FUSED_CBRP`、`FUSED_CBRP_BWD`、`FUSED_BOTTLENECK_FWD`、`FUSED_BOTTLENECK_BWD`、`FUSED_GAP_FC_FWD`、`FUSED_GAP_FC_BWD`、`FUSED_AXPY_FP32`），需要同步更新注册函数名，如 `register_op_FUSED_CBR()` → `register_op_CBR_AMP_FWD()`。
4. **新增 `_INF` 值**：`CBR_AMP_INF`、`CBRP_AMP_INF`、`BOTTLENECK_AMP_INF`、`GAP_FC_AMP_INF` 需在 op_kind.cpp 中添加对应的 `op_kind_to_string` 和 `format_params` 分支。
5. **枚举值顺序**：合并后的枚举需调整顺序，移除被合并的旧值，新增值插入对应分组位置。
6. **dtype 分发逻辑**：元素级算子（如 `RELU_FWD`、`ADD_FWD`）虽然不标精度，但 `execute_compute_node` 的 CUDA 分支需要根据输入 `DTensor::dtype` 自动分发到 FP32 或 AMP kernel。
7. **影响范围**：`op_kind.h`、`op_kind.cpp`、`compiler.cpp`、`task_base.cpp`、`op_all_register.cpp`、所有 `src/backend/ops/*.cpp` 的 OpDescriptor。
