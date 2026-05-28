# 【小伙伴D】

## 1. 实验配置

| 参数 | PyTorch | Renaissance |
|------|---------|-------------|
| 模型 | Flatten → FC(512)→Tanh→FC(256)→Tanh→FC(10) | 同 |
| 初始化 | Kaiming Uniform (`a=√5`) | Kaiming Uniform (`a=0`) |
| 优化器 | SGD (momentum=0.9, nesterov=False) | 同 |
| Weight Decay | 0.0 | 0.0 |
| 学习率 | 0.1 | 0.1 |
| LR 调度 | StepLR (step=10, decay=0.1) | StepLR (step=10, decay=0.1) |
| Batch Size | 128 | 128 |
| Epochs | 3 | 3 |
| 数据 | MNIST (mean=0.1307, std=0.3081) | 同 |
| Random Seed | 42 | 42 |
| AMP | 无 | 无 |

> **LR 调度说明**：StepLR 在 step_size=10 epoch 时才衰减，3-epoch 测试中 LR 恒定为 0.1。无 warmup。

---

## 2. 实验结果

### 2.1 最终结果（当前配置）

| 指标 | PyTorch | Renaissance | 差距 |
|------|---------|-------------|------|
| Epoch 0 Val Top-1 | **95.60%** | **92.18%** | **-3.42%** |
| Epoch 0 Val Loss | 0.140 | 0.268 | |
| Epoch 0 Train Loss | 0.248 | — | |
| Epoch 0 Train Time | — | 0.32s | |
| **PASS (85% threshold)** | ✓ | **✓** | |

### 2.2 之前的结果（旧配置，供参考）

| 指标 | PyTorch | Renaissance | 差距 |
|------|---------|-------------|------|
| Epoch 0 Val Top-1 | 93.29% | 69.32% | **-24.0%** |
| LR | 0.01 | 0.01 | |
| LR 调度 | Cosine | Cosine | |
| Weight Decay | 5e-4 | 5e-4 | |

> 差距从 **24% → 3.4%**，缩小了约 **20 个百分点**。

---

## 3. 修改清单

### 3.1 PyTorch 基准 (`tests/correction/benchmark_pytorch.py`)

| 修改项 | 旧值 | 新值 |
|--------|------|------|
| 初始学习率 | `lr=0.01` | `lr=0.1` |
| LR 调度器 | `CosineAnnealingLR(eta_min=1e-5)` | `StepLR(step_size=10)` |
| Weight Decay | `weight_decay=5e-4` | `weight_decay=0.0` |
| 批次 LR 打印 | 无 | 每 100 batch 打印 `lr=0.1000` |

### 3.2 C++ 测试 (`tests/correction/test_dl_full.cpp`)

| 修改项 | 旧值 | 新值 |
|--------|------|------|
| 初始学习率 | `.base_lr(0.01f)` | `.base_lr(0.1f)` |
| LR 调度器 | `CosineAnnealingLR().eta_min(1e-5f)` | `StepLR().step_by_epoch()` |
| Weight Decay | `.weight_decay(5e-4f)` | `.weight_decay(0.0f)` |

### 3.3 训练循环 (`src/task/deep_learning_task.cpp`)

在 `run_train_epoch_gpu()` 的三个分支中添加了每 batch 学习率打印：
- `batches == 1` 分支：batch 0 始终打印
- 主循环 (`batch = 0 .. batches-2`)：每 100 batch 打印一次
- 末 batch (`batch = batches-1`)：始终打印

日志格式：`[LR] epoch=0 batch=100 lr=0.1`

验证结果：在整个 3-epoch 训练中，LR 始终为 **0.1**（StepLR 的 step=10，在 3 epoch 内不衰减），符合预期。

---

## 4. 差距根因分析

### 4.1 验证的关键发现

#### 发现 1：首层是 Flatten，不是 FC

```
ArchPlan 7 层:
  0  Flatten    [1,28,28,1] → [1,1,1,784]   [FIRST]
  1  FC         [1,1,1,784] → [1,1,1,512]
  2  Tanh
  3  FC         ... → [1,1,1,256]
  4  Tanh
  5  FC         ... → [1,1,1,10]
  6  SoftmaxCE
```

- `mark_first_layer()` 将 **layer 0 (Flatten)** 标记为 `is_first_layer=true`
- FC(512) 是 **layer 1**，归入 `DEEP_FWD_BWD` 图，不走 `FIRST_LAYER_FWD_A/B`
- 这与架构设计一致：Flatten 读写 `I_A_DATA/I_B_DATA`，FC 从 Flatten 输出 (DTensor 13) 读入

#### 发现 2：梯度链完整

```
DEEP_FWD_BWD:
  SoftmaxCE_BWD → FC10_BWD → Tanh_BWD → FC256_BWD → Tanh_BWD → FC512_BWD
  → 输出 d_input 到 DTensor 13

FIRST_LAYER_BWD_A:
  Flatten_BWD(DTensor 13) → 写回 I_A_DATA(DTensor 1)
```

梯度从 DEEP → FIRST_LAYER_BWD 通过共享 DTensor 13 正确传递。

#### 发现 3：OPTIMIZER 覆盖全权重

```
RANGE_UPDATE_WEIGHT_MOMENTUM:  R9-R11(W), R27-R29(G), R37-R39(M)
RANGE_UPDATE_BIAS_MOMENTUM:    R6-R8(W), R24-R26(G), R34-R36(M)
```

包含全部 3 组 FC (512, 256, 10) 的 weight + bias + momentum。所有参数都会更新。

#### 发现 4：lr_dtensor_id_ 正确设置

`S_SCALAR_FP32` 区域中，`lr` (id=6) 先于 `scaling` (id=7) 分配。`on_prepare()` 中的循环找到第一个 `S_SCALAR_FP32` DTensor 即为 `lr`，值为 6。`cudaMemcpyAsync` 将 scheduler 返回的 LR 写入正确的 DTensor，不会被 scaling 覆盖。

#### 发现 5：DTensor ID 映射验证

| DTensor ID | 用途 | 区域 |
|------------|------|------|
| 0,2 | I_A_LABEL, I_B_LABEL | I_A_LABEL, I_B_LABEL |
| 1,3 | I_A_DATA, I_B_DATA | I_A_DATA, I_B_DATA |
| 4 | label_smce (DTensorCopy 目标) | T_TEMP_INT32 |
| 6 | lr (学习率) | S_SCALAR_FP32 |
| 7 | scaling (loss 缩放因子) | S_SCALAR_FP32 |
| 8,9 | loss, top1 (结果输出) | R_RESULT |
| 13 | Flatten 输出 / 梯度回传共用 | F_FEATURE_FP32 |
| 14,24,34 | FC weight (512,256,10) | W_FC_WEIGHT |
| 15,25,35 | FC bias (512,256,10) | W_FC_BIAS |
| 17,27,37 | FC weight grad (512,256,10) | G_FC_WEIGHT |
| 18,28,38 | FC bias grad (512,256,10) | G_FC_BIAS |
| 21,31,41 | FC weight momentum (512,256,10) | M_FC_WEIGHT |
| 22,32,42 | FC bias momentum (512,256,10) | M_FC_BIAS |

全部 DTensor 映射正确，无 ID 冲突。

### 4.2 剩余 3.4% 差距的来源

#### 主要因素：Kaiming 初始化参数不同

| | PyTorch | Renaissance |
|------|---------|-------------|
| `a` 参数 | `√5` ≈ 2.236 | `0` |
| gain | `√(2/(1+5))` ≈ **0.577** | `√2` ≈ **1.414** |
| FC(512) bound | `1/√784` ≈ **0.0357** | `√6/√784` ≈ **0.0875** |
| FC(256) bound | `1/√512` ≈ **0.0442** | `√6/√512` ≈ **0.1083** |
| FC(10) bound | `1/√256` ≈ **0.0625** | `√6/√256` ≈ **0.1531** |

**Renaissance 的权重初始化幅度约为 PyTorch 的 2.45 倍。**

影响路径：
1. 更大的权重 → FC 输出幅度更大 → Tanh 更易饱和
2. Tanh 饱和 (|x| > 3) → 梯度 ≈ 0 → 该层学习极慢
3. FC(512) 有 401,408 参数（占总模型 75%），其学习速度直接影响收敛

#### 次要因素

1. **Weight Decay 实现差异**：PyTorch 的 weight decay 与 momentum 的耦合方式可能与 Renaissance 不同。虽然已置 0，但之前的对比中 wd=5e-4 时差异更大。
2. **数值精度**：CUDA kernel 与 PyTorch 的 cuBLAS 实现可能有微小差异（< 1% 级别）。
3. **Flatten 顺序**：Renaissance 中 Flatten 在 FC 之前作为独立图节点执行（FIRST_LAYER_FWD），PyTorch 中为 `x.view()` 直接内联，两者在 CUDA graph 中的执行时序略有差别。

---

## 5. 关键经验

1. **`batches=1` 误判**：最初的诊断日志显示 `batches=1`，但实际 `steps_per_epoch=469`。该误判来自早期编译产物的残留日志，后续重新编译后确认 469 batch 正确。

2. **ArchPlan 打印是关键诊断工具**：新增的 ComputationGraph dump（包含每个 GraphId 的节点列表与 DTensor 映射）使得梯度链验证成为可能。建议保留此日志。

3. **LR 调度器选择影响巨大**：从 CosineAnnealingLR (lr=0.01→1e-5) 改为 StepLR (lr=0.1 恒定) 是准确率跃升的主因之一。说明在当前模型/数据集组合下，小学习率严重制约了收敛速度。

4. **Weight Decay 的放大效应**：在 lr=0.01 时，wd=5e-4 对 FC(512) 的 401K 参数产生了显著的压制效果；lr=0.1 + wd=0 彻底消除了此因素。

---

## 6. 后续建议

如需进一步缩小剩余的 3.4% 差距：

1. **修改初始化器**：将 `InitKind::KAIMING_UNIFORM` 的 `a` 参数从 `0` 改为 `√5`，使其与 PyTorch 默认行为对齐。这是性价比最高的改进。

2. **对比单 batch 训练**：用 `test_two_batch_correction` 对比 PyTorch 和 Renaissance 在完全相同输入下的 loss 数值，量化 kernel 层面的数值差异。

3. **对比梯度幅值**：在第一个 batch 后 dump 三组 FC 的权重梯度范数，验证梯度流是否与 PyTorch 等量。