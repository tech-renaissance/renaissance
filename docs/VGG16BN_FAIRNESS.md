# TR4 与 PyTorch 的 VGG16BN AMP 训练公平性对比报告（默认配置版）

> **文档版本**: DEFAULT-CFG-2026-06-21
> **日期**: 2026-06-21
> **测试任务**: ImageNet-1K VGG-16-BN, AMP (FP16), 100 epochs, global batch=2048, A100 × 8
> **TR4 入口**: `tests/integration/test_vgg16bn.cpp`（默认 `--amp`，不带 `--cpvs`/`--fully`/`--dts`）
> **PyTorch 入口**: `tests/integration/test_vgg16bn.py` (`torchrun --nproc_per_node=8`)
> **审计原则**: "无证据不指控"——仅陈述经代码审计确认的事实，不推测底层黑盒实现。

---

## 1. 执行摘要

本报告对 TR4（Renaissance）框架与 PyTorch 在 VGG16BN-ImageNet AMP 训练任务上的默认配置进行对比审计。

**结论: 在默认配置下，对比是公平的。**

- 核心算法、超参数、网络结构、数据增强、AMP 语义、计时口径、学习率调度均已严格对齐。
- 默认配置下已确认的差异共 10 项，均属于框架自身架构或实现能力差异，而非测试样例作弊。
- Channel Padding（首层通道 padding）已通过 EXY2 修复：`tests/integration/test_vgg16bn.cpp` 已移除首层 `channel_padding()`。默认 AMP 下，TR4 首层 Conv 仍以 3 通道逻辑输入开始（由 `ArchPlan` 自动对齐到 4 通道，FusedNormalization 将第 4 通道填 0），训练动态上与 PyTorch `Conv2d(3, 64, ...)` 等价；仅首层权重的 Kaiming `fan_in` 存在 `36` vs `27` 的微小残差。该问题不再作为差异项。
- CPVS 验证缓存、FULLY 训练集缓存与 DTS 压缩格式在默认配置下均已关闭，不再作为默认对比中的活跃差异项。
- 不存在系统性的单向不公平。所有差异的影响量级均在"极小"到"中等"范围内。

---

## 2. 审计原则与方法

### 2.1 审计原则

1. **无证据不指控**: 所有差异点均基于可阅读的源代码进行陈述；对无法确认的实现细节（如 cuDNN 内部 heuristics、PyTorch `autocast` 的逐算子决策、NCCL 底层调度）不做主观推断。
2. **公平 ≠ 完全一致**: 不同框架在内部实现上必然存在差异。本报告关注的是这些差异是否导致某一方获得系统性、不可辩护的优势，以及差异是否仍落在"公平可比"的范围内。
3. **不引用性能数据**: 本文档仅分析公平性，不涉及任何实测耗时或精度结果。

### 2.2 审计范围

| 层级 | 审计内容 |
|:---|:---|
| 测试入口 | `test_vgg16bn.cpp` (TR4), `test_vgg16bn.py` (PyTorch) |
| 网络结构 | blueprint 逐层对比 vs `nn.Module` 逐层对比 |
| 超参数 | 学习率、batch size、weight decay、momentum、dropout 等 |
| 调度器 | CosineAnnealingLR + Warmup 数学公式 |
| 优化器 | SGD + momentum + weight decay 排除 |
| 数据增强 | RRC、HFlip、ColorJitter、Normalize、RandomErasing |
| AMP 管线 | FP16 前向/反向、master weights、loss scaling |
| 计时 | 编译/图捕获排除、计时起点/终点 |
| 框架实现 | FusedNormalization、CPVS、CUDA Graph、内存布局、多 GPU 通信、数据加载 |

---

## 3. 测试配置

### 3.1 运行命令

```bash
# TR4（默认配置，--amp；CPVS/FULLY/DTS 均关闭）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /root/epfs/R/renaissance/build/bin/tests/integration/test_vgg16bn

# PyTorch
torchrun --nproc_per_node=8 /root/epfs/R/renaissance/tests/integration/test_vgg16bn.py
```

### 3.2 共同配置总览

| 参数 | 值 | 代码位置 |
|:---|---|:---|
| 网络结构 | VGG-16-BN: 5 block (2/2/3/3/3 conv+BN+ReLU) + 3 FC (4096/4096/1000) | C++ L162-198, Python L57-141 |
| Conv bias | false | C++ L164-188, Python L62-113 |
| FC bias | true | C++ L193-197, Python L117-123 |
| BatchNorm | affine=True, momentum=0.1, eps=1e-5 | C++ 默认, Python 默认 |
| Dropout | p=0.5 (FC1/FC2 之后) | C++ L194-196, Python L119-122 |
| 激活函数 | ReLU | C++ L164-197, Python L64-122 |
| Global batch size | 2048 (local=256 × 8 GPU) | C++ L124, Python L17 (local), L342 (global) |
| 优化器 | SGD momentum=0.9, nesterov=True | C++ L211-214, Python L234-235 |
| Weight decay | 1e-4 (BN/bias 排除) | C++ L211-214, Python L217-230 |
| 学习率调度 | CosineAnnealing + Warmup(10) | C++ L216-221, Python L45-51 |
| Peak LR | 0.36 | C++ L217, Python L18 |
| Warmup start LR | 0.01 | C++ L218, Python L19 |
| Eta min | 1e-6 | C++ L220, Python L20 |
| Label smoothing | 0.1 | C++ L202, Python L24 |
| 梯度裁剪 | 未启用 | — |
| 训练 epoch | 100 | C++ L42, Python L16 |
| 数据增强 | RRC(224,0.08~1.0) + HFlip + ColorJitter(0.2,0.2,0.2,0.1) + Normalize(ImageNet) + RandomErasing(p=0.25, scale=0.02~0.33, ratio=0.3~3.3, fill=0)（HFlip 与 ColorJitter 顺序因 FusedNormalization 融合而略有差异，详见 5.1 节） | C++ L143-150, Python L171-177 |
| 验证预处理 | Resize(256) + CenterCrop(224) + Normalize(ImageNet) | C++ L152-155, Python L179-184 |
| 权重初始化 | Kaiming Uniform fan_in | C++ L204-207, Python L131 |
| TF32 | 双方均开启 | C++ L127, Python L161-164 |

---

## 4. 已确认严格一致的方面

以下各项经代码对比确认，双方在数学语义和参数上完全一致，不存在任何差异。

### 4.1 网络结构

| 结构层 | TR4 | PyTorch | 一致性 |
|:---|:---|:---|:---:|
| Block1 | conv(64)×2+BN+ReLU → maxpool | Conv2d(3→64)×2+BN+ReLU → MaxPool2d | ✅ |
| Block2 | conv(128)×2+BN+ReLU → maxpool | Conv2d(64→128)×2+BN+ReLU → MaxPool2d | ✅ |
| Block3 | conv(256)×3+BN+ReLU → maxpool | Conv2d(128→256)×3+BN+ReLU → MaxPool2d | ✅ |
| Block4 | conv(512)×3+BN+ReLU → maxpool | Conv2d(256→512)×3+BN+ReLU → MaxPool2d | ✅ |
| Block5 | conv(512)×3+BN+ReLU → maxpool | Conv2d(512→512)×3+BN+ReLU → MaxPool2d | ✅ |
| Classifier | flatten → fc(4096) → ReLU → Dropout(0.5) → fc(4096) → ReLU → Dropout(0.5) → fc(1000) | 一致 | ✅ |

Conv kernel_size=3, stride=1, padding=1, bias=False 在所有层均一致。

### 4.2 学习率调度器

**PyTorch** (`test_vgg16bn.py:45-51`):

```python
def warmup_cosine_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        alpha = epoch / max(WARMUP_EPOCHS, 1)
        return (WARMUP_LR_START + (PEAK_LR - WARMUP_LR_START) * alpha) / PEAK_LR
    progress = (epoch - WARMUP_EPOCHS) / max(TOTAL_EPOCHS - WARMUP_EPOCHS - 1, 1)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (ETA_MIN + (PEAK_LR - ETA_MIN) * cos_val) / PEAK_LR
```

**TR4** (`src/algo/scheduler.cpp:102-130,195-208`):

- `get_lr_by_epoch(epoch_id)` 使用 `effective_step = epoch_id * steps_per_epoch_`。
- Warmup: `progress = effective_step / warmup_steps_`，其中 `warmup_steps_ = warmup_epochs_ * steps_per_epoch`。`warmup_epochs_=10` 覆盖内部 epoch `0..10`（共 11 个打印点），分母为 10。
- Cosine: `effective_total = total_decay - steps_per_epoch_ = (100-10)-1 = 89`，cosine 从内部 epoch 11 开始（打印 epoch 12），到内部 epoch 99（打印 epoch 100）到达 `eta_min`。

**数学等价性验证**:

PyTorch 的 `LambdaLR` 将 `lr_lambda` 返回值乘以 `base_lr`（即 `PEAK_LR`），因此实际 LR = `lr_lambda(epoch) * PEAK_LR`。

- PyTorch warmup: epoch `0..9` 线性，分母 10；epoch 10 进入 cosine，progress = 0，LR = `PEAK_LR`。
- TR4 warmup: 内部 epoch `0..10` 线性，分母 10；内部 epoch 11 开始 cosine，progress = 1/89。

虽然 PyTorch 的最后一个 warmup 点被 cosine progress=0 点替代，但两者打印出的 LR 序列完全一致：

| 打印 epoch | TR4 LR | PyTorch LR |
|---:|---:|---:|
| 1 | 0.010000 | 0.010000 |
| 2 | 0.045000 | 0.045000 |
| ... | ... | ... |
| 10 | 0.325000 | 0.325000 |
| 11 | 0.360000 | 0.360000 |
| 12 | 0.359888 | 0.359888 |
| 100 | 0.000001 | 0.000001 |

**结论: 数学完全等价。** ✅

### 4.3 Weight Decay 施加

**PyTorch** (`test_vgg16bn.py:217-230`): 通过 `param_groups` 显式分组，`ndim==1` 的参数（BN weight/bias、FC bias）weight_decay=0。

**TR4** (`test_vgg16bn.cpp:211-214`, `src/backend/ops/range/optimizer_op.cpp`): 框架底层通过 WEIGHT/BIAS 两套独立 CUDA kernel 实现分离。WEIGHT 路径传入 `wd`，BIAS 路径不传 `wd`（内部 `wd=0`）。

**结论: 数学完全等价。** ✅

### 4.4 BN 参数

**TR4** 使用 `bn()` 默认参数 `momentum=0.1, eps=1e-5`。

**PyTorch**: `nn.BatchNorm2d` 默认 `momentum=0.1, eps=1e-5`。

**结论: 完全一致。** ✅

### 4.5 Normalize 参数

**TR4**: `NormMode::IMAGENET` 对应 `mean={0.485,0.456,0.406}, std={0.229,0.224,0.225}`。

**PyTorch**: `transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))`。

**结论: 数值完全一致。** ✅

### 4.6 梯度裁剪

双方均未启用梯度裁剪。

**结论: ✅**

### 4.7 随机种子

**PyTorch** (`test_vgg16bn.py:167-168,287-288`): `torch.manual_seed(123 + rank)` 与 `random.seed(123 + rank)`，首次初始化与 `torch.compile` warm-up 后重新初始化均保持一致。

**TR4** (`test_vgg16bn.cpp:123`): `GLOBAL_SETTING.manual_seed(123)`，框架底层为每个 rank 派生独立种子。

**结论: 语义等价，均为 per-rank 独立种子。** ✅

### 4.8 drop_last 行为

TR4 默认 `using_drop_last_=false`，PyTorch `DistributedSampler` 与 `DataLoader` 默认 `drop_last=False`。双方均不丢弃最后不完整 batch。

**结论: ✅**

### 4.9 计时口径

PyTorch 先用 dummy batch 触发 `torch.compile(max-autotune)`，再 `torch.cuda.synchronize()` + `dist.barrier()`，然后重新初始化并开始计时。该 dummy batch 为完整 `LOCAL_BATCH_SIZE=256`；训练/验证的尾 batch（实际尺寸通常更小）未在 warm-up 中预热，计时循环中首次遇到这些形状时仍可能产生一次 `torch.compile` 重编译开销。

TR4 `task.compile()` 在计时起点之前完成，并且会构建 full/last batch 两套图变体。

**结论: 双方均将主要编译/图捕获开销排除在计时之外；PyTorch 存在尾 batch 编译未预热的微小残差。** ✅/⚠️

### 4.10 验证精度

PyTorch 验证在 `torch.no_grad()` + `torch.amp.autocast('cuda')` 下进行；TR4 AMP 推理图使用 FP16 权重和 FP16 计算。

**结论: ✅**

### 4.11 TF32 与精度设置

**TR4** (`test_vgg16bn.cpp:127`): `.use_tf32(true)`。

**PyTorch** (`test_vgg16bn.py:161-164`):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
```

- `allow_tf32`: 双方均开启 TF32 加速。
- `float32_matmul_precision('high')`: 使用 TF32 精度的矩阵乘法，与 `allow_tf32=True` 配合，双方语义一致。
- `cudnn.benchmark = True`: 启用 cuDNN auto-tuner，在固定输入形状下自动选择最优卷积算法。TR4 框架通过 cuDNN API 有等效的算法选择机制。

**结论: ✅**

### 4.12 权重初始化

| 参数类型 | TR4 | PyTorch | 一致性 |
|:---|:---|:---|:---:|
| Conv/FC 权重 | `InitKind::KAIMING_UNIFORM`, `FanMode::FAN_IN` | `nn.init.kaiming_uniform_(..., a=0, mode='fan_in')` | ✅ |
| BN weight | 初始化为 1 | `nn.init.ones_` | ✅ |
| BN bias | 初始化为 0 | `nn.init.zeros_` | ✅ |
| FC bias | 初始化为 0 | `nn.init.zeros_` | ✅ |

> **注**: 自 EXY2 修复后，`tests/integration/test_vgg16bn.cpp` 已移除首层 `channel_padding()`。AMP 下 `ArchPlan` 将首层 Conv 的输入逻辑 C 对齐到 4，首层 Conv 的 KRSC 权重物理 shape 变为 `[64,3,3,4]`（PyTorch 为 `[64,3,3,3]`）。FusedNormalization 将第 4 个输入通道填 0，conv 反向中该通道权重梯度为 0（仅受 weight decay 影响），因此前向/反向数学行为与 3 通道输入等价。残留的细微差别是 Kaiming `FanMode::FAN_IN` 按对齐后的 4 通道计算，得到 `4×3×3=36`，而 PyTorch 为 `3×3×3=27`，首层权重初始化幅度约有 `√27/√36 ≈ 0.87` 倍的差异，影响量级极小。

### 4.13 CrossEntropyLoss + Label Smoothing

双方均使用 label_smoothing=0.1，数学形式一致。

**结论: ✅**

### 4.14 数据增强参数

| 增强操作 | TR4 参数 | PyTorch 参数 | 一致性 |
|:---|:---|:---|:---:|
| RandomResizedCrop | `(224, 0.08f, 1.0f)`, ratio 默认 (3/4, 4/3) | `(224, scale=(0.08, 1.0))`, ratio 默认 (3/4, 4/3) | ✅ |
| RandomHorizontalFlip | 默认 50% | `RandomHorizontalFlip()` | ✅ |
| ColorJitter | `(0.2f, 0.2f, 0.2f, 0.1f)` | `(0.2, 0.2, 0.2, 0.1)` | ✅ |
| Normalize | ImageNet preset | `mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)` | ✅ |
| RandomErasing | `(0.25f, {0.02f, 0.33f}, {0.3f, 3.3f})` | `(p=0.25, scale=(0.02,0.33), ratio=(0.3,3.3), value=0)` | ✅ |

### 4.15 验证预处理 Resize + CenterCrop 的等价性

TR4 的 `Resize(256)` 与 PyTorch 的 `transforms.Resize(256, antialias=False)` 均保持短边比例，最终 `CenterCrop(224)` 覆盖区域一致。双方语义等价。PyTorch 在 `RandomResizedCrop` 和 `Resize` 中均显式设置 `antialias=False`，TR4 对应操作无抗锯齿，双方一致。

**结论: ✅**

---

## 5. 已确认差异及公平性分析

以下差异经代码审计确认存在，但均属于框架自身架构或实现能力差异，不构成不公平竞争。

---

### 5.1 FusedNormalization 融合预处理

| 属性 | 内容 |
|:---|:---|
| **差异描述** | TR4 的 `FusedNormalization` 将 ToTensor + RandomHorizontalFlip + Normalize + RandomErasing 融合为单次 CPU 内存遍历；PyTorch 的 `Compose` 链式调用每个 transform 独立遍历。ColorJitter 作为独立 PO 在 FusedNormalization 之前执行。因此 TR4 实际顺序为 RRC → ColorJitter → FusedNormalization(HFlip→ToTensor→Normalize→RandomErasing)，PyTorch 为 RRC → HFlip → ColorJitter → ToTensor → Normalize → RandomErasing。HFlip 与 ColorJitter 均为独立随机变换，交换顺序不产生系统性分布偏移。 |
| **代码证据** | TR4: `src/data/preprocessor.cpp:173-371`（注入逻辑）, `src/data/fused_normalization.cpp:445-570`（执行逻辑）; PyTorch: `test_vgg16bn.py:171-177`。 |
| **对谁有利** | 偏向 TR4 |
| **影响量级** | 中等（减少 CPU 预处理 cache miss 和内存带宽消耗） |
| **公平性** | ✅ 公平（架构优势） |

---

### 5.2 CPVS / FULLY / DTS 模式（默认关闭）

| 属性 | 内容 |
|:---|:---|
| **差异描述** | TR4 支持三个可选加速开关：`--cpvs`（验证集预处理缓存）、`--fully`（训练集全量缓存）与 `--dts`（DTS 压缩格式，level 3）。DTS 是本框架独有的数据集格式，将分散的 ImageNet 图片合并为一个二进制单文件；LV0 与原始数据集字节级一致，LV1~3 依次加大压缩强度（LV1 等比缩放；LV2 等比缩放+边缘裁剪；LV3 等比缩放+边缘裁剪+降低 JPEG 质量）。所有压缩级别均按固定规则对数据集做一视同仁的处理，不做特征增强或样本筛选。默认配置下 `.using_cpvs(false)`、`.fully_mode(false)`、`.using_dts_format(false, 3)`，三者均不启用。PyTorch 默认无等价缓存/压缩。 |
| **代码证据** | TR4: `test_vgg16bn.cpp:55-60,80-84,129-141`; `include/renaissance/core/global_registry.h:329-334`。 |
| **对谁有利** | 默认配置下无差异；启用后偏向 TR4 |
| **影响量级** | 默认情形下为 0 |
| **公平性** | ✅ 默认公平；启用时属于 TR4 架构能力 |

> 注：若未来使用 `--cpvs`、`--fully` 或 `--dts` 进行对比，应单独审计其公平性。

---

### 5.3 数据加载 Workers 架构

| 属性 | 内容 |
|:---|:---|
| **差异描述** | PyTorch 使用 `num_workers=16` per rank（共 128 个 DataLoader worker 子进程），每个 worker 完成完整"读文件→解码→预处理"流水线。TR4 使用 `load_workers=16`（全局文件读取线程）+ `preprocess_workers=128`（全局预处理线程），两级流水线分离 I/O 和计算。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:197-206`（含 `persistent_workers=True`, `prefetch_factor=8`）; TR4: `test_vgg16bn.cpp:129-141`（`.cpu_binding(false)`, `.fully_mode(false)`, `.using_cpvs(false)`, `.using_dts_format(false, 3)`）。 |
| **对谁有利** | 略微偏向 TR4（C++ 线程模型 + 共享内存 overhead 更低） |
| **影响量级** | 小 |
| **公平性** | ✅ 公平 |

---

### 5.4 计算图执行模型

| 属性 | 内容 |
|:---|:---|
| **差异描述** | TR4 使用 C++ 静态计算图，将训练 step 中的多个子序列捕获为 CUDA Graph 并通过 `cudaGraphLaunch` 回放。PyTorch 使用 Python 动态图 + `torch.compile(mode="max-autotune")` + DDP，通过 FX graph 捕获和 Triton 代码生成优化。 |
| **代码证据** | TR4: `src/graph/capture_cuda.cpp`, `src/task/deep_learning_task.cpp`; PyTorch: `test_vgg16bn.py:214`。 |
| **对谁有利** | 各自框架的核心竞争力 |
| **影响量级** | 核心维度 |
| **公平性** | ✅ 公平 |

---

### 5.5 AMP 策略

| 属性 | 内容 |
|:---|:---|
| **差异描述** | PyTorch 使用 `torch.amp.autocast('cuda')` + `GradScaler`（初始缩放 65536，动态增长）。TR4 使用显式 CAST 图 + 混合梯度缩放（初始 8192，仅回退不主动增长）。TR4 AMP 同时维护 FP32 master weights，优化器直接更新 FP32 权重，随后通过 `CAST_MAIN_FP32_TO_FP16` 同步到 FP16 working weights；PyTorch autocast 下模型权重本身即为 FP32，两者在“更新精度”上等价。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:240,367-376`; TR4: `include/renaissance/core/global_config.h`, `src/graph/compiler.cpp`, `src/backend/ops/range/grad_scaling_op.cu`。 |
| **对谁有利** | 互有抵消 |
| **影响量级** | 中等 |
| **公平性** | ✅ 公平 |

---

### 5.6 多 GPU 通信

| 属性 | 内容 |
|:---|:---|
| **差异描述** | PyTorch 使用 `DDP` + `nn.SyncBatchNorm.convert_sync_batchnorm()`。TR4 使用框架内置多 rank 通信机制，通过静态图内嵌 AllReduce 实现梯度同步和 BN 统计量同步。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:210-212`; TR4: `src/task/deep_learning_task.cpp`, `src/graph/compiler.cpp`, `src/backend/ops/range/allreduce_op.cpp`。 |
| **对谁有利** | 无显著偏向 |
| **影响量级** | 小 |
| **公平性** | ✅ 公平 |

---

### 5.7 内存布局

| 属性 | 内容 |
|:---|:---|
| **差异描述** | TR4 原生使用 NHWC 布局。PyTorch 原生 NCHW，通过 `.to(memory_format=torch.channels_last)` 在运行时转换。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:209,258,289,361,393`; TR4: 内部 DTensor 布局。 |
| **对谁有利** | 略微偏向 TR4 |
| **影响量级** | 小 |
| **公平性** | ✅ 公平（架构设计） |

---

### 5.8 梯度清零方式

| 属性 | 内容 |
|:---|:---|
| **差异描述** | PyTorch 使用 `optimizer.zero_grad(set_to_none=True)`。TR4 梯度清零在框架内部处理。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:364`。 |
| **对谁有利** | 略微偏向 PyTorch |
| **影响量级** | 极小 |
| **公平性** | ✅ 公平 |

---

### 5.9 ReLU 算子 (`inplace=True`)

| 属性 | 内容 |
|:---|:---|
| **差异描述** | PyTorch 使用 `nn.ReLU(inplace=True)`。TR4 的 `relu()` 算子当前不支持 inplace 模式。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:64-121`; TR4: `test_vgg16bn.cpp:164-197`。 |
| **对谁有利** | 略微偏向 PyTorch |
| **影响量级** | 小 |
| **公平性** | ✅ 公平（框架实现差异） |

---

### 5.10 torch.compile 对不完整 batch 的预热差异

| 属性 | 内容 |
|:---|:---|
| **差异描述** | PyTorch 的 warm-up 仅使用 `LOCAL_BATCH_SIZE=256` 的 dummy batch 触发 `torch.compile(max-autotune)`。ImageNet 训练/验证的最后一批实际大小通常小于 256（训练约 146，验证约 106），`torch.compile` 在遇到这些新形状时仍可能触发一次重新编译，开销落在计时循环内。TR4 的 `task.compile()` 在计时前已构建 full/last batch 两套图变体。 |
| **代码证据** | PyTorch: `test_vgg16bn.py:251-284`; TR4: `src/task/deep_learning_task.cpp`（compile 阶段构建 last-batch 变体）。 |
| **对谁有利** | 略微偏向 TR4（计时口径） |
| **影响量级** | 小~中（一次性编译开销，分摊到 100 epoch 后较小） |
| **公平性** | ⚠️ 建议改进：在 PyTorch warm-up 中补充 train/eval 的尾 batch 预热，或在报告中将该开销列为计时残差。 |

---

### 5.11 Channel Padding（首层通道 padding）—— 已消除

| 属性 | 内容 |
|:---|:---|
| **状态** | 已通过 EXY2 修复，不再作为差异项。 |
| **修复内容** | `tests/integration/test_vgg16bn.cpp` 已移除首层 `channel_padding()`。在 AMP 模式下，`ArchPlan` 会在形状推导阶段将首层 Conv/CBR 的输入逻辑 C 从 3 对齐到 4，满足 TensorCore 要求；FP32 模式与非 Conv/CBR 首层不受影响。 |
| **结果** | TR4 与 PyTorch 的首层 Conv 现在均有效处理 3 通道输入：TR4 通过 AMP 对齐将输入/权重物理通道扩展为 4，FusedNormalization 将第 4 通道填 0，conv 反向中该通道梯度为 0；PyTorch 直接为 `Conv2d(3, 64, kernel_size=3)`。双方在训练动态上等价，仅首层权重的 Kaiming 初始化 fan_in 有 `36` vs `27` 的微小残差（√27/√36 ≈ 0.87 倍幅度差异）。 |
| **公平性** | ✅ 已实现结构对齐 |

---

## 6. 不予置评的方面

以下方面涉及底层实现细节，无法从当前代码审计中直接验证，故本报告**不予置评**：

- cuDNN/cuBLAS 内部对 TF32/FP16 算子的具体选择与调优行为。
- PyTorch `torch.amp.autocast` 对每一个算子的逐层精度决策。
- `torch.compile(mode="max-autotune")` 是否持续进行后台 auto-tuning。
- NCCL 集合通信在特定 NVLink 拓扑下的调度细节。
- JPEG 解码库差异（TR4 使用 TurboJPEG，PyTorch 默认 PIL）。

---

## 7. 综合公平性判定

### 7.1 差异汇总

| 编号 | 差异项 | 偏向 | 影响量级 | 是否公平 |
|:---:|:---|:---|:---:|:---:|
| 1 | FusedNormalization 融合预处理 | 偏向 TR4 | 中等 | ✅ 架构优势 |
| 2 | CPVS / FULLY / DTS 模式 | 默认关闭，无差异 | 0 | ✅ |
| 3 | Workers 架构 | 略微偏向 TR4 | 小 | ✅ |
| 4 | 计算图执行模型 | 各自核心优势 | 核心维度 | ✅ |
| 5 | AMP 策略 | 互有抵消 | 中等 | ✅ |
| 6 | 多 GPU 通信 | 无显著偏向 | 小 | ✅ |
| 7 | 内存布局 (NHWC vs channels_last) | 略微偏向 TR4 | 小 | ✅ 架构设计 |
| 8 | 梯度清零方式 (`set_to_none`) | 略微偏向 PyTorch | 极小 | ✅ |
| 9 | ReLU `inplace=True` | 略微偏向 PyTorch | 小 | ✅ 框架实现差异 |
| 10 | torch.compile 尾 batch 预热 | 略微偏向 TR4（计时口径） | 小~中 | ⚠️ 建议补充预热或列为残差 |

### 7.2 对 TR4 有利的因素

| 因素 | 性质 | 影响量级 |
|:---|:---|:---:|
| FusedNormalization 融合预处理 | 架构级能力 | 中等 |
| 原生 NHWC 布局 | 框架设计选择 | 小 |
| C++ 静态图 + CUDA Graph | 核心架构优势 | 核心维度 |
| 线程模型 + 共享内存 | 框架设计选择 | 小 |

### 7.3 对 PyTorch 有利的因素

| 因素 | 性质 | 影响量级 |
|:---|:---|:---:|
| `set_to_none=True` 梯度清零 | 框架优化 | 极小 |
| ReLU `inplace=True` | 框架实现差异 | 小 |
| `torch.compile(max-autotune)` | 核心架构优势 | 核心维度 |

### 7.4 判定原则

1. **架构优势 ≠ 不公平**: 双方的核心优势（TR4 的 CUDA Graph + FusedNormalization，PyTorch 的 torch.compile + 动态 GradScaler）都是各自框架的核心竞争力。
2. **影响量级差异小**: 所有差异项的影响量级均在可接受范围内，不存在一方因单方面因素获得压倒性优势。
3. **核心超参数和算法完全对齐**: 学习率调度、batch size、weight decay、数据增强、网络结构、初始化等已完全一致。

---

## 8. 结论

**TR4 与 PyTorch 在 VGG16BN AMP 训练默认配置下的性能对比是公平的。**

在核心算法、超参数、网络结构、数据增强、AMP 语义、计时口径、学习率调度上做到了严格对齐。存在的差异均属于框架自身架构设计差异，而非测试样例作弊。双方都启用了各自框架的最优执行模式：

- **TR4**: CUDA Graph + FusedNormalization + 原生 NHWC + 静态图
- **PyTorch**: torch.compile(max-autotune) + DDP + SyncBN + channels_last + GradScaler

在各自最优配置下进行对比，结果可以真实反映两个框架在 VGG16BN ImageNet AMP 训练任务上的实际能力水平。

---

## 9. 参考文件

| 文件 | 用途 |
|:---|:---|
| `tests/integration/test_vgg16bn.cpp` | TR4 测试入口 |
| `tests/integration/test_vgg16bn.py` | PyTorch 测试入口 |
| `src/algo/scheduler.cpp` | TR4 CosineAnnealingLR 实现 |
| `src/backend/ops/range/optimizer_op.cpp` | Weight/Bias 优化器路径与 WD 排除 |
| `src/backend/ops/range/grad_scaling_op.cu` | AMP 梯度缩放 CUDA kernel |
| `src/backend/ops/range/allreduce_op.cpp` | NCCL AllReduce 实现 |
| `src/data/fused_normalization.cpp` | FusedNormalization 实现 |
| `src/data/preprocessor.cpp` | 预处理管线与 CPVS/FULLY/DTS 配置 |
| `src/graph/compiler.cpp` | 计算图编译与 CUDA Graph 构建 |
| `src/graph/capture_cuda.cpp` | CUDA Graph 捕获 |
| `src/task/deep_learning_task.cpp` | GPU 训练/验证循环 |
| `include/renaissance/core/global_registry.h` | drop_last、using_cpvs 默认值 |

---

*审计完成时间: 2026-06-21*  
*审计范围: 测试入口、核心超参数、网络结构、数据增强、AMP 实现、分布式通信、数据加载、图执行、计时口径、学习率调度*
