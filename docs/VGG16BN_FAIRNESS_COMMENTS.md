# 公平性检查结论



# 【小伙伴S】

  公平性结论：公平

  根据 docs/VGG16BN_FAIRNESS.md 的详细审计报告，在默认配置下（TR4 使用 AMP，不开启 CPVS 和 FULLY），对比是公平的。

  核心对齐项：

  - 网络结构: VGG-16-BN 完全一致（5个conv blocks + 3个FC层）
  - 超参数: 学习率 0.36、batch size 2048、weight decay 1e-4、momentum 0.9（Nesterov）
  - 调度器: CosineAnnealing + Warmup(10) 数学等价
  - 数据增强: RRC、HFlip、ColorJitter、RandomErasing 参数完全一致
  - AMP语义: 双方都使用FP16训练 + FP32 master weights
  - Weight Decay: BN/bias排除，数学等价

  已知差异（均为架构级差异）：

  1. FusedNormalization - TR4架构优势（中等影响）
  2. Channel Padding - TR4框架约束，略微偏向PyTorch（小~中影响）
  3. 计算图模型 - 各自核心竞争力（核心维度）
  4. 内存布局 - TR4原生NHWC vs PyTorch channels_last（小影响）

  所有差异均属于框架自身架构或实现能力差异，不存在系统性单向不公平，符合公平对比原则。



# 【小伙伴K】

• 结论：默认 AMP（不开启 --cpvs / --fully）配置下，TR4 与 PyTorch 的 VGG16BN 公平性对比是 ✅ 公平的。
  已审阅 docs/VGG16BN_FAIRNESS.md 并对 test_vgg16bn.cpp 与 test_vgg16bn.py 做了交叉核对。关键 LR warmup 已修复对齐（peak 出现在 epoc…
  11，打印 LR 0.36）。
  主要已识别差异均已落档且可辩护：
   #      差异                                                                                 影响    公平性
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   5.10   torch.compile 尾 batch 预热未覆盖，计时循环内可能重编译                              小~中   ⚠️ 已在文档中标记为建议改进
   5.11   channel_padding() 使首层 Conv 输入通道为 8，fan_in=72，Kaiming 初始化幅度约为 PyTo   小~中   ✅ 框架内部对齐约束，无法消除
          rch（fan_in=27）的 sqrt(27/72)≈0.61
  我已把 5.11 的文档补充完整，明确指出了 fan_in 初始化差异及其量级。

  其余超参数、网络结构、数据增强、BN、优化器、LR schedule、weight decay、label smoothing、TF32、drop_last、随机种子、验证口径等均一致
  或可忽略。因此，在“公平对比而非完全一致”的原则下，默认配置公平。



# 【小伙伴C】

• 结论：公平。
  默认配置（C++ --amp，不开启 --cpvs/--fully）下，两个入口在核心算法、超参数、网络结构、学习率调度、数据增强、AMP 语义、
  BN/WD 处理、随机种子等方面都已对齐，不存在单方面作弊。
  残留差异主要是两类：
  1. 框架实现差异（可接受）：channel_padding、NHWC vs channels_last、torch.compile vs CUDA Graph、FusedNormalization、in
     ce ReLU 等。
  2. 计时口径残差（轻微）：PyTorch warm-up 没预热最后不完整的 train/val batch，可能让计时里多一次 torch.compile 重编译。
     议要么在 warm-up 中补一个尾 batch 预热，要么在报告中明确列为残差。

  这两点不改变“公平”的总体判定。



# 【小伙伴D】

经过对三个文件的全面对比检查，结论如下：

## 结论：公平
核心超参数、网络结构、数据增强、学习率调度、AMP 语义、计时口径均已严格对齐。所有差异均属于框架自身架构或实现能力差异（如 FusedNormalization、CUDA Graph vs torch.compile、NHWC vs channels_last、channel_padding 等），不存在测试样例作弊或系统性单向不公平。