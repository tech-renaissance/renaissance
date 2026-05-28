

# 【今日话题：C=1时的FusedNormalization怎么了？】

众所周知，在AMP关闭的情况下，我们的Preprocessor的最后一步——FusedNormalization会照常输出FP32，颜色通道数与数据集本身一致；但在AMP开启的情况下，理应把C通道自动填充到4通道，不管原始是1通道还是3通道。如果是MNIST那应该是每4个FP16值就只有第一个非零，而CIFAR10那应该是每4个FP16值就只有前3个非零。但是现在似乎CIFAR是正常的，但MNIST有错。我们非常怀疑是FusedNormalization的计算和填充逻辑本身有问题，请小伙伴检查报告。



# 【小伙伴K】

• 你的判断完全正确。我已定位到 MNIST AMP 数据错乱的根本原因：
  src/data/fused_normalization.cpp 中 simd_process_4pixels_c1() 存在 AVX2 数据布局 bug（仅影响 C=1 的 AMP 路径）。
  该函数意图将 4 个单通道像素写成 h0,0,0,0, h1,0,0,0, h2,0,0,0, h3,0,0,0（每个像素 4 通道 FP16），但实际实现犯了两个错误
  ：
  1. 布局错误：_mm_unpacklo_epi32(h, zero) 只产生 h0,0,h1,0,h2,0,h3,0（每个像素仅占 2 通道），且后续 _mm_unpackhi_epi32
     合产生大量重复值。
  2. 步进错误：函数写了 32 个 uint16_t，但外层循环 simd_row_c1_noflip 按 dst + w * 4 步进（仅 16 个 uint16_t），导致相邻
     像素块重叠覆盖。
    结果就是：第一个像素偶尔能残留 b6ca，后续像素被反复覆盖成 0。

  CIFAR（C=3）走 simd_process_2pixels_c3，布局是 R,G,B,0,R,G,B,0，实现正确，所以数据正常。



# 【小伙伴D】

现在我看到了完整数据。让我仔细追踪 C=1 AMP 的 SIMD 逻辑。

关键证据：

解读（NHWC with padded C=4）：

- u16[0..3] = b6ca, 0, 0, 0 → Pixel 0 正确 （b6ca 是黑背景 -0.4243 的 FP16 值，padding 为零）
- u16[4..7] = 0, 0, 0, 0 → Pixel 1 全零！应该是 b6ca, 0, 0, 0
让我逐行追踪 simd_process_4pixels_c1 (line 106-124) 的 SIMD 运算：

然后关键的 mul/sub（line 536-537 设置了 mul_v 和 sub_v）：

继续 SIMD：

展开后结果就是 只有 pixel 0 有数据，pixel 1-3 全是零 。

## 🐛 Bug 定位
文件 ： r:\renaissance\src\data\fused_normalization.cpp
 行号 ：536-537
 函数 ： execute() 中 C==1 AMP 路径

根因 ：

- C=3 路径中： simd_process_2pixels_c3 每次只处理 1 个像素的 3 个通道 （展开成 4 float 其中第4通道=0），所以 mul_v = _mm_set_ps(0, mul[2], mul[1], mul[0]) 是正确的——3 个有效 mul 因子分别对应 3 个通道。
- C=1 路径中： simd_process_4pixels_c1 一次处理 4 个独立像素 。每个 float lane 都是独立像素，都需要同一个 mul[0] 和 sub[0] 。但当前代码只把 mul[0] / sub[0] 放在 lane[0]，lane[1..3] 为 0——导致像素 1-3 被乘以 0，直接归零！
正确的写法 ：应该广播到所有 4 个 lane：

或者等价：

同样的问题也存在于 do_flip 路径（第 542-543 行） ，虽然 C=1 flip 目前走的是 scalar 回退（line 179-187），但那两行 mul_v/sub_v 的构造方式是一样的 bug。

## 📊 影响范围
- MNIST AMP ：每 4 个像素只有第 1 个正确（像素 0, 4, 8, 12, ...），其余 3 个全是零
- CIFAR10 AMP / ImageNet AMP ：不受影响。C=3 路径中 mul_v[0..2] 分别对应 R/G/B 通道，第 4 个 float lane 本身就是 padding（不被使用），所以 mul_v[3]=0 是正确的
- FP32 路径 、 非 AVX2 路径 ：不受影响
## 📋 总结
项目 内容 Bug 位置 fused_normalization.cpp 第 536-537 行（+542-543 同理） Bug 现象 simd_process_4pixels_c1 每 4 像素只正确处理第 1 个，其余 3 个被清零 根因 _mm_set_ps(0,0,0,mul[0]) → 只有 lane[0] 有值；应改为 _mm_set1_ps(mul[0]) 广播到 4 个 lane 设计混淆 C=3 的 mul_v 按"1像素×3通道→4 float"设计是正确的；C=1 的 mul_v 需要"4像素×1通道→4 float"，但错误复用了 C=3 的写法模式



