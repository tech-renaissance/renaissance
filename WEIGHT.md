# 【问】

如果某个深度学习框架采用NHWC的张量布局、KRSC的权重格式，那么它的全连接层的权重应该是什么形式？哪两个维度是1？哪个维度是in_features？哪个维度是out_features？



# 【答1】

**（专家：DS）**

在采用NHWC数据布局和KRSC卷积核格式的框架中，全连接层的权重是一个形状为 **[OutFeatures, 1, 1, InFeatures]** 的四维张量。

*   **OutFeatures**：对应格式中的 **K (Kernel)**，代表输出特征数。
*   **InFeatures**：对应格式中的 **C (Channel)**，代表输入特征数。
*   **R, S (均为1)**：代表卷积核的**高度和宽度**。全连接层等效于1x1卷积，因此这两个空间维度都为1。

### 🔎 格式来源与映射关系

这种张量形态的映射关系，源于将全连接层视为一个特殊的 **1x1 卷积**操作：

*   **激活张量 (Input Tensor)**
    *   **数据布局**: **NHWC** = `[Batch, Height, Width, Channel]`
    *   **全连接层等效**: 当 `Height=1` 且 `Width=1` 时，NHWC布局下的输入就变成了 `[Batch, 1, 1, InFeatures]`，这正是全连接层期望的二维输入 `[Batch, InFeatures]` 在4D张量中的表示。

*   **权重张量 (Weight Tensor)**
    *   **数据布局**: **KRSC**。这是由NCHW (通常的卷积核格式) 经维度重排得到的对应格式。
    *   **维度映射**:
        *   **K (Kernel)**: NCHW中的N被K替换，代表**输出通道数** (`out_features`)。
        *   **R, S (Height, Width)**: NCHW中的H和W被R和S替换，代表**卷积核的空间维度**，在全连接层中均为1。
        *   **C (Channel)**: 保持为C，代表**输入通道数** (`in_features`)。

### 💎 总结

因此，在NHWC和KRSC的语境下，一个全连接层的权重形式为 **[out_features, 1, 1, in_features]**。其中空间维度（R和S）为1，K维度代表输出特征数，C维度代表输入特征数。

希望这能清晰地回答你的问题！

# 【答2】

**（专家：KM）**

在 NHWC 数据 + KRSC 权重 的框架里，全连接层（Fully Connected / Linear）本质上就是 1×1 卷积 的退化形式。

---

权重形状

```
[out_features, 1, 1, in_features]
    K          R  S      C
```

维度	值	含义	
K	`out_features`	输出通道数 = 全连接的输出特征维度	
R	1	卷积核高退化为 1（无空间维度）	
S	1	卷积核宽退化为 1（无空间维度）	
C	`in_features`	输入通道数 = 全连接的输入特征维度	

---

为什么是这样

NHWC 框架中，全连接层的输入通常也是 4D 的：

```
[N, 1, 1, in_features]   // H=1, W=1, C=in_features
```

此时用 KRSC 权重 `[out_features, 1, 1, in_features]` 做卷积，等价于矩阵乘法：

```
y = x @ W^T
```

其中：
- `x` 的形状 `[N, in_features]`（由 `[N,1,1,in_features]` 展平）
- `W` 的形状 `[out_features, in_features]`（由 `[K,1,1,C]` 展平）
- 输出为 `[N, out_features]`（即 `[N,1,1,out_features]`）

---

一句话

R 和 S 为 1，K 是 `out_features`，C 是 `in_features`。

这也是 cuDNN 内部把全连接层映射到卷积引擎的标准做法。