

咱们一次性解决这三个问题：第一是ZERO_GRAD和CHECK_NAN的连续性问题，第二是每次取回的结果是最后一个batch的结果的问题，第三是最后一个不完整的batch的更新的问题

• enum 连续性报告：
  G_BN_BIAS         = 25  ✓
  G_BN_WEIGHT       = 26  ✓
  G_FC_BIAS         = 27  ✓
  G_FC_WEIGHT       = 28  ✓
  G_FIRST_CONV      = 29  ✓
  G_DEEP_CONV       = 30  ✓
  R_RESULT          = 31  ← 结果区（loss/top1/top5 三个 scalar）
  G_FC_WEIGHT_FP16  = 32  ✓
  G_FIRST_CONV_FP16 = 33  ✓
  G_DEEP_CONV_FP16  = 34  ✓

为什么我要把R_RESULT放在31区、夹在FP32梯度和FP16梯度之间呢？
因为我强制要求通信最多只分两桶：DEEP_COMM和FIRST_COMM
DEEP_COMM处理的是G_DEEP_CONV区（非首层的卷积权重）和R_RESULT区
FIRST_COMM处理的是G_BN_BIAS~G_FC_WEIGHT这几个区
每个batch总共只启动2次通信！绝对没有第三次！这样R_RESULTS的结果就能跟着梯度一起完成通信，不需要多启动一次通信
然后就是，ZERO_GRAD的范围，大家都会疑惑。我的设计是：直接覆盖G_BN_BIAS~G_DEEP_CONV_FP16的全部区域。如果是AMP模式，G_FC_WEIGHT_FP16~G_DEEP_CONV_FP16会有张量，如果不是就没有。没有张量也不影响，因为计算范围的时候，没有张量的区域的大小就是0字节。对于范围化算子，只要你知道它需要处理的首地址和需要处理的字节数就行了

那么问题来了：R_RESULT被清零了怎么办？答案是：你还需要有一个不被清零的累积区R_RESULT_ACCUMULATED。
你的scalar区域还需要一个名为local_batch_size、一个名为last_train_batch_size、一个名为last_val_batch_size的常驻DTensor，在baseline里面添加。

然后呢，每个batch的最后，你需要有一个专门的新的GraphId，它的作用就是把R_RESULT的三个值分别乘上batch_size（具体是哪个，视情况而定，必须捕获成不同的Graph），累加到R_RESULT_ACCUMULATED的对应位置

这样，每个epoch从R_RESULT_ACCUMULATED取一次结果的话，取出来的就是正确的结果。
另外，last batch在更新权重的时候，需要用到batch size，这个时候也要使用last_train_batch_size。

总之，ZERO_GRAD和DEEP_COMM和CHECK_NAN，都必须是一个kernel launch完成任务。CHECK_NAN是不需要处理FP16梯度的，它只需要处理所有FP32梯度，也就是G_BN_BIAS~G_DEEP_CONV

请大家仔细分析，给出一套完整、科学、合理的方案。我说的这些肯定是能做到的，而且不难。因为我们的框架从一开始就被设计成这样