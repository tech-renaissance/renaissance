# 【今日话题：如何改造我们的RANGE算子】 



众所周知，我们的算子分为compute（dtensor）和range。

compute算子是操作DTensor，而range算子是操作一段内存。



但是我们的当前实现非常粗糙，而且不科学，需要改造。



 当前项目中的RANGE算子列表：

  数据传输类：

  1. RANGE_H2D_COPY_A - StagingPool A → I_A_LABEL(049) + I_A_DATA(050)
  2. RANGE_H2D_COPY_B - StagingPool B → I_B_LABEL(051) + I_B_DATA(052)
  3. RANGE_H2D_COPY_DTENSOR - Pinned memory → 单个 DTensor（SimpleTask专用）

  类型转换类：

  4. RANGE_CAST_W32_TO_W16 - 010-012 → 022-024
  5. RANGE_CAST_G16_TO_G32_FC - 031 → 028
  6. RANGE_CAST_G16_TO_G32_FIRST - 032 → 029
  7. RANGE_CAST_G16_TO_G32_DEEP - 033 → 030
  8. RANGE_CAST_EMA32_TO_EMA16 - 016-018 → 019-021

  梯度操作类：

  9. RANGE_ZERO_GRAD - 025-033 → 0（梯度清零）
  10. RANGE_NAN_CHECK_ALL_G - 025-030（NaN检测）

  通信类：

  11. RANGE_ALLREDUCE - 梯度AllReduce（桶1+桶2统一算子）
  12. RANGE_BN_STATS_ALLREDUCE - 003-004 in-place
  13. RANGE_BN_STATS_COPY - 003-004 → 001-002

  优化器更新类（Bias）：

  14. RANGE_UPDATE_BIAS_SGD - SGD专用
  15. RANGE_UPDATE_BIAS_MOMENTUM - SGD_MOMENTUM + LARS共用
  16. RANGE_UPDATE_BIAS_NESTEROV - SGD_NESTEROV + LARS_NESTEROV共用
  17. RANGE_UPDATE_BIAS_ADAM - ADAM + ADAMW共用

  优化器更新类（Weight）：

  18. RANGE_UPDATE_WEIGHT_SGD - SGD
  19. RANGE_UPDATE_WEIGHT_MOMENTUM - SGD_MOMENTUM
  20. RANGE_UPDATE_WEIGHT_NESTEROV - SGD_NESTEROV
  21. RANGE_UPDATE_WEIGHT_ADAM - ADAM
  22. RANGE_UPDATE_WEIGHT_ADAMW - ADAMW

  EMA和其他：

  23. RANGE_EMA_PARAM_UPDATE - W07-012 → E13-018
  24. RANGE_SEMA_SWITCH - E13-018 → W07-012

  总共24个RANGE算子

以上是当前的代码情况。

我说一下，这套方案有很大的问题

最主要的问题就是，它把算子的功能直接定死了，使得算子变成了只能操作固定区域、而不能接受参数来调整范围的僵化模式
正确的做法是，一个算子代表一种运算行为，是一个或几个简单的pointwise kernel，操作1~2片区域（输入输出），可以有1~2个标量参数

比如，同样是CAST，你并不需要根据它CAST的对象的语义来区分算子，你只需要区分它把什么类型转换成什么类型



双指定法：指参数有两套重载，一套是(源指针，目标指针，源字节数)，一套是(源起始区域，源结束区域，目标起始区域，目标结束区域)

大概可以约简成以下算子：

RANGE_H2D_COPY_A（只需指定target区域范围）
RANGE_H2D_COPY_B（只需指定target区域范围）

RANGE_H2D_POINT_TO_POINT（指定一个src的锁页内存指针和一个target的显存指针）

RANGE_CAST_FP32_TO_FP16（双指定法）
RANGE_CAST_FP16_TO_FP32（双指定法）

RANGE_CLEAR（指定目标指针+目标字节数，或指定目标起始区域+目标结束区域）
RANGE_D2D_COPY（双指定法）
RANGE_SUM_ALL_REDUCE（指定目标指针+目标字节数，或指定目标起始区域+目标结束区域）
RANGE_MEAN_ALL_REDUCE（指定目标指针+目标字节数，或指定目标起始区域+目标结束区域）

RANGE_CHECK_NAN（指定目标指针+目标字节数+HAS_NAN指针，或指定目标起始区域+目标结束区域+HAS_NAN指针）

以下的不但要指定操作范围，还要指定学习率的指针和动量的指针（视情况）

RANGE_UPDATE_BIAS_SGD - SGD专用
RANGE_UPDATE_BIAS_MOMENTUM - SGD_MOMENTUM + LARS共用
RANGE_UPDATE_BIAS_NESTEROV - SGD_NESTEROV + LARS_NESTEROV共用
RANGE_UPDATE_BIAS_ADAM - ADAM + ADAMW共用
RANGE_UPDATE_WEIGHT_SGD - SGD
RANGE_UPDATE_WEIGHT_MOMENTUM - SGD_MOMENTUM
RANGE_UPDATE_WEIGHT_NESTEROV - SGD_NESTEROV
RANGE_UPDATE_WEIGHT_ADAM - ADAM
RANGE_UPDATE_WEIGHT_ADAMW - ADAMW
RANGE_EMA_PARAM_UPDATE

需要说明的是，学习率和动量参数，都是固定分配好的标量，MemoryPlan持有其ID

依然是MemoryPlan和ComputationGraph平台无关
对于需要传递“指针”的，也是可以用DTensor的ID来代替
但对于锁页内存的指针，我认为可以在launch kernel调用时，从固定的变量里读取

请大家根据这个思路，提出更具体的算子改造方案。

我目前想到的调用形式：

ComputationGraph graph;
graph.append(RangeOp::RANGE_D2D_COPY,
                 {region_src_start.id, region_src_end.id},  // 闭区间
                 {region_target_start.id, region_target_end.id},  // 闭区间
                 );

graph.append(RangeOp::RANGE_CAST_FP32_TO_FP16, offset_src, offset_target, src_bytes);

请大家认真阅读代码，补充完善。