• 根据运行日志，我已经定位了核心问题。以下是完整分析：
  🔍 关键发现
  1. scaling 的递减过程已被捕获
    日志显示 SOFTMAX-FWD 的 scaling 值从 65536 开始，每个 batch 减半：
    scaling=65536 → 32768 → 16384 → ... → 1 → 0.000000
    这说明 第一个 batch 的 forward 时 scaling=65536 是正常的，scaling=0 是后续 backward 反复检测到 NaN 的结果，不是根因。
  2. logits 确实有 NaN
    Val 调试输出：
    VAL-logits TOTAL NaN count=1020 / 2000 first_nan_hex=7fff
    first_nan_hex=7fff 是标准的 FP16 quiet NaN。logits 中约 51% 的元素是 NaN。
  3. probs 的 NaN 是 logits 传播的
    probs 有 1473/2000 个 NaN，是因为 softmax kernel 读取了含 NaN 的 logits，导致：
    • max_val 在 reduce 中变成 NaN
    • expf(val - max_val) = NaN
    • sum = NaN, inv_sum = NaN
    • prob = NaN
    ❌ 排除的假设
      假设                 结论
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Padding 垃圾值       ❌ Kernel 只读 c < num_classes=10，padding (10-15) 不会被读取
      Softmax 计算 bug     ❌ 数学上不可能从有限值产生 NaN
      Scaling 初始化问题   ❌ INIT-CHECK 确认 scaling=65536，且递减过程正常
      H2D 数据有 NaN       ❌ MNIST 数据是 0-255 的 uint8，FP16 可精确表示
    🎯 根因定位：FC3 输出（logits）包含 NaN
    FC 层使用 cuDNN FE 1×1 卷积实现。如果输入、权重、偏置都是干净的，cuDNN 不可能产生 NaN。
    唯一可能：cuDNN 实际读取到了 NaN 值，原因可能是：
  1. 权重内存布局与 cuDNN 期望不一致，导致读取了相邻内存的 NaN 垃圾值
  2. cuDNN FE pointwise ADD（FP16 conv_out + FP32 bias）存在数值问题
  3. A_FC_WEIGHT（FP16 工作权重）在 CAST_MAIN 转换时只覆盖了部分内存，未覆盖区域含未初始化的 NaN
    我检查了 make_nhwc_stride 和 cuDNN FE 的 tensor descriptor 设置，理论上 stride 计算是正确的。但由于 StrReplaceFile 工
    具被系统拒绝，我无法在代码中添加更细粒度的调试（如逐层检查 FC1/FC2/FC3 输出的 NaN）。
    🔧 建议下一步
    需要确认 NaN 是从哪一层 FC 开始出现的。建议你在 deep_learning_task.cpp 的 training 循环中，在 每个 g_deep launch 后 添
    加对 F_FEATURE_FP16 区域各张量的 NaN 检查：
    // 在 g_deep 执行后检查各层输出
    for (int id : {22, 29, 32, 39, 42}) {  // FC1/FC2/FC3 输出
      const auto& dt = active_memory_plan_->get_dtensor(id);
      int64_t nelem = dt.n_stride_cuda() * dt.shape.n();  // 按 stride 检查全部
      bool has_nan = debug_check_buffer_nan(ctx.ptr_at(id), nelem, dt.dtype, s_c1, "TRAIN-fc-out");
      if (has_nan) LOG_INFO << "[NAN-CHECK] TRAIN batch=" << batch << " id=" << id << " HAS NAN";
    }
    这样可以确定 NaN 是从 FC1 就开始，还是只在 FC3 出现。
    或者，你也可以直接检查 A_FC_WEIGHT（FP16 权重） 是否有 NaN：
    for (int id : {25, 35, 45}) {  // A_FC_WEIGHT for FC1/FC2/FC3
      const auto& dt = active_memory_plan_->get_dtensor(id);
      int64_t nelem = dt.n_stride_cuda() * dt.shape.n();
      bool has_nan = debug_check_buffer_nan(ctx.ptr_at(id), nelem, dt.dtype, s_c1, "TRAIN-amp-weight");
    }

  如果 A_FC_WEIGHT 有 NaN，说明 CAST_MAIN (FP32→FP16) 有问题；如果 A_FC_WEIGHT 干净但 FC 输出有 NaN，说明 cuDNN 计算本身
  有问题。