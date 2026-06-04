# 【今日话题：卷积算子的最终检查】

各位小伙伴。

我们的卷积算子已经基本实施完毕。但还需要进一步检查。

检查时需要明确我们框架已有的约束。我们要做的不是万能的算子，但关键是在支持的范围内功能正确、性能优良。

我们明确卷积不支持bias。我们明确只支持正方形kernel，并且只支持1×1、3×3、5×5、7×7（顶层架构检查，算子内不检查）。所以这些限制不算在bug范围内。

我们明确要求dX覆盖X，也就是反向传播的梯度输出覆盖原始输入，复用同一个张量。

我们要求卷积有一个bn_stats的张量作为保留，尽管只有AMP会写入它，而CPU和GPU FP32版不写入（维持接口统一性）。

我们的张量布局是NHWC，卷积权重是KRSC。

好消息是小伙伴D声称已经完成了单元测试，结果正常。

但大家不要松懈，继续检查那些还存在的问题（如果确实还有问题的话），反复论证，给出修复意见。

如果发现问题，一定要再三确认它真的是个问题，不要误读，以免把对的改错。





# 【小伙伴K】

• 已重新检查。上一轮指出的 6 个问题已全部修复，但我发现了一个残留的 CPU BWD bug 和几个可改进项。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 1（CPU BWD 形状错误）：op_ctx->input_shape 被误用为 X 的形状
  位置：src/backend/ops/dtensor/conv_op.cpp line 663-666（launch_conv_bwd_cpu）
  代码：
  int N = op_ctx->input_shape.n;
  int H = op_ctx->input_shape.h;
  int IW = op_ctx->input_shape.w;
  int C = op_ctx->input_shape.c;
  原因：从 capture_cpu.cpp 可知，op_ctx->input_shape 取自 node.input_ids[0] 对应的 DTensor 形状。对于 Conv BWD，input_id
  s[0] = dY，所以 input_shape = dY.shape = {N, OH, OW, K}。
  但代码把 input_shape.h/w/c 当作 X 的高/宽/通道：
  • H = OH（应为 X.h = H）
  • IW = OW（应为 X.w = W）
  • C = K（应为 X.c = C）
  影响：
  • std::memset(dX, 0, N * H * IW * C * sizeof(float)) 只清零了 N*OH*OW*K 字节，而非完整的 N*H*W*C 字节
  • dgrad 和 wgrad 的循环边界、索引计算全部基于错误的 H/W/C
  • 当 stride>1 或 K≠C 时，CPU BWD 结果完全错误
  修复建议：output_ids[0] 是 dX（in-place 覆盖 X），所以 op_ctx->output_shape 就是 X 的形状。应改为：
  int N  = op_ctx->output_shape.n;
  int H  = op_ctx->output_shape.h;
  int IW = op_ctx->output_shape.w;
  int C  = op_ctx->output_shape.c;
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 2（代码可读性）：CPU FWD naive 中残留死代码
  位置：src/backend/ops/dtensor/conv_op.cpp line 569-570
  const float* B = nullptr;
  // 框架不支持卷积 bias，第 3 个输入是 bn_stats，不使用
  B 后续完全没有被使用（下面的 if (false) sum += 0.0f; 也永远不会执行）。建议直接删除 B 变量和那个死分支，而不是保留 if
  (false)。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  确认已修复的问题
   上一轮问题                                               状态
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   make_nhwc_stride/make_krsc_stride 重复定义导致编译错误   ✅ 已修复（通过 namespace 开关 + 删除 conv_op.cpp 中的定义
                                                            ）
   CPU FWD XNNPACK 将 bn_stats 误作 bias                    ✅ 已修复
   CPU FWD naive 将 bn_stats 误作 bias                      ✅ 已修复（但留了 if(false) 死代码）
   CPU BWD input_ids[1]/[2] 的 X 和 W 顺序颠倒              ✅ 已修复
   AMP FWD 中 bn_stats_ptr 死代码                           ✅ 已修复
   文件末尾缺少换行符                                       ✅ 已修复
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  确认正确的部分
   项目                                                                       状态
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CUDA 6 个 launch 函数 I/O 映射                                             ✅ 正确
   compiler.cpp 三处修改（get_grad_output_id、BWD GraphNode、grad_id 追踪）   ✅ 正确
   layer_descriptor_registry.cpp infer/build 函数                             ✅ 正确
   7 个 cuDNN FE graph builder                                                ✅ 正确
   workspace 管理（ensure_workspace_grow）                                    ✅ 正确
   BWD 跨流同步（wgrad COMP_1 → event → dgrad COMP_3）                        ✅ 正确
   bn_stats 内存布局与 Variant Pack 偏移                                      ✅ 正确
   warmup 走 launch_cuda + cudaDeviceSynchronize                              ✅ 正确
   op_registry.cpp 删除旧 extern 声明                                         ✅ 正确
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  总结

  唯一需要修复的是 CPU BWD 中的 input_shape 误用（应改用 output_shape）。其余均为代码风格/可读性小项，不影响功能。



# 【小伙伴C】

• 经过再次全面检查，以下是问题清单：
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 1: FWD 与 INF cache 仍未共享（之前指出，未修复）
  位置: src/backend/ops/dtensor/conv_op.cpp
   函数                        key 中的 op 字段
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   launch_conv_fp32_fwd_cuda   CONV_FP32_FWD
   launch_conv_fp32_inf_cuda   CONV_FP32_INF ← 不同
   launch_conv_amp_fwd_cuda    CONV_AMP_FWD
   launch_conv_amp_inf_cuda    CONV_AMP_INF ← 不同
  FWD 与 INF 的 cuDNN graph 结构完全相同（都是纯 conv_fprop），但 key 不同导致不命中同一 cache entry。每切换一次 FWD/IN…
  模式就会多构建一次 graph。
  建议: 将 INF 的 key 改为对应的 FWD 值（如 CONV_FP32_INF → CONV_FP32_FWD），共享同一条 cache。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 2: 首层 Conv 的 node.input_ids 缺少 X，导致越界访问（新发现，较严重）
  根因: compiler.cpp Phase 1 中，首层的 prev_output_id = -1。if (prev_output_id >= 0) 不注入 X。而 build_conv_forward/ba
  ckward 的 input_indices 只包含 weight（和 bn_stats），不含 X。只有 gn.input_ids.empty() 的首层算子（如 Flatten）才会 …
  baseline.data_a 注入分支，Conv 不满足此条件。
  后果:
   场景                node.input_ids 实际内容    launch 函数期望              结果
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   首层 FP32 FWD       {W} (1 个元素)             [0]=X, [1]=W                 node.input_ids[1] 越界
   首层 AMP FWD        {W, bn_stats} (2 个元素)   [0]=X, [1]=W, [2]=bn_stats   node.input_ids[0] 取到 W（被当作 X），[2
                                                                               ] 越界
   首层 FP32/AMP BWD   {dY, W} (2 个元素)         [0]=dY, [1]=W, [2]=X         node.input_ids[2] 越界
  越界细节:
  • 新代码没有 TR_CHECK(node.input_ids.size() >= ...) 保护，直接 mp.get_dtensor(node.input_ids[1]) / [2]
  • std::vector::operator[] 不检查边界，返回垃圾值，导致 get_dtensor 访问随机的 DTensor 条目，最终 cuDNN FE 报 BAD_PARAM
    或程序崩溃
  受影响范围: 仅首层 Conv。非首层不受影响（prev_output_id >= 0，X 正确注入）。
  修复方向（在 compiler.cpp 中，非 Conv 实现）:
  • 方案 A: Phase 1 中，所有首层算子（不限于 gn.input_ids.empty()）统一设置 layer_input_ids[l] = baseline.data_a，并在 p
    v_output_id < 0 && layer.is_first_layer 时把 baseline.data_a 注入 gn.input_ids 开头
  • 方案 B: 在 build_conv_forward 的 input_indices 中显式加入输入 X 的索引（但这与跨层链设计冲突）
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 3: bn_stats_ptr 冗余变量已删除（已修复 ✅）
  确认 launch_conv_amp_fwd_cuda 中不再声明未使用的 bn_stats_ptr。VariantPack 填充直接通过循环内的 ctx.ptr_at(tid) 获取指
  针，逻辑正确。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  其余设计点全部正确（再三确认）
   检查项                                                       结果
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   get_grad_output_id(Conv) = -1 + layer_input_ids 回退         ✅
   BWD 双图双流 + cudaEventRecord → cudaStreamWaitEvent         ✅
   AMP 操作数 set_dim 用 padded_c()，stride 用真实值            ✅
   conv_out->set_output(true) 后直接作为 genstats 输入          ✅
   sum/sq_sum 指针偏移 +K 个 float                              ✅
   6 个算子全部注册，CPU fallback 完整                          ✅
   require_warmup() 包含 INF，warmup 用 cudaDeviceSynchronize   ✅
   state.output_stream_idx = i_dx                               ✅
   infer_conv_tensors_with_bn_stats 返回 7 张量，注册正确       ✅
   build_conv_* 的 descs.size() < 7 检查                        ✅
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  总结: 小伙伴删除了 bn_stats_ptr（问题 1 已修），但未修复 FWD/INF cache 共享（问题 2 仍在）。此外，首层 Conv 的 X 注入
  缺失是一个较严重的运行时问题（问题 3），会导致 FWD/BWD 越界访问。这个问题根源在 compiler.cpp 的首层处理逻辑，不在 Con…
  实现本身，但 Conv 是最常见的首层算子，影响最直接。





# 【用户补充】

首层注入问题，非常关键，如果没有考虑某层作为首层的情况，导致I_A_DATA或I_B_DATA没有注入，导致输入缺失，放在DeepLearningTask里跑不通神经网络。我们的MaxPool的首层注入应该是正确的，请参考。包括训练图和推理图。



