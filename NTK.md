# 【小伙伴K】

• 我现在把四个路径获取 shape / stride 的方法逐一列清楚，方便后续判断问题。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  1. GPU FP32 FWD — launch_fc_fwd_cuda
      量                        获取方式
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      batch (N)                 dt_x.shape.n()
      in_features               dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c()
      out_features              p->out_features
      X / W / Y 的 row stride   dt_x.n_stride_cuda() / dt_w.n_stride_cuda() / dt_y.n_stride_cuda()
      实际调用                  cublasGemmEx(..., x_row_stride, w_row_stride, y_row_stride)
    结论：shape 用 logical 的 H*W*C，但 stride 用的是 DTensor 上物理 4D 的 n_stride_cuda()。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. GPU FP32 BWD — launch_fc_bwd_cuda
      量                               获取方式
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      batch                            dt_dy.shape.n()
      out_features                     dt_dy.shape.c()
      in_features                      dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c()
      dY / W / X / dX / dW 的 stride   dt_*.n_stride_cuda()
    结论：和 FWD 一样，shape 算的是 logical H*W*C，stride 用的是物理 4D 的 n_stride_cuda()。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3. AMP FWD — launch_fc_amp_fwd_cuda + build_fc_amp_fwd_conv_graph
      量                                         获取方式
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      运行时检查                                 TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1)
      cache key (N, C, out_features, has_bias)   dt_x.n(), dt_x.c(), dt_w.n(), has_bias
      构建 cuDNN FE graph 的 X 维度              to_fe_dim(dt_x.shape)，即 {s.n(), s.c(), s.h(), s.w()}
      构建 cuDNN FE graph 的 X stride            make_nhwc_stride(dt_x) = {dt.n_stride_cuda(), dt.c_stride_cuda(), dt.h_st
                                              ride_cuda(), dt.w_stride_cuda()}
      W / B / Y 维度                             to_fe_dim(dt_w.shape) / to_fe_dim(dt_b->shape) / to_fe_dim(dt_y.shape)
      运算本质                                   cuDNN FE conv_fprop(X, W)，padding=0, stride=1, dilation=1，用 1×1 conv
                                              模拟 FC
    结论：AMP FWD 硬性要求输入已经是 H=1, W=1，否则在 warmup 阶段直接 TR_DEBUG_CHECK 抛错；即使把 check 注释掉，1×1 conv
    的 graph 也无法接受 H>1 || W>1（会报 cuDNN FE code 11）。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. AMP BWD — launch_fc_amp_bwd_cuda
      量                               获取方式
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      运行时检查                       TR_DEBUG_CHECK(dt_dy.h() == 1 && dt_dy.w() == 1) 和 TR_DEBUG_CHECK(dt_x.h() == 1 &…
                                    dt_x.w() == 1)
      batch                            dt_dy.n()
      out_features                     dt_dy.c()
      in_features                      dt_x.c() ← ⚠️ 这里不是 H*W*C
      dY / W / X / dX / dW 的 stride   dt_*.n_stride_cuda()
      实际调用                         cublasGemmEx(..., x_ns, dy_ns, dx_ns, dw_ns)
    结论：AMP BWD 有两个明显问题：
  1. in_features 计算错误：当 FC 接收 4D 输入 [N,5,5,16] 时，dt_x.c() = 16，但真实 in_features 应该是 5*5*16 = 400。这会
     致 dW 维度是 [400, 120] 与算子声明不一致，GEMM 可能静默写出越界。
  2. stride 同样用的是物理 4D n_stride_cuda()，若 padded_c ≠ C 则 row stride 错误。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    核心问题总结
    所有 FC 算子存在一个共同假设：FC 的输入在 DTensor 层面已经是 [N,1,1,C]，即逻辑 2D 与物理 4D 完全一致。但当 test_lenet
    去掉 Flatten 后，FC 直接消费 [N,5,5,16]：
    • FP32 FWD/BWD：in_features = 400 是对的，但 x_row_stride = n_stride_cuda = padded_c*5*5。如果 padded_c == 16（紧凑）

    碰巧正确；如果 padded_c > 16，则 stride 错误，GEMM 读取的数据就不是连续的逻辑行。
  • AMP FWD：直接崩溃，因为 1×1 conv 无法接受 H=5, W=5。
  • AMP BWD：in_features = dt_x.c() = 16， outright 错误，且同样有 stride 问题。

  也就是说，FC 的四个实现里，没有一个真正支持 4D 输入 [N,H,W,C] 被逻辑展平为 [N,H*W*C] 的情况。这和你之前把 C3 从 Conv
  改成 FC 后仍然不学习的现象完全吻合。





# 【小伙伴C】

• 基于 src/backend/ops/dtensor/fc_op.cpp 的代码，各路径获取形状/stride 的方式如下：
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. FC FP32 GPU — FWD (launch_fc_fwd_cuda)
    形状获取：
    int batch        = dt_x.shape.n();
    int in_features  = dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c();
    int out_features = p->out_features;
    Stride 获取：
    size_t x_row_stride = dt_x.n_stride_cuda();   // 对应 dx 的 n_stride
    size_t w_row_stride = dt_w.n_stride_cuda();   // 对应 weight 的 n_stride
    size_t y_row_stride = dt_y.n_stride_cuda();   // 对应 output 的 n_stride
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. FC FP32 GPU — BWD (launch_fc_bwd_cuda)
    形状获取：
    int batch        = dt_dy.shape.n();
    int out_features = dt_dy.shape.c();           // dY 的 channel = output features
    int in_features  = dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c();
    Stride 获取：
    // dW GEMM
    size_t x_row_stride  = dt_x.n_stride_cuda();
    size_t dy_row_stride = dt_dy.n_stride_cuda();
    size_t dw_row_stride = dt_dw.n_stride_cuda();

  // dX GEMM
  size_t dx_row_stride = dt_dx.n_stride_cuda();
  size_t w_row_stride  = dt_w.n_stride_cuda();

  // db kernel
  size_t dy_row_stride = dt_dy.n_stride_cuda();
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3. FC AMP GPU — FWD (launch_fc_amp_fwd_cuda)
    形状获取：
    TR_DEBUG_CHECK(dt_x.h()==1 && dt_x.w()==1);   // 断言 H=1, W=1
    // 实际 shape 传入 cuDNN FE graph：
    auto X = graph->tensor(...set_dim(to_fe_dim(dt_x.shape))...);
    auto W = graph->tensor(...set_dim(to_fe_dim(dt_w.shape))...);
    Stride 获取（NHWC stride）：
    auto make_nhwc_stride = [](const DTensor& dt) -> std::vector<int64_t> {
      return {dt.n_stride_cuda(), dt.c_stride_cuda(),
              dt.h_stride_cuda(), dt.w_stride_cuda()};
    };
    // 分别传给 X 和 W：
    .set_stride(make_nhwc_stride(dt_x))
    .set_stride(make_nhwc_stride(dt_w))
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. FC AMP GPU — BWD (launch_fc_amp_bwd_cuda)
    形状获取：
    int batch        = dt_dy.n();
    int out_features = dt_dy.c();
    int in_features  = dt_x.c();   // ← 这里只取了 c()，不是 h()*w()*c()！
    Stride 获取：
    size_t dy_row_stride = dt_dy.n_stride_cuda();   // dY 的 n_stride
    size_t dx_row_stride = dt_x.n_stride_cuda();    // dX 的 n_stride
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  5. FC CPU Eigen (launch_fc_fwd_cpu_eigen / launch_fc_bwd_cpu_eigen)
    形状获取：
    int batch        = op_ctx->input_shape.n;
    int in_features  = op_ctx->input_shape.h * op_ctx->input_shape.w * op_ctx->input_shape.c;
    int out_features = p->out_features;
    CPU Eigen 直接通过 op_ctx->input_shape 获取，无 padding，所有维度都是紧凑布局。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    关键差异总结
      路径        in_features 计算                使用的 shape 方法
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      FP32 FWD    h()*w()*c()                     dt_x.shape.h/w/c/n
      FP32 BWD    h()*w()*c()                     dt_x.shape.h/w/c, dt_dy.shape.c/n
      AMP FWD     传完整 dt_x.shape 给 cuDNN FE   dt_x.shape + to_fe_dim()
      AMP BWD     dt_x.c() 而非 h()*w()*c()       dt_dy.n/c, dt_x.c
      CPU Eigen   h*w*c                           op_ctx->input_shape.h/w/c/n

  最明显的 bug 是 AMP BWD 的 int in_features = dt_x.c()。对于 4D 输入如 [128,5,5,16]，这会把 in_features 算成 16 而不是
  400，导致 cuBLAS GEMM 参数严重错误。



# 【小伙伴D】

## FC GPU/AMP 各实现获取 shape 和 stride 的方式对比
### FP32 FWD（第 480-486 行）
信息 获取方式 正确性 batch dt_x.shape.n() - in_features dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c() H×W×C out_features p->out_features （参数） - stride dt_x.n_stride_cuda() / dt_w.n_stride_cuda() / dt_y.n_stride_cuda() n_stride = H×W×C

### FP32 BWD（第 567-575 行）
信息 获取方式 正确性 batch dt_dy.shape.n() - out_features dt_dy.shape.c() - in_features dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c() H×W×C stride dt_dy.n_stride_cuda() / dt_w.n_stride_cuda() / dt_x.n_stride_cuda() / dt_dx.n_stride_cuda() / dt_dw.n_stride_cuda() n_stride = H×W×C

### AMP FWD（第 267-269 行 + 第 123-126 行）
信息 获取方式 正确性 cache key 的 in_features dt_x.c() 仅 C，不是 H×W×C cuDNN graph stride make_nhwc_stride(dt) = {dt.n_stride_cuda(), dt.c_stride_cuda(), dt.h_stride_cuda(), dt.w_stride_cuda()} stridden 但顺序是 {N,C,H,W} 配合 to_fe_dim 的 {N,C,H,W}

另外还有 DEBUG_CHECK： dt_x.h() == 1 && dt_x.w() == 1 （第 253 行）， 强制要求 H=W=1 。

### AMP BWD（第 355-363 行）
信息 获取方式 正确性 batch dt_dy.n() - out_features dt_dy.c() - in_features dt_x.c() 仅 C，不是 H×W×C stride dt_dy.n_stride_cuda() / dt_w.n_stride_cuda() / dt_x.n_stride_cuda() / dt_dx.n_stride_cuda() / dt_dw.n_stride_cuda() n_stride = H×W×C

另外也有 DEBUG_CHECK 限制 H=W=1（第 341-346 行）。

### CPU（Eigen + 朴素，第 687-833 行）
信息 获取方式 正确性 FWD in_features op_ctx->input_shape.h * op_ctx->input_shape.w * op_ctx->input_shape.c H×W×C BWD in_features op_ctx->output_shape.h * op_ctx->output_shape.w * op_ctx->output_shape.c H×W×C （dX shape = X shape）

### 结论
FP32 和 CPU 路径全部正确使用 H×W×C ，只有 AMP 的 FWD 和 BWD 错误地使用了 dt_x.c() ：

- AMP FWD：cache key 里的 in_features 用 dt_x.c() （269行）
- AMP BWD： in_features 用 dt_x.c() （357行）
对于 LeNet-5（H=W=1）， C == H×W×C ，所以这个 bug 恰好不暴露。但对于一般 4D 张量（H≠1 或 W≠1），这会直接导致维度错误。