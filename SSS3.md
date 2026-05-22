# 【今日话题：升级到真正的多流架构】



# 【小伙伴S】

  🔥 三流架构升级完整方案V2.0（基于SSS2.md深度优化）

  📊 一、核心设计原则（来自用户的关键补充）

  1.1 in-place梯度约束的致命性 ⚠️

  Legacy假设：X和dX共存（独立缓冲区）
  我们实际： dX覆盖X（节省内存）

  FC BWD的数据依赖链：
    dW = gemm(X, dY^T)  ← 需要读取X
    dX = gemm(W, dY)    ← 覆盖X的位置

  绝对约束：dW必须先于dX计算！

  1.2 惯用计算流分配（静态策略）

  FWD算子：
  - Conv/FC/GAP:  COMP_1
  - BN/Pool:      COMP_2
  - ReLU/Act:     COMP_3

  BWD算子（关键！）：
  - dW: COMP_1（必须先执行）
  - dB: COMP_2（与dW并行）
  - dX: COMP_3（必须等dW完成）

  设计优势：
  1. 相邻算子天然不同流（ResNet无连续同类型层）
  2. 事件同步目标静态确定（dX总在COMP_3）
  3. 零运行时决策开销

  1.3 三流加速的真谛 💡

  【反直觉发现】：
  并行 → 多流争抢、缓存互踢 → 可能更慢！
  串行 → 延迟隐藏 + L2缓存预热 → 真正收益！

  核心收益来源：
  1. 串行流水线化（3% FWD加速）
  2. 调度器提前就绪（微秒级热启动）
  3. L2缓存复用（减少HBM访问）

---
  🛠️ 二、最小侵入方案V2.0（完善版）

  核心思想：
  1. ✅ 不修改ComputationGraph IR（符合定律一）
  2. ✅ 不新增replay_cuda_capture函数指针（避免过度设计）
  3. ✅ 算子自治模式（每个算子内部处理自己的流依赖）
  4. ✅ 预注册三条计算流（解决多流屏障关键bug）

---
  📋 三、具体实施方案

  阶段1：capture基础设施修复（P0，0.5天）

  关键发现：小伙伴D发现了致命bug！

  ┌───────────────────────────────────────────┬──────────────────────────┬─────────────────────────────┬───────────┐
  │                    Bug                    │           位置           │            影响             │   修复    │
  ├───────────────────────────────────────────┼──────────────────────────┼─────────────────────────────┼───────────┤
  │ 只预注册primary_stream                    │ capture_cuda.cpp:L66     │ COMP_2/3的event不会被barrie │ 预注册3条 │
  │                                           │                          │ r等待                       │ 流        │
  ├───────────────────────────────────────────┼──────────────────────────┼─────────────────────────────┼───────────┤
  │ insert_cross_op_barrier缺少has_pending_wo │ capture_multi_stream.cpp │ 对未活跃流插入wait →        │ 新增检查  │
  │ rk检查                                    │ :L75                     │ undefined behavior          │           │
  └───────────────────────────────────────────┴──────────────────────────┴─────────────────────────────┴───────────┘

  修复代码：

  // capture_cuda.cpp:L66
  // 修改前：
  state.get_or_register(primary_stream);

  // 修改后：
  state.get_or_register(primary_stream);
  state.get_or_register(static_cast<cudaStream_t>(
      ctx.stream(StreamKind::COMP_2)));
  state.get_or_register(static_cast<cudaStream_t>(
      ctx.stream(StreamKind::COMP_3)));

  // capture_multi_stream.cpp:L72-78
  void insert_cross_op_barrier(...) {
      int out_idx = state.output_stream_idx;
      if (out_idx < 0) return;

      for (int i = 0; i < state.num_active; ++i) {
          if (i == out_idx) continue;
          if (!state.streams[i].has_pending_work) continue;  // ← 新增！
          cudaStreamWaitEvent(state.streams[i].stream,
                             state.streams[out_idx].last_done_event, 0);
      }
  }

---
  阶段2：算子内部依赖等待机制（P0，1天）

  关键创新：算子自治模式

  每个算子在自己的launch开头等待前一个算子的输出：

  // 添加到每个launch_cuda函数开头
  static inline void wait_for_prev_output(
      const GraphNode& node,
      const DeviceContext& ctx,
      MultiStreamCaptureState& state)
  {
      cudaStream_t my_stream = static_cast<cudaStream_t>(
          ctx.stream(get_op_default_stream(node.compute_op)));

      int out_idx = state.output_stream_idx;
      if (out_idx >= 0) {
          cudaStream_t prev_stream = state.streams[out_idx].stream;
          if (prev_stream != my_stream) {
              cudaStreamWaitEvent(my_stream,
                                 state.streams[out_idx].last_done_event, 0);
          }
      }
  }

  在每个launch_cuda函数开头调用：
  static void launch_fc_amp_fwd_cuda(...) {
      wait_for_prev_output(node, ctx, state);  // ← 新增

      // 原有代码...
      cudaStream_t s = static_cast<cudaStream_t>(
          ctx.stream(get_op_default_stream(node.compute_op)));
      // ...
  }

---
  阶段3：FC BWD三流分解（核心收益，1天）

  完整实现（符合用户约束）：

  static void launch_fc_amp_bwd_cuda(
      const GraphNode& node,
      const MemoryPlan& mp,
      const DeviceContext& ctx,
      MultiStreamCaptureState& state)
  {
      // ... 参数准备 ...

      // ===== 获取三条流 =====
      cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
      cudaStream_t s_db = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
      cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
    
      cublasHandle_t h_dw = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_1));
      cublasHandle_t h_db = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_2));
      cublasHandle_t h_dx = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_3));
    
      int i_dw = state.get_or_register(s_dw);
      int i_db = state.get_or_register(s_db);
      int i_dx = state.get_or_register(s_dx);
    
      // ===== COMP_2: dB (bias grad) =====
      if (has_bias) {
          launch_fc_bwd_db_amp_kernel(dy, db, batch, out_features, dy_ns, s_db);
          cudaEventRecord(state.streams[i_db].last_done_event, s_db);
          state.streams[i_db].has_pending_work = true;
      }
    
      // ===== COMP_1: dW (weight grad) - 必须先于dX =====
      cublasGemmEx(h_dw, CUBLAS_OP_N, CUBLAS_OP_T,
                   in_features, out_features, batch,
                   &alpha, x, CUDA_R_16F, x_ns,
                   dy, CUDA_R_16F, dy_ns,
                   &beta, dw, CUDA_R_16F, dw_ns,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
      // dW完成事件
      cudaEvent_t ev_dw_done = state.alloc_temp_event();
      cudaEventRecord(ev_dw_done, s_dw);
      cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
      state.streams[i_dw].has_pending_work = true;
    
      // ===== COMP_3: dX (input grad) - 等待dW完成 =====
      cudaStreamWaitEvent(s_dx, ev_dw_done, 0);
      cublasGemmEx(h_dx, CUBLAS_OP_N, CUBLAS_OP_N,
                   in_features, batch, out_features,
                   &alpha, w, CUDA_R_16F, w_ns,
                   dy, CUDA_R_16F, dy_ns,
                   &beta, dx, CUDA_R_16F, dx_ns,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
      cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
      state.streams[i_dx].has_pending_work = true;
    
      // 下游GAP_BWD依赖dX输出，设为COMP_3
      state.output_stream_idx = i_dx;
  }

---
  阶段4：惯用流映射表（新增文件，0.5天）

  创建src/backend/op_stream_policy.cpp：

  #include "renaissance/backend/op_stream_policy.h"
  #include "renaissance/core/op_kind.h"

  namespace tr {

  StreamKind get_op_default_stream(ComputeOp op) noexcept {
      switch (op) {
          // ===== 卷积/FC/GAP/池化 → COMP_1 =====
          case ComputeOp::CONV_FP32_FWD:
          case ComputeOp::CONV_AMP_FWD:
          case ComputeOp::CONV_FP32_BWD:
          case ComputeOp::CONV_AMP_BWD:
          case ComputeOp::FC_FP32_FWD:
          case ComputeOp::FC_AMP_FWD:
          case ComputeOp::GAP_FP32_FWD:
          case ComputeOp::GAP_AMP_FWD:
          case ComputeOp::GAP_FP32_BWD:
          case ComputeOp::GAP_AMP_BWD:
          case ComputeOp::MAXPOOL_FWD:
          case ComputeOp::MAXPOOL_BWD:
          case ComputeOp::AVGPOOL_FWD:
          case ComputeOp::AVGPOOL_BWD:
              return StreamKind::COMP_1;

          // ===== BN → COMP_2 =====
          case ComputeOp::BN1D_FP32_FWD:
          case ComputeOp::BN1D_AMP_FWD:
          case ComputeOp::BN1D_FP32_BWD:
          case ComputeOp::BN1D_AMP_BWD:
          case ComputeOp::BN2D_FP32_FWD:
          case ComputeOp::BN2D_AMP_FWD:
          case ComputeOp::BN2D_FP32_BWD:
          case ComputeOp::BN2D_AMP_BWD:
              return StreamKind::COMP_2;
    
          // ===== 激活函数 → COMP_3 =====
          case ComputeOp::RELU_FP32_FWD:
          case ComputeOp::RELU_AMP_FWD:
          case ComputeOp::RELU_FP32_BWD:
          case ComputeOp::RELU_AMP_BWD:
          case ComputeOp::TANH_FP32_FWD:
          case ComputeOp::TANH_AMP_FWD:
          case ComputeOp::TANH_FP32_BWD:
          case ComputeOp::TANH_AMP_BWD:
          case ComputeOp::SIGMOID_FP32_FWD:
          case ComputeOp::SIGMOID_AMP_FWD:
              return StreamKind::COMP_3;
    
          default:
              return StreamKind::COMP_1;
      }
  }

  } // namespace tr

---
  📊 四、完整修改清单

  ┌─────┬─────────────────────────────────────┬────────────────────────────────┬──────┬────────┐
  │  #  │                文件                 │            改动内容            │ 行数 │ 优先级 │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 1   │ capture_cuda.cpp:L66                │ 预注册3条计算流                │ +2   │ P0     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 2   │ capture_multi_stream.cpp:L75        │ insert增加has_pending_work检查 │ +1   │ P0     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 3   │ src/backend/op_stream_policy.h      │ 新增惯用流映射声明             │ ~15  │ P0     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 4   │ src/backend/op_stream_policy.cpp    │ 实现get_op_default_stream      │ ~60  │ P0     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 5   │ src/backend/ops/dtensor/gap_op.cpp  │ 4个函数用get_op_default_stream │ ~8   │ P1     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 6   │ src/backend/ops/dtensor/relu_op.cpp │ 6个函数用get_op_default_stream │ ~12  │ P1     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 7   │ src/backend/ops/dtensor/fc_op.cpp   │ FC BWD三流分解                 │ ~60  │ P0     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 8   │ 所有launch_cuda函数                 │ 开头增加wait_for_prev_output   │ ~48  │ P0     │
  ├─────┼─────────────────────────────────────┼────────────────────────────────┼──────┼────────┤
  │ 9   │ src/CMakeLists.txt                  │ 添加op_stream_policy.cpp       │ +2   │ P0     │
  └─────┴─────────────────────────────────────┴────────────────────────────────┴──────┴────────┘

  总计：~208行代码，预计3-4天工作量

---
  ✅ 五、预期收益与风险控制

  5.1 性能收益预测

  ┌────────────┬──────────┬──────────┬──────┬───────────────────┐
  │  测试场景  │ 当前耗时 │ 预期耗时 │ 提升 │       原理        │
  ├────────────┼──────────┼──────────┼──────┼───────────────────┤
  │ GAP+FC FWD │ 132.57μs │ ~125μs   │ ~6%  │ 调度延迟隐藏      │
  ├────────────┼──────────┼──────────┼──────┼───────────────────┤
  │ GAP+FC BWD │ 480.60μs │ ~425μs   │ ~12% │ dW                │
  ├────────────┼──────────┼──────────┼──────┼───────────────────┤
  │ 总计       │ 613.17μs │ ~550μs   │ ~10% │ 接近Legacy的469μs │
  └────────────┴──────────┴──────────┴──────┴───────────────────┘

  5.2 风险控制矩阵

  ┌─────────────────────┬──────┬──────┬────────────────────────────────────────────────┐
  │        风险         │ 概率 │ 影响 │                    缓解措施                    │
  ├─────────────────────┼──────┼──────┼────────────────────────────────────────────────┤
  │ dX覆盖X导致数据竞争 │ 低   │ 致命 │ 强制dW(S1)先于dX(S3)，事件同步保证             │
  ├─────────────────────┼──────┼──────┼────────────────────────────────────────────────┤
  │ 多流workspace不足   │ 低   │ 中   │ DeviceContext已支持per-stream动态增长          │
  ├─────────────────────┼──────┼──────┼────────────────────────────────────────────────┤
  │ 跨流barrier遗漏     │ 低   │ 高   │ ①预注册3流 ②has_pending_work检查 ③算子自治等待 │
  ├─────────────────────┼──────┼──────┼────────────────────────────────────────────────┤
  │ 正确性回归          │ 中   │ 高   │ test_gap_correction必须PASS，MSE验证           │
  ├─────────────────────┼──────┼──────┼────────────────────────────────────────────────┤
  │ cuBLAS多流冲突      │ 低   │ 低   │ per-stream handle隔离                          │
  └─────────────────────┴──────┴──────┴────────────────────────────────────────────────┘

---
  🎯 六、立即行动方案

  第一天（基础设施）：
  1. 修复capture预注册bug（3行代码）
  2. 修复insert_cross_op_barrier（1行代码）
  3. 创建op_stream_policy.cpp（60行代码）

  第二天（算子改造）：
  1. FC BWD三流分解（60行代码）
  2. 所有算子增加wait_for_prev_output（48行代码）
  3. 编译测试

  第三天（验证）：
  1. 正确性测试：test_gap_correction --amp
  2. 性能测试：test_gap_fc_perf --amp
  3. A100实测：目标BWD从480μs降至~425μs

  第一个里程碑：GAP+FC总计从613μs降至~550μs。

---
  ✅ 七、方案优势总结

  1. ✅ 最小侵入：不修改IR，不新增函数指针，不破坏现有架构
  2. ✅ 算子自治：每个算子自己处理流依赖，框架层只提供基础设施
  3. ✅ 符合用户约束：严格遵守dW先于dX、惯用流分配、串行延迟隐藏原则
  4. ✅ 工程可行：3-4天工作量，风险可控，收益明确
  5. ✅ 可扩展性：为未来CBR算子三流拆分预留接口



# 【小伙伴K】

  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一、代码实际状态的关键发现
  1.1 op_registry.h 没有 replay_cuda_capture 函数指针
  小伙伴 S 的方案 A 假设可以在 CuLaunchEntry 中增加 prefer_stream 和 replay_cuda_capture 字段。但代码实际状态是：
  // include/renaissance/backend/op_registry.h:L45-56
  struct ComputeOpEntry {
      ComputeOp op = ComputeOp::UNKNOWN;
      void (*launch_cpu)(CpuOpContext* ctx) = nullptr;
  #ifdef TR_USE_CUDA
      void (*launch_cuda)(const GraphNode&, const MemoryPlan&,
                          const DeviceContext&, MultiStreamCaptureState&) = nullptr;
  #endif
  };
  结论：新增 replay_cuda_capture 需要修改注册表结构、所有注册代码、以及 register_op_* 系列函数。这是一个架构级改动，与小
  伙伴 S 估计的"~100 行"相去甚远。
  1.2 insert_cross_op_barrier 存在根本语义缺陷（小伙伴 D 正确发现）
  当前已修复 self-wait，但仍有三个隐式 bug：
  // capture_multi_stream.cpp:L68-79
  void insert_cross_op_barrier(...) {
      int out_idx = state.output_stream_idx;
      if (out_idx < 0) return;
      for (int i = 0; i < state.num_active; ++i) {
          if (i == out_idx) continue;
          // ❌ BUG 1: 未检查 has_pending_work，对未记录事件的流调用 WaitEvent 是 UB
          cudaStreamWaitEvent(state.streams[i].stream,
                             state.streams[out_idx].last_done_event, 0);
      }
  }
  BUG 2： barrier 语义是"让已有 pending work 的其他流等上一个算子"，不是"让下一个算子等上一个算子"。
  BUG 3：如果下一个算子要用的流尚未注册，insert_cross_op_barrier 永远不会为它插入 wait。
  这导致 FWD 图 GAP(COMP_2) → FC(COMP_1) 中，FC 不会等 GAP！
  1.3 默认 replay 函数也硬编码 COMP_1
  // capture_cuda.cpp:L140-153
  static void replay_compute_node_default(...) {
      cudaStream_t s1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
      // ...
  }
  这些 fallback 路径如果不改，在多流图中会成为静默的正确性陷阱。
  1.4 cuBLAS handle 已是 per-stream
  cublasHandle_t h = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_1));
  DeviceContext 已为每个 StreamKind 维护独立的 cuBLAS/cuDNN handle。多流并发使用不同 handle 是线程安全且已就绪的。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  二、设计方案：框架集中控制 + 复合算子自治
  2.1 核心设计原则
   原则                   说明
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   不改 IR                符合 P_ULTIMATE.md 定律一，ComputationGraph 保持纯 IR
   不新增函数指针         不引入 replay_cuda_capture，避免注册表结构变动
   静态惯用流             零运行时决策开销，流分配是编译期确定的
   框架处理简单算子依赖   单流算子的跨流 wait 由 capture 循环统一处理，避免代码重复
   复合算子内部自治       FC BWD 内部 fork/join 逻辑由算子自行管理，框架不侵入
  2.2 惯用流映射（修正小伙伴 K 的错误）
  小伙伴 K 将 GAP 分到 COMP_1，但 SSS.md 原文明确："BN 和池化用 COMP_2"。GAP 是 Global Average Pooling，属于池化。
  // 新增 src/backend/op_stream_policy.h
  #pragma once
  #include <renaissance/core/types.h>
  #include <renaissance/backend/op_kind.h>

  namespace tr {

  StreamKind get_op_default_stream(ComputeOp op);

  } // namespace tr

  // 新增 src/backend/op_stream_policy.cpp
  StreamKind get_op_default_stream(ComputeOp op) {
      switch (op) {
          // ========== 卷积/FC/池化类 FWD → COMP_1 ==========
          case ComputeOp::CONV_FP32_FWD:  case ComputeOp::CONV_AMP_FWD:
          case ComputeOp::FC_FP32_FWD:    case ComputeOp::FC_AMP_FWD:
          case ComputeOp::MAXPOOL_FWD:    case ComputeOp::MAXPOOL_BWD:
              return StreamKind::COMP_1;

          // ========== GAP 属于池化 → COMP_2 ==========
          case ComputeOp::GAP_FP32_FWD:   case ComputeOp::GAP_AMP_FWD:
          case ComputeOp::GAP_FP32_BWD:   case ComputeOp::GAP_AMP_BWD:
              return StreamKind::COMP_2;
    
          // ========== BN 类 → COMP_2 ==========
          case ComputeOp::BN1D_FP32_FWD:  case ComputeOp::BN1D_AMP_FWD:
          case ComputeOp::BN1D_FP32_BWD:  case ComputeOp::BN1D_AMP_BWD:
          case ComputeOp::BN1D_FP32_INF:  case ComputeOp::BN1D_AMP_INF:
          case ComputeOp::BN2D_FP32_FWD:  case ComputeOp::BN2D_AMP_FWD:
          case ComputeOp::BN2D_FP32_BWD:  case ComputeOp::BN2D_AMP_BWD:
          case ComputeOp::BN2D_FP32_INF:  case ComputeOp::BN2D_AMP_INF:
              return StreamKind::COMP_2;
    
          // ========== 激活类 → COMP_3 ==========
          case ComputeOp::RELU_FP32_FWD:  case ComputeOp::RELU_AMP_FWD:
          case ComputeOp::RELU_FP32_BWD:  case ComputeOp::RELU_AMP_BWD:
          case ComputeOp::RELU_FP32_INF:  case ComputeOp::RELU_AMP_INF:
          case ComputeOp::TANH_FP32_FWD:  case ComputeOp::TANH_AMP_FWD:
          case ComputeOp::TANH_FP32_BWD:  case ComputeOp::TANH_AMP_BWD:
              return StreamKind::COMP_3;
    
          // ========== BWD 复合算子：代表流为 dX 输出流 ==========
          case ComputeOp::FC_FP32_BWD:    case ComputeOp::FC_AMP_BWD:
          case ComputeOp::CONV_FP32_BWD:  case ComputeOp::CONV_AMP_BWD:
              return StreamKind::COMP_3;  // dX 所在流，下游依赖它
    
          default:
              return StreamKind::COMP_1;
      }
  }
  ▌ 为什么 FC/CONV BWD 返回 COMP_3？ get_op_default_stream 的语义是"当前算子的代表流"——即 capture 循环用来等待上游算子
  ▌ 的那个流。FC BWD 的下游依赖的是 dX，而 dX 在 COMP_3 上。
  2.3 框架层：capture 循环增强
  // src/graph/capture_cuda.cpp
  void CapturedGraph::capture_cuda(...) {
      // ... 原有逻辑 ...

      MultiStreamCaptureState state;
      state.primary_stream = primary_stream;
    
      // 【新增】预注册全部三流，确保 insert_cross_op_barrier 能覆盖所有流
      state.get_or_register(primary_stream);  // COMP_1
      state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2)));
      state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3)));
    
      cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);
    
      for (size_t i = 0; i < nodes.size(); ++i) {
          const auto& node = nodes[i];
    
          if (i > 0) {
              insert_cross_op_barrier(nodes[i-1], node, state, ctx);
    
              // 【新增】显式同步：当前算子的代表流等待上一个算子的输出流
              // 这是 insert_cross_op_barrier 语义缺陷的根本修复
              if (state.output_stream_idx >= 0 &&
                  node.kind == GraphNode::Kind::COMPUTE) {
                  StreamKind target_sk = get_op_default_stream(node.compute_op);
                  cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
                  int target_idx = state.find_stream_index(target_s);
                  int out_idx = state.output_stream_idx;
                  // 不同流才需要 wait；同流自然有序
                  if (target_idx >= 0 && target_idx != out_idx) {
                      cudaStreamWaitEvent(target_s,
                                         state.streams[out_idx].last_done_event, 0);
                  }
              }
          }
    
          // 执行算子
          if (node.kind == GraphNode::Kind::COMPUTE) {
              auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
              if (entry.launch_cuda) {
                  entry.launch_cuda(node, mp, ctx, state);
              } else {
                  // 【修改】默认 replay 也使用惯用流
                  replay_compute_node_default(node, mp, ctx, state);
              }
          } else {
              // ... range op ...
          }
      }
    
      finalize_cross_stream_barrier(state);
      // ... 原有逻辑 ...
  }
  2.4 框架层：insert_cross_op_barrier 修复
  // src/graph/capture_multi_stream.cpp
  void insert_cross_op_barrier(...) {
      int out_idx = state.output_stream_idx;
      if (out_idx < 0) return;
      for (int i = 0; i < state.num_active; ++i) {
          if (i == out_idx) continue;
          // 【新增】跳过没有 pending work 的流，避免对未记录事件调用 WaitEvent
          if (!state.streams[i].has_pending_work) continue;
          cudaStreamWaitEvent(state.streams[i].stream,
                             state.streams[out_idx].last_done_event, 0);
      }
  }
  2.5 框架层：默认 replay 函数修正
  // capture_cuda.cpp
  static void replay_compute_node_default(
      const GraphNode& node,
      const MemoryPlan& /*mp*/,
      const DeviceContext& ctx,
      MultiStreamCaptureState& state)
  {
      // 【修改】使用惯用流，而非硬编码 COMP_1
      StreamKind sk = get_op_default_stream(node.compute_op);
      cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
      int i = state.get_or_register(s);
      state.output_stream_idx = i;
      state.streams[i].has_pending_work = true;
      cudaEventRecord(state.streams[i].last_done_event, s);
  }

  // range 节点同理
  2.6 算子层：简单算子改造（GAP 示例）
  // gap_op.cpp: launch_gap_fwd_cuda_impl
  static void launch_gap_fwd_cuda_impl(...) {
      // 【修改】COMP_1 → get_op_default_stream
      StreamKind sk = get_op_default_stream(node.compute_op);  // GAP_FWD → COMP_2
      cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
      int si = state.get_or_register(s);
      state.output_stream_idx = si;
      state.streams[si].has_pending_work = true;

      // 【修改】handle 和 workspace 也使用同一流
      cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
      ctx.ensure_workspace_grow(sk, cache.workspace_size);
      // ...
      cache.graph->execute(handle, vp, ctx.workspace(sk));
    
      cudaEventRecord(state.streams[si].last_done_event, s);
  }
  简单算子清单（所有只改 1 处 StreamKind::COMP_1 → get_op_default_stream(...)）：
  • gap_op.cpp: GAP FWD/BWD × 4 函数
  • relu_op.cpp: RELU FWD/BWD/INF × 6 函数
  • tanh_op.cpp: TANH FWD/BWD × 4 函数
  • fc_op.cpp: FC FWD × 2 函数
  2.7 算子层：FC BWD 三流化（核心，P0）
  这是整个方案中唯一需要内部多流逻辑的算子。设计如下：
  COMP_1: dW = gemm(X, dY^T) ──[ev_dw_done]────┐
  COMP_2: dB = reduce_sum(dY)  (与 dW 并行)      │
  COMP_3: wait(ev_dw_done) → dX = gemm(W, dY)   │
          └── output_stream_idx (下游依赖 dX)
  // fc_op.cpp: launch_fc_amp_bwd_cuda（launch_fc_bwd_cuda 同理）
  static void launch_fc_amp_bwd_cuda(...) {
      // ===== 参数解析（不变）=====
      const auto* p = std::get_if<FCParams>(&node.params.data);
      bool has_bias = p->bias;
      int x_idx = static_cast<int>(node.input_ids.size()) - 1;
      // ... 解析 dy, w, x, dx, dw, db, batch, in_features, out_features ...

      // ===== 获取三流 =====
      cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
      cudaStream_t s_db = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
      cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
    
      int i_dw = state.get_or_register(s_dw);
      int i_db = state.get_or_register(s_db);
      int i_dx = state.get_or_register(s_dx);
    
      cublasHandle_t h_dw = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_1));
      cublasHandle_t h_dx = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_3));
      // dB 使用自定义 kernel，不需要 cublas handle
    
      // =====【算子自治】让 dW 和 dB 等待上游输出（如果需要）=====
      // 上游算子的输出可能是 dY（外部输入）或中间层的 dX
      int out_idx = state.output_stream_idx;
      if (out_idx >= 0) {
          cudaStream_t prev_s = state.streams[out_idx].stream;
          if (prev_s != s_dw) {
              cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
          }
          if (prev_s != s_db) {
              cudaStreamWaitEvent(s_db, state.streams[out_idx].last_done_event, 0);
          }
      }
    
      // ===== COMP_2: dB (bias grad) =====
      // dB 只读 dY，与 dW 无冲突，可以和 dW 真实并行
      if (has_bias) {
          launch_fc_bwd_db_amp_kernel(dy, db, batch, out_features, dy_ns, s_db);
      }
      state.streams[i_db].has_pending_work = true;
      cudaEventRecord(state.streams[i_db].last_done_event, s_db);
    
      // ===== COMP_1: dW = X @ dY^T =====
      // dW 必须先于 dX，因为它需要读取 X（dX 会覆盖 X）
      cublasGemmEx(h_dw, CUBLAS_OP_N, CUBLAS_OP_T,
                   in_features, out_features, batch,
                   &alpha, x, CUDA_R_16F, x_ns,
                   dy, CUDA_R_16F, dy_ns,
                   &beta, dw, CUDA_R_16F, dw_ns,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      state.streams[i_dw].has_pending_work = true;
    
      // 记录 dW 完成事件，供 dX 等待
      cudaEvent_t ev_dw_done = state.alloc_temp_event();
      cudaEventRecord(ev_dw_done, s_dw);
      cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    
      // ===== COMP_3: dX = W @ dY =====
      // 【关键】必须等 dW 完成，因为 dX 会 in-place 覆盖 X
      cudaStreamWaitEvent(s_dx, ev_dw_done, 0);
      cublasGemmEx(h_dx, CUBLAS_OP_N, CUBLAS_OP_N,
                   in_features, batch, out_features,
                   &alpha, w, CUDA_R_16F, w_ns,
                   dy, CUDA_R_16F, dy_ns,
                   &beta, dx, CUDA_R_16F, dx_ns,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      state.streams[i_dx].has_pending_work = true;
      cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    
      // 下游 GAP_BWD 依赖的是 dX，所以 output_stream_idx 设为 dX 的流
      state.output_stream_idx = i_dx;
  }
  删除原有的 self-wait barrier（fc_op.cpp L343-347 和 L534-537）：
  // 删除以下无意义代码：
  // cudaEvent_t dw_done = state.alloc_temp_event();
  // cudaEventRecord(dw_done, s);
  // cudaStreamWaitEvent(s, dw_done, 0);  // 同流自等待，无实际效果
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  三、完整修改清单
   #    文件                                 改动                                            行数   风险
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1    src/backend/op_stream_policy.h       新增                                            ~15    无
   2    src/backend/op_stream_policy.cpp     新增                                            ~45    无
   3    src/backend/CMakeLists.txt           添加新文件                                      +1     无
   4    src/graph/capture_cuda.cpp           预注册三流 + pre-node wait + 默认 replay 修正   ~25    中
   5    src/graph/capture_multi_stream.cpp   insert_cross_op_barrier 加 has_pending_work     +1     低
   6    src/backend/ops/dtensor/gap_op.cpp   4 函数改 get_op_default_stream                  ~16    低
   7    src/backend/ops/dtensor/gap_op.cu    workspace/handle 改用同一流                     ~8     低
   8    src/backend/ops/dtensor/fc_op.cpp    FC FWD 改 get_op_default_stream                 ~4     低
   9    src/backend/ops/dtensor/fc_op.cpp    FC BWD 三流重构 + 删 self-wait                  ~50    高
   10   tests/perf/test_gap_fc_perf.cpp      验证测试                                        不变   -
  总计：约 165 行改动，分布在 5 个文件中，零 IR 变更。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  四、数据竞争安全性证明
  4.1 FWD 图：GAP(COMP_2) → FC(COMP_1)
  1. GAP_FWD@COMP_2: 读 gap_x → 写 gap_y
     → cudaEventRecord(last_done@COMP_2)
     → output_stream_idx = COMP_2

  2. insert_cross_op_barrier: COMP_1 无 pending → skip

  3. 【新增】pre-node wait: target=COMP_1, out_idx=COMP_2, 不同流
     → cudaStreamWaitEvent(COMP_1, last_done@COMP_2) ✅

  4. FC_FWD@COMP_1: 读 gap_y → 写 fc_y
     // gap_y 已就绪，无竞争
    4.2 BWD 图：FC_BWD(多流) → GAP_BWD(COMP_2)
  1. FC_BWD:
     COMP_1: dW = gemm(X, dY^T) → ev_dw_done
     COMP_2: dB = reduce(dY)
     COMP_3: wait(ev_dw_done) → dX = gemm(W, dY)
     → output_stream_idx = COMP_3

  2. insert_cross_op_barrier:
     COMP_1 pending → wait(COMP_3_done) ✅
     COMP_2 pending → wait(COMP_3_done) ✅

  3. 【新增】pre-node wait: target=COMP_2(GAP), out_idx=COMP_3
     → cudaStreamWaitEvent(COMP_2, last_done@COMP_3) ✅

  4. GAP_BWD@COMP_2: 读 d_fc_dx → 写 d_gap_dx
     // d_fc_dx 已就绪，无竞争
    4.3 FC BWD 内部 in-place 安全
      操作   读取    写入               约束
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      dB     dy      db                 与 dW 无共享缓冲区，可并行
      dW     x, dy   dw                 必须在 dX 之前（读 x）
      dX     w, dy   dx (即 x 缓冲区)   必须等 dW 完成
    保障：cudaStreamWaitEvent(s_dx, ev_dw_done, 0) 确保 dX 在 dW 完成后启动，此时 X 已被 dW 读取完毕，dX 覆盖 X 是安全的。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    五、预期收益与验证计划
    5.1 性能预测（基于 A100 基线）
      场景         当前 (μs)   预计 (μs)   原理
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      GAP+FC FWD   132.6       ~125        GAP(COMP_2) 与 FC(COMP_1) 调度延迟隐藏
      GAP+FC BWD   480.6       ~420        dW(COMP_1) ∥ dB(COMP_2)，dX(COMP_3) 提前就绪
      总计         613.2       ~545        ~11% 提升
    5.2 验证步骤
  1. 正确性：test_gap_correction --amp 必须 PASS（数值精度不变）
  2. 单算子回归：test_gap_perf / test_fc_perf 独立运行 PASS
  3. 联合性能：test_gap_fc_perf --amp A100 实测，目标 BWD < 430μs
  4. CBR 模型端到端：ResNet-50 完整训练 step 验证无死锁
    5.3 风险与缓解
      风险                              概率   缓解
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      cuBLAS 多流 workspace 冲突        低     DeviceContext 已有 per-stream workspace
      insert_cross_op_barrier 漏 wait   低     预注册 + has_pending_work + pre-node wait 三重保障
      FC BWD dW/dB 并发带宽饱和         中     dB 是 reduce 操作，带宽占用极小
      更复杂图（Conv→FC）依赖错误       中     FC BWD 内部自治 wait 上游事件
      默认 replay 路径行为变化          低     改为惯用流后语义更正确
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    六、与 SSS2.md 方案的本质改进
      维度           SSS2.md 小伙伴方案                              本方案
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      流映射         小伙伴 K 误把 GAP 分到 COMP_1                   GAP 作为池化正确分到 COMP_2
      跨流依赖机制   小伙伴 D 的"算子自治"导致每算子重复 wait 代码   框架统一处理简单算子，仅复合算子自治
      注册表改动     小伙伴 S 提议新增 replay_cuda_capture 指针      不新增函数指针，完全复用 launch_cuda
      barrier 修复   小伙伴 D 发现但未整合到框架层                   框架 pre-node wait + has_pending_work 双重修复
      默认 replay    SSS2.md 未提及                                  明确修正 fallback 路径，消除静默陷阱
      代表流语义     模糊                                            明确定义 get_op_default_stream 返回"代表流"（下游依赖
                                                                  的流）
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    结论：本方案在 SSS2.md 基础上，修正了流映射错误，避免了过度设计的注册表改动，通过"框架集中控制 + FC BWD 内部自治"的混
    合策略，以 ~165 行代码、零 IR 变更、5 个文件的代价，实现三流架构升级。核心收益来自 FC BWD 的 dW∥dB 真实并发 + dX 提前
    就绪的调度延迟隐藏，预期 GAP+FC 组合性能从 613μs 降至 ~545μs。



# 【小伙伴D】

## 一、三方关键分歧辨析
### 分歧 1：GAP 的流
来源 主张 与用户规则一致性 用户规则 BN 和 池化 用 COMP_2 — K 的 get_op_default_stream GAP → COMP_1（与 Conv/FC 同组） ❌ 违背 S Phase 2 GAP → COMP_1, FC → COMP_2 ❌ 违背 D 修正 GAP → COMP_2, FC → COMP_1 ✅ 遵从

裁决 ：GAP 是池化算子，必须用 COMP_2。K 把 GAP 归入 COMP_1 是错误的——如果 GAP 和 FC 同流， insert_cross_op_barrier 退化回单流，零加速。S 的 Phase 2 把 GAP 放 COMP_1、FC 放 COMP_2 是同样的问题（方向反了）。

### 分歧 2：预注册三流的必要性
来源 主张 分析 K capture 循环无需修改 依赖算子内自等，忽略了算子间跨流依赖 D 必须预注册 3 流 + has_pending_work 检查 正确识别了框架层的 gap S 修改 capture 分发逻辑，读 prefer_stream 方向对但方案复杂（需新增字段）

K 的"验证流拓扑"中假设 insert_cross_op_barrier 会让 COMP_2 等 COMP_3——但这只有 COMP_2 已被注册时才生效。如果 COMP_2 没有 pending work， has_pending_work=continue 会跳过。 预注册 + 算子间前向 wait 是必须的 。

### 分歧 3：实现封装方式
来源 主张 评价 S CuLaunchEntry.prefer_stream + capture 读取 + 分发 需改注册表结构 D replay_cuda_capture 函数指针 过度设计，每算子两套实现 K 只改算子内 launch_cuda 函数 ✅ 最轻量

裁决 ：走 K 的路径——不改函数签名，不改注册表结构，算子在自己函数内多流化。capture 循环用少量集中逻辑辅助 wait。

### 分歧 4：算子间前向等待的位置
来源 主张 D 初版 每个算子 launch_cuda 开头自等 D 终版 capture 循环集中 pre_wait K 隐式依赖 insert_cross_op_barrier S 框架分发时处理

裁决 ：走 capture 循环集中 pre_wait ——每个算子不用重复 boilerplate，且逻辑集中在 "capture 引擎负责跨流编排" 这个框架层职责内。

## 二、最终方案（7 文件，约 130 行）
### 2.1 新增：op_stream_policy.h（全局流映射函数，采纳 K，但修正 GAP→COMP_2）
```
// 静态惯用流：无需运行时代价
inline StreamKind get_op_default_stream(ComputeOp op) {
    switch (op) {
        // ===== Conv / FC → COMP_1 =====
        case ComputeOp::FC_FP32_FWD: case ComputeOp::FC_AMP_FWD:
        case ComputeOp::FC_FP32_BWD: case ComputeOp::FC_AMP_BWD:
        case ComputeOp::CONV_FP32_FWD: // ...
            return StreamKind::COMP_1;

        // ===== Pooling / BN → COMP_2 =====
        case ComputeOp::GAP_FP32_FWD: case ComputeOp::GAP_AMP_FWD:
        case ComputeOp::GAP_FP32_BWD: case ComputeOp::GAP_AMP_BWD:
        case ComputeOp::MAXPOOL_FWD: case ComputeOp::MAXPOOL_BWD:
        case ComputeOp::BN1D_FP32_FWD: // ...
            return StreamKind::COMP_2;

        // ===== Activation → COMP_3 =====
        case ComputeOp::RELU_FP32_FWD: case ComputeOp::RELU_AMP_FWD:
        case ComputeOp::TANH_FP32_FWD: case ComputeOp::TANH_AMP_FWD:
        // ...
            return StreamKind::COMP_3;

        default: return StreamKind::COMP_1;
    }
}
```
### 2.2 改动 capture_cuda.cpp（~12 行）
a) 预注册 3 流 （采纳 D，K 未提但必须）：

```
// 原：只注册 primary
state.get_or_register(primary_stream);
// 新增：
state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2)));
state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3)));
```
b) 算子间前向 wait （D 的设计，弥补 insert_cross_op_barrier 的语义缺陷）：

```
for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];

    // 如果这个节点是 COMPUTE，让它的惯用流等上一个算子的输出
    if (i > 0 && node.kind == GraphNode::Kind::COMPUTE
        && state.output_stream_idx >= 0) {
        StreamKind my_sk = get_op_default_stream(node.compute_op);
        cudaStream_t my_s = static_cast<cudaStream_t>(ctx.stream(my_sk));
        cudaStream_t prev_s = state.streams[state.output_stream_idx].stream;
        if (my_s != prev_s) {
            cudaStreamWaitEvent(my_s,
                state.streams[state.output_stream_idx].last_done_event, 0);
        }
    }

    if (i > 0) {
        insert_cross_op_barrier(nodes[i-1], node, state, ctx);
    }

    if (node.kind == GraphNode::Kind::COMPUTE) {
        auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
        if (entry.launch_cuda) {
            entry.launch_cuda(node, mp, ctx, state);
        }
    }
}
```
顺序的关键性 ：前向 wait 必须在 insert_cross_op_barrier 之前 ，因为 insert_cross_op_barrier 只在已注册且 has_pending_work=true 的流上插 wait。GAP+FC FWD 中，GAP 执行完时只有 COMP_2 有 pending，此时 insert_cross_op_barrier 不会给 COMP_1 插 wait。前向 wait 补上这个缺口。

为什么 insert_cross_op_barrier 仍然需要 ：BWD 场景——FC BWD 结束时 COMP_1/2/3 全有 pending。 insert_cross_op_barrier 收敛 COMP_1 和 COMP_2 到 COMP_3 的输出事件，确保 CUDA Graph 拓扑完整性。

### 2.3 改动 capture_multi_stream.cpp（+1 行，D + K 共识）
```
// insert_cross_op_barrier 加 has_pending_work 检查
for (int i = 0; i < state.num_active; ++i) {
    if (i == out_idx) continue;
    if (!state.streams[i].has_pending_work) continue;  // ← 新增
    cudaStreamWaitEvent(state.streams[i].stream, ...);
}
```
### 2.4 改动 GAP 算子 4 个 launch 函数（~4×4=16 行，采纳 D + K 共识部分，修正 K 的 COMP_1→COMP_2）
FWD 两函数 + BWD 两函数 ： COMP_1 → COMP_2 ，workspace、cuDNN handle 同步改。

```
// 以 launch_gap_amp_fwd_cuda 为例：
StreamKind sk = get_op_default_stream(node.compute_op);  // → COMP_2
cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
auto ws = ctx.workspace(sk);       // per-stream workspace
auto handle = ctx.cudnn_handle(sk); // per-stream cuDNN handle

int si = state.get_or_register(s);
// ... cuDNN FE graph execute ...
cudaEventRecord(state.streams[si].last_done_event, s);
state.streams[si].has_pending_work = true;
state.output_stream_idx = si;
```
### 2.5 改动 FC FWD 2 个 launch 函数（保持 COMP_1，仅取 get_op_default_stream，~4 行）
纯机械替换 StreamKind::COMP_1 → get_op_default_stream(node.compute_op) ，实际值不变。为未来统一风格。

### 2.6 改动 FC BWD 2 个 launch 函数（~2×50=100 行，采纳 K 核心逻辑，修正 S 的流分配）
三流分解 （K 的设计，D 验证的数据竞争分析）：

```
static void launch_fc_amp_bwd_cuda(node, mp, ctx, state) {
    // 三流获取
    cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);  // dW
    cudaStream_t s_db = ctx.stream(StreamKind::COMP_2);  // dB
    cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);  // dX

    int i_dw = state.get_or_register(s_dw);
    int i_db = state.get_or_register(s_db);
    int i_dx = state.get_or_register(s_dx);

    cublasHandle_t h_dw = ctx.cublas_handle(StreamKind::COMP_1);
    cublasHandle_t h_db = ctx.cublas_handle(StreamKind::COMP_2);
    cublasHandle_t h_dx = ctx.cublas_handle(StreamKind::COMP_3);

    // COMP_2: dB（极轻量，与 dW 无数据竞争，可并发）
    if (has_bias) launch_fc_bwd_db_kernel(..., s_db);

    // COMP_1: dW = X @ dY^T（必须先于 dX，因 dX 覆写 X）
    cublasGemmEx(..., s_dw);  // dW

    // dW 完成事件
    cudaEvent_t ev_dw = state.alloc_temp_event();
    cudaEventRecord(ev_dw, s_dw);

    // COMP_3: dX = W @ dY（等 dW 完成）
    cudaStreamWaitEvent(s_dx, ev_dw, 0);
    cublasGemmEx(..., s_dx);  // dX

    // 记录三流事件
    record_and_set(s_dw, i_dw, state);
    record_and_set(s_db, i_db, state);
    record_and_set(s_dx, i_dx, state);

    // 下游等 dX（COMP_3），不是 dW 或 dB
    state.output_stream_idx = i_dx;
}
```
与 K 版本的区别：

- K 把 output_stream_idx = i_dx 设为 COMP_3 → ✅ 正确
- K 只用 get_op_default_stream 但 GAP 映射错了（COMP_1 应改为 COMP_2）→ 本方案修正
- K 没有提 has_pending_work 检查 → 本方案加上（D + K 共识）
### 2.7 删除 fc_op.cpp 的同流 self-wait barrier（采纳 K，-3 行）
```
// 删除这三行：
// cudaEvent_t dw_done = state.alloc_temp_event();
// cudaEventRecord(dw_done, s);
// cudaStreamWaitEvent(s, dw_done, 0);
```
三流分解后用跨流 cudaStreamWaitEvent(s_dx, ev_dw, 0) 替代，正确表达跨流依赖。

## 三、TOP 验证：逐节点跟踪
### FWD GAP → FC
```
Primary = COMP_1（capture 流）
预注册: COMP_1(0)、COMP_2(1)、COMP_3(2)

节点 1 — GAP_FWD:
  sk = get_op_default_stream(GAP_FWD) = COMP_2
  s = ctx.stream(COMP_2)
  si = get_or_register(s) = 1（已预注册，返回 1）
  ... cuDNN FE execute on s ...
  cudaEventRecord(state.streams[1].last_done_event, COMP_2)
  state.streams[1].has_pending_work = true
  state.output_stream_idx = 1

  pending: COMP_2=✓, COMP_1=✗, COMP_3=✗

[pre_wait for FC_FWD]:
  sk = COMP_1, my_s = COMP_1
  prev_s = COMP_2（output_stream_idx=1）
  COMP_1 ≠ COMP_2 → cudaStreamWaitEvent(COMP_1, COMP_2_event) ✅

[insert_cross_op_barrier]:
  out_idx=1(COMP_2)
  COMP_1: no pending → skip ✅
  COMP_2: self → skip
  COMP_3: no pending → skip ✅

节点 2 — FC_FWD:
  sk = COMP_1, s = ctx.stream(COMP_1)
  si = 0
  COMP_1 已在 wait COMP_2 → FC 不会在 GAP 完成前启动 ✅
  ... cuDNN FE execute ...
  cudaEventRecord(COMP_1 event)
  has_pending_work = true
  output_stream_idx = 0

[finalize]: COMP_2 → join COMP_1, COMP_3(idle) skip → 图收敛 ✅
```
### BWD FC → GAP
```
节点 1 — FC_BWD（三流内部）:
  COMP_1: dW GEMM → ev_dw
  COMP_2: dB kernel
  COMP_3: wait(ev_dw) → dX GEMM → event
  output_stream_idx = COMP_3(index=2)

  pending: COMP_1=✓, COMP_2=✓, COMP_3=✓

[pre_wait for GAP_BWD]:
  sk = COMP_2, my_s = COMP_2
  prev_s = COMP_3（output_stream_idx=2）
  COMP_2 ≠ COMP_3 → cudaStreamWaitEvent(COMP_2, COMP_3_event) ✅

[insert_cross_op_barrier]:
  out_idx=2(COMP_3)
  COMP_1: pending → cudaWait(COMP_3_event) ✅
  COMP_2: self → skip
  COMP_3: self → skip

节点 2 — GAP_BWD:
  COMP_2 已在 wait COMP_3 → GAP_BWD 不会在 FC dX 完成前启动 ✅
  ... kernel ...
  output_stream_idx = COMP_2(1)

[finalize]: COMP_1/COMP_3 → join COMP_1(primary) → 图收敛 ✅
```
FWD 和 BWD 全部通过。无数据竞争，无悬挂流。

## 四、总结
来源采纳 内容 1 K get_op_default_stream() 全局流映射函数（修正 GAP→COMP_2） 2 D capture 循环预注册 3 流 + 前向 wait 3 D+K insert_cross_op_barrier 加 has_pending_work 检查 4 K FC BWD 三流分解核心逻辑 5 D GAP → COMP_2 (遵从用户规则) 6 K 删除 FC BWD 同流 self-wait 7 D+K output_stream_idx = COMP_3(dX) ，下游等 dX 而非 dW

否决 来源 理由 prefer_stream 字段 S 增加注册表结构，不如静态映射函数简洁 replay_cuda_capture 函数指针 D 初版 过度设计，改现有 launch 即可 GAP→COMP_1 K 违背用户规则 "池化用 COMP_2" 算子内自等 D 初版 各算子重复 boilerplate，集中在 capture 循环更好

总规模 ~130 行，7 文件， ComputationGraph IR 零变更。

