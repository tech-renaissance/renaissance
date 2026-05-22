# 【今日话题：讨论FC层的设计及workspace的申请】



# 【导言】

FC层必定申请bias张量，但不一定使用
FC的bias张量必定是FP32，即使在AMP模式下也是
cuDNN似乎要求矩阵乘法接受的输入不是2D，而是3D
如果上一个张量的C通道有padding，那么框架会自动在FC层前面插入元素重排的flatten层，所以传递给FC层的张量必定是在C通道连续的

workspace比较麻烦，似乎FC层和tanh和卷积层都要用到
它的关键点就是：1、每个流一个自己的workspace；2、一次申请，永久复用，绝对避免算子内部申请释放；3、只有warmup时才知道大小，所以无法分配到显存池内（ArenaKeeper）
warmup时会申请临时的workspace，warmup结束后把这个内存注销，但是报告需要多大的workspace
DeepLearningTask和SimpleTask类都会维护一个workspace的最大字节数
compile的时候，具体来说是预热后捕获前，会申请好这个最大workspace。Task或DeviceContext长期持有，一次申请永久复用，直到程序结束才释放，绝对不在多个epoch间反复申请
还有一种思路其实就是，warmup时先一次性申请一个足够大的常数值（比如512MB），然后warmup过程中遇到需要更大的workspace的情况就把旧的释放、申请新的——这样就确保得到的是最大值了，而且warmup结束后得到的是准确的指针值，捕获和运行的时候都知道指针和字节数

算法的实现是旧版的fc_op_legacy.cu和fc_op_legacy.cpp，但它们的对齐有可能不对，在stride上也没有使用本框架新架构的API

大家说说自己的方案。



# 【小伙伴S】

  FC 算子改进方案

  问题分析

  1. workspace 管理问题: 当前 conv_op.cpp 在每次执行时都临时 cudaMalloc/cudaFree
    workspace，违反了"一次申请，永久复用"原则
  2. FC 算子未使用 cuDNN Frontend: 当前 fc_op.cpp 使用的是自定义 kernel，没有利用 cuDNN 的优化实现
  3. bias 处理: bias 必须是 FP32，即使在 AMP 模式下
  4. 3D 张量要求: cuDNN 要求矩阵乘法输入是 3D 张量

  方案设计

  1. 统一 Workspace 管理

  参照 relu_op.cpp 的 per-handle 缓存模式，为需要 workspace 的算子（CONV、FC、TANH等）实现统一的 workspace 管理：

  // 在 warmup 阶段：
  // 1. 第一次运行时申请初始 workspace（如 512MB）
  // 2. 如果需要更大的 workspace，释放旧的，申请新的
  // 3. warmup 结束后得到最终大小和指针
  // 4. 在 compile() 阶段调用 ctx.ensure_workspace(kind, max_size)
  // 5. 后续执行直接使用 ctx.workspace(kind)，无需再申请/释放

  2. FC 算子 cuDNN Frontend 迁移

  参照 relu_op.cpp 和 tanh_op.cpp 的模式：

  // 结构
  struct FcFwdCache {
      cudnnHandle_t handle = nullptr;
      int64_t batch = -1, in_features = -1, out_features = -1;
      bool has_bias = false;
      std::shared_ptr<feg::Graph> graph;
      std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
  };

  // per-handle 缓存
  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcFwdCache>> s_fc_fwd_caches;

  3. Bias 处理策略

  - AMP 模式下，bias 始终使用 FP32 精度
  - 输入张量（X, W）使用 FP16，但 bias 使用 FP32
  - 在 pointwise add 操作时进行自动类型转换

  4. 3D 张量转换

  - 输入自动从 NHWC 4D 展平为 NC 2D
  - 使用 cuDNN 的 Matmul 操作而不是 GEMM
  - 维度映射：[N, H, W, C] → [N, H*W*C] → [N, 1, H*W*C]

  5. 实现优先级

  1. 高优先级: 修复 conv_op.cpp 的 workspace 管理问题（临时申请/释放改为复用）
  2. 中优先级: 将 fc_op.cpp 迁移到 cuDNN Frontend API（参照 tanh_op.cpp）
  3. 低优先级: 优化 alignment 计算，确保与新架构一致

  具体实现步骤

  Step 1: 修复 workspace 管理
  - 在 warmup 时检测最大 workspace 大小
  - 在 compile() 后调用 ctx.ensure_workspace()
  - 在执行时直接使用 ctx.workspace()，无需 cudaMalloc/cudaFree

  Step 2: 重构 FC 算子
  - 使用 cuDNN Frontend 的 Matmul_attributes
  - 支持 FP32 和 AMP 两种模式
  - 正确处理 bias 的 FP32 精度要求

  Step 3: 测试验证
  - 使用不同输入尺寸测试 workspace 扩展
  - 验证 bias 在 AMP 模式下的 FP32 精度
  - 确保 per-stream workspace 隔离

  这个方案可以确保：
  - ✅ Workspace 一次申请，永久复用
  - ✅ 支持 CUDA Graph 捕获
  - ✅ 多 GPU 环境下的正确性
  - ✅ 与现有 relu_op.cpp 和 tanh_op.cpp 架构一致

# 【小伙伴K】

  让我提出一个全面的方案。
• 现在我已掌握全部上下文。以下是针对 FC 算子及 workspace 管理的完整方案。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一、当前架构现状梳理
  1.1 DeviceContext 已具备的 workspace 基础设施
  device_context.h/cpp 中已经实现了完善的 workspace 生命周期管理：
  // 查询
  void*    workspace(StreamKind kind) const;      // 获取指针
  size_t   workspace_size(StreamKind kind) const; // 获取当前大小

  // 分配
  void pre_allocate_workspace(StreamKind kind, size_t size);  // 预分配（compile 阶段用）
  void ensure_workspace(StreamKind kind, size_t req_size);    // 懒分配/扩容（运行时用）

  // 释放
  void free_workspaces();  // 析构时统一释放
  特点：每个 DeviceContext 管理 5 个 stream 的 workspace，各自独立。
  1.2 现有算子的 workspace 使用模式
   算子                             模式                           问题
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   RELU / TANH / IDENTITY           execute(handle, vp, nullptr)   不需要 workspace，无问题
   Conv（op_registry.cpp warmup）   临时 cudaMalloc/cudaFree       ❌ capture 期间非法，且每次 epoch 重复申请
   Conv（conv_op.cpp launch）       临时 cudaMalloc/cudaFree       ❌ 同上
   FC（legacy）                     未实现 workspace 传递          ❌ Matmul 可能需要 workspace
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  二、Workspace 管理方案（推荐：方案 A + 兜底 B）
  2.1 核心设计原则（来自 FFF.md）
  1. 每个流一个 workspace
  2. 一次申请，永久复用，算子内部绝不 cudaMalloc/Free
  3. warmup 探测大小，compile 阶段预分配
  4. 所有算子统一从 ctx.workspace() 取指针传给 graph->execute()
    2.2 方案：Task 维护全局最大 workspace + DeviceContext 预分配
    ┌─────────────────────────────────────────────────────────────────┐
    │  Phase A: Warmup（逐 rank 串行）                                  │
    │  ─────────────────────────────                                   │
    │  对每个需要 workspace 的算子（Conv/FC/Matmul）：                    │
    │    1. 临时 cudaMalloc 执行 warmup（唯一允许临时分配的地方）          │
    │    2. ws = graph->get_workspace_size()                           │
    │    3. task_max_workspace = max(task_max_workspace, ws)           │
    │    4. cudaFree(temp_ws) ← 临时内存立即释放                         │
    │                                                                  │
    │  Phase B: Compile（预热后、捕获前）                                │
    │  ───────────────────────────────                                 │
    │  对每个 rank 的每个 stream：                                       │
    │    ctx.pre_allocate_workspace(stream_kind, task_max_workspace)   │
    │                                                                  │
    │  Phase C: Capture & Run（算子 launch）                            │
    │  ─────────────────────────────────                               │
    │  算子内部：                                                       │
    │    void* ws = ctx.workspace(StreamKind::COMP_1);                 │
    │    graph->execute(handle, vp, ws);  ← 永久复用，绝不申请释放        │
    └─────────────────────────────────────────────────────────────────┘
    2.3 代码层面的改动点
    改动 1：TaskBase / SimpleTask 新增 max_workspace_size_
    // task_base.h 或 simple_task.h
    class TaskBase {
    protected:
      size_t max_workspace_size_ = 0;  // 所有算子中最大的 workspace 需求

      void update_max_workspace(size_t ws) {
          if (ws > max_workspace_size_) max_workspace_size_ = ws;
      }
    };
    改动 2：warmup 阶段收集 workspace 大小
    目前 op_registry.cpp 中 Conv 的 warmup 已经调用了 get_workspace_size()，但只是打印日志。需要把值上报给 Task：
    // op_registry.cpp 中 Conv warmup 部分
    int64_t ws_bytes = graph->get_workspace_size();
    // ... 临时申请执行 warmup ...
    if (temp_ws) cudaFree(temp_ws);

  // 【新增】上报给 Task
  // 方式：通过 callback 或全局收集器
  // 例如：ctx.report_workspace_needed(ws_bytes);
  // 或：warmup 返回 ws_bytes，由调用方收集
  对于 RELU/FC 等通过 launch_cuda 预热的算子，如果它们内部使用了 cuDNN FE graph，也需要在 cache 重建时记录 get_workspace
  _size()：
  // 在算子 launch 函数中（如 launch_fc_amp_fwd_cuda）
  if (rebuild) {
      // ... build graph ...
      size_t ws = cache.graph->get_workspace_size();
      // 上报给 Task：ctx.report_workspace_needed(ws);
  }
  改动 3：compile 阶段预分配 workspace
  在 compile_capture_simple() 的 "warmup done" 和 "starting capture phase" 之间插入：
  // task_base.cpp 中 compile 流程
  // Phase B2: warmup 完成后
  for (int rank = 0; rank < num_gpus; ++rank) {
      auto& ctx = *device_contexts_[rank];
      for (int k = 0; k < 5; ++k) {
          StreamKind sk = static_cast<StreamKind>(k);
          ctx.pre_allocate_workspace(sk, max_workspace_size_);
      }
  }

  // Phase B3: capture（此时 workspace 已就绪）
  改动 4：所有算子 execute() 传入 ctx.workspace()
  Conv 算子（conv_op.cpp）：
  // 移除：
  // int64_t ws_bytes = graph->get_workspace_size();
  // void* temp_ws = nullptr;
  // if (ws_bytes > 0) cudaMalloc(&temp_ws, ws_bytes);
  // ...
  // if (temp_ws) cudaFree(temp_ws);

  // 改为：
  void* ws = ctx.workspace(StreamKind::COMP_1);
  TR_CUDNN_FE_CHECK(graph->execute(handle, vp, ws), "CONV execute");
  FC 算子（新版实现）：
  void* ws = ctx.workspace(StreamKind::COMP_1);
  TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "FC execute");
  TANH/RELU（不需要 workspace 的算子）：
  // 保持 nullptr 即可，不需要改动
  TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, nullptr), "RELU execute");
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  三、FC 算子新版实现方案
  3.1 设计约束（来自 FFF.md）
   约束                        说明
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Bias 必定申请，不一定使用   内存规划中预留 bias 张量，通过 FCParams::bias 标志控制
   Bias 永远是 FP32            即使 AMP 模式下，bias 数据类型固定为 FLOAT
   输入必定是 C 通道连续的     框架会自动在 FC 前插入 flatten 层，无需处理 padding
   cuDNN Matmul 要求 3D        输入从 {batch, in_features} 转换为 {batch, 1, in_features}
  3.2 算子注册表
  enum class ComputeOp {
      // ...
      FC_FP32_FWD,
      FC_FP32_BWD,
      FC_AMP_FWD,
      FC_AMP_BWD,
  };
  3.3 FC AMP FWD 的 cuDNN FE Graph 构建
  static std::shared_ptr<feg::Graph> build_fc_amp_fwd_graph(
      cudnnHandle_t handle,
      int64_t batch, int64_t in_features, int64_t out_features,
      bool has_bias,
      std::shared_ptr<feg::Tensor_attributes>& out_x,
      std::shared_ptr<feg::Tensor_attributes>& out_w,
      std::shared_ptr<feg::Tensor_attributes>& out_b,   // nullable
      std::shared_ptr<feg::Tensor_attributes>& out_y)
  {
      auto g = std::make_shared<feg::Graph>();

      // HALF I/O, FLOAT compute
      g->set_io_data_type(fe::DataType_t::HALF)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
    
      // cuDNN Matmul 要求 3D: {batch, 1, in_features}
      std::vector<int64_t> x_dim = {batch, 1, in_features};
      std::vector<int64_t> x_stride = {in_features, in_features, 1};  // 连续
    
      out_x = g->tensor(feg::Tensor_attributes()
          .set_name("x")
          .set_dim(x_dim).set_stride(x_stride)
          .set_data_type(fe::DataType_t::HALF));
    
      // W: {1, in_features, out_features} — cuDNN Matmul 的 B 矩阵
      std::vector<int64_t> w_dim = {1, in_features, out_features};
      std::vector<int64_t> w_stride = {in_features * out_features, out_features, 1};
    
      out_w = g->tensor(feg::Tensor_attributes()
          .set_name("w")
          .set_dim(w_dim).set_stride(w_stride)
          .set_data_type(fe::DataType_t::HALF));
    
      auto mm_attr = feg::Matmul_attributes()
          .set_name("FC_Matmul")
          .set_compute_data_type(fe::DataType_t::FLOAT);
    
      auto Y = g->matmul(out_x, out_w, mm_attr);
      Y->set_name("Y_mm")
        .set_dim({batch, 1, out_features})
        .set_stride({out_features, out_features, 1})
        .set_data_type(fe::DataType_t::HALF);
    
      if (has_bias) {
          // Bias: {1, 1, out_features}, FLOAT (即使 AMP 模式)
          std::vector<int64_t> b_dim = {1, 1, out_features};
          std::vector<int64_t> b_stride = {out_features, out_features, 1};
    
          out_b = g->tensor(feg::Tensor_attributes()
              .set_name("bias")
              .set_dim(b_dim).set_stride(b_stride)
              .set_data_type(fe::DataType_t::FLOAT));  // 【关键】Bias 永远是 FP32
    
          auto add_attr = feg::Pointwise_attributes()
              .set_name("FC_AddBias")
              .set_mode(fe::PointwiseMode_t::ADD)
              .set_compute_data_type(fe::DataType_t::FLOAT);
    
          auto Y_final = g->pointwise(Y, out_b, add_attr);
          Y_final->set_name("Y")
                  .set_dim({batch, 1, out_features})
                  .set_stride({out_features, out_features, 1})
                  .set_data_type(fe::DataType_t::HALF)
                  .set_output(true);
          out_y = Y_final;
      } else {
          Y->set_output(true);
          out_y = Y;
      }
    
      TR_CUDNN_FE_CHECK(g->validate(), "validate FC AMP FWD");
      TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build FC op graph");
      TR_CUDNN_FE_CHECK(g->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}),
                     "create FC plans");
      TR_CUDNN_FE_CHECK(g->check_support(handle), "check FC support");
      TR_CUDNN_FE_CHECK(g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE),
                     "build FC plans");
    
      return g;
  }
  3.4 FC AMP FWD Launch 函数
  struct FcAmpFwdCache {
      cudnnHandle_t handle = nullptr;
      int64_t batch = -1, in_features = -1, out_features = -1;
      bool has_bias = false;
      std::shared_ptr<feg::Graph> graph;
      std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
  };

  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>> s_fc_amp_fwd_caches;

  static void launch_fc_amp_fwd_cuda(
      const GraphNode& node,
      const MemoryPlan& mp,
      const DeviceContext& ctx,
      MultiStreamCaptureState& state)
  {
      TR_CHECK(node.input_ids.size() >= 2, ShapeError,
               "FC_AMP_FWD requires at least 2 inputs (x, w)");

      const auto* p = std::get_if<FCParams>(&node.params.data);
      TR_CHECK(p != nullptr, ValueError, "FC_AMP_FWD missing FCParams");
    
      const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
      const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    
      __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
      __half* w = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
      __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    
      float* b = nullptr;
      if (p->bias && node.input_ids.size() > 2) {
          b = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));
      }
    
      cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
      int si = state.get_or_register(s);
      state.output_stream_idx = si;
      state.streams[si].has_pending_work = true;
    
      cudnnHandle_t handle = static_cast<cudnnHandle_t>(
          ctx.cudnn_handle(StreamKind::COMP_1));
    
      int64_t batch = dt_x.n();
      int64_t in_features = dt_x.h() * dt_x.w() * dt_x.c();
      int64_t out_features = p->out_features;
    
      auto it = s_fc_amp_fwd_caches.find(handle);
      bool rebuild = (it == s_fc_amp_fwd_caches.end())
          || it->second->batch != batch
          || it->second->in_features != in_features
          || it->second->out_features != out_features
          || it->second->has_bias != p->bias;
    
      if (rebuild) {
          // 重建前检查是否在 capture 中（同 RELU/TANH）
          // ...
          auto cache = std::make_unique<FcAmpFwdCache>();
          cache->handle = handle;
          cache->batch = batch;
          cache->in_features = in_features;
          cache->out_features = out_features;
          cache->has_bias = p->bias;
          cache->graph = build_fc_amp_fwd_graph(
              handle, batch, in_features, out_features, p->bias,
              cache->x_attr, cache->w_attr, cache->b_attr, cache->y_attr);
    
          // 【关键】上报 workspace 大小给 Task
          // 方式 1: 通过 DeviceContext 的回调
          // 方式 2: 在 compile 阶段统一收集所有 cache 的 workspace
          size_t ws = cache->graph->get_workspace_size();
          (void)ws; // 待接入 Task 的 workspace 收集机制
    
          s_fc_amp_fwd_caches[handle] = std::move(cache);
      }
    
      auto& cache = *s_fc_amp_fwd_caches[handle];
      std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
          {cache.x_attr, x},
          {cache.w_attr, w},
          {cache.y_attr, y}
      };
      if (p->bias && b) {
          vp[cache.b_attr] = b;
      }
    
      // 【关键】从 DeviceContext 获取永久 workspace，不传 nullptr
      void* ws = ctx.workspace(StreamKind::COMP_1);
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
      cudaEventRecord(state.streams[si].last_done_event, s);
  }
  3.5 FC AMP BWD 的 cuDNN FE Graph
  FC BWD 需要计算 dX = dY @ W。同样使用 Matmul：
  static std::shared_ptr<feg::Graph> build_fc_amp_bwd_graph(
      cudnnHandle_t handle,
      int64_t batch, int64_t in_features, int64_t out_features,
      std::shared_ptr<feg::Tensor_attributes>& out_dy,
      std::shared_ptr<feg::Tensor_attributes>& out_w,
      std::shared_ptr<feg::Tensor_attributes>& out_dx)
  {
      auto g = std::make_shared<feg::Graph>();
      g->set_io_data_type(fe::DataType_t::HALF)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);

      // dY: {batch, 1, out_features}
      out_dy = g->tensor(feg::Tensor_attributes()
          .set_name("dy")
          .set_dim({batch, 1, out_features})
          .set_stride({out_features, out_features, 1})
          .set_data_type(fe::DataType_t::HALF));
    
      // W: {1, out_features, in_features} — 注意转置，Matmul(dY, W^T)
      // 实际上 cuDNN Matmul 的 B 矩阵需要是 {1, out_features, in_features}
      out_w = g->tensor(feg::Tensor_attributes()
          .set_name("w")
          .set_dim({1, out_features, in_features})
          .set_stride({out_features * in_features, in_features, 1})
          .set_data_type(fe::DataType_t::HALF));
    
      auto mm_attr = feg::Matmul_attributes()
          .set_name("FC_Bwd_Matmul")
          .set_compute_data_type(fe::DataType_t::FLOAT);
    
      auto dX = g->matmul(out_dy, out_w, mm_attr);
      dX->set_name("dx")
         .set_dim({batch, 1, in_features})
         .set_stride({in_features, in_features, 1})
         .set_data_type(fe::DataType_t::HALF)
         .set_output(true);
    
      // validate, build...
      return g;
  }
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  四、Workspace 收集机制的具体实现建议
  4.1 最小侵入式方案：Compile 前遍历所有 cache
  由于所有 cuDNN FE graph cache 都是全局 static 的（s_*_caches），可以在 compile 阶段通过辅助函数统一收集：
  // 新增：backend/workspace_collector.cpp 或在 op_registry.cpp 中

  size_t collect_max_workspace_from_all_caches() {
      size_t max_ws = 0;

      // 遍历所有算子的 cache
      for (auto& [handle, cache] : s_conv_fwd_caches) {
          max_ws = std::max(max_ws, static_cast<size_t>(cache->graph->get_workspace_size()));
      }
      for (auto& [handle, cache] : s_conv_bwd_caches) {
          max_ws = std::max(max_ws, static_cast<size_t>(cache->graph->get_workspace_size()));
      }
      for (auto& [handle, cache] : s_fc_amp_fwd_caches) {
          max_ws = std::max(max_ws, static_cast<size_t>(cache->graph->get_workspace_size()));
      }
      for (auto& [handle, cache] : s_fc_amp_bwd_caches) {
          max_ws = std::max(max_ws, static_cast<size_t>(cache->graph->get_workspace_size()));
      }
      // ... 其他算子
    
      return max_ws;
  }
  然后在 compile_capture_simple() 中：
  // warmup 完成后
  size_t max_ws = collect_max_workspace_from_all_caches();
  for (auto& ctx : device_contexts_) {
      for (int k = 0; k < 5; ++k) {
          ctx->pre_allocate_workspace(static_cast<StreamKind>(k), max_ws);
      }
  }
  4.2 更优雅的方案：Warmup 钩子回调
  在 DeviceContext 或 TaskBase 中增加一个 workspace 收集器：
  class TaskBase {
  public:
      void report_workspace_needed(size_t bytes) {
          max_workspace_size_ = std::max(max_workspace_size_, bytes);
      }

      size_t max_workspace_size() const { return max_workspace_size_; }

  private:
      size_t max_workspace_size_ = 0;
  };
  warmup 时算子调用 task.report_workspace_needed(ws)。但问题是 warmup 时 Task 指针不一定方便传递。
  折中方案：在 DeviceContext 中增加一个临时收集器：
  class DeviceContext {
  public:
      void report_workspace_needed(size_t bytes) {
          reported_max_ws_ = std::max(reported_max_ws_, bytes);
      }
      size_t reported_max_workspace() const { return reported_max_ws_; }
      void clear_reported_workspace() { reported_max_ws_ = 0; }
  private:
      size_t reported_max_ws_ = 0;
  };
  warmup 每个 rank 前 clear()，warmup 后 reported_max_workspace() 获取该 rank 的最大值，再取所有 rank 的最大值。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  五、总结：实施步骤
   步骤   文件                   内容
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1      device_context.h/cpp   确认现有 workspace API 满足需求（已完成 ✅）
   2      task_base.h/cpp        新增 max_workspace_size_ 和预分配逻辑
   3      op_registry.cpp        Conv warmup 中移除 cudaMalloc/Free，改为上报 ws 大小
   4      conv_op.cpp            launch 中改用 ctx.workspace() 传入 execute
   5      fc_op.cpp (新版)       实现 FC AMP FWD/BWD，使用 ctx.workspace()
   6      tanh_op.cpp            已修复（传 nullptr，不需要 ws）✅
   7      relu_op.cpp            保持不变（传 nullptr）✅
  核心口诀：

  ▌ Warmup 探测，Compile 预配，Launch 复用，绝不在内部分配。

# 【小伙伴D】

## FC 算子 + Workspace 方案
### 一、现状分析 1.1 已有基础设施
组件 状态 说明 DeviceContext::workspaces_[5] ✅ 已实现 5 个流各自有 {ptr, size} workspace pre_allocate_workspace(kind, size) ✅ 已实现 256B 对齐，cudaMalloc，idempotent ensure_workspace(kind, size) ✅ 已实现 不够大就抛异常（ 不支持 grow ） workspace(kind) / workspace_size(kind) ✅ 已实现 只读查询 per-handle cache map 模式 ✅ relu_op 已验证 unordered_map<cudnnHandle_t, unique_ptr<Cache>> launch_cuda(GraphNode, MP, DC, MCState) 新范式 ✅ fc_op.cpp 已采用 但 AMP 版本用的仍是 FP32 kernel require_warmup() 列表 ❌ FC 不在其中 FC 用自定义 kernel，不走 cuDNN warmup
 1.2 FC 算子当前问题
问题 文件 说明 FC_AMP_FWD/BWD 复用 FP32 kernel fc_op.cpp:L210-L224 launch_cuda = launch_fc_fwd_cuda ，用 float* 处理 FP16 数据 FC_AMP_BWD cuDNN graph 未实现 fc_op_legacy.cpp:L186-L196 TR_NOT_IMPLEMENTED 旧版自定义 get_row_stride() fc_op_legacy.cpp:L69-L75 手工 256B/128B 对齐，应改用框架 API 无 workspace 管理 全部 FC 文件 无

### 二、方案总览
```
┌─────────────────────────────────────────────────────────────┐
│                    FC 算子分层架构                            │
├─────────────────────────────────────────────────────────────┤
│  FC_FP32_FWD/BWD:  手写 CUDA kernel  (fc_op.cu)             │
│                     无 workspace，无 cuDNN                   │
│                     【保持不变】                               │
├─────────────────────────────────────────────────────────────┤
│  FC_AMP_FWD/BWD:   cuDNN Frontend Matmul graph              │
│                    需要 workspace                            │
│                    per-handle cache map                      │
│                    bias 始终 FP32                            │
│                    【全新实现】                                │
├─────────────────────────────────────────────────────────────┤
│  Workspace 管理:    融入 warmup → capture 生命周期            │
│                    采用 FFF.md 第15条替代方案                  │
└─────────────────────────────────────────────────────────────┘
```
### 三、FC_AMP 算子详细设计 3.1 Cache 结构（参考 relu_op.cpp 模式）
```
// per-handle cache —— 模板模式与 relu_op 一致
struct FcAmpFwdCache {
    cudnnHandle_t handle = nullptr;
    int batch = -1, in_features = -1, out_features = -1;
    int64_t x_n_stride = -1, w_w_stride = -1, y_n_stride = -1;
    bool has_bias = false;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
};

struct FcAmpBwdCache {
    cudnnHandle_t handle = nullptr;
    int batch = -1, in_features = -1, out_features = -1;
    int64_t dy_n_stride = -1, w_w_stride = -1, dx_n_stride = -1;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> dy_attr, w_attr, dx_attr;
};

// 静态 per-handle 缓存 —— 多 GPU 安全
static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>> 
s_fc_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>> 
s_fc_amp_bwd_caches;
``` 3.2 Forward Graph: Y = Matmul(X, W) + bias
```
IO 类型:    HALF (FP16)
Compute 类型: FLOAT (FP32, 利用 Tensor Core)
Bias 类型:  FLOAT (FP32) ← FFF.md 要求

Tensor 维度 (cuDNN 3D):
  X: {batch,  1, in_features}     stride: {x_n_stride, in_features, 1}
  W: {1, in_features, out_features} stride: {in_features*w_w_stride, w_w_stride, 1}
  B: {1, 1, out_features}         stride: {out_features, out_features, 1}  
  type=FLOAT
  Y: {batch, 1, out_features}     stride: {y_n_stride, out_features, 1}
```
关键点 ：

- W 布局取 dt_w.w_stride_cuda() 作为 row stride，这与手写 kernel 的 w_row_stride 语义一致
- cuDNN Matmul 的 batch × in_features × out_features 3D 布局天然匹配 FC 的需求
- Bias 在 AMP 图中 data_type = FLOAT ，由 cuDNN 自动做 FP32→FP16 转换后与 Y 相加 3.3 Backward Graph: dX = Matmul(dY, W)
```
IO 类型:    HALF (FP16)
Compute 类型: FLOAT (FP32)

dY: {batch, 1, out_features}   stride: {dy_n_stride, out_features, 1}
W:  需要通过 Matmul_attributes 设置转置
     或者构造为 {1, out_features, in_features}
dX: {batch, 1, in_features}    stride: {dx_n_stride, in_features, 1}
```
注 ：cuDNN Frontend Matmul 支持 Matmul_attributes 设置 transpose。BWD 不需要 bias（dX = dY × W，不涉及 bias 的梯度）。
 3.4 launch_fc_amp_fwd_cuda 流程
```
1. 从 GraphNode 解析 FCParams (out_features, bias flag)
2. 从 MemoryPlan 获取 DTensor (x, w, y)
3. 从 DeviceContext 获取指针 (x_ptr, w_ptr, b_ptr, y_ptr)
4. 获取 stride: n_stride_cuda(), w_stride_cuda()
5. 获取 cudnnHandle (per-stream)
6. 检查/重建 cache (per-handle, shape/stride mismatch → rebuild)
7. 获取 workspace: ctx.workspace(StreamKind::COMP_1) + ctx.workspace_size
(StreamKind::COMP_1)
8. graph->execute(handle, vp, workspace_ptr)
9. cudaEventRecord
``` 3.5 CPU 路径
FC_AMP CPU 路径应像 relu_op 一样设置 TR_TYPE_ERROR （CPU 不支持 FP16）：

```
static void launch_fc_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("FC_AMP_FWD is not supported on CPU (FP16 not available)");
}
```
### 四、Workspace 方案 4.1 采用 FFF.md 第 15 条的替代方案（推荐） "warmup 时先一次性申请一个足够大的常数值（比如 512MB），然后 warmup 过程中遇到需要更大的 workspace 的情况就把旧的释放、申请新的——这样就确保得到的是最大值了，而且 warmup 结束后得到的是准确的指针值，捕获和运行的时候都知道指针和字节数"
优势 ：无需 double-allocation（临时 → 永久），warmup 结束即获得最终 workspace。
 4.2 实现机制
新增方法 grow_workspace(kind, req_size) ：

```
void DeviceContext::grow_workspace(StreamKind kind, size_t req_size) {
    size_t aligned = align_up(req_size, 256);
    auto& ws = workspaces_[static_cast<size_t>(kind)];
    if (ws.size >= aligned) return;  // 已经足够大
    if (ws.ptr) { cudaFree(ws.ptr); }  // 释放旧的
    TR_CUDA_CHECK(cudaMalloc(&ws.ptr, aligned));
    ws.size = aligned;
}
``` 4.3 生命周期
```
时间线:
┌──────────────────────────────────────────────────────────────────┐
│ compile_alloc_hardware()                                          │
│   └→ DeviceContext 构造，workspaces_[] 初始化为 {nullptr, 0}      │
│                                                                  │
│ ★ Phase B1 — 捕获前准备                                          │
│                                                                  │
│ ★ Phase B2 — Warmup (每个 rank)                                  │
│   └→ launch_fc_amp_xxx_cuda() 被调用                             │
│      └→ ctx.grow_workspace(COMP_1, DEFAULT_WS)  // 首次分配512MB │
│      └→ graph->execute(handle, vp, ws)                           │
│      └→ (cuDNN 如果需要更大的, grow_workspace 自动扩容)           │
│   └→ cudaStreamSynchronize                                       │
│   └→ ★ 此时 workspace 已是最佳大小，指针有效                      │
│                                                                  │
│ ★ Phase B3 — Capture (每个 rank)                                 │
│   └→ launch_fc_amp_xxx_cuda() 被调用                             │
│      └→ ctx.workspace(COMP_1) 返回 warmup 确定的大小              │
│      └→ graph->execute(handle, vp, ws) —— 纯 cache hit           │
│                                                                  │
│ ★ Phase B4 — 首次 launch 验证                                    │
│                                                                  │
│ ★ Runtime — 每个 epoch 重复 launch                                │
│   └→ cudaGraphLaunch — workspace 自动复用                        │
│                                                                  │
│ ~DeviceContext() → free_workspaces() — 程序结束时释放              │
└──────────────────────────────────────────────────────────────────┘
``` 4.4 默认值选取
```
// 512 MB 初始 workspace，足够绝大多数 FC 层
static constexpr size_t kDefaultWorkspaceBytes = 512ULL * 1024 * 1024;
```
### 五、Warmup 集成 5.1 将 FC_AMP 加入 require_warmup()
op_registry.cpp 的 require_warmup() 函数需要添加：

```
case ComputeOp::FC_AMP_FWD:
case ComputeOp::FC_AMP_BWD:
    return true;
``` 5.2 Warmup 执行路径
现有的 warmup_single_cudnn_op() 对非 Conv 算子直接调用 entry.launch_cuda() 。FC_AMP 的 launch_cuda 内部会：

1. 首次调用 → grow_workspace(COMP_1, 512MB) → 构建 cuDNN graph → execute → cudaStreamSynchronize
2. 后续调用 → cache hit → 直接 execute
这与 relu_amp 的 warmup 模式完全一致，不需要特殊处理。

### 六、需修改的文件清单
文件 修改内容 优先级 fc_op.cpp 新增 launch_fc_amp_fwd_cuda / launch_fc_amp_bwd_cuda + cache + build_graph；分拆 FC_AMP 注册 🔴 核心 fc_op.cpp 新增 AMP CPU 不支持函数 🟡 device_context.h/cpp 新增 grow_workspace(kind, size) 方法 🔴 核心 op_registry.cpp require_warmup() 添加 FC_AMP_FWD/BWD 🟡 fc_op_legacy.cpp 标记为废弃（加注释），不删 🟢 文档 fc_op_legacy.cu 标记为废弃 🟢 文档

不需要修改的文件 ：

- fc_op.cu — FP32 kernel 保持不变
- capture_cuda.cpp — 不需要改动
- captured_graph.cpp — Warmup 路径已通用支持 launch_cuda
- task_base.cpp — compile_alloc_hardware() 不需要改动（workspace 在 warmup 中 lazy 分配）
### 七、与现有设计的一致性原则
原则 如何满足 C 通道连续 FFF.md 保证 FC 前必有 flatten，FC 算子本身不做处理 Bias FP32 cuDNN graph 中 bias tensor data_type = FLOAT 3D 输入 使用 {batch, 1, in_features} 维度 Stride API 使用框架 n_stride_cuda() / w_stride_cuda() ，废除旧版的 get_row_stride() Per-handle cache unordered_map<cudnnHandle_t, unique_ptr<Cache>> 与 relu_op 一致 每流一个 workspace DeviceContext::workspaces_[5] 天然支持 一次申请永久复用 grow 模式，warmup → capture → runtime 全程不释放 编译期防御检测 rebuild 路径的 #ifndef NDEBUG 检测（与 relu_op 一致）

### 八、实施步骤建议
1. Step 1 : 在 DeviceContext 中新增 grow_workspace() 方法
2. Step 2 : 在 fc_op.cpp 中实现 build_fc_amp_fwd_graph() 和 launch_fc_amp_fwd_cuda()
3. Step 3 : 实现 build_fc_amp_bwd_graph() 和 launch_fc_amp_bwd_cuda()
4. Step 4 : 修改 register_op_fc() 分拆 FC_AMP 注册（指向新的 launch 函数）
5. Step 5 : require_warmup() 添加 FC_AMP
6. Step 6 : 编译测试 FC FP32 + AMP
这就是我的完整方案。核心思路是： FP32 保持手写 kernel 不动 ， AMP 全新实现 cuDNN Frontend Matmul graph ， workspace 采用 "grow-on-demand" 策略在 warmup 中一次性确定 。整个过程与 relu_op 的成功模式保持一致，风险可控。
```

# 【用户补充】

我只是提个醒，仅供参考，不算是指导意见：

听说DeviceContext有workspace管理设施？如果有的话，大家考虑是不是把workspace的管理全权交给DeviceContext而非DeepLearningTask/SimpleTask。而且，鉴于workspace是每个流一个，是不是可以形成一个unordered map或别的数据结构，根据stream（每张卡上总共才5个）来获取它对应的workspace的指针和大小。warmup过程中如果遇到workspace不够大的情况，就realloc也是一个思路。毕竟只有传递了实际形状、且warmup实际进行了Heuristic搜索才知道workspace大小。申请初始workspace的办法、warmup阶段动态调整到最大或许不是最好的办法，但它至少是一个兜底的办法。希望小伙伴想出的办法至少不比这个差。

如果大家的方案已经比这个更好了，就忽略我说的。

