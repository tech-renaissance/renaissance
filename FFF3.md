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

  📋 FC算子和Workspace管理综合方案

  一、核心设计原则

  根据FFF.md要求和现有架构成功经验，遵循以下原则：

  1. 每流一个workspace：DeviceContext::workspaces_[5]已实现
  2. 一次申请，永久复用：绝对避免算子内部cudaMalloc/Free
  3. Warmup探测大小：唯一允许临时分配的阶段
  4. Bias永远FP32：即使AMP模式下也用FLOAT
  5. 3D张量输入：cuDNN Matmul要求{batch, 1, features}
  6. Per-handle缓存：参照relu_op.cpp的成功模式

  二、Workspace管理策略（推荐小伙D方案）

  采用FFF.md第15条的grow-on-demand方案，这是最科学合理的：

  // device_context.h 新增方法
  void DeviceContext::grow_workspace(StreamKind kind, size_t req_size) {
      size_t aligned = align_up(req_size, 256);  // 256字节对齐
      auto& ws = workspaces_[static_cast<size_t>(kind)];

      if (ws.size >= aligned) return;  // 已足够大，直接返回
    
      // 需要扩容：释放旧的，申请新的
      if (ws.ptr) {
          TR_CUDA_CHECK(cudaFree(ws.ptr));
      }
      TR_CUDA_CHECK(cudaMalloc(&ws.ptr, aligned));
      ws.size = aligned;
  }

  优势：
  - ✅ Warmup结束时即获得最终workspace大小和指针
  - ✅ 避免double-allocation（临时→永久）
  - ✅ 自动适应不同模型的workspace需求
  - ✅ 与CUDA Graph capture完美配合

  三、FC算子分层架构

  ┌─────────────────────────────────────────────────────┐
  │  FC_FP32_FWD/BWD:  保持现有手写CUDA kernel           │
  │  （fc_op.cu）无workspace，无cuDNN                   │
  │  【保持不变】                                        │
  ├─────────────────────────────────────────────────────┤
  │  FC_AMP_FWD/BWD:   全新cuDNN Frontend Matmul实现   │
  │  - 需要 workspace                                   │
  │  - Per-handle cache map                            │
  │  - Bias 始终 FP32                                   │
  │  - 【参照relu_op.cpp模式】                          │
  └─────────────────────────────────────────────────────┘

  四、FC_AMP算子详细设计

  4.1 Cache结构（完全参照relu_op.cpp）

  struct FcAmpFwdCache {
      cudnnHandle_t handle = nullptr;
      int64_t batch = -1, in_features = -1, out_features = -1;
      int64_t x_n_stride = -1, w_w_stride = -1, y_n_stride = -1;
      bool has_bias = false;
      std::shared_ptr<feg::Graph> graph;
      std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
  };

  struct FcAmpBwdCache {
      cudnnHandle_t handle = nullptr;
      int64_t batch = -1, in_features = -1, out_features = -1;
      int64_t dy_n_stride = -1, w_w_stride = -1, dx_n_stride = -1;
      std::shared_ptr<feg::Graph> graph;
      std::shared_ptr<feg::Tensor_attributes> dy_attr, w_attr, dx_attr;
  };

  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>> s_fc_amp_fwd_caches;
  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>> s_fc_amp_bwd_caches;

  4.2 FC_AMP_FWD Graph构建

  // Y = Matmul(X, W) + bias
  // IO类型: HALF, Compute类型: FLOAT, Bias类型: FLOAT
  static std::shared_ptr<feg::Graph> build_fc_amp_fwd_graph(
      cudnnHandle_t handle,
      int64_t batch, int64_t in_features, int64_t out_features, bool has_bias,
      std::shared_ptr<feg::Tensor_attributes>& out_x,
      std::shared_ptr<feg::Tensor_attributes>& out_w,
      std::shared_ptr<feg::Tensor_attributes>& out_b,
      std::shared_ptr<feg::Tensor_attributes>& out_y)
  {
      auto g = std::make_shared<feg::Graph>();
      g->set_io_data_type(fe::DataType_t::HALF)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);

      // X: {batch, 1, in_features}
      out_x = g->tensor(feg::Tensor_attributes()
          .set_name("x")
          .set_dim({batch, 1, in_features})
          .set_stride({in_features, in_features, 1})
          .set_data_type(fe::DataType_t::HALF));
    
      // W: {1, in_features, out_features}
      out_w = g->tensor(feg::Tensor_attributes()
          .set_name("w")
          .set_dim({1, in_features, out_features})
          .set_stride({in_features * out_features, out_features, 1})
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
          // Bias永远FP32
          out_b = g->tensor(feg::Tensor_attributes()
              .set_name("bias")
              .set_dim({1, 1, out_features})
              .set_stride({out_features, out_features, 1})
              .set_data_type(fe::DataType_t::FLOAT));  // 关键！
    
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
    
      // 标准cuDNN FE流程
      TR_CUDNN_FE_CHECK(g->validate(), "validate FC AMP FWD");
      TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build FC op graph");
      TR_CUDNN_FE_CHECK(g->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}),
                     "create FC plans");
      TR_CUDNN_FE_CHECK(g->check_support(handle), "check FC support");
      TR_CUDNN_FE_CHECK(g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE),
                     "build FC plans");
    
      return g;
  }

  4.3 Launch函数（关键部分）

  static void launch_fc_amp_fwd_cuda(
      const GraphNode& node,
      const MemoryPlan& mp,
      const DeviceContext& ctx,
      MultiStreamCaptureState& state)
  {
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
      cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
    
      int64_t batch = dt_x.n();
      int64_t in_features = dt_x.h() * dt_x.w() * dt_x.c();
      int64_t out_features = p->out_features;
    
      // Cache检查和重建
      auto it = s_fc_amp_fwd_caches.find(handle);
      bool rebuild = (it == s_fc_amp_fwd_caches.end())
          || it->second->batch != batch
          || it->second->in_features != in_features
          || it->second->out_features != out_features
          || it->second->has_bias != p->bias;
    
      if (rebuild) {
          // Debug检测：Capture期间禁止重建
          #ifndef NDEBUG
          {
              cudaStreamCaptureStatus cap_status;
              cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
              if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
                  TR_LOG_ERROR("fc") << "[AMP FWD] Rebuilding cuDNN FE graph inside CUDA Graph capture!";
              }
          }
          #endif
    
          auto cache = std::make_unique<FcAmpFwdCache>();
          cache->handle = handle;
          cache->batch = batch;
          cache->in_features = in_features;
          cache->out_features = out_features;
          cache->has_bias = p->bias;
    
          cache->graph = build_fc_amp_fwd_graph(
              handle, batch, in_features, out_features, p->bias,
              cache->x_attr, cache->w_attr, cache->b_attr, cache->y_attr);
    
          s_fc_amp_fwd_caches[handle] = std::move(cache);
      }
    
      auto& cache = *s_fc_amp_fwd_caches[handle];
    
      // 构建variant pack
      std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
          {cache.x_attr, x},
          {cache.w_attr, w},
          {cache.y_attr, y}
      };
      if (p->bias && b) {
          vp[cache.b_attr] = b;
      }
    
      // 关键：从DeviceContext获取workspace，如果不够大会自动grow
      size_t required_ws = cache.graph->get_workspace_size();
      ctx.grow_workspace(StreamKind::COMP_1, required_ws);
      void* ws = ctx.workspace(StreamKind::COMP_1);
    
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
    
      // MultiStreamCaptureState处理
      int si = state.get_or_register(s);
      state.output_stream_idx = si;
      state.streams[si].has_pending_work = true;
      cudaEventRecord(state.streams[si].last_done_event, s);
  }

  五、实施步骤

  阶段1：Workspace基础设施

  1. 在device_context.h/cpp中实现grow_workspace()方法
  2. 确认现有的workspace()和workspace_size()方法正常工作

  阶段2：FC_AMP算子实现

  1. 在fc_op.cpp中实现build_fc_amp_fwd_graph()和build_fc_amp_bwd_graph()
  2. 实现对应的launch函数，完全参照relu_op.cpp的模式
  3. 添加per-handle cache结构和静态map

  阶段3：注册和集成

  1. 修改register_op_fc()，让FC_AMP_FWD/BWD指向新的实现
  2. 在op_registry.cpp的require_warmup()中添加FC_AMP算子
  3. CPU版本设置为主动报TR_TYPE_ERROR

  阶段4：测试验证

  1. 编译test_fc_amp_fwd_bwd测试程序
  2. 验证workspace自动扩容机制
  3. 确认bias在AMP模式下使用FP32精度

  六、关键优势

  1. 与现有架构一致：完全参照relu_op.cpp的成功模式
  2. 最小侵入性：FP32版本保持不变，只重构AMP版本
  3. 自动workspace管理：grow-on-demand机制，无需手动计算最大值
  4. CUDA Graph友好：Warmup结束后workspace指针固定，perfect for capture
  5. 多GPU安全：Per-handle caching，每张卡独立缓存
  6. 符合FFF.md要求：bias永远FP32，3D张量输入，C通道连续

  七、风险控制

  1. FC_FP32保持不变：降低风险，AMP版本全新实现
  2. 渐进式测试：先FC_AMP_FWD，再FC_AMP_BWD
  3. Debug检测：Capture期间重建cache会报错
  4. 兜底方案：grow_workspace失败时明确错误信息

  这个方案综合了三位小伙伴的智慧，选择最科学合理的grow-on-demand
  workspace策略，完全遵循现有成功模式，风险可控，实施路径清晰。

# 【小伙伴K】

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

✨ 阅读根目录下的FFF2.md和有关文件，参考新版算子范例relu_op.cpp，然后针对FC算子和workspace的计算和申请，提出你的科学、合
理的方案

• 用户要求我阅读根目录下的 FFF2.md，以及有关文件，参考新版算子范例 relu_op.cpp，然后针对 FC 算子和 workspace 的计算和申
  请，提出科学合理的方案。

  让我先读取 FFF2.md，然后综合之前了解到的所有信息，给出一个经过深思熟虑的方案。
• Used ReadFile (FFF2.md)
• 用户要求我阅读 FFF2.md 后，综合所有信息，参考新版算子范例 relu_op.cpp，针对 FC 算子和 workspace 的计算和申请，提出一个
  科学合理、经过深思熟虑的方案。

  FFF2.md 包含了多个小伙伴的方案：

  • 小伙伴 S：基本的 per-handle cache 方案，建议 warmup 时申请初始 512MB，不够再释放申请新的
  • 小伙伴 K：详细的方案，包括 DeviceContext 已有的 workspace API、Task 维护 max_workspace_size、compile 阶段预分配等
  • 小伙伴 D：非常详细的方案，建议采用 "grow-on-demand" 策略，在 DeviceContext 中新增 grow_workspace() 方法，warmup 时一
    性确定最终大小

  用户补充提醒：

  1. workspace 管理全权交给 DeviceContext 而非 Task
  2. 每个流一个 workspace，可以用 unordered_map 或数组（共5个流）管理
  3. warmup 中遇到不够大的情况就 realloc
  4. 初始申请 + 动态调整是一个兜底办法，但希望有更好的方案

  现在让我重新审视所有信息，提出一个我自己的、科学合理的方案。

  关键分析：

  1. 关于 workspace 大小的不确定性

  • cuDNN FE get_workspace_size() 只有在 graph build 完成后才能知道
  • graph build 发生在 warmup 阶段（因为 build_plans 需要实际的 handle 和 heuristics）
  • 不同 shape、不同算子、不同 handle 可能需要不同大小的 workspace
  • 同一个 handle 的不同 graph 也可能需要不同大小的 workspace

  2. DeviceContext 已具备的基础设施

  • workspaces_[5] — 每个 stream 一个
  • pre_allocate_workspace(kind, size) — 预分配
  • ensure_workspace(kind, size) — 懒分配（但不够大时抛异常，不支持 grow）
  • workspace(kind) / workspace_size(kind) — 查询

  3. 现有问题

  • ensure_workspace 不支持扩容（不够大时抛异常）
  • pre_allocate_workspace 是 idempotent 的（已分配则跳过）
  • 没有 grow_workspace 或 realloc_workspace 方法

  4. 方案比较

  小伙伴 S/K 的方案：Task 维护 max_workspace_size，compile 阶段预分配

  • 优点：逻辑清晰，compile 后大小确定
  • 缺点：需要 Task 参与管理，耦合度高；warmup 中临时分配再释放，多一次分配开销

  小伙伴 D 的方案：DeviceContext 新增 grow_workspace，warmup 中动态调整

  • 优点：DeviceContext 全权管理，Task 不介入；warmup 结束即得最终大小，无额外分配
  • 缺点：初始 512MB 可能浪费显存；如果 warmup 中需要多次扩容，有多次 cudaFree/Malloc 开销

  用户补充：全权交给 DeviceContext，每个流一个，不够大就 realloc

  我的方案：

  综合考虑，我认为最佳方案是：

  "DeviceContext 全权管理 + Lazy Grow + Warmup 探测最终大小"

  核心思路：

  1. DeviceContext 全权负责 workspace 生命周期，Task 完全不介入
  2. 每个 stream 一个 workspace，用数组 workspaces_[5] 管理（已经是这样）
  3. 新增 ensure_workspace_grow() 方法：如果当前不够大，自动释放旧内存、申请新内存
  4. 算子 launch 函数中自动调用 ensure_workspace_grow()：如果 cache rebuild 时发现 workspace 不够大，自动扩容
  5. warmup 阶段自然探测最终大小：因为 warmup 会触发所有算子的 cache rebuild，每个算子都会检查 workspace 是否足够并自动
     ，warmup 结束后 workspace 就是所有算子中最大的那个
  6. capture 和 runtime 阶段直接复用：因为 capture 时 cache hit，workspace 已经确定且足够大

  这个方案的优点：

  • DeviceContext 全权管理，Task 零介入
  • 无需 Task 维护 max_workspace_size
  • 无需 compile 阶段显式预分配（warmup 中已经自然完成）
  • 无需初始 512MB 大buffer（按需分配，第一次需要多少就分配多少）
  • 与 relu_op/tanh_op 的 launch 模式完全一致
  • 如果后续新算子需要更大 workspace，ensure_workspace_grow 会自动处理

  具体实现：

  // device_context.h
  class DeviceContext {
  public:
      // ... 现有 API ...

      // 新增：确保 workspace 足够大，不够则自动扩容
      void ensure_workspace_grow(StreamKind kind, size_t req_size);
  };

  // device_context.cpp
  void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) {
  #ifdef TR_USE_CUDA
      size_t idx = stream_index(kind);
      auto& ws = workspaces_[idx];

      size_t aligned = ((req_size + 255) / 256) * 256;
      if (ws.size >= aligned) {
          return;  // 已经足够大
      }
    
      // 释放旧的
      if (ws.ptr) {
          cudaFree(ws.ptr);
          ws.ptr = nullptr;
          ws.size = 0;
      }
    
      // 申请新的
      int prev_device = -1;
      cudaGetDevice(&prev_device);
      cudaSetDevice(device_id_);
    
      cudaError_t err = cudaMalloc(&ws.ptr, aligned);
      cudaSetDevice(prev_device);
    
      if (err != cudaSuccess) {
          ws.ptr = nullptr;
          ws.size = 0;
          TR_GPU_OOM("Failed to grow workspace to " << aligned
                     << " bytes for stream " << static_cast<int>(kind));
      }
    
      ws.size = aligned;
      TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                             << ": workspace for stream " << static_cast<int>(kind)
                             << " grown to " << aligned << " bytes";
  #endif
  }

  算子 launch 函数中的使用：

  static void launch_fc_amp_fwd_cuda(...) {
      // ... cache rebuild check ...
      if (rebuild) {
          // ... build graph ...
          size_t ws_needed = cache.graph->get_workspace_size();
          if (ws_needed > 0) {
              ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_needed);
          }
          s_fc_amp_fwd_caches[handle] = std::move(cache);
      }

      // ... 构建 vp ...
    
      void* ws = ctx.workspace(StreamKind::COMP_1);
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
      // ...
  }

  这个方案的关键是：ensure_workspace_grow 在 warmup 阶段会被调用，自然地将 workspace 调整到所需大小。capture 和 runtime
  阶段不会再触发 rebuild（cache hit），所以不会再调用 ensure_workspace_grow。

  对于 Conv 算子，同样的模式：

  // conv_op.cpp 中
  void* ws = ctx.workspace(StreamKind::COMP_1);
  TR_CUDNN_FE_CHECK(graph->execute(handle, vp, ws), "CONV execute");

  而在 op_registry.cpp 的 warmup 中，Conv 的 warmup 临时分配可以改为：

  // 原来：
  int64_t ws_bytes = graph->get_workspace_size();
  void* temp_ws = nullptr;
  if (ws_bytes > 0) cudaMalloc(&temp_ws, ws_bytes);
  // ... execute ...
  if (temp_ws) cudaFree(temp_ws);

  // 改为：
  size_t ws_bytes = graph->get_workspace_size();
  if (ws_bytes > 0) {
      ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_bytes);
  }
  void* ws = ctx.workspace(StreamKind::COMP_1);
  TR_CUDNN_FE_CHECK(graph->execute(handle, vp, ws), "warmup execute");
  // 无需释放！workspace 永久复用

  这样 warmup 阶段使用的 workspace 就是最终复用的 workspace，没有临时分配/释放。

  这就是我认为最科学合理的方案。让我整理成文档。
• 我已完整阅读了 FFF2.md 以及所有相关文件。以下是我经过深度分析后提出的方案。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一、核心判断：现有方案的比较与取舍
  先对 FFF2.md 中三位小伙伴的方案做一个客观评估：
                                                       小伙伴 D（grow-on-demand + 512
   维度               小伙伴 S/K（Task 预分配）        MB 初始）                        我的评估
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   管理主体           Task 维护 max_workspace_size_    DeviceContext 全权管理           ✅ DeviceContext 管理更好（与用
                                                                                        户补充一致）
   分配时机           Warmup 临时分配 → Compile 预分   Warmup 中 512MB 初始，不够再扩   两者都有额外开销
                      配                               容
   额外开销           临时分配/释放一次 + 正式分配一   512MB 可能长期闲置；可能多次扩   都有优化空间
                      次                               容
   shape 变化适应性   需重新 compile                   自动 grow                        Grow 模式更灵活
   与现有架构一致性   需改 TaskBase                    只改 DeviceContext               DeviceContext 更内聚
  用户补充的关键提示：
  ▌ "workspace 的管理全权交给 DeviceContext 而非 Task"
  ▌ "warmup 过程中如果遇到 workspace 不够大的情况，就 realloc"
  这明确指向一个结论：DeviceContext 全权管理 + 自动扩容（grow/realloc） 是正确方向。
  但小伙伴 D 的 512MB 初始值 是一个不必要的保守策略——它浪费了显存，且如果模型很小，512MB 是巨大浪费；如果模型很大，512M…
  仍然不够。我们应该采用 "首次按需分配 + 不够自动扩容" 的精确策略。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  二、我的方案：DeviceContext 全权管理 + ensure_workspace_grow 精确扩容
  2.1 设计哲学
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  原则 1: DeviceContext 是 workspace 的唯一所有者，Task 完全不介入           │
  │  原则 2: 每个 stream 一个 workspace，天然由 workspaces_[5] 数组隔离        │
  │  原则 3: 首次使用时按需分配（需要多少分配多少），不是 512MB 盲猜            │
  │  原则 4: 不够用时自动释放旧内存、申请新内存，warmup 结束即得精确值         │
  │  原则 5: Capture 和 Runtime 直接复用 warmup 确定的 workspace，零额外分配   │
  │  原则 6: 算子内部绝不 cudaMalloc/Free，统一从 ctx.workspace() 取指针       │
  └─────────────────────────────────────────────────────────────────────────┘
  2.2 时间线对比
  传统方案（临时分配）          小伙伴 D 方案（512MB 初始）        我的方案（精确按需）
  ─────────────────────        ────────────────────────        ─────────────────────
  Warmup: cudaMalloc(ws)       Warmup: cudaMalloc(512MB)       Warmup:
          execute                      execute                         ensure_workspace_grow(实际大小)
          cudaFree(ws)                 (可能需要扩容)                    execute
                                                                ↓
  Compile: cudaMalloc(max_ws)  Compile: 复用 512MB              Compile: 复用已确定大小的 workspace
           (第二次分配)                                               (无额外分配)

  Runtime: 复用                 Runtime: 复用                    Runtime: 复用
  ─────────────────────        ────────────────────────        ─────────────────────
  问题: 分配两次                 问题: 512MB 可能浪费/不够        优势: 只分配一次，大小精确
  2.3 核心改动：DeviceContext 新增 ensure_workspace_grow
  // device_context.h — 新增一个方法
  class DeviceContext {
  public:
      // ... 现有 API 保持不变 ...

      /**
       * @brief 确保 workspace 足够大，不够时自动扩容（释放旧内存 + 申请新内存）
       * @note 这是唯一允许在运行时调整 workspace 大小的入口
       * @note 与 pre_allocate_workspace 不同：后者是 compile 阶段的一次性预分配
       *        而 ensure_workspace_grow 是在 warmup/launch 阶段按需扩容
       */
      void ensure_workspace_grow(StreamKind kind, size_t req_size);
  };
  // device_context.cpp
  void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) {
  #ifdef TR_USE_CUDA
      if (device_id_ < 0) return;  // CPU 模式无 workspace

      size_t idx = stream_index(kind);
      auto& ws = workspaces_[idx];
    
      // 256B 对齐（CUDA 要求）
      size_t aligned = ((req_size + 255) / 256) * 256;
    
      // 已经足够大，什么都不做
      if (ws.size >= aligned) {
          return;
      }
    
      // 释放旧的（如果存在）
      if (ws.ptr) {
          cudaFree(ws.ptr);
          ws.ptr = nullptr;
          ws.size = 0;
      }
    
      // 申请新的
      int prev_device = -1;
      cudaGetDevice(&prev_device);
      cudaSetDevice(device_id_);
    
      cudaError_t err = cudaMalloc(&ws.ptr, aligned);
      cudaSetDevice(prev_device);
    
      if (err != cudaSuccess) {
          ws.ptr = nullptr;
          ws.size = 0;
          TR_GPU_OOM("Failed to grow workspace to " << aligned
                     << " bytes for stream " << static_cast<int>(kind)
                     << " on device " << device_id_
                     << ": " << cudaGetErrorString(err));
      }
    
      ws.size = aligned;
      TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                             << ": workspace for stream " << static_cast<int>(kind)
                             << " grown to " << aligned << " bytes";
  #else
      (void)kind;
      (void)req_size;
  #endif
  }
  2.4 算子 launch 函数中的使用模式（通用模板）
  // 适用于所有需要 workspace 的算子：Conv, FC, Matmul 等
  static void launch_xxx_amp_xxx_cuda(...) {
      // ... cache 查找 ...

      if (rebuild) {
          // ... build graph ...
    
          // 【关键】在 graph 构建完成后，检查并确保 workspace 足够
          size_t ws_needed = cache.graph->get_workspace_size();
          if (ws_needed > 0) {
              ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_needed);
          }
    
          s_xxx_caches[handle] = std::move(cache);
      }
    
      // ... 构建 vp ...
    
      // 【关键】从 DeviceContext 获取永久 workspace 指针
      void* ws = ctx.workspace(StreamKind::COMP_1);
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute XXX");
    
      cudaEventRecord(state.streams[si].last_done_event, s);
  }
  2.5 对现有算子的具体改造
   算子                            当前问题                        改造方式
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Conv FWD/BWD (conv_op.cpp)      launch 中临时 cudaMalloc/Free   移除临时分配，改用 ctx.workspace() + ensure_workspac
                                                                   e_grow
   Conv warmup (op_registry.cpp)   warmup 中临时 cudaMalloc/Free   改用 ctx.ensure_workspace_grow() + ctx.workspace()
   FC AMP FWD/BWD (新增)           无 workspace 管理               全新实现，使用上述通用模板
   TANH AMP FWD/BWD                已传 nullptr                    ✅ 无需改动（Pointwise 不需要 workspace）
   RELU AMP FWD/BWD/INF            已传 nullptr                    ✅ 无需改动
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  三、FC 算子新版实现方案
  3.1 注册表扩展
  // op_kind.h / op_kind.cpp
  enum class ComputeOp {
      // ...
      FC_FP32_FWD,
      FC_FP32_BWD,
      FC_AMP_FWD,
      FC_AMP_BWD,
  };
  3.2 FC AMP FWD：cuDNN FE Graph 构建
  struct FcAmpFwdCache {
      cudnnHandle_t handle = nullptr;
      int64_t batch = -1, in_features = -1, out_features = -1;
      int64_t x_stride = -1, w_stride = -1, y_stride = -1;
      bool has_bias = false;
      std::shared_ptr<feg::Graph> graph;
      std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
  };

  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>>
      s_fc_amp_fwd_caches;

  static std::shared_ptr<feg::Graph> build_fc_amp_fwd_graph(
      cudnnHandle_t handle,
      int64_t batch, int64_t in_features, int64_t out_features,
      bool has_bias,
      std::shared_ptr<feg::Tensor_attributes>& out_x,
      std::shared_ptr<feg::Tensor_attributes>& out_w,
      std::shared_ptr<feg::Tensor_attributes>& out_b,
      std::shared_ptr<feg::Tensor_attributes>& out_y)
  {
      auto g = std::make_shared<feg::Graph>();

      // HALF I/O, FLOAT compute
      g->set_io_data_type(fe::DataType_t::HALF)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
    
      // cuDNN Matmul 要求 3D: {batch, 1, in_features}
      std::vector<int64_t> x_dim = {batch, 1, in_features};
      std::vector<int64_t> x_stride = {in_features, in_features, 1};
    
      out_x = g->tensor(feg::Tensor_attributes()
          .set_name("x")
          .set_dim(x_dim).set_stride(x_stride)
          .set_data_type(fe::DataType_t::HALF));
    
      // W: {1, in_features, out_features}
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
          // 【FFF.md 要求】Bias 永远是 FP32，即使在 AMP 模式下
          std::vector<int64_t> b_dim = {1, 1, out_features};
          std::vector<int64_t> b_stride = {out_features, out_features, 1};
    
          out_b = g->tensor(feg::Tensor_attributes()
              .set_name("bias")
              .set_dim(b_dim).set_stride(b_stride)
              .set_data_type(fe::DataType_t::FLOAT));  // ← 关键：FLOAT 不是 HALF
    
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
  3.3 FC AMP FWD Launch 函数
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
    
      // 从 DTensor 获取 shape（框架保证 C 通道连续）
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
  #ifndef NDEBUG
          {
              cudaStreamCaptureStatus cap_status;
              cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
              if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
                  TR_LOG_ERROR("fc") << "[AMP FWD] Rebuilding cuDNN FE graph inside CUDA Graph capture!";
              }
          }
  #endif
          auto cache = std::make_unique<FcAmpFwdCache>();
          cache->handle = handle;
          cache->batch = batch;
          cache->in_features = in_features;
          cache->out_features = out_features;
          cache->has_bias = p->bias;
          cache->graph = build_fc_amp_fwd_graph(
              handle, batch, in_features, out_features, p->bias,
              cache->x_attr, cache->w_attr, cache->b_attr, cache->y_attr);

          // 【关键】精确按需扩容 workspace
          size_t ws_needed = cache->graph->get_workspace_size();
          if (ws_needed > 0) {
              ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_needed);
          }
    
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
    
      void* ws = ctx.workspace(StreamKind::COMP_1);
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
      cudaEventRecord(state.streams[si].last_done_event, s);
  }
  3.4 FC AMP BWD：dX = dY @ W
  struct FcAmpBwdCache {
      cudnnHandle_t handle = nullptr;
      int64_t batch = -1, in_features = -1, out_features = -1;
      std::shared_ptr<feg::Graph> graph;
      std::shared_ptr<feg::Tensor_attributes> dy_attr, w_attr, dx_attr;
  };

  static std::shared_ptr<feg::Graph> build_fc_amp_bwd_graph(...) {
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
    
      // W: {1, out_features, in_features} — BWD 需要转置后的 W
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
  四、Conv 算子 workspace 修复
  当前 conv_op.cpp 和 op_registry.cpp 中的临时分配需要移除：
  // ===== conv_op.cpp: 移除临时分配 =====
  // 删除：
  //   int64_t ws_bytes = graph->get_workspace_size();
  //   void* temp_ws = nullptr;
  //   if (ws_bytes > 0) cudaMalloc(&temp_ws, ws_bytes);
  //   ...
  //   if (temp_ws) cudaFree(temp_ws);

  // 改为：
  void* ws = ctx.workspace(StreamKind::COMP_1);
  TR_CUDNN_FE_CHECK(graph->execute(handle, vp, ws), "CONV execute");

  // ===== op_registry.cpp: warmup 也改为复用 =====
  // 删除临时 cudaMalloc/cudaFree
  // 改为：
  size_t ws_bytes = graph->get_workspace_size();
  if (ws_bytes > 0) {
      ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_bytes);
  }
  void* ws = ctx.workspace(StreamKind::COMP_1);
  TR_CUDNN_FE_CHECK(graph->execute(handle, vp, ws), "warmup execute");
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  五、方案总结：为什么这个方案最优
   对比维度            小伙伴 S/K（Task 预分配）   小伙伴 D（512MB 初始）            我的方案（精确按需 grow）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   管理主体            Task + DeviceContext        DeviceContext                     ✅ 纯 DeviceContext
   分配次数            2 次（临时 + 正式）         1~N 次（初始 + 可能多次扩容）     ✅ 1 次（首次即精确值）
   显存浪费            无                          可能浪费 512MB - 实际值           ✅ 零浪费
   shape 变化适应性    需重新 compile              自动 grow                         ✅ 自动 grow
   实现复杂度          中（需改 Task + 算子）      低（只改 DeviceContext + 算子）   ✅ 低
   与 relu_op 一致性   高                          高                                ✅ 高
  核心口诀：

  ▌ Warmup 按需 grow，Capture 直接复用，Runtime 零分配，Task 零介入。

# 【小伙伴D】

## 🔬 FC 算子 + Workspace 科学方案
在仔细阅读了 FFF2.md 的全部讨论（导言 + 三位小伙伴的方案 + 用户补充）以及分析全部相关源文件后，以下是综合各方观点形成的最终方案。

### 一、设计原则
原则 来源 说明 workspace 管理全权归 DeviceContext 用户补充 不交给 Task，DeviceContext 已是唯一的所有者 每流一个 workspace FFF.md #11 DeviceContext 内置 workspaces_[5] 天然满足 一次申请，永久复用 FFF.md #11 warmup 确定大小 → pre-capture 分配 → 终生复用 warmup 发现大小，compile 分配 FFF.md #14-16 warmup 后 iterate 所有缓存 → pre_allocate_workspace grow-on-demand 作为兜底 用户补充 如果 iterate 遗漏，warmup 中 realloc 兜底 算子内部绝不 cudaMalloc/Free FFF.md #11 当前 conv_op 违反此规则，需修复 bias 始终 FP32 FFF.md #6 AMP 图中 bias tensor data_type = FLOAT

### 二、Workspace 管理方案：DeviceContext 中心化 2.1 核心思路
```
┌──────────────────────────────────────────────────────────────────┐
│  Phase B2: Warmup（每个 rank 串行）                               │
│  ───────────────────────────                                     │
│  1. launch_cuda 内部 build cuDNN graph → 填充 per-handle cache   │
│  2. cache.graph->get_workspace_size() 已知但不分配                │
│                                                                  │
│  Phase B2.5: Workspace 收集与分配（新增，B2 和 B3 之间）          │
│  ──────────────────────────────────────────────                  │
│  1. collect_max_workspace_from_all_caches() 遍历所有全局 cache    │
│  2. 对所有 rank × 所有 stream：                                    │
│       ctx.pre_allocate_workspace(stream_kind, max_ws)            │
│                                                                  │
│  Phase B3: Capture（每个 rank）                                   │
│  ─────────────────────────                                       │
│  launch_cuda 内部：                                               │
│    workspace = ctx.workspace(COMP_1)    ← 已分配，指针有效         │
│    graph->execute(handle, vp, workspace) ← 永不申请/释放           │
│                                                                  │
│  Runtime: 每个 epoch                                              │
│  ────────────────                                                │
│  cudaGraphLaunch → workspace 自动复用，零开销                      │
└──────────────────────────────────────────────────────────────────┘
``` 2.2 为什么选"遍历 cache 收集"？
小伙伴 K 提出了两种收集方案，我选择 方案 A（遍历 cache） ，原因是：

1. 不侵入 launch_cuda 签名 ： launch_cuda(const GraphNode&, const MemoryPlan&, const DeviceContext&, ...) 中 DeviceContext 是 const 引用，不能调用 report_workspace_needed() （非 const），也不需要改签名
2. 简单可靠 ：warmup 后所有 cache 已填充，遍历即得精确值
3. 对 conv 也生效 ：conv warmup 虽然走专用路径，但其 launch 时也有自己的 cache，遍历可覆盖 2.3 DeviceContext 修改
只需新增一个 grow_workspace 作为 const-safe 的 grow-on-demand 兜底机制：

```
// device_context.h — 新增
void grow_workspace(StreamKind kind, size_t req_size) const;
```
```
// device_context.cpp — 实现
void DeviceContext::grow_workspace(StreamKind kind, size_t req_size) const {
    constexpr size_t alignment = 256;
    size_t aligned = (req_size + alignment - 1) & ~(alignment - 1);
    size_t idx = stream_index(kind);
    auto& ws = const_cast<WSpace&>(workspaces_[idx]);  // const 安全：这是缓存扩容
    if (ws.size >= aligned) return;
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device_id_);
    if (ws.ptr) cudaFree(ws.ptr);
    cudaError_t err = cudaMalloc(&ws.ptr, aligned);
    if (prev >= 0) cudaSetDevice(prev);
    if (err != cudaSuccess) {
        ws.ptr = nullptr; ws.size = 0;
        TR_GPU_OOM("grow_workspace OOM: " << aligned << " bytes");
    }
    ws.size = aligned;
}
``` 2.4 中央收集器
在 op_registry.cpp 中新增：

```
// 每个有 workspace 需求的算子文件导出此函数
extern size_t collect_max_workspace_from_conv_caches();
extern size_t collect_max_workspace_from_fc_caches();
// 未来: tanh, matmul, ...

size_t collect_max_workspace_from_all_caches() {
    size_t max_ws = 0;
    max_ws = std::max(max_ws, collect_max_workspace_from_conv_caches());
    max_ws = std::max(max_ws, collect_max_workspace_from_fc_caches());
    return max_ws;
}
```
在 captured_graph.cpp 的 pre_capture() 中， Phase B2 和 B3 之间 插入：

```
// Phase B2.5: 分配 workspace（预热完成，cache 已填充）
if (is_cuda) {
    size_t max_ws = collect_max_workspace_from_all_caches();
    if (max_ws > 0) {
        for (int rank = 0; rank < num_ranks; ++rank) {
            auto& ctx = *contexts[rank];
            cudaSetDevice(ctx.device_id());
            for (int k = 0; k < 5; ++k) {
                ctx.pre_allocate_workspace(static_cast<StreamKind>(k), max_ws);
            }
        }
    }
}
```
### 三、FC_AMP 算子设计（cuDNN Frontend Matmul） 3.1 为何 AMP 必须用 cuDNN FE Matmul
当前问题 影响 FC_AMP_FWD 复用 launch_fc_fwd_cuda 用 float* 强转 __half* ，读取/写入字节数错误 手写 kernel 无 Tensor Core 加速 FP16 下性能严重低于 cuDNN FC_AMP_BWD cuDNN graph 未实现 一直报 TR_NOT_IMPLEMENTED

结论 ：FC_AMP 必须全新实现 cuDNN FE Matmul graph，FP32 保持手写 kernel 不变。
 3.2 Cache 结构
完全对标 relu_op.cpp / tanh_op.cpp 的成熟模式：

```
struct FcAmpFwdCache {
    cudnnHandle_t handle = nullptr;
    int64_t batch = -1, in_features = -1, out_features = -1;
    bool has_bias = false;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
};

struct FcAmpBwdCache {
    cudnnHandle_t handle = nullptr;
    int64_t batch = -1, in_features = -1, out_features = -1;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> dy_attr, w_attr, dx_attr;
};

static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>> 
s_fc_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>> 
s_fc_amp_bwd_caches;
``` 3.3 Forward Graph: Y = Matmul(X, W) + bias
```
IO 类型:      HALF  (FP16)
Compute 类型: FLOAT (FP32, 利用 Tensor Core)
Bias 类型:    FLOAT (FP32) ← FFF.md 硬性要求

cuDNN 3D 维度布局:
  X: {batch,  1, in_features}     ← cuDNN Matmul 要求 3D
  W: {1, in_features, out_features}
  B: {1, 1, out_features}         data_type=FLOAT
  Y: {batch, 1, out_features}

stride 使用框架 API: dt_x.n_stride_cuda() 等，不自己算对齐
```
关键 ：Bias 在图中 data_type 独立设为 FLOAT ，cuDNN 自动处理 FP32→FP16 转换。这与小伙伴 K 的方案一致。
 3.4 Backward Graph: dX = Matmul(dY, W)
```
dX = dY @ W  (其中 W.shape = [out_features, in_features])

dY: {batch, 1, out_features}
W:  {1, out_features, in_features}  ← W 的转置视图
dX: {batch, 1, in_features}
```
注意 ：BWD 不需要 bias（dX = dY × W，bias 不参与梯度计算）。
 3.5 Launch 函数骨架
```
static void launch_fc_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // 1. 解析参数
    const auto* p = std::get_if<FCParams>(&node.params.data);
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);

    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* w = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    float*  b = (p->bias && node.input_ids.size() > 2) 
                ? static_cast<float*>(ctx.ptr_at(node.input_ids[2])) : nullptr;
    
    // 2. 流注册
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    
    // 3. 获取 per-stream cuDNN handle
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));
    
    // 4. 检查/重建 cache
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
        // #ifndef NDEBUG capture内rebuild检测（同relu/tanh）
        auto cache = std::make_unique<FcAmpFwdCache>();
        cache->handle = handle;
        cache->batch = batch; cache->in_features = in_features;
        cache->out_features = out_features; cache->has_bias = p->bias;
        cache->graph = build_fc_amp_fwd_graph(
            handle, batch, in_features, out_features, p->bias,
            cache->x_attr, cache->w_attr, cache->b_attr, cache->y_attr);
        s_fc_amp_fwd_caches[handle] = std::move(cache);
    }
    
    // 5. 执行：使用 ctx.workspace() 获取预分配 workspace
    auto& cache = *s_fc_amp_fwd_caches[handle];
    std::unordered_map<...> vp = {{cache.x_attr, x}, {cache.w_attr, w}, {cache.
    y_attr, y}};
    if (p->bias && b) vp[cache.b_attr] = b;
    
    void* ws = ctx.workspace(StreamKind::COMP_1);          // 永久复用
    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "FC AMP FWD");
    
    cudaEventRecord(state.streams[si].last_done_event, s);
}
``` 3.6 Workspace 收集（供中央收集器调用）
在 fc_op.cpp 中导出：

```
size_t collect_max_workspace_from_fc_caches() {
    size_t max_ws = 0;
    for (auto& [h, c] : s_fc_amp_fwd_caches)
        max_ws = std::max(max_ws, static_cast<size_t>(c->graph->get_workspace_size
        ()));
    for (auto& [h, c] : s_fc_amp_bwd_caches)
        max_ws = std::max(max_ws, static_cast<size_t>(c->graph->get_workspace_size
        ()));
    return max_ws;
}
```
### 四、Conv 算子 Workspace 修复
当前 conv_op.cpp 在每次 launch 时 cudaMalloc / cudaFree ，违反核心原则。修复非常简单：

改前 ：

```
int64_t ws_bytes = graph->get_workspace_size();
void* temp_ws = nullptr;
if (ws_bytes > 0) cudaMalloc(&temp_ws, ws_bytes);
graph->execute(handle, vp, temp_ws);
if (temp_ws) cudaFree(temp_ws);
```
改后 ：

```
void* ws = ctx.workspace(StreamKind::COMP_1);   // 预分配好的
graph->execute(handle, vp, ws);
```
Workspace 大小已在 Phase B2.5 阶段由中央收集器统一分配，conv launch 时直接使用即可。

### 五、与现有算子的一致性对照
算子 IO 类型 Compute cuDNN 操作 Workspace Cache RELU_AMP HALF FLOAT Pointwise RELU_FWD+CMP_GT nullptr ✅ per-handle TANH_AMP HALF FLOAT Pointwise TANH_FWD nullptr ✅ per-handle FC_AMP (新) HALF FLOAT Matmul + Pointwise ADD ctx.workspace() ✅ per-handle CONV_AMP/FP32 FLOAT/HALF FLOAT Convolution FWD/BWD ctx.workspace() 需添加

FC_AMP 完全遵循 relu_op/tanh_op 的模式：per-handle 缓存 + launch_cuda(const GraphNode&, ...) + #ifndef NDEBUG 检测，唯一的区别是 graph->execute 传入 ctx.workspace() 而非 nullptr 。

### 六、实施步骤
步骤 文件 内容 风险 1 device_context.h/cpp 新增 grow_workspace(kind, size) const 方法 🟢 低 2 fc_op.cpp 实现 build_fc_amp_fwd/bwd_graph() + launch_fc_amp_fwd/bwd_cuda() + cache 🟡 中 3 fc_op.cpp 导出 collect_max_workspace_from_fc_caches() 🟢 低 4 conv_op.cpp 导出 collect_max_workspace_from_conv_caches() + 修复 temp alloc 🟡 中 5 op_registry.cpp require_warmup() 添加 FC_AMP_FWD/BWD + 中央收集器 🟢 低 6 captured_graph.cpp Phase B2.5 插入 workspace 预分配 🟡 中 7 fc_op.cpp 分拆 register_op_fc() ：FP32 指向旧 launch，AMP 指向新 launch 🟢 低 8 编译 + 测试 单 GPU + 多 GPU，FP32 + AMP 🟡 中

### 七、总结
核心公式 ：
 Warmup 发现 → Cache 提供大小 → DeviceContext 预分配 → Launch 永久复用
- FC_FP32 不动（手写 kernel，零依赖）
- FC_AMP 全新 cuDNN FE Matmul（bias FP32，Tensor Core 加速）
- Workspace 由 DeviceContext 全权管理，中央收集器 + pre_allocate_workspace + grow 兜底
- Conv 顺势修复（去掉临时 cudaMalloc/Free）
- 整个方案与 relu_op/tanh_op 的成功范式保持一致，不引入任何新的架构概念
```

# 【用户补充】

你先完全不用管Conv算子，你当它不存在。我们只有FC算子开发完毕之后才会考虑Conv怎么写、怎么改（现在的Conv全方位不符合要求，后期将直接删掉重写）。把注意力放在FC和workspace上。



