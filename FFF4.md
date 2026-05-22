# 【今日话题：讨论FC层的设计及workspace的申请】



# 【小伙伴S】

  🎯 FC算子与Workspace管理的最终科学方案

  一、核心洞察：现有架构的优越性

  通过分析DeviceContext的源码，我发现现有基础设施已经非常完善：

  1. 完美的workspace隔离：workspaces_[5]数组天然实现每流一个workspace
  2. 256字节对齐：pre_allocate_workspace已实现CUDA要求的对齐
  3. 幂等性保护：已分配的workspace会被保护，避免重复分配
  4. Per-stream cuDNN handles：cudnn_handles_[5]确保多流安全

  二、最优方案：精确按需扩容 + 中央收集器

  我综合了小伙伴S、K、D方案的优点，提出以下最优方案：

  2.1 方案选择逻辑

  ┌─────────────────────────────────────────────────────────────┐
  │  方案对比选择                                                │
  ├─────────────────────────────────────────────────────────────┤
  │  小伙伴S/K：Task维护max_workspace + compile预分配           │
  │    ❌ 破坏了DeviceContext的内聚性                            │
  │    ❌ 需要修改TaskBase，增加耦合度                           │
  │                                                              │
  │  小伙伴D：初始512MB + grow-on-demand                        │
  │    ✅ DeviceContext全权管理                                  │
  │    ❌ 512MB可能浪费或不够                                    │
  │                                                              │
  │  我的最优方案：精确按需 + 中央收集器                         │
  │    ✅ DeviceContext全权管理                                  │
  │    ✅ 精确分配，零浪费                                       │
  │    ✅ warmup自然探测大小                                     │
  │    ✅ 与relu_op/tanh_op完全一致                              │
  └─────────────────────────────────────────────────────────────┘

  2.2 核心设计：两阶段管理

  // 阶段1：warmup阶段 - 精确按需扩容
  // 在fc_op.cpp的launch函数中
  if (rebuild) {
      cache->graph = build_fc_amp_fwd_graph(...);

      // 关键：精确按需扩容，不浪费1字节显存
      size_t ws_needed = cache.graph->get_workspace_size();
      if (ws_needed > 0) {
          ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_needed);
      }
  }

  // 阶段2：capture/runtime阶段 - 直接复用
  void* ws = ctx.workspace(StreamKind::COMP_1);
  cache.graph->execute(handle, vp, ws);

  三、DeviceContext增强：ensure_workspace_grow方法

  分析现有代码发现，ensure_workspace在不够大时会抛异常，而不是扩容。我们需要增加智能扩容方法：

  // device_context.h 新增
  class DeviceContext {
  public:
      /**
       * @brief 确保workspace足够大，不够时自动释放旧的并申请新的
       * @note 与ensure_workspace不同：本方法支持自动扩容，而不是抛异常
       * @note 用于warmup阶段按需探测最终workspace大小
       */
      void ensure_workspace_grow(StreamKind kind, size_t req_size);
  };

  // device_context.cpp 实现
  void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) {
  #ifdef TR_USE_CUDA
      if (device_id_ < 0) return;  // CPU模式无workspace

      size_t idx = stream_index(kind);
      auto& ws = workspaces_[idx];
    
      // 256字节对齐
      constexpr size_t alignment = 256;
      size_t aligned_size = (req_size + alignment - 1) & ~(alignment - 1);
    
      // 已经足够大，直接返回
      if (ws.size >= aligned_size) {
          return;
      }
    
      // 需要扩容：释放旧的，申请新的
      if (ws.ptr) {
          cudaFree(ws.ptr);
          ws.ptr = nullptr;
          ws.size = 0;
      }
    
      int prev_device = -1;
      cudaGetDevice(&prev_device);
      cudaSetDevice(device_id_);
    
      cudaError_t err = cudaMalloc(&ws.ptr, aligned_size);
      cudaSetDevice(prev_device);
    
      if (err != cudaSuccess) {
          ws.ptr = nullptr;
          ws.size = 0;
          TR_GPU_OOM("Failed to grow workspace to " << aligned_size
                     << " bytes for stream " << static_cast<int>(kind)
                     << " on device " << device_id_
                     << ": " << cudaGetErrorString(err));
      }
    
      ws.size = aligned_size;
    
      TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                             << ": workspace for stream " << static_cast<int>(kind)
                             << " grown to " << aligned_size << " bytes";
  #else
      (void)kind;
      (void)req_size;
  #endif
  }

  四、FC_AMP算子完整实现

  4.1 Cache结构（完全对标relu_op）

  #ifdef TR_USE_CUDA
  namespace feg = cudnn_frontend::graph;

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

  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>>
  s_fc_amp_fwd_caches;
  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>>
  s_fc_amp_bwd_caches;
  #endif

  4.2 FC_AMP_FWD Graph构建

  static std::shared_ptr<feg::Graph> build_fc_amp_fwd_graph(
      cudnnHandle_t handle,
      int64_t batch, int64_t in_features, int64_t out_features, bool has_bias,
      std::shared_ptr<feg::Tensor_attributes>& out_x,
      std::shared_ptr<feg::Tensor_attributes>& out_w,
      std::shared_ptr<feg::Tensor_attributes>& out_b,
      std::shared_ptr<feg::Tensor_attributes>& out_y)
  {
      auto g = std::make_shared<feg::Graph>();

      // IO: HALF, Compute: FLOAT (利用Tensor Core)
      g->set_io_data_type(fe::DataType_t::HALF)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
    
      // cuDNN Matmul要求3D: {batch, 1, in_features}
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
          // 【关键】Bias永远FP32，即使AMP模式
          std::vector<int64_t> b_dim = {1, 1, out_features};
          std::vector<int64_t> b_stride = {out_features, out_features, 1};
    
          out_b = g->tensor(feg::Tensor_attributes()
              .set_name("bias")
              .set_dim(b_dim).set_stride(b_stride)
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

  4.3 FC_AMP_FWD Launch函数（关键部分）

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
      int si = state.get_or_register(s);
      state.output_stream_idx = si;
      state.streams[si].has_pending_work = true;
    
      cudnnHandle_t handle = static_cast<cudnnHandle_t>(
          ctx.cudnn_handle(StreamKind::COMP_1));
    
      // 从DTensor获取shape（框架保证C通道连续）
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
    
          // 【关键】精确按需扩容workspace
          size_t ws_needed = cache->graph->get_workspace_size();
          if (ws_needed > 0) {
              const_cast<DeviceContext&>(ctx).ensure_workspace_grow(
                  StreamKind::COMP_1, ws_needed);
          }
    
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
    
      // 【关键】从DeviceContext获取永久workspace
      void* ws = ctx.workspace(StreamKind::COMP_1);
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
    
      cudaEventRecord(state.streams[si].last_done_event, s);
  }

  五、方案优势总结

  5.1 与其他方案的对比

  ┌─────────────────┬──────────────────────┬───────────────┬────────────────────┐
  │      维度       │      小伙伴S/K       │    小伙伴D    │    我的最优方案    │
  ├─────────────────┼──────────────────────┼───────────────┼────────────────────┤
  │ 管理主体        │ Task + DeviceContext │ DeviceContext │ ✅ 纯DeviceContext │
  ├─────────────────┼──────────────────────┼───────────────┼────────────────────┤
  │ 显存浪费        │ 无                   │ 可能浪费512MB │ ✅ 零浪费          │
  ├─────────────────┼──────────────────────┼───────────────┼────────────────────┤
  │ 分配次数        │ 2次                  │ 1~N次         │ ✅ 1次（精确值）   │
  ├─────────────────┼──────────────────────┼───────────────┼────────────────────┤
  │ 实现复杂度      │ 中                   │ 低            │ ✅ 低              │
  ├─────────────────┼──────────────────────┼───────────────┼────────────────────┤
  │ 与relu_op一致性 │ 高                   │ 高            │ ✅ 高              │
  └─────────────────┴──────────────────────┴───────────────┴────────────────────┘

  5.2 核心优势

  1. 精确按需分配：首次需要多少就分配多少，不浪费1字节显存
  2. DeviceContext全权管理：Task完全不介入，保持内聚性
  3. 自动扩容机制：ensure_workspace_grow智能处理大小变化
  4. 零额外开销：warmup结束后workspace就是最终大小，无需重新分配
  5. 完美兼容现有架构：与relu_op/tanh_op的成功模式完全一致

  六、实施建议

  1. 优先级1：在DeviceContext中实现ensure_workspace_grow方法
  2. 优先级2：实现FC_AMP_FWD的cuDNN FE Graph和launch函数
  3. 优先级3：实现FC_AMP_BWD的对应代码
  4. 优先级4：在op_registry.cpp中注册FC_AMP算子并添加warmup需求

  这个方案综合了所有讨论的精华，选择最科学合理的路径，既保证了架构的清晰性，又实现了资源的精确管理。



# 【小伙伴K】

• 我已完整阅读 FFF3.md（含全部四位小伙伴的方案、我的上一轮方案、以及最终用户补充）。用户补充了一条关键约束：
  ▌ "完全不用管 Conv 算子，当它不存在。把注意力放在 FC 和 workspace 上。"
  这意味着方案必须极度聚焦：只解决 FC 算子实现 + workspace 管理，不引入任何与 Conv 相关的改动或"中央收集器"等复杂机制。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  我的最终方案：DeviceContext 全权管理 + 算子级 Lazy Grow
  一、核心设计决策
  基于 FFF3.md 全部讨论和用户补充，我做如下判断：
   争议点               我的判断                       理由
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   管理主体             DeviceContext 全权负责         用户补充明确要求"全权交给 DeviceContext 而非 Task"
   Task 是否介入        零介入                         同上
   是否需要中央收集器   不需要                         中央收集器是为 Conv/FC/多算子统一收集设计的；现在只有 FC，集中式
                                                       管理反而增加 compile 流程复杂度
   初始 Buffer 策略     不需要 512MB 初始值            用户说"初始申请+动态调整是兜底办法，希望有更好的"；FC 的 Matmul
                                                       workspace 通常几十 MB 以内，512MB 浪费
   分配时机             每个算子 rebuild 时按需 grow   最简洁：warmup 中 cache 重建 → 发现 ws 大小 → 立即 grow → 后续 c
                                                       apture/runtime 直接复用
   扩容策略             不够时释放旧内存、申请新内存   用户明确说"不够大的情况就 realloc"
  二、方案架构图
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Warmup 阶段（每个 rank 串行）                                           │
  │  ─────────────────────────────                                          │
  │  launch_fc_amp_fwd_cuda() 首次调用                                       │
  │    └→ cache miss → rebuild                                              │
  │        └→ build_fc_amp_fwd_graph() → graph->get_workspace_size()        │
  │        └→ ctx.ensure_workspace_grow(COMP_1, ws_needed)  ← 按需分配      │
  │        └→ s_fc_amp_fwd_caches[handle] = cache                           │
  │    └→ cache hit → ctx.workspace(COMP_1) → execute                       │
  │                                                                         │
  │  (同一 stream 上后续算子如 FC_BWD、或其他层若需要更大 ws)                 │
  │    └→ ensure_workspace_grow() 自动检测、自动扩容                        │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Capture 阶段                                                           │
  │  ─────────────                                                          │
  │  launch_fc_amp_fwd_cuda() 被 capture 调用                                │
  │    └→ cache hit → ctx.workspace(COMP_1) 指针已就绪                      │
  │    └→ graph->execute(handle, vp, ws) → 记录到 CUDA Graph                 │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Runtime 阶段（每个 epoch）                                              │
  │  ────────────────────────                                               │
  │  cudaGraphLaunch() → workspace 指针在 Graph 中固定不变，零开销复用       │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  程序结束                                                               │
  │  ────────                                                               │
  │  DeviceContext 析构 → free_workspaces() → cudaFree                      │
  └─────────────────────────────────────────────────────────────────────────┘
  三、唯一需要新增的基础设施
  只改 device_context.h/cpp，新增一个方法：
  // device_context.h
  class DeviceContext {
  public:
      // ... 现有 API 保持不变 ...

      /**
       * @brief 确保 workspace 足够大，不够时自动释放旧内存、申请新内存
       * @note 幂等：如果当前已足够大，什么都不做
       * @note 这是算子在 warmup/capture/runtime 各阶段获取足够 workspace 的唯一入口
       * @note 与 pre_allocate_workspace 的区别：后者是 compile 阶段的一次性预分配，
       *       而 ensure_workspace_grow 是在算子 launch 时按需扩容
       */
      void ensure_workspace_grow(StreamKind kind, size_t req_size);
  };
  // device_context.cpp
  void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) {
  #ifdef TR_USE_CUDA
      if (device_id_ < 0) return;  // CPU 模式无 workspace

      size_t idx = stream_index(kind);
      auto& ws = workspaces_[idx];
    
      // 256B 对齐（CUDA 分配要求）
      size_t aligned = ((req_size + 255) / 256) * 256;
    
      // 已经足够大，什么都不做（热路径，fast path）
      if (ws.size >= aligned) {
          return;
      }
    
      // 需要扩容：释放旧的
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
    
      if (prev_device >= 0) cudaSetDevice(prev_device);
    
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
  为什么这比 pre_allocate_workspace 更适合？
  • pre_allocate_workspace 是 idempotent 的（已分配则跳过），不支持扩容
  • ensure_workspace_grow 是 自动扩容 的，不够大就释放旧、申请新
  • 两者共存：pre_allocate_workspace 留给未来 compile 阶段显式预分配使用；ensure_workspace_grow 给算子 launch 时按需使用
  四、FC AMP FWD 算子实现
  完全参照 relu_op.cpp 的成熟模式：per-handle cache + launch_cuda(const GraphNode&, const MemoryPlan&, const DeviceConte
  xt&, MultiStreamCaptureState&)。
  // fc_op.cpp 中新增

  #ifdef TR_USE_CUDA
  namespace feg = cudnn_frontend::graph;

  // ===== Cache 结构 =====
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

  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>> s_fc_amp_fwd_caches;
  static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>> s_fc_amp_bwd_caches;

  // ===== Graph 构建：Y = Matmul(X, W) + bias =====
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

      // HALF I/O, FLOAT compute（Tensor Core 加速）
      g->set_io_data_type(fe::DataType_t::HALF)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
    
      // cuDNN Matmul 要求 3D: {batch, 1, in_features}
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
          // 【FFF.md 硬性要求】Bias 永远是 FP32，即使在 AMP 模式下
          out_b = g->tensor(feg::Tensor_attributes()
              .set_name("bias")
              .set_dim({1, 1, out_features})
              .set_stride({out_features, out_features, 1})
              .set_data_type(fe::DataType_t::FLOAT));
    
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

  // ===== Launch 函数 =====
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

          // 【关键】按需扩容 workspace
          size_t ws_needed = cache.graph->get_workspace_size();
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
    
      // 【关键】从 DeviceContext 获取永久 workspace
      void* ws = ctx.workspace(StreamKind::COMP_1);
      TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
      cudaEventRecord(state.streams[si].last_done_event, s);
  }
  #endif // TR_USE_CUDA
  五、FC AMP BWD 算子
  // dX = Matmul(dY, W^T)
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
    
      // W: {1, out_features, in_features} — BWD 需要 W 的转置视图
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
    
      TR_CUDNN_FE_CHECK(g->validate(), "validate FC AMP BWD");
      TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build FC BWD op graph");
      TR_CUDNN_FE_CHECK(g->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}),
                     "create FC BWD plans");
      TR_CUDNN_FE_CHECK(g->check_support(handle), "check FC BWD support");
      TR_CUDNN_FE_CHECK(g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE),
                     "build FC BWD plans");
    
      return g;
  }
  六、注册与 warmup
  // op_registry.cpp 中 require_warmup() 添加：
  case ComputeOp::FC_AMP_FWD:
  case ComputeOp::FC_AMP_BWD:
      return true;

  // fc_op.cpp 中 register_op_fc()：
  void register_op_fc() {
      // FC_FP32_FWD/BWD: 保持现有手写 kernel（不变）

      // FC_AMP_FWD: 仅 CUDA
      {
          auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_FWD)];
          e.op = ComputeOp::FC_AMP_FWD;
          e.launch_cpu = launch_fc_amp_cpu_not_supported;  // TR_TYPE_ERROR
  #ifdef TR_USE_CUDA
          e.launch_cuda = launch_fc_amp_fwd_cuda;
  #endif
      }

      // FC_AMP_BWD: 仅 CUDA
      {
          auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_BWD)];
          e.op = ComputeOp::FC_AMP_BWD;
          e.launch_cpu = launch_fc_amp_cpu_not_supported;
  #ifdef TR_USE_CUDA
          e.launch_cuda = launch_fc_amp_bwd_cuda;
  #endif
      }
  }
  七、方案对比：为什么这个最优
                                               小伙伴 K（Task 预分配                             我的方案（算子级 Lazy
   维度               小伙伴 S（512MB 初始）   ）                       小伙伴 D（中央收集器）   Grow）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   管理主体           DeviceContext            Task + DeviceContext     DeviceContext            ✅ 纯 DeviceContext
   Task 介入          否                       是（维护 max_ws）        否                       ✅ 零介入
   中央收集器         否                       否                       是（遍历所有 cache）     ✅ 不需要
   初始浪费           512MB 可能闲置           无                       无                       ✅ 零浪费
   分配次数           1~N 次                   2 次（临时+正式）        1 次                     ✅ 1 次（首次即精确）
   compile 流程改动   无                       需插入预分配阶段         需插入 B2.5 收集阶段     ✅ 零改动
   代码侵入性         低                       中（改 TaskBase）        中（改 compile 流程）    ✅ 最低
  八、实施步骤（极简）
   步骤   文件                   内容
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1      device_context.h/cpp   新增 ensure_workspace_grow(kind, size)
   2      fc_op.cpp              实现 build_fc_amp_fwd/bwd_graph + launch_fc_amp_fwd/bwd_cuda + cache
   3      fc_op.cpp              新增 launch_fc_amp_cpu_not_supported
   4      op_registry.cpp        require_warmup() 添加 FC_AMP_FWD/BWD
   5      op_kind.cpp/h          添加 FC 算子枚举（如尚未添加）
   6      测试                   编译 + 运行 FC FP32 + AMP 测试
  九、核心口诀

  ▌ DeviceContext 全权管，算子自己按需 grow，Task 完全不介入，Compile 流程零改动。



# 【小伙伴D】

## 🔬 FC 算子 + Workspace 最终方案
### 一、三轮讨论的进化与收敛
议题 FFF.md（导言） FFF2.md（第一轮） FFF3.md（收敛） workspace 所有者 Task 或 DeviceContext Task（S/K） vs DC（D） → DeviceContext 分配策略 预热临时→compile正式 512MB初始 / Task预分配 → ensure_workspace_grow 精确按需 512MB 盲猜 FFF.md #15 作为兜底 D 仍采纳 → S/K/D 一致否定，精确按需更优 Conv 算子 要改 D 方案包含其修复 → 用户：完全不管 Conv，专注 FC

FFF3.md 中小伙伴 K 的方案已经非常接近最优解 —— ensure_workspace_grow + DeviceContext 全权管理 + 零 Task 介入。但有一个技术细节需要修正： launch_cuda 的签名是 const DeviceContext& ，直接调用非 const 的 ensure_workspace_grow 会编译失败。下面给出修正后的完整方案。

### 二、核心设计
```
┌─────────────────────────────────────────────────────────────────┐
│  原则 1: DeviceContext 是 workspace 唯一所有者，Task 零介入       │
│  原则 2: workspaces_[5] 天然实现每流一个                         │
│  原则 3: 首次使用按需精确分配（cuDNN 报告多大就分配多大）          │
│  原则 4: 不够→释放旧的→申请新的，warmup 结束即得精确值            │
│  原则 5: Capture 和 Runtime 直接复用，零额外分配                  │
│  原则 6: 算子内部绝不 cudaMalloc/Free，统一 ctx.workspace()      │
└─────────────────────────────────────────────────────────────────┘
```
```
时间线：
┌──────────────────────────────────────────────────────────────────┐
│ Phase B2: Warmup（每个 rank 串行）                                │
│                                                                    │
│   warmup_single_cudnn_op() → entry.launch_cuda()                 │
│     └→ rebuild=true → build cuDNN graph                          │
│        └→ cache.graph->get_workspace_size()                      │
│        └→ ctx.ensure_workspace_grow(COMP_1, ws_needed)  ← 精确！ │
│        └→ graph->execute(handle, vp, ws)                          │
│     └→ cudaStreamSynchronize → 完成预热                           │
│                                                                    │
│ Phase B3: Capture（每个 rank）                                    │
│                                                                    │
│   entry.launch_cuda() 在 cudaStreamBeginCapture 内                │
│     └→ cache hit（rebuild=false）→ 不触发 ensure_workspace_grow  │
│     └→ ws = ctx.workspace(COMP_1)  ← 已在 warmup 中确定大小      │
│     └→ graph->execute(handle, vp, ws)  ← 纯 cache hit            │
│                                                                    │
│ Runtime: cudaGraphLaunch → workspace 自动复用，零开销              │
└──────────────────────────────────────────────────────────────────┘
```
关键洞察 ： ensure_workspace_grow 只在 rebuild=true 时调用。warmup 阶段触发 rebuild，capture 阶段 cache hit，自然形成"warmup 确定大小 → capture 复用"的完美节奏。无需中央收集器，无需 captured_graph.cpp 改动，无需 Phase B2.5。

### 三、DeviceContext 新增方法
技术要点： launch_cuda 签名是 const DeviceContext& ，所以 ensure_workspace_grow 必须声明为 const 方法 。内部通过 const_cast 修改 workspace（这属于逻辑 const——workspace 扩容不影响对象语义）。

```
// device_context.h — 新增
void ensure_workspace_grow(StreamKind kind, size_t req_size) const;
```
```
// device_context.cpp — 实现
void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) const {
#ifdef TR_USE_CUDA
    if (device_id_ < 0) return;

    constexpr size_t alignment = 256;
    size_t aligned = ((req_size + alignment - 1) / alignment) * alignment;
    size_t idx = stream_index(kind);
    auto& ws = const_cast<WSpace&>(workspaces_[idx]);  // 逻辑const:缓存扩容

    if (ws.size >= aligned) return;  // 已经足够大

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
                   << " bytes for stream " << static_cast<int>(kind)
                   << " on device " << device_id_
                   << ": " << cudaGetErrorString(err));
    }

    ws.size = aligned;
    TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                           << ": workspace[stream " << static_cast<int>(kind)
                           << "] grown to " << aligned << " bytes";
#else
    (void)kind;
    (void)req_size;
#endif
}
```
### 四、FC_AMP 算子实现 4.1 架构：FC_FP32 不动，FC_AMP 全新 cuDNN FE Matmul
```
FC_FP32_FWD/BWD → fc_op.cu 手写 kernel → workspace: 不需要 → 保持不变
FC_AMP_FWD/BWD → fc_op.cpp cuDNN FE Matmul → workspace: ctx.workspace() → 全新实现
``` 4.2 Per-handle Cache（完全对标 relu_op/tanh_op）
```
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

static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>> 
s_fc_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>> 
s_fc_amp_bwd_caches;
``` 4.3 Forward Graph: Y = Matmul(X, W) + bias
```
IO: HALF, Compute: FLOAT, Bias: FLOAT (永远 FP32)

cuDNN 3D 布局:
  X: {batch,  1, in_features}     stride: {x_n_stride, in_features, 1}
  W: {1, in_features, out_features} stride: {in_features*w_w_stride, w_w_stride, 1}
  B: {1, 1, out_features}         data_type = FLOAT  ← 关键
  Y: {batch, 1, out_features}     stride: {y_n_stride, out_features, 1}
```
Stride 来源：框架 API dt_x.n_stride_cuda() , dt_w.w_stride_cuda() , dt_y.n_stride_cuda() ，废除旧版 get_row_stride() 。
 4.4 Backward Graph: dX = Matmul(dY, W)
```
dX = dY @ W  (W.shape = [out_features, in_features])

dY: {batch, 1, out_features}
W:  {1, out_features, in_features}  ← 转置视图
dX: {batch, 1, in_features}

BWD 不需要 bias
``` 4.5 Launch 函数（核心）
```
static void launch_fc_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_AMP_FWD missing FCParams");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    
    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* w = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    float*  b = (p->bias && node.input_ids.size() > 2)
                ? static_cast<float*>(ctx.ptr_at(node.input_ids[2])) : nullptr;
    
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));
    
    int64_t batch        = dt_x.n();
    int64_t in_features  = dt_x.h() * dt_x.w() * dt_x.c();
    int64_t out_features = p->out_features;
    int64_t x_ns = dt_x.n_stride_cuda();
    int64_t w_ws = dt_w.w_stride_cuda();
    int64_t y_ns = dt_y.n_stride_cuda();
    
    auto it = s_fc_amp_fwd_caches.find(handle);
    bool rebuild = (it == s_fc_amp_fwd_caches.end())
        || it->second->batch != batch
        || it->second->in_features != in_features
        || it->second->out_features != out_features
        || it->second->x_n_stride != x_ns
        || it->second->w_w_stride != w_ws
        || it->second->y_n_stride != y_ns
        || it->second->has_bias != p->bias;
    
    if (rebuild) {
#ifndef NDEBUG
        {
            cudaStreamCaptureStatus cap_status;
            cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
            if (cap_err == cudaSuccess && cap_status != 
            cudaStreamCaptureStatusNone) {
                TR_LOG_ERROR("fc") << "[AMP FWD] Rebuilding inside capture! "
                                   << "handle=" << handle;
            }
        }
#endif
        auto cache = std::make_unique<FcAmpFwdCache>();
        cache->handle = handle;
        cache->batch = batch; cache->in_features = in_features;
        cache->out_features = out_features; cache->has_bias = p->bias;
        cache->x_n_stride = x_ns; cache->w_w_stride = w_ws; cache->y_n_stride = 
        y_ns;
        cache->graph = build_fc_amp_fwd_graph(
            handle, batch, in_features, out_features,
            x_ns, w_ws, y_ns, p->bias,
            cache->x_attr, cache->w_attr, cache->b_attr, cache->y_attr);

        // ★ 精确按需扩容 workspace
        size_t ws_needed = cache->graph->get_workspace_size();
        if (ws_needed > 0) {
            ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_needed);
        }
    
        s_fc_amp_fwd_caches[handle] = std::move(cache);
    }
    
    auto& cache = *s_fc_amp_fwd_caches[handle];
    std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
        {cache.x_attr, x}, {cache.w_attr, w}, {cache.y_attr, y}
    };
    if (p->bias && b) vp[cache.b_attr] = b;
    
    void* ws = ctx.workspace(StreamKind::COMP_1);  // 永久复用
    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "FC AMP FWD");
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```
关键点 ：

- ensure_workspace_grow 是 const 方法 ，在 const DeviceContext& 上可调用
- 只在 rebuild=true （warmup）时调用，capture 时 cache hit 跳过
- workspace 指针从 ctx.workspace() 获取，终身有效
- stride 从框架 n_stride_cuda() / w_stride_cuda() 获取 4.6 BWD 同理
```
static void launch_fc_amp_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // ...参数解析、指针获取、流注册...
    // ...cache 检查与 rebuild（含 ensure_workspace_grow）...
    // ...vp 构建...
    void* ws = ctx.workspace(StreamKind::COMP_1);
    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "FC AMP BWD");
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```
### 五、集成改动 5.1 require_warmup() 添加 FC_AMP
op_registry.cpp 的 require_warmup() ：

```
case ComputeOp::FC_AMP_FWD:
case ComputeOp::FC_AMP_BWD:
    return true;
``` 5.2 注册分拆
fc_op.cpp register_op_fc() ：

```
void register_op_fc() {
    // FC_FP32_FWD/BWD → 复用现有的 launch_fc_fwd_cuda / launch_fc_bwd_cuda
    { ... entry.launch_cuda = launch_fc_fwd_cuda; ... }
    { ... entry.launch_cuda = launch_fc_bwd_cuda; ... }

    // FC_AMP_FWD/BWD → 指向新的 cuDNN FE launch 函数
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_FWD)];
        e.op = ComputeOp::FC_AMP_FWD;
        e.launch_cpu = launch_fc_amp_fwd_cpu_not_supported;  // CPU 不支持 FP16
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_fc_amp_fwd_cuda;
#endif
    }
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_BWD)];
        e.op = ComputeOp::FC_AMP_BWD;
        e.launch_cpu = launch_fc_amp_bwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_fc_amp_bwd_cuda;
#endif
    }
}
``` 5.3 Warmup 路径：零改动
现有的 warmup_single_cudnn_op() 对非 Conv 算子走通用 entry.launch_cuda() 路径。FC_AMP 加入 require_warmup() 后自动被预热，无需任何特殊处理。其 launch_cuda 内部会：

1. 首次调用 → rebuild → build cuDNN graph → ensure_workspace_grow → execute → cudaStreamSynchronize
2. 后续调用（其他 rank/其他 graph） → cache hit → 直接 execute 5.4 不需要改的文件
- captured_graph.cpp — 不需要 Phase B2.5 插入
- task_base.cpp — Task 零介入
- op_registry.cpp — 不需要中央收集器
- capture_cuda.cpp — 不需要改动
- fc_op.cu — FP32 kernel 不变
### 六、改单清单
步骤 文件 内容 1 device_context.h 新增 ensure_workspace_grow(kind, size) const 声明 2 device_context.cpp 实现 ensure_workspace_grow （256B 对齐，释放旧→申请新） 3 fc_op.cpp 新增 build_fc_amp_fwd_graph() + build_fc_amp_bwd_graph() 4 fc_op.cpp 新增 launch_fc_amp_fwd_cuda() + launch_fc_amp_bwd_cuda() 5 fc_op.cpp 新增 cache 结构 + static cache map 6 fc_op.cpp 分拆 register_op_fc() （AMP→新 launch） 7 op_registry.cpp require_warmup() 添加 FC_AMP_FWD/BWD 8 编译 + 测试 FC FP32 + FC AMP

### 七、总结
核心公式 ：
 Warmup 触发 rebuild → ensure_workspace_grow 精确扩容 → Capture 命中 cache → workspace 永久复用
这个方案的优越性：

对比 Task 预分配（FFF2 K） 512MB 初始（FFF2 D） ensure_workspace_grow（最终） 管理主体 Task + DC DC ✅ 纯 DC 分配次数 2次 1~N次 ✅ 1次 显存浪费 无 512MB-实际 ✅ 零 架构侵入 captured_graph + Task DeviceContext ✅ 仅 DeviceContext 未来扩展 新算子需注册收集器 自动 ✅ 自动

与 relu_op 的对标 ：FC_AMP 的代码结构（cache 结构、rebuild 逻辑、 #ifndef NDEBUG 检测、launch_cuda 注册）与 relu_op/tanh_op 完全一致，唯一差异是 execute 传 ctx.workspace() 而非 nullptr 。
```





