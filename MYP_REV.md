# 【小伙伴K】

• 我已完整阅读 MYP_FINAL.md 并做了代码对照。方案整体方向正确，但发现 5 个需要修正的问题，其中 2 个会导致编译失败或运行时
  崩溃。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🔴 严重问题

  1. get_current_resolution_train() 函数不存在 — 编译错误
    方案在 3 处使用了 registry.get_current_resolution_train()：
    // MYP_FINAL.md:L404, L761, L961
    int current_res = registry.get_current_resolution_train();
    实际 API 是 registry.current_resolution_train()（无 get_ 前缀）。GlobalRegistry 只定义了：
    // global_registry.h:L758
    int current_resolution_train() const;
    修复：全部改为 registry.current_resolution_train()，或直接使用 DeepLearningTask 已有的成员函数 get_current_train_resol
    ution()（该函数内部已处理了 progressive resolution 逻辑）。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. memory_plan_ptr_ = std::move(variant_memory_plans_[0]) 导致 variant_memory_plans_[0] 变为 nullptr — 运行时崩溃
    方案 on_prepare() 中：
    variant_memory_plans_[0] = std::move(result.variants[0].memory_plan);  // [0] 有值
    // ...
    memory_plan_ptr_ = std::move(variant_memory_plans_[0]);  // [0] 变 nullptr
    然后 build_graph_atlas() 中：
    if (is_shape_invariant_graph(gid)) {
      sl.mp = variant_memory_plans_[0].get();  // ← nullptr!
      sl.shape_id = kShapeInvariant;
    }
    variant_memory_plans_[0] 在 move 给 memory_plan_ptr_ 后已经是 nullptr。capture_all_for_rank() 中 ctx.set_memory_plan(m
    p) 会拿到空指针，触发空指针解引用。
    修复（二选一）：
    方案 A（推荐）：build_graph_atlas() 中改用 active_memory_plan_：
    if (is_shape_invariant_graph(gid)) {
      sl.mp = active_memory_plan_;  // 始终指向 v0 的 MemoryPlan
      sl.shape_id = kShapeInvariant;
    }
    方案 B：on_prepare() 中不 move variant_memory_plans_[0]，而是让 memory_plan_ptr_ 单独持有：
    // 不这样做：
    // memory_plan_ptr_ = std::move(variant_memory_plans_[0]);
    // 改为：
    memory_plan_ptr_ = std::make_unique<MemoryPlan>(*variant_memory_plans_[0]);
    // 但 MemoryPlan 不可拷贝，此方案不可行
    所以 方案 A 是唯一可行的修复。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    🟡 逻辑缺陷
  3. run_train_epoch_cpu() 缺少 ACCUM_METRICS / ACCUM_METRICS_TRAIN_LAST 调用
    方案声称修复了 B3（从 accum_loss 读取 epoch 平均 loss），但 run_train_epoch_cpu() 的代码中：
    // 方案 Step 6 代码中...
    // Normal batch: 没有 launch ACCUM_METRICS
    // Last batch: 没有 launch ACCUM_METRICS_TRAIN_LAST
    后果：CLEAR_METRICS 在 epoch 开头把 accum_loss 清零，但没有任何图去累加它。最后读取出来永远是 0。
    修复：在 CPU 路径的 normal batch 循环中加入 ACCUM_METRICS，last batch 中加入 ACCUM_METRICS_TRAIN_LAST：
    // Normal batch (CPU):
    if (!frozen) launch(from_a ? idx_bwd_a_nb : idx_bwd_b_nb);
    launch(idx_dar);
    // ← 在这里加入：
    int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);
    if (idx_accum_nb >= 0) launch(idx_accum_nb);

  *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batch);
  launch(idx_far);
  launch(idx_opt);

  // Last batch (CPU):
  if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
  launch(idx_dar);
  // ← 在这里加入：
  int32_t idx_accum_lb = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST, v_last);
  if (idx_accum_lb >= 0) launch(idx_accum_lb);
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. run_train_epoch_cpu() 缺少 batch_size 标量写入
    GPU 路径中，ACCUM_METRICS 图的 kernel 通过 GraphNode 的 input_ids 自动绑定 batch_size_ptr（指向 MemoryPlan baseline 中
    的 local_batch_size DTensor）。但 CPU 路径中 graphs[idx].launch(0, nullptr) 直接执行，没有任何机制自动写入标量。
    如果 ACCUM_METRICS kernel 需要读取 batch_size_ptr 来除以 batch size，CPU 路径必须在 launch() 前手动写入：
    // 在 launch ACCUM_METRICS 前：
    const auto& bl = active_memory_plan_->baseline();
    if (bl.local_batch_size >= 0) {
      *static_cast<int32_t*>(ctx.ptr_at(bl.local_batch_size)) =
          registry.get_local_batch_size();  // normal batch
    }
    // last batch 同理，写入 last_train_batch_size
    如果不写入：kernel 读到的是初始值 0 或垃圾值，除法结果会是 NaN 或 Inf。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    🟢 文档不准确
  5. 4.2 节中 OPTIMIZER 被错误归类为"训练 Shape 相关"
    // computation_graph.h:L191
    // case GraphId::OPTIMIZER:  // 移除：LARS 下节点数依赖层数
    is_train_graph(OPTIMIZER) 返回 false，is_shape_invariant_graph(OPTIMIZER) 也返回 false。
    这意味着：
    • build_graph_atlas() 中交叉隔离条件不会跳过 OPTIMIZER（因为 is_train_graph 为 false）
    • val variant 也会填入 OPTIMIZER 的 slot（使用 train_cg_ + val variant 的 mp/shape_id）
    • 由于 Key{train_cg, OPTIMIZER, val_shape} 的 shape_id 与 train variant 不同，会多 capture 几张 OPTIMIZER 图
    这不是运行时错误（run_val_epoch_gpu() 不会解析 OPTIMIZER handle），但文档归类不准确。建议改为：
    ▌ 训练专属非 Shape 相关：OPTIMIZER — is_train_graph() 返回 false 但 is_shape_invariant_graph() 也返回 false。由于各
    ▌ variant 使用不同 shape_id，会被多次捕获，但拓扑相同。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    总结
      问题                                严重程度        位置                           修复
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      get_current_resolution_train() 不   🔴 编译错误     L404, L761, L961               改为 registry.current_resolution_
      存在                                                                               train()
      variant_memory_plans_[0] 变 nullp   🔴 运行时崩溃   build_graph_atlas() L293       改为 sl.mp = active_memory_plan_
      tr
      CPU 路径缺少 ACCUM_METRICS          🟡 逻辑错误     run_train_epoch_cpu() Step 6   加入 launch(idx_accum_nb) 和 laun
                                                                                      ch(idx_accum_lb)
      CPU 路径缺少标量写入                🟡 逻辑错误     run_train_epoch_cpu()          launch 前写入 local_batch_size /
                                                                                      last_train_batch_size
      OPTIMIZER 归类错误                  🟢 文档         4.2 节                         修正描述

  除去这 5 个问题，方案的整体架构、6-Variant 定义、交叉槽位隔离、GpuExecTable 扩展、pre-resolve 模式都是正确的，与代码基
  础设施完全兼容。