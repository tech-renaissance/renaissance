# 【小伙伴K】

• 已全量审查。实现整体与 MYP_FINAL 方案对齐，但发现 4 个必须修复的问题，其中 2 个会导致编译失败 / 运行时崩溃。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🔴 严重问题
  1. H2D 测试代码引用已删除的 gpu_exec_.graphs — 编译错误
    GpuExecTable 已改为 variant_graphs[6]，但 H2D 相关函数仍有 5 处引用旧字段 gpu_exec_.graphs[rank]：
      位置         函数                          当前代码                   应改为
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      L1899        test_h2d_copy_correctness()   gpu_exec_.graphs[r]        gpu_exec_.variant_graphs[0][r]
      L1969-1970   test_h2d_copy_correctness()   gpu_exec_.graphs[0][...]   gpu_exec_.variant_graphs[0][0][...]
      L2047        test_h2d_copy_bandwidth()     gpu_exec_.graphs[r]        gpu_exec_.variant_graphs[0][r]
      L2215        run_h2d_only_train_epoch()    gpu_exec_.graphs[rank]     gpu_exec_.variant_graphs[0][rank]
      L2394        run_h2d_only_val_epoch()      gpu_exec_.graphs[rank]     gpu_exec_.variant_graphs[0][rank]
    修复：全部替换为 gpu_exec_.variant_graphs[0][rank]（H2D 只使用 variant 0 的 XFER 图）。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. build_exec_table() 中 kRequired 校验对 val variant 不合法 — 运行时崩溃
    // L692-709
    static const GraphSlot kRequired[] = {
      GraphSlot::XFER_A, GraphSlot::XFER_B,
      GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
      GraphSlot::FIRST_LAYER_BWD_A, GraphSlot::FIRST_LAYER_BWD_B,
    };
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
      for (auto slot : kRequired) {
          if (!gpu_exec_.variant_graphs[v][rank][slot]) {
              TR_CHECK(false, ...);  // ← val variant (v=4,5) 会触发
          }
      }
    }
    问题：val variant (v=4,5) 的 build_exec_table() 逻辑中没有填充 FWD_BWD_DEEP_A/B 和 FIRST_LAYER_BWD_A/B（这些 slot 保 …
    resize 后的 nullptr）。但 kRequired 校验遍历了 全部 6 个 variant，导致 val variant 必然触发 TR_CHECK(false)。
    修复：将校验限制在 train variant：
    for (size_t v = 0; v < 4; ++v) {  // 只校验 train variant
      for (int rank = 0; rank < K; ++rank) {
          for (auto slot : kRequired) { ... }
      }
    }
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    🟡 逻辑问题
  3. run_val_epoch_gpu() 存在除以零风险
    // L1578, L1584, L1590
    avg_loss = accum_loss / static_cast<float>(registry.num_val_samples());
    avg_top1 = accum_top1 / static_cast<float>(registry.num_val_samples());
    avg_top5 = accum_top5 / static_cast<float>(registry.num_val_samples());
    问题：没有检查 num_val_samples() > 0。MYP_FINAL 方案中明确写了 if (n > 0) 保护。
    修复：统一添加 size_t n = registry.num_val_samples(); if (n > 0) 保护。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. run_train_epoch_cpu() 中 batches == 1 返回路径不一致
    // L1387-1389
    if (loss_id >= 0) {
      Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
      return h_loss.data<float>()[0];  // ← 直接返回 last batch 的 loss
    }
    问题：batches > 1 时返回 accum_loss / num_train_samples()（epoch 平均），但 batches == 1 时返回的是单 batch 的 loss_i…
    值，两者语义不一致。
    修复：统一为从 accum_loss 读取（ACCUM_METRICS 已发射，accum_loss 中已有值）：
    float train_loss = 0.0f;
    if (bl.accum_loss >= 0) {
      Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(bl.accum_loss), 0);
      float val = h.data<float>()[0];
      size_t total = registry.num_train_samples();
      if (total > 0) train_loss = val / static_cast<float>(total);
    }
    return train_loss;
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    总结
   #   问题                              严重度          文件                     行号
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   H2D 代码引用旧 gpu_exec_.graphs   🔴 编译失败     deep_learning_task.cpp   1899, 1969, 2047, 2215, 2394
   2   kRequired 校验包含 val variant    🔴 运行时崩溃   deep_learning_task.cpp   692-709
   3   run_val_epoch_gpu 除以零          🟡 逻辑缺陷     deep_learning_task.cpp   1578, 1584, 1590
   4   batches==1 返回路径不一致         🟡 逻辑缺陷     deep_learning_task.cpp   1387-1389