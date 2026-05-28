# 【小伙伴K】

• 我通读了最新版 OWR_FINAL.md（962 行），结合代码逐条验证。之前指出的 4 个问题已全部修复：
  • ✅ P1.8 last_batch_size 始终初始化（整除时写入 bs）
  • ✅ P2.7 VAL_RESULT_COMM 同时设置 input_ranges + output_ranges
  • ✅ P2.8b build_graph_atlas() 改为全遍历，infer_cg 的 VAL_RESULT_COMM 可被正常 capture
  • ✅ GraphSlot 编号全局一致
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  新发现 1 个问题
  🔴 P2.8 训练路径：循环内 batch == batches-2 误用 train_last 变体
  问题：P2.8 中训练路径的常规 batch 循环（batch = 0 .. batches-2）插入代码：
  bool is_last = (batch == batches - 2);
  cudaGraphExec_t g_accum_now = (is_last && last_bs != bs)
      ? g_accum_train_last : g_accum;
  batch == batches - 2 是循环中的倒数第二个 batch，不是 last batch。last batch 是 batches - 1，在循环之后单独处理（P2.8
  也已给出）。
  这意味着当 last_bs != bs 时，倒数第二个常规 batch 会被错误地用 last_train_batch_size 计算 accum，导致 epoch 总结果偏差
  。
  修正：
  // 循环内（batch = 0 .. batches-2）：全部用常规 g_accum
  if (g_accum) cudaGraphLaunch(g_accum, s_up);

  // 循环后的 last batch（batch = batches-1）：
  cudaGraphExec_t g_accum_now = (last_bs != bs)
      ? g_accum_train_last : g_accum;
  if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  其余部分（P1-P3 全部、数据流说明、风险清单、验证计划、文件清单）均正确，无其他意见。