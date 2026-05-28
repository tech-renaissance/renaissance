# 【小伙伴K】

• 直接说结论：有 2 个严重 race condition + 3 个中等问题 + 2 个轻微问题。最严重的是 FP32 训练路径和验证路径的并发竞争，会
  导致 ACCUM 读到错误数据。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🔴 严重问题 1：FP32 训练路径 race condition
  位置：src/task/deep_learning_task.cpp，常规 batch 循环末尾（~L1069-1070）
  当前代码：
  if (g_accum) cudaGraphLaunch(g_accum, s_up);
  if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
  问题：当 using_amp == false（FP32 模式）时，g_cm 不执行，sync_up() 也不执行。CPU 立即进入下一个 batch 的循环，启动 FWD
  _A 在 s_c1 上。s_c1 上的 FWD 会覆盖 R_RESULT（loss/top1/top5），而 s_up 上的 ACCUM 可能还在读取 R_RESULT。这是一个真实
  的并发竞争。
  修复：无论 AMP/FP32，循环末尾统一 sync_up()：
  if (g_accum) cudaGraphLaunch(g_accum, s_up);
  if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); }
  sync_up();  // ← 加这一行，确保 ACCUM 完成后再让下一 batch 覆盖 R_RESULT
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🔴 严重问题 2：验证路径 race condition
  位置：src/task/deep_learning_task.cpp，验证 batch 循环内（~L1525）
  当前代码：
  if (g_va) cudaGraphLaunch(g_va, s_up);
  // 没有 sync_up()！
  ts->set_buffer_readable(buf, false);
  ts->set_buffer_writeable(buf, true);
  问题：下一个 batch 开头会立即在 s_c1 上执行 cudaMemsetAsync(loss_id, 0, ..., s_c1) 清零 R_RESULT。s_c1 和 s_up 之间没
  有同步，memset 可能与 ACCUM 并发，导致 ACCUM 读到 0。
  修复：batch 循环内 ACCUM 后加 sync_up()：
  if (g_va) cudaGraphLaunch(g_va, s_up);
  sync_up();  // ← 加这一行
  ts->set_buffer_readable(buf, false);
  ts->set_buffer_writeable(buf, true);
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟡 中等问题 3：GRAD_CONVERT 去掉了 using_amp 保护
  位置：src/task/deep_learning_task.cpp 三处（~L1009, ~L1058, ~L1098）
  当前代码：
  if (g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }
  问题：原代码是 if (using_amp && g_gc)。虽然现在 GRAD_CONVERT graph 只在 amp_on 时编译（FP32 时 g_gc == nullptr），但这
  是隐式依赖。如果未来有人在 FP32 模式下添加 GRAD_CONVERT graph，它会被错误启动。
  修复：恢复 using_amp 条件：
  if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟡 中等问题 4：训练 last batch 无条件使用 g_accum_train_last
  位置：src/task/deep_learning_task.cpp（~L1108-1110）
  当前代码：
  cudaGraphExec_t g_accum_now = g_accum;
  if (g_accum_train_last) g_accum_now = g_accum_train_last;
  if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);
  问题：没有判断 last_bs != bs。只要 g_accum_train_last 存在就用它。虽然整除时 last_train_batch_size == local_batch_size
  ，数值结果一样，但逻辑不严谨。
  修复建议（可选但推荐）：在 run_train_epoch_gpu 开头声明 bs 和 last_bs：
  int32_t bs = registry.get_local_batch_size();
  int32_t last_bs = /* 从 active_memory_plan_->baseline().last_train_batch_size D2H 读取，或重新计算 */;
  然后：
  cudaGraphExec_t g_accum_now = (last_bs != bs) ? g_accum_train_last : g_accum;
  if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟡 中等问题 5：验证 last batch 同样问题
  位置：src/task/deep_learning_task.cpp（~L1522-1525）
  当前代码：
  cudaGraphExec_t g_va = (is_last && g_accum_val_last) ? g_accum_val_last : g_accum;
  问题：同问题 4，用 graph 存在性代替 val_last_bs != val_bs 判断。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟡 中等问题 6：accum_op.cpp CUDA launcher 缺少越界检查
  位置：src/backend/ops/range/accum_op.cpp（~L43-52）
  问题：直接访问 node.input_ids[0..3] 和 node.output_ids[0..2]，没有检查数组长度。如果 compiler 传错参数，会越界访问。
  修复：添加 defensive check（参考 CPU 路径已有）：
  TR_CHECK(node.input_ids.size() == 4, RuntimeError,
           "RANGE_ACCUM_METRICS requires 4 input ids, got " << node.input_ids.size());
  TR_CHECK(node.output_ids.size() == 3, RuntimeError,
           "RANGE_ACCUM_METRICS requires 3 output ids, got " << node.output_ids.size());
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟢 轻微问题 7：epoch 开始时多余的 CAST_MAIN
  位置：src/task/deep_learning_task.cpp（~L991, ~L1055 前）
  问题：epoch 开始和常规循环开始前各有一次 if (using_amp && g_cm) { ... }，加上每个 batch 结束又有一次。第一个 batch 前
  会多做一次 CAST_MAIN（幂等但浪费）。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟢 轻微问题 8：ACCUM_METRICS_VAL_LAST 挂在 train_cg 上
  位置：src/graph/compiler.cpp（~L1723）
  问题：验证用的 ACCUM_METRICS_VAL_LAST 被 append 到 train_cg，虽然 resolve() 能找到，但不符合直觉。建议挂到 infer_cg 上
  （如果验证路径只使用 infer_cg）。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  总结
   优先级   问题                             文件           行号
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   🔴       FP32 训练 race condition         dl_task.cpp    ~1069
   🔴       验证 race condition              dl_task.cpp    ~1525
   🟡       GRAD_CONVERT 去掉 using_amp      dl_task.cpp    ~1009,1058,1098
   🟡       last batch 无条件选 last graph   dl_task.cpp    ~1108,1523
   🟡       accum launcher 缺越界检查        accum_op.cpp   ~43
   🟢       多余 CAST_MAIN                   dl_task.cpp    ~991
   🟢       VAL_LAST 挂 train_cg             compiler.cpp   ~1723

  最紧急的修改：问题 1 和问题 2（加 sync_up()），以及问题 3（恢复 using_amp &&）。这三处改完再编译测试。