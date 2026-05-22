# 【小伙伴K】

• 经重新通读 Z_FINAL_K.md 全文并与当前源码逐行交叉验证，发现 3 处需要最小侵入修改的问题。整体方案已经高度成熟，核心架构（active_memor
  y_plan_、Atlas 直解、无状态 LR、4-phase 重叠）全部正确。
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🔴 P0：build_graph_atlas() const 编译错误
  位置：§3.2 build_graph_atlas() 实现
  问题：方法签名带 const，但函数体内调用 name_to_gid_.clear()。name_to_gid_ 是基类 TaskBase 的 protected 成员，非 mutable，在 const
  方法中修改会导致编译失败。
  最小修复：移除 const 修饰符
  // 修改前
  GraphAtlas DeepLearningTask::build_graph_atlas() const {

  // 修改后
  GraphAtlas DeepLearningTask::build_graph_atlas() {
  同时头文件声明同步移除 const：
  // include/renaissance/task/deep_learning_task.h
  GraphAtlas build_graph_atlas();   // ← 去掉 const
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟡 P1：run_gpu() 中 prep_thread 缺少异常捕获
  位置：§8.1 run_gpu() 主循环
  问题：std::thread prep_thread([&]() { prep.train(); }); 若 prep.train() 抛异常，std::terminate() 会被直接调用，程序崩溃且无法定位原
  因。run_train_epoch_gpu() 内的 K 个 rank 线程已有 exc[rank] 异常传播机制，但 Preprocessor 线程没有。
  最小修复：统一异常捕获
  // 修改前
  std::thread prep_thread([&]() { prep.train(); });
  run_train_epoch_gpu();
  prep_thread.join();

  // 修改后
  std::exception_ptr prep_exc;
  std::thread prep_thread([&]() {
      try { prep.train(); }
      catch (...) { prep_exc = std::current_exception(); }
  });
  run_train_epoch_gpu();
  prep_thread.join();
  if (prep_exc) std::rethrow_exception(prep_exc);
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟡 P1：compile_debug_print_and_return() dry-run 路径仍调用 build_simple_atlas()
  位置：§2.2 修改清单 与 task_base.cpp:597
  问题：compile_debug_print_and_return() 中对非 SimpleTask 调用 build_simple_atlas(name_to_gid_)。该函数对 DL 场景会：
  1. 将所有命名图填入 GraphId::SIMPLE_TASK_GRAPH（错误 ID）
  2. sl.mp = &memory_plan_（绑定空 plan）
  3. 导致 dry-run 输出的 Atlas 信息完全失真
    Z_FINAL_K.md 的 §2.2 修改清单已覆盖 memory_plan_ → active_memory_plan_ 的替换，但遗漏了 Atlas 构建函数的替换。
    最小修复：在 compile_debug_print_and_return() 中动态分派
    // 在 task_base.cpp compile_debug_print_and_return() 中
    // 将：
    GraphAtlas atlas = build_simple_atlas(name_to_gid_);

  // 替换为：
  GraphAtlas atlas;
  if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
      atlas = dl->build_graph_atlas();
  } else {
      atlas = build_simple_atlas(name_to_gid_);
  }
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🟢 P2（可选）：stream_for() 建议声明为 static
  位置：§3.2 stream_for() 实现
  问题：stream_for(GraphId) 不访问任何成员变量，但当前声明/实现为普通成员函数。虽不影响编译，但语义上应为工具函数。
  最小修复：头文件中加 static
  // include/renaissance/task/deep_learning_task.h
  static StreamKind stream_for(GraphId gid);
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  总结
   问题                                            严重度        是否阻塞编译          修改量
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   build_graph_atlas() const                       🔴 P0              是             删 1 个单词
   prep_thread 无异常捕获                          🟡 P1         否（健壮性）           +3 行
   compile_debug_print_and_return() Atlas 未替换   🟡 P1    否（dry-run 输出失真）      +4 行
   stream_for 未 static                            🟢 P2              否              +1 个单词

  除上述 4 处外，Z_FINAL_K.md 方案本身无其他问题。核心时序、内存安全、并发模型、LR 机制均已在源码层面验证通过。
