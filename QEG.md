# 【小伙伴K】

• 发现 1 个明显 bug，建议修复：
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Bug：run_h2d_only_val_epoch() GPU 路径异常时跳过 prep_thread.join()
  位置：src/task/deep_learning_task.cpp 约 line 2389-2391
  问题代码：
  for (auto& t : threads) t.join();
  for (int rank = 0; rank < K; ++rank)
      if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);  // ← 这里提前抛出
  } else
  #endif
  {
      // CPU 路径
  }

  auto t1 = std::chrono::steady_clock::now();
  prep_thread.join();        // ← 异常时永远执行不到
  if (prep_exc) std::rethrow_exception(prep_exc);
  如果某个 rank 线程抛异常，std::rethrow_exception 会直接跳出函数，prep_thread 没被 join，Preprocessor 可能还在跑，导致
  资源泄漏 + 状态不一致。
  对比 run_h2d_only_train_epoch()（正确的顺序）：
  auto t1 = std::chrono::steady_clock::now();
  prep_thread.join();
  if (prep_exc) std::rethrow_exception(prep_exc);
  #ifdef TR_USE_CUDA
      if (registry.using_gpu()) {
          for (int rank = 0; rank < K; ++rank)
              if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);
      }
  #endif