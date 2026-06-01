# 【小伙伴K】

• 好的，我找到了。这是一个非常隐蔽但致命的内存偏移错位问题。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  根本原因：Last Batch 的验证 metrics 累加到了错误的内存地址
  问题链
  1. Variant 内存布局差异
    MemoryPlan::finalize() 采用一遍线性累加布局。每个 Region 的 base_offset 取决于前面所有 Region 的总大小。
    • variant 4 (val_base): batch_size = local_bs（如 128 或 16）
    • variant 5 (val_last): batch_size = last_val_batch_size（如 16、8、4、2）
    由于 I_A_DATA、I_B_DATA 等输入缓冲区的大小直接正比于 batch_size，variant 4 和 variant 5 的前置 Region 总大小不同，导致
    后续 Region（包括 R_RESULT_ACCUMULATED）的 base_offset 产生偏移。
  2. 验证循环中的 Variant 切换
    在 run_val_epoch_gpu() 中：
    auto g_accum = is_last ? vl_accum : vb_accum;  // last batch → variant 5
    • 标准 batch：使用 vb_accum（variant 4 的 ACCUM_METRICS），写入 variant 4 的 accum_loss
    • 最后一个 batch：使用 vl_accum（variant 5 的 ACCUM_METRICS_VAL_LAST），写入 variant 5 的 accum_loss
  3. VAL_RESULT_ALLREDUCE 只读取 variant 4 的内存
    if (vb_comm) cudaGraphLaunch(vb_comm, s_up);  // variant 4 的 RANGE_MEAN_ALLREDUCE
    VAL_RESULT_ALLREDUCE 的 input_ranges 在编译时绑定到 R_RESULT_ACCUMULATED Region。由于 CUDA Graph capture 时解析的是var
    iant 4 的 MemoryPlan offset，它只会 AllReduce variant 4 的 accum_loss/top1/top5。
  4. 最终读取的又是 variant 0/4 的内存
    const auto& b = active_memory_plan_->baseline();  // ← variant 0（train_base）
    int32_t al_id = b.accum_loss;
    fetch_from_rank(al_dt, 0) 读取的是 active_memory_plan_（variant 0）的 accum_loss。由于 variant 0 和 variant 4 的 I/O s
    hape 相同（都是 local_bs），它们的 accum_loss offset 恰好相同，所以能读到标准 batches 的累加值。
    但 variant 5 的 accum_loss 落在完全不同的 offset，因此：
    • last batch 的 sample_loss * last_bs 写入了"无人区"
    • VAL_RESULT_ALLREDUCE 没有把它纳入
    • fetch_from_rank 也读不到它
    后果：每个 RANK 的最后一个 batch 被完全忽略
      RANK   per_rank   local_bs   last_bs   丢失样本数   丢失比例
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      1      10000      128        16        16           0.16%
      2      5000       64         8         8×2=16       0.16%
      4      2500       32         4         4×4=16       0.16%
      8      1250       16         2         2×8=16       0.16%
    总丢失样本数都是 16（因为 10000 样本），但具体丢失哪些样本完全不同：
    • 1 RANK：丢失第 9984~9999 号样本
    • 8 RANK：每个 RANK 丢失各自最后 2 个样本（如 RANK0 丢失第 1248~1249 号，RANK1 丢失第 2498~2499 号...）
    由于 MNIST 验证集的样本分布不均匀（某些数字的样本更容易被分到最后），不同 world_size 下丢失的"尾端样本"不同，导致：
    • 准确率呈现非单调波动（2 RANK 可能比 1 RANK 好）
    • 8 RANK 时尾端 batch 只剩 2 个样本，如果恰好包含难例，准确率下降更明显
    为什么 train loss 不受影响？
    训练时也有 train_last variant（variant 1），但训练 loss 的计算没有跨 RANK AllReduce：
    size_t per_rank = registry.train_samples_per_rank();
    train_loss = accum_val / static_cast<float>(per_rank);
    accum_val 是单个 RANK 本地的累加值，直接从该 RANK 的 variant 1 内存读取。不存在"从 variant 0 读取 variant 1 数据"的问
    题。
    而且训练的 ACCUM_METRICS_TRAIN_LAST 结果只用于打印，不影响权重更新（权重更新基于梯度，梯度 AllReduce 是正确的）。
    为什么单 GPU 测试通过？
    单 GPU 时 world_size=1，local_bs=128，last_val_batch_size=16。虽然 last batch 同样被丢失，但：
    • 丢失比例仅 0.16%
    • 对 99.5% 级别的准确率影响微乎其微（<0.02%）
    • 而且单 GPU 没有"不同 RANK 丢失不同样本"的问题
    建议修复方案
    方案 A（推荐）：统一验证 metrics 的 Region offset
    确保 variant 5 的 R_RESULT_ACCUMULATED Region 与 variant 4 的 offset 对齐。可以通过：
  1. 在 MemoryPlan::finalize() 中为 R_RESULT_ACCUMULATED 等关键 Region 强制固定偏移；或
  2. 将 ACCUM_METRICS_VAL_LAST 的 output_ranges 显式指向 variant 4 的 R_RESULT_ACCUMULATED Region（而非 variant 5 自己的
    方案 B：验证阶段不用 variant 5，用标量动态切换 batch size
    取消 val_last variant，所有验证 batch 都用 val_base variant，但通过运行时标量动态调整最后一个 batch 的 batch_size_ptr
    （前提是 SoftmaxCE kernel 真正使用 batch_size_ptr）。
    方案 C：验证结束后手动合并 last batch metrics
    在 run_val_epoch_gpu() 中，last batch 执行后，手动将 variant 5 的 accum_loss/top1/top5 加到 variant 4 的对应值上，然后
    再执行 VAL_RESULT_ALLREDUCE。