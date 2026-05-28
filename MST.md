# 【今日话题：解决AMP模式下的NaN问题】



众所周知，test_dl_full的AMP模式总是要么NaN要么不学习，但是CPU/GPU模式是正常的。大家一起来找找问题所在，调通AMP模式。



# 【小伙伴K】

  我卡住了。train_loss = 0 的原因仍然不完全清楚。

  但小伙伴发现的 optimizer bug 是确定无疑的。这个 bug 会导致 AMP 训练完全失败（因为 beta=65536），而且 FP32 下也有参数错
  位问题（虽然 scaling=1 时影响较小）。

  让我给出完整的分析意见，重点放在小伙伴发现的 bug 上，同时提及我分析到的其他问题。
• 根据你小伙伴的线索，我逐项做了代码级验证。结论：两个指控全部属实，而且问题比表面更严重——这不仅影响 AMP，FP32 的 Adam/A
  damW 同样在“带病运行”。以下是完整的技术分析。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一、指控 1：input_ids 索引错位（compiler.cpp vs optimizer_op.cpp）
  1.1 证据：compiler.cpp 的组装顺序
  src/graph/compiler.cpp 中构建非-LARS Weight/Bias 优化器节点时，无条件把 scaling 插到了固定位置：
  // Weight (line 1584-1597)
  node.input_ids.push_back(scalar_ids.lr);      // idx 0
  node.input_ids.push_back(scalar_ids.wd);      // idx 1
  node.input_ids.push_back(scalar_ids.scaling); // idx 2  ← 这里！
  if (weight_needs_m) {
      node.input_ids.push_back(scalar_ids.beta); // idx 3
  }
  if (weight_needs_v) {
      node.input_ids.push_back(scalar_ids.beta2); // idx 4
      node.input_ids.push_back(scalar_ids.eps);   // idx 5
  }
  node.input_ids.push_back(scalar_ids.has_nan); // last
  // Bias (line 1647-1658)
  node.input_ids.push_back(scalar_ids.lr);      // idx 0
  node.input_ids.push_back(scalar_ids.scaling); // idx 1  ← 这里！
  if (bias_needs_m) {
      node.input_ids.push_back(scalar_ids.beta); // idx 2
  }
  ...
  node.input_ids.push_back(scalar_ids.has_nan); // last
  1.2 证据：optimizer_op.cpp launcher 的读取顺序
   优化器            launcher 读取的 scalar 索引     compiler 实际放的值（idx 对应）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Weight SGD        0=lr, 1=wd, 2=scaling           0=lr, 1=wd, 2=scaling ✅ 唯一正确的
   Weight Momentum   0=lr, 1=wd, 2=beta              2=scaling ❌ beta = scaling
   Weight Nesterov   0=lr, 1=wd, 2=beta              2=scaling ❌ beta = scaling
   Weight Adam       0=lr, 1=wd, 2=b1, 3=b2, 4=eps   2=scaling, 3=beta, 4=beta2 ❌ 三级错位
   Weight AdamW      0=lr, 1=wd, 2=b1, 3=b2, 4=eps   2=scaling, 3=beta, 4=beta2 ❌ 三级错位
   Bias SGD          0=lr, 1=scaling                 0=lr, 1=scaling ✅
   Bias Momentum     0=lr, 1=beta                    1=scaling ❌ beta = scaling
   Bias Nesterov     0=lr, 1=beta                    1=scaling ❌ beta = scaling
   Bias Adam         0=lr, 1=b1, 2=b2, 3=eps         1=scaling, 2=beta, 3=beta2 ❌ 三级错位
  1.3 影响
  AMP 模式下（scaling = 65536）：
  • Momentum/Nesterov：beta = 65536，动量更新变成 m = m * 65536 + g，一 step 就爆炸。
  • Adam/AdamW：b1 = 65536，b2 = 0.9，eps = 0.999。一阶矩 m 几乎不接收梯度（(1-65536) * g 是负数且巨大），二阶矩 v 的 de
    y 被设成 0.9 而非 0.999，导致分母极小，更新量爆炸。
  FP32 模式下（scaling = 1.0）：
  • 因为 init_scaling = amp ? TR_AMP_INITIAL_SCALING : 1.0f，FP32 下 scaling = 1.0f。
  • Momentum：beta = 1.0，变成不衰减的累积动量（m = m + g），不是标准 Momentum。
  • Adam：b1 = 1.0，一阶矩完全冻结（m = m * 1.0 + 0 * g = m）；b2 = 0.9，eps = 0.999。训练还能跑是因为 m=0 时 Adam 退化
    0 / sqrt(v) + 0.999 的伪-RMSprop，但这不是 Adam。
  所以 FP32 下 97.61% 的“正常”是假象——Adam 实际在以一个完全错误的超参数组合运行。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  二、指控 2：Momentum/Nesterov/Adam/AdamW 均未做 unscaling
  2.1 CUDA kernel 层面（optimizer_op.cu）
  // SGD kernel — 唯一做了 unscaling 的
  float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
  float g_i = g[i] * _inv_scaling;   // ✅

  // Momentum kernel
  float g_i = g[i];                  // ❌ 直接用了 scaled 梯度

  // Nesterov kernel
  float g_i = g[i];                  // ❌

  // Adam kernel
  float g_i = g[i];                  // ❌

  // AdamW kernel
  float g_i = g[i];                  // ❌
  2.2 CPU launcher 层面（optimizer_op.cpp）
  sgd_update_cpu:     g[i] * inv_scaling   ✅
  momentum_update_cpu: g[i]                ❌
  nesterov_update_cpu: g[i]                ❌
  adam_update_cpu:     g[i]                ❌
  adamw_update_cpu:    g[i]                ❌
  2.3 影响
  AMP 的完整链路是：
  1. Forward：loss *= scaling（在你的 kernel 里是 atomicAdd(loss, sample_loss * inv_batch * scaling)）。
  2. Backward：由于 loss 被 scaled，所有梯度 g 天然携带 scaling 倍放大。
  3. Optimizer：必须在更新前将 g 除以 scaling（unscaling），否则权重更新步长是正常值的 65536 倍。
    SGD 做了 unscaling，所以 SGD 在 AMP 下能勉强工作（但 scaling 参数被放在了正确的位置，这是巧合）。
    Momentum/Adam/AdamW 既读错了参数又没做 unscaling，双重 bug 叠加，导致训练在 AMP 下完全不可收敛。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    三、我额外发现的关联问题
    3.1 SOFTMAX_CE_AMP_INF 在 val 时也乘了 scaling
    softmax_ce_inf_kernel（推理/验证专用）line 242：
    atomicAdd(loss, sample_loss * inv_batch * scaling);
    这导致 val loss 被放大了 scaling 倍。即使 ACCUM_METRICS 正常工作，val loss 也会是 true_loss * 65536。
    推理/验证不应乘以 scaling，因为 scaling 仅服务于 backward 的梯度放大。
    3.2 inv_scaling 写成了 1.0f / batch_size
    softmax_ce_fwd_kernel line 141 和 softmax_ce_inf_kernel line 299：
    *inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);  // BUG
    inv_scaling 的语义应该是 1.0f / scaling，供 backward 做 scale = scaling * inv_scaling。
    当前实现导致 backward 的 scale = scaling / batch_size，这不是 GradScaler 语义。
    3.3 g_accum 在 training loop 中的执行顺序错位
    src/task/deep_learning_task.cpp training loop（line 1197）：
    if (g_accum) cudaGraphLaunch(g_accum, s_up);   // 累加 metrics
    sync_up();
    ...
    cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);  // 清零 loss
    if (g_deep) cudaGraphLaunch(g_deep, s_c1);     // 计算 loss + backward
    g_accum 被放在了 g_deep 之前。这意味着：
    • Batch 0：g_accum 读取的是初始值 0（或残留值），累加 0。
    • Batch 1：g_accum 读取的是 Batch 0 的 loss，累加正确。
    • ...
    • 最后一个 batch 的 loss 永远不被累加。
    FP32 下因为 batches 很多（~300），漏一个误差只有 0.3%，看不出来。
    但在调试时如果 batches 少，这个偏差会被放大。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    四、修复方案（按优先级排序）
    P0：修复 input_ids 错位（立刻修复，否则任何 AMP 训练都失败）
    方案 A（最小改动，推荐）：修改 compiler.cpp，移除所有非-SGD 优化器节点中的 scaling 插入，让 input_ids 顺序与 launcher
    期望的一致。然后单独给所有 kernel 添加 scaling 参数并做 unscaling。
    但这需要改 8 个 launcher + 5 个 kernel 的签名。
    方案 B（更彻底，推荐）：统一所有 Weight/Bias 优化器节点的 input_ids 顺序为：
    [lr, wd, beta, beta2, eps, scaling, has_nan]
    缺少的参数跳过，但保持相对顺序。然后修改所有 launcher 按统一索引读取，并给所有 kernel 传入 scaling。这样以后新增优化器
    也不会再错位。
    我推荐方案 B，因为当前代码的“按需 push_back”是导致 bug 的根源，统一顺序可以根治。
    P0：给所有 optimizer kernel 添加 unscaling
    在 optimizer_op.cu 的 5 个 kernel 中统一添加 scaling 参数：
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    float g_i = g[i] * _inv_scaling;
    P1：修复 SOFTMAX_CE_AMP_INF 的 loss 缩放
    softmax_ce_inf_kernel 中：
    // 原来
    atomicAdd(loss, sample_loss * inv_batch * scaling);
    // 改为
    atomicAdd(loss, sample_loss * inv_batch);
    P1：修复 inv_scaling 的赋值
    *inv_scaling = 1.0f / (*scaling_ptr);  // 原来是 1.0f / batch_size
    P2：调整 training loop 中 g_accum 的位置
    把 g_accum 的 launch 移到 g_deep 之后（紧跟在 sync_comp() 后面），并确保 cudaMemsetAsync(loss, 0) 在下一个 batch 的 g_
    deep 之前。或者更简单地，把 g_accum 和 cudaMemsetAsync(loss, 0) 都放到 s_c1 上，与 g_deep 同 stream，利用 stream 顺序
    保证正确性。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    五、总结
    你小伙伴的直觉完全正确。当前代码存在系统性的 AMP 支持缺陷：
      问题                            严重性   影响
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      input_ids 错位                  致命     Momentum/Adam/AdamW 读错超参数，AMP 训练必炸；FP32 Adam 也在错误参数下运行
      缺失 unscaling                  致命     梯度未还原，权重更新步长放大 65536 倍
      SOFTMAX_CE_AMP_INF 乘 scaling   高       val loss 虚假爆炸
      inv_scaling 写错                高       backward scale 语义错误
      g_accum 顺序错位                中       漏统计最后一个 batch，metrics 有系统性偏差

  FP32 下 97.61% 的“正常”是因为 scaling=1.0 掩盖了参数错位（b1=1.0 时 Adam 还能勉强跑），并不代表代码正确。



# 【小伙伴D】

对，先分析清楚再动手。让我把所有 scaling 相关的路径完整梳理一遍。

## 完整根因分析
### 配置回顾
测试用的是 SGD(momentum=0.9, nesterov=false) ，对应 OptimizerKind::SGD_MOMENTUM 。

### Scaling 的完整链路（按 NVIDIA AMP 惯例）
scaling 在 (2) 和 (3) 中相互抵消，唯一作用是让 FP16 中间梯度更大（避免下溢）。

### Bug A：Momentum kernel 根本没有 unscaling
对比两个 kernel：

操作 SGD kernel Momentum kernel 参数 (w, g, n, lr, wd, has_nan, **scaling**) (w, g, m, n, lr, wd, beta, has_nan) unscaling g_i = g[i] * (1.0f/scaling) ✅ 不存在 ❌ 更新 w -= lr * g_i m = m*beta + g; w -= lr * m

Momentum/Nesterov/Adam/AdamW — 四个 kernel 全部没有 unscaling 。只有 SGD 有。

这意味着当 scaling=2 时：

- 反向传播梯度 ×2（正确）
- Momentum 直接把 ×2 的梯度累加进动量 buffer，不做反除
- 等效学习率变成 lr × 2 = 0.2
- 训练迅速发散 → NaN
### Bug B（更致命）：索引错位
compiler 为 weight 的 MOMENTUM 节点填入的 scalar input 顺序：

但 launcher 读取的顺序：

scaling=1.1 时，beta 被读成 1.1（而不是 0.9） 。动量公式变成：

几轮迭代后 m 发散 → NaN。

同理 bias momentum launcher 也读错： [lr, scaling, beta] → launcher 从 index 1 读 scaling 当作 beta。

而 scaling=1 时，beta 被读成 1.0，比正确的 0.9 稍大，虽然不至于 NaN 但动量系数不对，准确率只有 16.63%。

### 完整解释
scaling 实际 beta 值 实际 lr 等效倍率 结果 1.0 1.0 (≈0.9) ×1.0 (≈正常) 无 NaN，16.63%（动量系数偏高） 1.1 1.1 ×1.1 beta > 1 → 动量发散 → NaN 1.25 1.25 ×1.25 更严重，NaN 2.0 2.0 ×2.0 剧烈 NaN

这就完美解释了"只要 scaling ≠ 1 就炸"。

### 修复方案
涉及 3 个文件：

1. compiler.cpp — 修正 scalar 输入顺序

把 scaling 从 beta 前面移到 beta 后面（has_nan 前面）：

优化器 修正前 修正后 Weight SGD [lr, wd, scaling, has_nan] 不变 ✅ Weight MOMENTUM [lr, wd, scaling, beta, has_nan] [lr, wd, beta, scaling, has_nan] Weight ADAM [lr, wd, scaling, b1, b2, eps, has_nan] [lr, wd, b1, b2, eps, scaling, has_nan] Bias SGD [lr, scaling, has_nan] 不变 ✅ Bias MOMENTUM [lr, scaling, beta, has_nan] [lr, beta, scaling, has_nan] Bias ADAM [lr, scaling, b1, b2, eps, has_nan] [lr, b1, b2, eps, scaling, has_nan]

2. optimizer_op.cu — 4 个非 SGD kernel 全部加 unscaling

momentum/nesterov/adam/adamw kernel 各加 const float* scaling 参数，梯度先除 scaling：

```
float inv = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
float g_i = g[i] * inv;
```
以及 launcher extern 声明更新。

3. optimizer_op.cpp — 非 SGD launcher 读取 scaling 并传递

在各 launcher 中用 scalar_ptr<新索引> 读取 scaling，传给 kernel。CPU 更新函数同步修改。

### 预期效果
修复后 scaling=2~4 应该正常工作，梯度精度提升带来更高的准确率。如果 scaling 继续增大直到某个临界值导致 FP16 梯度溢出，CHECK_NAN + GRAD_SCALING 会正确地把 scaling 折半到安全范围，训练可以继续进行。