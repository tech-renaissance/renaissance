# 【小伙伴K】

• ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  根因报告：batch_size=199 时 CPU/GPU/AMP 全路径失败
  一、现象确认
   测试   batch=200（整除）   batch=199（不整除，last=101）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CPU    97.61% PASS         12.41% FAIL
   GPU    97.61% PASS         nan / 0% FAIL
   AMP    97.61% PASS         nan / 0% FAIL
  batch=200 时 60000/200=300 整除，没有 last batch，全程只走 v_base（v0 或 v2）。
  batch=199 时 60000/199=302 余 101，最后一个 batch 必须切换到 v_last（v1 或 v3）。问题只在 last batch 切换 variant 时出
  现。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  二、核心假设验证：MemoryPlan offset 真的跨 variant 一致吗？
  文档声称一致，memory_plan.h 第 316 行注释：
  ▌ "DTensor 内部存储此常数为 slot_bytes_，finalize 时所有变体返回同一值 → offset 一致"
  但这只对 Compiler::create_memory_plans 中显式传入 slot_bytes 的 tensor 成立（feature map、grad slot）。对于输入缓冲区
  和标量，走的是另一条路径：
  // memory_plan.cpp:359-362
  auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
  auto da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA);
  auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
  auto db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA);
  alloc_impl 三参数版本自己计算 slot_bytes：
  uint64_t effective = DTensor::compute_slot_bytes(shape, dtype, region);
  而 label_shape / data_shape 在 compiler.cpp:746 是variant 特定的：
  Shape label_shape{specs[s].batch_size, 1, 1, 1};
  结论：输入缓冲区（I_A_LABEL, I_A_DATA, I_B_LABEL, I_B_DATA）的 slot_bytes 随 batch_size 变化，不是最大值。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  三、offset 错位链式反应
  MemoryPlan::finalize()（memory_plan.cpp:606-642）的算法是按 Region 顺序线性累加：
  uint64_t cursor = 0;
  for (每个 Region) {
      info.base_offset = cursor;
      for (该 Region 的每个 DTensor) {
          dt.offset_ = cursor;
          cursor += dt.slot_bytes();   // ← 这里用了实际 slot_bytes
      }
  }
  以 MNIST FP32 为例对比 v0（batch=199）和 v1（batch=101）：
   Region      v0 slot_bytes      v1 slot_bytes      导致后果
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   I_A_LABEL   1,024              1,024              ✅ 相同
   I_A_DATA    627,712            319,648            ❌ v1 少了 308,064
   I_B_LABEL   1,024              1,024              但 base_offset 已错位
   I_B_DATA    627,712            319,648            ❌ 又少了 308,064
   F_FEATURE   1,874,944（max）   1,874,944（max）   相同，但 base_offset 错位了 616,128
   S_SCALAR    2,560              2,560              相同，但 base_offset 错位
   ...         ...                ...                所有后续 Region 全部错位
  总错位量：~616 KB，足以让 v1 的 captured graph 把 logits/weight/gradient 指针指到 DeviceContext 中完全错误的区域。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  四、Captured Graph 如何把错位地址固化进 GPU 指令
  capture_all_for_rank（captured_graph.cpp:367）：
  ctx.set_memory_plan(mp);   // ← 切换为 variant 自己的 MemoryPlan
  auto cg = CapturedGraph::capture(..., mp, ...);
  capture_cuda（capture_cuda.cpp:112）：
  entry.launch_cuda(node, mp, ctx, state);
  以 SOFTMAX_CE_FP32_FWD 为例（softmax_ce_op.cpp:346-360）：
  const float* logits = static_cast<const float*>(ctx.ptr_at(ids_in[0]));   // ← 用 mp 的 offset 算地址
  // ...
  const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
  int batch = logits_dt.shape.n();   // ← 硬编码进 captured graph！
  双重错误：
  1. 地址错误：ctx.ptr_at(ids_in[0]) 用的是 v1 的 offset，但 DeviceContext 实际内存是 v0 的布局 → 指针飞到错误位置
  2. batch 硬编码：batch = 101 被写死在 v1 的 cudaGraph 里，即使运行时想改也改不了
    GPU 读到随机内存 → softmax 遇到极大/极小值 → nan。
    CPU 读到错误内存 → 计算结果错误但不崩溃 → 12.41%（接近随机）。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    五、为什么 build_graph_atlas() 的设计加剧了问题
    当前 build_graph_atlas()：
    if (is_shape_invariant_graph(gid)) {
      sl.mp = active_memory_plan_;
    } else {
      sl.mp = mp;   // ← variant 自己的 MemoryPlan
    }
    这个分支的前提是："shape-dependent 图需要 variant 自己的 MemoryPlan 来获取正确的 shape 信息"。
    但这个前提是自相矛盾的：
    • 如果 sl.mp = mp（v1 的 MemoryPlan），capture 时地址按 v1 的 offset 计算 → 与 DeviceContext 内存布局不匹配
    • 如果 sl.mp = active_memory_plan_（v0 的 MemoryPlan），capture 时地址正确，但 batch = 199 被硬编码 → last batch 处理

    9 个样本而非 101 个
  无论选哪个，都有缺陷。 根本原因是 kernel launch 函数从 mp.get_dtensor(...).shape.n() 取 batch，而不是从运行时标量读取
  。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  六、汇总：失败链
  batch=199 → last batch 存在
      ↓
  run_train_epoch_gpu() 切换 v_base→v_last
      ↓
  build_graph_atlas() 给 v_last 分配 sl.mp = variant_memory_plans_[1]
      ↓
  pre_capture() 用 v1 的 MemoryPlan 做 key_to_mp
      ↓
  capture_cuda() 调用 ctx.ptr_at(id) → 按 v1 的 offset 算地址
      ↓
  v1 的输入缓冲区 I_A_DATA 只有 319,648 B（v0 是 627,712 B）
      ↓
  finalize() 累加后，v1 的 F_FEATURE 及之后所有 Region base_offset 错位 ~616 KB
      ↓
  captured graph 中 logits/weight/grad 指针全部指向 DeviceContext 错误位置
      ↓
  GPU kernel 读取未初始化/错误内存 → nan / 0%
  CPU kernel 读取错误内存 → 训练无效 → 12.41%
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  七、修复方向（按侵入性排序）
   方案   改动范围                                                说明
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   A      Compiler::create_memory_plans + MemoryPlan::alloc_bas   让输入缓冲区也使用跨变体最大 slot_bytes，从根本上保证
          eline_dtensors                                          所有 variant offset 完全一致。最正确，但涉及基础设施
                                                                  。
   B      build_graph_atlas() 一行修改                            强制所有 variant 的 sl.mp = active_memory_plan_。offs
                                                                  et 立即正确，但 last batch 的 kernel 硬编码 batch=199
                                                                  ，会多处理 98 个旧样本。不会崩溃，准确率可能微降。
   C      所有 kernel launch 函数                                 把 batch = mp.get_dtensor(...).shape.n() 改为从 batch
                                                                  _size 标量指针读取。工作量大，但最灵活。
   D      run_train_epoch_*() 运行时                              last batch 不切换 variant，继续使用 v_base 的 graph，
                                                                  只改标量。但 kernel 硬编码 batch 不变，效果同 B。

# 【小伙伴D】

## 🔴 根因分析报告：batch_size=199 时 loss=NaN, top1=0%
### 1. 问题链路
```
on_prepare():
  Compiler::compile() → 6 个独立 variant MemoryPlan
  active_memory_plan_ = variant_memory_plans_[0].get()   ← 只 finalize variant 0

compile_alloc_hardware():
  GPU buffer = ArenaKeeper.alloc(active_memory_plan_->total_bytes())  ← 按 variant 
  0 布局
  ctx.set_memory_plan(active_memory_plan_)                             ← DTensor→地
  址绑定用 variant 0 的 offset

pre_capture():
  对 variant 1 (last batch):
    ctx.set_memory_plan(variant_memory_plans_[1].get())  ← 切换为 variant 1 的 
    MemoryPlan
    捕获 CUDA Graph → 地址 baked in 来自 variant 1 的 offset 布局

运行时（last batch）:
  cudaGraphLaunch(l_deep_a, ...)  → GPU 以 variant 1 的 offset 访问内存
  但 GPU buffer 是按 variant 0 的 offset 布局的
  → 所有 DTensor 访问偏移 ~615KB → 读到垃圾数据 → NaN
```
### 2. 关键证据 2.1 Baseline 分配使用 shape 决定大小（导致 variant 间 offset 不同）
create_memory_plans() 中每个 variant 独立分配：

alloc_baseline_dtensors 调用的是 不带 slot_bytes 的 alloc_impl(shape, ...) ( memory_plan.cpp:L509-L528 )，内部调用 DTensor::compute_slot_bytes(shape, ...) ：

结论：baseline 分配的 slot_bytes 直接依赖 batch_size。

DTensor Variant 0 (n=199) Variant 1 (n=101) 差异 I_A_DATA 199×28×28×4 = 623,872 B 101×28×28×4 = 316,736 B -307,136 B I_A_LABEL 199×4 = 796 B 101×4 = 404 B -392 B I_B_DATA 623,872 B 316,736 B -307,136 B I_B_LABEL 796 B 404 B -392 B 累计 ≈1,250,000 B ≈635,000 B ≈615,000 B
 2.2 Layer tensor 的 offset 取决于前面累计的 cursor 位置
MemoryPlan::finalize() 按 Region 顺序线性推进 cursor：

Baseline DTensors 的 Region（ I_A_LABEL=050 , I_A_DATA=051 , I_B_LABEL , I_B_DATA , S_SCALAR_* , R_RESULT , R_RESULT_ACCUMULATED ）排在权重 DTensors（ W_FIRST_BN_WEIGHT 等）之前。variant 0 处理完 baseline 后 cursor=~1,250,000，variant 1 处理后 cursor=~635,000。

所有后续权重/特征 DTensor 的 offset 在两个 variant 中相差 ≈615KB。
 2.3 GPU buffer 只用 variant 0 的布局
compile_alloc_hardware() ：

且 L248 ctx.set_memory_plan(active_memory_plan_) 将 DTensor→GPU 地址绑定为 variant 0 的 offset。
 2.4 捕获时切换到 variant 1 的 MemoryPlan
capture_all_for_rank() ：

CUDA Graph 将 kernel 参数（即 DTensor GPU 地址）物理 baked in。这些地址 = gpu_base + variant_1_dtensor_offset 。但 GPU base 分配时是按 variant 0 的 offset 布局的。

### 3. 具体示例（MNIST MLP, fc_weight 的 offset）
当 last batch 启动 l_deep_a （variant 1 的 DEEP_FWD_BWD）：

- 它访问权重 fc_weight 时去到 gpu_base + 635,000 + w_offset_in_variant_1
- 但实际 fc_weight 在 gpu_base + 1,250,000 + w_offset_in_variant_1
- 读到的全是错位数据 → 梯度全错 → loss = NaN
### 4. 为什么 batch_size=200 正常？
batch_size=200 时，所有 batch 大小相同（60000/200=300，无 incomplete last batch）， get_last_train_batch_size() == get_local_batch_size() == 200 ，6 个 variant 全部产生同一个 ShapeId{200,28,28,1} 。pre_capture 去重后只捕获 1 张图（用 variant 0 的 MemoryPlan），variant 1-5 全走 reuse → 不存在跨 variant 的 offset 不一致问题。

### 5. 修复方向（建议）
根因在 Compiler 层 ： create_memory_plans 中 baseline 分配应使用跨 variant 的 max slot_bytes（类似 layer tensor 已有的 max_slots 机制），保证所有 variant 的 DTensor offset 一致。

具体来说：

1. Phase 2 扩展 compute_max_slot_bytes 覆盖 baseline Region（ I_A_LABEL , I_A_DATA , I_B_LABEL , I_B_DATA , scalars, result regions）
2. Phase 3 alloc_baseline_dtensors 接受 slot_bytes 参数，用 max 值分配
3. 或者在 on_prepare() 中 finalize 所有 variant，用 std::max(total_bytes) 分配 GPU buffer，并确保所有 variant 的 offset 一致
以上是根因分析。

# 【用户补充】

这不是很明显吗？输入缓冲区的slot bytes肯定要取最大值。我们的策略从根本上讲，就是要确保不同的变体之间，DTensor数量一样、相同ID的DTensor的offset和slot bytes一样（但shape允许不同）。

小伙伴K的方案B显然是不对的，因为那是强行把不完整batch视为完整batch来处理了，那根本就是数学错误，完全白费了variant设计的一片苦心。

大家认真研究，给出一个数学正确、完整、安全且高效的方案。

