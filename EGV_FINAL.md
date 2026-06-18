# CPU 路径调用顺序对齐 GPU — 最终方案

## 三方案对比

| 改动点 | EGV1 | EGV2 | EGV3 | **最终方案** |
|--------|------|------|------|:----------:|
| 删除 loss memset | ✅ 删 | ✅ 删 | ❌ 保留 | **保留** |
| `set_buffer` 后置 | ✅ | ✅ | ✅ | ✅ |
| `accum` 后置到 far 之后 | ✅ | ✅ | ✅ | ✅ |
| `ncg` 移到 first_bwd 组 | ❌ **错误** | ✅ 留在 UPDATE 组 | ✅ 留在 UPDATE 组 | **留在 UPDATE 组** |
| `batch_size` 移动 | ✅ 移到末尾 | ✅ 移到 far 后 | ❌ 不移动 | **不移动** |
| batches==1 分支 | ✅ | ✅ | ❌ 遗漏 | ✅ 对齐 |
| AMP 占位注释 | ✅ | ✅ | ❌ | ❌ 不添加 |

## 否决项理由

### 1. 删除 loss memset — 否决

**已确认：CPU 路径必须保留 loss memset。**

`softmax_ce_op.cpp` 第 152 行，CPU FWD 内核实现：

```cpp
// softmax_ce_fwd_inner — CPU 核心算法
*loss += sample_loss * inv_b;   // 逐样本累加，不是覆盖！
```

同样 `softmax_ce_inf_inner` 第 219 行也是 `*loss += ...`。

CPU 内核遍历 batch 内每个样本，用 `+=` 把 loss 累加到输出 buffer。内核本身不负责清零——它假设调用者已置零。GPU 能删 `cudaMemsetAsync` 是因为它用 `loss_partial` 中间缓冲 + reduction kernel 直接写入（`=` 非 `+=`），CPU 没有这层间接，直接对最终 buffer 做 `+=`。删除 memset 会导致上一 batch 的 loss 残留，新 batch 结果被累加在旧值之上，数值完全错误。

### 2. ncg 移到 first_bwd 组 — 否决

EGV1 混淆了 `n_cdg`（`CAST_DEEP_GRAD`，AMP 专用，在 first_bwd 组）和 `n_ncg`（`NAN_CHECK_GRAD_SCALE`，在 UPDATE 组）。

GPU 代码第 1148-1169 行明确：

```cpp
// first_bwd 组（line 1148-1150）
if (!frozen && g_first) cudaGraphLaunch(g_first, s_c1);   // first_bwd
if (using_amp && n_cdg) cudaGraphLaunch(n_cdg, s_up);      // cast_dg (AMP)
if (n_dar) cudaGraphLaunch(n_dar, s_up);                    // dar

// ... sync ...

// UPDATE 组（line 1165-1169）
if (using_amp && n_cfg) cudaGraphLaunch(n_cfg, s_up);       // cast_fg (AMP)
if (n_far) cudaGraphLaunch(n_far, s_up);                    // far
if (n_accum) cudaGraphLaunch(n_accum, s_up);               // accum
if (n_ncg) cudaGraphLaunch(n_ncg, s_up);                   // ncg ← 在这里！
if (n_sc) cudaGraphLaunch(n_sc, s_up);                     // sc
```

`n_ncg` 在 UPDATE 组，在 `far + accum` 之后。CPU 的 `idx_ncg` 对应 `n_ncg`，应留在原地。

### 3. batch_size 移动 — 不必要

GPU 路径不在循环内设置 batch_size（由 `init_variant_scalars` 在处理循环外完成）。CPU 因为不走 `init_variant_scalars`，需显式设置。这是 graph 级差异，移动它不会改善对齐，反而引入无意义的变动。

---

## 最终方案

### 只改 3 处，4 段代码

#### 改动 1：Batch 0 预传输 — clear 后置

**当前**（`deep_learning_task.cpp` L1486-1491）：
```cpp
if (idx_clear >= 0) launch(idx_clear);

ts->wait_buffer_readable(0);
launch(idx_xfer_a);
ts->set_buffer_readable(0, false);
ts->set_buffer_writeable(0, true);
```

**改为**：
```cpp
ts->wait_buffer_readable(0);
launch(idx_xfer_a);
ts->set_buffer_readable(0, false);
ts->set_buffer_writeable(0, true);

if (idx_clear >= 0) launch(idx_clear);
```

#### 改动 2：正常 batch 循环 — set_buffer 后置 + accum 后置

**当前**（L1524-1557）：
```cpp
for (int batch = 0; batch < batches - 1; ++batch) {
    bool from_a  = (batch % 2 == 0);
    int next_buf = from_a ? 1 : 0;

    launch(idx_zg);
    launch(from_a ? idx_fwd_a_nb : idx_fwd_b_nb);

    ts->wait_buffer_readable(next_buf);

    if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
    launch(idx_deep_nb);
    launch(from_a ? idx_xfer_b : idx_xfer_a);

    ts->set_buffer_readable(next_buf, false);       // ← 移走
    ts->set_buffer_writeable(next_buf, true);       // ← 移走

    if (!frozen) launch(from_a ? idx_bwd_a_nb : idx_bwd_b_nb);
    launch(idx_dar);

    if (local_bs_ptr) *static_cast<int32_t*>(local_bs_ptr) = registry.get_local_batch_size();
    int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);
    if (idx_accum_nb >= 0) launch(idx_accum_nb);    // ← 移到 far 之后

    if (is_step_by_batch_mode() || batch == 0) {
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batch);
    }
    launch(idx_far);
    if (idx_ncg >= 0) { launch(idx_ncg); }
    launch(idx_sc);
    launch(idx_opt);
    launch(idx_lars_fc);
    launch(idx_lars_fc2);
    launch(idx_lars_dc);
}
```

**改为**：
```cpp
for (int batch = 0; batch < batches - 1; ++batch) {
    bool from_a  = (batch % 2 == 0);
    int next_buf = from_a ? 1 : 0;

    // --- zg + fwd, 然后 wait + xfer（与 GPU 阶段1-2 对齐）---
    launch(idx_zg);
    launch(from_a ? idx_fwd_a_nb : idx_fwd_b_nb);

    ts->wait_buffer_readable(next_buf);

    // --- xfer + deep（与 GPU 阶段3-4 对齐：GPU 顺序为 wait → xfer → deep）---
    // 注意：CPU 必须保留 loss memset！
    // CPU 的 softmax_ce_fwd_inner 使用 *loss += ... 逐样本累加，
    // 内核不负责清零。若删掉，上一 batch 的 loss 残留会导致数值错误。
    // GPU 可删是因为 GPU 用 loss_partial 中间缓冲 + reduction 直接写入（=），
    // CPU 没有这层间接，直接对最终 buffer 做 +=。
    if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
    launch(from_a ? idx_xfer_b : idx_xfer_a);
    launch(idx_deep_nb);

    // --- first_bwd + dar（与 GPU 阶段6 对齐）---
    if (!frozen) launch(from_a ? idx_bwd_a_nb : idx_bwd_b_nb);
    launch(idx_dar);

    // --- lr（与 GPU 阶段7 对齐）---
    if (is_step_by_batch_mode() || batch == 0) {
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batch);
    }

    // --- set_buffer（与 GPU 阶段9 对齐，从 xfer 之后移至此）---
    ts->set_buffer_readable(next_buf, false);
    ts->set_buffer_writeable(next_buf, true);

    // --- far + accum + ncg + sc（与 GPU 阶段10 对齐，accum 从 far 之前移到 far 之后）---
    if (local_bs_ptr) *static_cast<int32_t*>(local_bs_ptr) = registry.get_local_batch_size();
    launch(idx_far);
    int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);
    if (idx_accum_nb >= 0) launch(idx_accum_nb);
    if (idx_ncg >= 0) { launch(idx_ncg); }
    launch(idx_sc);

    // --- opt + lars（与 GPU 阶段12 对齐）---
    launch(idx_opt);
    launch(idx_lars_fc);
    launch(idx_lars_fc2);
    launch(idx_lars_dc);
}
```

#### 改动 3：最后 batch — accum 后置

**当前**（L1559-1589）：
```cpp
{
    bool last_a = ((batches - 1) % 2 == 0);
    ctx.set_memory_plan(variant_memory_plans_[v_last].get());

    launch(idx_zg);
    launch(last_a ? idx_fwd_a_lb : idx_fwd_b_lb);

    if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
    launch(idx_deep_lb);

    if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
    launch(idx_dar);

    if (last_bs_ptr) *static_cast<int32_t*>(last_bs_ptr) = registry.get_last_train_batch_size();
    int32_t idx_accum_lb = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST, v_last);
    if (idx_accum_lb >= 0) launch(idx_accum_lb);   // ← 移到 far 之后

    if (is_step_by_batch_mode() || batches == 1) {
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batches - 1);
    }
    launch(idx_far);
    if (idx_ncg >= 0) { launch(idx_ncg); }
    launch(idx_sc);
    launch(idx_opt);
    launch(idx_lars_fc);
    launch(idx_lars_fc2);
    launch(idx_lars_dc);

    ctx.set_memory_plan(active_memory_plan_);
}
```

**改为**：
```cpp
{
    bool last_a = ((batches - 1) % 2 == 0);
    ctx.set_memory_plan(variant_memory_plans_[v_last].get());

    // --- zg + fwd ---
    launch(idx_zg);
    launch(last_a ? idx_fwd_a_lb : idx_fwd_b_lb);

    // --- deep（最后 batch 无 xfer）---
    // 注意：CPU 必须保留 loss memset！与上同理。
    if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
    launch(idx_deep_lb);

    // --- first_bwd + dar ---
    if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
    launch(idx_dar);

    // --- lr ---
    if (is_step_by_batch_mode() || batches == 1) {
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batches - 1);
    }

    // --- far + accum + ncg + sc（accum 移到 far 之后）---
    if (last_bs_ptr) *static_cast<int32_t*>(last_bs_ptr) = registry.get_last_train_batch_size();
    launch(idx_far);
    int32_t idx_accum_lb = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST, v_last);
    if (idx_accum_lb >= 0) launch(idx_accum_lb);
    if (idx_ncg >= 0) { launch(idx_ncg); }
    launch(idx_sc);

    // --- opt + lars ---
    launch(idx_opt);
    launch(idx_lars_fc);
    launch(idx_lars_fc2);
    launch(idx_lars_dc);

    ctx.set_memory_plan(active_memory_plan_);
}
```

#### 改动 4：batches == 1 分支 — accum 后置

**当前**（L1493-1522）：
```cpp
if (batches == 1) {
    launch(idx_zg);
    launch(idx_fwd_a_nb);
    if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
    launch(idx_deep_nb);
    if (!frozen) launch(idx_bwd_a_nb);
    launch(idx_dar);

    if (local_bs_ptr) *static_cast<int32_t*>(local_bs_ptr) = registry.get_local_batch_size();
    int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);
    if (idx_accum_nb >= 0) launch(idx_accum_nb);    // ← 移到 far 之后

    *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(0);
    launch(idx_far);
    if (idx_ncg >= 0) { launch(idx_ncg); }
    launch(idx_sc);
    launch(idx_opt);
    launch(idx_lars_fc);
    launch(idx_lars_fc2);
    launch(idx_lars_dc);
    ...
}
```

**改为**：
```cpp
if (batches == 1) {
    // --- zg + fwd ---
    launch(idx_zg);
    launch(idx_fwd_a_nb);

    // --- deep ---
    // 注意：CPU 必须保留 loss memset！与上同理。
    if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
    launch(idx_deep_nb);

    // --- first_bwd + dar ---
    if (!frozen) launch(idx_bwd_a_nb);
    launch(idx_dar);

    // --- lr ---
    *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(0);

    // --- far + accum + ncg + sc（accum 移到 far 之后）---
    if (local_bs_ptr) *static_cast<int32_t*>(local_bs_ptr) = registry.get_local_batch_size();
    launch(idx_far);
    int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);
    if (idx_accum_nb >= 0) launch(idx_accum_nb);
    if (idx_ncg >= 0) { launch(idx_ncg); }
    launch(idx_sc);

    // --- opt + lars ---
    launch(idx_opt);
    launch(idx_lars_fc);
    launch(idx_lars_fc2);
    launch(idx_lars_dc);
    ...
}
```

---

## 修改总结

| 代码段 | 改动 | 对应 GPU 阶段 |
|--------|------|:-----------:|
| Batch 0 预传输 | `clear` 从 `set_buffer` 前移到后 | 阶段9之后 |
| 正常 batch 循环 | `set_buffer` 从 xfer 后移到 lr 后；xfer 移到 deep 之前 | 阶段3-4, 9 |
| 正常 batch 循环 | `accum` 从 `far` 前移到 `far` 后 | 阶段10 |
| 最后 batch | `accum` 从 `far` 前移到 `far` 后 | 阶段10 |
| batches==1 分支 | `accum` 从 `far` 前移到 `far` 后 | 阶段10 |

**不改的项**（与三个方案的区别）：

| 项 | 不改的原因 |
|----|-----------|
| loss memset | CPU 内核使用 `*loss +=` 累加，不负责清零；GPU 用 `loss_partial` + reduction 直接写入，设计不同 |
| ncg 位置 | 对应 GPU 的 `n_ncg`（在 UPDATE 组），不应移到 first_bwd 组 |
| batch_size 位置 | GPU 由 `init_variant_scalars` 在循环外处理，CPU 是 graph 级差异，移动无益 |
| 新增 AMP 占位 | CPU 不支持 AMP，添加空注释是代码噪音 |

---

## 安全性分析

所有改动仅涉及 `launch()` 调用顺序的排列，且 CPU 是串行执行，因此：

- **set_buffer 后置**：唯一影响是 preprocessor 收到 buffer 可用的信号稍晚。由于 preprocessor 在独立线程运行，且 CPU 是串行的，preprocessor 的处理时间远小于 batch 计算时间，延迟几个 launch 调用不会造成 starvation。
- **accum 后置到 far 之后**：`accum` 和 `far` 操作不同内存区域（metrics vs gradients），顺序不影响正确性。
- **clear 后置**：`clear` 清零的是跨 batch 累积的 metrics，放在 batch 0 预传输之后、第一次 accum 之前执行即可，后置不影响正确性。

**不改变任何 graph 内部的执行逻辑，不改变任何数据依赖，仅调整 `launch()` 的相对顺序。**