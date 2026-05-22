# FC 层与 FC 算子最终修复方案

> 基于 FIXFC.md、FIXFC2.md、FIXFC3.md 的逐条分析，以及对 fc_op.cpp、compiler.cpp、
> layer_descriptor_registry.cpp、test_fc_fwd_bwd.cpp、test_fc_fwd_bwd.py 的完整审查，
> 以下是经核实的最终修复方案。

---

## 一、编译器侧的真实 I/O 约定（已全部核实）

### 1.1 张量描述符索引 (`infer_fc_tensors`)

[layer_descriptor_registry.cpp:247-287](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L247-L287)

| idx | 名称 | Region | DType | 备注 |
|-----|------|--------|-------|------|
| 0 | fc_weight | W_FC_WEIGHT | FP32 | 始终存在 |
| 1 | fc_bias | W_FC_BIAS | FP32 | no-bias 时 placeholder（Shape={1,1,1,1}, effective=0） |
| 2 | fc_output | F_FEATURE_* | varies | 正向输出 Y |
| 3 | fc_weight_grad | G_FC_WEIGHT | FP32 | 始终存在 |
| 4 | fc_bias_grad | G_FC_BIAS | FP32 | no-bias 时 placeholder |
| 5 | fc_amp_w_fp16 | A_FC_WEIGHT | FP16 | 非 AMP 时 placeholder |
| 6 | fc_amp_g_fp16 | G_FC_WEIGHT_FP16 | FP16 | 非 AMP 时 placeholder |

### 1.2 前向 graph (`build_fc_forward`)

[layer_descriptor_registry.cpp:289-298](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L289-L298)

```
build_fc_forward
  input_indices  = {0, 1}   → W, B
  output_indices = {2}      → Y
```

[compiler.cpp:1053-1055](file:///r:/renaissance/src/graph/compiler.cpp#L1053-L1055) 在开头插入 `prev_output_id`（前一层输出，即 X）：

**最终 FWD input_ids = {X, W, B}，恒为 3 个元素。**

### 1.3 反向 graph (`build_fc_backward`)

[layer_descriptor_registry.cpp:300-309](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L300-L309)

```
build_fc_backward
  input_indices  = {0, 2}   → W, Y_output
  output_indices = {3, 4}   → dW, dB
```

[compiler.cpp:1135-1146](file:///r:/renaissance/src/graph/compiler.cpp#L1135-L1146)：
- 在开头 `insert(begin, prev_grad_id)` → dY
- 在末尾 `push_back(layer_input_ids[l])` → X
- 在 outputs 开头 `insert(begin, layer_input_ids[l])` → X (in-place)

**最终 BWD input_ids = {dY, W, Y_output, X}，恒为 4 个元素。**
**最终 BWD output_ids = {X, dW, dB}，恒为 3 个元素。**

### 1.4 核心结论

无论 `FCParams::bias` 是 `true` 还是 `false`，compiler 构建的 graph I/O 数量
和布局完全一样。区分有无 bias 的唯一可靠来源是 `p->bias`。

---

## 二、当前代码真实存在的问题（已逐行核实）

### 🔴 P0-1：`has_bias` 判断逻辑错误（6 个入口全部受影响）

**根因**：用 `input_ids.size() >= N` 推断 bias 是否启用，但在真实网络中
`input_ids.size()` 是固定值（FWD=3, BWD=4），始终返回 `true`。

**受影响位置：**

| # | 函数 | 文件:行号 | 当前代码 |
|---|------|-----------|----------|
| 1 | `launch_fc_amp_fwd_cuda` | [fc_op.cpp:179](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L179) | `bool has_bias = (node.input_ids.size() >= 3);` |
| 2 | `launch_fc_amp_bwd_cuda` | [fc_op.cpp:290](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L290) | `bool has_bias = (node.input_ids.size() >= 4);` |
| 3 | `launch_fc_fwd_cuda` | [fc_op.cpp:419](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L419) | `bool has_bias = (node.input_ids.size() >= 3);` |
| 4 | `launch_fc_bwd_cuda` | [fc_op.cpp:470](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L470) | `bool has_bias = (node.input_ids.size() >= 4);` |
| 5 | `launch_fc_fwd_cpu` | [fc_op.cpp:549](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L549) | `bool has_bias = (op_ctx->num_inputs >= 3);` |
| 6 | `launch_fc_bwd_cpu` | [fc_op.cpp:587](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L587) | `bool has_bias = (op_ctx->num_inputs >= 4);` |

**后果**：
- 无 bias 时 FWD：仍会读取 `input_ids[2]`（B 的 placeholder）传入 kernel/cuDNN graph
- 无 bias 时 BWD：仍会调用 `launch_fc_bwd_db_*_kernel`，向 `output_ids[2]`
  （dB placeholder，`effective=0`）写入数据
- 违反用户明确约束："无 bias 情形不计算 bias 梯度"

**修复方案**：全部 6 处统一改为从 FCParams 读取：

```cpp
const auto* p = std::get_if<FCParams>(&node.params.data);
TR_CHECK(p != nullptr, ValueError, "FC_* missing FCParams");
bool has_bias = p->bias;  // ✅ 从参数读取，不再从 I/O 数量推断
```

同时更新 TR_CHECK 为固定常量：
```cpp
// FWD: 始终 3 输入，1 输出
TR_CHECK(node.input_ids.size() >= 3, ShapeError,
         "FC_FWD requires at least 3 inputs. Got " << node.input_ids.size());
TR_CHECK(node.output_ids.size() >= 1, ShapeError,
         "FC_FWD requires at least 1 output");

// BWD: 始终 4 输入，3 输出
TR_CHECK(node.input_ids.size() >= 4, ShapeError,
         "FC_BWD requires at least 4 inputs. Got " << node.input_ids.size());
TR_CHECK(node.output_ids.size() >= 3, ShapeError,
         "FC_BWD requires at least 3 outputs. Got " << node.output_ids.size());
```

**对应行号修改明细：**

| 原行号 | 修改 |
|--------|------|
| 179 | `bool has_bias = p->bias;` |
| 180-182 | TR_CHECK 改为固定 `>= 3` / `>= 1` |
| 290 | `bool has_bias = p->bias;` |
| 293-296 | TR_CHECK 改为固定 `>= 4` / `>= 3` |
| 419 | `bool has_bias = p->bias;` |
| 420-422 | TR_CHECK 改为固定 `>= 3` / `>= 1` |
| 470 | `bool has_bias = p->bias;` |
| 473-476 | TR_CHECK 改为固定 `>= 4` / `>= 3` |
| 549 | `bool has_bias = p->bias;` |
| 550-551 | TR_CHECK 改为固定 `>= 3` / `>= 1` |
| 587 | `bool has_bias = p->bias;` |
| 590-593 | TR_CHECK 改为固定 `>= 4` / `>= 3` |

---

### 🔴 P0-2：BWD 中 X 的索引逻辑脆弱（3 处）

**根因**：`x_idx = has_bias ? 3 : 2` 隐含假设了固定的 input 布局。
P0-1 修复后 `has_bias` 可变为 `false`，此时 `x_idx = 2`，会把
`input_ids[2]` 当作 X 读取。在真实网络中 `input_ids[2]` 是 **Y_output**
（正向输出），不是 X。

**后果**：`dW = dY^T @ Y_output`，权重梯度完全错误。

**受影响位置：**

| # | 函数 | 文件:行号 | 当前代码 |
|---|------|-----------|----------|
| 1 | `launch_fc_amp_bwd_cuda` | [fc_op.cpp:291](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L291) | `int x_idx = has_bias ? 3 : 2;` |
| 2 | `launch_fc_bwd_cuda` | [fc_op.cpp:471](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L471) | `int x_idx = has_bias ? 3 : 2;` |
| 3 | `launch_fc_bwd_cpu` | [fc_op.cpp:588](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L588) | `int x_idx = has_bias ? 3 : 2;` |

**修复方案**：X 永远由 compiler 的 `input_ids.push_back(layer_input_ids[l])`
添加到 input 列表末尾。使用动态索引：

```cpp
// X 总是 compiler push_back 添加的最后一个输入
int x_idx = static_cast<int>(node.input_ids.size()) - 1;
```

此写法的好处：
- 测试 no-bias 时 `input_ids = {dY, W, X}`（3 元素）→ `x_idx = 2` ✅
- 真实网络 `input_ids = {dY, W, Y_output, X}`（4 元素）→ `x_idx = 3` ✅
- 自文档化：X 总是最后一个输入
- 兼容未来可能的 I/O 布局变化

---

### 🔴 P0-3：测试 FWD graph 在 no-bias 时缺 B

**位置**：[test_fc_fwd_bwd.cpp:282-286](file:///r:/renaissance/tests/op/test_fc_fwd_bwd.cpp#L282-L286)

**当前代码：**
```cpp
if (cfg.has_bias) {
    g_fwd.append(fwd_op, {d_x.id, d_w.id, d_b.id}, {d_y.id}, fc_op_params);
} else {
    g_fwd.append(fwd_op, {d_x.id, d_w.id}, {d_y.id}, fc_op_params);  // ← 缺 B
}
```

**根因**：真实 compiler 始终生成 3 个 FWD 输入 `{X, W, B}`。
`build_fc_forward` 的 `input_indices = {0, 1}` 始终包含 B，
placeholder 的 ID 不会被 `map_indices` 过滤。测试 no-bias 时只给 2 个输入，
测试在不同于真实环境的 I/O 结构下运行。

**修复方案**：无论 bias 与否，FWD 始终构造 3 个输入：

```cpp
// has_bias 由 fc_params.bias 控制，算子内部决定是否使用 B
g_fwd.append(fwd_op, {d_x.id, d_w.id, d_b.id}, {d_y.id}, fc_op_params);
```

删除 `if (cfg.has_bias)` 分支。

---

### 🔴 P0-4：测试 BWD graph 的 I/O 约定与 compiler 不一致

**位置**：[test_fc_fwd_bwd.cpp:294-300](file:///r:/renaissance/tests/op/test_fc_fwd_bwd.cpp#L294-L300)

**当前代码：**
```cpp
if (cfg.has_bias) {
    g_bwd.append(bwd_op, {d_dy.id, d_w.id, d_b.id, d_x.id},
                 {d_dx.id, d_gw.id, d_gb.id}, fc_op_params);
} else {
    g_bwd.append(bwd_op, {d_dy.id, d_w.id, d_x.id},
                 {d_dx.id, d_gw.id}, fc_op_params);              // ← 缺 Y, dB
}
```

**与 compiler 真实约定的差异：**

| 项目 | 测试（当前） | compiler 真实约定 |
|------|-------------|-------------------|
| `input_ids[2]` | `d_b.id`（bias 张量） | `Y_output`（正向输出） |
| no-bias 输入数 | 3 个 | 4 个 |
| no-bias 输出数 | 2 个 | 3 个（dB 占位仍存在） |
| `output_ids[0]` | 独立 `d_dx` | 与 `X` 同 ID（in-place） |

虽然当前测试通过了（launch 函数完全不读取 `input_ids[2]`），但测试在不同
的 I/O 结构下运行，无法保证真实 compiler 路径的正确性。

**修复方案**：测试 BWD graph 与 compiler 完全对齐，始终 4 输入 3 输出：

```cpp
// d_y 已在前面分配（FWD 输出），作为 Y_output 传入 BWD
// has_bias 由 fc_params.bias 控制，算子内部决定是否计算 db
g_bwd.append(bwd_op,
             {d_dy.id, d_w.id, d_y.id, d_x.id},   // {dY, W, Y_output, X}
             {d_dx.id, d_gw.id, d_gb.id},          // {dX, dW, dB}
             fc_op_params);
```

删除 `if (cfg.has_bias)` 分支。

> **注意**：`d_dx` 在测试中仍独立分配（非 in-place），这没问题。
> 算子把梯度写入 `d_dx` 的内存而不是覆盖 `d_x`。这只是测试与真实网络在
> 内存布局上的差异，不影响数值正确性验证。

---

### 🟡 P1-1：测试缺少 dW / dB 数值验证（假阳性风险）

**位置**：[test_fc_fwd_bwd.cpp:341-364](file:///r:/renaissance/tests/op/test_fc_fwd_bwd.cpp#L341-L364)

**当前验证范围**：仅对比 FWD y 和 BWD dx 的 MSE。
dW 和 dB 从未被验证。当前所有 6 项测试"通过"是假阳性——即使 dW/dB 计算出错，
测试也不会失败。

#### Step 1 — 修改 test_fc_fwd_bwd.py，添加 dW/dB 参考值导出

在 `generate()` 函数中，BWD 计算段末尾添加（约第 74 行后）：

```python
# ── dW = dY^T @ X ──
# dY: [B, O], X: [B, I]
# dW = dY^T @ X, 结果 [O, I]，即 out_features × in_features
dW_2d = torch.mm(dY_2d.t(), X_2d)
dW = dW_2d.view(out_features, 1, 1, in_features).contiguous()

# ── dB = sum(dY, dim=0) ──
# dB: [O]，即沿 batch 维度求和
if bias:
    dB_2d = dY_2d.sum(dim=0)  # [O]
else:
    dB_2d = torch.zeros(out_features, dtype=torch.float32)
dB = dB_2d.view(1, 1, 1, out_features).contiguous()

# ── 导出 ──
save_tsr(os.path.join(ws, f'dw_ref_fwd_bwd{suffix}.tsr'),
         [dW.cpu().numpy().astype(np_dtype)], compress=False)
save_tsr(os.path.join(ws, f'db_ref_fwd_bwd{suffix}.tsr'),
         [dB.cpu().numpy().astype(np.float32)], compress=False)
```

并更新 `file_desc`（约第 101-108 行）：

```python
file_desc = (
    f"  x:   x_fwd_bwd{suffix}.tsr\n"
    f"  w:   w_fwd_bwd{suffix}.tsr\n"
    f"  b:   b_fwd_bwd{suffix}.tsr\n"
    f"  y:   y_ref_fwd_bwd{suffix}.tsr\n"
    f"  dy:  dy_fwd_bwd{suffix}.tsr\n"
    f"  dx:  dx_ref_fwd_bwd{suffix}.tsr\n"
    f"  dw:  dw_ref_fwd_bwd{suffix}.tsr\n"
    f"  db:  db_ref_fwd_bwd{suffix}.tsr\n"
)
```

#### Step 2 — 修改 test_fc_fwd_bwd.cpp，加载并验证

在 ref 数据加载处（约第 226 行后）添加：

```cpp
Tensor h_dw = Tensor::load_tensor(ws + "/dw_ref_fwd_bwd" + tsr_sfx + ".tsr");
Tensor h_db = Tensor::load_tensor(ws + "/db_ref_fwd_bwd" + tsr_sfx + ".tsr");
```

在验证循环中（约第 363 行后，`d_dx` 验证后）添加：

```cpp
// 验证 dW
Tensor h_dw_out = task.fetch_from_rank(d_gw, rank);
double mse_dw = is_amp ? compute_mse_fp16(h_dw_out, h_dw)
                       : compute_mse_fp32(h_dw_out, h_dw);
max_mse = (mse_dw > max_mse) ? mse_dw : max_mse;
const double mse_dw_thr = is_amp ? 1e-3 : 1e-5;
std::cout << "  Rank " << rank << " dW MSE = " << std::scientific << mse_dw;
if (mse_dw > mse_dw_thr) { std::cout << "  FAIL"; all_pass = false; }
std::cout << std::endl;

// 验证 dB（bias 梯度永远是 FP32）
Tensor h_db_out = task.fetch_from_rank(d_gb, rank);
double mse_db = compute_mse_fp32(h_db_out, h_db);
max_mse = (mse_db > max_mse) ? mse_db : max_mse;
const double mse_db_thr = 1e-5;
std::cout << "  Rank " << rank << " dB MSE = " << std::scientific << mse_db;
if (mse_db > mse_db_thr) { std::cout << "  FAIL"; all_pass = false; }
std::cout << std::endl;
```

---

### 🟡 P1-2：`build_fc_backward` 的 Y_output 是冗余输入（可选优化）

**位置**：[layer_descriptor_registry.cpp:305](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L305)

**当前代码：**
```cpp
n.input_indices  = {0, 2};   // weight, output(Y)
```

**问题**：BWD 算子计算 `db = reduce_sum(dY)`、`dW = X @ dY^T`、`dX = W @ dY`，
完全不需要正向输出 Y。Y 作为 BWD 输入是多余的。

**推荐修改**：
```cpp
n.input_indices  = {0, 1};   // weight, bias
```

> **实施验证**：`tensor_ids[1]` 已被 `build_fc_forward` 的 `input_indices = {0, 1}`
> 验证过对应 bias，无需额外检查。`map_indices` 对 `{0, 1}` 的处理与 FWD 完全一致，
> compiler 不会因索引从 `{0, 2}` 变为 `{0, 1}` 产生任何差异。

此修改的收益：
1. 去除无用的 Y_output 输入，graph 更干净
2. B 显式在 BWD 输入中，满足用户"bias 张量必须作为输入输出之一"的约束
   （B 在 inputs[2]，dB 在 outputs[2]，双重满足）
3. compiler 处理后：`input_ids = {dY, W, B, X}`（4 个，X 在末尾）
4. 测试 BWD graph 中原有的 `{d_dy, d_w, d_b, d_x}` convention 直接复用，
   无需额外添加 `d_y`

> **注意**：如果采纳此优化，测试 P0-4 的 BWD graph 应改为
> `{d_dy.id, d_w.id, d_b.id, d_x.id}` 而非
> `{d_dy.id, d_w.id, d_y.id, d_x.id}`。

---

### 🟢 P2：cuBLAS handle 资源泄漏

**位置**：[fc_op.cpp:333-345](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L333-L345)

**当前代码：**
```cpp
static std::unordered_map<cudaStream_t, cublasHandle_t> s_cublas_handles;
// cublasCreate 后从未 cublasDestroy
```

**修复方案**：添加 RAII 清理对象，利用 static 对象析构时自动销毁：

在 `s_cublas_handles` 声明后添加：
```cpp
static struct CublasHandleCleanup {
    ~CublasHandleCleanup() {
        for (auto& kv : s_cublas_handles) {
            if (kv.second) cublasDestroy(kv.second);
        }
        s_cublas_handles.clear();
    }
} s_cublas_cleanup;
```

> **已知局限**：static 析构函数在程序退出时调用，可能晚于 CUDA runtime 卸载。
> 若 CUDA context 已被销毁，`cublasDestroy` 可能产生 warning（非 crash）。
> 更理想的方案是让 `DeviceContext` 管理 handle 生命周期，但成本较高，
> 当前简化为 static RAII（严格优于现状的零清理），后续可在 DeviceContext
> 重构时统一处理。

---

## 三、完整修改清单

| 优先级 | 文件 | 行号 | 修改内容 |
|--------|------|------|----------|
| 🔴 P0 | fc_op.cpp | 179 | `has_bias = p->bias;`（AMP FWD） |
| 🔴 P0 | fc_op.cpp | 180-182 | TR_CHECK 固定化 |
| 🔴 P0 | fc_op.cpp | 290 | `has_bias = p->bias;`（AMP BWD） |
| 🔴 P0 | fc_op.cpp | 291 | `x_idx = size() - 1;` |
| 🔴 P0 | fc_op.cpp | 293-296 | TR_CHECK 固定化 |
| 🔴 P0 | fc_op.cpp | 419 | `has_bias = p->bias;`（FP32 FWD） |
| 🔴 P0 | fc_op.cpp | 420-422 | TR_CHECK 固定化 |
| 🔴 P0 | fc_op.cpp | 470 | `has_bias = p->bias;`（FP32 BWD） |
| 🔴 P0 | fc_op.cpp | 471 | `x_idx = size() - 1;` |
| 🔴 P0 | fc_op.cpp | 473-476 | TR_CHECK 固定化 |
| 🔴 P0 | fc_op.cpp | 549 | `has_bias = p->bias;`（CPU FWD） |
| 🔴 P0 | fc_op.cpp | 550-551 | TR_CHECK 固定化 |
| 🔴 P0 | fc_op.cpp | 587 | `has_bias = p->bias;`（CPU BWD） |
| 🔴 P0 | fc_op.cpp | 588 | `x_idx = size() - 1;` |
| 🔴 P0 | fc_op.cpp | 590-593 | TR_CHECK 固定化 |
| 🔴 P0 | test_fc_fwd_bwd.cpp | 282-286 | FWD 始终 3 输入，去掉分支 |
| 🔴 P0 | test_fc_fwd_bwd.cpp | 294-300 | BWD 始终 4 输入 3 输出，去掉分支 |
| 🟡 P1 | test_fc_fwd_bwd.py | 74 行后 | 添加 dW/dB 参考值生成 |
| 🟡 P1 | test_fc_fwd_bwd.cpp | 226 行后 | 加载 dW/dB 参考 |
| 🟡 P1 | test_fc_fwd_bwd.cpp | 363 行后 | 添加 dW/dB MSE 验证 |
| 🟡 P1 | layer_descriptor_registry.cpp | 305 | `input_indices = {0, 1}`（可选） |
| 🟢 P2 | fc_op.cpp | 333 行后 | 添加 CublasHandleCleanup RAII |

---

## 四、修改顺序建议

1. **第一批（P0）**：P0-1、P0-2、P0-3 — `has_bias`、`x_idx`、TR_CHECK
   - 修复后算子接口在真实网络中正确工作
2. **第二批（P0）**：P0-4、P0-5 — 测试 FWD/BWD graph 对齐 compiler
   - 确保测试在真实 I/O 结构下验证
3. **第三批（P1）**：测试 dW/dB 验证 + 编译运行 6 项测试
   - 消除假阳性，全面验证
4. **第四批（P1/P2）**：`build_fc_backward` 优化 + cuBLAS 清理
   - 架构优化和资源管理，可延后

---

## 五、不涉及的领域

- **FCBNReLU / FCReLU / Conv 等融合/高级算子**：用户明确指示这些尚未正式开发，
  本方案严格限定在独立 FC 算子范围内
- **MemoryPlan 的 W/G/M/V 校验逻辑**：当前逻辑已经正确，测试的 Region 分配
  已满足校验要求，无需修改