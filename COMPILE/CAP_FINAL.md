# CAP_FINAL — 图集（GraphAtlas）：预演去重三阶段架构

## 版本

V1.0  2026-05-13

## 摘要

本文档是对 CAP → CAP1 → CAP2 → CAP3 四轮讨论的最终综合方案。"图集"（GraphAtlas）解决两个核心问题：(1) **总共需要捕获多少个 CUDA Graph**；(2) **运行时不同情景下该调用哪个图**。

核心设计：**三阶段生命周期**——编译期声明逻辑归属 → 预演期遍历去重捕获 → 运行期纯索引调用。去重键使用 `ShapeId`（输入 shape 四元组），shape 无关图自动归并，shape 相关图仅按唯一 shape 捕获。全部捕获在 MLPerf 正式计时前完成。

---

## 一、问题定义

### 1.1 背景

COMP_ULTIMATE.md 定义了 6 个输入 shape 变体，15 个 GraphId（子图桶）。ComputationGraph 是 15 张子图的容器，CUDA Graph 缓存键为 `(ComputationGraph*, MemoryPlan*)`。

这个设计在 CUDA Graph 捕获层面有三个问题：

| 问题 | 描述 |
|------|------|
| **粒度太粗** | 缓存键是整图级别。shape 无关图（TRANSFER、COMM、OPTIMIZER）随 MemoryPlan 变化而被重复捕获——但它们根本不依赖 H×W 或 N |
| **缺少显式映射** | Executor 只靠 `v.train` / `v.inference` 两个指针选图，15 张子图中哪些用 train CG、哪些用 inference CG、哪些共用——没有声明 |
| **无物理去重** | 两个不同 MemoryPlan（如 val_base 和 train_lowres）即使 shape 完全相同，指针不同 → `(CG*, MP*)` 键不同 → 不会自动复用 |

### 1.2 cuDNN 预热成本约束

cuDNN 预热 + CUDA Graph 捕获极为耗时。如果在训练过程中触发（首次使用时捕获），预热时间会被计入 MLPerf 时钟。因此：**所有图的预热和捕获必须在正式训练前一次性完成。**

### 1.3 子图分类

15 个 GraphId 按对输入 shape 的敏感性分为两类：

```
shape 无关（7 张）                    shape 相关（8 张）
────────────────────────────        ────────────────────────────
TRANSFER_A      H2D 传输            FIRST_FWD_A     首层正向
TRANSFER_B      H2D 传输            FIRST_FWD_B     首层正向
FIRST_COMM      首层 AllReduce      DEEP_FWD_BWD    深层正反向
DEEP_COMM       深层 AllReduce      FIRST_BWD       首层反向
STATS_COMM      BN 统计量通信       INF_MAIN_A      主模型推理
OPTIMIZER       权重更新            INF_MAIN_B      主模型推理
EMA_UPDATE      EMA 更新            INF_EMA_A       EMA 推理
                                    INF_EMA_B       EMA 推理
```

shape 无关图：kernel 参数与 H×W、N 无关——传输是 raw memcpy，通信是 AllReduce 全量 buffer，优化器操作权重张量（形状固定，如 `[1,1,C_out,C_in/k]`）。**一份 MemoryPlan 下捕获的结果，所有变体通用。**

shape 相关图：kernel 可能使用 cuDNN descriptor（含 N/H/W/stride），不同输入 shape 需要不同捕获。

---

## 二、三阶段生命周期

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase A: 编译期（Compiler::compile）——不计 MLPerf 时间               │
│                                                                      │
│  输入: ArchPlan + 6 个 Config                                        │
│  输出: 6 个 MemoryPlan + 2 个 ComputationGraph + GraphAtlas           │
│        GraphAtlas 仅填入 (cg_ptr, mp_ptr, shape_id) —— 逻辑归属声明   │
│        不调用任何 CUDA API                                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase B: 预演捕获期（pre_capture）——不计 MLPerf 时间                 │
│                                                                      │
│  输入: GraphAtlas（含 cg/mp/shape_id）                                │
│  过程: 遍历 Atlas 每一格 → 构建去重键 (cg, gid, shape_id)             │
│        → 已存在则复用 → 不存在则调用 CaputredGraph::capture 捕获新图  │
│  输出: vector<CapturedGraph>（去重后，像 Python set 不重复）          │
│        GraphAtlas 的每格填入 captured_idx                             │
│        cuDNN 预热在此阶段完成                                         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase C: 运行期（Executor）——MLPerf 计时开始                         │
│                                                                      │
│  Executor::launch(gid):                                              │
│    captured_idx = atlas_.slot(active_variant, gid).captured_idx      │
│    captured_graphs_[captured_idx].launch(stream)                     │
│                                                                      │
│  纯数组索引 + 指针调用。零 hash，零分支，零捕获，零 cuDNN 调用。       │
└──────────────────────────────────────────────────────────────────────┘
```

**关键约束**：Phase A 和 Phase B 必须在 `MLPERF_LOG_START()` 之前完成。Phase C 中不允许任何形式的 CUDA Graph 捕获或 cuDNN 预热。

---

## 三、核心数据结构

### 3.1 ShapeId — 去重键的形状分量

```cpp
// include/renaissance/graph/shape_id.h

/// 输入 shape 标识：shape 相关图用于去重，shape 无关图恒为零
/// 用显式四元组替代 hash（K 的 shape_fp），语义直接，无需计算
struct ShapeId {
    int32_t n = 0, h = 0, w = 0, c = 0;

    bool operator==(const ShapeId& o) const noexcept {
        return n == o.n && h == o.h && w == o.w && c == o.c;
    }
};

struct ShapeIdHash {
    size_t operator()(const ShapeId& s) const {
        return (static_cast<uint64_t>(s.n) << 48) |
               (static_cast<uint64_t>(s.h) << 32) |
               (static_cast<uint64_t>(s.w) << 16) |
               static_cast<uint64_t>(s.c);
    }
};

/// shape 无关图专用：所有变体指向此值 → 必然碰撞 → 全局复用
constexpr ShapeId kShapeInvariant{0, 0, 0, 0};
```

#### ShapeId 与 K 的 shape_fp 对比

| | shape_fp（K） | ShapeId（最终采用） |
|---|---|---|
| 表达方式 | hash(DTensor[0].shape) 或 hash(所有 node 的 DTensor shape) | 显式 `{n, h, w, c}` |
| 可读性 | 16 进制哈希值，不可逆推 | 直接可读，知道是哪个 shape 组合 |
| 计算成本 | 需要遍历 node 列表 → hash | 直接从 Config 取，零成本 |
| 正确性保证 | hash 有碰撞风险（概率极低但非零） | 值相等，确定性无碰撞 |
| 充分性 | 过度精确：hash 了所有中间 DTensor | **充分且必要**：所有 DTensor shape 由 `(N,H,W,C)` + 固定层参数唯一确定 |

#### 为什么 ShapeId 是充分的

对任意 GraphId，其涉及的所有 DTensor 的 shape 是以下三项的确定性函数：
1. 输入 shape `(N, H, W, C)`
2. 层参数（卷积核大小、stride、padding 等——编译期固定，跨变体不变）
3. 网络拓扑（编译期固定）

因此，**输入 shape 相同 → 此 GraphId 涉及的所有 DTensor shape 相同 → 同一 Key → 应复用同一个 CUDA Graph**。ShapeId 直接捕获这个等价关系，不需要 DTensor 级别的 hash。

### 3.2 CapturedGraph — 后端无关的图抽象

```cpp
// include/renaissance/graph/captured_graph.h

/// 后端无关的"已捕获图"抽象
/// - CUDA:  内部持有 cudaGraphExec_t
/// - CPU:   内部持有有序函数指针序列（预留）
class CapturedGraph {
public:
    /// 从 ComputationGraph + MemoryPlan 在指定后端上捕获
    /// 内部执行 cuDNN 预热 + cudaGraphInstantiate
    static CapturedGraph capture(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  Device* device);

    /// 运行时调用
    void launch(cudaStream_t stream) const;

    /// 去重键：什么拓扑 + 哪张子图 + 什么输入形状
    struct Key {
        const ComputationGraph* cg;
        GraphId                 gid;
        ShapeId                 shape;

        bool operator==(const Key& o) const {
            return cg == o.cg && gid == o.gid && shape == o.shape;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h1 = reinterpret_cast<size_t>(k.cg) >> 4;
            size_t h2 = static_cast<size_t>(k.gid);
            size_t h3 = ShapeIdHash{}(k.shape);
            // hash_combine 风格：避免 XOR 相互抵消
            size_t h = h1;
            h ^= h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

private:
#ifdef USE_CUDA
    cudaGraphExec_t cuda_graph_ = nullptr;
#else
    std::vector<CpuOp> cpu_ops_;
#endif
};
```

### 3.3 GraphAtlas — 编译期声明 + 预演期索引

```cpp
// include/renaissance/graph/graph_atlas.h

class GraphAtlas {
public:
    static constexpr size_t kMaxVariants = 6;  // 6 个输入 shape 变体
    static constexpr size_t kMaxGraphIds =
        static_cast<size_t>(GraphId::COUNT);

    /// Atlas 中的一格
    struct Slot {
        const ComputationGraph* cg = nullptr;     // Phase A: Compiler 填入
        const MemoryPlan*       mp = nullptr;     // Phase A: Compiler 填入
        ShapeId                 shape_id{};       // Phase A: Compiler 填入
        int32_t  captured_idx = -1;               // Phase B: pre_capture 填入
    };

    /// Phase A: Compiler 声明逻辑归属
    Slot& slot(size_t v, GraphId gid) {
        return table_[v][static_cast<size_t>(gid)];
    }

    const Slot& slot(size_t v, GraphId gid) const {
        return table_[v][static_cast<size_t>(gid)];
    }

    /// Phase A: 构建 —— 填入 cg / mp / shape_id
    static GraphAtlas build(const Compiler::Result& compiled,
                             const std::array<ShapeId, 6>& variant_shapes);

    /// Phase C: Executor 运行时 —— O(1) 数组访问
    int32_t index(size_t v, GraphId gid) const {
        return table_[v][static_cast<size_t>(gid)].captured_idx;
    }

private:
    std::array<std::array<Slot, kMaxGraphIds>, kMaxVariants> table_;
};
```

---

## 四、Phase A：编译期 Atlas 构建

### 4.1 ShapeId 数组

Compiler 接收 6 个 Config，从中提取 6 个 ShapeId：

```cpp
// variant 0: train_base         ShapeId{N_std,  res_begin, res_begin, C}
// variant 1: train_last         ShapeId{N_last, res_begin, res_begin, C}
// variant 2: train_lowres       ShapeId{N_std,  res_end,   res_end,   C}
// variant 3: train_lowres_last  ShapeId{N_last, res_end,   res_end,   C}
// variant 4: val_base           ShapeId{N_val,  val_res,   val_res,   C}
// variant 5: val_last           ShapeId{N_vall, val_res,   val_res,   C}
```

### 4.2 构建逻辑

```cpp
// src/graph/graph_atlas.cpp

GraphAtlas GraphAtlas::build(const Compiler::Result& compiled,
                              const std::array<ShapeId, 6>& variant_shapes)
{
    GraphAtlas atlas;

    for (size_t v = 0; v < compiled.variants.size(); ++v) {
        for (size_t g = 0; g < kMaxGraphIds; ++g) {
            GraphId gid = static_cast<GraphId>(g);

            // ===== 适用性判定 =====
            bool is_train = is_train_graph(gid);     // DEEP_FWD_BWD, FIRST_*, COMM*, OPTIMIZER, EMA, TRANSFER
            bool is_infer = is_inference_graph(gid);  // INF_MAIN_*, INF_EMA_*
            bool is_xfer  = is_transfer_graph(gid);   // TRANSFER_A, TRANSFER_B

            bool is_train_var = (v <= 3);
            bool is_val_var   = (v >= 4);

            // val 变体不需要训练图
            if (is_val_var && is_train) continue;
            // train 变体不需要推理图（训练时不调用 INF_*）
            if (is_train_var && is_infer) continue;

            // ===== 适用：填入 Slot =====
            auto& s = atlas.slot(v, gid);

            if (is_train || is_xfer) {
                s.cg = compiled.variants[0].train;
            } else {
                s.cg = compiled.variants[0].inference;
            }

            if (is_shape_invariant(gid)) {
                // shape 无关图：全部变体指向 base MemoryPlan
                s.mp       = &compiled.variants[0].memory_plan;
                s.shape_id = kShapeInvariant;
            } else {
                // shape 相关图：各自用各自的 MemoryPlan
                s.mp       = &compiled.variants[v].memory_plan;
                s.shape_id = variant_shapes[v];
            }
        }
    }

    return atlas;
}
```

### 4.3 辅助判定函数

```cpp
static bool is_train_graph(GraphId gid) {
    switch (gid) {
        case GraphId::FIRST_FWD_A:   case GraphId::FIRST_FWD_B:
        case GraphId::DEEP_FWD_BWD:  case GraphId::FIRST_BWD:
        case GraphId::FIRST_COMM:    case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:    case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:    return true;
        default: return false;
    }
}

static bool is_inference_graph(GraphId gid) {
    switch (gid) {
        case GraphId::INF_MAIN_A:  case GraphId::INF_MAIN_B:
        case GraphId::INF_EMA_A:  case GraphId::INF_EMA_B:
            return true;
        default: return false;
    }
}

static bool is_transfer_graph(GraphId gid) {
    return gid == GraphId::TRANSFER_A || gid == GraphId::TRANSFER_B;
}

static bool is_shape_invariant(GraphId gid) {
    switch (gid) {
        case GraphId::TRANSFER_A:  case GraphId::TRANSFER_B:
        case GraphId::FIRST_COMM:  case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:  case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
            return true;
        default: return false;
    }
}
```

### 4.4 Atlas 填充结果（逻辑归属声明）

```
         │ TRANSFER_A  FIRST_FWD_A  DEEP_FWD_BWD  FIRST_BWD  ...  INF_MAIN_A  INF_EMA_A
─────────┼─────────────────────────────────────────────────────────────────────────────────
v=0 base │ train,MP[0] train,MP[0]  train,MP[0]   train,MP[0]           -           -
         │ shape={0}   shape=A       shape=A        shape=A
─────────┼─────────────────────────────────────────────────────────────────────────────────
v=1 last │ train,MP[0] train,MP[1]  train,MP[1]   train,MP[1]           -           -
         │ shape={0}   shape=B       shape=B        shape=B
─────────┼─────────────────────────────────────────────────────────────────────────────────
v=2 low  │ train,MP[0] train,MP[2]  train,MP[2]   train,MP[2]           -           -
         │ shape={0}   shape=C       shape=C        shape=C
─────────┼─────────────────────────────────────────────────────────────────────────────────
v=3 lwl  │ train,MP[0] train,MP[3]  train,MP[3]   train,MP[3]           -           -
         │ shape={0}   shape=D       shape=D        shape=D
─────────┼─────────────────────────────────────────────────────────────────────────────────
v=4 vbase│ train,MP[0]      -             -              -         inf,MP[4]   inf,MP[4]
         │ shape={0}                                              shape=E     shape=E
─────────┼─────────────────────────────────────────────────────────────────────────────────
v=5 vlast│ train,MP[0]      -             -              -         inf,MP[5]   inf,MP[5]
         │ shape={0}                                              shape=F     shape=F
```

- shape 无关图（TRANSFER、COMM、OPTIMIZER、EMA）：全部 6 变体 → `shape={0,0,0,0}` → Key 相同
- shape 相关训练图：4 个变体 × 各自的 ShapeId（A/B/C/D）
- val 训练图：`-`（不适用）
- 推理图：2 个 val 变体 × 各自的 ShapeId（E/F）；若 val batch 总是完整批，则 E==F
- train 变体推理图：`-`（训练迭代中 Executor 不调用 INF_*）

---

## 五、Phase B：预演去重捕获

### 5.1 PreCaptureResult

```cpp
// include/renaissance/graph/pre_capture.h

struct PreCaptureResult {
    std::vector<CapturedGraph> graphs;  // 去重后的图集（Python set 等价）
    GraphAtlas                  atlas;  // 每格的 captured_idx 已填入

    size_t total_slots   = 0;  // 有效格子总数
    size_t captured      = 0;  // 新捕获数
    size_t reused        = 0;  // 去重复用数
};
```

### 5.2 核心算法

```cpp
// src/graph/pre_capture.cpp

PreCaptureResult pre_capture(const GraphAtlas& compile_atlas, Device* device) {
    PreCaptureResult result;
    result.atlas = compile_atlas;

    // 去重映射：(CG*, GraphId, ShapeId) → vector 索引
    std::unordered_map<CapturedGraph::Key, int32_t,
                       CapturedGraph::KeyHash> seen;

    LOG_INFO << "=== Phase B: Pre-capture Rehearsal Start ===";

    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        for (size_t g = 0; g < GraphAtlas::kMaxGraphIds; ++g) {
            GraphId gid = static_cast<GraphId>(g);
            auto& s = result.atlas.slot(v, gid);

            if (!s.cg || !s.mp) continue;  // 不适用
            result.total_slots++;

            CapturedGraph::Key key{s.cg, gid, s.shape_id};

            // ===== 去重 =====
            auto it = seen.find(key);
            if (it != seen.end()) {
                s.captured_idx = it->second;
                result.reused++;
                continue;  // 复用已有图
            }

            // ===== 新图：预热 + 捕获 =====
            auto cg = CapturedGraph::capture(*s.cg, *s.mp, gid, device);
            int32_t idx = static_cast<int32_t>(result.graphs.size());
            result.graphs.push_back(std::move(cg));
            seen[key] = idx;
            s.captured_idx = idx;
            result.captured++;
        }
    }

    LOG_INFO << "=== Phase B Complete ===";
    LOG_INFO << "  Slots: " << result.total_slots
             << ", Captured: " << result.captured
             << ", Reused: " << result.reused
             << ", Unique: " << result.graphs.size();

    return result;
}
```

### 5.3 执行引擎主流程

```cpp
// main.cpp（示意）

int main() {
    // ===== Phase A: 编译期 =====
    auto compiled = Compiler::compile(arch_plan, base_cfg, variant_cfgs);
    std::array<ShapeId, 6> shapes = extract_shapes(variant_cfgs);
    GraphAtlas compile_atlas = GraphAtlas::build(compiled, shapes);

    // ===== Phase B: 预演捕获期（不计时） =====
    auto pre = pre_capture(compile_atlas, device);

    // ===== MLPerf 计时开始 =====
    MLPERF_LOG_START();

    // ===== Phase C: 运行期 =====
    cudaStream_t stream; cudaStreamCreate(&stream);
    Executor executor;
    executor.setup(pre);
    executor.run(stream);
    cudaStreamDestroy(stream);
}
```

---

## 六、Phase C：Executor 运行时

### 6.1 Executor 设计

```cpp
// include/renaissance/runtime/executor.h

class Executor {
public:
    void setup(const PreCaptureResult& pre) {
        atlas_  = &pre.atlas;
        graphs_ = &pre.graphs;
        activate_variant(0);
    }

    void activate_variant(size_t idx) {
        TR_CHECK(idx < GraphAtlas::kMaxVariants, IndexError,
                 "variant idx " << idx << " out of range [0, "
                 << GraphAtlas::kMaxVariants << ")");
        active_variant_ = idx;
    }
    void set_freeze_first_layer(bool freeze) { skip_first_bwd_ = freeze; }

    void launch(GraphId gid, cudaStream_t stream) {
        int32_t idx = atlas_->index(active_variant_, gid);
        if (idx < 0) return;  // 不适用
        (*graphs_)[idx].launch(stream);
    }

    /// 训练迭代（计算图部分）
    /// TRANSFER_B 由外部双缓冲调度器在上一迭代结束时异步触发，
    /// 与本迭代计算重叠，故不在 kOrder 中。
    void run_train_iteration(cudaStream_t stream) {
        static constexpr GraphId kOrder[] = {
            GraphId::TRANSFER_A,
            GraphId::FIRST_FWD_A,
            GraphId::DEEP_FWD_BWD,
            GraphId::FIRST_BWD,
            GraphId::FIRST_COMM,
            GraphId::DEEP_COMM,
            GraphId::STATS_COMM,
            GraphId::OPTIMIZER,
            GraphId::EMA_UPDATE,
        };
        for (auto gid : kOrder) {
            if (skip_first_bwd_ && gid == GraphId::FIRST_BWD) continue;
            launch(gid, stream);
        }
    }

    /// 验证迭代
    void run_val_iteration(cudaStream_t stream, bool use_ema = false) {
        launch(GraphId::TRANSFER_A,  stream);
        launch(GraphId::INF_MAIN_A,  stream);
        if (use_ema) {
            launch(GraphId::INF_EMA_A, stream);
        }
    }

    /// 示意：实际训练循环由 DataLoader 驱动，根据当前 batch 特征
    ///（分辨率, batch_size, is_last_batch, is_validation）动态选择 variant
    void run(cudaStream_t stream) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (auto& batch : dataloader) {
                size_t v = select_variant(batch);  // 0~3 训练, 4~5 验证
                activate_variant(v);
                if (v <= 3) run_train_iteration(stream);
                else        run_val_iteration(stream);
            }
        }
    }

private:
    const GraphAtlas*              atlas_  = nullptr;
    const std::vector<CapturedGraph>* graphs_ = nullptr;
    size_t                         active_variant_ = 0;
    bool                           skip_first_bwd_ = false;
};
```

### 6.2 运行时路径分析

```
用户视角的键值对：                    Executor 内部解析：
───────────────────────────          ────────────────────
"训练阶段、A区读取、begin分辨率、     activate_variant(0)
 标准batch、首层正向"                    ↓
                                   launch(FIRST_FWD_A)
                                        ↓
                                   atlas_.index(0, FIRST_FWD_A)
                                        ↓
                                   captured_idx = 7
                                        ↓
                                   graphs_[7].launch(stream)

零 hash，零 map 查找，零分支判断
纯：variant → 数组下标 → idx → vector 下标 → 虚函数调用
```

---

## 七、捕获数量精算

### 7.1 按类别的理论最大捕获数

```
类别              子图                                  shape 数   捕获数
──────────────────────────────────────────────────────────────────────────
shape 无关        7 张（TRANSFER×2, COMM×3,             1          7
                 OPTIMIZER, EMA）
训练 shape 相关   4 张（FIRST_FWD×2,                    4*         16
                 DEEP_FWD_BWD, FIRST_BWD）
推理 shape 相关   4 张（INF_MAIN×2, INF_EMA×2）         1~2        4~8
──────────────────────────────────────────────────────────────────────────
合计                                                              27~31
```

*\* 4 = (res_begin×std, res_begin×last, res_end×std, res_end×last)*

### 7.2 ShapeId 自动去重场景

**场景1**：val 的 last batch 为完整批（val_batch_size 整除数）
- val_base ShapeId = val_last ShapeId（N 相同）
- → 推理图从 8 降到 4（2 个 ShapeId → 1 个 ShapeId × 4 张图）

**场景2**：val_res == train_res_end 且 val_batch 碰巧等于某个 train batch
- 推理图 INF_* 使用 inference CG，训练图 DEEP_FWD_BWD 等使用 train CG——不同 CG* → Key 不同 → 即使 ShapeId 碰巧相同也不会碰撞
- 这是正确行为：推理图和训练图的 CUDA Graph 内容不同（推理无反向、可能用 EMA 权重），不应共享
- TRANSFER_A/B 是例外：train 和 val 变体共用同一 CG* + `kShapeInvariant` → Key 相同 → 自然共享 ✓

**场景3**：train_res_begin == train_res_end
- 变体 0 和 2 的 ShapeId 相同（同 N_std, 同 H×W）
- 变体 1 和 3 的 ShapeId 相同（同 N_last, 同 H×W）
- → 训练图从 16 降到 8

### 7.3 实际捕获数量预估

| 场景 | 条件 | 训练图 | 推理图 | 无关图 | 合计 |
|------|------|--------|--------|--------|------|
| 最坏 | 2 分辨率不同, val_last 不完整, EMA 启用 | 16 | 8 | 7 | **31** |
| 典型 | 2 分辨率不同, val_last 不完整, EMA 禁用 | 16 | 4 | 7 | **27** |
| 最优 | 1 分辨率, val_last 完整, EMA 禁用 | 8 | 2 | 7 | **17** |

---

## 八、方案对比与决策理由

### 8.1 三轮演进中的关键分歧与解决

| 分歧点 | S 方案 | K 方案 | D 方案 | **最终方案** | 理由 |
|--------|--------|--------|--------|-------------|------|
| 去重键 | `(CG*, MP*, gid)` | `(CG*, gid, shape_fp)` | `(CG*, gid, shape_id)` | **D 的 `(CG*, gid, shape_id)`** | 显式四元组 > hash；`shape_id` 比 `MP*` 可跨变体碰撞 |
| shape 标识 | 无（依赖 MP 指针） | `shape_fp`（hash 所有 DTensor） | `ShapeId`（输入 Config） | **D 的 `ShapeId`** | 输入 shape 是唯一的自由变量，hash 中间 DTensor 既不必要也有碰撞风险 |
| Atlas 分拆 | CompileAtlas + RuntimeAtlas | 统一 GraphAtlas | 统一 GraphAtlas | **统一 GraphAtlas** | 一表两阶段填入，比两表更简洁 |
| 图抽象 | 强绑定 CUDA | `GraphHandle`（类型别名） | `CapturedGraph`（类） | **D 的 `CapturedGraph`** | 统一封装 CPU/CUDA，虚函数 dispatch |
| 运行时 | hash 表查找 | 数组 O(1) | 数组 O(1) | **数组 index(v,gid)** | `int32_t` 数组，cache-friendly |
| frozen_first | 见于去重键 | 未涉及 | 纯运行时 | **纯运行时** | 用户确认不改变图，不触发独立建图 |
| kMaxVariants | 6 | 6 | 7 | **6** | frozen_first 不占变体槽 |

### 8.2 为什么不用 K 的 shape_fp

K 的 `compute_shape_fp` 遍历 GraphId 的所有 node，hash 所有 input/output DTensor shape。其出发点是 "DEEP_FWD_BWD 的某些 DTensor shape 可能不由输入唯一决定"——这个假设不正确：

对于给定的模型，每层的输出 shape 是输入 shape 和层参数的确定性函数。层参数在编译期固定，跨变体不变。因此，输入 `(N,H,W,C)` 唯一决定了该变体下此 GraphId 涉及的所有 DTensor shape。ShapeId 四元组完美捕获了这个等价关系，无需额外的 DTensor 遍历。

此外，hash 引入的碰撞概率（虽然是 `2^-64` 量级）在确定性系统中是不必要的风险。ShapeId 的值比较是零风险的。

---

## 九、对 COMP_ULTIMATE.md 的修改建议

### 9.1 新增文件

| 文件 | 内容 |
|------|------|
| `include/renaissance/graph/shape_id.h` | ShapeId 定义 |
| `include/renaissance/graph/captured_graph.h` | CapturedGraph 类 |
| `include/renaissance/graph/graph_atlas.h` | GraphAtlas 类 |
| `include/renaissance/graph/pre_capture.h` | PreCaptureResult + pre_capture() |
| `src/graph/graph_atlas.cpp` | GraphAtlas::build() |
| `src/graph/pre_capture.cpp` | pre_capture() 实现 |

### 9.2 修改现有文件

| 文件 | 修改 |
|------|------|
| `include/renaissance/graph/compiler.h` | `Result` 增加 `GraphAtlas compile_atlas` |
| `src/graph/compiler.cpp` | Phase 5 末尾调用 `GraphAtlas::build()` |
| `include/renaissance/runtime/executor.h` | 用 `atlas_` + `graphs_` 替代 `v.train` / `v.inference` |
| `COMP_ULTIMATE.md` | (1) Variant 的 `train*`/`inference*` 标记为可选 (2) 新增 §10 GraphAtlas 章节 (3) 变体矩阵下增加捕获数量说明 |

### 9.3 Variant 指针保留策略

`Variant` 中的 `train` / `inference` 指针**保留但标记为 `@deprecated`**。原因：
- 旧代码（如单元测试）可能通过 `v.train->nodes(GraphId::DEEP_FWD_BWD)` 读取节点列表做验证
- 新 Executor 通过 `GraphAtlas` 选择图，不再使用这两个指针执行
- 未来移除时机：所有使用者迁移到 GraphAtlas 后

---

## 十、实现路径（追加到 COMP_ULTIMATE.md §8）

```
第4周扩展:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 图集预演架构（5天）                                              │
  │ Day 1-2: ShapeId + CapturedGraph + GraphAtlas 数据结构          │
  │ Day 3:   GraphAtlas::build() + Compiler 集成                    │
  │ Day 4:   pre_capture() + cuDNN 预热 + CUDA Graph 捕获           │
  │ Day 5:   Executor 迁移（atlas_ + graphs_）+ 端到端测试           │
  └─────────────────────────────────────────────────────────────────┘

验收标准:
  □ Phase B 日志显示 captured ≤ 31, unique < valid_slots
  □ Phase C 无任何 cudaGraphBeginCapture / cudnnFind* 调用
  □ Phase C launch() 单次调用 < 50ns（纯数组索引 + 虚函数）
  □ 6 变体 × 15 GraphId 全组合通过 CRC 验证（无图错用）
  □ MLPerf 时钟期内无 CUDA Graph 创建 API 调用
```

---

## 十一、总结

```
编译期 Atlas  （Phase A）  → 预演期捕获 （Phase B）  → 运行期调用 （Phase C）
────────────────────────────────────────────────────────────────────────────
┌──────────────┐         ┌──────────────────┐        ┌──────────────┐
│ 声明逻辑归属  │         │ 遍历 → 去重 → 捕获│        │ 纯索引 + 指针 │
│ cg, mp,      │ ──────→ │ Key(cg,gid,      │ ────→ │ atlas[v][g]  │
│ shape_id     │         │     shape_id)     │        │   .captured  │
│              │         │                  │        │   _idx       │
│              │         │ vector<Graph>     │        │   → graph    │
│              │         │ (Python set)      │        │   .launch()  │
└──────────────┘         └──────────────────┘        └──────────────┘
  不计时                     不计时                      MLPerf 计时
```

**核心创新点：**

1. **ShapeId 显式去重**：用输入 shape 四元组而非 hash 或指针做去重键——语义直接、零碰撞、跨变体自动复用
2. **三阶段严格分离**：编译→预演→运行，时间边界清晰，MLPerf 计时期间零捕获
3. **GraphAtlas 统一管理**：一张二维表贯穿三阶段——Phase A 填逻辑归属、Phase B 填索引、Phase C 查索引
4. **CapturedGraph 后端抽象**：CUDA Graph 或 CPU 函数序列统一封装
5. **27~31 个图最小集**：shape 无关图全局共享 7 张，训练图按 4 种唯一 shape 捕获 16 张，推理图按 shape 捕获 4~8 张