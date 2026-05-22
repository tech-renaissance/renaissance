# Compiler 终极方案 — 从 ArchPlan 到 MemoryPlan + ComputationGraph 全流程

## 零、七轮辩论的终局

经过七轮迭代，所有关键分歧已消解。最终方案在每一个决策点上都有明确结论：

| # | 决策点 | 早期分歧 | 最终裁决 | 裁决者 |
|---|--------|----------|----------|--------|
| 1 | 类命名 | Compiler vs GraphBuilder | **Compiler** | 代码事实：`memory_plan.h:235` 已有 `friend class Compiler` |
| 2 | 知识形式 | 类/虚函数/自由函数 | **函数指针 `LayerDescriptor` + `switch` 注册表** | K 提出，全员共识 |
| 3 | 图归属 | Descriptor 内 vs Compiler 内 | **Compiler 内部策略** | K 纠正 D，全员共识 |
| 4 | Region | RegionResolver vs 硬编码 | **`TensorDesc::region` + `InferContext`** | D 论证 RegionResolver 是过度设计 |
| 5 | 形状变体 | ShapeTable/ShapeProfile | **多 MemoryPlan，DTensor 独占权威** | 用户在第五轮纠偏 |
| 6 | ComputationGraph | 含形状 vs 不含 | **纯算子拓扑 + tensor_id** | 用户在第五轮定义 |
| 7 | stride 计算 | K 主张用 max shape | **DTensor 自身 shape** | 用户在第七轮纠正 |
| 8 | MemoryPlan API | S:LayoutConfig / K:双shape / D:slot_bytes | **`alloc(shape, dtype, region, slot_bytes)`** | 用户第七轮裁决 |

### 0.1 第七轮用户的两条关键纠偏

**纠偏一（对 K 的 `alloc(alloc_shape, logical_shape)` 的批判）**：

> slot_bytes你可以比出最大值，alloc_shape你怎么比？？你拿谁做alloc_shape？？？

`slot_bytes` 是标量，取最大值是自然操作。但"取形状的最大值"需要定义原子语义——这不是形状的自然操作。虽然在此场景下（相同 ArchPlan 的不同分辨率）逐元素 max 成立，但 API 设计不应依赖这种隐含假设。

**纠偏二（对 stride 的错误论断的否定）**：

> stride就是DTensor本身的stride，每个形状下就只有一个stride！除了slot_bytes是取最大值以外，其他一切都是按具体形状来计算的！

K 的"stride 必须按最大形状计算"是错误的。DTensor 在不同变体中有不同的 shape，自然有不同的 stride：

```
base DTensor:   shape=[N,224,224,64] → n_stride=224*224*64, h_stride=224*64, w_stride=64
lowres DTensor: shape=[N,128,128,64] → n_stride=128*128*64, h_stride=128*64, w_stride=64
```

每个 DTensor 的 stride 按自己的 shape 计算，自洽且正确。**只有 `slot_bytes` 需要取最大值**来保证所有变体的 offset 一致。

---

## 一、总体架构

```
                          BluePrint
                              │
                       ArchPlan::build()
                       (fuse=true/false)
                              │
                       ArchPlan (layers_)
                              │
    ┌─────────────────────────▼──────────────────────────────────┐
    │                     Compiler::compile()                     │
    │                                                            │
    │  输入: ArchPlan + base_cfg + [variant_cfgs]                │
    │                                                            │
    │  Phase 1: derive_all_shapes()                              │
    │    对所有 config 推导全部 TensorDesc（shape+region+dtype）   │
    │                                                            │
    │  Phase 2: compute_max_slot_bytes()                         │
    │    对每个 tensor slot，取所有变体中 slot_bytes 的最大值      │
    │    这是唯一"跨变体取最大值"的操作                           │
    │                                                            │
    │  Phase 3: create_all_memory_plans()                        │
    │    每个变体独立 MemoryPlan，用 alloc(shape, dtype,          │
    │    region, max_slot_bytes) 分配 DTensor                    │
    │    → 所有变体 offset 一致，DTensor shape/stride 各自正确    │
    │                                                            │
    │  Phase 4: build_base_computation_graph()                   │
    │    正向遍历：infer → alloc → build_forward → graph.append  │
    │    反向遍历：build_backward → graph.append                 │
    │    辅助图：通信、优化器、EMA、推理                          │
    │    ComputationGraph 不存任何形状信息                       │
    │                                                            │
    │  Phase 5: build_variant_graphs()                           │
    │    shape-only 变体共享 base ComputationGraph               │
    │    graph-change 变体独立建图                               │
    │                                                            │
    │  输出: { base_mem, base_train, base_infer, variants[] }   │
    └────────────────────────────────────────────────────────────┘
```

---

## 二、核心组件设计

### 2.1 DTensor — 关键修改：slot_bytes 与 shape 解耦

用户第七轮纠偏确认：
1. **stride 按 DTensor 自身 shape 计算**，不同变体有不同 stride，各自正确
2. **slot_bytes 必须与 shape 解耦**——不同变体的 DTensor 有不同 shape 但相同 slot_bytes，说明它不是从 shape 实时计算出来的，而是构造时存储的常数

**修改要点**：
- 新增 `uint64_t slot_bytes_` 成员，构造时赋值，`slot_bytes()` 直接返回
- 原有计算逻辑移到 `static compute_slot_bytes(shape, dtype, region)` 纯函数
- 构造函数有两个重载：不带 slot_bytes（自动计算）和带 slot_bytes（显式指定）

```cpp
// include/renaissance/tensor/distributed_tensor.h

struct DistributedTensor {
    int32_t  id      = -1;
    Shape    shape;                   // 逻辑形状（创建后不可变）
    DType    dtype   = DType::FP32;
    Region   region  = Region::DEFAULT;
    uint64_t offset  = 0;            // 由 MemoryPlan::finalize 赋值

    int64_t  n_stride = 0;           // NHWC stride，按 shape 计算（创建后不可变）
    int64_t  h_stride = 0;
    int64_t  w_stride = 0;
    int64_t  c_stride = 0;

private:
    uint64_t slot_bytes_ = 0;        // 构造时存储的常数，与 shape 解耦

public:
    /// 返回构造时存储的 slot_bytes（非实时计算）
    uint64_t slot_bytes() const noexcept { return slot_bytes_; }

    /// 纯函数：对任意 (shape, dtype, region) 计算所需的最小 slot_bytes
    /// MemoryPlan 的 alloc() 和 Compiler 的 Phase 2 使用此函数
    static uint64_t compute_slot_bytes(const Shape& shape, DType dtype, Region region);

    // ... padded_c(), padded_bytes(), nbytes(), cuda_alignment() 等不变 ...

    // === 构造 ===

    DistributedTensor() = default;

    /// 标准构造：slot_bytes 从 shape 自动计算
    DistributedTensor(int32_t i, Shape s, DType d, Region r, uint64_t off)
        : id(i), shape(s), dtype(d), region(r), offset(off),
          slot_bytes_(compute_slot_bytes(s, d, r)) {
        init_strides();
    }

    /// 变体构造：slot_bytes 显式指定（所有变体传同一个 max 值）
    DistributedTensor(int32_t i, Shape s, DType d, Region r, uint64_t off, uint64_t slot_bytes)
        : id(i), shape(s), dtype(d), region(r), offset(off),
          slot_bytes_(slot_bytes) {
        init_strides();
    }

private:
    void init_strides() {
        int64_t ac = padded_c();
        c_stride = 1;
        w_stride = ac;
        h_stride = ac * shape.w();
        n_stride = ac * shape.w() * shape.h();
    }
};
```

**变体中两个 DTensor 的对比**：

```
base DTensor:
  shape = [N, 224, 224, 64], stride = [224*224*64, ...]
  slot_bytes_ = compute_slot_bytes([224,224,64]) = X
  offset = 按 X 计算 = 0x1234500

lowres DTensor:
  shape = [N, 128, 128, 64], stride = [128*128*64, ...]
  slot_bytes_ = X                                       ← 显式传入，与 base 相同！
  offset = 按 X 计算 = 0x1234500                         ← 与 base 相同！
```

每个 DTensor 的 stride 按自身 shape 计算（各自正确），slot_bytes 是构造时存储的常数（跨变体一致，保证 offset 相同）。`compute_slot_bytes()` 不依赖 `this`，纯函数可供 MemoryPlan 和 Compiler 任意调用。

### 2.2 MemoryPlan — 新增 `alloc(shape, dtype, region, slot_bytes)` 重载

```cpp
// include/renaissance/graph/memory_plan.h

class MemoryPlan {
public:
    // ====== 已有公开接口 ======
    [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype, Region region);
    // 内部调用 DTensor::compute_slot_bytes(shape, dtype, region)
    // 构造 DTensor 时 slot_bytes 自动从 shape 计算

    void finalize();
    [[nodiscard]] const DTensor& get_dtensor(int32_t id) const;
    int32_t entry_count() const;
    // ...

private:
    struct Entry {
        DTensor  dt;
        int8_t   grad_slot_idx = -1;
    };

    // ====== Compiler 专用（friend class Compiler 已声明） ======
    //
    // 在变体编译时使用：
    //   shape       — DTensor 的逻辑形状（决定 shape/stride/DType/Region）
    //   dtype       — 数据类型
    //   region      — 内存分区
    //   slot_bytes  — 预计算的槽位字节数（所有变体中的最大值）
    //
    // 直接将 slot_bytes 传入 DTensor 构造函数，DTensor 内部存储此常数。
    // finalize 时所有变体的 dt.slot_bytes() 返回同一值 → offset 完全一致。
    //
    [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype,
                                 Region region, uint64_t slot_bytes);

    friend class Compiler;
};
```

**`finalize()` 不再需要条件判断**：

```cpp
void MemoryPlan::finalize() {
    uint64_t cursor = 0;
    for (auto& entry : entries_) {
        uint64_t slot = entry.dt.slot_bytes();  // DTensor 内部存储的常数
        entry.dt.offset = align_up(cursor, 256);
        cursor = entry.dt.offset + slot;
    }
    // ...
}
```

**`alloc()` 的两个重载**：

```cpp
// 标准分配：slot_bytes 由 shape 自动计算
DTensor MemoryPlan::alloc(const Shape& shape, DType dtype, Region region) {
    uint64_t sb = DTensor::compute_slot_bytes(shape, dtype, region);
    DTensor dt(next_id_++, shape, dtype, region, /*offset placeholder*/ 0, sb);
    entries_.push_back({std::move(dt)});
    return entries_.back().dt;
}

// 变体分配：slot_bytes 由 Compiler 显式传入（max 值）
DTensor MemoryPlan::alloc(const Shape& shape, DType dtype, Region region, uint64_t slot_bytes) {
    DTensor dt(next_id_++, shape, dtype, region, /*offset placeholder*/ 0, slot_bytes);
    entries_.push_back({std::move(dt)});
    return entries_.back().dt;
}
```

### 2.3 ComputationGraph — 纯算子拓扑

```cpp
// include/renaissance/graph/computation_graph.h

class ComputationGraph {
public:
    enum class GraphId : uint8_t {
        TRANSFER_A, TRANSFER_B,     // 异步传输图
        FIRST_FWD_A, FIRST_FWD_B,   // 首层正向图
        DEEP_FWD_BWD,               // 深层正反向融合图
        FIRST_BWD,                  // 首层反向图
        FIRST_COMM, DEEP_COMM,      // 通信图
        STATS_COMM,                 // BN统计量通信图
        OPTIMIZER, EMA_UPDATE,      // 优化器/EMA图
        INF_MAIN_A, INF_MAIN_B,     // 主模型推理图
        INF_EMA_A, INF_EMA_B,       // EMA推理图
        COUNT
    };

    struct Node {
        ComputeOp op;
        OpParams  params;
        std::vector<int32_t> inputs;   // DTensor 全局 ID
        std::vector<int32_t> outputs;  // DTensor 全局 ID
        // 不存 Shape、Region、DType —— 全部从 MemoryPlan 的 DTensor 获取
    };

    void append(GraphId gid, Node node);
    void append(GraphId gid, const std::vector<Node>& nodes);
    const std::vector<Node>& nodes(GraphId gid) const;
    size_t total_node_count() const;

private:
    std::array<std::vector<Node>, static_cast<size_t>(GraphId::COUNT)> graphs_;
};
```

### 2.4 LayerDescriptor — 知识载体

```cpp
// include/renaissance/graph/layer_descriptor.h

enum class GraphMode { TRAIN_FWD, TRAIN_BWD, INFERENCE };

struct InferContext {
    GraphMode mode;
    bool      enable_amp;
    bool      is_first_layer;
};

/// @section TENSOR_CONSISTENCY 张量一致性铁律
///
/// **铁律1：infer_tensors 返回该 LayerKind 在所有模式下的张量并集**
///
/// infer_tensors 的返回值必须覆盖 TRAIN_FWD / TRAIN_BWD / INFERENCE 全部模式下
/// 该算子可能用到的所有张量。GraphMode 参数仅用于标注张量的角色（如 weight_fp16
/// 仅在 enable_amp 时存在），不做子集筛选。各 build_* 函数从同一 descs 列表中
/// 按需选取其子集。
///
/// **铁律2：返回数量必须与 Config 无关**
///
/// 无论 enable_amp、is_first_layer 如何变化，同一 LayerKind 的 infer_tensors
/// 返回的张量数量必须完全相同。某些 Config 下"不适用"的张量必须用占位符返回
/// （name 标记为 "unused_xxx"，shape 为 empty）。
///
/// **铁律3：返回顺序必须与 Config 无关**
///
/// 相同语义的张量在不同 Config 下的索引位置必须相同。例如 weight_fp32 在 AMP
/// 和非 AMP 配置下必须位于同一 descs 索引。
///
/// **铁律4：name 和 region 必须跨 Config 一致**
///
/// 相同索引位置的 TensorDesc，其 name 和 region 必须在所有 Config 中相同。
/// shape 和 dtype 可以因 Config 不同而变化。
///
/// **违规后果**：ComputationGraph 的 node.inputs 使用 tensor_id（= descs 索引），
/// 若不同 Config 下 descs 数量/顺序/语义不同，同一 tensor_id 在 base 和 lowres
/// MemoryPlan 中指向不同物理 DTensor，导致静默数据错乱，极难调试。

struct TensorDesc {
    std::string name;
    Shape       shape;
    Region      region;
    DType       dtype;
};

struct SubgraphPattern {
    struct Node {
        ComputeOp op;
        std::vector<size_t> input_indices;   // 在 TensorDesc 列表中的索引
        std::vector<size_t> output_indices;  // 在 TensorDesc 列表中的索引
    };
    std::vector<Node> nodes;
    // output_shape 已删除 —— 统一用 get_output_shape(tensors) 从 TensorDesc 推导
};

struct LayerDescriptor {
    using InferFn = std::vector<TensorDesc> (*)(const Shape& input_shape,
                                                  const OpParams& params,
                                                  const InferContext& ctx);
    using BuildFn = SubgraphPattern (*)(const OpParams& params,
                                         const std::vector<TensorDesc>& descs);

    InferFn infer_tensors;
    BuildFn build_forward;
    BuildFn build_backward;
    BuildFn build_inference;
};

const LayerDescriptor& get_layer_descriptor(LayerKind kind);

struct LayerContext {
    std::vector<TensorDesc> descs;      // 该层全部张量描述（三模式并集）
    std::vector<int32_t>    tensor_ids; // 分配后的 DTensor 全局 ID
};

/// @section STRIDE_CONSTRAINT stride 工程约束（铁律）
///
/// 所有 GPU kernel 和 cuDNN descriptor 必须使用 DTensor::n_stride /
/// h_stride / w_stride / c_stride。严禁基于 max_shape 硬编码 stride 计算。
/// stride 的唯一权威是 DTensor 自身。
///
/// 在不同变体（base 224×224 vs lowres 128×128）中：
/// - base DTensor: stride = [224*224*64, 224*64, 64, 1]
/// - lowres DTensor: stride = [128*128*64, 128*64, 64, 1]
/// kernel 用各自的 stride 在各自逻辑空间内寻址，完全正确。

// ====== 关键辅助函数声明 ======

/// 从 TensorDesc 列表推导该层的输出 shape
Shape get_output_shape(const std::vector<TensorDesc>& descs);

/// 为变体 Config 生成可读名称
std::string make_variant_name(const Compiler::Config& cfg);

/// 判断是否需要独立 ComputationGraph（图拓扑变化时返回 true）
/// 注意：freeze_first_layer 不触发独立建图，它是 Executor 运行时行为。
bool needs_separate_graph(const Compiler::Config& cfg);

/// 构建推理 ComputationGraph
ComputationGraph build_inference_graph(const ArchPlan& arch,
                                        const MemoryPlan& mem,
                                        const std::vector<LayerContext>& ctxs);
```

### 2.5 Compiler — 薄编排器

```cpp
// include/renaissance/graph/compiler.h

class Compiler {
public:
    struct Config {
        bool enable_amp;
        int  max_sample_resolution;   // MemoryPlan 构造用（最大槽位预留）
        int  actual_resolution;       // 形状推导用（变体可能不同）
        int  batch_size;
        int  num_color_channels;
        bool freeze_first_layer = false;  // 运行时首层冻结标记，不影响 MemoryPlan/图拓扑
        static Config from_global_registry();
    };

    struct Result {
        // variants[0] = base 配置，variants[1..] = 变体 —— 来自 S/K/D 共识
        struct Variant {
            std::string             name;
            MemoryPlan              memory_plan;
            // shape-only 变体指向 base 的图（指针共享，避免深拷贝）
            // graph-change 变体通过 owns_train_graph 管理自有图
            const ComputationGraph* train      = nullptr;
            const ComputationGraph* inference  = nullptr;
            ComputationGraph* own_train     = nullptr;  // 仅 graph-change 变体非空
            ComputationGraph* own_inference = nullptr;

            ~Variant() {
                delete own_train;
                delete own_inference;
            }
        };
        std::vector<Variant> variants;

        // 禁止移动 —— Variant 中的指针指向 variants[0] 内部
        Result() = default;
        Result(const Result&) = delete;
        Result& operator=(const Result&) = delete;
        Result(Result&&) = delete;
        Result& operator=(Result&&) = delete;
    };

    static Result compile(const ArchPlan& arch,
                          const Config& base_cfg,
                          const std::vector<Config>& variant_cfgs = {});
};
```

---

## 三、编译流程：五 Phase 详解

### Phase 1：对所有配置推导全部形状

```cpp
// 对每种配置，完整运行一遍形状推导。
// 输入相应配置的分辨率/batch_size，输出每层每个 tensor 的 TensorDesc。

struct AllShapes {
    // shapes[layer_idx][tensor_idx] = TensorDesc
    std::vector<std::vector<TensorDesc>> per_layer;
    Shape final_output_shape;
};

AllShapes derive_all_shapes(const ArchPlan& arch, const Config& cfg) {
    AllShapes result;
    // 根据 Config 构造该变体的实际输入形状 —— 来自 S/K/D 共识
    Shape cur(cfg.batch_size, cfg.actual_resolution,
              cfg.actual_resolution, cfg.num_color_channels);

    for (int i = 0; i < arch.layers().size(); ++i) {
        const auto& layer = arch.layers()[i];
        const auto& desc = get_layer_descriptor(layer.kind());

        InferContext ctx{GraphMode::TRAIN_FWD, cfg.enable_amp, layer.is_first_layer};
        auto tensors = desc.infer_tensors(cur, layer.params(), ctx);

        cur = get_output_shape(tensors);
        result.per_layer.push_back(std::move(tensors));
    }
    result.final_output_shape = cur;
    return result;
}
```

**注意**：输入 shape 因配置而异——低分辨率配置的 `in_shape` 是 128×128，last batch 的 batch_size 是 M。`infer_tensors` 按此推导，得到该变体正确的 shape。

### Phase 2：计算统一 slot_bytes（唯一跨变体比较）

```cpp
// 对每个 (layer_idx, tensor_idx) 位置，计算所有变体中 slot_bytes 的最大值。
// 这是唯一需要"跨变体取最大值"的操作。

std::vector<std::vector<uint64_t>> compute_max_slot_bytes(
    const std::vector<AllShapes>& all_shapes)
{
    // ===== 跨 Config 一致性断言（张量铁律的编译期保护） =====
    validate_tensor_consistency(all_shapes);

    size_t L = all_shapes[0].per_layer.size();
    std::vector<std::vector<uint64_t>> max_slots(L);

    for (size_t i = 0; i < L; ++i) {
        size_t T = all_shapes[0].per_layer[i].size();
        max_slots[i].resize(T, 0);

        for (size_t j = 0; j < T; ++j) {
            for (const auto& shapes : all_shapes) {
                const auto& td = shapes.per_layer[i][j];
                uint64_t s = DTensor::compute_slot_bytes(td.shape, td.dtype, td.region);
                max_slots[i][j] = std::max(max_slots[i][j], s);
            }
        }
    }
    return max_slots;
}

void validate_tensor_consistency(const std::vector<AllShapes>& all_shapes) {
    for (size_t ci = 1; ci < all_shapes.size(); ++ci) {
        TR_CHECK(all_shapes[ci].per_layer.size() == all_shapes[0].per_layer.size(),
                 ValueError, "Config " << ci << " layer count mismatch");
        for (size_t li = 0; li < all_shapes[0].per_layer.size(); ++li) {
            TR_CHECK(all_shapes[ci].per_layer[li].size() == all_shapes[0].per_layer[li].size(),
                     ValueError, "Config " << ci << " layer " << li << " tensor count mismatch");
            for (size_t ti = 0; ti < all_shapes[0].per_layer[li].size(); ++ti) {
                TR_CHECK(all_shapes[ci].per_layer[li][ti].name == all_shapes[0].per_layer[li][ti].name,
                         ValueError, "Config " << ci << " name mismatch at (" << li << "," << ti << ")");
                TR_CHECK(all_shapes[ci].per_layer[li][ti].region == all_shapes[0].per_layer[li][ti].region,
                         ValueError, "Config " << ci << " region mismatch at (" << li << "," << ti << ")");
            }
        }
    }
}
```

**为什么 `slot_bytes` 取最大值是正确的**：
- slot_bytes 仅用于 Offset 计算，定义槽位在 MemoryPlan 布局中的宽度
- 按最大宽度预留，所有变体的偏移一致
- shape / stride / dtype / region 等 DTensor 属性各自按自身配置计算，不受 slot_bytes 影响

### Phase 3：创建所有 MemoryPlan

```cpp
// 为每个变体创建 MemoryPlan。
// DTensor 的 shape/stride 按该变体的实际形状计算（各自正确）。
// DTensor 的 offset 由 max_slot_bytes 决定（所有变体一致）。

MemoryPlan create_memory_plan(
    const Config& cfg,
    const AllShapes& shapes,
    const std::vector<std::vector<uint64_t>>& max_slot_bytes,
    std::vector<LayerContext>& out_ctxs)
{
    MemoryPlan mem({cfg.batch_size, cfg.max_sample_resolution, cfg.num_color_channels});
    out_ctxs.clear();
    out_ctxs.reserve(shapes.per_layer.size());

    for (size_t i = 0; i < shapes.per_layer.size(); ++i) {
        LayerContext lctx;
        const auto& tensors = shapes.per_layer[i];
        lctx.descs = tensors;
        lctx.tensor_ids.reserve(tensors.size());

        for (size_t j = 0; j < tensors.size(); ++j) {
            // 用变体的 shape（各自正确）+ 统一的 max_slot_bytes（offset 一致）
            DTensor dt = mem.alloc(
                tensors[j].shape,        // DTensor 的权威形状 → stride 按此计算
                tensors[j].dtype,        // 数据类型
                tensors[j].region,       // 内存分区
                max_slot_bytes[i][j]     // 统一槽位宽度 → offset 一致
            );
            lctx.tensor_ids.push_back(dt.id());
        }
        out_ctxs.push_back(std::move(lctx));
    }

    mem.finalize();  // offset 此时确定
    return mem;
}
```

**关键**：两个不同变体中同一 tensor_id 的 DTensor：

```
base DTensor[5]:
  shape = [N, 224, 224, 64]
  stride = [224*224*64, 224*64, 64, 1]  ← 按 224 计算 ✓
  offset = 0x1234500                      ← 按 max_slot_bytes 决定

lowres DTensor[5]:
  shape = [N, 128, 128, 64]
  stride = [128*128*64, 128*64, 64, 1]  ← 按 128 计算 ✓
  offset = 0x1234500                      ← 相同的 max_slot_bytes = 相同的 offset ✓
```

每个变体的 DTensor 内部完全自洽：kernel 用 stride 在自己的逻辑空间内寻址，base_ptr + offset 指向正确位置。

### Phase 4：创建 base ComputationGraph

```cpp
ComputationGraph build_computation_graph(
    const ArchPlan& arch,
    const MemoryPlan& mem,
    const std::vector<LayerContext>& ctxs)
{
    ComputationGraph graph;
    int first_idx = arch.first_layer_index();

    // === 正向遍历 ===
    for (int i = 0; i < arch.layers().size(); ++i) {
        const auto& layer = arch.layers()[i];
        const auto& desc = get_layer_descriptor(layer.kind());
        const auto& lctx = ctxs[i];

        auto pattern = desc.build_forward(layer.params(), lctx.descs);
        auto gid = (i == first_idx) ? GraphId::FIRST_FWD_A : GraphId::DEEP_FWD_BWD;
        for (const auto& pn : pattern.nodes) {
            ComputationGraph::Node gn;
            gn.op = pn.op;
            gn.params = layer.params();
            for (auto idx : pn.input_indices)
                gn.inputs.push_back(lctx.tensor_ids[idx]);
            for (auto idx : pn.output_indices)
                gn.outputs.push_back(lctx.tensor_ids[idx]);
            graph.append(gid, gn);
        }
    }

    // === 反向遍历（逆序，自然形成 fwd_0..fwd_N, bwd_N..bwd_0） ===
    for (int i = arch.layers().size() - 1; i >= 0; --i) {
        const auto& layer = arch.layers()[i];
        const auto& desc = get_layer_descriptor(layer.kind());
        const auto& lctx = ctxs[i];

        auto pattern = desc.build_backward(layer.params(), lctx.descs);
        if (pattern.nodes.empty()) continue;
        auto gid = (i == first_idx) ? GraphId::FIRST_BWD : GraphId::DEEP_FWD_BWD;
        for (const auto& pn : pattern.nodes) {
            ComputationGraph::Node gn;
            gn.op = pn.op;
            gn.params = layer.params();
            for (auto idx : pn.input_indices)
                gn.inputs.push_back(lctx.tensor_ids[idx]);
            for (auto idx : pn.output_indices)
                gn.outputs.push_back(lctx.tensor_ids[idx]);
            graph.append(gid, gn);
        }
    }

    // === 辅助图 ===
    // 通信图、优化器图、EMA图、推理图 ...

    return graph;
}
```

### Phase 5：变体建图

```cpp
Compiler::Result Compiler::compile(
    const ArchPlan& arch,
    const Config& base_cfg,
    const std::vector<Config>& variant_cfgs)
{
    // === Phase 1 ===
    std::vector<Config> all_cfgs = {base_cfg};
    all_cfgs.insert(all_cfgs.end(), variant_cfgs.begin(), variant_cfgs.end());

    std::vector<AllShapes> all_shapes;
    for (const auto& cfg : all_cfgs)
        all_shapes.push_back(derive_all_shapes(arch, cfg));

    // === Phase 2 ===
    auto max_slot_bytes = compute_max_slot_bytes(all_shapes);

    // === Phase 3: 创建所有 MemoryPlan ===
    Result result;

    // base → variants[0]
    Result::Variant base_var;
    base_var.name       = "base";
    std::vector<LayerContext> base_ctxs;
    base_var.memory_plan = create_memory_plan(base_cfg, all_shapes[0], max_slot_bytes, base_ctxs);
    result.variants.push_back(std::move(base_var));

    // variant configs → variants[1..]
    std::vector<std::vector<LayerContext>> variant_ctxs;
    for (size_t vi = 0; vi < variant_cfgs.size(); ++vi) {
        std::vector<LayerContext> ctxs;
        auto mem = create_memory_plan(variant_cfgs[vi], all_shapes[vi + 1],
                                       max_slot_bytes, ctxs);
        variant_ctxs.push_back(std::move(ctxs));

        Result::Variant v;
        v.name        = make_variant_name(variant_cfgs[vi]);
        v.memory_plan  = std::move(mem);
        result.variants.push_back(std::move(v));
    }

    // === Phase 4: base ComputationGraph（放在 variants[0]） ===
    result.variants[0].own_train     = new ComputationGraph(build_computation_graph(arch, result.variants[0].memory_plan, base_ctxs));
    result.variants[0].own_inference = new ComputationGraph(build_inference_graph(arch, result.variants[0].memory_plan, base_ctxs));
    result.variants[0].train     = result.variants[0].own_train;
    result.variants[0].inference = result.variants[0].own_inference;

    // === Phase 5: 变体 ComputationGraph ===
    for (size_t vi = 0; vi < variant_cfgs.size(); ++vi) {
        auto& v = result.variants[vi + 1];

        if (needs_separate_graph(variant_cfgs[vi])) {
            // 图拓扑变化的变体：独立建图（当前无此场景，reserved for future）
            v.own_train     = new ComputationGraph(build_computation_graph(arch, v.memory_plan, variant_ctxs[vi]));
            v.own_inference = new ComputationGraph(build_inference_graph(arch, v.memory_plan, variant_ctxs[vi]));
            v.train     = v.own_train;
            v.inference = v.own_inference;
        } else {
            // shape-only 变体：指针指向 base 图（零拷贝）
            v.train     = result.variants[0].train;
            v.inference = result.variants[0].inference;
        }
    }

    return result;
}
```

---

## 四、运行时 Executor

### 4.1 变体切换

```cpp
class Executor {
    const Compiler::Result* compiled_ = nullptr;
    const MemoryPlan*       active_mem_ = nullptr;
    const ComputationGraph* active_train_ = nullptr;

public:
    void setup(const Compiler::Result& r) {
        compiled_     = &r;
        activate_variant(0);
    }

    // O(1) 指针切换，无分支
    void activate_variant(size_t idx) {
        auto& v  = compiled_->variants[idx];
        active_mem_   = &v.memory_plan;
        active_train_ = v.train;
    }

    void launch_train_iteration(cudaStream_t stream) {
        backend_.launch(*active_train_, *active_mem_, stream);
    }

    // 首层冻结：运行时跳过首层反向图，不触发独立建图
    // 正向图、更新图均不变——ZERO_GRAD 保证首层梯度为零，其他层正常更新
    void set_freeze_first_layer(bool freeze) {
        freeze_first_layer_ = freeze;
    }

private:
    bool freeze_first_layer_ = false;
};
```

### 4.2 CUDA Graph 捕获

```cpp
void Backend::launch(const ComputationGraph& graph, const MemoryPlan& mem,
                      cudaStream_t stream) {
    auto key = std::make_pair(&graph, &mem);
    if (auto it = cuda_graphs_.find(key); it != cuda_graphs_.end()) {
        cudaGraphLaunch(it->second, stream);
        return;
    }

    // 首次使用该 (graph, mem) 组合：捕获 CUDA Graph
    CudaGraph cg = capture_cuda_graph(graph, mem, stream);
    cuda_graphs_[key] = cg;
    cudaGraphLaunch(cg, stream);
}

CudaGraph Backend::capture_cuda_graph(const ComputationGraph& graph,
                                       const MemoryPlan& mem,
                                       cudaStream_t stream) {
    CudaGraph cg;
    cudaStreamBeginCapture(stream);

    for (const auto& node : graph.nodes(GraphId::DEEP_FWD_BWD)) {
        const DTensor& in_dt  = mem.get_dtensor(node.inputs[0]);
        const DTensor& out_dt = mem.get_dtensor(node.outputs[0]);

        // shape = DTensor 的权威值（该变体的实际形状）
        // stride = 按该变体 shape 计算（各自正确）
        // offset = 所有变体相同（由 max_slot_bytes 保证）
        void* in_ptr  = base_ptr_ + in_dt.offset;
        void* out_ptr = base_ptr_ + out_dt.offset;

        cudnnSetTensor4dDescriptorEx(in_desc, ...,
            in_dt.shape.n(),  in_dt.shape.h(),  in_dt.shape.w(),  in_dt.shape.c(),
            in_dt.n_stride,   in_dt.h_stride,   in_dt.w_stride,   in_dt.c_stride);

        launch_kernel(node.op, node.params,
                      in_desc, out_desc, in_ptr, out_ptr);
    }

    cudaStreamEndCapture(stream, &cg.handle);
    return cg;
}
```

---

## 五、变体矩阵（6 个输入 shape 变体）

输入 shape 变体由两个正交维度笛卡尔积产生：
- **分辨率**：train_res_begin、train_res_end、val_res（3 种）
- **batch size**：standard、train_last_batch、val_last_batch（3 种）

训练：2 × 2 = 4 种组合；验证：1 × 2 = 2 种组合。即使某两种组合的 shape 碰巧相同也不合并——每个变体有独立的语义和生命周期，避免运行时条件嵌套。

| 索引 | 变体 | 阶段 | 分辨率 | batch | shape 变化 | MemoryPlan | train 图 | inference 图 |
|---|---|---|---|---|---|---|---|---|
| 0 | train_base | 训练 | train_res_begin | standard | 基准 | 独立 | own | own |
| 1 | train_last | 训练 | train_res_begin | train_last_batch | N 变 | 独立（offset 同） | → [0] | → [0] |
| 2 | train_lowres | 训练 | train_res_end | standard | H×W 全变 | 独立（offset 同） | → [0] | → [0] |
| 3 | train_lowres_last | 训练 | train_res_end | train_last_batch | H×W+N 全变 | 独立（offset 同） | → [0] | → [0] |
| 4 | val_base | 验证 | val_res | standard | 视配置而定 | 独立（offset 同） | nullptr | → [0] |
| 5 | val_last | 验证 | val_res | val_last_batch | 视配置而定 | 独立（offset 同） | nullptr | → [0] |

**frozen_first（首层冻结）不属于输入 shape 变体。** 它不改变任何张量的 shape，与 MemoryPlan 生成完全无关。运行时通过 Executor 跳过首层反向图即可——正向图、更新图均不变，ZERO_GRAD 保证首层梯度为零，其他层正常更新。无需额外 ComputationGraph 实例，不触发独立建图。

6 个 MemoryPlan（~780KB）+ 1 个 ComputationGraph 实例（train）+ 1 个 ComputationGraph 实例（inference）+ 至多 7 套 CUDA Graph。

---

## 六、设计原则总结

| # | 原则 | 来源 |
|---|------|------|
| 1 | **DTensor 是形状的唯一权威**。无 ShapeTable、无 ShapeProfile | 用户第五轮 |
| 2 | **ComputationGraph 不存形状**。只存 (op, params, input_ids, output_ids) | 用户第五轮 |
| 3 | **stride 按 DTensor 自身 shape 计算**。不需要 stride_shape 或 max shape | 用户第七轮 |
| 4 | **slot_bytes 与 shape 解耦**：DTensor 存储构造时传入的 slot_bytes 常数，`slot_bytes()` 返回该常数，非实时计算 | 用户第七轮 + 本题纠正 |
| 5 | **多 MemoryPlan，每个 DTensor 自创自洽** | 用户第五轮 |
| 6 | **MemoryPlan API 传 slot_bytes 而非 alloc_shape**。标量比较是自然的 | 用户第七轮 |
| 7 | **编译成本极低**，不计入 MLPerf | 用户第三轮 |
| 8 | **图分 15 张有效图**（传输A/B、首层正反向A/B、深层正反向、首层反向、通信×3、优化器、EMA、推理×4），COUNT=16 | 用户第二轮 |

---

## 七、文件组织

```
include/renaissance/graph/
├── arch_plan.h              # 已有
├── memory_plan.h            # 修改：新增 alloc(shape, dtype, region, slot_bytes)
├── op_kind.h                # 已有
├── compiler.h               # 新建：Compiler + Config + Result + Variant
├── layer_descriptor.h       # 新建：LayerDescriptor + TensorDesc + SubgraphPattern + InferContext
└── computation_graph.h      # 新建：ComputationGraph + GraphId + Node（纯净，无 shape）

include/renaissance/tensor/
└── distributed_tensor.h     # 修改：新增 slot_bytes_ 成员 + static compute_slot_bytes() + 双构造函数

src/graph/compiler/
├── compiler.cpp             # 五 Phase 主流程
├── registry.cpp             # get_layer_descriptor() switch 集中注册
├── lower_common.cpp         # Identity, Flatten, ReLU, SoftmaxCE, Add2, Mul2
├── lower_conv.cpp           # Conv 族
├── lower_bn.cpp             # Bn2d 族
├── lower_pool.cpp           # MaxPool, AvgPool, GlobalAvgPool
├── lower_fc.cpp             # Linear
├── lower_fusion.cpp         # ConvBNReLU, CBRP 等融合算子
├── lower_block.cpp          # Bottleneck, BasicBlock, InvResidual
├── lower_upsample.cpp       # Upsample
├── lower_concat.cpp         # Concat
└── lower_comm.cpp           # AllReduce, BN_STATS_SYNC
```

---

## 八、实现路径

> **架构依赖**：arch_plan.h:179 `int first_layer_index() const;` — 确保此 getter 公开可用。若不存在，添加。
> **Config 字段**：`Config` 包含 `actual_resolution` 用于派生变体形状；`freeze_first_layer` 是运行时标记，不影响 Compiler 输出。

| 周次 | 目标 | 交付物 | 验收标准 |
|---|---|---|---|
| 第1周 | 头文件 + 骨架 | 全部 .h 文件 + `compute_slot_bytes` + `alloc(shape, dtype, region, slot_bytes)` + `compiler.cpp` Phase 1-3 骨架 | 编译通过 |
| 第2周 | LeNet-5 端到端 | `lower_common/conv/bn/pool/fc.cpp` + base + lowres 双变体 | base 和 lowres MemoryPlan offset 一致 |
| 第3周 | ResNet-50 + last batch | `lower_fusion.cpp` + 融合/非融合双路径 + last batch 变体 | 所有 shape variant 切换正确 |
| 第4周 | 首层冻结 + 通信 | Executor 跳过首层反向图 + 通信图 + 优化器图 | frozen_first 不独立建图 |
| 第5周 | 推理 + EMA | 推理图 + EMA 图 | 全部 15 张有效图有节点 |
| 第6周 | 集成测试 | CUDA Graph 捕获 + 多分辨率端到端 | CRC 验证通过 |

---

## 九、一句话总结

> **DTensor 是形状权威，slot_bytes 构造时存储为常数，与 shape 解耦。每变体独立 MemoryPlan（shape/stride 各自正确，offset 由 max_slot_bytes 一致）。ComputationGraph 只存算子拓扑 + tensor_id（15 张有效图），shape-only 变体指针共享 base 图。Compiler 五 Phase，Phase 2 含跨 Config 一致性断言保护。Result 通过 variants[0]=base 统一访问，禁止移动。infer_tensors 遵循三模式并集 + 数量/顺序/name/region 跨 Config 不变的四铁律。**





# 【例子】

## 【问】

我问一个问题：以现在的COMP_ULTIMATE.md的架构，我要实现一个FC层的话，我给它一个输入，它会怎么推断输出形状？描述流程

## 【答1】

  完整推断流程

  1. 编译期阶段

  第一步：LayerDescriptor注册
  // 在 registry.cpp 中注册FC层
  const LayerDescriptor& get_layer_descriptor(LayerKind kind) {
      switch (kind) {
          case LayerKind::LINEAR:
              static LayerDescriptor linear_desc = {
                  .infer_tensors = infer_linear_tensors,
                  .build_forward = build_linear_forward,
                  .build_backward = build_linear_backward,
                  .build_inference = build_linear_inference,
              };
              return linear_desc;
          // ...
      }
  }

  第二步：实现infer_linear_tensors函数
  // lower_fc.cpp
  std::vector<TensorDesc> infer_linear_tensors(
      const Shape& input_shape,
      const OpParams& params,
      const InferContext& ctx)
  {
      // 输入：input_shape = [N, H_in, W_in, C_in]
      // 对于FC层，通常输入已经是flatten后的 [N, 1, 1, features_in]

      auto& linear_params = params.get<LinearParams>();
      int features_out = linear_params.out_features;
    
      std::vector<TensorDesc> tensors;
    
      // 0: 输入张量
      tensors.push_back({"input", input_shape,
                         Region::F_FEATURE_FP16, DType::FP16});
    
      // 1: 权重张量 [1, 1, features_in, features_out]
      Shape weight_shape(1, 1, input_shape.c(), features_out);
      tensors.push_back({"weight", weight_shape,
                         Region::W_EQ_WEIGHT, DType::FP32});
    
      // 2: 偏置张量 [1, 1, 1, features_out]
      Shape bias_shape(1, 1, 1, features_out);
      tensors.push_back({"bias", bias_shape,
                         Region::W_EQ_BIAS, DType::FP32});
    
      // 3: 输出张量 [N, 1, 1, features_out]
      Shape output_shape(input_shape.n(), 1, 1, features_out);
      tensors.push_back({"output", output_shape,
                         Region::F_FEATURE_FP16, DType::FP16});
    
      // 4-6: 反向传播用张量（梯度）
      tensors.push_back({"d_input", input_shape,
                         Region::F_FEATURE_FP16, DType::FP16});
      tensors.push_back({"d_weight", weight_shape,
                         Region::G_FC_WEIGHT, DType::FP32});
      tensors.push_back({"d_bias", bias_shape,
                         Region::G_FC_BIAS, DType::FP32});
    
      return tensors;  // 遵循铁律：返回三模式并集
  }

  第三步：Compiler调用推导
  // compiler.cpp Phase 1
  AllShapes derive_all_shapes(const ArchPlan& arch, const Config& cfg) {
      AllShapes result;
      Shape cur(cfg.batch_size, cfg.actual_resolution,
                cfg.actual_resolution, cfg.num_color_channels);

      for (int i = 0; i < arch.layers().size(); ++i) {
          const auto& layer = arch.layers()[i];  // FC层
          const auto& desc = get_layer_descriptor(layer.kind());
    
          // 调用 infer_linear_tensors
          auto tensors = desc.infer_tensors(cur, layer.params(), ctx);
    
          // 提取输出形状 - 使用辅助函数
          cur = get_output_shape(tensors);  // 找到name="output"的TensorDesc的shape
    
          result.per_layer.push_back(std::move(tensors));
      }
      return result;
  }

  第四步：get_output_shape辅助函数
  Shape get_output_shape(const std::vector<TensorDesc>& descs) {
      for (const auto& td : descs) {
          if (td.name == "output") {
              return td.shape;
          }
      }
      TR_THROW(ValueError, "No output tensor found in TensorDesc list");
  }

  2. 运行时阶段

  第五步：实际计算
  // Backend执行时
  const DTensor& input_dt = mem.get_dtensor(node.inputs[0]);   // id=0
  const DTensor& weight_dt = mem.get_dtensor(node.inputs[1]);  // id=1
  const DTensor& output_dt = mem.get_dtensor(node.outputs[0]); // id=3

  // DTensor 包含正确的形状信息
  // input_dt.shape = [N, 1, 1, features_in]
  // output_dt.shape = [N, 1, 1, features_out]

  // 使用DTensor的形状调用cuDNN/cuBLAS
  cublasetGemmStridedBatchedEx(
      input_dt.shape.n(),     // batch size
      features_out,           // m
      features_in,            // k
      1,                      // batch count
      // ... 使用 input_dt.stride, output_dt.stride 等
  );

  关键点总结

  1. 形状权威在DTensor：最终输出形状由infer_linear_tensors返回的TensorDesc确定
  2. 遵循张量一致性铁律：无论Config如何变化，返回的张量数量(7个)、顺序、name、region都不变
  3. 辅助函数桥梁：get_output_shape从TensorDesc列表中提取输出形状作为下一层输入
  4. 解耦设计：形状推导(编译期)与实际计算(运行期)完全分离，DTensor存储权威形状信息

  这个流程确保了：
  - 编译期确定所有张量形状
  - 运行期直接使用预计算的形状信息
  - 不同变体(base/lowres)的同一tensor_id指向语义相同的张量

## 【答2】

FC 层的形状推断流程如下：
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  总体流程概览
  Config(batch=N, actual_resolution=224)
      │
      ▼
  Phase 1: derive_all_shapes()
      │
      ├── 构造输入 Shape: [N, 224, 224, 3]  ← 或前一层的输出
      │
      ├── 遍历到 FC 层 (LayerKind::LINEAR)
      │       │
      │       ▼
      │   get_layer_descriptor(LINEAR) ──→ registry.cpp 中的 switch case
      │       │
      │       ▼
      │   infer_tensors(input_shape=[N,H,W,C], op_params, ctx)
      │       │
      │       ▼
      │   返回 TensorDesc 列表（三模式并集）
      │       │
      │       ▼
      │   get_output_shape(tensors) ──→ 提取 "output" 张量的 shape
      │       │
      │       ▼
      │   下一层输入 shape = [N, 1, 1, out_features]
      │
      ▼
  Phase 2: compute_max_slot_bytes() ──→ 取跨变体最大值
  Phase 3: create_memory_plan() ──→ 按 output shape 分配 DTensor
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  详细步骤拆解
  步骤 1：FC 层的 infer_tensors 实现
  在 src/graph/compiler/lower_fc.cpp 中：
  std::vector<TensorDesc> infer_linear(const Shape& input_shape,
                                        const OpParams& params,
                                        const InferContext& ctx)
  {
      // OpParams 中包含 FC 层的配置，如 out_features
      int out_features = params.get<int>("out_features");

      // 输入可能是 4D [N,H,W,C] 或已经 Flatten 的 [N,1,1,C]
      // FC 输出总是 [N, 1, 1, out_features]
      Shape output_shape(input_shape.n(), 1, 1, out_features);
    
      std::vector<TensorDesc> descs;
    
      // 0: input —— 上一层的输出（无论是否 Flatten，逻辑 shape 由输入决定）
      descs.push_back({"input", input_shape, Region::F_FEATURE_FP32, DType::FP32});
    
      // 1: weight —— [in_features, out_features] 展平后存为 [1,1,in_features,out_features]
      int in_features = input_shape.h() * input_shape.w() * input_shape.c();
      Shape weight_shape(1, 1, in_features, out_features);
      descs.push_back({"weight", weight_shape, Region::W_FC_WEIGHT, DType::FP32});
    
      // 2: bias —— [out_features]
      Shape bias_shape(1, 1, 1, out_features);
      descs.push_back({"bias", bias_shape, Region::W_FC_BIAS, DType::FP32});
    
      // 3: output —— FC 结果
      descs.push_back({"output", output_shape, Region::F_FEATURE_FP32, DType::FP32});
    
      // ===== 三模式并集铁律：反向/推理需要的张量也必须返回 =====
      // 4: dX —— 输入梯度（反向需要）
      descs.push_back({"dX", input_shape, Region::F_GRAD_SLOT_FP32, DType::FP32});
    
      // 5: dW —— 权重梯度（反向需要）
      descs.push_back({"dW", weight_shape, Region::G_FC_WEIGHT, DType::FP32});
    
      // 6: dBias —— 偏置梯度（反向需要）
      descs.push_back({"dBias", bias_shape, Region::G_FC_BIAS, DType::FP32});
    
      // 若 enable_amp，额外返回 FP16 权重占位符（保持索引一致）
      if (ctx.enable_amp) {
          descs.push_back({"weight_fp16", weight_shape, Region::A_FC_WEIGHT, DType::FP16});
      } else {
          // 非 AMP 配置下用占位符保持数量/顺序一致（铁律2）
          descs.push_back({"unused_weight_fp16", Shape(), Region::DEFAULT, DType::FP32});
      }
    
      return descs;  // 共 8 个张量，所有 Config 下数量相同
  }
  关键：铁律约束
  • 铁律1：返回了 TRAIN_FWD（0-3）、TRAIN_BWD（4-6）、INFERENCE（0-3）的并集
  • 铁律2：无论 AMP 是否开启，都返回 8 个张量。AMP 时 descs[7] 是有效 FP16 权重，非 AMP 时是 unused_weight_fp16 占位符
  • 铁律3：weight_fp16 在 AMP 和非 AMP 配置下都位于索引 7，保证 tensor_id 映射一致
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  步骤 2：get_output_shape 提取输出形状
  Shape get_output_shape(const std::vector<TensorDesc>& descs) {
      for (const auto& t : descs) {
          if (t.name == "output")
              return t.shape;  // ← 返回 [N, 1, 1, out_features]
      }
      TR_UNREACHABLE();
  }
  derive_all_shapes 用这个 shape 作为 下一层的输入。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  步骤 3：Phase 2-3 中的跨变体一致性
  // Phase 2: validate_tensor_consistency 会验证：
  // - base (224×224) 和 lowres (128×128) 的 FC 层都有 8 个 TensorDesc
  // - descs[0].name == "input", descs[1].name == "weight" ... 完全一致
  // - descs[7].name 在 base 是 "weight_fp16" 或 "unused_weight_fp16"，
  //   但在所有 Config 下都位于索引 7
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  步骤 4：build_forward 构建计算图
  SubgraphPattern build_linear_forward(const OpParams& params,
                                        const std::vector<TensorDesc>& descs)
  {
      SubgraphPattern pattern;
      // 只需要 input(0), weight(1), bias(2), output(3)
      pattern.nodes.push_back({
          ComputeOp::LINEAR,
          {0, 1, 2},   // input_indices: input, weight, bias
          {3}          // output_indices: output
      });
      return pattern;
  }
  ComputationGraph::Node 存储的是 tensor_id（即 lctx.tensor_ids[3]），不是 shape。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  步骤 5：运行时 Kernel 使用 DTensor 自身的 stride
  当 Executor 切换到 lowres 变体时：
  // lowres MemoryPlan 中的 FC output DTensor:
  //   shape = [N, 1, 1, out_features]
  //   stride = [out_features, out_features, out_features, 1]  ← 按自身 shape 计算

  // base MemoryPlan 中的 FC output DTensor:
  //   shape = [N, 1, 1, out_features]  （如果 FC 前接 GAP，shape 不变）
  //   stride 相同

  // 若 FC 前接 Conv 输出（如 ResNet-50 最后的 2048 维）：
  //   base:   shape=[N,7,7,2048] → stride=[7*7*2048, 7*2048, 2048, 1]
  //   lowres: shape=[N,4,4,2048] → stride=[4*4*2048, 4*2048, 2048, 1]
  Kernel 必须使用 dt.n_stride / h_stride / w_stride / c_stride，禁止硬编码 max_shape 的 stride。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一句话总结

  ▌ FC 层的形状推断由 infer_tensors() 完成：根据输入 Shape 和 OpParams.out_features 计算出输出 [N, 1, 1, out_features]
  ▌ ，封装到 TensorDesc 列表中。get_output_shape() 从列表提取 "output" 的 shape 传递给下一层。所有 Config 下返回相同数
  ▌ 量/顺序的 TensorDesc（不用的占位），保证 tensor_id 跨变体映射一致。ComputationGraph 只存 tensor_id，实际 stride 从
  ▌ 各变体独立的 MemoryPlan DTensor 获取。

## 【答3】

以 FC 层为例追踪形状推断的完整流程。

## 场景设定
假设 ResNet-50 最后有一个 FC 分类层：

```
Input: [32, 1, 1, 2048]     (NHWC: N=32, C=2048)
FC params: in_features=2048, out_features=1000
AMP: 关闭
Is first layer: 否
```
## 流程图
```
┌─────────────────────────────────────────────────────────────────┐
│  ArchPlan.layers()[i] = { LayerKind::FC,                        │
│    params = FcParams{in=2048, out=1000},                        │
│    is_first_layer = false }                                     │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 1: Compiler 调用 desc.infer_tensors(input_shape, params, ctx)│
│                                                         │
│    input_shape = [32, 1, 1, 2048]      ← 上一层 get_output_shape │
│    params      = {in=2048, out=1000}                             │
│    ctx         = { mode: TRAIN_FWD, enable_amp: false,          │
│                    is_first_layer: false }                       │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 2: 在 lower_fc.cpp 的 fc_infer_tensors() 内部：              │
│                                                         │
│    int64_t in_f  = params.in;        // 2048                    │
│    int64_t out_f = params.out;       // 1000                    │
│    int64_t N     = input_shape.n();  // 32                      │
│                                                         │
│    计算输出形状:                                                  │
│    Shape out_shape{N, 1, 1, out_f};  // [32, 1, 1, 1000]       │
│                                                         │
│    返回 三模式并集（铁律1）:                                     │
│    ┌─────┬──────────┬───────────────────┬───────────────┐      │
│    │ idx │ name     │ shape             │ region        │      │
│    ├─────┼──────────┼───────────────────┼───────────────┤      │
│    │  0  │ "input"  │ [32,1,1,2048]     │ F_FEATURE_FP32│      │
│    │  1  │ "weight" │ [1000,1,1,2048]   │ W_FC_WEIGHT   │      │
│    │  2  │ "bias"   │ [1,1,1,1000]      │ W_FC_BIAS     │      │
│    │  3  │ "output" │ [32,1,1,1000]     │ F_FEATURE_FP32│      │
│    │  4  │ "wk"     │ ...(cuBLAS)       │ T_TEMP_FP32   │      │
│    │  5  │ "d_input" │ [32,1,1,2048]    │ F_GRAD_FC     │      │
│    │  6  │ "d_weight"│ [1000,1,1,2048]  │ G_FC_WEIGHT   │      │
│    │  7  │ "d_bias" │ [1,1,1,1000]      │ G_FC_BIAS     │      │
│    │  8  │ "unused" │ empty             │ UNUSED        │  ← 铁律2 占位│
│    └─────┴──────────┴───────────────────┴───────────────┘      │
│                                                         │
│    (若 AMP=true: 索引 8 变为 "grad_fp16"，shape=[1000,1,1,2048] │
│     铁律2 保证列表长度始终=9，铁律3 保证同一语义在固定索引)      │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 3: cur = get_output_shape(tensors)                       │
│                                                         │
│    遍历 descs[]，找到 region == F_FEATURE_FP32                  │
│    且 name == "output" 的条目                                   │
│    → 提取 descs[3].shape                                       │
│    → cur = [32, 1, 1, 1000]                                    │
│                                                         │
│    此值成为下一层的 input_shape                                │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 4: 下一轮迭代的 Step 1                                    │
│    desc.infer_tensors([32, 1, 1, 1000], params_next, ctx_next)  │
└────────────────────────────────────────────────────────────────┘
```
## 关键点
形状推导全在 LayerDescriptor 内完成。 infer_tensors 函数接收 input_shape + params + ctx ，纯数学计算得到 output_shape ，塞进 TensorDesc 列表。Compiler 不做任何形状计算——它只是"收菜"：调 infer_tensors ，拿返回值，用 get_output_shape 从列表中取出输出 shape，喂给下一层。

get_output_shape 的实现 ：遍历 TensorDesc 列表，找 region 为 F_FEATURE_* 且 name 为 "output" 的条目，返回其 shape 。

四条铁律在此流程中的体现 ：

- 铁律1（并集） ：FC 的 infer_tensors 返回 9 个条目，覆盖正向（input/weight/bias/output/workspace）和反向（d_input/d_weight/d_bias）及 AMP 占位
- 铁律2（等量） ：无论 enable_amp 、 is_first_layer 怎么变，始终返回 9 个条目
- 铁律3（等序） ：weight 永远在索引 1，output 永远在索引 3，梯度永远在索引 5-7
- 铁律4（等名等域） ：索引 1 永远叫 "weight" ，region 永远是 W_FC_WEIGHT ，跨所有 Config 不变