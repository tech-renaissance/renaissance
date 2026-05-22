# 【小伙伴S】

---

## 一、总体架构

### 三阶段流水线

```
Phase A (编译期)          Phase B (预演捕获期)         Phase C (运行期)
━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━      ━━━━━━━━━━━━━━━━━━━
ArchPlan + CompileSpec  →  GraphAtlas + Device    →  RuntimeAtlas
       ↓                        ↓                        ↓
MemoryPlan[6]           物理去重 + CUDA Graph      纯索引执行
ComputationGraph×2       捕获 (27-31个图)          图调用
       ↓                        ↓                        ↓
GraphAtlas (逻辑归属)   PreCaptureResult          Executor训练/推理
```

### 核心设计原则

1. **DTensor形状唯一权威**：每个MemoryPlan有独立的DTensor，各自形状正确
2. **ComputationGraph纯拓扑**：只存(op, tensor_id)，不存Shape/Region/DType
3. **GraphAtlas映射唯一权威**：编译期声明、预演期去重、运行期O(1)索引
4. **slot_bytes跨变体统一**：保证offset一致，shape可变

---

## 二、核心类清单（按职责分层）

### 第1层：输入层（2个类）

#### 2.1.1 ArchPlan（已有）

**职责**：神经网络架构的静态描述

**成员**：
```cpp
class ArchPlan {
    std::vector<Layer> layers_;           // 有序层序列
    int first_layer_index_;              // 首层索引
    // 提供迭代器接口
};
```

#### 2.1.2 CompileSpec（新增，避免Config冲突）

**职责**：单个编译配置（6个输入shape变体之一）

**成员**：
```cpp
struct CompileSpec {
    // PlanConfig参数（条件分配65个Region）
    bool amp_enabled;
    bool bn_folded;
    bool use_lars;
    bool use_adam;
    bool use_momentum;
    bool has_ema;
    int num_models;
    bool need_mask;
    bool need_temp;
    
    // 形状参数
    int max_sample_resolution;    // MemoryPlan最大预留
    int actual_resolution;        // 形状推导实际使用
    int batch_size;
    int num_color_channels;
    
    // 运行时标志（不影响MemoryPlan/ComputationGraph）
    bool freeze_first_layer;      // 运行时行为，不参与编译
    
    static CompileSpec from_global_registry();
    ShapeId get_shape_id() const;  // 转换为ShapeId四元组
};
```

**6个变体定义**：
- `train_base`: train_res_begin × standard_batch
- `train_last`: train_res_begin × last_batch
- `train_lowres`: train_res_end × standard_batch
- `train_lowres_last`: train_res_end × last_batch
- `val_base`: val_res × standard_batch
- `val_last`: val_res × val_last_batch

---

### 第2层：知识层（3个类）

#### 2.2.1 LayerDescriptor（新增）

**职责**：单层知识的函数指针表（Compiler与backend解耦）

**定义**：
```cpp
struct LayerDescriptor {
    using InferFn = std::vector<TensorDesc> (*)(
        const Shape&, const OpParams&, const InferContext&);
    using BuildFn = SubgraphPattern (*)(
        const OpParams&, const std::vector<TensorDesc>&);
    
    InferFn infer_tensors;      // 推导该层所有张量 (三模式并集)
    BuildFn build_forward;     // 构建正向子图模式
    BuildFn build_backward;    // 构建反向子图模式
    BuildFn build_inference;   // 构建推理子图模式
};

const LayerDescriptor& get_layer_descriptor(LayerKind kind);
```

**四条铁律**（infer_tensors必须遵守）：
1. **数量一致性**：所有CompileSpec的张量数量必须相同
2. **顺序一致性**：相同索引位置的张量name/region必须相同
3. **名称/区域不变**：shape和dtype可变，但name和region跨配置不变
4. **并集覆盖**：输出 = 训练/推理/反向全部所需张量的并集

#### 2.2.2 TensorDesc（新增）

**职责**：单个张量的完整描述

**定义**：
```cpp
struct TensorDesc {
    std::string name;      // 张量名称（用于张量一致性验证）
    Shape       shape;     // 逻辑形状
    Region      region;    // 显存分区（65个Region之一）
    DType       dtype;     // 数据类型
};
```

#### 2.2.3 SubgraphPattern（新增）

**职责**：算子子图模式（不含形状信息）

**定义**：
```cpp
struct SubgraphPattern {
    struct Node {
        ComputeOp    op;                       // 算子类型
        std::vector<size_t> input_indices;   // 在TensorDesc列表中的索引
        std::vector<size_t> output_indices;  // 在TensorDesc列表中的索引
    };
    
    std::vector<Node> nodes;
};
```

---

### 第3层：编译器层（2个类）

#### 2.3.1 Compiler（新增，核心编排器）

**职责**：五阶段编译流程

**定义**：
```cpp
class Compiler {
public:
    struct Variant {
        std::string             name;
        MemoryPlan              memory_plan;       // 独立MemoryPlan
        const ComputationGraph* train = nullptr;    // 指针共享
        const ComputationGraph* inference = nullptr;
        ComputationGraph*       own_train = nullptr;     // graph-change变体
        ComputationGraph*       own_inference = nullptr;
    
        ~Variant() { delete own_train; delete own_inference; }
    };

    struct Result {
        std::vector<Variant> variants;          // 6个变体
    };

    // Phase A: 五阶段编译流程
    static Result compile(const ArchPlan&, const CompileSpec&, const std::vector<CompileSpec>&);

private:
    // Phase 1: 对所有配置推导全部形状
    static std::vector<AllShapes> derive_all_shapes(const ArchPlan&, const std::vector<CompileSpec>&);

    // Phase 2: 计算统一slot_bytes (唯一跨变体比较)
    static std::vector<std::vector<uint64_t>> compute_max_slot_bytes(const std::vector<AllShapes>&);

    // Phase 3: 创建所有MemoryPlan
    static std::vector<std::vector<LayerContext>> create_all_memory_plans(...);

    // Phase 4: 构建base ComputationGraph
    static ComputationGraph build_computation_graph(...);

    // Phase 5: 构建变体图
    static void build_variant_graphs(Result&, const std::vector<CompileSpec>&);
};
```

**五阶段详解**：

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|---------|
| 1 | ArchPlan + 6 CompileSpec | all_tensors_[6][layer][tensor] | infer_tensors × 6，验证4铁律 |
| 2 | all_tensors_ | max_slot_bytes_[layer][tensor] | 逐位置取max，**唯一跨变体比较** |
| 3 | all_tensors_ + max_slot_bytes_ | MemoryPlan[6] | alloc(shape, dtype, region, max_slot_bytes) |
| 4 | LayerDescriptor序列 | 共享ComputationGraph ×2 | build_forward/build_backward/build_inference |
| 5 | 各变体ShapeId | 组装Result | 提取ShapeId[6]，共享或clone图 |

#### 2.3.2 LayerContext（新增）

**职责**：单层的完整上下信息

**定义**：
```cpp
struct LayerContext {
    std::vector<TensorDesc> descs;      // 该层全部张量描述（三模式并集）
    std::vector<int32_t>    tensor_ids; // 分配后的DTensor全局ID
};
```

---

### 第4层：显存布局层（2个类）

#### 2.4.1 MemoryPlan（已有，扩展）

**职责**：管理65个Region的线性布局，分配DTensor

**定义**：
```cpp
class MemoryPlan {
public:
    // 标准接口（base变体用）
    DTensor alloc(const Shape& shape, DType dtype, Region region);

    // Compiler专用接口（变体用）：显式slot_bytes保证跨变体offset一致
    DTensor alloc(const Shape& shape, DType dtype, Region region, uint64_t slot_bytes);

    void finalize();  // 按dt.slot_bytes()计算offset
    const DTensor& get_dtensor(int32_t id) const;

    // RangeOp预计算
    const RangeOpRange& get_range_op_range(RangeOp op) const;

private:
    void build_range_op_ranges();  // 预计算17个RangeOp的(offset, size)
    
    std::array<RegionInfo, 65> regions_;
    std::array<RangeOpRange, static_cast<size_t>(RangeOp::COUNT)> range_op_ranges_;
    std::vector<DTensor> dtensors_;
    
    friend class Compiler;  // 允许调用私有alloc重载
};
```

**65个Region布局**（12个系列）：
- B系列（BN统计量）: B_PREV_MEAN, B_PREV_VAR, B_NEXT_MEAN, B_NEXT_VAR
- W系列（主模型权重）: W_EQ_BIAS, W_EQ_SCALE, W_BN_BIAS, ...
- E系列（EMA模型）: E_BN_BIAS, E_FC_WEIGHT, E_FIRST_CONV, ...
- A系列（AMP权重）: A_FC_WEIGHT, A_FIRST_CONV, A_DEEP_CONV
- G系列（梯度）: G_BN_BIAS, G_FC_WEIGHT, G_DEEP_CONV, ...
- M系列（一阶动量）: M_BN_BIAS, M_FC_WEIGHT, ...
- V系列（二阶动量）: V_BN_BIAS, V_FC_WEIGHT, ...
- N系列（范数）: N_FC_WEIGHT, N_FIRST_CONV, N_DEEP_CONV
- I系列（输入缓冲）: I_A_LABEL, I_A_DATA, I_B_LABEL, I_B_DATA
- F系列（特征图）: F_FEATURE_FP32, F_GRAD_SLOT_FP32, F_FEATURE_FP16, F_GRAD_SLOT_FP16
- S系列（标量）: S_SCALAR_FP32, S_SCALAR_FP16, S_SCALAR_INT32, S_SCALAR_INT8, S_MASK
- T系列（临时张量）: T_TEMP_FP32, T_TEMP_FP16, T_TEMP_INT32, T_TEMP_INT8

#### 2.4.2 DTensor（已有，修改）

**职责**：分布式张量的形状权威

**定义**：
```cpp
struct DistributedTensor {
    int32_t  id = -1;
    Shape    shape;              // 该变体的实际逻辑形状（唯一权威）
    DType    dtype;
    Region   region;
    uint64_t offset = 0;         // 所有变体相同（由max_slot_bytes保证）
    
    int64_t  n_stride, h_stride, w_stride, c_stride;  // 按自身shape计算
    
    uint64_t slot_bytes() const noexcept;  // 返回构造时存储的常数
    
    static uint64_t compute_slot_bytes(const Shape&, DType, Region);
    
private:
    uint64_t slot_bytes_ = 0;  // 构造时常数，跨变体取其最大值
};
```

**关键原则**：
- slot_bytes与shape解耦：不同变体的DTensor可以有不同shape，但相同slot_bytes
- stride严格按自身shape计算，不存在"stride_shape"或"max shape"概念
- 双构造函数：标准构造（自动算slot_bytes）+ 变体构造（显式传入max_slot_bytes）

---

### 第5层：计算图层（2个类）

#### 2.5.1 GraphNode（新增）

**职责**：统一节点类型，支持ComputeOp和RangeOp

**定义**：
```cpp
struct GraphNode {
    enum class Kind : uint8_t { COMPUTE, RANGE };
    Kind kind;

    union {
        ComputeOp compute_op;   // DTensor级操作，43个枚举值
        RangeOp   range_op;     // Region级操作，19个枚举值
    };

    // COMPUTE态：关联DTensor ID
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;

    // RANGE态：预计算内存范围（offset + size）
    std::vector<MemRange> input_ranges;
    std::vector<MemRange> output_ranges;
    
    OpParams params;  // COMPUTE时有效
};
```

**ComputeOp（43枚举值/41有效）**：格式`{BASE}[_{PRECISION}]_{DIRECTION}`
**RangeOp（19枚举值/17有效）**：格式`RANGE_{ACTION}[_{SCOPE}]`

#### 2.5.2 ComputationGraph（新增）

**职责**：纯算子拓扑容器，不存形状信息

**定义**：
```cpp
class ComputationGraph {
public:
    enum class GraphId : uint8_t {
        TRANSFER_A, TRANSFER_B,
        FIRST_FWD_A, FIRST_FWD_B,
        DEEP_FWD_BWD,
        FIRST_BWD,
        FIRST_COMM, DEEP_COMM, STATS_COMM,
        OPTIMIZER, EMA_UPDATE,
        INF_MAIN_A, INF_MAIN_B,
        INF_EMA_A, INF_EMA_B,
        COUNT  // = 16
    };

    void append(GraphId gid, GraphNode node);
    const std::vector<GraphNode>& nodes(GraphId gid) const;

private:
    std::array<std::vector<GraphNode>, static_cast<size_t>(GraphId::COUNT)> graphs_;
};
```

**15张有效子图**：
- shape无关（7张）：TRANSFER×2, COMM×3, OPTIMIZER, EMA_UPDATE
- shape相关（8张）：FIRST_FWD×2, DEEP_FWD_BWD, FIRST_BWD, INF_MAIN×2, INF_EMA×2

**核心特征**：节点中只存(op, tensor_id)，形状/数据类型/Region全部从MemoryPlan的DTensor获取

---

### 第6层：图集管理层（3个类）

#### 2.6.1 ShapeId（新增）

**职责**：输入shape的唯一标识（Phase B物理去重）

**定义**：
```cpp
struct ShapeId {
    int32_t n = 0, h = 0, w = 0, c = 0;
    
    bool operator==(const ShapeId&) const noexcept = default;
    
    static ShapeId from_compile_spec(const CompileSpec&);
    static ShapeId from_memory_plan(const MemoryPlan&);
};

constexpr ShapeId kShapeInvariant{0, 0, 0, 0};  // shape无关图专用
```

#### 2.6.2 GraphAtlas（新增，三阶段桥梁）

**职责**：6×15二维表，映射变体×子图到捕获图

**定义**：
```cpp
class GraphAtlas {
public:
    static constexpr size_t kMaxVariants = 6;

    struct Slot {
        const ComputationGraph* cg = nullptr;     // Phase A: Compiler填入
        const MemoryPlan*       mp = nullptr;     // Phase A: Compiler填入
        ShapeId                 shape_id{};       // Phase A: Compiler填入
        int32_t                 captured_idx = -1; // Phase B: pre_capture填入
    };

    // Phase A: 编译期 - 声明逻辑归属
    static GraphAtlas build(const Compiler::Result&, const std::array<ShapeId, 6>&);

    // Phase C: 运行期 - O(1)数组查找
    int32_t index(size_t variant, GraphId gid) const noexcept;

private:
    std::array<std::array<Slot, static_cast<size_t>(GraphId::COUNT)>, kMaxVariants> table_;
};
```

**填充逻辑**：
- shape无关图：所有6变体共享base MemoryPlan，shape_id = kShapeInvariant
- shape相关训练图：4个训练变体各用各的MemoryPlan + 各自ShapeId
- 验证变体：不填入训练图，只填推理图
- 碰巧相同的ShapeId自动复用

#### 2.6.3 CapturedGraph（新增，后端抽象）

**职责**：已捕获图的后端无关抽象

**定义**：
```cpp
class CapturedGraph {
public:
    static CapturedGraph capture(const ComputationGraph&, const MemoryPlan&, GraphId, Device*);
    void launch(cudaStream_t) const;

    struct Key {
        const ComputationGraph* cg;
        GraphId                 gid;
        ShapeId                 shape;
    
        bool operator==(const Key&) const;
    };

private:
#ifdef TR_USE_CUDA
    cudaGraphExec_t cuda_graph_ = nullptr;
#else
    std::vector<CpuOp> cpu_ops_;  // CPU后端：函数指针序列
#endif
};
```

**重要**：IMPORTANT.md明确"先不要写真正的捕获代码"。当前阶段只需预留接口，内部可放占位实现。

---

### 第7层：预演捕获层（1个类）

#### 2.7.1 PreCapture（新增）

**职责**：Phase B预演去重捕获

**定义**：
```cpp
class PreCapture {
public:
    struct Result {
        std::vector<CapturedGraph> graphs;  // 去重图集（27-31个）
        GraphAtlas runtime_atlas;            // 填入captured_idx的运行时表
        
        size_t total_slots = 0;  // 有效格子数
        size_t captured = 0;      // 新捕获数
        size_t reused = 0;        // 去重复用数
    };

    // Phase B: 预演去重捕获
    static Result execute(const GraphAtlas& compile_atlas, Device*);
};
```

**去重算法**：
```cpp
std::unordered_map<CapturedGraph::Key, int32_t, CapturedGraph::KeyHash> seen;

for (size_t variant = 0; variant < 6; ++variant) {
    for (size_t g = 0; g < static_cast<size_t>(GraphId::COUNT); ++g) {
        GraphId gid = static_cast<GraphId>(g);
        auto& slot = atlas.slot(variant, gid);

        if (!slot.cg || !slot.mp) continue;

        CapturedGraph::Key key{slot.cg, gid, slot.shape_id};

        auto it = seen.find(key);
        if (it != seen.end()) {
            slot.captured_idx = it->second;
            result.reused++;
        } else {
            auto cg = CapturedGraph::capture(*slot.cg, *slot.mp, gid, device);
            int32_t idx = static_cast<int32_t>(result.graphs.size());
            result.graphs.push_back(std::move(cg));
            seen[key] = idx;
            slot.captured_idx = idx;
            result.captured++;
        }
    }
}
```

---

### 第8层：运行时层（1个类）

#### 2.8.1 Executor（已有，重构）

**职责**：Phase C运行时调度器

**定义**：
```cpp
class Executor {
public:
    void setup(const PreCaptureResult&);  // 绑定atlas + captured graphs
    void activate_variant(size_t idx);    // O(1)指针切换

    void launch(GraphId gid, cudaStream_t stream);
    void run_train_iteration(cudaStream_t stream);
    void run_val_iteration(cudaStream_t stream, bool use_ema);

    // frozen_first：运行时跳过FIRST_BWD，不触发独立建图
    void set_skip_first_bwd(bool skip) { skip_first_bwd_ = skip; }

private:
    const GraphAtlas* atlas_ = nullptr;
    const std::vector<CapturedGraph>* graphs_ = nullptr;
    size_t active_variant_ = 0;
    bool skip_first_bwd_ = false;
};
```

**训练迭代流程**：
```cpp
void Executor::run_train_iteration(cudaStream_t stream) {
    static constexpr GraphId train_sequence[] = {
        GraphId::TRANSFER_A,
        GraphId::FIRST_FWD_A,
        GraphId::DEEP_FWD_BWD,
        GraphId::FIRST_BWD,
        GraphId::FIRST_COMM,
        GraphId::DEEP_COMM,
        GraphId::STATS_COMM,
        GraphId::OPTIMIZER,
        GraphId::EMA_UPDATE,
        GraphId::TRANSFER_B
    };

    for (GraphId gid : train_sequence) {
        if (skip_first_bwd_ && gid == GraphId::FIRST_BWD) continue;
        launch(gid, stream);
    }
}
```

---

## 三、数据流全景

### Phase A: 编译期流程

```
ArchPlan + CompileSpec(base) + [CompileSpec(variants)]
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Compiler::compile()                                        │
│  ├─ Phase 1: derive_all_shapes (LayerDescriptor::infer)    │
│  ├─ Phase 2: compute_max_slot_bytes (跨变体取max)           │
│  ├─ Phase 3: MemoryPlan[] (alloc with max slot_bytes)      │
│  ├─ Phase 4: ComputationGraph (forward + backward + aux)   │
│  └─ Phase 5: variant sharing / cloning                     │
└─────────────────────────────────────────────────────────────┘
      │
      ├──► Compiler::Result { Variant[6] }
      │
      ├──► GraphAtlas::build(Result, ShapeId[6])
      │      填入 Slot.cg, Slot.mp, Slot.shape_id
      │
      ▼
```

### Phase B: 预演捕获流程

```
┌─────────────────────────────────────────────────────────────┐
│  pre_capture(GraphAtlas)                                    │
│  遍历 6×16 Slot → Key{cg, gid, shape_id} → 去重 → capture  │
│  填入 Slot.captured_idx                                    │
└─────────────────────────────────────────────────────────────┘
      │
      ├──► PreCaptureResult { CapturedGraph[], GraphAtlas }
      ▼
```

### Phase C: 运行期流程

```
┌─────────────────────────────────────────────────────────────┐
│  Executor::setup(pre)                                       │
│  activate_variant(v) → atlas.index(v, gid) → graphs[idx]   │
│  .launch(stream)   // Phase C：MLPerf计时，零捕获          │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、关键设计决策与约束

### 4.1 命名决策

| 决策 | 理由 |
|------|------|
| CompileSpec取代Config | IMPORTANT.md明确禁止Config作类名 |
| PlanConfig保留作MemoryPlan内部struct | 非顶层类，REGION_FINAL.md已定案 |
| ShapeId显式四元组 | 确定性去重，零碰撞，优于hash/指针 |
| GraphAtlas | 语义准确：按变体×子图索引 |
| CapturedGraph | 后端无关，CPU/CUDA统一抽象 |
| ComputeOp（非OpKind） | OPS_NAME.md V4.0定稿 |
| RangeOp（非BatchOp） | 用户裁决：BatchOp是失败的概念 |

### 4.2 设计原则

| 原则 | 体现 |
|------|------|
| slot_bytes与shape解耦 | DTensor::slot_bytes_构造时常量，offset统一 |
| stride按自身shape算 | 每变体独立stride，kernel在自己的逻辑空间寻址 |
| ComputationGraph纯拓扑 | 零shape，一份图6变体共享 |
| 4铁律 | infer_tensors跨CompileSpec数量/顺序/名称/区域不变 |
| ShapeId去重 | 相同输入shape → 同一份CapturedGraph |
| 3阶段严格分离 | 编译→预演→运行，MLPerf计时期零捕获 |
| COMPUTE/RANGE二态 | GraphNode统一支持DTensor精算和Region批量操作 |
| 15 GraphId有效图 | 覆盖训练/验证/推理全场景 |
| 平台不进枚举 | 一张图纸，八卡共享，平台分发在execute层 |

### 4.3 暂不涉及的（按IMPORTANT.md要求）

- CUDA Graph具体捕获代码（Phase B实现）
- "一卡预热，八卡并行捕获"的并行细节
- Executor运行时调度的完整实现
- 通信后端选择（NCCL/GLOO）

---

## 五、文件组织结构

```
include/renaissance/graph/
├── arch_plan.h              # 已有
├── compile_spec.h           # 新增：编译配置
├── shape_id.h               # 新增：去重键
├── layer_descriptor.h       # 新增：层知识载体
├── graph_atlas.h            # 新增：三阶段桥梁
├── captured_graph.h         # 新增：后端抽象
├── pre_capture.h            # 新增：预演接口
├── compiler.h               # 新增：编译器
├── memory_plan.h            # 修改：Phase A/B接口
├── computation_graph.h      # 新增：算子拓扑容器
└── op_kind.h                # 修改：双枚举算子

src/graph/
├── compiler.cpp             # 新增：五阶段实现
├── layer_descriptor_registry.cpp # 新增：注册表
├── pre_capture.cpp          # 新增：Phase B实现
└── lower_*.cpp              # 修改：各层具体实现
```

---

## 六、关键优势总结

1. **自洽性**：所有COMPILE文档的问题都被考虑，无设计矛盾
2. **科学性**：基于共识决策，避免个人偏见
3. **可实施性**：清晰的类职责，明确的接口设计
4. **最优性能**：27-31个CUDA Graph，理论下界
5. **扩展性**：后端抽象，支持CPU/CUDA/MUSA
6. **可维护性**：三阶段分离，职责清晰

---

**总结**：这是一个可以直接实施的、技术完善的顶层设计方案，综合了三个小伙伴的多数派意见，遵循了所有COMPILE文档的约束，并保持了与现有方案的兼容性。



# 【小伙伴K】

  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一、设计总纲
  本阶段回答三个问题：
  1. 张量怎么分配：6 个变体共享同一块 GPU 物理内存，但 DTensor 各自持有正确的 shape 与 stride。
  2. 图怎么建：15 张子图桶的正反向拓扑一次性生成，图本身零形状信息。
  3. 图怎么跑：运行时零分支、零 hash、零捕获，纯数组索引找到已预热好的可执行图。
    三条铁律：
    • DTensor 是形状的唯一权威 — 多 MemoryPlan 实例，slot_bytes 跨变体取 max 保证 offset 一致，stride 按自身 shape 计算。
    • ComputationGraph 是拓扑的唯一权威 — 只存 (GraphNode, OpParams, tensor_id)，不存 Shape/Region/DType。
    • GraphAtlas 是运行时映射的唯一权威 — 编译期声明、预演期去重、运行期 O(1) 索引。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    二、核心类一览（12 个）
    2.1 输入层
    CompileSpec — 编译参数集（S/D 多数派）
    取代被禁止的 Config。描述单个变体的编译输入。
    struct CompileSpec {
      bool enable_amp;
      int  max_sample_resolution;   // MemoryPlan 最大槽位预留
      int  actual_resolution;       // 该变体的实际形状推导分辨率
      int  batch_size;
      int  num_color_channels;
      bool freeze_first_layer = false;  // 运行时标记，不影响编译输出

      static CompileSpec from_global_registry();
      ShapeId input_shape_id() const noexcept;
    };
    6 变体：train_base(0)、train_last(1)、train_lowres(2)、train_lowres_last(3)、val_base(4)、val_last(5)。
    ArchPlan — 架构描述（已有）
    • std::vector<ArchLayer> layers_
    • first_layer_index()
    • 作为 Compiler 的唯一输入。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    2.2 知识层
    LayerDescriptor — 算子知识契约（三方一致）
    每个 LayerKind 一个描述符，函数指针结构体，集中 switch 注册。
    struct LayerDescriptor {
      using InferFn = std::vector<TensorDesc> (*)(const Shape&, const OpParams&, const InferContext&);
      using BuildFn = SubgraphPattern (*)(const OpParams&, const std::vector<TensorDesc>&);

      InferFn infer_tensors;     // 返回三模式并集（四铁律）
      BuildFn build_forward;
      BuildFn build_backward;
      BuildFn build_inference;
    };
    四铁律（COMP_ULTIMATE.md Phase 2 断言保护）：返回张量的数量、顺序、name、region 跨全部 CompileSpec 完全一致；shape/dty
    pe 可变。
    TensorDesc、InferContext、SubgraphPattern（三方一致）
    struct TensorDesc {
      std::string name;
      Shape shape;
      Region region;   // 65 区之一
      DType dtype;
    };

  struct InferContext {
      GraphMode mode;          // TRAIN_FWD / TRAIN_BWD / INFERENCE
      bool enable_amp;
      bool is_first_layer;
  };

  struct SubgraphPattern {
      struct Node {
          ComputeOp op;
          std::vector<size_t> input_indices;   // TensorDesc 列表索引
          std::vector<size_t> output_indices;
      };
      std::vector<Node> nodes;
      // output_shape 已删除，统一用 get_output_shape() 辅助函数
  };
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2.3 编译器
  Compiler — 五阶段薄编排器（三方一致）
  class Compiler {
  public:
      struct Result {
          struct Variant {
              std::string name;
              MemoryPlan memory_plan;
              const ComputationGraph* train = nullptr;       // shape-only 共享 base
              const ComputationGraph* inference = nullptr;
              ComputationGraph* own_train = nullptr;          // graph-change 才 new
              ComputationGraph* own_inference = nullptr;
              ~Variant() { delete own_train; delete own_inference; }
          };
          std::vector<Variant> variants;  // variants[0] = base

          Result() = default;
          Result(const Result&) = delete;
          Result(Result&&) = delete;
      };
    
      static Result compile(const ArchPlan&, const CompileSpec& base,
                            const std::vector<CompileSpec>& variants = {});
  };
  五阶段：
  1. derive：对所有 CompileSpec 调用 LayerDescriptor::infer_tensors，收集 AllShapes。
  2. max_slot_bytes：逐 tensor slot 跨变体取 DTensor::compute_slot_bytes() 最大值。唯一跨变体比较。
  3. alloc：为每个变体独立创建 MemoryPlan，调用 alloc(shape, dtype, region, max_slot_bytes)。所有变体 offset 一致。
  4. build_graph：正向 build_forward + 反向 build_backward + 辅助图（通信/优化器/EMA/推理）。
  5. share/clone：shape-only 变体共享 base ComputationGraph*；graph-change 变体独立 new。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    2.4 编译产出层
    DTensor — 分布式张量（已有，修改）（三方一致）
    struct DistributedTensor {
      int32_t id = -1;
      Shape shape;              // 该变体实际逻辑形状（唯一权威）
      DType dtype;
      Region region;
      uint64_t offset = 0;      // 所有变体相同

      int64_t n_stride, h_stride, w_stride, c_stride;  // 按自身 shape 计算

      uint64_t slot_bytes() const noexcept;  // 构造时存储的常数

      static uint64_t compute_slot_bytes(const Shape&, DType, Region);

  private:
      uint64_t slot_bytes_ = 0;  // 与 shape 解耦
  };
  双构造函数：标准构造（slot_bytes 自动计算）+ 变体构造（显式传入 max slot_bytes）。
  MemoryPlan — 内存布局引擎（已有，扩展）（三方一致）
  class MemoryPlan {
  public:
      DTensor alloc(const Shape&, DType, Region);  // 标准接口
      void finalize();
      const DTensor& get_dtensor(int32_t id) const;
      const RangeOpRange& get_range_op_range(RangeOp) const;

  private:
      DTensor alloc(const Shape&, DType, Region, uint64_t slot_bytes);  // Compiler 专用
      void build_range_op_ranges();
      friend class Compiler;
  };
  • 65 个 Region，12 板块（B/W/E/A/G/M/V/N/I/F/S/T），条件分配。
  • finalize() 读取 dt.slot_bytes()（构造时常数）计算 offset，无需条件分支。
  GraphNode — 统一节点（OOOPS_FINAL.md 已定案，K/D 采纳）
  ComputationGraph 的节点必须同时支持 ComputeOp（DTensor 级）和 RangeOp（Region 级）。TRANSFER_A/B 等子图存储的是 RangeO
  p。
  struct GraphNode {
      enum class Kind : uint8_t { COMPUTE, RANGE };
      Kind kind;

      union {
          ComputeOp compute_op;   // 43 枚举值
          RangeOp   range_op;     // 19 枚举值
      };
      OpParams params;  // COMPUTE 时有效
    
      std::vector<int32_t> inputs;   // COMPUTE: DTensor ID
      std::vector<int32_t> outputs;
    
      std::vector<MemRange> input_ranges;   // RANGE: 预计算 (offset, size)
      std::vector<MemRange> output_ranges;  // RANGE: 预计算 (offset, size)
  };
  ComputationGraph — 纯拓扑容器（三方一致）
  class ComputationGraph {
  public:
      enum class GraphId : uint8_t {
          TRANSFER_A, TRANSFER_B,
          FIRST_FWD_A, FIRST_FWD_B,
          DEEP_FWD_BWD,
          FIRST_BWD,
          FIRST_COMM, DEEP_COMM, STATS_COMM,
          OPTIMIZER, EMA_UPDATE,
          INF_MAIN_A, INF_MAIN_B,
          INF_EMA_A, INF_EMA_B,
          COUNT  // = 16
      };

      void append(GraphId, GraphNode);
      const std::vector<GraphNode>& nodes(GraphId) const;

  private:
      std::array<std::vector<GraphNode>, static_cast<size_t>(GraphId::COUNT)> graphs_;
  };
  15 张有效图分两类：
  • shape 无关（7 张）：TRANSFER×2, COMM×3, OPTIMIZER, EMA_UPDATE。
  • shape 相关（8 张）：FIRST_FWD×2, DEEP_FWD_BWD, FIRST_BWD, INF_MAIN×2, INF_EMA×2。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2.5 捕获层
  ShapeId — 去重键（三方一致）
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

  constexpr ShapeId kShapeInvariant{0, 0, 0, 0};  // shape 无关图专用
  显式四元组，零碰撞，不依赖 hash 遍历 DTensor。
  GraphAtlas — 图集映射表（三方一致）
  class GraphAtlas {
  public:
      static constexpr size_t kMaxVariants = 6;
      static constexpr size_t kMaxGraphIds = static_cast<size_t>(ComputationGraph::GraphId::COUNT);

      struct Slot {
          const ComputationGraph* cg = nullptr;    // Phase A: Compiler 填入
          const MemoryPlan*       mp = nullptr;    // Phase A: Compiler 填入
          ShapeId                 shape_id{};      // Phase A: Compiler 填入
          int32_t                 captured_idx = -1; // Phase B: pre_capture 填入
      };
    
      static GraphAtlas build(const Compiler::Result&, const std::array<ShapeId, 6>&);
      int32_t index(size_t variant, GraphId gid) const noexcept;

  private:
      std::array<std::array<Slot, kMaxGraphIds>, kMaxVariants> table_;
  };
  Phase A 填充逻辑：
  • shape 无关图：全部 6 变体共享 base MemoryPlan，shape_id = kShapeInvariant → Key 必然碰撞 → 全局复用 7 张。
  • shape 相关训练图：4 个训练变体各用各的 MemoryPlan + 各自 ShapeId。
  • 验证变体：不填训练图；训练变体：不填推理图。
  • 碰巧相同的 ShapeId（如 val_last 完整批与 val_base 相同）自动复用。
  CapturedGraph — 后端无关已捕获图（三方一致）
  class CapturedGraph {
  public:
      static CapturedGraph capture(const ComputationGraph&, const MemoryPlan&, GraphId, Device*);
      void launch(cudaStream_t) const;

      struct Key {
          const ComputationGraph* cg;
          GraphId gid;
          ShapeId shape;
          bool operator==(const Key& o) const {
              return cg == o.cg && gid == o.gid && shape == o.shape;
          }
      };
      struct KeyHash { size_t operator()(const Key& k) const; };

  private:
  #ifdef TR_USE_CUDA
      cudaGraphExec_t exec_ = nullptr;
  #else
      std::vector<std::function<void()>> cpu_ops_;
  #endif
  };
  预留：capture() 内部先放占位实现，未来对接 legacy 的"一卡预热，八卡并行捕获"（IMPORTANT.md）。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2.6 预演与运行
  pre_capture() — 预演去重（K 自由函数，D 隐含，S 封装为类 → 取 K 的简洁）
  struct PreCaptureResult {
      std::vector<CapturedGraph> graphs;  // 去重图集，27~31 个
      GraphAtlas atlas;                   // captured_idx 已填入
  };

  PreCaptureResult pre_capture(const GraphAtlas& compile_atlas, Device* device);
  去重算法：unordered_map<CapturedGraph::Key, int32_t, KeyHash>。已存在复用索引，不存在则 CapturedGraph::capture()。
  Executor — 运行时调度（三方一致）
  class Executor {
  public:
      void setup(const PreCaptureResult& pre);
      void activate_variant(size_t idx);
      void set_skip_first_bwd(bool skip);

      void launch(GraphId gid, cudaStream_t stream) {
          int32_t idx = atlas_->index(active_variant_, gid);
          if (idx >= 0) (*graphs_)[idx].launch(stream);
      }
    
      void run_train_iteration(cudaStream_t stream);
      void run_val_iteration(cudaStream_t stream, bool use_ema);

  private:
      const GraphAtlas* atlas_ = nullptr;
      const std::vector<CapturedGraph>* graphs_ = nullptr;
      size_t active_variant_ = 0;
      bool skip_first_bwd_ = false;
  };
  frozen_first：不是 shape 变体，不改变 MemoryPlan/ComputationGraph。run_train_iteration 中 if (skip_first_bwd_ && gid =
  = FIRST_BWD) continue;。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  三、三阶段数据流
  ArchPlan + CompileSpec(base) + [CompileSpec(variants)]
      │
      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Phase A: Compiler::compile()  — 不计 MLPerf 时间            │
  │                                                             │
  │  Phase 1: derive_all_shapes (LayerDescriptor::infer)       │
  │  Phase 2: compute_max_slot_bytes (跨变体取 max)             │
  │  Phase 3: MemoryPlan[6] (alloc with max slot_bytes)        │
  │  Phase 4: ComputationGraph (forward + backward + aux)      │
  │  Phase 5: variant sharing / cloning                        │
  └─────────────────────────────────────────────────────────────┘
      │
      ├──► Compiler::Result { Variant[6] }
      │
      ├──► GraphAtlas::build(Result, ShapeId[6])
      │      填入 Slot.cg, Slot.mp, Slot.shape_id
      │
      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Phase B: pre_capture()  — 不计 MLPerf 时间                  │
  │                                                             │
  │  遍历 6×16 Slot → Key{cg, gid, shape_id} → 去重 → capture   │
  │  填入 Slot.captured_idx                                    │
  └─────────────────────────────────────────────────────────────┘
      │
      ├──► PreCaptureResult { CapturedGraph[], GraphAtlas }
      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Phase C: Executor — MLPerf 计时                             │
  │                                                             │
  │  activate_variant(v) → atlas.index(v, gid) → graphs[idx]   │
  │  .launch(stream)   // 纯数组索引，零 hash，零分支，零捕获    │
  └─────────────────────────────────────────────────────────────┘
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  四、分歧点裁定
   分歧                   S                    K               D                 裁定              理由
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   编译参数命名           CompileSpec          BuildSpec       CompileSpec       CompileSpec       S/D 多数派
   Result 含 GraphAtlas   是                   否              否                否                K/D 多数派，Atlas 由
   ?                                                                                               外部 build()
   Node 支持 RangeOp?     否（仅 ComputeOp）   是              是（GraphNode）   是（GraphNode）   OOOPS_FINAL.md 已定
                                                                                                   案编译通过；TRANSFE…
                                                                                                   图必须存 RangeOp
   pre_capture 形式       PreCapture 类        自由函数        隐含              自由函数          K 简洁，符合项目风格
   ShapeId operator==     = default            noexcept 手写   noexcept 手写     noexcept 手写     K/D 多数派 + C++17
                                                                                                   兼容
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  五、文件组织
  include/renaissance/graph/
  ├── arch_plan.h              # 已有
  ├── compile_spec.h           # 新增：编译参数（取代 Config）
  ├── shape_id.h               # 新增：去重键
  ├── layer_descriptor.h       # 新增：LayerDescriptor + TensorDesc + SubgraphPattern + InferContext
  ├── computation_graph.h      # 新增：ComputationGraph + GraphNode
  ├── memory_plan.h            # 修改：新增私有 alloc(..., slot_bytes) + build_range_op_ranges
  ├── graph_atlas.h            # 新增：三阶段桥梁
  ├── captured_graph.h         # 新增：后端抽象
  ├── pre_capture.h            # 新增：pre_capture() + PreCaptureResult
  ├── compiler.h               # 新增：Compiler + Result + Variant
  └── op_kind.h                # 修改：ComputeOp + RangeOp + GraphNode

  include/renaissance/tensor/
  └── distributed_tensor.h     # 修改：slot_bytes_ 解耦 + 双构造函数

  src/graph/
  ├── compiler.cpp             # 五阶段主流程
  ├── layer_descriptor_registry.cpp  # switch 注册表
  ├── graph_atlas.cpp          # GraphAtlas::build()
  ├── pre_capture.cpp          # Phase B 去重捕获
  └── lower_*.cpp              # 各算子 Descriptor 实现
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  六、已纳入的全部约束
   约束                       来源               体现
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   禁止 Config 类名           IMPORTANT.md       CompileSpec
   DTensor 形状唯一权威       用户第五轮         多 MemoryPlan，DTensor 各自 shape
   slot_bytes 与 shape 解耦   用户第七轮         DTensor::slot_bytes_ 构造时常数
   stride 按自身 shape 计算   用户第七轮         init_strides() 用 shape.h()/w()/c()
   ComputationGraph 零形状    用户第五轮         Node 只存 op + tensor_id
   15 有效图 + COUNT=16       用户第二轮         GraphId 枚举
   四铁律                     COMP_ULTIMATE.md   Phase 2 validate_tensor_consistency()
   frozen_first 运行时行为    COMP_ULTIMATE.md   Executor skip_first_bwd_
   ComputeOp/RangeOp 双枚举   OOOPS_FINAL.md     GraphNode::Kind + union
   65 Region / 条件分配       REGION_FINAL.md    MemoryPlan 条件分配
   ShapeId 显式去重           CAP_FINAL.md       {n,h,w,c} 四元组
   三阶段严格分离             CAP_FINAL.md       Phase A/B/C 时间边界
   预留并行捕获               IMPORTANT.md       CapturedGraph::capture() 占位
   Result 禁止拷贝/移动       COMP_ULTIMATE.md   删除拷贝/移动构造



# 【小伙伴D】

> 

---

## 一、总体管线

```
ArchPlan + {CompileSpec base, CompileSpec variants[5]}
        │
        ▼
Compiler::compile()          ← 五阶段（Phase A 编译期，不计时）
        │
        ├──► Result.variants[6]  ← Variant{name, MemoryPlan, ComputationGraph*}
        │
        ▼
GraphAtlas::build(Result, ShapeId[6])      ← 填 cg/mp/shape_id
        │
        ▼
pre_capture()               ← 去重 + CUDA Graph 捕获（Phase B 预演期，不计时）
        │
        ├──► PreCaptureResult{ CapturedGraph[], GraphAtlas }
        │
        ▼
Executor::setup(pre) → launch()   ← Phase C 运行期（MLPerf 计时）
```

**一句话**：ArchPlan 是"有什么"，Compiler 算出"放哪里、做什么"，GraphAtlas 去重捕获成"可以直接跑的图"。

---

## 二、核心类清单（17 个类型，4 层）

---

### 第1层：输入层（3 个）

#### `CompileSpec`（新增）— 编译参数集

取代 Config（IMPORTANT.md 禁令）。只含**形状相关**字段，MemoryPlan 布局字段另入 PlanConfig。

| 字段                    | 类型   | 说明                                           |
| ----------------------- | ------ | ---------------------------------------------- |
| `amp_enabled`           | `bool` | 混合精度（影响 DTensor dtype）                 |
| `max_sample_resolution` | `int`  | MemoryPlan 最大槽位预留                        |
| `actual_resolution`     | `int`  | 该变体的形状推导分辨率                         |
| `batch_size`            | `int`  | 批次大小                                       |
| `num_color_channels`    | `int`  | 颜色通道数                                     |
| `freeze_first_layer`    | `bool` | 运行时标志，不影响 MemoryPlan/ComputationGraph |

```cpp
static CompileSpec from_global_registry();
ShapeId to_shape_id() const noexcept;
```

---

#### `ShapeId`（新增）— 形状去重键

显式四元组（N-H-W-C），零碰撞，零 hash。用作 CapturedGraph 物理去重的形状分量。

```cpp
struct ShapeId {
    int32_t n = 0, h = 0, w = 0, c = 0;
    bool operator==(const ShapeId&) const noexcept;
};

struct ShapeIdHash {
    size_t operator()(const ShapeId& s) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(s.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(s.h) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(s.w) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(s.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

constexpr ShapeId kShapeInvariant{0, 0, 0, 0};  // 形状无关图专用
```

充分性：输入 shape 相同 → 所有中间 DTensor shape 相同 → 同一 Key → 自动复用。

---

#### `ArchPlan`（已有）— 模型架构计划

Compiler 唯一输入。汇集模型全部结构信息。

```cpp
class ArchPlan {
    const std::vector<LayerDescriptor>& layers() const;
    const std::vector<DTensor>&         weights() const;
    int first_layer_index() const;
};
```

---

### 第2层：描述层（5 个）

#### `TensorDesc`（新增）— 单张量描述符

`LayerDescriptor::infer_tensors` 的返回值元素。Compiler Phase 1 逐层收集。

```cpp
struct TensorDesc {
    std::string name;    // 张量语义名称（4 铁律验证用）
    Shape       shape;   // 该变体下的逻辑形状
    Region      region;  // 所属 65 个 Region 之一（硬编码赋值，无需 RegionResolver）
    DType       dtype;   // 数据类型（受 amp 影响）
};
```

---

#### `InferContext`（新增）— 推导上下文

传递当前推导的模式和编译参数。

```cpp
struct InferContext {
    GraphMode mode;           // TRAIN_FWD / TRAIN_BWD / INFERENCE
    bool      enable_amp;
    bool      is_first_layer;
    bool      bn_folded;
};
```

---

#### `SubgraphPattern`（新增）— 子图模式

`LayerDescriptor::BuildFn` 的返回值。纯算子拓扑（不含形状/Region/DType）。

```cpp
struct SubgraphPattern {
    struct Node {
        ComputeOp              op;
        std::vector<size_t>    input_indices;   // 索引 → TensorDesc 列表
        std::vector<size_t>    output_indices;
    };
    std::vector<Node> nodes;
};
```

---

#### `LayerDescriptor`（新增）— 单层知识契约

每个 `LayerKind` 对应一个不可变描述符。4 函数指针 + switch 注册表。

```cpp
struct LayerDescriptor {
    using InferFn = std::vector<TensorDesc> (*)(const Shape& input,
                                                  const OpParams& params,
                                                  const InferContext& ctx);
    using BuildFn = SubgraphPattern (*)(const OpParams& params,
                                         const std::vector<TensorDesc>& descs);

    InferFn infer_tensors;    // 返回三模式（train fwd/bwd + inf）张量并集
    BuildFn build_forward;
    BuildFn build_backward;
    BuildFn build_inference;
};

const LayerDescriptor& get_layer_descriptor(LayerKind kind);  // switch 注册表
```

四条铁律（Compiler Phase 1 断言保护）：

1. **并集覆盖**：返回 = 训练前向/反向/推理全部所需张量的并集
2. **数量相等**：所有 CompileSpec 下返回 TensorDesc 数量相同
3. **顺序相等**：相同语义的张量在所有 CompileSpec 下位于相同索引
4. **名称/区域相等**：对应位置的 `name` 和 `region` 跨配置不变

shape 和 dtype 可变；name、region、数量、顺序不变。违规 → 不同变体同一 tensor_id 指向不同语义张量 → 静默数据错乱。

---

#### `DTensor` (DistributedTensor)（已有，修改）— 分布式张量

**关键修改**：`slot_bytes_` 与 `shape` 解耦。每个变体的 MemoryPlan 拥有一组独立 DTensor 实例。

```cpp
struct DistributedTensor {
    int32_t  id = -1;
    Shape    shape;            // 该变体的实际逻辑形状（唯一权威，创建后不可变）
    DType    dtype;
    Region   region;           // 所属 65 个 Region 之一
    uint64_t offset = 0;       // 所有变体相同（max_slot_bytes 保证）
    int64_t  n_stride, h_stride, w_stride, c_stride;  // 按自身 shape 计算

    uint64_t slot_bytes() const noexcept;  // 返回构造时存储的常数
    static uint64_t compute_slot_bytes(const Shape&, DType, Region);  // 纯函数
private:
    uint64_t slot_bytes_ = 0;  // 构造时常量，跨变体取最大值
};
```

核心原则：

- stride **严格按自身 shape** 计算，不存在 "max shape" 概念
- slot_bytes 跨 6 变体取 max → offset 一致
- 双构造函数：标准构造（自动算 slot_bytes）+ 变体构造（显式传入 max_slot_bytes）

---

### 第3层：产出层（3 个）

#### `PlanConfig`（已有）— MemoryPlan 内部布局参数

MemoryPlan 条件分配的开关集合。注意：这也叫 "Config" 但它是 MemoryPlan 的内部 struct，非顶层类，不受 IMPORTANT.md 禁令约束。

```cpp
struct PlanConfig {
    bool amp_enabled, bn_folded, use_lars, use_adam, use_momentum,
         has_ema, need_mask, need_temp;
    int  num_models;
};
```

---

#### `MemoryPlan`（已有，扩展）— 内存分配计划

管理 65 个 Region 的线性布局（12 板块：B_W_E_A_G_M_V_N_I_F_S_T），预计算 17 个 RangeOp 内存范围。

```cpp
class MemoryPlan {
public:
    void configure(const PlanConfig&);     // 条件分配 65 个 Region
    void finalize();                       // 按 dt.slot_bytes() 布局

    DTensor alloc(const Shape&, DType, Region);                  // 公开接口（自动算 slot_bytes）
    DTensor alloc(const Shape&, DType, Region, uint64_t max_slot_bytes);  // 私有：Compiler 专用

    const DTensor& get(int32_t id) const;
    const RangeOpRange& get_range_op_range(RangeOp) const;       // Phase B 用
private:
    void build_range_op_ranges();
    friend class Compiler;
};
```

---

#### `GraphNode`（新增）— 统一图节点

COMPUTE/RANGE 二态 union。来自 OOOPS_FINAL.md 终局决策。

```cpp
struct GraphNode {
    enum class Kind : uint8_t { COMPUTE, RANGE };
    Kind kind;

    union {
        ComputeOp compute_op;   // DTensor 级操作（43 枚举值，OPS_NAME.md）
        RangeOp   range_op;     // Region 级操作（19 枚举值）
    };

    OpParams params;            // COMPUTE 态有效

    SmallVector<int32_t, 4>  input_ids, output_ids;       // COMPUTE：关联 DTensor 全局 ID
    SmallVector<MemRange, 4> input_ranges, output_ranges; // RANGE：预计算 (offset, size)
};
```

- `ComputeOp`：格式 `{BASE}[_{AMP}]_{FWD/BWD/INF}`，平台不进枚举
- `RangeOp`：格式 `RANGE_{ACTION}[_{SCOPE}]`，已替代 BatchOp

---

#### `ComputationGraph`（新增）— 纯拓扑容器

零形状信息。一份图供多个 shape-only 变体共享。

```cpp
class ComputationGraph {
public:
    enum class GraphId : uint8_t {
        TRANSFER_A, TRANSFER_B,     // H2D 异步（双缓冲）
        FIRST_FWD_A, FIRST_FWD_B,   // 首层前向
        DEEP_FWD_BWD,               // 深层前向+反向融合
        FIRST_BWD,                  // 首层反向
        FIRST_COMM, DEEP_COMM,      // 梯度通信（分桶）
        STATS_COMM,                 // BN 统计量通信
        OPTIMIZER, EMA_UPDATE,      // 参数更新
        INF_MAIN_A, INF_MAIN_B,     // 主模型推理
        INF_EMA_A, INF_EMA_B,       // EMA 推理
        COUNT                       // = 16（15 张有效图 + 哨兵）
    };

    void append(GraphId, GraphNode);
    const std::vector<GraphNode>& nodes(GraphId) const;
private:
    std::array<std::vector<GraphNode>, static_cast<size_t>(GraphId::COUNT)> graphs_;
};
```

15 张子图分两类：

- **shape 无关**（7 张）：TRANSFER×2, COMM×3, OPTIMIZER, EMA_UPDATE — 全部 6 变体共享
- **shape 相关**（8 张）：FIRST_FWD×2, DEEP_FWD_BWD, FIRST_BWD, INF_MAIN×2, INF_EMA×2 — 按 ShapeId 去重捕获

---

### 第4层：捕获层（3 个）

#### `CapturedGraph`（新增）— 后端无关可执行图

封装平台相关的"已捕获图"对象。

```cpp
class CapturedGraph {
public:
    static CapturedGraph capture(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid, Device* device);
    void launch(cudaStream_t) const;

    struct Key {
        const ComputationGraph* cg;
        GraphId                 gid;
        ShapeId                 shape;
        bool operator==(const Key&) const;
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = std::hash<const ComputationGraph*>{}(k.cg);
            h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.gid)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= ShapeIdHash{}(k.shape) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
private:
#ifdef TR_USE_CUDA
    cudaGraphExec_t exec_ = nullptr;
#else
    std::vector<CpuOp> cpu_ops_;  // CPU 后端：函数指针序列（IMPORTANT.md 要求兼容 legacy）
#endif
};
```

去重键 = `(ComputationGraph*, GraphId, ShapeId)`。ShapeId 相同 → 自动复用，不重复捕获。

---

#### `GraphAtlas`（新增）— 图集映射表

**6×16 二维表**，三阶段桥梁。

```cpp
class GraphAtlas {
public:
    static constexpr size_t kMaxVariants = 6;
    static constexpr size_t kMaxGraphIds = static_cast<size_t>(ComputationGraph::GraphId::COUNT);

    struct Slot {
        const ComputationGraph* cg = nullptr;     // Phase A：Compiler 填入（shape-only 变体共享指针）
        const MemoryPlan*       mp = nullptr;     // Phase A：Compiler 填入（变体特有或共享 base）
        ShapeId                 shape_id{};       // Phase A：Compiler 填入（kShapeInvariant 用于无关图）
        int32_t                 captured_idx = -1; // Phase B：pre_capture 填入
    };

    static GraphAtlas build(const Compiler::Result&,
                             const std::array<ShapeId, 6>& input_shapes);
    int32_t index(size_t variant, ComputationGraph::GraphId) const noexcept;  // O(1)
private:
    std::array<std::array<Slot, kMaxGraphIds>, kMaxVariants> table_;
};
```

Phase A（build）填入逻辑归属：

- shape 无关图：6 变体都指向 base MemoryPlan，shape_id = `kShapeInvariant` → Phase B 必然碰撞 → 全局复用
- shape 相关训练图：4 训练变体各自 MemoryPlan + 各自 ShapeId
- val 变体不填训练图，train 变体不填推理图
- 碰巧相同的 ShapeId（如 val_base 和 val_last 可能同 shape）自动复用

Phase B（pre_capture）：遍历填 `captured_idx`（内部完成）。
Phase C（运行时）：纯数组索引，零 hash，零分支。

---

#### `PreCaptureResult`（新增）— Phase B 输出

去重后的图集 + 已填 captured_idx 的 Atlas。

```cpp
struct PreCaptureResult {
    std::vector<CapturedGraph> graphs;  // 去重后图集（典型 27 个）
    GraphAtlas atlas;                    // captured_idx 已填入
};

PreCaptureResult pre_capture(const GraphAtlas& compile_atlas, Device* device);
```

算法：`unordered_map<CapturedGraph::Key, int32_t, KeyHash>`。已存在则复用，不存在则 `CapturedGraph::capture()`。

捕获数量：7（shape 无关）+ 8×4（训练 shape 相关）= 39 逻辑槽 → 去重后典型 27 张（最坏 31）。

---

### 编译器：`Compiler` — 五阶段编排器

#### `LayerContext`（新增）— Compiler 内部类型

单层分配后的上下文。非公开 API。

```cpp
struct LayerContext {
    std::vector<TensorDesc> descs;       // 该层全部 TensorDesc（三模式并集）
    std::vector<int32_t>    tensor_ids;  // 分配后的 DTensor 全局 ID
};
```

---

#### `Compiler`（新增）— 五阶段编译编排器

```cpp
class Compiler {
public:
    struct Variant {
        std::string              name;
        MemoryPlan               memory_plan;        // 独立 MemoryPlan
        const ComputationGraph*  train = nullptr;     // 指针：shape-only 变体共享 base
        const ComputationGraph*  inference = nullptr;
        ComputationGraph*        own_train = nullptr;     // graph-change 变体独立 new
        ComputationGraph*        own_inference = nullptr;
        ~Variant() { delete own_train; delete own_inference; }
    };

    struct Result {
        std::vector<Variant> variants;  // variants[0] = base，禁止拷贝/移动
    };

    static Result compile(const ArchPlan& arch,
                           const CompileSpec& base_spec,
                           const std::vector<CompileSpec>& variant_specs = {});
};
```

五阶段：

| 阶段 | 方法                      | 输入                       | 输出                           | 说明                                                         |
| :--: | ------------------------- | -------------------------- | ------------------------------ | ------------------------------------------------------------ |
|  1   | `derive_all_shapes`       | ArchPlan + 6 CompileSpec   | `all_shapes[6][layer][tensor]` | `infer_tensors`×6 + 验证 4 铁律                              |
|  2   | `compute_max_slot_bytes`  | `all_shapes`               | `max_slots[layer][tensor]`     | 逐位置取 max，**唯一跨变体比较**                             |
|  3   | `create_memory_plans`     | `all_shapes` + `max_slots` | `Variant::memory_plan[6]`      | 调用私有 `alloc(shape, dtype, region, max_slot_bytes)`       |
|  4   | `build_computation_graph` | LayerDescriptor 序列       | 共享 `ComputationGraph` ×2     | `build_forward` → `build_backward` → `build_inference` + 辅助图（COMM/OPT/EMA） |
|  5   | `share_or_clone`          | 6 变体 ShapeId             | 组装 `Result`                  | shape-only 共享指针，graph-change 独立 new                   |

6 变体输入：

| 索引 | 名称              | 用途 | resolution      | batch_size     |
| :--: | ----------------- | ---- | --------------- | -------------- |
|  0   | train_base        | 训练 | train_res_begin | standard       |
|  1   | train_last        | 训练 | train_res_begin | last batch     |
|  2   | train_lowres      | 训练 | train_res_end   | standard       |
|  3   | train_lowres_last | 训练 | train_res_end   | last batch     |
|  4   | val_base          | 验证 | val_res         | standard       |
|  5   | val_last          | 验证 | val_res         | val_last_batch |

---

## 三、完整类关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│  输入层：CompileSpec + ShapeId + ArchPlan（已有）                     │
│                                                                     │
│  CompileSpec ──► ShapeId    ArchPlan ──► {LayerDescriptor[], DTensor}│
└────────────────────────────┬────────────────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                  ▼
    TensorDesc         InferContext        SubgraphPattern
           │                 │                  │
           └─────────┬───────┘──────────────────┘
                     ▼
            LayerDescriptor
            (infer_tensors / build_forward / build_bwd / build_inf)
                     │
           ┌─────────▼─────────┐
           │     DTensor        │  shape / dtype / region / offset / stride
           │  (slot_bytes_ 私有) │
           └─────────┬─────────┘
                     │
      ┌──────────────┼──────────────────────────────────────┐
      │               ▼                Compiler              │
      │  ┌─────────────────────────────────────────────┐    │
      │  │ Phase 1: derive_all_shapes    → all_shapes   │    │
      │  │ Phase 2: compute_max_slot_bytes → max_slots  │    │
      │  │ Phase 3: create_memory_plans   → MemoryPlan[6]│   │
      │  │ Phase 4: build_computation_graph → CG ×2     │    │
      │  │ Phase 5: share_or_clone        → Result      │    │
      │  └─────────────────────────────────────────────┘    │
      │                                                     │
      │  产出: PlanConfig → MemoryPlan[6] × 65 Region       │
      │        GraphNode (COMPUTE/RANGE二态)                 │
      │        ComputationGraph (15 GraphId)                │
      └───────────────────────┬─────────────────────────────┘
                              │
                  ┌───────────▼───────────┐
                  │    GraphAtlas          │  Phase A: build()
                  │    table_[6][16]        │  Phase B: captured_idx ← pre_capture
                  └───────────┬───────────┘
                              │
                  CapturedGraph[]  (27~31个)
                  CUDA: cudaGraphExec_t / CPU: 函数指针序列
                              │
                  ┌───────────▼───────────┐  (本次范围外)
                  │    Executor             │  activate_variant(v)
                  │    index(v, gid) → idx  │  → captured_[idx].launch()
                  └───────────────────────┘
```

---

## 四、关键设计决策

### 命名

| 决策                              | 理由                                    |
| --------------------------------- | --------------------------------------- |
| `CompileSpec`                     | S+D 多数，避免 Config（IMPORTANT.md）   |
| `PlanConfig` 保留（内部 struct）  | MemoryPlan 内部，非顶层类，不受禁令约束 |
| `Variant` 纳入 `Compiler::Result` | S+K 多数，从 COMP_ULTIMATE.md           |
| `GraphAtlas` 独立于 Result        | K+D 多数，Phase A/B 解耦                |
| `ShapeId` 显式四元组              | CAP_FINAL.md 共识，零碰撞零 hash        |
| `GraphNode` 含 Kind+union         | K+D 多数，OOOPS_FINAL.md 终局           |
| `RangeOp` 替代 `BatchOp`          | 用户裁决                                |

### 核心原则

| 原则                     | 体现                                            |
| ------------------------ | ----------------------------------------------- |
| slot_bytes 与 shape 解耦 | DTensor::slot_bytes_ = 构造时常量，跨变体取 max |
| stride 按自身 shape      | `init_strides()` 用 `shape.h/w/c`，不用 max     |
| ComputationGraph 纯拓扑  | 零 shape，一份图供 n 个 shape-only 变体共享     |
| 4 铁律                   | Phase 1 `validate_tensor_consistency()`         |
| 3 阶段分离               | 编译(不计时) → 预演(不计时) → 运行(MLPerf 计时) |
| 去重 = ShapeId           | 同一 ShapeId → 自动碰撞 → 复用捕获              |
| 平台不进枚举             | 一张图纸，八卡共享，execute 层分发              |

---

**版本**：V2.0 | **日期**：2026-05-13 | **作者**：技术觉醒团队

