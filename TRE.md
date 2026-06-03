# 【今日话题：提出卷积的实现方案】

**卷积权重的规则是KRSC！**

今天我们研究怎么实现卷积的六大算子：CONV_FP32_FWD、CONV_FP32_BWD、CONV_FP32_INF、CONV_AMP_FWD、CONV_AMP_BWD、CONV_AMP_INF。
我们的深度学习框架，对卷积是有非常严格的参数限制的。
首先说明一点，你并不需要在算子内部执行这样那样的参数合法性检查，因为参数的合法性检查应该在创建BluePrint或ArchPlan的时候执行，这样就确保了传递给算子的一定是合法的参数。这样就避免了热路径上的检查开销。
卷积有什么约束呢？首先一点，本框架只支持4种形状的kernel：1×1、3×3、5×5、7×7。正方形。kernel_size只允许1、3、5、7这四个整数，输入其他都算违规。然后，stride只允许1×1和2×2。也就是stride参数只允许1、2这两者。不支持dilation参数（或者说固定dilation=1），也就是普通的卷积。不支持分组卷积或Depthwise卷积（准确地说，是DW卷积会在未来用不同的算子来实现，但当前的卷积确定是普通卷积，不支持也不兼容Depthwise卷积），也就是不支持group参数。最后但也是最最重要的：卷积不支持bias。我们压根就不给卷积的bias进行分区，所以本框架内的所有卷积必定都是没有bias的。边缘padding倒是没有限制。
再次强调，这些参数限制的检查都是在BluePrint或ArchPlan的层级，算子内不应该做过多的检查，默认视为参数合格。
再然后就是我们的卷积的输入和输出。卷积的正向（FWD）应该有2个输入、2个输出——权重、特征图输入、特征图输出、bn_stats。
我们的算子有一个决定性的关键：dX覆盖X。也就是反向传播时的梯度输出会覆盖原始输入，它们用的是同一个DTensor。一定一定要确保这一点。
然后，关键来了——既然dX覆盖X，反向传播又要用到X，那如何确保反向传播过程中dX不污染X呢？答案是：永远先计算dW（wgrad），完成后再计算dX（dgrad）。具体可以参考FC的BWD实现。dW要用StreamKind::COMP_1来计算，dX要用StreamKind::COMP_3来计算（看清楚，是COMP_3而不是COMP_2！）。dX 启动前显式 cudaStreamWaitEvent(s_dx, dW_event)，即 dX 等待 dW 完成。至于为什么用COMP_3而不用COMP_2，一是为了跟FC算子对齐，二是因为Conv通常连着BN，我们要把COMP_2留给BN。至于卷积的FWD和INF用什么流，那当然是COMP_1。
AMP版的卷积算子，其反向传播应该是生成FP16权重梯度，而不是像FC那样直接生成FP32权重梯度。所以，AMP的反向的权重梯度是放到FP16权重梯度区的。我们后面有专门的CAST算子来转换，卷积算子不需要管。
这里重点讲一下Conv+GenStats。这是一个只支持FP16的算子。你在FP32是肯定不能用它的。但是我们的框架在张量申请时有一个“对齐原则”，那就是不管是CPU还是GPU，不管是FP32还是AMP，同一层所申请的输入输出张量的数量和种类和形状必须是一致的（虽然AMP的情形下会把特征图放到F_FEATURE_FP16区）。这意味着，即便GPU和CPU的FP32都不支持GenStats，依然要像FP16一样留出一个同形状的bn_stats张量。这个张量比较特别，因为它不使用padding，我建议放到T_TEMP_FP32区。
AMP卷积的实现当然是参考根目录下的cbr_fwd_fp16.cpp、cbr_bwd_fp16.cpp、cbr_inf_fp16.cpp了。需要注意的是，这些测试样例只是告诉你用什么算子、API怎么用，但它们的张量对齐、张量配置未必跟我们框架一样，不能完全照搬。我们的框架的张量都是NHWC的。AMP模式下可能需要用DTensor的padded_c而不是c，具体哪里用哪个，需要检查和研究，或参考FC、Flatten、MaxPool这些已经验证过的算子的AMP实现，认真思考。
FP32的GPU实现，你是没法照抄AMP了，因为FP32不支持Conv+GenStats。就用一般的cuDNN Frontend的卷积实现。你的算子不需要对bn_stats输出任何数据，但你依然要有这个张量，并且把它传给算子，以保持对齐。
最后就是“穷举式搜索”EXHAUSTIVE_C。具体来说，我们已经搜完了，只不过可以选择应用穷举式搜索的引擎罢了。选择这个模式就会从我们的include/generated目录下找出此前根据测试得到的最快引擎。这个只是针对特定平台、特定形状的优化，而且只支持AMP、只支持卷积。我们只有A100和RTX5090支持，其他平台碰到EXHAUSTIVE_C都是改用Heuristic Mode B。
EXHAUSTIVE_C枚举类型应该被定义在include/core/types.h。
GlobalRegistry类应该新增一个fixed_conv_search_mode_，然后提供setter和getter方法。setter方法是GlobalRegistry& conv_search_mode(类型枚举名)这样。
这里我不厌其烦地解释一下为什么我们的实现是NHWC，但是cbr_fwd_fp16.cpp里的cuDNN Frontend看上去像是NCHW。事实上cuDNN的物理布局支持NHWC，只不过它的set_stride的API顺序是N、C、H、W罢了。set_stride里面stride[1]=1就表示在C通道上连续，也就是NHWC。所以，cbr_*_fp16.cpp本质上就已经是NHWC的实现了。这些测试样例里面似乎不一定捕获为CUDA Graph，但我们的框架是必定捕获的。
你现在只需要管卷积这6个算子，完全不需要管涉及卷积的融合算子（比如CBR等）。
卷积的CPU实现，FWD和INF用XNNPACK实现（本项目已有支持），BWD先用朴素循环，后续再研究反向怎么优化。XNNPACK的实现可参考根目录下的conv_op_legacy.cpp。
卷积的权重是需要在init_all初始化的，具体参考FC权重的初始化。
再次提醒，bn_stats是所有类型的卷积算子都必备的张量，尽管只有AMP的FWD算子会真正往里面写东西（对于不需要写的算子，也不用对这个张量进行memset，不管就行）。
关于是否需要跨流同步，参考FC算子。
暂时不需要写测试样例。

**卷积权重的规则是KRSC！**（这个不能改变）

**还有一点，那就是关于你可能需要用到缓存的情况。首先，workspace我们是有的，每个stream有自己专属的workspace，由DeviceContext管理。我们禁止在算子内部申请临时内存，所有内存都必须实现申请好。但是workspace是cuDNN这个黑盒来调用的，所以你不能显式地往里面写东西，但是在cuDNN需要指定workspace的时候，就指定它。如果你确实需要用到临时张量，应该在Compiler阶段申请T_TEMP_FP32/FP16区的DTensor，并且计入张量的输入输出之一（这会比较麻烦）。CPU如果需要用到临时内存，可以手动操作workspace，类似于FC的转置。总之，千万记住不能在算子内申请内存。**



# 【小伙伴S】

# 

> 基于 ATN1/ATN2 算子添加流程，结合现有 CONV_FP32 实现与 CBR 参考代码，制定完整的六大卷积算子实现方案。

## 一、现状分析

### 1.1 现有实现

- **已实现**：`CONV_FP32_FWD`, `CONV_FP32_BWD`（仅 FP32 精度）
- **文件结构**：
  - `src/backend/ops/dtensor/conv_op.cpp` - CPU + CUDA launcher
  - `src/backend/ops/dtensor/conv_op_impl.cpp` - cuDNN FE 图构建
- **CPU实现**：XNNPACK（FWD）+ 朴素循环（BWD）
- **CUDA实现**：cuDNN Frontend API

### 1.2 缺失功能

根据 `op_kind.h` 枚举，需补充：

- `CONV_FP32_INF` - FP32 推理
- `CONV_AMP_FWD` - AMP 混合精度训练前向
- `CONV_AMP_BWD` - AMP 混合精度训练反向
- `CONV_AMP_INF` - AMP 混合精度推理

### 1.3 关键约束（来自 CVN4.md 要求）

- **布局**：NHWC 物理布局，NCHW 逻辑声明
- **dX-overwrites-X**：BWD 的 dX 必须原地覆盖 X
- **双流架构**：AMP BWD 的 dW 在 COMP_1，dX 在 COMP_3
- **AMP梯度**：AMP 版本输出 FP16 权重梯度（与 FC 不同）
- **EXHAUSTIVE_C**：A100/RTX5090 平台穷举搜索
- **GenStats**：仅在 AMP FWD 中，不用于 FP32
- **参数约束**：
  - `kernel_h == kernel_w` (正方形卷积核)
  - `stride_h == stride_w` (正方形步长)
  - `pad_h == pad_w` (对称填充)
  - `dilation_h == dilation_w == 1` (不支持空洞卷积)
  - `groups == 1` (仅标准卷积，不支持深度卷积/分组卷积)

## 二、实现路线图

### 阶段一：类型系统扩展（op_kind.h/op_kind.cpp）

#### 2.1 ComputeOp 枚举

```cpp
enum class ComputeOp : uint16_t {
    // 现有
    CONV_FP32_FWD,
    CONV_FP32_BWD,

    // 新增
    CONV_FP32_INF,           // FP32 推理

    CONV_AMP_FWD,            // AMP 训练前向
    CONV_AMP_BWD,            // AMP 训练反向
    CONV_AMP_INF,            // AMP 推理

    // ...
};
```

#### 2.2 字符串化（op_kind.cpp）

```cpp
std::string compute_op_to_string(ComputeOp op) {
    switch (op) {
        case ComputeOp::CONV_FP32_FWD: return "CONV_FP32_FWD";
        case ComputeOp::CONV_FP32_BWD: return "CONV_FP32_BWD";
        case ComputeOp::CONV_FP32_INF: return "CONV_FP32_INF";
        case ComputeOp::CONV_AMP_FWD:  return "CONV_AMP_FWD";
        case ComputeOp::CONV_AMP_BWD:  return "CONV_AMP_BWD";
        case ComputeOp::CONV_AMP_INF:  return "CONV_AMP_INF";
        // ...
    }
}

std::string format_params(ComputeOp op, const OpParams& p) {
    if (auto* params = std::get_if<ConvParams>(&p.data)) {
        return format_string("conv(out_channels=%d, kernel=%dx%d, stride=%dx%d, "
                             "pad=%dx%d, dilation=%dx%d, groups=%d)",
                             params->out_channels, params->kernel_h, params->kernel_w,
                             params->stride_h, params->stride_w,
                             params->pad_h, params->pad_w,
                             params->dilation_h, params->dilation_w,
                             params->groups);
    }
    return "";
}
```

### 阶段二：后端实现（conv_op.cpp）

#### 2.3 CPU 实现

**策略**：

- `CONV_FP32_INF`: 复用 `launch_conv_fwd_cpu_xnnpack()`
- `CONV_AMP_*`: 指向 `launch_not_supported_cpu`

```cpp
static void launch_conv_inf_cpu(CpuOpContext* op_ctx) {
    // 推理与前向计算相同，复用 XNNPACK 实现
    launch_conv_fwd_cpu_xnnpack(op_ctx);
}

#ifdef TR_USE_CUDA
static void launch_not_supported_amp_cpu(CpuOpContext* op_ctx) {
    TR_THROW_NOT_SUPPORTED_GPU("CONV_AMP operators on CPU");
}
#endif
```

#### 2.4 CUDA 实现 - FP32 INF

```cpp
static void launch_conv_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // 推理与前向计算相同，复用 conv_fwd 图构建逻辑
    launch_conv_fwd_cuda(node, mp, ctx, state);
}
```

#### 2.5 CUDA 实现 - AMP FWD

**关键特性**：

- 使用 FP16 张量（`DataType_t::HALF`）
- 集成 GenStats（sum + sq_sum）用于 BN 统计
- COMP_1 流执行
- 支持图缓存 + EXHAUSTIVE_C 搜索

**图结构**（参考 CBR FWD）：

```
X (FP16) + W (FP16) → [Conv] → Y (FP16)
                     ↓
                  [GenStats] → sum (FP32), sq_sum (FP32)
```

**实现框架**：

```cpp
struct ConvAmpFwdCacheKey {
    uint64_t handle_bits;
    int n, c, h, w, k, r, s, stride, pad;
    bool operator==(const ConvAmpFwdCacheKey&) const;
    // hash function
};

struct ConvAmpFwdCache {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph;
    int64_t workspace_size;
};

static std::unordered_map<ConvAmpFwdCacheKey, ConvAmpFwdCache> s_conv_amp_fwd_caches;

static void launch_conv_amp_fwd_cuda(...) {
    // 1. 检查图缓存
    // 2. 构建图（首次）
    //    - set_io_data_type(HALF)
    //    - Conv + GenStats
    //    - EXHAUSTIVE_C 搜索模式
    // 3. 执行图
    // 4. COMP_1 流注册
}
```

**图构建示例**（参考 `cbr_fwd_fp16.cpp`）：

```cpp
auto graph = std::make_shared<fe::Graph>();
graph->set_io_data_type(fe::DataType_t::HALF)
      ->set_compute_data_type(fe::DataType_t::FLOAT);

auto X = graph->tensor(attr_x);
auto W = graph->tensor(attr_w);

// Conv
fe::Conv_fprop_attributes conv_attr;
conv_attr.set_padding({pad, pad})
         ->set_stride({stride, stride});
auto Y = graph->conv_fprop(X, W, conv_attr);

// GenStats（用于 BN）
auto sum = graph->gen_stats(Y, fe::GenStats_attributes_t::REDUCTION_SUM);
auto sq_sum = graph->gen_stats(Y, fe::GenStats_attributes_t::REDUCTION_SQ_SUM);

// 设置输出
Y->set_output(true)->set_uid(102);
sum->set_output(true)->set_uid(103);
sq_sum->set_output(true)->set_uid(104);

// 执行计划构建
graph->build_operation_graph(handle);
auto heuristics = is_a100_or_rtx5090 ?
    fe::HeurMode_t::EXHAUSTIVE_C : fe::HeurMode_t::B;
graph->create_execution_plans({heuristics, fe::HeurMode_t::FALLBACK});
graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
```

#### 2.6 CUDA 实现 - AMP BWD

**关键特性**：

- **双流架构**：
  - dW → COMP_1 流（先完成）
  - dX → COMP_3 流（后完成，下游依赖）
- **显式同步**：COMP_3 等待 COMP_1 完成
- **FP16 dW 输出**（与 FC 的 FP32 不同）
- **双输出图**：单次 cuDNN FE 调用同时生成 dX + dW

**图结构**：

```
dY (FP16) + X (FP16) + W (FP16) → [Conv BWD] → dX (FP16), dW (FP16)
```

**流拓扑**：

```
COMP_1: [WGrad] → event_wgrad_done
COMP_3: wait(event_wgrad_done) → [DGrad]
```

**实现框架**：

```cpp
static void launch_conv_amp_bwd_cuda(...) {
    // 1. 构建双输出图
    auto graph = std::make_shared<fe::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF);

    auto dY = graph->tensor(attr_dy);
    auto X = graph->tensor(attr_x);
    auto W = graph->tensor(attr_w);

    // DGrad
    fe::Conv_dgrad_attributes dgrad_attr;
    dgrad_attr.set_padding({pad, pad})->set_stride({stride, stride});
    auto dX = graph->conv_dgrad(dY, W, dgrad_attr);

    // WGrad
    fe::Conv_wgrad_attributes wgrad_attr;
    wgrad_attr.set_padding({pad, pad})->set_stride({stride, stride});
    auto dW = graph->conv_wgrad(dY, X, wgrad_attr);

    dX->set_output(true)->set_uid(203);
    dW->set_output(true)->set_uid(204);

    graph->build_operation_graph(handle);
    graph->create_execution_plans({fe::HeurMode_t::B});
    graph->build_plans();

    // 2. 双流执行
    cudaStream_t stream_comp_1 = ctx.stream(StreamKind::COMP_1);
    cudaStream_t stream_comp_3 = ctx.stream(StreamKind::COMP_3);
    cudaEvent_t event_wgrad_done = ...;

    // WGrad on COMP_1
    cudnnSetStream(handle, stream_comp_1);
    graph->execute(handle, variant_pack, workspace);

    cudaEventRecord(event_wgrad_done, stream_comp_1);

    // DGrad on COMP_3
    cudaStreamWaitEvent(stream_comp_3, event_wgrad_done);
    cudnnSetStream(handle, stream_comp_3);
    graph->execute(handle, variant_pack, workspace);

    // 3. 注册流状态（COMP_3 为主输出流）
    int si = state.get_or_register(stream_comp_3);
    state.output_stream_idx = si;
}
```

#### 2.7 CUDA 实现 - AMP INF

```cpp
static void launch_conv_amp_inf_cuda(...) {
    // 推理与前向相同（无 GenStats）
    // 复用 AMP FWD 图，但移除 GenStats 节点
    // 或者使用独立图缓存
}
```

### 阶段三：算子注册（conv_op.cpp）

```cpp
void register_op_conv() {
    auto& table = g_compute_op_table;

    // FP32 FWD（已有）
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CONV_FP32_FWD)];
        e.op = ComputeOp::CONV_FP32_FWD;
        e.launch_cpu = launch_conv_fwd_cpu;
        e.launch_cuda = launch_conv_fwd_cuda;
    }

    // FP32 BWD（已有）
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CONV_FP32_BWD)];
        e.op = ComputeOp::CONV_FP32_BWD;
        e.launch_cpu = launch_conv_bwd_cpu;
        e.launch_cuda = launch_conv_bwd_cuda;
    }

    // FP32 INF（新增）
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CONV_FP32_INF)];
        e.op = ComputeOp::CONV_FP32_INF;
        e.launch_cpu = launch_conv_inf_cpu;
        e.launch_cuda = launch_conv_inf_cuda;
    }

#ifdef TR_USE_CUDA
    // AMP FWD（新增）
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CONV_AMP_FWD)];
        e.op = ComputeOp::CONV_AMP_FWD;
        e.launch_cpu = launch_not_supported_amp_cpu;
        e.launch_cuda = launch_conv_amp_fwd_cuda;
    }

    // AMP BWD（新增）
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CONV_AMP_BWD)];
        e.op = ComputeOp::CONV_AMP_BWD;
        e.launch_cpu = launch_not_supported_amp_cpu;
        e.launch_cuda = launch_conv_amp_bwd_cuda;
    }

    // AMP INF（新增）
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CONV_AMP_INF)];
        e.op = ComputeOp::CONV_AMP_INF;
        e.launch_cpu = launch_not_supported_amp_cpu;
        e.launch_cuda = launch_conv_amp_inf_cuda;
    }
#endif

    TR_LOG_DEBUG("backend") << "CONV operators registered (FP32+AMP, CPU+CUDA)";
}
```

### 阶段四：全局注册

#### 4.1 op_registry.h

```cpp
void register_op_conv();  // 已有声明
```

#### 4.2 op_registry.cpp

```cpp
// 在 require_warmup() 中添加
bool require_warmup(ComputeOp op) noexcept {
    switch (op) {
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_FP32_INF:
        case ComputeOp::CONV_AMP_FWD:
        case ComputeOp::CONV_AMP_INF:
            return true;  // cuDNN FE 需要 warmup
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_AMP_BWD:
            return false;  // cuDNN Legacy API 首次自初始化
        default:
            return false;
    }
}
```

#### 4.3 op_stream_policy.cpp

```cpp
StreamKind get_op_default_stream(ComputeOp op) noexcept {
    switch (op) {
        // CONV FWD → COMP_1（已有）
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_AMP_FWD:
            return StreamKind::COMP_1;

        // CONV BWD → COMP_3（已有，代表 dX 输出流）
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_AMP_BWD:
            return StreamKind::COMP_3;

        // CONV INF → COMP_1
        case ComputeOp::CONV_FP32_INF:
        case ComputeOp::CONV_AMP_INF:
            return StreamKind::COMP_1;

        default:
            return StreamKind::COMP_1;
    }
}
```

#### 4.4 CMakeLists.txt

```cmake
# src/CMakeLists.txt（已有，无需修改）
list(APPEND RENAISSANCE_SOURCES
    backend/ops/dtensor/conv_op.cpp
    backend/ops/dtensor/conv_op_impl.cpp
)
```

### 阶段五：图缓存与 EXHAUSTIVE_C 集成

#### 5.1 平台检测

```cpp
// 在 cuDNN_utils.h 或新文件 platform_utils.h 中
static bool is_exhaustive_c_supported() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // A100 (sm_80) 或 RTX 5090 (sm_100)
    return prop.major == 8 && prop.minor == 0 ||
           prop.major == 10;
}
```

#### 5.2 图缓存策略

```cpp
// 每个算子类型独立缓存
// ConvAmpFwdCache, ConvAmpBwdCache, ConvAmpInfCache
// 缓存键包含：handle_bits, shape, params
```

#### 5.3 执行计划构建

```cpp
auto heuristics = is_exhaustive_c_supported() ?
    fe::HeurMode_t::EXHAUSTIVE_C : fe::HeurMode_t::B;
graph->create_execution_plans({heuristics, fe::HeurMode_t::FALLBACK});
```

### 阶段六：参数验证

在所有 CUDA launcher 开头添加：

```cpp
const auto* p = std::get_if<ConvParams>(&node.params.data);
TR_CHECK(p != nullptr, ValueError, "CONV missing ConvParams");

// 参数约束检查
TR_CHECK(p->kernel_h == p->kernel_w, ValueError,
         "CONV requires square kernel (kernel_h == kernel_w)");
TR_CHECK(p->stride_h == p->stride_w, ValueError,
         "CONV requires square stride (stride_h == stride_w)");
TR_CHECK(p->pad_h == p->pad_w, ValueError,
         "CONV requires symmetric padding (pad_h == pad_w)");
TR_CHECK(p->dilation_h == 1 && p->dilation_w == 1, ValueError,
         "CONV does not support dilation (dilation must be 1)");
TR_CHECK(p->groups == 1, ValueError,
         "CONV does not support groups (groups must be 1)");
```

### 阶段七：ArchPlan 集成（可选，如需 DeepLearningTask 支持）

#### 7.1 arch_plan.h

```cpp
enum class LayerKind : uint16_t {
    Conv2d,  // 新增
    // ...
};

struct Conv2dLayerParams {
    int out_channels;
    int kernel_size;
    int stride = 1;
    int padding = 0;
    bool operator==(const Conv2dLayerParams&) const;
};
```

#### 7.2 layer_descriptor_registry.cpp

```cpp
static TensorIds infer_conv2d_tensors(const ArchLayer& layer) {
    auto* p = std::get_if<Conv2dLayerParams>(&layer.params);
    // 计算输出形状 OH, OW
    // 返回 {conv_output}
}

static SubgraphPattern build_conv2d_forward(...) {
    SubgraphPattern::Node n;
    n.op = using_amp() ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
    n.output_indices = {0};
    return {n};
}

static SubgraphPattern build_conv2d_backward(...) {
    SubgraphPattern::Node n;
    n.op = using_amp() ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
    n.input_indices = {1, 2};  // X, W（dY 由 Compiler prepend）
    n.output_indices = {};     // dX in-place
    return {n};
}

static SubgraphPattern build_conv2d_inference(...) {
    SubgraphPattern::Node n;
    n.op = using_amp() ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
    n.output_indices = {0};
    return {n};
}

static const LayerDescriptor conv2d_desc = {
    infer_conv2d_tensors, build_conv2d_forward,
    build_conv2d_backward, build_conv2d_inference
};

// 在 get_layer_descriptor() 中
case LayerKind::Conv2d: return conv2d_desc;
```

#### 7.3 compiler.cpp

```cpp
// 在 compile_primitive() 中添加
case LayerKind::Conv2d:
    return true;

// 在 build_computation_graph() FWD 中添加
case LayerKind::Conv2d:
    idx = 0;
    break;

// 在 build_computation_graph() BWD grad slot 中添加
case LayerKind::Conv2d:
    grad_slot_idx = -1;  // in-place
    break;

// 在 build_computation_graph() BWD 节点生成后处理中添加
if (gn.compute_op == ComputeOp::CONV_FP32_BWD ||
    gn.compute_op == ComputeOp::CONV_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.output_ids = {it->second};  // dX in-place
    }
}

// 在 grad_id 追踪中添加
case LayerKind::Conv2d:
    grad_id = layer_output_ids.at(l)[0];
    break;
```

#### 7.4 ArchPlan 辅助文件

- `arch_plan_expand.cpp`: `NodeKind::Conv2d → ArchLayer{LayerKind::Conv2d, ...}`
- `arch_plan_shape.cpp`: `recompute_shapes_from()` 计算 OH, OW
- `arch_plan_format.cpp`: `kind_name()`, `params_str()`
- `arch_plan_normalize.cpp`: `get_effective_output_c_at()`
- `arch_plan_yaml.cpp`: `kind_from_name()`, `to_yaml()`, `from_yaml()`

### 阶段八：测试

#### 8.1 正确性测试（tests/op/test_conv.cpp）

```cpp
// SimpleTask 验证 FWD/BWD/INF
// 支持 --cpu/--gpu/--amp
// PyTorch 参考数据对比

TEST(ConvOpTest, FP32_Fwd_GPU) {
    SimpleTask task(/*device=*/"gpu");
    // 构建 CONV_FP32_FWD 图
    // 运行并与 PyTorch 对比
}

TEST(ConvOpTest, AMP_Fwd_GPU) {
    SimpleTask task(/*device=*/"gpu", /*amp=*/true);
    // 构建 CONV_AMP_FWD 图（含 GenStats）
    // 验证 Y, sum, sq_sum 输出
}

TEST(ConvOpTest, AMP_Bwd_DualStream_GPU) {
    SimpleTask task(/*device=*/"gpu", /*amp=*/true);
    // 构建 CONV_AMP_BWD 图
    // 验证 dX in-place, dW FP16 输出
    // 验证双流同步
}
```

#### 8.2 性能测试（tests/perf/test_conv_perf.cpp）

```cpp
// 固定输入形状，预热 5 次，计时 100 次
// FWD/BWD/INF 分别计时
// 支持 --cpu/--gpu/--amp
// 对比 FP32 vs AMP 性能
```

#### 8.3 CMake 注册

```cmake
# tests/op/CMakeLists.txt
add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE renaissance)
target_compile_definitions(test_conv PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_conv)
endif()
set_target_properties(test_conv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
)
```

## 三、关键技术难点与解决方案

### 3.1 AMP BWD 双流同步

**难点**：dW 和 dX 在不同流上执行，需确保 COMP_3 的 DGrad 等待 COMP_1 的 WGrad 完成。

**解决方案**：

```cpp
cudaEvent_t event_wgrad_done;
cudaEventCreate(&event_wgrad_done);

// COMP_1: WGrad
cudaStream_t s1 = ctx.stream(StreamKind::COMP_1);
// ... 执行 WGrad ...
cudaEventRecord(event_wgrad_done, s1);

// COMP_3: DGrad
cudaStream_t s3 = ctx.stream(StreamKind::COMP_3);
cudaStreamWaitEvent(s3, event_wgrad_done);
// ... 执行 DGrad ...
```

### 3.2 cuDNN FE AMP 布局对齐

**难点**：AMP FP16 张量的 C 通道被 padding 到 8 的倍数，需使用 `padded_c()` 而非逻辑 `c()`。

**解决方案**：

```cpp
// 错误
attr_x->set_dim({N, dt_x.c(), H, W});

// 正确
int64_t PC = dt_x.padded_c();
attr_x->set_dim({N, PC, H, W})
       ->set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                     dt_x.h_stride_cuda(), dt_x.w_stride_cuda()});
```

### 3.3 EXHAUSTIVE_C 搜索模式

**难点**：仅在 A100/RTX5090 上启用，其他平台 fallback 到 HeurMode_t::B。

**解决方案**：

```cpp
auto heur_mode = is_exhaustive_c_supported() ?
    fe::HeurMode_t::EXHAUSTIVE_C : fe::HeurMode_t::B;
graph->create_execution_plans({heur_mode, fe::HeurMode_t::FALLBACK});
```

### 3.4 GenStats 集成

**难点**：AMP FWD 需输出 sum + sq_sum 用于 BN 统计，但 FP32 不需要。

**解决方案**：

- AMP FWD 图：Conv + GenStats（3 输出：Y, sum, sq_sum）
- AMP INF 图：仅 Conv（1 输出：Y）
- FP32 FWD/INF 图：仅 Conv（1 输出：Y）

### 3.5 图缓存键设计

**难点**：缓存键需包含所有影响图结构的参数，避免缓存失效。

**解决方案**：

```cpp
struct ConvAmpFwdCacheKey {
    uint64_t handle_bits;  // cuDNN handle 指针（区分不同 RANK）
    int n, c, h, w;        // 输入形状
    int k, r, s;           // 卷积核参数
    int stride, pad;       // 步长和填充
    bool operator==(const ConvAmpFwdCacheKey&) const = default;
};
```

## 四、实现优先级

### Phase 1: 核心功能（1-2 天）

1. `CONV_FP32_INF` - 复用现有 FWD 代码
2. `CONV_AMP_FWD` - 基于 CBR FWD 参考实现
3. 算子注册 + 流策略

### Phase 2: 反向传播（2-3 天）

4. `CONV_AMP_BWD` - 双流架构，难点
5. `CONV_AMP_INF` - 复用 AMP FWD（无 GenStats）

### Phase 3: 优化与测试（1-2 天）

6. 图缓存 + EXHAUSTIVE_C 集成
7. 参数验证
8. 正确性测试
9. 性能测试

### Phase 4: ArchPlan 集成（可选，1-2 天）

10. LayerKind + LayerParams
11. LayerDescriptor 注册
12. Compiler 集成

## 五、风险与缓解措施

| 风险                | 影响                  | 缓解措施                                   |
| ------------------- | --------------------- | ------------------------------------------ |
| 双流同步错误        | CUDA Graph 捕获失败   | 严格遵循 fork-join 拓扑，event 预分配      |
| AMP 布局对齐问题    | 读写错位，结果错误    | 统一使用 `padded_c()` + stride             |
| EXHAUSTIVE_C 不兼容 | A100/RTX5090 上性能差 | 添加平台检测，fallback 到 B 模式           |
| 图缓存失效          | 性能下降              | 缓存键包含所有关键参数                     |
| GenStats 集成错误   | BN 统计错误           | 参考 CBR FWD 实现，独立测试                |
| dX in-place 错误    | 内存覆盖问题          | BWD 节点 `output_ids` 指向 X 的 DTensor ID |

## 六、总结

本方案基于 ATN1/ATN2 标准流程，结合现有 CONV_FP32 实现与 CBR 参考代码，制定了完整的六大卷积算子实现路径。核心难点在于 AMP BWD 的双流同步与 cuDNN FE 的 AMP 布局对齐，通过参考 CBR 实现和严格遵循框架约定可降低风险。建议分阶段实现，先跑通 SimpleTask 验证，再接入 ArchPlan。

---

> **文档版本**: GRB4-v1.0
> **创建日期**: 2026-06-03
> **参考文档**: ATN1.md, ATN2.md, CVN4.md, cbr_fwd_fp16.cpp, cbr_bwd_fp16.cpp
> **预计工作量**: 5-8 天（不含 ArchPlan 集成）



# 【小伙伴K】



> 版本: 1.0.0
> 日期: 2026-06-02
> 范围: CONV_FP32_FWD / CONV_FP32_BWD / CONV_FP32_INF / CONV_AMP_FWD / CONV_AMP_BWD / CONV_AMP_INF
> 目标平台: Windows MSVC + CUDA 13.1 + cuDNN 9.17

---

## 1. 现状诊断

| 算子          | 现有状态 | 说明                                                         |
| ------------- | -------- | ------------------------------------------------------------ |
| CONV_FP32_FWD | 已注册   | launch_conv_fwd_cuda -> build_conv_fwd_graph（conv_op_impl.cpp），但输出 Y 的 stride 手写 align_up(K,8) 而非 DTensor 真实 stride |
| CONV_FP32_BWD | 已注册   | launch_conv_bwd_cuda -> build_conv_bwd_graph 单图单流（COMP_1）。需拆分为 wgrad/dgrad 双图并跨流同步 |
| CONV_FP32_INF | 缺失     | ComputeOp 枚举无此定义，get_op_default_stream/require_warmup 无此条目 |
| CONV_AMP_FWD  | 半残     | conv_op.cpp 中有旧代码（硬编码 FLOAT，未注册，无 launch）    |
| CONV_AMP_BWD  | 半残     | 同上                                                         |
| CONV_AMP_INF  | 缺失     | ComputeOp 枚举无此定义                                       |

CPU 侧：CONV_FP32_FWD 已有 XNNPACK/Naive；CONV_FP32_BWD 已有 Naive；AMP 算子 CPU 不支持（同 FC）。

---

## 2. 核心设计原则

### 2.1 参数约束（BluePrint/ArchPlan 层已保证，算子内不检查）

| 参数                | 约束                             |
| ------------------- | -------------------------------- |
| kernel_h / kernel_w | 仅 1, 3, 5, 7（正方形）          |
| stride_h / stride_w | 仅 1 或 2                        |
| dilation            | 固定 1                           |
| groups              | 固定 1（普通卷积，无 Depthwise） |
| bias                | 不支持                           |
| padding             | 无限制                           |

### 2.2 NHWC 物理布局与 cuDNN FE stride

框架张量物理布局为 NHWC。cuDNN FE 的 set_dim 顺序为 {N,C,H,W}（逻辑声明），set_stride 顺序也为 {N,C,H,W}。**stride[1]=1 表示 C 维度连续，即 NHWC。**

**铁律：所有 cuDNN FE Tensor 的 stride 必须使用 DTensor 真实 stride，禁止手写 align_up。**

```cpp
auto make_nhwc_stride = [](const DTensor& dt) -> std::vector<int64_t> {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
};

// 特征图 X/Y/dY/dX: dim={N,C,H,W}, stride=make_nhwc_stride(dt)
// 权重 W/dW:        dim={K,C,R,S}, stride=make_nhwc_stride(dt)
//                    (dt.shape 为 {K,R,S,C}，n_stride=R*S*C, c_stride=1, h_stride=S*C, w_stride=C)
```

### 2.3 dX 覆盖 X（In-Place BWD）

反向传播时 dX 复用 X 的同一 DTensor buffer。**必须先算 dW（wgrad），再算 dX（dgrad），避免 X 在 wgrad 消费前被覆盖。**

| 阶段              | StreamKind | 同步原语                                                     |
| ----------------- | ---------- | ------------------------------------------------------------ |
| wgrad             | COMP_1     | 完成后 cudaEventRecord(state.streams[i_dw].last_done_event, s_dw) |
| dgrad             | COMP_3     | 启动前 cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0) |
| output_stream_idx | -          | 设为 i_dx（dX 所在流）                                       |

**此规则适用于 FP32 BWD 和 AMP BWD。**现有 CONV_FP32_BWD 的单图单流实现需要拆分为双图。

### 2.4 内存管理铁律

- **禁止在算子内 cudaMalloc/cudaFree**。所有临时 workspace 通过 ctx.ensure_workspace_grow(StreamKind, size) 复用 DeviceContext 管理的 per-stream workspace。
- 如需额外临时张量，应在 Compiler 阶段申请 T_TEMP_FP32/FP16 区 DTensor。

---

## 3. 张量规格与 I/O 绑定

### 3.1 infer_conv_tensors 扩展（新增 bn_stats）

当前 infer_conv_tensors 返回 6 个张量（索引 0-5），需**新增第 6 个 bn_stats**：

| 索引  | 名称              | Shape (NHWC)    | Region                               | DType    | 说明                              |
| ----- | ----------------- | --------------- | ------------------------------------ | -------- | --------------------------------- |
| 0     | conv_weight       | {K,R,S,C}       | W_FIRST_CONV / W_DEEP_CONV           | FP32     | 主权重                            |
| 1     | conv_output       | {N,OH,OW,K}     | F_FEATURE_FP32/FP16                  | varies   | 特征图输出                        |
| 2     | conv_grad_slot    | {N,H,W,C}       | F_GRAD_SLOT_FP32/FP16                | varies   | 梯度槽（dX in-place）             |
| 3     | conv_weight_grad  | {K,R,S,C}       | G_FIRST_CONV / G_DEEP_CONV           | FP32     | FP32 权重梯度（经 CAST 后）       |
| 4     | conv_amp_w_fp16   | {K,R,S,C}       | A_FIRST_CONV / A_DEEP_CONV           | FP16     | AMP 工作权重                      |
| 5     | conv_amp_g_fp16   | {K,R,S,C}       | G_FIRST_CONV_FP16 / G_DEEP_CONV_FP16 | FP16     | AMP 权重梯度（BWD 直接产出 FP16） |
| **6** | **conv_bn_stats** | **{1,1,1,2*K}** | **T_TEMP_FP32**                      | **FP32** | **BN 统计量缓冲区（新增）**       |

**bn_stats 说明：**

- 大小为 2*K 个 float，前半段存 sum，后半段存 sq_sum。
- 不使用 padding（T_TEMP_FP32 的 cuda_alignment=1）。
- 所有 6 个 Conv 算子均将其作为**输入**传递（保持 I/O 种类一致），仅 CONV_AMP_FWD 真正写入。
- 非 AMP FWD 算子不对其执行 memset 或写入。

### 3.2 SubgraphPattern 修正

```cpp
// build_conv_forward
bool amp = GlobalRegistry::instance().using_amp();
n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
//   FP32: [X(跨层), weight(0), bn_stats(6)]
//   AMP:  [X(跨层), amp_w_fp16(4), bn_stats(6)]
n.output_indices = {1};   // output

// build_conv_backward
n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
n.input_indices  = amp
    ? std::vector<size_t>{4, 1, 2, 6}
    : std::vector<size_t>{0, 1, 2, 6};
//   [dY(跨层), weight, output, grad_slot(=X), bn_stats]
n.output_indices = amp
    ? std::vector<size_t>{2, 5}      // dX(in-place to grad_slot), dW to amp_g_fp16
    : std::vector<size_t>{2, 3};     // dX(in-place), dW to weight_grad

// build_conv_inference
n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
n.output_indices = {1};
```

### 3.3 各算子 I/O 速查表

| 算子          | Inputs                                 | Outputs           | Stream               |
| ------------- | -------------------------------------- | ----------------- | -------------------- |
| CONV_FP32_FWD | X, W(0), bn_stats(6)                   | Y(1)              | COMP_1               |
| CONV_FP32_INF | X, W(0), bn_stats(6)                   | Y(1)              | COMP_1               |
| CONV_FP32_BWD | dY, W(0), Y(1), X(2), bn_stats(6)      | dX(2), dW(3)      | dW:COMP_1, dX:COMP_3 |
| CONV_AMP_FWD  | X, W_fp16(4), bn_stats(6)              | Y(1)              | COMP_1               |
| CONV_AMP_INF  | X, W_fp16(4), bn_stats(6)              | Y(1)              | COMP_1               |
| CONV_AMP_BWD  | dY, W_fp16(4), Y(1), X(2), bn_stats(6) | dX(2), dW_fp16(5) | dW:COMP_1, dX:COMP_3 |

> 注：output(1) 在 BWD 中是 dY 的来源（正向输出），grad_slot(2) 是 X 同时也是 dX 的 in-place 目标。

---

## 4. CUDA 实现方案

### 4.1 通用设计：Per-Shape Graph Cache

参考 fc_op.cpp 的 FcAmpFwdCache 机制，为 Conv 构建 shape-keyed cache：

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C, K, R, S;
    int32_t  pad, stride;
    bool     is_amp;
    ComputeOp op;   // 区分 FWD/INF/BWD-wgrad/BWD-dgrad
    bool operator==(const ConvGraphCacheKey& o) const { ... }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    size_t workspace_size;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
};

namespace { std::unordered_map<ConvGraphCacheKey, ConvGraphCache> s_conv_caches; }
```

Cache 必须在 warmup 阶段预热（warmup_single_cudnn_op 中特殊处理 Conv）。

### 4.2 Graph 构建函数签名重构

现有 build_conv_fwd_graph 签名不接收输出 DTensor，导致 Y 的 stride 被手写。新签名直接传入全部 DTensor：

```cpp
// FP32 / AMP 共用（通过 dtype 区分）
std::shared_ptr<fe::graph::Graph> build_conv_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& params, cudnnHandle_t handle, DType dtype);

// BWD 拆分为两个独立 Graph
std::shared_ptr<fe::graph::Graph> build_conv_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& params, cudnnHandle_t handle, DType dtype);

std::shared_ptr<fe::graph::Graph> build_conv_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& params, cudnnHandle_t handle, DType dtype);

// AMP FWD 专用（Conv + GenStats）
std::shared_ptr<fe::graph::Graph> build_conv_amp_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const DTensor& dt_bn_stats,
    const ConvParams& params, cudnnHandle_t handle,
    ConvSearchMode search_mode);
```

### 4.3 FP32 FWD / INF（CONV_FP32_FWD、CONV_FP32_INF）

**Graph 构建（build_conv_fwd_graph with DType::FP32）：**

```cpp
auto graph = create_cudnn_graph(DType::FP32);
auto X = graph->tensor(Tensor_attributes()
    .set_name("X")
    .set_dim(to_fe_dim(dt_x.shape))
    .set_stride(make_nhwc_stride(dt_x))
    .set_data_type(DataType_t::FLOAT));

auto W = graph->tensor(Tensor_attributes()
    .set_name("W")
    .set_dim(to_fe_dim(dt_w.shape))
    .set_stride(make_nhwc_stride(dt_w))
    .set_data_type(DataType_t::FLOAT));

auto conv_opts = Conv_fprop_attributes()
    .set_padding({params.pad_h, params.pad_w})
    .set_stride({params.stride_h, params.stride_w})
    .set_dilation({1, 1});

auto Y = graph->conv_fprop(X, W, conv_opts);
Y->set_output(true)
  .set_name("Y")
  .set_dim(to_fe_dim(dt_y.shape))
  .set_stride(make_nhwc_stride(dt_y))
  .set_data_type(DataType_t::FLOAT);

finalize_cudnn_graph(graph.get(), handle);
```

**执行流程：**

```cpp
cudaStream_t s = ctx.stream(StreamKind::COMP_1);
cudnnHandle_t h = ctx.cudnn_handle(StreamKind::COMP_1);

auto& cache = get_or_build_cache(...);
void* ws = ctx.ensure_workspace_grow(StreamKind::COMP_1, cache.workspace_size);

std::unordered_map<int64_t, void*> vp;
vp[uid_x] = ctx.ptr_at(node.input_ids[0]);   // X
vp[uid_w] = ctx.ptr_at(node.input_ids[1]);   // W
vp[uid_y] = ctx.ptr_at(node.output_ids[0]);  // Y

graph->execute(h, vp, ws);

int si = state.get_or_register(s);
state.output_stream_idx = si;
state.streams[si].has_pending_work = true;
cudaEventRecord(state.streams[si].last_done_event, s);
```

CONV_FP32_INF 与 CONV_FP32_FWD 在 cuDNN 层面**完全等价**，可共用同一个 graph 构建函数和 cache，仅 ComputeOp 不同。

### 4.4 FP32 BWD（CONV_FP32_BWD）-- 拆双图 + 跨流同步

现有单图包含 conv_dgrad + conv_wgrad，但无法将 dgrad 绑定到 COMP_3。必须拆分为两个独立 Graph：

**WGrad Graph（COMP_1）：**

```cpp
auto graph = create_cudnn_graph(DType::FP32);
auto dY = graph->tensor(... FLOAT ...);
auto X  = graph->tensor(... FLOAT ...);
auto dW = graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   .set_dim(to_fe_dim(dt_dw.shape))
   .set_stride(make_nhwc_stride(dt_dw))
   .set_data_type(DataType_t::FLOAT);
```

**DGrad Graph（COMP_3）：**

```cpp
auto graph = create_cudnn_graph(DType::FP32);
auto dY = graph->tensor(... FLOAT ...);
auto W  = graph->tensor(... FLOAT ...);
auto dX = graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   .set_dim(to_fe_dim(dt_dx.shape))
   .set_stride(make_nhwc_stride(dt_dx))   // 必须与 X 完全一致
   .set_data_type(DataType_t::FLOAT);
```

**Launch 同步代码：**

```cpp
cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);
cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);
cudnnHandle_t h_dw = ctx.cudnn_handle(StreamKind::COMP_1);
cudnnHandle_t h_dx = ctx.cudnn_handle(StreamKind::COMP_3);
int i_dw = state.get_or_register(s_dw);
int i_dx = state.get_or_register(s_dx);

// 等待前序算子
int out_idx = state.output_stream_idx;
if (out_idx >= 0) {
    cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
    cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
}

// COMP_1: wgrad
auto& cache_w = get_or_build_wgrad_cache(...);
void* ws_w = ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);
cache_w.graph->execute(h_dw, vp_w, ws_w);
cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
state.streams[i_dw].has_pending_work = true;

// COMP_3: dgrad（等待 wgrad 完成，保护 X）
cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
auto& cache_x = get_or_build_dgrad_cache(...);
void* ws_x = ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);
cache_x.graph->execute(h_dx, vp_x, ws_x);
cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
state.streams[i_dx].has_pending_work = true;

state.output_stream_idx = i_dx;
```

### 4.5 AMP FWD（CONV_AMP_FWD）-- Conv + GenStats

**Graph 构建：**

```cpp
auto graph = create_cudnn_graph(DType::FP16);  // I/O = HALF

auto X = graph->tensor(... HALF, make_nhwc_stride(dt_x) ...);
auto W = graph->tensor(... HALF, make_nhwc_stride(dt_w) ...);

auto conv_out = graph->conv_fprop(X, W, conv_opts);
conv_out->set_is_virtual(true).set_data_type(DataType_t::FLOAT);

// GenStats 附加
auto genstats_opts = Genstats_attributes()
    .set_compute_data_type(DataType_t::FLOAT);
auto [sum, sq_sum] = graph->genstats(conv_out, genstats_opts);

// Y 输出
auto Y = conv_out;
Y->set_output(true)
  .set_name("Y")
  .set_dim(to_fe_dim(dt_y.shape))
  .set_stride(make_nhwc_stride(dt_y))
  .set_data_type(DataType_t::HALF);

// sum / sq_sum 输出到 bn_stats 的两个半区
sum->set_output(true)
    .set_dim({1, K, 1, 1})
    .set_stride({K, 1, K, K})
    .set_data_type(DataType_t::FLOAT);
sq_sum->set_output(true)
    .set_dim({1, K, 1, 1})
    .set_stride({K, 1, K, K})
    .set_data_type(DataType_t::FLOAT);
```

**Variant Pack 中 bn_stats 的映射：**

```cpp
float* bn_stats_ptr = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));  // bn_stats 是输入
vp[uid_sum]    = bn_stats_ptr;
vp[uid_sq_sum] = bn_stats_ptr + K;  // 后半段
```

**EXHAUSTIVE_C 分支（AMP FWD 专用）：**

```cpp
if (search_mode == ConvSearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
    std::string key = build_shape_key(
        "conv_genstats", dtype_str,
        N, H, W, C, K, R, S, stride, pad
    );
    auto exp_rec = ta_v4::experience::lookup(key);
    if (exp_rec != nullptr) {
        graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B});
        auto [status, matched] = match_and_build_plan(graph, candidates, exp_rec, handle);
        if (!matched) goto fallback;
    } else { goto fallback; }
#else
    goto fallback;
#endif
} else {
fallback:
    finalize_cudnn_graph(graph.get(), handle);  // 内部使用 HeurMode B + FALLBACK
}
```

### 4.6 AMP INF（CONV_AMP_INF）

纯 conv_fprop，**无 GenStats**。结构与 FP32 FWD 相同，仅数据类型为 HALF：

```cpp
auto graph = create_cudnn_graph(DType::FP16);
// X, W, Y 同 AMP FWD，但不创建 genstats 节点
finalize_cudnn_graph(graph.get(), handle);
```

bn_stats 作为输入传入算子，但 graph 中不引用（variant pack 中不需要映射 bn_stats 指针到 graph 节点）。

### 4.7 AMP BWD（CONV_AMP_BWD）-- 双图 + 跨流同步

与 FP32 BWD 完全相同的流架构，但数据类型为 HALF，且 dW 输出 FP16：

**WGrad Graph（COMP_1）：**

```cpp
auto graph = create_cudnn_graph(DType::FP16);
// dY(HALF), X(HALF)
auto dW = graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   .set_dim(to_fe_dim(dt_dw.shape))
   .set_stride(make_nhwc_stride(dt_dw))
   .set_data_type(DataType_t::HALF);   // FP16 输出
```

**DGrad Graph（COMP_3）：**

```cpp
auto graph = create_cudnn_graph(DType::FP16);
// dY(HALF), W(HALF)
auto dX = graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   .set_dim(to_fe_dim(dt_dx.shape))
   .set_stride(make_nhwc_stride(dt_dx))
   .set_data_type(DataType_t::HALF);
```

**流同步代码与 FP32 BWD 完全一致**（见 4.4）。

**EXHAUSTIVE_C：**

- WGrad 查 "conv_wgrad"
- DGrad 查 "conv_dgrad"

---

## 5. CPU 实现

### 5.1 FWD / INF

复用现有 launch_conv_fwd_cpu_xnnpack（XNNPACK）和 launch_conv_fwd_cpu_naive（fallback）。CONV_FP32_INF 与 CONV_FP32_FWD 共用同一 CPU launch 函数。

### 5.2 BWD

复用现有 launch_conv_bwd_cpu_naive。先算 dW、再算 dX（单线程顺序执行，天然满足 dX 覆盖 X 的约束）。

### 5.3 AMP CPU

不支持。launch_cpu 指向统一的无支持函数：

```cpp
static void launch_conv_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    TR_CHECK(false, NotImplementedError, "Conv AMP operators do not support CPU execution");
}
```

---

## 6. 枚举与注册表扩展

### 6.1 include/renaissance/graph/op_kind.h

```cpp
enum class ComputeOp : uint16_t {
    // ... 现有 ...
    CONV_FP32_FWD,
    CONV_FP32_BWD,
    CONV_FP32_INF,      // 新增
    CONV_AMP_FWD,
    CONV_AMP_BWD,
    CONV_AMP_INF,       // 新增
    // ...
    COUNT,
};
```

### 6.2 include/renaissance/core/types.h

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B = 0,   // 默认
    EXHAUSTIVE_C = 1   // 仅 AMP Conv，仅 A100/RTX5090
};
```

### 6.3 global_registry.h + global_registry.cpp

```cpp
class GlobalRegistry {
public:
    GlobalRegistry& conv_search_mode(ConvSearchMode mode);
    ConvSearchMode conv_search_mode() const;
private:
    std::atomic<int> fixed_conv_search_mode_{0};  // 0 = HEURISTIC_B
};
```

实现遵循 fixed 型变量规则：允许幂等赋值，禁止非幂等修改。

### 6.4 src/backend/op_stream_policy.cpp

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;
```

### 6.5 src/backend/op_registry.cpp

```cpp
bool require_warmup(ComputeOp op) noexcept {
    switch (op) {
        // ... 现有 ...
        case ComputeOp::CONV_FP32_INF:
        case ComputeOp::CONV_AMP_INF:
            return true;
        // ...
    }
}
```

warmup_single_cudnn_op 中 Conv 预热路径需扩展以覆盖 INF 和 AMP 变体。

---

## 7. Layer Descriptor 修改

### 7.1 src/graph/layer_descriptor_registry.cpp

**infer_conv_tensors** 新增 bn_stats（索引 6）：

```cpp
// 在现有 6 个张量之后追加
{ TensorDesc d;
  d.name = "conv_bn_stats";
  d.shape = Shape{1, 1, 1, cp.out_channels * 2};
  d.region = Region::T_TEMP_FP32;
  d.dtype = DType::FP32;
  descs.push_back(d);
}
```

**build_conv_forward / build_conv_backward / build_conv_inference** 修正 ComputeOp 选择（根据 using_amp()）和 input_indices / output_indices（见 3.2）。

### 7.2 src/graph/compiler.cpp

alloc_conv_group 中**无需**新增 bn_stats 分配，因为 infer_conv_tensors 返回的 TensorDesc 已由 Compiler 统一处理。若 Compiler 实际通过 alloc_conv_group 显式分配而非依赖 infer_conv_tensors，则需在 alloc_conv_group 中新增 memory_plan_.alloc_temp_fp32(Shape{1,1,1,shape.n()*2}) 或类似调用，并确保 infer_conv_tensors 与之一致。

---

## 8. 文件改动清单

| 文件                                       | 改动类型  | 说明                                                         |
| ------------------------------------------ | --------- | ------------------------------------------------------------ |
| include/renaissance/graph/op_kind.h        | 修改      | 新增 CONV_FP32_INF、CONV_AMP_INF                             |
| include/renaissance/core/types.h           | 修改      | 新增 ConvSearchMode 枚举                                     |
| include/renaissance/core/global_registry.h | 修改      | 新增 fixed_conv_search_mode_、setter/getter                  |
| src/core/global_registry.cpp               | 修改      | 实现 setter/getter（幂等检查）                               |
| src/backend/op_stream_policy.cpp           | 修改      | 新增 INF 默认流 = COMP_1                                     |
| src/backend/op_registry.cpp                | 修改      | require_warmup 新增 INF；预热路径扩展                        |
| src/graph/layer_descriptor_registry.cpp    | 修改      | infer_conv_tensors 新增 bn_stats；build_conv_* 修正 I/O 和 ComputeOp |
| src/backend/ops/dtensor/conv_op.cpp        | 重写      | 6 个 launch 函数、CPU kernel、注册入口                       |
| src/backend/ops/dtensor/conv_op_impl.cpp   | 重写/扩展 | FP32/AMP 的 cuDNN FE Graph 构建函数（含双图 BWD）            |

---

## 9. 实施优先级

按以下顺序实现，每一步都可独立编译验证：

1. **基础设施（P0，无风险）**
   - op_kind.h：新增 CONV_FP32_INF、CONV_AMP_INF
   - types.h：新增 ConvSearchMode
   - global_registry.h/cpp：新增 conv_search_mode
   - op_stream_policy.cpp：新增 INF 流
   - op_registry.cpp：require_warmup 扩展

2. **CONV_FP32_INF（P0，低风险）**
   - 复用 build_conv_fwd_graph，仅改枚举
   - CPU INF 共用 FWD 函数
   - 注册到 g_compute_op_table

3. **CONV_FP32_BWD 双流改造（P0，中风险）**
   - 将 build_conv_bwd_graph 拆为 build_conv_wgrad_graph + build_conv_dgrad_graph
   - launch_conv_bwd_cuda 改为双流 + event 同步
   - 验证 dX 覆盖 X 正确性

4. **CONV_AMP_FWD（P1，中风险）**
   - 实现 build_conv_amp_fwd_graph（Conv+GenStats，HALF I/O）
   - 处理 bn_stats 的 Variant Pack 映射（sum / sq_sum 分半区）
   - 使用 padded_c() 设置 dim（特征图）

5. **CONV_AMP_INF（P1，低风险）**
   - 纯 conv_fprop HALF，无 GenStats
   - 复用 AMP FWD 的大部分逻辑

6. **CONV_AMP_BWD（P1，高风险）**
   - 实现 build_conv_amp_wgrad_graph + build_conv_amp_dgrad_graph
   - 流同步与 FP32 BWD 一致
   - dW 输出 FP16（HALF），写入 G_FIRST_CONV_FP16
   - EXHAUSTIVE_C 分别查询 wgrad/dgrad

7. **EXHAUSTIVE_C 集成（P2）**
   - 接入 include/generated/ 下的经验表
   - 非 A100/RTX5090 自动 fallback Heuristic B

---

## 10. 风险与缓解

| 风险点                              | 缓解措施                                                     |
| ----------------------------------- | ------------------------------------------------------------ |
| dX 覆盖 X 的数据竞争                | BWD 必须拆双图：wgrad @ COMP_1 -> event -> dgrad @ COMP_3。单图无法满足跨流需求。 |
| Y/dX/dW stride 错误导致静默数据错乱 | 严格使用 DTensor 的 n_stride_cuda() 等系列，禁止手写 align_up。build_*_graph 签名需传入输出 DTensor。 |
| bn_stats 内存布局不匹配             | bn_stats 定义为 {1,1,1,2*K}（4D），T_TEMP_FP32 区无 padding。Variant Pack 中 sum 指向前半段、sq_sum 指向 +K 偏移。 |
| cuDNN FE Graph 数据类型不匹配       | AMP 所有 tensor 必须显式 set_data_type(HALF/FLOAT)；create_cudnn_graph(DType::FP16) 自动设置 io=HALF, intermediate/compute=FLOAT。 |
| Experience Key 不匹配               | build_shape_key 的字段顺序、数值必须与 include/generated/ 中的表完全一致（cbr_experience_*.hpp 格式）。 |
| Workspace 泄漏/每轮分配             | 通过 ctx.ensure_workspace_grow() 复用 DeviceContext 的 per-stream workspace；禁止 cudaMalloc/Free。 |
| CPU BWD 性能                        | 当前为 Naive 六重循环，仅用于 correctness。生产训练以 GPU 为主，CPU BWD 不在本次优化范围内。 |
| Graph Cache Key 冲突                | Cache key 必须包含 handle、全部 shape、params、is_amp、op type。BWD 的 wgrad 和 dgrad 使用独立 cache。 |

---

## 11. 参考代码索引

| 功能                          | 参考文件         | 关键函数/类                                                  |
| ----------------------------- | ---------------- | ------------------------------------------------------------ |
| AMP FWD Graph (Conv+GenStats) | cbr_fwd_fp16.cpp | build_conv_genstats_graph()                                  |
| AMP BWD Graph (WGrad/DGrad)   | cbr_bwd_fp16.cpp | build_conv_wgrad_graph() / build_conv_dgrad_graph()          |
| AMP INF Graph                 | cbr_inf_fp16.cpp | try_build_single_graph()                                     |
| FE 数据类型与 stride          | fc_op.cpp        | build_fc_amp_fwd_conv_graph(), make_nhwc_stride              |
| 双流同步 (dW->dX)             | fc_op.cpp        | launch_fc_amp_bwd_cuda()                                     |
| FP32 Graph 构建               | conv_op_impl.cpp | 现有 build_conv_fwd_graph() / build_conv_bwd_graph()（需重构） |
| CPU XNNPACK FWD               | conv_op.cpp      | launch_conv_fwd_cpu_xnnpack()                                |
| CPU Naive BWD                 | conv_op.cpp      | launch_conv_bwd_cpu()                                        |
| 算子注册                      | fc_op.cpp        | register_op_fc()                                             |
| Graph 公共辅助                | cudnn_utils.h    | create_cudnn_graph(), finalize_cudnn_graph(), to_fe_dim()    |



# 【小伙伴C】

> 基于 ATN1.md（需求原始定义 + 4 轮方案讨论）与 ATN2.md（算子添加标准流程）综合分析。  
> 版本: 2.0  
> 日期: 2026-06-02  

---

## 一、现状诊断（基于代码实际状态）

### 1.1 现有代码位置与功能

| 文件                                         | 当前状态                                | 说明                                                         |
| -------------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| `src/backend/ops/dtensor/conv_op.cpp`        | 有 FP32 FWD/BWD launch + CPU kernels    | 注册 `CONV_FP32_FWD/BWD`；**无 Graph Cache**；**每调用 cudaMalloc workspace**；BWD 单流 |
| `src/backend/ops/dtensor/conv_op_impl.cpp`   | 有 FP32 Graph 构建 + AMP `OpDescriptor` | `build_conv_fwd/bwd_graph`（供 conv_op.cpp 调用）；`build_graph_fwd/bwd`（AMP，数据类型硬编码 FLOAT） |
| `src/backend/ops/dtensor/conv_op_legacy.cpp` | 未跟踪的旧实现                          | 不在 CMakeLists 中，不编译                                   |
| `src/graph/layer_descriptor_registry.cpp`    | 6 张量，无 bn_stats                     | `infer_conv_tensors` 返回 6 个；`build_conv_*` 硬编码 `CONV_AMP_FWD/BWD` |
| `include/renaissance/graph/op_kind.h`        | 缺 INF 枚举                             | 有 `CONV_FP32_FWD/BWD`、`CONV_AMP_FWD/BWD`；**无 `CONV_FP32_INF`、`CONV_AMP_INF`** |
| `include/renaissance/core/types.h`           | 缺搜索模式枚举                          | 无 `ConvSearchMode`                                          |
| `include/renaissance/core/global_registry.h` | 缺搜索模式配置                          | 无 `conv_search_mode()`                                      |
| `src/backend/op_stream_policy.cpp`           | BWD 映射到 COMP_3                       | 但 `conv_op.cpp` launch 实际用 COMP_1（**不匹配**）          |
| `src/backend/op_registry.cpp`                | `require_warmup` 有 FWD/BWD             | **无 INF**；warmup 用 cudaMalloc 临时 workspace              |

### 1.2 关键缺陷

1. **无 cuDNN FE Graph Cache**：每次 launch 重建 Graph + malloc workspace，热路径性能差，CUDA Graph 捕获兼容性差。
2. **BWD 流分配错误**：`op_stream_policy.cpp` 声明 BWD 用 COMP_3，但 `conv_op.cpp` 实际用 COMP_1。
3. **缺少 INF 算子**：`ComputeOp` 枚举无 `CONV_FP32_INF`、`CONV_AMP_INF`。
4. **缺少 bn_stats 张量**：Conv 层 6 张量 → 需扩展为 7 张量。
5. **AMP 实现半残**：`conv_op_impl.cpp` 中 AMP graph builder 数据类型写死 FLOAT，未接入 launch 路径。
6. **双流 BWD 未实现**：用户明确要求 dW@COMP_1 → event → dX@COMP_3。

---

## 二、约束总览

### 2.1 卷积参数（BluePrint/ArchPlan 层已检查，算子内不做校验）

| 参数        | 约束                             |
| ----------- | -------------------------------- |
| kernel_size | 仅 1, 3, 5, 7（正方形）          |
| stride      | 仅 1, 2                          |
| dilation    | 固定 1                           |
| groups      | 固定 1（普通卷积，无 Depthwise） |
| bias        | **不支持**（无 bias 分区）       |
| padding     | 无限制                           |

### 2.2 框架铁律

- **dX 覆盖 X**：反向传播梯度输出覆盖原始输入，共用同一 DTensor。
- **先 dW 后 dX**：BWD 必须先完成 wgrad，再启动 dgrad。dW@COMP_1，dX@COMP_3，显式 `cudaStreamWaitEvent`。
- **张量对齐**：同一层输入输出张量的数量、种类、形状在 CPU/GPU/FP32/AMP 下完全一致。
- **禁止算子内临时内存分配**：workspace 用 `ctx.ensure_workspace_grow()`；DTensor 在 Compiler 阶段申请。
- **NHWC 物理布局**：cuDNN FE `set_stride({N_stride, C_stride, H_stride, W_stride})`，其中 `C_stride=1` 表示 NHWC。

---

## 三、张量布局（扩展后）

### 3.1 Conv 层张量清单（7 个）

| 索引  | 名称              | Region                               | Dtype    | 说明                         |
| ----- | ----------------- | ------------------------------------ | -------- | ---------------------------- |
| 0     | conv_weight       | W_FIRST_CONV / W_DEEP_CONV           | FP32     | 主权重                       |
| 1     | conv_output       | F_FEATURE_FP32 / F_FEATURE_FP16      | 随 AMP   | 输出特征图                   |
| 2     | conv_grad_slot    | F_GRAD_SLOT_FP32 / F_GRAD_SLOT_FP16  | 随 AMP   | 梯度槽（dX in-place 覆盖 X） |
| 3     | conv_weight_grad  | G_FIRST_CONV / G_DEEP_CONV           | FP32     | 权重梯度                     |
| 4     | conv_amp_w_fp16   | A_FIRST_CONV / A_DEEP_CONV           | FP16     | AMP 工作权重（!AMP 时占位）  |
| 5     | conv_amp_g_fp16   | G_FIRST_CONV_FP16 / G_DEEP_CONV_FP16 | FP16     | AMP 权重梯度（!AMP 时占位）  |
| **6** | **conv_bn_stats** | **T_TEMP_FP32**                      | **FP32** | **BN 统计量（新增）**        |

### 3.2 bn_stats 规格

- **形状**：`Shape{1, 1, 1, K}`（4D NHWC per-channel），或 `Shape{K}`（1D）。建议采用 **4D** `Shape{1, 1, 1, K}`，与 GenStats 输出 `{1, K, 1, 1}` 语义一致。
- **区域**：`T_TEMP_FP32`（不使用 padding）。
- **写入情况**：仅 `CONV_AMP_FWD` 通过 GenStats 写入 `sum` + `sq_sum`；其余 5 个算子不写入、不 memset。
- **Variant Pack 映射**：GenStats 输出两个 `{1, K, 1, 1}` 张量，`sum` 指向 bn_stats 首地址，`sq_sum` 指向 `bn_stats_ptr + K * sizeof(float)`。

### 3.3 算子 I/O 绑定

#### FWD / INF

```
input_indices  = {0}          // weight（X 由跨层链注入）
output_indices = {1, 6}       // output + bn_stats
```

- FP32: 算子枚举 `CONV_FP32_FWD` / `CONV_FP32_INF`，weight 用索引 0（FP32）
- AMP: 算子枚举 `CONV_AMP_FWD` / `CONV_AMP_INF`，weight 用索引 4（FP16）

#### BWD

```
input_indices  = {0, 1, 2}    // weight, output, grad_slot(=X)
output_indices = {2, 3}       // dX(in-place to grad_slot), dW(FP32)
// AMP 时:
// input_indices  = {4, 1, 2}    // amp_w_fp16, output, grad_slot
// output_indices = {2, 5}       // dX(in-place), dW(FP16 to amp_g_fp16)
```

- `output_indices` **不包含 bn_stats**（BWD 数学上不需要），bn_stats 仅在 layer 层面存在以维持对齐。

---

## 四、详细设计方案

### 4.1 基础设施扩展

#### (a) `include/renaissance/graph/op_kind.h`

```cpp
// ComputeOp 枚举新增
CONV_FP32_INF,   // 推理专用
CONV_AMP_INF,    // 推理专用

// ConvParams 已存在，无需新增
```

#### (b) `include/renaissance/core/types.h`

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B = 0,   // 默认
    EXHAUSTIVE_C = 1   // 仅 AMP Conv，仅 A100/RTX5090
};
```

#### (c) `include/renaissance/core/global_registry.h` + `src/core/global_registry.cpp`

```cpp
class GlobalRegistry {
public:
    GlobalRegistry& conv_search_mode(ConvSearchMode mode);
    ConvSearchMode conv_search_mode() const;
private:
    std::atomic<int> fixed_conv_search_mode_{
        static_cast<int>(ConvSearchMode::HEURISTIC_B)
    };
};
```

Setter 为链式调用风格，遵循 fixed 型变量规则（幂等赋值允许，非幂等赋值报错）。

#### (d) `src/backend/op_registry.cpp`

- `require_warmup()` 新增 `CONV_FP32_INF`、`CONV_AMP_INF`。
- `warmup_single_cudnn_op()`：当前只处理 FWD，需扩展为统一处理 FWD/INF（两者 Graph 结构相同）。BWD 的 warmup 走 generic `launch_cuda` 路径。

#### (e) `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::CONV_FP32_FWD:
case ComputeOp::CONV_AMP_FWD:
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;

case ComputeOp::CONV_FP32_BWD:
case ComputeOp::CONV_AMP_BWD:
    return StreamKind::COMP_3;   // 代表流 = dX 输出流
```

### 4.2 Layer Descriptor 与 Compiler 修改

#### `src/graph/layer_descriptor_registry.cpp`

**`infer_conv_tensors`**：新增第 7 个张量 bn_stats：

```cpp
{ TensorDesc d;
  d.name = "conv_bn_stats";
  d.shape = Shape{1, 1, 1, K};   // 或 Shape{K}，与 compile_bn 一致
  d.region = Region::T_TEMP_FP32;
  d.dtype = DType::FP32;
  descs.push_back(d);
}
```

**`build_conv_forward`**：

```cpp
bool amp = GlobalRegistry::instance().using_amp();
SubgraphPattern::Node n;
n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
n.input_indices  = amp ? std::vector<size_t>{4} : std::vector<size_t>{0};
n.output_indices = {1, 6};   // output + bn_stats
```

**`build_conv_backward`**：

```cpp
bool amp = GlobalRegistry::instance().using_amp();
SubgraphPattern::Node n;
n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
n.input_indices  = amp ? std::vector<size_t>{4, 1, 2}
                        : std::vector<size_t>{0, 1, 2};
n.output_indices = amp ? std::vector<size_t>{2, 5}
                        : std::vector<size_t>{2, 3};
```

**`build_conv_inference`**：

```cpp
bool amp = GlobalRegistry::instance().using_amp();
SubgraphPattern::Node n;
n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
n.input_indices  = amp ? std::vector<size_t>{4} : std::vector<size_t>{0};
n.output_indices = {1, 6};
```

#### `src/graph/compiler.cpp`

在以下位置添加 `LayerKind::Conv` 分支：

| 位置                                        | 操作                            |
| ------------------------------------------- | ------------------------------- |
| `compile_primitive()`                       | 标记 Conv 为 primitive          |
| `build_computation_graph()` FWD 索引映射    | `idx = 0`                       |
| `build_computation_graph()` BWD grad 槽映射 | `idx = -1`（in-place）          |
| `build_computation_graph()` BWD 节点后处理  | 追加 X 为额外输入（同 MaxPool） |
| grad_id 追踪                                | 加入 `LayerKind::Conv`          |

### 4.3 cuDNN FE Graph 缓存设计

参考 `fc_op.cpp` 的 `FcAmpFwdCache`，为 Conv 实现统一的 per-shape cache：

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C;       // 输入形状
    int32_t  K, R, S;          // 权重/输出形状
    int32_t  pad_h, pad_w;
    int32_t  stride_h, stride_w;
    bool     is_amp;           // 区分 FP32 / AMP
    ComputeOp op;              // 区分 FWD/BWD/INF
    bool operator==(const ConvGraphCacheKey& o) const { ... }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::vector<std::pair<int64_t, int64_t>> uid_to_dtensor_id;  // uid → DTensor id
    size_t workspace_size;
};

// 每个 (op, is_amp) 组合独立缓存
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_graph_caches[6];  // 对应 6 个算子
```

Cache 命中时：

1. 复用 Graph 对象
2. 通过 `uid_to_dtensor_id` 映射填充 Variant Pack（`ctx.ptr_at(id)` 取指针）
3. `ctx.ensure_workspace_grow(sk, cache.workspace_size)` 复用 workspace

### 4.4 NHWC ↔ cuDNN FE 的 stride 映射

```cpp
// 特征图 NHWC
tensor->set_dim({N, padded_C, H, W})       // cuDNN 逻辑顺序 N,C,H,W
       .set_stride({dt.n_stride_cuda(),     // N 步幅
                    dt.c_stride_cuda(),     // C 步幅 = 1（NHWC）
                    dt.h_stride_cuda(),     // H 步幅 = W * padded_C
                    dt.w_stride_cuda()});   // W 步幅 = padded_C

// 权重 KRSC → cuDNN dim {K, C, R, S}
tensor->set_dim({K, padded_C, R, S})
       .set_stride({dt_w.n_stride_cuda(),
                    dt_w.c_stride_cuda(),
                    dt_w.h_stride_cuda(),
                    dt_w.w_stride_cuda()});
```

**FP32**：`padded_C = C`。  
**AMP**：`padded_C = dt.padded_c()`（对齐到 8 的倍数）。**不可**用 `dt.c()`。

### 4.5 FP32 FWD / INF（`CONV_FP32_FWD`、`CONV_FP32_INF`）

#### Graph 构建

```cpp
auto graph = std::make_shared<fe::graph::Graph>();
graph->set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

auto X = graph->tensor(make_tensor_attr(dt_x, "X", fe::DataType_t::FLOAT));
auto W = graph->tensor(make_tensor_attr(dt_w, "W", fe::DataType_t::FLOAT));

auto conv_options = fe::graph::Conv_fprop_attributes()
    .set_padding({pad_h, pad_w})
    .set_stride({stride_h, stride_w})
    .set_dilation({1, 1});

auto Y = graph->conv_fprop(X, W, conv_options);
Y->set_output(true)
  .set_dim({N, K, OH, OW})
  .set_stride({OH*OW*K_aligned, 1, OW*K_aligned, K_aligned})
  .set_data_type(fe::DataType_t::FLOAT)
  .set_uid(UID_Y);

// 无需 GenStats（FP32 不支持）
// bn_stats 不在 Graph 中定义，但 output_indices 中保留以保持对齐
```

#### 执行流程

```cpp
cudaStream_t s = ctx.stream(StreamKind::COMP_1);
cudnnHandle_t h = ctx.cudnn_handle(StreamKind::COMP_1);
int si = state.get_or_register(s);

// 等待前序
if (state.output_stream_idx >= 0) {
    cudaStreamWaitEvent(s, state.streams[state.output_stream_idx].last_done_event, 0);
}

// Cache lookup / build
auto& cache = get_or_build_cache(key, h, ...);
ctx.ensure_workspace_grow(StreamKind::COMP_1, cache.workspace_size);

// Variant Pack
auto vp = fe::graph::VariantPack();
for (auto [uid, dt_id] : cache.uid_to_dtensor_id) {
    vp.set_workspace_pointer(uid, ctx.ptr_at(dt_id));
}

// Execute
cache.graph->execute(h, vp, ctx.workspace(StreamKind::COMP_1));

// Event
state.output_stream_idx = si;
state.streams[si].has_pending_work = true;
cudaEventRecord(state.streams[si].last_done_event, s);
```

**INF 与 FWD 的 Graph 结构完全相同**，可共用 cache（key 中包含 `op` 字段区分，或 INF 直接查 FWD 的 cache）。

### 4.6 FP32 BWD（`CONV_FP32_BWD`）

#### 双流拆分

用户明确要求 dW@COMP_1、dX@COMP_3。`op_stream_policy.cpp` 已映射 BWD 到 COMP_3（代表流）。

拆分为**两个独立 Graph**：

1. **WGrad Graph**（COMP_1）：`conv_wgrad(dY, X) → dW`
2. **DGrad Graph**（COMP_3）：`conv_dgrad(dY, W) → dX`

#### Graph 构建

**WGrad**：

```cpp
auto dW = graph->conv_wgrad(dY, X, wgrad_options);
dW->set_output(true)
   .set_dim({K, R, S, C})
   .set_stride({dt_dw.n_stride_cuda(), dt_dw.c_stride_cuda(),
                dt_dw.h_stride_cuda(), dt_dw.w_stride_cuda()})
   .set_data_type(fe::DataType_t::FLOAT)
   .set_uid(UID_DW);
```

**DGrad**：

```cpp
auto dX = graph->conv_dgrad(dY, W, dgrad_options);
dX->set_output(true)
   .set_dim({N, C, H, W})
   .set_stride({dt_dx.n_stride_cuda(), dt_dx.c_stride_cuda(),
                dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda()})
   .set_data_type(fe::DataType_t::FLOAT)
   .set_uid(UID_DX);
```

#### 执行与同步

```cpp
cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);
cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);
cudnnHandle_t h_dw = ctx.cudnn_handle(StreamKind::COMP_1);
cudnnHandle_t h_dx = ctx.cudnn_handle(StreamKind::COMP_3);
int i_dw = state.get_or_register(s_dw);
int i_dx = state.get_or_register(s_dx);

// 等待前序（两流都 wait）
if (state.output_stream_idx >= 0) {
    cudaStreamWaitEvent(s_dw, state.streams[state.output_stream_idx].last_done_event, 0);
    cudaStreamWaitEvent(s_dx, state.streams[state.output_stream_idx].last_done_event, 0);
}

// COMP_1: WGrad
auto& cache_w = get_or_build_wgrad_cache(...);
ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);
cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1));
cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
state.streams[i_dw].has_pending_work = true;

// COMP_3: DGrad（等待 WGrad 完成，保护 X）
cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
auto& cache_x = get_or_build_dgrad_cache(...);
ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);
cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3));
cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
state.streams[i_dx].has_pending_work = true;

state.output_stream_idx = i_dx;
```

### 4.7 AMP FWD（`CONV_AMP_FWD`）

#### Conv + GenStats 融合图

```cpp
graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

auto X = graph->tensor(make_tensor_attr(dt_x, "X", fe::DataType_t::HALF));
auto W = graph->tensor(make_tensor_attr(dt_w, "W", fe::DataType_t::HALF));

auto conv_out = graph->conv_fprop(X, W, conv_options);
// conv_out 为内部虚张量，不 set_output

auto genstats_attrs = fe::graph::Genstats_attributes()
    .set_name("genstats")
    .set_compute_data_type(fe::DataType_t::FLOAT);
auto [sum, sq_sum] = graph->genstats(conv_out, genstats_attrs);

// Y 输出
conv_out->set_output(true)
        ->set_dim({N, K, OH, OW})
        ->set_stride({OH*OW*K_aligned, 1, OW*K_aligned, K_aligned})
        ->set_data_type(fe::DataType_t::HALF)
        ->set_uid(UID_Y);

// sum / sq_sum 输出到 bn_stats
sum->set_output(true)
    ->set_dim({1, K, 1, 1})
    ->set_stride({K, 1, K, K})
    ->set_data_type(fe::DataType_t::FLOAT)
    ->set_uid(UID_SUM);

sq_sum->set_output(true)
      ->set_dim({1, K, 1, 1})
      ->set_stride({K, 1, K, K})
      ->set_data_type(fe::DataType_t::FLOAT)
      ->set_uid(UID_SQ_SUM);
```

#### Variant Pack 映射

```cpp
vp.set_workspace_pointer(UID_Y,      ctx.ptr_at(dt_y.id));
vp.set_workspace_pointer(UID_SUM,    ctx.ptr_at(dt_bn.id));           // sum → bn_stats 首地址
vp.set_workspace_pointer(UID_SQ_SUM, ctx.ptr_at(dt_bn.id) + K * sizeof(float));  // sq_sum → 偏移 K
```

**注意**：`bn_stats` 是单个 `{K}` 或 `{1,1,1,K}` buffer，GenStats 的两个输出通过指针偏移映射到同一块 buffer。

#### EXHAUSTIVE_C 搜索

```cpp
if (GlobalRegistry::instance().conv_search_mode() == ConvSearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
    std::string key = build_shape_key("conv_genstats", "fp16", N, H, W, C, K, R, S, stride, pad);
    auto exp_rec = ta_v4::experience::lookup(key);
    if (exp_rec) {
        graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B});
        auto [status, matched] = match_and_build_plan(graph, candidates, exp_rec, handle);
        if (matched) goto done;
    }
#endif
}
// fallback
graph->create_execution_plans({fe::HeurMode_t::B});
graph->check_support(handle);
graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
done:
```

### 4.8 AMP BWD（`CONV_AMP_BWD`）

#### 两个独立 Graph

**WGrad Graph**（COMP_1）：

```cpp
wgrad_graph->set_io_data_type(fe::DataType_t::HALF)
            ->set_compute_data_type(fe::DataType_t::FLOAT);
auto dY = wgrad_graph->tensor(...HALF...);
auto X  = wgrad_graph->tensor(...HALF...);
auto dW = wgrad_graph->conv_wgrad(dY, X, wgrad_options);
dW->set_output(true)
   ->set_dim({K, R, S, C})
   ->set_stride({...})
   ->set_data_type(fe::DataType_t::HALF)   // ★ AMP BWD 输出 FP16 dW
   ->set_uid(UID_DW);
```

**DGrad Graph**（COMP_3）：

```cpp
auto dY = dgrad_graph->tensor(...HALF...);
auto W  = dgrad_graph->tensor(...HALF...);
auto dX = dgrad_graph->conv_dgrad(dY, W, dgrad_options);
dX->set_output(true)
   ->set_dim({N, C, H, W})
   ->set_stride({...})  // 与 X 完全一致
   ->set_data_type(fe::DataType_t::HALF)
   ->set_uid(UID_DX);
```

#### 流同步

与 **4.6 FP32 BWD** 完全一致（`wgrad@COMP_1 → event → dgrad@COMP_3`）。

#### dW 输出 FP16

Conv AMP BWD 的 dW 输出到 `G_FIRST_CONV_FP16` / `G_DEEP_CONV_FP16`（索引 5）。后续由 Compiler 自动插入 `RANGE_CAST_FP16_TO_FP32` 转换为 FP32 的 G_*_CONV（索引 3）。

### 4.9 AMP INF（`CONV_AMP_INF`）

纯 `conv_fprop`，**无 GenStats**：

```cpp
graph->set_io_data_type(fe::DataType_t::HALF)
      ->set_compute_data_type(fe::DataType_t::FLOAT);
auto Y = graph->conv_fprop(X, W, conv_options);
Y->set_output(true)->set_data_type(fe::DataType_t::HALF);
```

- 输出 Y 的 stride 同 AMP FWD。
- bn_stats 作为 dummy 输出（`output_indices` 含 6），但 Graph 中不引用。

### 4.10 CPU 实现

#### FWD / INF

共用同一函数：`launch_conv_fwd_cpu()`。

- 优先 XNNPACK（`xnn_define_convolution_2d`，NHWC layout）。
- 若 XNNPACK 不支持当前参数组合（或 `TR_USE_XNNPACK` 未定义），fallback 到 naive 嵌套循环。
- INF 与 FWD 计算逻辑完全相同，仅算子枚举不同。

#### BWD

`launch_conv_bwd_cpu()`：

- Naive 6 重循环实现 dgrad + wgrad。
- 先 memset dX=0、dW=0，再累加。
- 单线程顺序执行（先 dW 后 dX，天然满足约束）。

#### AMP CPU

不支持，统一指向 `launch_conv_amp_cpu_not_supported()`，抛出异常。

---

## 五、文件变更清单（按 ATN2.md Checklist 顺序）

| #    | 文件                                         | 必须？ | 操作                                                         |
| ---- | -------------------------------------------- | :----: | ------------------------------------------------------------ |
| 1    | `include/renaissance/graph/op_kind.h`        |   ✅    | 新增 `CONV_FP32_INF`、`CONV_AMP_INF`                         |
| 2    | `src/graph/op_kind.cpp`                      |   ✅    | `compute_op_to_string()` + `format_params()` 补充            |
| 3    | `include/renaissance/core/types.h`           |   ✅    | 新增 `ConvSearchMode` 枚举                                   |
| 4    | `include/renaissance/core/global_registry.h` |   ✅    | 新增 `fixed_conv_search_mode_` + setter/getter               |
| 5    | `src/core/global_registry.cpp`               |   ✅    | 实现 setter/getter（幂等检查）                               |
| 6    | `src/backend/ops/dtensor/conv_op.cpp`        |   ✅    | **重写**：6 个 launch 函数 + CPU kernels + 注册              |
| 7    | `src/backend/ops/dtensor/conv_op_impl.cpp`   |   ✅    | **重写**：所有 cuDNN FE Graph builders（FP32 + AMP）         |
| 8    | `include/renaissance/backend/op_registry.h`  |   ✅    | 声明 `register_op_conv()`（已存在，确认即可）                |
| 9    | `src/backend/op_registry.cpp`                |   ✅    | `require_warmup()` 加 INF；`warmup_single_cudnn_op()` 更新   |
| 10   | `src/backend/op_stream_policy.cpp`           |   ✅    | 修正 BWD 流映射；新增 INF 映射                               |
| 11   | `src/CMakeLists.txt`                         |   ✅    | 确认 `conv_op.cpp`、`conv_op_impl.cpp` 已在源文件列表        |
| 12   | `src/graph/layer_descriptor_registry.cpp`    |   ✅    | `infer_conv_tensors` 加 bn_stats；`build_conv_*` 修正 ComputeOp 选择 |
| 13   | `src/graph/compiler.cpp`                     |   ✅    | Conv BWD 追加 X 输入、in-place dX 绑定                       |
| 14   | `src/graph/arch_plan_*.cpp`                  |  按需  | `kind_name()`、`params_str()`、YAML 序列化等                 |

---

## 六、实施优先级

```
Phase 1: 基础设施（无风险，先合入）
  1. op_kind.h: 加 CONV_FP32_INF, CONV_AMP_INF
  2. op_kind.cpp: 字符串化补充
  3. types.h: 加 ConvSearchMode
  4. global_registry.h/cpp: 加 conv_search_mode
  5. op_stream_policy.cpp: 修正流映射
  6. op_registry.cpp: require_warmup 加 INF

Phase 2: FP32 完整实现（中风险）
  7. layer_descriptor_registry.cpp: 加 bn_stats，修正 build_conv_*
  8. compiler.cpp: Conv BWD 追加 X 输入
  9. conv_op_impl.cpp: 重写 build_conv_fwd/bwd_graph（带 Graph Cache）
  10. conv_op.cpp: 重写 FP32 FWD/BWD/INF launch + CPU kernels

Phase 3: AMP 实现（高风险，需仔细验证）
  11. conv_op_impl.cpp: 新增 AMP FWD/BWD/INF Graph builders
  12. conv_op.cpp: 新增 AMP FWD/BWD/INF launch
  13. register_op_conv(): 注册全部 6 个算子

Phase 4: EXHAUSTIVE_C 与优化
  14. EXHAUSTIVE_C 搜索集成（仅 AMP FWD/INF）
  15. 性能测试与调优
```

---

## 七、关键风险与注意事项

| 风险                              | 影响                                    | 缓解措施                                                     |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| **dX 覆盖 X 数据竞争**            | 训练结果完全错误                        | BWD 严格 `wgrad(COMP_1) → event → dgrad(COMP_3)`；单流内 cuDNN FE execute 顺序执行 |
| **padded_c vs c 混用**            | cuDNN FE 找不到执行计划（error code 8） | AMP 模式所有 `set_dim` 必须用 `dt.padded_c()`；FP32 用 `dt.c()` |
| **Stride 手写错误**               | 静默数据错乱                            | 强制使用 `dt.n_stride_cuda()` 系列，禁止手算 stride          |
| **BN Stats 指针偏移**             | GenStats 写错位置                       | `sum` → `bn_stats_ptr`；`sq_sum` → `bn_stats_ptr + K * sizeof(float)`；确保 MemoryPlan 分配 `≥ 2*K*sizeof(float)` |
| **Workspace 越界/泄漏**           | 随机崩溃或 OOM                          | 禁止 `cudaMalloc/Free`；统一用 `ctx.ensure_workspace_grow()` |
| **INF 与 FWD Graph 混用**         | 意外触发 GenStats（INF 不应有）         | Cache key 包含 `op` 字段；或 INF 独立查 FWD cache 但构建时不加 GenStats |
| **CPU BWD 性能**                  | CPU 训练极慢                            | 当前 naive 实现仅保证 correctness；生产训练以 GPU 为主       |
| **Layer Descriptor 张量数不一致** | MemoryPlan 错位                         | 确保 `infer_conv_tensors` 返回 7 个；`build_conv_*` output_indices 正确引用 bn_stats |

---

## 八、参考索引

| 功能                          | 参考文件                | 关键函数                                                |
| ----------------------------- | ----------------------- | ------------------------------------------------------- |
| AMP FWD Graph (Conv+GenStats) | `cbr_fwd_fp16.cpp`      | `build_conv_genstats_graph()`                           |
| AMP BWD Graph (WGrad/DGrad)   | `cbr_bwd_fp16.cpp`      | `build_conv_wgrad_graph()` / `build_conv_dgrad_graph()` |
| AMP INF Graph                 | `cbr_inf_fp16.cpp`      | `try_build_single_graph()`                              |
| FE Graph Cache + Workspace    | `fc_op.cpp`             | `FcAmpFwdCache`、`launch_fc_amp_fwd_cuda()`             |
| 多流 BWD 同步                 | `fc_op.cpp`             | `launch_fc_amp_bwd_cuda()`                              |
| FP32 Graph 构建               | `conv_op_impl.cpp`      | `build_conv_fwd_graph()` / `build_conv_bwd_graph()`     |
| 算子注册标准流程              | `ATN2.md`               | 完整 25 项 checklist                                    |
| Experience 查询               | `ta_v4_common_fp16.hpp` | `build_shape_key()`、`match_and_build_plan()`           |



# 【小伙伴D】

# 

> 版本: 1.0.0  
> 日期: 2026-06-03  
> 作者: 技术觉醒团队  
> 状态: 实施方案（编码前最终版）

---

## 一、概述

### 1.1 六大算子

| 算子            | 含义     | 精度       | 后端                            | 关键特性                         |
| --------------- | -------- | ---------- | ------------------------------- | -------------------------------- |
| `CONV_FP32_FWD` | 正向传播 | FP32       | GPU (cuDNN FE) / CPU (XNNPACK)  | 标准conv_fprop，bn_stats占位     |
| `CONV_FP32_BWD` | 反向传播 | FP32       | GPU (cuDNN FE) / CPU (朴素循环) | dW+dX双流，dX覆盖X               |
| `CONV_FP32_INF` | 推理     | FP32       | GPU (cuDNN FE) / CPU (XNNPACK)  | 同FWD，无GenStats                |
| `CONV_AMP_FWD`  | 正向传播 | AMP (FP16) | GPU (cuDNN FE Conv+GenStats)    | GenStats输出sum+sq_sum到bn_stats |
| `CONV_AMP_BWD`  | 反向传播 | AMP (FP16) | GPU (cuDNN FE dgrad+wgrad)      | dW输出FP16，双流同步             |
| `CONV_AMP_INF`  | 推理     | AMP (FP16) | GPU (cuDNN FE Conv only)        | 无GenStats，纯conv_fprop         |

### 1.2 核心原则

1. **参数校验在BluePrint/ArchPlan层**，算子内部不检查合法性。
2. **dX覆盖X**：BWD时梯度输出dX与原始输入X共用同一DTensor，先算dW再算dX。
3. **张量对齐**：AMP与FP32的输入输出张量数量、种类、形状完全一致（bn_stats必有但不一定被写入）。
4. **CUDA Graph捕获**：所有GPU实现均需支持CUDA Graph。
5. **禁止算子内申请内存**：workspace通过`ctx.ensure_workspace_grow()`复用DeviceContext。

---

## 二、参数约束

| 参数        | 允许值     | 检查层级             |
| ----------- | ---------- | -------------------- |
| kernel_size | 1, 3, 5, 7 | BluePrint / ArchPlan |
| stride      | 1, 2       | BluePrint / ArchPlan |
| dilation    | 固定 1     | BluePrint / ArchPlan |
| groups      | 固定 1     | BluePrint / ArchPlan |
| bias        | **不支持** | BluePrint / ArchPlan |
| padding     | 无限制     | —                    |

算子内部不做任何检查，默认参数合格。

---

## 三、现状诊断

### 3.1 已有算子状态

| 算子            | 状态     | 说明                                                         |
| --------------- | -------- | ------------------------------------------------------------ |
| `CONV_FP32_FWD` | 已注册   | `launch_conv_fwd_cuda` + `launch_conv_fwd_cpu`（XNNPACK）    |
| `CONV_FP32_BWD` | 已注册   | `launch_conv_bwd_cuda`（单流单图）+ `launch_conv_bwd_cpu`（朴素） |
| `CONV_FP32_INF` | **缺失** | `ComputeOp`枚举中无此定义                                    |
| `CONV_AMP_FWD`  | 已注册   | 枚举已有，但无launch实现                                     |
| `CONV_AMP_BWD`  | 已注册   | 枚举已有，但无launch实现                                     |
| `CONV_AMP_INF`  | **缺失** | `ComputeOp`枚举中无此定义                                    |

### 3.2 现有代码问题

1. **FP32 BWD单流单图**：当前`launch_conv_bwd_cuda`在COMP_1单流上执行单图（dgrad+wgrad），未按规范拆分为COMP_1(dW)+COMP_3(dX)双流。
2. **Workspace内存泄漏**：当前`launch_conv_fwd_cuda`和`launch_conv_bwd_cuda`在热路径上使用`cudaMalloc`/`cudaFree`，必须改为`ctx.ensure_workspace_grow()`复用。
3. **无per-shape图缓存**：每次launch都重建cuDNN FE Graph，需添加缓存机制。
4. **bn_stats缺失**：`layer_descriptor_registry.cpp`中`infer_conv_tensors`只定义了6张量，缺少第7个bn_stats。
5. **build_conv_forward/backward/inference硬编码AMP**：`n.op`固定为`CONV_AMP_*`，需根据`using_amp()`动态选择。

---

## 四、枚举与注册表变更

### 4.1 ComputeOp枚举扩展（`include/renaissance/graph/op_kind.h`）

当前已有`CONV_FP32_FWD`、`CONV_FP32_BWD`、`CONV_AMP_FWD`、`CONV_AMP_BWD`。需新增两个INF枚举值：

```cpp
// 在 CONV_AMP_BWD 之后插入：
CONV_FP32_INF,   // FP32推理
CONV_AMP_INF,    // AMP推理
```

> 注意：`COUNT`自动递增，`g_compute_op_table`数组大小`kComputeOpCount`自动适配。

### 4.2 ConvSearchMode枚举（`include/renaissance/core/types.h`）

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B,    // 默认：cuDNN启发式Mode B
    EXHAUSTIVE_C    // 穷举式搜索：从include/generated读取预计算最优引擎
};
```

### 4.3 GlobalRegistry扩展（`include/renaissance/core/global_registry.h`）

```cpp
class GlobalRegistry {
public:
    GlobalRegistry& conv_search_mode(ConvSearchMode mode);  // Setter（链式调用）
    ConvSearchMode conv_search_mode() const;                // Getter

private:
    std::atomic<int> fixed_conv_search_mode_{
        static_cast<int>(ConvSearchMode::HEURISTIC_B)
    };
};
```

实现（`src/core/global_registry.cpp`）：

```cpp
GlobalRegistry& GlobalRegistry::conv_search_mode(ConvSearchMode mode) {
    fixed_conv_search_mode_.store(static_cast<int>(mode));
    return *this;
}
ConvSearchMode GlobalRegistry::conv_search_mode() const {
    return static_cast<ConvSearchMode>(fixed_conv_search_mode_.load());
}
```

### 4.4 require_warmup扩展（`src/backend/op_registry.cpp`）

在`require_warmup()`的switch中新增：

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return true;
```

### 4.5 op_stream_policy扩展（`src/backend/op_stream_policy.cpp`）

在`get_op_default_stream()`中新增：

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;  // FWD/INF同流
```

---

## 五、张量布局与内存规划

### 5.1 bn_stats张量新增（`src/graph/layer_descriptor_registry.cpp`）

在`infer_conv_tensors`中当前6张量后新增第7张量：

```cpp
// 6: bn_stats — 所有模式统一分配，仅AMP FWD真正写入
{
    TensorDesc d;
    d.name = "conv_bn_stats";
    d.shape = Shape{1, 1, 1, cp.out_channels};  // {1,1,1,K}，不使用padding
    d.region = Region::T_TEMP_FP32;
    d.dtype = DType::FP32;
    descs.push_back(d);
}
```

**Conv层张量完整清单（共7个）**：

| 索引 | 名称              | 形状             | Region                     | DType  | 说明                        |
| ---- | ----------------- | ---------------- | -------------------------- | ------ | --------------------------- |
| 0    | conv_weight       | `[O, KH, KW, I]` | W_FIRST_CONV / W_DEEP_CONV | FP32   | 主权重                      |
| 1    | conv_output       | `[N, OH, OW, O]` | F_FEATURE_FP32/FP16        | varies | 特征图输出                  |
| 2    | conv_grad_slot    | `[N, H, W, I]`   | F_GRAD_SLOT_FP32/FP16      | varies | 梯度槽（dX覆盖X，in-place） |
| 3    | conv_weight_grad  | `[O, KH, KW, I]` | G_FIRST_CONV / G_DEEP_CONV | FP32   | 权重梯度（FP32区）          |
| 4    | conv_amp_w_fp16   | `[O, KH, KW, I]` | A_FIRST_CONV / A_DEEP_CONV | FP16   | AMP工作权重（!amp时占位）   |
| 5    | conv_amp_g_fp16   | `[O, KH, KW, I]` | G_*_CONV_FP16              | FP16   | AMP权重梯度（!amp时占位）   |
| 6    | **conv_bn_stats** | `[1, 1, 1, O]`   | **T_TEMP_FP32**            | FP32   | BN统计量（仅AMP FWD写入）   |

### 5.2 build_conv_forward/backward/inference修正

根据`using_amp()`动态选择ComputeOp和输入索引：

```cpp
SubgraphPattern build_conv_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
    n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
    n.output_indices = {1};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_conv_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
    n.input_indices  = amp
        ? std::vector<size_t>{4, 1, 2}   // amp_w_fp16, output, grad_slot
        : std::vector<size_t>{0, 1, 2};  // weight, output, grad_slot
    n.output_indices = amp
        ? std::vector<size_t>{2, 5}      // dX(in-place), dW→G_*_CONV_FP16
        : std::vector<size_t>{2, 3};     // dX(in-place), dW→G_*_CONV
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_conv_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
    n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
    n.output_indices = {1};
    p.nodes.push_back(n);
    return p;
}
```

### 5.3 NHWC布局与cuDNN Frontend stride配置

框架张量均为NHWC物理布局。cuDNN FE的`set_stride`顺序为{N, C, H, W}，通过`stride[1]=1`表示C维度连续（即NHWC）。

**Stride计算规范**：

```cpp
// 特征图（NHWC）：dim={N, C, H, W}
auto make_nhwc_stride = [](const DTensor& dt) {
    return std::vector<int64_t>{
        dt.n_stride_cuda(),   // N步幅
        dt.c_stride_cuda(),   // C步幅（应为1）
        dt.h_stride_cuda(),   // H步幅
        dt.w_stride_cuda()    // W步幅
    };
};

// 权重（KRSC→cuDNN dim={K, C, R, S}）
auto make_krsc_stride = [](const DTensor& dt) {
    return std::vector<int64_t>{
        dt.n_stride_cuda(),   // K步幅
        dt.c_stride_cuda(),   // C步幅
        dt.h_stride_cuda(),   // R步幅
        dt.w_stride_cuda()    // S步幅
    };
};
```

**AMP模式下padded_c**：cuDNN FE的`set_dim`必须使用`dt.padded_c()`（对齐到8的倍数），stride使用`dt.c_stride_cuda()`等真实值。参考MaxPool AMP实现。

---

## 六、流分配与跨流同步

### 6.1 流分配表

| 算子          | 操作             | StreamKind |
| ------------- | ---------------- | ---------- |
| CONV_FP32_FWD | fprop            | COMP_1     |
| CONV_AMP_FWD  | fprop + GenStats | COMP_1     |
| CONV_FP32_INF | fprop            | COMP_1     |
| CONV_AMP_INF  | fprop            | COMP_1     |
| CONV_FP32_BWD | dW (wgrad)       | **COMP_1** |
| CONV_FP32_BWD | dX (dgrad)       | **COMP_3** |
| CONV_AMP_BWD  | dW (wgrad)       | **COMP_1** |
| CONV_AMP_BWD  | dX (dgrad)       | **COMP_3** |

**关键说明**：

- BWD的dX使用**COMP_3**（非COMP_2），COMP_2留给BN算子。
- 与FC算子的流分配策略完全对齐。
- dX启动前必须显式等待dW完成：`cudaStreamWaitEvent(s_dx, dW_event, 0)`。

### 6.2 BWD跨流同步代码模板

```cpp
// 两条流获取
cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);
cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);
cudnnHandle_t h_dw = ctx.cudnn_handle(StreamKind::COMP_1);
cudnnHandle_t h_dx = ctx.cudnn_handle(StreamKind::COMP_3);

int i_dw = state.get_or_register(s_dw);
int i_dx = state.get_or_register(s_dx);

// 等待前序算子完成
int out_idx = state.output_stream_idx;
if (out_idx >= 0) {
    cudaStream_t prev_s = state.streams[out_idx].stream;
    if (prev_s != s_dw) {
        cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
    }
    if (prev_s != s_dx) {
        cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
    }
}

// COMP_1: dW (wgrad)
wgrad_graph->execute(h_dw, vp_wgrad, ws_wgrad);
cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
state.streams[i_dw].has_pending_work = true;

// COMP_3: dX (dgrad) — 等待dW完成，保护X不被污染
cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
dgrad_graph->execute(h_dx, vp_dgrad, ws_dgrad);
cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
state.streams[i_dx].has_pending_work = true;

// dX是最后完成的，设为其输出流
state.output_stream_idx = i_dx;
```

**为什么拆分成两个独立Graph**：cuDNN FE Graph的`execute()`绑定到单个`cudnnHandle_t`（即单个stream），无法将wgrad和dgrad分配到不同stream。因此BWD必须拆成两个独立Graph，分别在不同stream上执行。

---

## 七、各算子详细实现

### 7.1 通用设计模式：Per-Shape Graph Cache

参考FC算子的`FcAmpFwdCache`机制，为Conv实现per-shape缓存：

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;    // cudnnHandle指针
    int32_t N, H, W, C;      // 输入形状
    int32_t K, R, S;         // 权重形状
    int32_t stride, pad;     // 卷积参数
    bool is_amp;             // AMP vs FP32
    ComputeOp op;            // 区分FWD/BWD/INF/AMP变体
    bool operator==(const ConvGraphCacheKey& o) const { ... }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_uid;
    size_t workspace_size;
};

// 静态缓存map
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher> s_conv_caches;
```

**缓存命中时**：

1. 直接复用已构建的Graph对象
2. 重新填充variant pack（指针从DeviceContext获取）
3. 复用已分配的workspace（通过`ctx.ensure_workspace_grow`）

**Workspace管理**（禁止算子内`cudaMalloc`/`cudaFree`）：

```cpp
size_t ws_bytes = cache.workspace_size;
void* workspace = ctx.ensure_workspace_grow(stream_kind, ws_bytes);
```

### 7.2 CONV_FP32_FWD / CONV_FP32_INF

**cuDNN操作**：conv_fprop only（无GenStats，FP32不支持GenStats）

**输入**：

- `input_ids[0]` — conv_weight (FP32, W_FIRST_CONV / W_DEEP_CONV)
- `input_ids[6]` — bn_stats (FP32, T_TEMP_FP32，占位，不写入)

**输出**：

- `output_ids[1]` — output (FP32, F_FEATURE_FP32)

**Graph构建**：

```cpp
graph->set_io_data_type(FLOAT).set_compute_data_type(FLOAT);

auto X = graph->tensor(...)
    ->set_dim({N, C, H, W})
    ->set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                  dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
    ->set_data_type(FLOAT);

auto W = graph->tensor(...)
    ->set_dim({K, C, R, S})
    ->set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(),
                  dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
    ->set_data_type(FLOAT);

auto conv_options = Conv_fprop_attributes()
    .set_padding({pad_h, pad_w})
    .set_stride({stride_h, stride_w})
    .set_dilation({1, 1});

auto Y = graph->conv_fprop(X, W, conv_options);
Y->set_output(true)->set_data_type(FLOAT);
// bn_stats不参与graph，仅占位对齐
```

**注意事项**：

- INF与FWD在cuDNN层面完全等价（都是conv_fprop），可共用缓存（通过`op`字段区分）。
- 使用`c()`而非`padded_c()`（FP32不需要padding）。
- 不需要EXHAUSTIVE_C查找（仅AMP支持）。
- 单流COMP_1。

**CPU实现**：XNNPACK（已有`launch_conv_fwd_cpu_xnnpack`），INF同FWD。

### 7.3 CONV_FP32_BWD

**cuDNN操作**：Conv_wgrad (dW) + Conv_dgrad (dX)，两个独立Graph双流执行

**输入**：

- `input_ids[0]` — conv_weight (FP32)
- `input_ids[1]` — output (FP32, =dY)
- `input_ids[2]` — grad_slot (FP32, =X)

**输出**：

- `output_ids[2]` — dX (FP32, in-place覆盖grad_slot)
- `output_ids[3]` — dW (FP32, G_FIRST_CONV / G_DEEP_CONV)

**WGrad Graph（COMP_1）**：

```cpp
wgrad_graph->set_io_data_type(FLOAT).set_compute_data_type(FLOAT);

auto dY = wgrad_graph->tensor(...)->set_data_type(FLOAT);
auto X  = wgrad_graph->tensor(...)->set_data_type(FLOAT);

auto dW = wgrad_graph->conv_wgrad(dY, X, wgrad_options);
dW->set_output(true)
    ->set_dim({K, R, S, C})  // 输出KRSC
    ->set_data_type(FLOAT);
```

**DGrad Graph（COMP_3）**：

```cpp
dgrad_graph->set_io_data_type(FLOAT).set_compute_data_type(FLOAT);

auto dY = dgrad_graph->tensor(...)->set_data_type(FLOAT);
auto W  = dgrad_graph->tensor(...)->set_data_type(FLOAT);

auto dX = dgrad_graph->conv_dgrad(dY, W, dgrad_options);
dX->set_output(true)
    ->set_dim({N, C, H, W})
    ->set_stride({...})  // 与X完全一致（dX覆盖X）
    ->set_data_type(FLOAT);
```

**关键点**：

- dX的stride必须与X完全一致（使用X的`padded_c`）。
- FP32 BWD不需要EXHAUSTIVE_C。
- 两个独立Graph，分别缓存。

**CPU实现**：朴素循环（已有`launch_conv_bwd_cpu`），先dW后dX，单线程顺序执行天然满足。

### 7.4 CONV_AMP_FWD

**cuDNN操作**：Conv_fprop + GenStats（融合）

**参考**：`cbr_fwd_fp16.cpp`的`build_conv_genstats_graph()`

**输入**：

- `input_ids[4]` — amp_weight_fp16 (FP16, A_FIRST_CONV / A_DEEP_CONV)
- `input_ids[6]` — bn_stats (FP32, T_TEMP_FP32，写入目标)

**输出**：

- `output_ids[1]` — output (FP16, F_FEATURE_FP16)

**Graph构建**：

```cpp
graph->set_io_data_type(HALF)
     ->set_intermediate_data_type(FLOAT)
     ->set_compute_data_type(FLOAT);

auto X = graph->tensor(...)
    ->set_dim({N, dt_x.padded_c(), H, W})  // AMP: 使用padded_c()
    ->set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                  dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
    ->set_data_type(HALF);

auto W = graph->tensor(...)
    ->set_dim({K, dt_w.padded_c(), R, S})
    ->set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(),
                  dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
    ->set_data_type(HALF);

auto conv_out = graph->conv_fprop(X, W, conv_options);

// GenStats
auto genstats_attrs = Genstats_attributes()
    .set_name("genstats")
    .set_compute_data_type(FLOAT);
auto [sum, sq_sum] = graph->genstats(conv_out, genstats_attrs);

// Y输出
conv_out->set_output(true)
    ->set_dim({N, K, OH, OW})
    ->set_stride({OH*OW*K_aligned, 1, OW*K_aligned, K_aligned})
    ->set_data_type(HALF);

// sum/sq_sum输出到bn_stats
sum->set_output(true)
    ->set_dim({1, K, 1, 1})
    ->set_stride({K, 1, K, K})
    ->set_data_type(FLOAT);
sq_sum->set_output(true)
    ->set_dim({1, K, 1, 1})
    ->set_stride({K, 1, K, K})
    ->set_data_type(FLOAT);
```

**bn_stats内存Mapping**：在Variant Pack中，`sum`指向`bn_stats_ptr`，`sq_sum`指向`bn_stats_ptr + K * sizeof(float)`。

**EXHAUSTIVE_C**：若模式为`EXHAUSTIVE_C`且平台为A100/RTX5090，生成shape key查表，未找到则fallback到Heuristic B。

**单流COMP_1**。

### 7.5 CONV_AMP_BWD

**最复杂算子**，需跨流同步和dX覆盖X保护。

**cuDNN操作**：Conv_wgrad (dW) + Conv_dgrad (dX)，两个独立Graph双流执行

**输入**：

- `input_ids[4]` — amp_weight_fp16 (FP16)
- `input_ids[1]` — output (FP16, =dY)
- `input_ids[2]` — grad_slot (FP16, =X)

**输出**：

- `output_ids[2]` — dX (FP16, in-place覆盖grad_slot)
- `output_ids[5]` — dW (FP16, G_*_CONV_FP16)

**WGrad Graph（COMP_1）**：

```cpp
wgrad_graph->set_io_data_type(HALF)
           ->set_compute_data_type(FLOAT);

auto dY = wgrad_graph->tensor(...)->set_data_type(HALF);
auto X  = wgrad_graph->tensor(...)->set_data_type(HALF);

auto dW = wgrad_graph->conv_wgrad(dY, X, wgrad_options);
dW->set_output(true)
    ->set_dim({K, R, S, C})
    ->set_stride({dt_dw.n_stride_cuda(), dt_dw.c_stride_cuda(),
                  dt_dw.h_stride_cuda(), dt_dw.w_stride_cuda()})
    ->set_data_type(HALF);  // ⚠️ AMP BWD输出FP16 dW
```

**DGrad Graph（COMP_3）**：

```cpp
dgrad_graph->set_io_data_type(HALF)
           ->set_compute_data_type(FLOAT);

auto dY = dgrad_graph->tensor(...)->set_data_type(HALF);
auto W  = dgrad_graph->tensor(...)->set_data_type(HALF);

auto dX = dgrad_graph->conv_dgrad(dY, W, dgrad_options);
dX->set_output(true)
    ->set_dim({N, dt_dx.padded_c(), H, W})
    ->set_stride({dt_dx.n_stride_cuda(), dt_dx.c_stride_cuda(),
                  dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda()})
    ->set_data_type(HALF);
```

**关键点**：

- dW输出FP16到`G_*_CONV_FP16`区（不同于FC的FP32 dW），后续由compiler插入`RANGE_CAST_FP16_TO_FP32`批量转换。
- dX的stride必须与X完全一致（使用X的`padded_c`）。
- AMP使用`padded_c()`作为`set_dim`参数。
- dgrad_graph的输入X（grad_slot）也需要用`padded_c`作为set_dim，但stride使用实际stride。
- EXHAUSTIVE_C分别对wgrad和dgrad查询。

### 7.6 CONV_AMP_INF

**cuDNN操作**：conv_fprop only（无GenStats）

**输入**：

- `input_ids[4]` — amp_weight_fp16 (FP16)
- `input_ids[6]` — bn_stats (FP32, T_TEMP_FP32，占位，不写入)

**输出**：

- `output_ids[1]` — output (FP16, F_FEATURE_FP16)

**Graph构建**：

```cpp
graph->set_io_data_type(HALF)
     ->set_intermediate_data_type(HALF)
     ->set_compute_data_type(FLOAT);

auto X = graph->tensor(...)->set_data_type(HALF);
auto W = graph->tensor(...)->set_data_type(HALF);
auto Y = graph->conv_fprop(X, W, conv_options);
Y->set_output(true)->set_data_type(HALF);
// 无GenStats，bn_stats仅占位
```

**注意事项**：

- 纯conv_fprop，无GenStats。
- AMP使用`padded_c()`。
- 单流COMP_1。
- EXHAUSTIVE_C key使用`"conv_fprop"`。

---

## 八、EXHAUSTIVE_C穷举式搜索

### 8.1 平台支持

| 平台           | EXHAUSTIVE_C | 回退        |
| -------------- | ------------ | ----------- |
| A100-SXM4-80GB | 支持         | Heuristic B |
| RTX 5090       | 支持         | Heuristic B |
| 其他           | 不支持       | Heuristic B |

### 8.2 集成流程

```cpp
ConvSearchMode mode = GlobalRegistry::instance().conv_search_mode();

if (mode == ConvSearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
    std::string key = build_shape_key(op_name, "fp16",
        N, H, W, C, K, R, S, stride, pad);
    auto exp_rec = ta_v4::experience::lookup(key);
    if (exp_rec != nullptr) {
        graph->create_execution_plans({HeurMode_t::A, HeurMode_t::B});
        // 匹配预计算engine plan
        auto [status, matched] = match_and_build_plan(graph, candidates, exp_rec, handle);
        if (!matched) goto fallback;
    } else { goto fallback; }
#else
    goto fallback;
#endif
} else {
fallback:
    graph->create_execution_plans({heur_mode});
    graph->check_support(handle);
    graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE);
}
```

### 8.3 Shape Key格式

各算子使用的key前缀：

- FWD: `"conv_genstats"`
- BWD wgrad: `"conv_wgrad"`
- BWD dgrad: `"conv_dgrad"`
- INF: `"conv_fprop"`

---

## 九、CPU实现

### 9.1 FWD / INF

沿用现有XNNPACK实现（`launch_conv_fwd_cpu_xnnpack`），NHWC格式，KRSC权重。INF与FWD调用同一函数。

### 9.2 BWD

沿用现有朴素循环实现（`launch_conv_bwd_cpu`），先dW后dX，单线程顺序执行。后续可优化为im2col+GEMM或OpenMP并行化。

### 9.3 AMP

CPU不支持AMP，`launch_cpu = launch_conv_amp_cpu_not_supported`，直接抛异常。

---

## 十、CUDA Graph捕获兼容性

### 10.1 捕获策略

- **Warmup阶段**：首次调用构建Graph并缓存，直接执行（非Graph模式）。
- **Capture阶段**：框架的`CaptureBuilder`在capture stream上重新执行算子。算子内部调用`graph->execute()`被CUDA Graph自动捕获。
- **Replay阶段**：`cudaGraphLaunch`替换执行路径。

### 10.2 多流捕获（BWD）

BWD涉及COMP_1和COMP_3两条流：

- 两条流分别`BeginCapture`。
- 跨流event同步（`cudaStreamWaitEvent`）在捕获时表现为依赖边。
- 分别`EndCapture`，得到两个`cudaGraphExec_t`。

### 10.3 bn_stats同步（仅AMP FWD）

GenStats写入bn_stats后，capture阶段需确保统计量对后续BN可见。参考`cbr_fwd_fp16.cpp`中的处理方式。

---

## 十一、权重初始化

权重初始化由`Initializer::derive(Region)`和`Initializer::apply_to_tensor()`统一处理，算子层无需改动：

| 区域                                     | 初始化方式   |
| ---------------------------------------- | ------------ |
| `W_FIRST_CONV` / `W_DEEP_CONV`           | TRUNC_NORMAL |
| `A_FIRST_CONV` / `A_DEEP_CONV`           | TRUNC_NORMAL |
| `G_FIRST_CONV` / `G_DEEP_CONV`           | ZEROS        |
| `G_FIRST_CONV_FP16` / `G_DEEP_CONV_FP16` | ZEROS        |

---

## 十二、算子I/O速查表

### 正向 (FWD)

| 算子          | 输入 (input_ids)              | 输出 (output_ids) |
| ------------- | ----------------------------- | ----------------- |
| CONV_FP32_FWD | [0]=fp32_weight, [6]=bn_stats | [1]=output        |
| CONV_AMP_FWD  | [4]=fp16_weight, [6]=bn_stats | [1]=output        |

### 反向 (BWD)

| 算子          | 输入 (input_ids)                               | 输出 (output_ids)              |
| ------------- | ---------------------------------------------- | ------------------------------ |
| CONV_FP32_BWD | [0]=fp32_weight, [1]=output, [2]=grad_slot(=X) | [2]=dX(in-place), [3]=dW(fp32) |
| CONV_AMP_BWD  | [4]=fp16_weight, [1]=output, [2]=grad_slot(=X) | [2]=dX(in-place), [5]=dW(fp16) |

### 推理 (INF)

| 算子          | 输入 (input_ids)              | 输出 (output_ids) |
| ------------- | ----------------------------- | ----------------- |
| CONV_FP32_INF | [0]=fp32_weight, [6]=bn_stats | [1]=output        |
| CONV_AMP_INF  | [4]=fp16_weight, [6]=bn_stats | [1]=output        |

> 注：FWD和INF的特征图输入（X）不在descs的input_indices中，由Compiler Phase 4通过跨层链注入。BWD的dY也不在列表中，由Compiler注入为input_ids[0]。

---

## 十三、文件变更清单

| 文件                                         | 变更类型 | 说明                                                         |
| -------------------------------------------- | -------- | ------------------------------------------------------------ |
| `include/renaissance/graph/op_kind.h`        | 修改     | 新增`CONV_FP32_INF`、`CONV_AMP_INF`枚举值                    |
| `include/renaissance/core/types.h`           | 修改     | 新增`ConvSearchMode`枚举                                     |
| `include/renaissance/core/global_registry.h` | 修改     | 新增`fixed_conv_search_mode_`成员和`conv_search_mode()`接口  |
| `src/core/global_registry.cpp`               | 修改     | 实现`conv_search_mode()` setter/getter                       |
| `src/graph/layer_descriptor_registry.cpp`    | 修改     | `infer_conv_tensors`新增bn_stats（第7张量）；`build_conv_*`修正ComputeOp选择 |
| `src/backend/op_registry.cpp`                | 修改     | `require_warmup()`新增INF；`register_default_ops()`已有`register_op_conv()` |
| `src/backend/op_stream_policy.cpp`           | 修改     | 新增INF的stream策略                                          |
| `src/backend/ops/dtensor/conv_op.cpp`        | **重写** | 6个launch函数（CUDA+CPU）、per-shape缓存、注册入口           |
| `src/backend/ops/dtensor/conv_op_impl.cpp`   | **重写** | cuDNN FE Graph构建函数（FP32+AMP全部6个图）                  |

---

## 十四、实施顺序

按以下顺序实现，每一步都可编译验证：

### Phase 1: 基础设施（无风险）

1. `op_kind.h`：新增`CONV_FP32_INF`、`CONV_AMP_INF`
2. `types.h`：新增`ConvSearchMode`枚举
3. `global_registry.h/cpp`：新增`conv_search_mode` setter/getter
4. `op_registry.cpp`：`require_warmup()`扩展
5. `op_stream_policy.cpp`：新增INF stream策略
6. `layer_descriptor_registry.cpp`：新增bn_stats张量 + 修正`build_conv_*`

### Phase 2: FP32 INF（低风险）

7. 实现`launch_conv_fp32_inf_cuda`（复用FWD的Graph构建函数）
8. CPU INF 共用`launch_conv_fwd_cpu`
9. 注册到`g_compute_op_table`

### Phase 3: FP32 BWD双流改造（中风险）

10. 将现有单图BWD拆分为`wgrad_graph`(COMP_1) + `dgrad_graph`(COMP_3)
11. 添加跨流同步（`cudaStreamWaitEvent`）
12. 添加per-shape缓存 + workspace复用
13. 验证dX覆盖X正确性

### Phase 4: FP32 FWD改造（低风险）

14. 修复workspace管理（`cudaMalloc`/`cudaFree` → `ensure_workspace_grow`）
15. 添加per-shape缓存

### Phase 5: AMP FWD（中风险）

16. 实现`build_conv_amp_fwd_graph`（Conv+GenStats，HALF类型）
17. 实现`launch_conv_amp_fwd_cuda`
18. bn_stats的Variant Pack映射
19. EXHAUSTIVE_C分支

### Phase 6: AMP INF（低风险）

20. 纯`conv_fprop` HALF，无GenStats
21. 共用AMP FWD大部分逻辑

### Phase 7: AMP BWD（高风险）

22. 实现`build_conv_amp_wgrad_graph` + `build_conv_amp_dgrad_graph`
23. 严格流同步：wgrad(COMP_1) → event → dgrad(COMP_3)
24. dW输出FP16（HALF）
25. dX覆盖X，确保等待barrier
26. EXHAUSTIVE_C分别对wgrad/dgrad查询

### Phase 8: 统一注册与验证

27. 完善`register_op_conv()`，注册全部6个算子
28. 编译并通过现有测试（确保FP32 FWD/BWD不被破坏）

---

## 十五、风险与注意事项

| 风险点                             | 缓解措施                                                     |
| ---------------------------------- | ------------------------------------------------------------ |
| **dX覆盖X的数据竞争**              | BWD必须严格执行`wgrad → event → dgrad`顺序；dgrad在COMP_3上等待wgrad的event |
| **cuDNN FE Graph数据类型不匹配**   | AMP所有tensor必须显式`set_data_type(HALF)`，compute type用FLOAT |
| **Stride设置错误导致静默数据错乱** | 严格使用DTensor的`n_stride_cuda()`等真实值；不可手写stride计算 |
| **Workspace内存泄漏**              | 通过`ctx.ensure_workspace_grow()`复用DeviceContext workspace；禁止`cudaMalloc/Free` |
| **AMP padded_c遗漏**               | cuDNN FE的`set_dim`必须用`dt.padded_c()`；参考MaxPool AMP实现 |
| **Experience Key不匹配**           | `build_shape_key`的字段顺序、数值必须与`include/generated/`中的表完全一致 |
| **CPU BWD性能过低**                | 当前为Naive六重循环，后续可研究im2col+GEMM或Eigen优化，不在本次范围内 |
| **BN Stats内存对齐**               | `T_TEMP_FP32`区的bn_stats不使用padding，大小严格`2*K*sizeof(float)`；Variant Pack中sum/sq_sum分别指向首地址和偏移`K`处 |
| **bn_stats对齐一致性**             | 即使FP32不写bn_stats，也必须持有这个张量，否则AMP/FP32切换时memory plan错位 |

---

## 十六、参考代码索引

| 功能                          | 参考文件                | 关键函数/类                                             |
| ----------------------------- | ----------------------- | ------------------------------------------------------- |
| AMP FWD Graph (Conv+GenStats) | `cbr_fwd_fp16.cpp`      | `build_conv_genstats_graph()`                           |
| AMP BWD Graph (WGrad/DGrad)   | `cbr_bwd_fp16.cpp`      | `build_conv_wgrad_graph()` / `build_conv_dgrad_graph()` |
| AMP INF Graph                 | `cbr_inf_fp16.cpp`      | `try_build_single_graph()`                              |
| FE 数据类型与stride           | `fc_op.cpp`             | `build_fc_amp_fwd_conv_graph()`                         |
| 双流同步 (dW→dX)              | `fc_op.cpp`             | `launch_fc_amp_bwd_cuda()`                              |
| FP32 Graph 构建               | `conv_op_impl.cpp`      | `build_conv_fwd_graph()` / `build_conv_bwd_graph()`     |
| CPU XNNPACK FWD               | `conv_op.cpp`           | `launch_conv_fwd_cpu_xnnpack()`                         |
| CPU Naive BWD                 | `conv_op.cpp`           | `launch_conv_bwd_cpu()`                                 |
| AMP padded_c使用              | `maxpool_op.cpp`        | MaxPool AMP FWD/INF实现                                 |
| Experience查询                | `ta_v4_common_fp16.hpp` | `build_shape_key()`, `match_and_build_plan()`           |
| 算子注册                      | `fc_op.cpp`             | `register_op_fc()`                                      |
| Per-shape缓存                 | `fc_op.cpp`             | `FcAmpFwdCache`                                         |
| 添加新算子流程                | `ATN2.md`               | 全流程checklist                                         |

---

*方案制定完成。经充分调研ATN文档与现有代码，本方案与框架现有基础设施（DTensor对齐、Region体系、MultiStreamCaptureState、GlobalRegistry fixed变量体系、Initializer初始化链）完全兼容，可直接进入编码阶段。*



