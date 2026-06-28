/**
 * @file conv_op_impl.cpp
 * @brief CONV算子的cuDNN FE Graph构建实现（FP32 + AMP，6个算子变体）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cudnn_frontend.h, cudnn_utils.h, distributed_tensor.h
 * @note 所属系列: backend/ops/dtensor
 * @note 本文件包含7个graph builder函数，供conv_op.cpp中的launch函数调用
 */

#ifdef TR_USE_CUDA

#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>

#include "renaissance/backend/cudnn_utils.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/op_kind.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>

namespace tr {

// ============================================================================
// 通用辅助函数：stride映射
// ============================================================================

/**
 * @brief NHWC stride 映射（特征图）
 *
 * dim={N, C, H, W}，stride 通过 DTensor 真实值隐式表达 NHWC 物理布局
 */
inline std::vector<int64_t> make_nhwc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

/**
 * @brief KRSC stride 映射（权重）
 *
 * dim={K, C, R, S}，stride 通过 DTensor 真实值隐式表达 KRSC 物理布局。
 * 实现与 make_nhwc_stride 完全一致，因为 DTensor 的 stride 方法已基于
 * padded_c 隐式表达了 KRSC 布局。两个函数等价但语义不同，保留以区分特征图和权重。
 */
inline std::vector<int64_t> make_krsc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

// ============================================================================
// Per-Shape Graph Cache 结构
// ============================================================================

struct ConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C, K, R, S;
    int32_t  pad_h, pad_w;
    int32_t  stride_h, stride_w;
    bool     is_amp;
    ComputeOp op;

    bool operator==(const ConvGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W &&
               C == o.C && K == o.K && R == o.R && S == o.S &&
               pad_h == o.pad_h && pad_w == o.pad_w &&
               stride_h == o.stride_h && stride_w == o.stride_w &&
               is_amp == o.is_amp && op == o.op;
    }
};

struct ConvGraphCacheKeyHasher {
    size_t operator()(const ConvGraphCacheKey& k) const {
        size_t h = std::hash<uint64_t>()(k.handle_bits);
        h ^= std::hash<int32_t>()(k.N)  << 1;
        h ^= std::hash<int32_t>()(k.H)  << 2;
        h ^= std::hash<int32_t>()(k.W)  << 3;
        h ^= std::hash<int32_t>()(k.C)  << 4;
        h ^= std::hash<int32_t>()(k.K)  << 5;
        h ^= std::hash<int32_t>()(k.R)  << 6;
        h ^= std::hash<int32_t>()(k.S)  << 7;
        h ^= std::hash<int32_t>()(k.pad_h) << 8;
        h ^= std::hash<int32_t>()(k.pad_w) << 9;
        h ^= std::hash<int32_t>()(k.stride_h) << 10;
        h ^= std::hash<int32_t>()(k.stride_w) << 11;
        h ^= std::hash<bool>()(k.is_amp) << 12;
        h ^= std::hash<int>()(static_cast<int>(k.op)) << 13;
        return h;
    }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

// 静态缓存：按算子类型分独立 map
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_fwd_cache;     // FWD + INF 共用
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_wgrad_cache;   // BWD wgrad
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_dgrad_cache;   // BWD dgrad

// ============================================================================
// Cache 查找/构建模板
// ============================================================================

template<typename CacheMap>
static ConvGraphCache& get_or_build_cache(
    CacheMap& cache_map,
    const ConvGraphCacheKey& key,
    std::function<ConvGraphCache()> builder)
{
    auto it = cache_map.find(key);
    if (it != cache_map.end()) return it->second;
    auto [inserted_it, _] = cache_map.emplace(key, builder());
    return inserted_it->second;
}

/**
 * @brief 根据当前 capture variant 的实际 DTensor，更新共享 cache 中的 tensor_to_id 映射。
 *
 * 背景：A/B 双缓冲共享同一个 cache entry 和同一个 fe::graph::Graph 对象，但各自
 * 的 input/output buffer 不同（I_A_DATA vs I_B_DATA）。在 capture 阶段，必须在
 * execute() 之前更新 tensor_to_id，确保 variant pack 中的指针指向正确的 buffer。
 */
static void update_conv_tensor_to_id(
    ConvGraphCache& cache,
    int64_t x_id, int64_t w_id, int64_t y_id,
    int64_t dy_id = -1, int64_t dx_id = -1, int64_t dw_id = -1,
    int64_t sum_id = -1, int64_t sq_sum_id = -1)
{
    for (auto& [ta, tid] : cache.tensor_to_id) {
        const std::string& name = ta->get_name();
        if (name == "X")       tid = x_id;
        else if (name == "W")  tid = w_id;
        else if (name == "Y")  tid = y_id;
        else if (name == "dY") tid = dy_id;
        else if (name == "dX") tid = dx_id;
        else if (name == "dW") tid = dw_id;
        else if (name == "sum")     tid = sum_id;
        else if (name == "sq_sum")  tid = sq_sum_id;
    }
}

// ============================================================================
// FP32 Graph 构建函数
// ============================================================================

/**
 * @brief 构建 CONV_FP32_FWD / CONV_FP32_INF 的 cuDNN FE Graph
 *
 * FP32 精度：dim 使用 dt.c()（逻辑C），stride 使用 DTensor 真实值。
 * INF 与 FWD 在 cuDNN 层面完全等价（都是 conv_fprop），共用此函数和 cache。
 */
ConvGraphCache build_conv_fp32_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP32);

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim(to_fe_dim(dt_x.shape))
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::FLOAT));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim(to_fe_dim(dt_w.shape))
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto Y = graph->conv_fprop(X, W, opts);
    Y->set_output(true)
      .set_name("Y")
      .set_dim(to_fe_dim(dt_y.shape))
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    cache.tensor_to_id[Y] = dt_y.id;
    return cache;
}

/**
 * @brief 构建 CONV_FP32_BWD wgrad 的 cuDNN FE Graph
 *
 * 在 COMP_1 流上执行。dW 输出维度从 dt_dw.shape 推导，stride 使用 make_krsc_stride。
 */
ConvGraphCache build_conv_fp32_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP32);

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim(to_fe_dim(dt_dy.shape))
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::FLOAT));

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim(to_fe_dim(dt_x.shape))
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = Conv_wgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dW = graph->conv_wgrad(dY, X, opts);
    dW->set_output(true)
       .set_name("dW")
       .set_dim(to_fe_dim(dt_dw.shape))
       .set_stride(make_krsc_stride(dt_dw))
       .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[X]  = dt_x.id;
    cache.tensor_to_id[dW] = dt_dw.id;
    return cache;
}

/**
 * @brief 构建 CONV_FP32_BWD dgrad 的 cuDNN FE Graph
 *
 * 在 COMP_3 流上执行。dX 的 stride 必须与 X 完全一致（in-place）。
 */
ConvGraphCache build_conv_fp32_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP32);

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim(to_fe_dim(dt_dy.shape))
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::FLOAT));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim(to_fe_dim(dt_w.shape))
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = Conv_dgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dX = graph->conv_dgrad(dY, W, opts);
    dX->set_output(true)
       .set_name("dX")
       .set_dim(to_fe_dim(dt_dx.shape))
       .set_stride(make_nhwc_stride(dt_dx))
       .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[W]  = dt_w.id;
    cache.tensor_to_id[dX] = dt_dx.id;
    return cache;
}

// ============================================================================
// AMP Graph 构建函数
// ============================================================================

/**
 * @brief 构建 CONV_AMP_FWD 的 cuDNN FE Graph（Conv + GenStats）
 *
 * AMP 精度：dim 使用 dt.padded_c()（对齐后的C），stride 使用 DTensor 真实值。
 * conv_out 直接 set_output(true)，然后 genstats 消费 conv_out。
 * GenStats 输出 sum 和 sq_sum，均为 FP32，形状 {1, K, 1, 1}。
 * sum/sq_sum 分别绑定到独立的 DTensor（index 6 / 7）。
 */
ConvGraphCache build_conv_amp_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const DTensor& dt_sum, const DTensor& dt_sq_sum,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    int64_t PC_x = dt_x.padded_c();
    int64_t PC_w = dt_w.padded_c();
    int64_t K = dt_y.c();

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), PC_x, dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), PC_w, dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto conv_out = graph->conv_fprop(X, W, opts);

    // conv_out 设为 output 后仍可作为 genstats 的输入
    conv_out->set_output(true)
             .set_name("Y")
             .set_dim(to_fe_dim(dt_y.shape))
             .set_stride(make_nhwc_stride(dt_y))
             .set_data_type(fe::DataType_t::HALF);

    // GenStats：保持原有行为，只改输出绑定方式
    auto genstats_opts = Genstats_attributes()
        .set_name("genstats")
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto genstats_outputs = graph->genstats(conv_out, genstats_opts);
    auto sum    = genstats_outputs[0];
    auto sq_sum = genstats_outputs[1];

    sum->set_output(true)
        .set_name("sum")
        .set_dim({1, K, 1, 1})
        .set_stride({K, 1, K, K})
        .set_data_type(fe::DataType_t::FLOAT);

    sq_sum->set_output(true)
          .set_name("sq_sum")
          .set_dim({1, K, 1, 1})
          .set_stride({K, 1, K, K})
          .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X]        = dt_x.id;
    cache.tensor_to_id[W]        = dt_w.id;
    cache.tensor_to_id[conv_out] = dt_y.id;
    cache.tensor_to_id[sum]      = dt_sum.id;
    cache.tensor_to_id[sq_sum]   = dt_sq_sum.id;
    return cache;
}

/**
 * @brief 构建 CONV_AMP_INF 的 cuDNN FE Graph（纯 conv_fprop，无 GenStats）
 *
 * 与 AMP FWD 的区别：不包含 GenStats 节点，仅输出 Y。
 */
ConvGraphCache build_conv_amp_inf_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.padded_c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.padded_c(), dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto Y = graph->conv_fprop(X, W, opts);
    Y->set_output(true)
      .set_name("Y")
      .set_dim(to_fe_dim(dt_y.shape))
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    cache.tensor_to_id[Y] = dt_y.id;
    return cache;
}

/**
 * @brief 构建 CONV_AMP_BWD wgrad 的 cuDNN FE Graph
 *
 * 在 COMP_1 流上执行。dW 输出 FP16 到 G_*_CONV_FP16。
 * dim 使用 dt.padded_c()，stride 使用 DTensor 真实值。
 */
ConvGraphCache build_conv_amp_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({dt_dy.n(), dt_dy.padded_c(), dt_dy.h(), dt_dy.w()})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.padded_c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_wgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dW = graph->conv_wgrad(dY, X, opts);
    dW->set_output(true)
       .set_name("dW")
       .set_dim(to_fe_dim(dt_dw.shape))
       .set_stride(make_krsc_stride(dt_dw))
       .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[X]  = dt_x.id;
    cache.tensor_to_id[dW] = dt_dw.id;
    return cache;
}

/**
 * @brief 构建 CONV_AMP_BWD dgrad 的 cuDNN FE Graph
 *
 * 在 COMP_3 流上执行。dX 的 stride 必须与 X 完全一致（in-place）。
 * dim 使用 dt.padded_c()，stride 使用 DTensor 真实值。
 */
ConvGraphCache build_conv_amp_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({dt_dy.n(), dt_dy.padded_c(), dt_dy.h(), dt_dy.w()})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.padded_c(), dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_dgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dX = graph->conv_dgrad(dY, W, opts);
    dX->set_output(true)
       .set_name("dX")
       .set_dim({dt_dx.n(), dt_dx.padded_c(), dt_dx.h(), dt_dx.w()})
       .set_stride(make_nhwc_stride(dt_dx))
       .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[W]  = dt_w.id;
    cache.tensor_to_id[dX] = dt_dx.id;
    return cache;
}

} // namespace tr

#endif // TR_USE_CUDA