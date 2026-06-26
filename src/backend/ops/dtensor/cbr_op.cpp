/**
 * @file cbr_op.cpp
 * @brief CBR（Conv+BN2D+ReLU）融合算子 AMP 实现
 * @version 1.0.0
 * @date 2026-06-20
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, cudnn_utils.h
 * @note 所属系列: backend/ops/dtensor
 * @note 仅实现 AMP 版本（CBR_AMP_FWD/BWD/BWD_FIRST_LAYER/INF）
 * @note 内部通过复制 Conv/BN/ReLU 核心逻辑实现，确保字节级一致
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/cudnn_utils.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <functional>
#include <sstream>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#endif

// Mode C 经验搜索头文件
#if defined(USING_A100)
#include "generated/cbr_experience_a100_fp16.hpp"
#elif defined(USING_RTX5090)
#include "generated/cbr_experience_rtx5090_fp16.hpp"
#endif

namespace tr {

// ============================================================================
// 外部 kernel 声明
// ============================================================================
#ifdef TR_USE_CUDA

extern cudaError_t launch_relu_amp_inf_kernel(
    const __half* x, __half* y, int64_t n, cudaStream_t stream);

// BN INF kernel（定义在 bn_op.cu）
extern "C" cudaError_t launch_tr_bn_inf_kernel(
    const void* x,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, void* y,
    int N, int C, int H, int W,
    bool is_fp16, cudaStream_t stream);

extern "C" cudaError_t launch_tr_bn_inf_eq_kernel(
    const void* x,
    const float* eq_scale,
    const float* eq_bias,
    void* y,
    int N, int C, int H, int W,
    bool is_fp16,
    cudaStream_t stream);

// ============================================================================
// BN 子操作：Graph Cache 结构（复制自 bn_op.cpp，重命名避免符号冲突）
// ============================================================================

namespace {

inline uint32_t float_to_bits(float f) {
    uint32_t u;
    static_assert(sizeof(f) == sizeof(u), "float size mismatch");
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

struct CBRBNGraphCacheKey {
    uint64_t handle_bits;
    int32_t N, H, W, C;
    bool is_fp16;
    uint32_t eps_bits;
    uint32_t momentum_bits;

    bool operator==(const CBRBNGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W && C == o.C
            && is_fp16 == o.is_fp16 && eps_bits == o.eps_bits && momentum_bits == o.momentum_bits;
    }
};

struct CBRBNGraphCacheKeyHash {
    size_t operator()(const CBRBNGraphCacheKey& k) const noexcept {
        size_t h = std::hash<uint64_t>{}(k.handle_bits);
        h = h * 31 + std::hash<int32_t>{}(k.N);
        h = h * 31 + std::hash<int32_t>{}(k.H);
        h = h * 31 + std::hash<int32_t>{}(k.W);
        h = h * 31 + std::hash<int32_t>{}(k.C);
        h = h * 31 + std::hash<bool>{}(k.is_fp16);
        h = h * 31 + std::hash<uint32_t>{}(k.eps_bits);
        h = h * 31 + std::hash<uint32_t>{}(k.momentum_bits);
        return h;
    }
};

struct CBRBNGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

std::unordered_map<CBRBNGraphCacheKey, CBRBNGraphCache, CBRBNGraphCacheKeyHash> s_cbr_bn_bwd_caches;

static void update_cbr_bn_tensor_to_id(
    CBRBNGraphCache& cache,
    const std::unordered_map<std::string, int64_t>& name_to_id)
{
    for (auto& [ta, tid] : cache.tensor_to_id) {
        const std::string& name = ta->get_name();
        auto it = name_to_id.find(name);
        if (it != name_to_id.end()) {
            tid = it->second;
        }
    }
}

} // anonymous namespace

// ============================================================================
// Conv 子操作：Graph Cache 结构与辅助函数（复制自 conv_op_impl.cpp，重命名）
// ============================================================================

inline std::vector<int64_t> make_nhwc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

inline std::vector<int64_t> make_krsc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

struct CBRConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C, K, R, S;
    int32_t  pad_h, pad_w;
    int32_t  stride_h, stride_w;
    bool     is_amp;
    ComputeOp op;

    bool operator==(const CBRConvGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W &&
               C == o.C && K == o.K && R == o.R && S == o.S &&
               pad_h == o.pad_h && pad_w == o.pad_w &&
               stride_h == o.stride_h && stride_w == o.stride_w &&
               is_amp == o.is_amp && op == o.op;
    }
};

struct CBRConvGraphCacheKeyHasher {
    size_t operator()(const CBRConvGraphCacheKey& k) const {
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

struct CBRConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

static std::unordered_map<CBRConvGraphCacheKey, CBRConvGraphCache, CBRConvGraphCacheKeyHasher>
    s_cbr_conv_fwd_cache;
static std::unordered_map<CBRConvGraphCacheKey, CBRConvGraphCache, CBRConvGraphCacheKeyHasher>
    s_cbr_conv_wgrad_cache;
static std::unordered_map<CBRConvGraphCacheKey, CBRConvGraphCache, CBRConvGraphCacheKeyHasher>
    s_cbr_conv_dgrad_cache;

// ============================================================================
// BNFinalize 子图：Cache 结构与辅助函数
// ============================================================================

struct CBRBNFinalizeCacheKey {
    uint64_t handle_bits;
    int32_t N, H, W, C;
    uint32_t eps_bits;
    uint32_t momentum_bits;

    bool operator==(const CBRBNFinalizeCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W && C == o.C
            && eps_bits == o.eps_bits && momentum_bits == o.momentum_bits;
    }
};

struct CBRBNFinalizeCacheKeyHash {
    size_t operator()(const CBRBNFinalizeCacheKey& k) const noexcept {
        size_t h = std::hash<uint64_t>{}(k.handle_bits);
        h = h * 31 + std::hash<int32_t>{}(k.N);
        h = h * 31 + std::hash<int32_t>{}(k.H);
        h = h * 31 + std::hash<int32_t>{}(k.W);
        h = h * 31 + std::hash<int32_t>{}(k.C);
        h = h * 31 + std::hash<uint32_t>{}(k.eps_bits);
        h = h * 31 + std::hash<uint32_t>{}(k.momentum_bits);
        return h;
    }
};

struct CBRBNFinalizeGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

static std::unordered_map<CBRBNFinalizeCacheKey, CBRBNFinalizeGraphCache, CBRBNFinalizeCacheKeyHash>
    s_cbr_bn_finalize_caches;

// ============================================================================
// BN Apply+ReLU 子图：Cache 结构与辅助函数
// ============================================================================

struct CBRBNReluCacheKey {
    uint64_t handle_bits;
    int32_t N, H, W, C;

    bool operator==(const CBRBNReluCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W && C == o.C;
    }
};

struct CBRBNReluCacheKeyHash {
    size_t operator()(const CBRBNReluCacheKey& k) const noexcept {
        size_t h = std::hash<uint64_t>{}(k.handle_bits);
        h = h * 31 + std::hash<int32_t>{}(k.N);
        h = h * 31 + std::hash<int32_t>{}(k.H);
        h = h * 31 + std::hash<int32_t>{}(k.W);
        h = h * 31 + std::hash<int32_t>{}(k.C);
        return h;
    }
};

struct CBRBNReluGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

static std::unordered_map<CBRBNReluCacheKey, CBRBNReluGraphCache, CBRBNReluCacheKeyHash>
    s_cbr_bn_relu_caches;

// ============================================================================
// CBR_AMP_INF 融合图 Cache（Conv + MUL(eq_scale) + ADD(eq_bias) + ReLU 单图）
// ============================================================================

struct CBRAmpInfFusedGraphCacheKey {
    uint64_t handle_bits;
    int32_t N, H, W, C, K, R, S;
    int32_t pad_h, pad_w, stride_h, stride_w;

    bool operator==(const CBRAmpInfFusedGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W &&
               C == o.C && K == o.K && R == o.R && S == o.S &&
               pad_h == o.pad_h && pad_w == o.pad_w &&
               stride_h == o.stride_h && stride_w == o.stride_w;
    }
};

struct CBRAmpInfFusedGraphCacheKeyHasher {
    size_t operator()(const CBRAmpInfFusedGraphCacheKey& k) const {
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
        return h;
    }
};

struct CBRAmpInfFusedGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    size_t workspace_size = 0;
    std::shared_ptr<fe::graph::Tensor_attributes> input_ta;
    std::shared_ptr<fe::graph::Tensor_attributes> weight_ta;
    std::shared_ptr<fe::graph::Tensor_attributes> eq_scale_ta;
    std::shared_ptr<fe::graph::Tensor_attributes> eq_bias_ta;
    std::shared_ptr<fe::graph::Tensor_attributes> output_ta;
};

static std::unordered_map<CBRAmpInfFusedGraphCacheKey, CBRAmpInfFusedGraphCache,
                          CBRAmpInfFusedGraphCacheKeyHasher>
    s_cbr_amp_inf_fused_cache;

template<typename CacheMap>
static CBRConvGraphCache& get_or_build_cbr_conv_cache(
    CacheMap& cache_map,
    const CBRConvGraphCacheKey& key,
    std::function<CBRConvGraphCache()> builder)
{
    auto it = cache_map.find(key);
    if (it != cache_map.end()) return it->second;
    auto [inserted_it, _] = cache_map.emplace(key, builder());
    return inserted_it->second;
}

static void update_cbr_conv_tensor_to_id(
    CBRConvGraphCache& cache,
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
// Mode C 经验搜索辅助函数
// ============================================================================

#if defined(USING_A100) || defined(USING_RTX5090)
static std::string build_shape_key(
    const std::string& op_type, const std::string& dtype,
    int64_t N, int64_t H, int64_t W, int64_t C, int64_t K,
    int64_t R, int64_t S, int64_t stride, int64_t padding)
{
    std::ostringstream oss;
#if defined(USING_A100)
    oss << "A100-SXM4-80GB";
#elif defined(USING_RTX5090)
    oss << "RTX5090";
#endif
    oss << "|SM80"
        << "|cuDNN9.17.0"
        << "|CUDA13.1"
        << "|" << op_type << "_" << dtype
        << "|N" << N << "|H" << H << "|W" << W
        << "|C" << C << "|K" << K
        << "|R" << R << "|S" << S
        << "|U1|V1"
        << "|P" << padding << "|Q" << padding
        << "|D" << stride << "|E" << stride
        << "|NHWC|FP16|FP32";
    return oss.str();
}

static std::pair<int, bool> match_and_build_plan(
    std::shared_ptr<fe::graph::Graph>& graph,
    const std::vector<int64_t>& candidates,
    const ta_v4::experience::ExperienceRecord* exp_rec,
    cudnnHandle_t handle)
{
    (void)handle;
    // Level 1: Winner
    for (int64_t idx : candidates) {
        std::string tag;
        auto status = graph->get_plan_name_at_index(idx, tag);
        if (status.is_bad()) continue;
        if (tag == std::string(exp_rec->winner_tag)) {
            auto build_status = graph->build_plan_at_index(idx);
            if (!build_status.is_bad()) return {0, true};
        }
    }
    // Level 2: Backup1
    if (strlen(exp_rec->backup1_tag) > 0) {
        for (int64_t idx : candidates) {
            std::string tag;
            auto status = graph->get_plan_name_at_index(idx, tag);
            if (status.is_bad()) continue;
            if (tag == std::string(exp_rec->backup1_tag)) {
                auto build_status = graph->build_plan_at_index(idx);
                if (!build_status.is_bad()) return {0, true};
            }
        }
    }
    // Level 3: Backup2
    if (strlen(exp_rec->backup2_tag) > 0) {
        for (int64_t idx : candidates) {
            std::string tag;
            auto status = graph->get_plan_name_at_index(idx, tag);
            if (status.is_bad()) continue;
            if (tag == std::string(exp_rec->backup2_tag)) {
                auto build_status = graph->build_plan_at_index(idx);
                if (!build_status.is_bad()) return {0, true};
            }
        }
    }
    return {0, false};
}
#endif

// ============================================================================
// BNFinalize 子图：cuDNN FE Graph 构建函数
// ============================================================================

static CBRBNFinalizeGraphCache build_cbr_bn_finalize_graph(
    const DTensor& dt_sum, const DTensor& dt_sq_sum,
    const DTensor& dt_scale, const DTensor& dt_bias,
    const DTensor& dt_prev_mean, const DTensor& dt_prev_var,
    const DTensor& dt_next_mean, const DTensor& dt_next_var,
    const DTensor& dt_saved_mean, const DTensor& dt_saved_inv_var,
    const DTensor& dt_eq_scale_bias,  // 复用 bn_output 缓冲区
    float eps, float momentum, int64_t accum_count,
    cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP32);
    int64_t C = dt_sum.c();

    auto sum = graph->tensor(Tensor_attributes()
        .set_name("sum")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto sq_sum = graph->tensor(Tensor_attributes()
        .set_name("sq_sum")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto scale = graph->tensor(Tensor_attributes()
        .set_name("scale")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto bias = graph->tensor(Tensor_attributes()
        .set_name("bias")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto prev_rm = graph->tensor(Tensor_attributes()
        .set_name("prev_running_mean")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto prev_rv = graph->tensor(Tensor_attributes()
        .set_name("prev_running_var")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto EPS = graph->tensor(fe::graph::Tensor_attributes(eps).set_name("epsilon"));
    auto MOM = graph->tensor(fe::graph::Tensor_attributes(momentum).set_name("momentum"));

    auto bn_finalize_attrs = BN_finalize_attributes()
        .set_name("bn_finalize")
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_previous_running_stats(prev_rm, prev_rv, MOM);

    auto ACC = graph->tensor(fe::graph::Tensor_attributes(accum_count).set_name("accum_count"));

    auto outputs = graph->bn_finalize(sum, sq_sum, scale, bias, EPS, ACC, bn_finalize_attrs);
    // outputs[0]=eq_scale, [1]=eq_bias, [2]=saved_mean, [3]=saved_inv_var,
    // [4]=next_running_mean, [5]=next_running_var

    const char* out_names[] = {"eq_scale", "eq_bias", "saved_mean",
                                "saved_inv_var", "next_rm", "next_rv"};
    for (int i = 0; i < 6; ++i) {
        outputs[i]->set_name(out_names[i])
                   .set_output(true)
                   .set_dim({1, C, 1, 1})
                   .set_stride({C, 1, C, C})
                   .set_data_type(fe::DataType_t::FLOAT);
    }

    TR_CUDNN_FE_CHECK(graph->validate(), "CBR_AMP_FWD bn_finalize validate");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "CBR_AMP_FWD bn_finalize build op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}), "CBR_AMP_FWD bn_finalize create exec plans");
    graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
    TR_CUDNN_FE_CHECK(graph->check_support(handle), "CBR_AMP_FWD bn_finalize check support");
    TR_CUDNN_FE_CHECK(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "CBR_AMP_FWD bn_finalize build plans");

    CBRBNFinalizeGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[sum]      = dt_sum.id;
    cache.tensor_to_id[sq_sum]   = dt_sq_sum.id;
    cache.tensor_to_id[scale]    = dt_scale.id;
    cache.tensor_to_id[bias]     = dt_bias.id;
    cache.tensor_to_id[prev_rm]  = dt_prev_mean.id;
    cache.tensor_to_id[prev_rv]  = dt_prev_var.id;
    cache.tensor_to_id[outputs[0]] = dt_eq_scale_bias.id;
    cache.tensor_to_id[outputs[1]] = dt_eq_scale_bias.id;  // eq_bias 在同一缓冲区
    cache.tensor_to_id[outputs[2]] = dt_saved_mean.id;
    cache.tensor_to_id[outputs[3]] = dt_saved_inv_var.id;
    cache.tensor_to_id[outputs[4]] = dt_next_mean.id;
    cache.tensor_to_id[outputs[5]] = dt_next_var.id;
    return cache;
}

// ============================================================================
// BN Apply+ReLU 子图：cuDNN FE Graph 构建函数
// ============================================================================

static CBRBNReluGraphCache build_cbr_bn_relu_graph(
    const DTensor& dt_x,       // conv_output
    const DTensor& dt_y,       // relu_output
    const DTensor& dt_eq_scale_bias,  // eq_scale+eq_bias 缓冲区
    cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);
    graph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::HALF)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t N = dt_x.n(), C = dt_x.c(), H = dt_x.h(), W = dt_x.w();

    auto x_ta = graph->tensor(Tensor_attributes()
        .set_name("bn_relu_x")
        .set_dim({N, C, H, W})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto eq_scale_ta = graph->tensor(Tensor_attributes()
        .set_name("eq_scale")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    auto eq_bias_ta = graph->tensor(Tensor_attributes()
        .set_name("eq_bias")
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));

    // scaled = X * eq_scale
    auto scaled = graph->pointwise(x_ta, eq_scale_ta,
        Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    scaled->set_data_type(fe::DataType_t::HALF);

    // shifted = scaled + eq_bias
    auto shifted = graph->pointwise(scaled, eq_bias_ta,
        Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    shifted->set_data_type(fe::DataType_t::HALF);

    // relu = ReLU(shifted)
    auto relu = graph->pointwise(shifted,
        Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    relu->set_name("relu_output")
        .set_output(true)
        .set_dim(to_fe_dim(dt_y.shape))
        .set_stride(make_nhwc_stride(dt_y))
        .set_data_type(fe::DataType_t::HALF);

    // mask = shifted > 0 (CMP_GT)
    auto zero = graph->tensor(0.0f);
    auto mask = graph->pointwise(shifted, zero,
        Pointwise_attributes().set_mode(fe::PointwiseMode_t::CMP_GT)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    mask->set_name("relu_mask")
        .set_output(true)
        .set_dim(to_fe_dim(dt_y.shape))
        .set_stride(make_nhwc_stride(dt_y))
        .set_data_type(fe::DataType_t::BOOLEAN);

    TR_CUDNN_FE_CHECK(graph->validate(), "CBR_AMP_FWD bn_relu validate");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "CBR_AMP_FWD bn_relu build op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}), "CBR_AMP_FWD bn_relu create exec plans");
    graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
    TR_CUDNN_FE_CHECK(graph->check_support(handle), "CBR_AMP_FWD bn_relu check support");
    TR_CUDNN_FE_CHECK(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "CBR_AMP_FWD bn_relu build plans");

    CBRBNReluGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[x_ta]         = dt_x.id;
    cache.tensor_to_id[eq_scale_ta]  = dt_eq_scale_bias.id;
    cache.tensor_to_id[eq_bias_ta]   = dt_eq_scale_bias.id;
    cache.tensor_to_id[relu]         = dt_y.id;
    cache.tensor_to_id[mask]         = -1;  // 由 launch 时动态绑定
    return cache;
}

// ============================================================================
// Conv 子操作：cuDNN FE Graph 构建函数（复制自 conv_op_impl.cpp，重命名）
// ============================================================================

CBRConvGraphCache build_cbr_conv_amp_fwd_graph(
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

    conv_out->set_output(true)
             .set_name("Y")
             .set_dim(to_fe_dim(dt_y.shape))
             .set_stride(make_nhwc_stride(dt_y))
             .set_data_type(fe::DataType_t::HALF);

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

    // Conv+GenStats: 先 build_operation_graph，再尝试 Mode C 经验搜索
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "CBR_AMP_FWD conv build op graph");

    bool mode_c_matched = false;
#if defined(USING_A100) || defined(USING_RTX5090)
    {
        std::string key = build_shape_key(
            "conv_genstats", "fp16",
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(), K,
            dt_w.h(), dt_w.w(), cp.stride_h, cp.pad_h);
        auto exp_rec = ta_v4::experience::lookup(key);
        if (exp_rec) {
            TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
                "CBR_AMP_FWD conv create exec plans (mode C)");
            graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
            std::vector<int64_t> candidates;
            int64_t count = graph->get_execution_plan_count();
            for (int64_t i = 0; i < count; ++i) candidates.push_back(i);
            auto [status, matched] = match_and_build_plan(graph, candidates, exp_rec, handle);
            mode_c_matched = matched;
        }
    }
#endif
    if (!mode_c_matched) {
        TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}), "CBR_AMP_FWD conv create exec plans");
        graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
        TR_CUDNN_FE_CHECK(graph->check_support(handle), "CBR_AMP_FWD conv check support");
        TR_CUDNN_FE_CHECK(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "CBR_AMP_FWD conv build plans");
    }

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X]        = dt_x.id;
    cache.tensor_to_id[W]        = dt_w.id;
    cache.tensor_to_id[conv_out] = dt_y.id;
    cache.tensor_to_id[sum]      = dt_sum.id;
    cache.tensor_to_id[sq_sum]   = dt_sq_sum.id;
    return cache;
}

CBRConvGraphCache build_cbr_conv_amp_inf_graph(
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

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    cache.tensor_to_id[Y] = dt_y.id;
    return cache;
}

CBRConvGraphCache build_cbr_conv_amp_wgrad_graph(
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

    TR_CUDNN_FE_CHECK(graph->validate(),              "CBR_AMP_BWD wgrad validate");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "CBR_AMP_BWD wgrad build op graph");

    bool mode_c_matched = false;
#if defined(USING_A100) || defined(USING_RTX5090)
    {
        std::string key = build_shape_key(
            "conv_wgrad", "fp16",
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(), dt_dw.n(),
            dt_dw.h(), dt_dw.w(), cp.stride_h, cp.pad_h);
        auto exp_rec = ta_v4::experience::lookup(key);
        if (exp_rec) {
            TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
                "CBR_AMP_BWD wgrad create exec plans (mode C)");
            graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
            std::vector<int64_t> candidates;
            int64_t count = graph->get_execution_plan_count();
            for (int64_t i = 0; i < count; ++i) candidates.push_back(i);
            auto [status, matched] = match_and_build_plan(
                graph, candidates, exp_rec, handle);
            mode_c_matched = matched;
        }
    }
#endif

    if (!mode_c_matched) {
        // 注意：不能用 finalize_cudnn_graph，因为它会再次调用 create_execution_plans，
        // 而 Mode C 分支已经调用过一次。此处仿照 FWD 路径的显式 fallback 写法。
        TR_CUDNN_FE_CHECK(graph->create_execution_plans(
            {fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}),
            "CBR_AMP_BWD wgrad create exec plans");
        graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
        TR_CUDNN_FE_CHECK(graph->check_support(handle),
            "CBR_AMP_BWD wgrad check support");
        TR_CUDNN_FE_CHECK(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE),
            "CBR_AMP_BWD wgrad build plans");
    }

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[X]  = dt_x.id;
    cache.tensor_to_id[dW] = dt_dw.id;
    return cache;
}

CBRConvGraphCache build_cbr_conv_amp_dgrad_graph(
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

    TR_CUDNN_FE_CHECK(graph->validate(),              "CBR_AMP_BWD dgrad validate");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "CBR_AMP_BWD dgrad build op graph");

    bool mode_c_matched = false;
#if defined(USING_A100) || defined(USING_RTX5090)
    {
        std::string key = build_shape_key(
            "conv_dgrad", "fp16",
            dt_dx.n(), dt_dx.h(), dt_dx.w(), dt_dx.c(), dt_w.n(),
            dt_w.h(), dt_w.w(), cp.stride_h, cp.pad_h);
        auto exp_rec = ta_v4::experience::lookup(key);
        if (exp_rec) {
            TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
                "CBR_AMP_BWD dgrad create exec plans (mode C)");
            graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
            std::vector<int64_t> candidates;
            int64_t count = graph->get_execution_plan_count();
            for (int64_t i = 0; i < count; ++i) candidates.push_back(i);
            auto [status, matched] = match_and_build_plan(
                graph, candidates, exp_rec, handle);
            mode_c_matched = matched;
        }
    }
#endif

    if (!mode_c_matched) {
        TR_CUDNN_FE_CHECK(graph->create_execution_plans(
            {fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}),
            "CBR_AMP_BWD dgrad create exec plans");
        graph->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
        TR_CUDNN_FE_CHECK(graph->check_support(handle),
            "CBR_AMP_BWD dgrad check support");
        TR_CUDNN_FE_CHECK(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE),
            "CBR_AMP_BWD dgrad build plans");
    }

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[W]  = dt_w.id;
    cache.tensor_to_id[dX] = dt_dx.id;
    return cache;
}

// ============================================================================
// CBR_AMP_INF 融合图构建器（Conv + MUL(eq_scale) + ADD(eq_bias) + ReLU 单图）
// ============================================================================

static CBRAmpInfFusedGraphCache
build_cbr_amp_inf_fused_graph(
    const DTensor& dt_x, const DTensor& dt_w,
    const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;
    auto graph = create_cudnn_graph(DType::FP16);

    int64_t K = dt_y.c();

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

    // eq_scale / eq_bias: per-channel, {1,K,1,1}, stride {K,1,K,K}, FP32
    auto eq_scale = graph->tensor(Tensor_attributes()
        .set_name("eq_scale")
        .set_dim({1, K, 1, 1})
        .set_stride({K, 1, K, K})
        .set_data_type(fe::DataType_t::FLOAT));

    auto eq_bias = graph->tensor(Tensor_attributes()
        .set_name("eq_bias")
        .set_dim({1, K, 1, 1})
        .set_stride({K, 1, K, K})
        .set_data_type(fe::DataType_t::FLOAT));

    // Conv fprop
    auto conv_opts = Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto conv_out = graph->conv_fprop(X, W, conv_opts);
    conv_out->set_dim(to_fe_dim(dt_y.shape))
            .set_stride(make_nhwc_stride(dt_y))
            .set_data_type(fe::DataType_t::HALF);   // 虚拟张量，不 set_output

    // MUL(eq_scale)
    auto mul_opts = Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::MUL)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto scaled = graph->pointwise(conv_out, eq_scale, mul_opts);
    scaled->set_data_type(fe::DataType_t::HALF);

    // ADD(eq_bias)
    auto add_opts = Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::ADD)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto shifted = graph->pointwise(scaled, eq_bias, add_opts);
    shifted->set_data_type(fe::DataType_t::HALF);

    // ReLU
    auto relu_opts = Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::RELU_FWD)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto relu_out = graph->pointwise(shifted, relu_opts);

    relu_out->set_output(true)
            .set_name("relu_output")
            .set_dim(to_fe_dim(dt_y.shape))
            .set_stride(make_nhwc_stride(dt_y))
            .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    CBRAmpInfFusedGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.input_ta    = X;
    cache.weight_ta   = W;
    cache.eq_scale_ta = eq_scale;
    cache.eq_bias_ta  = eq_bias;
    cache.output_ta   = relu_out;
    return cache;
}

// ============================================================================
// CBR_AMP_FWD
// ============================================================================

static void launch_cbr_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=amp_w, [2]=bn_w, [3]=bn_b,
    //             [4]=prev_mean, [5]=prev_var, [6]=eps, [7]=mom
    // output_ids: [0]=conv_output, [1]=bn_sum, [2]=bn_sq_sum, [3]=bn_output,
    //             [4]=saved_mean, [5]=saved_inv_var, [6]=relu_output, [7]=relu_mask,
    //             [8]=next_mean, [9]=next_var

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;
    const auto& bp = cbrp.bn;

    float eps_val = bp.eps;
    float mom_val = bp.momentum;

    int64_t K = mp.get_dtensor(node.output_ids[0]).c();  // conv_output channels

    // ── 1) Conv+GenStats on COMP_1 ─────────────────────────────────────
    {
        const DTensor& dt_x      = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_w      = mp.get_dtensor(node.input_ids[1]);
        const DTensor& dt_y      = mp.get_dtensor(node.output_ids[0]);
        const DTensor& dt_sum    = mp.get_dtensor(node.output_ids[1]);
        const DTensor& dt_sq_sum = mp.get_dtensor(node.output_ids[2]);

        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        int si = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != s) {
                cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_FWD
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_fwd_cache, key, [&]() {
            return build_cbr_conv_amp_fwd_graph(dt_x, dt_w, dt_y, dt_sum, dt_sq_sum, cp, h);
        });

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* ws = ctx.workspace(sk);

        update_cbr_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id,
                                     -1, -1, -1, dt_sum.id, dt_sq_sum.id);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                          "CBR_AMP_FWD conv execute");

        cudaEventRecord(state.streams[si].last_done_event, s);
    }

    // ── 2) BNFinalize on COMP_2 ────────────────────────────────────────
    {
        const DTensor& dt_sum         = mp.get_dtensor(node.output_ids[1]);
        const DTensor& dt_sq_sum      = mp.get_dtensor(node.output_ids[2]);
        const DTensor& dt_scale       = mp.get_dtensor(node.input_ids[2]);
        const DTensor& dt_bias        = mp.get_dtensor(node.input_ids[3]);
        const DTensor& dt_prev_mean   = mp.get_dtensor(node.input_ids[4]);
        const DTensor& dt_prev_var    = mp.get_dtensor(node.input_ids[5]);
        const DTensor& dt_next_mean   = mp.get_dtensor(node.output_ids[8]);
        const DTensor& dt_next_var    = mp.get_dtensor(node.output_ids[9]);
        const DTensor& dt_saved_mean  = mp.get_dtensor(node.output_ids[4]);
        const DTensor& dt_saved_inv_var = mp.get_dtensor(node.output_ids[5]);
        const DTensor& dt_eq_scale_bias = mp.get_dtensor(node.output_ids[3]);  // bn_output 复用

        const DTensor& dt_conv_out_bn = mp.get_dtensor(node.output_ids[0]);
        int N = dt_conv_out_bn.n();
        int H = dt_conv_out_bn.h();
        int W = dt_conv_out_bn.w();
        int C_conv = dt_conv_out_bn.c();
        int64_t accum_count = static_cast<int64_t>(N) * H * W;

        // bn_output 缓冲区必须足够容纳 eq_scale + eq_bias (2*K*sizeof(float))
        TR_CHECK(dt_eq_scale_bias.nbytes() >= 2 * C_conv * sizeof(float),
                 ValueError,
                 "CBR_AMP_FWD: bn_output buffer too small for eq_scale/eq_bias");

        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        uint64_t handle_bits = reinterpret_cast<uint64_t>(handle);

        CBRBNFinalizeCacheKey key{handle_bits, N, H, W, C_conv,
                                  float_to_bits(eps_val), float_to_bits(mom_val)};
        auto it = s_cbr_bn_finalize_caches.find(key);
        if (it == s_cbr_bn_finalize_caches.end()) {
            auto cache = build_cbr_bn_finalize_graph(
                dt_sum, dt_sq_sum, dt_scale, dt_bias,
                dt_prev_mean, dt_prev_var,
                dt_next_mean, dt_next_var,
                dt_saved_mean, dt_saved_inv_var,
                dt_eq_scale_bias,
                eps_val, mom_val, accum_count, handle);
            it = s_cbr_bn_finalize_caches.emplace(key, std::move(cache)).first;
        }

        CBRBNFinalizeGraphCache& cache = it->second;
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* workspace = ctx.workspace(sk);

        // Update tensor IDs
        std::unordered_map<std::string, int64_t> name_to_id = {
            {"sum",       dt_sum.id},
            {"sq_sum",    dt_sq_sum.id},
            {"scale",     dt_scale.id},
            {"bias",      dt_bias.id},
            {"prev_running_mean", dt_prev_mean.id},
            {"prev_running_var",  dt_prev_var.id},
            {"eq_scale",  dt_eq_scale_bias.id},
            {"eq_bias",   dt_eq_scale_bias.id},
            {"saved_mean",     dt_saved_mean.id},
            {"saved_inv_var",  dt_saved_inv_var.id},
            {"next_rm",   dt_next_mean.id},
            {"next_rv",   dt_next_var.id},
        };
        for (auto& [ta, tid] : cache.tensor_to_id) {
            const std::string& name = ta->get_name();
            auto it2 = name_to_id.find(name);
            if (it2 != name_to_id.end()) tid = it2->second;
        }

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            if (ta->get_name() == "eq_bias") {
                vp[ta] = static_cast<char*>(ctx.ptr_at(static_cast<int>(tid)))
                         + K * sizeof(float);
            } else {
                vp[ta] = ctx.ptr_at(static_cast<int>(tid));
            }
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, workspace),
                          "CBR_AMP_FWD bn_finalize execute");

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 3) BN Apply+ReLU on COMP_3 ─────────────────────────────────────
    {
        const DTensor& dt_x      = mp.get_dtensor(node.output_ids[0]);  // conv_output
        const DTensor& dt_y      = mp.get_dtensor(node.output_ids[6]);  // relu_output
        const DTensor& dt_eq_scale_bias = mp.get_dtensor(node.output_ids[3]);  // eq_scale/eq_bias

        int64_t mask_id = mp.get_dtensor(node.output_ids[7]).id;

        int N = dt_x.n(), H = dt_x.h(), W = dt_x.w(), C = dt_x.c();

        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        uint64_t handle_bits = reinterpret_cast<uint64_t>(handle);

        CBRBNReluCacheKey key{handle_bits, N, H, W, C};
        auto it = s_cbr_bn_relu_caches.find(key);
        if (it == s_cbr_bn_relu_caches.end()) {
            auto cache = build_cbr_bn_relu_graph(dt_x, dt_y, dt_eq_scale_bias, handle);
            it = s_cbr_bn_relu_caches.emplace(key, std::move(cache)).first;
        }

        CBRBNReluGraphCache& cache = it->second;
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* workspace = ctx.workspace(sk);

        // Update tensor IDs
        std::unordered_map<std::string, int64_t> name_to_id = {
            {"bn_relu_x",   dt_x.id},
            {"eq_scale",    dt_eq_scale_bias.id},
            {"eq_bias",     dt_eq_scale_bias.id},
            {"relu_output", dt_y.id},
            {"relu_mask",   mask_id},
        };
        for (auto& [ta, tid] : cache.tensor_to_id) {
            const std::string& name = ta->get_name();
            auto it2 = name_to_id.find(name);
            if (it2 != name_to_id.end()) tid = it2->second;
        }

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            if (ta->get_name() == "eq_bias") {
                vp[ta] = static_cast<char*>(ctx.ptr_at(static_cast<int>(tid)))
                         + K * sizeof(float);
            } else {
                vp[ta] = ctx.ptr_at(static_cast<int>(tid));
            }
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, workspace),
                          "CBR_AMP_FWD bn_relu execute");

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    }

// ============================================================================
// BN+ReLU BWD 融合图：cuDNN FE Graph 构建函数
// ============================================================================

// 注意：dt_bwd_out 绑定到 bn_output 缓冲区，但其中存储的是 dReLU+dBN 的输出，
// 即 conv_out 的梯度（dL/d(conv_out)），而非 bn_output 的梯度。
// 这是复用显存的技巧，因为 bn_output 在 BWD 阶段不再需要。
static CBRBNGraphCache build_cbr_bn_bwd_fused_graph(
    const DTensor& dt_dy,        // input_ids[0]
    const DTensor& dt_mask,      // input_ids[5], INT8 buffer, BOOLEAN semantics
    const DTensor& dt_x,         // output_ids[4], conv_output (BN forward input)
    const DTensor& dt_scale,     // input_ids[2], bn_weight
    const DTensor& dt_saved_mean,
    const DTensor& dt_saved_inv_var,
    const DTensor& dt_bwd_out,   // output_ids[5], bn_output reused as bn_bwd_out
                                  // 注意：存的是 dL/d(conv_out)，不是 dL/d(bn_output)
    const DTensor& dt_dscale,    // output_ids[2]
    const DTensor& dt_dbias,     // output_ids[3]
    cudnnHandle_t handle)
{
    using namespace fe::graph;
    auto graph = create_cudnn_graph(DType::FP16);

    int64_t N = dt_dy.n(), C = dt_dy.c(), H = dt_dy.h(), W = dt_dy.w();

    auto dy_ta = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({N, C, H, W})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto mask_ta = graph->tensor(Tensor_attributes()
        .set_name("mask")
        .set_dim({N, C, H, W})
        .set_stride(make_nhwc_stride(dt_mask))
        .set_data_type(fe::DataType_t::BOOLEAN));

    auto x_ta = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto make_param = [&](const char* name) {
        return graph->tensor(Tensor_attributes()
            .set_name(name)
            .set_dim({1, C, 1, 1})
            .set_stride({C, 1, C, C})
            .set_data_type(fe::DataType_t::FLOAT));
    };
    auto scale_ta = make_param("scale");
    auto mean_ta  = make_param("saved_mean");
    auto ivar_ta  = make_param("saved_inv_var");

    // MUL(dY, BOOLEAN mask) 正确解码 bit-packed mask
    auto mul_opts = Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::MUL)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto dy_masked = graph->pointwise(dy_ta, mask_ta, mul_opts);
    dy_masked->set_name("dy_masked")
              .set_data_type(fe::DataType_t::HALF);

    auto dbn_opts = Batchnorm_backward_attributes()
        .set_saved_mean_and_inv_variance(mean_ta, ivar_ta)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto [dx, dscale, dbias] = graph->batchnorm_backward(
        dy_masked, x_ta, scale_ta, dbn_opts);

    dx->set_output(true)
       .set_name("dX")
       .set_dim({N, C, H, W})
       .set_stride(make_nhwc_stride(dt_bwd_out))
       .set_data_type(fe::DataType_t::HALF);

    dscale->set_output(true)
          .set_name("dS")
          .set_dim({1, C, 1, 1})
          .set_stride({C, 1, C, C})
          .set_data_type(fe::DataType_t::FLOAT);

    dbias->set_output(true)
         .set_name("dB")
         .set_dim({1, C, 1, 1})
         .set_stride({C, 1, C, C})
         .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    CBRBNGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dy_ta]      = dt_dy.id;
    cache.tensor_to_id[mask_ta]    = dt_mask.id;
    cache.tensor_to_id[x_ta]       = dt_x.id;
    cache.tensor_to_id[scale_ta]   = dt_scale.id;
    cache.tensor_to_id[mean_ta]    = dt_saved_mean.id;
    cache.tensor_to_id[ivar_ta]    = dt_saved_inv_var.id;
    cache.tensor_to_id[dx]         = dt_bwd_out.id;
    cache.tensor_to_id[dscale]     = dt_dscale.id;
    cache.tensor_to_id[dbias]      = dt_dbias.id;
    return cache;
}

// ============================================================================
// CBR_AMP_BWD
// ============================================================================

static void launch_cbr_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=dY, [1]=amp_w, [2]=bn_w, [3]=saved_mean, [4]=saved_inv_var, [5]=mask, [6]=X
    // output_ids: [0]=dX, [1]=conv_amp_g, [2]=dγ, [3]=dβ, [4]=conv_output, [5]=bn_output

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;

    // ── 1) BN+ReLU BWD on COMP_1 ───────────────────────────────────────
    int i_bn;
    {
        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        i_bn = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0 && state.streams[out_idx].stream != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
        state.streams[i_bn].has_pending_work = true;

        const DTensor& dt_dy    = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_mask  = mp.get_dtensor(node.input_ids[5]);
        const DTensor& dt_x     = mp.get_dtensor(node.output_ids[4]); // conv_output
        const DTensor& dt_scale = mp.get_dtensor(node.input_ids[2]);
        const DTensor& dt_sm    = mp.get_dtensor(node.input_ids[3]);
        const DTensor& dt_siv   = mp.get_dtensor(node.input_ids[4]);
        const DTensor& dt_bout  = mp.get_dtensor(node.output_ids[5]); // bn_output -> bn_bwd_out
        const DTensor& dt_dscale= mp.get_dtensor(node.output_ids[2]);
        const DTensor& dt_dbias = mp.get_dtensor(node.output_ids[3]);

        CBRBNGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_dy.n(), dt_dy.h(), dt_dy.w(), dt_dy.c(),
            (dt_dy.dtype == DType::FP16),
            0u, 0u   // eps/momentum not used
        };

        auto it = s_cbr_bn_bwd_caches.find(key);
        if (it == s_cbr_bn_bwd_caches.end()) {
            auto cache = build_cbr_bn_bwd_fused_graph(
                dt_dy, dt_mask, dt_x, dt_scale, dt_sm, dt_siv,
                dt_bout, dt_dscale, dt_dbias, h);
            it = s_cbr_bn_bwd_caches.emplace(key, std::move(cache)).first;
        }
        auto& cache = it->second;

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        std::unordered_map<std::string, int64_t> name_to_id = {
            {"dY", dt_dy.id}, {"mask", dt_mask.id}, {"X", dt_x.id},
            {"scale", dt_scale.id}, {"saved_mean", dt_sm.id},
            {"saved_inv_var", dt_siv.id}, {"dX", dt_bout.id},
            {"dS", dt_dscale.id}, {"dB", dt_dbias.id}
        };
        update_cbr_bn_tensor_to_id(cache, name_to_id);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ctx.workspace(sk)),
                          "CBR_AMP_BWD bn_bwd fused");

        cudaEventRecord(state.streams[i_bn].last_done_event, s);
        state.output_stream_idx = i_bn;
    }

    // ── 2) WGrad on COMP_3 ─────────────────────────────────────────────
    int i_wg;
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        i_wg = state.get_or_register(s);

        cudaStreamWaitEvent(s, state.streams[i_bn].last_done_event, 0);
        state.streams[i_wg].has_pending_work = true;

        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[5]); // bn_bwd_out
        const DTensor& dt_x  = mp.get_dtensor(node.input_ids[6]);  // X
        const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]); // conv_amp_g

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_dw.n(), dt_dw.h(), dt_dw.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_BWD
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_wgrad_cache, key, [&]() {
            return build_cbr_conv_amp_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h);
        });
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        update_cbr_conv_tensor_to_id(cache, dt_x.id, dt_dw.id, -1, dt_dy.id, -1, dt_dw.id);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ctx.workspace(sk)),
                          "CBR_AMP_BWD wgrad");

        cudaEventRecord(state.streams[i_wg].last_done_event, s);
        state.output_stream_idx = i_wg;
    }

    // ── 3) DGrad on COMP_2 ─────────────────────────────────────────────
    int i_dg;
    {
        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        i_dg = state.get_or_register(s);

        // 等待 WGrad，确保 WGrad 已读完原始 X 后 DGrad 才覆盖 X
        cudaStreamWaitEvent(s, state.streams[i_wg].last_done_event, 0);
        state.streams[i_dg].has_pending_work = true;

        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[5]); // bn_bwd_out
        const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);  // amp_w
        const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]); // dX target (=X)

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_dx.n(), dt_dx.h(), dt_dx.w(), dt_dx.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_BWD
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_dgrad_cache, key, [&]() {
            return build_cbr_conv_amp_dgrad_graph(dt_dy, dt_w, dt_dx, cp, h);
        });
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        update_cbr_conv_tensor_to_id(cache, dt_dx.id, dt_w.id, -1, dt_dy.id, dt_dx.id, -1);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ctx.workspace(sk)),
                          "CBR_AMP_BWD dgrad");

        cudaEventRecord(state.streams[i_dg].last_done_event, s);
        state.output_stream_idx = i_dg;
    }

    // ── 4) Join back to COMP_1 ─────────────────────────────────────────
    {
        cudaStream_t s1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
        int i_s1 = state.get_or_register(s1);

        cudaStreamWaitEvent(s1, state.streams[i_wg].last_done_event, 0);
        cudaStreamWaitEvent(s1, state.streams[i_dg].last_done_event, 0);

        state.streams[i_s1].has_pending_work = true;
        cudaEventRecord(state.streams[i_s1].last_done_event, s1);
        state.output_stream_idx = i_s1;
    }
}

// ============================================================================
// CBR_AMP_BWD_FIRST_LAYER
// ============================================================================

static void launch_cbr_amp_bwd_first_layer_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=dY, [1]=amp_w, [2]=bn_w, [3]=saved_mean, [4]=saved_inv_var, [5]=mask, [6]=X
    // output_ids: [0]=dX target (Compiler 注入 data_a/data_b, 不写入), [1]=conv_amp_g, [2]=dγ, [3]=dβ,
    //             [4]=conv_output, [5]=bn_output

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;

    // ── 1) BN+ReLU BWD on COMP_1 ───────────────────────────────────────
    int i_bn;
    {
        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        i_bn = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0 && state.streams[out_idx].stream != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
        state.streams[i_bn].has_pending_work = true;

        const DTensor& dt_dy    = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_mask  = mp.get_dtensor(node.input_ids[5]);
        const DTensor& dt_x     = mp.get_dtensor(node.output_ids[4]); // conv_output
        const DTensor& dt_scale = mp.get_dtensor(node.input_ids[2]);
        const DTensor& dt_sm    = mp.get_dtensor(node.input_ids[3]);
        const DTensor& dt_siv   = mp.get_dtensor(node.input_ids[4]);
        const DTensor& dt_bout  = mp.get_dtensor(node.output_ids[5]); // bn_output -> bn_bwd_out
        const DTensor& dt_dscale= mp.get_dtensor(node.output_ids[2]);
        const DTensor& dt_dbias = mp.get_dtensor(node.output_ids[3]);

        CBRBNGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_dy.n(), dt_dy.h(), dt_dy.w(), dt_dy.c(),
            (dt_dy.dtype == DType::FP16),
            0u, 0u   // eps/momentum not used
        };

        auto it = s_cbr_bn_bwd_caches.find(key);
        if (it == s_cbr_bn_bwd_caches.end()) {
            auto cache = build_cbr_bn_bwd_fused_graph(
                dt_dy, dt_mask, dt_x, dt_scale, dt_sm, dt_siv,
                dt_bout, dt_dscale, dt_dbias, h);
            it = s_cbr_bn_bwd_caches.emplace(key, std::move(cache)).first;
        }
        auto& cache = it->second;

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        std::unordered_map<std::string, int64_t> name_to_id = {
            {"dY", dt_dy.id}, {"mask", dt_mask.id}, {"X", dt_x.id},
            {"scale", dt_scale.id}, {"saved_mean", dt_sm.id},
            {"saved_inv_var", dt_siv.id}, {"dX", dt_bout.id},
            {"dS", dt_dscale.id}, {"dB", dt_dbias.id}
        };
        update_cbr_bn_tensor_to_id(cache, name_to_id);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ctx.workspace(sk)),
                          "CBR_AMP_BWD_FIRST_LAYER bn_bwd fused");

        cudaEventRecord(state.streams[i_bn].last_done_event, s);
        state.output_stream_idx = i_bn;
    }

    // ── 2) WGrad on COMP_3 ─────────────────────────────────────────────
    int i_wg;
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        i_wg = state.get_or_register(s);

        cudaStreamWaitEvent(s, state.streams[i_bn].last_done_event, 0);
        state.streams[i_wg].has_pending_work = true;

        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[5]); // bn_bwd_out
        const DTensor& dt_x  = mp.get_dtensor(node.input_ids[6]);  // X
        const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]); // conv_amp_g

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_dw.n(), dt_dw.h(), dt_dw.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_BWD_FIRST_LAYER
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_wgrad_cache, key, [&]() {
            return build_cbr_conv_amp_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h);
        });
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        update_cbr_conv_tensor_to_id(cache, dt_x.id, dt_dw.id, -1, dt_dy.id, -1, dt_dw.id);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ctx.workspace(sk)),
                          "CBR_AMP_BWD_FIRST_LAYER wgrad");

        cudaEventRecord(state.streams[i_wg].last_done_event, s);
        state.output_stream_idx = i_wg;
    }

    // ── 3) Join back to COMP_1 (no DGrad for first layer) ──────────────
    {
        cudaStream_t s1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
        int i_s1 = state.get_or_register(s1);

        cudaStreamWaitEvent(s1, state.streams[i_wg].last_done_event, 0);

        state.streams[i_s1].has_pending_work = true;
        cudaEventRecord(state.streams[i_s1].last_done_event, s1);
        state.output_stream_idx = i_s1;
    }
}

// ============================================================================
// CBR_AMP_INF
// ============================================================================

static void launch_cbr_amp_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=amp_w, [2]=eq_scale, [3]=eq_bias
    // output_ids: [0]=conv_output  (fused 路径虚拟化，fallback 仍写入)
    //             [1]=bn_sum       (reserved, INF 不使用)
    //             [2]=bn_sq_sum    (reserved, INF 不使用)
    //             [3]=bn_output    (fused 路径虚拟化，fallback 仍写入)
    //             [4]=relu_output  (唯一真实输出)
    //             [5]=relu_mask    (INF 不生成)

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[4]);  // relu_output

    // 以下输出张量由 memory plan 分配并保留在接口中，但 fused 路径不再写入：
    // [0] conv_output / [3] bn_output 在 fused 路径中虚拟化；
    // [1] bn_sum / [2] bn_sq_sum / [5] relu_mask 在 INF 中始终不使用。
    (void)node.output_ids[1];
    (void)node.output_ids[2];
    (void)node.output_ids[5];

    // ── 尝试融合图（Conv + MUL + ADD + ReLU 单图） ──────────────────
    {
        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        int si = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != s) {
                cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        try {
            CBRAmpInfFusedGraphCacheKey key{
                reinterpret_cast<uint64_t>(h),
                dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
                dt_w.n(), dt_w.h(), dt_w.w(),
                cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w
            };
            auto it = s_cbr_amp_inf_fused_cache.find(key);
            if (it == s_cbr_amp_inf_fused_cache.end()) {
                auto cache = build_cbr_amp_inf_fused_graph(
                    dt_x, dt_w, dt_y, cp, h);
                it = s_cbr_amp_inf_fused_cache.emplace(key, std::move(cache)).first;
            }
            auto& cache = it->second;
            ctx.ensure_workspace_grow(sk, cache.workspace_size);
            void* ws = ctx.workspace(sk);

            std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
            vp[cache.input_ta]    = ctx.ptr_at(node.input_ids[0]);
            vp[cache.weight_ta]   = ctx.ptr_at(node.input_ids[1]);
            vp[cache.eq_scale_ta] = ctx.ptr_at(node.input_ids[2]);
            vp[cache.eq_bias_ta]  = ctx.ptr_at(node.input_ids[3]);
            vp[cache.output_ta]   = ctx.ptr_at(node.output_ids[4]);

            TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                              "CBR_AMP_INF fused execute");

            cudaEventRecord(state.streams[si].last_done_event, s);
            return;
        } catch (const std::exception&) {
            // fused graph 构建/执行失败，回退到三段式
        }
    }

    // ── fallback: 三段式 (Conv + BN + ReLU) ──────────────────────────
    // 1) Conv INF on COMP_1
    {
        const DTensor& dt_conv_x = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_conv_w = mp.get_dtensor(node.input_ids[1]);
        const DTensor& dt_conv_y = mp.get_dtensor(node.output_ids[0]);

        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        int si = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != s) {
                cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_conv_x.n(), dt_conv_x.h(), dt_conv_x.w(), dt_conv_x.c(),
            dt_conv_w.n(), dt_conv_w.h(), dt_conv_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_INF
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_fwd_cache, key, [&]() {
            return build_cbr_conv_amp_inf_graph(dt_conv_x, dt_conv_w, dt_conv_y, cp, h);
        });

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* ws = ctx.workspace(sk);

        update_cbr_conv_tensor_to_id(cache, dt_conv_x.id, dt_conv_w.id, dt_conv_y.id);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                          "CBR_AMP_INF conv execute");

        cudaEventRecord(state.streams[si].last_done_event, s);
    }

    // 2) BN INF on COMP_2
    {
        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const DTensor& dt_bn_x = mp.get_dtensor(node.output_ids[0]);  // conv_output
        Shape shape = dt_bn_x.shape;
        int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
        bool is_fp16 = (dt_bn_x.dtype == DType::FP16);

        const float* eq_scale = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
        const float* eq_bias  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
        void* y = ctx.ptr_at(node.output_ids[3]);  // bn_output

        cudaError_t err = launch_tr_bn_inf_eq_kernel(
            ctx.ptr_at(node.output_ids[0]),  // x = conv_output
            eq_scale, eq_bias,
            y, N, C, H, W, is_fp16, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_INF bn kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // 3) ReLU INF on COMP_3
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const __half* x = static_cast<const __half*>(ctx.ptr_at(node.output_ids[3]));  // bn_output
        __half* y       = static_cast<__half*>(ctx.ptr_at(node.output_ids[4]));         // relu_output

        const DTensor& dt_bn_out = mp.get_dtensor(node.output_ids[3]);
        int64_t n = static_cast<int64_t>(dt_bn_out.padded_elems());

        cudaError_t err = launch_relu_amp_inf_kernel(x, y, n, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_INF relu kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }
}

#endif // TR_USE_CUDA

// ============================================================================
// CPU 不支持
// ============================================================================

static void launch_cbr_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_CHECK(false, NotImplementedError, "CBR AMP is CUDA-only.");
}

// ============================================================================
// 算子注册
// ============================================================================

void register_op_cbr() {
    // CBR_AMP_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_FWD)];
        e.op = ComputeOp::CBR_AMP_FWD;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_fwd_cuda;
#endif
    }

    // CBR_AMP_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_BWD)];
        e.op = ComputeOp::CBR_AMP_BWD;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_bwd_cuda;
#endif
    }

    // CBR_AMP_BWD_FIRST_LAYER
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_BWD_FIRST_LAYER)];
        e.op = ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_bwd_first_layer_cuda;
#endif
    }

    // CBR_AMP_INF
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_INF)];
        e.op = ComputeOp::CBR_AMP_INF;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_inf_cuda;
#endif
    }

    TR_LOG_DEBUG("backend") << "CBR operators registered (AMP_FWD/BWD/BWD_FIRST_LAYER/INF)";
}

} // namespace tr