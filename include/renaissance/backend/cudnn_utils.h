/**
 * @file cudnn_utils.h
 * @brief cuDNN辅助函数：消除代码重复，提供公共工具
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: cudnn_frontend.h, types.h, exception.h
 * @note 所属系列: backend
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/core/tr_exception.h"

#ifdef TR_USE_CUDA
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cuda_runtime.h>

namespace tr {

namespace fe = cudnn_frontend;

// ─────────────────────────────────────────────────────────
// TR4 cuDNN Frontend 错误检查宏
// ─────────────────────────────────────────────────────────

#define TR_CUDNN_FE_CHECK(call, msg) \
    do { \
        auto err = (call); \
        if (err.is_bad()) { \
            TR_DEVICE_ERROR("cuDNN FE [" msg "]: " #call \
                           " failed with code: " << static_cast<int>(err.get_code())); \
        } \
    } while(0)

// ─────────────────────────────────────────────────────────
// DType ↔ cuDNN Frontend DataType 转换
// ─────────────────────────────────────────────────────────

inline fe::DataType_t to_fe_dtype(DType dtype) {
    switch (dtype) {
        case DType::FP32:  return fe::DataType_t::FLOAT;
        case DType::FP16:  return fe::DataType_t::HALF;
        case DType::INT8:  return fe::DataType_t::INT8;
        case DType::INT32: return fe::DataType_t::INT32;
        default:
            TR_TYPE_ERROR("Unsupported DType for cuDNN Frontend");
    }
}

// ─────────────────────────────────────────────────────────
// Shape → cuDNN Frontend 维度/stride 转换
// ─────────────────────────────────────────────────────────

inline std::vector<int64_t> to_fe_dim(const Shape& s) {
    return {static_cast<int64_t>(s.n()), static_cast<int64_t>(s.c()),
            static_cast<int64_t>(s.h()), static_cast<int64_t>(s.w())};
}

inline std::vector<int64_t> to_fe_stride_nhwc(const Shape& s, DType dt) {
    size_t elem_size = 0;
    switch (dt) {
        case DType::INT8:  elem_size = 1; break;
        case DType::FP16:  elem_size = 2; break;
        case DType::FP32:  elem_size = 4; break;
        case DType::INT32: elem_size = 4; break;
        default:
            TR_TYPE_ERROR("Unsupported DType for cuDNN Frontend stride calculation");
    }
    uint64_t row_bytes = static_cast<uint64_t>(s.w()) * s.c() * elem_size;
    uint64_t align = (dt == DType::FP16 || dt == DType::INT8) ? 128 : 256;
    uint64_t aligned_row_bytes = (row_bytes + align - 1) & ~(align - 1);
    int64_t row_elems = static_cast<int64_t>(aligned_row_bytes / elem_size);

    return {
        static_cast<int64_t>(s.h()) * row_elems,
        1,
        row_elems,
        static_cast<int64_t>(s.c())
    };
}

inline Shape shape_from_fe_dim(const std::vector<int64_t>& fe_dim) {
    return Shape(static_cast<int>(fe_dim[0]), static_cast<int>(fe_dim[2]),
                 static_cast<int>(fe_dim[3]), static_cast<int>(fe_dim[1]));
}

// ─────────────────────────────────────────────────────────
// cuDNN 图公共构建/收尾
// ─────────────────────────────────────────────────────────

inline std::shared_ptr<fe::graph::Graph> create_cudnn_graph(DType dtype) {
    auto graph = std::make_shared<fe::graph::Graph>();
    // 注：试验证明 intermediate_data_type 设为 FLOAT 或 HALF 均不影响精度。
    //     原因在于当前 graph 中每个张量的 data_type 已通过 set_data_type() 显式指定，
    //     intermediate_data_type 实际上未被使用。为节省内存/带宽，AMP 场景仍用 HALF。
    auto intermediate_dt = (dtype == DType::FP16)
        ? fe::DataType_t::HALF
        : fe::DataType_t::FLOAT;
    graph->set_io_data_type(to_fe_dtype(dtype))
          .set_intermediate_data_type(intermediate_dt)
          .set_compute_data_type(fe::DataType_t::FLOAT);
    return graph;
}

inline void finalize_cudnn_graph(fe::graph::Graph* graph, cudnnHandle_t handle) {
    TR_CUDNN_FE_CHECK(graph->validate(),                "graph validate");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "build op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}), "create exec plans");
    TR_CUDNN_FE_CHECK(graph->check_support(handle),       "check support");
    TR_CUDNN_FE_CHECK(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build plans");
}

// ─────────────────────────────────────────────────────────
// DType ↔ cuDNN Legacy DataType 转换
// ─────────────────────────────────────────────────────────

inline cudnnDataType_t to_cudnn_dtype(DType dtype) {
    switch (dtype) {
        case DType::FP32:  return CUDNN_DATA_FLOAT;
        case DType::FP16:  return CUDNN_DATA_HALF;
        case DType::INT8:  return CUDNN_DATA_INT8;
        case DType::INT32: return CUDNN_DATA_INT32;
        default:
            TR_TYPE_ERROR("Unsupported DType for cuDNN Legacy API");
    }
}

// ─────────────────────────────────────────────────────────
// TR4 cuDNN Legacy 错误检查宏
// ─────────────────────────────────────────────────────────

#define TR_CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t err = (call); \
        if (err != CUDNN_STATUS_SUCCESS) { \
            TR_DEVICE_ERROR("cuDNN Legacy [" #call "] failed: " << cudnnGetErrorString(err)); \
        } \
    } while(0)

} // namespace tr

#endif // TR_USE_CUDA
