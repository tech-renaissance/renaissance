/**
 * @file infra_kernels.cpp
 * @brief CPU 基础设施 kernel（通用工具，不对应具体算子）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend
 * @note 依赖项: Eigen3 (可选)
 */

#include "renaissance/core/tr_exception.h"

#ifdef TR_USE_EIGEN
#if __has_include(<Eigen/Core>)
#include <Eigen/Core>
#elif __has_include(<eigen3/Eigen/Core>)
#include <eigen3/Eigen/Core>
#else
#error "TR_USE_EIGEN is defined but Eigen3 headers not found"
#endif
#endif

namespace tr {

void launch_tr_fill_fp32_kernel_cpu(
    float* __restrict dst,
    float value,
    int64_t n) {

    TR_CHECK(dst != nullptr, ValueError, "launch_tr_fill_fp32_kernel_cpu: destination array is null");
    TR_CHECK(n > 0, ValueError, "launch_tr_fill_fp32_kernel_cpu: invalid element count " << n);

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf> dst_vec(dst, n);
    dst_vec.setConstant(value);
#else
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = value;
    }
#endif
}

} // namespace tr
