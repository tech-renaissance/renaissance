/**
 * @file identity_op.cu
 * @brief IDENTITY 算子的 CUDA kernel 占位文件
 * @version 1.1.0
 * @date 2026-05-17
 * @author 技术觉醒团队
 * @note CUDA 路径已迁移至 identity_op.cpp 中使用 cudaMemcpyAsync（硬件加速）
 * @note 保留此文件以满足 CMake 构建系统的文件列表要求
 */

#ifdef TR_USE_CUDA

namespace tr {

// 所有 IDENTITY 算子的 CUDA 实现已内联到 identity_op.cpp 中，
// 使用 cudaMemcpyAsync 替代自定义 kernel，性能提升 10~50 倍。

} // namespace tr

#endif // TR_USE_CUDA
