/**
 * @file cpu_comp.cpp
 * @brief CPU器件张量运算实现
 * @version 3.6.26
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/device/device_manager.h"
#include "renaissance/data/tensor.h"
#include "renaissance/base/dtype.h"
#include <cmath>
#include <algorithm>

namespace tr {

// ===== 张量运算 =====

void CpuDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // 1. 验证
    check_on_device(a);
    check_on_device(b);
    check_on_device(result);
    check_same_shape(a, b);
    check_same_shape(a, result);

    // 2. 检查数据类型一致
    if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
        TR_TYPE_ERROR("Dtype mismatch in add_into");
    }

    // 3. 执行加法（根据数据类型）
    size_t count = static_cast<size_t>(a.shape().numel());

    switch (a.dtype()) {
        case DType::FP32: {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
            break;
        }
        case DType::BF16: {
            // BF16加法：先转FP32，相加，再转回BF16（使用RNE舍入）
            const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
            const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
            uint16_t* r_data = static_cast<uint16_t*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                float a_fp32 = bf16_to_fp32(a_data[i]);
                float b_fp32 = bf16_to_fp32(b_data[i]);
                r_data[i] = fp32_to_bf16_rne(a_fp32 + b_fp32);  // RNE：舍入到最近
            }
            break;
        }
        case DType::INT32: {
            const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
            const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
            int32_t* r_data = static_cast<int32_t*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
            break;
        }
        case DType::INT8: {
            // INT8加法：需要饱和运算防止溢出
            const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
            const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
            int8_t* r_data = static_cast<int8_t*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                int16_t sum = static_cast<int16_t>(a_data[i]) + static_cast<int16_t>(b_data[i]);
                r_data[i] = static_cast<int8_t>(std::clamp(sum, int16_t(-128), int16_t(127)));
            }
            break;
        }
        default:
            TR_TYPE_ERROR("Unsupported dtype in add_into: " << dtype_name(a.dtype()));
    }
}

// ===== 张量比较 =====

bool CpuDevice::equal(const Tensor& a, const Tensor& b) {
    // 检查设备
    check_on_device(a);
    check_on_device(b);

    // 检查形状
    check_same_shape(a, b);

    // 检查dtype
    if (a.dtype() != b.dtype()) {
        TR_TYPE_ERROR("Cannot compare tensors with different dtypes: "
                 << dtype_name(a.dtype()) << " vs " << dtype_name(b.dtype()));
    }

    // 仅支持INT8和INT32
    if (a.dtype() == DType::FP32 || a.dtype() == DType::BF16) {
        TR_TYPE_ERROR("equal() only supports INT8 and INT32. "
                 "For FP32/BF16 comparison, use is_close() instead.");
    }

    // 处理空张量
    int64_t numel = a.numel();
    if (numel == 0) {
        return b.numel() == 0;
    }

    // 逐元素比较
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::INT32) {
        const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
        const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            if (a_data[i] != b_data[i]) {
                return false;
            }
        }
        return true;
    }
    else if (a.dtype() == DType::INT8) {
        const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
        const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            if (a_data[i] != b_data[i]) {
                return false;
            }
        }
        return true;
    }

    // 不应该到达这里
    TR_TYPE_ERROR("Unsupported dtype in equal: " << dtype_name(a.dtype()));
}

bool CpuDevice::is_close(const Tensor& a, const Tensor& b, float eps) {
    // 检查设备
    check_on_device(a);
    check_on_device(b);

    // 检查形状
    check_same_shape(a, b);

    // 检查dtype
    if (a.dtype() != b.dtype()) {
        TR_TYPE_ERROR("Cannot compare tensors with different dtypes: "
                 << dtype_name(a.dtype()) << " vs " << dtype_name(b.dtype()));
    }

    // 仅支持FP32和BF16
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        TR_TYPE_ERROR("is_close() only supports FP32 and BF16. "
                 "For INT8/INT32 comparison, use equal() instead.");
    }

    // 处理空张量
    int64_t numel = a.numel();
    if (numel == 0) {
        return b.numel() == 0;
    }

    // 确定容差
    float tolerance;
    if (eps < 0.0f) {
        // 使用默认容差
        tolerance = (a.dtype() == DType::FP32) ? 1e-6f : 1e-3f;
    } else {
        tolerance = eps;
    }

    // 逐元素比较
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::FP32) {
        const float* a_data = static_cast<const float*>(a.data_ptr());
        const float* b_data = static_cast<const float*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }
    else if (a.dtype() == DType::BF16) {
        // BF16存储为uint16，需要转换为FP32比较
        const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
        const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            float a_fp32 = bf16_to_fp32(a_data[i]);
            float b_fp32 = bf16_to_fp32(b_data[i]);
            float diff = std::abs(a_fp32 - b_fp32);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }

    // 不应该到达这里
    TR_TYPE_ERROR("Unsupported dtype in is_close: " << dtype_name(a.dtype()));
}

} // namespace tr
