/**
 * @file tensor.cpp
 * @brief Tensor类实现
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#include "renaissance/data/tensor.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/device_type.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <vector>

namespace tr {

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief BF16转FP32（辅助函数）
 * @param bf16 BF16值（uint16_t）
 * @return FP32值
 */
inline float bf16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// ============================================================================
// 构造函数
// ============================================================================

Tensor::Tensor() noexcept
    : shape_()
    , dtype_(DType::INVALID)
    , padding1_{0}
    , device_type_(DeviceType::cpu())
    , storage_(nullptr)
    , offset_(0)
    , is_view_(false)
    , padding2_{0}
    , grad_(nullptr)
{
}

Tensor::Tensor(const Shape& shape, DType dtype, DeviceType device_type,
                 std::shared_ptr<Storage> storage,
                 size_t offset, bool is_view)
    : shape_(shape)
    , dtype_(dtype)
    , padding1_{0}
    , device_type_(device_type)
    , storage_(storage)
    , offset_(offset)
    , is_view_(is_view)
    , padding2_{0}
    , grad_(nullptr)
{
}

// ============================================================================
// 字节数计算
// ============================================================================

size_t Tensor::nbytes() const noexcept {
    return static_cast<size_t>(numel()) * dtype_size(dtype_);
}

// ============================================================================
// 数据访问
// ============================================================================

void* Tensor::data_ptr() {
    if (dtype_ == DType::INVALID) {
        TR_VALUE_ERROR("Cannot access data of invalid Tensor");
    }
    if (!is_bound()) {
        TR_DEVICE_ERROR("Tensor not bound to storage");
    }
    return static_cast<char*>(storage_->data()) + offset_;
}

const void* Tensor::data_ptr() const {
    if (dtype_ == DType::INVALID) {
        TR_VALUE_ERROR("Cannot access data of invalid Tensor");
    }
    if (!is_bound()) {
        TR_DEVICE_ERROR("Tensor not bound to storage");
    }
    return static_cast<const char*>(storage_->data()) + offset_;
}

// 类型安全访问（FP32特化）
template<>
float* Tensor::typed_data<float>() {
    if (dtype_ != DType::FP32) {
        TR_TYPE_ERROR("Expected fp32, got " << dtype_name(dtype_));
    }
    return static_cast<float*>(data_ptr());
}

template<>
const float* Tensor::typed_data<float>() const {
    if (dtype_ != DType::FP32) {
        TR_TYPE_ERROR("Expected fp32, got " << dtype_name(dtype_));
    }
    return static_cast<const float*>(data_ptr());
}

// 类型安全访问（BF16特化）
template<>
uint16_t* Tensor::typed_data<uint16_t>() {
    if (dtype_ != DType::BF16) {
        TR_TYPE_ERROR("Expected bf16, got " << dtype_name(dtype_));
    }
    return static_cast<uint16_t*>(data_ptr());
}

template<>
const uint16_t* Tensor::typed_data<uint16_t>() const {
    if (dtype_ != DType::BF16) {
        TR_TYPE_ERROR("Expected bf16, got " << dtype_name(dtype_));
    }
    return static_cast<const uint16_t*>(data_ptr());
}

// 类型安全访问（INT32特化）
template<>
int32_t* Tensor::typed_data<int32_t>() {
    if (dtype_ != DType::INT32) {
        TR_TYPE_ERROR("Expected int32, got " << dtype_name(dtype_));
    }
    return static_cast<int32_t*>(data_ptr());
}

template<>
const int32_t* Tensor::typed_data<int32_t>() const {
    if (dtype_ != DType::INT32) {
        TR_TYPE_ERROR("Expected int32, got " << dtype_name(dtype_));
    }
    return static_cast<const int32_t*>(data_ptr());
}

// 类型安全访问（INT8特化）
template<>
int8_t* Tensor::typed_data<int8_t>() {
    if (dtype_ != DType::INT8) {
        TR_TYPE_ERROR("Expected int8, got " << dtype_name(dtype_));
    }
    return static_cast<int8_t*>(data_ptr());
}

template<>
const int8_t* Tensor::typed_data<int8_t>() const {
    if (dtype_ != DType::INT8) {
        TR_TYPE_ERROR("Expected int8, got " << dtype_name(dtype_));
    }
    return static_cast<const int8_t*>(data_ptr());
}

// ============================================================================
// Storage绑定
// ============================================================================

void Tensor::bind_storage(std::shared_ptr<Storage> storage, size_t offset) {
    if (!storage) {
        TR_VALUE_ERROR("Cannot bind null storage");
    }

    if (storage->device_type().index() != device_type_.index() ||
        storage->device_type().kind() != device_type_.kind()) {
        TR_DEVICE_ERROR("Storage device mismatch");
    }

    size_t required = offset + nbytes();
    if (storage->capacity() < required) {
        TR_VALUE_ERROR("Storage too small: " << storage->capacity()
                       << " bytes, need " << required << " bytes");
    }

    storage_ = std::move(storage);
    offset_ = offset;
}

bool Tensor::storage_fits() const noexcept {
    return storage_ && (offset_ + nbytes() <= storage_->capacity());
}

// ============================================================================
// 视图操作
// ============================================================================

Tensor Tensor::view(const Shape& new_shape) const {
    if (!is_bound()) {
        TR_DEVICE_ERROR("Cannot view unbound Tensor");
    }

    if (new_shape.numel() != numel()) {
        TR_SHAPE_ERROR("view numel mismatch: " << numel() << " -> " << new_shape.numel());
    }

    // 禁止view-of-view以防止offset计算错误
    if (is_view_) {
        TR_NOT_IMPLEMENTED("Viewing a view is not supported in MVP (to avoid offset errors).");
    }

    // 创建视图（共享Storage）
    Tensor view_tensor;
    view_tensor.shape_ = new_shape;
    view_tensor.dtype_ = dtype_;
    view_tensor.device_type_ = device_type_;
    view_tensor.storage_ = storage_;  // 关键：共享
    view_tensor.offset_ = offset_;
    view_tensor.is_view_ = true;

    return view_tensor;
}

Tensor Tensor::flatten() const {
    int64_t total_elements = numel();
    int32_t size = static_cast<int32_t>(total_elements);

    // 检查是否可以安全转换
    if (total_elements != static_cast<int64_t>(size)) {
        TR_SHAPE_ERROR("Cannot flatten: too many elements (" << total_elements << ")");
    }

    return view(Shape(size));
}

// ============================================================================
// 设备转换
// ============================================================================

Tensor Tensor::cpu() const {
    return to(DeviceType::cpu());
}

Tensor Tensor::cuda(int device_id) const {
    return to(DeviceType::cuda(device_id));
}

Tensor Tensor::to(const DeviceType& target) const {
    if (device_type_.kind() == target.kind() &&
        device_type_.index() == target.index()) {
        return *this;  // 浅拷贝（共享Storage）
    }

    // TODO: 跨设备拷贝需要Device支持
    // 暂时抛出未实现异常
    TR_NOT_IMPLEMENTED("Cross-device copy not yet implemented");
}

// ============================================================================
// 梯度管理
// ============================================================================

Tensor& Tensor::grad() {
    if (!grad_) {
        // 延迟创建（通过Device）
        // TODO: 需要Device支持
        TR_NOT_IMPLEMENTED("Gradient creation not yet implemented");
    }
    return *grad_;
}

const Tensor& Tensor::grad() const {
    if (!grad_) {
        TR_VALUE_ERROR("Gradient not initialized. Call non-const grad() first.");
    }
    return *grad_;
}

void Tensor::zero_grad() {
    if (grad_ && grad_->is_bound()) {
        // TODO: 需要Device支持
        TR_NOT_IMPLEMENTED("Gradient zeroing not yet implemented");
    }
}

// ============================================================================
// 调试输出
// ============================================================================

std::string Tensor::to_string() const {
    std::ostringstream oss;

    // 设备信息
    std::string device_str;
    if (device_type_.is_cpu()) {
        device_str = "CPU";
    } else if (device_type_.is_cuda()) {
        device_str = "CUDA(" + std::to_string(device_type_.index()) + ")";
    } else if (device_type_.is_musa()) {
        device_str = "MUSA(" + std::to_string(device_type_.index()) + ")";
    }

    // dtype信息
    std::string dtype_str = dtype_name(dtype_);

    oss << "Tensor(shape=" << shape_.to_string()
        << ", dtype=" << dtype_str
        << ", device=" << device_str;

    if (is_empty()) {
        oss << ", empty";
    } else {
        oss << ", numel=" << numel();

        // 显示张量数据内容（类似print()但不显示名称）
        if (numel() <= 16) {  // 只对小张量显示完整内容
            oss << ", data=";
            // 使用临时ostringstream捕获输出
            std::ostringstream temp_oss;
            format_tensor_content(temp_oss, 4);  // 使用默认4位精度
            oss << temp_oss.str();
        } else {  // 大张量只显示摘要信息，不读取具体数据
            oss << ", data=[...large tensor, use print() for details...]";
        }
    }

    oss << ")";
    return oss.str();
}

void Tensor::print(const char* name) const {
    // 默认精度4位小数
    print(name, 4);
}

void Tensor::print(const char* name, int precision) const {
    // 打印名称
    if (name && name[0] != '\0') {
        std::cout << name << ":" << std::endl;
    }

    // 打印内容（format_tensor_content已包含"tensor("前缀）
    if (is_empty()) {
        std::cout << "tensor([])" << std::endl;
    } else {
        // 格式化数据内容
        format_tensor_content(std::cout, precision);
        std::cout << std::endl;
    }
}

void Tensor::format_tensor_content(std::ostream& os, int precision) const {
    if (!is_bound()) {
        os << "[unbound]";
        return;
    }

    if (is_empty()) {
        os << "[]";
        return;
    }

    // GPU张量需要先传输到CPU
    std::vector<float> fp32_data;
    std::vector<int32_t> int32_data;
    std::vector<int8_t> int8_data;

    const void* raw_data = nullptr;

    if (is_cpu()) {
        raw_data = data_ptr();
    } else {
        // GPU张量：分配CPU buffer并传输数据
        // TODO: 需要通过Device的memcpy方法来实现
        // 暂时显示提示信息
        os << "[...GPU tensor: " << numel() << " elements, use to_cpu() first...]";
        return;
    }

    // 辅助函数：生成dtype字符串（FP32默认不显示）
    auto get_dtype_str = [&]() -> const char* {
        if (dtype_ == DType::FP32) {
            return nullptr;  // FP32是默认类型，不显示
        } else if (dtype_ == DType::BF16) {
            return "dtype=BF16";
        } else if (dtype_ == DType::INT32) {
            return "dtype=INT32";
        } else if (dtype_ == DType::INT8) {
            return "dtype=INT8";
        } else {
            return nullptr;
        }
    };

    // 根据维度格式化输出（完全参照tensor_old.cpp）
    if (ndim() == 0) {
        // 标量
        if (dtype_ == DType::FP32) {
            const float* data = reinterpret_cast<const float*>(raw_data);
            os << std::fixed << std::setprecision(precision) << data[0];
        } else if (dtype_ == DType::BF16) {
            const uint16_t* data = reinterpret_cast<const uint16_t*>(raw_data);
            os << std::fixed << std::setprecision(precision) << bf16_to_float(data[0]);
        } else if (dtype_ == DType::INT32) {
            const int32_t* data = reinterpret_cast<const int32_t*>(raw_data);
            os << data[0];
        } else if (dtype_ == DType::INT8) {
            const int8_t* data = reinterpret_cast<const int8_t*>(raw_data);
            os << static_cast<int>(data[0]);
        } else {
            os << "?";
        }

        // 添加dtype信息（FP32默认不显示）
        const char* dtype_str = get_dtype_str();
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
    } else if (ndim() == 1) {
        // 1D张量 - 完全匹配PyTorch风格
        int32_t d0 = shape_.dim(0);

        os << "tensor([";
        for (int32_t i = 0; i < d0; ++i) {
            if (i > 0) os << ", ";
            // 每8个元素换行（PyTorch默认）
            if (i == 8) os << std::endl << "        ";  // 8个空格缩进

            if (dtype_ == DType::FP32) {
                const float* data = reinterpret_cast<const float*>(raw_data);
                float val = data[i];
                if (val >= 0) os << " ";
                os << std::fixed << std::setprecision(precision) << val;
            } else if (dtype_ == DType::BF16) {
                const uint16_t* data = reinterpret_cast<const uint16_t*>(raw_data);
                float val = bf16_to_float(data[i]);
                if (val >= 0) os << " ";
                os << std::fixed << std::setprecision(precision) << val;
            } else if (dtype_ == DType::INT32) {
                const int32_t* data = reinterpret_cast<const int32_t*>(raw_data);
                int32_t val = data[i];
                // 整数右对齐，宽度根据数值范围动态调整
                // INT8: 3位（-128到127），INT32: 至少4位
                os << std::setw(4) << val;
            } else if (dtype_ == DType::INT8) {
                const int8_t* data = reinterpret_cast<const int8_t*>(raw_data);
                int val = static_cast<int>(data[i]);
                // INT8范围: -128到127，需要3位字符
                os << std::setw(3) << val;
            } else {
                os << "?";
            }
        }
        os << "]";

        // 添加dtype信息（FP32默认不显示）
        const char* dtype_str = get_dtype_str();
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else if (ndim() == 2) {
        // 2D张量 - 完全匹配PyTorch风格
        int32_t d0 = shape_.dim(0);
        int32_t d1 = shape_.dim(1);

        os << "tensor([";
        for (int32_t i = 0; i < d0; ++i) {
            if (i > 0) os << "        ";  // 8个空格缩进
            os << "[";
            for (int32_t j = 0; j < d1; ++j) {
                if (j > 0) os << ", ";
                int64_t idx = i * d1 + j;

                if (dtype_ == DType::FP32) {
                    const float* data = reinterpret_cast<const float*>(raw_data);
                    float val = data[idx];
                    if (val >= 0) os << " ";
                    os << std::fixed << std::setprecision(precision) << val;
                } else if (dtype_ == DType::BF16) {
                    const uint16_t* data = reinterpret_cast<const uint16_t*>(raw_data);
                    float val = bf16_to_float(data[idx]);
                    if (val >= 0) os << " ";
                    os << std::fixed << std::setprecision(precision) << val;
                } else if (dtype_ == DType::INT32) {
                    const int32_t* data = reinterpret_cast<const int32_t*>(raw_data);
                    int32_t val = data[idx];
                    os << std::setw(4) << val;
                } else if (dtype_ == DType::INT8) {
                    const int8_t* data = reinterpret_cast<const int8_t*>(raw_data);
                    int val = static_cast<int>(data[idx]);
                    os << std::setw(3) << val;
                } else {
                    os << "?";
                }
            }
            os << "]";
            // PyTorch格式：非最后一行才加逗号
            if (i < d0 - 1) {
                os << "," << std::endl;
            }
        }
        os << "]";

        // 添加dtype信息（FP32默认不显示）
        const char* dtype_str = get_dtype_str();
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else if (ndim() == 3) {
        // 3D张量 - 完全匹配PyTorch风格
        int32_t d0 = shape_.dim(0);
        int32_t d1 = shape_.dim(1);
        int32_t d2 = shape_.dim(2);

        os << "tensor([";
        for (int32_t i = 0; i < d0; ++i) {
            if (i > 0) {
                os << std::endl << "        ";  // 8个空格，d0块之间有空行
            }
            os << "[";
            for (int32_t j = 0; j < d1; ++j) {
                if (j > 0) os << "         ";  // 9个空格
                os << "[";
                for (int32_t k = 0; k < d2; ++k) {
                    if (k > 0) os << ", ";
                    int64_t idx = i * d1 * d2 + j * d2 + k;

                    if (dtype_ == DType::FP32) {
                        const float* data = reinterpret_cast<const float*>(raw_data);
                        float val = data[idx];
                        if (val >= 0) os << " ";
                        os << std::fixed << std::setprecision(precision) << val;
                    } else if (dtype_ == DType::BF16) {
                        const uint16_t* data = reinterpret_cast<const uint16_t*>(raw_data);
                        float val = bf16_to_float(data[idx]);
                        if (val >= 0) os << " ";
                        os << std::fixed << std::setprecision(precision) << val;
                    } else if (dtype_ == DType::INT32) {
                        const int32_t* data = reinterpret_cast<const int32_t*>(raw_data);
                        int32_t val = data[idx];
                        os << std::setw(4) << val;
                    } else if (dtype_ == DType::INT8) {
                        const int8_t* data = reinterpret_cast<const int8_t*>(raw_data);
                        int val = static_cast<int>(data[idx]);
                        os << std::setw(3) << val;
                    } else {
                        os << "?";
                    }
                }
                os << "]";
                // PyTorch格式：非最后一行才加逗号
                if (j < d1 - 1) {
                    os << "," << std::endl;
                }
            }
            // d0块结束：需要关闭d0的左括号
            // 非最后一行：d0右括号 + 外层逗号
            // 最后一行：d0右括号
            if (i < d0 - 1) {
                os << "]," << std::endl << std::endl;
            }
        }
        os << "]]";

        // 添加dtype信息（FP32默认不显示）
        const char* dtype_str = get_dtype_str();
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else if (ndim() == 4) {
        // 4D张量 - 完全匹配PyTorch风格
        int32_t d0 = shape_.dim(0);
        int32_t d1 = shape_.dim(1);
        int32_t d2 = shape_.dim(2);
        int32_t d3 = shape_.dim(3);

        os << "tensor([";
        for (int32_t i = 0; i < d0; ++i) {
            if (i > 0) os << "        ";  // 8个空格
            os << "[";
            for (int32_t j = 0; j < d1; ++j) {
                if (j > 0) os << "         ";  // 9个空格
                os << "[";
                for (int32_t k = 0; k < d2; ++k) {
                    if (k > 0) os << "          ";  // 10个空格
                    os << "[";
                    for (int32_t l = 0; l < d3; ++l) {
                        if (l > 0) os << ", ";
                        // NHWC到线性索引的转换
                        int64_t idx = i * d1 * d2 * d3 + j * d2 * d3 + k * d3 + l;

                        // 使用固定宽度7位（右对齐）：符号+数字+小数点+4位小数
                        if (dtype_ == DType::FP32) {
                            const float* data = reinterpret_cast<const float*>(raw_data);
                            float val = data[idx];
                            // 右对齐，正数前面加空格，负数直接输出
                            if (val >= 0) os << " ";
                            os << std::fixed << std::setprecision(precision) << val;
                        } else if (dtype_ == DType::BF16) {
                            const uint16_t* data = reinterpret_cast<const uint16_t*>(raw_data);
                            float val = bf16_to_float(data[idx]);
                            if (val >= 0) os << " ";
                            os << std::fixed << std::setprecision(precision) << val;
                        } else if (dtype_ == DType::INT32) {
                            const int32_t* data = reinterpret_cast<const int32_t*>(raw_data);
                            int32_t val = data[idx];
                            os << std::setw(4) << val;
                        } else if (dtype_ == DType::INT8) {
                            const int8_t* data = reinterpret_cast<const int8_t*>(raw_data);
                            int val = static_cast<int>(data[idx]);
                            os << std::setw(3) << val;
                        } else {
                            os << "?";
                        }
                    }
                    os << "]";
                    if (k < d2 - 1) os << "," << std::endl;
                }
                os << "]";
                if (j < d1 - 1) os << "," << std::endl << std::endl;
            }
            os << "]";
            if (i < d0 - 1) os << "," << std::endl << std::endl;
        }
        os << "]";
        // 添加dtype信息（FP32默认不显示）
        const char* dtype_str = get_dtype_str();
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else {
        // 更高维度暂时不支持
        os << "[...unsupported dimensions...]";
    }
}

void Tensor::summary() const {
    std::cout << to_string();

    // 添加内存大小信息
    if (storage_) {
        std::cout << ", memory_size=" << storage_->capacity() << " bytes";
    }

    std::cout << std::endl;
}

// ============================================================================
// 比较
// ============================================================================

bool Tensor::operator==(const Tensor& other) const noexcept {
    return shape_ == other.shape_ &&
           dtype_ == other.dtype_ &&
           device_type_.kind() == other.device_type_.kind() &&
           device_type_.index() == other.device_type_.index() &&
           storage_ == other.storage_;
}

} // namespace tr
