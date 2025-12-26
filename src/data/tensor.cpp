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
#include <sstream>
#include <cstring>
#include <iomanip>
#include <cmath>

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
        TR_TYPE_ERROR("Expected fp32, got ", dtype_name(dtype_));
    }
    return static_cast<float*>(data_ptr());
}

template<>
const float* Tensor::typed_data<float>() const {
    if (dtype_ != DType::FP32) {
        TR_TYPE_ERROR("Expected fp32, got ", dtype_name(dtype_));
    }
    return static_cast<const float*>(data_ptr());
}

// 类型安全访问（BF16特化）
template<>
uint16_t* Tensor::typed_data<uint16_t>() {
    if (dtype_ != DType::BF16) {
        TR_TYPE_ERROR("Expected bf16, got ", dtype_name(dtype_));
    }
    return static_cast<uint16_t*>(data_ptr());
}

template<>
const uint16_t* Tensor::typed_data<uint16_t>() const {
    if (dtype_ != DType::BF16) {
        TR_TYPE_ERROR("Expected bf16, got ", dtype_name(dtype_));
    }
    return static_cast<const uint16_t*>(data_ptr());
}

// 类型安全访问（INT32特化）
template<>
int32_t* Tensor::typed_data<int32_t>() {
    if (dtype_ != DType::INT32) {
        TR_TYPE_ERROR("Expected int32, got ", dtype_name(dtype_));
    }
    return static_cast<int32_t*>(data_ptr());
}

template<>
const int32_t* Tensor::typed_data<int32_t>() const {
    if (dtype_ != DType::INT32) {
        TR_TYPE_ERROR("Expected int32, got ", dtype_name(dtype_));
    }
    return static_cast<const int32_t*>(data_ptr());
}

// 类型安全访问（INT8特化）
template<>
int8_t* Tensor::typed_data<int8_t>() {
    if (dtype_ != DType::INT8) {
        TR_TYPE_ERROR("Expected int8, got ", dtype_name(dtype_));
    }
    return static_cast<int8_t*>(data_ptr());
}

template<>
const int8_t* Tensor::typed_data<int8_t>() const {
    if (dtype_ != DType::INT8) {
        TR_TYPE_ERROR("Expected int8, got ", dtype_name(dtype_));
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
        TR_VALUE_ERROR("Storage too small: ", storage->capacity(),
                       " bytes, need ", required, " bytes");
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
        TR_SHAPE_ERROR("view numel mismatch: ", numel(), " -> ", new_shape.numel());
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
        TR_SHAPE_ERROR("Cannot flatten: too many elements (", total_elements, ")");
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
    oss << "Tensor(";
    oss << "shape=" << shape_.to_string();
    oss << ", dtype=" << dtype_name(dtype_);

    if (device_type_.is_cpu()) {
        oss << ", device=CPU";
    } else if (device_type_.is_cuda()) {
        oss << ", device=CUDA(" << device_type_.index() << ")";
    } else if (device_type_.is_musa()) {
        oss << ", device=MUSA(" << device_type_.index() << ")";
    }

    if (is_bound()) {
        oss << ", bound";
        if (is_view_) oss << ", view";
    } else {
        oss << ", unbound";
    }

    if (has_grad()) oss << ", has_grad";

    oss << ")";
    return oss.str();
}

void Tensor::print(const char* name) const {
    // 默认精度4位小数
    print(name, 4);
}

void Tensor::print(const char* name, int precision) const {
    std::ostringstream oss;

    // 打印名称
    if (name && name[0] != '\0') {
        oss << name << ":\n";
    }

    // 打印内容
    oss << "tensor(";

    if (is_empty()) {
        oss << "[]";
    } else {
        // 格式化数据内容
        format_tensor_content(oss, precision);
    }

    // 打印设备和类型信息（仅非默认值）
    if (!device_type_.is_cpu()) {
        if (device_type_.is_cuda()) {
            oss << ", device='cuda(" << device_type_.index() << ")'";
        } else if (device_type_.is_musa()) {
            oss << ", device='musa(" << device_type_.index() << ")'";
        }
    }

    if (dtype_ != DType::FP32) {
        oss << ", dtype=" << dtype_name(dtype_);
    }

    oss << ")";

    LOG_INFO << oss.str();
}

void Tensor::format_tensor_content(std::ostringstream& oss, int precision) const {
    // 简化实现：仅支持小张量打印
    if (!is_bound()) {
        oss << "[unbound]";
        return;
    }

    // 大张量不打印数据
    if (numel() > 64) {
        oss << "[...large tensor: " << numel() << " elements, use print() for details...]";
        return;
    }

    // GPU张量暂不支持打印（TODO: 需要Device支持to_cpu）
    if (!is_cpu()) {
        oss << "[...GPU tensor: " << numel() << " elements...]";
        return;
    }

    // 打印小张量数据
    oss << "[";
    const char* data = static_cast<const char*>(data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        if (i > 0) oss << ", ";

        if (dtype_ == DType::FP32) {
            const float* fp32_data = reinterpret_cast<const float*>(data);
            oss << std::fixed << std::setprecision(precision) << fp32_data[i];
        } else if (dtype_ == DType::BF16) {
            const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(data);
            oss << std::fixed << std::setprecision(precision) << bf16_to_float(bf16_data[i]);
        } else if (dtype_ == DType::INT32) {
            const int32_t* int32_data = reinterpret_cast<const int32_t*>(data);
            oss << int32_data[i];
        } else if (dtype_ == DType::INT8) {
            const int8_t* int8_data = reinterpret_cast<const int8_t*>(data);
            oss << static_cast<int>(int8_data[i]);
        } else {
            oss << "?";
        }
    }

    oss << "]";

    // TODO: 完整实现PyTorch风格格式化（参照tensor_old.cpp format_tensor_content）
    // 需要支持：
    // 1. 多维张量的嵌套打印（1D/2D/3D/4D）
    // 2. NHWC布局适配
    // 3. 更美观的缩进和换行
}

void Tensor::summary() const {
    LOG_INFO << to_string();
    if (storage_) {
        LOG_INFO << "  Storage: " << storage_->capacity() << " bytes"
                 << ", refs=" << storage_->use_count()
                 << ", " << (storage_->is_owned() ? "owned" : "borrowed");
    }
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
