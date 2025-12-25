/**
 * @file memory_plan.cpp
 * @brief MemoryPlan实现（整数句柄机制 + 256字节对齐）
 * @version 3.8.1
 * @date 2025-12-25
 */

#include "renaissance/base/memory_plan.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include <algorithm>

namespace tr {

// 统一对齐基准：256字节 (适配 Cache Line, AVX2, CUDA)
constexpr size_t MEMORY_ALIGNMENT = 256;

int MemoryPlan::register_tensor(const std::string& tensor_id, size_t size, bool is_param) {
    // 检查是否已注册
    if (has_tensor(tensor_id)) {
        TR_LOG_WARN("MemoryPlan") << "Tensor already registered: " << tensor_id;
        return id_to_handle_[tensor_id];
    }

    // 计算当前基准偏移
    size_t current_offset = is_param ? param_size_ : (param_size_ + temp_size_);

    // 核心修正：执行256字节对齐
    // 公式：(offset + 255) & ~255
    size_t aligned_offset = (current_offset + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);

    // 更新各部分大小
    if (is_param) {
        // 参数内存：顺序分配，不复用
        param_size_ = aligned_offset + size;
    } else {
        // 临时内存：记录峰值需求（简化版，后续可优化生命周期分析）
        size_t temp_end = (aligned_offset - param_size_) + size;
        temp_size_ = std::max(temp_size_, temp_end);
    }

    // 更新总大小
    total_size_ = param_size_ + temp_size_;

    // 存入 Vector (确保 O(1) 访问)
    TensorSlot slot;
    slot.offset = aligned_offset;
    slot.size = size;
    slot.is_param = is_param;
    slots_.push_back(slot);

    // 建立 ID -> Handle 映射 (仅编译期使用)
    int handle = static_cast<int>(slots_.size()) - 1;
    id_to_handle_[tensor_id] = handle;

    TR_LOG_DEBUG("MemoryPlan") << "Registered tensor: " << tensor_id
                               << " size=" << size << " bytes"
                               << " offset=" << aligned_offset
                               << " handle=" << handle
                               << " is_param=" << (is_param ? "true" : "false");

    return handle;
}

void MemoryPlan::reserve_scratch_buffer(size_t size) {
    // 同样需要对齐
    size_t aligned_total = (total_size_ + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);
    scratch_offset_ = aligned_total;
    scratch_size_ = size;

    // 最终总显存/内存需求
    total_size_ = scratch_offset_ + scratch_size_;

    TR_LOG_INFO("MemoryPlan") << "Reserved ScratchBuffer: "
                              << scratch_size_ / (1024.0 * 1024.0) << " MB"
                              << " at offset " << scratch_offset_;
}

int MemoryPlan::get_handle(const std::string& tensor_id) const {
    auto it = id_to_handle_.find(tensor_id);
    if (it == id_to_handle_.end()) {
        return -1;  // 未注册
    }
    return it->second;
}

bool MemoryPlan::has_tensor(const std::string& tensor_id) const {
    return id_to_handle_.find(tensor_id) != id_to_handle_.end();
}

void MemoryPlan::print() const {
    TR_LOG_INFO("MemoryPlan") << "=== Memory Plan ===";
    TR_LOG_INFO("MemoryPlan") << "Total memory: " << total_size_ / (1024.0 * 1024.0) << " MB";
    TR_LOG_INFO("MemoryPlan") << "  Persistent (params/grads/optimizer): "
                               << param_size_ / (1024.0 * 1024.0) << " MB";
    TR_LOG_INFO("MemoryPlan") << "  Temporary (activations): "
                               << temp_size_ / (1024.0 * 1024.0) << " MB";
    if (scratch_size_ > 0) {
        TR_LOG_INFO("MemoryPlan") << "  ScratchBuffer: "
                                   << scratch_size_ / (1024.0 * 1024.0) << " MB";
    }
    TR_LOG_INFO("MemoryPlan") << "Registered tensors: " << slots_.size();
    TR_LOG_INFO("MemoryPlan") << "====================";
}

} // namespace tr
