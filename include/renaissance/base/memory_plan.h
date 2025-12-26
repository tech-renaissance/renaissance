/**
 * @file memory_plan.h
 * @brief 内存规划表（静态图编译期生成）
 * @details 使用整数句柄机制消除热路径字符串查找
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: base/memory
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace tr {

/**
 * @struct TensorSlot
 * @brief 张量内存槽位信息
 */
struct TensorSlot {
    size_t offset;      ///< 在Arena中的偏移（字节）
    size_t size;        ///< 字节数
    bool is_param;      ///< 是否是模型参数（持久内存）
};

/**
 * @class MemoryPlan
 * @brief 内存规划表（compile阶段生成，run阶段使用）
 *
 * 核心优化（V3.8.1）：
 * - 使用vector存储槽位（O(1)数组访问）
 * - 整数句柄替代字符串哈希（消除热路径开销）
 * - 256字节对齐（适配AVX2 + CUDA）
 *
 * 性能提升：
 * - 字符串哈希20-50ns → 数组索引1-2ns
 * - 单次epoch可节省毫秒级
 *
 * 工作流程：
 * 1. 编译期：register_tensor() → 返回int句柄
 * 2. 运行期：get_offset(handle) → 直接数组索引
 */
class MemoryPlan {
public:
    MemoryPlan() = default;
    ~MemoryPlan() = default;

    /**
     * @brief 注册张量（返回整数句柄）
     * @param tensor_id 张量唯一标识（如"layer1.weight"）
     * @param size 字节数
     * @param is_param 是否是参数（持久内存）
     * @return int句柄（vector索引）
     *
     * 关键：应用256字节对齐算法
     * 公式：(offset + 255) & ~255
     */
    int register_tensor(const std::string& tensor_id, size_t size, bool is_param);

    /**
     * @brief 预留暂存缓冲区（V3.8.1新增）
     * @param size ScratchBuffer大小（如ResNet-50建议512MB）
     *
     * ScratchBuffer用于cuDNN算法搜索等临时计算
     * 在所有张量注册完成后调用
     */
    void reserve_scratch_buffer(size_t size);

    /**
     * @brief 通过句柄获取偏移（性能关键！）
     * @param handle 整数句柄
     * @return 偏移字节数
     *
     * 热路径：内联优化，纯数组访问
     */
    inline size_t get_offset(int handle) const noexcept {
        return slots_[handle].offset;
    }

    /**
     * @brief 获取ScratchBuffer偏移
     * @return 偏移字节数
     */
    size_t get_scratch_offset() const noexcept { return scratch_offset_; }

    /**
     * @brief 通过ID获取句柄（仅编译期使用）
     * @param tensor_id 张量ID
     * @return 句柄，未注册返回-1
     *
     * 注意：此方法仅在编译期使用，运行期应直接用句柄
     */
    int get_handle(const std::string& tensor_id) const;

    /**
     * @brief 检查张量是否已注册
     * @param tensor_id 张量ID
     * @return 是否已注册
     */
    bool has_tensor(const std::string& tensor_id) const;

    /**
     * @brief 获取所需总内存
     * @return 总字节数
     */
    size_t total_size() const noexcept { return total_size_; }

    /**
     * @brief 获取参数内存大小
     * @return 持久内存字节数
     */
    size_t param_size() const noexcept { return param_size_; }

    /**
     * @brief 获取临时内存大小
     * @return 临时内存字节数
     */
    size_t temp_size() const noexcept { return temp_size_; }

    /**
     * @brief 获取ScratchBuffer大小
     * @return ScratchBuffer字节数
     */
    size_t scratch_size() const noexcept { return scratch_size_; }

    /**
     * @brief 获取已注册张量数量
     * @return 张量数量
     */
    size_t tensor_count() const noexcept { return slots_.size(); }

    /**
     * @brief 打印规划详情（调试用）
     */
    void print() const;

    // 禁止拷贝和移动
    MemoryPlan(const MemoryPlan&) = delete;
    MemoryPlan& operator=(const MemoryPlan&) = delete;
    MemoryPlan(MemoryPlan&&) = delete;
    MemoryPlan& operator=(MemoryPlan&&) = delete;

private:
    std::vector<TensorSlot> slots_;                    ///< 核心优化：用vector替代map
    std::unordered_map<std::string, int> id_to_handle_;///< 仅编译期使用
    size_t total_size_ = 0;                            ///< 总内存
    size_t param_size_ = 0;                            ///< 持久内存（参数、梯度、优化器状态）
    size_t temp_size_ = 0;                             ///< 临时内存（激活值，可复用）
    size_t scratch_offset_ = 0;                        ///< ScratchBuffer偏移
    size_t scratch_size_ = 0;                          ///< ScratchBuffer大小
};

} // namespace tr
