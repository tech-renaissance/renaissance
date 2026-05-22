/**
 * @file memory_arena.h
 * @brief 统一内存/显存池管理系统
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: mimalloc, CUDA (可选)
 * @note 所属系列: backend
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <mimalloc.h>

#include "renaissance/core/tr_exception.h"

namespace tr {

// ============================================================================
// MemoryArena - 统一内存/显存池抽象基类
// ============================================================================

/**
 * @class MemoryArena
 * @brief 内存/显存池抽象基类，提供统一的对齐分配接口
 *
 * 核心契约：
 * 1. allocate() 只能成功调用一次，再次调用将抛出异常
 * 2. 不提供public释放接口，资源随对象析构自动回收
 * 3. usable_size 必须能被 alignment 整除，否则抛出异常
 * 4. 实际分配 usable_size + alignment 字节，返回空间内最小对齐地址
 */
class MemoryArena {
public:
    /**
     * @brief 构造函数
     * @param alignment 对齐字节数（必须是2的幂）
     */
    explicit MemoryArena(size_t alignment);

    /**
     * @brief 虚析构函数（基类不释放资源，由派生类处理）
     */
    virtual ~MemoryArena();

    /**
     * @brief 分配内存/显存
     * @param usable_size 用户请求的可用字节数
     * @return 满足对齐要求的可用区域基地址
     * @throw TRException 如果多次调用、参数无效或分配失败
     */
    void* allocate(size_t usable_size);

    // 查询接口（训练热路径，inline零开销）
    [[nodiscard]] void* base_ptr() const noexcept { return aligned_ptr_; }
    [[nodiscard]] size_t usable_size() const noexcept { return usable_size_; }
    [[nodiscard]] bool is_allocated() const noexcept {
        return allocated_.load(std::memory_order_acquire);
    }
    [[nodiscard]] size_t alignment() const noexcept { return alignment_; }

    // 禁止拷贝/移动
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) = delete;
    MemoryArena& operator=(MemoryArena&&) = delete;

protected:
    /**
     * @brief 派生类实现的原始分配接口
     * @param total_size 总共需要分配的字节数
     * @return 分配的内存指针，失败返回nullptr
     */
    virtual void* do_allocate(size_t total_size) = 0;

    /**
     * @brief 将指针向上取整到2的幂对齐边界
     * @param ptr 原始指针
     * @param alignment 对齐字节数
     * @return 对齐后的指针
     */
    [[nodiscard]] static void* align_up(void* ptr, size_t alignment) noexcept;

    size_t alignment_;                      ///< 对齐字节数
    void* raw_ptr_ = nullptr;               ///< 原始分配地址（派生类析构时释放）
    void* aligned_ptr_ = nullptr;           ///< 对齐后的可用基地址
    size_t usable_size_ = 0;                ///< 用户请求的可用大小
    std::atomic<bool> allocated_{false};    ///< 是否已分配
};

// ============================================================================
// CpuArena - CPU内存池（后端：mimalloc）
// ============================================================================

/**
 * @class CpuArena
 * @brief CPU内存池实现，使用mimalloc作为后端分配器
 *
 * @note 构造函数为private，只能通过ArenaKeeper创建
 *       这确保了所有内存池都通过统一管理器创建
 */
class CpuArena : public MemoryArena {
    // 声明ArenaKeeper为友元，允许其访问private构造函数
    friend class ArenaKeeper;

public:
    /**
     * @brief 析构函数，释放CPU内存
     */
    ~CpuArena() override;

protected:
    void* do_allocate(size_t total_size) override;

private:
    /**
     * @brief 构造函数（私有，仅ArenaKeeper可访问）
     * @param alignment 对齐字节数，默认256
     */
    explicit CpuArena(size_t alignment = 256);
};

// ============================================================================
// CudaArena - GPU显存池（后端：CUDA Runtime）
// ============================================================================

#ifdef TR_USE_CUDA
/**
 * @class CudaArena
 * @brief GPU显存池实现，使用CUDA Runtime作为后端
 *
 * @note 构造函数为private，只能通过ArenaKeeper创建
 *       这确保了所有显存池都通过统一管理器创建
 */
class CudaArena : public MemoryArena {
    // 声明ArenaKeeper为友元，允许其访问private构造函数
    friend class ArenaKeeper;

public:
    /**
     * @brief 析构函数，释放GPU显存
     */
    ~CudaArena() override;

    /**
     * @brief 获取设备ID
     * @return GPU设备ID
     */
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

    /**
     * @brief 预热指定设备的CUDA上下文
     * @details cudaSetDevice首次调用会触发上下文懒加载，该过程在驱动层持有全局锁。
     *          主线程预先为每个设备完成上下文初始化，可消除多线程分配时的锁竞争。
     * @param device_id GPU设备ID
     */
    static void warm_context(int device_id);

protected:
    void* do_allocate(size_t total_size) override;

private:
    /**
     * @brief 构造函数（私有，仅ArenaKeeper可访问）
     * @param device_id GPU设备ID
     * @param alignment 对齐字节数，默认256
     */
    CudaArena(int device_id, size_t alignment = 256);

    int device_id_;  ///< GPU设备ID
};
#endif

// ============================================================================
// ArenaKeeper - 多卡并行、统一管理、跨平台全局单例
// ============================================================================

/**
 * @class ArenaKeeper
 * @brief 内存池/显存池的全局管理器，实现多卡并行分配和统一查询
 *
 * 核心特性：
 * 1. Mayer单例模式，全局唯一
 * 2. 支持CPU和GPU两种模式
 * 3. 多GPU并行分配，显著提升初始化速度
 * 4. 初始化后查询接口完全无锁，零开销
 * 5. 异常安全，分配失败自动回滚
 */
class ArenaKeeper {
public:
    /**
     * @brief 获取单例实例
     * @return 全局唯一的ArenaKeeper实例
     */
    static ArenaKeeper& instance();

    /**
     * @brief 触发单例初始化（可选调用）
     */
    static void init();

    /**
     * @brief 初始化内存池/显存池
     * @param using_gpu 是否使用GPU
     * @param device_ids 设备ID列表
     * @param usable_size_per_device 每设备的可用字节数
     * @param alignment 对齐字节数，默认256
     * @throw TRException 如果参数无效或分配失败
     */
    void initialize(bool using_gpu,
                    const std::vector<int>& device_ids,
                    size_t usable_size_per_device,
                    size_t alignment = 256);

    // 查询接口（init完成后完全无锁、线程安全）
    [[nodiscard]] void* base_ptr(int rank) const;
    [[nodiscard]] void* base_ptr_by_device(int device_id) const;
    [[nodiscard]] size_t usable_size(int rank) const;
    [[nodiscard]] MemoryArena* arena(int rank) const;
    [[nodiscard]] int device_id(int rank) const;
    [[nodiscard]] int rank_of_device(int device_id) const;

    /**
     * @brief 根据rank和偏移量获取实际指针
     * @param rank 设备rank（0 ~ world_size-1）
     * @param offset 相对Arena基地址的字节偏移
     * @return 实际设备/内存指针
     * @note 训练期热路径，Release模式下零分支、零虚函数调用
     *       Debug模式下保留完整的边界检查和异常诊断
     */
    [[nodiscard]] void* ptr_at(int rank, size_t offset) const noexcept;

    [[nodiscard]] size_t world_size() const noexcept { return world_size_; }
    [[nodiscard]] bool is_gpu_mode() const noexcept { return using_gpu_; }
    [[nodiscard]] bool is_initialized() const noexcept {
        return initialized_.load(std::memory_order_acquire);
    }

    // 禁止拷贝/移动
    ArenaKeeper(const ArenaKeeper&) = delete;
    ArenaKeeper& operator=(const ArenaKeeper&) = delete;
    ArenaKeeper(ArenaKeeper&&) = delete;
    ArenaKeeper& operator=(ArenaKeeper&&) = delete;

private:
    ArenaKeeper() = default;
    ~ArenaKeeper() = default;

    // 仅保护初始化过程，训练阶段永不触碰
    std::mutex init_mutex_;
    std::atomic<bool> initialized_{false};

    bool using_gpu_ = false;
    size_t world_size_ = 0;
    size_t alignment_ = 256;
    std::vector<int> device_ids_;                       ///< rank -> device_id
    std::vector<std::unique_ptr<MemoryArena>> arenas_;  ///< rank -> arena
    std::unordered_map<int, int> device_to_rank_;       ///< device_id -> rank
};

} // namespace tr