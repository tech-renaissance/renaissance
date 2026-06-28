/**
 * @file memory_arena.cpp
 * @brief 统一内存/显存池管理系统实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend
 */

#include "renaissance/backend/memory_arena.h"

#include <exception>
#include <limits>

namespace tr {

// ============================================================================
// MemoryArena基类实现
// ============================================================================

MemoryArena::MemoryArena(size_t alignment) : alignment_(alignment) {
    // 对齐值必须是2的幂
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        TR_CHECK(false, ValueError,
                 "MemoryArena alignment must be power of 2 and > 0, got " << alignment);
    }
}

MemoryArena::~MemoryArena() = default;

void* MemoryArena::align_up(void* ptr, size_t alignment) noexcept {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t mask = alignment - 1;
    uintptr_t aligned = (addr + mask) & ~mask;
    return reinterpret_cast<void*>(aligned);
}

void* MemoryArena::allocate(size_t usable_size) {
    // CAS保证严格只调用一次，即使多线程误用也安全
    bool expected = false;
    if (!allocated_.compare_exchange_strong(
            expected, true,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        TR_CHECK(false, RuntimeError, "MemoryArena::allocate() can only be called once");
    }

    TR_CHECK(usable_size > 0, ValueError, "MemoryArena::allocate() usable_size must be > 0");
    TR_CHECK(usable_size % alignment_ == 0, ValueError,
             "MemoryArena::allocate() usable_size (" << usable_size << ") must be divisible by alignment (" << alignment_ << ")");

    // 防御性溢出检查
    TR_CHECK(usable_size <= std::numeric_limits<size_t>::max() - alignment_,
             MemoryError, "MemoryArena::allocate() size overflow");

    const size_t total_size = usable_size + alignment_;

    raw_ptr_ = do_allocate(total_size);
    TR_CHECK(raw_ptr_ != nullptr, MemoryError, "MemoryArena::allocate() allocation failed");

    aligned_ptr_ = align_up(raw_ptr_, alignment_);
    usable_size_ = usable_size;
    return aligned_ptr_;
}

// ============================================================================
// CpuArena实现
// ============================================================================

CpuArena::CpuArena(size_t alignment) : MemoryArena(alignment) {}

CpuArena::~CpuArena() {
    if (raw_ptr_) {
        mi_free(raw_ptr_);
        raw_ptr_     = nullptr;
        aligned_ptr_ = nullptr;
    }
}

void* CpuArena::do_allocate(size_t total_size) {
    void* ptr = mi_malloc(total_size);
    TR_CHECK(ptr != nullptr, MemoryError,
             "CpuArena::do_allocate() mi_malloc failed for " << total_size << " bytes");
    return ptr;
}

// ============================================================================
// CudaArena实现（条件编译）
// ============================================================================

#ifdef TR_USE_CUDA

CudaArena::CudaArena(int device_id, size_t alignment)
    : MemoryArena(alignment), device_id_(device_id) {}

CudaArena::~CudaArena() {
    if (raw_ptr_) {
        // 必须切回原设备上下文，否则可能误释其他设备显存
        cudaError_t err = cudaSetDevice(device_id_);
        if (err == cudaSuccess) {
            cudaFree(raw_ptr_);
        }
        // 若设备已离线/被重置，无法安全释放，静默处理避免UB
        raw_ptr_     = nullptr;
        aligned_ptr_ = nullptr;
    }
}

void CudaArena::warm_context(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err == cudaSuccess) {
        // 使用cudaDeviceSynchronize()触发Context创建
        // CUDA规范保证：同步操作必然需要设备上下文，无副作用
        // 相比cudaFree(0)（语义为"no operation"），此方式有明确规范保证
        cudaDeviceSynchronize();
    }
    // 若 cudaSetDevice 失败，说明设备不可用。
    // 此处静默处理，由后续 do_allocate 中的严格校验负责报错。
}

void* CudaArena::do_allocate(size_t total_size) {
    cudaError_t err = cudaSetDevice(device_id_);
    TR_CHECK(err == cudaSuccess, DeviceError,
             "CudaArena::do_allocate() cudaSetDevice(" << device_id_ << ") failed: " << cudaGetErrorString(err));

    void* ptr = nullptr;
    err = cudaMalloc(&ptr, total_size);
    TR_CHECK(err == cudaSuccess && ptr != nullptr, DeviceError,
             "CudaArena::do_allocate() cudaMalloc failed on device " << device_id_ <<
             " for " << total_size << " bytes: " << cudaGetErrorString(err));
    return ptr;
}

#endif // TR_USE_CUDA

// ============================================================================
// ArenaKeeper实现
// ============================================================================

ArenaKeeper& ArenaKeeper::instance() {
    static ArenaKeeper keeper;
    return keeper;
}

void ArenaKeeper::init() {
    (void)instance();  // 仅触发Mayer单例构造
}

void ArenaKeeper::initialize(bool using_gpu,
                             const std::vector<int>& device_ids,
                             size_t usable_size_per_device,
                             size_t alignment) {
    // 初始化全程互斥，确保仅执行一次；训练阶段此锁永不触碰
    std::lock_guard<std::mutex> lock(init_mutex_);
    TR_CHECK(!initialized_.load(std::memory_order_relaxed), RuntimeError,
             "ArenaKeeper::initialize() already called");

    TR_CHECK(!device_ids.empty(), ValueError, "ArenaKeeper::initialize() device_ids cannot be empty");
    TR_CHECK(usable_size_per_device > 0, ValueError,
             "ArenaKeeper::initialize() usable_size_per_device must be > 0");

    const size_t world_size = device_ids.size();
    std::vector<std::unique_ptr<MemoryArena>> temp_arenas;
    std::unordered_map<int, int> temp_map;
    temp_arenas.resize(world_size);

    if (using_gpu) {
#ifndef TR_USE_CUDA
        (void)alignment;
        (void)usable_size_per_device;
        TR_CHECK(false, RuntimeError,
                 "ArenaKeeper::initialize() GPU mode requested but TR_USE_CUDA is not defined");
#else
        // ====================================================================
        // CUDA上下文预热（关键性能优化）
        //
        // 问题：cudaSetDevice首次调用会触发上下文懒加载，驱动层持有全局锁。
        //       若让8个子线程首次调用时懒加载，会竞争同一把锁，强制串行。
        //       实测：8卡耗时 ≈ 8×单卡耗时（上下文创建占主导）。
        // 解决：主线程预先为每个设备串行完成上下文初始化。
        //       后续子线程的cudaSetDevice只需绑定已有上下文，零竞争。
        // ====================================================================
        for (int dev_id : device_ids) {
            CudaArena::warm_context(dev_id);
        }

        if (world_size > 1) {
            // 多线程并行显存申请：每线程独立绑定设备，零共享状态竞争
            std::vector<std::thread> threads;
            threads.reserve(world_size);
            std::vector<std::exception_ptr> exceptions(world_size);

            for (size_t i = 0; i < world_size; ++i) {
                int dev_id = device_ids[i];
                threads.emplace_back([&, i, dev_id]() {
                    try {
                        auto arena = std::unique_ptr<MemoryArena>(new CudaArena(dev_id, alignment));
                        arena->allocate(usable_size_per_device);
                        temp_arenas[i] = std::move(arena);
                    } catch (...) {
                        exceptions[i] = std::current_exception();
                    }
                });
            }

            for (auto& t : threads) {
                t.join();
            }

            // 任一卡失败，立即抛出，局部temp_arenas自动释放已分配资源
            for (const auto& e : exceptions) {
                if (e) {
                    std::rethrow_exception(e);
                }
            }
        } else {
            // 单卡：无需多线程开销
            auto arena = std::unique_ptr<MemoryArena>(new CudaArena(device_ids[0], alignment));
            arena->allocate(usable_size_per_device);
            temp_arenas[0] = std::move(arena);
        }

        for (size_t i = 0; i < world_size; ++i) {
            temp_map[device_ids[i]] = static_cast<int>(i);
        }
#endif
    } else {
        // CPU模式：强制world_size = 1
        TR_CHECK(world_size == 1, ValueError,
                 "ArenaKeeper::initialize() CPU mode requires exactly 1 device");
        auto arena = std::unique_ptr<MemoryArena>(new CpuArena(alignment));
        arena->allocate(usable_size_per_device);
        temp_arenas[0] = std::move(arena);
        temp_map[device_ids[0]] = 0;
    }

    // 所有分配成功，一次性移动到成员变量；此后进入只读状态
    using_gpu_ = using_gpu;
    device_ids_ = device_ids;
    world_size_ = world_size;
    alignment_ = alignment;
    arenas_ = std::move(temp_arenas);
    device_to_rank_ = std::move(temp_map);

    // Release语义：此前的所有写入对后续acquire load可见
    initialized_.store(true, std::memory_order_release);
}

void* ArenaKeeper::base_ptr(int rank) const {
    // 防御性检查：Release模式下由上层保证正确性
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && static_cast<size_t>(rank) < world_size_, IndexError,
                  "ArenaKeeper::base_ptr() invalid rank " << rank << ", world_size=" << world_size_);
    return arenas_[rank]->base_ptr();
}

void* ArenaKeeper::base_ptr_by_device(int device_id) const {
    return base_ptr(rank_of_device(device_id));
}

size_t ArenaKeeper::usable_size(int rank) const {
    // 防御性检查：Release模式下由上层保证正确性
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && static_cast<size_t>(rank) < world_size_, IndexError,
                  "ArenaKeeper::usable_size() invalid rank " << rank << ", world_size=" << world_size_);
    return arenas_[rank]->usable_size();
}

MemoryArena* ArenaKeeper::arena(int rank) const {
    // 防御性检查：Release模式下由上层保证正确性
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && static_cast<size_t>(rank) < world_size_, IndexError,
                  "ArenaKeeper::arena() invalid rank " << rank << ", world_size=" << world_size_);
    return arenas_[rank].get();
}

int ArenaKeeper::device_id(int rank) const {
    // 防御性检查：Release模式下由上层保证正确性
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && static_cast<size_t>(rank) < world_size_, IndexError,
                  "ArenaKeeper::device_id() invalid rank " << rank << ", world_size=" << world_size_);
    return device_ids_[rank];
}

int ArenaKeeper::rank_of_device(int device_id) const {
    TR_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
             "ArenaKeeper not initialized");
    auto it = device_to_rank_.find(device_id);
    TR_CHECK(it != device_to_rank_.end(), IndexError,
             "ArenaKeeper::rank_of_device() invalid device_id " << device_id);
    return it->second;
}

void* ArenaKeeper::ptr_at(int rank, size_t offset) const noexcept {
    // 防御性检查：Release模式下由上层保证正确性，Debug模式下保留完整诊断
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && static_cast<size_t>(rank) < world_size_, IndexError,
                  "ArenaKeeper::ptr_at() invalid rank " << rank << ", world_size=" << world_size_);

    // 热路径：纯指针运算，零分支、零虚函数调用
    return static_cast<char*>(arenas_[rank]->base_ptr()) + offset;
}

} // namespace tr