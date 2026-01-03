/**
 * @file cpu_device.cpp
 * @brief CPU器件实现
 * @version 3.6.26
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include <mimalloc.h>
#include <cstring>

namespace tr {

// ===== 构造/析构 =====

CpuDevice::CpuDevice() {
    LOG_INFO << "CpuDevice initialized on " << hardware_name();
}

CpuDevice::~CpuDevice() {
    LOG_INFO << "CpuDevice destroyed";
}

// ===== 器件信息 =====

DeviceType CpuDevice::type() const noexcept {
    return DeviceType::cpu();
}

std::string CpuDevice::hardware_name() const {
#if defined(TR_CPU_ARCH_X86_64)
    return "x86_64 CPU";
#elif defined(TR_CPU_ARCH_ARM64)
    return "ARM64 CPU";
#elif defined(TR_CPU_ARCH_RISCV64)
    return "RISC-V64 CPU";
#else
    return "Unknown CPU";
#endif
}

bool CpuDevice::is_available() const {
    return true;
}

size_t CpuDevice::memory_available() const {
    // 简化实现：返回16GB
    return 16ULL * 1024 * 1024 * 1024;
}

// ===== 内存管理（基于mimalloc）=====

std::shared_ptr<void> CpuDevice::allocate(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Cannot allocate 0 bytes");
    }

    // 调用mimalloc分配（CpuArena会处理对齐）
    void* ptr = mi_malloc(size);
    if (!ptr) {
        TR_MEMORY_ERROR("CPU allocation failed: " << size << " bytes");
    }

    return std::shared_ptr<void>(ptr, [](void* p) {
        mi_free(p);
    });
}

void CpuDevice::deallocate(void* ptr) {
    if (ptr) {
        mi_free(ptr);
    }
}

void CpuDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        TR_VALUE_ERROR("Null pointer in memcpy");
    }
    std::memcpy(dst, src, size);
}

void CpuDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in memset");
    }
    std::memset(ptr, value, size);
}

} // namespace tr
