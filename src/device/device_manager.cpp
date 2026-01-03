/**
 * @file device_manager.cpp
 * @brief 器件管理器实现
 * @version 3.6.4
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/device_manager.h"
#include "renaissance/device/cpu_device.h"

#ifdef TR_USE_CUDA
#include "renaissance/device/cuda_device.h"
#include <cuda_runtime.h>
#endif

#ifdef TR_USE_MUSA
#include "renaissance/device/musa_device.h"
#include <musa_runtime.h>
#endif

namespace tr {

DeviceManager& DeviceManager::instance() noexcept {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    LOG_INFO << "Initializing DeviceManager...";
    initialize();
    LOG_INFO << "DeviceManager initialized. CUDA: " << cuda_count_
             << ", MUSA: " << musa_count_;
}

void DeviceManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return;

    // 1. 创建CPU器件（索引0，必定存在）
    devices_[0] = std::make_unique<CpuDevice>();
    LOG_INFO << "CPU device created: " << devices_[0]->hardware_name();

    // 2. 检测并创建CUDA器件（索引1~8）
#ifdef TR_USE_CUDA
    cuda_count_ = detect_cuda();
#endif

    // 3. 检测并创建MUSA器件（索引9~16）
#ifdef TR_USE_MUSA
    musa_count_ = detect_musa();
#endif

    // 4. 设置默认器件
    if (cuda_count_ > 0) {
        default_device_ = DeviceType::cuda(0);
        LOG_INFO << "Default device: CUDA:0";
    } else if (musa_count_ > 0) {
        default_device_ = DeviceType::musa(0);
        LOG_INFO << "Default device: MUSA:0";
    } else {
        default_device_ = DeviceType::cpu();
        LOG_INFO << "Default device: CPU";
    }

    // 5. 打印器件信息
    print_devices();

    initialized_ = true;
}

// ===== 器件访问实现 =====

Device& DeviceManager::get(const DeviceType& type) {
    int idx = device_index(type);

    int max_devices = 1 + cuda_count_ + musa_count_;
    if (idx < 0 || idx >= max_devices) {
        TR_VALUE_ERROR("Invalid device type: " << type.to_string());
    }

    auto& device_ptr = devices_[idx];

    if (!device_ptr) {
        TR_VALUE_ERROR("Device not available: " << type.to_string());
    }

    if (!device_ptr->is_available()) {
        TR_VALUE_ERROR("Device offline at runtime: " << type.to_string());
    }

    return *device_ptr;
}

const Device& DeviceManager::get(const DeviceType& type) const {
    return const_cast<DeviceManager*>(this)->get(type);
}

CpuDevice& DeviceManager::cpu() noexcept {
    // 直接访问，无需检查（CPU必定存在）
    return *static_cast<CpuDevice*>(devices_[0].get());
}

// ===== 默认设备管理 =====

void DeviceManager::set_default(const DeviceType& type) {
    // 验证设备存在
    get(type);

    std::lock_guard<std::mutex> lock(mutex_);
    default_device_ = type;

    LOG_INFO << "Default device changed to: " << type.to_string();
}

Device& DeviceManager::default_device() {
    return get(default_device_);
}

// ===== 调试信息 =====

void DeviceManager::print_devices() const {
    LOG_INFO << "=== Available Devices ===";

    // CPU
    if (devices_[0]) {
        LOG_INFO << "[0] CPU - " << devices_[0]->hardware_name();
    }

#ifdef TR_USE_CUDA
    // CUDA
    if (cuda_count_ > 0) {
        LOG_INFO << "[CUDA] " << cuda_count_ << " device(s):";
        for (int i = 0; i < cuda_count_; ++i) {
            if (devices_[1 + i]) {
                LOG_INFO << "  [" << i << "] " << devices_[1 + i]->hardware_name();
            }
        }
    }
#endif

#ifdef TR_USE_MUSA
    // MUSA
    if (musa_count_ > 0) {
        LOG_INFO << "[MUSA] " << musa_count_ << " device(s):";
        for (int i = 0; i < musa_count_; ++i) {
            if (devices_[9 + i]) {
                LOG_INFO << "  [" << i << "] " << devices_[9 + i]->hardware_name();
            }
        }
    }
#endif

    LOG_INFO << "Default: " << default_device_.to_string();
}

// ===== 设备索引计算 =====

int DeviceManager::device_index(const DeviceType& type) noexcept {
    if (type.is_cpu()) return 0;
    if (type.is_cuda()) return 1 + type.index();
    if (type.is_musa()) return 9 + type.index();
    return -1;
}

#ifdef TR_USE_CUDA
int DeviceManager::detect_cuda() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        LOG_WARN << "CUDA not available: " << cudaGetErrorString(err);
        return 0;
    }

    if (count > 8) {
        LOG_WARN << "Found " << count << " CUDA devices, limiting to 8";
        count = 8;
    }

    LOG_INFO << "Detected " << count << " CUDA device(s)";

    // 创建设备实例
    for (int i = 0; i < count; ++i) {
        try {
            int slot_index = 1 + i;  // CUDA[0]在索引1
            devices_[slot_index] = std::make_unique<CudaDevice>(i);

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            LOG_INFO << "  CUDA:" << i << " - " << prop.name
                     << " (" << prop.totalGlobalMem / (1024*1024) << " MB)";
        } catch (const std::exception& e) {
            LOG_ERROR << "Failed to initialize CUDA device " << i
                      << ": " << e.what();
            return i;  // 返回成功初始化的数量
        }
    }

    return count;
}
#else
int DeviceManager::detect_cuda() {
    LOG_INFO << "CUDA support not compiled (TR_USE_CUDA=OFF)";
    return 0;
}
#endif

#ifdef TR_USE_MUSA
int DeviceManager::detect_musa() {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);

    if (err != musaSuccess) {
        LOG_WARN << "MUSA not available";
        return 0;
    }

    if (count > 8) {
        LOG_WARN << "Found " << count << " MUSA devices, limiting to 8";
        count = 8;
    }

    LOG_INFO << "Detected " << count << " MUSA device(s)";

    for (int i = 0; i < count; ++i) {
        try {
            int slot_index = 9 + i;  // MUSA[0]在索引9
            devices_[slot_index] = std::make_unique<MusaDevice>(i);

            musaDeviceProp prop;
            musaGetDeviceProperties(&prop, i);
            LOG_INFO << "  MUSA:" << i << " - " << prop.name
                     << " (" << prop.totalGlobalMem / (1024*1024) << " MB)";
        } catch (const std::exception& e) {
            LOG_ERROR << "Failed to initialize MUSA device " << i;
            return i;
        }
    }

    return count;
}
#else
int DeviceManager::detect_musa() {
    LOG_INFO << "MUSA support not compiled (TR_USE_MUSA=OFF)";
    return 0;
}
#endif

#ifdef TR_USE_CUDA
CudaDevice& DeviceManager::cuda(int index) {
    if (index < 0 || index >= cuda_count_) {
        TR_VALUE_ERROR("CUDA device index out of range: " << index
                        << " (available: " << cuda_count_ << ")");
    }

    return *static_cast<CudaDevice*>(devices_[1 + index].get());
}
#endif

#ifdef TR_USE_MUSA
MusaDevice& DeviceManager::musa(int index) {
    if (index < 0 || index >= musa_count_) {
        TR_VALUE_ERROR("MUSA device index out of range: " << index
                        << " (available: " << musa_count_ << ")");
    }

    return *static_cast<MusaDevice*>(devices_[9 + index].get());
}
#endif

// ============================================================================
// NCCL通信管理实现
// ============================================================================

#ifdef TR_USE_NCCL
#include "renaissance/device/cuda_device.h"
#include <nccl.h>

void DeviceManager::setup_nccl(int gpu_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (nccl_active_) {
        LOG_WARN << "NCCL already active";
        return;
    }

    TR_CHECK(gpu_count >= 2, ValueError,
            "NCCL requires at least 2 GPUs, got " << gpu_count);
    TR_CHECK(gpu_count <= cuda_count_, ValueError,
            "Requested " << gpu_count << " GPUs but only " << cuda_count_ << " available");

    // ===== 性能调优环境变量设置（V3.6.30新增）=====
    // 强制使用Ring算法（双GPU场景最优）
    // 启用P2P直接访问
    // 优化通道数以提升带宽利用率
#ifdef _WIN32
    _putenv_s("NCCL_ALGO", "Ring");
    _putenv_s("NCCL_MIN_NCHANNELS", "4");
    _putenv_s("NCCL_MAX_NCHANNELS", "16");
    _putenv_s("NCCL_P2P_DISABLE", "0");
    _putenv_s("NCCL_SHM_DISABLE", "0");
    _putenv_s("NCCL_IB_DISABLE", "1");
    _putenv_s("NCCL_NET_GDR_LEVEL", "PHB");
#else
    setenv("NCCL_ALGO", "Ring", 1);
    setenv("NCCL_MIN_NCHANNELS", "4", 1);
    setenv("NCCL_MAX_NCHANNELS", "16", 1);
    setenv("NCCL_P2P_DISABLE", "0", 1);
    setenv("NCCL_SHM_DISABLE", "0", 1);
    setenv("NCCL_IB_DISABLE", "1", 1);
    setenv("NCCL_NET_GDR_LEVEL", "PHB", 1);
#endif

    LOG_INFO << "Setting up NCCL for " << gpu_count << " GPUs (Performance tuning enabled)";

    // 1. 准备设备和通信器数组
    std::vector<int> devices(gpu_count);
    std::vector<ncclComm_t> comms(gpu_count);

    for (int i = 0; i < gpu_count; ++i) {
        devices[i] = i;
    }

    // 2. 使用ncclCommInitAll一次性初始化所有通信器（避免死锁）
    ncclResult_t result = ncclCommInitAll(comms.data(), gpu_count, devices.data());

    // 专家评审建议：如果初始化失败，清理已创建的通信器，避免资源泄漏
    if (result != ncclSuccess) {
        // 清理可能已经部分初始化的通信器
        for (int i = 0; i < gpu_count; ++i) {
            if (comms[i] != nullptr) {
                ncclCommAbort(comms[i]);  // ncclCommAbort用于异常情况下清理
                comms[i] = nullptr;
            }
        }

        TR_DEVICE_ERROR("ncclCommInitAll failed: " << ncclGetErrorString(result)
                        << ", cleaned up partially initialized communicators");
    }

    // 3. 将通信器设置到每个CudaDevice
    for (int rank = 0; rank < gpu_count; ++rank) {
        CudaDevice* cuda_dev = static_cast<CudaDevice*>(devices_[1 + rank].get());
        cuda_dev->enable_nccl(gpu_count, rank, comms[rank]);
    }

    nccl_active_ = true;
    nccl_world_size_ = gpu_count;

    LOG_INFO << "NCCL initialized for " << gpu_count << " GPUs";
}

void DeviceManager::cleanup_nccl() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!nccl_active_) {
        LOG_WARN << "NCCL not active";
        return;
    }

    LOG_INFO << "Cleaning up NCCL";

    // 销毁所有CUDA设备的NCCL通信器
    for (int i = 0; i < cuda_count_; ++i) {
        CudaDevice* cuda_dev = static_cast<CudaDevice*>(devices_[1 + i].get());
        cuda_dev->cleanup_nccl();
    }

    nccl_active_ = false;
    nccl_world_size_ = 0;

    LOG_INFO << "NCCL cleaned up";
}
#endif

} // namespace tr
