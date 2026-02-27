/**
 * @file initializer.cpp
 * @brief 全局单例初始化器实现
 * @version 1.0.0
 * @date 2026-02-27
 * @author 技术觉醒团队
 */

#include "renaissance/utils/initializer.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/global_registry.h"
#include "renaissance/device/device_manager.h"
#include "renaissance/data/preprocessor.h"
#include <algorithm>
#include <cctype>

namespace tr {

void Initializer::init(const std::string& deep_learning_device) {
    if (initialized_) {
        return;  // 幂等性：多次调用只生效一次
    }

    // ========================================================================
    // 处理设备字符串
    // ========================================================================

    // 1. 转大写
    std::string device = deep_learning_device;
    std::transform(device.begin(), device.end(), device.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    // 2. CUDA/MUSA -> GPU
    if (device == "CUDA" || device == "MUSA") {
        device = "GPU";
    }

    // 3. 场景宏检查：如果定义了EDGE_ARM/EDGE_RISCV/CPU_CLOUD，强制使用CPU
#if defined(TR_SCENE_EDGE_ARM) || defined(TR_SCENE_EDGE_RISCV) || defined(TR_SCENE_CPU_CLOUD)
    device = "CPU";
    LOG_WARN << "Framework running in CPU-only scene (EDGE_ARM/EDGE_RISCV/CPU_CLOUD), "
             << "deep learning device forced to CPU";
#endif

    // 4. 验证设备类型
    if (device != "CPU" && device != "GPU") {
        TR_VALUE_ERROR("Invalid deep learning device: '" << deep_learning_device
                       << "' (normalized to: '" << device << "')"
                       << "\n  Supported values: CPU, GPU, CUDA, MUSA");
    }

    LOG_INFO << "Initializing framework with deep learning device: " << device;

    // ========================================================================
    // 按优先级顺序初始化单例
    // ========================================================================

    // Logger (priority 10)
    Logger::instance().init();

    // GlobalRegistry (priority 20)
    GlobalRegistry::instance().init();

    // 注册设备类型到 GlobalRegistry
    GlobalRegistry::instance().set_using_gpu(device == "GPU");

    // 设置框架初始化标志（此时 GlobalRegistry 已初始化，但其他组件尚未初始化）
    GlobalRegistry::instance().set_initializer_inited(true);

    // DeviceManager (priority 30)
    DeviceManager::instance().init();

    // Preprocessor (priority 40)
    Preprocessor::instance().init();

    initialized_ = true;

    LOG_INFO << "Framework initialized successfully (INIT_FRAMEWORK completed)";
}

Initializer::ScopeGuard::~ScopeGuard() {
    // 析构时自动清理所有单例
    cleanup_all();
}

void Initializer::cleanup_all() {
    if (!initialized_) {
        return;
    }

    // 按优先级逆序清理：Preprocessor → DeviceManager → GlobalRegistry → Logger
    // Preprocessor (priority 40)
    Preprocessor::instance().cleanup();

    // DeviceManager (priority 30)
    DeviceManager::instance().cleanup();

    // GlobalRegistry (priority 20)
    GlobalRegistry::instance().cleanup();

    // Logger (priority 10)
    Logger::instance().cleanup();

    initialized_ = false;
}

} // namespace tr
