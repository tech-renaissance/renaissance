/**
 * @file renaissance.h
 * @brief renAIssance Deep Learning Framework Main Header
 * @details €/É’ñ¦f`F¶8Ã4‡ö
 * @version 3.0.4
 * @date 2025-12-20
 * @author €/É’â
 */

#ifndef RENAISSANCE_H
#define RENAISSANCE_H

#include <iostream>
#include <string>

// ============================================================================
// Framework Information
// ============================================================================

#define RENAISSANCE_VERSION_MAJOR 3
#define RENAISSANCE_VERSION_MINOR 0
#define RENAISSANCE_VERSION_PATCH 4
#define RENAISSANCE_VERSION "3.0.4"

#define RENAISSANCE_NAME "renAIssance"
#define RENAISSANCE_FULL_NAME "€/É’ñ¦f`F¶"

// ============================================================================
// Platform Detection
// ============================================================================

#ifdef _WIN32
    #define RENAISSANCE_PLATFORM_WINDOWS
#elif defined(__linux__)
    #define RENAISSANCE_PLATFORM_LINUX
#else
    #define RENAISSANCE_PLATFORM_UNKNOWN
#endif

// ============================================================================
// CUDA Support Detection
// ============================================================================

#ifdef TR_USE_CUDA
    #include <cuda_runtime.h>
    #include <cudnn.h>
    #define RENAISSANCE_CUDA_ENABLED
#endif

// ============================================================================
// Core Namespace
// ============================================================================

namespace renAIssance {

/**
 * @brief Framework initialization class
 */
class Framework {
public:
    /**
     * @brief Initialize the renAIssance framework
     */
    static void initialize() {
        std::cout << "[" << RENAISSANCE_NAME << "] Framework v" << RENAISSANCE_VERSION << " initializing..." << std::endl;
        printSystemInfo();
        std::cout << "[" << RENAISSANCE_NAME << "] Framework initialized successfully!" << std::endl;
    }

    /**
     * @brief Get framework version
     */
    static std::string getVersion() {
        return RENAISSANCE_VERSION;
    }

    /**
     * @brief Get framework name
     */
    static std::string getName() {
        return RENAISSANCE_NAME;
    }

    /**
     * @brief Print welcome message
     */
    static void printWelcome() {
        std::cout << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  Welcome to " << RENAISSANCE_FULL_NAME << std::endl;
        std::cout << "  Version: " << RENAISSANCE_VERSION << std::endl;
        std::cout << "  €/É’z*e" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << std::endl;
    }

private:
    /**
     * @brief Print system information
     */
    static void printSystemInfo() {
#ifdef RENAISSANCE_PLATFORM_WINDOWS
        std::cout << "[" << RENAISSANCE_NAME << "] Platform: Windows" << std::endl;
#elif defined(RENAISSANCE_PLATFORM_LINUX)
        std::cout << "[" << RENAISSANCE_NAME << "] Platform: Linux" << std::endl;
#else
        std::cout << "[" << RENAISSANCE_NAME << "] Platform: Unknown" << std::endl;
#endif

#ifdef RENAISSANCE_CUDA_ENABLED
        std::cout << "[" << RENAISSANCE_NAME << "] CUDA: Enabled" << std::endl;

        // ÀåCUDA¾pÏ
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess && deviceCount > 0) {
            std::cout << "[" << RENAISSANCE_NAME << "] CUDA Devices: " << deviceCount << std::endl;
        } else {
            std::cout << "[" << RENAISSANCE_NAME << "] CUDA Devices: None available" << std::endl;
        }
#else
        std::cout << "[" << RENAISSANCE_NAME << "] CUDA: Disabled" << std::endl;
#endif
    }
};

/**
 * @brief Simple logging utility
 */
class Logger {
public:
    static void info(const std::string& message) {
        std::cout << "[" << RENAISSANCE_NAME << "][INFO] " << message << std::endl;
    }

    static void warn(const std::string& message) {
        std::cout << "[" << RENAISSANCE_NAME << "][WARN] " << message << std::endl;
    }

    static void error(const std::string& message) {
        std::cerr << "[" << RENAISSANCE_NAME << "][ERROR] " << message << std::endl;
    }
};

/**
 * @brief Basic tensor class placeholder
 */
class Tensor {
private:
    std::string name_;

public:
    Tensor(const std::string& name = "tensor") : name_(name) {}

    const std::string& getName() const { return name_; }

    void setName(const std::string& name) { name_ = name; }

    void print() const {
        std::cout << "[" << RENAISSANCE_NAME << "] Tensor '" << name_ << "'" << std::endl;
    }
};

} // namespace renAIssance

#endif // RENAISSANCE_H