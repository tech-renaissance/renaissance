#pragma once

#include <cstdint>
#include <string>
#include <sstream>

namespace tr {

/**
 * @brief 设备类型枚举
 *
 * 支持的设备类型：
 * - CPU: 中央处理器（x86/ARM）
 * - CUDA: NVIDIA GPU（cuDNN）
 * - MUSA: 摩尔线程GPU（muDNN）
 *
 * 设计要点：
 * - 使用 enum class 确保类型安全
 * - 底层类型为 uint8_t，节省内存
 * - 未来可扩展 FPGA、NPU等其他加速器
 */
enum class DeviceKind : uint8_t {
    INVALID = 0,  ///< 无效设备类型
    CPU    = 1,   ///< CPU设备（支持x86/ARM/RISC-V，使用oneDNN或XNNPACK）
    CUDA   = 2,   ///< NVIDIA GPU（使用cuDNN）
    MUSA   = 3    ///< 摩尔线程GPU（使用muDNN）
};

/**
 * @brief CPU架构枚举
 *
 * 用于调试和日志输出，方便识别CPU架构类型。
 * 在需要选择不同后端库时（oneDNN vs XNNPACK）很有用。
 */
enum class Arch : uint8_t {
    UNKNOWN = 0,   ///< 未知架构
    X86_64  = 1,   ///< x86-64架构（AMD64/Intel 64）
    ARM64   = 2,   ///< ARM64架构（AArch64）
    RISCV64 = 3    ///< RISC-V 64位架构
};

/**
 * @brief 设备类型标识类
 *
 * 设备类型是张量的元数据之一，用于标识张量存储在哪个设备上。
 * 框架的核心抽象：Device类根据DeviceType创建对应设备，并在该设备上分配和操作张量。
 *
 * 设计要点：
 * - 8字节POD类型（Plain Old Data），可直接memcpy
 * - 使用位域保持8字节总大小（kind: 8bit, arch: 8bit, reserved: 16bit, index: 32bit）
 * - 设备索引用于多设备场景（如多GPU、多CPU NUMA节点）
 * - Arch字段用于调试和识别CPU架构（x86/ARM/RISC-V）
 * - 默认为CPU:0（主CPU设备）
 *
 * 内存布局：
 * - kind_ (1字节位域): 设备类型枚举
 * - arch_ (1字节位域): CPU架构类型
 * - reserved_ (2字节位域): 保留字段
 * - index_ (4字节): 设备索引（如GPU ID）
 * - 总计：8字节
 *
 * 性能优化：
 * - POD类型可按值传递，无堆内存开销
 * - constexpr构造函数支持编译期初始化
 * - 内联比较函数，零抽象开销
 */
class DeviceType {
public:
    /**
     * @brief 默认构造函数
     * @return 默认设备（CPU:0，未知架构）
     *
     * 默认使用主CPU设备，兼容现有代码
     */
    constexpr DeviceType() noexcept
        : kind_(static_cast<uint8_t>(DeviceKind::CPU)),
          arch_(static_cast<uint8_t>(Arch::UNKNOWN)),
          reserved_(0),
          index_(0) {
        static_assert(sizeof(DeviceType) == 8, "DeviceType must be exactly 8 bytes");
    }

    /**
     * @brief 构造函数（指定设备类型）
     * @param kind 设备类型枚举
     * @param index 设备索引（默认为0）
     * @param arch CPU架构（默认为UNKNOWN，仅对CPU有效）
     *
     * @note index必须有效（如GPU ID必须小于设备总数）
     */
    constexpr DeviceType(DeviceKind kind, uint32_t index = 0, Arch arch = Arch::UNKNOWN) noexcept
        : kind_(static_cast<uint8_t>(kind)),
          arch_(static_cast<uint8_t>(arch)),
          reserved_(0),
          index_(index) {
        static_assert(sizeof(DeviceType) == 8, "DeviceType must be exactly 8 bytes");
    }

    /**
     * @brief 获取设备类型
     * @return 设备类型枚举
     */
    constexpr DeviceKind kind() const noexcept {
        return static_cast<DeviceKind>(kind_);
    }

    /**
     * @brief 获取设备索引
     * @return 设备索引（如GPU ID）
     */
    constexpr uint32_t index() const noexcept {
        return index_;
    }

    /**
     * @brief 获取CPU架构
     * @return CPU架构枚举
     *
     * @note 仅对CPU设备有效，GPU设备返回UNKNOWN
     */
    constexpr Arch arch() const noexcept {
        return static_cast<Arch>(arch_);
    }

    /**
     * @brief 判断是否为CPU设备
     * @return true表示CPU设备，false表示其他设备
     */
    constexpr bool is_cpu() const noexcept {
        return kind_ == static_cast<uint8_t>(DeviceKind::CPU);
    }

    /**
     * @brief 判断是否为CUDA设备
     * @return true表示NVIDIA GPU，false表示其他设备
     */
    constexpr bool is_cuda() const noexcept {
        return kind_ == static_cast<uint8_t>(DeviceKind::CUDA);
    }

    /**
     * @brief 判断是否为MUSA设备
     * @return true表示摩尔线程GPU，false表示其他设备
     */
    constexpr bool is_musa() const noexcept {
        return kind_ == static_cast<uint8_t>(DeviceKind::MUSA);
    }

    /**
     * @brief 判断是否为GPU设备（CUDA或MUSA）
     * @return true表示GPU设备，false表示其他设备
     */
    constexpr bool is_gpu() const noexcept {
        return is_cuda() || is_musa();
    }

    /**
     * @brief 相等比较运算符
     * @param other 另一个设备类型
     * @return true表示设备类型和索引都相同
     */
    constexpr bool operator==(const DeviceType& other) const noexcept {
        return kind_ == other.kind_ && index_ == other.index_;
    }

    /**
     * @brief 不等比较运算符
     * @param other 另一个设备类型
     * @return true表示设备类型或索引不同
     */
    constexpr bool operator!=(const DeviceType& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief 获取设备类型名称（包含索引和架构信息）
     * @return 设备类型字符串（如"CPU:0:x86_64"、"CUDA:1"）
     *
     * 格式：
     * - CPU设备："{TYPE}:{index}:{arch}"（如"CPU:0:x86_64"）
     * - GPU设备："{TYPE}:{index}"（如"CUDA:0"、"MUSA:1"）
     *
     * @note 返回std::string确保线程安全（不支持返回静态缓冲区）
     */
    std::string to_string() const {
        std::ostringstream oss;
        if (is_cpu()) {
            oss << "CPU:" << index_;
            // CPU设备添加架构信息
            switch (static_cast<Arch>(arch_)) {
                case Arch::X86_64:
                    oss << ":x86_64";
                    break;
                case Arch::ARM64:
                    oss << ":arm64";
                    break;
                case Arch::RISCV64:
                    oss << ":riscv64";
                    break;
                default:
                    oss << ":unknown";
                    break;
            }
        } else if (is_cuda()) {
            oss << "CUDA:" << index_;
        } else if (is_musa()) {
            oss << "MUSA:" << index_;
        } else {
            oss << "INVALID:" << index_;
        }
        return oss.str();
    }

    /**
     * @brief 创建CPU设备类型
     * @param index CPU索引（默认为0，用于NUMA节点）
     * @param arch CPU架构（默认为UNKNOWN）
     * @return CPU设备类型
     */
    static constexpr DeviceType cpu(uint32_t index = 0, Arch arch = Arch::UNKNOWN) noexcept {
        return DeviceType(DeviceKind::CPU, index, arch);
    }

    /**
     * @brief 创建CUDA设备类型
     * @param index GPU ID（必须小于CUDA设备总数）
     * @return CUDA设备类型
     */
    static constexpr DeviceType cuda(uint32_t index = 0) noexcept {
        return DeviceType(DeviceKind::CUDA, index);
    }

    /**
     * @brief 创建MUSA设备类型
     * @param index GPU ID（必须小于MUSA设备总数）
     * @return MUSA设备类型
     */
    static constexpr DeviceType musa(uint32_t index = 0) noexcept {
        return DeviceType(DeviceKind::MUSA, index);
    }

private:
    uint8_t kind_;       ///< 设备类型（1字节）
    uint8_t arch_;       ///< CPU架构（1字节）
    uint16_t reserved_;  ///< 保留字段（2字节）
    uint32_t index_;     ///< 设备索引（4字节）
    // 总计：8字节（无padding）
};

/**
 * @brief 默认设备类型
 *
 * 全局默认设备，用于简化API调用。
 * 例如：Device::ones({2, 3}) 使用默认设备创建张量。
 */
inline constexpr DeviceType kDefaultDevice = DeviceType::cpu(0);

// ==================== 下标语法支持 ====================

/**
 * @brief CUDA设备数组代理类
 *
 * 提供 tr::CUDA[i] 下标语法，与PyTorch等先进框架接轨。
 *
 * 使用示例：
 * - tr::CUDA[0] → DeviceType::cuda(0)
 * - tr::CUDA[1] → DeviceType::cuda(1)
 */
struct CudaDeviceArray {
    /**
     * @brief 下标运算符
     * @param index GPU ID
     * @return CUDA设备类型
     */
    constexpr DeviceType operator[](int index) const noexcept {
        return DeviceType::cuda(static_cast<uint32_t>(index));
    }
};

/**
 * @brief MUSA设备数组代理类
 *
 * 提供 tr::MUSA[i] 下标语法。
 *
 * 使用示例：
 * - tr::MUSA[0] → DeviceType::musa(0)
 * - tr::MUSA[1] → DeviceType::musa(1)
 */
struct MusaDeviceArray {
    /**
     * @brief 下标运算符
     * @param index GPU ID
     * @return MUSA设备类型
     */
    constexpr DeviceType operator[](int index) const noexcept {
        return DeviceType::musa(static_cast<uint32_t>(index));
    }
};

/**
 * @brief CUDA设备数组全局常量
 *
 * 提供 tr::CUDA[i] 语法。
 *
 * @note 使用方法：auto device = tr::CUDA[0];
 */
inline constexpr CudaDeviceArray CUDA{};

/**
 * @brief MUSA设备数组全局常量
 *
 * 提供 tr::MUSA[i] 语法。
 *
 * @note 使用方法：auto device = tr::MUSA[0];
 */
inline constexpr MusaDeviceArray MUSA{};

} // namespace tr
