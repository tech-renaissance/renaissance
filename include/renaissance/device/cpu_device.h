/**
 * @file cpu_device.h
 * @brief CPU器件实现
 * @version 3.6.4
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/device/device.h"
#include "renaissance/data/tsr_format.h"
#include <vector>
#include <string>

namespace tr {

/**
 * @class CpuDevice
 * @brief CPU器件实现（基于mimalloc）
 */
class CpuDevice final : public Device {
public:
    CpuDevice();
    ~CpuDevice() override;

    // ===== 禁止拷贝和移动 =====
    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(const CpuDevice&) = delete;
    CpuDevice(CpuDevice&&) = delete;
    CpuDevice& operator=(CpuDevice&&) = delete;

    // ===== 器件信息 =====
    DeviceType type() const noexcept override;
    std::string hardware_name() const override;
    bool is_available() const override;
    size_t memory_available() const override;

    // ===== 内存管理（基于mimalloc）=====
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_internal(void* dst, const void* src, size_t size) override;
    void memset_internal(void* ptr, int value, size_t size) override;

    // ===== 张量创建 =====
    Tensor empty(const Shape& shape, DType dtype) override;
    Tensor zeros(const Shape& shape, DType dtype) override;
    Tensor ones(const Shape& shape, DType dtype) override;
    Tensor null_tensor() override;
    void zeros_inplace(Tensor& tensor_a) override;
    void ones_inplace(Tensor& tensor_a) override;

    // ===== 全值填充方法（V3.6.21新增）=====
    Tensor full_fp32(const Shape& shape, float value) override;
    Tensor full_bf16(const Shape& shape, float value) override;
    Tensor full_int32(const Shape& shape, int32_t value) override;
    Tensor full_int8(const Shape& shape, int8_t value) override;
    void full_fp32_inplace(Tensor& tensor_a, float value) override;
    void full_bf16_inplace(Tensor& tensor_a, float value) override;
    void full_int32_inplace(Tensor& tensor_a, int32_t value) override;
    void full_int8_inplace(Tensor& tensor_a, int8_t value) override;

    // ===== 随机数生成（高级接口，调用默认Generator）=====
    Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f,
                  DType dtype = DType::FP32) override;
    void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f,
                         DType dtype = DType::FP32) override;

    Tensor randn(const Shape& shape, float mean = 0.0f, float stddev = 1.0f,
                 DType dtype = DType::FP32) override;
    void randn_inplace(Tensor& tensor_a, float mean = 0.0f, float stddev = 1.0f,
                       DType dtype = DType::FP32) override;

    Tensor randint(const Shape& shape, int low = 0, int high = 10,
                  DType dtype = DType::FP32) override;
    void randint_inplace(Tensor& tensor_a, int low = 0, int high = 10,
                         DType dtype = DType::FP32) override;

    Tensor randbool(const Shape& shape, float rate_of_zeros = 0.5,
                   DType dtype = DType::FP32) override;
    void randbool_inplace(Tensor& tensor_a, float rate_of_zeros = 0.5,
                          DType dtype = DType::FP32) override;

    // ===== 张量运算（加法和复制）=====
    void add_into(const Tensor& a, const Tensor& b, Tensor& result) override;
    void copy_into(const Tensor& tensor_a, Tensor& tensor_b) override;
    void transfer_into(const Tensor& tensor_a, Tensor& tensor_b) override;
    void cast_into(const Tensor& tensor_a, Tensor& tensor_b,
                  StreamType stream = TR_DEFAULT_STREAM) override;
    void trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b,
                        StreamType stream = TR_DEFAULT_STREAM) override;

    // ===== 张量比较 =====
    bool equal(const Tensor& a, const Tensor& b) override;
    bool is_close(const Tensor& a, const Tensor& b, float eps = -1.0f) override;

    // =========================================================================
    // TSR V3 文件导入导出
    // =========================================================================

    /**
     * @brief 导出张量到TSR V3文件
     * @param tensor 要导出的张量（必须在CPU上且已绑定存储）
     * @param filename 目标文件路径
     * @param compress true使用ZLIB压缩模式，false使用RAW快速模式
     * @throws DeviceError 张量不在CPU上
     * @throws ValueError 张量无效或未绑定
     * @throws FileNotFoundError 无法创建文件
     *
     * 模式选择建议：
     * - RAW模式：大模型权重、需要频繁加载的场景
     * - ZLIB模式：稀疏张量、网络传输、磁盘空间受限
     */
    void export_tensor(const Tensor& tensor, const std::string& filename,
                       bool compress = false) const;

    /**
     * @brief 从TSR V3文件导入张量
     * @param filename 源文件路径
     * @param using_mmap 是否使用mmap零拷贝加载（仅RAW模式，默认true）
     * @return 加载到CPU的Tensor对象
     * @throws FileNotFoundError 文件不存在
     * @throws ValueError 文件格式无效或数据损坏
     *
     * 加载策略：
     * - RAW模式：using_mmap=true时使用mmap零拷贝，false时使用常规读取
     * - ZLIB模式：解压到新分配的内存（using_mmap参数无效）
     */
    Tensor import_tensor(const std::string& filename, bool using_mmap = true);

private:
    // =========================================================================
    // TSR辅助方法
    // =========================================================================

    /**
     * @brief 验证TSR文件头有效性
     * @param header 文件头引用
     * @throws ValueError 头部无效时
     */
    void validate_tsr_header(const TSRHeaderV3& header) const;

    /**
     * @brief 计算CRC32校验和
     * @param data 数据指针
     * @param size 数据字节数
     * @return CRC32值
     */
    uint32_t calculate_crc32(const void* data, size_t size) const;

    /**
     * @brief zlib压缩数据
     * @param data 原始数据指针
     * @param size 原始数据大小
     * @return 压缩后的数据向量
     * @throws ValueError 压缩失败时
     */
    std::vector<uint8_t> compress_zlib(const void* data, size_t size) const;

    /**
     * @brief zlib解压数据
     * @param src 压缩数据指针
     * @param src_size 压缩数据大小
     * @param dst 目标缓冲区指针
     * @param dst_size 期望的解压后大小
     * @throws ValueError 解压失败时
     */
    void decompress_zlib(const uint8_t* src, size_t src_size,
                         void* dst, size_t dst_size) const;
};

} // namespace tr
