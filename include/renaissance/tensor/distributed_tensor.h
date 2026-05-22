/**
 * @file distributed_tensor.h
 * @brief 分布式张量 — "一张图纸，八卡共享"的内存抽象
 * @version 4.20.2
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/core/types.h
 * @note 所属系列: tensor
 * @note 关键修改：slot_bytes_ 与 shape 解耦为构造时常量，支持跨变体 offset 一致
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/core/init_config.h"
#include "renaissance/core/tr_exception.h"

namespace tr {

constexpr inline int64_t align_up(int64_t x, int64_t alignment) noexcept {
    return (x + alignment - 1) / alignment * alignment;
}

/**
 * @brief 分布式张量 — "一张图纸，八卡共享"的内存抽象
 *
 * DTensor 是一个纯虚拟概念：只存形状/偏移量/stride，不持有内存、不存指针。
 * 同一个DTensor指代所有卡上相同的物理内存区域，布局完全相同，但数据可以有别。
 *
 * 职责分离：
 *   DTensor       — 描述单张量在多卡上的统一内存视图（本文件）
 *   MemoryPlan    — 分配 DTensor、计算偏移、确保无间隙和 256B 对齐
 *   DeviceContext — 将 DTensor.offset 解析为真实 GPU/CPU 指针
 *
 * 关键约定：
 *   - 全部 4D NHWC 物理布局，Shape 存逻辑维度（padding前）
 *   - 首地址必定 256 字节对齐，slot_bytes() 必定为 256 的整数倍
 *   - 创建后 id/shape/dtype/region 不可变
 *   - offset_ 由 MemoryPlan::finalize 赋值，此前恒为 -1（哨兵）
 *   - stride 由 shape + dtype 计算，创建后不可变
 *
 * 多变体设计（V4.20.2）：
 *   - slot_bytes_ 与 shape 解耦：跨变体取 max 保证 offset 一致
 *   - stride 严格按自身 shape 计算，不存在 "max shape" 概念
 *   - 双构造函数：标准（自动算 slot_bytes）+ 变体（显式传入 max_slot_bytes）
 *   - compute_slot_bytes() 为静态纯函数，不依赖 this
 */
struct DistributedTensor {
    int32_t  id      = -1;
    Shape    shape;                   // [N, H, W, C] 逻辑维度（该变体唯一权威）
    DType    dtype   = DType::FP32;
    Region   region  = Region::DEFAULT;
    // 快速访问成员变量（紧跟shape变化，不可单独赋值）
    int32_t  n_ = 1;                 // shape.n()的缓存值，构造/赋值时自动同步
    int32_t  h_ = 1;                 // shape.h()的缓存值
    int32_t  w_ = 1;                 // shape.w()的缓存值
    int32_t  c_ = 1;                 // shape.c()的缓存值

    /**
     * @brief 内部方法：shape变化时自动同步n_/h_/w_/c_
     * @note 仅在MemoryPlan等内部模块中调用，外部不应直接修改shape
     */
    void sync_shape_from(const Shape& s) noexcept {
        shape = s;
        n_ = s.n();
        h_ = s.h();
        w_ = s.w();
        c_ = s.c();
    }

    InitConfig init_config;          // 初始化配置（V4.20.3 新增）

    // ====================
    // stride getter — 显式区分 CUDA 对齐 stride 与 CPU 紧凑 stride
    // ====================

    /**
     * @brief CUDA 对齐 stride — 由 padded_c()（cuda_alignment）推导
     *
     * 非紧凑时（FP16+I_A_DATA等），c_stride_cuda=1, w_stride_cuda=padded_c(>C),
     * h_stride_cuda=padded_c*W, n_stride_cuda=padded_c*W*H。
     * 仅 CUDA 路径使用。CPU 路径不应调用此系列——CPU 上 DTensor 必定紧凑。
     */
    int64_t n_stride_cuda() const noexcept { return n_stride_cuda_; }
    int64_t h_stride_cuda() const noexcept { return h_stride_cuda_; }
    int64_t w_stride_cuda() const noexcept { return w_stride_cuda_; }
    int64_t c_stride_cuda() const noexcept { return c_stride_cuda_; }

    /**
     * @brief CPU 紧凑 stride — 框架保证 CPU 上所有 DTensor 必定紧凑
     *
     * 恒为 H*W*C / W*C / C / 1，与 cuda_alignment 无关。
     * CPU capture、CPU 算子、oneDNN 路径使用此系列。
     */
    int64_t n_stride_cpu() const noexcept { return n_stride_cpu_; }
    int64_t h_stride_cpu() const noexcept { return h_stride_cpu_; }
    int64_t w_stride_cpu() const noexcept { return w_stride_cpu_; }
    int64_t c_stride_cpu() const noexcept { return c_stride_cpu_; }

    // ====================
    // 基本属性
    // ====================

    /**
     * @brief 获取N维度（批次大小）
     * @return shape.n()的缓存值，提供O(1)访问且避免shape对象构造
     * @note n_成员变量严格跟随shape变化，shape赋值时自动同步，不可单独修改
     */
    int32_t n() const noexcept { return n_; }

    /**
     * @brief 获取H维度（高度）
     * @return shape.h()的缓存值
     * @note h_成员变量严格跟随shape变化，shape赋值时自动同步，不可单独修改
     */
    int32_t h() const noexcept { return h_; }

    /**
     * @brief 获取W维度（宽度）
     * @return shape.w()的缓存值
     * @note w_成员变量严格跟随shape变化，shape赋值时自动同步，不可单独修改
     */
    int32_t w() const noexcept { return w_; }

    /**
     * @brief 获取C维度（通道数）
     * @return shape.c()的缓存值
     * @note c_成员变量严格跟随shape变化，shape赋值时自动同步，不可单独修改
     */
    int32_t c() const noexcept { return c_; }

    int64_t numel() const noexcept { return shape.numel(); }

    /**
     * @brief 有效数据的逻辑字节数（不含 padding）
     *
     * 恒等关系：nbytes = N × H × W × C × sizeof(dtype)
     * 用于数据拷贝、CRC 校验等需要精确数据长度的场景。
     * 与 slot_bytes() 不同：slot_bytes 含 C 通道填充 + 16B 预留 + 256B 对齐。
     */
    uint64_t nbytes() const noexcept {
        return static_cast<uint64_t>(numel()) * dsize(dtype);
    }

    bool valid() const noexcept { return id >= 0; }

    /**
     * @brief 判断该 DTensor 是否为紧凑布局
     * @return true 当且仅当 padded_c == shape.c()
     *
     * 紧凑的 DTensor 与 Tensor 具有完全相同的内存排布，可直接 memcpy。
     * 非紧凑时（FP16 + 特定 Region，且 c 不是 alignment 的倍数），
     * C 通道有 padding，需要 layout conversion。
     */
    bool is_compact() const noexcept {
        return padded_c() == shape.c();
    }

    // ====================
    // 对齐推导
    // ====================

    /**
     * @brief C 通道对齐因子（1/4/8）
     *
     * 完全由 dtype + region 决定，不依赖运行时环境：
     *   - 非 FP16：一律 1（紧凑）
     *   - FP16 + 输入缓冲区(I_A_DATA/I_B_DATA)：4
     *   - FP16 + 特征图区(F_FEATURE_FP16/F_GRAD_SLOT_FP16)（仅CUDA）：8
     *   - 其余 FP16 情形：1
     *
     * CPU 场景下 cuda_alignment 永远为 1（REGION_FINAL.md #19）。
     */
    uint8_t cuda_alignment() const noexcept {
        if (dtype == DType::FP16) {
            switch (region) {
                case Region::I_A_DATA:
                case Region::I_B_DATA:  return 4;
                case Region::F_FEATURE_FP16:
                case Region::F_GRAD_SLOT_FP16:
#ifdef TR_USE_CUDA
                    return 8;
#else
                    return 1;
#endif
                default: return 1;
            }
        }
        if (dtype == DType::INT8 && region == Region::S_MASK) {
#ifdef TR_USE_CUDA
            return 8;
#else
            return 1;
#endif
        }
        return 1;
    }

    /**
     * @brief padding 之后的 C 通道元素数
     *
     * padded_c = align_up(C, cuda_alignment)
     * w_stride / h_stride / n_stride 均基于此值。
     */
    int64_t padded_c() const noexcept {
        return align_up(static_cast<int64_t>(shape.c()),
                        static_cast<int64_t>(cuda_alignment()));
    }

    // ====================
    // 填充元素数
    // ====================

    /**
     * @brief padding 之后的总元素数（不含 dtype 转换）
     *
     * padded_elems = N × H × W × padded_c
     *
     * 用场：cuDNN setDim 维度传入、slot_bytes 的 FP16 基准计算、stride 逻辑校验。
     */
    uint64_t padded_elems() const noexcept {
        return static_cast<uint64_t>(shape.n()) * shape.h() * shape.w()
               * static_cast<uint64_t>(padded_c());
    }

    // ====================
    // 填充字节数（MP.md #12 三别名同义）
    // ====================

    /**
     * @brief padding 之后的总字节数
     *
     * padded_bytes = N × H × W × padded_c × sizeof(dtype)
     *
     * 用场：cuDNN workspace 大小、CUDA 显存需求估算、slot_bytes 计算输入。
     */
    uint64_t padded_bytes() const noexcept {
        return padded_elems() * dsize(dtype);
    }

    /// @brief 同 padded_bytes()，对接 cuDNN API 时的推荐方法名
    uint64_t cudnn_bytes() const noexcept { return padded_bytes(); }

    /// @brief 同 padded_bytes()，对接 CUDA API 时的推荐方法名
    uint64_t cuda_bytes()  const noexcept { return padded_bytes(); }

    // ====================
    // slot_bytes — 物理槽位大小（V4.20.2 核心修改）
    // ====================

    /**
     * @brief MemoryPlan 划分该 DTensor 所需的最小字节槽位
     *
     * V4.20.2 关键修改：slot_bytes() 不再根据当前 shape/dtype 实时计算，
     * 而是返回构造时存储的常数 slot_bytes_。
     *
     * 这保证了跨变体的 offset 一致性：
     *   - 标准构造：slot_bytes_ = compute_slot_bytes(shape, dtype, region)
     *   - 变体构造：slot_bytes_ = Compiler 传入的 max_slot_bytes
     *
     * 公式（不变）：
     *   FP16/INT8:   slot = align_up_256(padded_bytes() + 16)
     *   FP32/INT32:  slot = 2 × align_up_256(FP16等效padded_bytes + 16)
     */
    uint64_t slot_bytes() const noexcept { return slot_bytes_; }

    /**
     * @brief 静态纯函数 —— 计算指定 Shape/DType/Region 对应的槽位字节数
     *
     * 不依赖 this，可供 MemoryPlan 和 Compiler 在分配前任意调用。
     * Phase 2（compute_max_slot_bytes）的核心依赖。
     */
    static uint64_t compute_slot_bytes(const Shape& shape, DType dtype, Region region) noexcept {
        // 重用 cuda_alignment 逻辑：从 region + dtype 推断 padded_c
        uint8_t alignment = 1;
        if (dtype == DType::INT8 && region == Region::S_MASK) {
#ifdef TR_USE_CUDA
            alignment = 8;
#else
            alignment = 1;
#endif
        } else if (dtype == DType::FP16) {
            switch (region) {
                case Region::I_A_DATA:
                case Region::I_B_DATA:  alignment = 4; break;
                case Region::F_FEATURE_FP16:
                case Region::F_GRAD_SLOT_FP16:
#ifdef TR_USE_CUDA
                    alignment = 8; break;
#else
                    alignment = 1; break;
#endif
                default: alignment = 1; break;
            }
        }
        int64_t pc = align_up(static_cast<int64_t>(shape.c()),
                              static_cast<int64_t>(alignment));
        uint64_t elems = static_cast<uint64_t>(shape.n()) * shape.h() * shape.w()
                         * static_cast<uint64_t>(pc);

        if (dtype == DType::FP16) {
            return utils::align_up_256(elems * 2 + 16);  // FP16: 2 bytes per element
        } else if (dtype == DType::INT8) {
            return utils::align_up_256(elems * 1 + 16);  // INT8: 1 byte per element
        } else {
            return 2 * utils::align_up_256(elems * 2 + 16);  // FP32 基准 = 2×FP16
        }
    }

    // ====================
    // cuDNN 消歧义接口（MP.md #11）
    // ====================

    /**
     * @brief 以 cuDNN 的 NCHW API 顺序返回逻辑维度
     *
     * dim_index: 0→N, 1→C, 2→H, 3→W
     *
     * 关键：cuDNN API 期望 NCHW 顺序，但物理布局是 NHWC。
     * 此方法填正确值即可——cuDNN 通过 cudnn_stride(1)==1 自动识别 NHWC。
     */
    int64_t cudnn_dim(int dim_index) const noexcept {
        switch (dim_index) {
            case 0: return shape.n();
            case 1: return shape.c();
            case 2: return shape.h();
            case 3: return shape.w();
            default: return 0;
        }
    }

    /**
     * @brief 以 cuDNN 的 NCHW API 顺序返回 stride
     *
     * dim_index: 0→n_stride, 1→c_stride, 2→h_stride, 3→w_stride
     *
     * cudnn_stride(1) 返回 c_stride=1——cuDNN 由此知道 C 在最内层 = NHWC。
     */
    int64_t cudnn_stride(int dim_index) const noexcept {
        switch (dim_index) {
            case 0: return n_stride_cuda_;
            case 1: return c_stride_cuda_;
            case 2: return h_stride_cuda_;
            case 3: return w_stride_cuda_;
            default: return 0;
        }
    }

    friend class MemoryPlan;  // offset_ 的唯一写入者

    /**
     * @brief 获取 MemoryPlan 中的字节偏移（finalize 后才能访问）
     */
    uint64_t offset() const {
        TR_DEBUG_CHECK(offset_ >= 0, RuntimeError,
                 "DTensor offset accessed before MemoryPlan::finalize: id=" << id);
        return static_cast<uint64_t>(offset_);
    }

    // ====================
    // 构造（V4.20.2 双构造函数）
    // ====================

    DistributedTensor() = default;

    /**
     * @brief 标准构造函数 —— slot_bytes 从 shape 自动推导
     *
     * @param i  全局 ID（由 MemoryPlan 分配）
     * @param s  逻辑形状 [N, H, W, C]
     * @param d  数据类型
     * @param r  所在显存区域（决定 cuda_alignment）
     *
     * slot_bytes_ 由 compute_slot_bytes(s, d, r) 自动计算。
     * 适用于 base 变体 MemoryPlan 的第一次分配。
     */
    DistributedTensor(int32_t i, Shape s, DType d, Region r)
        : id(i), shape(s), dtype(d), region(r),
          n_(s.n()), h_(s.h()), w_(s.w()), c_(s.c()) {
        slot_bytes_ = compute_slot_bytes(s, d, r);
        int64_t ac = padded_c();
        c_stride_cuda_ = 1;
        w_stride_cuda_ = ac;
        h_stride_cuda_ = ac * s.w();
        n_stride_cuda_ = ac * s.w() * s.h();
        c_stride_cpu_ = 1;
        w_stride_cpu_ = s.c();
        h_stride_cpu_ = s.w() * s.c();
        n_stride_cpu_ = s.h() * s.w() * s.c();
    }

    /**
     * @brief 变体构造函数 —— slot_bytes 由 Compiler 显式传入
     *
     * @param i  全局 ID
     * @param s  逻辑形状（该变体的实际 shape，用于 stride 推导）
     * @param d  数据类型
     * @param r  所在显存区域
     * @param sb 跨变体最大 slot_bytes（保证 offset 一致）
     *
     * slot_bytes_ = sb（不是从 shape 推导！）
     * stride 仍按 s（自身 shape）计算——与 slot_bytes_ 无关。
     *
     * 仅 Compiler 通过 MemoryPlan 的私有 alloc 重载调用。
     */
    DistributedTensor(int32_t i, Shape s, DType d, Region r, uint64_t sb)
        : id(i), shape(s), dtype(d), region(r),
          n_(s.n()), h_(s.h()), w_(s.w()), c_(s.c()),
          slot_bytes_(sb) {
        int64_t ac = padded_c();
        c_stride_cuda_ = 1;
        w_stride_cuda_ = ac;
        h_stride_cuda_ = ac * s.w();
        n_stride_cuda_ = ac * s.w() * s.h();
        c_stride_cpu_ = 1;
        w_stride_cpu_ = s.c();
        h_stride_cpu_ = s.w() * s.c();
        n_stride_cpu_ = s.h() * s.w() * s.c();
    }

    static constexpr size_t dsize(DType dt) noexcept {
        switch (dt) {
            case DType::INT8:  return 1;
            case DType::FP16:  return 2;
            case DType::FP32: return 4;
            case DType::INT32: return 4;
        }
        return 4;
    }

private:
    int64_t offset_ = -1;  ///< -1 = 未 finalize 哨兵值，finalize 后 >= 0
    uint64_t slot_bytes_ = 0;  ///< 构造时常量，与 shape 解耦，跨变体取 max

    /// CUDA 对齐 stride（由 padded_c 推导，非紧凑时有 padding）
    int64_t n_stride_cuda_ = 0;
    int64_t h_stride_cuda_ = 0;
    int64_t w_stride_cuda_ = 0;
    int64_t c_stride_cuda_ = 0;

    /// CPU 紧凑 stride（框架保证 CPU 上 DTensor 必定紧凑，恒为 HWC/WC/C/1）
    int64_t n_stride_cpu_ = 0;
    int64_t h_stride_cpu_ = 0;
    int64_t w_stride_cpu_ = 0;
    int64_t c_stride_cpu_ = 0;
};

using DTensor = DistributedTensor;

}  // namespace tr