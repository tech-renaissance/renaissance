/**
 * @file types.h
 * @brief 基础类型定义：Shape、DType、Phase、StreamKind、Region（65个）、PlanConfig、Metric、InputSpec、TrainingResult、OptimizerKind
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: <cstdint>, <string>
 * @note 所属系列: core
 */

#pragma once

#include <cstdint>
#include <string>
#include <initializer_list>

// 跨平台编译器宏定义
#if defined(_MSC_VER)
    // MSVC: 使用 __declspec(noinline)
    #define TR_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: 使用 __attribute__((noinline))
    #define TR_NOINLINE __attribute__((noinline))
#else
    #define TR_NOINLINE
#endif

namespace tr {

/// 通用工具函数
namespace utils {

/**
 * @brief 256字节向上对齐
 * @param size 原始大小
 * @return 对齐后的最小256字节倍数
 */
constexpr inline size_t align_up_256(size_t size) noexcept {
    return (size + 255) & ~static_cast<size_t>(255);
}

} // namespace utils

struct Shape {
    int dims[4] = {1, 1, 1, 1};  // NHWC格式：dims[0]=N, dims[1]=H, dims[2]=W, dims[3]=C

    /**
     * @brief 默认构造函数：创建标量shape [1,1,1,1]
     */
    Shape() = default;

    /**
     * @brief 从4个元素构造Shape [N,H,W,C]
     *
     * 【重要编译器优化问题】
     * 本构造函数必须使用 TR_NOINLINE 和 volatile 写入，原因如下：
     *
     * 1. 问题现象：在MSVC Release模式(/O2 /Ob2 /arch:AVX2)下，普通的初始化列表会被错误优化，
     *    导致 Shape(2,3,4,5) 构造出垃圾值如 [554307520,32762,554307520,32762]
     *
     * 2. 根本原因：编译器在优化简单POD结构体时，可能：
     *    - 错误地使用寄存器传递参数而非栈传递
     *    - 重排序赋值操作导致内存布局错误
     *    - 过度激进地内联和优化构造函数
     *
     * 3. 解决方案：
     *    - TR_NOINLINE: 禁止内联，强制生成函数调用，避免优化错误
     *    - volatile 写入: 强制生成实际的内存写入指令，防止编译器优化赋值顺序
     *
     * 4. 警告：请勿删除 TR_NOINLINE 或 volatile 写入，否则会导致Shape构造失败
     *
     * @bug MSVC /O2 优化bug，影响简单POD结构体的构造函数
     * @date 2026-05-01 技术觉醒团队调试发现并修复
     */
    TR_NOINLINE Shape(int n, int h, int w, int c) : dims{n, h, w, c} {
        // 强制内存写入，防止编译器优化导致的参数传递错误
        // 不能依赖初始化列表 dims{n, h, w, c}，需要显式的volatile写入
        volatile int* ptr = dims;
        ptr[0] = n;
        ptr[1] = h;
        ptr[2] = w;
        ptr[3] = c;
        normalize_dims();
    }

    /**
     * @brief 从初始化列表构造Shape - 修复版本
     * @param list 形状列表，支持1-4个元素，右对齐填充
     *
     * 填充规则（右对齐）：
     * - 1个元素 {C} -> [1,1,1,C]
     * - 2个元素 {W,C} -> [1,1,W,C]
     * - 3个元素 {H,W,C} -> [1,H,W,C]
     * - 4个元素 {N,H,W,C} -> [N,H,W,C]
     */
    Shape(std::initializer_list<int> list) {
        // 先将所有维度初始化为1
        for (int i = 0; i < 4; ++i) {
            dims[i] = 1;
        }

        // 计算起始写入位置（右对齐：从右侧开始填充）
        size_t write_idx = (list.size() < 4) ? (4 - list.size()) : 0;

        // 从list中复制值到dims，严格防止越界
        for (auto it = list.begin(); it != list.end() && write_idx < 4; ++it, ++write_idx) {
            int val = *it;
            // 非正值（包括0和负数）都修正为1
            dims[write_idx] = (val <= 0) ? 1 : val;
        }
    }

    /**
     * @brief 访问第i个维度
     */
    int& operator[](size_t i) { return dims[i]; }
    const int& operator[](size_t i) const { return dims[i]; }

    /**
     * @brief 获取批量大下 N
     */
    int n() const { return dims[0]; }

    /**
     * @brief 获取高度 H
     */
    int h() const { return dims[1]; }

    /**
     * @brief 获取宽度 W
     */
    int w() const { return dims[2]; }

    /**
     * @brief 获取通道数 C
     */
    int c() const { return dims[3]; }

    /**
     * @brief 计算总元素数量 N*H*W*C
     */
    int64_t numel() const noexcept {
        return static_cast<int64_t>(dims[0]) * dims[1] * dims[2] * dims[3];
    }

    /**
     * @brief 转换为字符串表示
     */
    std::string to_string() const {
        return "[" + std::to_string(dims[0]) + "," +
               std::to_string(dims[1]) + "," +
               std::to_string(dims[2]) + "," +
               std::to_string(dims[3]) + "]";
    }

    /**
     * @brief 获取维度数量（总是返回4）
     */
    static constexpr size_t ndim() { return 4; }

    /**
     * @brief 相等比较运算符
     * @param other 另一个Shape
     * @return 两个Shape的所有维度是否完全相同
     */
    bool operator==(const Shape& other) const noexcept {
        return dims[0] == other.dims[0] &&
               dims[1] == other.dims[1] &&
               dims[2] == other.dims[2] &&
               dims[3] == other.dims[3];
    }

    /**
     * @brief 不等比较运算符
     * @param other 另一个Shape
     * @return 两个Shape是否有任何维度不同
     */
    bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

private:
    /**
     * @brief 规范化维度：确保所有维度都是正值
     * - 将任何非正值（<=0）修正为1
     */
    void normalize_dims() {
        for (int i = 0; i < 4; ++i) {
            if (dims[i] <= 0) {
                dims[i] = 1;
            }
        }
    }
};

enum class DType : uint8_t { FP32, FP16, INT8, INT32 };

enum class NormMode : uint8_t { NO_NORM, MLPERF, IMAGENET, MNIST, CIFAR };

/**
 * @brief 归一化预设配置枚举
 *
 * 用于图像归一化预处理操作，定义常用的数据集归一化参数
 */
enum class NormalizePreset : uint8_t {
    NO_NORM,  ///< 不归一化: mean=(0,0,0), std=(1,1,1) — 仅ToTensor（除以255）
    MNIST,    ///< MNIST数据集: 1通道, mean=(0.1307,), std=(0.3081,)
    CIFAR,    ///< CIFAR-10/100数据集: 3通道, mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)
    IMAGENET, ///< ImageNet数据集: 3通道, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
    MLPERF    ///< MLPerf基准测试: 3通道, mean=(123.68/255,116.78/255,103.94/255), std=(1/255,1/255,1/255)
};

// 图模式
enum class GraphMode : uint8_t { TRAIN_FORWARD, TRAIN_BACKWARD, INFERENCE };

// 任务生命周期
enum class Phase : uint8_t { PLANNING, MEMORY_LOCKED, COMPILED };

// 5个物理非阻塞流（STREAM_COMP为逻辑别名，指向COMP_1）
enum class StreamKind : uint8_t { TRANS, COMP_1, COMP_2, COMP_3, UPDATE };

// 优化器类型枚举
enum class OptimizerKind : uint8_t {
    SGD,             ///< 随机梯度下降
    SGD_MOMENTUM,    ///< SGD with Momentum
    SGD_NESTEROV,    ///< SGD with Nesterov momentum
    LARS,            ///< Layer-wise Adaptive Rate Scaling
    LARS_NESTEROV,   ///< LARS with Nesterov momentum
    ADAM,            ///< Adaptive Moment Estimation
    ADAMW            ///< Adam with decoupled weight decay
};

// ============================================================================
// 显存区域铁律（65个Region，低地址→高地址，001-065）
// 基于REGION_FINAL.md V2.9规范，技术觉醒团队2026-05-12
// ============================================================================
enum class Region : uint8_t {
    // B-Series: BN统计量（epoch级生命周期）
    B_PREV_MEAN  = 0,   // 001
    B_PREV_VAR,          // 002
    B_NEXT_MEAN,         // 003
    B_NEXT_VAR,          // 004

    // W-Series: 主模型权重（batch级）
    W_EQ_BIAS,           // 005
    W_EQ_SCALE,          // 006
    W_BN_BIAS,           // 007
    W_BN_WEIGHT,         // 008
    W_FC_BIAS,           // 009
    W_FC_WEIGHT,         // 010
    W_FIRST_CONV,        // 011
    W_DEEP_CONV,         // 012

    // E-Series: EMA权重
    E_BN_BIAS,           // 013
    E_BN_WEIGHT,         // 014
    E_FC_BIAS,           // 015
    E_FC_WEIGHT,         // 016
    E_FIRST_CONV,        // 017
    E_DEEP_CONV,         // 018
    E_FC_WEIGHT_FP16,    // 019
    E_FIRST_CONV_FP16,   // 020
    E_DEEP_CONV_FP16,    // 021

    // A-Series: AMP FP16权重
    A_FC_WEIGHT,         // 022
    A_FIRST_CONV,        // 023
    A_DEEP_CONV,         // 024

    // G-Series: 梯度（FP32+FP16）
    G_BN_BIAS,           // 025  桶2起点
    G_BN_WEIGHT,         // 026  桶2
    G_FC_BIAS,           // 027  桶2
    G_FC_WEIGHT,         // 028  桶2
    G_FIRST_CONV,        // 029  桶2终点
    G_DEEP_CONV,         // 030  桶1

    // R-Series: 结果区
    R_RESULT,            // 031  结果区（FP32 三标量：loss + top1 正确率 + top5 正确率）

    G_FC_WEIGHT_FP16,    // 032
    G_FIRST_CONV_FP16,   // 033
    G_DEEP_CONV_FP16,    // 034

    // M-Series: 一阶动量
    M_BN_BIAS,           // 035
    M_BN_WEIGHT,         // 036
    M_FC_BIAS,           // 037
    M_FC_WEIGHT,         // 038
    M_FIRST_CONV,        // 039
    M_DEEP_CONV,         // 040

    // V-Series: 二阶动量（Adam）
    V_BN_BIAS,           // 041
    V_BN_WEIGHT,         // 042
    V_FC_BIAS,           // 043
    V_FC_WEIGHT,         // 044
    V_FIRST_CONV,        // 045
    V_DEEP_CONV,         // 046

    // N-Series: LARS范数
    N_FC_WEIGHT,         // 047
    N_FIRST_CONV,        // 048
    N_DEEP_CONV,         // 049

    // I-Series: 输入缓冲区
    I_A_LABEL,           // 050
    I_A_DATA,            // 051
    I_B_LABEL,           // 052
    I_B_DATA,            // 053

    // F-Series: 特征图与梯度槽
    F_FEATURE_FP32,      // 054
    F_GRAD_SLOT_FP32,    // 055
    F_FEATURE_FP16,      // 056
    F_GRAD_SLOT_FP16,    // 057

    // S-Series: 标量与掩码
    S_SCALAR_FP32,       // 058
    S_SCALAR_FP16,       // 059
    S_SCALAR_INT32,      // 060
    S_SCALAR_INT8,       // 061
    S_MASK,              // 062

    // T-Series: 临时张量
    T_TEMP_FP32,         // 063
    T_TEMP_FP16,         // 064
    T_TEMP_INT32,        // 065
    T_TEMP_INT8,         // 066

    R_PREDICTED_LABEL,        // 067  推理标签值（[batch] INT32）
    R_RESULT_ACCUMULATED,     // 068  累积结果区（FP32：sum_loss, sum_top1, sum_top5）

    DEFAULT = B_PREV_MEAN,
    NUM_REGIONS = 69
};

// ============================================================================
// MemoryPlan相关数据结构
// ============================================================================
struct PlanConfig {
    bool bn_folded    = true;
    bool use_lars     = false;
    bool use_adam     = false;
    bool use_momentum = true;    // Adam隐含需要M_和V_系列
    bool has_ema      = false;
    int  num_models   = 1;       // 预留，当前固定1
    bool need_mask    = false;
    bool need_temp    = false;
};

// 指标位掩码
enum class Metric : uint32_t {
    NONE       = 0,
    TRAIN_LOSS = 1 << 0,
    VAL_LOSS   = 1 << 1,
    VAL_TOP1   = 1 << 2,
    VAL_TOP5   = 1 << 3,
    EMA_TOP1   = 1 << 4,
    EMA_TOP5   = 1 << 5
};

inline Metric operator|(Metric a, Metric b) {
    return static_cast<Metric>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool has_metric(Metric flags, Metric m) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(m)) != 0;
}

/**
 * @brief TTA（Test Time Augmentation）模式枚举
 *
 * TTA是测试时数据增强技术，通过对输入图像进行多次不同的变换并聚合预测结果，
 * 来提升模型的泛化能力和准确率。
 *
 * 各模式说明：
 * - DISABLED: 禁用TTA，使用常规单次前向传播
 * - LR: 左右翻转（Left-Right），对图像进行水平翻转后再次预测，与原始预测聚合
 * - SHIFT_1PX: 一像素平移（针对MNIST等小图像），进行左移1px、右移1px、上移1px、下移1px四个方向平移，
 *   加上中心位置共5次预测，然后聚合结果
 *
 * @note 当前版本仅提供API支持，具体实现留为TODO（在Task::on_prepare()中实现）
 * @note TTA会增加推理时的计算量，通常用于最终评估和竞赛场景
 */
enum class TTA : uint8_t {
    DISABLED,   ///< 禁用TTA（默认）
    LR,         ///< 左右翻转（Left-Right）
    SHIFT_1PX   ///< 一像素平移（四个方向+中心，适用于MNIST等小图像）
};

struct InputSpec {
    int n = 1, c = 3, h = 224, w = 224;
};

struct TrainingResult {
    float best_top1     = 0.0f;
    float best_top5     = 0.0f;
    float best_ema_top1 = 0.0f;
    float best_ema_top5 = 0.0f;
    int   best_epoch    = -1;

    static TrainingResult debug_stub() { return {}; }
};

} // namespace tr
