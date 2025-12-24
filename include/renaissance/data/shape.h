#pragma once

#include <array>
#include <cstdint>
#include "renaissance/base/tr_exception.h"

namespace tr {

/**
 * @brief 张量形状类（NHWC语义）
 *
 * Shape类描述张量的维度信息，是Tensor的元数据之一。
 * 采用原生NHWC（Batch, Height, Width, Channel）布局，支持1-4维张量。
 *
 * 设计要点：
 * - 右对齐存储：1D→[0,0,0,C], 2D→[0,0,H,W], 3D→[0,H,W,C], 4D→[N,H,W,C]
 * - 内存占用20字节（16字节数组 + 4字节ndim）
 * - NHWC语义与硬件对齐（AVX2/AVX-512、Tensor Core）
 * - 2D张量始终表示(H, W)，不表示(N, C)，避免歧义
 *
 * 右对齐优势：
 * - 形状语义清晰：n()/h()/w()/c()返回值与实际维度对应
 * - 避免PyTorch的歧义：torch.Size([2])可能代表(N)或(C)，需要额外信息
 * - 便于实现卷积、池化等操作：最后一个维度始终是Channel
 *
 * 内存布局：
 * - dims_: std::array<int32_t, 4>，右对齐存储
 * - ndim_: 实际维度数（1-4），0表示标量
 *
 * 性能优化：
 * - constexpr构造函数，支持编译期初始化
 * - 内联访问函数，零抽象开销
 * - numel()缓存计算结果（如果需要）
 */
class Shape {
public:
    /**
     * @brief 默认构造函数（标量）
     * @return 标量形状（ndim=0，dims=[0,0,0,0]）
     *
     * 标量表示0维张量，如损失值、标量参数等。
     */
    constexpr Shape() noexcept : dims_{0, 0, 0, 0}, ndim_(0) {}

    /**
     * @brief 1D构造函数（仅通道维度）
     * @param c 通道数
     * @return 1D形状，语义为(,C)，右对齐存储为[0,0,0,C]
     *
     * 使用场景：BatchNorm的参数（scale、bias）、特征图展平后的特征向量等。
     */
    explicit constexpr Shape(int32_t c) noexcept
        : dims_{0, 0, 0, c}, ndim_(1) {}

    /**
     * @brief 2D构造函数（高度、宽度）
     * @param h 高度（Height）
     * @param w 宽度（Width）
     * @return 2D形状，语义为(H,W)，右对齐存储为[0,0,H,W]
     *
     * 使用场景：特征图（无batch、无通道）、2D卷积核权重等。
     *
     * @note 2D始终表示(H,W)，不表示(N,C)，避免歧义。
     */
    constexpr Shape(int32_t h, int32_t w) noexcept
        : dims_{0, 0, h, w}, ndim_(2) {}

    /**
     * @brief 3D构造函数（高度、宽度、通道数）
     * @param h 高度（Height）
     * @param w 宽度（Width）
     * @param c 通道数（Channel）
     * @return 3D形状，语义为(H,W,C)，右对齐存储为[0,H,W,C]
     *
     * 使用场景：单张特征图（无batch）、3D卷积核权重等。
     */
    constexpr Shape(int32_t h, int32_t w, int32_t c) noexcept
        : dims_{0, h, w, c}, ndim_(3) {}

    /**
     * @brief 4D构造函数（批量、高度、宽度、通道数）
     * @param n 批量大小（Number/Batch）
     * @param h 高度（Height）
     * @param w 宽度（Width）
     * @param c 通道数（Channel）
     * @return 4D形状，语义为(N,H,W,C)，存储为[N,H,W,C]
     *
     * 使用场景：输入图像batch、卷积层输出特征图等。
     *
     * @note 这是深度学习中最常见的形状。
     */
    constexpr Shape(int32_t n, int32_t h, int32_t w, int32_t c) noexcept
        : dims_{n, h, w, c}, ndim_(4) {}

    /**
     * @brief 获取维度数
     * @return 实际维度数（0-4）
     */
    constexpr int32_t ndim() const noexcept {
        return ndim_;
    }

    /**
     * @brief 获取第i维的大小（内联实现以优化性能）
     * @param i 维度索引（0-based，支持负索引）
     * @return 维度大小
     *
     * @note i为0时返回第一个有效维度（根据ndim确定）
     * @note 支持Python风格负索引：dim(-1)返回最后一维
     * @note 越界访问会抛出异常
     */
    constexpr int32_t dim(int32_t i) const {
        // 支持负索引（Python风格）
        if (i < 0) {
            i += ndim_;
        }

        // 边界检查
        if (i < 0 || i >= ndim_) {
            TR_THROW(IndexError, "Shape index out of bounds: index=", i, ", ndim=", ndim_);
        }

        // 根据右对齐规则返回对应维度
        return dims_[4 - ndim_ + i];
    }

    /**
     * @brief 获取批量维度（N）
     * @return 批量大小，非4D张量返回1
     *
     * NHWC语义中的N（Number/Batch）维度。
     */
    constexpr int32_t n() const noexcept {
        return (ndim_ == 4) ? dims_[0] : 1;
    }

    /**
     * @brief 获取高度维度（H）
     * @return 高度，非2D以上张量返回1
     *
     * NHWC语义中的H（Height）维度。
     */
    constexpr int32_t h() const noexcept {
        if (ndim_ == 4) return dims_[1];
        if (ndim_ == 3) return dims_[1];
        if (ndim_ == 2) return dims_[2];
        return 1;
    }

    /**
     * @brief 获取宽度维度（W）
     * @return 宽度，非2D以上张量返回1
     *
     * NHWC语义中的W（Width）维度。
     */
    constexpr int32_t w() const noexcept {
        if (ndim_ == 4) return dims_[2];
        if (ndim_ == 3) return dims_[2];
        if (ndim_ == 2) return dims_[3];
        return 1;
    }

    /**
     * @brief 获取通道维度（C）
     * @return 通道数，非3D以上张量返回1
     *
     * NHWC语义中的C（Channel）维度。
     * - 4D: dims_[3] = C
     * - 3D: dims_[3] = C
     * - 2D: 无通道维度，返回1
     * - 1D: dims_[3] = C
     */
    constexpr int32_t c() const noexcept {
        if (ndim_ == 4) return dims_[3];
        if (ndim_ == 3) return dims_[3];
        if (ndim_ == 1) return dims_[3];
        return 1;  // 2D或标量
    }

    /**
     * @brief 获取元素总数
     * @return 所有维度的乘积（标量返回1）
     *
     * 示例：
     * - Shape(2,3) → 6
     * - Shape(2,3,4) → 24
     * - Shape(2,3,4,5) → 120
     */
    constexpr int64_t numel() const noexcept {
        int64_t total = 1;
        for (int32_t i = 0; i < ndim_; ++i) {
            total *= dims_[4 - ndim_ + i];
        }
        return total;
    }

    /**
     * @brief 相等比较运算符
     * @param other 另一个形状
     * @return true表示所有维度都相同
     */
    constexpr bool operator==(const Shape& other) const noexcept {
        return ndim_ == other.ndim_ && dims_ == other.dims_;
    }

    /**
     * @brief 不等比较运算符
     * @param other 另一个形状
     * @return true表示任意维度不同
     */
    constexpr bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

    // ==================== 形状推断工具函数 ====================

    /**
     * @brief 计算卷积输出形状
     * @param input 输入形状（4D或3D）
     * @param kernel_size 卷积核大小（正方形）
     * @param out_channels 输出通道数（卷积核个数）
     * @param stride 步长（默认为1）
     * @param padding 填充（默认为0）
     * @return 输出形状
     *
     * 公式：output = (input - kernel + 2*padding) / stride + 1
     *
     * 示例：
     * - input: Shape(1,28,28,3), kernel: 5, out_channels: 16
     *   → output: Shape(1,24,24,16) （假设padding=0, stride=1）
     */
    static Shape conv_output_shape(const Shape& input, int32_t kernel_size,
                                    int32_t out_channels, int32_t stride = 1, int32_t padding = 0);

    /**
     * @brief 计算池化输出形状
     * @param input 输入形状（4D或3D）
     * @param kernel_size 池化窗口大小（默认为2）
     * @param stride 步长（默认为2，与kernel_size相同）
     * @return 输出形状
     *
     * 公式：output = (input - kernel) / stride + 1
     *
     * 示例：
     * - input: Shape(1,56,56,64), kernel: 2, stride: 2
     *   → output: Shape(1,28,28,64)
     */
    static Shape pool_output_shape(const Shape& input, int32_t kernel_size = 2, int32_t stride = 2);

    /**
     * @brief 计算全局平均池化输出形状
     * @param input 输入形状（4D或3D）
     * @return 输出形状（2D：(1,C)或1D：(C)）
     *
     * 全局平均池化将H×W维度池化为1×1，输出形状为(1,C)或。
     * 用于分类网络的最终特征聚合。
     *
     * 示例：
     * - input: Shape(1,7,7,512) → output: Shape(512)
     * - input: Shape(1,7,7,512) （4D） → output: Shape(1,512) （2D）
     */
    static Shape gap_output_shape(const Shape& input);

    /**
     * @brief 计算全连接层输出形状
     * @param input 输入形状（任意维度）
     * @param out_features 输出特征数
     * @return 输出形状（2D：(N,out_features)或1D：(out_features)）
     *
     * 全连接层将输入展平为一维向量，然后线性映射到out_features维。
     * 如果输入有batch维度(N)，输出为(N, out_features)；否则为。
     *
     * 示例：
     * - input: Shape(1,7,7,512), out_features: 1000
     *   → output: Shape(1,1000)
     * - input: Shape(512), out_features: 1000
     *   → output: Shape(1000)
     */
    static Shape linear_output_shape(const Shape& input, int32_t out_features);

    /**
     * @brief 计算展平后的形状
     * @param input 输入形状（任意维度）
     * @param start_dim 开始展平的维度（默认为1，保留batch维度）
     * @return 展平后的形状
     *
     * 将start_dim到最后一维展平为一维。
     * 默认保留batch维度(N)，展平HWC为一维特征向量。
     *
     * 示例：
     * - input: Shape(1,28,28,3) → output: Shape(1, 2352)
     * - input: Shape(1,7,7,512) → output: Shape(1, 25088)
     */
    static Shape flatten_shape(const Shape& input, int32_t start_dim = 1);

    /**
     * @brief 计算重塑后的形状（仅支持已知维度的情况）
     * @param input 输入形状
     * @param new_shape 新形状（可以包含-1表示推导）
     * @return 重塑后的形状
     *
     * 将张量重塑为新形状，元素总数必须相同。
     * -1表示自动推导该维度（只能有一个-1）。
     *
     * 示例：
     * - input: Shape(1,28,28,3), new_shape: (1,-1)
     *   → output: Shape(1,2352)
     */
    static Shape reshape_shape(const Shape& input, const std::array<int32_t, 4>& new_shape);

private:
    std::array<int32_t, 4> dims_;  ///< 维度数组（右对齐）
    int32_t ndim_;                  ///< 实际维度数（0-4）
};

} // namespace tr
