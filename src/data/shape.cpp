#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

#include "renaissance/data/shape.h"

#include <string>

namespace tr {

Shape Shape::conv_output_shape(const Shape& input, int32_t kernel_size,
                               int32_t out_channels, int32_t stride, int32_t padding) {
    int32_t h = input.h();
    int32_t w = input.w();

    // 计算输出的高度和宽度
    // 公式: output = (input - kernel + 2*padding) / stride + 1
    int32_t out_h = (h - kernel_size + 2 * padding) / stride + 1;
    int32_t out_w = (w - kernel_size + 2 * padding) / stride + 1;

    // 验证输出形状的有效性
    if (out_h <= 0 || out_w <= 0) {
        TR_THROW(ValueError, "Invalid convolution output shape: input=(", h, ",", w, "), "
                       "kernel=", kernel_size, ", padding=", padding, ", stride=", stride,
                       ", output=(", out_h, ",", out_w, ")");
    }

    // 根据输入维度决定输出维度（保持语义一致性）
    if (input.ndim() == 3) {
        // 3D输入(H,W,C) -> 3D输出(H',W',C')
        return Shape(out_h, out_w, out_channels);
    } else {
        // 4D输入(N,H,W,C) -> 4D输出(N,H',W',C')
        return Shape(input.n(), out_h, out_w, out_channels);
    }
}

Shape Shape::pool_output_shape(const Shape& input, int32_t kernel_size, int32_t stride) {
    int32_t h = input.h();
    int32_t w = input.w();
    int32_t c = input.c();

    // 计算输出的高度和宽度
    // 公式: output = (input - kernel) / stride + 1
    int32_t out_h = (h - kernel_size) / stride + 1;
    int32_t out_w = (w - kernel_size) / stride + 1;

    // 验证输出形状的有效性
    if (out_h <= 0 || out_w <= 0) {
        TR_THROW(ValueError, "Invalid pooling output shape: input=(", h, ",", w, "), "
                       "kernel=", kernel_size, ", stride=", stride,
                       ", output=(", out_h, ",", out_w, ")");
    }

    // 根据输入维度决定输出维度（保持语义一致性）
    if (input.ndim() == 3) {
        // 3D输入(H,W,C) -> 3D输出(H',W',C)
        return Shape(out_h, out_w, c);
    } else {
        // 4D输入(N,H,W,C) -> 4D输出(N,H',W',C)
        return Shape(input.n(), out_h, out_w, c);
    }
}

Shape Shape::gap_output_shape(const Shape& input) {
    int32_t n = input.n();
    int32_t c = input.c();

    // 全局平均池化: (N,H,W,C) -> (N,1,1,C) -> 展平为 (N,C) 或
    if (n > 1) {
        // 保留batch维度: (N, C)
        return Shape(n, c);
    } else {
        // 单个样本: (C)
        return Shape(c);
    }
}

Shape Shape::linear_output_shape(const Shape& input, int32_t out_features) {
    int32_t n = input.n();

    // 全连接层将输入展平为一维向量，然后线性映射到out_features维
    if (n > 1) {
        // 保留batch维度: (N, out_features)
        return Shape(n, out_features);
    } else {
        // 单个样本: (out_features)
        return Shape(out_features);
    }
}

Shape Shape::flatten_shape(const Shape& input, int32_t start_dim) {
    if (start_dim < 0 || start_dim >= input.ndim_) {
        TR_THROW(ValueError, "Invalid start_dim for flatten: start_dim=", start_dim,
                       ", ndim=", input.ndim_);
    }

    // 默认保留batch维度（start_dim=1），展平HWC为一维
    if (start_dim == 0) {
        // 展平所有维度为一维
        return Shape(static_cast<int32_t>(input.numel()));
    } else if (start_dim == 1 && input.ndim_ == 4) {
        // (N,H,W,C) -> (N, H*W*C)
        int32_t n = input.n();
        int32_t flattened_dim = input.h() * input.w() * input.c();
        return Shape(n, flattened_dim);
    } else {
        // 通用情况: 计算展平后的维度
        int32_t leading_dim = 1;
        for (int32_t i = 0; i < start_dim; ++i) {
            leading_dim *= input.dim(i);
        }

        int32_t flattened_dim = 1;
        for (int32_t i = start_dim; i < input.ndim_; ++i) {
            flattened_dim *= input.dim(i);
        }

        if (leading_dim > 1 && flattened_dim > 1) {
            return Shape(leading_dim, flattened_dim);
        } else if (flattened_dim > 1) {
            // 保留第一个维度（即使为1，也创建2D形状）
            return Shape(leading_dim > 0 ? leading_dim : 1, flattened_dim);
        } else {
            return Shape();  // 标量
        }
    }
}

Shape Shape::reshape_shape(const Shape& input, const std::array<int32_t, 4>& new_shape) {
    int64_t total_elements = input.numel();
    int32_t new_ndim = 0;
    int32_t inferred_dim = -1;
    int64_t known_product = 1;

    // 计算已知维度的乘积，并找到需要推导的维度（-1）
    for (int32_t i = 0; i < 4; ++i) {
        if (new_shape[i] == 0) {
            // 0表示保持原维度（仅当输入维度数>=i+1时有效）
            if (i < input.ndim_) {
                known_product *= input.dim(i);
                new_ndim = i + 1;
            }
        } else if (new_shape[i] == -1) {
            // -1表示自动推导
            if (inferred_dim != -1) {
                TR_THROW(ValueError, "Reshape can only infer one dimension, but found multiple -1");
            }
            inferred_dim = i;
            new_ndim = i + 1;
        } else if (new_shape[i] > 0) {
            known_product *= new_shape[i];
            new_ndim = i + 1;
        }
        // 忽略负数（除了-1）
    }

    // 推导-1维度的值
    if (inferred_dim != -1) {
        if (known_product == 0) {
            TR_THROW(ValueError, "Cannot infer dimension when known product is 0");
        }
        int64_t inferred_size = total_elements / known_product;
        if (inferred_size * known_product != total_elements) {
            TR_THROW(ValueError, "Cannot reshape tensor of size ", total_elements,
                           " to shape with incompatible dimensions");
        }
        if (inferred_size > INT32_MAX) {
            TR_THROW(ValueError, "Inferred dimension size exceeds INT32_MAX: ", inferred_size);
        }

        // 构造最终的形状
        std::array<int32_t, 4> final_shape = new_shape;
        final_shape[inferred_dim] = static_cast<int32_t>(inferred_size);

        // 右对齐存储
        std::array<int32_t, 4> right_aligned{0, 0, 0, 0};
        for (int32_t i = 0; i < new_ndim; ++i) {
            right_aligned[4 - new_ndim + i] = final_shape[i];
        }

        Shape result;
        result.dims_ = right_aligned;
        result.ndim_ = new_ndim;
        return result;
    } else {
        // 无需推导，直接验证
        if (known_product != total_elements) {
            TR_THROW(ValueError, "Cannot reshape tensor of size ", total_elements,
                           " to shape with size ", known_product);
        }

        // 右对齐存储
        std::array<int32_t, 4> right_aligned{0, 0, 0, 0};
        for (int32_t i = 0; i < new_ndim; ++i) {
            right_aligned[4 - new_ndim + i] = new_shape[i];
        }

        Shape result;
        result.dims_ = right_aligned;
        result.ndim_ = new_ndim;
        return result;
    }
}

std::string Shape::to_string() const {
    if (ndim_ == 0) {
        return "()";  // 标量
    }

    // 找到第一个非零维度的索引
    int32_t first_idx = 4 - ndim_;

    std::string result = "(";
    for (int32_t i = first_idx; i < 4; ++i) {
        if (i != first_idx) {
            result += ",";
        }
        result += std::to_string(dims_[i]);
    }
    result += ")";
    return result;
}

} // namespace tr
