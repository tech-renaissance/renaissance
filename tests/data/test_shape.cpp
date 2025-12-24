/**
 * @file test_shape.cpp
 * @brief Shape类测试样例
 * @details 测试NHWC语义和右对齐存储是否正确
 * @version 3.6.0
 * @date 2025-12-24
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <cassert>
#include <iostream>

using namespace tr;

// 辅助函数：打印测试结果
void print_test_result(const char* test_name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << test_name << std::endl;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
    }
}

// 测试1: 右对齐存储 - 2D Shape测试
void test_right_alignment_2d() {
    std::cout << "\n=== Test 1: Right Alignment (2D Shape) ===" << std::endl;

    // 创建2D Shape (32, 64)
    Shape s(32, 64);

    // 验证维度数
    bool test_ndim = (s.ndim() == 2);
    print_test_result("  2D shape has ndim=2", test_ndim);

    // 验证右对齐: [0, 0, 32, 64]
    bool test_h = (s.h() == 32);
    bool test_w = (s.w() == 64);
    print_test_result("  2D shape.h() == 32", test_h);
    print_test_result("  2D shape.w() == 64", test_w);

    // 验证N和C都返回1（因为不是4D）
    bool test_n = (s.n() == 1);
    bool test_c = (s.c() == 1);
    print_test_result("  2D shape.n() == 1 (non-4D)", test_n);
    print_test_result("  2D shape.c() == 1 (non-1D)", test_c);

    // 验证元素总数
    bool test_numel = (s.numel() == 32 * 64);
    print_test_result("  numel() == 32 * 64", test_numel);

    // 验证不是4D表现形式(32, 64, 1, 1)
    // 右对齐应该是(1, 1, 32, 64)
    bool test_alignment = (s.n() == 1 && s.h() == 32 && s.w() == 64 && s.c() == 1);
    print_test_result("  Right-aligned as (1, 1, 32, 64) not (32, 64, 1, 1)", test_alignment);

    assert(test_ndim && test_h && test_w && test_n && test_c && test_numel && test_alignment);
}

// 测试2: 右对齐存储 - 4D Shape测试
void test_right_alignment_4d() {
    std::cout << "\n=== Test 2: Right Alignment (4D Shape) ===" << std::endl;

    // 创建4D Shape (N=2, H=28, W=28, C=3)
    Shape s(2, 28, 28, 3);

    // 验证维度数
    bool test_ndim = (s.ndim() == 4);
    print_test_result("  4D shape has ndim=4", test_ndim);

    // 验证NHWC语义
    bool test_n = (s.n() == 2);
    bool test_h = (s.h() == 28);
    bool test_w = (s.w() == 28);
    bool test_c = (s.c() == 3);
    print_test_result("  shape.n() == 2 (N)", test_n);
    print_test_result("  shape.h() == 28 (H)", test_h);
    print_test_result("  shape.w() == 28 (W)", test_w);
    print_test_result("  shape.c() == 3 (C)", test_c);

    // 验证dim()访问器 (NHWC语义)
    bool test_dim_0 = (s.dim(0) == 2);   // N
    bool test_dim_1 = (s.dim(1) == 28);  // H
    bool test_dim_2 = (s.dim(2) == 28);  // W
    bool test_dim_3 = (s.dim(3) == 3);   // C
    print_test_result("  shape.dim(0) returns N (2)", test_dim_0);
    print_test_result("  shape.dim(1) returns H (28)", test_dim_1);
    print_test_result("  shape.dim(2) returns W (28)", test_dim_2);
    print_test_result("  shape.dim(3) returns C (3)", test_dim_3);

    assert(test_ndim && test_n && test_h && test_w && test_c &&
           test_dim_0 && test_dim_1 && test_dim_2 && test_dim_3);
}

// 测试3: 标量Shape
void test_scalar_shape() {
    std::cout << "\n=== Test 3: Scalar Shape ===" << std::endl;

    // 创建标量Shape
    Shape s;

    // 验证维度数
    bool test_ndim = (s.ndim() == 0);
    print_test_result("  Scalar shape has ndim=0", test_ndim);

    // 验证所有访问器返回1
    bool test_n = (s.n() == 1);
    bool test_h = (s.h() == 1);
    bool test_w = (s.w() == 1);
    bool test_c = (s.c() == 1);
    print_test_result("  shape.n() == 1", test_n);
    print_test_result("  shape.h() == 1", test_h);
    print_test_result("  shape.w() == 1", test_w);
    print_test_result("  shape.c() == 1", test_c);

    // 验证元素总数
    bool test_numel = (s.numel() == 1);
    print_test_result("  numel() == 1", test_numel);

    assert(test_ndim && test_n && test_h && test_w && test_c && test_numel);
}

// 测试4: 1D Shape (仅通道维度)
void test_1d_shape() {
    std::cout << "\n=== Test 4: 1D Shape (Channel only) ===" << std::endl;

    // 创建1D Shape (C=128)
    Shape s(128);

    // 验证维度数
    bool test_ndim = (s.ndim() == 1);
    print_test_result("  1D shape has ndim=1", test_ndim);

    // 验证右对齐: [0, 0, 0, 128]
    bool test_c = (s.c() == 128);
    bool test_h = (s.h() == 1);
    bool test_w = (s.w() == 1);
    print_test_result("  shape.c() == 128 (C)", test_c);
    print_test_result("  shape.h() == 1 (non-2D)", test_h);
    print_test_result("  shape.w() == 1 (non-2D)", test_w);

    // 验证元素总数
    bool test_numel = (s.numel() == 128);
    print_test_result("  numel() == 128", test_numel);

    assert(test_ndim && test_c && test_h && test_w && test_numel);
}

// 测试5: 3D Shape (H, W, C)
void test_3d_shape() {
    std::cout << "\n=== Test 5: 3D Shape (H, W, C) ===" << std::endl;

    // 创建3D Shape (H=224, W=224, C=3)
    Shape s(224, 224, 3);

    // 验证维度数
    bool test_ndim = (s.ndim() == 3);
    print_test_result("  3D shape has ndim=3", test_ndim);

    // 验证右对齐: [0, 224, 224, 3]
    bool test_n = (s.n() == 1);  // 不是4D
    bool test_h = (s.h() == 224);
    bool test_w = (s.w() == 224);
    bool test_c = (s.c() == 3);
    print_test_result("  shape.n() == 1 (non-4D)", test_n);
    print_test_result("  shape.h() == 224 (H)", test_h);
    print_test_result("  shape.w() == 224 (W)", test_w);
    print_test_result("  shape.c() == 3 (C)", test_c);

    // 验证元素总数
    bool test_numel = (s.numel() == 224 * 224 * 3);
    print_test_result("  numel() == 224 * 224 * 3", test_numel);

    assert(test_ndim && test_n && test_h && test_w && test_c && test_numel);
}

// 测试6: 负索引支持
void test_negative_indexing() {
    std::cout << "\n=== Test 6: Negative Indexing ===" << std::endl;

    // 创建4D Shape (N=2, H=28, W=28, C=3)
    Shape s(2, 28, 28, 3);

    // 验证负索引
    bool test_neg1 = (s.dim(-1) == 3);   // C
    bool test_neg2 = (s.dim(-2) == 28);  // W
    bool test_neg3 = (s.dim(-3) == 28);  // H
    bool test_neg4 = (s.dim(-4) == 2);   // N
    print_test_result("  shape.dim(-1) == C (3)", test_neg1);
    print_test_result("  shape.dim(-2) == W (28)", test_neg2);
    print_test_result("  shape.dim(-3) == H (28)", test_neg3);
    print_test_result("  shape.dim(-4) == N (2)", test_neg4);

    assert(test_neg1 && test_neg2 && test_neg3 && test_neg4);
}

// 测试7: 形状相等性
void test_shape_equality() {
    std::cout << "\n=== Test 7: Shape Equality ===" << std::endl;

    Shape s1(32, 32, 3);
    Shape s2(32, 32, 3);
    Shape s3(32, 32);

    bool test_eq = (s1 == s2);
    bool test_ne = (s1 != s3);
    print_test_result("  Shape(32,32,3) == Shape(32,32,3)", test_eq);
    print_test_result("  Shape(32,32,3) != Shape(32,32)", test_ne);

    assert(test_eq && test_ne);
}

// 测试8: 形状推断 - 卷积输出
void test_conv_output_shape() {
    std::cout << "\n=== Test 8: Conv Output Shape Inference ===" << std::endl;

    // 输入: (N=1, H=28, W=28, C=3)
    // 卷积: kernel=5, stride=1, padding=0, out_channels=16
    Shape input(1, 28, 28, 3);
    Shape output = Shape::conv_output_shape(input, 5, 16, 1, 0);

    // 期望输出: (1, 24, 24, 16)
    // 公式: (28 - 5 + 2*0) / 1 + 1 = 24
    bool test_n = (output.n() == 1);
    bool test_h = (output.h() == 24);
    bool test_w = (output.w() == 24);
    bool test_c = (output.c() == 16);
    print_test_result("  Conv output N == 1", test_n);
    print_test_result("  Conv output H == 24", test_h);
    print_test_result("  Conv output W == 24", test_w);
    print_test_result("  Conv output C == 16", test_c);

    assert(test_n && test_h && test_w && test_c);
}

// 测试9: 形状推断 - 池化输出
void test_pool_output_shape() {
    std::cout << "\n=== Test 9: Pool Output Shape Inference ===" << std::endl;

    // 输入: (N=1, H=56, W=56, C=64)
    // 池化: kernel=2, stride=2
    Shape input(1, 56, 56, 64);
    Shape output = Shape::pool_output_shape(input, 2, 2);

    // 期望输出: (1, 28, 28, 64)
    // 公式: (56 - 2) / 2 + 1 = 28
    bool test_n = (output.n() == 1);
    bool test_h = (output.h() == 28);
    bool test_w = (output.w() == 28);
    bool test_c = (output.c() == 64);
    print_test_result("  Pool output N == 1", test_n);
    print_test_result("  Pool output H == 28", test_h);
    print_test_result("  Pool output W == 28", test_w);
    print_test_result("  Pool output C == 64", test_c);

    assert(test_n && test_h && test_w && test_c);
}

// 测试10: 形状推断 - 展平
void test_flatten_shape() {
    std::cout << "\n=== Test 10: Flatten Shape Inference ===" << std::endl;

    // 输入: (N=1, H=28, W=28, C=3)
    Shape input(1, 28, 28, 3);

    // 展平: start_dim=1 (保留batch维度)
    Shape output = Shape::flatten_shape(input, 1);

    // 期望输出: (1, 28*28*3) = (1, 2352) - 2D Shape
    // 对于2D Shape(H, W): H=1, W=2352
    bool test_n = (output.n() == 1);
    bool test_h = (output.h() == 1);      // 2D的第一个维度
    bool test_w = (output.w() == 2352);   // 2D的第二个维度
    print_test_result("  Flatten output N == 1", test_n);
    print_test_result("  Flatten output H == 1 (2D first dim)", test_h);
    print_test_result("  Flatten output W == 2352 (28*28*3)", test_w);

    assert(test_n && test_h && test_w);
}

// 测试11: 形状推断 - 全局平均池化
void test_gap_shape() {
    std::cout << "\n=== Test 11: GAP Shape Inference ===" << std::endl;

    // 输入: (N=1, H=7, W=7, C=512)
    Shape input(1, 7, 7, 512);

    // 全局平均池化
    Shape output = Shape::gap_output_shape(input);

    // 期望输出: (512) - 1D向量
    bool test_ndim = (output.ndim() == 1);
    bool test_c = (output.c() == 512);
    print_test_result("  GAP output ndim == 1", test_ndim);
    print_test_result("  GAP output C == 512", test_c);

    assert(test_ndim && test_c);
}

// 测试12: 形状推断 - 全连接层
void test_linear_shape() {
    std::cout << "\n=== Test 12: Linear Shape Inference ===" << std::endl;

    // 输入: (512) - 展平的特征向量
    Shape input(512);

    // 全连接层: out_features=1000
    Shape output = Shape::linear_output_shape(input, 1000);

    // 期望输出: (1000)
    bool test_ndim = (output.ndim() == 1);
    bool test_c = (output.c() == 1000);
    print_test_result("  Linear output ndim == 1", test_ndim);
    print_test_result("  Linear output C == 1000", test_c);

    assert(test_ndim && test_c);
}

// 主函数
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Shape Class Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 基础测试
        test_scalar_shape();
        test_1d_shape();
        test_right_alignment_2d();
        test_right_alignment_4d();
        test_3d_shape();

        // 高级测试
        test_negative_indexing();
        test_shape_equality();
        test_conv_output_shape();
        test_pool_output_shape();
        test_flatten_shape();
        test_gap_shape();
        test_linear_shape();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test FAILED with exception: " << e.what() << std::endl;
        std::cout << "========================================" << std::endl;
        return 1;
    }
}
