#!/usr/bin/env python3
"""
生成CONV测试数据 - PyTorch实现 (修复版)
版本：V4.20.1
日期：2026-05-06
作者：技术觉醒团队
"""

import torch
import numpy as np
import sys
import os

# 添加python目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'scripts'))
from tsr_v4 import save_tsr

def get_project_root():
    """获取项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', '..'))

def generate_conv_test_data():
    """生成CONV测试数据并保存为TSR格式"""

    # 设置随机种子保证可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 禁用TF32，使用纯FP32 IEEE精度，消除与TR4 cuDNN的精度差异
    try:
        torch.backends.cudnn.conv.fp32_precision = 'ieee'  # PyTorch 2.9+ 新API
    except AttributeError:
        torch.backends.cudnn.allow_tf32 = False  # 旧API兼容
    print(f"PyTorch TF32: disabled (pure FP32)")

    # 测试参数
    batch_size = 16
    in_channels = 64
    out_channels = 64
    height = 56
    width = 56
    kernel_h = 3
    kernel_w = 3
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1

    print(f"生成CONV测试数据:")
    print(f"  输入: [{batch_size}, {height}, {width}, {in_channels}]")
    print(f"  权重: [{out_channels}, {in_channels}, {kernel_h}, {kernel_w}]")
    print(f"  输出: [{batch_size}, {height}, {width}, {out_channels}]")

    # 生成输入张量 X: NCHW -> NHWC
    X_nchw = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
    X_nhwc = X_nchw.transpose(0, 2, 3, 1).copy()

    # 生成权重张量 W: PyTorch OIHW格式 (K,C,R,S)
    W_kcrs = np.random.randn(out_channels, in_channels, kernel_h, kernel_w).astype(np.float32) * 0.1

    # cuDNN conv_fprop 要求权重为 KRSC 物理布局 [K,R,S,C]
    # 转置: (K,C,R,S) → (K,R,S,C)
    W_krsc = W_kcrs.transpose(0, 2, 3, 1).copy()
    W_output = W_krsc  # shape [K,R,S,C]，紧凑存储即为KRSC

    # 用PyTorch计算参考输出 (PyTorch 使用 KCRS/OIHW)
    X_torch = torch.from_numpy(X_nchw)
    W_torch = torch.from_numpy(W_kcrs)
    Y_torch = torch.nn.functional.conv2d(
        X_torch, W_torch, bias=None,
        stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    Y_nchw = Y_torch.numpy().astype(np.float32)
    Y_nhwc = Y_nchw.transpose(0, 2, 3, 1).copy()

    print(f"\n输入X shape: {X_nhwc.shape}")
    print(f"权重W shape: {W_output.shape}")
    print(f"输出Y shape: {Y_nhwc.shape}")

    # 保存TSR文件
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "workspace", "conv_test_data")
    os.makedirs(output_dir, exist_ok=True)

    save_tsr(f"{output_dir}/X.tsr", X_nhwc, compress=False)
    save_tsr(f"{output_dir}/W.tsr", W_output, compress=False)  # KRSC格式 [K,R,S,C]
    save_tsr(f"{output_dir}/Y_ref.tsr", Y_nhwc, compress=False)

    print(f"\nTSR文件已保存到: {output_dir}")
    print(f"  X.tsr     - 输入张量 (NHWC)")
    print(f"  W.tsr     - 权重张量 (KRSC layout, shape=[K,R,S,C])")
    print(f"  Y_ref.tsr - PyTorch参考输出 (NHWC)")

    # 保存参数
    params_file = os.path.join(output_dir, "params.txt")
    with open(params_file, 'w') as f:
        f.write(f"{batch_size} {in_channels} {out_channels} "
                f"{height} {width} "
                f"{kernel_h} {kernel_w} "
                f"{stride_h} {stride_w} "
                f"{pad_h} {pad_w}\n")

    print(f"参数信息已保存到: {params_file}")

if __name__ == "__main__":
    generate_conv_test_data()
