#!/usr/bin/env python3
"""
生成FC测试数据 - PyTorch实现
版本：V4.20.1
日期：2026-04-20
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

# 获取项目根目录（脚本所在目录的上级的上级）
def get_project_root():
    """获取项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 脚本在 python/scripts/ 下，所以需要向上两级
    return os.path.abspath(os.path.join(script_dir, '..', '..'))

def generate_fc_test_data():
    """生成FC测试数据并保存为TSR格式"""

    # 设置随机种子保证可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 测试参数 - 扩大规格以便更全面验证
    batch_size = 4
    in_features = 10
    out_features = 11

    print(f"生成FC测试数据:")
    print(f"  batch_size = {batch_size}")
    print(f"  in_features = {in_features}")
    print(f"  out_features = {out_features}")

    # 生成输入张量 X: [batch, in_features]
    # 注意：需要转换为NHWC格式 [N, H, W, C] = [batch, 1, 1, in_features]
    X_np = np.random.randn(batch_size, in_features).astype(np.float32)
    X_nhwc = X_np.reshape(batch_size, 1, 1, in_features)

    # 生成权重张量 W: [out_features, in_features]
    # 转换为NHWC格式 [out_features, 1, 1, in_features]
    W_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    W_nhwc = W_np.reshape(out_features, 1, 1, in_features)

    # 生成偏置张量 b: [out_features]
    # 转换为NHWC格式 [1, 1, 1, out_features]
    b_np = np.random.randn(out_features).astype(np.float32) * 0.1
    b_nhwc = b_np.reshape(1, 1, 1, out_features)

    # 用PyTorch计算FC输出作为参考
    X_torch = torch.from_numpy(X_np)
    W_torch = torch.from_numpy(W_np)
    b_torch = torch.from_numpy(b_np)

    # FC前向: Y = X @ W^T + b
    Y_torch = torch.nn.functional.linear(X_torch, W_torch, bias=b_torch)
    Y_np = Y_torch.numpy().astype(np.float32)

    # 转换为NHWC格式 [batch, 1, 1, out_features]
    Y_nhwc = Y_np.reshape(batch_size, 1, 1, out_features)

    print(f"\n输入X shape: {X_nhwc.shape}")
    print(f"权重W shape: {W_nhwc.shape}")
    print(f"偏置b shape: {b_nhwc.shape}")
    print(f"输出Y shape: {Y_nhwc.shape}")

    print(f"\nPyTorch输出Y前5个元素:")
    print(f"  {Y_np[0, :5]}")

    # 调试：打印W的数据
    print(f"\n权重W的前20个元素（NHWC展平）:")
    print(f"  {W_nhwc.flatten()[:20]}")
    print(f"权重W原始形状 {W_np.shape}:")
    print(f"  {W_np}")

    # 保存为TSR文件（使用项目根目录的绝对路径）
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "workspace", "fc_test_data")
    os.makedirs(output_dir, exist_ok=True)

    save_tsr(f"{output_dir}/X.tsr", X_nhwc, compress=False)
    save_tsr(f"{output_dir}/W.tsr", W_nhwc, compress=False)
    save_tsr(f"{output_dir}/b.tsr", b_nhwc, compress=False)
    save_tsr(f"{output_dir}/Y_ref.tsr", Y_nhwc, compress=False)

    print(f"\nTSR文件已保存到: {output_dir}")
    print(f"  X.tsr     - 输入张量")
    print(f"  W.tsr     - 权重张量")
    print(f"  b.tsr     - 偏置张量")
    print(f"  Y_ref.tsr - PyTorch参考输出")

    # 保存参数信息供C++测试使用
    params_file = os.path.join(output_dir, "params.txt")
    with open(params_file, 'w') as f:
        f.write(f"{batch_size} {in_features} {out_features}\n")

    print(f"\n参数信息已保存到: {params_file}")

if __name__ == "__main__":
    generate_fc_test_data()
