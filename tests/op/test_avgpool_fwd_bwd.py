#!/usr/bin/env python3
"""
AvgPool FWD+BWD 串接测试 — PyTorch 参考数据生成
FWD:  y = avgpool2d(x, count_include_pad=False)
BWD:  dx = avgpool2d_backward(dy, x)
输出: FWD 的输入 x、输出 y、BWD 的输入 dy、输出 dx。
支持 --dtype fp16 | fp32。
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate(shape: tuple, kernel: int, stride: int, padding: int, seed: int, ws: str, dtype: str):
    N, H, W, C = shape

    if dtype == 'fp16':
        torch_dtype = torch.float16
        np_dtype    = np.float16
        suffix      = '_amp'
        dtype_label = 'FP16'
        is_fp16     = True
    else:
        torch_dtype = torch.float32
        np_dtype    = np.float32
        suffix      = '_fp32'
        dtype_label = 'FP32'
        is_fp16     = False

    if is_fp16:
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.backends.cudnn.fp32_precision = 'ieee'
        except AttributeError:
            pass

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(ws, exist_ok=True)

    # 计算输出形状
    OH = (H + 2 * padding - kernel) // stride + 1
    OW = (W + 2 * padding - kernel) // stride + 1

    # PyTorch 使用 NCHW 格式
    # count_include_pad=False 对齐 CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    pool = torch.nn.AvgPool2d(
        kernel_size=kernel, stride=stride, padding=padding,
        count_include_pad=False)

    X = torch.randn(N, C, H, W, dtype=torch_dtype) * 2.0
    Y = pool(X)

    torch.manual_seed(seed + 1000)
    dY = torch.randn(N, C, OH, OW, dtype=torch_dtype) * 2.0

    # 计算梯度
    X_grad = X.detach().requires_grad_(True)
    Y_grad = pool(X_grad)
    Y_grad.backward(dY)
    dX = X_grad.grad

    # 转换为 NHWC 格式保存
    X_npy  = X.permute(0, 2, 3, 1).contiguous()
    Y_npy  = Y.permute(0, 2, 3, 1).contiguous()
    dY_npy = dY.permute(0, 2, 3, 1).contiguous()
    dX_npy = dX.permute(0, 2, 3, 1).contiguous()

    save_tsr(os.path.join(ws, f'x_fwd_bwd{suffix}.tsr'),
             [X_npy.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'y_ref_fwd_bwd{suffix}.tsr'),
             [Y_npy.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dy_fwd_bwd{suffix}.tsr'),
             [dY_npy.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dx_ref_fwd_bwd{suffix}.tsr'),
             [dX_npy.cpu().numpy().astype(np_dtype)], compress=False)

    file_desc = (
        f"  x:   x_fwd_bwd{suffix}.tsr\n"
        f"  y:   y_ref_fwd_bwd{suffix}.tsr\n"
        f"  dy:  dy_fwd_bwd{suffix}.tsr\n"
        f"  dx:  dx_ref_fwd_bwd{suffix}.tsr\n"
    )

    yaml = (
        f"op: avgpool_fwd_bwd\n"
        f"dtype: {dtype_label}\n"
        f"shape: [{N}, {H}, {W}, {C}]\n"
        f"output_shape: [{N}, {OH}, {OW}, {C}]\n"
        f"pool_params:\n"
        f"  kernel: {kernel}\n"
        f"  stride: {stride}\n"
        f"  padding: {padding}\n"
        f"seed: {seed}\n"
        f"mse_threshold: 1e-3\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"AvgPool FWD+BWD {dtype_label} reference generated in: {ws}")
    print(f"  Input shape: (N,C,H,W)=({N},{C},{H},{W}) -> Output: (N,C,OH,OW)=({N},{C},{OH},{OW})")

def main():
    parser = argparse.ArgumentParser(description='AvgPool FWD+BWD test data generator')
    parser.add_argument('--shape', default='8,16,16,64',
                        help='NHWC shape (comma-separated, e.g. 8,16,16,64)')
    parser.add_argument('--kernel', type=int, default=2,
                        help='Kernel size (default: 2)')
    parser.add_argument('--stride', type=int, default=2,
                        help='Stride (default: 2)')
    parser.add_argument('--padding', type=int, default=0,
                        help='Padding (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'],
                        help='Data type: fp16 (AMP) or fp32 (CPU/GPU)')
    args = parser.parse_args()

    dims = tuple(int(x) for x in args.shape.split(','))
    if len(dims) != 4:
        print(f"Invalid shape '{args.shape}': must be N,H,W,C (4D)")
        sys.exit(1)

    generate(dims, args.kernel, args.stride, args.padding, args.seed, args.workspace, args.dtype)

if __name__ == '__main__':
    main()
