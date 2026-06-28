#!/usr/bin/env python3
"""
ReLU FWD+BWD 串接测试 — PyTorch 参考数据生成
FWD:  y = relu(x)
BWD:  dx = dy * (x > 0).to(dtype)   [BWD 复用 FWD 的位掩码]
输出: FWD 的输入 x、输出 y、BWD 的输入 dy、输出 dx。不输出 bitmask TSR。
支持 --dtype fp16 | fp32。
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate(shape: tuple, seed: int, ws: str, dtype: str):
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

    # ── FWD ──
    X = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
    Y = torch.relu(X)

    # ── BWD ──
    torch.manual_seed(seed + 1000)
    dY = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
    mask_val = (X > 0).to(torch_dtype)
    dX = dY * mask_val

    save_tsr(os.path.join(ws, f'x_fwd_bwd{suffix}.tsr'),
             [X.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'y_ref_fwd_bwd{suffix}.tsr'),
             [Y.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dy_fwd_bwd{suffix}.tsr'),
             [dY.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dx_ref_fwd_bwd{suffix}.tsr'),
             [dX.cpu().numpy().astype(np_dtype)], compress=False)

    file_desc = (
        f"  x:   x_fwd_bwd{suffix}.tsr\n"
        f"  y:   y_ref_fwd_bwd{suffix}.tsr\n"
        f"  dy:  dy_fwd_bwd{suffix}.tsr\n"
        f"  dx:  dx_ref_fwd_bwd{suffix}.tsr\n"
    )

    yaml = (
        f"op: relu_fwd_bwd\n"
        f"dtype: {dtype_label}\n"
        f"shape: [{N}, {H}, {W}, {C}]\n"
        f"seed: {seed}\n"
        f"mse_threshold: 1e-3\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"ReLU FWD+BWD {dtype_label} reference generated in: {ws}")

def main():
    parser = argparse.ArgumentParser(description='ReLU FWD+BWD test data generator')
    parser.add_argument('--shape', default='8,1024,1024,8',
                        help='NHWC shape (comma-separated, e.g. 8,1024,1024,8)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'],
                        help='Data type: fp16 (AMP) or fp32 (CPU/GPU)')
    args = parser.parse_args()

    dims = tuple(int(x) for x in args.shape.split(','))
    if len(dims) != 4:
        print(f"Invalid shape '{args.shape}': must be N,H,W,C (4D)")
        sys.exit(1)

    generate(dims, args.seed, args.workspace, args.dtype)

if __name__ == '__main__':
    main()
