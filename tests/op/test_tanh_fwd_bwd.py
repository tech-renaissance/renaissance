#!/usr/bin/env python3
"""
Tanh FWD+BWD 串接测试 — PyTorch 参考数据生成

数学公式：
  FWD: Y = tanh(X)
  BWD: dX = dY * (1 - Y^2)
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr


def generate(shape, seed, ws, dtype):
    N, H, W, C = shape

    if dtype == 'fp16':
        torch_dtype = torch.float16
        np_dtype = np.float16
        suffix = '_amp'
        dtype_label = 'FP16'
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
        suffix = '_fp32'
        dtype_label = 'FP32'

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(ws, exist_ok=True)

    # NHWC layout (PyTorch shape semantics only, element-wise ops are layout-agnostic)
    X = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0

    # FWD: Y = tanh(X)
    Y = torch.tanh(X)

    # BWD: dX = dY * (1 - Y^2)
    torch.manual_seed(seed + 1000)
    dY = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
    dX = dY * (1.0 - Y ** 2)

    save_tsr(os.path.join(ws, f'x_fwd_bwd{suffix}.tsr'),
             [X.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'y_ref_fwd_bwd{suffix}.tsr'),
             [Y.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dy_fwd_bwd{suffix}.tsr'),
             [dY.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dx_ref_fwd_bwd{suffix}.tsr'),
             [dX.cpu().numpy().astype(np_dtype)], compress=False)

    print(f"Tanh FWD+BWD {dtype_label} reference generated in: {ws}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='8,1024,1024,8', help='N,H,W,C')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()

    shape = [int(x) for x in args.shape.split(',')]
    generate(shape, args.seed, args.workspace, args.dtype)
