#!/usr/bin/env python3
"""
GAP FWD+BWD 数学正确性测试 — PyTorch 参考数据生成
FWD: y[n,c] = mean(x[n,:, :, c])  (NHWC视角)
BWD: dx[n,h,w,c] = dy[n,c] / (H*W)
支持 --dtype fp16 | fp32
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate(batch: int, H: int, W: int, C: int, seed: int, ws: str, dtype: str):
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

    # ── FWD: y = adaptive_avg_pool2d(x, 1) ──
    # PyTorch 默认 NCHW
    x_nchw = torch.randn(batch, C, H, W, dtype=torch_dtype) * 2.0
    y_nchw = F.adaptive_avg_pool2d(x_nchw, (1, 1))  # [N, C, 1, 1]

    # ── BWD: dx = dy / (H*W) broadcast ──
    torch.manual_seed(seed + 1000)
    dy_nchw = torch.randn(batch, C, 1, 1, dtype=torch_dtype) * 2.0
    dx_nchw = dy_nchw.expand(-1, -1, H, W) / (H * W)

    # ── 转换为 NHWC 并保存 ──
    x_nhwc  = x_nchw.permute(0, 2, 3, 1).contiguous()
    y_nhwc  = y_nchw.permute(0, 2, 3, 1).contiguous()
    dy_nhwc = dy_nchw.permute(0, 2, 3, 1).contiguous()
    dx_nhwc = dx_nchw.permute(0, 2, 3, 1).contiguous()

    save_tsr(os.path.join(ws, f'x_gap{suffix}.tsr'),
             [x_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'y_gap{suffix}.tsr'),
             [y_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dy_gap{suffix}.tsr'),
             [dy_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dx_gap{suffix}.tsr'),
             [dx_nhwc.cpu().numpy().astype(np_dtype)], compress=False)

    file_desc = (
        f"  x:   x_gap{suffix}.tsr\n"
        f"  y:   y_gap{suffix}.tsr\n"
        f"  dy:  dy_gap{suffix}.tsr\n"
        f"  dx:  dx_gap{suffix}.tsr\n"
    )

    yaml = (
        f"op: gap_fwd_bwd\n"
        f"dtype: {dtype_label}\n"
        f"batch: {batch}\n"
        f"H: {H}\n"
        f"W: {W}\n"
        f"C: {C}\n"
        f"seed: {seed}\n"
        f"mse_threshold: {1e-3 if is_fp16 else 1e-6}\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"GAP FWD+BWD {dtype_label} reference generated in: {ws}")
    print(f"  x shape:  [{batch}, {H}, {W}, {C}] (NHWC)")
    print(f"  y shape:  [{batch}, 1, 1, {C}] (NHWC)")


def main():
    parser = argparse.ArgumentParser(description='GAP FWD+BWD test data generator')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--H', type=int, default=7)
    parser.add_argument('--W', type=int, default=7)
    parser.add_argument('--C', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp32', choices=['fp16', 'fp32'])
    args = parser.parse_args()

    generate(args.batch, args.H, args.W, args.C,
             args.seed, args.workspace, args.dtype)


if __name__ == '__main__':
    main()
