#!/usr/bin/env python3
"""
Conv INF 推理测试 — PyTorch 参考数据生成
INF: Y = Conv2d(X, W, stride, pad) — 无 bias，推理模式
支持 --dtype fp16 | fp32。
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr


def generate(batch: int, IH: int, IW: int, C: int, K: int,
             kernel: int, stride: int, pad: int,
             seed: int, ws: str, dtype: str):
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

    R = kernel
    S = kernel
    OH = (IH + 2 * pad - R) // stride + 1
    OW = (IW + 2 * pad - S) // stride + 1

    # ── 创建张量 ──
    # X: NHWC [B, H, W, C] → NCHW [B, C, H, W] for PyTorch conv2d
    X_nhwc = torch.randn(batch, IH, IW, C, dtype=torch_dtype) * 0.5
    X = X_nhwc.permute(0, 3, 1, 2).contiguous()

    # W: KRSC [K, R, S, C] → OIHW [K, C, R, S] for PyTorch conv2d
    W_krsc = torch.randn(K, R, S, C, dtype=torch_dtype) * 0.1
    W_conv = W_krsc.permute(0, 3, 1, 2).contiguous()

    # ── INF: Y = Conv2d(X, W, stride, pad) ──
    Y = F.conv2d(X, W_conv, stride=stride, padding=pad)  # NCHW [B, K, OH, OW]
    Y_nhwc = Y.permute(0, 2, 3, 1).contiguous()           # NHWC [B, OH, OW, K]

    print(f"Conv INF {dtype_label} reference:")
    print(f"  Input:  [{batch}, {IH}, {IW}, {C}] (NHWC)")
    print(f"  Weight: [{K}, {R}, {S}, {C}] (KRSC)")
    print(f"  Output: [{batch}, {OH}, {OW}, {K}] (NHWC)")

    # ── 导出 .tsr ──
    save_tsr(os.path.join(ws, f'x_conv{suffix}.tsr'),
             [X_nhwc.numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'w_conv{suffix}.tsr'),
             [W_krsc.numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'y_conv_ref{suffix}.tsr'),
             [Y_nhwc.detach().numpy().astype(np_dtype)], compress=False)

    file_desc = (
        f"  x:        x_conv{suffix}.tsr\n"
        f"  w:        w_conv{suffix}.tsr\n"
        f"  y_ref:    y_conv_ref{suffix}.tsr\n"
    )

    yaml = (
        f"op: conv_inf\n"
        f"dtype: {dtype_label}\n"
        f"batch: {batch}\n"
        f"ih: {IH}\n"
        f"iw: {IW}\n"
        f"c: {C}\n"
        f"k: {K}\n"
        f"kernel: {kernel}\n"
        f"stride: {stride}\n"
        f"pad: {pad}\n"
        f"seed: {seed}\n"
        f"mse_threshold: 1e-3\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"  Workspace: {ws}")


def main():
    parser = argparse.ArgumentParser(description='Conv INF test data generator')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--IH', type=int, default=32)
    parser.add_argument('--IW', type=int, default=32)
    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--pad', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'])
    args = parser.parse_args()

    generate(args.batch, args.IH, args.IW, args.C, args.K,
             args.kernel, args.stride, args.pad,
             args.seed, args.workspace, args.dtype)


if __name__ == '__main__':
    main()