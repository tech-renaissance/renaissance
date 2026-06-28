#!/usr/bin/env python3
"""
Flatten+FC+ReLU+FC 复合算子正确性测试 — PyTorch 参考数据生成
FWD: X[7,28,28,1] → Flatten → [7,1,1,784] → FC1(784→512) → ReLU → FC2(512→256)
BWD: dY2[7,1,1,256] → FC2_BWD → ReLU_BWD → FC1_BWD → Flatten_BWD → dX[7,28,28,1]
支持 --dtype fp16 | fp32
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr


def generate(seed: int, ws: str, dtype: str):
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

    # ── 固定形状 ──
    batch = 7
    H, W, C = 28, 28, 1
    fc1_in, fc1_out_dim = 784, 512
    fc2_in, fc2_out_dim = 512, 256

    fc1 = torch.nn.Linear(fc1_in, fc1_out_dim, bias=True)
    fc2 = torch.nn.Linear(fc2_in, fc2_out_dim, bias=True)
    fc1 = fc1.to(dtype=torch_dtype)
    fc2 = fc2.to(dtype=torch_dtype)

    with torch.no_grad():
        fc1.weight.copy_(torch.randn(fc1_out_dim, fc1_in, dtype=torch_dtype) * 0.1)
        fc1.bias.copy_(torch.randn(fc1_out_dim, dtype=torch.float32) * 0.1)
        fc2.weight.copy_(torch.randn(fc2_out_dim, fc2_in, dtype=torch_dtype) * 0.1)
        fc2.bias.copy_(torch.randn(fc2_out_dim, dtype=torch.float32) * 0.1)

    # ── FWD ──
    X_nhwc = torch.randn(batch, H, W, C, dtype=torch_dtype) * 2.0

    X_flat_2d = X_nhwc.reshape(batch, fc1_in)
    flat_out = X_flat_2d.view(batch, 1, 1, fc1_in)

    # FC1 FWD
    fc1_out_2d = fc1(X_flat_2d)
    fc1_out = fc1_out_2d.view(batch, 1, 1, fc1_out_dim)

    # ReLU FWD
    relu_out_2d = F.relu(fc1_out_2d)
    relu_out = relu_out_2d.view(batch, 1, 1, fc2_in)
    mask_2d = (fc1_out_2d > 0).to(torch.int8)
    mask = mask_2d.view(batch, 1, 1, fc2_in)

    # FC2 FWD
    fc2_out_2d = fc2(relu_out_2d)
    fc2_out = fc2_out_2d.view(batch, 1, 1, fc2_out_dim)

    # ── BWD ──
    torch.manual_seed(seed + 1000)
    dY2_nhwc = torch.randn(batch, 1, 1, fc2_out_dim, dtype=torch_dtype) * 2.0
    dY2_2d = dY2_nhwc.view(batch, fc2_out_dim)

    # FC2 BWD
    W2 = fc2.weight.data
    dX2_2d = torch.mm(dY2_2d, W2)
    dX2 = dX2_2d.view(batch, 1, 1, fc2_in)
    dW2_2d = torch.mm(dY2_2d.t(), relu_out_2d)
    dW2 = dW2_2d.view(fc2_out_dim, 1, 1, fc2_in).contiguous()
    dB2_2d = dY2_2d.sum(dim=0).to(torch.float32)
    dB2 = dB2_2d.view(1, 1, 1, fc2_out_dim).contiguous()

    # ReLU BWD
    dX1_2d = torch.where(mask_2d > 0, dX2_2d, torch.zeros_like(dX2_2d))
    dX1 = dX1_2d.view(batch, 1, 1, fc1_out_dim)

    # FC1 BWD
    W1 = fc1.weight.data
    X_2d = X_flat_2d
    dX0_2d = torch.mm(dX1_2d, W1)
    dX0 = dX0_2d.view(batch, 1, 1, fc1_in)
    dW1_2d = torch.mm(dX1_2d.t(), X_2d)
    dW1 = dW1_2d.view(fc1_out_dim, 1, 1, fc1_in).contiguous()
    dB1_2d = dX1_2d.sum(dim=0).to(torch.float32)
    dB1 = dB1_2d.view(1, 1, 1, fc1_out_dim).contiguous()

    # Flatten BWD
    dX_spatial = dX0_2d.reshape(batch, H, W, C).contiguous()

    # ── Weight NHWC ──
    W1_4d = W1.view(fc1_out_dim, 1, 1, fc1_in).contiguous()
    W2_4d = W2.view(fc2_out_dim, 1, 1, fc2_in).contiguous()
    B1_4d = fc1.bias.data.to(torch.float32).view(1, 1, 1, fc1_out_dim).contiguous()
    B2_4d = fc2.bias.data.to(torch.float32).view(1, 1, 1, fc2_out_dim).contiguous()

    # ── 保存 ──
    def s(name, data, np_dtype_override=None):
        dt = np_dtype_override if np_dtype_override else np_dtype
        save_tsr(os.path.join(ws, f'{name}{suffix}.tsr'),
                 [data.detach().cpu().numpy().astype(dt)], compress=False)

    s('x',         X_nhwc)
    s('flat_out',  flat_out)
    s('w1',        W1_4d)
    s('b1',        B1_4d, np.float32)
    s('fc1_out',   fc1_out)
    s('relu_out',  relu_out)
    s('mask',      mask, np.int8)
    s('w2',        W2_4d)
    s('b2',        B2_4d, np.float32)
    s('fc2_out',   fc2_out)

    s('dy2',       dY2_nhwc)
    s('dx2_ref',   dX2)
    s('dw2_ref',   dW2)
    s('db2_ref',   dB2, np.float32)
    s('dx1_ref',   dX1)
    s('dx0_ref',   dX0)
    s('dw1_ref',   dW1)
    s('db1_ref',   dB1, np.float32)
    s('dx_ref',    dX_spatial)

    file_desc = (
        f"  x:         x{suffix}.tsr\n"
        f"  flat_out:  flat_out{suffix}.tsr\n"
        f"  w1:        w1{suffix}.tsr\n"
        f"  b1:        b1{suffix}.tsr\n"
        f"  fc1_out:   fc1_out{suffix}.tsr\n"
        f"  relu_out:  relu_out{suffix}.tsr\n"
        f"  mask:      mask{suffix}.tsr\n"
        f"  w2:        w2{suffix}.tsr\n"
        f"  b2:        b2{suffix}.tsr\n"
        f"  fc2_out:   fc2_out{suffix}.tsr\n"
        f"  dy2:       dy2{suffix}.tsr\n"
        f"  dx2_ref:   dx2_ref{suffix}.tsr\n"
        f"  dw2_ref:   dw2_ref{suffix}.tsr\n"
        f"  db2_ref:   db2_ref{suffix}.tsr\n"
        f"  dx1_ref:   dx1_ref{suffix}.tsr\n"
        f"  dx0_ref:   dx0_ref{suffix}.tsr\n"
        f"  dw1_ref:   dw1_ref{suffix}.tsr\n"
        f"  db1_ref:   db1_ref{suffix}.tsr\n"
        f"  dx_ref:    dx_ref{suffix}.tsr\n"
    )

    yaml = (
        f"op: flatten_fc_relu_fc\n"
        f"dtype: {dtype_label}\n"
        f"shape: [{batch}, {H}, {W}, {C}] (NHWC)\n"
        f"fc1: {fc1_in}→{fc1_out}\n"
        f"fc2: {fc2_in}→{fc2_out}\n"
        f"seed: {seed}\n"
        f"mse_threshold: {'1e-3' if is_fp16 else '1e-5'}\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"Flatten+FC+ReLU+FC {dtype_label} reference generated in: {ws}")


def main():
    parser = argparse.ArgumentParser(
        description='Flatten+FC+ReLU+FC composite test data generator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp32', choices=['fp16', 'fp32'])
    args = parser.parse_args()
    generate(args.seed, args.workspace, args.dtype)


if __name__ == '__main__':
    main()
