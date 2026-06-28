#!/usr/bin/env python3
"""
FC FWD+BWD 串接测试 — PyTorch 参考数据生成
使用 torch.nn.Linear 进行真正的全连接层计算
FWD: y = fc(x)  where fc = nn.Linear(in_features, out_features)
BWD: dx = dy @ fc.weight  (dX = dY @ W, 其中 W 是 O×I 矩阵)
支持 --dtype fp16 | fp32。
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate(batch: int, in_features: int, out_features: int, bias: bool,
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

    # ── 创建 torch.nn.Linear ──
    fc = torch.nn.Linear(in_features, out_features, bias=bias)
    fc = fc.to(dtype=torch_dtype)

    # 用固定种子初始化权重和bias（复现性）
    with torch.no_grad():
        fc.weight.copy_(torch.randn(out_features, in_features, dtype=torch_dtype) * 0.1)
        if bias:
            fc.bias.copy_(torch.randn(out_features, dtype=torch.float32) * 0.1)

    # ── FWD: y = fc(x) ──
    X = torch.randn(batch, 1, 1, in_features, dtype=torch_dtype) * 2.0  # NHWC
    X_2d = X.view(batch, in_features)  # Linear 接受 [B, I]
    Y = fc(X_2d)                       # 真正的 torch.nn.Linear.forward
    Y_4d = Y.view(batch, 1, 1, out_features)  # 恢复 NHWC

    # ── BWD: dX = dY @ W ──
    torch.manual_seed(seed + 1000)
    dY = torch.randn(batch, 1, 1, out_features, dtype=torch_dtype) * 2.0
    dY_2d = dY.view(batch, out_features)         # [B, O]
    dX_2d = torch.mm(dY_2d, fc.weight.data)      # dX = dY @ W,  W is [O, I]
    dX = dX_2d.view(batch, 1, 1, in_features)    # 恢复 NHWC

    # 计算梯度
    X_2d = X.view(batch, in_features)            # [B, I]
    dW_2d = torch.mm(dY_2d.t(), X_2d)            # dW = dY^T @ X  [O, I]
    dW = dW_2d.view(out_features, 1, 1, in_features).contiguous()

    if bias:
        dB_2d = dY_2d.sum(dim=0)                 # dB = sum(dY, dim=0) [O]
    else:
        dB_2d = torch.zeros(out_features, dtype=torch.float32)
    dB = dB_2d.view(1, 1, 1, out_features).contiguous()

    # 权重转化为 NHWC [out_features, 1, 1, in_features]
    # PyTorch Linear 的 weight 是 [O, I]，在 NHWC/KRSC 中应映射为 [O, 1, 1, I]
    # K=out_features, R=1, S=1, C=in_features（符合 WEIGHT.md 规范）
    W_4d = fc.weight.data.view(out_features, 1, 1, in_features).contiguous()

    # Bias 永远是 FP32
    if bias:
        B_4d = fc.bias.data.to(torch.float32).view(1, 1, 1, out_features).contiguous()
    else:
        B_4d = torch.zeros(1, 1, 1, out_features, dtype=torch.float32)

    # 导出 .tsr
    save_tsr(os.path.join(ws, f'x_fwd_bwd{suffix}.tsr'),
             [X.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'w_fwd_bwd{suffix}.tsr'),
             [W_4d.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'b_fwd_bwd{suffix}.tsr'),
             [B_4d.cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'y_ref_fwd_bwd{suffix}.tsr'),
             [Y_4d.detach().cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dy_fwd_bwd{suffix}.tsr'),
             [dY.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dx_ref_fwd_bwd{suffix}.tsr'),
             [dX.detach().cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dw_ref_fwd_bwd{suffix}.tsr'),
             [dW.detach().cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'db_ref_fwd_bwd{suffix}.tsr'),
             [dB.detach().cpu().numpy().astype(np.float32)], compress=False)

    file_desc = (
        f"  x:   x_fwd_bwd{suffix}.tsr\n"
        f"  w:   w_fwd_bwd{suffix}.tsr\n"
        f"  b:   b_fwd_bwd{suffix}.tsr\n"
        f"  y:   y_ref_fwd_bwd{suffix}.tsr\n"
        f"  dy:  dy_fwd_bwd{suffix}.tsr\n"
        f"  dx:  dx_ref_fwd_bwd{suffix}.tsr\n"
        f"  dw:  dw_ref_fwd_bwd{suffix}.tsr\n"
        f"  db:  db_ref_fwd_bwd{suffix}.tsr\n"
    )

    yaml = (
        f"op: fc_fwd_bwd\n"
        f"dtype: {dtype_label}\n"
        f"batch: {batch}\n"
        f"in_features: {in_features}\n"
        f"out_features: {out_features}\n"
        f"bias: {'true' if bias else 'false'}\n"
        f"seed: {seed}\n"
        f"mse_threshold: 1e-3\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"FC FWD+BWD {dtype_label} reference generated in: {ws}")
    print(f"  Weight shape: [{out_features}, 1, 1, {in_features}] (NHWC)")


def main():
    parser = argparse.ArgumentParser(description='FC FWD+BWD test data generator')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--in', dest='in_features', type=int, default=1024)
    parser.add_argument('--out', dest='out_features', type=int, default=512)
    parser.add_argument('--no-bias', action='store_true', dest='no_bias',
                        help='Disable bias in the Linear layer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'],
                        help='Data type: fp16 (AMP) or fp32 (CPU/GPU)')
    args = parser.parse_args()

    generate(args.batch, args.in_features, args.out_features,
             not args.no_bias, args.seed, args.workspace, args.dtype)


if __name__ == '__main__':
    main()
