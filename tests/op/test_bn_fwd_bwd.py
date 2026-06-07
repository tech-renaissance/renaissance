#!/usr/bin/env python3
"""
BN FWD+BWD+INF 测试 — PyTorch 参考数据生成
支持 BN1D (H=W=1) 和 BN2D (H,W>1)
FWD: 训练模式前向，更新 running stats
BWD: 标准 BN 反向
INF: 推理模式，使用更新后的 running stats
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr


def generate(is_bn1d, batch, H, W, C, eps, momentum, seed, ws, dtype):
    if dtype == 'fp16':
        torch_dtype = torch.float16
        np_dtype = np.float16
        suffix = '_amp'
        dtype_label = 'FP16'
        is_fp16 = True
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
        suffix = '_fp32'
        dtype_label = 'FP32'
        is_fp16 = False

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

    if is_bn1d:
        H, W = 1, 1

    # PyTorch uses NCHW internally
    X_nchw = torch.randn(batch, C, H, W, dtype=torch.float32) * 0.5
    X_nchw = X_nchw.to(torch_dtype)

    # Parameters (FP32, same as our initializer convention)
    gamma = torch.ones(C, dtype=torch.float32) + torch.randn(C, dtype=torch.float32) * 0.2
    beta = torch.randn(C, dtype=torch.float32) * 0.2
    running_mean = torch.zeros(C, dtype=torch.float32)
    running_var = torch.ones(C, dtype=torch.float32)

    gamma_t = gamma.to(torch_dtype)
    beta_t = beta.to(torch_dtype)

    # ========================================================================
    # FWD — compute in FP32 to match cuDNN FE compute_data_type=FLOAT
    # ========================================================================
    X_fp32 = X_nchw.float()
    mean = X_fp32.mean(dim=(0, 2, 3))
    # PyTorch BatchNorm uses biased variance (divides by N*H*W)
    var = X_fp32.var(dim=(0, 2, 3), unbiased=False)
    inv_std = 1.0 / torch.sqrt(var + eps)

    Y_fp32 = (X_fp32 - mean.view(1, C, 1, 1)) * \
             inv_std.view(1, C, 1, 1) * gamma.view(1, C, 1, 1) + beta.view(1, C, 1, 1)
    Y_nchw = Y_fp32.to(torch_dtype)

    # Update running stats (same semantics as our implementation)
    running_mean_new = (1.0 - momentum) * running_mean + momentum * mean
    # cuDNN uses unbiased variance for running_var update
    running_var_new = (1.0 - momentum) * running_var + momentum * X_fp32.var(dim=(0, 2, 3), unbiased=True)

    saved_mean = mean
    saved_inv_var = inv_std

    # ========================================================================
    # BWD — compute in FP32 to match cuDNN FE compute_data_type=FLOAT
    # ========================================================================
    torch.manual_seed(seed + 1000)
    dY_nchw = torch.randn(batch, C, H, W, dtype=torch.float32) * 0.5
    dY_nchw = dY_nchw.to(torch_dtype)

    X_var = X_fp32.clone().detach().requires_grad_(True)     # FP32
    gamma_var = gamma.clone().detach().requires_grad_(True)   # FP32
    beta_var = beta.clone().detach().requires_grad_(True)     # FP32

    mean_v = X_var.mean(dim=(0, 2, 3))
    var_v = X_var.var(dim=(0, 2, 3), unbiased=False)
    inv_std_v = 1.0 / torch.sqrt(var_v + eps)

    Y_var = (X_var - mean_v.view(1, C, 1, 1)) * \
            inv_std_v.view(1, C, 1, 1) * gamma_var.view(1, C, 1, 1) + beta_var.view(1, C, 1, 1)

    dY_fp32 = dY_nchw.float()
    Y_var.backward(dY_fp32)
    dX_nchw = X_var.grad.to(torch_dtype)
    dgamma = gamma_var.grad
    dbeta = beta_var.grad

    # ========================================================================
    # INF — using updated running stats
    # ========================================================================
    eq_scale = gamma / torch.sqrt(running_var_new + eps)
    eq_bias = beta - running_mean_new * eq_scale
    Y_inf_nchw = X_fp32 * eq_scale.view(1, C, 1, 1) + eq_bias.view(1, C, 1, 1)
    Y_inf_nchw = Y_inf_nchw.to(torch_dtype)

    # ========================================================================
    # Convert NCHW -> NHWC for TSR export
    # ========================================================================
    X_nhwc = X_nchw.permute(0, 2, 3, 1).contiguous()
    Y_nhwc = Y_nchw.permute(0, 2, 3, 1).contiguous()
    dY_nhwc = dY_nchw.permute(0, 2, 3, 1).contiguous()
    dX_nhwc = dX_nchw.permute(0, 2, 3, 1).contiguous()
    Y_inf_nhwc = Y_inf_nchw.permute(0, 2, 3, 1).contiguous()

    def to_nhwc_1d(t):
        return t.view(1, 1, 1, C).contiguous()

    # ========================================================================
    # Export .tsr files
    # ========================================================================
    save_tsr(os.path.join(ws, f'x_bn{suffix}.tsr'),
             [X_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'y_bn_ref{suffix}.tsr'),
             [Y_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dy_bn{suffix}.tsr'),
             [dY_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'dx_bn_ref{suffix}.tsr'),
             [dX_nhwc.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(ws, f'gamma_bn{suffix}.tsr'),
             [to_nhwc_1d(gamma).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'beta_bn{suffix}.tsr'),
             [to_nhwc_1d(beta).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_mean_init_bn{suffix}.tsr'),
             [to_nhwc_1d(running_mean).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_var_init_bn{suffix}.tsr'),
             [to_nhwc_1d(running_var).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_mean_new_bn{suffix}.tsr'),
             [to_nhwc_1d(running_mean_new).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_var_new_bn{suffix}.tsr'),
             [to_nhwc_1d(running_var_new).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'saved_mean_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(saved_mean).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'saved_inv_var_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(saved_inv_var).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'dgamma_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(dgamma.float()).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'dbeta_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(dbeta.float()).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'y_inf_bn_ref{suffix}.tsr'),
             [Y_inf_nhwc.cpu().numpy().astype(np_dtype)], compress=False)

    # Scalars
    eps_np = np.array([[[[eps]]]], dtype=np.float32)
    momentum_np = np.array([[[[momentum]]]], dtype=np.float32)
    save_tsr(os.path.join(ws, f'eps_bn{suffix}.tsr'), [eps_np], compress=False)
    save_tsr(os.path.join(ws, f'momentum_bn{suffix}.tsr'), [momentum_np], compress=False)

    print(f"BN {'1D' if is_bn1d else '2D'} FWD+BWD+INF {dtype_label} reference generated in: {ws}")
    print(f"  Input shape:  [{batch}, {H}, {W}, {C}] (NHWC)")
    print(f"  Output shape: [{batch}, {H}, {W}, {C}] (NHWC)")
    print(f"  eps={eps}, momentum={momentum}")


def main():
    parser = argparse.ArgumentParser(description='BN FWD+BWD+INF test data generator')
    parser.add_argument('--bn1d', action='store_true', help='Generate BN1D data (H=W=1)')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--H', type=int, default=4, help='Height (ignored for BN1D)')
    parser.add_argument('--W', type=int, default=4, help='Width (ignored for BN1D)')
    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp32', choices=['fp16', 'fp32'])
    args = parser.parse_args()

    generate(args.bn1d, args.batch, args.H, args.W, args.C,
             args.eps, args.momentum, args.seed, args.workspace, args.dtype)


if __name__ == '__main__':
    main()
