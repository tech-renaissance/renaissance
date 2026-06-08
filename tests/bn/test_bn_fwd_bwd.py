#!/usr/bin/env python3
"""
BN FWD+BWD+INF 测试 — PyTorch 内置 BatchNorm2d 参考数据生成

与 legacy 版 test_bn_fwd_bwd.py（tests/op/test_bn_fwd_bwd_legacy.py）的关键区别：
  - FWD: 使用 nn.BatchNorm2d (training=True)，即 PyTorch C++ batch_norm 前向 kernel
  - BWD: 使用 F.batch_norm(training=True) + autograd，即 PyTorch C++ batch_norm_backward kernel
  - INF: 使用 F.batch_norm(training=False)，即 PyTorch eval 模式推理路径
  - saved_mean / saved_invstd 从 NativeBatchNormBackward0 提取（PyTorch 内部实际使用的值）
  - 不再手动拼装 mean/var/inv_std 计算图，彻底消除与 PyTorch 内置实现的数值偏差

支持 BN1D (H=W=1) 和 BN2D (H,W>1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr


def generate(is_bn1d, batch, H, W, C, eps, momentum, seed, ws, dtype, device='cpu'):
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

    # Disable TF32 for exact FP32 reference alignment
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cudnn.fp32_precision = 'ieee'
    except AttributeError:
        pass
    try:
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    except AttributeError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.makedirs(ws, exist_ok=True)

    if is_bn1d:
        H, W = 1, 1

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---- 生成输入数据 (NCHW) — 先在CPU上生成确保种子一致，再移动到目标设备 ----
    X_nchw = (torch.randn(batch, C, H, W, dtype=torch.float32) * 0.5).to(dev).to(torch_dtype)

    # ---- 生成参数 (FP32) ----
    gamma = (torch.ones(C, dtype=torch.float32) + torch.randn(C, dtype=torch.float32) * 0.2).to(dev)
    beta  = (torch.randn(C, dtype=torch.float32) * 0.2).to(dev)
    running_mean_init = torch.zeros(C, dtype=torch.float32, device=dev)
    running_var_init  = torch.ones(C, dtype=torch.float32, device=dev)

    # ====================================================================
    # FWD — 使用 nn.BatchNorm2d (training=True)
    # 这会调用 PyTorch C++ batch_norm 前向 kernel，
    # running_mean / running_var 被原地更新。
    # ====================================================================
    bn_fwd = nn.BatchNorm2d(
        C, eps=eps, momentum=momentum,
        affine=True, track_running_stats=True
    )
    bn_fwd.weight.data = gamma.clone()
    bn_fwd.bias.data   = beta.clone()
    bn_fwd.train()

    # 初始 running stats 设为 0 / 1（与 C++ 初始化一致）
    bn_fwd.running_mean.data = running_mean_init.clone()
    bn_fwd.running_var.data  = running_var_init.clone()

    X_fwd = X_nchw.clone().detach()
    Y_nchw = bn_fwd(X_fwd)

    # 提取 PyTorch 内部实际使用的 saved_mean / saved_invstd
    # NativeBatchNormBackward0._saved_result1 = saved_mean
    # NativeBatchNormBackward0._saved_result2 = saved_invstd
    saved_mean   = Y_nchw.grad_fn._saved_result1.detach().clone()
    saved_invstd = Y_nchw.grad_fn._saved_result2.detach().clone()

    # 提取更新后的 running stats
    running_mean_new = bn_fwd.running_mean.detach().clone()
    running_var_new  = bn_fwd.running_var.detach().clone()

    # ====================================================================
    # BWD — 使用 F.batch_norm (training=True) + autograd
    # 这会调用 PyTorch C++ batch_norm_backward kernel。
    # 使用 momentum=0.0 + dummy running stats，避免再次更新 running stats。
    # ====================================================================
    torch.manual_seed(seed + 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1000)
    dY_nchw = (torch.randn(batch, C, H, W, dtype=torch.float32) * 0.5).to(dev).to(torch_dtype)

    X_bwd     = X_nchw.clone().detach().requires_grad_(True)
    gamma_bwd = gamma.clone().detach().requires_grad_(True)
    beta_bwd  = beta.clone().detach().requires_grad_(True)

    # dummy running stats：内容不重要，training=True 时 PyTorch 只使用 batch stats 计算梯度
    rm_dummy = torch.zeros(C, dtype=torch.float32, device=dev)
    rv_dummy = torch.ones(C, dtype=torch.float32, device=dev)

    Y_bwd = F.batch_norm(
        X_bwd, rm_dummy, rv_dummy,
        weight=gamma_bwd, bias=beta_bwd,
        training=True, momentum=0.0, eps=eps
    )

    dY_fp32 = dY_nchw.float()
    Y_bwd.backward(dY_fp32)

    dX_nchw = X_bwd.grad.to(torch_dtype)
    dgamma  = gamma_bwd.grad
    dbeta   = beta_bwd.grad

    # ====================================================================
    # INF — 使用 F.batch_norm (training=False)
    # 这是 PyTorch eval 模式推理路径，直接使用更新后的 running_mean / running_var。
    # ====================================================================
    with torch.no_grad():
        Y_inf_nchw = F.batch_norm(
            X_nchw, running_mean_new, running_var_new,
            weight=gamma, bias=beta,
            training=False, eps=eps
        )

    # eq_scale / eq_bias：C++ fused 推理路径的预计算参数
    with torch.no_grad():
        eq_scale = gamma / torch.sqrt(running_var_new + eps)
        eq_bias  = beta - running_mean_new * eq_scale

    # ====================================================================
    # Convert NCHW -> NHWC for TSR export
    # ====================================================================
    X_nhwc     = X_nchw.permute(0, 2, 3, 1).contiguous()
    Y_nhwc     = Y_nchw.detach().permute(0, 2, 3, 1).contiguous()
    dY_nhwc    = dY_nchw.detach().permute(0, 2, 3, 1).contiguous()
    dX_nhwc    = dX_nchw.detach().permute(0, 2, 3, 1).contiguous()
    Y_inf_nhwc = Y_inf_nchw.detach().permute(0, 2, 3, 1).contiguous()

    def to_nhwc_1d(t):
        return t.view(1, 1, 1, C).contiguous()

    # ====================================================================
    # Export .tsr files
    # ====================================================================
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
             [to_nhwc_1d(running_mean_init).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_var_init_bn{suffix}.tsr'),
             [to_nhwc_1d(running_var_init).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_mean_new_bn{suffix}.tsr'),
             [to_nhwc_1d(running_mean_new).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'running_var_new_bn{suffix}.tsr'),
             [to_nhwc_1d(running_var_new).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'saved_mean_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(saved_mean).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'saved_inv_var_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(saved_invstd).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'dgamma_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(dgamma.float()).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'dbeta_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(dbeta.float()).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'y_inf_bn_ref{suffix}.tsr'),
             [Y_inf_nhwc.cpu().numpy().astype(np_dtype)], compress=False)

    # eq_scale / eq_bias 参考值（供 C++ fused 推理路径验证）
    save_tsr(os.path.join(ws, f'eq_scale_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(eq_scale).cpu().numpy().astype(np.float32)], compress=False)
    save_tsr(os.path.join(ws, f'eq_bias_bn_ref{suffix}.tsr'),
             [to_nhwc_1d(eq_bias).cpu().numpy().astype(np.float32)], compress=False)

    # Scalars
    eps_np      = np.array([[[[eps]]]], dtype=np.float32)
    momentum_np = np.array([[[[momentum]]]], dtype=np.float32)
    save_tsr(os.path.join(ws, f'eps_bn{suffix}.tsr'),      [eps_np],      compress=False)
    save_tsr(os.path.join(ws, f'momentum_bn{suffix}.tsr'), [momentum_np], compress=False)

    print(f"BN {'1D' if is_bn1d else '2D'} FWD+BWD+INF {dtype_label} reference generated in: {ws}")
    print(f"  Input shape:  [{batch}, {H}, {W}, {C}] (NHWC)")
    print(f"  Output shape: [{batch}, {H}, {W}, {C}] (NHWC)")
    print(f"  eps={eps}, momentum={momentum}")
    print(f"  NOTE: Using nn.BatchNorm2d / F.batch_norm (PyTorch built-in C++ kernels)")


def main():
    parser = argparse.ArgumentParser(
        description='BN FWD+BWD+INF test data generator (PyTorch built-in, v2)'
    )
    parser.add_argument('--bn1d', action='store_true', help='Generate BN1D data (H=W=1)')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--H', type=int, default=224, help='Height (ignored for BN1D)')
    parser.add_argument('--W', type=int, default=224, help='Width (ignored for BN1D)')
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp32', choices=['fp16', 'fp32'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    generate(args.bn1d, args.batch, args.H, args.W, args.C,
             args.eps, args.momentum, args.seed, args.workspace, args.dtype, args.device)


if __name__ == '__main__':
    main()
