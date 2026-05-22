#!/usr/bin/env python3
"""
ReLU 算子数学正确性验证 — PyTorch 参考数据生成
输出: TSR 张量 + YAML 配置 → workspace/
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate(op_type: str, dtype_str: str, shape: tuple, seed: int, ws: str):
    N, H, W, C = shape
    is_fp16 = (dtype_str == "FP16")
    torch_dtype = torch.float16 if is_fp16 else torch.float32
    np_dtype    = np.float16    if is_fp16 else np.float32

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cudnn.fp32_precision = 'ieee'
    except AttributeError:
        pass

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(ws, exist_ok=True)

    # 文件名后缀：同时区分dtype和op_type，避免6个测试相互覆盖
    # 格式: {op_type}_{dtype后缀}.tsr
    dtype_suffix = "amp" if is_fp16 else "fp32"

    if op_type == 'fwd':
        X = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
        Y = torch.relu(X)
        mask = (X > 0).to(torch.int8)

        save_tsr(os.path.join(ws, f'input_fwd_{dtype_suffix}.tsr'),
                 [X.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'output_ref_fwd_{dtype_suffix}.tsr'),
                 [Y.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'mask_ref_fwd_{dtype_suffix}.tsr'),
                 [mask.cpu().numpy().astype(np.int8)], compress=False)

        file_desc = (
            f"  input: input_fwd_{dtype_suffix}.tsr\n"
            f"  output: output_ref_fwd_{dtype_suffix}.tsr\n"
            f"  mask: mask_ref_fwd_{dtype_suffix}.tsr\n"
        )
    else:
        X = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
        mask = (X > 0).to(torch.int8)

        torch.manual_seed(seed + 1000)
        dY = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
        dX = dY * mask.float().to(torch_dtype)

        save_tsr(os.path.join(ws, f'input_bwd_{dtype_suffix}.tsr'),
                 [dY.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'output_ref_bwd_{dtype_suffix}.tsr'),
                 [dX.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'mask_ref_bwd_{dtype_suffix}.tsr'),
                 [mask.cpu().numpy().astype(np.int8)], compress=False)

        file_desc = (
            f"  input: input_bwd_{dtype_suffix}.tsr\n"
            f"  output: output_ref_bwd_{dtype_suffix}.tsr\n"
            f"  mask: mask_ref_bwd_{dtype_suffix}.tsr\n"
        )

    yaml = (
        f"op: relu\n"
        f"op_type: {op_type}\n"
        f"dtype: {dtype_str}\n"
        f"shape: [{N}, {H}, {W}, {C}]\n"
        f"seed: {seed}\n"
        f"mse_threshold: {'1e-3' if is_fp16 else '1e-5'}\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"ReLU {op_type} {dtype_str} reference generated in: {ws}")

def main():
    parser = argparse.ArgumentParser(description='ReLU test data generator')
    parser.add_argument('--op_type', required=True, choices=['fwd', 'bwd'])
    parser.add_argument('--dtype',   default='FP32', choices=['FP32', 'FP16'])
    parser.add_argument('--shape',   default='8,1024,1024,8',   # 67M elements = 256MB FP32 / 128MB FP16
                        help='NHWC shape (comma-separated, e.g. 1,1024,1,1)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()

    dims = tuple(int(x) for x in args.shape.split(','))
    if len(dims) != 4:
        print(f"Invalid shape '{args.shape}': must be N,H,W,C (4D)")
        sys.exit(1)

    generate(args.op_type, args.dtype, dims, args.seed, args.workspace)

if __name__ == '__main__':
    main()