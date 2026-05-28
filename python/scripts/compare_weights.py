#!/usr/bin/env python3
"""
对比 GPU (FP32) 与 AMP (FP16) 初始化后的 FC 权重/偏置 TSR 文件，输出 MSE。
"""

import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from tsr_v4 import load_single_tensor

WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "workspace")

pairs = [
    ("flatten_output", "Flatten Output [200,784]"),
    ("fc1_weight", "FC1 Weight [512,784]"),
    ("fc1_bias",   "FC1 Bias   [512]"),
    ("fc2_weight", "FC2 Weight [256,512]"),
    ("fc2_bias",   "FC2 Bias   [256]"),
    ("fc3_weight", "FC3 Weight [10,256]"),
    ("fc3_bias",   "FC3 Bias   [10]"),
]

print("=" * 60)
print("GPU (FP32) vs AMP (FP16) 初始化权重/偏置 MSE 对比")
print("Workspace:", WORKSPACE)
print("=" * 60)

for base, desc in pairs:
    gpu_path = os.path.join(WORKSPACE, f"{base}_gpu.tsr")
    amp_path = os.path.join(WORKSPACE, f"{base}_amp.tsr")

    if not os.path.exists(gpu_path):
        print(f"[SKIP] {desc}: {gpu_path} not found")
        continue
    if not os.path.exists(amp_path):
        print(f"[SKIP] {desc}: {amp_path} not found")
        continue

    gpu_arr = load_single_tensor(gpu_path)
    amp_arr = load_single_tensor(amp_path)

    # AMP is FP16, promote to FP32 for fair comparison
    amp_arr_f = amp_arr.astype(np.float32)
    gpu_arr_f = gpu_arr.astype(np.float32)

    diff = gpu_arr_f - amp_arr_f
    mse = np.mean(diff ** 2)
    max_err = np.max(np.abs(diff))
    mean_err = np.mean(np.abs(diff))

    print(f"\n{desc}")
    print(f"  GPU  shape={gpu_arr.shape} dtype={gpu_arr.dtype}")
    print(f"  AMP  shape={amp_arr.shape} dtype={amp_arr.dtype}")
    print(f"  MSE       = {mse:.6e}")
    print(f"  Mean|err| = {mean_err:.6e}")
    print(f"  Max |err| = {max_err:.6e}")

print("\n" + "=" * 60)
print("对比完成")
print("=" * 60)
