#!/usr/bin/env python3
"""
compare_detailed.py - 对比 GPU (FP32) 与 AMP (FP16) 模式下的张量差异

用法:
    python compare_detailed.py                    # 使用默认 workspace 对比预设张量
    python compare_detailed.py <workspace_dir>    # 指定 workspace 目录
    python compare_detailed.py <gpu.tsr> <amp.tsr> [name]  # 对比两个指定文件

依赖: tsr_v4.py (同目录)
"""

import sys
import os
import numpy as np

# 确保能导入同目录的 tsr_v4
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tsr_v4 import load_single_tensor

DEFAULT_WORKSPACE = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "workspace")


def compare_tensors(name: str, gpu_path: str, amp_path: str) -> bool:
    """对比两个 TSR 文件，打印统计信息。"""
    if not os.path.exists(gpu_path):
        print(f"[MISSING] GPU: {gpu_path}")
        return False
    if not os.path.exists(amp_path):
        print(f"[MISSING] AMP: {amp_path}")
        return False

    gpu = load_single_tensor(gpu_path)
    amp = load_single_tensor(amp_path)

    diff = np.abs(gpu.astype(np.float64) - amp.astype(np.float64))

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  GPU  shape={gpu.shape} dtype={gpu.dtype}")
    print(f"  AMP  shape={amp.shape} dtype={amp.dtype}")
    print(f"  MSE       = {np.mean(diff**2):.6e}")
    print(f"  Mean|err| = {np.mean(diff):.6e}")
    print(f"  Max |err| = {np.max(diff):.6e}")

    n = min(16, gpu.size)
    gpu_flat = gpu.reshape(-1)
    amp_flat = amp.reshape(-1)
    print(f"  GPU first {n}: {gpu_flat[:n]}")
    print(f"  AMP first {n}: {amp_flat[:n]}")
    return True


def compare_preset(workspace_dir: str) -> None:
    """对比 workspace 中预设的一组张量。"""
    pairs = [
        ("I_A_DATA",       "ia_data_gpu.tsr",        "ia_data_amp.tsr"),
        ("I_B_DATA",       "ib_data_gpu.tsr",        "ib_data_amp.tsr"),
        ("Flatten Output", "flatten_output_gpu.tsr", "flatten_output_amp.tsr"),
        ("FC1 Weight",     "fc1_weight_gpu.tsr",     "fc1_weight_amp.tsr"),
        ("FC1 Bias",       "fc1_bias_gpu.tsr",       "fc1_bias_amp.tsr"),
        ("FC2 Weight",     "fc2_weight_gpu.tsr",     "fc2_weight_amp.tsr"),
        ("FC2 Bias",       "fc2_bias_gpu.tsr",       "fc2_bias_amp.tsr"),
        ("FC3 Weight",     "fc3_weight_gpu.tsr",     "fc3_weight_amp.tsr"),
        ("FC3 Bias",       "fc3_bias_gpu.tsr",       "fc3_bias_amp.tsr"),
    ]

    found_any = False
    for name, gpu_name, amp_name in pairs:
        gpu_path = os.path.join(workspace_dir, gpu_name)
        amp_path = os.path.join(workspace_dir, amp_name)
        if compare_tensors(name, gpu_path, amp_path):
            found_any = True

    if not found_any:
        print(f"\n警告: workspace 目录 '{workspace_dir}' 中未找到任何对比文件。")
        print("请先运行 test_dl_full_gpu 和 test_dl_full_amp 生成 .tsr 文件。")


def main():
    if len(sys.argv) == 3:
        # 对比两个指定文件
        gpu_path = sys.argv[1]
        amp_path = sys.argv[2]
        name = os.path.basename(gpu_path)
        compare_tensors(name, gpu_path, amp_path)
    elif len(sys.argv) == 4:
        # 对比两个指定文件，带名称
        gpu_path = sys.argv[1]
        amp_path = sys.argv[2]
        name = sys.argv[3]
        compare_tensors(name, gpu_path, amp_path)
    elif len(sys.argv) == 2:
        # 指定 workspace 目录
        compare_preset(sys.argv[1])
    else:
        # 默认 workspace
        compare_preset(DEFAULT_WORKSPACE)


if __name__ == "__main__":
    main()
