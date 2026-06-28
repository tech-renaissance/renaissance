#!/usr/bin/env python3
"""
test_relu_inf.py
Generate reference data for ReLU INF (inference) tests.

This script generates input and reference output tensors for ReLU inference testing.
Unlike forward+backward tests, inference tests don't need mask or backward pass data.
"""

import argparse
import torch
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate_reference_data(shape, dtype_str, seed, workspace):
    """
    Generate reference data for ReLU inference testing.

    Args:
        shape: Tuple of (N, H, W, C)
        dtype_str: "fp32" or "fp16"
        seed: Random seed
        workspace: Path to workspace directory
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Parse shape
    N, H, W, C = shape

    # Create dtype mapping
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16
    }
    np_dtype_map = {
        "fp32": np.float32,
        "fp16": np.float16
    }
    dtype = dtype_map[dtype_str]
    np_dtype = np_dtype_map[dtype_str]

    # Print configuration
    print(f"Shape: {N}x{H}x{W}x{C}")
    print(f"Dtype: {dtype_str}")
    print(f"Seed: {seed}")

    # Generate random input tensor
    # Use normal distribution for more realistic test data
    x = torch.randn(N, C, H, W, dtype=dtype)

    # Apply ReLU (forward inference only)
    y = torch.relu(x)

    # Convert from NCHW to NHWC format (our framework uses NHWC)
    x = x.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    y = y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    # Create workspace directory if it doesn't exist
    os.makedirs(workspace, exist_ok=True)

    # Determine file suffix based on dtype
    suffix = "_amp" if dtype_str == "fp16" else "_fp32"

    # Save tensors using TSR format
    save_tsr(os.path.join(workspace, f"x_inf{suffix}.tsr"),
             [x.cpu().numpy().astype(np_dtype)], compress=False)
    save_tsr(os.path.join(workspace, f"y_ref_inf{suffix}.tsr"),
             [y.cpu().numpy().astype(np_dtype)], compress=False)

    # Save configuration as YAML (for compatibility with C++ test)
    config = (
        f"shape:\n"
        f"  - {N}\n"
        f"  - {H}\n"
        f"  - {W}\n"
        f"  - {C}\n"
        f"dtype: {dtype_str}\n"
        f"seed: {seed}\n"
    )
    with open(os.path.join(workspace, "config.yaml"), "w") as f:
        f.write(config)

    print(f"ReLU INF {dtype_str.upper()} reference generated in: {workspace}")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")

def main():
    parser = argparse.ArgumentParser(description="Generate reference data for ReLU INF tests")
    parser.add_argument("--shape", type=str, default="8,1024,1024,8",
                       help="Tensor shape as N,H,W,C (default: 8,1024,1024,8)")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"],
                       help="Data type (default: fp32)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--workspace", type=str, default="./workspace",
                       help="Workspace directory (default: ./workspace)")

    args = parser.parse_args()

    # Parse shape
    shape = tuple(map(int, args.shape.split(",")))

    if len(shape) != 4:
        print("Error: Shape must be N,H,W,C (4 dimensions)")
        sys.exit(1)

    generate_reference_data(shape, args.dtype, args.seed, args.workspace)

if __name__ == "__main__":
    main()
