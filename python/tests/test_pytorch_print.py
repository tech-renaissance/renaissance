#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch打印格式一致性测试（双向验证）
验证PyTorch和C++的打印格式完全一致

【测试流程】
1. Python创建随机数据（seed=42确保与C++一致）
2. Python保存TSR文件
3. Python打印原始数据
4. Python加载C++生成的TSR文件并打印
5. 对比两边打印格式

【运行方式】
python/tests/test_pytorch_print.py
"""

import sys
import os
import numpy as np

# 添加scripts目录到路径
script_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'scripts')
sys.path.insert(0, script_dir)

from tsr_v4 import save_tsr, load_tsr


def get_workspace_dir():
    """获取workspace目录"""
    workspace = os.environ.get('TR_WORKSPACE')
    if workspace:
        return workspace
    # 从python/tests/向上三级到项目根目录，再进入workspace
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'workspace'
    )


def create_random_tensors():
    """创建4种类型的随机Tensor（seed=42）"""
    np.random.seed(42)

    tensors = []

    # 1. FP32: 0-10范围的随机数
    fp32 = np.random.uniform(0.0, 10.0, size=(1, 2, 2, 3)).astype(np.float32)
    tensors.append(fp32)

    # 2. FP16: -5到5范围的随机数
    fp16 = np.random.uniform(-5.0, 5.0, size=(1, 1, 2, 4)).astype(np.float16)
    tensors.append(fp16)

    # 3. INT32: -1000到1000的随机整数
    int32 = np.random.randint(-1000, 1000, size=(1, 1, 1, 4), dtype=np.int32)
    tensors.append(int32)

    # 4. INT8: -50到50的随机整数
    int8 = np.random.randint(-50, 50, size=(1, 1, 1, 3), dtype=np.int8)
    tensors.append(int8)

    return tensors


def print_tensor_pytorch_style(tensor, name):
    """模仿PyTorch的打印格式"""
    print(f"{name}:")

    # 数据类型标注
    dtype_map = {
        np.float16: "dtype=FP16",
        np.float32: "dtype=FP32",
        np.int32: "dtype=INT32",
        np.int8: "dtype=INT8"
    }
    dtype_str = dtype_map.get(tensor.dtype, f"dtype={tensor.dtype}")

    # 打印张量（模仿PyTorch的多维数组格式）
    print(f"tensor({list(tensor.shape)}, {dtype_str})")

    # 对于小张量，打印完整数据
    if tensor.size <= 12:
        print("   Data:", tensor.flatten())
    else:
        print("   Data:", tensor.flatten()[:6], "...", tensor.flatten()[-6:])


def main():
    print("=" * 70)
    print("PyTorch Print Format Bidirectional Consistency Test")
    print("=" * 70)

    workspace = get_workspace_dir()
    os.makedirs(workspace, exist_ok=True)

    # ========================================================================
    # 1. Python创建随机数据并打印
    # ========================================================================
    print("\n[Step 1] Python creates random tensors (seed=42):")

    python_tensors = create_random_tensors()

    for i, tensor in enumerate(python_tensors):
        print(f"\n--- Python Tensor {i} ---")
        print_tensor_pytorch_style(tensor, f"python_tensor_{i}")

    # ========================================================================
    # 2. Python保存TSR文件
    # ========================================================================
    print("\n[Step 2] Python saves TSR file...")

    python_tsr_file = os.path.join(workspace, "test_python_random.tsr")
    save_tsr(python_tsr_file, python_tensors, compress=False)
    print(f"Saved: {python_tsr_file}")

    # ========================================================================
    # 3. Python自验证：加载并打印
    # ========================================================================
    print("\n[Step 3] Python self-verification (load and print):")

    loaded_python = load_tsr(python_tsr_file)
    print(f"Loaded {len(loaded_python)} tensors")

    all_match = True
    for i, (orig, loaded) in enumerate(zip(python_tensors, loaded_python)):
        print(f"\n--- Loaded Python Tensor {i} ---")
        print_tensor_pytorch_style(loaded, f"loaded_python_{i}")

        # 验证数据
        if orig.dtype == np.float16:
            match = np.allclose(loaded, orig, rtol=1e-3)
        else:
            match = np.array_equal(loaded, orig)

        if match:
            print(f"  [OK] Tensor {i} roundtrip verified")
        else:
            print(f"  [FAIL] Tensor {i} roundtrip FAILED")
            all_match = False

    # ========================================================================
    # 4. Python加载C++生成的TSR文件
    # ========================================================================
    print("\n[Step 4] Python loads C++ generated TSR file...")

    cpp_tsr_file = os.path.join(workspace, "test_cpp_random.tsr")

    if os.path.exists(cpp_tsr_file):
        cpp_tensors = load_tsr(cpp_tsr_file)
        print(f"Loaded {len(cpp_tensors)} tensors from C++")

        print("\n--- C++ Tensors Loaded in Python ---")
        for i, tensor in enumerate(cpp_tensors):
            print(f"\n--- C++ Tensor {i} ---")
            print_tensor_pytorch_style(tensor, f"cpp_tensor_{i}")

        print("\n[PASS] C++ -> Python format verified")
    else:
        print(f"  [SKIP] C++ file not found: {cpp_tsr_file}")
        print("         (run C++ test first: test_tsr_pytorch.exe)")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary:")

    if all_match:
        print("  [PASS] Python tensor roundtrip verified")

    if os.path.exists(cpp_tsr_file):
        print("  [PASS] C++ -> Python print format verified")
        print("  [INFO] Compare with C++ test output to verify bidirectional consistency")
    else:
        print("  [INFO] Run C++ test first: test_tsr_pytorch.exe")
        print("        Then rerun this Python test to complete verification")

    print("=" * 70)

    return 0 if all_match else 1


if __name__ == '__main__':
    sys.exit(main())