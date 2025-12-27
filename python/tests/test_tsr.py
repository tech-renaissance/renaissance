#!/usr/bin/env python3
"""
TSR V3 Python测试（PyTorch版本）
@version 2.0.0
@date 2025-12-27
@author 技术觉醒团队

测试流程：
1. 读取C++生成的TSR文件（4个dtype）→ torch.Tensor
2. 验证数据一致性
3. 删除C++生成的文件
4. 导出新的TSR文件（torch.Tensor → TSR）
5. 重新导入验证一致性
6. 清理所有生成的文件
"""

import sys
import os
import numpy as np

# 添加scripts目录到path
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
sys.path.insert(0, scripts_dir)

from tsr_io import import_tensor, export_tensor


def get_workspace_dir():
    """获取workspace目录"""
    # 尝试从环境变量读取
    workspace = os.environ.get('TR_WORKSPACE')
    if workspace:
        return workspace

    # 默认路径
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'workspace')


def tensors_equal(a, b, epsilon: float = 1e-6) -> bool:
    """
    比较两个PyTorch张量是否相等
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")

    if a.shape != b.shape:
        return False

    if a.dtype != b.dtype:
        return False

    # 浮点数使用allclose，整数使用equal
    if torch.is_floating_point(a):
        return torch.allclose(a, b, atol=epsilon)
    else:
        return torch.equal(a, b)


def cleanup_files(files):
    """删除文件列表"""
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed: {os.path.basename(f)}")
        except Exception as e:
            print(f"  Warning: Could not remove {os.path.basename(f)}: {e}")


def main():
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print("Install with: pip install torch")
        return 1

    print("=== TSR V3 Python Interoperability Test (PyTorch) ===")

    workspace_dir = get_workspace_dir()
    print(f"Workspace directory: {workspace_dir}")

    if not os.path.exists(workspace_dir):
        print(f"ERROR: Workspace directory does not exist: {workspace_dir}")
        print("Please run test_tsr_python.exe first to generate test files.")
        return 1

    passed = 0
    failed = 0

    cpp_generated_files = []
    python_generated_files = []

    # ==========================================================================
    # 第一部分：读取C++生成的文件
    # ==========================================================================

    print("\n=== Part 1: Reading C++ Generated Files ===")

    # -----------------------------------------------------------------
    # Test 1: FP32 (2D: 4x6, 全为1.0)
    # -----------------------------------------------------------------
    print("[Test 1] Reading FP32 tensor from C++...")
    filename = os.path.join(workspace_dir, "test_python_fp32.tsr")

    try:
        tensor = import_tensor(filename)

        # 验证
        expected_shape = (4, 6)
        expected_dtype = torch.float32

        if tuple(tensor.shape) != expected_shape:
            print(f"  FAILED: Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
            failed += 1
        elif tensor.dtype != expected_dtype:
            print(f"  FAILED: Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            failed += 1
        elif not torch.allclose(tensor, torch.ones_like(tensor), atol=1e-6):
            print(f"  FAILED: Data values incorrect")
            failed += 1
        else:
            print(f"  PASSED: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, all values≈1.0")
            passed += 1

        cpp_generated_files.append(filename)

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # -----------------------------------------------------------------
    # Test 2: BF16 (3D: 2x3x4, 全为1.0)
    # -----------------------------------------------------------------
    print("[Test 2] Reading BF16 tensor from C++...")
    filename = os.path.join(workspace_dir, "test_python_bf16.tsr")

    try:
        tensor = import_tensor(filename)

        # 验证（BF16读取为torch.bfloat16）
        expected_shape = (2, 3, 4)
        expected_dtype = torch.bfloat16  # BF16保持为bfloat16

        if tuple(tensor.shape) != expected_shape:
            print(f"  FAILED: Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
            failed += 1
        elif tensor.dtype != expected_dtype:
            print(f"  FAILED: Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            failed += 1
        elif not torch.allclose(tensor.float(), torch.ones_like(tensor.float()), atol=1e-3):  # 转float比较精度
            print(f"  FAILED: Data values incorrect")
            failed += 1
        else:
            print(f"  PASSED: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, all values≈1.0")
            passed += 1

        cpp_generated_files.append(filename)

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # -----------------------------------------------------------------
    # Test 3: INT32 (1D: 20, 全为1)
    # -----------------------------------------------------------------
    print("[Test 3] Reading INT32 tensor from C++...")
    filename = os.path.join(workspace_dir, "test_python_int32.tsr")

    try:
        tensor = import_tensor(filename)

        # 验证
        expected_shape = (20,)
        expected_dtype = torch.int32

        if tuple(tensor.shape) != expected_shape:
            print(f"  FAILED: Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
            failed += 1
        elif tensor.dtype != expected_dtype:
            print(f"  FAILED: Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            failed += 1
        elif not (tensor == 1).all().item():
            print(f"  FAILED: Data values incorrect")
            failed += 1
        else:
            print(f"  PASSED: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, all values=1")
            passed += 1

        cpp_generated_files.append(filename)

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # -----------------------------------------------------------------
    # Test 4: INT8 (2D: 8x8, 全为1)
    # -----------------------------------------------------------------
    print("[Test 4] Reading INT8 tensor from C++...")
    filename = os.path.join(workspace_dir, "test_python_int8.tsr")

    try:
        tensor = import_tensor(filename)

        # 验证
        expected_shape = (8, 8)
        expected_dtype = torch.int8

        if tuple(tensor.shape) != expected_shape:
            print(f"  FAILED: Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
            failed += 1
        elif tensor.dtype != expected_dtype:
            print(f"  FAILED: Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            failed += 1
        elif not (tensor == 1).all().item():
            print(f"  FAILED: Data values incorrect")
            failed += 1
        else:
            print(f"  PASSED: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, all values=1")
            passed += 1

        cpp_generated_files.append(filename)

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # ==========================================================================
    # 第二部分：删除C++生成的文件
    # ==========================================================================

    print("\n=== Part 2: Cleaning up C++ Generated Files ===")
    cleanup_files(cpp_generated_files)

    # ==========================================================================
    # 第三部分：PyTorch导出测试
    # ==========================================================================

    print("\n=== Part 3: PyTorch Export/Import Round-trip Test ===")

    # -----------------------------------------------------------------
    # Test 5: FP32导出/导入
    # -----------------------------------------------------------------
    print("[Test 5] FP32 export/import round-trip...")
    filename = os.path.join(workspace_dir, "test_python_fp32_export.tsr")

    try:
        # 创建测试tensor
        original = torch.ones((3, 4), dtype=torch.float32)

        # 导出
        export_tensor(original, filename, compress=False)
        python_generated_files.append(filename)

        # 导入
        loaded = import_tensor(filename)

        # 验证
        if not tensors_equal(original, loaded):
            print(f"  FAILED: Round-trip data mismatch")
            failed += 1
        else:
            print(f"  PASSED: FP32 round-trip successful")
            passed += 1

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # -----------------------------------------------------------------
    # Test 6: INT32导出/导入
    # -----------------------------------------------------------------
    print("[Test 6] INT32 export/import round-trip...")
    filename = os.path.join(workspace_dir, "test_python_int32_export.tsr")

    try:
        # 创建测试tensor
        original = torch.arange(12, dtype=torch.int32).reshape(3, 4)

        # 导出
        export_tensor(original, filename, compress=False)
        python_generated_files.append(filename)

        # 导入
        loaded = import_tensor(filename)

        # 验证
        if not tensors_equal(original, loaded):
            print(f"  FAILED: Round-trip data mismatch")
            failed += 1
        else:
            print(f"  PASSED: INT32 round-trip successful")
            passed += 1

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # -----------------------------------------------------------------
    # Test 7: INT8导出/导入
    # -----------------------------------------------------------------
    print("[Test 7] INT8 export/import round-trip...")
    filename = os.path.join(workspace_dir, "test_python_int8_export.tsr")

    try:
        # 创建测试tensor
        original = torch.arange(16, dtype=torch.int8).reshape(4, 4)

        # 导出
        export_tensor(original, filename, compress=False)
        python_generated_files.append(filename)

        # 导入
        loaded = import_tensor(filename)

        # 验证
        if not tensors_equal(original, loaded):
            print(f"  FAILED: Round-trip data mismatch")
            failed += 1
        else:
            print(f"  PASSED: INT8 round-trip successful")
            passed += 1

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # -----------------------------------------------------------------
    # Test 8: ZLIB压缩模式
    # -----------------------------------------------------------------
    print("[Test 8] ZLIB compression mode...")
    filename = os.path.join(workspace_dir, "test_python_zlib.tsr")

    try:
        # 创建稀疏tensor（很多0）
        original = torch.zeros((10, 10), dtype=torch.float32)
        original[5, 5] = 1.0  # 只有一个非零元素

        # 导出（ZLIB压缩）
        export_tensor(original, filename, compress=True)
        python_generated_files.append(filename)

        # 导入
        loaded = import_tensor(filename)

        # 验证
        if not tensors_equal(original, loaded):
            print(f"  FAILED: Round-trip data mismatch")
            failed += 1
        else:
            print(f"  PASSED: ZLIB mode successful")
            passed += 1

    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # ==========================================================================
    # 第四部分：清理所有文件
    # ==========================================================================

    print("\n=== Part 4: Final Cleanup ===")
    all_files = cpp_generated_files + python_generated_files
    cleanup_files(all_files)

    # ==========================================================================
    # 总结
    # ==========================================================================

    print("\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n========================================")
        print("     All PyTorch tests passed!")
        print("========================================")
        return 0
    else:
        print(f"\n{failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
