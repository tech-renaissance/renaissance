#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSR-V4.20 格式回环测试（简化版，避免Windows编码问题）
"""

import sys
import os
import numpy as np
import tempfile

# 添加scripts目录到路径
script_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, script_dir)

from tsr_v4 import save_tsr, load_single_tensor, load_tsr


def test_single_tensor_raw():
    """测试单张量RAW模式"""
    print("\n=== Test 1: Single Tensor RAW ===")

    shape = (2, 3, 4, 5)  # NHWC
    arr_orig = np.random.randn(*shape).astype(np.float32)

    print(f"Original: shape={arr_orig.shape}, dtype={arr_orig.dtype}")
    print(f"Stats: min={arr_orig.min():.6f}, max={arr_orig.max():.6f}")

    with tempfile.NamedTemporaryFile(suffix='.tsr', delete=False) as f:
        temp_file = f.name

    try:
        # 保存
        save_tsr(temp_file, arr_orig, compress=False)
        print(f"[PASS] Saved to {temp_file}")

        # 读取
        arr_loaded = load_single_tensor(temp_file)
        print(f"[PASS] Loaded")

        # 验证
        assert arr_loaded.shape == arr_orig.shape
        assert arr_loaded.dtype == arr_orig.dtype
        assert np.allclose(arr_loaded, arr_orig, rtol=1e-6)
        print(f"[PASS] Verified: shape={arr_loaded.shape}, dtype={arr_loaded.dtype}")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_single_tensor_zlib():
    """测试单张量ZLIB模式"""
    print("\n=== Test 2: Single Tensor ZLIB ===")

    shape = (4, 8, 16, 32)
    arr_orig = np.random.randn(*shape).astype(np.float16)

    print(f"Original: shape={arr_orig.shape}, dtype={arr_orig.dtype}")

    with tempfile.NamedTemporaryFile(suffix='.tsr', delete=False) as f:
        temp_file = f.name

    try:
        # 保存（压缩）
        save_tsr(temp_file, arr_orig, compress=True)
        file_size = os.path.getsize(temp_file)
        print(f"[PASS] Saved (compressed): {file_size} bytes")

        # 读取
        arr_loaded = load_single_tensor(temp_file)
        print(f"[PASS] Loaded")

        # 验证
        assert arr_loaded.shape == arr_orig.shape
        assert arr_loaded.dtype == arr_orig.dtype
        assert np.allclose(arr_loaded, arr_orig, rtol=1e-3)
        print(f"[PASS] Verified")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_multi_tensor_raw():
    """测试多张量RAW模式"""
    print("\n=== Test 3: Multi-Tensor RAW ===")

    arrays = [
        np.random.randn(1, 64, 7, 7).astype(np.float32),
        np.random.randn(1, 1, 1, 64).astype(np.float32),
        np.random.randn(1, 1, 1, 64).astype(np.float16),
        np.random.randint(-128, 127, (2, 3, 4, 5), dtype=np.int8),
        np.random.randint(0, 1000, (1, 1, 1, 10), dtype=np.int32),
    ]

    print(f"Created {len(arrays)} tensors")

    with tempfile.NamedTemporaryFile(suffix='.tsr', delete=False) as f:
        temp_file = f.name

    try:
        # 保存
        save_tsr(temp_file, arrays, compress=False)
        print(f"[PASS] Saved {len(arrays)} tensors")

        # 读取
        loaded_arrays = load_tsr(temp_file)
        print(f"[PASS] Loaded {len(loaded_arrays)} tensors")

        # 验证
        assert len(loaded_arrays) == len(arrays)
        for i, (orig, loaded) in enumerate(zip(arrays, loaded_arrays)):
            assert loaded.shape == orig.shape
            assert loaded.dtype == orig.dtype
            if orig.dtype == np.float16:
                assert np.allclose(loaded, orig, rtol=1e-3)
            else:
                assert np.array_equal(loaded, orig)
            print(f"  [{i}] OK: {loaded.shape}, {loaded.dtype}")

        print("[PASS] All tensors verified")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_multi_tensor_zlib():
    """测试多张量ZLIB模式（模拟ResNet-50首层）"""
    print("\n=== Test 4: Multi-Tensor ZLIB (ResNet-50 Stage 0) ===")

    arrays = [
        np.random.randn(64, 3, 7, 7).astype(np.float16) * 0.1,  # conv1 weight
        np.zeros((1, 1, 1, 64), dtype=np.float32),              # conv1 bias
        np.ones((1, 1, 1, 64), dtype=np.float32),               # bn1 scale
        np.zeros((1, 1, 1, 64), dtype=np.float32),              # bn1 bias
    ]

    print(f"Created {len(arrays)} tensors (ResNet-50 stage 0)")
    for i, arr in enumerate(arrays):
        print(f"  [{i}] {arr.shape}, {arr.dtype}")

    with tempfile.NamedTemporaryFile(suffix='.tsr', delete=False) as f:
        temp_file = f.name

    try:
        # 保存（压缩）
        save_tsr(temp_file, arrays, compress=True)
        file_size_raw = sum(arr.nbytes for arr in arrays)
        file_size_compressed = os.path.getsize(temp_file)
        ratio = file_size_compressed / file_size_raw
        print(f"[PASS] Saved: {file_size_raw} -> {file_size_compressed} bytes ({ratio:.1%})")

        # 读取
        loaded_arrays = load_tsr(temp_file)
        print(f"[PASS] Loaded {len(loaded_arrays)} tensors")

        # 验证
        assert len(loaded_arrays) == len(arrays)
        for i, (orig, loaded) in enumerate(zip(arrays, loaded_arrays)):
            assert loaded.shape == orig.shape
            assert loaded.dtype == orig.dtype
            if orig.dtype == np.float16:
                assert np.allclose(loaded, orig, rtol=1e-3)
            else:
                assert np.allclose(loaded, orig, rtol=1e-6)
            print(f"  [{i}] OK: {loaded.shape}, {loaded.dtype}")

        print("[PASS] All tensors verified")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("TSR-V4.20 Loopback Test Suite")
    print("="*60)

    tests = [
        ("Single Tensor RAW", test_single_tensor_raw),
        ("Single Tensor ZLIB", test_single_tensor_zlib),
        ("Multi-Tensor RAW", test_multi_tensor_raw),
        ("Multi-Tensor ZLIB", test_multi_tensor_zlib),
    ]

    results = []
    for name, func in tests:
        try:
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总结果
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print("\n" + "="*60)
    print(f"Total: {passed}/{total} passed")
    print("="*60)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
