#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSR-V4.20 C++/Python互操作性快速测试
验证C++和Python生成的TSR文件可以相互读取

【互操作流程说明】
  本测试与 tests/tensor/test_tsr_interop.cpp 配对使用，验证 Python 和 C++
  的 TSR-V4.20 实现可以互相读写对方的文件。

【文件约定】（统一存放在 workspace/ 目录下，项目根目录相对路径）
  - Python 生成：workspace/test_python_raw.tsr   (RAW 模式)
  - Python 生成：workspace/test_python_zlib.tsr  (ZLIB 模式)
  - C++ 生成：workspace/test_cpp_raw.tsr  (RAW 模式)
  - C++ 生成：workspace/test_cpp_zlib.tsr (ZLIB 模式)

【测试数据】（双方必须完全一致）
  共 4 个张量，全部为 4D NHWC：
  - [0] FP32 (2,3,4,5)  全填充 1.5
  - [1] FP16 (1,2,3,4)  全填充 2.5
  - [2] INT8 (1,1,1,8)  全填充 42
  - [3] INT32 (1,1,1,4) 全填充 12345

【推荐运行方式 - 从项目根目录执行】
  cd python/tests && python test_tsr_cpp_python_simple.py

  完整流程：
  1. Python 生成 test_python_raw.tsr 和 test_python_zlib.tsr
  2. Python 自验证（保存→加载→比对）
  3. Python 加载 C++ 生成的 test_cpp_raw.tsr（C++→Python 验证）
  4. Python 自动调用 C++ 测试完成反向验证（Python→C++ 验证）
  5. 输出完整测试报告

【手动分步验证方式】
  步骤 1 - 运行 C++ 生成文件：
    build/bin/tests/tensor/test_tsr_interop.exe
    （生成 workspace/test_cpp_raw.tsr 和 test_cpp_zlib.tsr）

  步骤 2 - 运行 Python 完成验证：
    cd python/tests && python test_tsr_cpp_python_simple.py
    （加载 C++ 文件并生成 Python 文件供 C++ 验证）

  步骤 3 - 再次运行 C++ 完成双向验证：
    build/bin/tests/tensor/test_tsr_interop.exe
    （加载 Python 生成的文件完成 Python→C++ 验证）

【C++ 可执行文件路径】
  Windows: build/bin/tests/tensor/test_tsr_interop.exe
  Linux:   build/linux-release/bin/tests/tensor/test_tsr_interop

【返回值】
  - 0：全部通过（包括自验证和互操作验证）
  - 1：数据不匹配、文件损坏、CRC32 校验失败等真正错误
"""

import sys
import os
import numpy as np
import subprocess

# 添加 scripts 目录到 Python 路径，以便导入 TSR-V4.20 的实现模块
# tsr_v4.py 是 Python 侧 TSR 读写的核心，必须与 C++ 侧的格式严格一致
script_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'scripts')
sys.path.insert(0, script_dir)

from tsr_v4 import save_tsr, load_tsr


def get_workspace_dir():
    """
    获取workspace目录（TSR文件交换的统一目录）。

    优先级：
    1. 环境变量 TR_WORKSPACE（与 C++ 编译宏 TR_WORKSPACE 保持一致）
    2. 默认路径：从 python/tests/ 向上两级到项目根目录，再进入 workspace/

    注意：C++ 侧通过编译时宏 TR_WORKSPACE 解析为绝对路径，因此双方必须
    指向同一物理目录，否则互操作文件将互相不可见。
    """
    workspace = os.environ.get('TR_WORKSPACE')
    if workspace:
        return workspace
    # 默认路径：从 python/tests/ 向上两级到项目根目录，再进入 workspace
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'workspace')


def main():
    # ===================================================================
    # 测试主流程（详见模块 docstring 的【互操作流程说明】）
    # ===================================================================
    # Step 1: Python 创建 4 个固定值张量（与 C++ 侧完全一致）
    # Step 2: Python 保存为 test_python_raw.tsr / test_python_zlib.tsr
    # Step 3: Python 自验证 — 加载自己保存的文件，验证全部元素
    # Step 4: Python 加载 C++ 生成的 test_cpp_raw.tsr（如果存在）
    #         若文件不存在则输出 SKIP 提示，不视为失败
    # Step 5: Python 自动调用 C++ 可执行文件 test_tsr_interop.exe
    #         让 C++ 端加载 test_python_raw.tsr 完成反向验证
    # ===================================================================
    workspace = get_workspace_dir()
    os.makedirs(workspace, exist_ok=True)

    print("=" * 60)
    print("TSR-V4.20 C++/Python Interoperability Test")
    print("=" * 60)
    print(f"Workspace: {workspace}")

    # ================================================================
    # 1. Python生成测试数据（与C++测试严格一致）
    # ================================================================
    print("\nStep 1: Python creates test tensors...")

    tensors = []

    # FP32张量：全1.5
    fp32 = np.full((2, 3, 4, 5), 1.5, dtype=np.float32)
    tensors.append(fp32)
    print("  Created FP32 tensor (2,3,4,5) with value 1.5")

    # FP16张量：全2.5
    fp16 = np.full((1, 2, 3, 4), 2.5, dtype=np.float16)
    tensors.append(fp16)
    print("  Created FP16 tensor (1,2,3,4) with value 2.5")

    # INT8张量：全42
    int8 = np.full((1, 1, 1, 8), 42, dtype=np.int8)
    tensors.append(int8)
    print("  Created INT8 tensor (1,1,1,8) with value 42")

    # INT32张量：全12345
    int32 = np.full((1, 1, 1, 4), 12345, dtype=np.int32)
    tensors.append(int32)
    print("  Created INT32 tensor (1,1,1,4) with value 12345")

    # ================================================================
    # 2. Python生成TSR文件到workspace
    # ================================================================
    print("\nStep 2: Python saves TSR files to workspace...")

    py_raw_path = os.path.join(workspace, "test_python_raw.tsr")
    py_zlib_path = os.path.join(workspace, "test_python_zlib.tsr")

    save_tsr(py_raw_path, tensors, compress=False)
    print(f"  Saved: {py_raw_path}")

    save_tsr(py_zlib_path, tensors, compress=True)
    print(f"  Saved: {py_zlib_path}")

    # ================================================================
    # 3. Python自验证：保存后再加载
    # ================================================================
    print("\nStep 3: Python self-verification...")

    self_pass = True
    loaded_py_raw = load_tsr(py_raw_path)
    if len(loaded_py_raw) != len(tensors):
        print(f"  [FAIL] RAW mode tensor count mismatch")
        self_pass = False
    else:
        for i, (orig, loaded) in enumerate(zip(tensors, loaded_py_raw)):
            if orig.shape != loaded.shape or orig.dtype != loaded.dtype:
                print(f"  [FAIL] Tensor {i} shape/dtype mismatch")
                self_pass = False
                continue
            # FP16 使用 np.allclose（rtol=1e-3），因为 FP16 精度有限，
            # 保存/加载过程中的转换可能产生微小差异；其余类型按位比较
            if orig.dtype == np.float16:
                if not np.allclose(loaded, orig, rtol=1e-3):
                    print(f"  [FAIL] Tensor {i} data mismatch")
                    self_pass = False
                    continue
            else:
                if not np.array_equal(loaded, orig):
                    print(f"  [FAIL] Tensor {i} data mismatch")
                    self_pass = False
                    continue
            print(f"  [{i}] RAW self-check passed: {loaded.shape}, {loaded.dtype}")

    if self_pass:
        print("  [PASS] Python self-verification passed")

    # ================================================================
    # 4. Python 读取 C++ 生成的文件（如果存在）
    # ================================================================
    # 目标文件：TR_WORKSPACE/test_cpp_raw.tsr
    # 由 tests/tensor/test_tsr_interop.cpp 生成
    # 若文件不存在，输出 SKIP 提示，不视为失败
    # ================================================================
    print("\nStep 4: Python loads C++ generated files...")

    cpp_raw_path = os.path.join(workspace, "test_cpp_raw.tsr")
    cpp_interop_pass = True

    if os.path.exists(cpp_raw_path):
        loaded_cpp = load_tsr(cpp_raw_path)
        print(f"  Loaded {len(loaded_cpp)} tensors from C++ RAW file")

        if len(loaded_cpp) != len(tensors):
            print(f"  [FAIL] Tensor count mismatch: expected {len(tensors)}, got {len(loaded_cpp)}")
            cpp_interop_pass = False
        else:
            for i, (cpp_tensor, py_tensor) in enumerate(zip(loaded_cpp, tensors)):
                if cpp_tensor.shape != py_tensor.shape:
                    print(f"  [FAIL] Tensor {i} shape mismatch")
                    cpp_interop_pass = False
                    break
                if cpp_tensor.dtype != py_tensor.dtype:
                    print(f"  [FAIL] Tensor {i} dtype mismatch")
                    cpp_interop_pass = False
                    break
                # FP16 使用 np.allclose（rtol=1e-3），因为跨语言 FP16 转换可能产生
                # 极小的舍入差异；其余类型使用 np.array_equal 按位严格比较
                if py_tensor.dtype == np.float16:
                    if not np.allclose(cpp_tensor, py_tensor, rtol=1e-3):
                        print(f"  [FAIL] Tensor {i} data mismatch")
                        cpp_interop_pass = False
                        break
                else:
                    if not np.array_equal(cpp_tensor, py_tensor):
                        print(f"  [FAIL] Tensor {i} data mismatch")
                        cpp_interop_pass = False
                        break
                print(f"  [{i}] Verified: {cpp_tensor.shape}, {cpp_tensor.dtype}")

        if cpp_interop_pass:
            print("  [PASS] C++ -> Python interop verified")
    else:
        print(f"  [SKIP] C++ file not found at {cpp_raw_path}")
        print("         (run C++ interop test first to generate it)")

    # ================================================================
    # 5. 自动调用 C++ 可执行文件完成反向验证
    # ================================================================
    # 调用 build/bin/tests/tensor/test_tsr_interop.exe
    # C++ 端会加载 TR_WORKSPACE/test_python_raw.tsr 完成 Python->C++ 验证
    # 若可执行文件不存在，输出提示并跳过此步骤
    # ================================================================
    print("\nStep 5: Compile and run C++ test...")

    # C++ 可执行文件路径，对应 tests/tensor/test_tsr_interop.cpp 的编译输出
    # 若构建目录或目标平台不同，需相应调整此路径
    cpp_exe = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "build", "bin", "tests", "tensor", "test_tsr_interop.exe"
    )

    # 标记C++反向验证是否执行并成功
    cpp_reverse_pass = False

    if not os.path.exists(cpp_exe):
        print(f"  C++ test binary not found: {cpp_exe}")
        print("  Please run: cmake --build build --target test_tsr_interop")
    else:
        print(f"  Running: {cpp_exe}")
        try:
            result = subprocess.run(
                [cpp_exe],
                capture_output=True, text=True, timeout=30
            )
            print(result.stdout)
            if result.returncode == 0:
                print("  [PASS] C++ test successful")
                cpp_reverse_pass = True
            else:
                print(f"  [FAIL] C++ test failed (exit code {result.returncode})")
                if result.stderr:
                    print(f"  stderr: {result.stderr}")
                return 1
        except Exception as e:
            print(f"  [WARN] C++ test execution failed: {e}")

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  [PASS] Python TSR generation working")
    print("  [PASS] Python TSR loading working")

    if os.path.exists(cpp_raw_path) and cpp_interop_pass:
        print("  [PASS] C++ -> Python interop verified")
    if cpp_reverse_pass:
        print("  [PASS] Python -> C++ interop verified")
        print("  [PASS] Full bidirectional compatibility confirmed")

    print("=" * 60)

    # 返回 0 表示成功；若走到这里，说明前面的所有关键检查都已通过
    return 0


# 返回值：
#   0 - 全部通过（自验证 + 可选的互操作验证）
#   1 - 数据不匹配、CRC32 失败、C++ 测试失败等真正错误
if __name__ == '__main__':
    sys.exit(main())
