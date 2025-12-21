#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file configure.py
@brief 技术觉醒深度学习框架 V3.1.0 配置脚本
@details 根目录配置脚本，支持6大场景和16依赖项的完整配置系统
@version 3.01.00
@date 2025-12-21
@author 技术觉醒团队
"""

import os
import sys

# 添加python/scripts到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "scripts"))

# 导入配置数据
from dependency_data import SCENE_DEPS, DEP_CONFIG

# ============================================================================
# 可自定义的配置变量（用户可根据需要修改）
# ============================================================================

# vcpkg路径（如果使用vcpkg管理依赖）
VCPKG_ROOT = os.environ.get("VCPKG_ROOT", "T:/Softwares/vcpkg")

# 构建目录
BUILD_DIR = "build/cmake-build-release"

# 并行构建线程数
PARALLEL_JOBS = 30

# CMake生成器
CMAKE_GENERATOR = "Ninja"

# 构建类型
CMAKE_BUILD_TYPE = "Release"

# 是否启用详细输出
VERBOSE_OUTPUT = False

# ============================================================================
# 调用智能配置模块
# ============================================================================

def main():
    """主函数 - 调用智能配置模块"""

    # 创建全局变量供smart_config.py使用
    global VCPKG_ROOT, BUILD_DIR, PARALLEL_JOBS, CMAKE_GENERATOR
    global CMAKE_BUILD_TYPE, VERBOSE_OUTPUT

    # 导入并运行智能配置模块
    from smart_config import run_smart_config

    success = run_smart_config()

    if success:
        # 构建脚本名称根据平台确定
        script_name = ".\\build.bat" if sys.platform == "win32" else "./build.sh"
        print(f"\nNext steps:")
        print(f"  1. Run {script_name} in the project root directory")
        print(f"  2. Find examples in the build directory: cd {BUILD_DIR}")
    else:
        print("\n[FAIL] Configuration incomplete")

    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)