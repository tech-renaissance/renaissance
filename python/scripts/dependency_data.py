# -*- coding: utf-8 -*-
"""
@file dependency_data.py
@brief 技术觉醒框架依赖配置数据（Phase 1：核心场景精简版）
@details 只包含PC-CUDA和GPU云场景的必需依赖
@version 3.00.04
@date 2025-01-01
@author 技术觉醒团队
"""

# ============================================================================
# 场景定义（Phase 1：三大核心场景）
# ============================================================================

SCENE_DEPS = {
    "pc_cuda_win": {
        "name": "PC-CUDA (Windows)",
        "description": "Windows + NVIDIA GPU + MSVC",
        "required": ["cmake", "ninja", "msvc", "cuda", "cudnn", "onednn", "xnnpack", "zlib", "libcurl", "python", "numpy"],
        "optional": [],
        "cmake_opts": ["-DTR_USE_CUDA=ON", "-DTR_USE_MUSA=OFF"]
    },
    "pc_cuda_linux": {
        "name": "PC-CUDA (Linux)",
        "description": "Linux + NVIDIA GPU + GCC",
        "required": ["cmake", "ninja", "gcc", "cuda", "cudnn", "onednn", "xnnpack", "zlib", "libcurl", "python", "numpy"],
        "optional": [],
        "cmake_opts": ["-DTR_USE_CUDA=ON", "-DTR_USE_MUSA=OFF"]
    },
    "gpu_cloud": {
        "name": "GPU云服务器",
        "description": "Linux + Multi-NVIDIA GPU + GCC",
        "required": ["cmake", "ninja", "gcc", "cuda", "cudnn", "onednn", "xnnpack", "zlib", "libcurl", "python", "numpy"],
        "optional": [],
        "cmake_opts": ["-DTR_USE_CUDA=ON", "-DTR_USE_MUSA=OFF"]
    }
}

# ============================================================================
# 依赖配置（Phase 1：精简版）
# ============================================================================

DEP_CONFIG = {
    # 基础构建工具
    "cmake": {
        "name": "CMake",
        "exe": ["cmake", "cmake.exe"],
        "env": ["CMAKE_ROOT"],
        "paths_win": [
            "C:/Program Files/CMake/bin",
            "T:/Softwares/CMake/bin",
            "C:/Program Files/Microsoft Visual Studio/*/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
        ],
        "paths_linux": ["/usr/bin", "/usr/local/bin"],
        "version_cmd": ["cmake", "--version"],
        "version_pattern": r"cmake version (\d+\.\d+\.\d+)",
        "min_version": "3.24.0",
        "user_selection": True,  # 保留user_selection，但会在search_dependency中处理平台差异
        "install_hint": "Windows: https://cmake.org/download/ | Linux: sudo apt install cmake"
    },

    "ninja": {
        "name": "Ninja",
        "exe": ["ninja", "ninja.exe"],
        "env": [],
        "paths_win": [
            "C:/Program Files/Ninja",
            "C:/Program Files/Microsoft Visual Studio/2022/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja",
            "B:/Softwares/JetBrains/CLion */bin/ninja/win/x64",
            "T:/Softwares/msys64/mingw64/bin",
            "C:/Python314"
        ],
        "paths_linux": ["/usr/bin", "/usr/local/bin"],
        "version_cmd": ["ninja", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "user_selection": True,  # 保留user_selection，但会在search_dependency中处理平台差异
        "install_hint": "pip install ninja | Linux: sudo apt install ninja-build"
    },

    # 编译器
    "msvc": {
        "name": "MSVC",
        "exe": ["cl.exe"],
        "env": [],
        "paths_win": [
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/*/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/*/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/*/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2019/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64",
            "C:/Program Files/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC/*/bin/Hostx64/x64"
        ],
        "search_pattern": "*/bin/Hostx64/x64/cl.exe",
        "version_cmd": ["cl", "--version"],
        "user_selection": True,
        "install_hint": "Install Visual Studio 2022 with C++ development tools"
    },

    "gcc": {
        "name": "GCC",
        "exe": ["g++", "gcc"],
        "env": [],
        "paths_linux": ["/usr/bin", "/usr/local/bin"],
        "version_cmd": ["g++", "--version"],
        "version_pattern": r"g\+\+.*?(\d+\.\d+\.\d+)",
        "min_version": "13.0.0",
        "user_selection": True,
        "install_hint": "Linux: sudo apt install g++-13"
    },

    # GPU工具链
    "cuda": {
        "name": "CUDA Toolkit",
        "exe": ["nvcc", "nvcc.exe"],
        "env": ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"],
        "paths_win": [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.*"
        ],
        "paths_linux": ["/usr/local/cuda", "/usr/local/cuda-13.*"],
        "bin_subdir": "bin",
        "version_cmd": ["nvcc", "--version"],
        "version_pattern": r"release (\d+\.\d+)",
        "min_version": "13.0",
        "install_hint": "https://developer.nvidia.com/cuda-toolkit"
    },

    "cudnn": {
        "name": "cuDNN",
        "exe": [],  # cuDNN主要通过头文件和库文件检测
        "header": "cudnn.h",
        "lib_files": ["cudnn64.dll", "libcudnn.so"],
        "env": ["CUDNN_ROOT", "CUDNN_PATH"],
        "paths_win": [
            "C:/Program Files/NVIDIA/CUDNN",
            "C:/Program Files/NVIDIA/CUDNN/v9.*"
        ],
        "paths_linux": ["/usr/local/cuda", "/usr", "/usr/local/cuda/lib64"],
        "version_cmd": ["grep", "CUDNN_MAJOR", "-A", "2", "{include}/cudnn_version.h"],
        "version_pattern": r"#define CUDNN_MAJOR\s+(\d+).*#define CUDNN_MINOR\s+(\d+).*#define CUDNN_PATCHLEVEL\s+(\d+)",
        "min_version": "9.17",
        "install_hint": "https://developer.nvidia.com/cudnn"
    },

    # Python生态
    "python": {
        "name": "Python",
        "exe": ["python3", "python3.14", "python", "python.exe"],
        "env": ["PYTHON_HOME", "VIRTUAL_ENV"],
        "paths_win": [
            "C:/Python314",
            "C:/Python313",
            "C:/Python312",
            "C:/Users/*/AppData/Local/Programs/Python/Python3*",
            "T:/Softwares/msys64/mingw64/bin"
        ],
        "paths_linux": ["/home/ubuntu/venv/py314/bin", "/usr/bin", "/usr/local/bin"],
        "version_cmd": ["python", "--version"],
        "version_pattern": r"Python (\d+\.\d+\.\d+)",
        "min_version": "3.12.0",
        "user_selection": True,
        "install_hint": "https://python.org | conda create -n renaissance python=3.12+"
    },

    "numpy": {
        "name": "NumPy",
        "exe": [],  # 通过Python命令检查
        "check_cmd": ["python", "-c", "import numpy; print(numpy.__version__)"],
        "min_version": "1.24.0",
        "install_hint": "pip install numpy"
    },

    # 基础库
    "onednn": {
        "name": "oneDNN",
        "exe": [],  # 通过头文件检测
        "header": ["dnnl.hpp", "dnn.hpp"],
        "env": ["ONEDNN_ROOT", "DNNL_ROOT"],
        "vcpkg_packages": ["dnnl"],
        "install_hint": "vcpkg install dnnl"
    },

    "zlib": {
        "name": "zlib",
        "exe": [],  # 通过头文件检测
        "header": "zlib.h",
        "lib_files": ["zlib1.dll", "libz.so"],
        "env": ["ZLIB_ROOT"],
        "vcpkg_packages": ["zlib"],
        "install_hint": "vcpkg install zlib"
    },

    "libcurl": {
        "name": "libcurl",
        "exe": ["curl"],  # 通过curl命令检测系统安装
        "header": "curl/curl.h",
        "lib_files": ["libcurl.dll", "libcurl.so"],
        "env": ["CURL_ROOT"],
        "paths_linux": ["/usr", "/usr/local", "/usr/local/curl"],
        "vcpkg_packages": ["curl"],
        "install_hint": "vcpkg install curl"
    },

    "xnnpack": {
        "name": "XNNPACK",
        "exe": [],  # 通过头文件检测
        "header": "xnnpack.h",
        "lib_files": ["xnnpack.dll", "libxnnpack.so"],
        "env": ["XNNPACK_ROOT"],
        "vcpkg_packages": ["xnnpack"],
        "install_hint": "vcpkg install xnnpack"
    }
}

# ============================================================================
# 安装建议模板
# ============================================================================

INSTALL_SUGGESTIONS = {
    "cmake": {
        "windows": "Download from https://cmake.org/download/ and install to C:/Program Files/CMake",
        "linux": "Run: sudo apt install cmake"
    },
    "ninja": {
        "windows": "Run: pip install ninja",
        "linux": "Run: sudo apt install ninja-build"
    },
    "msvc": {
        "windows": "Install Visual Studio 2022 with C++ development tools"
    },
    "gcc": {
        "linux": "Run: sudo apt install g++-13"
    },
    "cuda": {
        "both": "Download from https://developer.nvidia.com/cuda-toolkit"
    },
    "cudnn": {
        "both": "Download cuDNN 9.17.0+ for CUDA 13.x from: https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/ Then extract and copy files to CUDA directory"
    },
    "python": {
        "both": "Download from https://python.org or run: conda create -n renaissance python=3.11"
    },
    "numpy": {
        "both": "Run: pip install numpy"
    },
    "onednn": {
        "both": "Run: vcpkg install dnnl"
    },
    "zlib": {
        "windows": "Run: vcpkg install zlib",
        "linux": "Run: sudo apt install zlib1g-dev"
    },
    "libcurl": {
        "windows": "Run: vcpkg install curl",
        "linux": "Run: sudo apt install libcurl4-openssl-dev"
    }
}

# ============================================================================
# 详细安装指南
# ============================================================================

DETAILED_INSTALL_GUIDES = {
    "cudnn": {
        "title": "cuDNN 9.17.0+ Installation Guide (Linux x86_64, CUDA 13.x)",
        "steps": [
            "1. Visit NVIDIA cuDNN redistribution page:",
            "   https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/",
            "2. Find cuDNN version 9.17.0 or higher for CUDA 13.x",
            "3. Download the file (usually .tar.xz format) named like:",
            "   cudnn-linux-x86_64-9.17.0_cuda13-archive.tar.xz",
            "4. Extract the archive:",
            "   tar -xf cudnn-linux-x86_64-9.17.0_cuda13-archive.tar.xz",
            "5. Copy files to your CUDA installation directory:",
            "   cd cudnn-linux-x86_64-9.17.0_cuda13-archive",
            "   sudo cp include/cudnn*.h /usr/local/cuda-13.0/include",
            "   sudo cp -P lib/libcudnn* /usr/local/cuda-13.0/lib64",
            "   sudo chmod a+r /usr/local/cuda-13.0/include/cudnn*.h /usr/local/cuda-13.0/lib64/libcudnn*",
            "6. Verify installation:",
            "   cat /usr/local/cuda-13.0/include/cudnn_version.h | grep CUDNN_MAJOR -A 2"
        ],
        "notes": [
            "Make sure to use cuDNN version 9.17.0+ for CUDA 13.x compatibility",
            "If CUDA is installed in different location, adjust paths accordingly",
            "The cuDNN library files should match your CUDA version exactly"
        ]
    }
}