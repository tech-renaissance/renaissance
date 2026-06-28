# -*- coding: utf-8 -*-
"""
@file dependency_data.py
@brief 技术觉醒框架依赖配置数据（V3.1.0：六大场景升级版）
@details 支持6大场景、16个依赖项的完整配置系统
@version 3.01.00
@date 2025-12-21
@author 技术觉醒团队
"""

# ============================================================================
# 场景定义（Phase 1：六大核心场景）
# ============================================================================

SCENE_DEPS = {
    "pc_cuda": {
        "name": "PC-CUDA",
        "description": "Windows + NVIDIA GPU + MSVC",
        "required": ["cmake", "ninja", "msvc", "cuda", "cudnn", "cudnn-frontend", "eigen", "xnnpack", "zlib", "libcurl", "libarchive", "libjpeg-turbo", "mimalloc", "stb", "simd", "python", "numpy"],
        "optional": [],
        "cmake_opts": [
            "-DTR_SCENE_PC_CUDA=ON",
            "-DTR_USE_CUDA=ON",
            "-DTR_USE_MUSA=OFF",
            "-DTR_USE_EIGEN=ON",
            "-DTR_USE_XNNPACK=ON",
            "-DTR_USE_STB=ON",
            "-DTR_USE_LIBJPEG=ON",
            "-DTR_USE_ZLIB=ON",
            "-DTR_USE_LIBCURL=ON",
            "-DTR_USE_LIBARCHIVE=ON",
            "-DTR_USE_MIMALLOC=ON",
            "-DTR_USE_SIMD=ON"
        ]
    },
    "gpu_cloud": {
        "name": "GPU_CLOUD",
        "description": "Linux + Multi-NVIDIA GPU + GCC",
        "required": ["cmake", "ninja", "gcc", "cuda", "cudnn", "nccl", "cudnn-frontend", "eigen", "xnnpack", "zlib", "libcurl", "libarchive", "libnuma", "libjpeg-turbo", "mimalloc", "stb", "simd", "python", "numpy"],
        "optional": [],
        "cmake_opts": [
            "-DTR_SCENE_GPU_CLOUD=ON",
            "-DTR_USE_CUDA=ON",
            "-DTR_USE_MUSA=OFF",
            "-DTR_USE_EIGEN=ON",
            "-DTR_USE_XNNPACK=ON",
            "-DTR_USE_NCCL=ON",
            "-DTR_USE_MULTI_GPU=ON",
            "-DTR_USE_STB=ON",
            "-DTR_USE_LIBJPEG=ON",
            "-DTR_USE_ZLIB=ON",
            "-DTR_USE_LIBCURL=ON",
            "-DTR_USE_LIBARCHIVE=ON",
            "-DTR_USE_LIBNUMA=ON",
            "-DTR_USE_MIMALLOC=ON",
            "-DTR_USE_SIMD=ON"
        ]
    },
    "pc_musa": {
        "name": "PC-MUSA",
        "description": "Linux + Moore Threads GPU + GCC",
        "required": ["cmake", "ninja", "gcc", "musa", "mudnn", "eigen", "xnnpack", "zlib", "libcurl", "libarchive", "libjpeg-turbo", "mimalloc", "stb", "simd", "python", "numpy"],
        "optional": [],
        "cmake_opts": [
            "-DTR_SCENE_PC_MUSA=ON",
            "-DTR_USE_CUDA=OFF",
            "-DTR_USE_MUSA=ON",
            "-DTR_USE_EIGEN=ON",
            "-DTR_USE_XNNPACK=ON",
            "-DTR_USE_STB=ON",
            "-DTR_USE_LIBJPEG=ON",
            "-DTR_USE_ZLIB=ON",
            "-DTR_USE_LIBCURL=ON",
            "-DTR_USE_LIBARCHIVE=ON",
            "-DTR_USE_MIMALLOC=ON",
            "-DTR_USE_SIMD=ON"
        ]
    },
    "cpu_cloud": {
        "name": "CPU_CLOUD",
        "description": "Windows/Linux x86 CPU Cloud Server + GCC/MSVC",
        "required": ["cmake", "ninja", "eigen", "xnnpack", "zlib", "libcurl", "libarchive", "libjpeg-turbo", "mimalloc", "stb", "simd", "python", "numpy"],
        "optional": ["gcc", "msvc"],  # 编译器为可选，根据平台选择
        "cmake_opts": [
            "-DTR_SCENE_CPU_CLOUD=ON",
            "-DTR_USE_CUDA=OFF",
            "-DTR_USE_MUSA=OFF",
            "-DTR_USE_EIGEN=ON",
            "-DTR_USE_XNNPACK=ON",
            "-DTR_USE_STB=ON",
            "-DTR_USE_LIBJPEG=ON",
            "-DTR_USE_ZLIB=ON",
            "-DTR_USE_LIBCURL=ON",
            "-DTR_USE_LIBARCHIVE=ON",
            "-DTR_USE_MIMALLOC=ON",
            "-DTR_USE_SIMD=ON"
        ]
    },
    "edge_arm": {
        "name": "EDGE_ARM",
        "description": "ARM Embedded Linux + GCC",
        "required": ["cmake", "ninja", "gcc", "xnnpack", "zlib", "libcurl", "libarchive", "libjpeg-turbo", "mimalloc", "stb", "simd", "python", "numpy"],
        "optional": [],
        "cmake_opts": [
            "-DTR_SCENE_EDGE_ARM=ON",
            "-DTR_USE_CUDA=OFF",
            "-DTR_USE_MUSA=OFF",
            "-DTR_USE_XNNPACK=ON",
            "-DTR_USE_STB=ON",
            "-DTR_USE_LIBJPEG=ON",
            "-DTR_USE_ZLIB=ON",
            "-DTR_USE_LIBCURL=ON",
            "-DTR_USE_LIBARCHIVE=ON",
            "-DTR_USE_MIMALLOC=ON",
            "-DTR_USE_SIMD=ON",
            "-DTR_IS_EDGE_DEVICE=ON"
        ]
    },
    "edge_riscv": {
        "name": "EDGE_RISCV",
        "description": "RISC-V Embedded Linux + GCC",
        "required": ["cmake", "ninja", "gcc", "xnnpack", "zlib", "libcurl", "libarchive", "libjpeg-turbo", "mimalloc", "stb", "python", "numpy"],
        "optional": [],
        "cmake_opts": [
            "-DTR_SCENE_EDGE_RISCV=ON",
            "-DTR_USE_CUDA=OFF",
            "-DTR_USE_MUSA=OFF",
            "-DTR_USE_XNNPACK=ON",
            "-DTR_USE_STB=ON",
            "-DTR_USE_LIBJPEG=ON",
            "-DTR_USE_ZLIB=ON",
            "-DTR_USE_LIBCURL=ON",
            "-DTR_USE_LIBARCHIVE=ON",
            "-DTR_USE_MIMALLOC=ON",
            "-DTR_USE_SIMD=OFF",
            "-DTR_IS_EDGE_DEVICE=ON"
        ]
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
        "min_version": "3.20.0",
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
        "min_version": "13.1",
        "install_hint": "https://developer.nvidia.com/cuda-toolkit"
    },

    "cudnn": {
        "name": "cuDNN",
        "exe": [],  # cuDNN主要通过头文件和库文件检测
        "header": ["cudnn.h", "cudnn_version.h", "cudnn_version_v9.h"],  # 支持多个版本头文件
        "lib_files": ["cudnn64.dll", "libcudnn.so"],
        "env": ["CUDNN_ROOT", "CUDNN_PATH"],
        "paths_win": [
            "C:/Program Files/NVIDIA/CUDNN",
            "C:/Program Files/NVIDIA/CUDNN/v9.*",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/include"
        ],
        "paths_linux": [
            "/usr/local/cuda",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/include",
            "/usr/include/cudnn",
            "/usr/include/x86_64-linux-gnu",  # 添加系统级cuDNN头文件路径
            "/usr/local/cudnn"
        ],
        "version_cmd": ["grep", "CUDNN_MAJOR", "-A", "2", "{include}/cudnn_version.h"],  # 默认使用cudnn_version.h
        "version_pattern": r"#define CUDNN_MAJOR\s+(\d+).*#define CUDNN_MINOR\s+(\d+).*#define CUDNN_PATCHLEVEL\s+(\d+)",
        "min_version": "9.17",
        "install_hint": "https://developer.nvidia.com/cudnn",
        "use_path_version": True  # 使用路径推导版本号
    },

    "musa": {
        "name": "MUSA Toolkit",
        "exe": ["mcc", "mcc.exe"],
        "env": ["MUSA_INSTALL_PATH"],
        "paths_linux": ["/usr/local/musa", "/usr/local/musa-3.*"],
        "bin_subdir": "bin",
        "version_cmd": ["mcc", "--version"],
        "version_pattern": r"mcc version (\d+\.\d+)",
        "min_version": "3.0",
        "install_hint": "https://www.mthreads.com/download"
    },

    "mudnn": {
        "name": "MUDNN",
        "exe": [],  # MUDNN主要通过头文件和库文件检测
        "header": "mudnn_version.h",
        "lib_files": ["libmudnn.so"],
        "env": ["DNN_PATH"],
        "paths_linux": ["/usr/local/musa", "/usr/local/musa/lib", "/usr"],
        "version_cmd": ["grep", "MUDNN_VERSION_MAJOR", "-A", "2", "{include}/mudnn_version.h"],
        "version_pattern": r"#define MUDNN_VERSION_MAJOR\s+(\d+).*#define MUDNN_VERSION_MINOR\s+(\d+).*#define MUDNN_VERSION_PATCH\s+(\d+)",
        "min_version": "2.8",
        "install_hint": "https://www.mthreads.com/download"
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
        "paths_linux": ["/opt/venv/bin", "~/venv/py314/bin", "/home/tech-renaissance/venv/py314/bin", "/usr/bin", "/usr/local/bin", "/opt/python/bin"],
        "version_cmd": ["python", "--version"],
        "version_pattern": r"Python (\d+\.\d+\.\d+)",
        "min_version": "3.12.0",
        "user_selection": True,
        "install_hint": "https://python.org | conda create -n renaissance python=3.12+"
    },

    "numpy": {
        "name": "NumPy",
        "exe": [],  # 通过Python命令检查
        "check_cmd": ["python", "-c", "import numpy; print(numpy.__version__); print(numpy.__file__)"],
        "min_version": "1.24.0",
        "install_hint": "pip install numpy"
    },

    # 基础库
    "eigen": {
        "name": "Eigen",
        "exe": [],  # 通过头文件检测
        "header": ["Eigen/Core", "eigen3/Eigen/Core"],
        "env": ["EIGEN3_ROOT", "EIGEN_ROOT"],
        "vcpkg_packages": ["eigen3"],
        "min_version": "3.4",
        "version_pattern": r"EIGEN_WORLD_VERSION\s+(\d+).*EIGEN_MAJOR_VERSION\s+(\d+).*EIGEN_MINOR_VERSION\s+(\d+)",
        "version_cmd": ["grep", "-E", "EIGEN_WORLD_VERSION|EIGEN_MAJOR_VERSION|EIGEN_MINOR_VERSION", "{include}/Eigen/src/Core/util/Macros.h"],
        "install_hint": "vcpkg install eigen3"
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
        "version_cmd": ["curl", "--version"],
        "version_pattern": r"curl (\d+\.\d+\.\d+)",
        "install_hint": "vcpkg install curl (Windows) | sudo apt install libcurl4-openssl-dev (Linux)"
    },

    "libarchive": {
        "name": "libarchive",
        "exe": [],  # 通过头文件检测
        "header": "archive.h",
        "lib_files": ["archive.lib", "libarchive.a", "libarchive.so"],
        "env": ["ARCHIVE_ROOT"],
        "paths_linux": ["/usr", "/usr/local"],
        "vcpkg_packages": ["libarchive"],
        "version_cmd": ["bsdtar", "--version"],
        "version_pattern": r"bsdtar (\d+\.\d+\.\d+)",
        "install_hint": "vcpkg install libarchive (Windows) | sudo apt install libarchive-dev (Linux)"
    },

    "libnuma": {
        "name": "libnuma",
        "exe": [],  # 不通过可执行文件检测，直接检查头文件
        "header": "numa.h",
        "lib_files": ["libnuma.so", "libnuma.a"],
        "env": [],
        "paths_linux": ["/usr", "/usr/local"],
        "version_cmd": ["numactl", "--version"],
        "version_pattern": r"numactl version (\d+\.\d+\.\d+)",
        "install_hint": "sudo apt install libnuma-dev"
    },

    "xnnpack": {
        "name": "XNNPACK",
        "exe": [],  # 通过头文件检测
        "header": "xnnpack.h",
        "lib_files": ["xnnpack.dll", "libxnnpack.so"],
        "env": ["XNNPACK_ROOT"],
        "paths_win": ["C:\\XNNPACK", "C:/Program Files/XNNPACK*"],  # Windows手动安装路径（静态库）
        "paths_linux": ["/opt/xnnpack"],  # Linux自编译路径（优先于vcpkg）
        "vcpkg_packages": ["xnnpack"],
        "install_hint": "vcpkg install xnnpack (Windows) | Build from https://github.com/google/XNNPACK (Linux)"
    },

    # 新增依赖项
    "mimalloc": {
        "name": "mimalloc",
        "exe": [],  # 通过头文件检测
        "header": "mimalloc.h",
        "lib_files": ["mimalloc.dll", "libmimalloc.so"],
        "env": ["MIMALLOC_ROOT"],
        "vcpkg_packages": ["mimalloc"],
        "install_hint": "vcpkg install mimalloc"
    },

    "simd": {
        "name": "Simd",
        "exe": [],  # 通过头文件检测
        "header": ["Simd/SimdLib.h", "Simd/SimdVersion.h"],  # 检测SimdLib.h或SimdVersion.h
        "env": ["SIMD_ROOT"],
        "paths_linux": ["/opt/simd/src"],  # Linux自编译路径，头文件在src目录下
        "vcpkg_packages": ["simd"],
        "version_cmd": ["grep", "SIMD_VERSION", "{path}/Simd/SimdVersion.h"],
        "version_pattern": r"#define\s+SIMD_VERSION\s+\"([^\"]+)\"",  # 版本字符串格式: "6.2.159"
        "min_version": "6.0",
        "install_hint": "vcpkg install simd (Windows) | Build from https://github.com/ermig1979/Simd (Linux)"
    },

    "stb": {
        "name": "STB",
        "exe": [],  # 通过头文件检测 (单头文件库)
        "header": "stb_image.h",  # 主要检测stb_image.h
        "env": ["STB_ROOT"],
        "vcpkg_packages": ["stb"],
        "version_pattern": r"(\d{4}-\d{2}-\d{2})",  # 日期格式: 2024-07-29
        "install_hint": "vcpkg install stb"
    },

    "libjpeg-turbo": {
        "name": "libjpeg-turbo",
        "exe": ["jpegtran", "cjpeg", "djpeg"],  # 通过可执行文件检测
        "header": "jpeglib.h",
        "lib_files": ["libjpeg-turbo.dll", "libjpeg.so", "libjpeg-turbo.so"],
        "env": ["JPEG_ROOT"],
        "vcpkg_packages": ["libjpeg-turbo"],
        "paths_win": [
            "C:/Program Files/libjpeg-turbo",
            "C:/Program Files/libjpeg-turbo64"
        ],
        "paths_linux": ["/usr", "/usr/local", "/opt/libjpeg-turbo"],
        "version_cmd": ["jpegtran", "-version"],
        "version_pattern": r"version (\d+\.\d+\.\d+)",
        "install_hint": "vcpkg install libjpeg-turbo"
    },

    "nccl": {
        "name": "NCCL",
        "exe": [],  # 通过头文件检测
        "header": "nccl.h",
        "lib_files": ["nccl.dll", "libnccl.so"],
        "env": ["NCCL_ROOT", "NCCL_HOME"],
        "paths_linux": ["/usr/local/nccl", "/usr", "/opt/nccl", "/usr/local/cuda"],
        "version_cmd": ["grep", "-E", "NCCL_MAJOR|NCCL_MINOR|NCCL_PATCH", "{include}/nccl.h"],
        "version_pattern": r"#define NCCL_MAJOR\s+(\d+).*#define NCCL_MINOR\s+(\d+).*#define NCCL_PATCH\s+(\d+)",
        "min_version": "2.29",
        "install_hint": "Download from https://developer.nvidia.com/nccl"
    },

    "cudnn-frontend": {
        "name": "cuDNN Frontend",
        "exe": [],  # 通过头文件检测
        "header": ["cudnn_frontend.h", "cudnn_frontend_version.h"],
        "env": ["CUDNN_FRONTEND_ROOT"],
        "paths_win": ["C:/Program Files/cudnn-frontend*"],
        "paths_linux": ["/opt/cudnn-frontend"],
        "version_cmd": ["grep", "-E", "CUDNN_FRONTEND_MAJOR_VERSION|CUDNN_FRONTEND_MINOR_VERSION|CUDNN_FRONTEND_PATCH_VERSION", "{include}/cudnn_frontend_version.h"],
        "version_pattern": r"#define CUDNN_FRONTEND_MAJOR_VERSION\s+(\d+).*#define CUDNN_FRONTEND_MINOR_VERSION\s+(\d+).*#define CUDNN_FRONTEND_PATCH_VERSION\s+(\d+)",
        "min_version": "1.0",
        "install_hint": "Download from https://github.com/NVIDIA/cudnn-frontend"
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
    "eigen": {
        "both": "Run: vcpkg install eigen3"
    },
    "zlib": {
        "windows": "Run: vcpkg install zlib",
        "linux": "Run: sudo apt install zlib1g-dev"
    },
    "libcurl": {
        "windows": "Run: vcpkg install curl",
        "linux": "Run: sudo apt install libcurl4-openssl-dev"
    },
    "libarchive": {
        "windows": "Run: vcpkg install libarchive",
        "linux": "Run: sudo apt install libarchive-dev"
    },
    "libnuma": {
        "linux": "Run: sudo apt install libnuma-dev"
    },
    "musa": {
        "both": "Download from https://www.mthreads.com/download"
    },
    "mudnn": {
        "both": "Download from https://www.mthreads.com/download"
    },
    "mimalloc": {
        "both": "Run: vcpkg install mimalloc"
    },
    "simd": {
        "both": "Run: vcpkg install simd"
    },

    "stb": {
        "both": "Run: vcpkg install stb"
    },
    "libjpeg-turbo": {
        "windows": "Run: vcpkg install libjpeg-turbo",
        "linux": "Run: sudo apt install libjpeg-turbo8-dev OR vcpkg install libjpeg-turbo"
    },
    "nccl": {
        "linux": "Download from https://developer.nvidia.com/nccl"
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
            "   sudo cp include/cudnn*.h /usr/local/cuda-13.1/include",
            "   sudo cp -P lib/libcudnn* /usr/local/cuda-13.1/lib64",
            "   sudo chmod a+r /usr/local/cuda-13.1/include/cudnn*.h /usr/local/cuda-13.1/lib64/libcudnn*",
            "6. Verify installation:",
            "   cat /usr/local/cuda-13.1/include/cudnn_version.h | grep CUDNN_MAJOR -A 2"
        ],
        "notes": [
            "Make sure to use cuDNN version 9.17.0+ for CUDA 13.x compatibility",
            "If CUDA is installed in different location, adjust paths accordingly",
            "The cuDNN library files should match your CUDA version exactly"
        ]
    }
}
