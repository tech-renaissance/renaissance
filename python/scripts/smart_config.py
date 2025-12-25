#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file smart_config.py
@brief 技术觉醒深度学习框架 V3.1.0 智能配置模块
@details 系统检测、6大场景判断、16依赖项搜索、完整编译宏生成的核心逻辑
@version 3.01.00
@date 2025-12-21
@author 技术觉醒团队
"""

import os
import sys
import json
import platform
import subprocess
import shutil
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 导入依赖配置
from dependency_data import SCENE_DEPS, DEP_CONFIG, INSTALL_SUGGESTIONS

# 全局变量：存储已经找到的Python依赖路径
global_deps = {}

# 全局变量：缓存vcpkg包列表，避免重复调用
vcpkg_cache = {}
vcpkg_full_output_cache = None  # 缓存完整的vcpkg list输出字符串

# 导入配置变量（来自根目录的configure.py）
try:
    from configure import VCPKG_ROOT, BUILD_DIR, PARALLEL_JOBS, CMAKE_GENERATOR, CMAKE_BUILD_TYPE, VERBOSE_OUTPUT
except ImportError:
    # 默认配置 - 统一为build目录（与Linux保持一致）
    VCPKG_ROOT = os.environ.get("VCPKG_ROOT", "T:/Softwares/vcpkg")
    BUILD_DIR = "build"
    PARALLEL_JOBS = 30
    CMAKE_GENERATOR = "Ninja"
    CMAKE_BUILD_TYPE = "Release"
    VERBOSE_OUTPUT = False

# ============================================================================
# 颜色系统和编码设置 (参考 configure_old_part1.py)
# ============================================================================

# Set output encoding to UTF-8 for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    OK = '\033[92m'  # Same as GREEN for success messages
    WARN = '\033[93m'
    INFO = '\033[96m'  # Using CYAN color for INFO
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message: str, color: str):
    """打印带颜色的消息"""
    print(f"{color}{message}{Colors.ENDC}")

# ============================================================================
# 工具函数
# ============================================================================

def print_header(title: str):
    """打印标题"""
    # 读取版本号
    try:
        with open("version.txt", "r") as f:
            version = f.readline().strip()
    except:
        version = "3.0.4"

    print_colored(f"\n{'='*46}", Colors.BOLD)
    print_colored(f"  {title}", Colors.BOLD)
    print_colored(f"  renAIssance Deep Learning Framework v{version}", Colors.BOLD)
    print_colored(f"{'='*46}", Colors.BOLD)

def print_step(step: int, total: int, msg: str):
    """打印步骤"""
    print_colored(f"\n[Step {step}/{total}] {msg}", Colors.BLUE)

def print_ok(msg: str):
    """打印成功信息"""
    print_colored(f"[OK] {msg}", Colors.OK)

def print_fail(msg: str):
    """打印失败信息"""
    print_colored(f"[FAIL] {msg}", Colors.FAIL)

def format_dependency_output(name: str, version: str = None, path: str = None) -> str:
    """格式化依赖项输出

    Args:
        name: 依赖项名称
        version: 版本号 (可选)
        path: 路径 (可选)

    Returns:
        格式化后的字符串
    """
    # 依赖项名占14字符，左对齐
    name_part = f"{name:<14}"

    # 版本号部分 - 完整版本信息在括号内，括号后补空格
    if version:
        version_part = f"[v{version}]"  # 完整版本信息在括号内
    else:
        version_part = "               "

    # 计算需要的空格数来保持总宽度一致
    if version:
        needed_spaces = 15 - len(version_part)
        version_part += " " * needed_spaces

    # 路径部分 - 统一使用斜线分隔
    if path and path != "system PATH" and path != "system":
        # 统一转换为斜线分隔
        path_part = path.replace("\\", "/")
        result = f"{name_part}{version_part} - {path_part}"
    else:
        result = f"{name_part}{version_part}"

    return result

def print_info(msg: str):
    """打印信息"""
    print_colored(f"[INFO] {msg}", Colors.INFO)

def print_warn(msg: str):
    """打印警告"""
    print_colored(f"[WARN] {msg}", Colors.WARN)

def run_cmd(cmd: List[str], capture: bool = True) -> Tuple[bool, str]:
    """运行命令并返回结果"""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout + result.stderr
        else:
            result = subprocess.run(cmd, timeout=30)
            return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)

def get_user_choice(options: List[Dict], prompt: str) -> Dict:
    """获取用户选择"""
    print_info(f"{prompt}")
    print()

    # 列出选项
    for i, option in enumerate(options, 1):
        version = option.get("version", "")
        path = option.get("path", "")
        # 使用统一的格式化函数，但不要加上[OK]前缀
        formatted = format_dependency_output(option['name'], version, path)
        print(f"  {i}. {formatted}")

    # 添加"其他"选项
    print(f"  {len(options) + 1}. Other (enter custom path)")
    print()

    while True:
        try:
            choice = input(f"  Please select [1-{len(options) + 1}]: ").strip()
            if not choice:  # 空输入默认为1
                choice_num = 1
            else:
                choice_num = int(choice)

            if 1 <= choice_num <= len(options):
                selected = options[choice_num - 1]
                version = selected.get('version', 'unknown')
                path = selected.get('path')
                formatted_output = format_dependency_output(selected['name'], version, path)
                print_info(f"Selected: {formatted_output}")
                return selected
            elif choice_num == len(options) + 1:
                custom_path = input("  Enter custom path: ").strip()
                if custom_path and os.path.exists(custom_path):
                    return {
                        "name": options[0]["name"],
                        "path": custom_path,
                        "version": None,
                        "from_vcpkg": False
                    }
                else:
                    print_fail("Path does not exist!")
            else:
                print_fail(f"Invalid choice. Please enter 1-{len(options) + 1}")
        except EOFError:
            # 非交互模式，Windows默认选第一个，Linux默认选第二个
            import platform
            is_windows = platform.system() == "Windows"

            if is_windows or len(options) < 2:
                print_info(f"Non-interactive mode: selecting first {options[0]['name']} installation")
                selected = options[0]
            else:
                # Linux：默认选第二个（通常是有完整库的虚拟环境）
                print_info(f"Non-interactive mode: selecting second {options[1]['name']} installation")
                selected = options[1]
            selected["found"] = True
            return selected
        except ValueError:
            print_fail("Please enter a valid number")
        except KeyboardInterrupt:
            print_fail("\nConfiguration cancelled by user")
            sys.exit(1)

def find_exe_in_path(exe_names: List[str]) -> Optional[str]:
    """在系统PATH中查找可执行文件"""
    for exe in exe_names:
        path = shutil.which(exe)
        if path:
            return path
    return None

def find_file_in_dirs(filename: str, dirs: List[str], subdirs: List[str] = None) -> Optional[str]:
    """在目录列表中查找文件"""
    subdirs = subdirs or ["", "include", "bin", "lib"]
    for d in dirs:
        if not os.path.exists(d):
            continue
        for sub in subdirs:
            path = os.path.join(d, sub, filename) if sub else os.path.join(d, filename)
            if os.path.exists(path):
                return d
        # 支持通配符搜索
        if '*' in d:
            matches = glob.glob(d)
            for match in matches:
                for sub in subdirs:
                    path = os.path.join(match, sub, filename) if sub else os.path.join(match, filename)
                    if os.path.exists(path):
                        return match
    return None

def extract_version(output: str, pattern: str) -> Optional[str]:
    """从输出中提取版本号"""
    match = re.search(pattern, output, re.DOTALL)
    if match:
        # 如果有多个捕获组，组合成版本号
        if len(match.groups()) > 1:
            return ".".join(match.groups())
        else:
            return match.group(1)
    return None

def compare_version(v1: str, v2: str) -> int:
    """比较版本号，返回 -1, 0, 1"""
    def parse(v):
        return [int(x) for x in v.split('.')[:3]]
    p1, p2 = parse(v1), parse(v2)
    for a, b in zip(p1, p2):
        if a < b: return -1
        if a > b: return 1
    return 0

def extract_cudnn_version_from_path(path: str) -> str:
    """从cuDNN路径推导版本号 - Windows看路径，Linux看头文件"""
    # 标准化路径
    path = path.replace("\\", "/")
    import re
    import platform

    if platform.system() == "Windows":
        # Windows下：从路径中提取版本号
        # 模式1: CUDNN/v9.17
        pattern1 = r"/v(\d+\.\d+)"
        match = re.search(pattern1, path)
        if match:
            return match.group(1)

        # 模式2: CUDNN/v9.17.0
        pattern2 = r"/v(\d+\.\d+\.\d+)"
        match = re.search(pattern2, path)
        if match:
            return match.group(1)

        # 模式3: 检查常见的cuDNN版本目录名
        common_versions = ["9.17", "9.16", "9.15", "9.14", "9.13", "9.12", "9.11", "9.10"]
        for version in common_versions:
            if version in path:
                return version

    else:
        # Linux下：从CUDA路径的include/cudnn_version.h读取版本号
        # 假设cuDNN安装在CUDA目录中，先找到CUDA目录
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-13.0",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8"
        ]

        # 如果当前路径就是CUDA路径，直接使用
        if "cuda" in path.lower():
            cuda_paths.insert(0, path)

        for cuda_path in cuda_paths:
            cudnn_version_file = os.path.join(cuda_path, "include", "cudnn_version.h")
            if os.path.exists(cudnn_version_file):
                try:
                    # 使用grep命令读取版本信息
                    success, output = run_cmd(["cat", cudnn_version_file])
                    if success:
                        lines = output.split('\n')
                        major = minor = patch = None

                        for line in lines:
                            if '#define CUDNN_MAJOR' in line:
                                major_match = re.search(r'#define\s+CUDNN_MAJOR\s+(\d+)', line)
                                if major_match:
                                    major = major_match.group(1)
                            elif '#define CUDNN_MINOR' in line:
                                minor_match = re.search(r'#define\s+CUDNN_MINOR\s+(\d+)', line)
                                if minor_match:
                                    minor = minor_match.group(1)
                            elif '#define CUDNN_PATCHLEVEL' in line:
                                patch_match = re.search(r'#define\s+CUDNN_PATCHLEVEL\s+(\d+)', line)
                                if patch_match:
                                    patch = patch_match.group(1)

                        if major and minor:
                            return f"{major}.{minor}.{patch if patch else '0'}"

                except Exception:
                    pass

        # 备用方案：从头文件解析
        include_paths = [
            path + "/include",
            path,
            path + "/include/cudnn",
            path + "/cudnn"
        ]

        for include_path in include_paths:
            version_file = os.path.join(include_path, "cudnn_version.h")
            if os.path.exists(version_file):
                try:
                    with open(version_file, 'r') as f:
                        content = f.read()

                    # 解析版本号
                    major_match = re.search(r'#define\s+CUDNN_MAJOR\s+(\d+)', content)
                    minor_match = re.search(r'#define\s+CUDNN_MINOR\s+(\d+)', content)
                    patch_match = re.search(r'#define\s+CUDNN_PATCHLEVEL\s+(\d+)', content)

                    if major_match and minor_match:
                        major = major_match.group(1)
                        minor = minor_match.group(1)
                        patch = patch_match.group(1) if patch_match else "0"
                        return f"{major}.{minor}.{patch}"

                except Exception:
                    pass

    return "unknown"

def extract_date_version_from_header(path: str, header_names: List[str]) -> str:
    """从头文件中提取日期格式的版本号"""
    import re

    # 构建可能的头文件路径
    include_paths = [
        path + "/include",
        path,
        path + "/include/xnnpack",
        path + "/include/stb"
    ]

    for include_path in include_paths:
        for header_name in header_names:
            header_path = os.path.join(include_path, header_name)
            if os.path.exists(header_path):
                try:
                    with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # 查找日期格式的版本号 (YYYY-MM-DD)
                    date_patterns = [
                        r'(\d{4}-\d{2}-\d{2})',  # 2024-08-20
                        r'(\d{1,2}/\d{1,2}/\d{4})',  # 8/20/2024
                        r'(20\d{2}-[A-Za-z]{3}-\d{1,2})',  # 2024-Aug-20
                    ]

                    for pattern in date_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            # 验证这是一个合理的日期版本号
                            if re.match(r'20\d{2}-\d{2}-\d{2}', match):
                                return match
                            elif re.match(r'\d{1,2}/\d{1,2}/20\d{2}', match):
                                # 转换 M/D/YYYY 为 YYYY-MM-DD
                                parts = match.split('/')
                                if len(parts) == 3:
                                    year = parts[2]
                                    month = parts[0].zfill(2)
                                    day = parts[1].zfill(2)
                                    return f"{year}-{month}-{day}"

                except Exception:
                    pass

    return "unknown"

def extract_version_from_path(path: str, dep_name: str) -> str:
    """通过执行--version命令从路径获取准确版本信息"""
    if not path:
        return "unknown"

    # 获取对应的依赖配置
    config = DEP_CONFIG.get(dep_name.lower())
    if not config:
        return "unknown"

    # 特殊处理：cuDNN使用路径推导版本号
    if dep_name.lower() == "cudnn" and config.get("use_path_version"):
        return extract_cudnn_version_from_path(path)

    # 特殊处理：XNNPACK和STB尝试从文件内容解析日期版本
    if dep_name.lower() in ["xnnpack", "stb"] and "header" in config:
        version = extract_date_version_from_header(path, config["header"])
        if version != "unknown":
            return version

    # 构建版本命令
    if "version_cmd" in config:
        cmd = config["version_cmd"]
    elif "exe" in config:
        # 默认使用 --version 参数
        exe_name = config["exe"][0]
        if exe_name.endswith('.exe'):
            exe_name = exe_name[:-4]
        cmd = [exe_name, "--version"]
    else:
        return "unknown"

    # 构建完整的可执行文件路径
    exe_path = None
    if "exe" in config:
        for exe in config["exe"]:
            full_exe_path = os.path.join(path, exe)
            if os.path.exists(full_exe_path):
                exe_path = full_exe_path
                break
            # 也尝试在bin子目录中查找
            bin_exe_path = os.path.join(path, "bin", exe)
            if os.path.exists(bin_exe_path):
                exe_path = bin_exe_path
                break

    # 如果在指定路径中没找到可执行文件，尝试在PATH中查找
    if not exe_path and "exe" in config:
        exe_path = find_exe_in_path(config["exe"])

    if not exe_path:
        return "unknown"

    # 执行版本命令
    if cmd[0] in config.get("exe", []):
        # 如果命令使用的是可执行文件名，替换为完整路径
        cmd[0] = exe_path

    success, output = run_cmd(cmd, capture=True)
    if not success:
        return "unknown"

    # 使用配置中的版本模式提取版本号
    version_pattern = config.get("version_pattern")
    if version_pattern:
        match = re.search(version_pattern, output)
        if match:
            return match.group(1)

    # 如果没有模式匹配，尝试常见的版本号模式
    common_patterns = [
        r'(\d+\.\d+\.\d+)',      # x.y.z
        r'(\d+\.\d+)',          # x.y
        r'version\s+(\d+\.\d+\.\d+)',  # version x.y.z
        r'version\s+(\d+\.\d+)',       # version x.y
        r'v(\d+\.\d+\.\d+)',     # vx.y.z
        r'v(\d+\.\d+)',          # vx.y
    ]

    for pattern in common_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1)

    # 如果都失败了，返回输出的前几行作为参考
    lines = output.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        # 如果第一行包含数字，尝试提取
        if re.search(r'\d', first_line):
            return first_line[:100]  # 限制长度

    return "unknown"

# ============================================================================
# 系统检测
# ============================================================================

def detect_system() -> Dict:
    """检测系统信息"""
    info = {
        "os": platform.system(),
        "arch": platform.machine().lower(),
        "is_windows": platform.system() == "Windows",
        "is_linux": platform.system() == "Linux"
    }

    # 标准化架构名称
    if info["arch"] in ["x86_64", "amd64"]:
        info["arch"] = "x86_64"
    elif info["arch"] in ["aarch64", "arm64"]:
        info["arch"] = "arm64"
    elif info["arch"].startswith("riscv"):
        info["arch"] = "riscv"

    return info

def detect_gpu() -> Dict:
    """检测GPU信息 - 增强GPU数量和版本检查"""
    gpu_info = {"type": None, "name": None, "count": 0, "detected": False, "driver_version": None}

    # 检测NVIDIA GPU
    success, output = run_cmd(["nvidia-smi", "--query-gpu=name,count,driver_version", "--format=csv,noheader,nounits"])
    if success:
        lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
        if lines:
            gpu_info["type"] = "nvidia"
            gpu_info["count"] = len(lines)

            # 解析第一行GPU信息获取名称和驱动版本
            first_line = lines[0]
            parts = [p.strip() for p in first_line.split(',')]
            if len(parts) >= 3:
                gpu_info["name"] = parts[0]
                gpu_info["driver_version"] = parts[2]
                # 如果有多张相同型号的GPU，显示为"GPU名称 x N"
                if gpu_info["count"] > 1:
                    gpu_info["name"] = f"{parts[0]} x {gpu_info['count']}"
                else:
                    gpu_info["name"] = parts[0]
            else:
                # 兼容旧格式
                gpu_info["name"] = parts[0].split(',')[0].strip()

            gpu_info["detected"] = True
            return gpu_info

    # 检测摩尔线程GPU
    success, output = run_cmd(["mthreads-gmi"])
    if success and "MTT" in output:
        gpu_info["type"] = "mthreads"
        gpu_info["count"] = 1

        # 解析摩尔线程GPU信息
        lines = output.split('\n')
        for line in lines:
            if "MTT" in line and "ID" in line:
                # 提取GPU型号
                parts = line.split()
                for i, part in enumerate(parts):
                    if "MTT" in part:
                        gpu_info["name"] = part
                        break
                break

        # 提取驱动版本
        for line in lines:
            if "Driver Version" in line:
                version_parts = line.split(':')
                if len(version_parts) >= 2:
                    gpu_info["driver_version"] = version_parts[1].strip()
                break

        if not gpu_info["name"]:
            gpu_info["name"] = "Moore Threads GPU"

        gpu_info["detected"] = True
        return gpu_info

    return gpu_info

# ============================================================================
# 场景判断（严格遵循文档【二十三】）
# ============================================================================

def determine_scene(sys_info: Dict, gpu_info: Dict) -> str:
    """根据系统和GPU信息判断使用场景 - 严格按照文档6步流程"""

    # Step 1: CPU架构判断
    if sys_info["arch"] in ["arm64", "aarch64"] or sys_info["arch"].startswith("arm"):
        if not sys_info["is_linux"]:
            print_fail("ARM devices must run Linux!")
            sys.exit(1)
        return "edge_arm"

    if sys_info["arch"].startswith("riscv") or sys_info["arch"] == "riscv64":
        if not sys_info["is_linux"]:
            print_fail("RISC-V devices must run Linux!")
            sys.exit(1)
        return "edge_riscv"

    # 非x86/ARM/RISC-V架构报错
    if sys_info["arch"] not in ["x86_64", "i686", "i386"]:
        print_fail(f"Unsupported architecture: {sys_info['arch']}")
        print_info("Supported architectures: x86_64, i686, i386, arm64, aarch64, riscv64")
        sys.exit(1)

    # Step 2: 询问用户是否使用GPU（默认Y）
    try:
        has_gpu = input("\nDo you want to use GPU acceleration? ([Y]/N):")
        if not has_gpu:  # 空输入默认为Y
            has_gpu = "Y"
        else:
            has_gpu = has_gpu.strip().upper()
    except EOFError:
        print_info("Non-interactive mode: continuing with GPU mode")
        has_gpu = "Y"

    if has_gpu != "Y":
        # 用户选择不使用GPU
        return "cpu_cloud"

    # 用户选择使用GPU，检查GPU检测情况
    if gpu_info["detected"]:
        # GPU已检测到，根据类型判断场景
        if gpu_info["type"] == "nvidia":
            # NVIDIA GPU
            if sys_info["is_windows"]:
                return "pc_cuda"
            else:
                # Linux下根据GPU数量判断
                if gpu_info["count"] > 1:
                    return "gpu_cloud"
                else:
                    return "pc_cuda"
        elif gpu_info["type"] == "mthreads":
            # 摩尔线程GPU只支持Linux
            if not sys_info["is_linux"]:
                print_fail("Moore Threads GPU only supports Linux!")
                sys.exit(1)
            return "pc_musa"
        else:
            print_fail(f"Unknown GPU type detected: {gpu_info['type']}")
            sys.exit(1)
    else:
        # 未检测到GPU，询问用户GPU类型并提供驱动安装建议
        print_fail("No GPU found. You may need to properly install a GPU driver.")

        print("\nWhat type of GPU do you have?")
        print("  1. NVIDIA GPU (GeForce, RTX, Tesla, A10, etc.)")
        print("  2. Moore Threads GPU (MTT S80, etc.)")

        try:
            print("Please select [1-2]:")
            choice = input().strip()
        except EOFError:
            choice = "1"  # 默认NVIDIA

        if choice == "2":
            # 摩尔线程GPU
            print_info("For Moore Threads GPU, please install drivers from:")
            print_info("  https://www.mthreads.com/download")
            print_fail("Please install drivers and run configure.py again")
            sys.exit(1)
        else:
            # NVIDIA GPU
            print_info("For NVIDIA GPU, please install drivers from:")
            print_info("  https://developer.nvidia.com/cuda-downloads")
            print_fail("Please install drivers and run configure.py again")
            sys.exit(1)

# ============================================================================
# 依赖搜索
# ============================================================================

def find_all_dependency_versions(name: str, sys_info: Dict) -> List[Dict]:
    """找到所有符合版本要求的依赖版本"""
    config = DEP_CONFIG.get(name)
    if not config:
        return []

    is_win = sys_info["is_windows"]
    versions = []
    min_version = config.get("min_version")

    # 特殊处理：Python包检测 - 返回一个版本
    if "check_cmd" in config:
        # 对于Python本身，优先使用当前运行的Python解释器
        if name == "python":
            current_python = sys.executable
            if current_python:
                success, output = run_cmd([current_python, "--version"])
                if success:
                    version = extract_version(output, config["version_pattern"])
                    if version and (not min_version or compare_version(version, min_version) >= 0):
                        versions.append({
                            "name": config["name"],
                            "path": os.path.dirname(current_python),
                            "version": version,
                            "from_vcpkg": False,
                            "exe_path": current_python
                        })

        # 对于NumPy等Python包，使用默认python命令
        success, output = run_cmd(config["check_cmd"])
        if success:
            version = output.strip()
            if not min_version or compare_version(version, min_version) >= 0:
                # 获取Python可执行文件路径
                python_exe = find_exe_in_path(config["exe"])
                python_path = os.path.dirname(python_exe) if python_exe else "system PATH"

                # 避免重复添加相同的Python路径
                is_duplicate = False
                for existing in versions:
                    if existing["path"] == python_path and existing["version"] == version:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    versions.append({
                        "name": config["name"],
                        "path": python_path,
                        "version": version,
                        "from_vcpkg": False,
                        "exe_path": python_exe
                    })
        return versions

    # 对于Python，总是优先使用当前运行的Python解释器
    if name == "python":
        current_python = sys.executable
        if current_python:
            # 检查当前Python是否满足版本要求
            success, output = run_cmd([current_python, "--version"])
            if success:
                version = extract_version(output, config["version_pattern"])
                if version and (not min_version or compare_version(version, min_version) >= 0):
                    # 计算正确的安装路径
                    install_path = os.path.dirname(current_python)

                    # 如果在 Windows 的 Python 目录中
                    if is_win and not install_path.endswith(('Scripts', 'bin')):
                        # 直接使用包含 python.exe 的目录
                        pass
                    elif is_win and install_path.endswith('Scripts'):
                        # 从 Scripts 目录回到 Python 安装根目录
                        install_path = os.path.dirname(install_path)

                    versions.append({
                        "name": config["name"],
                        "path": install_path,
                        "version": version,
                        "from_vcpkg": False,
                        "exe_path": current_python
                    })

    # 搜索所有可能的版本
    paths = config.get("paths_win" if is_win else "paths_linux", [])

    for path_pattern in paths:
        # 展开 ~ 符号（用户主目录）
        if path_pattern.startswith("~"):
            expanded_path = os.path.expanduser(path_pattern)
        else:
            expanded_path = path_pattern

        # 展开通配符
        if "*" in expanded_path:
            import glob
            matches = glob.glob(expanded_path)
        else:
            matches = [expanded_path] if os.path.exists(expanded_path) else []

        for path in matches:
            # 检查版本
            if "exe" in config:
                exe_path = None
                for exe in config["exe"]:
                    # 在特定路径中查找（优先）
                    full_exe_path = os.path.join(path, exe)
                    if os.path.exists(full_exe_path):
                        exe_path = full_exe_path
                        break

                    # Windows: 直接在根目录查找（Windows Python通常直接安装路径）
                    if is_win and not exe.endswith('.exe'):
                        full_exe_exe = os.path.join(path, exe + '.exe')
                        if os.path.exists(full_exe_exe):
                            exe_path = full_exe_exe
                            break
                    # Linux/macOS: 尝试在bin子目录中查找
                    elif not is_win:
                        bin_exe_path = os.path.join(path, "bin", exe)
                        if os.path.exists(bin_exe_path):
                            exe_path = bin_exe_path
                            break

                if exe_path:
                    # 使用改进的版本提取方法
                    version = extract_version_from_path(os.path.dirname(exe_path), name)
                    if version != "unknown" and (not min_version or compare_version(version, min_version) >= 0):
                        # 避免重复添加当前Python解释器
                        is_duplicate = False
                        if name == "python":
                            for existing in versions:
                                # 检查可执行文件路径是否相同
                                if existing.get("exe_path") == exe_path:
                                    is_duplicate = True
                                    break

                                # 跨平台重复检测：检查是否为同一Python安装目录
                                existing_path = existing.get("path", "")
                                current_path = os.path.dirname(exe_path)

                                # 标准化路径格式进行比较
                                existing_normalized = os.path.normpath(existing_path).lower()
                                current_normalized = os.path.normpath(current_path).lower()

                                if existing_normalized == current_normalized:
                                    is_duplicate = True
                                    break

                                # 对于Linux，还要检查bin目录结构
                                if not is_win:
                                    # 如果一个是bin目录，一个不是，但指向同一安装
                                    if current_path.endswith('/bin') and not existing_path.endswith('/bin'):
                                        current_install = current_path[:-4]  # 移除 /bin
                                        if existing_normalized == os.path.normpath(current_install).lower():
                                            is_duplicate = True
                                            break
                                    elif existing_path.endswith('/bin') and not current_path.endswith('/bin'):
                                        existing_install = existing_path[:-4]  # 移除 /bin
                                        if current_normalized == os.path.normpath(existing_install).lower():
                                            is_duplicate = True
                                            break

                        if not is_duplicate:
                            # 计算正确的安装路径
                            if is_win:
                                # Windows: 直接使用包含python.exe的目录
                                install_path = os.path.dirname(exe_path)
                            else:
                                # Linux: 使用bin目录的父目录
                                install_path = os.path.dirname(os.path.dirname(exe_path))

                            versions.append({
                                "name": config["name"],
                                "path": install_path,
                                "version": version,
                                "from_vcpkg": False,
                                "exe_path": exe_path
                            })

            # 头文件/库文件检测
            elif "header" in config:
                header_path = find_file_in_dirs(config["header"], [path], ["include"])
                if header_path:
                    # 对于没有可执行文件的依赖，尝试从路径获取版本
                    version = extract_version_from_path(path, name)

                    # 特殊处理：对于XNNPACK等库，即使版本未知也接受
                    # 因为这些库主要是通过头文件使用的
                    allow_unknown = name.lower() in ["xnnpack", "stb", "cpuinfo", "pthreadpool"]

                    if version != "unknown" or allow_unknown:
                        if version == "unknown" or not min_version or compare_version(version, min_version) >= 0:
                            versions.append({
                                "name": config["name"],
                                "path": path,
                                "version": version if version != "unknown" else None,
                                "from_vcpkg": False
                            })

    # vcpkg版本
    # 特殊处理：Windows下XNNPACK跳过vcpkg，优先使用本地安装
    if VCPKG_ROOT and config.get("vcpkg_packages"):
        if not (is_win and name == "xnnpack"):
            vcpkg_found = search_in_vcpkg(name, config, is_win)
            if vcpkg_found["found"]:
                vcpkg_found["from_vcpkg"] = True
                versions.append(vcpkg_found)

    # 环境变量版本
    for env in config.get("env", []):
        env_val = os.environ.get(env)
        if env_val and os.path.exists(env_val):
            version = extract_version_from_path(env_val, name)
            if version != "unknown" and (not min_version or compare_version(version, min_version) >= 0):
                versions.append({
                    "name": config["name"],
                    "path": env_val,
                    "version": version,
                    "from_vcpkg": False
                })

    return versions

def search_dependency(name: str, sys_info: Dict, suppress_print: bool = False) -> Dict:
    """搜索单个依赖"""
    config = DEP_CONFIG.get(name)
    if not config:
        return {"found": False, "error": f"Unknown dependency: {name}"}

    
    # 特殊处理：MSVC使用专门的查找函数
    if name == "msvc" and sys_info["is_windows"]:
        return find_msvc()

    # 检查是否需要用户选择
    if config.get("user_selection", False):
        # Linux下的CMake和Ninja直接使用，不需要用户选择
        if sys_info.get("is_linux") and name in ["cmake", "ninja"]:
            # 直接进入原有逻辑进行自动选择
            pass
        else:
            versions = find_all_dependency_versions(name, sys_info)
            if versions:
                # 如果只找到一个版本，直接使用；如果找到多个版本，让用户选择
                if len(versions) == 1:
                    version = versions[0].get('version', 'unknown')
                    path = versions[0].get('path')
                    if not suppress_print:
                        formatted_output = format_dependency_output(config['name'], version, path)
                        print_ok(formatted_output)
                    versions[0]["found"] = True
                    return versions[0]
                else:
                    selected = get_user_choice(versions, f"Found {config['name']} installations:")
                    selected["found"] = True
                    return selected
            else:
                # 对于Python等用户选择的依赖，即使没找到也提供手动输入选项
                print_info(f"No {config['name']} installation found with required version")
                print_info("You can enter a custom path for {config['name']}")
                try:
                    custom_path = input(f"  Enter {config['name']} path (or press Enter to skip): ").strip()
                    if custom_path and os.path.exists(custom_path):
                        return {
                            "found": True,
                            "name": config["name"],
                            "path": os.path.dirname(custom_path),
                            "version": None,
                            "from_vcpkg": False,
                            "exe_path": custom_path
                        }
                except (EOFError, KeyboardInterrupt):
                    pass

                return {"found": False, "name": config["name"], "error": f"No {config['name']} installation found with required version"}

    # 原有逻辑：自动选择最佳版本
    result = {"found": False, "name": config["name"], "path": None, "version": None, "from_vcpkg": False}
    is_win = sys_info["is_windows"]

    # 特殊处理：Python包检测
    if "check_cmd" in config:
        # 对于NumPy等Python包，需要使用已选择的Python解释器
        if name == "numpy":
            python_exe = None

            # 1. 从全局Python依赖中获取已选择的Python路径
            from smart_config import global_deps
            if global_deps and "python" in global_deps and global_deps["python"]["found"]:
                python_exe = global_deps["python"].get("exe_path")

            # 2. 如果没有找到全局Python依赖，尝试从主流程中已检测的Python获取
            if not python_exe:
                # 检查主流程中是否已经检测了Python
                all_found = locals().get('all_found', {})  # 尝试从主循环获取
                if "python" in all_found and all_found["python"]["found"]:
                    python_exe = all_found["python"].get("exe_path")

                # 如果还没找到，检查found_other（如果已经在循环中）
                if not python_exe:
                    found_other = locals().get('found_other', {})
                    if "python" in found_other and found_other["python"]["found"]:
                        python_exe = found_other["python"].get("exe_path")

            # 3. 优先使用当前运行的Python解释器
            if not python_exe:
                current_python = sys.executable
                if current_python:
                    python_exe = current_python

            # 4. 根据不同平台尝试从常见路径查找Python
            if not python_exe:
                if sys_info.get("is_linux"):
                    python_paths = [
                        # 虚拟环境常见路径（从配置文件获取）
                        "/home/tech-renaissance/venv/py314/bin/python3",
                        "/home/ubuntu/venv/py314/bin/python3",
                        # 标准系统路径
                        "/usr/bin/python3",
                        "/usr/local/bin/python3",
                        "python3"
                    ]
                elif sys_info.get("is_windows"):
                    python_paths = [
                        "python",
                        "python3",
                        "py",
                        # 常见Windows Python安装路径
                        "C:/Python314/python.exe",
                        "C:/Python313/python.exe",
                        "C:/Python312/python.exe"
                    ]
                else:
                    python_paths = ["python3", "python"]

                for py_path in python_paths:
                    if os.path.exists(py_path):
                        python_exe = py_path
                        break
                    elif not py_path.endswith('.exe'):  # 可能是命令名，尝试运行
                        success, _ = run_cmd([py_path, "--version"])
                        if success:
                            python_exe = py_path
                            break

            if python_exe:
                # 使用找到的Python解释器检测NumPy
                cmd = [python_exe, "-c", "import numpy; print(numpy.__version__); print(numpy.__file__)"]
                success, output = run_cmd(cmd)
                if success:
                    lines = output.strip().split('\n')
                    if len(lines) >= 2:
                        result["found"] = True
                        result["version"] = lines[0].strip()
                        numpy_file = lines[1].strip()
                        # 获取NumPy的site-packages路径
                        result["path"] = os.path.dirname(numpy_file)
                        # 如果路径以site-packages结尾，保留它；否则向上移动
                        if not result["path"].endswith("site-packages"):
                            # 尝试找到site-packages目录
                            import pathlib
                            current_path = pathlib.Path(numpy_file)
                            for parent in current_path.parents:
                                if parent.name == "site-packages":
                                    result["path"] = str(parent)
                                    break
                    else:
                        result["found"] = True
                        result["version"] = lines[0].strip()
                        result["path"] = os.path.dirname(python_exe)  # 回退到Python路径

                    if "min_version" in config and compare_version(result["version"], config["min_version"]) < 0:
                        result["found"] = False
                        result["error"] = f"Version {result['version']} < {config['min_version']}"
                    return result
                else:
                    # 如果用找到的Python检测失败，继续使用原始逻辑
                    pass
            else:
                # 回退到原始逻辑
                cmd = config["check_cmd"]
        else:
            cmd = config["check_cmd"]

        success, output = run_cmd(cmd)
        if success:
            lines = output.strip().split('\n')
            if name == "numpy" and len(lines) >= 2:
                # NumPy特殊处理：解析版本和文件路径
                result["found"] = True
                result["version"] = lines[0].strip()
                numpy_file = lines[1].strip()
                # 获取NumPy的site-packages路径
                result["path"] = os.path.dirname(numpy_file)
                # 如果路径以site-packages结尾，保留它；否则向上移动
                if not result["path"].endswith("site-packages"):
                    # 尝试找到site-packages目录
                    import pathlib
                    current_path = pathlib.Path(numpy_file)
                    for parent in current_path.parents:
                        if parent.name == "site-packages":
                            result["path"] = str(parent)
                            break
            else:
                result["found"] = True
                result["version"] = output.strip()

            if "min_version" in config and compare_version(result["version"], config["min_version"]) < 0:
                result["found"] = False
                result["error"] = f"Version {result['version']} < {config['min_version']}"
        return result

    # 第一层：vcpkg搜索 (最高优先级)
    # 特殊处理：Windows下XNNPACK跳过vcpkg，优先使用本地安装
    if VCPKG_ROOT and config.get("vcpkg_packages"):
        if not (is_win and name == "xnnpack"):
            vcpkg_found = search_in_vcpkg(name, config, is_win)
            if vcpkg_found["found"]:
                result = vcpkg_found
                result["from_vcpkg"] = True
                return result

    # 第二层：环境变量
    for env in config.get("env", []):
        env_val = os.environ.get(env)
        if env_val and os.path.exists(env_val):
            result["path"] = env_val
            result["found"] = True
            break

    # 第三层：常见路径搜索
    if not result["found"]:
        paths = config.get("paths_win" if is_win else "paths_linux", [])

        # 可执行文件搜索
        if "exe" in config and config["exe"]:  # 确保exe列表不为空
            for exe in config["exe"]:
                # 先检查PATH
                exe_path = find_exe_in_path([exe])
                if exe_path:
                    result["path"] = os.path.dirname(os.path.dirname(exe_path))
                    result["found"] = True
                    break

                # 再检查常见路径
                for p in paths:
                    bin_sub = config.get("bin_subdir", "")
                    check_path = os.path.join(p, bin_sub, exe) if bin_sub else os.path.join(p, exe)

                    # 处理通配符路径
                    if '*' in check_path:
                        matches = glob.glob(check_path)
                        if matches:
                            result["path"] = os.path.dirname(os.path.dirname(matches[0]))
                            result["found"] = True
                            break
                    elif os.path.exists(check_path):
                        result["path"] = p
                        result["found"] = True
                        break
                if result["found"]:
                    break

        # 头文件搜索（支持多个头文件）
        elif "header" in config:
            headers = config["header"] if isinstance(config["header"], list) else [config["header"]]
            for header in headers:
                found_path = find_file_in_dirs(header, paths)
                if found_path:
                    result["path"] = found_path
                    result["found"] = True
                    break

        # 库文件搜索
        elif "lib_files" in config:
            for lib_file in config["lib_files"]:
                found_path = find_file_in_dirs(lib_file, paths)
                if found_path:
                    result["path"] = found_path
                    result["found"] = True
                    break

    # 版本检测 - 只有在result["found"]为True且尚未设置version时才执行
    if result["found"] and not result.get("version"):
        # 特殊处理：使用路径推导版本号（如cuDNN）
        if config.get("use_path_version"):
            if result["path"]:
                result["version"] = extract_version_from_path(result["path"], name)
        # 标准版本命令检测
        elif "version_cmd" in config:
            cmd = config["version_cmd"].copy()
            if result["path"]:
                # 处理路径替换占位符
                for i, arg in enumerate(cmd):
                    if "{path}" in arg:
                        cmd[i] = arg.replace("{path}", result["path"])
                    elif "{include}" in arg:
                        include_path = os.path.join(result["path"], "include")
                        cmd[i] = arg.replace("{include}", include_path)

                # 处理可执行文件路径替换
                bin_sub = config.get("bin_subdir", "bin")
                exe_path = os.path.join(result["path"], bin_sub, cmd[0])
                if os.path.exists(exe_path):
                    cmd[0] = exe_path
                elif os.path.exists(exe_path + ".exe"):
                    cmd[0] = exe_path + ".exe"

            success, output = run_cmd(cmd)
            if success and "version_pattern" in config:
                result["version"] = extract_version(output, config["version_pattern"])

            # 特殊处理：对于NCCL，需要从多行输出中组合版本号
            if name == "nccl" and success:
                major_match = re.search(r"#define NCCL_MAJOR\s+(\d+)", output)
                minor_match = re.search(r"#define NCCL_MINOR\s+(\d+)", output)
                patch_match = re.search(r"#define NCCL_PATCH\s+(\d+)", output)
                if major_match and minor_match:
                    patch = patch_match.group(1) if patch_match else "0"
                    result["version"] = f"{major_match.group(1)}.{minor_match.group(1)}.{patch}"

            # 版本检查
            if result["version"] and "min_version" in config:
                if compare_version(result["version"], config["min_version"]) < 0:
                    result["found"] = False
                    result["error"] = f"Version {result['version']} < {config['min_version']}"

    return result


def get_vcpkg_full_output() -> str:
    """获取完整的vcpkg list输出并缓存（只执行一次）"""
    global vcpkg_full_output_cache

    if vcpkg_full_output_cache is not None:
        return vcpkg_full_output_cache

    vcpkg_exe = os.path.join(VCPKG_ROOT, "vcpkg.exe") if sys.platform == "win32" else os.path.join(VCPKG_ROOT, "vcpkg")

    if not os.path.exists(vcpkg_exe):
        vcpkg_full_output_cache = ""
        return ""

    success, output = run_cmd([vcpkg_exe, "list"])
    if success:
        vcpkg_full_output_cache = output
        return output
    else:
        vcpkg_full_output_cache = ""
        return ""

def get_vcpkg_package_version(package_name: str, triplet: str = None) -> Optional[str]:
    """通过缓存字符串搜索获取包版本号（全平台统一优化版本）"""
    global vcpkg_cache

    # 检查缓存
    if package_name in vcpkg_cache:
        return vcpkg_cache[package_name]

    # 获取完整的vcpkg list输出（带缓存）
    full_output = get_vcpkg_full_output()
    if not full_output:
        vcpkg_cache[package_name] = None
        return None

    # 在输出字符串中搜索包含包名的行
    lines = full_output.strip().split('\n')
    import re

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 查找包含 "包名:" 的行
        if f"{package_name}:" not in line:
            continue

        # 分割第一部分（包名+架构）
        first_space_idx = line.find(' ')
        if first_space_idx == -1:
            continue

        package_with_arch = line[:first_space_idx]

        # 确保是我们要的包名开头（避免误匹配）
        if not package_with_arch.startswith(f"{package_name}:"):
            continue

        # 提取版本号部分（在第一个空格和第二个空格之间）
        version_part = line[first_space_idx:].strip()
        version_space_idx = version_part.find(' ')
        if version_space_idx == -1:
            version_str = version_part
        else:
            version_str = version_part[:version_space_idx].strip()

        # 验证版本号格式
        result_version = None
        if re.match(r'^\d+\.\d+', version_str):  # 如 3.7, 3.7.0
            # 移除可能的#n后缀
            if '#' in version_str:
                version_str = version_str.split('#')[0]
            result_version = version_str.strip()
        elif re.match(r'^\d{4}-\d{2}-\d{2}', version_str):  # 如 2024-08-20
            # 移除可能的#n后缀
            if '#' in version_str:
                version_str = version_str.split('#')[0]
            result_version = version_str.strip()

        # 缓存结果
        vcpkg_cache[package_name] = result_version
        return result_version

    vcpkg_cache[package_name] = None
    return None

def search_in_vcpkg(name: str, config: Dict, is_win: bool) -> Dict:
    """在vcpkg中搜索依赖"""
    if not VCPKG_ROOT or not os.path.exists(VCPKG_ROOT):
        return {"found": False}

    # 根据系统架构确定triplet
    if is_win:
        triplet = "x64-windows"
    else:
        # 检测系统架构
        import platform
        machine = platform.machine().lower()
        if machine == "x86_64":
            # x64平台：默认使用 x64-linux
            triplet = "x64-linux"
            # 只有 XNNPACK 需要动态库（由于 microkernel 链接问题）
            if not is_win and name == "xnnpack":  # Linux + XNNPACK
                dynamic_path = os.path.join(VCPKG_ROOT, "installed", "x64-linux-dynamic")
                if os.path.exists(dynamic_path):
                    triplet = "x64-linux-dynamic"
        elif machine in ["arm64", "aarch64"]:
            if name == "xnnpack":
                dynamic_path = os.path.join(VCPKG_ROOT, "installed", "arm64-linux-dynamic")
                if os.path.exists(dynamic_path):
                    triplet = "arm64-linux-dynamic"
            else:
                triplet = "arm64-linux"
        elif machine.startswith("arm"):
            if name == "xnnpack":
                dynamic_path = os.path.join(VCPKG_ROOT, "installed", "arm-linux-dynamic")
                if os.path.exists(dynamic_path):
                    triplet = "arm-linux-dynamic"
            else:
                triplet = "arm-linux"
        elif machine.startswith("aarch"):
            if name == "xnnpack":
                dynamic_path = os.path.join(VCPKG_ROOT, "installed", "aarch64-linux-dynamic")
                if os.path.exists(dynamic_path):
                    triplet = "aarch64-linux-dynamic"
            else:
                triplet = "aarch64-linux"
        elif machine.startswith("riscv"):
            # RISC-V特殊处理：查找所有riscv*开头的triplet
            vcpkg_installed_root = os.path.join(VCPKG_ROOT, "installed")
            triplet = None

            # 查找所有以riscv开头的triplet目录
            for item in os.listdir(vcpkg_installed_root):
                item_path = os.path.join(vcpkg_installed_root, item)
                if os.path.isdir(item_path) and item.startswith('riscv'):
                    triplet = item
                    break

            # 如果没找到，使用默认
            if not triplet:
                triplet = "riscv64-linux"

    vcpkg_installed = os.path.join(VCPKG_ROOT, "installed", triplet)

    # 检查头文件 - 使用统一的字符串搜索逻辑
    if "header" in config:
        headers = config["header"] if isinstance(config["header"], list) else [config["header"]]

        # 首先在默认triplet中查找
        for header in headers:
            header_path = os.path.join(vcpkg_installed, "include", header)
            if os.path.exists(header_path):
                result = {
                    "found": True,
                    "name": config["name"],
                    "path": vcpkg_installed,
                    "from_vcpkg": True
                }

                # 获取vcpkg包的版本号（使用统一的全平台算法）
                vcpkg_packages = config.get("vcpkg_packages", [])
                if vcpkg_packages:
                    package_name = vcpkg_packages[0]  # 取第一个包名
                    version = get_vcpkg_package_version(package_name)
                    if version:
                        result["version"] = version

                return result

        # 如果在默认triplet中没找到，且是RISC-V，尝试在其他riscv* triplet中查找
        if machine and machine.startswith("riscv"):
            for item in os.listdir(vcpkg_installed_root):
                item_path = os.path.join(vcpkg_installed_root, item)
                if os.path.isdir(item_path) and item.startswith('riscv') and item != os.path.basename(vcpkg_installed):
                    for header in headers:
                        header_path = os.path.join(item_path, "include", header)
                        if os.path.exists(header_path):
                            result = {
                                "found": True,
                                "name": config["name"],
                                "path": item_path,
                                "from_vcpkg": True
                            }

                            # 获取vcpkg包的版本号（使用统一的全平台算法）
                            vcpkg_packages = config.get("vcpkg_packages", [])
                            if vcpkg_packages:
                                package_name = vcpkg_packages[0]  # 取第一个包名
                                version = get_vcpkg_package_version(package_name)
                                if version:
                                    result["version"] = version

                            return result

    return {"found": False}

def search_dependencies(scene: str, sys_info: Dict) -> Dict:
    """搜索场景所需的所有依赖"""
    scene_config = SCENE_DEPS[scene]
    all_deps = scene_config["required"] + scene_config.get("optional", [])

    found_deps = {}

    print_info(f"Checking dependencies for {scene_config['name']}...")

    for dep_name in all_deps:
        is_required = dep_name in scene_config["required"]
        status = "required" if is_required else "optional"

        result = search_dependency(dep_name, sys_info)
        found_deps[dep_name] = result

        if result["found"]:
            version = result.get("version", "")
            path = result.get("path")
            if not path or path == "system":
                # 尝试获取可执行文件路径作为路径信息
                if result.get("exe_path"):
                    path = os.path.dirname(result["exe_path"])
                else:
                    path = "system PATH"
            formatted_output = format_dependency_output(result['name'], version, path)
            print_ok(f"{formatted_output} ({status})")
        else:
            if is_required:
                print_fail(f"{DEP_CONFIG[dep_name]['name']} ({status}) - NOT FOUND")
                reason = result.get("error", "Not found")
                print_info(f"  Reason: {reason}")
            else:
                print_warn(f"{DEP_CONFIG[dep_name]['name']} ({status}) - not found (optional)")

    return found_deps

def check_required_deps(scene: str, found_deps: Dict) -> List[str]:
    """检查必需依赖是否都找到了"""
    scene_config = SCENE_DEPS[scene]
    missing_required = []

    for dep_name in scene_config["required"]:
        if not found_deps.get(dep_name, {}).get("found", False):
            missing_required.append(dep_name)

    return missing_required

# ============================================================================
# 配置文件生成
# ============================================================================

def generate_cmake_config(scene: str, deps: Dict, sys_info: Dict) -> str:
    """生成CMake配置文件内容 - 支持新的编译宏系统"""
    lines = [
        "# Auto-generated by configure.py",
        f"# Scene: {SCENE_DEPS[scene]['name']}",
        f"# Generated on: {platform.node()}",
        ""
    ]

    # vcpkg工具链
    if VCPKG_ROOT and os.path.exists(VCPKG_ROOT):
        toolchain = os.path.join(VCPKG_ROOT, "scripts/buildsystems/vcpkg.cmake").replace("\\", "/")
        lines.append(f'set(CMAKE_TOOLCHAIN_FILE "{toolchain}")')

    # CUDA配置
    if "cuda" in deps and deps["cuda"]["found"]:
        cuda_path = deps["cuda"]["path"].replace("\\", "/")
        lines.append(f'set(CUDAToolkit_ROOT "{cuda_path}")')

    # cuDNN配置
    if "cudnn" in deps and deps["cudnn"]["found"]:
        cudnn_path = deps["cudnn"]["path"].replace("\\", "/")
        lines.append(f'set(CUDNN_ROOT "{cudnn_path}")')

    # MUSA配置
    if "musa" in deps and deps["musa"]["found"]:
        musa_path = deps["musa"]["path"].replace("\\", "/")
        lines.append(f'set(MUSA_ROOT "{musa_path}")')

    # muDNN配置
    if "mudnn" in deps and deps["mudnn"]["found"]:
        mudnn_path = deps["mudnn"]["path"].replace("\\", "/")
        lines.append(f'set(MUDNN_ROOT "{mudnn_path}")')

    # Python配置
    if "python" in deps and deps["python"]["found"]:
        python_path = deps["python"].get("exe_path")
        if python_path:
            # 使用正斜杠替换Windows反斜杠，避免转义问题
            lines.append(f'set(Python3_EXECUTABLE "{python_path.replace(chr(92), "/")}" )')
        else:
            # 回退到PATH查找
            python_exe = find_exe_in_path(["python3", "python", "python.exe"])
            if python_exe:
                lines.append(f'set(Python3_EXECUTABLE "{python_exe.replace(chr(92), "/")}" )')

    # MSVC配置已通过build.bat中的vcvars64.bat处理，无需在cmake_paths.cmake中重复设置

    # 添加GPU数量信息
    gpu_count = 0
    if "cuda" in deps and deps["cuda"]["found"]:
        # 尝试从GPU检测信息获取GPU数量
        try:
            from smart_config import detect_gpu
            gpu_info = detect_gpu()
            if gpu_info["detected"] and gpu_info["type"] == "nvidia":
                gpu_count = gpu_info["count"]
        except:
            pass

    # 基础CMake选项
    lines.append("# Scene-specific options")
    for opt in SCENE_DEPS[scene]["cmake_opts"]:
        key, val = opt.split("=")
        lines.append(f'set({key.replace("-D", "")} {val})')

    # 添加GPU数量宏
    if gpu_count > 0:
        lines.append(f'set(TR_NUM_GPUS {gpu_count})')
    else:
        lines.append('set(TR_NUM_GPUS 0)')

    # 添加CPU架构宏
    arch = sys_info["arch"]
    if arch == "x86_64":
        lines.append('set(TR_CPU_ARCH_X86_64 ON)')
    elif arch == "arm64":
        lines.append('set(TR_CPU_ARCH_ARM64 ON)')
    elif arch == "riscv":
        lines.append('set(TR_CPU_ARCH_RISCV64 ON)')

    # 添加所有依赖项的具体路径配置（V3.2.0新增）
    lines.append("")
    lines.append("# Dependency library paths (V3.2.0 enhancement)")

    # CUDA/cuDNN路径（已在前面设置，这里确保一致性）
    if "cuda" in deps and deps["cuda"]["found"]:
        cuda_path = deps["cuda"]["path"].replace("\\", "/")
        lines.append(f'set(TR_CUDA_PATH "{cuda_path}")')

    if "cudnn" in deps and deps["cudnn"]["found"]:
        cudnn_path = deps["cudnn"]["path"].replace("\\", "/")
        lines.append(f'set(TR_CUDNN_PATH "{cudnn_path}")')
        # 确保include和lib路径
        if sys_info["is_windows"]:
            lines.append(f'set(TR_CUDNN_INCLUDE_DIR "{cudnn_path}/include")')
            lines.append(f'set(TR_CUDNN_LIBRARY_DIR "{cudnn_path}/lib")')
        else:
            lines.append(f'set(TR_CUDNN_INCLUDE_DIR "{cudnn_path}/include")')
            lines.append(f'set(TR_CUDNN_LIBRARY_DIR "{cudnn_path}/lib64")')

    # MUSA/muDNN路径
    if "musa" in deps and deps["musa"]["found"]:
        musa_path = deps["musa"]["path"].replace("\\", "/")
        lines.append(f'set(TR_MUSA_PATH "{musa_path}")')

    if "mudnn" in deps and deps["mudnn"]["found"]:
        mudnn_path = deps["mudnn"]["path"].replace("\\", "/")
        lines.append(f'set(TR_MUDNN_PATH "{mudnn_path}")')
        lines.append(f'set(TR_MUDNN_INCLUDE_DIR "{mudnn_path}/include")')

        # 检测MUDNN库目录（MUSA使用lib，不是lib64）
        if sys_info["is_windows"]:
            # Windows: 使用lib
            mudnn_lib_dir = f"{mudnn_path}/lib"
        else:
            # Linux: MUSA平台实际使用lib目录，不是lib64
            # 需要检测实际存在的目录
            mudnn_lib_dir = f"{mudnn_path}/lib"  # MUSA默认使用lib
            # 如果lib不存在，回退到lib64（兼容其他可能的MUSA安装）
            # 注意：这里不进行实际文件系统检测，因为在配置阶段可能尚未安装MUSA

        lines.append(f'set(TR_MUDNN_LIBRARY_DIR "{mudnn_lib_dir}")')

    # NCCL路径（仅Linux，GPU_CLOUD场景专用）
    if "nccl" in deps and deps["nccl"]["found"]:
        nccl_path = deps["nccl"]["path"].replace("\\", "/")
        lines.append(f'set(TR_NCCL_PATH "{nccl_path}")')
        if sys_info["is_linux"]:
            # NCCL通常安装在/usr/include和/usr/lib
            lines.append('set(TR_NCCL_INCLUDE_DIR "/usr/include")')
            lines.append('set(TR_NCCL_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu")')
        else:
            # Windows暂不支持NCCL
            lines.append(f'set(TR_NCCL_INCLUDE_DIR "{nccl_path}/include")')
            lines.append(f'set(TR_NCCL_LIBRARY_DIR "{nccl_path}/lib")')

    # vcpkg安装的依赖项路径 (使用实际的vcpkg包名)
    vcpkg_packages = {
        "onednn": "oneDNN",           # vcpkg包名: onednn
        "xnnpack": "XNNPACK",         # vcpkg包名: xnnpack
        "mimalloc": "mimalloc",       # vcpkg包名: mimalloc
        "zlib": "zlib",              # vcpkg包名: zlib
        "libcurl": "CURL",           # vcpkg包名: curl
        "libjpeg-turbo": "JPEG",     # vcpkg包名: libjpeg-turbo
        "stb": "STB",               # vcpkg包名: stb
        "simd": "Simd"              # vcpkg包名: simd
    }

    for dep_key, cmake_name in vcpkg_packages.items():
        if dep_key in deps and deps[dep_key]["found"]:
            dep_path = deps[dep_key]["path"].replace("\\", "/")
            lines.append(f'set(TR_{cmake_name.upper()}_PATH "{dep_path}")')

            # 根据平台确定include和lib子目录
            if sys_info["is_windows"]:
                # 特殊处理：Windows + XNNPACK 使用本地安装（C:\XNNPACK）
                if dep_key == "xnnpack" and not deps[dep_key].get("from_vcpkg", False):
                    # 使用本地安装路径，包含include和lib子目录
                    include_dir = f"{dep_path}/include"
                    lib_dir = f"{dep_path}/lib"
                    lines.append(f'set(TR_{cmake_name.upper()}_TRIPLET "x64-windows")')
                else:
                    # Windows下使用packages目录，需要映射到实际的vcpkg包名
                    vcpkg_package_name = {
                        "libcurl": "curl",
                        "libjpeg-turbo": "libjpeg-turbo",
                        "onednn": "onednn",
                        "xnnpack": "xnnpack",
                        "mimalloc": "mimalloc",
                        "zlib": "zlib",
                        "stb": "stb",
                        "simd": "simd"
                    }.get(dep_key, dep_key)

                    installed_triplet = get_installed_triplet(dep_key, sys_info) or "x64-windows"
                    package_name = f"{vcpkg_package_name}_{installed_triplet}"
                    include_dir = f"{VCPKG_ROOT.replace(chr(92), '/')}/packages/{package_name}/include"
                    lib_dir = f"{VCPKG_ROOT.replace(chr(92), '/')}/packages/{package_name}/lib"
                    lines.append(f'set(TR_{cmake_name.upper()}_TRIPLET "{installed_triplet}")')
            else:
                # Linux下优先使用installed目录，如果不存在则检查packages目录
                installed_triplet = get_installed_triplet(dep_key, sys_info)
                if installed_triplet:
                    # 先检查installed目录
                    installed_include = f"{VCPKG_ROOT}/installed/{installed_triplet}/include"
                    installed_lib = f"{VCPKG_ROOT}/installed/{installed_triplet}/lib"

                    # 如果installed目录不存在，检查packages目录（某些架构如RISC-V）
                    if not os.path.exists(installed_include):
                        # 映射依赖名到vcpkg包名
                        vcpkg_package_name = {
                            "libcurl": "curl",
                            "libjpeg-turbo": "libjpeg-turbo",
                            "onednn": "onednn",
                            "xnnpack": "xnnpack",
                            "mimalloc": "mimalloc",
                            "zlib": "zlib",
                            "stb": "stb",
                            "simd": "simd"
                        }.get(dep_key, dep_key)

                        packages_include = f"{VCPKG_ROOT}/packages/{vcpkg_package_name}_{installed_triplet}/include"
                        packages_lib = f"{VCPKG_ROOT}/packages/{vcpkg_package_name}_{installed_triplet}/lib"

                        if os.path.exists(packages_include):
                            include_dir = packages_include
                            lib_dir = packages_lib
                        else:
                            # 回退到installed目录
                            include_dir = installed_include
                            lib_dir = installed_lib
                    else:
                        include_dir = installed_include
                        lib_dir = installed_lib

                    lines.append(f'set(TR_{cmake_name.upper()}_TRIPLET "{installed_triplet}")')
                else:
                    # 回退到默认triplet
                    default_triplet = get_default_triplet(sys_info)
                    default_include = f"{VCPKG_ROOT}/installed/{default_triplet}/include"
                    default_lib = f"{VCPKG_ROOT}/installed/{default_triplet}/lib"

                    # 检查packages目录
                    vcpkg_package_name = {
                        "libcurl": "curl",
                        "libjpeg-turbo": "libjpeg-turbo",
                        "onednn": "onednn",
                        "xnnpack": "xnnpack",
                        "mimalloc": "mimalloc",
                        "zlib": "zlib",
                        "stb": "stb",
                        "simd": "simd"
                    }.get(dep_key, dep_key)

                    packages_include = f"{VCPKG_ROOT}/packages/{vcpkg_package_name}_{default_triplet}/include"
                    packages_lib = f"{VCPKG_ROOT}/packages/{vcpkg_package_name}_{default_triplet}/lib"

                    if os.path.exists(packages_include):
                        include_dir = packages_include
                        lib_dir = packages_lib
                    else:
                        include_dir = default_include
                        lib_dir = default_lib

                    lines.append(f'set(TR_{cmake_name.upper()}_TRIPLET "{default_triplet}")')

            lines.append(f'set(TR_{cmake_name.upper()}_INCLUDE_DIR "{include_dir}")')
            lines.append(f'set(TR_{cmake_name.upper()}_LIBRARY_DIR "{lib_dir}")')

    # STB现在通过vcpkg_packages统一处理，无需特殊处理

    # 添加项目路径宏
    project_root = os.path.abspath(os.getcwd()).replace("\\", "/")

    lines.append("")
    lines.append("# Project path macros")
    lines.append(f'set(TR_PROJECT_ROOT "{project_root}")')
    lines.append('set(TR_WORKSPACE "${TR_PROJECT_ROOT}/workspace")')

    return "\n".join(lines)

def generate_build_script(scene: str, deps: Dict, sys_info: Dict) -> Tuple[str, str]:
    """生成编译脚本，返回(文件名, 内容)"""
    cmake_opts = " ".join(SCENE_DEPS[scene]["cmake_opts"])

    # 添加vcpkg工具链选项
    if VCPKG_ROOT and os.path.exists(VCPKG_ROOT):
        toolchain_opt = f'-DCMAKE_TOOLCHAIN_FILE="{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"'
    else:
        toolchain_opt = ""

    if sys_info["is_windows"]:
        # Windows脚本 - 2025-12-23验证版本
        # 使用正确的构建目录：build/windows-msvc-release
        windows_build_dir = "build/windows-msvc-release"

        # VS路径推导
        vcvars_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
        if "msvc" in deps and deps["msvc"]["found"]:
            msvc_path = deps["msvc"]["path"]
            # 从MSVC路径推导VS根目录，统一使用正斜杠
            msvc_path = msvc_path.replace("\\", "/")
            if "/VC/Tools/MSVC" in msvc_path:
                vs_root = msvc_path.split("/VC/Tools/MSVC")[0]
            else:
                # 默认路径
                vs_root = "C:/Program Files/Microsoft Visual Studio/2022/Community"
            vcvars_path = vs_root + "/VC/Auxiliary/Build/vcvars64.bat"

        # vcpkg路径统一使用正斜杠
        vcpkg_root = VCPKG_ROOT.replace("\\", "/") if VCPKG_ROOT else "T:/Softwares/vcpkg"

        script = f'''@echo off
REM ================================================================
REM renAIssance Framework - Build Script (Windows)
REM Generated by configure.py - Verified 2025-12-23
REM Build Directory: {windows_build_dir}
REM ================================================================

setlocal enabledelayedexpansion

REM Get script directory (project root) and remove trailing backslash
set PROJECT_ROOT=%~dp0
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

echo [INFO] renAIssance Framework Build (Windows)
echo [INFO] Build Directory: {windows_build_dir}
echo [INFO] Parallel Jobs: {PARALLEL_JOBS}

REM Key step: Initialize Visual Studio Developer Command Prompt environment
echo [INFO] Initializing Visual Studio Developer Command Prompt...
call "{vcvars_path}"

if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize Visual Studio environment
    echo [INFO] Please check Visual Studio installation
    exit /b 1
)

REM Clean and create build directory (using correct directory name)
if exist {windows_build_dir} (
    echo [INFO] Cleaning existing build directory...
    rmdir /s /q {windows_build_dir}
)
mkdir {windows_build_dir}

REM Configure CMake - using verified parameters from 2025-12-23
REM Note: Use -B and -S to explicitly specify build and source directories
echo [INFO] Configuring with CMake...
cmake -G Ninja ^
    -S "%PROJECT_ROOT%" ^
    -B {windows_build_dir} ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_TOOLCHAIN_FILE={vcpkg_root}/scripts/buildsystems/vcpkg.cmake ^
    {cmake_opts}

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed
    echo [INFO] This might be due to missing dependencies or path issues
    exit /b 1
)

REM Build all targets
echo [INFO] Building all targets with Ninja...
cmake --build {windows_build_dir} --parallel {PARALLEL_JOBS}

if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    echo [INFO] This might be due to missing dependencies or compilation errors
    exit /b 1
)

echo [OK] Build completed successfully!
echo [INFO] Build artifacts are in: {windows_build_dir}

REM Return to root directory
cd "%PROJECT_ROOT%"
endlocal
'''
        return "build.bat", script

    else:
        # Linux脚本 - 使用特定构建目录名避免时间戳问题
        linux_build_dir = "build/linux-gcc-release"
        script = f'''#!/bin/bash
# ================================================================
# renAIssance Framework - Build Script (Linux)
# Generated by configure.py
# Version: V3.3.4
# ================================================================

set -e  # 遇到错误时退出

echo "[INFO] renAIssance Framework Alpha Build (Linux)"
echo "[INFO] Build Directory: {BUILD_DIR}"
echo "[INFO] Parallel Jobs: {PARALLEL_JOBS}"

# 清理并创建构建目录（使用特定目录名避免时间戳冲突）
if [ -d "{BUILD_DIR}" ]; then
    echo "[INFO] Cleaning existing build directory..."
    rm -rf {BUILD_DIR}
fi
mkdir -p {BUILD_DIR}
cd {BUILD_DIR}

# 配置CMake - 使用Ninja构建系统，避免时间戳问题
echo "[INFO] Configuring with CMake..."
cmake .. \\
    -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_CXX_COMPILER=g++ \\
    -DCMAKE_TOOLCHAIN_FILE="{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \\
    -DCMAKE_DISABLE_SOURCE_CHANGES=ON \\
    -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \\
    -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON \\
    {cmake_opts}

if [ $? -ne 0 ]; then
    echo "[ERROR] CMake configuration failed"
    exit 1
fi

# 编译项目 - 使用Ninja并行构建（通用版）
echo "[INFO] Building with Ninja..."
ninja -j{PARALLEL_JOBS}

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed"
    exit 1
fi

echo "[OK] Build completed successfully!"
echo "[INFO] Build artifacts are in: {BUILD_DIR}"

# 返回根目录
cd ..
'''
        return "build.sh", script

def generate_config_files(scene: str, found_deps: Dict, sys_info: Dict):
    """生成所有配置文件"""
    # 创建config目录（在根目录下）
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # 生成cmake_paths.cmake
    cmake_content = generate_cmake_config(scene, found_deps, sys_info)
    cmake_path = config_dir / "cmake_paths.cmake"
    cmake_path.write_text(cmake_content)
    print_ok(f"Generated: {cmake_path}")

    # 获取GPU数量信息
    gpu_count = 0
    try:
        gpu_info = detect_gpu()
        if gpu_info["detected"]:
            gpu_count = gpu_info["count"]
    except:
        pass

    # 生成project_config.json
    json_config = {
        "system": sys_info,
        "scene": scene,
        "scene_name": SCENE_DEPS[scene]["name"],
        "build_config": {
            "build_dir": BUILD_DIR,
            "parallel_jobs": PARALLEL_JOBS,
            "cmake_generator": CMAKE_GENERATOR,
            "cmake_build_type": CMAKE_BUILD_TYPE,
            "vcpkg_root": VCPKG_ROOT
        },
        "gpu_info": {
            "detected": gpu_count > 0,
            "count": gpu_count,
            "type": gpu_info.get("type") if gpu_count > 0 else None,
            "name": gpu_info.get("name") if gpu_count > 0 else None
        },
        "dependencies": {k: {
            "found": v["found"],
            "path": v.get("path"),
            "version": v.get("version"),
            "from_vcpkg": v.get("from_vcpkg", False)
        } for k, v in found_deps.items()},
        "cmake_options": SCENE_DEPS[scene]["cmake_opts"],
        "compilation_macros": {
            "scene_macros": [opt for opt in SCENE_DEPS[scene]["cmake_opts"] if opt.startswith("-DTR_SCENE_")],
            "feature_macros": [opt for opt in SCENE_DEPS[scene]["cmake_opts"] if opt.startswith("-DTR_USE_")],
            "system_macros": {
                "TR_NUM_GPUS": gpu_count,
                "TR_IS_EDGE_DEVICE": scene in ["edge_arm", "edge_riscv"]
            }
        }
    }
    json_path = config_dir / "project_config.json"
    json_path.write_text(json.dumps(json_config, indent=2))
    print_ok(f"Generated: {json_path}")

    # 生成编译脚本（在根目录下）
    script_name, script_content = generate_build_script(scene, found_deps, sys_info)
    script_path = Path(script_name)
    script_path.write_text(script_content)
    if not sys_info["is_windows"]:
        os.chmod(script_path, 0o755)
    print_ok(f"Generated: {script_path}")

def find_msvc():
    """专门查找MSVC编译器"""
    print_info("Looking for MSVC...")

    # 1. 直接搜索技术觉醒2中确认的路径
    known_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
    if os.path.exists(known_path):
        version = "14.44.35207"
        path = os.path.dirname(os.path.dirname(known_path))
        formatted_output = format_dependency_output("MSVC", version, path)
        print_ok(formatted_output)
        return {
            "found": True,
            "name": "MSVC",
            "path": path,
            "exe_path": known_path,
            "version": version,
            "from_vcpkg": False
        }

    # 2. 使用glob搜索所有可能的MSVC路径
    import glob

    base_patterns = [
        "C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe",
        "C:/Program Files/Microsoft Visual Studio/2019/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe",
    ]

    for pattern in base_patterns:
        matches = glob.glob(pattern)
        if matches:
            # 选择最新的版本（按路径排序）
            latest_cl = max(matches)
            version = extract_version_from_path(os.path.dirname(latest_cl), "msvc")
            path = os.path.dirname(os.path.dirname(latest_cl))
            formatted_output = format_dependency_output("MSVC", version, path)
            print_ok(formatted_output)
            return {
                "found": True,
                "name": "MSVC",
                "path": path,
                "exe_path": latest_cl,
                "version": version,
                "from_vcpkg": False
            }

    # 3. 在PATH中查找
    cl_path = find_exe_in_path(["cl.exe"])
    if cl_path:
        path = os.path.dirname(os.path.dirname(cl_path))
        formatted_output = format_dependency_output("MSVC", "", path)
        print_ok(formatted_output)
        return {
            "found": True,
            "name": "MSVC",
            "path": path,
            "exe_path": cl_path,
            "version": "unknown",
            "from_vcpkg": False
        }

    print_warn("MSVC not found")
    return {"found": False, "name": "MSVC", "error": "MSVC compiler not found"}

def check_vcpkg():
    """检查vcpkg是否可用"""
    if not VCPKG_ROOT:
        print_fail("VCPKG_ROOT not set")
        return False

    # 跨平台vcpkg可执行文件名
    vcpkg_exe_name = "vcpkg.exe" if sys.platform == "win32" else "vcpkg"
    vcpkg_exe = os.path.join(VCPKG_ROOT, vcpkg_exe_name)

    if not os.path.exists(vcpkg_exe):
        print_fail(f"vcpkg not found at {vcpkg_exe}")
        return False

    return True

def get_installed_triplet(dep_name: str, sys_info: Dict) -> str:
    """从vcpkg list输出中获取实际安装的triplet"""
    global vcpkg_full_output_cache

    if not vcpkg_full_output_cache:
        return None

    # 搜索依赖项的实际安装记录
    # 支持多种包名格式：onednn, libcurl, libjpeg-turbo等
    import re

    # 可能的包名列表
    possible_names = []
    if dep_name == "onednn":
        possible_names = ["onednn"]
    elif dep_name == "libcurl":
        possible_names = ["curl"]
    elif dep_name == "libjpeg-turbo":
        possible_names = ["libjpeg-turbo"]
    elif dep_name == "xnnpack":
        possible_names = ["xnnpack"]
    elif dep_name == "mimalloc":
        possible_names = ["mimalloc"]
    elif dep_name == "zlib":
        possible_names = ["zlib"]
    elif dep_name == "stb":
        possible_names = ["stb"]

    for name in possible_names:
        # 查找 pattern: "name:triplet"
        pattern = rf"{name}:([^\s]+)"
        match = re.search(pattern, vcpkg_full_output_cache, re.IGNORECASE)
        if match:
            return match.group(1)

    return None

def get_default_triplet(sys_info: Dict) -> str:
    """获取默认triplet（作为回退方案）"""
    arch = sys_info.get("arch", "x86_64")

    if arch == "x86_64":
        return "x64-linux" if not sys_info.get("is_windows", False) else "x64-windows"
    elif arch == "aarch64" or arch == "arm64":
        return "arm64-linux"
    elif arch.startswith("arm"):
        return "arm-linux"
    elif arch.startswith("riscv64"):
        return "riscv64-linux"
    else:
        return "x64-linux"  # 默认

# ============================================================================
# 主函数
# ============================================================================

def run_smart_config():
    """智能配置主函数"""
    print_header("renAIssance Configuration Wizard")

    total_steps = 8

    # Step 1: 检查CPU架构
    print_step(1, total_steps, "Checking CPU architecture...")
    sys_info = detect_system()
    print_ok(f"Architecture: {sys_info['arch']}")

    # Step 2: 检查操作系统
    print_step(2, total_steps, "Checking operating system...")
    print_ok(f"OS: {sys_info['os']}")

    # Step 3: 检查GPU硬件
    print_step(3, total_steps, "Detecting GPU hardware...")
    gpu_info = detect_gpu()
    if gpu_info["detected"]:
        version_info = f", Driver: {gpu_info['driver_version']}" if gpu_info.get("driver_version") else ""
        print_ok(f"GPU: {gpu_info['name']} (x{gpu_info['count']}){version_info}")
    else:
        print_info("No GPU detected")

    # Step 4: 检查vcpkg包管理器
    print_step(4, total_steps, "Checking vcpkg package manager...")
    vcpkg_ok = check_vcpkg()
    if not vcpkg_ok:
        print_fail("vcpkg is required but not found!")
        print_info("Please install vcpkg or set VCPKG_ROOT environment variable")
        print_info("Download from: https://github.com/microsoft/vcpkg")
        return False
    print_ok(format_dependency_output('vcpkg', None, VCPKG_ROOT))

    # Step 5: 检查C++工具链 (编译器、CMake、Ninja)
    print_step(5, total_steps, "Checking C++ toolchain...")

    # 5a: 检查MSVC (Windows) 或 GCC (Linux)
    if sys_info["is_windows"]:
        msvc_result = find_msvc()
        if not msvc_result["found"]:
            print_fail("MSVC compiler is required but not found!")
            return False
        found_toolchain = {"msvc": msvc_result}
    else:
        # Linux下检查GCC
        gcc_path = find_exe_in_path(["g++", "gcc"])
        if not gcc_path:
            print_fail("GCC compiler is required but not found!")
            return False

        # 获取GCC版本信息
        gcc_result = {"found": True, "name": "GCC", "exe_path": gcc_path}
        version_cmd = ["gcc", "--version"]
        success, output = run_cmd(version_cmd)
        if success:
            # 从输出中提取版本号
            version_match = re.search(r"gcc.*?(\d+\.\d+\.\d+)", output)
            if version_match:
                gcc_result["version"] = version_match.group(1)

        # 打印GCC结果
        version = gcc_result.get("version", "")
        path = os.path.dirname(gcc_path)
        formatted_output = format_dependency_output(gcc_result['name'], version, path)
        print_ok(formatted_output)

        found_toolchain = {"gcc": gcc_result}

    # 5b: 检查CMake和Ninja (通过依赖搜索)
    toolchain_deps = ["cmake", "ninja"]
    # 注意：MSVC和GCC已经在5a步检查过了，find_msvc函数已经打印了MSVC结果

    for dep in toolchain_deps:
        if dep in found_toolchain:
            continue  # 跳过已经查找过的依赖
        result = search_dependency(dep, sys_info, suppress_print=True)  # 抑制内部打印
        found_toolchain[dep] = result
        if result["found"]:
            version = result.get("version", "")
            path = result.get("path", "")
            formatted_output = format_dependency_output(result['name'], version, path)
            print_ok(formatted_output)
        else:
            print_fail(f"{result['name']} - NOT FOUND")

    # Step 6: 智能确定使用场景并检查相应依赖
    print_step(6, total_steps, "Determining usage scenario...")
    scene = determine_scene(sys_info, gpu_info)
    scene_info = SCENE_DEPS.get(scene)
    if not scene_info:
        print_fail(f"Unsupported scene: {scene}")
        return False

    # Step 7: 检查场景特定依赖
    if scene in ["pc_cuda", "gpu_cloud", "pc_musa"]:
        print_step(7, total_steps, "Checking GPU acceleration libraries...")
    else:
        print_step(7, total_steps, "Checking system libraries...")

    # 只检查CUDA/MUSA相关依赖（如果场景需要）
    found_cuda = {}
    if scene in ["pc_cuda", "gpu_cloud", "pc_musa"]:
        if scene in ["pc_cuda", "gpu_cloud"]:
            cuda_deps = ["cuda", "cudnn"]
            if scene == "gpu_cloud":
                cuda_deps.append("nccl")  # GPU云服务器需要NCCL
        elif scene == "pc_musa":
            cuda_deps = ["musa", "mudnn"]

        for dep in cuda_deps:
            result = search_dependency(dep, sys_info)
            found_cuda[dep] = result
            if result["found"]:
                version = result.get("version", "")
                path = result.get("path", "")
                formatted_output = format_dependency_output(result['name'], version, path)
                print_ok(formatted_output)
            else:
                print_fail(f"{result['name']} - NOT FOUND")

    # Step 8: 检查其他库
    print_step(8, total_steps, "Checking other libraries...")

    # 搜索所有依赖
    all_found = {**found_toolchain, **found_cuda}
    remaining_deps = [dep for dep in scene_info["required"] if dep not in all_found]

    found_other = {}
    for dep in remaining_deps:
        result = search_dependency(dep, sys_info)
        found_other[dep] = result

        # 保存Python依赖到全局变量，供NumPy检测使用
        if dep == "python" and result["found"]:
            global_deps["python"] = result

        if result["found"]:
            version = result.get("version", "")
            path = result.get("path", "")
            formatted_output = format_dependency_output(result['name'], version, path)
            print_ok(formatted_output)
        else:
            print_fail(f"{result['name']} - NOT FOUND")

    # 合并所有找到的依赖
    all_deps = {**all_found, **found_other}
    missing_required = check_required_deps(scene, all_deps)

    if missing_required:
        print_fail("Configuration incomplete.")
        print_info("Please install the missing components and run 'python configure.py' again. We are waiting for you.")
        print("\nRequired components missing:")
        for dep in missing_required:
            config = DEP_CONFIG.get(dep, {})
            name = config.get("name", dep)
            min_version = config.get("min_version")
            version_str = f" (version {min_version}+)" if min_version else ""
            hint = config.get("install_hint", "Please check documentation")
            print(f"  • {name}{version_str}: {hint}")

            # 显示详细安装指南（如果有的话）
            from dependency_data import DETAILED_INSTALL_GUIDES
            if dep in DETAILED_INSTALL_GUIDES:
                guide = DETAILED_INSTALL_GUIDES[dep]
                print(f"\n  [Detailed Guide] {guide['title']}:")
                for step in guide['steps']:
                    print(f"    {step}")
                if guide.get('notes'):
                    print(f"    [Important Notes]:")
                    for note in guide['notes']:
                        print(f"      {note}")
                print()  # 空行分隔

        return False

    # Step 8: 生成配置文件
    print_step(8, total_steps, "Generating configuration files...")
    print_ok(f"Build Directory: {BUILD_DIR}")

    generate_config_files(scene, all_deps, sys_info)

    # 完成提示
    print_colored("\n==============================================", Colors.BOLD)
    print_colored("  Configuration Completed! Congratulations!", Colors.BOLD)
    print_colored("==============================================", Colors.BOLD)

    return True

