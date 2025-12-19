#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file smart_config.py
@brief 技术觉醒深度学习框架 V3.0.4 智能配置模块
@details 系统检测、场景判断、依赖搜索、配置文件生成的核心逻辑
@version 3.00.04
@date 2025-01-01
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

# 导入配置变量（来自根目录的configure.py）
try:
    from configure import VCPKG_ROOT, BUILD_DIR, PARALLEL_JOBS, CMAKE_GENERATOR, CMAKE_BUILD_TYPE, VERBOSE_OUTPUT
except ImportError:
    # 默认配置
    VCPKG_ROOT = os.environ.get("VCPKG_ROOT", "T:/Softwares/vcpkg")
    BUILD_DIR = "build/cmake-build-release"
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

    print_colored(f"\n{'='*60}", Colors.BOLD)
    print_colored(f"  {title}", Colors.BOLD)
    print_colored(f"  renAIssance Deep Learning Framework v{version}", Colors.BOLD)
    print_colored(f"{'='*60}", Colors.BOLD)

def print_step(step: int, total: int, msg: str):
    """打印步骤"""
    print_colored(f"\n[Step {step}/{total}] {msg}", Colors.BLUE)

def print_ok(msg: str):
    """打印成功信息"""
    print_colored(f"[OK] {msg}", Colors.OK)

def print_fail(msg: str):
    """打印失败信息"""
    print_colored(f"[FAIL] {msg}", Colors.FAIL)

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
        version_str = f" v{option['version']}" if option.get("version") else ""
        path_str = f" ({option['path']})" if option.get("path") else ""
        print(f"  {i}. {option['name']}{version_str}{path_str}")

    # 添加"其他"选项
    print(f"  {len(options) + 1}. Other (enter custom path)")
    print()

    while True:
        try:
            choice = input(f"  Please select [1-{len(options) + 1}]: ").strip()
            choice_num = int(choice)

            if 1 <= choice_num <= len(options):
                selected = options[choice_num - 1]
                print_ok(f"Selected: {selected['name']} v{selected.get('version', 'unknown')}")
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
            # 非交互模式，默认选择第一个选项
            print_info(f"Non-interactive mode: selecting first {options[0]['name']} installation")
            selected = options[0]
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

def extract_version_from_path(path: str, dep_name: str) -> str:
    """通过执行--version命令从路径获取准确版本信息"""
    if not path:
        return "unknown"

    # 获取对应的依赖配置
    config = DEP_CONFIG.get(dep_name.lower())
    if not config:
        return "unknown"

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
    """检测GPU信息"""
    gpu_info = {"type": None, "name": None, "count": 0, "detected": False}

    # 检测NVIDIA GPU
    success, output = run_cmd(["nvidia-smi", "--query-gpu=name,count", "--format=csv,noheader"])
    if success:
        lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
        if lines:
            gpu_info["type"] = "nvidia"
            gpu_info["name"] = lines[0].split(',')[0].strip()
            gpu_info["count"] = len(lines)
            gpu_info["detected"] = True
            return gpu_info

    # 检测摩尔线程GPU
    success, output = run_cmd(["mthreads-gmi"])
    if success and "MTT" in output:
        gpu_info["type"] = "mthreads"  # 统一使用mthreads作为类型标识
        gpu_info["name"] = "Moore Threads GPU"
        gpu_info["count"] = 1
        gpu_info["detected"] = True
        return gpu_info

    return gpu_info

# ============================================================================
# 场景判断（严格遵循文档【二十三】）
# ============================================================================

def determine_scene(sys_info: Dict, gpu_info: Dict) -> str:
    """根据系统和GPU信息判断使用场景"""

    # 1. ARM/RISC-V → 嵌入式
    if sys_info["arch"] in ["arm64", "aarch64", "riscv"]:
        if not sys_info["is_linux"]:
            print_fail("Embedded devices must run Linux!")
            sys.exit(1)
        return "embedded"

    # 2. 非x86架构报错
    if sys_info["arch"] != "x86_64":
        print_fail(f"Unsupported architecture: {sys_info['arch']}")
        sys.exit(1)

    # 3. Windows → PC-CUDA（唯一选择）
    if sys_info["is_windows"]:
        return "pc_cuda_win"

    # 4. Linux下智能判断GPU使用
    if gpu_info["detected"]:
        # 检测到GPU硬件，直接使用，无需询问
        if gpu_info["type"] == "nvidia":
            # 根据GPU数量选择场景
            if gpu_info["count"] > 1:
                return "gpu_cloud"
            else:
                return "pc_cuda_linux"
        elif gpu_info["type"] == "mthreads":
            return "pc_musa"

    else:
        # x86架构但没检测到GPU，询问用户是否有GPU但检测不到
        try:
            print("\nNo GPU detected. Do you have a GPU that wasn't detected? (Y/N): ", end="")
            has_gpu = input().strip().upper()
        except EOFError:
            print_info("Non-interactive mode: continuing with CPU-only mode")
            has_gpu = "N"

        if has_gpu == "Y":
            print_warn("GPU not detected. Please install GPU drivers first:")
            print_info("  - For NVIDIA: https://developer.nvidia.com/cuda-downloads")
            print_info("  - For Moore Threads: https://www.mthreads.com/download")
            print_fail("Please install GPU drivers and run configure.py again")
            sys.exit(1)
        else:
            return "cpu_cloud"

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
        success, output = run_cmd(config["check_cmd"])
        if success:
            version = output.strip()
            if not min_version or compare_version(version, min_version) >= 0:
                # 获取Python可执行文件路径
                python_exe = find_exe_in_path(config["exe"])
                python_path = os.path.dirname(python_exe) if python_exe else "system PATH"

                versions.append({
                    "name": config["name"],
                    "path": python_path,
                    "version": version,
                    "from_vcpkg": False,
                    "exe_path": python_exe
                })
        return versions

    # 搜索所有可能的版本
    paths = config.get("paths_win" if is_win else "paths_linux", [])

    for path_pattern in paths:
        # 展开通配符
        if "*" in path_pattern:
            import glob
            matches = glob.glob(path_pattern)
        else:
            matches = [path_pattern] if os.path.exists(path_pattern) else []

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
                    # 也尝试在bin子目录中查找
                    bin_exe_path = os.path.join(path, "bin", exe)
                    if os.path.exists(bin_exe_path):
                        exe_path = bin_exe_path
                        break

                if exe_path:
                    # 使用改进的版本提取方法
                    version = extract_version_from_path(os.path.dirname(exe_path), name)
                    if version != "unknown" and (not min_version or compare_version(version, min_version) >= 0):
                        versions.append({
                            "name": config["name"],
                            "path": os.path.dirname(os.path.dirname(exe_path)),
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
                    if version != "unknown" and (not min_version or compare_version(version, min_version) >= 0):
                        versions.append({
                            "name": config["name"],
                            "path": path,
                            "version": version,
                            "from_vcpkg": False
                        })

    # vcpkg版本
    if VCPKG_ROOT and config.get("vcpkg_packages"):
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

def search_dependency(name: str, sys_info: Dict) -> Dict:
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
                    print_ok(f"Found {config['name']}: {versions[0]['name']} v{versions[0].get('version', 'unknown')}")
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

            # 3. 根据不同平台尝试从常见路径查找Python
            if not python_exe:
                if sys_info.get("is_linux"):
                    python_paths = [
                        "/home/tech-renaissance/venv/py314/bin/python3.14",
                        "/home/ubuntu/venv/py314/bin/python3.14",
                        "/usr/bin/python3",
                        "python3"
                    ]
                elif sys_info.get("is_windows"):
                    python_paths = [
                        "python",
                        "python3",
                        "py",
                        # 尝试常见的Windows Python安装路径
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
                cmd = [python_exe, "-c", "import numpy; print(numpy.__version__)"]
                success, output = run_cmd(cmd)
                if success:
                    result["found"] = True
                    result["version"] = output.strip()
                    result["path"] = os.path.dirname(python_exe)  # 保存Python路径
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
            result["found"] = True
            result["version"] = output.strip()
            if "min_version" in config and compare_version(result["version"], config["min_version"]) < 0:
                result["found"] = False
                result["error"] = f"Version {result['version']} < {config['min_version']}"
        return result

    # 第一层：vcpkg搜索 (最高优先级)
    if VCPKG_ROOT and config.get("vcpkg_packages"):
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

    # 版本检测
    if result["found"] and "version_cmd" in config:
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

            # 版本检查
            if result["version"] and "min_version" in config:
                if compare_version(result["version"], config["min_version"]) < 0:
                    result["found"] = False
                    result["error"] = f"Version {result['version']} < {config['min_version']}"

    return result

def search_in_vcpkg(name: str, config: Dict, is_win: bool) -> Dict:
    """在vcpkg中搜索依赖"""
    if not VCPKG_ROOT or not os.path.exists(VCPKG_ROOT):
        return {"found": False}

    triplet = "x64-windows" if is_win else "x64-linux"
    vcpkg_installed = os.path.join(VCPKG_ROOT, "installed", triplet)

    if not os.path.exists(vcpkg_installed):
        return {"found": False}

    # 检查头文件
    if "header" in config:
        headers = config["header"] if isinstance(config["header"], list) else [config["header"]]
        for header in headers:
            header_path = os.path.join(vcpkg_installed, "include", header)
            if os.path.exists(header_path):
                return {
                    "found": True,
                    "name": config["name"],
                    "path": vcpkg_installed,
                    "from_vcpkg": True
                }

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
            ver = f" v{result['version']}" if result.get("version") else ""
            path = result.get("path")
            if not path or path == "system":
                # 尝试获取可执行文件路径作为路径信息
                if result.get("exe_path"):
                    path = os.path.dirname(result["exe_path"])
                else:
                    path = "system PATH"
            # 如果路径为None或者等于"system PATH"，就不显示路径
            if path and path != "system PATH":
                print_ok(f"{result['name']}{ver} ({status}) - {path}")
            else:
                print_ok(f"{result['name']}{ver} ({status})")
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
    """生成CMake配置文件内容"""
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

    # Python配置
    if "python" in deps and deps["python"]["found"]:
        python_exe = find_exe_in_path(["python3", "python", "python.exe"])
        if python_exe:
            lines.append(f'set(Python3_EXECUTABLE "{python_exe.replace("\\\\", "/")}" )')

    # MSVC配置 - 使用Alpha编译标准路径
    if "msvc" in deps and deps["msvc"]["found"]:
        msvc_path = deps["msvc"]["path"].replace("\\", "/")
        # 查找vcvars64.bat - Alpha编译标准路径
        vs_root = msvc_path.split("/VC/Tools/MSVC")[0] if "/VC/Tools/MSVC" in msvc_path else msvc_path
        vcvars = os.path.join(vs_root, "VC/Auxiliary/Build/vcvars64.bat")
        if os.path.exists(vcvars):
            lines.append(f'set(VCVARS_PATH "{vcvars.replace("\\\\", "/")}" )')

        # cl.exe路径 - Alpha编译优化
        cl_path = os.path.join(msvc_path, "cl.exe").replace("\\\\", "/")
        if os.path.exists(cl_path):
            lines.append(f'set(cl_path "{cl_path}")')

    lines.append("")

    # CMake选项
    for opt in SCENE_DEPS[scene]["cmake_opts"]:
        key, val = opt.split("=")
        lines.append(f'set({key.replace("-D", "")} {val})')

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
        # Windows脚本 - 基于alpha编译成功经验
        vcvars_path = ""
        cl_path = ""
        if "msvc" in deps and deps["msvc"]["found"]:
            msvc_path = deps["msvc"]["path"]
            vs_root = msvc_path.split("\\VC\\Tools\\MSVC")[0] if "\\VC\\Tools\\MSVC" in msvc_path else msvc_path
            vcvars_path = os.path.join(vs_root, "VC\\Auxiliary\\Build\\vcvars64.bat")
            if os.path.exists(vcvars_path):
                vcvars_path = vcvars_path.replace("\\", "/")
            cl_path = os.path.join(msvc_path, "cl.exe").replace("\\", "/")

        script = f'''@echo off
REM ================================================================
REM renAIssance Framework - Alpha Build Script (Windows)
REM Generated by configure.py - High Performance Build
REM ================================================================

echo [INFO] renAIssance Framework Alpha Build
echo [INFO] Build Directory: {BUILD_DIR}
echo [INFO] Parallel Jobs: {PARALLEL_JOBS}

REM 设置环境变量
set CMAKE="T:\\Softwares\\CMake\\bin\\cmake.exe"
set NINJA="B:\\Softwares\\JetBrains\\CLion 2025.2\\bin\\ninja\\win\\x64\\ninja.exe"

REM 设置构建目录
if not exist {BUILD_DIR} mkdir {BUILD_DIR}
cd {BUILD_DIR}

REM 配置CMake (Alpha编译标准)
echo [INFO] Configuring with CMake...
%CMAKE% ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_TOOLCHAIN_FILE="{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ^
    {cmake_opts} ^
    ../..

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed
    exit /b 1
)

REM 编译项目
echo [INFO] Building with Ninja...
%CMAKE% --build . --parallel {PARALLEL_JOBS}

if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    exit /b 1
)

echo [OK] Build completed successfully!
echo [INFO] Build artifacts are in: {BUILD_DIR}
echo [INFO] Run tests: %cd%\\bin\\tests\\*.exe
cd ../..
'''
        return "build.bat", script

    else:
        # Linux脚本
        script = f'''#!/bin/bash
# ================================================================
# renAIssance Framework - Build Script (Linux)
# Generated by configure.py
# ================================================================

echo "[INFO] renAIssance Framework Build"
echo "[INFO] Build Directory: {BUILD_DIR}"
echo "[INFO] Parallel Jobs: {PARALLEL_JOBS}"

# 设置构建目录
mkdir -p {BUILD_DIR}
cd {BUILD_DIR}

# 配置CMake
echo "[INFO] Configuring with CMake..."
cmake ../.. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_TOOLCHAIN_FILE="{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \
    {cmake_opts}

if [ $? -ne 0 ]; then
    echo "[ERROR] CMake configuration failed"
    exit 1
fi

# 编译项目
echo "[INFO] Building with Ninja..."
cmake --build . --parallel {PARALLEL_JOBS}

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed"
    exit 1
fi

echo "[OK] Build completed successfully!"
echo "[INFO] Build artifacts are in: {BUILD_DIR}"
echo "[INFO] Run tests: $PWD/bin/tests/*.exe"
cd ../..
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
        "dependencies": {k: {
            "found": v["found"],
            "path": v.get("path"),
            "version": v.get("version"),
            "from_vcpkg": v.get("from_vcpkg", False)
        } for k, v in found_deps.items()},
        "cmake_options": SCENE_DEPS[scene]["cmake_opts"]
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
        print_ok(f"Found MSVC: {known_path}")
        return {
            "found": True,
            "name": "MSVC",
            "path": os.path.dirname(os.path.dirname(known_path)),
            "exe_path": known_path,
            "version": "14.44.35207",
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
            print_ok(f"Found MSVC: {latest_cl}")
            return {
                "found": True,
                "name": "MSVC",
                "path": os.path.dirname(os.path.dirname(latest_cl)),
                "exe_path": latest_cl,
                "version": version,
                "from_vcpkg": False
            }

    # 3. 在PATH中查找
    cl_path = find_exe_in_path(["cl.exe"])
    if cl_path:
        print_ok(f"Found MSVC in PATH: {cl_path}")
        return {
            "found": True,
            "name": "MSVC",
            "path": os.path.dirname(os.path.dirname(cl_path)),
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
    if gpu_info["type"]:
        print_ok(f"GPU: {gpu_info['name']} (x{gpu_info['count']})")
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
    print_ok(f"vcpkg: {VCPKG_ROOT}")

    # Step 5: 检查C++工具链 (编译器、CMake、Ninja)
    print_step(5, total_steps, "Checking C++ toolchain...")

    # 5a: 检查MSVC (Windows) 或 GCC (Linux)
    if sys_info["is_windows"]:
        msvc_result = find_msvc()
        if not msvc_result["found"]:
            print_fail("MSVC compiler is required but not found!")
            return False
    else:
        # Linux下检查GCC
        gcc_path = find_exe_in_path(["g++", "gcc"])
        if not gcc_path:
            print_fail("GCC compiler is required but not found!")
            return False
        print_ok(f"GCC: {gcc_path}")

    # 5b: 检查CMake和Ninja (通过依赖搜索)
    toolchain_deps = ["cmake", "ninja"]
    # 注意：MSVC和GCC已经在5a步检查过了，不需要重复检查

    found_toolchain = {}
    if sys_info["is_windows"]:
        found_toolchain["msvc"] = msvc_result
        # 显示MSVC结果
        ver = f" v{msvc_result['version']}" if msvc_result.get("version") else ""
        path = msvc_result.get("path", "system PATH")
        if path and path != "system PATH":
            print_ok(f"{msvc_result['name']}{ver} - {path}")
        else:
            print_ok(f"{msvc_result['name']}{ver}")
    else:
        found_toolchain["gcc"] = {"found": True, "name": "GCC", "exe_path": gcc_path}

    for dep in toolchain_deps:
        result = search_dependency(dep, sys_info)
        found_toolchain[dep] = result
        if result["found"]:
            ver = f" v{result['version']}" if result.get("version") else ""
            path = result.get("path", "system PATH")
            # 如果路径为None或者等于"system PATH"，就不显示路径
            if path and path != "system PATH":
                print_ok(f"{result['name']}{ver} - {path}")
            else:
                print_ok(f"{result['name']}{ver}")
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
    print_step(7, total_steps, "Checking GPU acceleration libraries...")

    # 只检查CUDA/MUSA相关依赖（如果场景需要）
    found_cuda = {}
    if scene in ["pc_cuda_linux", "gpu_cloud", "pc_musa"]:
        if scene in ["pc_cuda_linux", "gpu_cloud"]:
            cuda_deps = ["cuda", "cudnn"]
        elif scene == "pc_musa":
            cuda_deps = ["musa", "mudnn"]

        for dep in cuda_deps:
            result = search_dependency(dep, sys_info)
            found_cuda[dep] = result
            if result["found"]:
                ver = f" v{result['version']}" if result.get("version") else ""
                path = result.get("path", "system")
                # 如果路径为None或者等于"system"，就不显示路径
                if path and path != "system":
                    print_ok(f"{result['name']}{ver} - {path}")
                else:
                    print_ok(f"{result['name']}{ver}")
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
            ver = f" v{result['version']}" if result.get("version") else ""
            path = result.get("path", "system")
            # 如果路径为None或者等于"system"，就不显示路径
            if path and path != "system":
                print_ok(f"{result['name']}{ver} - {path}")
            else:
                print_ok(f"{result['name']}{ver}")
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

