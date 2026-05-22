#!/usr/bin/env python3
"""
run_search.py - 自动化执行 ResNet-50 各层的穷举式搜索

读取 resnet50_conv.json，对每层运行四大搜索器（fprop/genstats/dgrad/wgrad）
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# 常量定义
JSON_CONFIG_PATH = Path("/root/epfs/R/renaissance/scripts/resnet50_conv.json")
SEARCHER_DIR = Path("/root/epfs/R/renaissance/build/bin/tests/search")
GENERATED_DIR = Path("/root/epfs/R/renaissance/include/generated")

# 四大搜索器（使用简短名称用于显示）
SEARCHERS = [
    ("search_conv_fprop", "fprop"),
    ("search_conv_genstats", "genstats"),
    ("search_conv_dgrad", "dgrad"),
    ("search_conv_wgrad", "wgrad")
]

# 全局变量：保存每个 experience 文件的当前内容
# 格式：{short_name: file_content}
experience_backups: Dict[str, str] = {}


def detect_gpu_platform() -> str:
    """
    检测当前GPU平台

    Returns:
        GPU平台名称: RTX5090, A100-SXM4-80GB, A100-SXM4-40GB, 或 UNKNOWN_GPU
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        # 只取第一个GPU的名称（多GPU环境）
        gpu_name = result.stdout.strip().split('\n')[0].strip()

        # 标准化命名（与搜索器保持一致）
        if "A100" in gpu_name:
            # 检查显存大小
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            # 只取第一个GPU的显存（多GPU环境）
            memory_mb = int(result.stdout.strip().split('\n')[0].strip())
            if memory_mb >= 79 * 1024:  # 约80GB
                return "A100-SXM4-80GB"
            else:
                return "A100-SXM4-40GB"
        elif "RTX 5090" in gpu_name or "5090" in gpu_name:
            return "RTX5090"
        else:
            return "UNKNOWN_GPU"
    except Exception as e:
        print(f"Warning: Failed to detect GPU: {e}", file=sys.stderr)
        return "UNKNOWN_GPU"


def get_experience_filename(short_name: str, gpu_name: str, dtype: str) -> Path:
    """
    获取 experience 文件名

    Args:
        short_name: 搜索器简短名称 (fprop/genstats/dgrad/wgrad)
        gpu_name: GPU 平台名称
        dtype: 数据类型 (fp16/bf16)

    Returns:
        experience 文件路径
    """
    filename = f"experience_conv_{short_name}_{gpu_name}_{dtype}.txt"
    return GENERATED_DIR / filename


def backup_experience_file(short_name: str, gpu_name: str, dtype: str) -> None:
    """
    备份 experience 文件内容到全局字典

    Args:
        short_name: 搜索器简短名称
        gpu_name: GPU 平台名称
        dtype: 数据类型
    """
    exp_file = get_experience_filename(short_name, gpu_name, dtype)

    if exp_file.exists():
        with open(exp_file, 'r') as f:
            experience_backups[short_name] = f.read()
    else:
        experience_backups[short_name] = ""


def restore_experience_file(short_name: str, gpu_name: str, dtype: str) -> None:
    """
    从备份恢复 experience 文件

    Args:
        short_name: 搜索器简短名称
        gpu_name: GPU 平台名称
        dtype: 数据类型
    """
    exp_file = get_experience_filename(short_name, gpu_name, dtype)

    if short_name in experience_backups:
        with open(exp_file, 'w') as f:
            f.write(experience_backups[short_name])


def clear_old_experience(dtype: str) -> None:
    """
    删除旧的经验文件

    Args:
        dtype: 数据类型 (fp16/bf16)
    """
    pattern = f"*{dtype}.txt"
    files = list(GENERATED_DIR.glob(pattern))

    if files:
        print(f"[CLEAN] Deleting {len(files)} old experience files: {pattern}")
        for file in files:
            file.unlink()
        print(f"[CLEAN] Cleanup completed")
    else:
        print(f"[CLEAN] No old experience files found for {dtype}")


def load_configurations(json_path: Path) -> List[Dict]:
    """
    加载 ResNet-50 卷积配置

    Args:
        json_path: JSON 配置文件路径

    Returns:
        配置字典列表
    """
    with open(json_path, 'r') as f:
        configs = json.load(f)
    return configs


def run_searcher(searcher_name: str, short_name: str, dtype: str, batch_size: int,
                 input_size: int, in_channels: int, out_channels: int,
                 kernel_size: int, conv_stride: int, padding: int, gpu_name: str) -> Tuple[bool, str]:
    """
    运行单个搜索器

    Args:
        searcher_name: 搜索器完整名称
        short_name: 搜索器简短名称
        dtype: 数据类型 (fp16/bf16)
        batch_size: 批大小
        input_size: 输入特征图大小
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        conv_stride: 卷积步长
        padding: 填充大小
        gpu_name: GPU 平台名称

    Returns:
        (成功状态, 错误信息)
    """
    binary_path = SEARCHER_DIR / searcher_name

    cmd = [
        str(binary_path),
        "--dtype", dtype,
        "--n", str(batch_size),
        "--h", str(input_size),
        "--w", str(input_size),
        "--c", str(in_channels),
        "--k", str(out_channels),
        "--r", str(kernel_size),
        "--s", str(kernel_size),
        "--stride", str(conv_stride),
        "--padding", str(padding)
    ]

    # 步骤1: 备份当前的 experience 文件
    backup_experience_file(short_name, gpu_name, dtype)

    try:
        # 步骤2: 运行搜索器
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minutes timeout per searcher
        )
        # 步骤3: 成功后更新备份（隐式完成，因为文件已被搜索器更新）
        return (True, "")
    except subprocess.TimeoutExpired:
        error_msg = f"{short_name} timed out after 10 minutes"
        # 恢复文件
        restore_experience_file(short_name, gpu_name, dtype)
        return (False, error_msg)
    except subprocess.CalledProcessError as e:
        # 提取错误信息的关键部分
        stderr_lines = e.stderr.strip().split('\n')
        # 查找包含 "FATAL ERROR" 或 "CUDA error" 的行
        for line in stderr_lines:
            if "FATAL ERROR" in line or "CUDA error" in line:
                parts = line.split('-')
                if len(parts) > 1:
                    error_msg = f"{short_name}: {parts[-1].strip()}"
                    # 恢复文件
                    restore_experience_file(short_name, gpu_name, dtype)
                    return (False, error_msg)
        # 如果没找到，返回通用错误信息
        error_msg = f"{short_name} failed with exit code {e.returncode}"
        # 恢复文件
        restore_experience_file(short_name, gpu_name, dtype)
        return (False, error_msg)


def search_layer(config: Dict, batch_size: int, dtype: str, gpu_name: str) -> List[Tuple[str, str, bool, str]]:
    """
    对单个配置运行四大搜索

    Args:
        config: 配置字典
        batch_size: 批大小
        dtype: 数据类型
        gpu_name: GPU 平台名称

    Returns:
        搜索结果列表：(signature, short_name, success, error_message)
    """
    params = config["parameters"]
    signature = config['signature']
    results = []

    # 打印基本信息
    print(f"[{signature}] "
          f"C={params['in_channels']:4d} → K={params['out_channels']:4d}, "
          f"H=W={params['input_size']:3d}, "
          f"kernel={params['kernel_size']}×{params['kernel_size']}, "
          f"stride={params['conv_stride']}, "
          f"count={config['count']}")

    # 运行四大搜索器
    for searcher_name, short_name in SEARCHERS:
        success, error_msg = run_searcher(
            searcher_name=searcher_name,
            short_name=short_name,
            dtype=dtype,
            batch_size=batch_size,
            input_size=params["input_size"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            kernel_size=params["kernel_size"],
            conv_stride=params["conv_stride"],
            padding=params["padding"],
            gpu_name=gpu_name
        )

        results.append((signature, short_name, success, error_msg))

        # 打印搜索器结果
        if success:
            print(f"    [{short_name:^10}] ✓ OK")
        else:
            print(f"    [{short_name:^10}] ✗ FAILED: {error_msg}")

    return results


def print_summary(all_results: List[Tuple[str, str, bool, str]], total_configs: int) -> None:
    """
    打印搜索结果汇总

    Args:
        all_results: 所有搜索结果列表
        total_configs: 总配置数
    """
    total_searches = len(all_results)
    successful = sum(1 for _, _, success, _ in all_results if success)
    failed = total_searches - successful

    print()
    print("=" * 80)
    print("SEARCH SUMMARY")
    print("=" * 80)
    print(f"Total configurations: {total_configs}")
    print(f"Total searches: {total_searches} ({len(SEARCHERS)} searchers × {total_configs} configs)")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("FAILED SEARCHES:")
        print("-" * 80)
        print(f"{'Layer':<20} {'Searcher':<12} {'Error'}")
        print("-" * 80)

        for signature, short_name, success, error_msg in all_results:
            if not success:
                print(f"{signature:<20} {short_name:<12} {error_msg}")

        print("-" * 80)

    print()
    print(f"[NEXT] Run 'python scripts/generate_experience_hpp.py' to generate C++ headers")
    if failed > 0:
        print(f"[WARN] Some searches failed - experience tables will be incomplete")


def main():
    parser = argparse.ArgumentParser(
        description="Run exhaustive search for all ResNet-50 convolution layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s                           # 使用默认参数（fp16, batch_size=512）
  python %(prog)s --dtype bf16              # BF16 搜索
  python %(prog)s --batch_size 256          # 指定批大小
  python %(prog)s --clear                   # 清理旧经验文件后搜索
  python %(prog)s --dtype bf16 --clear      # BF16 + 清理
  python %(prog)s --debug                   # Debug 模式：只搜索第一层
  python %(prog)s --debug --clear           # Debug 模式 + 清理
        """
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for search (default: 512)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="Data type (default: fp16)"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear old experience files before searching"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only search the first layer (conv1)"
    )

    args = parser.parse_args()

    # 验证搜索器目录存在
    if not SEARCHER_DIR.exists():
        print(f"[ERROR] Searcher directory not found: {SEARCHER_DIR}", file=sys.stderr)
        print("[ERROR] Please build the searchers first", file=sys.stderr)
        sys.exit(1)

    # 验证搜索器文件存在
    missing_searchers = []
    for searcher_name, _ in SEARCHERS:
        binary_path = SEARCHER_DIR / searcher_name
        if not binary_path.exists():
            missing_searchers.append(searcher_name)

    if missing_searchers:
        print(f"[ERROR] Missing searchers: {', '.join(missing_searchers)}", file=sys.stderr)
        sys.exit(1)

    # 检测 GPU 平台
    gpu_name = detect_gpu_platform()
    print(f"[GPU] Detected platform: {gpu_name}")

    # 清理旧经验文件
    if args.clear:
        clear_old_experience(args.dtype)

    # 加载配置
    configurations = load_configurations(JSON_CONFIG_PATH)

    # Debug 模式：只搜索第一层
    if args.debug:
        configurations = [configurations[0]]
        print(f"[DEBUG] Debug mode enabled: only searching first layer ({configurations[0]['signature']})")

    print(f"[LOAD] Loaded {len(configurations)} configurations from {JSON_CONFIG_PATH}")
    print(f"[CONFIG] batch_size={args.batch_size}, dtype={args.dtype}")
    print(f"[START] Running {len(SEARCHERS)} searchers for each configuration")
    print(f"[INFO] Experience files will be protected on failure")
    print()

    # 收集所有搜索结果
    all_results = []

    # 对每个配置运行搜索
    for idx, config in enumerate(configurations, 1):
        print(f"[{idx}/{len(configurations)}] ", end="", flush=True)
        results = search_layer(config, args.batch_size, args.dtype, gpu_name)
        all_results.extend(results)
        print()  # 空行分隔不同层

    # 打印汇总
    print_summary(all_results, len(configurations))


if __name__ == "__main__":
    main()
