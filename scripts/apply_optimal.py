#!/usr/bin/env python3
"""
apply_optimal.py - 将 JSON 中的最优引擎应用到 experience *.txt 文件

功能：
  1. 读取 scripts/optimal_engines_{GPU}_{dtype}.json
  2. 匹配 JSON 配置与 include/generated/*.txt 中的记录
  3. 将 JSON 中的最优引擎替换到 txt 文件的 WINNER_TAG（最左边的引擎名）
  4. 输出匹配统计

作者：技术觉醒团队
版本：1.0.0
日期：2026-04-11
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

# 常量定义
SCRIPTS_DIR = Path("/root/epfs/R/renaissance/scripts")
GENERATED_DIR = Path("/root/epfs/R/renaissance/include/generated")

# 操作类型映射（JSON → TXT）
OP_TYPE_MAPPING = {
    "conv_genstats": "conv_genstats",
    "dgrad": "conv_dgrad",
    "wgrad": "conv_wgrad",
}


def detect_gpu_platform() -> str:
    """
    检测当前 GPU 平台

    Returns:
        GPU 平台名称: RTX5090, A100-SXM4-80GB, A100-SXM4-40GB, 或 UNKNOWN_GPU
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


def get_json_path(gpu_name: str, dtype: str) -> Path:
    """
    获取 JSON 文件路径

    Args:
        gpu_name: GPU 平台名称
        dtype: 数据类型 (fp16/bf16)

    Returns:
        JSON 文件路径
    """
    filename = f"optimal_engines_{gpu_name}_{dtype}.json"
    return SCRIPTS_DIR / filename


def build_new_line_from_json(config: Dict, dtype: str, batch_size: int,
                               txt_op: str, optimal_engine: str, gpu_name: str) -> str:
    """
    从 JSON 配置构建新的 experience 记录行

    当 txt 文件中没有匹配行时，调用此函数创建新行。
    winner, backup1, backup2 都使用最优引擎。

    Args:
        config: JSON 配置字典
        dtype: 数据类型 (fp16/bf16)
        batch_size: 批大小
        txt_op: 操作类型 (conv_fprop/conv_genstats/conv_dgrad/conv_wgrad)
        optimal_engine: 最优引擎名称
        gpu_name: GPU 平台名称

    Returns:
        完整的 27 字段行字符串
    """
    params = config["parameters"]

    # 提取参数
    n = batch_size
    h = params["input_size"]
    w = params["input_size"]
    c = params["in_channels"]
    k = params["out_channels"]
    r = params["kernel_size"]
    s = params["kernel_size"]
    stride = params["conv_stride"]
    pad = params.get("padding", params.get("conv_pad", 0))
    dilation = params.get("conv_dilation", 1)

    # 构建完整的 27 字段行
    # 格式：GPU|SM|cuDNN|CUDA|op_type_dtype|N|H|W|C|K|R|S|U|V|P|Q|D|E|NHWC|dtype|FP32|WINNER|BACKUP1|BACKUP2|WS|TIME|SOURCE

    # 获取 SM 和 cuDNN 版本（从现有 txt 文件推断，或使用默认值）
    if "A100" in gpu_name:
        sm = "SM80"
        cudnn_version = "cuDNN9.17.0"
        cuda_version = "CUDA13.1"
    elif "RTX" in gpu_name:
        sm = "SM80"
        cudnn_version = "cuDNN9.17.0"
        cuda_version = "CUDA13.1"
    else:
        sm = "SM80"
        cudnn_version = "cuDNN9.17.0"
        cuda_version = "CUDA13.1"

    op_type_dtype = f"{txt_op}_{dtype}"

    # 构建字段
    fields = [
        gpu_name,           # 0: GPU
        sm,                 # 1: SM
        cudnn_version,      # 2: cuDNN
        cuda_version,       # 3: CUDA
        op_type_dtype,      # 4: op_type_dtype
        f"N{n}",            # 5: N
        f"H{h}",            # 6: H
        f"W{w}",            # 7: W
        f"C{c}",            # 8: C
        f"K{k}",            # 9: K
        f"R{r}",            # 10: R
        f"S{s}",            # 11: S
        f"U{dilation}",     # 12: U (dilation_h)
        f"V{dilation}",     # 13: V (dilation_w)
        f"P{pad}",          # 14: P (padding_h)
        f"Q{pad}",          # 15: Q (padding_w)
        f"D{stride}",       # 16: D (stride_h)
        f"E{stride}",       # 17: E (stride_w)
        "NHWC",             # 18: layout
        dtype.upper(),      # 19: dtype
        "FP32",             # 20: compute precision
        optimal_engine,     # 21: WINNER_TAG
        optimal_engine,     # 22: BACKUP1_TAG (使用最优引擎)
        optimal_engine,     # 23: BACKUP2_TAG (使用最优引擎)
        "0",                # 24: workspace_bytes (未知，设为 0)
        "0.0",              # 25: benchmark_time_ms (未知，设为 0)
        "heur_a"            # 26: SOURCE (标识为启发式)
    ]

    return "|".join(fields)


def get_txt_path(op_type: str, gpu_name: str, dtype: str) -> Path:
    """
    获取 experience txt 文件路径

    Args:
        op_type: 操作类型 (conv_fprop/conv_genstats/conv_dgrad/conv_wgrad)
        gpu_name: GPU 平台名称
        dtype: 数据类型 (fp16/bf16)

    Returns:
        txt 文件路径
    """
    filename = f"experience_{op_type}_{gpu_name}_{dtype}.txt"
    return GENERATED_DIR / filename


def build_match_key_from_json(config: Dict, dtype: str, batch_size: int) -> str:
    """
    从 JSON 配置构建匹配键（用于与 txt 文件记录匹配）

    只匹配输入相关参数：N, H, W, C, K, R, S, padding, stride, dilation
    不匹配输出尺寸

    Args:
        config: JSON 配置字典
        dtype: 数据类型
        batch_size: 批大小

    Returns:
        匹配键字符串
    """
    params = config["parameters"]

    # 提取参数
    n = batch_size
    h = params["input_size"]
    w = params["input_size"]
    c = params["in_channels"]
    k = params["out_channels"]
    r = params["kernel_size"]
    s = params["kernel_size"]
    stride = params["conv_stride"]
    pad = params.get("padding", params.get("conv_pad", 0))  # 兼容两种命名
    dilation = params.get("conv_dilation", 1)  # 默认为 1（ResNet-50 通常使用 dilation=1）

    # 构建 key（只匹配输入相关的字段）
    # txt 文件字段顺序（从字段 6 开始）：
    # N|H|W|C|K|R|S|U|V|P|Q|D|E|NHWC|dtype|FP32
    key = f"N{n}|H{h}|W{w}|C{c}|K{k}|R{r}|S{s}|U{dilation}|V{dilation}|P{pad}|Q{pad}|D{stride}|E{stride}|NHWC|{dtype.upper()}|FP32"

    return key


def find_matching_line(lines: list, match_key: str, op_type: str) -> Tuple[int, Optional[str]]:
    """
    在 txt 文件中查找匹配的行

    Args:
        lines: txt 文件的所有行
        match_key: 匹配键
        op_type: 操作类型（用于验证）

    Returns:
        (行索引, 原始行内容) 或 (-1, None)
    """
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # 分割字段
        parts = line.split('|')
        if len(parts) != 27:
            continue

        # 提取形状部分（字段 5-20，即 N512 到 FP32）
        # txt 文件字段索引（0-indexed）：
        # 0-4: GPU|SM|cuDNN|CUDA|op_type
        # 5: N (batch size)
        # 6-7: H, W
        # 8-9: C, K
        # 10-11: R, S (kernel)
        # 12-13: U, V (dilation)
        # 14-15: P, Q (padding)
        # 16-17: D, E (stride)
        # 18: NHWC
        # 19: dtype (FP16/BF16)
        # 20: FP32 (compute precision)
        shape_part = '|'.join(parts[5:21])  # 字段 5-20（0-indexed）

        # 验证操作类型匹配
        txt_op_type = parts[4]  # 字段 5（0-indexed）
        if txt_op_type != op_type:
            continue

        # 匹配形状
        if shape_part == match_key:
            return (idx, line)

    return (-1, None)


def apply_optimal_engines(dtype: str, gpu_name: str, batch_size: int, dry_run: bool) -> None:
    """
    应用最优引擎到 txt 文件

    Args:
        dtype: 数据类型
        gpu_name: GPU 平台名称
        batch_size: 批大小（从 JSON metadata 读取）
        dry_run: 是否只显示不实际修改
    """
    # 加载 JSON 文件
    json_path = get_json_path(gpu_name, dtype)
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        sys.exit(1)

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    print(f"[LOAD] Loaded {len(json_data['results'])} configurations from {json_path}")
    print(f"[INFO] GPU: {gpu_name}, Data Type: {dtype}, Batch Size: {batch_size}")
    print()

    # 统计信息
    total_attempts = 0
    successful_matches = 0
    failed_matches = 0

    # 按操作类型分组处理
    # 需要处理的 txt 文件映射
    op_to_txt = {
        "conv_genstats": "conv_genstats",
        "dgrad": "conv_dgrad",
        "wgrad": "conv_wgrad",
    }

    # 存储需要修改的文件内容：{txt_path: [(line_idx, old_line, new_line), ...]}
    file_modifications = {}

    # 遍历所有配置
    for config in json_data["results"]:
        signature = config["signature"]
        params = config["parameters"]
        optimal_engines = config["optimal_engine"]

        # 构建匹配键
        match_key = build_match_key_from_json(config, dtype, batch_size)

        print(f"[{signature}] C={params['in_channels']:4d} → K={params['out_channels']:4d}, "
              f"H=W={params['input_size']:3d}, kernel={params['kernel_size']}×{params['kernel_size']}, "
              f"stride={params['conv_stride']}")

        # 处理每个操作类型
        for json_op, txt_op in op_to_txt.items():
            if json_op not in optimal_engines or not optimal_engines[json_op]:
                continue

            optimal_engine = optimal_engines[json_op]
            total_attempts += 1

            # 构建完整的操作类型名称
            full_op_type = f"{txt_op}_{dtype}"

            # 读取对应的 txt 文件
            txt_path = get_txt_path(txt_op, gpu_name, dtype)
            if not txt_path.exists():
                print(f"    [{json_op:^10}] ✗ TXT file not found: {txt_path.name}")
                failed_matches += 1
                continue

            # 如果文件还没有被读取过，读取并缓存
            if txt_path not in file_modifications:
                with open(txt_path, 'r') as f:
                    file_modifications[txt_path] = (f.readlines(), [])

            lines, _ = file_modifications[txt_path]

            # 查找匹配的行
            line_idx, old_line = find_matching_line(lines, match_key, full_op_type)

            if line_idx == -1:
                # 找不到匹配行，新增一行
                # 构建完整的 27 字段行
                # 格式：GPU|SM|cuDNN|CUDA|op_type_dtype|N|H|W|C|K|R|S|U|V|P|Q|D|E|NHWC|dtype|FP32|WINNER|BACKUP1|BACKUP2|WS|TIME|SOURCE
                new_line = build_new_line_from_json(config, dtype, batch_size, txt_op, optimal_engine, gpu_name)
                print(f"    [{json_op:^10}] + Added new line: {optimal_engine}")

                # 记录新增（使用特殊标记 -1 表示新增行）
                file_modifications[txt_path][1].append((-1, None, new_line.strip()))

                successful_matches += 1
                continue

            # 解析原始行
            parts = old_line.strip().split('|')
            if len(parts) != 27:
                print(f"    [{json_op:^10}] ✗ Invalid line format")
                failed_matches += 1
                continue

            old_engine = parts[21]  # WINNER_TAG（字段 22，0-indexed 为 21）

            # 替换 WINNER_TAG（字段 22，0-indexed 为 21）
            parts[21] = optimal_engine

            # 重新组合成新行
            new_line = '|'.join(parts) + '\n'

            # 记录修改
            file_modifications[txt_path][1].append((line_idx, old_line.strip(), new_line.strip()))

            # 输出结果
            if old_engine == optimal_engine:
                print(f"    [{json_op:^10}] = Already optimal: {optimal_engine}")
            else:
                print(f"    [{json_op:^10}] → Updated: {old_engine} → {optimal_engine}")

            successful_matches += 1

        print()

    # 输出汇总
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total attempts: {total_attempts}")
    print(f"Successful matches: {successful_matches}")
    print(f"Failed matches: {failed_matches}")
    print()

    # 实际写入文件
    if not dry_run and successful_matches > 0:
        print("[WRITE] Writing changes to files...")
        for txt_path, (lines, modifications) in file_modifications.items():
            if not modifications:
                continue

            # 分离新增行和修改行
            new_lines = [m for m in modifications if m[0] == -1]
            update_lines = [m for m in modifications if m[0] >= 0]

            # 先应用修改行（按索引从大到小排序，避免索引错位）
            update_lines.sort(key=lambda x: x[0], reverse=True)
            for line_idx, old_line, new_line in update_lines:
                lines[line_idx] = new_line + '\n'

            # 再追加新增行到文件末尾
            for line_idx, old_line, new_line in new_lines:
                lines.append(new_line + '\n')

            # 写回文件
            with open(txt_path, 'w') as f:
                f.writelines(lines)

            print(f"  Updated {len(update_lines)} lines, added {len(new_lines)} new lines in {txt_path.name}")

        print()
        print("[SUCCESS] All changes applied successfully!")
    elif dry_run:
        print("[DRY RUN] No files were modified (use --no-dry-run to apply changes)")


def main():
    parser = argparse.ArgumentParser(
        description="Apply optimal engines from JSON to experience txt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s                           # Dry run with detected GPU and fp16
  python %(prog)s --dtype bf16              # Dry run with bf16
  python %(prog)s --no-dry-run              # Actually apply changes
  python %(prog)s --gpu RTX5090 --dtype fp16 --no-dry-run
        """
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="Data type (default: fp16)"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU platform (auto-detect if not specified)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (read from JSON metadata if not specified)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only show what would be changed without modifying files (default: enabled)"
    )

    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually apply changes to files"
    )

    args = parser.parse_args()

    # 检测 GPU
    if args.gpu:
        gpu_name = args.gpu
    else:
        gpu_name = detect_gpu_platform()
        print(f"[GPU] Detected platform: {gpu_name}")

    if gpu_name == "UNKNOWN_GPU":
        print("[ERROR] Could not detect GPU platform. Please specify --gpu manually.")
        sys.exit(1)

    # 获取批大小
    batch_size = args.batch_size
    if batch_size is None:
        # 从 JSON metadata 读取
        json_path = get_json_path(gpu_name, args.dtype)
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            batch_size = json_data["metadata"]["batch_size"]
            print(f"[INFO] Batch size from JSON: {batch_size}")
        else:
            print(f"[ERROR] Cannot determine batch size. Please specify --batch-size or ensure JSON exists.")
            sys.exit(1)

    # 设置 dry_run
    dry_run = args.dry_run and not args.no_dry_run
    if dry_run:
        print("[DRY RUN] Showing changes without modifying files...")
    print()

    # 执行
    apply_optimal_engines(args.dtype, gpu_name, batch_size, dry_run)


if __name__ == "__main__":
    main()
