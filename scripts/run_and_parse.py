#!/usr/bin/env python3
"""
run_and_parse.py - Run test_cbr_fp16_merged/test_cbr_bf16_merged with ResNet-50 conv configurations and output optimal engines
"""

import subprocess
import sys
import re
import json
import os
from pathlib import Path
from statistics import mean, median

# 常量定义
NUM_ITERATIONS = 32
JSON_CONFIG_PATH = Path("/root/epfs/R/renaissance/scripts/resnet50_conv.json")
OUTPUT_DIR = Path("/root/epfs/R/renaissance/scripts")

# 数据类型选择：fp16 或 bf16（可通过 --dtype 参数覆盖）
DEFAULT_DATA_TYPE = "fp16"

# 调试模式：设为True时只运行一个配置并输出详细信息
DEBUG_MODE = False

# 全局变量（通过命令行参数设置）
DATA_TYPE = DEFAULT_DATA_TYPE
GPU_NAME = None


def detect_gpu_platform():
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

        # 标准化命名（与run_search.py和ta_v4_search_common.hpp保持一致）
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


def get_output_filename(dtype: str, gpu_name: str) -> Path:
    """
    生成输出文件名，格式：optimal_engines_{GPU}_{dtype}.json

    Args:
        dtype: 数据类型 (fp16/bf16)
        gpu_name: GPU平台名称

    Returns:
        输出文件路径
    """
    filename = f"optimal_engines_{gpu_name}_{dtype}.json"
    return OUTPUT_DIR / filename


def run_test(mode: str, batch_size: int, input_size: int, in_channels: int,
             out_channels: int, kernel_size: int, conv_stride: int) -> dict:
    """
    运行 test_cbr_fp16_merged 或 test_cbr_bf16_merged 并解析输出

    Args:
        mode: 搜索模式 (A/B/C)
        batch_size: 批大小
        input_size: 输入特征图大小
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        conv_stride: 卷积步长

    Returns:
        包含6个信息的字典
    """
    binary_path = Path(f"/root/epfs/R/renaissance/build/bin/tests/search/test_cbr_{DATA_TYPE}_merged")

    cmd = [
        str(binary_path),
        "--mode", mode,
        "--batch_size", str(batch_size),
        "--input_size", str(input_size),
        "--in_channels", str(in_channels),
        "--out_channels", str(out_channels),
        "--kernel_size", str(kernel_size),
        "--conv_stride", str(conv_stride)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        print(f"Error: Test timed out after 5 minutes", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Test failed with return code {e.returncode}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    return parse_output(output)


def parse_output(output: str) -> dict:
    """
    解析测试程序输出

    新格式示例:
        Conv Forward Type: Conv+GenStats
        Forward Experience: Found
        WGrad Experience: Found
        Forward Engine: eng3_k24=13
        DGrad Engine: eng57_k2=0_k5=3_k14=2
        WGrad Engine: eng62_k25=1

        Time (Unit: ms):
        Forward:  1.68483
        Backward: 1.85778
        Total:    3.54261
    """
    result = {
        "conv_genstats_engine": None,
        "dgrad_engine": None,
        "wgrad_engine": None,
        "forward_time": None,
        "backward_time": None,
        "total_time": None,
        "experience_found_count": 0  # 新增：统计找到的 Experience 数量
    }

    # 解析 Experience Found（统计 "Experience: Found" 出现的次数）
    experience_found_count = len(re.findall(r'Experience: Found', output))
    result["experience_found_count"] = experience_found_count

    # 解析引擎名称 - 新格式
    forward_match = re.search(r'Forward Engine:\s+(\S+)', output)
    if forward_match:
        result["conv_genstats_engine"] = forward_match.group(1)

    dgrad_match = re.search(r'DGrad Engine:\s+(\S+)', output)
    if dgrad_match:
        result["dgrad_engine"] = dgrad_match.group(1)

    wgrad_match = re.search(r'WGrad Engine:\s+(\S+)', output)
    if wgrad_match:
        result["wgrad_engine"] = wgrad_match.group(1)

    # 解析时间（这部分保持不变）
    time_section = re.search(r'Time \(Unit: ms\):(.*?)(?====|\Z)', output, re.DOTALL)
    if time_section:
        time_text = time_section.group(1)
        forward_match = re.search(r'Forward:\s+([\d.]+)', time_text)
        if forward_match:
            result["forward_time"] = float(forward_match.group(1))

        backward_match = re.search(r'Backward:\s+([\d.]+)', time_text)
        if backward_match:
            result["backward_time"] = float(backward_match.group(1))

        total_match = re.search(r'Total:\s+([\d.]+)', time_text)
        if total_match:
            result["total_time"] = float(total_match.group(1))

    return result


def test_parsing():
    """
    测试解析函数 - 使用示例输出验证解析逻辑
    """
    # FP16 格式
    sample_output_fp16 = """Conv Forward Type: Conv+GenStats

Forward Engine: eng3_k24=13
DGrad Engine: eng57_k2=0_k5=3_k14=2
WGrad Engine: eng62_k25=1

Time (Unit: ms):
Forward:  1.68483
Backward: 1.85778
Total:    3.54261
"""

    # BF16 格式
    sample_output_bf16 = """Conv Forward Type: Conv FProp

Forward Engine: eng58_k25=2
DGrad Engine: eng22_k2=0_k13=0_k14=3_k23=0
WGrad Engine: eng62_k25=2

Time (Unit: ms):
Forward:  1.00715
Backward:  1.86006
Total:    2.86721
"""

    print("FP16格式解析测试:")
    result_fp16 = parse_output(sample_output_fp16)
    print(f"  Forward Engine: {result_fp16['conv_genstats_engine']}")
    print(f"  DGrad Engine: {result_fp16['dgrad_engine']}")
    print(f"  WGrad Engine: {result_fp16['wgrad_engine']}")
    print(f"  Forward Time: {result_fp16['forward_time']}")
    print(f"  Backward Time: {result_fp16['backward_time']}")
    print(f"  Total Time: {result_fp16['total_time']}")

    # 验证FP16解析结果
    assert result_fp16['conv_genstats_engine'] == 'eng3_k24=13', "FP16 Forward engine解析失败"
    assert result_fp16['dgrad_engine'] == 'eng57_k2=0_k5=3_k14=2', "FP16 DGrad engine解析失败"
    assert result_fp16['wgrad_engine'] == 'eng62_k25=1', "FP16 WGrad engine解析失败"
    assert result_fp16['forward_time'] == 1.68483, "FP16 Forward time解析失败"
    assert result_fp16['backward_time'] == 1.85778, "FP16 Backward time解析失败"
    assert result_fp16['total_time'] == 3.54261, "FP16 Total time解析失败"

    print("\nBF16格式解析测试:")
    result_bf16 = parse_output(sample_output_bf16)
    print(f"  Forward Engine: {result_bf16['conv_genstats_engine']}")
    print(f"  DGrad Engine: {result_bf16['dgrad_engine']}")
    print(f"  WGrad Engine: {result_bf16['wgrad_engine']}")
    print(f"  Forward Time: {result_bf16['forward_time']}")
    print(f"  Backward Time: {result_bf16['backward_time']}")
    print(f"  Total Time: {result_bf16['total_time']}")

    # 验证BF16解析结果
    assert result_bf16['conv_genstats_engine'] == 'eng58_k25=2', "BF16 Forward engine解析失败"
    assert result_bf16['dgrad_engine'] == 'eng22_k2=0_k13=0_k14=3_k23=0', "BF16 DGrad engine解析失败"
    assert result_bf16['wgrad_engine'] == 'eng62_k25=2', "BF16 WGrad engine解析失败"
    assert result_bf16['forward_time'] == 1.00715, "BF16 Forward time解析失败"
    assert result_bf16['backward_time'] == 1.86006, "BF16 Backward time解析失败"
    assert result_bf16['total_time'] == 2.86721, "BF16 Total time解析失败"

    print("\n✓ 所有解析测试通过! (FP16和BF16格式)")


def compute_stats(results: list) -> dict:
    """
    计算统计信息

    Args:
        results: 多次运行的结果列表

    Returns:
        统计信息字典
    """
    forward_times = [r["forward_time"] for r in results]
    backward_times = [r["backward_time"] for r in results]

    # 找到前向最快的索引
    fastest_forward_idx = forward_times.index(min(forward_times))
    # 找到反向最快的索引
    fastest_backward_idx = backward_times.index(min(backward_times))

    return {
        "forward_mean": mean(forward_times),
        "forward_median": median(forward_times),
        "backward_mean": mean(backward_times),
        "backward_median": median(backward_times),
        "fastest_forward_engine": results[fastest_forward_idx]["conv_genstats_engine"],
        "fastest_backward_dgrad_engine": results[fastest_backward_idx]["dgrad_engine"],
        "fastest_backward_wgrad_engine": results[fastest_backward_idx]["wgrad_engine"],
    }


def find_optimal_engines(all_stats: dict) -> dict:
    """
    找出最优引擎

    Args:
        all_stats: 包含所有模式统计信息的字典

    Returns:
        最优引擎字典
    """
    # 1. 决定 conv+genstats 最优引擎（基于正向用时）
    optimal_conv_genstats_engine = all_stats["C"]["fastest_forward_engine"]
    optimal_forward_mean = all_stats["C"]["forward_mean"]
    optimal_forward_median = all_stats["C"]["forward_median"]

    if (optimal_forward_mean > all_stats["B"]["forward_mean"] and
        optimal_forward_median > all_stats["B"]["forward_median"]):
        optimal_conv_genstats_engine = all_stats["B"]["fastest_forward_engine"]
        optimal_forward_mean = all_stats["B"]["forward_mean"]
        optimal_forward_median = all_stats["B"]["forward_median"]

    if (optimal_forward_mean > all_stats["A"]["forward_mean"] and
        optimal_forward_median > all_stats["A"]["forward_median"]):
        optimal_conv_genstats_engine = all_stats["A"]["fastest_forward_engine"]

    # 2. 决定 dgrad、wgrad 最优引擎（基于反向用时）
    optimal_dgrad_engine = all_stats["C"]["fastest_backward_dgrad_engine"]
    optimal_wgrad_engine = all_stats["C"]["fastest_backward_wgrad_engine"]
    optimal_backward_mean = all_stats["C"]["backward_mean"]
    optimal_backward_median = all_stats["C"]["backward_median"]

    if (optimal_backward_mean > all_stats["B"]["backward_mean"] and
        optimal_backward_median > all_stats["B"]["backward_median"]):
        optimal_dgrad_engine = all_stats["B"]["fastest_backward_dgrad_engine"]
        optimal_wgrad_engine = all_stats["B"]["fastest_backward_wgrad_engine"]
        optimal_backward_mean = all_stats["B"]["backward_mean"]
        optimal_backward_median = all_stats["B"]["backward_median"]

    if (optimal_backward_mean > all_stats["A"]["backward_mean"] and
        optimal_backward_median > all_stats["A"]["backward_median"]):
        optimal_dgrad_engine = all_stats["A"]["fastest_backward_dgrad_engine"]
        optimal_wgrad_engine = all_stats["A"]["fastest_backward_wgrad_engine"]

    return {
        "conv_genstats": optimal_conv_genstats_engine,
        "dgrad": optimal_dgrad_engine,
        "wgrad": optimal_wgrad_engine,
    }


def load_configurations(json_path: Path) -> list:
    """加载 ResNet-50 卷积配置"""
    with open(json_path, 'r') as f:
        configs = json.load(f)
    return configs


def test_configuration(config: dict, batch_size: int) -> dict:
    """
    对单个配置运行测试并返回最优引擎

    Args:
        config: 配置字典
        batch_size: 批大小

    Returns:
        包含最优引擎的字典
    """
    params = config["parameters"]
    modes = ["A", "B", "C"]
    all_stats = {}

    # 收集所有模式的统计信息
    for mode in modes:
        results = []
        for i in range(NUM_ITERATIONS):
            result = run_test(
                mode=mode,
                batch_size=batch_size,
                input_size=params["input_size"],
                in_channels=params["in_channels"],
                out_channels=params["out_channels"],
                kernel_size=params["kernel_size"],
                conv_stride=params["conv_stride"]
            )

            # 第一次运行 Mode C 时，检查 Experience Found 数量
            if mode == "C" and i == 0:
                found_count = result["experience_found_count"]
                print(f"    Experience Found: [{found_count}/3]", flush=True)

                # 如果找到的经验少于2个，报错退出
                if found_count < 2:
                    print(f"\n[ERROR] Mode C found only {found_count}/3 experiences for this configuration!")
                    print(f"[ERROR] This is insufficient for reliable optimization.")
                    print(f"[ERROR] Please re-run 'python scripts/run_search.py' to perform exhaustive search.")
                    print(f"\n[DEBUG] Configuration: {config['signature']}")
                    print(f"[DEBUG] Parameters: C={params['in_channels']}, K={params['out_channels']}, "
                          f"H=W={params['input_size']}, kernel={params['kernel_size']}x{params['kernel_size']}, "
                          f"stride={params['conv_stride']}, padding={params.get('padding', 'N/A')}", flush=True)
                    sys.exit(1)

            results.append(result)

        stats = compute_stats(results)
        all_stats[mode] = stats

    # 找出最优引擎
    optimal = find_optimal_engines(all_stats)

    return {
        "signature": config["signature"],
        "count": config["count"],
        "layer_numbers": config["layer_numbers"],
        "parameters": params,
        "optimal_engine": optimal
    }


def main():
    global DATA_TYPE, NUM_ITERATIONS, GPU_NAME

    # 解析命令行参数
    if "--test" in sys.argv:
        test_parsing()
        return

    # 检测GPU平台
    GPU_NAME = detect_gpu_platform()
    print(f"Detected GPU: {GPU_NAME}", file=sys.stderr)

    # 默认值
    batch_size = 512
    data_type = DEFAULT_DATA_TYPE
    debug_mode = DEBUG_MODE
    num_iterations = NUM_ITERATIONS

    # 解析参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--iter":
            if i + 1 < len(sys.argv):
                try:
                    num_iterations = int(sys.argv[i + 1])
                    if num_iterations <= 0:
                        print("Error: --iter must be a positive integer", file=sys.stderr)
                        sys.exit(1)
                    i += 2
                except ValueError:
                    print(f"Error: --iter requires an integer, got '{sys.argv[i + 1]}'", file=sys.stderr)
                    sys.exit(1)
            else:
                print("Error: --iter requires an argument (positive integer)", file=sys.stderr)
                sys.exit(1)

        elif arg == "--dtype":
            if i + 1 < len(sys.argv):
                data_type = sys.argv[i + 1].lower()
                if data_type not in ["fp16", "bf16"]:
                    print(f"Error: --dtype must be 'fp16' or 'bf16', got '{data_type}'", file=sys.stderr)
                    sys.exit(1)
                i += 2
            else:
                print("Error: --dtype requires an argument (fp16 or bf16)", file=sys.stderr)
                sys.exit(1)

        elif arg == "--debug":
            debug_mode = True
            i += 1

        elif arg == "--help" or arg == "-h":
            print(f"Usage: python {sys.argv[0]} [options]", file=sys.stderr)
            print("", file=sys.stderr)
            print("Options:", file=sys.stderr)
            print("  --dtype {{fp16|bf16}}  Data type (default: fp16)", file=sys.stderr)
            print("  --iter N              Number of iterations per mode (default: 32)", file=sys.stderr)
            print("  --debug               Only run first configuration with 1 iteration", file=sys.stderr)
            print("  --test                Test parsing logic and exit", file=sys.stderr)
            print("  [batch_size]          Batch size (default: 512)", file=sys.stderr)
            print("", file=sys.stderr)
            print("Examples:", file=sys.stderr)
            print(f"  python {sys.argv[0]} --dtype fp16 512", file=sys.stderr)
            print(f"  python {sys.argv[0]} --dtype bf16 --iter 5", file=sys.stderr)
            print(f"  python {sys.argv[0]} --debug", file=sys.stderr)
            print(f"  python {sys.argv[0]} 256", file=sys.stderr)
            sys.exit(0)

        else:
            # 尝试解析为 batch_size
            try:
                batch_size = int(arg)
                i += 1
            except ValueError:
                print(f"Error: Unknown argument '{arg}'", file=sys.stderr)
                print(f"Use --help for usage information", file=sys.stderr)
                sys.exit(1)

    # 应用参数
    DATA_TYPE = data_type
    NUM_ITERATIONS = num_iterations

    # 加载配置
    configurations = load_configurations(JSON_CONFIG_PATH)

    # 调试模式：只运行第一个配置
    if debug_mode:
        configurations = [configurations[0]]
        NUM_ITERATIONS = 1  # 调试时只运行1次

    print(f"Loaded {len(configurations)} configurations from {JSON_CONFIG_PATH}", file=sys.stderr)
    print(f"Using batch_size={batch_size}", file=sys.stderr)
    print(f"Data type: {DATA_TYPE}", file=sys.stderr)
    print(f"Running {NUM_ITERATIONS} iterations per mode", file=sys.stderr)
    if debug_mode:
        print("DEBUG MODE: Only running first configuration with 1 iteration", file=sys.stderr)
    print(file=sys.stderr)

    # 测试每个配置
    results = []
    for idx, config in enumerate(configurations, 1):
        print(f"[{idx}/{len(configurations)}] Testing config: {config['signature']}", file=sys.stderr)
        result = test_configuration(config, batch_size)
        results.append(result)
        print(f"  Optimal: conv_genstats={result['optimal_engine']['conv_genstats']}, "
              f"dgrad={result['optimal_engine']['dgrad']}, "
              f"wgrad={result['optimal_engine']['wgrad']}", file=sys.stderr)

    # 生成输出文件名：optimal_engines_{GPU}_{dtype}.json
    output_path = get_output_filename(DATA_TYPE, GPU_NAME)

    # 添加元数据
    output_data = {
        "metadata": {
            "gpu_name": GPU_NAME,
            "data_type": DATA_TYPE,
            "batch_size": batch_size,
            "num_iterations": NUM_ITERATIONS,
            "total_configurations": len(results)
        },
        "results": results
    }

    # 保存 JSON 结果到文件
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}", file=sys.stderr)
    print(f"  GPU: {GPU_NAME}, Data Type: {DATA_TYPE}", file=sys.stderr)


if __name__ == "__main__":
    main()
