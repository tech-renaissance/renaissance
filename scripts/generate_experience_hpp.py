#!/usr/bin/env python3
"""
generate_experience_hpp.py - 将搜索结果文本固化为C++17 constexpr头文件

设计目标：
  1. 解析4个搜索器的简化DSL文本输出
  2. 合并所有操作类型到统一的Experience表
  3. 生成C++17 constexpr静态表（编译期常量）
  4. 提供二分查找函数（<10ns查询）
  5. 支持多GPU平台（A100/RTX5090）和多精度（FP16/BF16）

输入：
  experience_conv_fprop_<GPU>_<dtype>.txt
  experience_conv_genstats_<GPU>_<dtype>.txt
  experience_conv_dgrad_<GPU>_<dtype>.txt
  experience_conv_wgrad_<GPU>_<dtype>.txt

输出：
  generated/cbr_experience_<GPU>_<dtype>.hpp

文本格式（27字段，用'|'分隔）：
  GPU|SM|cuDNN|CUDA|op_type_dtype|N|H|W|C|K|R|S|U|V|P|Q|D|E|NHWC|dtype|FP32|WINNER|BACKUP1|BACKUP2|WS|TIME|SOURCE

  字段说明（字段 12-17 的重要含义）：
    U, V = dilation_h, dilation_w (扩张率)
    P, Q = padding_h, padding_w   (填充)
    D, E = stride_h, stride_w     (步长)

示例：
  RTX5090|SM80|cuDNN9.17.0|CUDA13.1|conv_genstats_fp16|N512|H56|W56|C64|K256|R1|S1|U1|V1|P0|Q0|D1|E1|NHWC|FP16|FP32|cudnnEngine_123|cudnnEngine_456|cudnnEngine_789|536870912|0.7685|exhaustive
         ↑                                     ↑                                                                                           ↑
      字段 0-4                              字段 5-20 (KEY)                                                                              字段 21-26

用法：
  python3 generate_experience_hpp.py <input_dir> <output_dir>

示例（使用绝对路径）：
  # 从 tests/search/ 目录读取，输出到 include/generated/ 目录
  python3 /root/epfs/R/renaissance/scripts/generate_experience_hpp.py /root/epfs/R/renaissance/tests/search/ /root/epfs/R/renaissance/include/generated/

示例（从项目根目录执行）：
  cd /root/epfs/R/renaissance
  python3 scripts/generate_experience_hpp.py tests/search/ include/generated/

作者：技术觉醒团队
版本：1.0.0
日期：2026-04-07
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict


### ════════════════════════════════════════════════════════════════════════════
### §1. 数据结构定义
### ════════════════════════════════════════════════════════════════════════════

class ExperienceRecord:
    """
    Experience记录（与C++端ExperienceRecord结构体对齐）

    字段说明：
      - shape_key: 完整的查询键（GPU|SM|...|Layout|Dtype）
      - winner_tag: 最优Plan的Tag指纹
      - backup1_tag: 备选Plan 1（winner失效时）
      - backup2_tag: 备选Plan 2（backup1也失效时）
      - workspace_bytes: Workspace需求（字节）
      - benchmark_time_ms: Graph模式精测耗时（ms）
      - source: 来源（"exhaustive"/"heur_a"/"heur_b"）
    """

    def __init__(self, shape_key: str, winner_tag: str, backup1_tag: str,
                 backup2_tag: str, workspace_bytes: int, benchmark_time_ms: float,
                 source: str):
        self.shape_key = shape_key
        self.winner_tag = winner_tag
        self.backup1_tag = backup1_tag
        self.backup2_tag = backup2_tag
        self.workspace_bytes = workspace_bytes
        self.benchmark_time_ms = benchmark_time_ms
        self.source = source

    def __lt__(self, other):
        """排序键：按shape_key字典序"""
        return self.shape_key < other.shape_key

    def __eq__(self, other):
        """相等判断：仅比较shape_key"""
        return self.shape_key == other.shape_key

    def __repr__(self):
        return f"ExperienceRecord({self.shape_key[:50]}...)"


### ════════════════════════════════════════════════════════════════════════════
### §2. 文本解析函数
### ════════════════════════════════════════════════════════════════════════════

def parse_experience_file(filepath: Path) -> List[ExperienceRecord]:
    """
    解析单个Experience文本文件

    格式：
      KEY|WINNER_TAG|BACKUP1_TAG|BACKUP2_TAG|WS|TIME|SOURCE

    参数：
      filepath: 文本文件路径

    返回：
      ExperienceRecord列表

    异常：
      RuntimeError: 文件格式错误
    """
    records = []

    if not filepath.exists():
        print(f"[WARNING] File not found: {filepath}")
        return records

    print(f"[PARSING] {filepath}")

    # [编码健壮性改进] 采纳专家建议：添加errors='replace'处理潜在非ASCII字符
    # 参考：PRO/ISSUE/EXP_SNX.md 问题3.2
    # 原因：某些Plan Tag可能包含非ASCII字符（cuDNN内部生成），导致解析失败
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            ### 跳过空行和注释行
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            ### 按'|'分割
            parts = line.split('|')

            ### 支持两种格式：
            ### 1. 完整格式（27字段）：KEY(21字段)|winner|backup1|backup2|ws|time|source
            ### 2. 简化格式（7字段）：KEY|winner|backup1|backup2|ws|time|source
            if len(parts) == 27:
                # 完整格式：前21个字段组成shape_key，后6个是数据
                #
                # ════════════════════════════════════════════════════════════════════════════
                # 字段顺序说明（0-indexed，必须与 ta_v4_search_common.hpp 的 to_key_string() 一致）：
                # 0-4:   GPU|SM|cuDNN|CUDA|op_type_dtype
                # 5:     N (batch size)
                # 6-7:   H, W (输入尺寸)
                # 8-9:   C, K (通道数)
                # 10-11: R, S (卷积核尺寸)
                # 12-13: U, V (dilation，扩张率) ← 修复后：dilation
                # 14-15: P, Q (padding，填充)
                # 16-17: D, E (stride，步长)     ← 修复后：stride
                # 18-20: NHWC|dtype|FP32
                # 21:    winner_tag (最优引擎)
                # 22-23: backup1_tag, backup2_tag (备选引擎)
                # 24:    workspace_bytes (workspace 需求)
                # 25:    benchmark_time_ms (测试耗时)
                # 26:    source ("exhaustive"/"heur_a"/"heur_b")
                # ════════════════════════════════════════════════════════════════════════════
                shape_key = '|'.join(parts[:21]).strip()
                winner_tag = parts[21].strip()
                backup1_tag = parts[22].strip()
                backup2_tag = parts[23].strip()
                workspace_bytes = int(parts[24].strip())
                benchmark_time_ms = float(parts[25].strip())
                source = parts[26].strip()
            elif len(parts) == 7:
                # 简化格式
                shape_key = parts[0].strip()
                winner_tag = parts[1].strip()
                backup1_tag = parts[2].strip()
                backup2_tag = parts[3].strip()
                workspace_bytes = int(parts[4].strip())
                benchmark_time_ms = float(parts[5].strip())
                source = parts[6].strip()
            else:
                print(f"[ERROR] Line {line_num}: Invalid format (expected 7 or 27 fields, got {len(parts)})")
                print(f"  Line content: {line[:100]}...")
                continue

            try:

                ### 创建记录
                record = ExperienceRecord(
                    shape_key=shape_key,
                    winner_tag=winner_tag,
                    backup1_tag=backup1_tag,
                    backup2_tag=backup2_tag,
                    workspace_bytes=workspace_bytes,
                    benchmark_time_ms=benchmark_time_ms,
                    source=source
                )
                records.append(record)

            except (ValueError, IndexError) as e:
                print(f"[ERROR] Line {line_num}: Failed to parse fields: {e}")
                print(f"  Line content: {line[:100]}...")
                continue

    print(f"  Parsed {len(records)} records from {filepath.name}")
    return records


def collect_all_records(input_dir: Path, gpu: str, dtype: str) -> List[ExperienceRecord]:
    """
    收集指定GPU和精度的所有操作类型的Experience记录

    参数：
      input_dir: 输入目录
      gpu: GPU型号（"a100"/"rtx5090"）
      dtype: 精度类型（"fp16"/"bf16"）

    返回：
      合并后的ExperienceRecord列表
    """
    all_records = []

    ### 操作类型列表
    op_types = ["conv_fprop", "conv_genstats", "conv_dgrad", "conv_wgrad"]

    ### GPU全称映射
    gpu_full_names = {
        "a100": "A100-SXM4-80GB",
        "rtx5090": "RTX5090"
    }

    gpu_full = gpu_full_names.get(gpu.lower(), gpu.upper())

    for op_type in op_types:
        ### 文件名格式：experience_<op_type>_<GPU>_<dtype>.txt
        filename = f"experience_{op_type}_{gpu_full}_{dtype}.txt"
        filepath = input_dir / filename

        ### 解析文件
        records = parse_experience_file(filepath)
        all_records.extend(records)

    ### 去重：对于重复的shape_key，只保留性能最好的（时间最短的）
    seen_keys = {}
    deduped_records = []
    for rec in all_records:
        key = rec.shape_key
        if key not in seen_keys:
            seen_keys[key] = rec
            deduped_records.append(rec)
        else:
            # 如果已存在，比较性能，保留时间更短的
            existing = seen_keys[key]
            if rec.benchmark_time_ms < existing.benchmark_time_ms:
                # 替换为性能更好的记录
                idx = deduped_records.index(existing)
                deduped_records[idx] = rec
                seen_keys[key] = rec
                print(f"  [DEDUP] Replaced {key[:50]}... ({existing.benchmark_time_ms}ms → {rec.benchmark_time_ms}ms, {rec.source})")

    ### 按shape_key排序（必须！为了二分查找）
    deduped_records.sort()

    print(f"\n[SUMMARY] Collected {len(deduped_records)} total records for {gpu} {dtype.upper()}")

    return deduped_records


### ════════════════════════════════════════════════════════════════════════════
### §3. C++头文件生成
### ════════════════════════════════════════════════════════════════════════════

def escape_cpp_string(s: str) -> str:
    """
    转义C++字符串中的特殊字符

    参数：
      s: 原始字符串

    返回：
      转义后的字符串
    """
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def generate_hpp(records: List[ExperienceRecord], output_path: Path,
                 gpu: str, dtype: str) -> None:
    """
    生成C++17 constexpr头文件

    参数：
      records: ExperienceRecord列表（已排序）
      output_path: 输出文件路径
      gpu: GPU型号（"a100"/"rtx5090"）
      dtype: 精度类型（"fp16"/"bf16"）
    """

    ### GPU全称映射
    gpu_full_names = {
        "a100": "A100-SXM4-80GB",
        "rtx5090": "RTX5090"
    }
    gpu_full = gpu_full_names.get(gpu.lower(), gpu.upper())

    ### 平台宏映射
    platform_macros = {
        "a100": "USING_A100",
        "rtx5090": "USING_RTX5090"
    }
    platform_macro = platform_macros.get(gpu.lower(), f"USING_{gpu.upper()}")

    ### 文件头
    lines = [
        f'// Auto-generated by generate_experience_hpp.py',
        f'// DO NOT EDIT MANUALLY',
        f'//',
        f'// Generated at: {datetime.now().isoformat()}',
        f'// GPU: {gpu_full}',
        f'// Data Type: {dtype.upper()}',
        f'// Total Records: {len(records)}',
        f'//',
        f'// Purpose: Provide constexpr Experience table for Mode C穷举式搜索',
        f'//          Zero-overhead binary search lookup (<10ns)',
        f'//          Platform macro: {platform_macro}',
        f'//',
        f'// Usage:',
        f'//   #ifdef {platform_macro}',
        f'//   #include "cbr_experience_{gpu.lower()}_{dtype}.hpp"',
        f'//   namespace exp = ta_v4::experience;',
        f'//   auto record = exp::lookup(key);',
        f'//   if (record) match_and_build_plan(graph, candidates, record, handle);',
        f'//   #endif',
        f'',
        f'#pragma once',
        f'',
        f'#include <array>',
        f'#include <string_view>',
        # [GMX-003 FIX] 删除optional头文件，因为lookup()返回原始指针而非std::optional
        f'#include <algorithm>',
        f'#include <cstdint>',
        f'',
        f'namespace ta_v4 {{',
        f'namespace experience {{',
        f'',
        f'/**',
        f' * @brief Experience记录结构（与搜索器输出格式对齐）',
        f' * @note 所有字符串均为编译期常量，零运行时分配',
        f' */',
        f'struct ExperienceRecord {{',
        f'    const char* shape_key;         // 完整查询键',
        f'    const char* winner_tag;        // 最优Plan指纹',
        f'    const char* backup1_tag;      // 备选Plan 1',
        f'    const char* backup2_tag;      // 备选Plan 2',
        f'    uint64_t workspace_bytes;     // Workspace需求（字节）',
        f'    float benchmark_time_ms;     // Graph精测耗时（ms）',
        f'    const char* source;            // 来源："exhaustive"/"heur_a"/"heur_b"',
        f'}};',
        f'',
        f'/**',
        f' * @brief Experience表（已排序，支持二分查找）',
        f' * @note 总记录数：{len(records)}',
        f' */',
        f'inline constexpr std::array<ExperienceRecord, {len(records)}> TABLE = {{{{',
    ]

    ### 遍历所有记录，生成表项
    for idx, record in enumerate(records):
        ### 转义字符串
        key_esc = escape_cpp_string(record.shape_key)
        winner_esc = escape_cpp_string(record.winner_tag)
        backup1_esc = escape_cpp_string(record.backup1_tag)
        backup2_esc = escape_cpp_string(record.backup2_tag)

        ### 生成表项（不添加注释）
        lines.append(
            f'    {{"{key_esc}", "{winner_esc}", '
            f'"{backup1_esc}", "{backup2_esc}", '
            f'{record.workspace_bytes}ULL, {record.benchmark_time_ms:.4f}f, "{record.source}"}},'
        )

    ### 文件尾
    lines.extend([
        f'}}}};',
        f'',
        f'/**',
        f' * @brief 零开销二分查找（时间复杂度：O(log N)）',
        f' * @param key Shape Key字符串（必须与搜索器输出格式完全一致）',
        # [GMX-003 FIX] 修改原因：返回原始指针而非std::optional，与主程序fallback实现类型一致
        f' * @return 找到返回记录指针，否则返回nullptr',
        f' * @note 平均查询时间：< 10ns',
        f' */',
        # [GMX-003 FIX] 修改返回类型：std::optional<const ExperienceRecord*> → const ExperienceRecord*
        # 原因：主程序fallback路径返回const ExperienceRecord*，需保持类型一致
        # 原错误：返回std::optional导致exp_rec != nullptr编译失败
        f'inline const ExperienceRecord* lookup(std::string_view key) {{',
        f'    auto it = std::lower_bound(',
        f'        TABLE.begin(), TABLE.end(), key,',
        f'        [](const ExperienceRecord& rec, std::string_view k) {{',
        f'            return std::string_view(rec.shape_key) < k;',
        f'        }});',
        f'    ',
        f'    if (it != TABLE.end() && std::string_view(it->shape_key) == key) {{',
        f'        return &(*it);',
        f'    }}',
        # [GMX-003 FIX] 返回nullptr而非std::nullopt
        f'    return nullptr;',
        f'}}',
        f'',
        f'/**',
        f' * @brief 统计信息',
        f' */',
        f'inline void print_statistics() {{',
        f'    std::cout << "[Experience Table Statistics]" << std::endl;',
        f'    std::cout << "  Total records: " << TABLE.size() << std::endl;',
        f'    ',
        f'    // 统计各操作类型数量',
        f'    int fprop_count = 0, genstats_count = 0, dgrad_count = 0, wgrad_count = 0;',
        f'    for (const auto& rec : TABLE) {{',
        f'        std::string_view key(rec.shape_key);',
        f'        if (key.find("conv_fprop") != std::string_view::npos) ++fprop_count;',
        f'        else if (key.find("conv_genstats") != std::string_view::npos) ++genstats_count;',
        f'        else if (key.find("conv_dgrad") != std::string_view::npos) ++dgrad_count;',
        f'        else if (key.find("conv_wgrad") != std::string_view::npos) ++wgrad_count;',
        f'    }}',
        f'    ',
        f'    std::cout << "  By operation type:" << std::endl;',
        f'    std::cout << "    Conv Fprop:    " << fprop_count << std::endl;',
        f'    std::cout << "    Conv GenStats: " << genstats_count << std::endl;',
        f'    std::cout << "    Conv DGrad:    " << dgrad_count << std::endl;',
        f'    std::cout << "    Conv WGrad:   " << wgrad_count << std::endl;',
        f'    ',
        f'    // 统计来源分布',
        f'    int exhaustive_count = 0, heur_a_count = 0, heur_b_count = 0;',
        f'    for (const auto& rec : TABLE) {{',
        f'        std::string_view src(rec.source);',
        f'        if (src == "exhaustive") ++exhaustive_count;',
        f'        else if (src == "heur_a") ++heur_a_count;',
        f'        else if (src == "heur_b") ++heur_b_count;',
        f'    }}',
        f'    ',
        f'    std::cout << "  By source:" << std::endl;',
        f'    std::cout << "    Exhaustive:    " << exhaustive_count << std::endl;',
        f'    std::cout << "    Heuristic A:   " << heur_a_count << std::endl;',
        f'    std::cout << "    Heuristic B:   " << heur_b_count << std::endl;',
        f'}}',
        f'',
        f'}} // namespace experience',
        f'}} // namespace ta_v4',
    ])

    ### 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[SUCCESS] Generated: {output_path}")
    print(f"  Total records: {len(records)}")


### ════════════════════════════════════════════════════════════════════════════
### §4. 验证函数
### ════════════════════════════════════════════════════════════════════════════

def validate_records(records: List[ExperienceRecord], gpu: str, dtype: str) -> bool:
    """
    验证Experience记录的完整性

    参数：
      records: ExperienceRecord列表
      gpu: GPU型号
      dtype: 精度类型

    返回：
      是否验证通过
    """
    print(f"\n[VALIDATION] Checking {len(records)} records for {gpu} {dtype.upper()}...")

    issues = []

    ### 检查是否有无效记录
    for idx, rec in enumerate(records):
        if not rec.winner_tag:
            issues.append(f"记录{idx}缺少winner_tag")
        # 注意：workspace_bytes可以为0（很多最优引擎不需要额外workspace）
        # 注意：benchmark_time_ms可以为0（新增的启发式记录，时间未知）
        # 只检查是否为负数，不检查0
        if rec.benchmark_time_ms < 0:
            issues.append(f"记录{idx}的benchmark_time_ms异常: {rec.benchmark_time_ms}")

    ### 统计来源分布
    sources = defaultdict(int)
    for rec in records:
        sources[rec.source] += 1

    print(f"  来源分布:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count} ({count/len(records)*100:.1f}%)")

    ### 检查是否有缺失的操作类型
    op_types_found = set()
    for rec in records:
        if "conv_fprop" in rec.shape_key:
            op_types_found.add("conv_fprop")
        elif "conv_genstats" in rec.shape_key:
            op_types_found.add("conv_genstats")
        elif "conv_dgrad" in rec.shape_key:
            op_types_found.add("conv_dgrad")
        elif "conv_wgrad" in rec.shape_key:
            op_types_found.add("conv_wgrad")

    expected_ops = {"conv_fprop", "conv_genstats", "conv_dgrad", "conv_wgrad"}
    missing_ops = expected_ops - op_types_found
    if missing_ops:
        # 只警告，不阻止生成（部分操作类型的经验仍然有价值）
        print(f"  [WARNING] 缺少操作类型: {missing_ops}")
        print(f"  [INFO] 将使用现有记录继续生成（缺失的操作将fallback到Heuristic B）")

    ### 输出结果
    if issues:
        print(f"  [ERROR] 发现{len(issues)}个问题:")
        for issue in issues[:10]:
            print(f"    - {issue}")
        return False
    else:
        print(f"  [OK] 验证通过，无重复记录，来源分布正常")
        return True


### ════════════════════════════════════════════════════════════════════════════
### §5. 批量生成函数
### ════════════════════════════════════════════════════════════════════════════

def generate_all_platforms(input_dir: Path, output_dir: Path) -> None:
    """
    为所有平台生成Experience表

    参数：
      input_dir: 输入目录（包含所有搜索器的txt文件）
      output_dir: 输出目录（生成的hpp文件）
    """
    platforms = [
        ("a100", "fp16"),
        ("a100", "bf16"),
        ("rtx5090", "fp16"),
        ("rtx5090", "bf16"),
    ]

    print("\n" + "="*70)
    print("TA-V4 Experience Table Generator")
    print("="*70)
    print(f"Input Directory: {input_dir.absolute()}")
    print(f"Output Directory: {output_dir.absolute()}")
    print("="*70)

    success_count = 0

    for gpu, dtype in platforms:
        print(f"\n[{gpu.upper()} {dtype.upper()}]")
        print("-"*70)

        ### 收集所有记录
        records = collect_all_records(input_dir, gpu, dtype)

        if not records:
            print(f"[WARNING] No records found for {gpu} {dtype}, skipping...")
            continue

        ### 验证记录
        if not validate_records(records, gpu, dtype):
            print(f"[ERROR] Validation failed for {gpu} {dtype}, skipping...")
            continue

        ### 生成头文件
        output_filename = f"cbr_experience_{gpu}_{dtype}.hpp"
        output_path = output_dir / output_filename

        generate_hpp(records, output_path, gpu, dtype)
        success_count += 1

    print("\n" + "="*70)
    print(f"Generation Complete: {success_count}/{len(platforms)} platforms")
    print("="*70)


### ════════════════════════════════════════════════════════════════════════════
### §6. 主函数
### ════════════════════════════════════════════════════════════════════════════

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("Usage: python3 generate_experience_hpp.py <input_dir> <output_dir>")
        print("\nArguments:")
        print("  input_dir   : Directory containing experience_*.txt files")
        print("  output_dir  : Directory to output generated .hpp files")
        print("\nExample:")
        print("  python3 generate_experience_hpp.py . generated/")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    ### 检查输入目录
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    ### 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    ### 生成所有平台的头文件
    generate_all_platforms(input_dir, output_dir)


if __name__ == "__main__":
    main()
