#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crc_csv_merger.py
作用：将指定目录下所有 crc_*.csv 文件首尾相连合并为一个大的 CSV 文件
"""

import os
import sys
import glob
import argparse


def merge_crc_csv_files(input_dir, output_file):
    """
    合并指定目录下所有 crc_*.csv 文件
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出文件路径
    """
    # 查找所有匹配的文件并按名称排序
    pattern = os.path.join(input_dir, "crc_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"警告: 在 '{input_dir}' 中未找到任何 crc_*.csv 文件")
        return False
    
    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # 开始合并
    print(f"\n正在合并到: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        for i, filepath in enumerate(files, 1):
            print(f"[{i}/{len(files)}] 处理: {os.path.basename(filepath)}")
            with open(filepath, 'r', newline='', encoding='utf-8') as infile:
                # 直接原文本复制，不做任何处理
                outfile.write(infile.read())
    
    # 显示结果
    output_size = os.path.getsize(output_file)
    print(f"\n✅ 完成！")
    print(f"   输出文件: {output_file}")
    print(f"   文件大小: {output_size / 1024:.2f} KB")
    
    return True


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description='合并 crc_*.csv 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python crc_csv_merger.py                    # 使用默认参数（当前目录，输出到 merged_crc.csv）
  python crc_csv_merger.py -i ./data          # 指定输入目录
  python crc_csv_merger.py -o result.csv      # 指定输出文件名
  python crc_csv_merger.py -i ./data -o out.csv  # 指定输入目录和输出文件
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        default='.',
        help='输入目录路径 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='merged_crc.csv',
        help='输出文件名 (默认: merged_crc.csv)'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input):
        print(f"错误: 目录 '{args.input}' 不存在")
        sys.exit(1)
    
    # 执行合并
    success = merge_crc_csv_files(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

