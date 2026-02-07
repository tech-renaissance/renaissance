#!/usr/bin/env python3
"""
通用文件排序工具

对文本文件按字典序进行排序。

用法:
    python sort_crc.py <input_file>

示例:
    python sort_crc.py data.csv
    python sort_crc.py data.txt

输出:
    data_sorted.csv (如果输入是data.csv)
    data_sorted.txt (如果输入是data.txt)
"""

import os
import sys
import argparse

def sort_file(input_path, output_path):
    """
    对文件进行字典序排序

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径

    Returns:
        bool: 成功返回True,失败返回False
    """
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在: {input_path}", file=sys.stderr)
        return False

    print(f"正在读取: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"错误: 无法读取文件: {e}", file=sys.stderr)
        return False

    print(f"  读取了 {len(lines)} 行")

    # 去除每行末尾的换行符,排序后再添加回去
    lines_stripped = [line.rstrip('\n') for line in lines]

    print(f"正在排序...")
    lines_sorted = sorted(lines_stripped)

    # 写入排序后的内容,每行末尾添加换行符
    print(f"正在写入: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines_sorted:
                f.write(line + '\n')
    except Exception as e:
        print(f"错误: 无法写入文件: {e}", file=sys.stderr)
        return False

    print(f"  完成! 共写入 {len(lines_sorted)} 行")
    return True

def get_output_path(input_path):
    """
    根据输入文件路径生成输出文件路径

    规则:
    - input.csv -> input_sorted.csv
    - input.txt -> input_sorted.txt
    - 其他文件 -> input.sorted

    Args:
        input_path: 输入文件路径

    Returns:
        str: 输出文件路径
    """
    # 获取文件名(不含路径)
    basename = os.path.basename(input_path)

    # 分离文件名和扩展名
    name, ext = os.path.splitext(basename)

    # 生成输出文件名
    output_name = f"{name}_sorted{ext}"

    # 如果没有扩展名,使用 .sorted 后缀
    if not ext:
        output_name = f"{name}.sorted"

    # 组合完整路径
    dirname = os.path.dirname(input_path)
    if dirname:
        return os.path.join(dirname, output_name)
    else:
        return output_name

def main():
    parser = argparse.ArgumentParser(
        description='通用文件排序工具 - 按字典序对文本文件进行排序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s data.csv          # 生成 data_sorted.csv
  %(prog)s data.txt          # 生成 data_sorted.txt
  %(prog)s /path/to/file.csv  # 生成 /path/to/file_sorted.csv
        '''
    )

    parser.add_argument(
        'input_file',
        help='输入文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='输出文件路径(默认自动生成)',
        default=None
    )

    args = parser.parse_args()

    # 检查输入文件
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)

    # 确定输出文件路径
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = get_output_path(input_file)

    # 显示信息
    print("=" * 60)
    print("文件排序工具")
    print("=" * 60)
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print()

    # 执行排序
    if sort_file(input_file, output_file):
        print()
        print("=" * 60)
        print("排序完成!")
        print("=" * 60)
        print(f"输出文件: {output_file}")
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("排序失败!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
