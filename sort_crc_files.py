#!/usr/bin/env python3
"""
排序CRC文件工具

对workspace目录下的CRC文件进行字典序排序:
- crc_0.csv: 按字典序排序
- crc.txt: 按字典序排序

注意: 两个文件都有1281168行(最后一行是空行)
"""

import os
import sys

def sort_file(input_path, output_path):
    """
    对文件进行字典序排序

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在: {input_path}")
        return False

    print(f"正在读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  读取了 {len(lines)} 行")

    # 去除每行末尾的换行符,排序后再添加回去
    lines_stripped = [line.rstrip('\n') for line in lines]

    print(f"正在排序...")
    lines_sorted = sorted(lines_stripped)

    # 写入排序后的内容,每行末尾添加换行符
    print(f"正在写入: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines_sorted:
            f.write(line + '\n')

    print(f"  完成! 共写入 {len(lines_sorted)} 行")
    return True

def main():
    workspace_dir = "workspace"

    # 检查workspace目录
    if not os.path.exists(workspace_dir):
        print(f"错误: workspace目录不存在: {workspace_dir}")
        sys.exit(1)

    # 定义文件路径
    crc_0_csv = os.path.join(workspace_dir, "crc_0.csv")
    crc_0_csv_sorted = os.path.join(workspace_dir, "crc_0_sorted.csv")

    crc_txt = os.path.join(workspace_dir, "crc.txt")
    crc_txt_sorted = os.path.join(workspace_dir, "crc_sorted.txt")

    print("=" * 60)
    print("CRC文件排序工具")
    print("=" * 60)
    print()

    # 排序 crc_0.csv
    if os.path.exists(crc_0_csv):
        print(f"1. 处理 crc_0.csv")
        print(f"   输入: {crc_0_csv}")
        print(f"   输出: {crc_0_csv_sorted}")
        if sort_file(crc_0_csv, crc_0_csv_sorted):
            print(f"   [OK] 成功!")
        else:
            print(f"   [FAIL] 失败!")
        print()
    else:
        print(f"1. 跳过 crc_0.csv (文件不存在)")
        print()

    # 排序 crc.txt
    if os.path.exists(crc_txt):
        print(f"2. 处理 crc.txt")
        print(f"   输入: {crc_txt}")
        print(f"   输出: {crc_txt_sorted}")
        if sort_file(crc_txt, crc_txt_sorted):
            print(f"   [OK] 成功!")
        else:
            print(f"   [FAIL] 失败!")
        print()
    else:
        print(f"2. 跳过 crc.txt (文件不存在)")
        print()

    print("=" * 60)
    print("排序完成!")
    print("=" * 60)
    print()
    print("输出文件:")
    if os.path.exists(crc_0_csv_sorted):
        print(f"  - {crc_0_csv_sorted}")
    if os.path.exists(crc_txt_sorted):
        print(f"  - {crc_txt_sorted}")

if __name__ == "__main__":
    main()
