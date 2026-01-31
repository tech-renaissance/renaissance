#!/usr/bin/env python3
"""
读取二进制文件的第27个int

用法:
    python read_info.py <binary_file_path>

说明:
    - 读取第27个int（索引26，前面有26个int）
    - 每个int占4字节，所以偏移量是 26 * 4 = 104 字节
    - 使用小端序解析
    - 只读取需要的4个字节，不加载整个文件
"""

import sys
import os


def read_27th_int(file_path: str) -> int:
    """
    读取二进制文件的第27个int

    Args:
        file_path: 二进制文件路径

    Returns:
        第27个int的值（小端序，无符号32位整数）
    """
    # 计算偏移量：第27个int前面有26个int，每个int 4字节
    offset = 26 * 4  # 104 字节

    # 以二进制读模式打开文件
    with open(file_path, 'rb') as f:
        # seek到偏移位置（不读取前面的内容）
        f.seek(offset)

        # 只读取4个字节
        data = f.read(4)

        # 检查是否成功读取了4个字节
        if len(data) < 4:
            raise ValueError(f"文件太小，无法读取第27个int（只读取到{len(data)}个字节）")

        # 按小端序解析为无符号32位整数
        value = int.from_bytes(data, byteorder='little', signed=False)

        return value


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <binary_file_path>", file=sys.stderr)
        print(f"\n示例: {sys.argv[0]} /path/to/file.bin", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}", file=sys.stderr)
        sys.exit(1)

    # 检查是否为文件
    if not os.path.isfile(file_path):
        print(f"错误: 路径不是文件: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # 读取并打印第27个int
        value = read_27th_int(file_path)
        print(value)

    except FileNotFoundError:
        print(f"错误: 文件未找到: {file_path}", file=sys.stderr)
        sys.exit(1)

    except PermissionError:
        print(f"错误: 没有权限读取文件: {file_path}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
