#!/usr/bin/env python3
"""
@file collect_crc.py
@brief 计算目录下所有JPEG文件的CRC32值
@details 递归遍历目录，计算每个JPEG文件的CRC32校验值，保存到CSV文件
@version 1.0.0
@date 2026-02-05
@author 技术觉醒团队

Usage:
    python collect_crc.py /root/dataset/imagenet/train
    python collect_crc.py T:/Dataset/imagenet/train
    python collect_crc.py /path/to/directory --output custom.txt

Output format (one CRC32 per line):
    A1B2C3D4
    E5F6G7H8
    ...
"""

import os
import sys
import argparse
import zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 全局锁，用于保护CSV文件写入
write_lock = Lock()


def calculate_crc32(file_path):
    """
    计算文件的CRC32值

    Args:
        file_path: 文件路径（Path对象）

    Returns:
        str: CRC32值（8位16进制大写字符串）或 None
    """
    try:
        with open(file_path, 'rb') as f:
            # 读取文件内容并计算CRC32
            data = f.read()
            crc_value = zlib.crc32(data) & 0xffffffff  # 确保是无符号32位整数

            # 转换为8位16进制大写字符串
            crc_hex = f"{crc_value:08X}"

            return crc_hex
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}", file=sys.stderr)
        return None


def find_jpeg_files(root_dir):
    """
    递归查找目录下所有JPEG文件

    Args:
        root_dir: 根目录路径

    Yields:
        Path: JPEG文件的路径
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_dir}")

    # 常见的JPEG文件扩展名
    jpeg_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.jpe', '.jpe'}

    # 递归查找所有JPEG文件
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in jpeg_extensions:
            yield file_path


def collect_crc_single_threaded(root_dir, output_file):
    """
    单线程收集CRC32值（适合调试或小数据集）

    Args:
        root_dir: 根目录路径
        output_file: 输出文件路径

    Returns:
        int: 处理的文件总数
    """
    print(f"[INFO] Scanning directory: {root_dir}")
    print(f"[INFO] Mode: Single-threaded")

    processed_count = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        # 查找并处理所有JPEG文件
        for file_path in find_jpeg_files(root_dir):
            crc_hex = calculate_crc32(file_path)

            if crc_hex:
                f.write(f"{crc_hex}\n")
                processed_count += 1

                # 每1000个文件打印一次进度
                if processed_count % 1000 == 0:
                    print(f"[PROGRESS] Processed {processed_count} files...")
            else:
                error_count += 1

    print(f"[INFO] Completed: {processed_count} files processed, {error_count} errors")
    return processed_count


def collect_crc_multi_threaded(root_dir, output_file, num_workers=None):
    """
    多线程收集CRC32值（适合大数据集，如ImageNet）

    Args:
        root_dir: 根目录路径
        output_file: 输出文件路径
        num_workers: 线程数（默认为CPU核心数）

    Returns:
        int: 处理的文件总数
    """
    import multiprocessing

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"[INFO] Scanning directory: {root_dir}")
    print(f"[INFO] Mode: Multi-threaded ({num_workers} workers)")

    # 先收集所有文件路径
    print(f"[INFO] Collecting file list...")
    all_files = list(find_jpeg_files(root_dir))
    total_files = len(all_files)

    if total_files == 0:
        print("[WARNING] No JPEG files found in the directory!")
        return 0

    print(f"[INFO] Found {total_files} JPEG files")

    # 使用线程池处理
    processed_count = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        def process_file(index, file_path):
            """处理单个文件"""
            crc_hex = calculate_crc32(file_path)
            return index, crc_hex

        # 提交所有任务
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有文件处理任务
            futures = {
                executor.submit(process_file, i, file_path): i
                for i, file_path in enumerate(all_files)
            }

            # 收集结果并实时写入
            for future in as_completed(futures):
                index, crc_hex = future.result()

                if crc_hex:
                    # 使用锁保护文件写入
                    with write_lock:
                        f.write(f"{crc_hex}\n")

                    processed_count += 1

                    # 每10000个文件打印一次进度
                    if processed_count % 10000 == 0:
                        progress = processed_count / total_files * 100
                        print(f"[PROGRESS] {processed_count}/{total_files} files ({progress:.1f}%)")
                else:
                    error_count += 1

    print(f"[INFO] Completed: {processed_count} files processed, {error_count} errors")
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description='Calculate CRC32 values for all JPEG files in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python collect_crc.py /root/dataset/imagenet/train

  # Custom output file
  python collect_crc.py T:/Dataset/imagenet/train --output train_crc.txt

  # Single-threaded mode (slower but uses less resources)
  python collect_crc.py /path/to/dir --single-threaded

  # Specify number of worker threads
  python collect_crc.py /path/to/dir --workers 16
        """
    )

    parser.add_argument(
        'directory',
        help='Root directory containing JPEG files'
    )

    parser.add_argument(
        '--output', '-o',
        default='crc.txt',
        help='Output file path (default: crc.txt)'
    )

    parser.add_argument(
        '--single-threaded',
        action='store_true',
        help='Use single-threaded mode (slower but uses less resources)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of worker threads (default: CPU count)'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify output file by counting rows'
    )

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.isdir(args.directory):
        print(f"[ERROR] Directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("CRC32 Collection Tool")
    print("=" * 70)
    print(f"Input directory: {args.directory}")
    print(f"Output file: {args.output}")
    print(f"Single-threaded: {args.single_threaded}")
    print("=" * 70)
    print()

    # 选择处理模式
    start_time = __import__('time').time()

    if args.single_threaded:
        total_files = collect_crc_single_threaded(args.directory, args.output)
    else:
        total_files = collect_crc_multi_threaded(args.directory, args.output, args.workers)

    elapsed_time = __import__('time').time() - start_time

    print()
    print("=" * 70)
    print(f"[SUCCESS] CRC32 collection completed!")
    print(f"[STATS] Total files: {total_files}")
    print(f"[STATS] Elapsed time: {elapsed_time:.2f} seconds")
    if total_files > 0:
        print(f"[STATS] Throughput: {total_files/elapsed_time:.1f} files/sec")
    print(f"[OUTPUT] CRC32 values saved to: {args.output}")
    print("=" * 70)

    # 验证输出文件
    if args.verify:
        print()
        print("[VERIFY] Checking output file...")

        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)

            print(f"[VERIFY] Total lines in file: {line_count}")

            if line_count == total_files:
                print("[VERIFY] ✓ Line count matches processed files!")
            else:
                print(f"[VERIFY] ✗ Warning: Line count mismatch! Expected {total_files}, got {line_count}")
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
