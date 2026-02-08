"""
DTS样本CRC-32提取工具
@version 1.0.0
@date 2026-02-08
@author 技术觉醒团队

从DTS文件中提取每个样本图像的CRC-32码
每行一个CRC-32码，格式为8位16进制大写
"""

import argparse
import struct
import zlib
import os
from pathlib import Path


# 常量定义
FILE_MAGIC = b'.DTS'
CIFAR_MNIST_HEADER_SIZE = 256  # MNIST和CIFAR的Header大小


def read_dts_header(dts_path):
    """
    读取DTS文件头，返回样本数量和图像信息
    """
    if not os.path.exists(dts_path):
        raise FileNotFoundError(f"DTS file not found: {dts_path}")

    with open(dts_path, 'rb') as f:
        # 读取前144字节的header
        header_data = f.read(144)
        if len(header_data) < 144:
            raise ValueError(f"Invalid DTS file: {dts_path}")

        # 解包header (参考make_dataset.py的struct格式)
        header = struct.unpack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI', header_data)

        magic = header[0]
        dataset_type = header[5].decode('ascii').strip()
        num_samples = header[15]
        num_channels = header[13]
        image_width = header[11]
        image_height = header[12]

        if magic != FILE_MAGIC:
            raise ValueError(f"Invalid DTS magic number in {dts_path}")

        # 计算每个样本的字节数
        # MNIST: 28*28*1 = 784 bytes
        # CIFAR-10/100: 32*32*3 = 3072 bytes
        bytes_per_sample = image_width * image_height * num_channels

        return num_samples, bytes_per_sample, dataset_type


def extract_sample_crcs(dts_path, output_txt):
    """
    从DTS文件中提取每个样本的CRC-32码
    """
    print(f"Processing: {dts_path}")

    # 读取header信息
    num_samples, bytes_per_sample, dataset_type = read_dts_header(dts_path)

    print(f"  Dataset: {dataset_type}")
    print(f"  Samples: {num_samples}")
    print(f"  Bytes per sample: {bytes_per_sample}")

    # 打开DTS文件
    with open(dts_path, 'rb') as f:
        # 跳过header (256 bytes for MNIST/CIFAR)
        f.seek(CIFAR_MNIST_HEADER_SIZE)

        # 读取所有labels (num_samples bytes)
        labels_data = f.read(num_samples)
        if len(labels_data) != num_samples:
            raise ValueError(f"Failed to read labels from {dts_path}")

        # 读取所有images
        images_data = f.read(num_samples * bytes_per_sample)
        if len(images_data) != num_samples * bytes_per_sample:
            raise ValueError(f"Failed to read images from {dts_path}")

    # 计算每个样本的CRC-32
    crc_list = []
    for i in range(num_samples):
        # 提取单个样本的图像数据
        start = i * bytes_per_sample
        end = start + bytes_per_sample
        sample_data = images_data[start:end]

        # 计算CRC-32
        crc = zlib.crc32(sample_data) & 0xFFFFFFFF
        crc_hex = f"{crc:08X}"  # 8位16进制大写
        crc_list.append(crc_hex)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")

    # 保存到文本文件
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, 'w') as f:
        for crc in crc_list:
            f.write(crc + '\n')

    print(f"  Saved {len(crc_list)} CRC-32 codes to: {output_txt}")
    print(f"  Done!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CRC-32 codes of each sample from DTS files"
    )
    parser.add_argument(
        'dts_files',
        nargs='+',
        help='DTS files to process (e.g., mnist_train.dts mnist_test.dts)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for CRC text files (default: current directory)'
    )

    args = parser.parse_args()

    # 处理每个DTS文件
    for dts_file in args.dts_files:
        if not os.path.exists(dts_file):
            print(f"Warning: File not found: {dts_file}")
            continue

        # 生成输出文件名
        # mnist_train.dts -> mnist_train_crc.txt
        base_name = Path(dts_file).stem  # 去掉.dts扩展名
        output_name = f"{base_name}_crc.txt"
        output_path = os.path.join(args.output_dir, output_name)

        try:
            extract_sample_crcs(dts_file, output_path)
        except Exception as e:
            print(f"Error processing {dts_file}: {e}\n")


if __name__ == '__main__':
    main()
