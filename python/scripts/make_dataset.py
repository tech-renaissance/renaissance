"""
DTS 数据集导出工具
@version 3.0.0
@date 2026-01-08
@author 技术觉醒团队

将MNIST、CIFAR-10、CIFAR-100、ImageNet导出为.dts格式
"""

import argparse
import os
from urllib import request, error
from pathlib import Path
import zlib
import gzip
import tarfile
import shutil
import subprocess
import platform
from PIL import Image
from io import BytesIO, SEEK_END
import pickle
import random
import struct
import mmap
import numpy as np


# ********************************************************
# ImageNet数据集的压缩等级：L0 ~ L3
# L0：完全无压缩，数据集原图字节流拼接，只是规划了对齐
# L1：仅缩放。策略是把所有短边超过400px的图片按比例缩放成短边400px
# L2：缩放+裁剪。短边400~440px的情况采取裁剪，且限制长边到600px以内
# L3：缩放+裁剪+降低JPEG质量。在L2的基础上，把JPEG质量从90降低到80
# ********************************************************

# 全局设定
RAND_SEED = 42
BLOCK_SIZE = 16 * 1024 * 1024  # 16 MB
USING_PIC_ALIGNMENT = True
PIC_ALIGNMENT_BYTES = 64  # Bytes
FILE_MAGIC = b'.DTS'
DTS_VERSION = [3, 0, 0, 0]
IMAGENET_HEADER_SIZE = 16 * 1024 * 1024  # 16 MB
CIFAR_MNIST_HEADER_SIZE = 256  # 256 Bytes

# LV0 参数
HEADER_SIZE_LV0 = 4 * 1024  # 4 KB, 256 elements
MINIMUM_TRAIN_BLOCKS_LV0 = 8700
MINIMUM_VAL_BLOCKS_LV0 = 400
MARGINAL_FILL_ITERATIONS = 32
BLOCK_MAGIC_LV0 = b'LV0B'

# LV1 ~ LV3 参数
HEADER_SIZE_LV123 = 16 * 1024  # 16 KB, 1024 elements
BLOCK_MAGIC_LV1 = b'LV1B'
BLOCK_MAGIC_LV2 = b'LV2B'
BLOCK_MAGIC_LV3 = b'LV3B'

SCALING_ALGORITHM = 'LANCZOS'  # BICUBIC OR LANCZOS
SHORT_LIMIT = 400
LONG_LIMIT = 600
CROP_BOUNDARY = 440

algorithm = None
jpeg_quality = 90
crop_boundary = CROP_BOUNDARY
dataset_name = ''
print_name = ''
enable_download = False
enable_extract = False
enable_concat = False
input_path = ''
output_path = ''
compress_level = 0
train_set_only = False
val_set_only = False
remove_source = False
preprocess_val_set = False
skip_download_verification = False
view_header = False
downloaded_files = list()
mnist_urls = [
    'https://tech-renaissance.cn/download/mnist/',
    'https://ossci-datasets.s3.amazonaws.com/mnist/'
]
mnist_targets = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]
cifar10_urls = [
    'https://tech-renaissance.cn/download/cifar-10/',
    'https://www.cs.toronto.edu/~kriz/'
]
cifar10_targets = [
    'cifar-10-binary.tar.gz'
]
cifar100_urls = [
    'https://tech-renaissance.cn/download/cifar-100/',
    'https://www.cs.toronto.edu/~kriz/'
]
cifar100_targets = [
    'cifar-100-binary.tar.gz'
]
imagenet_targets = [
    'ILSVRC2012_img_train.tar',
    'ILSVRC2012_img_val.tar'
]
script_urls_linux = [
    'https://tech-renaissance.cn/download/imagenet/',
    'https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/'
]
script_urls_windows = [
    'https://tech-renaissance.cn/download/imagenet/'
]
correct_crc = {
    'mnist': {
        'train-images-idx3-ubyte.gz': 'eb392171',
        'train-labels-idx1-ubyte.gz': '28ee680a',
        't10k-images-idx3-ubyte.gz': 'df9322ee',
        't10k-labels-idx1-ubyte.gz': '5c1cf43b'
    },
    'cifar10': {
        'cifar-10-binary.tar.gz': 'f709d4ba'
    },
    'cifar100': {
        'cifar-100-binary.tar.gz': 'b2274685'
    },
    'imagenet': {
        'ILSVRC2012_img_train.tar': '01f6e856',
        'ILSVRC2012_img_val.tar': '7c951e53'
    }
}
imagenet_folder_check_code = '007d4f42'
mnist_extracted_targets = [
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte'
]
cifar10_extracted_targets = [
    'data_batch_1.bin',
    'data_batch_2.bin',
    'data_batch_3.bin',
    'data_batch_4.bin',
    'data_batch_5.bin',
    'test_batch.bin'
]
cifar100_extracted_targets = [
    'train.bin',
    'test.bin'
]


def parse_args():
    # 初始化解析器
    parser = argparse.ArgumentParser(description="Simple Argument Parser")

    # 1. 必选参数 (dataset_name)
    parser.add_argument('-n', dest='dataset_name', required=True, help='Dataset Name')

    # 2. 布尔开关 (出现在命令行即为True，否则为False)
    parser.add_argument('-d', dest='enable_download', action='store_true', help='Download')
    parser.add_argument('-x', dest='enable_extract', action='store_true', help='Extract')
    parser.add_argument('-c', dest='enable_concat',   action='store_true', help='Concat')

    # 3. 路径参数 (带默认值)
    parser.add_argument('-i', dest='input_path',  default='.', help='Input Path')
    # output_path 默认值设为 None，以便后续逻辑处理
    parser.add_argument('-o', dest='output_path', default=None, help='Output Path')

    # 4. 带范围限制的整数
    parser.add_argument('-L', dest='compress_level', type=int, choices=[0, 1, 2, 3], default=0, help='Compression Level (for ImageNet Only)')

    # 5. 其他布尔开关
    parser.add_argument('-t', dest='train_set_only',     action='store_true', help='Training Set Only (for ImageNet Only)')
    parser.add_argument('-v', dest='val_set_only',       action='store_true', help='Validation Set Only (for ImageNet Only)')
    parser.add_argument('-r', dest='remove_source',      action='store_true', help='Remove Source')
    parser.add_argument('-p', dest='preprocess_val_set', action='store_true', help='Preprocess the Validation Set (for ImageNet Only)')
    parser.add_argument('-s', dest='skip_download_verification', action='store_true', help='Skip the verification of downloaded files')

    parser.add_argument('-g', dest='view_header',     action='store_true', help='View the header of .dts files')

    args = parser.parse_args()

    # 特殊逻辑：如果 -o 未指定，则默认等于 input_path
    if args.output_path is None:
        args.output_path = args.input_path

    return args


def initialize_args(user_args):
    global dataset_name, print_name, enable_download, enable_extract, enable_concat, input_path, output_path
    global compress_level, train_set_only, val_set_only, remove_source, preprocess_val_set, downloaded_files
    global skip_download_verification, algorithm, jpeg_quality, crop_boundary, view_header

    user_dict = vars(user_args)
    user_dataset_name = user_dict['dataset_name'].lower()
    if user_dataset_name in ['imagenet', 'image-net', 'image_net']:
        dataset_name = 'imagenet'
        print_name = 'ImageNet'
    elif user_dataset_name in ['cifar10', 'cifar-10', 'cifar_10', 'cifar']:
        dataset_name = 'cifar10'
        print_name = 'CIFAR-10'
    elif user_dataset_name in ['cifar100', 'cifar-100', 'cifar_100']:
        dataset_name = 'cifar100'
        print_name = 'CIFAR-100'
    elif user_dataset_name == 'mnist':
        dataset_name = 'mnist'
        print_name = 'MNIST'
    else:
        print('Unsupported Dataset')
        exit(1)
    enable_download = user_dict['enable_download']
    enable_extract = user_dict['enable_extract']
    enable_concat = user_dict['enable_concat']
    input_path = user_dict['input_path']
    output_path = user_dict['output_path']
    compress_level = user_dict['compress_level']
    train_set_only = user_dict['train_set_only']
    val_set_only = user_dict['val_set_only']
    remove_source = user_dict['remove_source']
    preprocess_val_set = user_dict['preprocess_val_set']
    skip_download_verification = user_dict['skip_download_verification']
    view_header = user_dict['view_header']
    algorithm = Image.Resampling.BICUBIC if SCALING_ALGORITHM == 'BICUBIC' else Image.Resampling.LANCZOS
    jpeg_quality = 80 if compress_level == 3 else 90
    crop_boundary = SHORT_LIMIT if compress_level == 1 else CROP_BOUNDARY

    if dataset_name == 'mnist':
        downloaded_files = mnist_targets
    elif dataset_name == 'cifar10':
        downloaded_files = cifar10_targets
    elif dataset_name == 'cifar100':
        downloaded_files = cifar100_targets
    else:
        downloaded_files = imagenet_targets


def suggest_imagenet_download():
    print('Please download the ImageNet dataset from the official website:')
    print('\thttps://image-net.org/')
    print('Two files are required:')
    print('\tILSVRC2012_img_train.tar (147,897,477,120 Bytes)')
    print('\tILSVRC2012_img_val.tar (6,744,924,160 Bytes)')


def download_dataset():
    Path(input_path).mkdir(exist_ok=True)
    urls = list()
    if dataset_name == 'mnist':
        urls = mnist_urls
    elif dataset_name == 'cifar10':
        urls = cifar10_urls
    elif dataset_name == 'cifar100':
        urls = cifar100_urls
    else:
        failed = False
        if not val_set_only:
            if os.path.exists(os.path.join(input_path, 'ILSVRC2012_img_train.tar')):
                print('Using cached ILSVRC2012_img_train.tar ...')
            else:
                dir_name = os.path.join(output_path, 'train')
                if enable_extract and os.path.exists(dir_name) and os.path.isdir(dir_name):
                    pass
                else:
                    print('Training set missing.')
                    failed = True
        if not train_set_only:
            if os.path.exists(os.path.join(input_path, 'ILSVRC2012_img_val.tar')):
                print('Using cached ILSVRC2012_img_val.tar ...')
            else:
                dir_name = os.path.join(output_path, 'val')
                if enable_extract and os.path.exists(dir_name) and os.path.isdir(dir_name):
                    pass
                else:
                    print('Validation set missing.')
                    failed = True
        if failed:
            suggest_imagenet_download()
            exit(0)
    success = True
    for each_target in downloaded_files:
        if dataset_name == 'imagenet':
            continue
        if not download_impl(urls, each_target):
            success = False
    if success:
        print(f'Successfully downloaded the {print_name} dataset.')


def download_impl(urls, target_file_name):
    dir_name = os.path.abspath(input_path)
    success = False
    full_name = os.path.join(dir_name, target_file_name)
    if os.path.exists(full_name) and os.path.isfile(full_name):
        print(f'Using cached {target_file_name} ...')
        success = True
        return success
    for url in urls:
        try:
            print(f'Downloading {target_file_name} from {url} ...')
            with request.urlopen(url + target_file_name) as resp, open(full_name, 'wb') as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            print(f'{target_file_name} downloaded to {dir_name}.')
            success = True
            break
        except (error.URLError, error.HTTPError) as e:
            print(e)
    else:
        raise RuntimeError(f'All Download links failed when downloading {target_file_name}.')
    return success


def _compute_crc32(file_path, offset=0, return_str=True):
    """
    (内部函数) 计算文件 CRC32 的通用逻辑
    :param file_path: 文件路径
    :param offset: 起始字节位置 (0表示从头开始)
    """
    crc_value = 0
    chunk_size = 65536  # 64KB 分块读取

    try:
        with open(file_path, 'rb') as f:
            # 关键点：移动文件指针到第 N 个字节
            f.seek(offset)

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                crc_value = zlib.crc32(chunk, crc_value)

        # 返回 8位 16进制字符串
        if return_str:
            return f"{crc_value & 0xFFFFFFFF:08x}"
        else:
            return int(crc_value & 0xFFFFFFFF)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None


def get_full_crc32(file_path, return_str=True):
    """
    校验整个文件
    """
    return _compute_crc32(file_path, offset=0, return_str=return_str)


def get_partial_crc32(file_path, start_byte_index, return_str=True):
    """
    从文件的第 N 个字节（含）开始校验直到文件结束
    :param start_byte_index: 偏移量 (例如 100 表示跳过前100个字节，从索引100开始)
    """
    return _compute_crc32(file_path, offset=start_byte_index, return_str=return_str)


def decompress_gz_file(target_file_name):
    """
    将 .gz 文件解压为二进制文件
    """
    gz_path = os.path.join(input_path, target_file_name)
    if not os.path.exists(gz_path):
        print(f"File {gz_path} does not exist.")
        return

    output_file = os.path.join(output_path, target_file_name.replace('.gz', ''))

    print(f"Extracting {target_file_name} to {os.path.abspath(output_file)} ...")

    try:
        # 'rb' 模式读取 gzip 流
        with gzip.open(gz_path, 'rb') as f_in:
            # 'wb' 模式写入普通二进制文件
            with open(output_file, 'wb') as f_out:
                # copyfileobj 会自动分块复制，内存占用极低
                shutil.copyfileobj(f_in, f_out)
        print(f"Successfully extracted {target_file_name}.")

    except Exception as e:
        print(f"Failed to extract {target_file_name}. {e}")


def decompress_tar_gz_file(target_file_name):
    """
    将 .tar.gz 文件解压为二进制文件
    """
    file_path = os.path.join(input_path, target_file_name)
    abs_output_path = os.path.abspath(output_path)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    print(f"Extracting {target_file_name} to {abs_output_path} ...")

    try:
        with tarfile.open(file_path, "r:gz") as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=output_path, filter='data')
            else:
                tar.extractall(path=output_path)

            root_dir_name = os.path.commonprefix(tar.getnames())
            print(f"Successfully extracted {target_file_name} to {os.path.join(abs_output_path, root_dir_name)}")

    except Exception as e:
        print(f"Failed to extract {target_file_name}. {e}")


def extract_dataset():
    Path(input_path).mkdir(exist_ok=True)

    # 验证数据集的CRC-32
    if not skip_download_verification:
        success = True
        for each_target in downloaded_files:
            if dataset_name == 'imagenet' and train_set_only and each_target == 'ILSVRC2012_img_val.tar':
                continue
            if dataset_name == 'imagenet' and val_set_only and each_target == 'ILSVRC2012_img_train.tar':
                continue
            p = os.path.join(input_path, each_target)
            print(f'Verifying the downloaded {each_target} ...')
            crc32_code = get_full_crc32(p)
            if not crc32_code == correct_crc[dataset_name][each_target]:
                success = False
        if success:
            print(f'Successfully verified the {print_name} dataset.')
        else:
            print(f'Failed to verify the {print_name} dataset. Please download the dataset again.')
            exit(1)

    # 解压数据集
    if dataset_name == 'mnist':
        for each_target in downloaded_files:
            decompress_gz_file(each_target)
    elif dataset_name in ['cifar10', 'cifar100']:
        for each_target in downloaded_files:
            decompress_tar_gz_file(each_target)
    else:
        if not val_set_only:
            extract_imagenet_train_set()
        # 下载 valprep 脚本
        system_platform = platform.system()
        if not train_set_only:
            if system_platform == "Windows":
                script_name = "valprep.bat"
                download_impl(script_urls_windows, script_name)
            else:
                script_name = "valprep.sh"
                download_impl(script_urls_linux, script_name)
            extract_imagenet_val_set()


def remove_downloaded_files():
    for each_file in downloaded_files:
        full_path = os.path.join(os.path.abspath(input_path), each_file)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            os.remove(full_path)
    print('Removed downloaded files.')


def extract_imagenet_train_set():
    root_dir = Path(input_path)
    train_dir = Path(output_path) / 'train'
    tar_filename = 'ILSVRC2012_img_train.tar'
    source_tar = root_dir / tar_filename
    target_tar = train_dir / tar_filename

    # 创建 train 目录
    train_dir.mkdir(exist_ok=True)

    # 检查源文件是否存在
    if not source_tar.exists():
        print(f"Error: Cannot find {tar_filename} in {root_dir}.")
        exit(1)

    # mv ILSVRC2012_img_train.tar train/
    shutil.move(str(source_tar), str(target_tar))

    # cd train && tar -xvf ...
    print(f"Extracting the training set tar file ...")
    with tarfile.open(target_tar) as tar:
        tar.extractall(path=train_dir)

    # mv ILSVRC2012_img_train.tar ..
    shutil.move(str(target_tar), str(source_tar))

    # 处理子压缩包 (n01440764.tar 等)
    # find . -name "*.tar" | while read NAME ; do ...
    print("Extracting tar files of each class ...")
    sub_tars = list(train_dir.glob("*.tar"))
    total_sub_tars = len(sub_tars)

    for i, sub_tar in enumerate(sub_tars, 1):
        # 获取文件名作为目录名 (n01440764)
        class_dir_name = sub_tar.stem
        class_dir = train_dir / class_dir_name

        if i % 100 == 0:
            print(f"    Progress: {i / 10}%")

        # mkdir -p "${NAME%.tar}"
        class_dir.mkdir(exist_ok=True)

        # tar -xvf "${NAME}" -C "${NAME%.tar}"
        with tarfile.open(sub_tar) as tar:
            tar.extractall(path=class_dir)

        # rm -f "${NAME}"
        os.remove(sub_tar)

    print("Successfully extracted the ImageNet training set.")


def extract_imagenet_val_set():
    root_dir = Path(input_path)
    val_dir = Path(output_path) / 'val'
    tar_filename = 'ILSVRC2012_img_val.tar'
    source_tar = root_dir / tar_filename
    target_tar = val_dir / tar_filename

    # 创建 val 目录
    val_dir.mkdir(exist_ok=True)

    if not source_tar.exists():
        print(f"Error: Cannot find {tar_filename} in {root_dir}.")
        exit(1)

    # mv ILSVRC2012_img_val.tar val/
    shutil.move(str(source_tar), str(target_tar))

    # cd val && tar -xvf ...
    print(f"Extracting the validation set tar file ...")
    with tarfile.open(target_tar) as tar:
        tar.extractall(path=val_dir)

    # mv ILSVRC2012_img_val.tar ..
    shutil.move(str(target_tar), str(source_tar))

    # 处理 valprep 脚本
    system_platform = platform.system()

    if system_platform == "Windows":
        script_name = "valprep.bat"
        shell_cmd = True
    else:
        script_name = "valprep.sh"
        shell_cmd = False

    source_script = root_dir / script_name
    target_script = val_dir / script_name

    if not source_script.exists():
        print(f"Error: Cannot find script {script_name}. Skipping the organization of the validation set.")
        exit(1)

    # mv ../valprep.sh .
    print(f"Organizing the validation set ...")
    shutil.move(str(source_script), str(target_script))

    try:
        if system_platform == "Windows":
            cmd = [script_name]
            subprocess.check_call(cmd, cwd=val_dir, shell=shell_cmd)
        else:
            full_path = str(os.path.join(os.path.abspath(output_path), 'val', script_name))
            os.chmod(full_path, 0o755)
            sh_file = Path(full_path).resolve()
            subprocess.run(['bash', sh_file.name], cwd=sh_file.parent, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed to run scripts. {e}")
    else:
        print("Successfully extracted the ImageNet validation set.")
    finally:
        # mv valprep.sh ..
        if target_script.exists():
            shutil.move(str(target_script), str(source_script))


def check_imagenet_folder(trainning=True):
    """
    通过文件夹个数、文件个数、文件名简单检查ImageNet数据集合法性
    """
    dir_name = os.path.join(os.path.abspath(output_path), 'train' if trainning else 'val')
    folder_names = [p.name for p in Path(dir_name).iterdir() if p.is_dir()]
    check_code = 0
    num_subfolders = 0
    for each_name in folder_names:
        if each_name[0] == 'n':
            num_subfolders += 1
            name_id = int(each_name[1:])
            check_code = check_code ^ name_id
    if num_subfolders == 1000 and f"{check_code & 0xFFFFFFFF:08x}" == imagenet_folder_check_code:
        cnt = sum(1 for p in Path(os.path.join(dir_name, 'n02097298')).iterdir()
                  if p.is_file() and p.suffix.lower() =='.jpeg' and p.name[0] != '.')
        if trainning:
            return cnt == 1300
        else:
            return cnt == 50
    else:
        return False


def search_for_file(dir_path, file_name):
    target_path = Path(dir_path)
    pass
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file == file_name:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(target_path)
                return str(relative_path)
    return None


def check_dataset_folder():
    file_dict = dict()
    if dataset_name == 'cifar10':
        targets = cifar10_extracted_targets
    elif dataset_name == 'cifar100':
        targets = cifar100_extracted_targets
    else:
        targets = mnist_extracted_targets
    for each_file in targets:
        rel_path = search_for_file(os.path.abspath(output_path), each_file)
        if rel_path:
            file_dict[each_file] = rel_path
        else:
            print(f'Error: Invalid {print_name} folder.')
            exit(1)
    return file_dict


def read_cifar_bin(file_path):
    # 1. 定义每条数据的字节结构
    if dataset_name == 'cifar10':
        label_bytes = 1
        data_bytes = 3072  # 32*32*3
    elif dataset_name == 'cifar100':
        label_bytes = 2  # coarse + fine
        data_bytes = 3072
    else:
        raise ValueError("Invalid dataset type")

    record_bytes = label_bytes + data_bytes

    # 2. 读取二进制文件
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return None, None

    # 直接读入 uint8 数组
    buffer = np.fromfile(file_path, dtype=np.uint8)

    # 3. 重塑数组 (N, record_length)
    num_images = len(buffer) // record_bytes
    buffer = buffer.reshape(num_images, record_bytes)

    # 4. 切片分离标签和图像
    if dataset_name == 'cifar10':
        labels = buffer[:, 0]  # 第1个字节是标签
        img_flat = buffer[:, 1:]
    else:
        # CIFAR-100: Byte 0=Coarse, Byte 1=Fine
        labels = buffer[:, 1]  # 通常只需要细分类标签
        img_flat = buffer[:, 2:]

    # 5. 转换图像格式
    # 原始: (N, 3072) 平铺, 顺序 RRR...GGG...BBB...
    # 目标: (N, 32, 32, 3) 标准图像格式
    images = img_flat.reshape(num_images, 3, 32, 32).transpose(0, 2, 3, 1)

    return images, labels


def load_mnist_images(filename):
    """
    读取 MNIST 图像文件 ('train-images-idx3-ubyte' / 't10k-images-idx3-ubyte')
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    with open(filename, 'rb') as f:
        # 1. 读取文件头 (16字节)
        # >IIII: > 表示大端序, I 表示 unsigned int (4字节)
        # 分别是: Magic Number, 图片数量, 行数, 列数
        bin_data = f.read(16)
        magic, num_images, rows, cols = struct.unpack('>IIII', bin_data)

        if magic != 2051:  # 0x00000803
            raise ValueError(f"Invalid magic number {magic} in image file")

        # 2. 读取像素数据
        # 剩下的数据就是像素本身，直接用 numpy 读入
        data = np.fromfile(f, dtype=np.uint8)

        # 3. 重塑形状 (N, 28, 28)
        # 注意: MNIST 是单通道灰度图，通常不需要最后一维的 1，
        # 如果为了兼容 CIFAR 格式，可以 reshape 为 (num_images, rows, cols, 1)
        images = data.reshape(num_images, rows, cols, 1)

    return images


def load_mnist_labels(filename):
    """
    读取 MNIST 标签文件 ('train-labels-idx1-ubyte' / 't10k-labels-idx1-ubyte')
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    with open(filename, 'rb') as f:
        # 1. 读取文件头 (8字节)
        # Magic Number, 标签数量
        bin_data = f.read(8)
        magic, num_items = struct.unpack('>II', bin_data)

        if magic != 2049:  # 0x00000801
            raise ValueError(f"Invalid magic number {magic} in label file")

        # 2. 读取标签数据
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


def export_to_bytes(images, labels):
    """
    将 NHWC 格式的图像数组和标签数组导出为 BytesIO 流。
    """

    # 1. 安全检查与类型转换
    # 确保是 uint8 类型，节省空间且符合图像标准
    if images.dtype != np.uint8:
        images = images.astype(np.uint8)

    # 确保标签也是 uint8 (CIFAR 类别数 < 255)
    if labels.dtype != np.uint8:
        labels = labels.astype(np.uint8)

    # 2. 转换为二进制
    # numpy.tobytes() 默认使用 'C' order (C语言顺序/行优先)
    # 对于 (N, H, W, C) 的数组：
    # 最内层维度是 C (R,G,B 紧挨着)
    # 其次是 W (一行中的像素)
    # 再次是 H (不同行)
    # 最外层是 N (不同图片)
    # 这完全符合 "最里面是颜色通道维度C...然后是下一张图片" 的要求
    img_bytes = images.tobytes()
    lbl_bytes = labels.tobytes()

    # 3. 封装进 BytesIO
    img_stream = BytesIO(img_bytes)
    lbl_stream = BytesIO(lbl_bytes)

    return img_stream.getvalue(), lbl_stream.getvalue()


def concat_mnist():
    target_file_dict = check_dataset_folder()
    train_images = load_mnist_images(os.path.join(os.path.abspath(output_path), target_file_dict['train-images-idx3-ubyte']))
    train_labels = load_mnist_labels(os.path.join(os.path.abspath(output_path), target_file_dict['train-labels-idx1-ubyte']))
    test_images = load_mnist_images(os.path.join(os.path.abspath(output_path), target_file_dict['t10k-images-idx3-ubyte']))
    test_labels = load_mnist_labels(os.path.join(os.path.abspath(output_path), target_file_dict['t10k-labels-idx1-ubyte']))
    write_cifar_mnist_dts(train_images, train_labels, train=True)
    write_cifar_mnist_dts(test_images, test_labels, train=False)

    if remove_source:
        for each in target_file_dict.values():
            remove_target = os.path.join(os.path.abspath(output_path), each)
            if os.path.exists(remove_target) and os.path.isfile(remove_target):
                os.remove(remove_target)

    print(f'Successfully exported mnist_train.dts and mnist_test.dts to {output_path}.')


def concat_cifar10():
    target_file_dict = check_dataset_folder()
    train_image_temp = list()
    train_label_temp = list()
    test_images, test_labels = read_cifar_bin(os.path.join(os.path.abspath(output_path), target_file_dict['test_batch.bin']))
    for i in range(1, 6):
        full_path = os.path.join(os.path.abspath(output_path), target_file_dict[f'data_batch_{i}.bin'])
        images, labels = read_cifar_bin(full_path)
        train_image_temp.append(images)
        train_label_temp.append(labels)
    train_images = np.concatenate(train_image_temp, axis=0)
    train_labels = np.concatenate(train_label_temp, axis=0)
    write_cifar_mnist_dts(train_images, train_labels, train=True)
    write_cifar_mnist_dts(test_images, test_labels, train=False)

    if remove_source:
        remove_target = os.path.join(os.path.abspath(output_path), 'cifar-10-batches-bin')
        if os.path.exists(remove_target) and os.path.isdir(remove_target):
            shutil.rmtree(remove_target)

    print(f'Successfully exported cifar10_train.dts and cifar10_test.dts to {output_path}.')


def concat_cifar100():
    target_file_dict = check_dataset_folder()
    train_images, train_labels = read_cifar_bin(os.path.join(os.path.abspath(output_path), target_file_dict['train.bin']))
    test_images, test_labels = read_cifar_bin(os.path.join(os.path.abspath(output_path), target_file_dict['test.bin']))
    write_cifar_mnist_dts(train_images, train_labels, train=True)
    write_cifar_mnist_dts(test_images, test_labels, train=False)

    if remove_source:
        remove_target = os.path.join(os.path.abspath(output_path), 'cifar-100-binary')
        if os.path.exists(remove_target) and os.path.isdir(remove_target):
            shutil.rmtree(remove_target)

    print(f'Successfully exported cifar100_train.dts and cifar100_test.dts to {output_path}.')


def write_cifar_mnist_dts(images, labels, train):
    image_bytes, label_bytes = export_to_bytes(images, labels)
    main_stream = BytesIO()
    main_stream.write(make_empty_file_header(CIFAR_MNIST_HEADER_SIZE))
    main_stream.seek(0, SEEK_END)
    main_stream.write(label_bytes)
    main_stream.seek(0, SEEK_END)
    main_stream.write(image_bytes)
    target_path = os.path.join(output_path, dataset_name + ('_train.dts' if train else '_test.dts'))
    with open(target_path, 'wb') as f:
        f.write(main_stream.getvalue())
    if dataset_name in ['cifar10', 'cifar100']:
        modify_file_header(target_path, make_cifar_file_header(file_path=target_path, train=train))
    else:
        modify_file_header(target_path, make_mnist_file_header(file_path=target_path, train=train))


def get_file_dicts(root_dir):
    file_name_dict = dict()
    file_num_dict = dict()
    for root, dirs, files in os.walk(root_dir):
        # 获取当前子目录的名称（即字典的 key）
        subdir_name = os.path.basename(root)

        # 这是一个简单的保护措施，防止把 'train' 根目录本身也作为一个 key 加入
        if root == root_dir:
            continue

        # 使用列表推导式筛选 .jpg 文件 (使用 lower() 兼容 .JPG 和 .jpg)
        jpg_files = [f for f in files if f.lower().endswith('.jpeg') and f[0] != '.']

        # 如果该目录下有 jpg 文件，则存入字典
        if jpg_files:
            file_name_dict[subdir_name] = sorted(jpg_files)
            file_num_dict[subdir_name] = len(jpg_files)

    return file_name_dict, file_num_dict


# def concat_imagenet_train(train_file_name_list):
#     random.seed(RAND_SEED)
#     print('Start concatenating the ImageNet training set ...')
#
#     process_num = 2048  # TODO:
#     a = train_file_name_list
#     random.shuffle(a)
#
#     part = a[:process_num]  # TODO:
#     for i, each in enumerate(part):  # TODO:
#         src_path = os.path.join(output_path, 'train', each[1], each[2])  # TODO:
#
#         # folder_name = 'COPY'
#         folder_name = f'LEVEL{compress_level}'
#         if not os.path.exists(os.path.join(output_path, folder_name)):
#             os.makedirs(os.path.join(output_path, folder_name))
#
#         # Copy
#         # dst_path = os.path.join(output_path, folder_name, f'{i}.jpg')
#         # shutil.copy(src_path, dst_path)
#
#         dst_path = os.path.join(output_path, f'LEVEL{compress_level}', f'{i}.jpg')  # TODO:
#         byte_stream = process_imagenet_image(src_path)  # TODO:
#         with open(dst_path, 'wb') as f:  # TODO:
#             f.write(byte_stream)  # TODO:


def process_imagenet_image(src_path):
    if compress_level == 0:  # 压缩级别为0时不做任何处理，直接读取原文件字节流
        with open(src_path, 'rb') as f:
            file_content = f.read()
            buf = BytesIO(file_content)
            return buf.getvalue(), 0

    else:  # 压缩级别为1、2、3时，均为有损压缩
        with Image.open(src_path) as im:
            w, h = im.size
            if w > h:
                if h > crop_boundary:
                    scale = SHORT_LIMIT * 1.0 / h  # 短边缩放到SHORT_LIMIT
                    new_w, new_h = int(w * scale), int(h * scale)
                    im = im.resize((new_w, new_h), algorithm)
                else:
                    new_w, new_h = w, h
                if new_w <= LONG_LIMIT and new_h <= SHORT_LIMIT:
                    pass  # 缩放后合规就不裁剪
                else:
                    if compress_level >= 2:  # 压缩级别2以上才裁剪
                        final_w = min(new_w, LONG_LIMIT)
                        final_h = min(new_h, SHORT_LIMIT)
                        margin_w = (new_w - final_w) // 2
                        margin_h = (new_h - final_h) // 2
                        left = margin_w
                        top = margin_h
                        right = margin_w + final_w
                        bottom = margin_h + final_h
                        box = (left, top, right, bottom)
                        im = im.crop(box)
            else:
                if w > crop_boundary:
                    scale = SHORT_LIMIT * 1.0 / w  # 短边缩放到SHORT_LIMIT
                    new_w, new_h = int(w * scale), int(h * scale)
                    im = im.resize((new_w, new_h), algorithm)
                else:
                    new_w, new_h = w, h
                if new_w <= SHORT_LIMIT and new_h <= LONG_LIMIT:
                    pass  # 缩放后合规就不裁剪
                else:
                    if compress_level >= 2:  # 压缩级别2以上才裁剪
                        final_w = min(new_w, SHORT_LIMIT)
                        final_h = min(new_h, LONG_LIMIT)
                        margin_w = (new_w - final_w) // 2
                        margin_h = (new_h - final_h) // 2
                        left = margin_w
                        top = margin_h
                        right = margin_w + final_w
                        bottom = margin_h + final_h
                        box = (left, top, right, bottom)
                        im = im.crop(box)

            output_w, output_h = im.size

            buf = BytesIO()
            if im.mode in ('RGBA', 'LA', 'P'):
                im = im.convert('RGB')
            im.save(buf, format='JPEG', quality=jpeg_quality)
            return buf.getvalue(), output_w * output_h


def concat_imagenet():
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    print('Checking ImageNet directories ...')
    if not val_set_only:
        if (not os.path.exists(os.path.join(output_path, 'train_file_name_list.pkl'))
                or not os.path.exists(os.path.join(output_path, 'train_info_list.pkl'))):
            if os.path.exists(train_dir) and check_imagenet_folder(trainning=True):
                print('Training set OK.')
            else:
                print('Error: Invalid training set directory.')
                exit(1)
            train_file_name_dict, train_file_num_dict = get_file_dicts(train_dir)
            train_file_num_list = list(train_file_num_dict.values())
            train_class_name_list = sorted(list(train_file_num_dict.keys()))
            class_dict = dict(zip(train_class_name_list, [i for i in range(1000)]))
            total_train_file_num = sum(train_file_num_list)
            if total_train_file_num != 1281167:
                print('Error: Invalid training set directory.')
                exit(1)
            flat_data = ((key, item) for key, lst in train_file_name_dict.items() for item in lst)
            train_file_name_list = [[i, key, item, class_dict[key]] for i, (key, item) in enumerate(flat_data)]
            train_file_name_list.sort(key=lambda t: t[2])
            for k, v in enumerate(train_file_name_list):
                v[0] = k
            save_list(train_file_name_list, os.path.join(os.path.abspath(output_path),
                                                         'train_file_name_list.pkl'))
            train_info_list = create_info_list(train_file_name_list, train=True)
            save_list(train_info_list, os.path.join(os.path.abspath(output_path), 'train_info_list.pkl'))

        else:
            train_file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'train_file_name_list.pkl'))
            train_info_list = load_list(os.path.join(os.path.abspath(output_path), 'train_info_list.pkl'))
        if compress_level == 0:
            greedy_arrange(train_info_list, MINIMUM_TRAIN_BLOCKS_LV0, train=True)
            write_imagenet_dts_lv0(train=True)
        else:
            write_imagenet_dts_lv123(train=True)

    if not train_set_only:
        if (not os.path.exists(os.path.join(output_path, 'val_file_name_list.pkl'))
                or not os.path.exists(os.path.join(output_path, 'val_info_list.pkl'))):
            if os.path.exists(val_dir) and check_imagenet_folder(trainning=False):
                print('Validation set OK.')
            else:
                print('Error: Invalid validation set directory.')
                exit(1)
            val_file_name_dict, val_file_num_dict = get_file_dicts(val_dir)
            val_file_num_list = list(val_file_num_dict.values())
            val_class_name_list = sorted(list(val_file_num_dict.keys()))
            class_dict = dict(zip(val_class_name_list, [i for i in range(1000)]))
            total_val_file_num = sum(val_file_num_list)
            if total_val_file_num != 50000:
                print('Error: Invalid validation set directory.')
                exit(1)

            flat_data = ((key, item) for key, lst in val_file_name_dict.items() for item in lst)
            val_file_name_list = [[i, key, item, class_dict[key]] for i, (key, item) in enumerate(flat_data)]
            val_file_name_list.sort(key=lambda t: t[2])
            for k, v in enumerate(val_file_name_list):
                v[0] = k
            save_list(val_file_name_list, os.path.join(os.path.abspath(output_path),
                                                         'val_file_name_list.pkl'))
            val_info_list = create_info_list(val_file_name_list, train=False)
            save_list(val_info_list, os.path.join(os.path.abspath(output_path), 'val_info_list.pkl'))

        else:
            val_file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'val_file_name_list.pkl'))
            val_info_list = load_list(os.path.join(os.path.abspath(output_path), 'val_info_list.pkl'))

        if compress_level == 0:
            greedy_arrange(val_info_list, MINIMUM_VAL_BLOCKS_LV0, train=False)
            write_imagenet_dts_lv0(train=False)
        else:
            write_imagenet_dts_lv123(train=False)


def create_info_list(file_name_list, train):
    info_list = list()
    pref = os.path.join(os.path.abspath(output_path), 'train' if train else 'val')
    for each in file_name_list:
        p = os.path.join(pref, each[1], each[2])
        s = os.path.getsize(p)
        info_list.append([each[0], s])
    return info_list


def save_list(obj, file_path: str | Path, *, overwrite=True):
    file_path = Path(file_path)
    if file_path.exists() and not overwrite:
        raise FileExistsError(file_path)
    with file_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_list(file_path: str | Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def make_blocks_l0(block_size, header_size, minimum_num_blocks):
    blocks = list()
    block_margin = [block_size - header_size] * minimum_num_blocks
    return blocks, block_margin


def ffd_fill(blocks, block_margin, info_list, minimum_num_blocks):
    info_list.sort(reverse=False, key=lambda t: t[1])  # 由小到大排序
    for i in range(minimum_num_blocks):
        blocks.append([info_list[-1]])
        block_margin[i] -= info_list[-1][1]
        info_list.pop()


def random_fill(blocks, block_margin, info_list, minimum_num_blocks):
    random.seed(RAND_SEED)
    random.shuffle(info_list)
    block_index = 0
    while block_index < minimum_num_blocks:
        while True:
            if info_list[-1][1] <= block_margin[block_index]:  # 如果block还能容纳
                blocks[block_index].append(info_list[-1])
                block_margin[block_index] -= info_list[-1][1]  # 扣除剩余空间
                info_list.pop()  # 从尾部排除
            else:
                break
        block_index += 1  # 指向下一个block


def marginal_fill(blocks, block_margin, info_list, minimum_num_blocks):
    info_list.sort(reverse=True, key=lambda t: t[1])  # 依然是由大到小排序
    block_index = 0
    while block_index < minimum_num_blocks:
        if info_list[-1][1] <= block_margin[block_index]:  # 如果block还能容纳
            blocks[block_index].append(info_list[-1])
            block_margin[block_index] -= info_list[-1][1]  # 扣除剩余空间
            info_list.pop()  # 从尾部排除
        else:  # 容纳不了，就什么也不干，直接指向下一个block
            pass
        block_index += 1


def extra_block_fill(blocks, block_margin, info_list, minimum_num_blocks):
    random.seed(RAND_SEED)
    random.shuffle(info_list)
    block_index = minimum_num_blocks - 1
    while info_list:  # 一直循环到所有图片装完
        if info_list[-1][1] <= block_margin[block_index]:  # 如果block还能容纳
            blocks[block_index].append(info_list[-1])
            block_margin[block_index] -= info_list[-1][1]  # 扣除剩余空间
            info_list.pop()  # 从尾部排除
        else:  # 容纳不了，就新增一个block
            blocks.append(list())
            block_margin.append(BLOCK_SIZE - HEADER_SIZE_LV0)
            block_index += 1


def apply_pic_alignment(pic_size_byte, alignment_byte):
    margin = pic_size_byte % alignment_byte
    return pic_size_byte - margin + alignment_byte if margin != 0 else pic_size_byte


def apply_pic_alignment_to_all(info_list, alignment_byte):
    for each in info_list:
        each[1] = apply_pic_alignment(each[1], alignment_byte)


def report_block_info(blocks, block_margin, original_size, train):
    print('*' * 45)
    if train:
        print('Training set blocks info:')
    else:
        print('Validation set blocks info:')
    print('*' * 45)
    print('Number of blocks:', len(blocks))
    print('*' * 45)
    print('Maximum block margins:')
    print(f'{max(block_margin) / 1024.0:.2f} KB')
    print('Average block margins:')
    print(f'{sum(block_margin) / len(block_margin) / 1024.0:.2f} KB')

    print('*' * 45)
    total_file_size = len(block_margin) * BLOCK_SIZE
    print(f'Total file size: {total_file_size} bytes.')
    size_inc = (total_file_size / sum(original_size) - 1.0) * 100.0
    print(f'Size increment: {size_inc:.2f}%')

    print('*' * 45)
    num_pic_per_block = list()
    for each in blocks:
        num_pic_per_block.append(len(each))
    print(f'Maximum number of pictures per block: {max(num_pic_per_block)}')
    print(f'Average number of pictures per block: {sum(num_pic_per_block) / len(num_pic_per_block):.2f}')
    print('*' * 45)


def greedy_arrange(info_list, minimum_num_blocks, train):
    random.seed(RAND_SEED)
    # 数据集大小信息列表
    # [index, file_size]
    train_size_original = [s[1] for s in info_list]

    if USING_PIC_ALIGNMENT:
        apply_pic_alignment_to_all(info_list, PIC_ALIGNMENT_BYTES)

    # 第一步：建立blocks
    blocks, block_margin = make_blocks_l0(BLOCK_SIZE, HEADER_SIZE_LV0, minimum_num_blocks)
    # 当前训练集block平均余量：16380.00 KB
    # 当前验证集block平均余量：16380.00 KB

    # 第二步，First-Fit Decreasing（FFD）填充
    ffd_fill(blocks, block_margin, info_list, minimum_num_blocks)
    # 当前训练集block平均余量：15247.79 KB
    # 当前验证集block平均余量：15441.91 KB

    # 第三步：随机填充
    random_fill(blocks, block_margin, info_list, minimum_num_blocks)
    # 当前训练集block平均余量：69.99 KB
    # 当前验证集block平均余量：70.00 KB

    # 第四步：余量填充（这一步非常强大，反复对余量空间进行填充，直至所有的block都再也装不下任何的图片）
    for k in range(MARGINAL_FILL_ITERATIONS):
        # 按余量从小到大重排
        block_margin, blocks = map(list, (zip(*sorted(zip(block_margin, blocks)))))
        marginal_fill(blocks, block_margin, info_list, minimum_num_blocks)
    # 当前训练集block平均余量：11.07 KB
    # 当前验证集block平均余量：28.87 KB

    # 第五步：新增block以容纳剩余图片
    extra_block_fill(blocks, block_margin, info_list, minimum_num_blocks)
    # 当前训练集block平均余量：13.19 KB
    # 当前验证集block平均余量：43.54 KB

    # report_block_info(blocks, block_margin, train_size_original, train=train)

    if train:
        save_list(blocks, os.path.join(output_path, 'train_blocks.pkl'))
        save_list(block_margin, os.path.join(output_path, 'train_block_margin.pkl'))
    else:
        save_list(blocks, os.path.join(output_path, 'val_blocks.pkl'))
        save_list(block_margin, os.path.join(output_path, 'val_block_margin.pkl'))

    return blocks, block_margin

# *********************************************
# Training set blocks info:
# *********************************************
# Number of blocks: 8768
# *********************************************
# Maximum block margins:
# 13860.31 KB
# Average block margins:
# 13.19 KB
# *********************************************
# Total file size: 147102629888 bytes.
# Size increment: 0.13%
# *********************************************
# Maximum number of pictures per block: 177
# Average number of pictures per block: 146.12
# *********************************************
# *********************************************
# Validation set blocks info:
# *********************************************
# Number of blocks: 401
# *********************************************
# Maximum block margins:
# 5912.88 KB
# Average block margins:
# 43.54 KB
# *********************************************
# Total file size: 6727663616 bytes.
# Size increment: 0.31%
# *********************************************
# Maximum number of pictures per block: 142
# Average number of pictures per block: 124.69
# *********************************************


def concat_dataset():
    if dataset_name == 'mnist':
        concat_mnist()
    elif dataset_name == 'cifar10':
        concat_cifar10()
    elif dataset_name == 'cifar100':
        concat_cifar100()
    else:
        concat_imagenet()


def make_empty_file_header(header_size):
    stream = BytesIO()
    data = struct.pack('<4sBBBB',
                       FILE_MAGIC,  # 4s
                       DTS_VERSION[0],  # B
                       DTS_VERSION[1],  # B
                       DTS_VERSION[2],  # B
                       DTS_VERSION[3]   # B
                       )
    stream.write(data)
    stream.write(b'\x00' * (header_size - 8))
    return stream.getvalue()


def make_mnist_file_header(file_path, train):
    stream = BytesIO()
    dataset_type = b'   MNIST'
    is_training_set = 1 if train else 0
    val_set_prep = 1 if preprocess_val_set else 0
    num_classes = 10
    tensor_layout = b'NHWC'
    image_width, image_height = 28, 28
    num_channels = 1
    color_channel_type = b'GRAY'
    num_samples = 60000 if train else 10000
    num_volumes = 1
    volume_id = 0
    num_blocks = 1
    block_bytes = num_samples * (28 * 28 * 3 + 1)
    total_bytes = CIFAR_MNIST_HEADER_SIZE + block_bytes
    block_header_size = 0
    pic_alignment = 0
    maximum_pic_area = 28 * 28
    max_pic_per_block = num_samples
    compression_ratio = 1.0
    normalize_mean = [0.0, 0.0, 0.0]
    normalize_std = [0.0, 0.0, 0.0]
    crc_code = get_partial_crc32(file_path, CIFAR_MNIST_HEADER_SIZE, return_str=False)

    # 144 Bytes
    data = struct.pack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI',
                       FILE_MAGIC,  # 4s
                       DTS_VERSION[0],  # B
                       DTS_VERSION[1],  # B
                       DTS_VERSION[2],  # B
                       DTS_VERSION[3],  # B
                       dataset_type,    # 8s
                       is_training_set, # I
                       compress_level,  # I
                       val_set_prep,    # I
                       num_classes,     # I
                       tensor_layout,   # 4s
                       image_width,     # I
                       image_height,    # I
                       num_channels,    # I
                       color_channel_type,  # 4s
                       num_samples,     # I
                       num_volumes,     # I
                       volume_id,       # I
                       num_blocks,      # I
                       num_blocks,      # I
                       total_bytes,       # Q
                       CIFAR_MNIST_HEADER_SIZE,      # I
                       block_bytes,      # Q
                       block_bytes,       # I
                       block_header_size, # I
                       pic_alignment,    # I
                       maximum_pic_area, # I
                       max_pic_per_block,  # I
                       compression_ratio,  # f
                       normalize_mean[0],  # f
                       normalize_mean[1],  # f
                       normalize_mean[2],  # f
                       normalize_std[0],  # f
                       normalize_std[1],  # f
                       normalize_std[2],  # f
                       crc_code,          # I
                       )
    stream.write(data)
    stream.seek(0, SEEK_END)
    stream.write(b'\x00' * (CIFAR_MNIST_HEADER_SIZE - 144))
    return stream.getvalue()


def make_cifar_file_header(file_path, train):
    stream = BytesIO()
    dataset_type = b' CIFAR10' if dataset_name == 'cifar10' else b'CIFAR100'
    is_training_set = 1 if train else 0
    val_set_prep = 1 if preprocess_val_set else 0
    num_classes = 10 if dataset_name == 'cifar10' else 100
    tensor_layout = b'NHWC'
    image_width, image_height = 32, 32
    num_channels = 3
    color_channel_type = b' RGB'
    num_samples = 50000 if train else 10000
    num_volumes = 1
    volume_id = 0
    num_blocks = 1
    block_bytes = num_samples * (32 * 32 * 3 + 1)
    total_bytes = CIFAR_MNIST_HEADER_SIZE + block_bytes
    block_header_size = 0
    pic_alignment = 0
    maximum_pic_area = 32 * 32
    max_pic_per_block = num_samples
    compression_ratio = 1.0
    normalize_mean = [0.0, 0.0, 0.0]
    normalize_std = [0.0, 0.0, 0.0]
    crc_code = get_partial_crc32(file_path, CIFAR_MNIST_HEADER_SIZE, return_str=False)

    # 144 Bytes
    data = struct.pack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI',
                       FILE_MAGIC,  # 4s
                       DTS_VERSION[0],  # B
                       DTS_VERSION[1],  # B
                       DTS_VERSION[2],  # B
                       DTS_VERSION[3],  # B
                       dataset_type,    # 8s
                       is_training_set, # I
                       compress_level,  # I
                       val_set_prep,    # I
                       num_classes,     # I
                       tensor_layout,   # 4s
                       image_width,     # I
                       image_height,    # I
                       num_channels,    # I
                       color_channel_type,  # 4s
                       num_samples,     # I
                       num_volumes,     # I
                       volume_id,       # I
                       num_blocks,      # I
                       num_blocks,      # I
                       total_bytes,       # Q
                       CIFAR_MNIST_HEADER_SIZE,      # I
                       block_bytes,      # Q
                       block_bytes,       # I
                       block_header_size, # I
                       pic_alignment,    # I
                       maximum_pic_area, # I
                       max_pic_per_block,  # I
                       compression_ratio,  # f
                       normalize_mean[0],  # f
                       normalize_mean[1],  # f
                       normalize_mean[2],  # f
                       normalize_std[0],  # f
                       normalize_std[1],  # f
                       normalize_std[2],  # f
                       crc_code,          # I
                       )
    stream.write(data)
    stream.write(b'\x00' * (CIFAR_MNIST_HEADER_SIZE - 144))
    return stream.getvalue()


def make_imagenet_file_header(file_path, train):
    if train:
        file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'train_file_name_list.pkl'))
        info_list = load_list(os.path.join(os.path.abspath(output_path), 'train_info_list.pkl'))
        blocks = load_list(os.path.join(output_path, 'train_blocks.pkl'))
    else:
        file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'val_file_name_list.pkl'))
        info_list = load_list(os.path.join(os.path.abspath(output_path), 'val_info_list.pkl'))
        blocks = load_list(os.path.join(output_path, 'val_blocks.pkl'))

    stream = BytesIO()
    dataset_type = b'IMAGENET'
    is_training_set = 1 if train else 0
    val_set_prep = 1 if preprocess_val_set else 0
    num_classes = 1000
    tensor_layout = b'NHWC'
    image_width, image_height = 0, 0  # Default
    num_channels = 3
    color_channel_type = b' RGB'
    num_samples = len(file_name_list)
    num_volumes = 1
    volume_id = 0
    num_blocks = len(blocks)
    total_bytes = num_blocks * BLOCK_SIZE + IMAGENET_HEADER_SIZE
    block_bytes = num_blocks * BLOCK_SIZE
    if compress_level == 0:
        block_header_size = HEADER_SIZE_LV0
    else:
        block_header_size = HEADER_SIZE_LV123
    pic_alignment = PIC_ALIGNMENT_BYTES if USING_PIC_ALIGNMENT else 0
    maximum_pic_area = 31667328 if train else 18248230
    max_pic_per_block = max([len(s) for s in blocks])
    compression_ratio = total_bytes / sum([s[1] for s in info_list])
    normalize_mean = [0.0, 0.0, 0.0]
    normalize_std = [0.0, 0.0, 0.0]
    crc_code = get_partial_crc32(file_path, IMAGENET_HEADER_SIZE, return_str=False)

    # 144 Bytes
    data = struct.pack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI',
                       FILE_MAGIC,  # 4s
                       DTS_VERSION[0],  # B
                       DTS_VERSION[1],  # B
                       DTS_VERSION[2],  # B
                       DTS_VERSION[3],  # B
                       dataset_type,    # 8s
                       is_training_set, # I
                       compress_level,  # I
                       val_set_prep,    # I
                       num_classes,     # I
                       tensor_layout,   # 4s
                       image_width,     # I
                       image_height,    # I
                       num_channels,    # I
                       color_channel_type,  # 4s
                       num_samples,     # I
                       num_volumes,     # I
                       volume_id,       # I
                       num_blocks,      # I
                       num_blocks,      # I
                       total_bytes,       # Q
                       IMAGENET_HEADER_SIZE,      # I
                       block_bytes,      # Q
                       BLOCK_SIZE,       # I
                       block_header_size, # I
                       pic_alignment,    # I
                       maximum_pic_area, # I
                       max_pic_per_block,  # I
                       compression_ratio,  # f
                       normalize_mean[0],  # f
                       normalize_mean[1],  # f
                       normalize_mean[2],  # f
                       normalize_std[0],  # f
                       normalize_std[1],  # f
                       normalize_std[2],  # f
                       crc_code,          # I
                       )
    stream.write(data)
    stream.write(b'\x00' * (IMAGENET_HEADER_SIZE - 144))
    return stream.getvalue()


def load_dts_header_info(file_path):
    """ 读取文件头，显示数据集信息 """
    header_size = 144
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DTS file not found: {file_path}")
    with open(file_path, 'rb') as f:
        header_data = f.read(256)
        if len(header_data) < 256:
            raise ValueError(f"Invalid DTS file.")
        header = struct.unpack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI', header_data[:header_size])
        magic, version_0, version_1, version_2, _, \
        dataset_type, is_training_set, compress_level_read, val_set_prep, num_classes, \
        tensor_layout, image_width, image_height, num_channels, color_channel_type, \
        num_samples, num_volumes, volume_id, total_blocks, num_blocks, \
        total_bytes, header_bytes, block_bytes, bytes_per_block, block_header_size, \
        pic_alignment, maximum_pic_area, max_pic_per_block, compression_ratio, \
        normalize_mean_0, normalize_mean_1, normalize_mean_2, \
        normalize_std_0, normalize_std_1, normalize_std_2, crc_code = header
        print('Magic:\t\t\t', f'{magic}')
        print('Version:\t\t', f'{version_0:d}.{version_1:d}.{version_2:d}')
        print('Dataset type:\t\t', f'{dataset_type}')
        print('Is training set:\t', 'True' if is_training_set else 'False')
        print('Compression level:\t', f'{compress_level_read}')
        print('Val set preprocess:\t', 'True' if val_set_prep else 'False')
        print('Number of classes:\t', f'{num_classes}')
        print('Tensor layout:\t\t', f'{tensor_layout}')
        print('Image width:\t\t', f'{image_width}')
        print('Image height:\t\t', f'{image_height}')
        print('Number of channels:\t', f'{num_channels}')
        print('Color channel type:\t', f'{color_channel_type[1:]}')
        print('Number of samples:\t', f'{num_samples}')
        print('Number of volumes:\t', f'{num_volumes}')
        print('Volume ID:\t\t', f'{volume_id}')
        print('Total blocks:\t\t', f'{total_blocks}')
        print('Blocks in this file:\t', f'{num_blocks}')
        print('Total bytes:\t\t', f'{total_bytes}')
        print('Header bytes:\t\t', f'{header_bytes}')
        print('Block bytes:\t\t', f'{block_bytes}')
        print('Bytes per block:\t', f'{bytes_per_block}')
        print('Block header size:\t', f'{block_header_size}')
        print('Picture alignment:\t', f'{pic_alignment}')
        print('Max picture area:\t', f'{maximum_pic_area}')
        print('Max num of pic per block: ', f'{max_pic_per_block}')
        print('Compression ratio:\t', f'{100 * compression_ratio:.2f}')
        print('Normalization mean R:\t', f'{normalize_mean_0:.2f}')
        print('Normalization mean G:\t', f'{normalize_mean_1:.2f}')
        print('Normalization mean B:\t', f'{normalize_mean_2:.2f}')
        print('Normalization std R:\t', f'{normalize_std_0:.2f}')
        print('Normalization std G:\t', f'{normalize_std_1:.2f}')
        print('Normalization std B:\t', f'{normalize_std_2:.2f}')
        print('CRC-32 code:\t\t', f'{crc_code & 0xFFFFFFFF:08x}')


def modify_file_header(file_path, byte_stream):
    with open(file_path, 'r+b') as f:
        f.seek(0)
        f.write(byte_stream)


def make_imagenet_blocks(block_id, train):
    if compress_level == 0:
        block_stream = BytesIO()
        block_header_steam = BytesIO()
        if train:
            # file_name_list: index, folder_name, jpeg_name, label
            file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'train_file_name_list.pkl'))
            info_list = load_list(os.path.join(os.path.abspath(output_path), 'train_info_list.pkl'))
            blocks = load_list(os.path.join(output_path, 'train_blocks.pkl'))
        else:
            file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'val_file_name_list.pkl'))
            info_list = load_list(os.path.join(os.path.abspath(output_path), 'val_info_list.pkl'))
            blocks = load_list(os.path.join(output_path, 'val_blocks.pkl'))
        info_list.sort()
        # [index, file_size]
        current_block = blocks[block_id]
        num_pics = len(current_block)
        offset = HEADER_SIZE_LV0
        offset_list = list()
        size_list = list()
        label_list = list()
        for each_pic in current_block:
            offset_list.append(offset)
            offset += each_pic[1]
            pic_id = each_pic[0]
            label_list.append(file_name_list[pic_id][3])
            pic_size = info_list[pic_id][1]
            size_list.append(pic_size)
            padding = each_pic[1] - pic_size
            if not info_list[pic_id][0] == file_name_list[pic_id][0] == pic_id:
                raise IndexError
            src_path = os.path.join(output_path, 'train' if train else 'val', file_name_list[pic_id][1], file_name_list[pic_id][2])
            with open(src_path, 'rb') as f:
                file_content = f.read()
                buf = BytesIO(file_content)
                buf.seek(0)
                block_stream.seek(0, SEEK_END)
                block_stream.write(buf.read())
                block_stream.seek(0, SEEK_END)
                block_stream.write(b'\x00' * padding)

        fmt = f'<{len(offset_list)}I'
        header_data = struct.pack('<4sII', BLOCK_MAGIC_LV0, block_id, num_pics)
        block_header_steam.write(header_data)
        block_header_steam.seek(0, SEEK_END)
        block_header_steam.write(struct.pack(fmt, *offset_list))
        block_header_steam.seek(0, SEEK_END)
        block_header_steam.write(struct.pack(fmt, *size_list))
        block_header_steam.seek(0, SEEK_END)
        block_header_steam.write(struct.pack(fmt, *label_list))
        block_header_steam.seek(0, SEEK_END)
        padding_header = HEADER_SIZE_LV0 - 12 - 3 * num_pics * 4
        block_header_steam.write(b'\x00' * padding_header)

        block_header_steam.seek(0, SEEK_END)
        block_stream.seek(0)
        block_header_steam.write(block_stream.read())

        final_bytes = block_header_steam.getvalue()
        final_padding = BLOCK_SIZE - len(final_bytes)
        if final_padding == 0:
            return final_bytes
        else:
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(b'\x00' * final_padding)
            return block_header_steam.getvalue()
    else:
        return BytesIO().getvalue()


def read_block(file_path, block_id, pic_id):
    """ 读取指定block的内容。如果block数是401，那么能选的最大block_id是400 """
    start_pos = BLOCK_SIZE * (block_id + 1)
    end_pos = BLOCK_SIZE * (block_id + 2)
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            chunk = mm[start_pos:end_pos]
            BytesIO(chunk)
            header = struct.unpack('<4sII', chunk[:12])
            magic, block_id, num_pics = header
            print('Magic:\t\t\t', f'{magic}')
            print('Block ID:\t\t', f'{block_id}')
            print('Num pics:\t\t', f'{num_pics}')
            fmt = f'<{num_pics}I'
            offset_0 = 12
            offset_1 = offset_0 + num_pics * 4
            offset_2 = offset_1 + num_pics * 4
            offset_3 = offset_2 + num_pics * 4
            offset_list = struct.unpack(fmt, chunk[offset_0:offset_1])
            size_list = struct.unpack(fmt, chunk[offset_1:offset_2])
            label_list = struct.unpack(fmt, chunk[offset_2:offset_3])
            if pic_id < num_pics:
                byte_stream = chunk[offset_list[pic_id]:offset_list[pic_id] + size_list[pic_id]]
                dst_path = os.path.join(output_path, f'B{block_id}_P{pic_id}_L{label_list[pic_id]}.jpg')
                with open(dst_path, 'wb') as f:
                    f.write(byte_stream)
            else:
                print(f'Pic ID out of range. Total pics in the block: {num_pics}.')


def write_imagenet_dts_lv0(train):
    blocks = load_list(os.path.join(output_path, 'train_blocks.pkl' if train else 'val_blocks.pkl'))
    target_path = os.path.join(output_path, 'imagenet_train_lv0.dts' if train else 'imagenet_val_lv0.dts')
    byte_stream = make_empty_file_header(IMAGENET_HEADER_SIZE)
    num_blocks = len(blocks)
    with open(target_path, 'wb') as f:
        f.write(byte_stream)
    for block_id in range(num_blocks):
        with open(target_path, 'ab') as f:
            buf = make_imagenet_blocks(block_id=block_id, train=train)
            f.write(buf)
    modify_file_header(target_path, make_imagenet_file_header(file_path=target_path, train=train))
    if train:
        remove_list = [os.path.join(output_path, n) \
                       for n in ['train_block_margin.pkl', 'train_blocks.pkl',
                                 'train_file_name_list.pkl', 'train_info_list.pkl']]
        for each in remove_list:
            if os.path.exists(each):
                os.remove(each)
        if remove_source:
            train_dir = os.path.join(output_path, 'train')
            if os.path.exists(train_dir) and os.path.isdir(train_dir):
                shutil.rmtree(train_dir)
        print(f'Successfully exported imagenet_train_lv0.dts to {output_path}.')
    else:
        remove_list = [os.path.join(output_path, n) \
                       for n in ['val_block_margin.pkl', 'val_blocks.pkl',
                                 'val_file_name_list.pkl', 'val_info_list.pkl']]
        for each in remove_list:
            if os.path.exists(each):
                os.remove(each)
        if remove_source:
            val_dir = os.path.join(output_path, 'val')
            if os.path.exists(val_dir) and os.path.isdir(val_dir):
                shutil.rmtree(val_dir)
        print(f'Successfully exported imagenet_val_lv0.dts to {output_path}.')


def write_imagenet_dts_lv123(train):
    # file_name_list: index, folder_name, jpeg_name, label
    file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'train_file_name_list.pkl' if train else 'val_file_name_list.pkl'))
    info_list = load_list(os.path.join(os.path.abspath(output_path), 'train_info_list.pkl' if train else 'val_info_list.pkl'))
    target_path = os.path.join(output_path, f'imagenet_train_lv{compress_level}.dts' if train else f'imagenet_val_lv{compress_level}.dts')
    byte_stream = make_empty_file_header(IMAGENET_HEADER_SIZE)
    with open(target_path, 'wb') as f:
        f.write(byte_stream)

    random.seed(RAND_SEED)
    random.shuffle(file_name_list)

    if compress_level == 1:
        block_magic = BLOCK_MAGIC_LV1
    elif compress_level == 2:
        block_magic = BLOCK_MAGIC_LV2
    elif compress_level == 3:
        block_magic = BLOCK_MAGIC_LV3
    else:
        block_magic = BLOCK_MAGIC_LV0
    original_margin = BLOCK_SIZE - HEADER_SIZE_LV123
    num_pics = 0
    offset = HEADER_SIZE_LV123
    offset_list = list()
    size_list = list()
    label_list = list()
    block_id = 0
    block_margin = original_margin
    total_pics = len(file_name_list)
    block_header_steam = BytesIO()
    block_stream = BytesIO()
    maximum_area = 0
    max_pic_per_block = 0
    for index, each_file in enumerate(file_name_list):
        full_path = os.path.join(os.path.abspath(output_path), 'train' if train else 'val', each_file[1], each_file[2])
        output_bytes, img_area = process_imagenet_image(full_path)
        if img_area > maximum_area:
            maximum_area = img_area
        output_size = len(output_bytes)
        output_size_aligned = apply_pic_alignment(output_size, PIC_ALIGNMENT_BYTES)
        label = each_file[3]
        if output_size_aligned > block_margin:  # 如果容纳不下
            fmt = f'<{len(offset_list)}I'
            header_data = struct.pack('<4sII', block_magic, block_id, num_pics)
            block_header_steam.write(header_data)
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(struct.pack(fmt, *offset_list))
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(struct.pack(fmt, *size_list))
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(struct.pack(fmt, *label_list))
            block_header_steam.seek(0, SEEK_END)
            padding_header = HEADER_SIZE_LV123 - 12 - 3 * num_pics * 4
            block_header_steam.write(b'\x00' * padding_header)

            block_header_steam.seek(0, SEEK_END)
            block_stream.seek(0)
            block_header_steam.write(block_stream.read())

            final_bytes = block_header_steam.getvalue()
            final_padding = BLOCK_SIZE - len(final_bytes)
            if final_padding != 0:
                block_header_steam.seek(0, SEEK_END)
                block_header_steam.write(b'\x00' * final_padding)
            with open(target_path, 'ab') as f:
                f.write(block_header_steam.getvalue())

            if num_pics > max_pic_per_block:
                max_pic_per_block = num_pics
            block_id += 1

            # 清空所有列表
            offset_list = list()
            size_list = list()
            label_list = list()
            offset = HEADER_SIZE_LV123
            block_margin = original_margin
            num_pics = 0
            block_header_steam = BytesIO()
            block_stream = BytesIO()

        offset_list.append(offset)
        offset += output_size_aligned
        size_list.append(output_size)
        label_list.append(label)
        block_margin -= output_size_aligned
        num_pics += 1
        block_stream.seek(0, SEEK_END)
        block_stream.write(output_bytes)
        if output_size_aligned - output_size != 0:
            block_stream.seek(0, SEEK_END)
            block_stream.write(b'\x00' * (output_size_aligned - output_size))

        if index == total_pics - 1:
            fmt = f'<{len(offset_list)}I'
            header_data = struct.pack('<4sII', block_magic, block_id, num_pics)
            block_header_steam.write(header_data)
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(struct.pack(fmt, *offset_list))
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(struct.pack(fmt, *size_list))
            block_header_steam.seek(0, SEEK_END)
            block_header_steam.write(struct.pack(fmt, *label_list))
            block_header_steam.seek(0, SEEK_END)
            padding_header = HEADER_SIZE_LV123 - 12 - 3 * num_pics * 4
            block_header_steam.write(b'\x00' * padding_header)

            block_header_steam.seek(0, SEEK_END)
            block_stream.seek(0)
            block_header_steam.write(block_stream.read())

            final_bytes = block_header_steam.getvalue()
            final_padding = BLOCK_SIZE - len(final_bytes)
            if final_padding != 0:
                block_header_steam.seek(0, SEEK_END)
                block_header_steam.write(b'\x00' * final_padding)
            with open(target_path, 'ab') as f:
                f.write(block_header_steam.getvalue())

    total_blocks = block_id + 1  # 总BLOCK数，但不包括文件头

    stream = BytesIO()
    dataset_type = b'IMAGENET'
    is_training_set = 1 if train else 0
    val_set_prep = 1 if preprocess_val_set else 0
    num_classes = 1000
    tensor_layout = b'NHWC'
    image_width, image_height = 0, 0  # Default
    num_channels = 3
    color_channel_type = b' RGB'
    num_samples = len(file_name_list)
    num_volumes = 1
    volume_id = 0
    num_blocks = total_blocks
    total_bytes = num_blocks * BLOCK_SIZE + IMAGENET_HEADER_SIZE
    block_bytes = num_blocks * BLOCK_SIZE
    if compress_level == 0:
        block_header_size = HEADER_SIZE_LV0
    else:
        block_header_size = HEADER_SIZE_LV123
    pic_alignment = PIC_ALIGNMENT_BYTES if USING_PIC_ALIGNMENT else 0
    maximum_pic_area = maximum_area
    compression_ratio = total_bytes / sum([s[1] for s in info_list])
    normalize_mean = [0.0, 0.0, 0.0]
    normalize_std = [0.0, 0.0, 0.0]
    crc_code = get_partial_crc32(target_path, IMAGENET_HEADER_SIZE, return_str=False)

    # 144 Bytes
    data = struct.pack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI',
                       FILE_MAGIC,  # 4s
                       DTS_VERSION[0],  # B
                       DTS_VERSION[1],  # B
                       DTS_VERSION[2],  # B
                       DTS_VERSION[3],  # B
                       dataset_type,    # 8s
                       is_training_set, # I
                       compress_level,  # I
                       val_set_prep,    # I
                       num_classes,     # I
                       tensor_layout,   # 4s
                       image_width,     # I
                       image_height,    # I
                       num_channels,    # I
                       color_channel_type,  # 4s
                       num_samples,     # I
                       num_volumes,     # I
                       volume_id,       # I
                       num_blocks,      # I
                       num_blocks,      # I
                       total_bytes,       # Q
                       IMAGENET_HEADER_SIZE,      # I
                       block_bytes,      # Q
                       BLOCK_SIZE,       # I
                       block_header_size, # I
                       pic_alignment,    # I
                       maximum_pic_area, # I
                       max_pic_per_block,  # I
                       compression_ratio,  # f
                       normalize_mean[0],  # f
                       normalize_mean[1],  # f
                       normalize_mean[2],  # f
                       normalize_std[0],  # f
                       normalize_std[1],  # f
                       normalize_std[2],  # f
                       crc_code,          # I
                       )
    stream.write(data)
    stream.seek(0, SEEK_END)
    stream.write(b'\x00' * (IMAGENET_HEADER_SIZE - 144))
    stream.seek(0)
    modify_file_header(target_path, stream.getvalue())

    if train:
        remove_list = [os.path.join(output_path, n) \
                       for n in ['train_block_margin.pkl', 'train_blocks.pkl',
                                 'train_file_name_list.pkl', 'train_info_list.pkl']]
        for each in remove_list:
            if os.path.exists(each):
                os.remove(each)
        if remove_source:
            train_dir = os.path.join(output_path, 'train')
            if os.path.exists(train_dir) and os.path.isdir(train_dir):
                shutil.rmtree(train_dir)
        print(f'Successfully exported imagenet_train_lv{compress_level}.dts to {output_path}.')
    else:
        remove_list = [os.path.join(output_path, n) \
                       for n in ['val_block_margin.pkl', 'val_blocks.pkl',
                                 'val_file_name_list.pkl', 'val_info_list.pkl']]
        for each in remove_list:
            if os.path.exists(each):
                os.remove(each)
        if remove_source:
            val_dir = os.path.join(output_path, 'val')
            if os.path.exists(val_dir) and os.path.isdir(val_dir):
                shutil.rmtree(val_dir)
        print(f'Successfully exported imagenet_val_lv{compress_level}.dts to {output_path}.')


def view_dts_header():
    abs_path = os.path.abspath(output_path)
    folder = Path(abs_path)
    dts_files = [f.name for f in folder.glob('*.dts')]
    if dts_files:
        for each in dts_files:
            print('*' * 45)
            print(each)
            print('')
            load_dts_header_info(os.path.join(abs_path, each))
        print('*' * 45)
    else:
        print(f'Error: Cannot find any .dts files in {abs_path}.')
        exit(1)


def main():
    random.seed(RAND_SEED)
    if view_header:
        view_dts_header()
        # read_block(os.path.join(output_path, 'imagenet_val_lv3.dts'), 1, 405)
    else:
        if enable_download:
            download_dataset()
        if enable_extract:
            extract_dataset()
            if remove_source:
                remove_downloaded_files()
        if enable_concat:
            concat_dataset()


if __name__ == '__main__':
    initialize_args(parse_args())
    main()
