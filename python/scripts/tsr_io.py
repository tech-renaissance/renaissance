#!/usr/bin/env python3
"""
TSR V3 文件读写工具
@version 1.0.0
@date 2025-12-27
@author 技术觉醒团队

提供TSR V3格式文件的导入导出功能，与C++实现完全兼容
"""

import struct
import zlib
import numpy as np
from typing import Tuple, Union
import os


# ============================================================================
# 常量定义
# ============================================================================

TSR_MAGIC = b'TSR3'
TSR_VERSION = 3
TSR_HEADER_SIZE = 128
TSR_RAW_DATA_OFFSET = 256

# dtype映射 (TSR enum → numpy dtype)
DTYPE_MAP = {
    1: np.float32,   # FP32
    2: np.float16,   # BF16 (存储为uint16，读取时转换)
    3: np.int32,     # INT32
    4: np.int8,      # INT8
}

# dtype名称映射
DTYPE_NAME = {
    1: 'FP32',
    2: 'BF16',
    3: 'INT32',
    4: 'INT8',
}

# numpy dtype → TSR enum
DTYPE_TO_ENUM = {
    np.float32: 1,
    np.float16: 2,  # BF16存储
    np.int32: 3,
    np.int8: 4,
}


# ============================================================================
# 辅助函数
# ============================================================================

def bf16_to_fp32(bf16_data: np.ndarray) -> np.ndarray:
    """
    将BF16（存储为uint16）转换为FP32
    """
    # 查看为uint16
    bf16_bits = bf16_data.view(np.uint16)
    # 转换为uint32（左移16位）
    fp32_bits = bf16_bits.astype(np.uint32) << 16
    # 查看为float32
    return fp32_bits.view(np.float32)


def fp32_to_bf16(fp32_data: np.ndarray) -> np.ndarray:
    """
    将FP32转换为BF16（存储为uint16）
    """
    # 查看为uint32
    fp32_bits = fp32_data.view(np.uint32)
    # 右移16位，保留高16位
    bf16_bits = (fp32_bits >> 16).astype(np.uint16)
    return bf16_bits


def validate_header(header: dict) -> None:
    """
    验证TSR文件头有效性
    """
    if header['magic'] != TSR_MAGIC:
        raise ValueError(f"Invalid magic number: {header['magic']}, expected {TSR_MAGIC}")

    if header['version'] != TSR_VERSION:
        raise ValueError(f"Unsupported version: {header['version']}, expected {TSR_VERSION}")

    if header['header_size'] != TSR_HEADER_SIZE:
        raise ValueError(f"Invalid header size: {header['header_size']}, expected {TSR_HEADER_SIZE}")

    if header['dtype'] not in DTYPE_MAP:
        raise ValueError(f"Invalid dtype: {header['dtype']}, expected 1-4")

    if header['ndim'] > 4:
        raise ValueError(f"Invalid ndim: {header['ndim']}, max is 4")


# ============================================================================
# 导入函数（NumPy）
# ============================================================================

def import_ndarray(filename: str) -> Tuple[np.ndarray, dict]:
    """
    从TSR V3文件导入NumPy数组

    Args:
        filename: TSR文件路径

    Returns:
        (numpy数组, 元数据字典)

    元数据字典包含:
        - dtype: 数据类型名称 ('FP32', 'BF16', 'INT32', 'INT8')
        - shape: 形状元组
        - file_mode: 文件模式 (0=RAW, 1=ZLIB)
        - numel: 元素总数
    """
    return _import_ndarray(filename)


def _import_ndarray(filename: str) -> Tuple[np.ndarray, dict]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"TSR file not found: {filename}")

    with open(filename, 'rb') as f:
        # ========== 读取头部 (128字节) ==========
        header_data = f.read(TSR_HEADER_SIZE)
        if len(header_data) < TSR_HEADER_SIZE:
            raise ValueError(f"File too small to contain TSR header: {len(header_data)} bytes")

        # 解析头部（小端序）
        header = struct.unpack('<4sIIIBBBBIIIIQQQQII', header_data[:76])

        magic, header_size, version, file_mode, dtype, ndim, reserved1, reserved2, \
            dim0, dim1, dim2, dim3, numel, data_offset, payload_size, raw_data_size, \
            crc32, _ = header

        header_dict = {
            'magic': magic,
            'header_size': header_size,
            'version': version,
            'file_mode': file_mode,
            'dtype': dtype,
            'ndim': ndim,
            'dims': [dim0, dim1, dim2, dim3],
            'numel': numel,
            'data_offset': data_offset,
            'payload_size': payload_size,
            'raw_data_size': raw_data_size,
            'crc32': crc32,
        }

        # 验证头部
        validate_header(header_dict)

        # ========== 解析形状（NHWC右对齐）==========
        dims = header_dict['dims']
        if ndim == 0:
            shape = ()
        elif ndim == 1:
            shape = (dims[3],)
        elif ndim == 2:
            shape = (dims[2], dims[3])
        elif ndim == 3:
            shape = (dims[1], dims[2], dims[3])
        elif ndim == 4:
            shape = (dims[0], dims[1], dims[2], dims[3])
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")

        # ========== 读取数据 ==========
        f.seek(data_offset)

        if file_mode == 0:  # RAW模式
            data_bytes = f.read(raw_data_size)
            if len(data_bytes) < raw_data_size:
                raise ValueError(f"File truncated: expected {raw_data_size} bytes, got {len(data_bytes)}")

        else:  # ZLIB模式
            compressed_data = f.read(payload_size)
            if len(compressed_data) < payload_size:
                raise ValueError(f"File truncated: expected {payload_size} bytes, got {len(compressed_data)}")

            # 解压
            data_bytes = zlib.decompress(compressed_data)
            if len(data_bytes) != raw_data_size:
                raise ValueError(f"Decompression size mismatch: expected {raw_data_size}, got {len(data_bytes)}")

        # ========== 转换为numpy数组 ==========
        np_dtype = DTYPE_MAP[dtype]

        if dtype == 2:  # BF16特殊处理
            # BF16存储为uint16
            array = np.frombuffer(data_bytes, dtype=np.uint16).reshape(shape)
            # 转换为FP32
            array = bf16_to_fp32(array).astype(np.float32)
        else:
            array = np.frombuffer(data_bytes, dtype=np_dtype).reshape(shape)

        # ========== CRC32校验（可选）==========
        computed_crc = zlib.crc32(data_bytes) & 0xffffffff
        if computed_crc != crc32:
            raise ValueError(f"CRC32 mismatch: expected {crc32}, got {computed_crc}")

        # ========== 返回结果 ==========
        metadata = {
            'dtype': DTYPE_NAME[dtype],
            'shape': shape,
            'file_mode': 'RAW' if file_mode == 0 else 'ZLIB',
            'numel': numel,
        }

        return array.copy(), metadata  # copy以释放底层bytes


# ============================================================================
# 导出函数（NumPy）
# ============================================================================

def export_ndarray(array: np.ndarray, filename: str, compress: bool = False) -> None:
    """
    将NumPy数组导出为TSR V3文件

    Args:
        array: NumPy数组
        filename: 目标文件路径
        compress: 是否使用ZLIB压缩（默认False=RAW模式）
    """
    return _export_ndarray(array, filename, compress)


def _export_ndarray(array: np.ndarray, filename: str, compress: bool = False) -> None:
    # ========== 检查输入 ==========
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(array)}")

    # 支持的dtype
    supported_dtypes = [np.float32, np.int32, np.int8]

    # FP32/INT32/INT8直接支持
    dtype = array.dtype
    dtype_enum = None

    # 使用dtype.type进行比较
    if dtype.type in DTYPE_TO_ENUM:
        dtype_enum = DTYPE_TO_ENUM[dtype.type]
    elif dtype == np.float16:
        # BF16: 如果输入是float16，假设已经是BF16格式
        dtype_enum = 2
    else:
        raise TypeError(f"Unsupported dtype: {dtype}, expected one of {supported_dtypes}")

    # ========== 准备数据 ==========
    if dtype_enum == 2:  # BF16
        # 将FP32转换为BF16（存储为uint16）
        if array.dtype == np.float16:
            # 假设已经是BF16格式，转换为uint16存储
            data_bytes = array.view(np.uint16).tobytes()
        else:
            # 从FP32转换为BF16
            bf16_array = fp32_to_bf16(array.astype(np.float32))
            data_bytes = bf16_array.tobytes()
    else:
        data_bytes = array.tobytes()

    raw_data_size = len(data_bytes)

    # ========== 计算CRC32 ==========
    crc32 = zlib.crc32(data_bytes) & 0xffffffff

    # ========== 构造头部（NHWC右对齐）==========
    shape = array.shape
    ndim = array.ndim

    dims = [0, 0, 0, 0]
    if ndim == 0:
        pass  # 全0
    elif ndim == 1:
        dims[3] = shape[0]
    elif ndim == 2:
        dims[2], dims[3] = shape
    elif ndim == 3:
        dims[1], dims[2], dims[3] = shape
    elif ndim == 4:
        dims = list(shape)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}, max is 4")

    numel = array.size

    # ========== 压缩（可选）==========
    file_mode = 1 if compress else 0

    if compress:
        # ZLIB压缩
        compressed_data = zlib.compress(data_bytes, level=zlib.Z_DEFAULT_COMPRESSION)
        payload_size = len(compressed_data)
        data_offset = TSR_HEADER_SIZE
    else:
        # RAW模式
        compressed_data = None
        payload_size = raw_data_size
        data_offset = TSR_RAW_DATA_OFFSET

    # ========== 写入文件 ==========
    with open(filename, 'wb') as f:
        # 写入头部（128字节）
        # 格式: magic(4) + header_size(4) + version(4) + file_mode(4) +
        #       dtype(1) + ndim(1) + reserved1(1) + reserved2(1) + reserved3(1) + reserved4(1) +
        #       dims[4](16) + numel(8) + data_offset(8) + payload_size(8) +
        #       raw_data_size(8) + crc32(4) + reserved5(4)
        header = struct.pack('<4sIIIBBBBIIIIQQQQII',
                            TSR_MAGIC,           # 4s
                            TSR_HEADER_SIZE,     # I
                            TSR_VERSION,         # I
                            file_mode,           # I
                            dtype_enum,          # B
                            ndim,                # B
                            0,                   # B (reserved1)
                            0,                   # B (reserved2)
                            dims[0], dims[1], dims[2], dims[3],  # IIII
                            numel,               # Q
                            data_offset,         # Q
                            payload_size,        # Q
                            raw_data_size,       # Q
                            crc32,               # I
                            0,                   # I (reserved3)
                            )

        f.write(header)
        f.write(b'\x00' * (TSR_HEADER_SIZE - len(header)))  # 填充到128字节

        # 写入数据
        if compress:
            # ZLIB模式：紧随头部
            f.write(compressed_data)
        else:
            # RAW模式：256字节对齐
            padding_size = TSR_RAW_DATA_OFFSET - TSR_HEADER_SIZE
            f.write(b'\x00' * padding_size)
            f.write(data_bytes)


# ============================================================================
# PyTorch接口
# ============================================================================

def import_tensor(filename: str):
    """
    从TSR V3文件导入PyTorch张量

    Args:
        filename: TSR文件路径

    Returns:
        torch.Tensor: PyTorch张量
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")

    # 调用内部导入函数获取元数据
    if not os.path.exists(filename):
        raise FileNotFoundError(f"TSR file not found: {filename}")

    with open(filename, 'rb') as f:
        # ========== 读取头部 (128字节) ==========
        header_data = f.read(TSR_HEADER_SIZE)
        if len(header_data) < TSR_HEADER_SIZE:
            raise ValueError(f"File too small to contain TSR header: {len(header_data)} bytes")

        # 解析头部（小端序）
        header = struct.unpack('<4sIIIBBBBIIIIQQQQII', header_data[:76])

        magic, header_size, version, file_mode, dtype, ndim, reserved1, reserved2, \
            dim0, dim1, dim2, dim3, numel, data_offset, payload_size, raw_data_size, \
            crc32, _ = header

        # 验证头部
        header_dict = {
            'magic': magic,
            'header_size': header_size,
            'version': version,
            'file_mode': file_mode,
            'dtype': dtype,
            'ndim': ndim,
            'dims': [dim0, dim1, dim2, dim3],
            'numel': numel,
            'data_offset': data_offset,
            'payload_size': payload_size,
            'raw_data_size': raw_data_size,
            'crc32': crc32,
        }
        validate_header(header_dict)

        # ========== 解析形状（NHWC右对齐）==========
        dims = header_dict['dims']
        if ndim == 0:
            shape = ()
        elif ndim == 1:
            shape = (dims[3],)
        elif ndim == 2:
            shape = (dims[2], dims[3])
        elif ndim == 3:
            shape = (dims[1], dims[2], dims[3])
        elif ndim == 4:
            shape = (dims[0], dims[1], dims[2], dims[3])
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")

        # ========== 读取数据 ==========
        f.seek(data_offset)

        if file_mode == 0:  # RAW模式
            data_bytes = f.read(raw_data_size)
            if len(data_bytes) < raw_data_size:
                raise ValueError(f"File truncated: expected {raw_data_size} bytes, got {len(data_bytes)}")

        else:  # ZLIB模式
            compressed_data = f.read(payload_size)
            if len(compressed_data) < payload_size:
                raise ValueError(f"File truncated: expected {payload_size} bytes, got {len(compressed_data)}")

            # 解压
            data_bytes = zlib.decompress(compressed_data)
            if len(data_bytes) != raw_data_size:
                raise ValueError(f"Decompression size mismatch: expected {raw_data_size}, got {len(data_bytes)}")

        # ========== 转换为torch tensor ==========
        if dtype == 2:  # BF16特殊处理 - 直接转为torch.bfloat16
            # BF16存储为uint16
            array = np.frombuffer(data_bytes, dtype=np.uint16).reshape(shape)
            # 转换为torch tensor（uint16）
            tensor = torch.from_numpy(array.copy())
            # reinterpret为bfloat16
            tensor = tensor.view(torch.bfloat16)
        else:
            # 其他dtype正常处理
            np_dtype = DTYPE_MAP[dtype]
            array = np.frombuffer(data_bytes, dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(array.copy())

        return tensor


def export_tensor(tensor, filename: str, compress: bool = False) -> None:
    """
    将PyTorch张量导出为TSR V3文件

    Args:
        tensor: torch.Tensor
        filename: 目标文件路径
        compress: 是否使用ZLIB压缩（默认False=RAW模式）
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")

    # 转换为NumPy数组
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # 特殊处理bfloat16 - 转换为uint16存储
    if tensor.dtype == torch.bfloat16:
        # 转为CPU，view为uint16（BF16底层存储），再转numpy
        array = tensor.cpu().view(torch.uint16).numpy()
    else:
        # 其他dtype正常处理
        array = tensor.cpu().numpy()

    # 调用NumPy导出函数
    _export_ndarray(array, filename, compress)


# ============================================================================
# 主函数（命令行测试）
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Read:  python tsr_io.py read <file.tsr>")
        print("  Write: python tsr_io.py write <file.tsr> <shape> <dtype>")
        print()
        print("Examples:")
        print("  python tsr_io.py read tensor.tsr")
        print("  python tsr_io.py write tensor.tsr 2,3,4 float32")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'read':
        if len(sys.argv) != 3:
            print("Usage: python tsr_io.py read <file.tsr>")
            sys.exit(1)

        filename = sys.argv[2]
        array, metadata = import_ndarray(filename)

        print(f"Shape: {metadata['shape']}")
        print(f"Dtype: {metadata['dtype']}")
        print(f"Mode: {metadata['file_mode']}")
        print(f"Numel: {metadata['numel']}")
        print(f"\nArray preview:")
        print(array)

    elif command == 'write':
        if len(sys.argv) != 5:
            print("Usage: python tsr_io.py write <file.tsr> <shape> <dtype>")
            print("Example: python tsr_io.py write tensor.tsr 2,3,4 float32")
            sys.exit(1)

        filename = sys.argv[2]
        shape = tuple(map(int, sys.argv[3].split(',')))
        dtype_str = sys.argv[4]

        dtype_map = {
            'float32': np.float32,
            'int32': np.int32,
            'int8': np.int8,
        }

        if dtype_str not in dtype_map:
            print(f"Unsupported dtype: {dtype_str}, supported: {list(dtype_map.keys())}")
            sys.exit(1)

        dtype = dtype_map[dtype_str]
        array = np.random.randn(*shape).astype(dtype)

        export_ndarray(array, filename, compress=False)
        print(f"Exported to {filename}: shape={shape}, dtype={dtype_str}")

    else:
        print(f"Unknown command: {command}")
        print("Supported commands: read, write")
        sys.exit(1)

