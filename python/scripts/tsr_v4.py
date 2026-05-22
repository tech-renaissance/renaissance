#!/usr/bin/env python3
"""
TSR-V4.20 张量存储格式 Python 读写实现
版本：V4.20
日期：2026-05-01
作者：技术觉醒团队

支持特性：
- 读写TSR-V4.20格式文件
- RAW模式和ZLIB压缩模式
- 多张量打包存储
- 完整的CRC32校验
- NumPy数组与TSR格式互转
- 强制4D NHWC布局
- 严格的内存对齐要求
"""

import numpy as np
import struct
import zlib
import os
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import IntEnum


class TSRDtype(IntEnum):
    """TSR协议数据类型映射"""
    FP32 = 0
    FP16 = 1
    INT8 = 2
    INT32 = 3


# NumPy dtype 与 TSR dtype 的双向映射
NUMPY_DTYPE_TO_TSR = {
    np.float32: TSRDtype.FP32,
    np.float16: TSRDtype.FP16,
    np.int8: TSRDtype.INT8,
    np.int32: TSRDtype.INT32,
}

TSR_DTYPE_TO_NUMPY = {
    TSRDtype.FP32: np.float32,
    TSRDtype.FP16: np.float16,
    TSRDtype.INT8: np.int8,
    TSRDtype.INT32: np.int32,
}

# 数据类型元素大小（字节）
DTYPE_ELEM_SIZE = {
    TSRDtype.FP32: 4,
    TSRDtype.FP16: 2,
    TSRDtype.INT8: 1,
    TSRDtype.INT32: 4,
}

# 行步幅对齐要求（字节）
ROW_STRIDE_ALIGNMENT = {
    TSRDtype.FP32: 256,
    TSRDtype.FP16: 128,
    TSRDtype.INT8: 128,
    TSRDtype.INT32: 256,
}


@dataclass
class TSR4FileHeader:
    """TSR-V4.20 文件头结构（128字节）"""
    magic: bytes = b'TSR4'
    version_major: int = 4
    version_minor: int = 20
    header_size: int = 128
    file_mode: int = 0  # 0=RAW, 1=ZLIB
    tensor_count: int = 1
    entry_size: int = 64
    dir_offset: int = 128
    data_offset: int = 0  # 计算后填充
    header_crc32: int = 0
    reserved_0: int = 0
    reserved_1: bytes = b'\x00' * 80

    def pack(self) -> bytes:
        """打包为二进制格式"""
        # 计算header_crc32（先置0）
        self.header_crc32 = 0

        # 正确的格式字符串: <4sHHIIIIQQII80s
        # 4s + H + H + I + I + I + I + Q + Q + I + I + 80s = 12个参数
        packed = struct.pack(
            '<4sHHIIIIQQII80s',
            self.magic,              # 4s
            self.version_major,      # H
            self.version_minor,      # H
            self.header_size,        # I
            self.file_mode,          # I
            self.tensor_count,       # I
            self.entry_size,         # I
            self.dir_offset,         # Q
            self.data_offset,        # Q
            self.header_crc32,       # I (置0)
            self.reserved_0,         # I
            self.reserved_1          # 80s
        )

        # 计算CRC32
        self.header_crc32 = zlib.crc32(packed) & 0xFFFFFFFF

        # 重新打包（带CRC32）
        return struct.pack(
            '<4sHHIIIIQQII80s',
            self.magic,              # 4s
            self.version_major,      # H
            self.version_minor,      # H
            self.header_size,        # I
            self.file_mode,          # I
            self.tensor_count,       # I
            self.entry_size,         # I
            self.dir_offset,         # Q
            self.data_offset,        # Q
            self.header_crc32,       # I
            self.reserved_0,         # I
            self.reserved_1          # 80s
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'TSR4FileHeader':
        """从二进制数据解包"""
        if len(data) != 128:
            raise ValueError(f"Invalid header size: {len(data)}, expected 128")

        unpacked = struct.unpack('<4sHHIIIIQQII80s', data)

        header = cls()
        header.magic = unpacked[0]
        header.version_major = unpacked[1]
        header.version_minor = unpacked[2]
        header.header_size = unpacked[3]
        header.file_mode = unpacked[4]
        header.tensor_count = unpacked[5]
        header.entry_size = unpacked[6]
        header.dir_offset = unpacked[7]
        header.data_offset = unpacked[8]
        header.header_crc32 = unpacked[9]
        header.reserved_0 = unpacked[10]
        header.reserved_1 = unpacked[11]

        return header

    def verify_crc32(self) -> bool:
        """验证头部CRC32"""
        packed_without_crc = struct.pack(
            '<4sHHIIIIQQII80s',
            self.magic,              # 4s
            self.version_major,      # H
            self.version_minor,      # H
            self.header_size,        # I
            self.file_mode,          # I
            self.tensor_count,       # I
            self.entry_size,         # I
            self.dir_offset,         # Q
            self.data_offset,        # Q
            0,                      # I - CRC32字段置0
            self.reserved_0,         # I
            self.reserved_1          # 80s
        )
        computed_crc = zlib.crc32(packed_without_crc) & 0xFFFFFFFF
        return computed_crc == self.header_crc32


@dataclass
class TSR4TensorEntry:
    """TSR-V4.20 张量目录项结构（64字节）"""
    dtype: TSRDtype = TSRDtype.FP32
    reserved_0: bytes = b'\x00' * 3
    shape: Tuple[int, int, int, int] = (1, 1, 1, 1)  # NHWC
    numel: int = 1
    row_stride: int = 0
    nbytes: int = 0
    data_offset: int = 0
    payload_size: int = 0
    data_crc32: int = 0

    def pack(self) -> bytes:
        """打包为二进制格式"""
        return struct.pack(
            '<B3xiiiiqqQQQI',
            self.dtype,
            self.shape[0], self.shape[1], self.shape[2], self.shape[3],
            self.numel,
            self.row_stride,
            self.nbytes,
            self.data_offset,
            self.payload_size,
            self.data_crc32
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'TSR4TensorEntry':
        """从二进制数据解包"""
        if len(data) != 64:
            raise ValueError(f"Invalid entry size: {len(data)}, expected 64")

        unpacked = struct.unpack('<B3xiiiiqqQQQI', data)

        entry = cls()
        entry.dtype = TSRDtype(unpacked[0])
        entry.shape = tuple(unpacked[1:5])
        entry.numel = unpacked[5]
        entry.row_stride = unpacked[6]
        entry.nbytes = unpacked[7]
        entry.data_offset = unpacked[8]
        entry.payload_size = unpacked[9]
        entry.data_crc32 = unpacked[10]

        return entry


def compute_row_stride(shape: Tuple[int, int, int, int], dtype: TSRDtype) -> int:
    """
    计算TR4规范的行步幅（字节）
    NHWC布局：row_stride = ALIGN_UP(W × C × sizeof(dtype), alignment)
    """
    _, _, W, C = shape
    elem_size = DTYPE_ELEM_SIZE[dtype]
    alignment = ROW_STRIDE_ALIGNMENT[dtype]

    logical_row_bytes = W * C * elem_size
    return ((logical_row_bytes + alignment - 1) // alignment) * alignment


def compute_nbytes(shape: Tuple[int, int, int, int], dtype: TSRDtype) -> int:
    """
    计算总分配字节数（V4.21紧凑布局）
    V4.21变更：Tensor强制紧凑布局，nbytes = numel * sizeof(dtype)
    """
    N, H, W, C = shape
    elem_size = DTYPE_ELEM_SIZE[dtype]
    return N * H * W * C * elem_size


def create_clean_buffer(arr: np.ndarray, dtype: TSRDtype) -> bytes:
    """
    创建干净缓冲区（V4.21紧凑布局）
    将有效数据按NHWC紧密排列，无任何行间padding
    """
    N, H, W, C = arr.shape
    elem_size = DTYPE_ELEM_SIZE[dtype]

    # V4.21：紧凑布局，总字节数 = numel * elem_size
    total_bytes = N * H * W * C * elem_size
    buffer = bytearray(total_bytes)

    # 直接复制整个数组（已经是紧凑的）
    arr.tobytes()

    # 按NHWC顺序复制数据
    buffer = bytearray(arr.tobytes())

    return bytes(buffer)


def load_from_clean_buffer(data: bytes, shape: Tuple[int, int, int, int],
                          dtype: TSRDtype) -> np.ndarray:
    """
    从干净缓冲区加载到NumPy数组（V4.21紧凑布局）
    数据紧密排列，直接reshape即可
    """
    np_dtype = TSR_DTYPE_TO_NUMPY[dtype]
    N, H, W, C = shape

    # V4.21：紧凑布局，直接从字节数组创建NumPy数组
    arr = np.frombuffer(data, dtype=np_dtype).reshape(N, H, W, C)

    return arr


def align_up(size: int, alignment: int) -> int:
    """向上对齐到指定边界"""
    return ((size + alignment - 1) // alignment) * alignment


class TSRWriter:
    """TSR-V4.20 文件写入器"""

    def __init__(self, compress: bool = False):
        self.compress = compress
        self.arrays: List[np.ndarray] = []

    def add_tensor(self, arr: np.ndarray):
        """添加张量到文件"""
        if arr.ndim != 4:
            raise ValueError(f"Tensor must be 4D NHWC, got {arr.ndim}D")

        # 使用dtype.type进行比较
        if arr.dtype.type not in NUMPY_DTYPE_TO_TSR:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        # 验证所有维度≥1
        if any(dim < 1 for dim in arr.shape):
            raise ValueError(f"All dimensions must be >= 1, got shape {arr.shape}")

        self.arrays.append(arr)

    def save(self, filename: str):
        """保存到TSR文件"""
        if not self.arrays:
            raise ValueError("No tensors to save")

        tensor_count = len(self.arrays)

        # 1. 构造文件头
        header = TSR4FileHeader()
        header.tensor_count = tensor_count
        header.file_mode = 1 if self.compress else 0

        # 2. 构造目录项
        entries = []
        for arr in self.arrays:
            entry = TSR4TensorEntry()
            entry.dtype = NUMPY_DTYPE_TO_TSR[arr.dtype.type]
            entry.shape = tuple(arr.shape)
            entry.numel = arr.size

            row_stride = compute_row_stride(arr.shape, entry.dtype)
            entry.row_stride = row_stride
            entry.nbytes = compute_nbytes(arr.shape, entry.dtype)

            entries.append(entry)

        # 3. 计算数据偏移
        header_dir_size = 128 + tensor_count * 64
        if not self.compress:
            # RAW模式：数据区256字节对齐
            header.data_offset = align_up(header_dir_size, 256)
        else:
            # ZLIB模式：紧密排列
            header.data_offset = header_dir_size

        # 4. 准备数据并计算CRC32
        payloads = []
        for i, arr in enumerate(self.arrays):
            # 创建干净缓冲区
            clean_buffer = create_clean_buffer(arr, entries[i].dtype)

            # 计算CRC32（仅有效数据）
            valid_data_size = arr.size * DTYPE_ELEM_SIZE[entries[i].dtype]
            entries[i].data_crc32 = zlib.crc32(clean_buffer[:valid_data_size]) & 0xFFFFFFFF

            # 准备payload
            if not self.compress:
                # RAW模式：直接使用干净缓冲区
                payload = clean_buffer
                entries[i].payload_size = entries[i].nbytes
            else:
                # ZLIB模式：仅压缩有效数据
                payload = zlib.compress(clean_buffer[:valid_data_size])
                entries[i].payload_size = len(payload)

            payloads.append(payload)

        # 5. 计算每个数据块的文件偏移
        current_offset = header.data_offset
        for i in range(tensor_count):
            entries[i].data_offset = current_offset
            current_offset += entries[i].payload_size

            if not self.compress:
                # RAW模式：每个数据块256字节对齐
                current_offset = align_up(current_offset, 256)

        # 6. 计算头部CRC32
        header_packed = header.pack()

        # 7. 写入文件
        with open(filename, 'wb') as f:
            # 写入文件头
            f.write(header_packed)

            # 写入目录项
            for entry in entries:
                f.write(entry.pack())

            # RAW模式：写入对齐填充
            if not self.compress:
                align_padding = header.data_offset - header_dir_size
                if align_padding > 0:
                    f.write(b'\x00' * align_padding)

            # 写入数据块
            for i, payload in enumerate(payloads):
                f.write(payload)

                # RAW模式：块间对齐填充
                if not self.compress and i < tensor_count - 1:
                    next_offset = entries[i + 1].data_offset
                    current_end = entries[i].data_offset + entries[i].payload_size
                    align_padding = next_offset - current_end
                    if align_padding > 0:
                        f.write(b'\x00' * align_padding)


class TSRReader:
    """TSR-V4.20 文件读取器"""

    def __init__(self, filename: str):
        self.filename = filename
        self.header: Optional[TSR4FileHeader] = None
        self.entries: List[TSR4TensorEntry] = []

    def read_header(self):
        """读取并验证文件头"""
        with open(self.filename, 'rb') as f:
            header_data = f.read(128)

        if len(header_data) != 128:
            raise ValueError(f"File too small to be valid TSR: {self.filename}")

        self.header = TSR4FileHeader.unpack(header_data)

        # 验证魔数
        if self.header.magic != b'TSR4':
            raise ValueError(f"Invalid magic number (expected 'TSR4'): {self.filename}")

        # 验证版本
        if self.header.version_major != 4 or self.header.version_minor != 20:
            raise ValueError(
                f"Unsupported TSR version: {self.header.version_major}.{self.header.version_minor}"
            )

        # 验证头部大小
        if self.header.header_size != 128:
            raise ValueError(f"Invalid header size: {self.header.header_size}")

        # 验证张量数量
        if self.header.tensor_count == 0:
            raise ValueError(f"TSR file contains zero tensors: {self.filename}")

        # 验证模式
        if self.header.file_mode > 1:
            raise ValueError(f"Invalid file mode: {self.header.file_mode}")

        # 验证目录项大小
        if self.header.entry_size != 64:
            raise ValueError(f"Invalid entry size: {self.header.entry_size}")

        # 验证dir_offset
        if self.header.dir_offset != 128:
            raise ValueError(f"Invalid dir_offset: {self.header.dir_offset}")

        # 验证保留字段
        if self.header.reserved_0 != 0:
            raise ValueError(f"Header reserved_0 must be 0, got {self.header.reserved_0}")
        if self.header.reserved_1 != b'\x00' * 80:
            raise ValueError("Header reserved_1 must be all zeros")

        # 验证CRC32
        if not self.header.verify_crc32():
            computed_crc = zlib.crc32(struct.pack('<4sHHIIIIIIQI80s',
                self.header.magic,
                self.header.version_major,
                self.header.version_minor,
                self.header.header_size,
                self.header.file_mode,
                self.header.tensor_count,
                self.header.entry_size,
                self.header.dir_offset,
                self.header.data_offset,
                0,  # CRC32置0
                self.header.reserved_0,
                self.header.reserved_1
            )) & 0xFFFFFFFF
            raise ValueError(
                f"Header CRC32 mismatch: expected={self.header.header_crc32}, "
                f"computed={computed_crc}"
            )

        return self.header

    def read_entries(self):
        """读取所有目录项"""
        if self.header is None:
            self.read_header()

        with open(self.filename, 'rb') as f:
            f.seek(self.header.dir_offset)

            for i in range(self.header.tensor_count):
                entry_data = f.read(64)
                if len(entry_data) != 64:
                    raise ValueError(f"Failed to read tensor entry at index {i}")

                entry = TSR4TensorEntry.unpack(entry_data)

                # 验证保留字段
                if entry.reserved_0 != b'\x00' * 3:
                    raise ValueError(f"Entry reserved_0 must be zeros at index {i}")

                # 验证Shape维度
                if any(dim < 1 for dim in entry.shape):
                    raise ValueError(
                        f"Invalid shape at index {i}: all dimensions must be >= 1, "
                        f"got {entry.shape}"
                    )

                # 验证numel冗余校验
                expected_numel = entry.shape[0] * entry.shape[1] * entry.shape[2] * entry.shape[3]
                if entry.numel != expected_numel:
                    raise ValueError(
                        f"numel mismatch at index {i}: expected={expected_numel}, "
                        f"got={entry.numel}"
                    )

                # RAW模式额外验证
                if self.header.file_mode == 0:
                    if entry.data_offset % 256 != 0:
                        raise ValueError(
                            f"RAW mode data_offset not 256-byte aligned at index {i}: "
                            f"{entry.data_offset}"
                        )
                    if entry.payload_size != entry.nbytes:
                        raise ValueError(
                            f"RAW mode payload_size != nbytes at index {i}: "
                            f"expected={entry.nbytes}, got={entry.payload_size}"
                        )

                self.entries.append(entry)

        return self.entries

    def load_all(self) -> List[np.ndarray]:
        """加载所有张量"""
        if self.header is None:
            self.read_header()
        if not self.entries:
            self.read_entries()

        results = []

        with open(self.filename, 'rb') as f:
            for i, entry in enumerate(self.entries):
                # 读取数据
                f.seek(entry.data_offset)

                if self.header.file_mode == 0:
                    # RAW模式：读取整个干净缓冲区
                    payload = f.read(entry.payload_size)
                    if len(payload) != entry.payload_size:
                        raise ValueError(f"Failed to read tensor data at index {i}")

                    # 加载数据
                    arr = load_from_clean_buffer(payload, entry.shape, entry.dtype)

                else:
                    # ZLIB模式：读取压缩数据
                    compressed = f.read(entry.payload_size)
                    if len(compressed) != entry.payload_size:
                        raise ValueError(f"Failed to read compressed data at index {i}")

                    # 解压
                    valid_data_size = entry.numel * DTYPE_ELEM_SIZE[entry.dtype]
                    decompressed = zlib.decompress(compressed)
                    if len(decompressed) != valid_data_size:
                        raise ValueError(
                            f"Decompressed size mismatch at index {i}: "
                            f"expected={valid_data_size}, got={len(decompressed)}"
                        )

                    # 加载数据
                    arr = load_from_clean_buffer(decompressed, entry.shape, entry.dtype)

                # 验证数据CRC32
                clean_buffer = create_clean_buffer(arr, entry.dtype)
                valid_data_size = arr.size * DTYPE_ELEM_SIZE[entry.dtype]
                computed_crc = zlib.crc32(clean_buffer[:valid_data_size]) & 0xFFFFFFFF

                if computed_crc != entry.data_crc32:
                    raise ValueError(
                        f"Data CRC32 mismatch at index {i}: "
                        f"expected={entry.data_crc32}, computed={computed_crc}"
                    )

                results.append(arr)

            # 验证文件总大小一致性（使用实际文件大小，与C++实现对齐）
            last_entry = self.entries[-1]
            expected_end = last_entry.data_offset + last_entry.payload_size
            actual_file_size = os.path.getsize(self.filename)
            if actual_file_size != expected_end:
                raise ValueError(
                    f"File size mismatch: expected {expected_end} bytes "
                    f"based on last tensor entry, but file is {actual_file_size} bytes "
                    f"(file may be truncated or have extra data)"
                )

        return results

    def load_first(self) -> np.ndarray:
        """加载首个张量"""
        arrays = self.load_all()
        return arrays[0]

    def load_single(self) -> np.ndarray:
        """严格加载单个张量（文件必须只包含1个张量）"""
        if self.header is None:
            self.read_header()

        if self.header.tensor_count != 1:
            raise ValueError(
                f"Expected single tensor file, got {self.header.tensor_count} tensors"
            )

        return self.load_first()


# 便捷函数
def save_tsr(filename: str, arrays: Union[np.ndarray, List[np.ndarray]], compress: bool = False):
    """保存张量到TSR文件（便捷接口）"""
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    writer = TSRWriter(compress)
    for arr in arrays:
        writer.add_tensor(arr)
    writer.save(filename)


def load_tsr(filename: str) -> List[np.ndarray]:
    """加载TSR文件中的所有张量（便捷接口）"""
    reader = TSRReader(filename)
    return reader.load_all()


def load_first_tensor(filename: str) -> np.ndarray:
    """加载TSR文件中的首个张量（便捷接口）"""
    reader = TSRReader(filename)
    return reader.load_first()


def load_single_tensor(filename: str) -> np.ndarray:
    """严格加载单个张量（便捷接口）"""
    reader = TSRReader(filename)
    return reader.load_single()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Read:  python tsr_v4.py read <file.tsr>")
        print("  Write: python tsr_v4.py write <file.tsr> <shape> <dtype>")
        print()
        print("Examples:")
        print("  python tsr_v4.py read tensor.tsr")
        print("  python tsr_v4.py write tensor.tsr 2,3,4,5 float32")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'read':
        if len(sys.argv) != 3:
            print("Usage: python tsr_v4.py read <file.tsr>")
            sys.exit(1)

        filename = sys.argv[2]
        arrays = load_tsr(filename)

        print(f"Tensor count: {len(arrays)}")
        for i, arr in enumerate(arrays):
            print(f"\nTensor {i}:")
            print(f"  Shape: {arr.shape} (NHWC)")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Numel: {arr.size}")
            print(f"  Min: {arr.min():.6f}, Max: {arr.max():.6f}")
            print(f"  Mean: {arr.mean():.6f}, Std: {arr.std():.6f}")

    elif command == 'write':
        if len(sys.argv) != 5:
            print("Usage: python tsr_v4.py write <file.tsr> <shape> <dtype>")
            print("Example: python tsr_v4.py write tensor.tsr 2,3,4,5 float32")
            sys.exit(1)

        filename = sys.argv[2]
        shape = tuple(map(int, sys.argv[3].split(',')))
        dtype_str = sys.argv[4]

        dtype_map = {
            'float32': np.float32,
            'float16': np.float16,
            'int32': np.int32,
            'int8': np.int8,
        }

        if dtype_str not in dtype_map:
            print(f"Unsupported dtype: {dtype_str}, supported: {list(dtype_map.keys())}")
            sys.exit(1)

        dtype = dtype_map[dtype_str]
        array = np.random.randn(*shape).astype(dtype)

        save_tsr(filename, array, compress=False)
        print(f"Exported to {filename}: shape={shape}, dtype={dtype_str}")

    else:
        print(f"Unknown command: {command}")
        print("Supported commands: read, write")
        sys.exit(1)
