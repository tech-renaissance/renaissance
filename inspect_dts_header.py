#!/usr/bin/env python3
import struct

# 打开DTS文件
with open('T:/dataset/imagenet/imagenet_val_lv0.dts', 'rb') as f:
    # 跳过文件头（16MB）
    f.seek(16 * 1024 * 1024)

    # 读取block 0的前100字节
    block_data = f.read(100)

    print("Block 0前100字节（十六进制）：")
    for i in range(0, 100, 16):
        hex_str = ' '.join(f'{b:02x}' for b in block_data[i:i+16])
        print(f'  {i:04d}: {hex_str}')

    print("\n解析block header:")
    # magic (4B)
    magic = block_data[0:4]
    print(f"  magic: {magic}")

    # block_id (4B)
    block_id = struct.unpack('<I', block_data[4:8])[0]
    print(f"  block_id: {block_id}")

    # num_pics (4B)
    num_pics = struct.unpack('<I', block_data[8:12])[0]
    print(f"  num_pics: {num_pics}")

    # offsets数组
    offsets = struct.unpack(f'<{num_pics}I', block_data[12:12+num_pics*4])
    print(f"  前5个offsets: {offsets[:5]}")
    print(f"  HEADER_SIZE_LV123应该=16384 (16KB)")
