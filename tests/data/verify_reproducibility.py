#!/usr/bin/env python3
"""
验证随机可复现性脚本

用法:
    python verify_reproducibility.py run1_logs run2_logs

验证步骤:
    1. 读取两次运行的日志
    2. 按worker_id分组
    3. 对比每个worker的样本序列
    4. 报告是否完全一致
"""

import sys
import os

def read_log_file(log_file):
    """读取日志文件"""
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) >= 2:
                worker_id = int(parts[0])
                data_size = int(parts[1])
                label = int(parts[2]) if len(parts) > 2 else -1
                records.append((worker_id, data_size, label))

    return records

def verify_logs(log_dir1, log_dir2):
    """验证两次运行的日志是否一致"""

    print("=" * 60)
    print("随机可复现性验证工具")
    print("=" * 60)

    # 读取日志
    print(f"\n读取日志目录1: {log_dir1}")
    records1 = read_log_file(os.path.join(log_dir1, "worker_0_log.txt"))

    # 扫描所有worker文件
    for i in range(1, 16):
        try:
            file_path = os.path.join(log_dir1, f"worker_{i}_log.txt")
            if os.path.exists(file_path):
                records1.extend(read_log_file(file_path))
        except:
            pass

    print(f"读取日志目录2: {log_dir2}")
    records2 = read_log_file(os.path.join(log_dir2, "worker_0_log.txt"))

    for i in range(1, 16):
        try:
            file_path = os.path.join(log_dir2, f"worker_{i}_log.txt")
            if os.path.exists(file_path):
                records2.extend(read_log_file(file_path))
        except:
            pass

    print(f"目录1: {len(records1)} 条记录")
    print(f"目录2: {len(records2)} 条记录")

    # 验证数量
    if len(records1) != len(records2):
        print(f"\n❌ 验证失败：记录数量不一致")
        print(f"   运行1: {len(records1)} 条")
        print(f"   运行2: {len(records2)} 条")
        return False

    # 按worker_id分组
    worker_samples1 = {}
    worker_samples2 = {}

    for wid, size, label in records1:
        if wid not in worker_samples1:
            worker_samples1[wid] = []
        worker_samples1[wid].append((size, label))

    for wid, size, label in records2:
        if wid not in worker_samples2:
            worker_samples2[wid] = []
        worker_samples2[wid].append((size, label))

    # 打印每个worker的统计
    print(f"\nWorker统计:")
    for wid in sorted(worker_samples1.keys()):
        count1 = len(worker_samples1[wid])
        count2 = len(worker_samples2.get(wid, []))
        print(f"  Worker {wid:2d}: 运行1={count1:6d} 运行2={count2:6d}", end="")
        if count1 == count2:
            print(" ✓")
        else:
            print(" ✗")

    # 逐条对比
    print(f"\n逐条对比中...")
    mismatch_count = 0

    for i in range(len(records1)):
        r1 = records1[i]
        r2 = records2[i]

        if r1[0] != r2[0] or r1[1] != r2[1] or r1[2] != r2[2]:
            if mismatch_count < 10:  # 只打印前10个错误
                print(f"  记录 {i} 不匹配:")
                print(f"    运行1: worker={r1[0]}, size={r1[1]}, label={r1[2]}")
                print(f"    运行2: worker={r2[0]}, size={r2[1]}, label={r2[2]}")
            mismatch_count += 1

    # 结果
    print("\n" + "=" * 60)
    if mismatch_count == 0:
        print("✅ 验证通过：两次运行完全一致！")
        print("=" * 60)
        return True
    else:
        print(f"❌ 验证失败：发现 {mismatch_count} 处不匹配")
        print("=" * 60)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python verify_reproducibility.py <log_dir1> <log_dir2>")
        print("\n示例:")
        print("  python verify_reproducibility.py run1_logs run2_logs")
        sys.exit(1)

    log_dir1 = sys.argv[1]
    log_dir2 = sys.argv[2]

    if not os.path.exists(log_dir1):
        print(f"错误：日志目录不存在: {log_dir1}")
        sys.exit(1)

    if not os.path.exists(log_dir2):
        print(f"错误：日志目录不存在: {log_dir2}")
        sys.exit(1)

    success = verify_logs(log_dir1, log_dir2)

    sys.exit(0 if success else 1)
