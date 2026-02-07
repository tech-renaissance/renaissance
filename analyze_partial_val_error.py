#!/usr/bin/env python3
# 分析PARTIAL val的错误

# 读取文件
with open('R:/renaissance/new_results/official_val_sorted.txt', 'r') as f:
    official = f.read().strip().split('\n')

with open('R:/renaissance/new_results/partial_val.csv', 'r') as f:
    partial = f.read().strip().split('\n')

print(f"Official: {len(official)} samples")
print(f"Partial:  {len(partial)} samples")
print()

# 转换为set进行比较
official_set = set(official)
partial_set = set(partial)

print(f"Official unique values: {len(official_set)}")
print(f"Partial unique values:  {len(partial_set)}")
print()

# 找出差异
only_in_official = official_set - partial_set
only_in_partial = partial_set - official_set

print(f"Only in official: {len(only_in_official)} values")
print(f"Only in partial:  {len(only_in_partial)} values")
print()

if only_in_official:
    print("First 10 values only in official:")
    for val in list(only_in_official)[:10]:
        print(f"  {val}")

if only_in_partial:
    print("First 10 values only in partial:")
    for val in list(only_in_partial)[:10]:
        print(f"  {val}")

# 检查是否有重复值
from collections import Counter
official_counts = Counter(official)
partial_counts = Counter(partial)

official_duplicates = [val for val, count in official_counts.items() if count > 1]
partial_duplicates = [val for val, count in partial_counts.items() if count > 1]

print()
print(f"Official has {len(official_duplicates)} duplicated values")
print(f"Partial has {len(partial_duplicates)} duplicated values")

if partial_duplicates:
    print("First 10 duplicated values in partial:")
    for val in partial_duplicates[:10]:
        print(f"  {val}: {partial_counts[val]} times")
