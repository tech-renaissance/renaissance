#!/usr/bin/env python3
import hashlib

# 读取两个文件
with open('R:/renaissance/crc-official-val_sorted.txt', 'r', encoding='utf-8') as f:
    standard = f.read().strip().split('\n')

with open('R:/renaissance/workspace/output_sorted_unix.csv', 'r', encoding='utf-8') as f:
    ours = f.read().strip().split('\n')

print(f"标准答案行数: {len(standard)}")
print(f"我们的输出行数: {len(ours)}")
print()

# 查找差异
diffs = []
for i in range(min(len(standard), len(ours))):
    if standard[i] != ours[i]:
        diffs.append((i, standard[i], ours[i]))

print(f"不同行数: {len(diffs)}")
print()

if diffs:
    print("前10个差异:")
    for i, s, o in diffs[:10]:
        print(f"  行{i}: 标准='{s}', 我们的='{o}'")
else:
    print("所有内容完全一致！")

# 检查是否有长度差异
if len(standard) != len(ours):
    print(f"\n警告: 文件长度不同！差异={abs(len(standard) - len(ours))}行")
