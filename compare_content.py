#!/usr/bin/env python3
# 比较两个文件的内容（忽略换行符）
with open('R:/renaissance/crc-official-val_sorted.txt', 'r', encoding='utf-8') as f:
    standard = f.read().strip().split('\n')

with open('R:/renaissance/workspace/output_sorted.csv', 'r', encoding='utf-8') as f:
    ours = f.read().strip().split('\n')

print(f"Standard lines: {len(standard)}")
print(f"Our lines: {len(ours)}")
print()

# 找差异
diffs = []
for i in range(min(len(standard), len(ours))):
    if standard[i] != ours[i]:
        diffs.append((i, standard[i], ours[i]))

print(f"Different lines: {len(diffs)}")
print()

if diffs:
    print("First 10 differences:")
    for i, s, o in diffs[:10]:
        print(f"  Line {i}: standard='{s}', ours='{o}'")
else:
    print("All content is IDENTICAL!")
