#!/usr/bin/env python3
import hashlib

print("=" * 60)
print("Validation Set Comparison (50000 samples)")
print("=" * 60)

# 读取3个val文件
with open('R:/renaissance/crc_results/crc-official-val_sorted.txt', 'r') as f:
    val_official = f.read().strip().split('\n')

with open('R:/renaissance/crc_results/fully_val.csv', 'r') as f:
    val_fully = f.read().strip().split('\n')

with open('R:/renaissance/crc_results/partial_val.csv', 'r') as f:
    val_partial = f.read().strip().split('\n')

print(f"Official val:  {len(val_official)} lines")
print(f"FULLY val:     {len(val_fully)} lines")
print(f"PARTIAL val:   {len(val_partial)} lines")
print()

# 对比Official vs FULLY
diffs_official_fully = [(i, val_official[i], val_fully[i]) for i in range(min(len(val_official), len(val_fully))) if val_official[i] != val_fully[i]]
print(f"Official vs FULLY:   {len(diffs_official_fully)} differences")

# 对比Official vs PARTIAL
diffs_official_partial = [(i, val_official[i], val_partial[i]) for i in range(min(len(val_official), len(val_partial))) if val_official[i] != val_partial[i]]
print(f"Official vs PARTIAL: {len(diffs_official_partial)} differences")

# 对比FULLY vs PARTIAL
diffs_fully_partial = [(i, val_fully[i], val_partial[i]) for i in range(min(len(val_fully), len(val_partial))) if val_fully[i] != val_partial[i]]
print(f"FULLY vs PARTIAL:     {len(diffs_fully_partial)} differences")
print()

if len(diffs_official_fully) == 0:
    print("✅ FULLY val matches Official!")
else:
    print(f"❌ FULLY val has {len(diffs_official_fully)} differences")

if len(diffs_official_partial) == 0:
    print("✅ PARTIAL val matches Official!")
else:
    print(f"❌ PARTIAL val has {len(diffs_official_partial)} differences")

if len(diffs_fully_partial) == 0:
    print("✅ FULLY val and PARTIAL val are identical!")
else:
    print(f"❌ FULLY val and PARTIAL val have {len(diffs_fully_partial)} differences")

print()
print("=" * 60)
print("Training Set Comparison (1281167 samples)")
print("=" * 60)

# 读取3个train文件
with open('R:/renaissance/crc_results/crc-official-train_sorted.txt', 'r') as f:
    train_official = f.read().strip().split('\n')

with open('R:/renaissance/crc_results/fully_train.csv', 'r') as f:
    train_fully = f.read().strip().split('\n')

with open('R:/renaissance/crc_results/partial_train.csv', 'r') as f:
    train_partial = f.read().strip().split('\n')

print(f"Official train:  {len(train_official)} lines")
print(f"FULLY train:     {len(train_fully)} lines")
print(f"PARTIAL train:   {len(train_partial)} lines")
print()

# 对比Official vs FULLY (只检查前1000个样本，避免太慢)
sample_size = min(1000, len(train_official), len(train_fully))
diffs_official_fully_train = [(i, train_official[i], train_fully[i]) for i in range(sample_size) if train_official[i] != train_fully[i]]
print(f"Official vs FULLY (first {sample_size}):   {len(diffs_official_fully_train)} differences")

# 对比Official vs PARTIAL (只检查前1000个样本)
diffs_official_partial_train = [(i, train_official[i], train_partial[i]) for i in range(sample_size) if train_official[i] != train_partial[i]]
print(f"Official vs PARTIAL (first {sample_size}): {len(diffs_official_partial_train)} differences")

# 对比FULLY vs PARTIAL (只检查前1000个样本)
diffs_fully_partial_train = [(i, train_fully[i], train_partial[i]) for i in range(sample_size) if train_fully[i] != train_partial[i]]
print(f"FULLY vs PARTIAL (first {sample_size}):     {len(diffs_fully_partial_train)} differences")
print()

if len(diffs_official_fully_train) == 0:
    print("✅ FULLY train matches Official (first 1000)!")
else:
    print(f"❌ FULLY train has {len(diffs_official_fully_train)} differences (first 1000)")

if len(diffs_official_partial_train) == 0:
    print("✅ PARTIAL train matches Official (first 1000)!")
else:
    print(f"❌ PARTIAL train has {len(diffs_official_partial_train)} differences (first 1000)")

if len(diffs_fully_partial_train) == 0:
    print("✅ FULLY train and PARTIAL train are identical (first 1000)!")
else:
    print(f"❌ FULLY train and PARTIAL train have {len(diffs_fully_partial_train)} differences (first 1000)")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)

if len(diffs_official_fully) == 0 and len(diffs_official_fully_train) == 0:
    print("✅✅✅ FULLY mode (both train and val) is CORRECT!")
if len(diffs_official_partial) == 0 and len(diffs_official_partial_train) == 0:
    print("✅✅✅ PARTIAL mode (both train and val) is CORRECT!")
