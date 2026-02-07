#!/usr/bin/env python3
import hashlib

def calc_md5(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

print("=" * 60)
print("Validation Set (50000 samples)")
print("=" * 60)

val_official_md5 = calc_md5('R:/renaissance/crc_results/crc-official-val_sorted.txt')
val_fully_md5 = calc_md5('R:/renaissance/crc_results/fully_val.csv')
val_partial_md5 = calc_md5('R:/renaissance/crc_results/partial_val.csv')

print(f"Official val MD5:  {val_official_md5}")
print(f"FULLY val MD5:     {val_fully_md5}")
print(f"PARTIAL val MD5:   {val_partial_md5}")
print()

if val_official_md5 == val_fully_md5:
    print("[VAL] FULLY mode:     MATCH official!")
else:
    print("[VAL] FULLY mode:     DIFFERENT from official")

if val_official_md5 == val_partial_md5:
    print("[VAL] PARTIAL mode:   MATCH official!")
else:
    print("[VAL] PARTIAL mode:   DIFFERENT from official")

if val_fully_md5 == val_partial_md5:
    print("[VAL] FULLY == PARTIAL")
else:
    print("[VAL] FULLY != PARTIAL")

print()
print("=" * 60)
print("Training Set (1281167 samples)")
print("=" * 60)

train_official_md5 = calc_md5('R:/renaissance/crc_results/crc-official-train_sorted.txt')
train_fully_md5 = calc_md5('R:/renaissance/crc_results/fully_train.csv')
train_partial_md5 = calc_md5('R:/renaissance/crc_results/partial_train.csv')

print(f"Official train MD5:  {train_official_md5}")
print(f"FULLY train MD5:     {train_fully_md5}")
print(f"PARTIAL train MD5:   {train_partial_md5}")
print()

if train_official_md5 == train_fully_md5:
    print("[TRAIN] FULLY mode:     MATCH official!")
else:
    print("[TRAIN] FULLY mode:     DIFFERENT from official")

if train_official_md5 == train_partial_md5:
    print("[TRAIN] PARTIAL mode:   MATCH official!")
else:
    print("[TRAIN] PARTIAL mode:   DIFFERENT from official")

if train_fully_md5 == train_partial_md5:
    print("[TRAIN] FULLY == PARTIAL")
else:
    print("[TRAIN] FULLY != PARTIAL")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)

val_fully_ok = (val_official_md5 == val_fully_md5)
val_partial_ok = (val_official_md5 == val_partial_md5)
train_fully_ok = (train_official_md5 == train_fully_md5)
train_partial_ok = (train_official_md5 == train_partial_md5)

if val_fully_ok and val_partial_ok and train_fully_ok and train_partial_ok:
    print("ALL MODES CORRECT!")
elif val_fully_ok and train_fully_ok:
    print("FULLY mode is CORRECT for both train and val")
    print("PARTIAL mode has issues")
elif val_partial_ok and train_partial_ok:
    print("PARTIAL mode is CORRECT for both train and val")
    print("FULLY mode has issues")
else:
    print("Mixed results - see details above")
