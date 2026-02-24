# 二、报错情况描述

首先，我们运行的是很简单的脚本：

## com.sh

```shell
echo "[Non-reproducible #0]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #0]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #1]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #1]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #2]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #2]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #3]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #3]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #4]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #4]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #5]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #5]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #6]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #6]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #7]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #7]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #8]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #8]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

echo "[Non-reproducible #9]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
echo "[Reproducible #9]"
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible

```

在服务器A的运行是非常正常的，至少在我们测试次数范围内从未出错：

## 服务器A输出（部分）

```shell
--   e_v4.cpp - 当前最优方案：全局洗牌 + 顺序读取（无需Direct I/O）
-- 
-- === Loading Performance Tests Configuration Complete ===
-- Test executables will be built in: /root/epfs/R/renaissance/build/bin/tests/loading
-- Including loading tests
-- === NUMA Binding Tests Configured ===
-- bind_a  : Expert Solution A (GMX)
-- bind_f  : Expert Solution F (SNX)
-- Use: cd build && make bind_a bind_f
-- Or:  cd build && cmake .. && make -CMakeLists.txt=tests/bind/bind_f
-- Including bind tests
-- === Unit Tests Configuration Complete ===
-- === Unit Tests Configuration Complete ===
-- Unit tests directory included
-- 
-- === renAIssance Framework Build Configuration ===
-- Platform: Linux
-- Build Type: Release
-- C++ Standard: 17
-- Architecture: x86_64
-- Compiler: GNU
-- 
-- CUDA: Enabled (75;80;86;89;90)
-- cuDNN: Found (/usr/lib/x86_64-linux-gnu/libcudnn.so)
-- OpenMP: Enabled
-- vcpkg: Enabled
-- 
-- Configuring done (18.6s)
-- Generating done (0.1s)
-- Build files have been written to: /root/epfs/R/renaissance/build
[INFO] Building with Ninja...
[232/232] Linking CXX executable bin/tests/pw/test_two_po
[OK] Build completed successfully!
[INFO] Build artifacts are in: build
root@tech-renaissance:~/epfs# rm -rf com.sh
root@tech-renaissance:~/epfs# rm -rf rep.sh
root@tech-renaissance:~/epfs# nano com.sh
root@tech-renaissance:~/epfs# chmod a+x com.sh
root@tech-renaissance:~/epfs# ./com.sh
[Non-reproducible #0]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 45.399 s, Samples: 1281167, Throughput: 28220.3 samples/s

[VAL]
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 2.310 s, Samples: 50000, Throughput: 21640.7 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.641 s, Samples: 1281167, Throughput: 780801.7 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 686988.6 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 42.436 s, Samples: 1281167, Throughput: 30190.8 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 0.087 s, Samples: 50000, Throughput: 576779.0 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.885 s, Samples: 1281167, Throughput: 679485.2 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 681392.4 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 91.361 s (5124668 samples)
Total val time: 2.543 s (200000 samples)
Total time: 93.904 s
Avg train time: 22.840 s
Avg val time: 0.636 s
Avg epoch time: 23.476 s

=== Test Completed Successfully ===

[Reproducible #0]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 58.457 s, Samples: 1281167, Throughput: 21916.3 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 2.605 s, Samples: 50000, Throughput: 19193.1 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 1.670 s, Samples: 1281167, Throughput: 767160.6 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 0.078 s, Samples: 50000, Throughput: 637459.4 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 55.550 s, Samples: 1281167, Throughput: 23063.5 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.100 s, Samples: 50000, Throughput: 499511.2 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 1.939 s, Samples: 1281167, Throughput: 660805.3 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.077 s, Samples: 50000, Throughput: 645844.1 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 117.616 s (5124668 samples)
Total val time: 2.861 s (200000 samples)
Total time: 120.477 s
Avg train time: 29.404 s
Avg val time: 0.715 s
Avg epoch time: 30.119 s

=== Test Completed Successfully ===

[Non-reproducible #1]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 45.737 s, Samples: 1281167, Throughput: 28011.7 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 2.323 s, Samples: 50000, Throughput: 21521.7 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.636 s, Samples: 1281167, Throughput: 783286.7 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 687234.9 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 42.863 s, Samples: 1281167, Throughput: 29889.9 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 0.086 s, Samples: 50000, Throughput: 582558.8 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 1.901 s, Samples: 1281167, Throughput: 673925.4 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 681248.4 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 92.136 s (5124668 samples)
Total val time: 2.555 s (200000 samples)
Total time: 94.692 s
Avg train time: 23.034 s
Avg val time: 0.639 s
Avg epoch time: 23.673 s

=== Test Completed Successfully ===

[Reproducible #1]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 58.221 s, Samples: 1281167, Throughput: 22005.4 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 2.694 s, Samples: 50000, Throughput: 18557.4 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #[EngineBuffer #0] reset and updated.4
] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 1.670 s, Samples: 1281167, Throughput: 767090.7 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 0.078 s, Samples: 50000, Throughput: 639184.7 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 54.522 s, Samples: 1281167, Throughput: 23498.3 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.089 s, Samples: 50000, Throughput: 561533.5 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.795 s, Samples: 1281167, Throughput: 713634.3 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 0.078 s, Samples: 50000, Throughput: 643580.4 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 116.208 s (5124668 samples)
Total val time: 2.939 s (200000 samples)
Total time: 119.147 s
Avg train time: 29.052 s
Avg val time: 0.735 s
Avg epoch time: 29.787 s

=== Test Completed Successfully ===

[Non-reproducible #2]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 44.884 s, Samples: 1281167, Throughput: 28543.8 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 2.348 s, Samples: 50000, Throughput: 21293.5 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 1.636 s, Samples: 1281167, Throughput: 783007.5 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 684790.2 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 42.310 s, Samples: 1281167, Throughput: 30280.2 samples/s

[VAL]
[EngineBuffer #[EngineBuffer #76] reset and updated.] reset and updated.

[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.083 s, Samples: 50000, Throughput: 600094.4 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #[EngineBuffer #7] reset and updated.
2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.884 s, Samples: 1281167, Throughput: 680101.1 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 682793.5 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 90.715 s (5124668 samples)
Total val time: 2.578 s (200000 samples)
Total time: 93.292 s
Avg train time: 22.679 s
Avg val time: 0.644 s
Avg epoch time: 23.323 s

=== Test Completed Successfully ===

[Reproducible #2]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 58.268 s, Samples: 1281167, Throughput: 21987.5 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 2.648 s, Samples: 50000, Throughput: 18882.3 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #[EngineBuffer #7] reset and updated.
1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 1.670 s, Samples: 1281167, Throughput: 767169.8 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 0.078 s, Samples: 50000, Throughput: 638874.8 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 54.715 s, Samples: 1281167, Throughput: 23415.5 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 0.101 s, Samples: 50000, Throughput: 495631.2 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 1.982 s, Samples: 1281167, Throughput: 646414.5 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 0.078 s, Samples: 50000, Throughput: 641102.3 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 116.635 s (5124668 samples)
Total val time: 2.905 s (200000 samples)
Total time: 119.540 s
Avg train time: 29.159 s
Avg val time: 0.726 s
Avg epoch time: 29.885 s

=== Test Completed Successfully ===

[Non-reproducible #3]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 45.220 s, Samples: 1281167, Throughput: 28332.0 samples/s

[VAL]
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 2.343 s, Samples: 50000, Throughput: 21343.4 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.634 s, Samples: 1281167, Throughput: 784108.0 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 688270.7 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 42.337 s, Samples: 1281167, Throughput: 30260.9 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.[EngineBuffer #
1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.083 s, Samples: 50000, Throughput: 600879.9 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 1.877 s, Samples: 1281167, Throughput: 682687.3 samples/s

[VAL]
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.073 s, Samples: 50000, Throughput: 682651.9 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 91.068 s (5124668 samples)
Total val time: 2.572 s (200000 samples)
Total time: 93.639 s
Avg train time: 22.767 s
Avg val time: 0.643 s
Avg epoch time: 23.410 s

=== Test Completed Successfully ===

[Reproducible #3]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 58.376 s, Samples: 1281167, Throughput: 21946.9 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 2.615 s, Samples: 50000, Throughput: 19122.9 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.[EngineBuffer #
6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 1.790 s, Samples: 1281167, Throughput: 715824.0 samples/s

[VAL]
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.079 s, Samples: 50000, Throughput: 630206.1 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 55.387 s, Samples: 1281167, Throughput: 23131.1 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 0.091 s, Samples: 50000, Throughput: 549517.9 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 1.792 s, Samples: 1281167, Throughput: 715075.7 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 0.086 s, Samples: 50000, Throughput: 584751.7 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 117.344 s (5124668 samples)
Total val time: 2.871 s (200000 samples)
Total time: 120.215 s
Avg train time: 29.336 s
Avg val time: 0.718 s
Avg epoch time: 30.054 s

=== Test Completed Successfully ===

[Non-reproducible #4]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #[EngineBuffer #6] reset and updated.
5] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 45.278 s, Samples: 1281167, Throughput: 28295.7 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 2.364 s, Samples: 50000, Throughput: 21147.4 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 1.644 s, Samples: 1281167, Throughput: 779223.6 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #[EngineBuffer #7] reset and updated.4] reset and updated.

[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 0.072 s, Samples: 50000, Throughput: 691982.9 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.

```

服务器B就不同了，报错概率还挺高：

## 服务器B输出（部分）

```shell
========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #[EngineBuffer #3] reset and updated.7] reset and updated.
[EngineBuffer #0] reset and updated.

[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 99.268 s, Samples: 1281167, Throughput: 12906.2 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #[EngineBuffer #[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
7] reset and updated.
5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 4.779 s, Samples: 50000, Throughput: 10462.6 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 2.059 s, Samples: 1281167, Throughput: 622118.2 samples/s

[VAL]
[EngineBuffer #[EngineBuffer #7] reset and updated.6] reset and updated.

[EngineBuffer #1] reset and updated.
[EngineBuffer #[EngineBuffer #0] reset and updated.
2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.102 s, Samples: 50000, Throughput: 490691.0 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #[EngineBuffer #76] reset and updated.] reset and updated.

  Time: 96.821 s, Samples: 1281167, Throughput: 13232.3 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 0.137 s, Samples: 50000, Throughput: 365586.8 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 1.323 s, Samples: 1281167, Throughput: 968049.0 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 0.093 s, Samples: 50000, Throughput: 537513.2 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 199.472 s (5124668 samples)
Total val time: 5.111 s (200000 samples)
Total time: 204.582 s
Avg train time: 49.868 s
Avg val time: 1.278 s
Avg epoch time: 51.146 s

=== Test Completed Successfully ===

[Non-reproducible #6]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 70.420 s, Samples: 1281167, Throughput: 18193.2 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #[EngineBuffer #03] reset and updated.] reset and updated.

[EngineBuffer #1] reset and updated.
[EngineBuffer #[EngineBuffer #67] reset and updated.] reset and updated.

  Time: 3.302 s, Samples: 50000, Throughput: 15140.8 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 1.354 s, Samples: 1281167, Throughput: 946061.3 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #[EngineBuffer #2] reset and updated.3
] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #[EngineBuffer #76] reset and updated.
] reset and updated.
  Time: 0.081 s, Samples: 50000, Throughput: 619547.0 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #[EngineBuffer #1] reset and updated.
2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 66.655 s, Samples: 1281167, Throughput: 19221.0 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 0.068 s, Samples: 50000, Throughput: 738470.5 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 1.195 s, Samples: 1281167, Throughput: 1072432.4 samples/s

[VAL]
./com.sh: line 32: 77307 Segmentation fault      (core dumped) /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
[Reproducible #6]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 99.382 s, Samples: 1281167, Throughput: 12891.3 samples/s

[VAL]
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 4.785 s, Samples: 50000, Throughput: 10449.4 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #[EngineBuffer #67] reset and updated.] reset and updated.

[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 2.105 s, Samples: 1281167, Throughput: 608749.4 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 0.123 s, Samples: 50000, Throughput: 406951.3 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #[EngineBuffer #23] reset and updated.] reset and updated.
[EngineBuffer #5] reset and updated.

[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 96.819 s, Samples: 1281167, Throughput: 13232.5 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 0.141 s, Samples: 50000, Throughput: 355537.6 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 1.310 s, Samples: 1281167, Throughput: 978136.6 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 0.093 s, Samples: 50000, Throughput: 537670.9 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 199.616 s (5124668 samples)
Total val time: 5.141 s (200000 samples)
Total time: 204.757 s
Avg train time: 49.904 s
Avg val time: 1.285 s
Avg epoch time: 51.189 s

=== Test Completed Successfully ===

[Non-reproducible #7]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 70.484 s, Samples: 1281167, Throughput: 18176.6 samples/s

[VAL]
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 3.302 s, Samples: 50000, Throughput: 15140.1 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 1.338 s, Samples: 1281167, Throughput: 957423.7 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #[EngineBuffer #2] reset and updated.
3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 0.092 s, Samples: 50000, Throughput: 542262.5 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 66.610 s, Samples: 1281167, Throughput: 19233.9 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.[EngineBuffer #
5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 0.069 s, Samples: 50000, Throughput: 723525.1 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 1.224 s, Samples: 1281167, Throughput: 1046795.9 samples/s

[VAL]
./com.sh: line 37: 81720 Segmentation fault      (core dumped) /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
[Reproducible #7]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 99.769 s, Samples: 1281167, Throughput: 12841.4 samples/s

[VAL]
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 4.763 s, Samples: 50000, Throughput: 10497.3 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 1.766 s, Samples: 1281167, Throughput: 725329.0 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 0.114 s, Samples: 50000, Throughput: 440100.9 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 97.042 s, Samples: 1281167, Throughput: 13202.3 samples/s

[VAL]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 0.138 s, Samples: 50000, Throughput: 363577.9 samples/s
========== Epoch 3 ==========

[TRAIN]
./com.sh: line 39: 83872 Segmentation fault      (core dumped) /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible
[Non-reproducible #8]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
  Time: 70.366 s, Samples: 1281167, Throughput: 18207.2 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 3.300 s, Samples: 50000, Throughput: 15152.1 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 1.382 s, Samples: 1281167, Throughput: 926990.2 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 0.090 s, Samples: 50000, Throughput: 557828.0 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #[EngineBuffer #4] reset and updated.
[EngineBuffer #[EngineBuffer #6] reset and updated.
[EngineBuffer #52] reset and updated.
1] reset and updated.
[EngineBuffer #7] reset and updated.
] reset and updated.
  Time: 66.791 s, Samples: 1281167, Throughput: 19181.8 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 0.063 s, Samples: 50000, Throughput: 789067.4 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 1.206 s, Samples: 1281167, Throughput: 1062077.6 samples/s

[VAL]
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
  Time: 0.080 s, Samples: 50000, Throughput: 623612.3 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 139.745 s (5124668 samples)
Total val time: 3.533 s (200000 samples)
Total time: 143.278 s
Avg train time: 34.936 s
Avg val time: 0.883 s
Avg epoch time: 35.820 s

=== Test Completed Successfully ===

[Reproducible #8]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 99.270 s, Samples: 1281167, Throughput: 12905.9 samples/s

[VAL]
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 4.779 s, Samples: 50000, Throughput: 10462.1 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 1.989 s, Samples: 1281167, Throughput: 644270.7 samples/s

[VAL]
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 0.125 s, Samples: 50000, Throughput: 400240.8 samples/s
========== Epoch 2 ==========

[TRAIN]
[EngineBuffer #4] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
  Time: 97.036 s, Samples: 1281167, Throughput: 13203.0 samples/s

[VAL]
[EngineBuffer #1] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
  Time: 0.137 s, Samples: 50000, Throughput: 365890.3 samples/s
========== Epoch 3 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 1.325 s, Samples: 1281167, Throughput: 966569.8 samples/s

[VAL]
./com.sh: line 44: 88173 Segmentation fault      (core dumped) /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible
[Non-reproducible #9]
Reproducible mode: DISABLED (performance optimized)

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #0] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #4] reset and updated.
  Time: 70.709 s, Samples: 1281167, Throughput: 18118.8 samples/s

[VAL]
[EngineBuffer #[EngineBuffer #10] reset and updated.] reset and updated.

[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
4] reset and updated.
[EngineBuffer #6] reset and updated.
  Time: 3.313 s, Samples: 50000, Throughput: 15090.4 samples/s
========== Epoch 1 ==========

[TRAIN]
./com.sh: line 47: 90350 Segmentation fault      (core dumped) /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
[Reproducible #9]
Reproducible mode: ENABLED

=== Configuring Dataset ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet

=== Configuring DataLoader ===
Load workers: 16
Preprocess workers: 112
Mode: Partial

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuring Preprocessor ===
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
Test mode: false (NORMAL mode with EngineBuffer)
Device configured: GPU, GPUs=8, Auto CPU Binding: True
GPU IDs: 0,1,2,3,4,5,6,7

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip (inference from previous PO)
Val PO 1: Resize (224)
Val PO 2: CenterCrop (224)
Random seed: 42
[EngineBuffer #1] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #7] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #0] reset and updated.

```

## 补充说明

服务器A和B是完全一样的操作系统，运行的是完全一样的代码，所用的依赖版本全都一致。

最最重要的是，在我们开发PreprocessorWorker和EngineBuffer之前，这个段错误从未发生。印象中是只在开发EngineBuffer之后发生过。

我们看到，这个段错误发生的位置具有随机性，而且与机器高度相关，服务器A从未发生过这个错误。

对于服务器B，段错误可以发生在任何phase（train/val）、任何epoch、任何模式（可复现模式、不可复现模式），而且，同样的命令，可能出错也可能不出错。这是在我们没有修改随机种子的情形下出现的随机性，因此我们初步认为跟多线程竞态有关。极有可能是多线程竞争导致指针/索引/计数器值计算错误，然后引发越界访问。当然也有可能是别的原因。如果是普通的参数设置错误（比如分配的内存不足或计数器边界条件出现逻辑错误），那么这个段错误应该会必然发生，而且会在早期发生。现在的情况，更像是一种很偶然的多线程竞争导致某个值出错，最后程序“跑飞了”。