# 预处理器性能测试



测试日期：2026年2月28日

测试版本：V3.27.1

测试说明：先清空缓存，然后用同样命令的1个epoch进行缓存预热，然后再执行以下测试

测试epoch数：28（符合训练ResNet-50到75.9%的Top-1准确率所需）

```shell
# 清理缓存命令
sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
sudo sh -c 'sync; echo 2 > /proc/sys/vm/drop_caches'
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

```



## RRC, PARTIAL, SDMP = 1

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Partial
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 1
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 28.976 s, Samples: 1281167, Throughput: 44214.7 samples/s

[VAL]
  Time: 2.224 s, Samples: 50000, Throughput: 22486.7 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 26.578 s, Samples: 1281167, Throughput: 48203.6 samples/s

[VAL]
  Time: 0.091 s, Samples: 50000, Throughput: 547347.5 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 26.190 s, Samples: 1281167, Throughput: 48918.0 samples/s

[VAL]
  Time: 0.086 s, Samples: 50000, Throughput: 579981.5 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 27.770 s, Samples: 1281167, Throughput: 46134.1 samples/s

[VAL]
  Time: 0.077 s, Samples: 50000, Throughput: 650649.9 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 26.948 s, Samples: 1281167, Throughput: 47542.4 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 706940.3 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 26.041 s, Samples: 1281167, Throughput: 49198.7 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 716137.3 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 26.158 s, Samples: 1281167, Throughput: 48977.7 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 712915.1 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 26.454 s, Samples: 1281167, Throughput: 48429.2 samples/s

[VAL]
  Time: 0.079 s, Samples: 50000, Throughput: 630153.4 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 26.847 s, Samples: 1281167, Throughput: 47721.7 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 726301.3 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 26.089 s, Samples: 1281167, Throughput: 49107.5 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 732631.0 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 26.734 s, Samples: 1281167, Throughput: 47922.0 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 726490.7 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 27.121 s, Samples: 1281167, Throughput: 47238.8 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711602.6 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 26.566 s, Samples: 1281167, Throughput: 48225.9 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 709435.8 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 26.237 s, Samples: 1281167, Throughput: 48830.7 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711529.2 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 26.417 s, Samples: 1281167, Throughput: 48498.7 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 732455.2 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 26.890 s, Samples: 1281167, Throughput: 47644.5 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 736275.8 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 26.602 s, Samples: 1281167, Throughput: 48160.6 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 738673.7 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 26.146 s, Samples: 1281167, Throughput: 49000.9 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 708609.5 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 26.472 s, Samples: 1281167, Throughput: 48397.6 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 709124.0 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 26.263 s, Samples: 1281167, Throughput: 48782.2 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 722810.4 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 26.464 s, Samples: 1281167, Throughput: 48412.4 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 718134.5 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 26.426 s, Samples: 1281167, Throughput: 48482.1 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 736234.4 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 26.098 s, Samples: 1281167, Throughput: 49089.8 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 743627.9 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 26.305 s, Samples: 1281167, Throughput: 48703.5 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 744026.9 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 26.175 s, Samples: 1281167, Throughput: 48946.7 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 708961.4 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 25.915 s, Samples: 1281167, Throughput: 49437.2 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 745713.8 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 25.915 s, Samples: 1281167, Throughput: 49436.4 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 742600.4 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 26.151 s, Samples: 1281167, Throughput: 48991.2 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 745448.2 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 742.948 s (35872676 samples)
Total val time: 4.143 s (1400000 samples)
Total time: 747.091 s
Avg train time: 26.534 s
Avg val time: 0.148 s
Avg epoch time: 26.682 s

=== Test Completed Successfully ===


```



## RRC, PARTIAL, SDMP = 2

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Partial
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 46.259 s, Samples: 1281167, Throughput: 27695.3 samples/s

[VAL]
  Time: 2.240 s, Samples: 50000, Throughput: 22324.4 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 1.902 s, Samples: 1281167, Throughput: 673570.8 samples/s

[VAL]
  Time: 0.074 s, Samples: 50000, Throughput: 678694.9 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 42.963 s, Samples: 1281167, Throughput: 29820.1 samples/s

[VAL]
  Time: 0.082 s, Samples: 50000, Throughput: 607251.9 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 1.786 s, Samples: 1281167, Throughput: 717350.3 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 747744.4 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 42.814 s, Samples: 1281167, Throughput: 29923.9 samples/s

[VAL]
  Time: 0.081 s, Samples: 50000, Throughput: 615930.3 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 1.834 s, Samples: 1281167, Throughput: 698396.7 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 705533.6 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 43.160 s, Samples: 1281167, Throughput: 29683.9 samples/s

[VAL]
  Time: 0.079 s, Samples: 50000, Throughput: 633391.7 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 1.851 s, Samples: 1281167, Throughput: 692289.0 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 734933.4 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 43.366 s, Samples: 1281167, Throughput: 29543.0 samples/s

[VAL]
  Time: 0.081 s, Samples: 50000, Throughput: 614168.0 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 1.878 s, Samples: 1281167, Throughput: 682100.2 samples/s

[VAL]
  Time: 0.075 s, Samples: 50000, Throughput: 668215.2 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 42.702 s, Samples: 1281167, Throughput: 30002.8 samples/s

[VAL]
  Time: 0.080 s, Samples: 50000, Throughput: 622826.0 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 1.819 s, Samples: 1281167, Throughput: 704311.0 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 728935.4 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 42.750 s, Samples: 1281167, Throughput: 29968.7 samples/s

[VAL]
  Time: 0.081 s, Samples: 50000, Throughput: 617908.7 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 1.872 s, Samples: 1281167, Throughput: 684464.4 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 731824.6 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 42.707 s, Samples: 1281167, Throughput: 29999.3 samples/s

[VAL]
  Time: 0.079 s, Samples: 50000, Throughput: 630702.0 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 1.874 s, Samples: 1281167, Throughput: 683625.5 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 736376.1 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 42.775 s, Samples: 1281167, Throughput: 29951.3 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 637892.5 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 1.809 s, Samples: 1281167, Throughput: 708361.3 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 741841.9 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 42.809 s, Samples: 1281167, Throughput: 29927.5 samples/s

[VAL]
  Time: 0.083 s, Samples: 50000, Throughput: 605833.1 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 1.847 s, Samples: 1281167, Throughput: 693718.1 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 735998.4 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 42.743 s, Samples: 1281167, Throughput: 29973.7 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 637478.6 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 1.867 s, Samples: 1281167, Throughput: 686365.6 samples/s

[VAL]
  Time: 0.074 s, Samples: 50000, Throughput: 671508.7 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 42.827 s, Samples: 1281167, Throughput: 29914.9 samples/s

[VAL]
  Time: 0.082 s, Samples: 50000, Throughput: 606764.4 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 1.883 s, Samples: 1281167, Throughput: 680366.4 samples/s

[VAL]
  Time: 0.073 s, Samples: 50000, Throughput: 680927.5 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 42.821 s, Samples: 1281167, Throughput: 29919.5 samples/s

[VAL]
  Time: 0.079 s, Samples: 50000, Throughput: 636528.6 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 1.831 s, Samples: 1281167, Throughput: 699868.3 samples/s

[VAL]
  Time: 0.066 s, Samples: 50000, Throughput: 754472.3 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 42.795 s, Samples: 1281167, Throughput: 29937.4 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 638969.6 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 1.848 s, Samples: 1281167, Throughput: 693410.2 samples/s

[VAL]
  Time: 0.065 s, Samples: 50000, Throughput: 763731.4 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 629.391 s (35872676 samples)
Total val time: 4.257 s (1400000 samples)
Total time: 633.647 s
Avg train time: 22.478 s
Avg val time: 0.152 s
Avg epoch time: 22.630 s

=== Test Completed Successfully ===


```



## RRC, FULLY, SDMP = 1

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode fully --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Fully
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 1
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 38.317 s, Samples: 1281167, Throughput: 33436.0 samples/s

[VAL]
  Time: 2.253 s, Samples: 50000, Throughput: 22191.7 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 18.487 s, Samples: 1281167, Throughput: 69302.0 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 728599.7 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 18.513 s, Samples: 1281167, Throughput: 69203.5 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 729184.1 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 18.380 s, Samples: 1281167, Throughput: 69702.5 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 745611.9 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 18.213 s, Samples: 1281167, Throughput: 70343.0 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 743025.0 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 18.495 s, Samples: 1281167, Throughput: 69270.4 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 740403.4 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 18.453 s, Samples: 1281167, Throughput: 69430.2 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 737528.8 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 18.319 s, Samples: 1281167, Throughput: 69938.0 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 743658.0 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 18.358 s, Samples: 1281167, Throughput: 69788.5 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 740068.7 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 18.377 s, Samples: 1281167, Throughput: 69715.2 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 739107.9 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 18.476 s, Samples: 1281167, Throughput: 69344.0 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 744926.2 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 18.301 s, Samples: 1281167, Throughput: 70005.5 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 737641.8 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 18.265 s, Samples: 1281167, Throughput: 70142.2 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 719733.9 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 18.170 s, Samples: 1281167, Throughput: 70509.1 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 743096.8 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 18.368 s, Samples: 1281167, Throughput: 69751.4 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 733561.9 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 18.202 s, Samples: 1281167, Throughput: 70384.5 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 702610.2 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 18.232 s, Samples: 1281167, Throughput: 70272.1 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711434.2 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 18.370 s, Samples: 1281167, Throughput: 69741.7 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 709051.0 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 18.351 s, Samples: 1281167, Throughput: 69816.3 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 710518.8 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 18.354 s, Samples: 1281167, Throughput: 69801.3 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 718241.4 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 18.381 s, Samples: 1281167, Throughput: 69701.4 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 703553.0 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 18.367 s, Samples: 1281167, Throughput: 69755.3 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 704761.3 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 18.324 s, Samples: 1281167, Throughput: 69916.0 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711761.8 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 18.475 s, Samples: 1281167, Throughput: 69345.9 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 702353.9 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 18.301 s, Samples: 1281167, Throughput: 70005.3 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711092.7 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 18.347 s, Samples: 1281167, Throughput: 69830.1 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 709649.0 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 18.091 s, Samples: 1281167, Throughput: 70816.0 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 709342.1 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 18.216 s, Samples: 1281167, Throughput: 70331.7 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 724497.5 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 533.503 s (35872676 samples)
Total val time: 4.118 s (1400000 samples)
Total time: 537.621 s
Avg train time: 19.054 s
Avg val time: 0.147 s
Avg epoch time: 19.201 s

=== Test Completed Successfully ===


```



## RRC, FULLY, SDMP = 2

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode fully --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Fully
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: RandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 57.222 s, Samples: 1281167, Throughput: 22389.6 samples/s

[VAL]
  Time: 2.282 s, Samples: 50000, Throughput: 21909.6 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 2.205 s, Samples: 1281167, Throughput: 580984.2 samples/s

[VAL]
  Time: 0.130 s, Samples: 50000, Throughput: 385690.9 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 32.499 s, Samples: 1281167, Throughput: 39422.3 samples/s

[VAL]
  Time: 0.107 s, Samples: 50000, Throughput: 465482.8 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 2.233 s, Samples: 1281167, Throughput: 573814.4 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 541037.0 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 32.489 s, Samples: 1281167, Throughput: 39434.0 samples/s

[VAL]
  Time: 0.106 s, Samples: 50000, Throughput: 473609.1 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 2.190 s, Samples: 1281167, Throughput: 584899.9 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 540815.1 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 32.451 s, Samples: 1281167, Throughput: 39479.9 samples/s

[VAL]
  Time: 0.105 s, Samples: 50000, Throughput: 474935.1 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 2.190 s, Samples: 1281167, Throughput: 584881.6 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 543053.8 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 32.426 s, Samples: 1281167, Throughput: 39510.9 samples/s

[VAL]
  Time: 0.106 s, Samples: 50000, Throughput: 473077.7 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 2.255 s, Samples: 1281167, Throughput: 568214.0 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 544568.0 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 32.421 s, Samples: 1281167, Throughput: 39516.0 samples/s

[VAL]
  Time: 0.105 s, Samples: 50000, Throughput: 475812.2 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 2.194 s, Samples: 1281167, Throughput: 583833.7 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 545000.5 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 32.430 s, Samples: 1281167, Throughput: 39505.8 samples/s

[VAL]
  Time: 0.104 s, Samples: 50000, Throughput: 481428.4 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 2.246 s, Samples: 1281167, Throughput: 570460.4 samples/s

[VAL]
  Time: 0.096 s, Samples: 50000, Throughput: 522489.0 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 32.426 s, Samples: 1281167, Throughput: 39511.0 samples/s

[VAL]
  Time: 0.105 s, Samples: 50000, Throughput: 476798.6 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 2.233 s, Samples: 1281167, Throughput: 573698.5 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 542045.1 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 32.447 s, Samples: 1281167, Throughput: 39485.1 samples/s

[VAL]
  Time: 0.105 s, Samples: 50000, Throughput: 478328.6 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 2.179 s, Samples: 1281167, Throughput: 588027.2 samples/s

[VAL]
  Time: 0.092 s, Samples: 50000, Throughput: 542306.3 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 32.420 s, Samples: 1281167, Throughput: 39517.8 samples/s

[VAL]
  Time: 0.105 s, Samples: 50000, Throughput: 477505.0 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 2.254 s, Samples: 1281167, Throughput: 568366.3 samples/s

[VAL]
  Time: 0.091 s, Samples: 50000, Throughput: 549287.4 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 32.474 s, Samples: 1281167, Throughput: 39452.2 samples/s

[VAL]
  Time: 0.100 s, Samples: 50000, Throughput: 499652.4 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 2.203 s, Samples: 1281167, Throughput: 581642.6 samples/s

[VAL]
  Time: 0.090 s, Samples: 50000, Throughput: 558161.1 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 32.404 s, Samples: 1281167, Throughput: 39537.1 samples/s

[VAL]
  Time: 0.103 s, Samples: 50000, Throughput: 484298.0 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 2.257 s, Samples: 1281167, Throughput: 567688.5 samples/s

[VAL]
  Time: 0.101 s, Samples: 50000, Throughput: 495011.8 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 32.399 s, Samples: 1281167, Throughput: 39542.9 samples/s

[VAL]
  Time: 0.106 s, Samples: 50000, Throughput: 472495.1 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 2.232 s, Samples: 1281167, Throughput: 574100.5 samples/s

[VAL]
  Time: 0.090 s, Samples: 50000, Throughput: 558062.1 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 32.396 s, Samples: 1281167, Throughput: 39547.1 samples/s

[VAL]
  Time: 0.099 s, Samples: 50000, Throughput: 502812.2 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 2.150 s, Samples: 1281167, Throughput: 595929.9 samples/s

[VAL]
  Time: 0.089 s, Samples: 50000, Throughput: 561754.3 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 509.924 s (35872676 samples)
Total val time: 4.968 s (1400000 samples)
Total time: 514.892 s
Avg train time: 18.212 s
Avg val time: 0.177 s
Avg epoch time: 18.389 s

=== Test Completed Successfully ===


```



## FRRC, PARTIAL, SDMP = 1

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Partial
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 1
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: FastRandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 29.255 s, Samples: 1281167, Throughput: 43792.6 samples/s

[VAL]
  Time: 2.297 s, Samples: 50000, Throughput: 21770.3 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 26.893 s, Samples: 1281167, Throughput: 47639.2 samples/s

[VAL]
  Time: 0.090 s, Samples: 50000, Throughput: 555579.7 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 26.565 s, Samples: 1281167, Throughput: 48227.0 samples/s

[VAL]
  Time: 0.084 s, Samples: 50000, Throughput: 595137.1 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 28.084 s, Samples: 1281167, Throughput: 45618.6 samples/s

[VAL]
  Time: 0.077 s, Samples: 50000, Throughput: 650594.0 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 27.432 s, Samples: 1281167, Throughput: 46703.3 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 721007.7 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 26.388 s, Samples: 1281167, Throughput: 48551.9 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 725655.8 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 26.435 s, Samples: 1281167, Throughput: 48465.2 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 723860.9 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 26.505 s, Samples: 1281167, Throughput: 48337.6 samples/s

[VAL]
  Time: 0.080 s, Samples: 50000, Throughput: 622185.0 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 27.081 s, Samples: 1281167, Throughput: 47308.6 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 639129.5 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 26.811 s, Samples: 1281167, Throughput: 47786.0 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 724345.1 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 26.264 s, Samples: 1281167, Throughput: 48780.8 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 742367.2 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 26.574 s, Samples: 1281167, Throughput: 48211.0 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 744855.5 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 26.529 s, Samples: 1281167, Throughput: 48292.6 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 743913.0 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 26.458 s, Samples: 1281167, Throughput: 48422.3 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 733927.3 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 26.585 s, Samples: 1281167, Throughput: 48190.6 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 745040.3 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 26.398 s, Samples: 1281167, Throughput: 48532.8 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 734674.4 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 27.239 s, Samples: 1281167, Throughput: 47033.7 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 742744.7 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 27.494 s, Samples: 1281167, Throughput: 46598.7 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 744877.7 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 26.679 s, Samples: 1281167, Throughput: 48022.2 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 709936.4 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 26.429 s, Samples: 1281167, Throughput: 48476.2 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 718308.4 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 26.710 s, Samples: 1281167, Throughput: 47965.0 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 744588.2 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 26.667 s, Samples: 1281167, Throughput: 48042.8 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 733954.1 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 26.379 s, Samples: 1281167, Throughput: 48567.7 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 729192.3 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 26.579 s, Samples: 1281167, Throughput: 48201.5 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 731956.5 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 26.447 s, Samples: 1281167, Throughput: 48442.0 samples/s

[VAL]
  Time: 0.077 s, Samples: 50000, Throughput: 652775.4 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 26.072 s, Samples: 1281167, Throughput: 49139.4 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 709754.8 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 26.132 s, Samples: 1281167, Throughput: 49027.3 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 717866.0 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 26.487 s, Samples: 1281167, Throughput: 48368.8 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711515.7 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 749.572 s (35872676 samples)
Total val time: 4.221 s (1400000 samples)
Total time: 753.794 s
Avg train time: 26.770 s
Avg val time: 0.151 s
Avg epoch time: 26.921 s

=== Test Completed Successfully ===


```



## FRRC, PARTIAL, SDMP = 2

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Partial
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: FastRandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 35.461 s, Samples: 1281167, Throughput: 36129.1 samples/s

[VAL]
  Time: 2.220 s, Samples: 50000, Throughput: 22522.3 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 1.852 s, Samples: 1281167, Throughput: 691610.8 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 741417.1 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 32.516 s, Samples: 1281167, Throughput: 39401.6 samples/s

[VAL]
  Time: 0.085 s, Samples: 50000, Throughput: 585740.0 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 1.807 s, Samples: 1281167, Throughput: 709151.8 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 734689.4 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 32.212 s, Samples: 1281167, Throughput: 39773.3 samples/s

[VAL]
  Time: 0.084 s, Samples: 50000, Throughput: 597094.0 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 1.801 s, Samples: 1281167, Throughput: 711195.1 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 734064.8 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 32.308 s, Samples: 1281167, Throughput: 39654.6 samples/s

[VAL]
  Time: 0.083 s, Samples: 50000, Throughput: 601346.0 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 1.795 s, Samples: 1281167, Throughput: 713671.2 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 737073.4 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 32.084 s, Samples: 1281167, Throughput: 39931.9 samples/s

[VAL]
  Time: 0.082 s, Samples: 50000, Throughput: 606370.3 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 1.789 s, Samples: 1281167, Throughput: 716272.8 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 739283.7 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 32.304 s, Samples: 1281167, Throughput: 39660.1 samples/s

[VAL]
  Time: 0.083 s, Samples: 50000, Throughput: 603992.3 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 1.784 s, Samples: 1281167, Throughput: 718036.8 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 747166.5 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 32.559 s, Samples: 1281167, Throughput: 39348.6 samples/s

[VAL]
  Time: 0.081 s, Samples: 50000, Throughput: 618424.1 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 1.783 s, Samples: 1281167, Throughput: 718539.6 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 740096.0 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 32.370 s, Samples: 1281167, Throughput: 39578.5 samples/s

[VAL]
  Time: 0.081 s, Samples: 50000, Throughput: 614080.8 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 1.787 s, Samples: 1281167, Throughput: 716906.8 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 741886.5 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 32.275 s, Samples: 1281167, Throughput: 39695.8 samples/s

[VAL]
  Time: 0.080 s, Samples: 50000, Throughput: 621336.5 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 1.806 s, Samples: 1281167, Throughput: 709248.9 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 738328.3 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 32.231 s, Samples: 1281167, Throughput: 39749.8 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 638232.7 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 1.820 s, Samples: 1281167, Throughput: 703872.7 samples/s

[VAL]
  Time: 0.065 s, Samples: 50000, Throughput: 773128.6 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 32.178 s, Samples: 1281167, Throughput: 39815.6 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 638836.6 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 1.811 s, Samples: 1281167, Throughput: 707609.3 samples/s

[VAL]
  Time: 0.065 s, Samples: 50000, Throughput: 767941.9 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 31.973 s, Samples: 1281167, Throughput: 40070.4 samples/s

[VAL]
  Time: 0.080 s, Samples: 50000, Throughput: 628313.2 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 1.844 s, Samples: 1281167, Throughput: 694841.1 samples/s

[VAL]
  Time: 0.065 s, Samples: 50000, Throughput: 770730.3 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 32.149 s, Samples: 1281167, Throughput: 39850.7 samples/s

[VAL]
  Time: 0.079 s, Samples: 50000, Throughput: 636162.8 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 1.828 s, Samples: 1281167, Throughput: 701022.4 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 726890.0 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 32.180 s, Samples: 1281167, Throughput: 39812.8 samples/s

[VAL]
  Time: 0.078 s, Samples: 50000, Throughput: 639010.5 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 1.826 s, Samples: 1281167, Throughput: 701636.2 samples/s

[VAL]
  Time: 0.066 s, Samples: 50000, Throughput: 753683.9 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 480.131 s (35872676 samples)
Total val time: 4.212 s (1400000 samples)
Total time: 484.343 s
Avg train time: 17.148 s
Avg val time: 0.150 s
Avg epoch time: 17.298 s

=== Test Completed Successfully ===


```



## FRRC, FULLY, SDMP = 1

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode fully --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Fully
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 1
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: FastRandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 36.689 s, Samples: 1281167, Throughput: 34920.1 samples/s

[VAL]
  Time: 2.268 s, Samples: 50000, Throughput: 22048.9 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 18.363 s, Samples: 1281167, Throughput: 69769.8 samples/s

[VAL]
  Time: 0.067 s, Samples: 50000, Throughput: 749851.0 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 18.429 s, Samples: 1281167, Throughput: 69519.4 samples/s

[VAL]
  Time: 0.129 s, Samples: 50000, Throughput: 387613.7 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 18.376 s, Samples: 1281167, Throughput: 69718.9 samples/s

[VAL]
  Time: 0.079 s, Samples: 50000, Throughput: 630856.2 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 18.165 s, Samples: 1281167, Throughput: 70530.6 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 713021.2 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 18.513 s, Samples: 1281167, Throughput: 69202.7 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 713020.7 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 18.466 s, Samples: 1281167, Throughput: 69380.5 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 710460.2 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 18.333 s, Samples: 1281167, Throughput: 69884.5 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 711374.2 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 18.382 s, Samples: 1281167, Throughput: 69696.9 samples/s

[VAL]
  Time: 0.068 s, Samples: 50000, Throughput: 730492.3 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 18.391 s, Samples: 1281167, Throughput: 69662.4 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 716161.6 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 18.508 s, Samples: 1281167, Throughput: 69220.5 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 722178.3 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 18.362 s, Samples: 1281167, Throughput: 69773.5 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 704977.4 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 18.331 s, Samples: 1281167, Throughput: 69892.1 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 717991.4 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 18.190 s, Samples: 1281167, Throughput: 70431.0 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 715954.6 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 18.437 s, Samples: 1281167, Throughput: 69489.8 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 728446.7 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 18.242 s, Samples: 1281167, Throughput: 70231.7 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 710543.7 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 18.287 s, Samples: 1281167, Throughput: 70057.2 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 713754.2 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 18.446 s, Samples: 1281167, Throughput: 69454.5 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 708947.8 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 18.378 s, Samples: 1281167, Throughput: 69711.4 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 710660.6 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 18.349 s, Samples: 1281167, Throughput: 69822.0 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 714665.8 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 18.367 s, Samples: 1281167, Throughput: 69753.8 samples/s

[VAL]
  Time: 0.069 s, Samples: 50000, Throughput: 728997.3 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 18.412 s, Samples: 1281167, Throughput: 69584.2 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 715332.1 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 18.353 s, Samples: 1281167, Throughput: 69807.2 samples/s

[VAL]
  Time: 0.071 s, Samples: 50000, Throughput: 707618.9 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 18.459 s, Samples: 1281167, Throughput: 69407.1 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 717567.8 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 18.329 s, Samples: 1281167, Throughput: 69899.1 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 714560.7 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 18.347 s, Samples: 1281167, Throughput: 69828.6 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 712383.6 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 18.105 s, Samples: 1281167, Throughput: 70763.4 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 712952.9 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 18.247 s, Samples: 1281167, Throughput: 70212.5 samples/s

[VAL]
  Time: 0.070 s, Samples: 50000, Throughput: 715304.5 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 532.255 s (35872676 samples)
Total val time: 4.220 s (1400000 samples)
Total time: 536.476 s
Avg train time: 19.009 s
Avg val time: 0.151 s
Avg epoch time: 19.160 s

=== Test Completed Successfully ===


```



## FRRC, FULLY, SDMP = 2

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode fully --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 28 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
Framework initialized with device: GPU
Reproducible mode: DISABLED (performance optimized)

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 112
Mode: Fully
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 2
CPVS: enabled
Device: GPU
GPU IDs: 0,1,2,3,4,5,6,7
CPU binding: enabled
Test mode: false (NORMAL mode with EngineBuffer)

=== Setting Transforms ===
Train PO 1: FastRandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
Device configured: GPU, GPUs=8, Auto CPU Binding: True

=== Running 28 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
  Time: 45.447 s, Samples: 1281167, Throughput: 28190.3 samples/s

[VAL]
  Time: 2.307 s, Samples: 50000, Throughput: 21674.9 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 2.411 s, Samples: 1281167, Throughput: 531415.9 samples/s

[VAL]
  Time: 0.154 s, Samples: 50000, Throughput: 324203.4 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 23.545 s, Samples: 1281167, Throughput: 54414.5 samples/s

[VAL]
  Time: 0.115 s, Samples: 50000, Throughput: 434836.0 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 2.338 s, Samples: 1281167, Throughput: 548051.0 samples/s

[VAL]
  Time: 0.095 s, Samples: 50000, Throughput: 526090.2 samples/s
========== Epoch 4 ==========

[TRAIN]
  Time: 23.572 s, Samples: 1281167, Throughput: 54350.2 samples/s

[VAL]
  Time: 0.111 s, Samples: 50000, Throughput: 451675.7 samples/s
========== Epoch 5 ==========

[TRAIN]
  Time: 2.398 s, Samples: 1281167, Throughput: 534184.2 samples/s

[VAL]
  Time: 0.099 s, Samples: 50000, Throughput: 507612.2 samples/s
========== Epoch 6 ==========

[TRAIN]
  Time: 23.569 s, Samples: 1281167, Throughput: 54359.0 samples/s

[VAL]
  Time: 0.110 s, Samples: 50000, Throughput: 452937.4 samples/s
========== Epoch 7 ==========

[TRAIN]
  Time: 2.370 s, Samples: 1281167, Throughput: 540690.5 samples/s

[VAL]
  Time: 0.095 s, Samples: 50000, Throughput: 527384.8 samples/s
========== Epoch 8 ==========

[TRAIN]
  Time: 23.444 s, Samples: 1281167, Throughput: 54647.9 samples/s

[VAL]
  Time: 0.110 s, Samples: 50000, Throughput: 452712.6 samples/s
========== Epoch 9 ==========

[TRAIN]
  Time: 2.423 s, Samples: 1281167, Throughput: 528838.4 samples/s

[VAL]
  Time: 0.096 s, Samples: 50000, Throughput: 518658.8 samples/s
========== Epoch 10 ==========

[TRAIN]
  Time: 23.411 s, Samples: 1281167, Throughput: 54724.9 samples/s

[VAL]
  Time: 0.111 s, Samples: 50000, Throughput: 452146.0 samples/s
========== Epoch 11 ==========

[TRAIN]
  Time: 2.395 s, Samples: 1281167, Throughput: 534907.2 samples/s

[VAL]
  Time: 0.097 s, Samples: 50000, Throughput: 515444.9 samples/s
========== Epoch 12 ==========

[TRAIN]
  Time: 23.625 s, Samples: 1281167, Throughput: 54228.9 samples/s

[VAL]
  Time: 0.111 s, Samples: 50000, Throughput: 451930.2 samples/s
========== Epoch 13 ==========

[TRAIN]
  Time: 2.396 s, Samples: 1281167, Throughput: 534777.1 samples/s

[VAL]
  Time: 0.095 s, Samples: 50000, Throughput: 523841.2 samples/s
========== Epoch 14 ==========

[TRAIN]
  Time: 23.548 s, Samples: 1281167, Throughput: 54407.0 samples/s

[VAL]
  Time: 0.111 s, Samples: 50000, Throughput: 450467.3 samples/s
========== Epoch 15 ==========

[TRAIN]
  Time: 2.468 s, Samples: 1281167, Throughput: 519140.9 samples/s

[VAL]
  Time: 0.096 s, Samples: 50000, Throughput: 518745.9 samples/s
========== Epoch 16 ==========

[TRAIN]
  Time: 23.435 s, Samples: 1281167, Throughput: 54669.9 samples/s

[VAL]
  Time: 0.110 s, Samples: 50000, Throughput: 453227.3 samples/s
========== Epoch 17 ==========

[TRAIN]
  Time: 2.416 s, Samples: 1281167, Throughput: 530311.3 samples/s

[VAL]
  Time: 0.097 s, Samples: 50000, Throughput: 515493.0 samples/s
========== Epoch 18 ==========

[TRAIN]
  Time: 23.456 s, Samples: 1281167, Throughput: 54620.6 samples/s

[VAL]
  Time: 0.100 s, Samples: 50000, Throughput: 497874.6 samples/s
========== Epoch 19 ==========

[TRAIN]
  Time: 2.454 s, Samples: 1281167, Throughput: 522135.1 samples/s

[VAL]
  Time: 0.108 s, Samples: 50000, Throughput: 463294.8 samples/s
========== Epoch 20 ==========

[TRAIN]
  Time: 23.554 s, Samples: 1281167, Throughput: 54393.3 samples/s

[VAL]
  Time: 0.111 s, Samples: 50000, Throughput: 451203.7 samples/s
========== Epoch 21 ==========

[TRAIN]
  Time: 2.434 s, Samples: 1281167, Throughput: 526298.8 samples/s

[VAL]
  Time: 0.097 s, Samples: 50000, Throughput: 514440.7 samples/s
========== Epoch 22 ==========

[TRAIN]
  Time: 23.392 s, Samples: 1281167, Throughput: 54768.6 samples/s

[VAL]
  Time: 0.101 s, Samples: 50000, Throughput: 497291.0 samples/s
========== Epoch 23 ==========

[TRAIN]
  Time: 2.409 s, Samples: 1281167, Throughput: 531933.1 samples/s

[VAL]
  Time: 0.098 s, Samples: 50000, Throughput: 511013.5 samples/s
========== Epoch 24 ==========

[TRAIN]
  Time: 23.406 s, Samples: 1281167, Throughput: 54735.7 samples/s

[VAL]
  Time: 0.109 s, Samples: 50000, Throughput: 459113.6 samples/s
========== Epoch 25 ==========

[TRAIN]
  Time: 2.382 s, Samples: 1281167, Throughput: 537888.9 samples/s

[VAL]
  Time: 0.097 s, Samples: 50000, Throughput: 516188.1 samples/s
========== Epoch 26 ==========

[TRAIN]
  Time: 23.321 s, Samples: 1281167, Throughput: 54935.6 samples/s

[VAL]
  Time: 0.109 s, Samples: 50000, Throughput: 460411.6 samples/s
========== Epoch 27 ==========

[TRAIN]
  Time: 2.410 s, Samples: 1281167, Throughput: 531553.4 samples/s

[VAL]
  Time: 0.097 s, Samples: 50000, Throughput: 514522.7 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 384.427 s (35872676 samples)
Total val time: 5.147 s (1400000 samples)
Total time: 389.574 s
Avg train time: 13.730 s
Avg val time: 0.184 s
Avg epoch time: 13.913 s

```



### 测试结果汇总（单位：s/epoch）：

|                             | Train  |  Val  | Train+Val |
| :-------------------------: | :----: | :---: | :-------: |
| **RRC, PARTIAL, SDMP = 1**  | 26.534 | 0.148 |  26.682   |
| **RRC, PARTIAL, SDMP = 2**  | 22.478 | 0.152 |  22.630   |
|  **RRC, FULLY, SDMP = 1**   | 19.054 | 0.147 |  19.201   |
|  **RRC, FULLY, SDMP = 2**   | 18.212 | 0.177 |  18.389   |
| **FRRC, PARTIAL, SDMP = 1** | 26.770 | 0.151 |  26.921   |
| **FRRC, PARTIAL, SDMP = 2** | 17.148 | 0.150 |  17.298   |
|  **FRRC, FULLY, SDMP = 1**  | 19.009 | 0.151 |  19.160   |
|  **FRRC, FULLY, SDMP = 2**  | 13.730 | 0.184 |  13.913   |

（注：在SDMP=1的情形下，FRRC与RRC的行为完全一致，测试结果如有差别纯属正常波动）

RRC在SDMP=2的情形下速度提升不明显的原因：RRC在SDMP=1时是局部解码，但在SDMP=2时必须切换为完整解码，由此造成了更多的时间开销，抵消了SDMP的优势。FRRC算法的主要优势就在于它在SDMP=1和SDMP=2时都是局部解码。





