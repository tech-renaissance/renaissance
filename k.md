





最新的卡死和段错误情况检测到了。

实验发现，卡死和段错误的情况，都具有**机器相关性**。我们在其中一台机器上从未发生过这两个问题（而且没发生问题的这台机器内存还更小）。另一台机器内存更大，但是同样的代码同样的命令却碰到了错误。

此外，段错误具有偶发性，已确认相同的命令有可能导致段错误也有可能不导致。奇怪的是，这还是在使用reproducible的模式下。

在会卡死的机器上，使用会卡死的命令，卡死发生的概率极高（目前所有3次测试均卡死），但我们仍无法判断卡死是必然还是偶然。卡死的时候，永远不会退出，也完全没有报错信息（release模式下）。



## 卡死的情况：

```shell
root@cuda13-cudnn919-pytorch290-960gb-100m:~/epfs# /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
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
[EngineBuffer #2] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #1] reset and updated.
# 卡死，手动中断，下一条命令

```

## 段错误的情况：

```shell
root@cuda13-cudnn919-pytorch290-960gb-100m:~/epfs# /root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true --reproducible
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
[EngineBuffer #2] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[EngineBuffer #5] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 177.844 s, Samples: 1281167, Throughput: 7203.9 samples/s

[VAL]
[EngineBuffer #4] reset and updated.
[EngineBuffer #0] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #3] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #1] reset and updated.
  Time: 8.294 s, Samples: 50000, Throughput: 6028.4 samples/s
========== Epoch 1 ==========

[TRAIN]
[EngineBuffer #0] reset and updated.
[EngineBuffer #1] reset and updated.
[EngineBuffer #7] reset and updated.
[EngineBuffer #5] reset and updated.
[EngineBuffer #4] reset and updated.
[EngineBuffer #6] reset and updated.
[EngineBuffer #2] reset and updated.
[EngineBuffer #3] reset and updated.
  Time: 9.972 s, Samples: 1281167, Throughput: 128480.6 samples/s

[VAL]
Segmentation fault (core dumped)

```



