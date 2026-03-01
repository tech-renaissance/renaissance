# DataLoader性能测试

测试日期：2026年3月1日

测试版本：V3.28.8

## Windows

```shell
PS R:\renaissance\build\windows-msvc-release\bin\tests\pw> .\test_dataloader_performance --dataset imagenet --format dts --lv 0 --path T:/dataset/imagenet --mode partial --workers 8 --preproc 24

========================================================================
Preprocessor V4.0 Performance Test
========================================================================

Configuration:
  Dataset path: T:/dataset/imagenet
  Format: DTS LV0
  Load workers: 8
  Preprocess workers: 24

Testing imagenet dts partial...
Device configured: GPU, GPUs=1

========================================
Training Set Test
========================================
Load time:        41.118 s
Total samples:    1281167
Expected samples: 1281167
Throughput:       3.332 GB/s
Integrity:        PASSED

========================================
Validation Set Test
========================================
Load time:        1.661 s
Total samples:    50000
Expected samples: 50000
Throughput:       3.772 GB/s
Integrity:        PASSED

========================================================================
All tests completed!
========================================================================

```



## 树莓派

```shell
root@tech-renaissance:/home/tech-renaissance/Downloads# /root/epfs/R/renaissance/build/bin/tests/pw/test_dataloader_performance --dataset imagenet --format dts --lv 3 --path /root/epfs/dataset/imagenet --mode partial --workers 4 --preproc 4
[2026-03-01 11:21:48.784] [WARN ] [TR] Framework running in CPU-only scene (EDGE_ARM/EDGE_RISCV/CPU_CLOUD), deep learning device forced to CPU

========================================================================
Preprocessor V4.0 Performance Test
========================================================================

Configuration:
  Dataset path: /root/epfs/dataset/imagenet
  Format: DTS LV3
  Load workers: 4
  Preprocess workers: 4

Testing imagenet dts partial...
Device configured: CPU, GPUs=0

========================================
Training Set Test
========================================
Load time:        143.599 s
Total samples:    1281167
Expected samples: 1281167
Throughput:       0.310 GB/s
Integrity:        PASSED

========================================
Validation Set Test
========================================
Load time:        5.633 s
Total samples:    50000
Expected samples: 50000
Throughput:       0.338 GB/s
Integrity:        PASSED

========================================================================
All tests completed!
========================================================================

```



## 智星云

```shell
root@tech-renaissance:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_dataloader_performance --dataset imagenet --format dts --lv 0 --path /root/epfs/dataset/imagenet --mode partial --workers 16 --preproc 112

========================================================================
Preprocessor V4.0 Performance Test
========================================================================

Configuration:
  Dataset path: /root/epfs/dataset/imagenet
  Format: DTS LV0
  Load workers: 16
  Preprocess workers: 112

Testing imagenet dts partial...
Device configured: GPU, GPUs=8, Auto CPU Binding: True

========================================
Training Set Test
========================================
Load time:        4.258 s
Total samples:    1281167
Expected samples: 1281167
Throughput:       32.175 GB/s
Integrity:        PASSED

========================================
Validation Set Test
========================================
Load time:        0.295 s
Total samples:    50000
Expected samples: 50000
Throughput:       21.239 GB/s
Integrity:        PASSED

========================================================================
All tests completed!
========================================================================


```



## 并行智算云

```shell
root@cuda13-cudnn919-pytorch290-960gb-100m:~/epfs# /root/epfs/R/renaissance/build/bin/tests/pw/test_dataloader_performance --dataset imagenet --format dts --lv 0 --path /root/epfs/dataset/imagenet --mode partial --workers 16 --preproc 112

========================================================================
Preprocessor V4.0 Performance Test
========================================================================

Configuration:
  Dataset path: /root/epfs/dataset/imagenet
  Format: DTS LV0
  Load workers: 16
  Preprocess workers: 112

Testing imagenet dts partial...
Device configured: GPU, GPUs=8, Auto CPU Binding: True

========================================
Training Set Test
========================================
Load time:        4.863 s
Total samples:    1281167
Expected samples: 1281167
Throughput:       28.172 GB/s
Integrity:        PASSED

========================================
Validation Set Test
========================================
Load time:        0.318 s
Total samples:    50000
Expected samples: 50000
Throughput:       19.703 GB/s
Integrity:        PASSED

========================================================================
All tests completed!
========================================================================


```

