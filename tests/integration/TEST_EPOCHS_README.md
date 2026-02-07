# Multi-Epoch Test (test_epochs)

## 测试目的

验证DataLoader能够在训练集和验证集之间正确切换，这是训练过程中的核心功能。

## 测试流程

```
验证集(Epoch -1) → 训练集(Epoch 0) → 验证集(Epoch 0) → 训练集(Epoch 1) → 验证集(Epoch 1)
```

### 详细说明

1. **验证集 (Epoch -1)**: 训练前的验证，用于评估模型初始状态
2. **训练集 (Epoch 0)**: 第一个训练epoch（从0开始计数）
3. **验证集 (Epoch 0)**: 第一个训练epoch后的验证
4. **训练集 (Epoch 1)**: 第二个训练epoch
5. **验证集 (Epoch 1)**: 第二个训练epoch后的验证

## 关键验证点

### 1. 集合切换正确性
- ✅ 从验证集切换到训练集
- ✅ 从训练集切换到验证集
- ✅ 多次往返切换

### 2. 样本完整性
- ✅ 每次加载的样本数匹配预期值
- ✅ Worker负载均衡（最多相差1个样本）

### 3. 状态管理
- ✅ `begin_epoch()` 正确设置集合类型
- ✅ `end_epoch()` 正确清理状态
- ✅ 静态分配机制在切换时仍然有效

## 使用方法

### 基本用法

```bash
# 测试ImageNet DTS LV0 (PARTIAL模式)
./test_epochs --dataset imagenet --format dts --path T:/Datasets --lv 0

# 测试CIFAR-10 RAW (FULLY模式)
./test_epochs --dataset cifar10 --format raw --path T:/Datasets --mode fully

# 测试MNIST DTS
./test_epochs --dataset mnist --format dts --path T:/Datasets
```

### 完整参数

```
Required Options:
  --dataset <NAME>     Dataset to test: mnist, cifar10, cifar100, imagenet
  --format <FMT>       Format: raw, dts
  --path <PATH>        Dataset root path (default: T:/Datasets)

Optional Options:
  --mode <MODE>        Mode: partial, fully (default: partial)
  --lv <0-3>           DTS compression level for ImageNet (default: 0)
  --workers <N>        Number of load workers (default: 8)
  --preproc <N>        Number of preprocess workers (default: 16)
  --help               Show help message
```

## 输出示例

```
========================================================================
Multi-Epoch Test: imagenet dts partial
========================================================================

Configuration:
  Dataset: ImageNet
  Format: DTS LV0
  Mode: PARTIAL
  Load workers: 8
  Preprocess workers: 16
  Train samples: 1281167
  Val samples: 50000

Test Flow:
  1. Validation Set (Epoch -1) - Pre-training validation
  2. Training Set (Epoch 0)
  3. Validation Set (Epoch 0)
  4. Training Set (Epoch 1)
  5. Validation Set (Epoch 1)

Running epochs...

Epoch -1 (Pre-training validation):
  Validation Epoch -1: 50000 samples (PASSED), 0.546 s

Epoch 0:
  Training Epoch 0: 1281167 samples (PASSED), 2.445 s

Epoch 0 (Validation):
  Validation Epoch 0: 50000 samples (PASSED), 0.543 s

Epoch 1:
  Training Epoch 1: 1281167 samples (PASSED), 2.438 s

Epoch 1 (Validation):
  Validation Epoch 1: 50000 samples (PASSED), 0.545 s

========================================================================
SUCCESS: All epochs completed with integrity check PASSED!
========================================================================
```

## 技术细节

### 实现原理

1. **静态分配机制**: 每个worker静态领取样本，不需要锁
   - Worker i的第k次调用 → 读取第 `(i + k×M)` 个样本
   - M = num_preproc_workers

2. **状态管理**:
   - `begin_epoch(epoch_id, is_train)`: 切换到指定集合
   - `end_epoch()`: 清理当前epoch状态
   - `get_next_sample(worker_id, ...)`: 获取下一个样本

3. **零拷贝设计**:
   - 返回的指针直接指向DataLoader内部缓冲区
   - 避免数据复制，提升性能

### 与其他测试的区别

| 测试 | 目的 | 特点 |
|------|------|------|
| `test_dataloader_performance` | 测试加载性能 | 只测试一个方向（train→val或val→train） |
| `test_epochs` | 测试集合切换 | 测试多次往返切换（val→train→val→train→val） |
| `test_reproducibility` | 测试可复现性 | 验证随机seed和跨epoch一致性 |

## 故障排查

### 问题1: 样本数不匹配

**症状**: 显示 "FAILED" 而不是 "PASSED"

**可能原因**:
- DataLoader配置错误
- 文件路径不正确
- 数据集文件损坏

**解决方案**:
- 检查文件路径是否正确
- 使用`test_dataloader_performance`验证数据集完整性
- 检查DataLoader配置参数

### 问题2: Worker负载不均衡

**症状**: 显示 "WARNING: Worker load imbalance detected"

**可能原因**:
- 样本总数不能被worker数整除
- 静态分配机制实现错误

**说明**:
- 差异≤1是正常的（样本数不能被worker数整除）
- 差异>1表示实现有bug

## 版本历史

- **V1.0.0** (2026-02-05): 初始版本
  - 支持所有6种数据集（MNIST/CIFAR-10/CIFAR-100/ImageNet × RAW/DTS）
  - 支持PARTIAL和FULLY模式
  - 验证完整性检查和负载均衡

## 作者

技术觉醒团队

## 许可

MIT License
