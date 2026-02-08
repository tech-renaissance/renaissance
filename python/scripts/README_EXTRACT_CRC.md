# DTS样本CRC-32提取工具使用说明

## 功能描述
从DTS文件中提取每个样本图像的CRC-32码，每行一个CRC-32码（8位16进制大写），类似于`official_val_sorted.txt`格式。

## 支持的DTS文件
- mnist_train.dts
- mnist_test.dts
- cifar10_train.dts
- cifar10_test.dts
- cifar100_train.dts
- cifar100_test.dts

## 使用方法

### 1. 基本用法
```bash
# 提取单个DTS文件的CRC-32码
python python/scripts/extract_dts_crc.py path/to/mnist_train.dts

# 提取多个DTS文件
python python/scripts/extract_dts_crc.py path/to/mnist_train.dts path/to/mnist_test.dts

# 提取所有MNIST和CIFAR的DTS文件
python python/scripts/extract_dts_crc.py \
    path/to/mnist_train.dts \
    path/to/mnist_test.dts \
    path/to/cifar10_train.dts \
    path/to/cifar10_test.dts \
    path/to/cifar100_train.dts \
    path/to/cifar100_test.dts
```

### 2. 指定输出目录
```bash
# 将CRC文件保存到指定目录
python python/scripts/extract_dts_crc.py \
    path/to/mnist_train.dts \
    path/to/mnist_test.dts \
    -o path/to/output_dir
```

### 3. 完整示例（假设DTS文件在T:/Datasets）
```bash
python python/scripts/extract_dts_crc.py \
    T:/Datasets/mnist/mnist_train.dts \
    T:/Datasets/mnist/mnist_test.dts \
    T:/Datasets/cifar-10/cifar10_train.dts \
    T:/Datasets/cifar-10/cifar10_test.dts \
    T:/Datasets/cifar-100/cifar100_train.dts \
    T:/Datasets/cifar-100/cifar100_test.dts \
    -o R:/renaissance/crc_lists
```

## 输出格式

### 文件命名
输入文件 → 输出文件
- mnist_train.dts → mnist_train_crc.txt
- mnist_test.dts → mnist_test_crc.txt
- cifar10_train.dts → cifar10_train_crc.txt
- cifar10_test.dts → cifar10_test_crc.txt
- cifar100_train.dts → cifar100_train_crc.txt
- cifar100_test.dts → cifar100_test_crc.txt

### CRC文件格式
每行一个CRC-32码，8位16进制大写：
```
0000CF19
0000E406
0001CC3B
0004100B
00041127
0005DC08
0006A623
0006DF6D
0006E325
0007146C
...
```

## 技术细节

### DTS文件结构（MNIST/CIFAR）
```
+-------------------+
| Header (256 bytes)|  <- SmallDtsHeader
+-------------------+
| Labels (N bytes)  |  <- N个label，每个1字节
+-------------------+
| Images (N×S bytes)|  <- N个样本，每个S字节
+-------------------+
```

- **MNIST**: 每个样本 28×28×1 = 784 字节
- **CIFAR-10/100**: 每个样本 32×32×3 = 3072 字节

### CRC-32计算
- 对每个样本的图像数据（不包括label）计算CRC-32
- 使用zlib.crc32()函数
- 输出为8位16进制大写（例如：0000CF19）

## 示例输出
```
Processing: T:/Datasets/mnist/mnist_train.dts
  Dataset: MNIST
  Samples: 60000
  Bytes per sample: 784
  Processed 10000/60000 samples...
  Processed 20000/60000 samples...
  Processed 30000/60000 samples...
  Processed 40000/60000 samples...
  Processed 50000/60000 samples...
  Processed 60000/60000 samples...
  Saved 60000 CRC-32 codes to: R:/renaissance/crc_lists/mnist_train_crc.txt
  Done!
```

## 注意事项
1. 确保DTS文件已经生成（使用`python/scripts/make_dataset.py`生成）
2. 输出目录会自动创建（如果不存在）
3. 脚本会跳过256字节的header
4. 只计算图像数据的CRC-32，不包括label
5. 每个样本的CRC-32独立计算，不是整个文件的CRC-32

## 验证
提取完成后，可以通过以下方式验证：
```bash
# 检查行数是否等于样本数
wc -l mnist_train_crc.txt

# MNIST train应该有60000行
# MNIST test应该有10000行
# CIFAR-10 train应该有50000行
# CIFAR-10 test应该有10000行
# CIFAR-100 train应该有50000行
# CIFAR-100 test应该有10000行
```
