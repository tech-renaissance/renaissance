# collect_crc.py 使用指南

## 功能描述

`collect_crc.py` 是一个用于计算目录下所有 JPEG 文件 CRC32 校验值的 Python 脚本。它可以：
- 递归遍历目录及其子目录
- 支持多种 JPEG 格式（.jpg, .jpeg, .JPG, .JPEG 等）
- 多线程并行处理，提高计算速度
- 输出标准 CSV 格式，便于后续分析

## 安装依赖

脚本使用 Python 标准库，无需额外安装依赖包。需要 Python 3.6+。

```bash
# 检查 Python 版本
python --version
```

## 使用方法

### 基本用法

```bash
# 计算指定目录下所有JPEG文件的CRC32值
python collect_crc.py /path/to/directory

# 示例：ImageNet训练集
python collect_crc.py /root/dataset/imagenet/train

# 示例：Windows路径
python collect_crc.py T:\Dataset\imagenet\train
```

### 高级选项

```bash
# 指定输出文件名
python collect_crc.py /path/to/dir --output custom_crc.csv

# 单线程模式（资源受限时使用）
python collect_crc.py /path/to/dir --single-threaded

# 指定线程数（默认为CPU核心数）
python collect_crc.py /path/to/dir --workers 16

# 验证输出文件（自动检查行数）
python collect_crc.py /path/to/dir --verify

# 组合使用
python collect_crc.py /path/to/dir --output results.csv --workers 8 --verify
```

## 输出格式

### CSV 文件格式

生成的 CSV 文件包含两列：

```csv
filepath,crc32
n01440764/n01440764_10026.JPEG,77C8EBEC
n01440764/n01440764_10027.JPEG,4D4B9383
n01440764/n01440764_10028.JPEG,99744B7C
...
```

**字段说明**：
- `filepath`: 相对于输入目录的文件路径（使用 `/` 分隔符）
- `crc32`: 8 位 16 进制大写的 CRC32 校验值（例如：`77C8EBEC`）

### 文件路径说明

- 路径是相对于输入目录的相对路径
- 使用正斜杠 `/` 作为分隔符（跨平台兼容）
- 保留原始目录结构

例如，对于输入目录 `T:\Dataset\imagenet\train`：
- 文件：`T:\Dataset\imagenet\train\n01440764\n01440764_10026.JPEG`
- 输出：`n01440764/n01440764_10026.JPEG`

## 性能参考

不同数据集的性能参考（使用默认多线程模式）：

| 数据集 | 文件数 | 总大小 | 耗时 | 吞吐量 |
|--------|--------|--------|------|--------|
| MNIST 训练集 | 60,000 | ~50 MB | ~3 秒 | ~20K files/s |
| CIFAR-10 训练集 | 50,000 | ~160 MB | ~5 秒 | ~10K files/s |
| ImageNet 训练集 | 1,281,167 | ~140 GB | ~300 秒 | ~4.3K files/s |

*注：性能取决于硬件配置（CPU、磁盘速度等）*

## 使用示例

### 示例 1：MNIST 训练集

```bash
python python/tests/collect_crc.py T:\Dataset\mnist\train --output mnist_train_crc.csv --verify
```

**预期输出**：
```
======================================================================
CRC32 Collection Tool
======================================================================
Input directory: T:\Dataset\mnist\train
Output file: mnist_train_crc.csv
Single-threaded: False
======================================================================

[INFO] Scanning directory: T:\Dataset\mnist\train
[INFO] Mode: Multi-threaded (16 workers)
[INFO] Found 60000 JPEG files
[PROGRESS] 10000/60000 files (16.7%)
[PROGRESS] 20000/60000 files (33.3%)
[PROGRESS] 30000/60000 files (50.0%)
[PROGRESS] 40000/60000 files (66.7%)
[PROGRESS] 50000/60000 files (83.3%)
[INFO] Completed: 60000 files processed, 0 errors

======================================================================
[SUCCESS] CRC32 collection completed!
[STATS] Total files: 60000
[STATS] Elapsed time: 3.25 seconds
[STATS] Throughput: 18461.5 files/sec
[OUTPUT] CRC32 values saved to: mnist_train_crc.csv
======================================================================

[VERIFY] Checking output file...
[VERIFY] Total rows in CSV: 60000
[VERIFY] ✓ Row count matches processed files!
```

### 示例 2：ImageNet 训练集

```bash
python python/tests/collect_crc.py T:\Dataset\imagenet\train --output imagenet_train_crc.csv --workers 16 --verify
```

**预期结果**：
- 文件数：1,281,167
- 输出文件大小：约 60-80 MB
- 耗时：约 5-10 分钟（取决于硬件）

### 示例 3：单线程模式（调试用）

```bash
# 单线程模式，便于调试或资源受限环境
python python/tests/collect_crc.py /path/to/small_dataset --single-threaded
```

## 常见问题

### Q1: 脚本支持哪些图片格式？

A: 支持常见的 JPEG 格式扩展名：
- `.jpg`, `.jpeg`, `.JPG`, `.JPEG`
- `.jpe`, `.JPE`

不支持 PNG、BMP 等其他格式。

### Q2: 如何验证 CRC32 计算是否正确？

A: 可以使用 `--verify` 选项自动验证，或手动检查：

```bash
# 方法1：使用 --verify 选项
python collect_crc.py /path/to/dir --verify

# 方法2：手动检查行数
wc -l crc.csv  # Linux/Mac
Get-Content crc.csv | Measure-Object -Line  # Windows PowerShell

# 方法3：使用Python验证
python -c "import csv; print(sum(1 for _ in csv.reader(open('crc.csv')))-1)"
```

### Q3: 内存占用会很高吗？

A: 不会。脚本采用流式处理，不会一次性加载所有文件到内存。内存占用主要取决于线程数。

### Q4: 如何中断正在运行的脚本？

A: 按 `Ctrl+C` 安全中断。已处理的结果会保存在输出文件中（但可能不完整）。

### Q5: 输出文件格式可以修改吗？

A: 可以修改脚本中的 `calculate_crc()` 函数来改变输出格式。当前格式：
- CSV 格式
- 第一列：文件相对路径
- 第二列：8 位 16 进制大写 CRC32 值

## 与 Renaissance 框架集成

这个脚本可以用于：
1. **数据完整性验证**：对比 RAW 格式和 DTS 格式的 CRC32 值
2. **数据集验证**：检查数据集是否损坏
3. **格式转换验证**：验证数据格式转换的正确性

### 使用 Renaissance 生成的 CRC32 对比

```bash
# 1. 使用 Renaissance 生成 DTS 格式的 CRC32
test_crc_logging.exe --dataset imagenet --format dts --lv 0 --path T:/Dataset/imagenet --preproc 8 --samples 0

# 2. 使用 Python 脚本生成 RAW 格式的 CRC32
python python/tests/collect_crc.py T:\Dataset\imagenet\train --output raw_train_crc.csv --workers 16

# 3. 对比两个 CSV 文件（确保样本顺序一致）
# 可以编写简单的 Python 脚本进行对比
```

## 技术细节

### CRC32 计算方法

使用 Python 标准库 `zlib.crc32()`：
```python
import zlib
crc_value = zlib.crc32(data) & 0xffffffff  # 确保是无符号32位整数
crc_hex = f"{crc_value:08X}"  # 转换为8位16进制大写
```

### 多线程实现

使用 `concurrent.futures.ThreadPoolExecutor`：
- 默认线程数：CPU 核心数
- 使用线程锁保护文件写入
- 实时写入，不占用内存

### 跨平台兼容性

- Windows：支持 `\\` 和 `/` 路径分隔符
- Linux/Mac：支持 `/` 路径分隔符
- 输出统一使用 `/` 作为路径分隔符

## 许可证

本脚本是 Renaissance 深度学习框架的一部分，遵循框架的许可证。

## 作者

技术觉醒团队 (Technical Awakening Team)

## 版本历史

- **V1.0.0** (2026-02-05): 初始版本
  - 支持递归遍历目录
  - 多线程并行处理
  - CSV 格式输出
  - 8 位 16 进制大写 CRC32 值
