# ImageNet数据加载器测试快速指南

## 一、快速编译和测试

### 1. 一键配置并编译

```bash
cd R:\renaissance
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake -G Ninja -S . -B build/windows-msvc-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake && cmake --build build/windows-msvc-debug --target test_imagenet_loader --parallel 30' }"
```

### 2. 测试验证集（约2秒）

```bash
cd R:\renaissance
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && build\windows-msvc-debug\bin\tests\integration\test_imagenet_loader.exe --dts --val --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16 2>&1' }"
```

**预期结果**:
- 加载时间: ~1.6秒
- Group数: 51个
- 速度: ~3.9 GB/s

### 3. 测试训练集（约40秒）

```bash
cd R:\renaissance
timeout 60 powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && build\windows-msvc-debug\bin\tests\integration\test_imagenet_loader.exe --dts --train --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16 2>&1' }"
```

**预期结果**:
- 加载时间: ~39秒
- Group数: 1,096个（全部完成）
- 速度: ~3.5 GB/s
- 样本数: 1,281,167张图片

### 4. 分析Group时间（需要先保存日志）

```bash
# 先运行测试并保存日志
cd R:\renaissance
build\windows-msvc-debug\bin\tests\integration\test_imagenet_loader.exe --dts --train --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16 > train_test.log 2>&1

# 然后分析
python analyze_group_timing.py train_test.log 50
```

## 二、测试参数说明

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dts` | 使用DTS格式 | 必需 |
| `--val` / `--train` | 加载验证集/训练集 | 必需 |
| `--lv 0-3` | 压缩级别（0=无压缩） | 可选，默认0 |
| `--path PATH` | 数据集路径 | 可选，默认I:/imagenet |
| `--workers N` | IO线程数 | 可选，默认8 |
| `--preprocess N` | 预处理线程数 | 可选，默认16 |
| `--simulate N` | 模拟预处理时间（ms） | 可选，默认0 |
| `--full` | 全量加载模式 | 可选，默认PARTIAL |
| `--no-shuffle` | 不打乱数据 | 可选，默认shuffle |
| `--save worker sample` | 保存图片 | 可选，默认不保存 |

### 推荐配置

#### 快速测试（验证集）
```bash
--dts --val --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16
```

#### 完整测试（训练集）
```bash
--dts --train --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16
```

#### 模拟预处理（测试流水线）
```bash
--dts --train --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16 --simulate 5
```

#### 保存第一张图片
```bash
--dts --val --lv 0 --path T:/dataset/imagenet --workers 8 --preprocess 16 --save 0 0
```

## 三、预期输出

### 验证集输出示例

```
========================================
Test Results
========================================
Load time: 1.64315 s
Total bytes: 0 GB
Speed: 0 MB/s
Total samples processed: 50000
Samples per second: 30429.4
========================================

Label distribution (first 20 classes):
  Label 0: 49 samples
  Label 1: 46 samples
  ...
```

### 训练集输出示例

```
[INFO] Group 0 ready: 1180 samples
[INFO] Group 1 ready: 1164 samples
[INFO] Group 2 ready: 1171 samples
...
[INFO] Group 1095 ready: 1209 samples

========================================
Test Results
========================================
Load time: 39.1234 s
Total bytes: 0 GB
Speed: 0 MB/s
Total samples processed: 1281167
Samples per second: 32754.2
========================================
```

## 四、故障排查

### 问题1: 找不到文件

**错误**: `系统找不到指定的路径`

**解决**:
- 检查数据集路径是否正确: `--path T:/dataset/imagenet`
- 检查文件是否存在: `ls T:/dataset/imagenet/*.dts`

### 问题2: 加载速度慢

**现象**: 训练集加载超过60秒

**检查**:
1. 查看磁盘I/O性能
2. 减少IO线程数: `--workers 4`
3. 检查是否在正确磁盘上（T盘 vs I盘）

### 问题3: Group数量不对

**现象**: Group数量少于预期

**检查**:
1. 查看日志中的Group ready数量
2. 检查是否被timeout中断
3. 增加timeout时间: `timeout 120`

### 问题4: Windows宏冲突

**错误**: `ERROR` 宏定义冲突

**状态**: 已在代码中处理，如果仍有问题，查看WINDOWS_PROBLEM.md

## 五、性能基准

### T盘性能基准

| 数据集 | 样本数 | Group数 | 平均间隔 | 速度 | 总时间 |
|--------|--------|---------|----------|------|--------|
| 验证集 | 50,000 | 51 | 32 ms | 3.9 GB/s | 1.6秒 |
| 训练集 | 1,281,167 | 1,096 | 36 ms | 3.5 GB/s | 39秒 |

### I盘性能参考

| 数据集 | 速度 | 说明 |
|--------|------|------|
| 验证集 | 14.9 GB/s | 可能是NVMe SSD |
| 训练集 | 0.13 GB/s | 有问题，已弃用 |

## 六、Python分析脚本

### analyze_group_timing.py 使用方法

```bash
# 基本用法
python analyze_group_timing.py <log_file> [num_groups_to_show]

# 示例
python analyze_group_timing.py train_test_T_90s_full.log 50
python analyze_group_timing.py val_test_T_full.log 20

# 输出内容
# - Group总数
# - 平均、最小、最大、中位数间隔
# - 加载速度（GB/s）
# - 预估总时间
# - 前N个Group的详细间隔
# - 最慢的10个Group
```

## 七、下一步

### 测试通过后

1. ✅ 集成到实际训练流程
2. ✅ 测试LV1-LV3压缩级别
3. ✅ 测试全量加载模式（--full）
4. ✅ 测试多GPU场景
5. ✅ 性能profiling和优化

### 联系方式

如有问题，查看:
- `docs/dataloader.md` - 完整测试报告
- `INFO_2.md` - 设计文档
- `WINDOWS_PROBLEM.md` - Windows平台问题

---

**最后更新**: 2026-01-09 22:50
**测试状态**: ✅ 全部通过
