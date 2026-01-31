# Windows平台测试计划（V4.0共享双缓冲验证）

**日期**: 2026-01-26
**目的**: 验证train/val在PARTIAL模式下共用双缓冲的正确性
**平台**: Windows 11
**数据集路径**: `T:/dataset/imagenet`

---

## 📋 测试清单

### 测试1-2: PARTIAL模式性能+完整性测试

**目标**: 验证PARTIAL模式的吞吐量和数据完整性

| 测试 | 数据集 | 命令 | 预期结果 |
|------|--------|------|---------|
| 1 | Val LV0 (6.4GB) | `.\test_partial_mode.exe --dts --path T:/dataset/imagenet --val --lv 0 --workers 16` | 吞吐量≥70 GB/s, 50000/50000样本 |
| 2 | Train LV0 (137GB) | `.\test_partial_mode.exe --dts --path T:/dataset/imagenet --train --lv 0 --workers 16` | 吞吐量≥40 GB/s, 1281167/1281167样本 |

**验证点**:
- ✅ 吞吐量符合预期（缓存场景）
- ✅ 样本数100%完整
- ✅ Worker样本分布均匀（diff≤1）
- ✅ 零数据丢失、零重复

---

### 测试3-4: FULLY模式性能+完整性测试

**目标**: 验证FULLY模式的吞吐量和数据完整性

| 测试 | 数据集 | 命令 | 预期结果 |
|------|--------|------|---------|
| 3 | Val LV0 (6.4GB) | `.\test_fully_mode.exe --dts --path T:/dataset/imagenet --val --lv 0 --workers 16` | 吞吐量≥3 GB/s, 50000/50000样本 |
| 4 | Train LV0 (137GB) | `.\test_fully_mode.exe --dts --path T:/dataset/imagenet --train --lv 0 --workers 16` | 吞吐量≥2.5 GB/s, 1281167/1281167样本 |

**验证点**:
- ✅ 吞吐量符合预期（真实IO）
- ✅ 样本数100%完整
- ✅ Worker样本分布均匀（diff≤1）
- ✅ FULLY模式后续epoch加速（如果测试）

---

### 测试5: 不shuffle可复现性 - 不同seed

**目标**: 验证不同seed产生不同的结果

**命令**:
```powershell
# seed 42
.\test_reproducibility.exe --dts --path T:/dataset/imagenet --train --lv 3 --workers 16 --preprocess 64 --seed 42 --out R:/renaissance/workspace/final_seed_42

# seed 12345 (第1次)
.\test_reproducibility.exe --dts --path T:/dataset/imagenet --train --lv 3 --workers 16 --preprocess 64 --seed 12345 --out R:/renaissance/workspace/final_seed_12345_run1
```

**验证**:
```powershell
# 计算MD5哈希
Get-FileHash R:\renaissance\workspace\final_seed_42\worker_0.csv -Algorithm MD5
Get-FileHash R:\renaissance\workspace\final_seed_12345_run1\worker_0.csv -Algorithm MD5
```

**预期结果**: 两个MD5哈希应该**不同**

---

### 测试6: 不shuffle可复现性 - 相同seed

**目标**: 验证相同seed产生相同的结果

**命令**:
```powershell
# seed 12345 (第2次)
.\test_reproducibility.exe --dts --path T:/dataset/imagenet --train --lv 3 --workers 16 --preprocess 64 --seed 12345 --out R:/renaissance/workspace/final_seed_12345_run2
```

**验证**:
```powershell
# 计算MD5哈希
Get-FileHash R:\renaissance\workspace\final_seed_12345_run1\worker_0.csv -Algorithm MD5
Get-FileHash R:\renaissance\workspace\final_seed_12345_run2\worker_0.csv -Algorithm MD5
```

**预期结果**: 两个MD5哈希应该**相同**

---

### 测试7: epoch间可复现性

**目标**: 验证FULLY模式下不shuffle时，多个epoch的内容完全一致

**命令**:
```powershell
.\test_cross_epoch_reproducibility.exe --dts --path T:/dataset/imagenet --val --lv 0 --io-workers 16 --preprocess 64
```

**预期结果**:
- ✅ 两个epoch的日志完全匹配
- ✅ 所有worker的样本序列一致
- ✅ 50000/50000样本，零差异

---

## 🔍 手动验证步骤

### MD5哈希验证脚本

保存以下内容为 `verify_reproducibility.ps1`:

```powershell
# 验证不同seed产生不同结果
Write-Host "=== 测试5: 不同seed验证 ===" -ForegroundColor Cyan
$hash1 = (Get-FileHash "R:\renaissance\workspace\final_seed_42\worker_0.csv" -Algorithm MD5).Hash
$hash2 = (Get-FileHash "R:\renaissance\workspace\final_seed_12345_run1\worker_0.csv" -Algorithm MD5).Hash
Write-Host "seed 42:            $hash1"
Write-Host "seed 12345 (run1):  $hash2"
if ($hash1 -eq $hash2) {
    Write-Host "[FAILED] 不同seed产生了相同结果！" -ForegroundColor Red
} else {
    Write-Host "[PASSED] 不同seed产生了不同结果" -ForegroundColor Green
}

# 验证相同seed产生相同结果
Write-Host ""
Write-Host "=== 测试6: 相同seed验证 ===" -ForegroundColor Cyan
$hash3 = (Get-FileHash "R:\renaissance\workspace\final_seed_12345_run2\worker_0.csv" -Algorithm MD5).Hash
Write-Host "seed 12345 (run1):  $hash2"
Write-Host "seed 12345 (run2):  $hash3"
if ($hash2 -eq $hash3) {
    Write-Host "[PASSED] 相同seed产生了相同结果" -ForegroundColor Green
} else {
    Write-Host "[FAILED] 相同seed产生了不同结果！" -ForegroundColor Red
}
```

---

## 📊 测试结果记录模板

### 性能测试结果

| 测试 | 数据集 | 吞吐量 | 时间 | 样本数 | Worker分布 | 结果 |
|------|--------|--------|------|--------|-----------|------|
| 1 | Val LV0 | __ GB/s | __ s | 50000/50000 | min=___, max=___, diff=___ | ⬜ |
| 2 | Train LV0 | __ GB/s | __ s | 1281167/1281167 | min=___, max=___, diff=___ | ⬜ |
| 3 | Val LV0 | __ GB/s | __ s | 50000/50000 | min=___, max=___, diff=___ | ⬜ |
| 4 | Train LV0 | __ GB/s | __ s | 1281167/1281167 | min=___, max=___, diff=___ | ⬜ |

### 可复现性测试结果

| 测试 | seed | MD5哈希 | 预期 | 实际 | 结果 |
|------|------|---------|------|------|------|
| 5 | 42 vs 12345 | 不同 | 应该不同 | ⬜ | ⬜ |
| 6 | 12345 × 2 | 相同 | 应该相同 | ⬜ | ⬜ |
| 7 | epoch 0 vs 1 | 一致 | 应该一致 | ⬜ | ⬜ |

---

## ⚠️ 注意事项

1. **缓存影响**:
   - 测试1-2可能会显示80+ GB/s的吞吐量（这是正常的，因为数据在系统缓存中）
   - 如果要测试真实IO性能，需要重启系统或使用`--clear-cache`选项

2. **内存要求**:
   - 验证集LV0测试需要约8GB内存
   - 训练集LV0 FULLY模式需要至少160GB内存（如果测试失败，使用PARTIAL模式）

3. **worker数量**:
   - 本脚本使用16 IO workers和64 preprocess workers
   - 这是推荐配置，可以根据系统调整

4. **工作目录**:
   - 测试会在 `R:/renaissance/workspace/` 下生成日志
   - 确保该目录存在且可写

---

## 🚀 执行测试

### 方法1: 自动执行所有测试

```cmd
cd R:\renaissance
run_windows_tests.bat
```

### 方法2: 手动逐个测试

```cmd
cd R:\renaissance\build\windows-msvc-release\bin\tests\data

# 测试1-4: 性能测试
.\test_partial_mode.exe --dts --path T:/dataset/imagenet --val --lv 0 --workers 16
.\test_partial_mode.exe --dts --path T:/dataset/imagenet --train --lv 0 --workers 16
.\test_fully_mode.exe --dts --path T:/dataset/imagenet --val --lv 0 --workers 16
.\test_fully_mode.exe --dts --path T:/dataset/imagenet --train --lv 0 --workers 16

# 测试5-7: 可复现性测试
.\test_reproducibility.exe --dts --path T:/dataset/imagenet --train --lv 3 --workers 16 --preprocess 64 --seed 42 --out R:/renaissance/workspace/final_seed_42
.\test_reproducibility.exe --dts --path T:/dataset/imagenet --train --lv 3 --workers 16 --preprocess 64 --seed 12345 --out R:/renaissance/workspace/final_seed_12345_run1
.\test_reproducibility.exe --dts --path T:/dataset/imagenet --train --lv 3 --workers 16 --preprocess 64 --seed 12345 --out R:/renaissance/workspace/final_seed_12345_run2
.\test_cross_epoch_reproducibility.exe --dts --path T:/dataset/imagenet --val --lv 0 --io-workers 16 --preprocess 64
```

---

## 📝 测试报告模板

测试完成后，请填写以下报告：

```
测试日期：2026-01-26
测试人员：_________
编译版本：V4.0 (共享双缓冲)

测试结果总结：
- 测试通过数：___ / 7
- 测试失败数：___ / 7

性能指标：
- PARTIAL模式验证集：___ GB/s
- PARTIAL模式训练集：___ GB/s
- FULLY模式验证集：___ GB/s
- FULLY模式训练集：___ GB/s

可复现性：
- 不同seed：⬜ 通过 / ⬜ 失败
- 相同seed：⬜ 通过 / ⬜ 失败
- epoch间：⬜ 通过 / ⬜ 失败

问题记录：
（如有问题请详细记录）

结论：
⬜ 所有测试通过，共享双缓冲修改正确
⬜ 部分测试失败，需要修复
```

---

**文档版本**: 1.0
**创建日期**: 2026-01-26
**作者**: Renaissance 技术觉醒团队
