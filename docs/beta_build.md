# Renaissance Framework 编译指南

## 概述

本文档记录了Renaissance框架的Debug和Release模式编译方法。

**关键发现**：Debug模式编译需要显式指定 `-march=x86-64-v3` 选项，否则会因为AVX指令集导致编译失败。

---

## Release模式编译（使用build.sh脚本）

直接使用项目根目录的 `build.sh` 脚本：

```bash
cd /root/epfs/R/renaissance
./build.sh
```

这个脚本使用Ninja构建系统，编译速度更快。

---

## Debug模式编译

### 问题背景

默认的Debug模式编译会失败，错误信息：

```
cc1plus: error: inlining failed in call to 'always_inline' '__m256i _mm256_cvtps_epi32(__m256)': target specific option mismatch
```

这是因为代码中使用了AVX2指令（如 `_mm256_cvtps_epi32`），但Debug模式默认没有启用AVX指令集支持。

### 解决方法

在CMake配置时显式添加 `-march=x86-64-v3` 选项：

```bash
cd /root/epfs/R/renaissance
rm -rf build
mkdir build
cd build

cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -march=x86-64-v3" \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_DISABLE_SOURCE_CHANGES=ON \
  -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
  -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON \
  -DTR_SCENE_GPU_CLOUD=ON \
  -DTR_USE_CUDA=ON \
  -DTR_USE_MUSA=OFF \
  -DTR_USE_ONEDNN=ON \
  -DTR_USE_XNNPACK=ON \
  -DTR_USE_NCCL=ON \
  -DTR_USE_MULTI_GPU=ON \
  -DTR_USE_STB=ON \
  -DTR_USE_LIBJPEG=ON \
  -DTR_USE_ZLIB=ON \
  -DTR_USE_LIBCURL=ON \
  -DTR_USE_LIBARCHIVE=ON \
  -DTR_USE_MIMALLOC=ON \
  -DTR_USE_SIMD=ON

# 编译
ninja -j30
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `-G Ninja` | - | 使用Ninja构建系统（更快） |
| `-DCMAKE_BUILD_TYPE` | `Debug` | Debug模式 |
| `-DCMAKE_CXX_FLAGS_DEBUG` | `-g -O0 -march=x86-64-v3` | **关键**：添加x86-64-v3架构支持 |
| `-DCMAKE_TOOLCHAIN_FILE` | vcpkg.cmake | 使用vcpkg包管理 |

**重要**：`-march=x86-64-v3` 中的是**连字符** `x86-64-v3`，不是下划线 `x86_64-v3`！

---

## Release模式带调试符号编译

如果需要在Release模式下保留调试符号（用于gdb调试），使用：

```bash
cd /root/epfs/R/renaissance
rm -rf build
mkdir build
cd build

cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-g -O3" \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_DISABLE_SOURCE_CHANGES=ON \
  -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
  -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON \
  -DTR_SCENE_GPU_CLOUD=ON \
  -DTR_USE_CUDA=ON \
  -DTR_USE_MUSA=OFF \
  -DTR_USE_ONEDNN=ON \
  -DTR_USE_XNNPACK=ON \
  -DTR_USE_NCCL=ON \
  -DTR_USE_MULTI_GPU=ON \
  -DTR_USE_STB=ON \
  -DTR_USE_LIBJPEG=ON \
  -DTR_USE_ZLIB=ON \
  -DTR_USE_LIBCURL=ON \
  -DTR_USE_LIBARCHIVE=ON \
  -DTR_USE_MIMALLOC=ON \
  -DTR_USE_SIMD=ON

# 编译
ninja -j30
```

---

## 编译单个测试程序

```bash
# 在build目录下
ninja test_crc_logging -j30

# 或使用make（如果使用Unix Makefiles）
make test_crc_logging -j$(nproc)
```

---

## 常见问题

### Q1: 为什么Debug模式需要 `-march=x86-64-v3`？

A: 代码中使用了AVX2指令集（如 `_mm256_cvtps_epi32`），这些指令需要在编译时启用对应的CPU指令集支持。`x86-64-v3` level包含了AVX2指令集。

### Q2: `x86-64-v3` 和 `x86_64-v3` 有什么区别？

A: 前者是正确的（连字符），后者是错误的（下划线），会导致编译错误：
```
cc1plus: error: bad value 'x86_64-v3' for '-march=' switch
```

### Q3: Release模式为什么不需要显式指定？

A: Release模式在 `CMakeLists.txt` 中已经设置了 `-march=native`（第255行），会自动启用当前CPU支持的所有指令集。

---

## 架构级别说明

| Level | 特性 | 最低CPU |
|-------|------|---------|
| x86-64-v2 | SSE3/SSE4 | 任何x86_64 CPU |
| x86-64-v3 | AVX2 | Haswell (2013+) |
| x86-64-v4 | AVX-512 | Skylake-X/Cannonlake |

本框架使用 `x86-64-v3`，需要Haswell或更新的CPU。

---

## 版本信息

- 创建日期: 2026-02-06
- 框架版本: V3.11.1
- CMake版本: 3.28.3
- GCC版本: 13.x
