# renAIssance框架编译指南

**版本**: V3.5.1
**更新日期**: 2025-12-24
**状态**: ✅ Windows + Linux 全平台验证通过

## 概述

本文档提供renAIssance深度学习框架的编译指南,包含Windows和Linux两个平台的正确编译方法。

**前置条件**: 必须先运行 `python configure.py` 生成配置文件

---

## 🪟 Windows平台编译方法

### 方法一:一键编译(推荐)

#### ✅ 正确的执行方法

**在PowerShell或cmd中直接执行**（推荐）:
```cmd
# 在PowerShell中
.\build.bat

# 或在cmd中
build.bat
```

**说明**:
- build.bat**内部已经包含**了vcvars64.bat调用
- 自动初始化Visual Studio环境
- 自动清理旧的构建目录
- 自动配置CMake并编译所有目标
- 输出到 `build/windows-msvc-release/`

**预期输出**:
```
[INFO] renAIssance Framework Build (Windows)
[INFO] Initializing Visual Studio Developer Command Prompt...
[vcvarsall.bat] Environment initialized for: 'x64'
[INFO] Configuring with CMake...
...
[OK] Build completed successfully!
```

#### ⚠️ 常见错误

**错误1**: 在Git Bash中执行`./build.bat`失败
- **原因**: Git Bash无法正确执行Windows批处理文件
- **解决**: 必须在PowerShell或cmd中执行

**错误2**: 提示"找不到build.bat"
- **原因**: 当前目录不在项目根目录
- **解决**: 先`cd R:/renaissance`再执行

**错误3**: 编译时出现"找不到iostream"
- **原因**: build.bat中的vcvars64.bat调用失败
- **解决**: 检查Visual Studio安装路径是否正确

---

### 方法二:PowerShell/CMake手动编译

#### ⚠️ 重要说明

**何时使用手动编译？**
- 需要自定义CMake参数
- 只编译特定目标（如只编译hello_cuda）
- 调试编译问题

**关键**: 手动编译**必须先在VS环境**中执行，在Git Bash中无法直接执行！

---

#### 方法2A：在VS Developer Command Prompt中执行（推荐）

**步骤1**: 打开"Developer Command Prompt for VS 2022"
- 开始菜单 → Visual Studio 2022 → Developer Command Prompt for VS 2022

**步骤2**: 切换到项目目录并编译
```cmd
cd R:\renaissance
cmake -G Ninja -S . -B build/windows-msvc-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build/windows-msvc-release --parallel 30
```

**优点**: 环境已预初始化，命令简单直接

---

#### 方法2B：在PowerShell中手动执行

**一键配置+编译**:
```powershell
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake -G Ninja -S . -B build/windows-msvc-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake && cmake --build build/windows-msvc-release --parallel 30' }"
```

**分步执行**（用于调试）:

步骤1 - 配置:
```powershell
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake -G Ninja -S . -B build/windows-msvc-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake' }"
```

步骤2 - 编译:
```powershell
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake --build build/windows-msvc-release --parallel 30' }"
```

---

#### 只编译特定目标

在VS Developer Command Prompt中:
```cmd
cmake --build build/windows-msvc-release --target hello_cuda
```

或在PowerShell中:
```powershell
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake --build build/windows-msvc-release --target hello_cuda' }"
```

---

### Windows关键技术要点

#### 1. Alpha编译方法

**核心**: vcvars64.bat初始化 + CMake编译

**为什么必须调用vcvars64.bat?**
- ✅ 初始化MSVC标准库路径
- ✅ 设置Windows SDK环境
- ✅ 配置链接器库路径
- ✅ 避免"找不到iostream"错误

#### 2. 目录命名规范

推荐使用 `build/windows-msvc-release` 而不是 `build` 或 `cmake-build-release`,明确标识平台和编译器。

---

## 🐧 Linux平台编译方法

### 方法一:一键编译(推荐)

```bash
./build.sh
```

**说明**:
- 自动清理旧的构建目录
- 使用Ninja构建系统
- 并行编译(30线程)
- 输出到 `build/`

---

### 方法二:CMake手动编译

#### 步骤1:配置CMake

```bash
cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

#### 步骤2:编译

```bash
cmake --build build --parallel $(nproc)
```

或者只编译特定目标:
```bash
cmake --build build --target hello_cuda
```

---

## 运行测试

### Windows

```cmd
.\build\windows-msvc-release\bin\dependency\hello_cuda.exe
```

### Linux

```bash
./build/bin/dependency/hello_cuda
```

**预期性能**:
- Windows (RTX 4060): 15.2 TFLOPS
- Linux (RTX 5090): 50.1 TFLOPS

---

## 常见问题

### Q1: Windows编译报错"找不到iostream"

**原因**: MSVC环境未初始化

**解决**: 在编译前必须执行
```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

或者直接使用 `build.bat`,它已经包含了这一步。

---

### Q2: cuDNN库找不到

**原因**: cuDNN 9.x使用了新的目录结构(`include/13.1/cudnn.h`)

**解决**: CMakeLists.txt已实现自动递归搜索,无需手动处理。

---

### Q3: 如何只编译某个测试程序?

**Windows**:
```cmd
cmake --build build/windows-msvc-release --target hello_cuda
```

**Linux**:
```bash
cmake --build build --target hello_cuda
```

---

## 编译流程对比

| 步骤 | Windows (一键) | Windows (手动-VS CMD) | Windows (手动-PowerShell) | Linux (一键) | Linux (手动) |
|------|--------------|----------------------|--------------------------|-------------|-------------|
| **1. 配置** | `python configure.py` | `python configure.py` | `python configure.py` | `python3 configure.py` | `python3 configure.py` |
| **2. 编译** | `build.bat` | 见方法2A | 见方法2B | `./build.sh` | 见方法二步骤1-2 |
| **3. 运行** | `build\...\bin\...\hello_cuda.exe` | 同左 | 同左 | `build/bin/dependency/hello_cuda` | 同左 |

**推荐方案**:
- **Windows**: 优先使用方法2A（VS Developer Command Prompt）
- **Linux**: 使用一键编译`./build.sh`

---

## 技术原理说明

### Alpha编译方法的核心价值

**问题**: MSVC编译器找不到标准库头文件(iostream等)

**根源**: Visual Studio的环境变量(INCLUDE, LIB等)未初始化

**解决**: 调用vcvars64.bat初始化MSVC Developer Command Prompt环境

**vcvars64.bat做了什么**:
1. 设置INCLUDE环境变量 → 指向MSVC标准库头文件
2. 设置LIB环境变量 → 指向MSVC库文件
3. 添加MSVC工具链到PATH
4. 初始化Windows SDK环境

**验证成功**: 如果看到输出 `[vcvarsall.bat] Environment initialized for: 'x64'`,说明环境初始化成功。

---

## 总结

### Windows编译的正确方法

**核心原理**:
- build.bat**内部已经调用**了vcvars64.bat
- **不需要**在外部再调用vcvars64.bat
- 只要在PowerShell或cmd中执行`build.bat`即可

**三种编译方式对比**:

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **一键: build.bat** | 日常编译 | 简单直接，环境自动初始化 | 无法自定义参数 |
| **手动: VS CMD** | 调试、自定义 | 环境预初始化，命令清晰 | 需要打开特殊终端 |
| **手动: PowerShell** | 脚本自动化 | 可以一次性执行多条命令 | 命令复杂 |

**推荐工作流程**:
1. **日常开发**: 使用`build.bat`（在PowerShell中执行）
2. **调试问题**: 使用VS Developer Command Prompt手动执行
3. **CI/CD**: 使用PowerShell命令行

**常见误区**:
- ❌ 错误：在Git Bash中执行`./build.bat`
- ✅ 正确：在PowerShell或cmd中执行`build.bat`
- ❌ 错误：手动执行时忘记调用vcvars64.bat
- ✅ 正确：使用VS Developer Command Prompt（已预初始化）

---

### Linux编译要点

1. ✅ 使用GCC 13+编译器
2. ✅ 使用Ninja构建系统
3. ✅ 并行编译加速

---

### 最佳实践

**首次使用**:
1. 运行`python configure.py`生成配置文件
2. 验证config/目录和build.bat/build.sh已生成
3. Windows: 在PowerShell中执行`build.bat`
4. Linux: 执行`./build.sh`

**遇到问题时**:
1. 检查是否运行了configure.py
2. Windows: 检查是否在PowerShell/cmd中执行
3. 查看错误信息，参考常见问题部分

---

**文档版本**: V3.5.1
**最后更新**: 2025-12-24
**作者**: renAIssance开发团队
