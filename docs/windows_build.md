# Windows 增量编译指南

## 问题现象

在 PowerShell 或普通 `cmd.exe` 中，如果直接运行：

```powershell
cd R:\renaissance\build\windows-msvc-release
ninja mlp_test
# 或
cmake --build . --target mlp_test --config Release
```

链接阶段会报错：

```
LINK : fatal error LNK1181: 无法打开输入文件"kernel32.lib"
ninja: build stopped: subcommand failed.
```

## 根本原因

Visual Studio 的链接器 `link.exe` 依赖系统库（`kernel32.lib`、`user32.lib` 等），这些库的路径由 **Visual Studio Developer Command Prompt** 的环境变量（`LIB`、`LIBPATH` 等）提供。

普通 PowerShell / cmd 没有加载这些环境变量，因此链接器找不到系统库。项目根目录下的 `build.bat` 之所以不会遇到这个问题，是因为它的开头显式调用了：

```bat
call "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
```

而 `rb.bat` 又会调用 `build.bat`，所以走 `rb.bat` 全量重建时不会遇到此错误。

## 正确做法：增量编译单目标

### Kimi Code 使用方法

如果你只想编译修改过的单个目标（如 `mlp_test`），不想执行 `rb.bat` 删掉整个 build 目录重新 configure，请在 **PowerShell** 中执行：

```powershell
& cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cd /d R:\renaissance\build\windows-msvc-release && ninja mlp_test'
```

### Claude Code 使用方法

在Claude Code环境中，由于运行在bash shell中，需要使用PowerShell.exe来调用正确的MSVC环境：

```bash
powershell.exe -Command "& cmd /c '\"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance\build\windows-msvc-release && ninja mlp_test'"
```

**关键区别**：
- **Claude Code环境**：bash shell → 需要通过 `powershell.exe -Command` 调用
- **直接PowerShell环境**：直接执行命令即可

**验证编译成功**：
看到类似以下输出表示编译成功：
```
[1/1] Linking CUDA executable bin\tests\simple\mlp_test.exe; Copying cuDNN DLL to target directory
```

### 命令拆解

| 部分 | 作用 |
|------|------|
 `"C:\Program Files\...\vcvars64.bat"` | 初始化 x64 MSVC 环境变量（`LIB`、`INCLUDE`、`PATH` 等） |
 `&& cd /d R:\renaissance\build\windows-msvc-release` | 切换到构建目录 |
 `&& ninja mlp_test` | 只增量编译 `mlp_test` 目标及其依赖 |

### 为什么用 `cmd /c '...'` 包裹？

PowerShell 的引号规则与 cmd 不同。`vcvars64.bat` 路径包含空格，必须被 cmd 正确解析。`cmd /c '"..." && ...'` 这种写法确保：

1. 外层的单引号让 PowerShell 把整个字符串传给 cmd
2. 内层的 `"..."` 让 cmd 把带空格的路径当作一个参数

### 验证环境已加载

执行后如果看到如下输出，说明 MSVC 环境初始化成功：

```
**********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.14.18
** Copyright (c) 2025 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'
```

## 全量重建（官方推荐）

如果你不确定构建状态是否干净，或 CMake 配置有变更，仍应使用项目提供的：

```bat
R:\rb.bat
```

这会：
1. 删除 `config/`、`build/`、`workspace/` 目录
2. 运行 `python configure.py` 重新生成构建配置
3. 调用 `build.bat`（内部自动加载 `vcvars64.bat`）全量编译

> ⚠️ `rb.bat` 是全量重建，耗时较长，适合首次构建或 CMake 配置变更后使用。日常开发修改少量 `.cu`/`.cpp` 源文件时，建议使用上文的 `ninja` 增量编译。

## 常见问题补充

### cuDNN 9.x DLL 依赖

Windows 下运行依赖 cuDNN 的可执行文件时，如果报错：

```
Invalid handle. Cannot load symbol cudnnGetVersion
```

即使 `cudnn64_9.dll` 已放在同目录，也可能是因为 cuDNN 9.x **拆分成了多个 DLL**：

- `cudnn64_9.dll`（stub loader，仅 266KB）
- `cudnn_graph64_9.dll`（图 API）
- `cudnn_ops64_9.dll`（ops API）
- `cudnn_cnn64_9.dll`（CNN API）
- 其他 engine/heuristic DLL

**解决方案**：将 `C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1\cudnn*.dll` 全部复制到可执行文件目录。项目 `build.bat` 已包含此复制步骤，但如果是手动拷贝二进制，容易遗漏。
