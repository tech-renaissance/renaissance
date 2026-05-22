# ds_build — 纯 Python TR4 构建脚本

> 为无法使用 PowerShell / CMD 的自动化环境设计。全部操作由 Python 标准库完成，零外部依赖。
> **验证状态**：`--clear` ✅ `--config` ✅ `--rebuild` ✅ `--target` ✅

---

## 前置条件

- Python 3.10+
- Visual Studio 2022 Community（已安装 MSVC + Windows SDK）
- CUDA Toolkit 13.1、cuDNN 9.17、Ninja、CMake 已按 `configure.py` 要求部署

---

## 脚本位置

```
python/scripts/build_helper.py
```

---

## 用法总览

```bash
python python/scripts/build_helper.py [选项]
```

| 选项 | 作用 |
|---|---|
| `--clear` | 删除 `build/`、`config/`、`build.bat` |
| `--config` | 运行 `configure.py`（自动回答 `Y` + `1`） |
| `--rebuild` | **CMake 配置 + Ninja 全量编译** |
| `--target <name>` | **CMake 配置 + Ninja 增量编译**指定目标 |

---

## 典型工作流

### 1. 从零开始全量构建

```bash
python python/scripts/build_helper.py --clear --config --rebuild
```

等价于人工执行：
1. 清理旧构建
2. `configure.py` 生成路径配置
3. `vcvars64.bat` → `cmake -G Ninja ...` → `ninja -j30`

### 2. 增量编译单个测试

```bash
python python/scripts/build_helper.py --target test_relu_multigpu
```

仅重新构建指定目标及其依赖，未变更的目标不会触碰。

### 3. 只跑配置向导（不编译）

```bash
python python/scripts/build_helper.py --config
```

---

## 技术原理

`vcvars64.bat` 会向当前 cmd 进程注入 `INCLUDE`、`LIB`、`PATH` 等环境变量。普通 `subprocess.run(["cmake", ...])` 拿不到这些变量，导致 `cl.exe` 找不到标准库头文件（`C1083: cstdint No such file`）。

脚本内部通过**临时批处理文件**解决：

```bat
@echo off
call "C:\Program Files\...\vcvars64.bat"
set
```

用 `subprocess.run(['cmd.exe', '/c', tmp_bat])` 执行，捕获 `set` 输出的所有 `KEY=VALUE`，合并进 `os.environ` 后再传给 `cmake` 和 `ninja` 的 `env` 参数。同时检查 `INCLUDE`/`LIB` 是否存在来验证 vcvars 是否真正生效。

因此后续编译完全在正确的 MSVC 环境中进行，无需用户手动进入 Developer Command Prompt。

---

## 运行测试示例

```python
import os
import subprocess

os.environ["PATH"] = (
    r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;"
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;"
    + os.environ["PATH"]
)

subprocess.run([
    r"R:\renaissance\build\windows-msvc-release\bin\tests\model\test_relu_multigpu.exe"
])
```
