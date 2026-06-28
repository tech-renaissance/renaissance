# Renaissance Windows 构建指南

**版本**: V1.0  
**更新日期**: 2026-05-05  
**状态**: ✅ 验证通过

---

## 📋 完整构建流程（三步法）

### **第一步：清理旧配置**

**意义**：删除旧的构建产物和配置，避免冲突

```bash
cd R:/renaissance && rm -rf build config workspace
```

---

### **第二步：运行配置向导**

**意义**：检测硬件环境，生成 CMake 路径配置和编译脚本

```bash
cd R:/renaissance && (echo Y && echo 1) | python configure.py
```

**输入说明**：
- `Y` - 启用 GPU 加速
- `1` - 选择 Python 3.14.0

**生成文件**：
- `config/cmake_paths.cmake` - CMake 路径配置
- `config/project_config.json` - 项目配置
- `build.bat` - 编译脚本

---

### **第三步：初始化 MSVC 环境并 CMake 配置**

**意义**：初始化 Visual Studio 编译器环境变量（LIB、INCLUDE、PATH），生成 `build.ninja` 构建文件

```powershell
powershell.exe -Command "& cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake -G Ninja -S . -B build/windows-msvc-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake -DTR_SCENE_PC_CUDA=ON -DTR_USE_CUDA=ON -DTR_USE_MUSA=OFF -DTR_USE_EIGEN=ON -DTR_USE_XNNPACK=ON -DTR_USE_STB=ON -DTR_USE_LIBJPEG=ON -DTR_USE_ZLIB=ON -DTR_USE_LIBCURL=ON -DTR_USE_LIBARCHIVE=ON -DTR_USE_MIMALLOC=ON -DTR_USE_SIMD=ON'"
```

**关键作用**：
- `vcvars64.bat` - 设置 MSVC 链接器环境变量（解决 `kernel32.lib` 找不到的问题）
- `cmake -G Ninja` - 生成 `build.ninja` 构建文件
- 自动生成 `setup_cuda_env.bat` - 运行时环境设置脚本

**预期输出**：
```
-- Generated: R:/renaissance/setup_cuda_env.bat
-- Configuring done (10.8s)
-- Generating done (0.1s)
```

---

### **第四步：编译全项目**

**意义**：编译所有源文件和测试程序，生成可执行文件

```powershell
powershell.exe -Command "& cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake --build build/windows-msvc-release --parallel 30'"
```

**关键作用**：
- `cmake --build` - 调用 ninja 执行编译
- `--parallel 30` - 30 个并行编译任务，加速编译

**预期输出**：
```
[153/153] Linking CXX executable bin\tests\model\mnist_mlp.exe
```

**编译时间**：约 2-3 分钟（取决于硬件）

---

### **增量编译单个测试**

**意义**：只重新编译修改过的目标，节省时间

```powershell
powershell.exe -Command "& cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance\build\windows-msvc-release && ninja test_relu_multigpu'"
```

**使用场景**：
- 修改了某个测试的源代码
- 某个测试的可执行文件被删除
- 只想快速验证某个测试的编译

**预期输出**：
```
[1/1] Linking CXX executable bin\tests\model\test_relu_multigpu.exe
```

**编译时间**：约 1-5 秒（只链接一个目标）

---

### **第五步：运行测试**

**意义**：执行测试程序，验证功能正确性

```powershell
cd R:/renaissance/build/windows-msvc-release/bin/tests/model

powershell.exe -Command "\$env:PATH = 'C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + \$env:PATH; .\test_relu_multigpu.exe"
```

**关键作用**：
- 设置 PATH 环境变量，让系统找到 CUDA/cuDNN DLL
- `C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1` - cuDNN DLL 位置
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin` - CUDA Runtime DLL 位置
- `.\test_relu_multigpu.exe` - 运行测试程序

**预期输出**：
```
========================================
ReLU Multi-GPU Test Results:
  Forward:  PASS
  Backward:  PASS
  Overall:  PASS
========================================
```

**运行时间**：约 5-10 秒（取决于测试规模）

---

## ✅ 验证要点

- ✅ 第一步后：目录被清空
- ✅ 第二步后：生成 `config/` 目录和 `build.bat`
- ✅ 第三步后：生成 `build/windows-msvc-release/build.ninja` 和 `setup_cuda_env.bat`

---

## 🚀 下一步

配置完成后，执行编译：

```powershell
powershell.exe -Command "& cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake --build build/windows-msvc-release --parallel 30'"
```

详见后续章节（TODO: 编译、运行测试）



## 运行并重定向到文件的示例

如果你要运行C++程序并把结果重定向到某个txt文件：

```shell
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + $env:PATH; cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\integration\test_nin.exe --gpu > R:\renaissance\test_nin_gpu_mp2.txt 2>&1'
```

如果你要运行python脚本并把结果重定向到某个txt文件：

```shell
powershell.exe -Command "$env:PATH = 'C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + $env:PATH; Start-Process -FilePath 'B:\Softwares\miniconda3\envs\py313\python.exe' -ArgumentList 'r:\renaissance\tests\bn\mnist_best_adamw.py --gpu' -NoNewWindow -Wait -RedirectStandardOutput 'r:\renaissance\mnist_best_adamw_gpu_run1.txt' -RedirectStandardError 'r:\renaissance\mnist_best_adamw_gpu_run1_err.txt'; Write-Output 'GPU_RUN1_DONE'"
```
