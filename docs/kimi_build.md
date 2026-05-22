# Kimi 验证通过的 TR4 Windows 构建命令

## 1. Python 配置向导

```powershell
cd R:\renaissance
"Y`n1" | python configure.py
```

## 2. CMake 配置（必须在同一个 cmd 进程内调用 vcvars）

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cd /d R:\renaissance && cmake -G Ninja -S . -B build/windows-msvc-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake -DTR_SCENE_PC_CUDA=ON -DTR_USE_CUDA=ON -DTR_USE_MUSA=OFF -DTR_USE_EIGEN=ON -DTR_USE_XNNPACK=ON -DTR_USE_STB=ON -DTR_USE_LIBJPEG=ON -DTR_USE_ZLIB=ON -DTR_USE_LIBCURL=ON -DTR_USE_LIBARCHIVE=ON -DTR_USE_MIMALLOC=ON -DTR_USE_SIMD=ON'
```

## 3. Ninja 全量编译（同样必须在 vcvars 环境中）

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cd /d R:\renaissance\build\windows-msvc-release && ninja -j30'
```

## 4. Ninja 增量编译单个目标

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cd /d R:\renaissance\build\windows-msvc-release && ninja -j30 test_relu_multigpu'
```

> 只编译/链接指定目标，不碰其他未变更的目标。示例中删除 `test_relu_multigpu.exe` 后重新执行，结果：`[1/1] Linking`，即只重链接、不重复编译 `.obj`。

## 5. 运行 test_relu_multigpu（需先注入 CUDA/cuDNN 到 PATH）

```powershell
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + $env:PATH
R:\renaissance\build\windows-msvc-release\bin\tests\model\test_relu_multigpu.exe
```

> 验证结果：Forward PASS / Backward PASS / Overall PASS
