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

---

## 6. 常见陷阱：PowerShell 管道导致 exit code 误报

当用以下方式运行测试时，即使程序本身返回 0，Shell 工具也可能报告 `exit code: 1`：

```powershell
# 错误示范：管道会触发 PowerShell NativeCommandError
& .\bin\tests\correction\test_dl_full.exe --gpu 2>&1 | Select-String -Pattern "PASS|FAIL" | Select-Object -Last 10
```

**原因：** `Select-String` 在没有匹配到任何行时会返回 `$? = false`，而 `2>&1` 把 stderr 重定向到 stdout 后，管道关闭时机问题会进一步放大此现象。此时 `test_dl_full.exe` 本身其实已经正常退出并打印了 `PASS`。

**正确做法：** 用 `Start-Process -Wait` 直接运行，避免管道：

```powershell
$p = Start-Process -FilePath ".\bin\tests\correction\test_dl_full.exe" `
    -ArgumentList "--cpu" -PassThru -Wait -NoNewWindow
Write-Host "Exit code: $($p.ExitCode)"
```

或直接用 cmd（不经过 PowerShell 管道）：

```powershell
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\correction\test_dl_full.exe --cpu'
```

> 已确认：`test_dl_full.exe --cpu` 和 `--gpu` 均返回 **Exit code: 0**，训练结果一致（97.61% top1）。
