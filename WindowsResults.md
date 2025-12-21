```shell
PS R:\renaissance> python configure.py

==============================================
  renAIssance Configuration Wizard
  renAIssance Deep Learning Framework v3.1.0
==============================================

[Step 1/8] Checking CPU architecture...
[OK] Architecture: x86_64

[Step 2/8] Checking operating system...
[OK] OS: Windows

[Step 3/8] Detecting GPU hardware...
[OK] GPU: NVIDIA GeForce RTX 4060 Laptop GPU (x1), Driver: 591.44

[Step 4/8] Checking vcpkg package manager...
[OK] vcpkg: T:\Softwares\vcpkg

[Step 5/8] Checking C++ toolchain...
[INFO] Looking for MSVC...
[OK] MSVC          [v14.44.35207]  - C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64
[OK] CMake         [v4.1.0]        - T:/Softwares/CMake/bin
[OK] Ninja         [v1.12.1]       - B:/Softwares/JetBrains/CLion 2025.2/bin/ninja/win/x64

[Step 6/8] Determining usage scenario...

Do you want to use GPU acceleration? ([Y]/N):y

[Step 7/8] Checking GPU acceleration libraries...
[OK] CUDA Toolkit  [v13.0]         - C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0
[OK] cuDNN         [v9.17]         - C:/Program Files/NVIDIA/CUDNN/v9.17

[Step 8/8] Checking other libraries...
[OK] oneDNN        [v3.7]          - T:/Softwares/vcpkg/installed/x64-windows
[OK] XNNPACK       [v2024-08-20]   - T:/Softwares/vcpkg/installed/x64-windows
[OK] zlib          [v1.3.1]        - T:/Softwares/vcpkg/installed/x64-windows
[OK] libcurl       [v8.16.0]       - T:/Softwares/vcpkg/installed/x64-windows
[OK] libjpeg-turbo [v3.1.2]        - T:/Softwares/vcpkg/installed/x64-windows
[OK] mimalloc      [v2.2.3]        - T:/Softwares/vcpkg/installed/x64-windows
[OK] STB           [v2024-07-29]   - T:/Softwares/vcpkg/installed/x64-windows
[INFO] Found Python installations:

  1. Python        [v3.14.0]       - C:/Python314
  2. Python        [v3.14.0]       - T:/Softwares/msys64/mingw64/bin
  3. Other (enter custom path)

  Please select [1-3]:
[INFO] Selected: Python        [v3.14.0]       - C:/Python314
[OK] Python        [v3.14.0]       - C:/Python314
[OK] NumPy         [v2.3.4]        - C:/Python314

[Step 8/8] Generating configuration files...
[OK] Build Directory: build/cmake-build-release
[OK] Generated: config\cmake_paths.cmake
[OK] Generated: config\project_config.json
[OK] Generated: build.bat

==============================================
  Configuration Completed! Congratulations!
==============================================

Next steps:
  1. Run .\build.bat in the project root directory
  2. Find examples in the build directory: cd build/cmake-build-release
```

