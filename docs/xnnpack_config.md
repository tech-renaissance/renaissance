# XNNPACK 跨平台配置完整指南

**项目**: renAIssance 深度学习框架 V3.3.5
**日期**: 2025-12-23
**状态**: ✅ 全平台验证通过
**维护者**: renAIssance 开发团队

---

## 📋 目录

1. [概述](#概述)
2. [Linux平台配置](#linux平台配置)
3. [Windows平台配置](#windows平台配置)
4. [踩过的坑与解决方案](#踩过的坑与解决方案)
5. [快速参考](#快速参考)
6. [FAQ](#faq)

---

## 概述

XNNPACK 是 Google 开发的高性能神经网络推理库，在集成到 renAIssance 框架时，不同平台遇到了完全不同的问题：

| 平台 | 主要问题 | 解决方案 | 最终配置 |
|------|---------|----------|----------|
| **Linux** | 静态库缺少 microkernel 符号 | 使用动态库 | `x64-linux-dynamic` |
| **Windows** | vcpkg 安装不完整 + CRT 冲突 | 手动安装 + WHOLE_ARCHIVE | `C:\XNNPACK` + `/MD` |

---

## Linux平台配置

### 最终方案：使用动态库

**Triplet**: `x64-linux-dynamic`
**库文件**: `libXNNPACK.so`
**链接方式**: 普通动态链接

### 安装步骤

```bash
# 1. 卸载静态版（如果已安装）
cd /path/to/vcpkg
./vcpkg remove xnnpack:x64-linux

# 2. 安装动态版
./vcpkg install xnnpack:x64-linux-dynamic

# 3. 验证安装
ls /path/to/vcpkg/installed/x64-linux-dynamic/lib/libXNNPACK.so
```

### 配置脚本修改

**文件**: `python/scripts/smart_config.py`

在 `search_in_vcpkg()` 函数中添加动态库优先逻辑：

```python
# Linux x64平台：默认使用 x64-linux
triplet = "x64-linux"
# 只有 XNNPACK 需要动态库（由于 microkernel 链接问题）
if not is_win and name == "xnnpack":  # Linux + XNNPACK
    dynamic_path = os.path.join(VCPKG_ROOT, "installed", "x64-linux-dynamic")
    if os.path.exists(dynamic_path):
        triplet = "x64-linux-dynamic"
```

### CMakeLists.txt 配置

```cmake
elseif(TEST_NAME MATCHES "xnnpack")
    if(XNNPACK_FOUND)
        target_include_directories(${TEST_NAME} PRIVATE ${XNNPACK_INCLUDE_DIRS})
        target_link_directories(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARY_DIRS})

        # Linux/macOS使用动态库
        target_link_libraries(${TEST_NAME} PRIVATE XNNPACK)

        # 设置运行时 RPATH 以便找到动态库
        if(UNIX AND NOT APPLE)
            set_target_properties(${TEST_NAME} PROPERTIES
                INSTALL_RPATH "${XNNPACK_LIBRARY_DIRS}"
                BUILD_RPATH "${XNNPACK_LIBRARY_DIRS}"
            )
        endif()
        message(STATUS "  - Linked XNNPACK (dynamic, C++17)")
    endif()
```

### 问题1：402个未定义引用错误

**错误现象**:
```
undefined reference to `xnn_f32_gemm_minmax_ukernel_4x2c4__sse'
undefined reference to `xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast'
... (数百个类似错误)
```

**根本原因**:
vcpkg 构建的静态版 XNNPACK **关闭了 ASM microkernels**，但调度代码仍然引用这些内核。

**验证方法**:
```bash
nm -C /path/to/vcpkg/installed/x64-linux/lib/libXNNPACK.a | grep xnn_f32_gemm_minmax_ukernel_4x2c4__sse
# 输出: U xnn_f32_gemm_minmax_ukernel_4x2c4__sse (U = Undefined)
```

**解决方案**: 切换到动态库（见上述安装步骤）

**验证动态库**:
```bash
nm -D /path/to/vcpkg/installed/x64-linux-dynamic/lib/libXNNPACK.so | grep xnn_f32_gemm_minmax_ukernel_4x2c4__sse
# 输出: T xnn_f32_gemm_minmax_ukernel_4x2c4__sse (T = Text/已定义)
```

### 问题2：旧 API 已废弃

**错误现象**:
```
error: 'xnn_create_add_nd_f32' was not declared in this scope
```

**根本原因**: XNNPACK v2024-08-20 中旧 Operator API 已废弃

**解决方案**: 使用 Subgraph API

```cpp
// 步骤1: 创建子图
xnn_subgraph_t subgraph = nullptr;
xnn_create_subgraph(4, 0, &subgraph);

// 步骤2: 定义张量值
uint32_t input_id = 0;
xnn_define_tensor_value(
    subgraph,
    xnn_datatype_fp32,
    2, input_dims,
    nullptr,
    UINT32_MAX,  // external_id (XNN_INVALID_VALUE_ID)
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    &input_id
);

// 步骤3: 定义算子
xnn_define_fully_connected(
    subgraph,
    -std::numeric_limits<float>::infinity(),
    +std::numeric_limits<float>::infinity(),
    input_id,
    kernel_id,
    bias_id,
    output_id,
    0
);

// 步骤4: 创建运行时
xnn_runtime_t runtime = nullptr;
xnn_create_runtime_v3(subgraph, nullptr, nullptr, 0, &runtime);

// 步骤5: 设置并执行
xnn_external_value externals[] = {
    {input_id, input.data()},
    {output_id, output.data()}
};
xnn_setup_runtime(runtime, 2, externals);
xnn_invoke_runtime(runtime);
```

### 问题3：API 参数错误

**错误现象**: 段错误或输出全是 0

**根本原因**:
1. 最后参数传的是值而不是指针
2. `external_id` 使用 `0` 而非 `UINT32_MAX`

**解决方案**:
```cpp
// ❌ 错误
xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 2, dims, nullptr, 0, flags, 0);

// ✅ 正确
uint32_t input_id = 0;
xnn_define_tensor_value(
    subgraph,
    xnn_datatype_fp32,
    2, dims,
    nullptr,
    UINT32_MAX,  // XNN_INVALID_VALUE_ID - 让系统自动分配
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    &input_id    // 接收分配的 ID（指针）
);
```

### 测试结果

**环境**: Ubuntu 24.04 + 2×RTX 5090
**性能**: 98.31 GFLOPS
**状态**: ✅ PASS

---

## Windows平台配置

### 最终方案：手动安装 + WHOLE_ARCHIVE

**安装路径**: `C:\XNNPACK`
**库类型**: 静态库 (.lib)
**链接方式**: `/WHOLEARCHIVE` + `/MD` CRT

### 安装步骤

```powershell
# 1. 下载或从源码编译 XNNPACK
# 2. 安装到 C:\XNNPACK 目录
#    C:\XNNPACK\
#    ├── include\
#    │   └── xnnpack.h
#    └── lib\
#        ├── XNNPACK.lib
#        ├── pthreadpool.lib
#        ├── cpuinfo.lib
#        └── xnnpack-microkernels-prod.lib
```

### 配置脚本修改

#### 1. 添加搜索路径

**文件**: `python/scripts/dependency_data.py`

```python
"xnnpack": {
    "name": "XNNPACK",
    "exe": [],
    "header": "xnnpack.h",
    "lib_files": ["xnnpack.dll", "libxnnpack.so"],
    "env": ["XNNPACK_ROOT"],
    "paths_win": ["C:\\XNNPACK"],  # ← 添加 Windows 手动安装路径
    "vcpkg_packages": ["xnnpack"],
    "version_pattern": r"(\d{4}-\d{2}-\d{2})",
    "install_hint": "vcpkg install xnnpack"
},
```

#### 2. 跳过 vcpkg 搜索

**文件**: `python/scripts/smart_config.py`

**位置1**: `find_all_dependency_versions` 函数（第894-898行）
```python
# vcpkg版本
# 特殊处理：Windows下XNNPACK跳过vcpkg，优先使用本地安装
if VCPKG_ROOT and config.get("vcpkg_packages"):
    if not (is_win and name == "xnnpack"):  # ← 添加条件
        vcpkg_found = search_in_vcpkg(name, config, is_win)
        if vcpkg_found["found"]:
            vcpkg_found["from_vcpkg"] = True
            versions.append(vcpkg_found)
```

**位置2**: `search_dependency` 函数（第1106-1108行）
```python
# 第一层：vcpkg搜索 (最高优先级)
# 特殊处理：Windows下XNNPACK跳过vcpkg，优先使用本地安装
if VCPKG_ROOT and config.get("vcpkg_packages"):
    if not (is_win and name == "xnnpack"):  # ← 添加条件
        vcpkg_found = search_in_vcpkg(name, config, is_win)
        if vcpkg_found["found"]:
            result = vcpkg_found
            result["from_vcpkg"] = True
            return result
```

#### 3. 生成正确的 CMake 路径

**文件**: `python/scripts/smart_config.py`（第1592-1597行）

```python
# 根据平台确定include和lib子目录
if sys_info["is_windows"]:
    # 特殊处理：Windows + XNNPACK 使用本地安装（C:\XNNPACK）
    if dep_key == "xnnpack" and not deps[dep_key].get("from_vcpkg", False):
        # 使用本地安装路径，包含include和lib子目录
        include_dir = f"{dep_path}/include"
        lib_dir = f"{dep_path}/lib"
        lines.append(f'set(TR_{cmake_name.upper()}_TRIPLET "x64-windows")')
    else:
        # Windows下其他库使用packages目录...
```

#### 4. 允许 unknown 版本

**文件**: `python/scripts/smart_config.py`（第880-891行）

```python
# 头文件/库文件检测
elif "header" in config:
    header_path = find_file_in_dirs(config["header"], [path], ["include"])
    if header_path:
        version = extract_version_from_path(path, name)

        # 特殊处理：对于XNNPACK等库，即使版本未知也接受
        allow_unknown = name.lower() in ["xnnpack", "stb", "cpuinfo", "pthreadpool"]

        if version != "unknown" or allow_unknown:
            if version == "unknown" or not min_version or compare_version(version, min_version) >= 0:
                versions.append({
                    "name": config["name"],
                    "path": path,
                    "version": version if version != "unknown" else None,
                    "from_vcpkg": False
                })
```

### CMakeLists.txt 配置

**文件**: `unit_tests/dependency/CMakeLists.txt`（第540-586行）

```cmake
elseif(TEST_NAME MATCHES "xnnpack")
    if(XNNPACK_FOUND)
        target_include_directories(${TEST_NAME} PRIVATE ${XNNPACK_INCLUDE_DIRS})
        target_link_directories(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARY_DIRS})

        # XNNPACK使用C标准库，需要降低C++标准版本以避免兼容性问题
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)

        if(MSVC)
            # Windows上使用WHOLE_ARCHIVE强制包含所有microkernel符号
            target_link_libraries(${TEST_NAME} PRIVATE
                XNNPACK
                pthreadpool
                cpuinfo
                xnnpack-microkernels-prod
            )

            # MSVC需要使用/WHOLEARCHIVE来强制链接所有符号
            target_link_options(${TEST_NAME} PRIVATE
                "/WHOLEARCHIVE:XNNPACK.lib"
                "/WHOLEARCHIVE:pthreadpool.lib"
                "/WHOLEARCHIVE:cpuinfo.lib"
                "/WHOLEARCHIVE:xnnpack-microkernels-prod.lib"
            )

            # XNNPACK需要_CRT_SECURE_NO_WARNINGS来避免某些C标准库函数的警告
            target_compile_definitions(${TEST_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

            # C:\XNNPACK的库使用动态CRT(/MD)，与项目默认设置一致
            message(STATUS "  - Linked XNNPACK (static with WHOLE_ARCHIVE, /MD CRT, C++17)")
        else()
            # Linux/macOS使用动态库
            target_link_libraries(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARIES})

            # 设置运行时 RPATH 以便找到动态库
            if(UNIX AND NOT APPLE)
                set_target_properties(${TEST_NAME} PROPERTIES
                    INSTALL_RPATH "${XNNPACK_LIBRARY_DIRS}"
                    BUILD_RPATH "${XNNPACK_LIBRARY_DIRS}"
                )
            endif()
            message(STATUS "  - Linked XNNPACK (dynamic, C++17)")
        endif()
    else()
        message(WARNING "  - XNNPACK not available")
    endif()
```

### 问题1：vcpkg 路径错误

**错误现象**:
```
fatal error C1083: 无法打开包括文件: "pthreadpool.h": No such file or directory
```

**错误路径**:
```
T:/Softwares/vcpkg/packages/xnnpack_x64-windows/include  # ❌ 错误
```

**正确路径**:
```
T:/Softwares/vcpkg/installed/x64-windows/include  # ✅ 正确
```

**根本原因**: `packages/` 是临时打包目录，不包含依赖库的头文件

**解决方案**: 使用 `C:\XNNPACK` 手动安装，跳过 vcpkg

### 问题2：402个 Microkernel 链接错误

**错误现象**:
```
error LNK2001: 无法解析的外部符号 xnn_f32_gemm_minmax_ukernel__*
```

**根本原因**:
1. XNNPACK 使用静态库（Windows 官方不支持 DLL）
2. MSVC 链接器优化掉了未直接引用的 microkernel 符号
3. 这些符号通过函数指针表调度，链接器认为"无用"

**解决方案**: 使用 `/WHOLEARCHIVE` 强制链接

```cmake
target_link_options(${TEST_NAME} PRIVATE
    "/WHOLEARCHIVE:XNNPACK.lib"
    "/WHOLEARCHIVE:pthreadpool.lib"
    "/WHOLEARCHIVE:cpuinfo.lib"
    "/WHOLEARCHIVE:xnnpack-microkernels-prod.lib"
)
```

### 问题3：CRT 不匹配（已解决！）

**历史问题**:
- 旧版 XNNPACK 使用 `/MT` 静态 CRT
- 项目使用 `/MD` 动态 CRT
- 需要强制使用 `/MT` 产生警告

**最终解决方案**: 重装 XNNPACK 使用 `/MD`！

**验证方法**:
```cmd
dumpbin /DIRECTIVES C:\XNNPACK\lib\XNNPACK.lib | findstr -i "libcmt\|msvcrt"

# 如果看到 /DEFAULTLIB:libcmt.lib → 使用 /MT（旧版）
# 如果看到 /DEFAULTLIB:msvcrt.lib → 使用 /MD（新版）✅
```

**无警告编译**:
```
[1/2] Building CXX object unit_tests\dependency\CMakeFiles\hello_xnnpack.dir\hello_xnnpack.cpp.obj
[2/2] Linking CXX executable bin\dependency\hello_xnnpack.exe
# ✅ 零警告！
```

### 问题4：找不到 stdbool.h

**错误现象**:
```
fatal error C1083: 无法打开包括文件: "stdbool.h": No such file or directory
```

**根本原因**: 编译环境未初始化（缺少 `vcvars64.bat`）

**解决方案**: 通过 `build.bat` 或 Alpha 编译方法

```batch
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --build . --target hello_xnnpack
```

### 测试结果

**环境**: Windows 11 + RTX 4060
**性能**: 85.82 GFLOPS
**状态**: ✅ PASS
**编译**: 零错误、零警告

---

## 踩过的坑与解决方案

### 坑1：vcpkg 静态库不自洽

**症状**: 静态库引用了不存在的符号
**原因**: vcpkg 构建 XNNPACK 时关闭了 ASM microkernels
**影响**: Linux 平台
**解决**: 使用动态库 `x64-linux-dynamic`

### 坑2：Windows vcpkg 安装不完整

**症状**: pthreadpool.h 找不到
**原因**: vcpkg `packages/` 目录不完整
**影响**: Windows 平台
**解决**: 手动安装到 `C:\XNNPACK`

### 坑3：MSVC 链接器优化激进

**症状**: 402个 microkernel 未定义引用
**原因**: 函数指针调用的符号被优化删除
**影响**: Windows 平台
**解决**: `/WHOLEARCHIVE` 强制链接

### 坑4：CRT 版本冲突

**症状**: LNK2038 Runtime Library 不匹配
**原因**: XNNPACK 用 `/MT`，项目用 `/MD`
**影响**: Windows 平台
**解决**: 重装 XNNPACK 使用 `/MD`

### 坑5：API 废弃

**症状**: 旧 API 函数未声明
**原因**: XNNPACK v2024+ Operator API 废弃
**影响**: 所有平台
**解决**: 使用 Subgraph API

### 坑6：API 参数错误

**症状**: 段错误或输出全是 0
**原因**: `external_id` 使用错误
**影响**: 所有平台
**解决**: 使用 `UINT32_MAX` + 指针

---

## 快速参考

### 平台配置对比

| 配置项 | Linux | Windows |
|--------|-------|---------|
| **安装方式** | vcpkg | 手动安装 |
| **Triplet** | `x64-linux-dynamic` | N/A |
| **安装路径** | `$VCPKG_ROOT/installed/x64-linux-dynamic` | `C:\XNNPACK` |
| **库文件** | `libXNNPACK.so` | `XNNPACK.lib` (静态) |
| **库类型** | 动态 | 静态 |
| **链接方式** | 普通 | `/WHOLEARCHIVE` |
| **CRT** | N/A | `/MD` |
| **C++标准** | C++17 | C++17 |
| **RPATH** | 需要 | 不需要 |

### 关键命令

#### Linux
```bash
# 检查动态库符号
nm -D libXNNPACK.so | grep xnn_f32_gemm_minmax_ukernel_4x2c4__sse

# 检查依赖
ldd build/bin/dependency/hello_xnnpack

# 运行测试
./build/bin/dependency/hello_xnnpack
```

#### Windows
```cmd
REM 检查 CRT 类型
dumpbin /DIRECTIVES C:\XNNPACK\lib\XNNPACK.lib | findstr -i "libcmt\|msvcrt"

REM 检查依赖
dumpbin /DEPENDENTS build\windows-msvc-release\bin\dependency\hello_xnnpack.exe

REM 运行测试
build\windows-msvc-release\bin\dependency\hello_xnnpack.exe
```

### 修改文件清单

| 文件 | 修改内容 | 影响平台 |
|------|---------|----------|
| `python/scripts/dependency_data.py` | 添加 `paths_win: ["C:\\XNNPACK"]` | Windows |
| `python/scripts/smart_config.py` | XNNPACK 跳过 vcpkg 搜索 | Windows |
| `python/scripts/smart_config.py` | 动态库优先逻辑 | Linux |
| `python/scripts/smart_config.py` | allow_unknown 版本支持 | Windows |
| `python/scripts/smart_config.py` | CMake 路径特殊处理 | Windows |
| `unit_tests/dependency/CMakeLists.txt` | WHOLE_ARCHIVE + /MD | Windows |
| `unit_tests/dependency/CMakeLists.txt` | 动态库 + RPATH | Linux |
| `unit_tests/dependency/hello_xnnpack.cpp` | Subgraph API | 所有平台 |

---

## FAQ

### Q1: 为什么 Linux 用动态库，Windows 用静态库？

**A**:
- **Linux**: vcpkg 静态库不自洽（缺少 microkernel 符号），动态库是官方推荐
- **Windows**: vcpkg 安装不完整，手动安装的静态库通过 `/WHOLEARCHIVE` 可用

### Q2: `/WHOLEARCHIVE` 会不会影响性能？

**A**: 不会。它只是强制链接所有符号，不改变运行时代码。性能完全相同。

### Q3: 为什么需要 C++17？

**A**: XNNPACK 是 C 库，使用 C 标准库。C++20 的新特性可能与 C 代码不兼容。

### Q4: 动态库会影响部署吗？

**A**: Linux 需要 `libXNNPACK.so`，可通过 RPATH 自动查找。Windows 无额外依赖。

### Q5: 如何升级 XNNPACK？

**A**:
- **Linux**: `vcpkg install xnnpack:x64-linux-dynamic --force-rebuild`
- **Windows**: 替换 `C:\XNNPACK` 目录内容

### Q6: Windows 为什么不直接用 vcpkg？

**A**: vcpkg 的 `packages/` 目录不完整，缺少 pthreadpool 等依赖。手动安装更可靠。

### Q7: `/MD` 和 `/MT` 有什么区别？

**A**:
- `/MD`: 动态链接 CRT（微软推荐），文件小，安全更新
- `/MT`: 静态链接 CRT，文件大，独立部署
- 现在 XNNPACK 用 `/MD`，与项目一致！

### Q8: 会不会影响其他测试程序？

**A**: 不会。所有配置都限定在 `hello_xnnpack` 目标，不影响其他程序。

---

## 总结

### 核心经验

1. **Linux**: 优先使用动态库 `x64-linux-dynamic`
2. **Windows**: 手动安装 + `/WHOLEARCHIVE` + `/MD`
3. **跨平台**: Subgraph API + `UINT32_MAX` + C++17
4. **配置**: 自动检测 + 平台差异化处理

### 最终状态

✅ **Linux GPU_CLOUD**: 98.31 GFLOPS
✅ **Windows PC_CUDA**: 85.82 GFLOPS
✅ **零错误零警告**: 所有平台完美编译
✅ **自动配置**: `python configure.py` 一键完成

### 相关文档

- `docs/xnnpack_dynamic_fix.md` - Linux 动态库切换教程
- `docs/xnnpack_issue.md` - Windows 问题记录
- `docs/xnnpack_find.md` - Windows 手动安装指南
- `docs/MD_MT_ISSUE.md` - CRT 冲突问题分析
- `docs/auto_config.md` - 自动配置系统文档

---

**文档版本**: V1.0
**最后更新**: 2025-12-23
**维护者**: renAIssance 开发团队
