# ============================================================================
# Renaissance Framework Unit Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "tests must be included from the parent project")
endif()

# ============================================================================
# 确保workspace目录存在
# ============================================================================

if(DEFINED TR_WORKSPACE)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create workspace directory: ${TR_WORKSPACE}")
    endif()
else()
    # 如果未定义TR_WORKSPACE，使用默认路径
    set(TR_WORKSPACE "${CMAKE_SOURCE_DIR}/workspace")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create default workspace directory: ${TR_WORKSPACE}")
    endif()
endif()

# 设置单元测试的输出目录
set(TESTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests)

# ============================================================================
# GPU 运行时环境设置（不使用 DLL 复制，而是配置 PATH）
# ============================================================================
#
# 设计原则：
#   业界最佳实践（PyTorch、TensorFlow、CUDA 官方示例等）从不将 cuDNN/CUDA DLL
#   复制到 build 目录。原因：
#     1. cuDNN 9.x 的 DLL 集合超过 1GB，多次复制会导致 build 目录急剧膨胀
#     2. 并行构建时多目标同时复制到同一目录会引发文件占用冲突
#     3. 复制无法处理 DLL 的间接依赖（如 cudart 还需要 nvrtc 等运行时组件）
#   正确做法：
#     - 构建期不复制任何 DLL，让 OS 通过 PATH 环境变量在运行时查找
#     - VS IDE 调试：设置 VS_DEBUGGER_ENVIRONMENT 自动注入 PATH
#     - CTest 运行：设置 ENVIRONMENT 测试属性
#     - 命令行直接运行：执行 setup_cuda_env.bat 或在系统环境变量中配置
#
#   关于 vcpkg DLL：
#     vcpkg 的 applocal.ps1 在链接时自动分析导入表并复制所需的 vcpkg DLL
#     （mimalloc、turbojpeg、Simd、zlib、libcurl、libarchive 等）到输出目录。
#     这个机制与本函数互不冲突，无需额外处理。
#

# 查找 cuDNN DLL 目录（仅执行一次，供所有 GPU 测试目标共用）
if(WIN32 AND TR_USE_CUDA AND DEFINED TR_CUDNN_PATH)
    set(_cudnn_dll_dir "")
    file(GLOB _cudnn_ver_dirs "${TR_CUDNN_PATH}/bin/*")
    foreach(_ver_dir ${_cudnn_ver_dirs})
        if(IS_DIRECTORY ${_ver_dir})
            file(GLOB _cudnn_dlls "${_ver_dir}/cudnn*.dll")
            if(_cudnn_dlls)
                set(_cudnn_dll_dir ${_ver_dir})
                break()
            endif()
        endif()
    endforeach()
    if(NOT _cudnn_dll_dir)
        file(GLOB _cudnn_dlls "${TR_CUDNN_PATH}/bin/cudnn*.dll")
        if(_cudnn_dlls)
            set(_cudnn_dll_dir "${TR_CUDNN_PATH}/bin")
        endif()
    endif()
    if(_cudnn_dll_dir)
        set(TR_CUDNN_DLL_DIR "${_cudnn_dll_dir}" CACHE INTERNAL "cuDNN DLL directory")
        message(STATUS "cuDNN DLL directory: ${TR_CUDNN_DLL_DIR}")
    endif()
endif()

# 为 GPU 测试目标设置运行时 PATH 环境
#   - VS IDE: 通过 VS_DEBUGGER_ENVIRONMENT 自动注入
#   - 命令行: 运行前先执行 setup_cuda_env.bat
#   - CTest:  通过 ENVIRONMENT 测试属性（需在 add_test 时调用）
function(setup_gpu_runtime_env target_name)
    if(NOT WIN32)
        return()
    endif()

    set(_extra_path "")

    # cuDNN DLL 目录
    if(DEFINED TR_CUDNN_DLL_DIR)
        if(_extra_path)
            set(_extra_path "${_extra_path}\;${TR_CUDNN_DLL_DIR}")
        else()
            set(_extra_path "${TR_CUDNN_DLL_DIR}")
        endif()
    endif()

    # CUDA Toolkit bin 目录（cudart、cublas、nvrtc 等运行时 DLL）
    if(TR_USE_CUDA AND DEFINED TR_CUDA_PATH)
        if(_extra_path)
            set(_extra_path "${_extra_path}\;${TR_CUDA_PATH}/bin")
        else()
            set(_extra_path "${TR_CUDA_PATH}/bin")
        endif()
    endif()

    if(_extra_path)
        set_target_properties(${target_name} PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=${_extra_path}\;%PATH%"
        )
    endif()
endfunction()

# ============================================================================
# 包含core子目录（Logger、TRException等核心组件测试）
# ============================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/core)
    add_subdirectory(core)
    message(STATUS "[Tests] Core tests: included")
else()
    if(TR_CMAKE_VERBOSE)
        message(STATUS "Optional test suite 'core' not present")
    endif()
endif()

# ============================================================================
# 包含prep子目录（Preprocessor相关测试）
# ============================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/prep AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/prep)
    add_subdirectory(prep)
    message(STATUS "[Tests] Prep tests: included")
else()
    if(TR_CMAKE_VERBOSE)
        message(STATUS "Optional test suite 'prep' not present")
    endif()
endif()

# ============================================================================
# 包含graph子目录（BluePrint、Compiler、Flow等测试）
# ============================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/graph AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/graph)
    add_subdirectory(graph)
    message(STATUS "[Tests] Graph tests: included")
else()
    if(TR_CMAKE_VERBOSE)
        message(STATUS "Optional test suite 'graph' not present")
    endif()
endif()

# ============================================================================
# 包含top子目录（ResNet-50等顶级集成测试）
# ============================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/top AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/top)
    add_subdirectory(top)
    message(STATUS "[Tests] Top integration tests: included")
endif()

# ============================================================================
# mlp 子目录（MLP原型测试，仅CUDA平台）
# ============================================================================

# 注意：mlp_test 是原型验证性质的独立实验，不链接 renaissance 库
# 已通过功能测试，暂时排除在常规编译之外
# 如需手动编译，请使用：
#   powershell.exe -Command "& cmd /c '\"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl && cd build && ninja mlp_test'"
#
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mlp AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mlp)
#     if(TR_USE_CUDA)
#         add_subdirectory(mlp)
#         message(STATUS "Including mlp test")
#     else()
#         message(STATUS "Skipping mlp test (TR_USE_CUDA is OFF)")
#     endif()
# endif()

# ============================================================================
# fp16 子目录（FP16算子测试，仅CUDA平台）
# ============================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fp16 AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fp16)
    if(TR_USE_CUDA)
        add_subdirectory(fp16)
        message(STATUS "[Tests] FP16 tests: included")
    else()
        message(STATUS "Skipping fp16 tests (TR_USE_CUDA is OFF)")
    endif()
endif()

# ============================================================================
# 未来添加其他测试板块时的示例：
# ============================================================================
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)
#     add_subdirectory(data)
#     message(STATUS "Including data tests")
# endif()
#
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/device AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/device)
#     add_subdirectory(device)
#     message(STATUS "Including device tests")
# endif()

# ============================================================================
# 单元测试（V4.20.3 新增，已移除）
# ============================================================================

# ============================================================================
# correction 子目录（算子数学正确性验证）
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/correction AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/correction)
    add_subdirectory(correction)
    message(STATUS "[Tests] Correction tests: included")
endif()

# ============================================================================
# perf 子目录（算子性能测试）
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/perf AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/perf)
    add_subdirectory(perf)
    message(STATUS "[Tests] Perf tests: included")
endif()

# ============================================================================
# integration 子目录（端到端集成测试）
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/integration AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/integration)
    add_subdirectory(integration)
    message(STATUS "[Tests] Integration tests: included")
endif()

# ============================================================================
# bn 子目录（BatchNorm 算子数学正确性验证测试）
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bn AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bn)
    add_subdirectory(bn)
    message(STATUS "[Tests] BN tests: included")
endif()

# ============================================================================
# op 子目录（算子数学正确性验证测试）
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/op)
    add_subdirectory(op)
    message(STATUS "[Tests] Op tests: included")
endif()

# ============================================================================
# ref 子目录（MNIST MLP 参考实现）
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ref AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ref)
    if(TR_USE_CUDA)
        add_executable(mnist_mlp_3 ref/mnist_mlp_3.cpp)
        target_compile_definitions(mnist_mlp_3 PRIVATE TR_LOG_LEVEL=1)
        target_link_libraries(mnist_mlp_3 PRIVATE renaissance)
        setup_gpu_runtime_env(mnist_mlp_3)
        set_target_properties(mnist_mlp_3 PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/ref
        )
        if(WIN32)
            set_target_properties(mnist_mlp_3 PROPERTIES WIN32_EXECUTABLE FALSE)
            target_compile_options(mnist_mlp_3 PRIVATE /W4 /utf-8
                /wd4244 /wd4505 /wd4458 /wd4127)
        else()
            target_compile_options(mnist_mlp_3 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
        endif()
        message(STATUS "  - mnist_mlp_3: MNIST 3-Layer MLP (784-512-256-10)")
    endif()
endif()
