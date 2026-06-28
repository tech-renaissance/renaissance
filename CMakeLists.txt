cmake_minimum_required(VERSION 3.22)

# ============================================================================
# renAIssance Deep Learning Framework CMake Configuration
# Version 3.0.4 - based on Tech Renaissance Alpha Build optimizations
# ============================================================================

project(renAIssance VERSION 3.0.4 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 平台检测
if(WIN32)
    set(TR_PLATFORM "Windows")
elseif(UNIX)
    set(TR_PLATFORM "Linux")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# ============================================================================
# 包含生成的配置文件
# ============================================================================

# 包含用户路径配置
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config/cmake_paths.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config/cmake_paths.cmake)
endif()

# 自动修复 Eigen3 include 路径（处理 Linux vcpkg 的嵌套 include/eigen3 结构）
if(DEFINED TR_EIGEN_INCLUDE_DIR)
    if(NOT EXISTS "${TR_EIGEN_INCLUDE_DIR}/Eigen/Core" AND EXISTS "${TR_EIGEN_INCLUDE_DIR}/eigen3/Eigen/Core")
        set(TR_EIGEN_INCLUDE_DIR "${TR_EIGEN_INCLUDE_DIR}/eigen3")
        message(STATUS "Auto-corrected Eigen3 include path: ${TR_EIGEN_INCLUDE_DIR}")
    endif()
endif()

# 确保workspace目录存在（在项目根目录）
if(DEFINED TR_WORKSPACE)
    # 使用execute_process创建目录，绕过CMake的源码目录限制
    if(TR_WORKSPACE MATCHES "\\$\\{TR_PROJECT_ROOT\\}")
        # 需要展开变量，先设置临时路径
        set(TR_WORKSPACE_TEMP "${CMAKE_CURRENT_SOURCE_DIR}/workspace")
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE_TEMP}
                        RESULT_VARIABLE TR_WORKSPACE_RESULT)
        if(NOT TR_WORKSPACE_RESULT EQUAL 0)
            message(WARNING "Failed to create workspace directory")
        endif()
        # 更新TR_WORKSPACE为实际路径
        set(TR_WORKSPACE ${TR_WORKSPACE_TEMP})
    else()
        # 直接使用已定义的路径
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                        RESULT_VARIABLE TR_WORKSPACE_RESULT)
        if(NOT TR_WORKSPACE_RESULT EQUAL 0)
            message(WARNING "Failed to create workspace directory")
        endif()
    endif()
    message(STATUS "Workspace directory ensured: ${TR_WORKSPACE}")
elseif(DEFINED TR_PROJECT_ROOT)
    # 如果只有TR_PROJECT_ROOT，则在源码目录中创建workspace
    set(TR_WORKSPACE "${CMAKE_CURRENT_SOURCE_DIR}/workspace")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE TR_WORKSPACE_RESULT)
    if(NOT TR_WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create workspace directory")
    endif()
    message(STATUS "Workspace directory created: ${TR_WORKSPACE}")
else()
    # 回退到默认workspace目录（在构建目录中）
    set(TR_WORKSPACE "${CMAKE_BINARY_DIR}/workspace")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE TR_WORKSPACE_RESULT)
    if(NOT TR_WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create workspace directory")
    endif()
    message(STATUS "Default workspace directory created: ${TR_WORKSPACE}")
endif()

# ============================================================================
# 运行时库搜索路径 (RPATH) 配置
# ============================================================================

# 让可执行文件在构建目录中运行时，能自动找到非标准路径下的动态库（如 /opt/xnnpack/lib）
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(DEFINED TR_XNNPACK_LIBRARY_DIR)
    list(APPEND CMAKE_BUILD_RPATH ${TR_XNNPACK_LIBRARY_DIR})
    message(STATUS "CMAKE_BUILD_RPATH appended: ${TR_XNNPACK_LIBRARY_DIR}")
endif()

# ============================================================================
# 项目全局宏定义
# ============================================================================

# 定义项目根目录和工作空间目录宏，使所有源文件可见（包括src和tests目录）
if(DEFINED TR_PROJECT_ROOT)
    add_compile_definitions(TR_PROJECT_ROOT="${TR_PROJECT_ROOT}")
    message(STATUS "Global macro TR_PROJECT_ROOT defined: ${TR_PROJECT_ROOT}")
endif()

if(DEFINED TR_WORKSPACE)
    add_compile_definitions(TR_WORKSPACE="${TR_WORKSPACE}")
    message(STATUS "Global macro TR_WORKSPACE defined: ${TR_WORKSPACE}")
endif()

# 定义CPU架构宏，使所有源文件可见
if(DEFINED TR_CPU_ARCH_X86_64)
    add_compile_definitions(TR_CPU_ARCH_X86_64=1)
    message(STATUS "Global macro TR_CPU_ARCH_X86_64 defined")
endif()

if(DEFINED TR_CPU_ARCH_ARM64)
    add_compile_definitions(TR_CPU_ARCH_ARM64=1)
    message(STATUS "Global macro TR_CPU_ARCH_ARM64 defined")
endif()

if(DEFINED TR_CPU_ARCH_RISCV64)
    add_compile_definitions(TR_CPU_ARCH_RISCV64=1)
    message(STATUS "Global macro TR_CPU_ARCH_RISCV64 defined")
endif()

# 定义场景宏，使所有源文件可见（只当值为ON时才定义）
if(TR_SCENE_PC_CUDA)
    add_compile_definitions(TR_SCENE_PC_CUDA=1)
    message(STATUS "Global macro TR_SCENE_PC_CUDA defined")
endif()

if(TR_SCENE_PC_MUSA)
    add_compile_definitions(TR_SCENE_PC_MUSA=1)
    message(STATUS "Global macro TR_SCENE_PC_MUSA defined")
endif()

if(TR_SCENE_CPU_CLOUD)
    add_compile_definitions(TR_SCENE_CPU_CLOUD=1)
    message(STATUS "Global macro TR_SCENE_CPU_CLOUD defined")
endif()

if(TR_SCENE_GPU_CLOUD)
    add_compile_definitions(TR_SCENE_GPU_CLOUD=1)
    message(STATUS "Global macro TR_SCENE_GPU_CLOUD defined")
    message(STATUS "DEBUG: AUTO_CPU_BINDING='${AUTO_CPU_BINDING}'")
    if(AUTO_CPU_BINDING STREQUAL "ON")
        add_compile_definitions(AUTO_CPU_BINDING=1)
        message(STATUS "Global macro AUTO_CPU_BINDING defined")
    else()
        message(STATUS "DEBUG: AUTO_CPU_BINDING is NOT ON, skipping")
    endif()
endif()

if(TR_SCENE_EDGE_ARM)
    add_compile_definitions(TR_SCENE_EDGE_ARM=1)
    message(STATUS "Global macro TR_SCENE_EDGE_ARM defined")
endif()

if(TR_SCENE_EDGE_RISCV)
    add_compile_definitions(TR_SCENE_EDGE_RISCV=1)
    message(STATUS "Global macro TR_SCENE_EDGE_RISCV defined")
endif()

if(TR_IS_EDGE_DEVICE)
    add_compile_definitions(TR_IS_EDGE_DEVICE=${TR_IS_EDGE_DEVICE})
    message(STATUS "Global macro TR_IS_EDGE_DEVICE defined: ${TR_IS_EDGE_DEVICE}")
endif()

# 定义特性宏，使所有源文件可见（只当值为ON时才定义）
if(TR_USE_CUDA)
    add_compile_definitions(TR_USE_CUDA=1)
    message(STATUS "Global macro TR_USE_CUDA defined: 1")
endif()

if(TR_USE_MUSA)
    add_compile_definitions(TR_USE_MUSA=1)
    message(STATUS "Global macro TR_USE_MUSA defined: 1")
endif()

if(TR_USE_ONEDNN)
    add_compile_definitions(TR_USE_ONEDNN=1)
    message(STATUS "Global macro TR_USE_ONEDNN defined: 1")
endif()

if(TR_USE_XNNPACK)
    add_compile_definitions(TR_USE_XNNPACK=1)
    message(STATUS "Global macro TR_USE_XNNPACK defined: 1")
endif()

if(TR_USE_STB)
    add_compile_definitions(TR_USE_STB=1)
    message(STATUS "Global macro TR_USE_STB defined: 1")
endif()

if(TR_USE_LIBJPEG)
    add_compile_definitions(TR_USE_LIBJPEG=1)
    message(STATUS "Global macro TR_USE_LIBJPEG defined: 1")
endif()

if(TR_USE_ZLIB)
    add_compile_definitions(TR_USE_ZLIB=1)
    message(STATUS "Global macro TR_USE_ZLIB defined: 1")
endif()

if(TR_USE_LIBCURL)
    add_compile_definitions(TR_USE_LIBCURL=1)
    message(STATUS "Global macro TR_USE_LIBCURL defined: 1")
endif()

if(TR_USE_LIBARCHIVE)
    add_compile_definitions(TR_USE_LIBARCHIVE=1)
    message(STATUS "Global macro TR_USE_LIBARCHIVE defined: 1")
endif()

if(TR_USE_LIBNUMA)
    add_compile_definitions(TR_USE_LIBNUMA=1)
    message(STATUS "Global macro TR_USE_LIBNUMA defined: 1")
endif()

if(TR_USE_MIMALLOC)
    add_compile_definitions(TR_USE_MIMALLOC=1)
    message(STATUS "Global macro TR_USE_MIMALLOC defined: 1")
endif()

if(TR_USE_SIMD)
    add_compile_definitions(TR_USE_SIMD=1)
    message(STATUS "Global macro TR_USE_SIMD defined: 1")
endif()

if(TR_USE_NCCL)
    add_compile_definitions(TR_USE_NCCL=1)
    message(STATUS "Global macro TR_USE_NCCL defined: 1")
endif()

option(TR_NCCL_SINGLE_GPU "Enable NCCL initialization for single-GPU tasks (default OFF)" OFF)
if(TR_NCCL_SINGLE_GPU)
    add_compile_definitions(TR_NCCL_SINGLE_GPU=1)
    message(STATUS "Global macro TR_NCCL_SINGLE_GPU defined: 1")
endif()

# ============================================================================
# GPU特定型号宏定义
# ============================================================================
# 定义GPU型号特定的宏，使所有源文件可见
if(TR_GPU_MODEL_A100)
    add_compile_definitions(USING_A100=1)
    message(STATUS "Global macro USING_A100 defined: 1")
endif()

if(TR_GPU_MODEL_RTX5090)
    add_compile_definitions(USING_RTX5090=1)
    message(STATUS "Global macro USING_RTX5090 defined: 1")
endif()

# ============================================================================
# PythonSession支持（仅在特定场景下启用）
# ============================================================================
# 同时设置CMake变量和编译定义
set(TR_USE_PYTHON_SESSION OFF)
if(TR_SCENE_CPU_CLOUD OR TR_SCENE_GPU_CLOUD OR TR_SCENE_PC_CUDA)
    set(TR_USE_PYTHON_SESSION ON)
    add_compile_definitions(TR_USE_PYTHON_SESSION=1)
    message(STATUS "Global macro TR_USE_PYTHON_SESSION defined: 1")

    # 定义Python解释器路径
    if(DEFINED Python3_EXECUTABLE)
        add_compile_definitions(TR_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
        message(STATUS "Global macro TR_PYTHON_EXECUTABLE defined: ${Python3_EXECUTABLE}")
    else()
        message(WARNING "Python3_EXECUTABLE not defined, PythonSession may not work correctly")
    endif()
endif()

if(DEFINED TR_NUM_GPUS)
    add_compile_definitions(TR_NUM_GPUS=${TR_NUM_GPUS})
    message(STATUS "Global macro TR_NUM_GPUS defined: ${TR_NUM_GPUS}")
endif()

# ============================================================================
# 构建选项配置
# ============================================================================

# 编译器标志 - Alpha编译优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # MSVC Release优化
    if(MSVC)
        # 添加UTF-8编译标志以支持中文注释（避免C4819警告）
        # 使用生成器表达式，只应用于C++，不应用于CUDA
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG /arch:AVX2")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
    endif()

    # GCC Release优化
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # RISC-V场景：添加向量扩展支持
        if(TR_SCENE_EDGE_RISCV)
            # RISC-V向量扩展：启用RVV 1.0（march=rv64gcv）
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=rv64gcv")
            message(STATUS "EDGE_RISCV scene: using -O3 with -march=rv64gcv (vector extension)")
        else()
            # 其他场景：可以使用-march=native
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
        endif()
        # 注意：不使用-flto（链接时优化），因为会导致静态库链接问题
        # LTO要求所有编译单元在链接时可见，对于包含多个静态库的项目会增加复杂度
        # 如果需要启用LTO，需要在每个静态库target中使用-ffat-lto-objects
    endif()
endif()

# 调试模式
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        # 添加UTF-8编译标志以支持中文注释（避免C4819警告）
        # 使用生成器表达式，只应用于C++，不应用于CUDA
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        # 注意：不使用/RTC1，因为它与OpenMP和mimalloc不兼容
        # MSVC文档明确说明：/RTC（运行时错误检查）不能与/openmp同时使用
        # 添加/arch:AVX2以与Release模式保持一致
        # 清空默认的debug标志，重新设置（避免/RTC1）
        set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Ob0 /Od /arch:AVX2")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # RISC-V场景：添加向量扩展支持
        if(TR_SCENE_EDGE_RISCV)
            # RISC-V向量扩展：启用RVV 1.0（march=rv64gcv）
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -march=rv64gcv")
            message(STATUS "EDGE_RISCV Debug: using -g -O0 with -march=rv64gcv (vector extension)")
        else()
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
        endif()
    endif()
endif()

# ============================================================================
# 依赖库查找
# ============================================================================

# CUDA支持 - V3.2.0简化版本（跳过编译器测试）
option(TR_USE_CUDA "Enable CUDA support" OFF)  # 默认关闭，由配置脚本控制
if(TR_USE_CUDA)
    # 如果已经定义了CUDAToolkit_ROOT，直接启用CUDA语言，跳过find_package
    if(DEFINED CUDAToolkit_ROOT)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # 设置CUDA为静默模式（减少nvcc的冗余输出）
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --disable-warnings" CACHE STRING "CUDA compile flags" FORCE)
        message(STATUS "CUDA language enabled for .cu files")

        # 查找CUDA Toolkit
        set(CMAKE_CUDA_TOOLKIT_ROOT ${CUDAToolkit_ROOT})
        set(CUDA_INCLUDE_DIRS ${CUDAToolkit_ROOT}/include)
        set(CUDA_LIBRARIES ${CUDAToolkit_ROOT}/lib/x64)

        if(WIN32)
            set(CUDA_cudart_LIBRARY ${CUDAToolkit_ROOT}/lib/x64/cudart.lib)
            set(CUDA_cublas_LIBRARY ${CUDAToolkit_ROOT}/lib/x64/cublas.lib)
        else()
            set(CUDA_cudart_LIBRARY ${CUDAToolkit_ROOT}/lib64/libcudart.so)
            set(CUDA_cublas_LIBRARY ${CUDAToolkit_ROOT}/lib64/libcublas.so)
        endif()

        # 设置CUDA架构（支持RTX 20/30/40/50系列、A100/H100等）
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;100" CACHE STRING "CUDA Architecture" FORCE)

        message(STATUS "CUDA support enabled (library-only mode)")
        message(STATUS "CUDA Toolkit: ${CUDAToolkit_ROOT}")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "TR_USE_CUDA=ON but CUDAToolkit_ROOT not defined")
    endif()

    # 检查cuDNN - 支持Windows和Linux的不同路径
    if(DEFINED CUDNN_ROOT)
        if(WIN32)
            # Windows下的cuDNN库查找 - 支持9.x版本的子目录结构
            # 首先尝试标准路径
            find_library(CUDNN_LIBRARY cudnn
                PATHS ${CUDNN_ROOT}
                PATH_SUFFIXES lib lib64 lib/x64
                NO_DEFAULT_PATH
            )
            # 如果没找到，搜索版本子目录 (如 lib/12.9/x64)
            if(NOT CUDNN_LIBRARY)
                file(GLOB CUDNN_VERSION_SUBDIRS "${CUDNN_ROOT}/lib/*")
                foreach(CUDNN_VER_DIR ${CUDNN_VERSION_SUBDIRS})
                    if(IS_DIRECTORY ${CUDNN_VER_DIR})
                        file(GLOB CUDNN_ARCH_DIRS "${CUDNN_VER_DIR}/*")
                        foreach(CUDNN_ARCH_DIR ${CUDNN_ARCH_DIRS})
                            if(IS_DIRECTORY ${CUDNN_ARCH_DIR})
                                find_library(CUDNN_LIBRARY cudnn
                                    PATHS ${CUDNN_ARCH_DIR}
                                    NO_DEFAULT_PATH
                                )
                                if(CUDNN_LIBRARY)
                                    message(STATUS "Found cuDNN in subdirectory: ${CUDNN_LIBRARY}")
                                    break()
                                endif()
                            endif()
                        endforeach()
                        if(CUDNN_LIBRARY)
                            break()
                        endif()
                    endif()
                endforeach()
            endif()
            # 查找头文件 - cuDNN 9.x也在include下有版本子目录
            find_path(CUDNN_INCLUDE_DIR cudnn.h
                PATHS ${CUDNN_ROOT}
                PATH_SUFFIXES include
                NO_DEFAULT_PATH
            )
            # 如果没找到，搜索版本子目录
            if(NOT CUDNN_INCLUDE_DIR)
                file(GLOB CUDNN_INCLUDE_VERSION_DIRS "${CUDNN_ROOT}/include/*")
                foreach(CUDNN_VER_DIR ${CUDNN_INCLUDE_VERSION_DIRS})
                    if(IS_DIRECTORY ${CUDNN_VER_DIR})
                        find_path(CUDNN_INCLUDE_DIR cudnn.h
                            PATHS ${CUDNN_VER_DIR}
                            NO_DEFAULT_PATH
                        )
                        if(CUDNN_INCLUDE_DIR)
                            message(STATUS "Found cuDNN headers in subdirectory: ${CUDNN_INCLUDE_DIR}")
                            break()
                        endif()
                    endif()
                endforeach()
            endif()
        else()
            # Linux下的cuDNN库查找
            find_library(CUDNN_LIBRARY cudnn
                PATHS ${CUDNN_ROOT}
                PATH_SUFFIXES lib lib64
                NO_DEFAULT_PATH
            )

            # 特殊处理：系统级安装（/usr）不搜索include路径，避免污染系统头文件搜索顺序
            if(NOT CUDNN_ROOT STREQUAL "/usr" AND
               NOT CUDNN_ROOT STREQUAL "/usr/local" AND
               NOT CUDNN_ROOT MATCHES "^/usr/local/cuda")
                # 非系统安装才搜索include路径
                find_path(CUDNN_INCLUDE_DIR cudnn.h
                    PATHS ${CUDNN_ROOT}
                    PATH_SUFFIXES include
                    NO_DEFAULT_PATH
                )
            else()
                # 系统安装：不设置include路径，让编译器使用默认搜索路径
                message(STATUS "cuDNN installed in system directory, using compiler default include paths")
                set(CUDNN_INCLUDE_DIR "")
            endif()
        endif()
    endif()

    # 对于系统安装，只检查库文件是否存在
    if(CUDNN_ROOT STREQUAL "/usr" OR
       CUDNN_ROOT STREQUAL "/usr/local" OR
       CUDNN_ROOT MATCHES "^/usr/local/cuda")
        # 系统安装：只检查库文件
        if(CUDNN_LIBRARY)
            message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
            message(STATUS "cuDNN include: using compiler default search paths")
            add_definitions(-DTR_USE_CUDNN)
        else()
            message(WARNING "cuDNN library not found")
        endif()
    else()
        # 非系统安装：检查库和头文件
        if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
            message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
            message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
            add_definitions(-DTR_USE_CUDNN)
        else()
            message(WARNING "cuDNN not found - CUDA tests may not work")
            if(DEFINED CUDNN_ROOT)
                message(WARNING "  CUDNN_ROOT was set to: ${CUDNN_ROOT}")
            else()
                message(WARNING "  CUDNN_ROOT was not defined")
            endif()
        endif()
    endif()
else()
    message(STATUS "CUDA support disabled")
endif()

# MUSA支持 - V3.6.5 MUSA kernel编译支持
if(TR_USE_MUSA)
    # 查找MUSA包
    if(DEFINED TR_MUSA_PATH)
        set(MUSA_ROOT ${TR_MUSA_PATH})
    elseif(DEFINED ENV{MUSA_ROOT})
        set(MUSA_ROOT $ENV{MUSA_ROOT})
    elseif(EXISTS /usr/local/musa)
        set(MUSA_ROOT /usr/local/musa)
    endif()

    if(MUSA_ROOT)
        # 添加MUSA CMake模块路径
        list(APPEND CMAKE_MODULE_PATH ${MUSA_ROOT}/cmake)
        find_package(MUSA REQUIRED)

        message(STATUS "MUSA package found: ${MUSA_ROOT}")
        message(STATUS "MUSA CMake modules added to CMAKE_MODULE_PATH")
    else()
        message(WARNING "TR_USE_MUSA=ON but MUSA_ROOT not found")
    endif()
endif()

# OpenMP支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_definitions(-DTR_USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP enabled")
else()
    message(WARNING "OpenMP not found")
endif()

# vcpkg集成
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# ============================================================================
# 框架核心组件配置
# ============================================================================

# 设置全局输出目录（必须在 add_subdirectory 之前，否则对子目录无效）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含src目录（统一的renaissance库在src/CMakeLists.txt中创建）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt)
        add_subdirectory(src)
        message(STATUS "Framework renaissance library enabled")
    else()
        message(STATUS "Framework disabled - src/CMakeLists.txt not found")
    endif()
else()
    message(STATUS "Framework disabled - src directory not found")
endif()

# ============================================================================
# 单元测试配置
# ============================================================================

# 检查tests目录是否存在
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    # 包含tests目录，由tests/CMakeLists.txt统一管理所有测试
    add_subdirectory(tests)
    message(STATUS "Unit tests directory included")
else()
    message(STATUS "Unit tests disabled - tests directory not found")
endif()

# ============================================================================
# 安装配置
# ============================================================================

# 为所有目标设置输出目录（辅助函数）
function(set_output_directories target)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endfunction()

# 设置CMAKE生成的文件位置
set(CMAKE_CACHEFILE_DIR ${CMAKE_BINARY_DIR})
set(CMAKE_CPACK_OUTPUT_CONFIG_FILE ${CMAKE_BINARY_DIR}/CPackConfig.cmake)

# ============================================================================
# 输出配置信息
# ============================================================================

message(STATUS "")
message(STATUS "=== renAIssance Framework Build Configuration ===")
message(STATUS "Platform: ${TR_PLATFORM}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")

if(TR_USE_CUDA)
    message(STATUS "CUDA: Enabled (${CMAKE_CUDA_ARCHITECTURES})")
    if(CUDNN_LIBRARY)
        message(STATUS "cuDNN: Found (${CUDNN_LIBRARY})")
    endif()
else()
    message(STATUS "CUDA: Disabled")
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Enabled")
else()
    message(STATUS "OpenMP: Disabled")
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "vcpkg: Enabled")
endif()
message(STATUS "")

# 查找 cuDNN DLL 目录（在生成 setup_cuda_env.bat 之前完成）
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
    endif()
endif()

# ============================================================================
# 生成 CUDA 运行时环境设置脚本（不复制 DLL，通过 PATH 查找）
# ============================================================================
# 业界最佳实践：不将 cuDNN/CUDA DLL 复制到 build 目录。
# 而是在运行时通过 PATH 环境变量查找 DLL。
# VS IDE 用户无需此脚本（VS_DEBUGGER_ENVIRONMENT 已自动配置）。
if(WIN32 AND TR_USE_CUDA)
    set(_env_bat "${CMAKE_SOURCE_DIR}/setup_cuda_env.bat")
    set(_bat_content "@echo off\r\n")
    set(_bat_content "${_bat_content}rem ============================================================\r\n")
    set(_bat_content "${_bat_content}rem TR4 CUDA/cuDNN Runtime Environment Setup\r\n")
    set(_bat_content "${_bat_content}rem ============================================================\r\n")

    if(DEFINED TR_CUDA_PATH)
        set(_bat_content "${_bat_content}set \"PATH=${TR_CUDA_PATH}\\bin;%PATH%\"\r\n")
    endif()

    if(DEFINED TR_CUDNN_DLL_DIR)
        set(_bat_content "${_bat_content}set \"PATH=${TR_CUDNN_DLL_DIR};%PATH%\"\r\n")
    endif()

    set(_bat_content "${_bat_content}\r\n")
    set(_bat_content "${_bat_content}%*\r\n")

    file(WRITE ${_env_bat} "${_bat_content}")
    message(STATUS "Generated: ${_env_bat}")
    message(STATUS "  Usage: setup_cuda_env.bat build\\windows-msvc-release\\bin\\tests\\top\\resnet50_c.exe")
endif()
