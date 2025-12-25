cmake_minimum_required(VERSION 3.22)

# ============================================================================
# renAIssance Deep Learning Framework CMake Configuration
# Version 3.0.4 - based on Tech Renaissance Alpha Build optimizations
# ============================================================================

project(renAIssance VERSION 3.0.4 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
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

if(TR_USE_MIMALLOC)
    add_compile_definitions(TR_USE_MIMALLOC=1)
    message(STATUS "Global macro TR_USE_MIMALLOC defined: 1")
endif()

if(TR_USE_SIMD)
    add_compile_definitions(TR_USE_SIMD=1)
    message(STATUS "Global macro TR_USE_SIMD defined: 1")
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
        add_compile_options(/utf-8)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG /arch:AVX2")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
    endif()

    # GCC Release优化
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # RISC-V场景：不使用-march=native
        if(TR_SCENE_EDGE_RISCV)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
            message(STATUS "EDGE_RISCV scene: using -O3 without -march=native")
        else()
            # 其他场景：可以使用-march=native
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
        endif()
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
    endif()
endif()

# 调试模式
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        # 添加UTF-8编译标志以支持中文注释（避免C4819警告）
        add_compile_options(/utf-8)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Ob0 /Od /RTC1")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    endif()
endif()

# ============================================================================
# 依赖库查找
# ============================================================================

# CUDA支持 - V3.2.0简化版本（跳过编译器测试）
option(TR_USE_CUDA "Enable CUDA support" OFF)  # 默认关闭，由配置脚本控制
if(TR_USE_CUDA)
    # 查找CUDA Toolkit
    if(DEFINED CUDAToolkit_ROOT)
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
    endif()

    # 设置CUDA架构（不启用CUDA语言，仅用于库链接）
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA Architecture")

    message(STATUS "CUDA support enabled (library-only mode)")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_ROOT}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

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
            find_path(CUDNN_INCLUDE_DIR cudnn.h
                PATHS ${CUDNN_ROOT}
                PATH_SUFFIXES include
                NO_DEFAULT_PATH
            )
        endif()
    endif()

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
else()
    message(STATUS "CUDA support disabled")
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

# 包含基础组件目录
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/base AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/base)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/base/CMakeLists.txt)
        add_subdirectory(src/base)
        message(STATUS "Framework base components enabled")
    else()
        message(STATUS "Framework base components disabled - src/base/CMakeLists.txt not found")
    endif()
else()
    message(STATUS "Framework base components disabled - src/base directory not found")
endif()

# 包含数据组件目录
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/data AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/data)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/data/CMakeLists.txt)
        add_subdirectory(src/data)
        message(STATUS "Framework data components enabled")
    else()
        message(STATUS "Framework data components disabled - src/data/CMakeLists.txt not found")
    endif()
else()
    message(STATUS "Framework data components disabled - src/data directory not found")
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

# 设置输出目录 - 确保所有输出都在build目录中
# 注意：CMAKE_BINARY_DIR在build目录中运行时就是build目录本身
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 为所有目标设置输出目录
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