cmake_minimum_required(3.24.0)

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

# ============================================================================
# 构建选项配置
# ============================================================================

# 编译器标志 - Alpha编译优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # MSVC Release优化
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG /arch:AVX2")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
    endif()

    # GCC Release优化
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
    endif()
endif()

# 调试模式
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Ob0 /Od /RTC1")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    endif()
endif()

# ============================================================================
# 依赖库查找
# ============================================================================

# CUDA支持
option(TR_USE_CUDA "Enable CUDA support" ON)
if(TR_USE_CUDA)
    # 查找CUDA
    if(DEFINED CUDAToolkit_ROOT)
        set(CMAKE_CUDA_TOOLKIT_ROOT ${CUDAToolkit_ROOT})
    endif()

    enable_language(CUDA)

    # 设置CUDA架构
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA Architecture")

    message(STATUS "CUDA support enabled for architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # 检查cuDNN
    find_library(CUDNN_LIBRARY cudnn PATHS ${CUDNN_ROOT} ${CUDA_TOOLKIT_ROOT} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDNN_ROOT} ${CUDA_TOOLKIT_ROOT}/include NO_DEFAULT_PATH)

    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
        add_definitions(-DTR_USE_CUDNN)

        # 设置cuDNN变量
        set(CUDNN_LIBRARY ${CUDNN_LIBRARY})
        set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR})
    else()
        message(WARNING "cuDNN not found")
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

# ============================================================================
# 单元测试配置
# ============================================================================

# 检查unit_tests目录是否存在
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests)
    # 检查unit_tests/CMakeLists.txt是否存在
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/CMakeLists.txt)
        # 包含单元测试目录
        add_subdirectory(unit_tests)
        message(STATUS "Unit tests enabled")
    else()
        message(STATUS "Unit tests disabled - unit_tests/CMakeLists.txt not found")
    endif()
else()
    message(STATUS "Unit tests disabled - unit_tests directory not found")
endif()

# ============================================================================
# 安装配置
# ============================================================================

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

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