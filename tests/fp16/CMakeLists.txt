# ============================================================================
# FP16 Tests (Legacy — CBR fusion targets removed)
# ============================================================================

message(STATUS "Configuring FP16 tests (no active targets)")

# Only build when TR_USE_CUDA is enabled
if(NOT TR_USE_CUDA)
    message(STATUS "FP16 tests disabled (TR_USE_CUDA is OFF)")
    return()
endif()

# Ensure workspace directory exists
if(DEFINED TR_WORKSPACE)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create workspace directory: ${TR_WORKSPACE}")
    endif()
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define header file paths
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/generated
)

# Add cuDNN include directory
if(DEFINED CUDNN_INCLUDE_DIR)
    include_directories(${CUDNN_INCLUDE_DIR})
    message(STATUS "FP16 tests: cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
endif()

# Add CUDA include directory (if defined by main CMake)
if(DEFINED CUDA_INCLUDE_DIRS)
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "FP16 tests: CUDA include directories: ${CUDA_INCLUDE_DIRS}")
endif()

# ============================================================================
# 通用MSVC编译选项 - 禁用第三方库警告
# ============================================================================
if(MSVC)
    set(FP16_TEST_COMMON_COMPILE_OPTIONS
        /W4 /utf-8
        # 禁用cuDNN Frontend库警告（第三方代码质量问题）
        /wd4458  # 变量声明隐藏类成员
        /wd4127  # 条件表达式是常量
        /wd4244  # 类型转换可能丢失数据（cuDNN Frontend和我们的double->float转换）
        # 禁用CUDA/cuDNN未引用函数警告（正常优化行为）
        /wd4505  # 已删除具有内部链接的未引用函数
    )
    message(STATUS "FP16 tests: Disabled third-party library warnings (C4458, C4127, C4244, C4505)")
endif()

# ============================================================================
# cuDNN Frontend API Include Directory
# ============================================================================

# Add cuDNN Frontend include directory if defined
if(DEFINED TR_CUDNN_FRONTEND_INCLUDE_DIR)
    include_directories(${TR_CUDNN_FRONTEND_INCLUDE_DIR})
    message(STATUS "cuDNN Frontend include directory: ${TR_CUDNN_FRONTEND_INCLUDE_DIR}")
elseif(DEFINED TR_CUDNN_FRONTEND_PATH)
    include_directories(${TR_CUDNN_FRONTEND_PATH}/include)
    message(STATUS "cuDNN Frontend include directory: ${TR_CUDNN_FRONTEND_PATH}/include")
else()
    message(WARNING "TR_CUDNN_FRONTEND_INCLUDE_DIR and TR_CUDNN_FRONTEND_PATH not defined - FP16 tests may not compile")
endif()

message(STATUS "FP16 tests configuration complete")
