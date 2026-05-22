# ============================================================================
# FP16 Tests (CBR Fusion Performance Tests)
# ============================================================================

message(STATUS "Configuring FP16 tests (CBR Fusion)")

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

# ============================================================================
# CBR FP16 Backward Test (Unified: Traditional + CUDA Graph + Mode C + Bidirectional L2)
# ============================================================================

message(STATUS "Configuring CBR FP16 Backward test (FP16)")

add_executable(cbr_bwd_fp16
    cbr_bwd_fp16.cpp
)

target_include_directories(cbr_bwd_fp16 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(cbr_bwd_fp16 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/fp16
)

if(WIN32)
    set_target_properties(cbr_bwd_fp16 PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(cbr_bwd_fp16 PRIVATE ${FP16_TEST_COMMON_COMPILE_OPTIONS})
else()
    target_compile_options(cbr_bwd_fp16 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

target_compile_definitions(cbr_bwd_fp16 PRIVATE TR_LOG_LEVEL=0)

if(DEFINED CUDA_cudart_LIBRARY)
    target_link_libraries(cbr_bwd_fp16 PRIVATE ${CUDA_cudart_LIBRARY})
    message(STATUS "cbr_bwd_fp16: Linking CUDA runtime")
endif()

if(DEFINED CUDNN_LIBRARY)
    target_link_libraries(cbr_bwd_fp16 PRIVATE ${CUDNN_LIBRARY})
    message(STATUS "cbr_bwd_fp16: Linking cuDNN")
else()
    message(WARNING "CUDNN_LIBRARY not defined - cbr_bwd_fp16 may not link correctly")
endif()

if(WIN32)
    # DLL paths for cbr_bwd_fp16
    set(DLL_PATHS_CBR_BWD_FP16 "")

    if(TR_USE_MIMALLOC AND DEFINED TR_MIMALLOC_LIBRARY_DIR)
        get_filename_component(MIMALLOC_DLL_DIR "${TR_MIMALLOC_LIBRARY_DIR}" DIRECTORY)
        get_filename_component(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}" DIRECTORY)
        set(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}/bin")
        list(APPEND DLL_PATHS_CBR_BWD_FP16 "${MIMALLOC_DLL_DIR}")
        message(STATUS "cbr_bwd_fp16: mimalloc.dll directory: ${MIMALLOC_DLL_DIR}")
    endif()

    if(DEFINED TR_CUDA_PATH)
        list(APPEND DLL_PATHS_CBR_BWD_FP16 "${TR_CUDA_PATH}/bin")
        message(STATUS "cbr_bwd_fp16: CUDA bin directory: ${TR_CUDA_PATH}/bin")
    endif()

    if(DLL_PATHS_CBR_BWD_FP16)
        string(REPLACE ";" ";" DLL_PATH_STR_BWD "${DLL_PATHS_CBR_BWD_FP16}")
        set_target_properties(cbr_bwd_fp16 PROPERTIES
            ENVIRONMENT "PATH=${DLL_PATH_STR_BWD};$ENV{PATH}"
        )
        message(STATUS "cbr_bwd_fp16: DLL paths added to environment")
    endif()
endif()

# ============================================================================
# CBR FP16 Backward Test SEQUENTIAL (DGrad waits WGrad)
# ============================================================================

message(STATUS "Configuring CBR FP16 Backward SEQUENTIAL test (FP16)")

add_executable(cbr_bwd_fp16_seq
    cbr_bwd_fp16_seq.cpp
)

target_include_directories(cbr_bwd_fp16_seq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(cbr_bwd_fp16_seq PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/fp16
)

if(WIN32)
    set_target_properties(cbr_bwd_fp16_seq PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(cbr_bwd_fp16_seq PRIVATE ${FP16_TEST_COMMON_COMPILE_OPTIONS})
else()
    target_compile_options(cbr_bwd_fp16_seq PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

target_compile_definitions(cbr_bwd_fp16_seq PRIVATE TR_LOG_LEVEL=0)

if(DEFINED CUDA_cudart_LIBRARY)
    target_link_libraries(cbr_bwd_fp16_seq PRIVATE ${CUDA_cudart_LIBRARY})
    message(STATUS "cbr_bwd_fp16_seq: Linking CUDA runtime")
endif()

if(DEFINED CUDNN_LIBRARY)
    target_link_libraries(cbr_bwd_fp16_seq PRIVATE ${CUDNN_LIBRARY})
    message(STATUS "cbr_bwd_fp16_seq: Linking cuDNN")
else()
    message(WARNING "CUDNN_LIBRARY not defined - cbr_bwd_fp16_seq may not link correctly")
endif()

if(WIN32)
    # DLL paths for cbr_bwd_fp16_seq
    set(DLL_PATHS_CBR_BWD_SEQ_FP16 "")

    if(TR_USE_MIMALLOC AND DEFINED TR_MIMALLOC_LIBRARY_DIR)
        get_filename_component(MIMALLOC_DLL_DIR "${TR_MIMALLOC_LIBRARY_DIR}" DIRECTORY)
        get_filename_component(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}" DIRECTORY)
        set(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}/bin")
        list(APPEND DLL_PATHS_CBR_BWD_SEQ_FP16 "${MIMALLOC_DLL_DIR}")
        message(STATUS "cbr_bwd_fp16_seq: mimalloc.dll directory: ${MIMALLOC_DLL_DIR}")
    endif()

    if(DEFINED TR_CUDA_PATH)
        list(APPEND DLL_PATHS_CBR_BWD_SEQ_FP16 "${TR_CUDA_PATH}/bin")
        message(STATUS "cbr_bwd_fp16_seq: CUDA bin directory: ${TR_CUDA_PATH}/bin")
    endif()

    if(DLL_PATHS_CBR_BWD_SEQ_FP16)
        string(REPLACE ";" ";" DLL_PATH_STR_BWD_SEQ "${DLL_PATHS_CBR_BWD_SEQ_FP16}")
        set_target_properties(cbr_bwd_fp16_seq PROPERTIES
            ENVIRONMENT "PATH=${DLL_PATH_STR_BWD_SEQ};$ENV{PATH}"
        )
        message(STATUS "cbr_bwd_fp16_seq: DLL paths added to environment")
    endif()
endif()

# ============================================================================
# CBR FP16 Backward Test SINGLE STREAM (All operations on one stream)
# ============================================================================

message(STATUS "Configuring CBR FP16 Backward SINGLE STREAM test (FP16)")

add_executable(cbr_bwd_fp16_single
    cbr_bwd_fp16_single.cpp
)

target_include_directories(cbr_bwd_fp16_single PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(cbr_bwd_fp16_single PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/fp16
)

if(WIN32)
    set_target_properties(cbr_bwd_fp16_single PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(cbr_bwd_fp16_single PRIVATE ${FP16_TEST_COMMON_COMPILE_OPTIONS})
else()
    target_compile_options(cbr_bwd_fp16_single PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

target_compile_definitions(cbr_bwd_fp16_single PRIVATE TR_LOG_LEVEL=0)

if(DEFINED CUDA_cudart_LIBRARY)
    target_link_libraries(cbr_bwd_fp16_single PRIVATE ${CUDA_cudart_LIBRARY})
    message(STATUS "cbr_bwd_fp16_single: Linking CUDA runtime")
endif()

if(DEFINED CUDNN_LIBRARY)
    target_link_libraries(cbr_bwd_fp16_single PRIVATE ${CUDNN_LIBRARY})
    message(STATUS "cbr_bwd_fp16_single: Linking cuDNN")
else()
    message(WARNING "CUDNN_LIBRARY not defined - cbr_bwd_fp16_single may not link correctly")
endif()

if(WIN32)
    # DLL paths for cbr_bwd_fp16_single
    set(DLL_PATHS_CBR_BWD_SINGLE_FP16 "")

    if(TR_USE_MIMALLOC AND DEFINED TR_MIMALLOC_LIBRARY_DIR)
        get_filename_component(MIMALLOC_DLL_DIR "${TR_MIMALLOC_LIBRARY_DIR}" DIRECTORY)
        get_filename_component(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}" DIRECTORY)
        set(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}/bin")
        list(APPEND DLL_PATHS_CBR_BWD_SINGLE_FP16 "${MIMALLOC_DLL_DIR}")
        message(STATUS "cbr_bwd_fp16_single: mimalloc.dll directory: ${MIMALLOC_DLL_DIR}")
    endif()

    if(DEFINED TR_CUDA_PATH)
        list(APPEND DLL_PATHS_CBR_BWD_SINGLE_FP16 "${TR_CUDA_PATH}/bin")
        message(STATUS "cbr_bwd_fp16_single: CUDA bin directory: ${TR_CUDA_PATH}/bin")
    endif()

    if(DLL_PATHS_CBR_BWD_SINGLE_FP16)
        string(REPLACE ";" ";" DLL_PATH_STR_BWD_SINGLE "${DLL_PATHS_CBR_BWD_SINGLE_FP16}")
        set_target_properties(cbr_bwd_fp16_single PROPERTIES
            ENVIRONMENT "PATH=${DLL_PATH_STR_BWD_SINGLE};$ENV{PATH}"
        )
        message(STATUS "cbr_bwd_fp16_single: DLL paths added to environment")
    endif()
endif()

message(STATUS "FP16 tests configuration complete")
