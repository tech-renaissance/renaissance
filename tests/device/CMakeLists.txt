# ============================================================================
# Device Component Unit Tests
# ============================================================================

message(STATUS "Configuring device component unit tests")

# 确保workspace目录存在
if(DEFINED TR_WORKSPACE)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create workspace directory: ${TR_WORKSPACE}")
    else()
        message(STATUS "Workspace directory ensured: ${TR_WORKSPACE}")
    endif()
else()
    # 如果未定义TR_WORKSPACE，使用默认路径
    set(TR_WORKSPACE "${CMAKE_SOURCE_DIR}/workspace")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create default workspace directory: ${TR_WORKSPACE}")
    else()
        message(STATUS "Default workspace directory created: ${TR_WORKSPACE}")
    endif()
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# 创建CpuDevice测试可执行文件
add_executable(test_cpu_device
    test_cpu_device.cpp
)

# 为测试目标单独设置头文件路径
target_include_directories(test_cpu_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 设置输出目录
set_target_properties(test_cpu_device PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

# Windows下为控制台程序设置子系统
if(WIN32)
    set_target_properties(test_cpu_device PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

# 编译选项
if(MSVC)
    target_compile_options(test_cpu_device PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_device PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 设置日志级别
target_compile_definitions(test_cpu_device PRIVATE TR_LOG_LEVEL=0)

# 链接renaissance统一库（包含base、data、device及所有依赖）
target_link_libraries(test_cpu_device PRIVATE renaissance)

message(STATUS "Device component unit tests configured successfully")

# RNG测试（CPU随机数生成器）
add_executable(test_cpu_rng
    test_cpu_rng.cpp
)

target_include_directories(test_cpu_rng PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_cpu_rng PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(WIN32)
    set_target_properties(test_cpu_rng PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(test_cpu_rng PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_rng PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_cpu_rng PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_cpu_rng PRIVATE renaissance)

# CpuDevice高级RNG接口测试
add_executable(test_cpu_rand_tensor
    test_cpu_rand_tensor.cpp
)

target_include_directories(test_cpu_rand_tensor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_cpu_rand_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(WIN32)
    set_target_properties(test_cpu_rand_tensor PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(test_cpu_rand_tensor PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_rand_tensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_cpu_rand_tensor PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_cpu_rand_tensor PRIVATE renaissance)

# CPU张量复制测试
add_executable(test_cpu_copy
    test_cpu_copy.cpp
)

target_include_directories(test_cpu_copy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_cpu_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(WIN32)
    set_target_properties(test_cpu_copy PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(test_cpu_copy PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_copy PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_cpu_copy PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_cpu_copy PRIVATE renaissance)

# CUDA Device测试（条件编译）
if(TR_USE_CUDA)
    add_executable(test_cuda_device test_cuda_device.cpp)

    # CUDA RNG测试
    add_executable(test_cuda_rng test_cuda_rng.cpp)

    # CUDA张量复制测试
    add_executable(test_cuda_copy test_cuda_copy.cpp)

    # CUDA跨设备传输测试
    add_executable(test_cuda_transfer test_cuda_transfer.cpp)

    # CUDA流创建测试
    add_executable(test_cuda_stream_creation test_cuda_stream_creation.cpp)

    # CudaDevice高级RNG接口测试
    add_executable(test_cuda_rand_tensor
        test_cuda_rand_tensor.cpp
    )

    target_include_directories(test_cuda_rand_tensor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_rand_tensor PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_rand_tensor PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_rand_tensor PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_rand_tensor PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_rand_tensor PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_rand_tensor PRIVATE renaissance)
endif()

# MUSA Device测试（条件编译）
if(TR_USE_MUSA)
    add_executable(test_musa_device test_musa_device.cpp)

    # MUSA RNG测试
    add_executable(test_musa_rng test_musa_rng.cpp)

    # MUSA张量复制测试
    add_executable(test_musa_copy test_musa_copy.cpp)

    # MUSA跨设备传输测试
    add_executable(test_musa_transfer test_musa_transfer.cpp)
endif()


if(TR_USE_CUDA)
    set_target_properties(test_cuda_device PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_device PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_device PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_device PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_device PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_device PRIVATE renaissance)

    # ===== test_cuda_rng配置 =====
    set_target_properties(test_cuda_rng PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_rng PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_rng PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_rng PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_rng PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_rng PRIVATE renaissance)

    # ===== test_cuda_copy配置 =====
    target_include_directories(test_cuda_copy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_copy PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_copy PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_copy PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_copy PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_copy PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_copy PRIVATE renaissance)

    # ===== test_cuda_transfer配置 =====
    target_include_directories(test_cuda_transfer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_transfer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_transfer PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_transfer PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_transfer PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_transfer PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_transfer PRIVATE renaissance)

    # ===== test_cuda_stream_creation配置 =====
    target_include_directories(test_cuda_stream_creation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_stream_creation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_stream_creation PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_stream_creation PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_stream_creation PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_stream_creation PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_stream_creation PRIVATE renaissance)

    # ===== NCCL AllReduce测试 =====
    if(TR_USE_NCCL)
        add_executable(test_nccl_allreduce test_nccl_allreduce.cpp)

        target_include_directories(test_nccl_allreduce PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        )

        set_target_properties(test_nccl_allreduce PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
        )

        if(WIN32)
            set_target_properties(test_nccl_allreduce PROPERTIES
                WIN32_EXECUTABLE FALSE
            )
        endif()

        if(MSVC)
            target_compile_options(test_nccl_allreduce PRIVATE /W4 /utf-8)
        else()
            target_compile_options(test_nccl_allreduce PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        target_compile_definitions(test_nccl_allreduce PRIVATE TR_LOG_LEVEL=0)
        target_link_libraries(test_nccl_allreduce PRIVATE renaissance)

        # ===== NCCL Broadcast测试 =====
        add_executable(test_nccl_broadcast test_nccl_broadcast.cpp)

        target_include_directories(test_nccl_broadcast PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        )

        set_target_properties(test_nccl_broadcast PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
        )

        if(WIN32)
            set_target_properties(test_nccl_broadcast PROPERTIES
                WIN32_EXECUTABLE FALSE
            )
        endif()

        if(MSVC)
            target_compile_options(test_nccl_broadcast PRIVATE /W4 /utf-8)
        else()
            target_compile_options(test_nccl_broadcast PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        target_compile_definitions(test_nccl_broadcast PRIVATE TR_LOG_LEVEL=0)
        target_link_libraries(test_nccl_broadcast PRIVATE renaissance)

        # NCCL AllReduce性能测试
        add_executable(test_allreduce_speed test_allreduce_speed.cpp)

        target_include_directories(test_allreduce_speed PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        )

        set_target_properties(test_allreduce_speed PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
        )

        if(WIN32)
            set_target_properties(test_allreduce_speed PROPERTIES
                WIN32_EXECUTABLE FALSE
            )
        endif()

        if(MSVC)
            target_compile_options(test_allreduce_speed PRIVATE /W4 /utf-8)
        else()
            target_compile_options(test_allreduce_speed PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        target_compile_definitions(test_allreduce_speed PRIVATE TR_LOG_LEVEL=0)
        target_link_libraries(test_allreduce_speed PRIVATE renaissance)

        # NCCL Broadcast性能测试
        add_executable(test_broadcast_speed test_broadcast_speed.cpp)

        target_include_directories(test_broadcast_speed PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        )

        set_target_properties(test_broadcast_speed PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
        )

        if(WIN32)
            set_target_properties(test_broadcast_speed PROPERTIES
                WIN32_EXECUTABLE FALSE
            )
        endif()

        if(MSVC)
            target_compile_options(test_broadcast_speed PRIVATE /W4 /utf-8)
        else()
            target_compile_options(test_broadcast_speed PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        target_compile_definitions(test_broadcast_speed PRIVATE TR_LOG_LEVEL=0)
        target_link_libraries(test_broadcast_speed PRIVATE renaissance)
    endif()

    # 为测试设置运行时PATH环境变量（合并所有DLL路径）
    if(WIN32)
        set(DLL_PATHS "")

        # 添加mimalloc.dll目录
        if(TR_USE_MIMALLOC AND DEFINED TR_MIMALLOC_LIBRARY_DIR)
            get_filename_component(MIMALLOC_DLL_DIR "${TR_MIMALLOC_LIBRARY_DIR}" DIRECTORY)
            get_filename_component(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}" DIRECTORY)
            set(MIMALLOC_DLL_DIR "${MIMALLOC_DLL_DIR}/bin")
            list(APPEND DLL_PATHS "${MIMALLOC_DLL_DIR}")
            message(STATUS "CUDA device test: mimalloc.dll directory: ${MIMALLOC_DLL_DIR}")
        endif()

        # 添加CUDA DLL目录
        if(DEFINED TR_CUDA_BIN_DIR)
            list(APPEND DLL_PATHS "${TR_CUDA_BIN_DIR}/x64")
            message(STATUS "CUDA device test: CUDA Toolkit bin: ${TR_CUDA_BIN_DIR}/x64")
        endif()

        # 添加cuDNN DLL目录
        if(DEFINED TR_CUDNN_BIN_DIR)
            list(APPEND DLL_PATHS "${TR_CUDNN_BIN_DIR}")
            message(STATUS "CUDA device test: cuDNN bin: ${TR_CUDNN_BIN_DIR}")
        endif()

        # 设置完整的PATH环境变量（对test_cuda_device）
        if(DLL_PATHS)
            # 将列表转换为分号分隔的路径
            string(REPLACE ";" ";" DLL_PATH_STR "${DLL_PATHS}")
            set_target_properties(test_cuda_device PROPERTIES
                ENVIRONMENT "PATH=${DLL_PATH_STR};$ENV{PATH}"
            )
            # 为test_cuda_rng也设置相同的PATH
            set_target_properties(test_cuda_rng PROPERTIES
                ENVIRONMENT "PATH=${DLL_PATH_STR};$ENV{PATH}"
            )
            # 为test_cuda_copy也设置相同的PATH
            set_target_properties(test_cuda_copy PROPERTIES
                ENVIRONMENT "PATH=${DLL_PATH_STR};$ENV{PATH}"
            )
            # 为test_cuda_transfer也设置相同的PATH
            set_target_properties(test_cuda_transfer PROPERTIES
                ENVIRONMENT "PATH=${DLL_PATH_STR};$ENV{PATH}"
            )
            # 为test_cuda_stream_creation也设置相同的PATH
            set_target_properties(test_cuda_stream_creation PROPERTIES
                ENVIRONMENT "PATH=${DLL_PATH_STR};$ENV{PATH}"
            )
            message(STATUS "CUDA device tests: All DLL paths added to environment")
        endif()
    endif()
endif()

if(TR_USE_MUSA)
    set_target_properties(test_musa_device PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_device PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_device PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_device PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_device PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_device PRIVATE renaissance)

    # ===== test_musa_rng配置 =====
    set_target_properties(test_musa_rng PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_rng PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_rng PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_rng PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_rng PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_rng PRIVATE renaissance)

    # ===== test_musa_copy配置 =====
    target_include_directories(test_musa_copy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_musa_copy PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_copy PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_copy PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_copy PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_copy PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_copy PRIVATE renaissance)

    # ===== test_musa_transfer配置 =====
    target_include_directories(test_musa_transfer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_musa_transfer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_transfer PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_transfer PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_transfer PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_transfer PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_transfer PRIVATE renaissance)

    # ===== test_musa_rand_tensor配置 =====
    add_executable(test_musa_rand_tensor
        test_musa_rand_tensor.cpp
    )

    target_include_directories(test_musa_rand_tensor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_musa_rand_tensor PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_rand_tensor PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_rand_tensor PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_rand_tensor PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_rand_tensor PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_rand_tensor PRIVATE renaissance)
endif()

# ============================================================================
# TSR V3 格式测试
# ============================================================================
add_executable(test_tsr_format
    test_tsr_format.cpp
)

target_include_directories(test_tsr_format PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_tsr_format PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(WIN32)
    set_target_properties(test_tsr_format PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(test_tsr_format PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_tsr_format PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_tsr_format PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_tsr_format PRIVATE renaissance)

message(STATUS "Device tests: TSR V3 format test configured")

# TSR Python互操作性测试（生成测试文件）
add_executable(test_tsr_python
    test_tsr_python.cpp
)

target_include_directories(test_tsr_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_tsr_python PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(WIN32)
    set_target_properties(test_tsr_python PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

if(MSVC)
    target_compile_options(test_tsr_python PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_tsr_python PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_tsr_python PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_tsr_python PRIVATE renaissance)

message(STATUS "Device tests: TSR Python interoperability test configured")

# ==============================================================================
# CUDA异步传输测试（V3.6.18新增）
# ==============================================================================
if(TR_USE_CUDA)
    add_executable(test_cuda_async
        test_cuda_async.cpp
    )

    target_include_directories(test_cuda_async PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_async PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_async PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_async PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_async PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_async PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_async PRIVATE renaissance)

    message(STATUS "Device tests: CUDA async transfer test configured (V3.6.18)")

    # 同步vs异步性能对比测试（V3.6.18）
    add_executable(test_sync_vs_async_perf
        test_sync_vs_async_perf.cpp
    )

    target_include_directories(test_sync_vs_async_perf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_sync_vs_async_perf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_sync_vs_async_perf PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_sync_vs_async_perf PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_sync_vs_async_perf PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_sync_vs_async_perf PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_sync_vs_async_perf PRIVATE renaissance)

    message(STATUS "Device tests: Sync vs async performance test configured (V3.6.18)")
endif()

# ============================================================================
# 数据类型转换测试（V3.6.18新增）
# ============================================================================

# CPU cast测试
add_executable(test_cpu_cast
    test_cpu_cast.cpp
)

target_include_directories(test_cpu_cast PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_cpu_cast PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(MSVC)
    target_compile_options(test_cpu_cast PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_cast PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_cpu_cast PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_cpu_cast PRIVATE renaissance)

message(STATUS "Device tests: CPU cast test configured (V3.6.18)")

# CUDA cast测试
if(TR_USE_CUDA)
    add_executable(test_cuda_cast
        test_cuda_cast.cpp
    )

    target_include_directories(test_cuda_cast PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_cast PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_cast PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_cast PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_cast PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_cast PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_cast PRIVATE renaissance)

    message(STATUS "Device tests: CUDA cast test configured (V3.6.18)")
endif()

# MUSA cast测试
if(TR_USE_MUSA)
    add_executable(test_musa_cast
        test_musa_cast.cpp
    )

    target_include_directories(test_musa_cast PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_musa_cast PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(MSVC)
        target_compile_options(test_musa_cast PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_cast PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_cast PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_cast PRIVATE renaissance)

    message(STATUS "Device tests: MUSA cast test configured (V3.6.18)")

    # ===== MUSA异步传输测试 =====
    add_executable(test_musa_async
        test_musa_async.cpp
    )

    target_include_directories(test_musa_async PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_musa_async PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_async PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_async PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_async PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_async PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_async PRIVATE renaissance)

    message(STATUS "Device tests: MUSA async transfer test configured (V3.6.18)")

    # 同步vs异步性能对比测试（MUSA版本）
    add_executable(test_sync_vs_async_perf_musa
        test_sync_vs_async_perf_musa.cpp
    )

    target_include_directories(test_sync_vs_async_perf_musa PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_sync_vs_async_perf_musa PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_sync_vs_async_perf_musa PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_sync_vs_async_perf_musa PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_sync_vs_async_perf_musa PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_sync_vs_async_perf_musa PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_sync_vs_async_perf_musa PRIVATE renaissance)

    message(STATUS "Device tests: MUSA sync vs async performance test configured (V3.6.18)")
endif()

message(STATUS "Device tests: CUDA and MUSA device tests configured")

# ============================================================================
# 数据类型转换性能测试（V3.6.18新增）
# ============================================================================

# CPU cast性能测试
add_executable(test_cpu_cast_perf
    test_cpu_cast_perf.cpp
)

target_include_directories(test_cpu_cast_perf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(test_cpu_cast_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

if(MSVC)
    target_compile_options(test_cpu_cast_perf PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_cast_perf PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(test_cpu_cast_perf PRIVATE TR_LOG_LEVEL=0)
target_link_libraries(test_cpu_cast_perf PRIVATE renaissance)

message(STATUS "Device tests: CPU cast performance test configured (V3.6.18)")

# CUDA cast性能测试
if(TR_USE_CUDA)
    add_executable(test_cuda_cast_perf
        test_cuda_cast_perf.cpp
    )

    target_include_directories(test_cuda_cast_perf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_cuda_cast_perf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_cuda_cast_perf PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_cuda_cast_perf PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_cast_perf PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_cuda_cast_perf PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_cuda_cast_perf PRIVATE renaissance)

    message(STATUS "Device tests: CUDA cast performance test configured (V3.6.18)")
endif()

# MUSA cast性能测试
if(TR_USE_MUSA)
    add_executable(test_musa_cast_perf
        test_musa_cast_perf.cpp
    )

    target_include_directories(test_musa_cast_perf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    set_target_properties(test_musa_cast_perf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
    )

    if(WIN32)
        set_target_properties(test_musa_cast_perf PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()

    if(MSVC)
        target_compile_options(test_musa_cast_perf PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_musa_cast_perf PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    target_compile_definitions(test_musa_cast_perf PRIVATE TR_LOG_LEVEL=0)
    target_link_libraries(test_musa_cast_perf PRIVATE renaissance)

    message(STATUS "Device tests: MUSA cast performance test configured (V3.6.18)")
endif()

message(STATUS "Device tests: All cast performance tests configured (V3.6.18)")

