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

# CUDA Device测试（条件编译）
if(TR_USE_CUDA)
    add_executable(test_cuda_device test_cuda_device.cpp)

    # CUDA RNG测试
    add_executable(test_cuda_rng test_cuda_rng.cpp)

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

message(STATUS "Device tests: CUDA and MUSA device tests configured")
