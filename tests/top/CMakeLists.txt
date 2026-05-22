# ============================================================================
# Top-Level Integration Tests
# ============================================================================

message(STATUS "Configuring top-level integration tests")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# 添加CUDA头文件路径（如果启用CUDA）
if(TR_USE_CUDA AND DEFINED TR_CUDA_PATH)
    include_directories(${TR_CUDA_PATH}/include)
endif()

# ============================================================================
# 创建演示可执行文件（仅包含实际存在的文件）
# ============================================================================

# Staging调试测试
add_executable(staging_debug staging_debug.cpp)

# ArchPlan ResNet-50 End-to-End Test
add_executable(test_arch_plan_resnet50 test_arch_plan_resnet50.cpp)

# ArchPlan Multi-Model Test
add_executable(test_arch_plan_models test_arch_plan_models.cpp)

# ArchPlan FP32 + Unfused Test
add_executable(test_arch_plan_fp32_unfused test_arch_plan_fp32_unfused.cpp)

# ResNet-50 YAML Checksum Consistency Test
add_executable(test_resnet50_checksum test_resnet50_checksum.cpp)

# ============================================================================
# 设置输出目录
# ============================================================================

set_target_properties(staging_debug PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/top
)

set_target_properties(test_arch_plan_resnet50 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/top
)

set_target_properties(test_arch_plan_models PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${C_BINARY_DIR}/bin/tests/top
)

set_target_properties(test_arch_plan_fp32_unfused PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/top
)

set_target_properties(test_resnet50_checksum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/top
)

# ============================================================================
# Windows下为控制台程序设置子系统
# ============================================================================

if(WIN32)
    set_target_properties(staging_debug PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_arch_plan_resnet50 PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_arch_plan_models PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_arch_plan_fp32_unfused PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_resnet50_checksum PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

# ============================================================================
# 编译选项
# ============================================================================

if(MSVC)
    target_compile_options(staging_debug PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_arch_plan_resnet50 PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_arch_plan_models PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_arch_plan_fp32_unfused PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_resnet50_checksum PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(staging_debug PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_arch_plan_resnet50 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_arch_plan_models PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_arch_plan_fp32_unfused PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_resnet50_checksum PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

# ============================================================================
# 设置日志级别
# ============================================================================

target_compile_definitions(staging_debug PRIVATE TR_LOG_LEVEL=0)  # DEBUG级别
target_compile_definitions(test_arch_plan_resnet50 PRIVATE TR_LOG_LEVEL=1)  # INFO级别
target_compile_definitions(test_arch_plan_models PRIVATE TR_LOG_LEVEL=1)  # INFO级别
target_compile_definitions(test_arch_plan_fp32_unfused PRIVATE TR_LOG_LEVEL=1)  # INFO级别
target_compile_definitions(test_resnet50_checksum PRIVATE TR_LOG_LEVEL=1)  # INFO级别

# ============================================================================
# 链接renaissance统一库
# ============================================================================

target_link_libraries(staging_debug PRIVATE renaissance)
target_link_libraries(test_arch_plan_resnet50 PRIVATE renaissance)
target_link_libraries(test_arch_plan_models PRIVATE renaissance)
target_link_libraries(test_arch_plan_fp32_unfused PRIVATE renaissance)
target_link_libraries(test_resnet50_checksum PRIVATE renaissance)

# ============================================================================
# GPU运行时环境设置
# ============================================================================

setup_gpu_runtime_env(staging_debug)
setup_gpu_runtime_env(test_arch_plan_resnet50)
setup_gpu_runtime_env(test_arch_plan_models)
setup_gpu_runtime_env(test_arch_plan_fp32_unfused)
setup_gpu_runtime_env(test_resnet50_checksum)

message(STATUS "Top-level integration tests configured successfully")
message(STATUS "  - staging_debug: StagingBufferPool + NUMA binding debug test")
message(STATUS "  - test_arch_plan_resnet50: ArchPlan ResNet-50 end-to-end + YAML round-trip")
message(STATUS "  - test_arch_plan_models: ArchPlan multi-model test")
message(STATUS "  - test_arch_plan_fp32_unfused: FP32 unfused minimum granularity test")
message(STATUS "  - test_resnet50_checksum: ResNet-50 YAML checksum consistency test")
