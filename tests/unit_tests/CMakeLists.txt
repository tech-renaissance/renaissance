# ============================================================================
# Renaissance Framework Unit Tests
# ============================================================================

message(STATUS "Configuring unit tests")

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
# Core 模块单元测试
# ============================================================================

# Initializer 单元测试
add_executable(test_initializer
    core/test_initializer.cpp
)

target_link_libraries(test_initializer
    renaissance
)

# Bias初始化修复验证测试
add_executable(test_bias_init
    core/test_bias_init.cpp
)

target_link_libraries(test_bias_init
    renaissance
)

# 综合初始化修复验证测试
add_executable(test_comprehensive_init
    core/test_comprehensive_init.cpp
)

target_link_libraries(test_comprehensive_init
    renaissance
)

# ============================================================================
# Algo 模块单元测试
# ============================================================================

# 学习率调度器单元测试
add_executable(test_scheduler
    algo/test_scheduler.cpp
)

target_link_libraries(test_scheduler
    renaissance
)

# PolynomialLR数学精确性验证测试
add_executable(test_polynomial_lr_math_verify
    algo/test_polynomial_lr_math_verify.cpp
)

target_link_libraries(test_polynomial_lr_math_verify
    renaissance
)

# renAIssance PolynomialLR输出测试（用于与PyTorch对比）
add_executable(renaissance_polynomial_lr
    algo/renaissance_polynomial_lr.cpp
)

target_link_libraries(renaissance_polynomial_lr
    renaissance
)

# 设置输出目录
set_target_properties(test_initializer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/core
    FOLDER "tests/unit_tests"
)

set_target_properties(test_scheduler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/algo
    FOLDER "tests/unit_tests"
)

set_target_properties(test_polynomial_lr_math_verify PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/algo
    FOLDER "tests/unit_tests"
)

set_target_properties(renaissance_polynomial_lr PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/algo
    FOLDER "tests/unit_tests"
)

message(STATUS "Unit tests configured successfully")
