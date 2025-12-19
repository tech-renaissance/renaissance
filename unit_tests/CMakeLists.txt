# ============================================================================
# renAIssance Framework Unit Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "unit_tests must be included from the parent project")
endif()

# 设置单元测试的输出目录
set(UNIT_TESTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unit_tests)

# ============================================================================
# CUDA测试样例
# ============================================================================

# 检查CUDA是否可用
if(TR_USE_CUDA AND CUDNN_LIBRARY)
    message(STATUS "Building CUDA unit tests with cuDNN support")

    # hello_cuda.cpp 测试可执行文件
    add_executable(hello_cuda unit_tests/dependence/hello_cuda.cpp)

    # 设置CUDA/cuDNN包含目录
    target_include_directories(hello_cuda PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${CUDNN_INCLUDE_DIR}
    )

    # 链接CUDA库和框架库
    target_link_libraries(hello_cuda PRIVATE
        ${CUDA_CUDART_LIBRARY}
        ${CUDNN_LIBRARY}
        renaissance_base
    )

    # 设置输出目录
    set_target_properties(hello_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${UNIT_TESTS_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TESTS_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TESTS_OUTPUT_DIR}
    )

    # 添加编译定义
    target_compile_definitions(hello_cuda PRIVATE
        TR_USE_CUDA
        TR_USE_CUDNN
    )

    # 为MSVC编译器添加特定设置
    if(MSVC)
        target_compile_options(hello_cuda PRIVATE /W3)
    else()
        target_compile_options(hello_cuda PRIVATE -Wall -Wextra)
    endif()

    message(STATUS "CUDA test 'hello_cuda' will be built")

else()
    message(STATUS "CUDA/cuDNN not available - skipping CUDA unit tests")
endif()

# ============================================================================
# 单元测试目标
# ============================================================================

# 创建一个总的测试目标（如果需要的话）
if(TR_USE_CUDA AND CUDNN_LIBRARY)
    # 添加运行测试的自定义目标
    add_custom_target(run_cuda_tests
        COMMAND ${UNIT_TESTS_OUTPUT_DIR}/hello_cuda
        DEPENDS hello_cuda
        WORKING_DIRECTORY ${UNIT_TESTS_OUTPUT_DIR}
        COMMENT "Running CUDA unit tests"
    )

    message(STATUS "Added 'run_cuda_tests' target")
endif()

# ============================================================================
# 输出单元测试信息
# ============================================================================

message(STATUS "")
message(STATUS "=== Unit Tests Configuration ===")
if(TR_USE_CUDA AND CUDNN_LIBRARY)
    message(STATUS "CUDA Tests: Enabled")
    message(STATUS "Test Output Directory: ${UNIT_TESTS_OUTPUT_DIR}")
else()
    message(STATUS "CUDA Tests: Disabled (CUDA/cuDNN not available)")
endif()
message(STATUS "")