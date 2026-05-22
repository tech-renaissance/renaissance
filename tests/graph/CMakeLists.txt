# ============================================================================
# Graph Layer Tests
# ============================================================================

message(STATUS "Configuring graph layer tests")

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
    # 添加cuDNN Frontend头文件路径
    if(DEFINED TR_CUDNN_FRONTEND_INCLUDE_DIR)
        include_directories(${TR_CUDNN_FRONTEND_INCLUDE_DIR})
        message(STATUS "Adding cuDNN Frontend include path: ${TR_CUDNN_FRONTEND_INCLUDE_DIR}")
    endif()
endif()

# ============================================================================
# 创建图层测试可执行文件
# ============================================================================

# AXPY测试（GPU和CPU版本）
if(TR_USE_CUDA)
    add_executable(test_axpy_gpu axpy.cpp)
    target_compile_definitions(test_axpy_gpu PRIVATE TR_LOG_LEVEL=1)
    target_link_libraries(test_axpy_gpu PRIVATE renaissance)
    setup_gpu_runtime_env(test_axpy_gpu)
    set_target_properties(test_axpy_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
    )
    message(STATUS "  - test_axpy_gpu: AXPY GPU test")
endif()

add_executable(test_axpy_cpu axpy_cpu.cpp)
target_compile_definitions(test_axpy_cpu PRIVATE TR_LOG_LEVEL=1)
target_link_libraries(test_axpy_cpu PRIVATE renaissance)
set_target_properties(test_axpy_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
)
message(STATUS "  - test_axpy_cpu: AXPY CPU test")

# Tensor-DTensor回环测试（GPU和CPU版本）
if(TR_USE_CUDA)
    add_executable(test_tensor_dtensor_loopback_gpu tensor_dtensor_loopback.cpp)
    target_compile_definitions(test_tensor_dtensor_loopback_gpu PRIVATE TR_LOG_LEVEL=1)
    target_link_libraries(test_tensor_dtensor_loopback_gpu PRIVATE renaissance)
    setup_gpu_runtime_env(test_tensor_dtensor_loopback_gpu)
    set_target_properties(test_tensor_dtensor_loopback_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
    )
    message(STATUS "  - test_tensor_dtensor_loopback_gpu: Tensor-DTensor loopback GPU test")
endif()

add_executable(test_tensor_dtensor_loopback_cpu tensor_dtensor_loopback.cpp)
target_compile_definitions(test_tensor_dtensor_loopback_cpu PRIVATE TR_LOG_LEVEL=1)
target_link_libraries(test_tensor_dtensor_loopback_cpu PRIVATE renaissance)
set_target_properties(test_tensor_dtensor_loopback_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
)
message(STATUS "  - test_tensor_dtensor_loopback_cpu: Tensor-DTensor loopback CPU test")

# MLP测试（GPU和CPU版本）
if(TR_USE_CUDA)
    add_executable(test_mlp_amp mlp_amp.cpp)
    target_compile_definitions(test_mlp_amp PRIVATE TR_LOG_LEVEL=1)
    target_link_libraries(test_mlp_amp PRIVATE renaissance)
    setup_gpu_runtime_env(test_mlp_amp)
    set_target_properties(test_mlp_amp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
    )
    message(STATUS "  - test_mlp_amp: MLP AMP GPU test")
endif()

add_executable(test_mlp mlp.cpp)
target_compile_definitions(test_mlp PRIVATE TR_LOG_LEVEL=1)
target_link_libraries(test_mlp PRIVATE renaissance)
set_target_properties(test_mlp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
)
message(STATUS "  - test_mlp: MLP CPU test")

# 双图测试（GPU和CPU版本）
if(TR_USE_CUDA)
    add_executable(test_dual_graph_gpu dual_graph.cpp)
    target_compile_definitions(test_dual_graph_gpu PRIVATE TR_LOG_LEVEL=1)
    target_link_libraries(test_dual_graph_gpu PRIVATE renaissance)
    setup_gpu_runtime_env(test_dual_graph_gpu)
    set_target_properties(test_dual_graph_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
    )
    message(STATUS "  - test_dual_graph_gpu: Dual graph GPU test")
endif()

add_executable(test_dual_graph_cpu dual_graph_cpu.cpp)
target_compile_definitions(test_dual_graph_cpu PRIVATE TR_LOG_LEVEL=1)
target_link_libraries(test_dual_graph_cpu PRIVATE renaissance)
set_target_properties(test_dual_graph_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/graph
)
message(STATUS "  - test_dual_graph_cpu: Dual graph CPU test")

# ============================================================================
# Windows下为控制台程序设置子系统
# ============================================================================

if(WIN32)
    if(TR_USE_CUDA)
        set_target_properties(test_axpy_gpu PROPERTIES WIN32_EXECUTABLE FALSE)
        set_target_properties(test_tensor_dtensor_loopback_gpu PROPERTIES WIN32_EXECUTABLE FALSE)
        set_target_properties(test_mlp_amp PROPERTIES WIN32_EXECUTABLE FALSE)
        set_target_properties(test_dual_graph_gpu PROPERTIES WIN32_EXECUTABLE FALSE)
    endif()
    set_target_properties(test_axpy_cpu PROPERTIES WIN32_EXECUTABLE FALSE)
    set_target_properties(test_tensor_dtensor_loopback_cpu PROPERTIES WIN32_EXECUTABLE FALSE)
    set_target_properties(test_mlp PROPERTIES WIN32_EXECUTABLE FALSE)
    set_target_properties(test_dual_graph_cpu PROPERTIES WIN32_EXECUTABLE FALSE)
endif()

# ============================================================================
# 编译选项
# ============================================================================

if(MSVC)
    if(TR_USE_CUDA)
        target_compile_options(test_axpy_gpu PRIVATE /W4 /utf-8
            /wd4244 /wd4505 /wd4458 /wd4127)
        target_compile_options(test_tensor_dtensor_loopback_gpu PRIVATE /W4 /utf-8
            /wd4244 /wd4505 /wd4458 /wd4127)
        target_compile_options(test_mlp_amp PRIVATE /W4 /utf-8
            /wd4244 /wd4505 /wd4458 /wd4127)
        target_compile_options(test_dual_graph_gpu PRIVATE /W4 /utf-8
            /wd4244 /wd4505 /wd4458 /wd4127)
    endif()
    target_compile_options(test_axpy_cpu PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_tensor_dtensor_loopback_cpu PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_mlp PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
    target_compile_options(test_dual_graph_cpu PRIVATE /W4 /utf-8
        /wd4244 /wd4505 /wd4458 /wd4127)
else()
    if(TR_USE_CUDA)
        target_compile_options(test_axpy_gpu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
        target_compile_options(test_tensor_dtensor_loopback_gpu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
        target_compile_options(test_mlp_amp PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
        target_compile_options(test_dual_graph_gpu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
    target_compile_options(test_axpy_cpu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_tensor_dtensor_loopback_cpu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_mlp PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    target_compile_options(test_dual_graph_cpu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "Graph layer tests configured successfully")