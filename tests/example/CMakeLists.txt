set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# MLP MNIST Example — 完整训练流程示例（对比 PyTorch 版本）
# ============================================================================
add_executable(mlp_mnist mlp_mnist.cpp)
target_link_libraries(mlp_mnist PRIVATE renaissance)
target_compile_definitions(mlp_mnist PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_mnist)
endif()

set_target_properties(mlp_mnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/example
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_mnist PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_mnist PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_mnist: MLP MNIST training example")
