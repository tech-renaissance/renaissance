set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# BN FWD+BWD+INF 串接测试 — BN1D/BN2D 算子验证
# ============================================================================
add_executable(test_bn_fwd_bwd test_bn_fwd_bwd.cpp)
target_link_libraries(test_bn_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_bn_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_bn_fwd_bwd)
endif()

set_target_properties(test_bn_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/bn
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_bn_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_bn_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_bn_fwd_bwd: BN FWD+BWD+INF correctness test")

# ============================================================================
# MNIST MLP ultimate accuracy test (AdamW) — 含 BatchNorm 的完整训练验证
# ============================================================================
add_executable(mnist_best_adamw mnist_best_adamw.cpp)
target_link_libraries(mnist_best_adamw PRIVATE renaissance)
target_compile_definitions(mnist_best_adamw PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mnist_best_adamw)
endif()

set_target_properties(mnist_best_adamw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/bn
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mnist_best_adamw PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mnist_best_adamw PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mnist_best_adamw: MNIST MLP ultimate accuracy test (AdamW)")

# ============================================================================
# MNIST MLP ultimate accuracy test with BatchNorm (AdamW)
# ============================================================================
add_executable(mnist_best_adamw_bn mnist_best_adamw_bn.cpp)
target_link_libraries(mnist_best_adamw_bn PRIVATE renaissance)
target_compile_definitions(mnist_best_adamw_bn PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mnist_best_adamw_bn)
endif()

set_target_properties(mnist_best_adamw_bn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/bn
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mnist_best_adamw_bn PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mnist_best_adamw_bn PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mnist_best_adamw_bn: MNIST MLP with BN ultimate accuracy test (AdamW)")

# ============================================================================
# MNIST mini-LeNet accuracy test (AdamW) — 无BN的CNN训练验证
# ============================================================================
add_executable(mini_lenet mini_lenet.cpp)
target_link_libraries(mini_lenet PRIVATE renaissance)
target_compile_definitions(mini_lenet PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mini_lenet)
endif()

set_target_properties(mini_lenet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/bn
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mini_lenet PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mini_lenet PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mini_lenet: MNIST mini-LeNet end-to-end accuracy test (AdamW)")

# ============================================================================
# MNIST mini-LeNet with BatchNorm accuracy test (AdamW)
# ============================================================================
add_executable(mini_lenet_bn mini_lenet_bn.cpp)
target_link_libraries(mini_lenet_bn PRIVATE renaissance)
target_compile_definitions(mini_lenet_bn PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mini_lenet_bn)
endif()

set_target_properties(mini_lenet_bn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/bn
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mini_lenet_bn PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mini_lenet_bn PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mini_lenet_bn: MNIST mini-LeNet with BN end-to-end accuracy test (AdamW)")
