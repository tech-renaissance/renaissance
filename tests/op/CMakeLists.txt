set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(test_relu test_relu.cpp)
target_link_libraries(test_relu PRIVATE renaissance)
target_compile_definitions(test_relu PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_relu)
endif()

set_target_properties(test_relu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_relu PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_relu PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_relu: ReLU operator correctness verification")

# ============================================================================
# ReLU FWD+BWD 串接测试 — BWD 复用 FWD 的 BOOLEAN 掩码
# ============================================================================
add_executable(test_relu_fwd_bwd test_relu_fwd_bwd.cpp)
target_link_libraries(test_relu_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_relu_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_relu_fwd_bwd)
endif()

set_target_properties(test_relu_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_relu_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_relu_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_relu_fwd_bwd: ReLU FWD+BWD chained correctness test")

# ============================================================================
# ReLU INF (inference) test — no mask verification needed
# ============================================================================
add_executable(test_relu_inf test_relu_inf.cpp)
target_link_libraries(test_relu_inf PRIVATE renaissance)
target_compile_definitions(test_relu_inf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_relu_inf)
endif()

set_target_properties(test_relu_inf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_relu_inf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_relu_inf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_relu_inf: ReLU INF (inference) correctness test")

# ============================================================================
# IDENTITY FWD+BWD 串接测试 — 恒等映射验证
# ============================================================================
add_executable(test_identity_fwd_bwd test_identity_fwd_bwd.cpp)
target_link_libraries(test_identity_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_identity_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_identity_fwd_bwd)
endif()

set_target_properties(test_identity_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_identity_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_identity_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_identity_fwd_bwd: IDENTITY FWD+BWD correctness test")

# ============================================================================
# TANH FWD+BWD 串接测试 — Tanh 激活函数验证
# ============================================================================
add_executable(test_tanh_fwd_bwd test_tanh_fwd_bwd.cpp)
target_link_libraries(test_tanh_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_tanh_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_tanh_fwd_bwd)
endif()

set_target_properties(test_tanh_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_tanh_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_tanh_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_tanh_fwd_bwd: TANH FWD+BWD correctness test")

# ============================================================================
# SiLU FWD+BWD 串接测试
# ============================================================================
add_executable(test_silu_fwd_bwd test_silu_fwd_bwd.cpp)
target_link_libraries(test_silu_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_silu_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_silu_fwd_bwd)
endif()

set_target_properties(test_silu_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_silu_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_silu_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_silu_fwd_bwd: SiLU FWD+BWD correctness test")

# ============================================================================
# ReLU6 FWD+BWD 串接测试
# ============================================================================
add_executable(test_relu6_fwd_bwd test_relu6_fwd_bwd.cpp)
target_link_libraries(test_relu6_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_relu6_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_relu6_fwd_bwd)
endif()

set_target_properties(test_relu6_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_relu6_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_relu6_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_relu6_fwd_bwd: ReLU6 FWD+BWD correctness test")

# ============================================================================
# LeakyReLU FWD+BWD 串接测试
# ============================================================================
add_executable(test_leaky_relu_fwd_bwd test_leaky_relu_fwd_bwd.cpp)
target_link_libraries(test_leaky_relu_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_leaky_relu_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_leaky_relu_fwd_bwd)
endif()

set_target_properties(test_leaky_relu_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_leaky_relu_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_leaky_relu_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_leaky_relu_fwd_bwd: LeakyReLU FWD+BWD correctness test")

# ============================================================================
# HardSwish FWD+BWD 串接测试
# ============================================================================
add_executable(test_hardswish_fwd_bwd test_hardswish_fwd_bwd.cpp)
target_link_libraries(test_hardswish_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_hardswish_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_hardswish_fwd_bwd)
endif()

set_target_properties(test_hardswish_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_hardswish_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_hardswish_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_hardswish_fwd_bwd: HardSwish FWD+BWD correctness test")

# ============================================================================
# ELU FWD+BWD 串接测试
# ============================================================================
add_executable(test_elu_fwd_bwd test_elu_fwd_bwd.cpp)
target_link_libraries(test_elu_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_elu_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_elu_fwd_bwd)
endif()

set_target_properties(test_elu_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_elu_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_elu_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_elu_fwd_bwd: ELU FWD+BWD correctness test")

# ============================================================================
# Sigmoid FWD+BWD 串接测试
# ============================================================================
add_executable(test_sigmoid_fwd_bwd test_sigmoid_fwd_bwd.cpp)
target_link_libraries(test_sigmoid_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_sigmoid_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_sigmoid_fwd_bwd)
endif()

set_target_properties(test_sigmoid_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_sigmoid_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_sigmoid_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_sigmoid_fwd_bwd: Sigmoid FWD+BWD correctness test")

# ============================================================================
# FLATTEN FWD+BWD 串接测试 — Flatten 算子验证
# ============================================================================
add_executable(test_flatten_fwd_bwd test_flatten_fwd_bwd.cpp)
target_link_libraries(test_flatten_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_flatten_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_flatten_fwd_bwd)
endif()

set_target_properties(test_flatten_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_flatten_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_flatten_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_flatten_fwd_bwd: FLATTEN FWD+BWD correctness test")

# ============================================================================
# FC FWD+BWD 测试
# ============================================================================
add_executable(test_fc_fwd_bwd test_fc_fwd_bwd.cpp)
target_link_libraries(test_fc_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_fc_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_fc_fwd_bwd)
endif()

set_target_properties(test_fc_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_fc_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_fc_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_fc_fwd_bwd: FC FWD+BWD correctness test")

# ============================================================================
# Conv FWD+BWD 测试
# ============================================================================
add_executable(test_conv_fwd_bwd test_conv_fwd_bwd.cpp)
target_link_libraries(test_conv_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_conv_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_conv_fwd_bwd)
endif()

set_target_properties(test_conv_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_conv_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_conv_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_conv_fwd_bwd: Conv FWD+BWD correctness test")

# ============================================================================
# Conv INF 推理测试
# ============================================================================
add_executable(test_conv_inf test_conv_inf.cpp)
target_link_libraries(test_conv_inf PRIVATE renaissance)
target_compile_definitions(test_conv_inf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_conv_inf)
endif()

set_target_properties(test_conv_inf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_conv_inf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_conv_inf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_conv_inf: Conv INF correctness test")

# ============================================================================
# FC AMP 性能测试 —— 专用耗时测试
# ============================================================================
add_executable(test_fc_amp test_fc_amp.cpp)
target_link_libraries(test_fc_amp PRIVATE renaissance)
target_compile_definitions(test_fc_amp PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_fc_amp)
endif()

set_target_properties(test_fc_amp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_fc_amp PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_fc_amp PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_fc_amp: FC AMP performance test (FWD+BWD)")

# ============================================================================
# FC FP32 性能测试 —— 专用耗时测试
# ============================================================================
add_executable(test_fc_fp32 test_fc_fp32.cpp)
target_link_libraries(test_fc_fp32 PRIVATE renaissance)
target_compile_definitions(test_fc_fp32 PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_fc_fp32)
endif()

set_target_properties(test_fc_fp32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_fc_fp32 PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_fc_fp32 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_fc_fp32: FC FP32 performance test (FWD+BWD)")
# ============================================================================
# ReLU cuDNN Frontend standalone performance test
# ============================================================================
if(TR_USE_CUDA)
    add_executable(test_relu_cudnn_frontend test_relu_cudnn_frontend.cpp)

    target_include_directories(test_relu_cudnn_frontend PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/fp16
        ${CMAKE_SOURCE_DIR}/include
    )

    # 添加CUDA头文件路径（cuDNN依赖CUDA头文件）
    if(DEFINED TR_CUDA_PATH)
        target_include_directories(test_relu_cudnn_frontend SYSTEM PRIVATE ${TR_CUDA_PATH}/include)
    endif()

    # 添加cuDNN头文件路径（cuDNN Frontend依赖cuDNN头文件）
    if(DEFINED TR_CUDNN_INCLUDE_DIR)
        target_include_directories(test_relu_cudnn_frontend SYSTEM PRIVATE ${TR_CUDNN_INCLUDE_DIR})
        # cuDNN 9.x版本化子目录
        if(DEFINED TR_CUDNN_INCLUDE_DIR_VERSIONED)
            target_include_directories(test_relu_cudnn_frontend SYSTEM PRIVATE ${TR_CUDNN_INCLUDE_DIR_VERSIONED})
        endif()
    endif()

    if(DEFINED TR_CUDNN_FRONTEND_INCLUDE_DIR)
        target_include_directories(test_relu_cudnn_frontend SYSTEM PRIVATE ${TR_CUDNN_FRONTEND_INCLUDE_DIR})
    elseif(DEFINED TR_CUDNN_FRONTEND_PATH)
        target_include_directories(test_relu_cudnn_frontend SYSTEM PRIVATE ${TR_CUDNN_FRONTEND_PATH}/include)
    endif()

    if(DEFINED CUDA_cudart_LIBRARY)
        target_link_libraries(test_relu_cudnn_frontend PRIVATE ${CUDA_cudart_LIBRARY})
    endif()
    if(DEFINED CUDNN_LIBRARY)
        target_link_libraries(test_relu_cudnn_frontend PRIVATE ${CUDNN_LIBRARY})
    endif()

    set_target_properties(test_relu_cudnn_frontend PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
        WIN32_EXECUTABLE FALSE
    )

    if(MSVC)
        target_compile_options(test_relu_cudnn_frontend PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(test_relu_cudnn_frontend PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()

    message(STATUS "  - test_relu_cudnn_frontend: ReLU standalone cuDNN Frontend performance")

    # ============================================================================
    # ReLU cuDNN Frontend BOOLEAN bitmask对比测试
    # ============================================================================
    add_executable(test_relu_cudnn_bitmask test_relu_cudnn_bitmask.cpp)

    target_include_directories(test_relu_cudnn_bitmask PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/fp16
        ${CMAKE_SOURCE_DIR}/include
    )

    # 添加CUDA头文件路径（cuDNN依赖CUDA头文件）
    if(DEFINED TR_CUDA_PATH)
        target_include_directories(test_relu_cudnn_bitmask SYSTEM PRIVATE ${TR_CUDA_PATH}/include)
    endif()

    # 添加cuDNN头文件路径（cuDNN Frontend依赖cuDNN头文件）
    if(DEFINED TR_CUDNN_INCLUDE_DIR)
        target_include_directories(test_relu_cudnn_bitmask SYSTEM PRIVATE ${TR_CUDNN_INCLUDE_DIR})
        # cuDNN 9.x版本化子目录
        if(DEFINED TR_CUDNN_INCLUDE_DIR_VERSIONED)
            target_include_directories(test_relu_cudnn_bitmask SYSTEM PRIVATE ${TR_CUDNN_INCLUDE_DIR_VERSIONED})
        endif()
    endif()

    if(DEFINED TR_CUDNN_FRONTEND_INCLUDE_DIR)
        target_include_directories(test_relu_cudnn_bitmask SYSTEM PRIVATE ${TR_CUDNN_FRONTEND_INCLUDE_DIR})
    elseif(DEFINED TR_CUDNN_FRONTEND_PATH)
        target_include_directories(test_relu_cudnn_bitmask SYSTEM PRIVATE ${TR_CUDNN_FRONTEND_PATH}/include)
    endif()

    if(DEFINED CUDA_cudart_LIBRARY)
        target_link_libraries(test_relu_cudnn_bitmask PRIVATE ${CUDA_cudart_LIBRARY})
    endif()
    if(DEFINED CUDNN_LIBRARY)
        target_link_libraries(test_relu_cudnn_bitmask PRIVATE ${CUDNN_LIBRARY})
    endif()

    set_target_properties(test_relu_cudnn_bitmask PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
        WIN32_EXECUTABLE FALSE
    )

    if(MSVC)
        target_compile_options(test_relu_cudnn_bitmask PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(test_relu_cudnn_bitmask PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()

    message(STATUS "  - test_relu_cudnn_bitmask: ReLU cuDNN Frontend BOOLEAN bitmask implementation")
endif()

# ============================================================================
# MaxPool FWD+BWD 串接测试 — MaxPool 算子验证
# ============================================================================
add_executable(test_maxpool_fwd_bwd test_maxpool_fwd_bwd.cpp)
target_link_libraries(test_maxpool_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_maxpool_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_maxpool_fwd_bwd)
endif()

set_target_properties(test_maxpool_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_maxpool_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_maxpool_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_maxpool_fwd_bwd: MaxPool FWD+BWD correctness test")

# ============================================================================
# MaxPool Visual 测试 — 可视化调试工具
# ============================================================================
add_executable(test_maxpool_visual test_maxpool_visual.cpp)
target_link_libraries(test_maxpool_visual PRIVATE renaissance)
target_compile_definitions(test_maxpool_visual PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_maxpool_visual)
endif()

set_target_properties(test_maxpool_visual PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_maxpool_visual PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_maxpool_visual PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_maxpool_visual: MaxPool visual debugging test")

# ============================================================================
# AvgPool FWD+BWD 串接测试 — AvgPool 算子验证
# ============================================================================
add_executable(test_avgpool_fwd_bwd test_avgpool_fwd_bwd.cpp)
target_link_libraries(test_avgpool_fwd_bwd PRIVATE renaissance)
target_compile_definitions(test_avgpool_fwd_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_avgpool_fwd_bwd)
endif()

set_target_properties(test_avgpool_fwd_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_avgpool_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_avgpool_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_avgpool_fwd_bwd: AvgPool FWD+BWD correctness test")

# ============================================================================
# Dropout FWD+BWD+INF 数学属性测试 — 与 PyTorch 无关，自包含验证
# ============================================================================
add_executable(test_dropout test_dropout.cpp)
target_link_libraries(test_dropout PRIVATE renaissance)
target_compile_definitions(test_dropout PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dropout)
endif()

set_target_properties(test_dropout PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_dropout PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_dropout PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_dropout: Dropout FWD+BWD+INF property test")

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
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_bn_fwd_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_bn_fwd_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_bn_fwd_bwd: BN FWD+BWD+INF correctness test")
