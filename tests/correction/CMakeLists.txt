# GAP算子数学正确性测试 (correction)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(test_gap test_gap.cpp)
target_link_libraries(test_gap PRIVATE renaissance)
target_compile_definitions(test_gap PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_gap)
endif()

set_target_properties(test_gap PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_gap PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_gap PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_gap: GAP FWD+BWD correction test (TSR/PyTorch)")

add_executable(test_flatten_fc_relu_fc test_flatten_fc_relu_fc.cpp)
target_link_libraries(test_flatten_fc_relu_fc PRIVATE renaissance)
target_compile_definitions(test_flatten_fc_relu_fc PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_flatten_fc_relu_fc)
endif()

set_target_properties(test_flatten_fc_relu_fc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_flatten_fc_relu_fc PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_flatten_fc_relu_fc PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_flatten_fc_relu_fc: Flatten+FC+ReLU+FC composite correction test")

add_executable(test_mlp_final test_mlp_final.cpp)
target_link_libraries(test_mlp_final PRIVATE renaissance)
target_compile_definitions(test_mlp_final PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_mlp_final)
endif()

set_target_properties(test_mlp_final PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_mlp_final PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_mlp_final PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_mlp_final: MLP 784-512-256-10 (Tanh+SoftmaxCE) composite correction test")

add_executable(test_softmax_ce test_softmax_ce.cpp)
target_link_libraries(test_softmax_ce PRIVATE renaissance)
target_compile_definitions(test_softmax_ce PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_softmax_ce)
endif()

set_target_properties(test_softmax_ce PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_softmax_ce PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_softmax_ce PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_softmax_ce: SOFTMAX_CE FWD+BWD correction test")

# SOFTMAX_CE INF (推理) 数学正确性测试 (correction)
add_executable(test_softmax_ce_inf test_softmax_ce_inf.cpp)
target_link_libraries(test_softmax_ce_inf PRIVATE renaissance)
target_compile_definitions(test_softmax_ce_inf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_softmax_ce_inf)
endif()

set_target_properties(test_softmax_ce_inf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_softmax_ce_inf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_softmax_ce_inf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_softmax_ce_inf: SOFTMAX_CE INF correction test")

# SoftmaxCE FWD vs INF loss 差异测试
add_executable(test_softmax_ce_loss test_softmax_ce_loss.cpp)
target_link_libraries(test_softmax_ce_loss PRIVATE renaissance)
target_compile_definitions(test_softmax_ce_loss PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_softmax_ce_loss)
endif()
set_target_properties(test_softmax_ce_loss PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_softmax_ce_loss PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_softmax_ce_loss PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_softmax_ce_loss: SOFTMAX_CE FWD vs INF loss diff test")

# RANGE_CLEAR 数学正确性测试
add_executable(test_clear test_clear.cpp)
target_link_libraries(test_clear PRIVATE renaissance)
target_compile_definitions(test_clear PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_clear)
endif()

set_target_properties(test_clear PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_clear PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_clear PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_clear: RANGE_CLEAR correction test")

# RANGE_D2D_COPY 数学正确性测试
add_executable(test_d2d_copy test_d2d_copy.cpp)
target_link_libraries(test_d2d_copy PRIVATE renaissance)
target_compile_definitions(test_d2d_copy PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_d2d_copy)
endif()
set_target_properties(test_d2d_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_d2d_copy PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_d2d_copy PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_d2d_copy: RANGE_D2D_COPY correction test")

# RANGE_CAST_FP32_TO_FP16 数学正确性测试
add_executable(test_cast_fp32_to_fp16 test_cast_fp32_to_fp16.cpp)
target_link_libraries(test_cast_fp32_to_fp16 PRIVATE renaissance)
target_compile_definitions(test_cast_fp32_to_fp16 PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_cast_fp32_to_fp16)
endif()
set_target_properties(test_cast_fp32_to_fp16 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_cast_fp32_to_fp16 PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_cast_fp32_to_fp16 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_cast_fp32_to_fp16: RANGE_CAST_FP32_TO_FP16 correction test")

# RANGE_CAST_FP16_TO_FP32 数学正确性测试
add_executable(test_cast_fp16_to_fp32 test_cast_fp16_to_fp32.cpp)
target_link_libraries(test_cast_fp16_to_fp32 PRIVATE renaissance)
target_compile_definitions(test_cast_fp16_to_fp32 PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_cast_fp16_to_fp32)
endif()
set_target_properties(test_cast_fp16_to_fp32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_cast_fp16_to_fp32 PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_cast_fp16_to_fp32 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_cast_fp16_to_fp32: RANGE_CAST_FP16_TO_FP32 correction test")

# RANGE_CHECK_NAN 数学正确性测试
add_executable(test_check_nan test_check_nan.cpp)
target_link_libraries(test_check_nan PRIVATE renaissance)
target_compile_definitions(test_check_nan PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_check_nan)
endif()
set_target_properties(test_check_nan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_check_nan PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_check_nan PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_check_nan: RANGE_CHECK_NAN correction test")

# RANGE_MEAN_ALLREDUCE 数学正确性测试
add_executable(test_mean_allreduce test_mean_allreduce.cpp)
target_link_libraries(test_mean_allreduce PRIVATE renaissance)
target_compile_definitions(test_mean_allreduce PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_mean_allreduce)
endif()
set_target_properties(test_mean_allreduce PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_mean_allreduce PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_mean_allreduce PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_mean_allreduce: RANGE_MEAN_ALLREDUCE correction test")

# RANGE_UPDATE_WEIGHT_SGD 数学正确性测试
add_executable(test_sgd_weight test_sgd_weight.cpp)
target_link_libraries(test_sgd_weight PRIVATE renaissance)
target_compile_definitions(test_sgd_weight PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_sgd_weight)
endif()
set_target_properties(test_sgd_weight PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_sgd_weight PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_sgd_weight PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_sgd_weight: RANGE_UPDATE_WEIGHT_SGD correction test")

# 优化器 RANGE_OP 正确性测试（Weight 系列）
set(OPT_WEIGHT_TESTS
    test_sgd_bias
    test_momentum_weight
    test_momentum_bias
    test_nesterov_weight
    test_nesterov_bias
    test_lars_weight
    test_lars_nesterov_weight
    test_adam_weight
    test_adamw_weight
    test_adam_bias
    test_adamw_bias
)
foreach(test_name IN LISTS OPT_WEIGHT_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE renaissance)
    target_compile_definitions(${test_name} PRIVATE TR_LOG_LEVEL=1)
    if(TR_USE_CUDA)
        setup_gpu_runtime_env(${test_name})
    endif()
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
        WIN32_EXECUTABLE FALSE
    )
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
    message(STATUS "  - ${test_name}")
endforeach()

# RANGE_H2D_COPY 系列：数学正确性测试 + 性能测试
set(H2D_CORRECTION_TESTS
    test_h2d_copy_a
    test_h2d_copy_b
    test_h2d_copy_dtensor
)

foreach(test_name IN LISTS H2D_CORRECTION_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE renaissance)
    target_compile_definitions(${test_name} PRIVATE TR_LOG_LEVEL=1)
    if(TR_USE_CUDA)
        setup_gpu_runtime_env(${test_name})
    endif()
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
        WIN32_EXECUTABLE FALSE
    )
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
    message(STATUS "  - ${test_name}: RANGE_H2D_COPY correction test")
endforeach()

# TRANSFER_A 性能测试
add_executable(perf_h2d_copy_a perf_h2d_copy_a.cpp)
target_link_libraries(perf_h2d_copy_a PRIVATE renaissance)
target_compile_definitions(perf_h2d_copy_a PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_h2d_copy_a)
endif()
set_target_properties(perf_h2d_copy_a PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(perf_h2d_copy_a PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_h2d_copy_a PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - perf_h2d_copy_a: RANGE_H2D_COPY_A performance test")

# TRANSFER_B 性能测试
add_executable(perf_h2d_copy_b perf_h2d_copy_b.cpp)
target_link_libraries(perf_h2d_copy_b PRIVATE renaissance)
target_compile_definitions(perf_h2d_copy_b PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_h2d_copy_b)
endif()
set_target_properties(perf_h2d_copy_b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(perf_h2d_copy_b PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_h2d_copy_b PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - perf_h2d_copy_b: RANGE_H2D_COPY_B performance test")

# ============================================================================
# DeepLearningTask GPU 测试
# ============================================================================

function(setup_dl_test name)
    target_link_libraries(${name} PRIVATE renaissance)
    target_compile_definitions(${name} PRIVATE TR_LOG_LEVEL=1)
    if(TR_USE_CUDA)
        setup_gpu_runtime_env(${name})
    endif()
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
        WIN32_EXECUTABLE FALSE)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
endfunction()

add_executable(test_dl_dry test_dl_dry.cpp)
setup_dl_test(test_dl_dry)
message(STATUS "  - test_dl_dry: DeepLearningTask DRY-RUN test")

add_executable(perf_dl_xfer_ab perf_dl_xfer_ab.cpp)
setup_dl_test(perf_dl_xfer_ab)
message(STATUS "  - perf_dl_xfer_ab: AB transfer bandwidth test")

add_executable(test_dl_full test_dl_full.cpp)
setup_dl_test(test_dl_full)
message(STATUS "  - test_dl_full: DeepLearningTask full training test (CPU/GPU/AMP)")

add_executable(mnist_best_adamw mnist_best_adamw.cpp)
setup_dl_test(mnist_best_adamw)
message(STATUS "  - mnist_best_adamw: MNIST MLP ultimate accuracy test (AdamW)")

add_executable(mnist_best_lars mnist_best_lars.cpp)
setup_dl_test(mnist_best_lars)
message(STATUS "  - mnist_best_lars: MNIST MLP ultimate accuracy test (LARS reference)")

add_executable(test_h2d_copy_correctness test_h2d_copy_correctness.cpp)
setup_dl_test(test_h2d_copy_correctness)
message(STATUS "  - test_h2d_copy_correctness: H2D COPY A/B correctness (preprocessor + fetch)")

add_executable(test_h2d_copy_bandwidth test_h2d_copy_bandwidth.cpp)
setup_dl_test(test_h2d_copy_bandwidth)
message(STATUS "  - test_h2d_copy_bandwidth: H2D COPY A/B bandwidth (full epoch)")

add_executable(test_h2d_only_epoch test_h2d_only_epoch.cpp)
setup_dl_test(test_h2d_only_epoch)
message(STATUS "  - test_h2d_only_epoch: H2D-only epoch test (compile_h2d_only + run_h2d_only)")

add_executable(test_two_batch_correction
    test_two_batch_correction.cpp
    ${CMAKE_SOURCE_DIR}/src/task/deep_learning_task.cpp
)
setup_dl_test(test_two_batch_correction)
target_compile_definitions(test_two_batch_correction PRIVATE TR_TEST_TWO_BATCH_CORRECTION)
message(STATUS "  - test_two_batch_correction: Two-batch H2D correction test")

add_executable(test_pw_ultimate
    test_pw_ultimate.cpp
    ${CMAKE_SOURCE_DIR}/src/data/transfer_station.cpp
)
setup_dl_test(test_pw_ultimate)
target_compile_definitions(test_pw_ultimate PRIVATE FOR_TRANSFER_STATION_UNIT_TESTS_ONLY)
target_include_directories(test_pw_ultimate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
)
message(STATUS "  - test_pw_ultimate: PW ultimate test")

# ============================================================================
# FusedNormalization 测试
# ============================================================================

add_executable(test_fused_normalization test_fused_normalization.cpp)
target_link_libraries(test_fused_normalization PRIVATE renaissance)
target_compile_definitions(test_fused_normalization PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_fused_normalization)
endif()
set_target_properties(test_fused_normalization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_fused_normalization PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_fused_normalization PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_fused_normalization: FusedNormalization FP32 vs AMP numerical comparison test")

# ============================================================================
# MNIST Dropout 测试
# ============================================================================

add_executable(mnist_best_dropout mnist_best_dropout.cpp)
target_link_libraries(mnist_best_dropout PRIVATE renaissance)
target_compile_definitions(mnist_best_dropout PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mnist_best_dropout)
endif()

set_target_properties(mnist_best_dropout PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mnist_best_dropout PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mnist_best_dropout PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mnist_best_dropout: MNIST MLP with Dropout ultimate accuracy test")

add_executable(mnist_best_maxpool mnist_best_maxpool.cpp)
target_link_libraries(mnist_best_maxpool PRIVATE renaissance)
target_compile_definitions(mnist_best_maxpool PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mnist_best_maxpool)
endif()

set_target_properties(mnist_best_maxpool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mnist_best_maxpool PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mnist_best_maxpool PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mnist_best_maxpool: MNIST MLP with MaxPool ultimate accuracy test")