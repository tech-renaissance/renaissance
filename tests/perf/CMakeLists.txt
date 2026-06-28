# GAP算子性能测试 (perf)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(test_gap_perf test_gap_perf.cpp)
target_link_libraries(test_gap_perf PRIVATE renaissance)
target_compile_definitions(test_gap_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_gap_perf)
endif()

set_target_properties(test_gap_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_gap_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_gap_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_gap_perf: GAP FWD+BWD performance test (randn)")

# GAP+FC联合性能测试
add_executable(test_gap_fc_perf test_gap_fc_perf.cpp)
target_link_libraries(test_gap_fc_perf PRIVATE renaissance)
target_compile_definitions(test_gap_fc_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_gap_fc_perf)
endif()

set_target_properties(test_gap_fc_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_gap_fc_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_gap_fc_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_gap_fc_perf: GAP+FC joint performance test (randn)")

# SOFTMAX_CE算子性能测试 (perf)
add_executable(test_softmax_ce_perf test_softmax_ce_perf.cpp)
target_link_libraries(test_softmax_ce_perf PRIVATE renaissance)
target_compile_definitions(test_softmax_ce_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_softmax_ce_perf)
endif()

set_target_properties(test_softmax_ce_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_softmax_ce_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_softmax_ce_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_softmax_ce_perf: SOFTMAX_CE FWD+BWD performance test (randn)")

# SOFTMAX_CE INF (推理) 性能测试 (perf)
add_executable(test_softmax_ce_perf_inf test_softmax_ce_perf_inf.cpp)
target_link_libraries(test_softmax_ce_perf_inf PRIVATE renaissance)
target_compile_definitions(test_softmax_ce_perf_inf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_softmax_ce_perf_inf)
endif()

set_target_properties(test_softmax_ce_perf_inf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_softmax_ce_perf_inf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_softmax_ce_perf_inf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_softmax_ce_perf_inf: SOFTMAX_CE INF performance test (randn)")

# RANGE_CLEAR 性能测试
add_executable(test_clear_perf test_clear_perf.cpp)
target_link_libraries(test_clear_perf PRIVATE renaissance)
target_compile_definitions(test_clear_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_clear_perf)
endif()

set_target_properties(test_clear_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_clear_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_clear_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_clear_perf: RANGE_CLEAR performance test")

# RANGE_D2D_COPY 性能测试
add_executable(test_d2d_copy_perf test_d2d_copy_perf.cpp)
target_link_libraries(test_d2d_copy_perf PRIVATE renaissance)
target_compile_definitions(test_d2d_copy_perf PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_d2d_copy_perf)
endif()
set_target_properties(test_d2d_copy_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_d2d_copy_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_d2d_copy_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_d2d_copy_perf: RANGE_D2D_COPY performance test")

# RANGE_CAST_FP32_TO_FP16 性能测试
add_executable(test_cast_fp32_to_fp16_perf test_cast_fp32_to_fp16_perf.cpp)
target_link_libraries(test_cast_fp32_to_fp16_perf PRIVATE renaissance)
target_compile_definitions(test_cast_fp32_to_fp16_perf PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_cast_fp32_to_fp16_perf)
endif()
set_target_properties(test_cast_fp32_to_fp16_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_cast_fp32_to_fp16_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_cast_fp32_to_fp16_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_cast_fp32_to_fp16_perf: RANGE_CAST_FP32_TO_FP16 performance test")

# RANGE_CAST_FP16_TO_FP32 性能测试
add_executable(test_cast_fp16_to_fp32_perf test_cast_fp16_to_fp32_perf.cpp)
target_link_libraries(test_cast_fp16_to_fp32_perf PRIVATE renaissance)
target_compile_definitions(test_cast_fp16_to_fp32_perf PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_cast_fp16_to_fp32_perf)
endif()
set_target_properties(test_cast_fp16_to_fp32_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_cast_fp16_to_fp32_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_cast_fp16_to_fp32_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_cast_fp16_to_fp32_perf: RANGE_CAST_FP16_TO_FP32 performance test")

# RANGE_CHECK_NAN 性能测试
add_executable(test_check_nan_perf test_check_nan_perf.cpp)
target_link_libraries(test_check_nan_perf PRIVATE renaissance)
target_compile_definitions(test_check_nan_perf PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_check_nan_perf)
endif()
set_target_properties(test_check_nan_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_check_nan_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_check_nan_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_check_nan_perf: RANGE_CHECK_NAN performance test")

# MaxPool 性能测试
add_executable(test_maxpool_perf test_maxpool_perf.cpp)
target_link_libraries(test_maxpool_perf PRIVATE renaissance)
target_compile_definitions(test_maxpool_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_maxpool_perf)
endif()

set_target_properties(test_maxpool_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_maxpool_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_maxpool_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_maxpool_perf: MaxPool FWD+BWD+INF performance test")

# Dropout 性能测试
add_executable(test_dropout_perf test_dropout_perf.cpp)
target_link_libraries(test_dropout_perf PRIVATE renaissance)
target_compile_definitions(test_dropout_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dropout_perf)
endif()

set_target_properties(test_dropout_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_dropout_perf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_dropout_perf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - test_dropout_perf: Dropout FWD+BWD+INF performance test")

# CBR_AMP_FWD 性能测试
add_executable(perf_cbr_fwd perf_cbr_fwd.cpp)
target_link_libraries(perf_cbr_fwd PRIVATE renaissance)
target_compile_definitions(perf_cbr_fwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_cbr_fwd)
endif()

set_target_properties(perf_cbr_fwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(perf_cbr_fwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_cbr_fwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - perf_cbr_fwd: CBR_AMP_FWD performance test")

# CBR_AMP_BWD 性能测试
add_executable(perf_cbr_bwd perf_cbr_bwd.cpp)
target_link_libraries(perf_cbr_bwd PRIVATE renaissance)
target_compile_definitions(perf_cbr_bwd PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_cbr_bwd)
endif()

set_target_properties(perf_cbr_bwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(perf_cbr_bwd PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_cbr_bwd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - perf_cbr_bwd: CBR_AMP_BWD performance test")

# CBR_AMP_BWD_FIRST_LAYER 性能测试
add_executable(perf_cbr_bwd_first_layer perf_cbr_bwd_first_layer.cpp)
target_link_libraries(perf_cbr_bwd_first_layer PRIVATE renaissance)
target_compile_definitions(perf_cbr_bwd_first_layer PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_cbr_bwd_first_layer)
endif()

set_target_properties(perf_cbr_bwd_first_layer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(perf_cbr_bwd_first_layer PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_cbr_bwd_first_layer PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - perf_cbr_bwd_first_layer: CBR_AMP_BWD_FIRST_LAYER performance test")

# CBR_AMP_INF 性能测试
add_executable(perf_cbr_inf perf_cbr_inf.cpp)
target_link_libraries(perf_cbr_inf PRIVATE renaissance)
target_compile_definitions(perf_cbr_inf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_cbr_inf)
endif()

set_target_properties(perf_cbr_inf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/perf
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(perf_cbr_inf PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_cbr_inf PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - perf_cbr_inf: CBR_AMP_INF performance test")

