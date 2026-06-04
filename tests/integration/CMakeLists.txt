# Integration tests: end-to-end model training
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(mlp_adamw mlp_adamw.cpp)
target_link_libraries(mlp_adamw PRIVATE renaissance)
target_compile_definitions(mlp_adamw PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_adamw)
endif()

set_target_properties(mlp_adamw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_adamw PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_adamw PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_adamw: MNIST MLP end-to-end accuracy test (AdamW)")

add_executable(mlp_adamw_larger mlp_adamw_larger.cpp)
target_link_libraries(mlp_adamw_larger PRIVATE renaissance)
target_compile_definitions(mlp_adamw_larger PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_adamw_larger)
endif()

set_target_properties(mlp_adamw_larger PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_adamw_larger PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_adamw_larger PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_adamw_larger: MNIST MLP 32x32 end-to-end accuracy test (AdamW)")

add_executable(mlp_adamw_small mlp_adamw_small.cpp)
target_link_libraries(mlp_adamw_small PRIVATE renaissance)
target_compile_definitions(mlp_adamw_small PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_adamw_small)
endif()

set_target_properties(mlp_adamw_small PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_adamw_small PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_adamw_small PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_adamw_small: MNIST MLP 24x24 end-to-end accuracy test (AdamW)")

add_executable(mlp_adamw_larger2 mlp_adamw_larger2.cpp)
target_link_libraries(mlp_adamw_larger2 PRIVATE renaissance)
target_compile_definitions(mlp_adamw_larger2 PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_adamw_larger2)
endif()

set_target_properties(mlp_adamw_larger2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_adamw_larger2 PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_adamw_larger2 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_adamw_larger2: MNIST MLP 32x32 with Crop end-to-end accuracy test (AdamW)")

add_executable(mlp_adamw_median mlp_adamw_median.cpp)
target_link_libraries(mlp_adamw_median PRIVATE renaissance)
target_compile_definitions(mlp_adamw_median PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_adamw_median)
endif()

set_target_properties(mlp_adamw_median PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_adamw_median PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_adamw_median PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_adamw_median: MNIST MLP 28x28 with Pad(2) val end-to-end accuracy test (AdamW)")

add_executable(mlp_adamw_median2 mlp_adamw_median2.cpp)
target_link_libraries(mlp_adamw_median2 PRIVATE renaissance)
target_compile_definitions(mlp_adamw_median2 PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(mlp_adamw_median2)
endif()

set_target_properties(mlp_adamw_median2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(mlp_adamw_median2 PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(mlp_adamw_median2 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()

message(STATUS "  - mlp_adamw_median2: MNIST MLP 28x28 with Resize val end-to-end accuracy test (AdamW)")
