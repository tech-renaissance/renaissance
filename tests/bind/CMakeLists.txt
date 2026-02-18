# ============================================================================
# renAIssance Framework - NUMA Binding Experiment
# ============================================================================
# This is a standalone test directory, similar to loading tests
# NOTE: These tests are Linux-only (requires SysFS, NUMA, unistd.h)

cmake_minimum_required(VERSION 3.22)

project(NUMA_Bind VERSION 0.1.0 LANGUAGES CXX)

# Skip on Windows (these tests require Linux SysFS/NUMA APIs)
if(WIN32)
    message(STATUS "NUMA binding tests are Linux-only, skipping on Windows")
    return()
endif()

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory
set(BIND_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests/bind)

# ============================================================================
# CUDA Runtime Linking
# ============================================================================
# bind tests inherit CUDA configuration from parent project
# Parent CMakeLists.txt already sets CUDA_cudart_LIBRARY to correct lib64 path

# 仅在具有CUDA环境时才编译bind测试
if(TR_USE_CUDA)
    # Expert Solution A - GMX
    # ============================================================================
    add_executable(bind_a a.cpp)
    target_link_libraries(bind_a PRIVATE ${CUDA_cudart_LIBRARY})

    # Platform-specific settings
    if(MSVC)
        target_compile_options(bind_a PRIVATE /utf-8)
    else()
        target_link_libraries(bind_a PRIVATE pthread)
        target_compile_options(bind_a PRIVATE -std=c++17 -O3)
    endif()

    set_target_properties(bind_a PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIND_OUTPUT_DIR}
    )

    # Expert Solution F - SNX
    # ============================================================================
    add_executable(bind_f f.cpp)
    target_link_libraries(bind_f PRIVATE ${CUDA_cudart_LIBRARY})

    # Platform-specific settings
    if(MSVC)
        target_compile_options(bind_f PRIVATE /utf-8)
    else()
        target_link_libraries(bind_f PRIVATE pthread)
        target_compile_options(bind_f PRIVATE -std=c++17 -O3)
    endif()

    set_target_properties(bind_f PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIND_OUTPUT_DIR}
    )

    # ============================================================================
    # Build Summary
    # ============================================================================
    message(STATUS "=== NUMA Binding Tests Configured ===")
    message(STATUS "bind_a  : Expert Solution A (GMX)")
    message(STATUS "bind_f  : Expert Solution F (SNX)")
    message(STATUS "Use: cd build && make bind_a bind_f")
    message(STATUS "Or:  cd build && cmake .. && make -CMakeLists.txt=tests/bind/bind_f")
else()
    message(STATUS "CUDA not available (TR_USE_CUDA=OFF), skipping NUMA binding tests")
endif()
