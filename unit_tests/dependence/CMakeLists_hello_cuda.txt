# Unit Tests CMakeLists.txt
message(STATUS "=== Configuring Unit Tests ===")

# cuDNN manual discovery section (only if CUDA is enabled)
if(TR_USE_CUDA)
    # Use cuDNN from parent CMakeLists.txt configuration
    if(DEFINED CUDNN_ROOT)
        set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include")

        # Set platform-specific library path
        if(WIN32)
            if(EXISTS "${CUDNN_ROOT}/lib/x64/cudnn.lib")
                set(CUDNN_LIBRARY "${CUDNN_ROOT}/lib/x64/cudnn.lib")
            elseif(DEFINED CUDNN_LIB AND EXISTS "${CUDNN_LIB}/cudnn.lib")
                set(CUDNN_LIBRARY "${CUDNN_LIB}/cudnn.lib")
            else()
                message(FATAL_ERROR "cuDNN library not found in ${CUDNN_ROOT}")
            endif()
        else()
            # Try both lib64 and lib for Linux
            if(EXISTS "${CUDNN_ROOT}/lib64/libcudnn.so")
                set(CUDNN_LIBRARY "${CUDNN_ROOT}/lib64/libcudnn.so")
            elseif(EXISTS "${CUDNN_ROOT}/lib/libcudnn.so")
                set(CUDNN_LIBRARY "${CUDNN_ROOT}/lib/libcudnn.so")
            else()
                message(FATAL_ERROR "cuDNN library not found in ${CUDNN_ROOT}")
            endif()
        endif()

        if(NOT EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h" AND NOT EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
            message(FATAL_ERROR "cuDNN headers not found in: ${CUDNN_INCLUDE_DIR}")
        else()
            message(STATUS "Found cuDNN include: ${CUDNN_INCLUDE_DIR}")
            message(STATUS "Found cuDNN lib: ${CUDNN_LIBRARY}")
        endif()
    else()
        message(STATUS "CUDNN_ROOT not defined - using system cuDNN")
    endif()
else()
    message(STATUS "CUDA is disabled - skipping CUDA tests")
endif()

# Gather test sources
if(TR_USE_CUDA)
    # Include CUDA files when CUDA is enabled
    file(GLOB TEST_SOURCES "*.cpp" "*.cu")
else()
    # Exclude CUDA files when CUDA is disabled
    file(GLOB TEST_SOURCES "*.cpp")
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*cuda.*")  # Remove files with "cuda" in name
endif()
message(STATUS "Found test sources: ${TEST_SOURCES}")

# -----------------------------
# Iterate all test files
# -----------------------------
foreach(TEST_SOURCE ${TEST_SOURCES})

    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    get_filename_component(TEST_EXT ${TEST_SOURCE} EXT)
    message(STATUS "Creating test executable: ${TEST_NAME}")

    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # C++17 standard
    set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    # -----------------------------
    # CUDA tests (学习自CMakeLists_ref.txt第223行)
    # -----------------------------
    if(TEST_NAME MATCHES "cuda" OR TEST_NAME MATCHES "test_cuda_gemm")
        # Only build CUDA tests if CUDA is enabled
        if(TR_USE_CUDA)
            # 关键配置：设置CUDA运行时库为静态
            set_property(TARGET ${TEST_NAME} PROPERTY CUDA_RUNTIME_LIBRARY static)

            # 关键：添加UTF-8编码支持，解决C4819警告
            if(MSVC)
                target_compile_options(${TEST_NAME} PRIVATE /utf-8)
            endif()

            # 包含头文件目录
            target_include_directories(${TEST_NAME} PRIVATE
                ${CUDNN_INCLUDE_DIR}
            )

            # 链接CUDA库
            target_link_libraries(${TEST_NAME} PRIVATE
                CUDA::cudart
                CUDA::cublas
                ${CUDNN_LIBRARY}
            )

            message(STATUS "  - CUDA runtime library set to static for ${TEST_NAME}")
            message(STATUS "  - Added UTF-8 encoding support for ${TEST_NAME}")
            message(STATUS "  - Linked CUDA, cuBLAS, cuDNN for ${TEST_NAME}")
        endif()

    else()
        target_include_directories(${TEST_NAME} PRIVATE
            ${CUDNN_INCLUDE_DIR}
        )

        # Handle Eigen3 linking for both vcpkg and system installations
        target_link_libraries(${TEST_NAME} PRIVATE
            CUDA::cudart
            CUDA::cublas
            ${CUDNN_LIBRARY}
            OpenMP::OpenMP_CXX
        )

        # Add Eigen3 (now always available as Eigen3::Eigen)
        target_link_libraries(${TEST_NAME} PRIVATE Eigen3::Eigen)

        message(STATUS "  - Linked All libraries for ${TEST_NAME}")
    endif()

    # Output directory
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/unit_tests"
    )

    # Compiler warnings
    if(MSVC)
        target_compile_options(${TEST_NAME} PRIVATE /W4)
    else()
        target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

endforeach()

message(STATUS "=== Unit Tests Configuration Complete ===")
message(STATUS "Test executables will be built in: ${CMAKE_BINARY_DIR}/tests/unit_tests/")