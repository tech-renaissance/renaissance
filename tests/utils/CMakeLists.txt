# ============================================================================
# renAIssance Framework Utils Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "utils tests must be included from the parent project")
endif()

message(STATUS "=== Utils Tests Configuration ===")

# ============================================================================
# Profiler测试
# ============================================================================

add_executable(test_profiler
    test_profiler.cpp
)

target_link_libraries(test_profiler
    renaissance
)

# 设置输出目录到tests/utils子目录
set_target_properties(test_profiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/utils
)

message(STATUS "Profiler test configured: test_profiler")

# ============================================================================
# PythonSession测试
# ============================================================================

if(TR_USE_PYTHON_SESSION)
    add_executable(test_python_session
        test_python_session.cpp
    )

    target_link_libraries(test_python_session
        renaissance
    )

    # 设置输出目录到tests/utils子目录
    set_target_properties(test_python_session PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/utils
    )

    message(STATUS "PythonSession test configured: test_python_session")
else()
    message(STATUS "PythonSession test skipped (not in PC_CUDA/CPU_CLOUD/GPU_CLOUD scene)")
endif()

message(STATUS "=== Utils Tests Configuration Complete ===")
