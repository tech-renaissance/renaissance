# ============================================================================
# renAIssance Framework Integration Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "integration tests must be included from the parent project")
endif()

message(STATUS "=== Integration Tests Configuration ===")

# ============================================================================
# Python print tensor test
# ============================================================================

if(TR_USE_PYTHON_SESSION)
    add_executable(test_python_print
        test_python_print.cpp
    )

    target_link_libraries(test_python_print
        renaissance
    )

    # 设置输出目录到tests/integration子目录
    set_target_properties(test_python_print PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    )

    add_test(NAME test_python_print COMMAND test_python_print)

    message(STATUS "Python print tensor test configured: test_python_print")
else()
    message(STATUS "Python print tensor test skipped (TR_USE_PYTHON_SESSION not defined)")
endif()

message(STATUS "=== Integration Tests Configuration Complete ===")
