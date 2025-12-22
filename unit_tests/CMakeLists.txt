# ============================================================================
# renAIssance Framework Unit Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "unit_tests must be included from the parent project")
endif()

# 设置单元测试的输出目录
set(UNIT_TESTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unit_tests)

message(STATUS "=== Unit Tests Configuration ===")
message(STATUS "Unit tests will be built in: ${UNIT_TESTS_OUTPUT_DIR}")

# ============================================================================
# 包含dependency子目录
# ============================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dependency AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dependency)
    # 包含dependency目录，该目录有自己完整的CMakeLists.txt
    add_subdirectory(dependency)
    message(STATUS "Including dependency tests")
else()
    message(STATUS "Dependency tests directory not found")
endif()

# ============================================================================
# 输出单元测试信息
# ============================================================================

message(STATUS "=== Unit Tests Configuration Complete ===")