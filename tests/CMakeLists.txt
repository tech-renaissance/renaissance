# ============================================================================
# renAIssance Framework Unit Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "tests must be included from the parent project")
endif()

# ============================================================================
# 确保workspace目录存在
# ============================================================================

if(DEFINED TR_WORKSPACE)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create workspace directory: ${TR_WORKSPACE}")
    else()
        message(STATUS "Workspace directory ensured: ${TR_WORKSPACE}")
    endif()
else()
    # 如果未定义TR_WORKSPACE，使用默认路径
    set(TR_WORKSPACE "${CMAKE_SOURCE_DIR}/workspace")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TR_WORKSPACE}
                    RESULT_VARIABLE WORKSPACE_RESULT)
    if(NOT WORKSPACE_RESULT EQUAL 0)
        message(WARNING "Failed to create default workspace directory: ${TR_WORKSPACE}")
    else()
        message(STATUS "Default workspace directory created: ${TR_WORKSPACE}")
    endif()
endif()

# 设置单元测试的输出目录
set(TESTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests)

message(STATUS "=== Unit Tests Configuration ===")
message(STATUS "Unit tests will be built in: ${TESTS_OUTPUT_DIR}")

# ============================================================================
# 包含dependency子目录（依赖库Hello World测试）
# ============================================================================

# 注意：dependency测试（hello_cuda, hello_xnnpack等）已被暂时关闭
# 如果需要重新开启dependency测试，请取消下面第4-6行的注释
#
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dependency AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dependency)
#     add_subdirectory(dependency)
#     message(STATUS "Including dependency tests")
# endif()

message(STATUS "Dependency tests are temporarily disabled (see tests/CMakeLists.txt:40-46)")

# ============================================================================
# 包含base子目录（Logger和TRException测试）
# ============================================================================

# 注意：base测试当前已开启。如果需要暂时关闭，请注释掉下面第3-4行
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/base AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/base)
    add_subdirectory(base)
    message(STATUS "Including base tests")
else()
    message(STATUS "Base tests directory not found")
endif()

# ============================================================================
# 输出单元测试信息
# ============================================================================

message(STATUS "=== Unit Tests Configuration Complete ===")