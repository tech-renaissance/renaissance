# ============================================================================
# Logger and TRException Unit Tests
# ============================================================================

message(STATUS "Configuring base component unit tests")

# 确保workspace目录存在
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

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义源文件
set(BASE_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/base/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/base/tr_exception.cpp
)

# 定义头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# 创建Logger测试可执行文件
add_executable(test_logger
    test_logger.cpp
    ${BASE_TEST_SOURCES}
)

# 创建Exception测试可执行文件
add_executable(test_exception
    test_exception.cpp
    ${BASE_TEST_SOURCES}
)

# 设置输出目录
set_target_properties(test_logger PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_exception PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

# Windows下为控制台程序设置子系统
if(WIN32)
    set_target_properties(test_logger PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_exception PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

# 编译选项
if(MSVC)
    target_compile_options(test_logger PRIVATE /W4 /utf-8)
    target_compile_options(test_exception PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_logger PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_exception PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 设置日志级别（Debug模式全开，Release模式只保留WARN/ERROR）
target_compile_definitions(test_logger PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_exception PRIVATE TR_LOG_LEVEL=0)

# 传递TR_WORKSPACE宏到测试代码
target_compile_definitions(test_logger PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
target_compile_definitions(test_exception PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
message(STATUS "TR_WORKSPACE propagated to base tests: ${TR_WORKSPACE}")

message(STATUS "Base component unit tests configured successfully")
