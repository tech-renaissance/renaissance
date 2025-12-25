# ============================================================================
# Device Component Unit Tests
# ============================================================================

message(STATUS "Configuring device component unit tests")

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

# 定义源文件（需要包含base、data、device的所有源文件）
set(DEVICE_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/base/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/base/tr_exception.cpp
    ${CMAKE_SOURCE_DIR}/src/base/memory_arena.cpp
    ${CMAKE_SOURCE_DIR}/src/base/memory_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/base/cpu_arena.cpp
    ${CMAKE_SOURCE_DIR}/src/data/shape.cpp
    ${CMAKE_SOURCE_DIR}/src/data/storage.cpp
    ${CMAKE_SOURCE_DIR}/src/data/tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/device/device.cpp
    ${CMAKE_SOURCE_DIR}/src/device/device_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/device/cpu_device.cpp
)

# 定义头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# 添加mimalloc支持
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_INCLUDE_DIR)
        include_directories(${TR_MIMALLOC_INCLUDE_DIR})
        message(STATUS "Device tests: Using mimalloc headers from ${TR_MIMALLOC_INCLUDE_DIR}")
    endif()
endif()

# 创建CpuDevice测试可执行文件
add_executable(test_cpu_device
    test_cpu_device.cpp
    ${DEVICE_TEST_SOURCES}
)

# 为测试目标单独设置头文件路径（确保#include "renaissance/renaissance.h"能找到）
target_include_directories(test_cpu_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 设置输出目录
set_target_properties(test_cpu_device PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/device
)

# Windows下为控制台程序设置子系统
if(WIN32)
    set_target_properties(test_cpu_device PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

# 编译选项
if(MSVC)
    target_compile_options(test_cpu_device PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_cpu_device PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 设置日志级别
# 注意：TR_WORKSPACE已在主CMakeLists.txt中全局定义，此处无需重复传递
target_compile_definitions(test_cpu_device PRIVATE TR_LOG_LEVEL=0)

# 链接mimalloc库
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_LIBRARY_DIR)
        if(WIN32)
            # Windows: 使用完整路径
            target_link_libraries(test_cpu_device PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
        else()
            # Linux/其他: 指定库搜索路径 + 库名
            target_link_directories(test_cpu_device PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_libraries(test_cpu_device PRIVATE mimalloc)
        endif()
        message(STATUS "Device tests: Linking mimalloc from ${TR_MIMALLOC_LIBRARY_DIR}")
    endif()
endif()

message(STATUS "TR_WORKSPACE propagated to device tests: ${TR_WORKSPACE}")
message(STATUS "Device component unit tests configured successfully")
