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

# 定义头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# 创建Logger测试可执行文件
add_executable(test_logger test_logger.cpp)

# 创建Exception测试可执行文件
add_executable(test_exception test_exception.cpp)

# 创建Terminate Handler测试可执行文件
add_executable(test_terminate_handler test_terminate_handler.cpp)

# 创建内存池测试可执行文件
add_executable(test_memory_alignment test_memory_alignment.cpp)

add_executable(test_arena test_arena.cpp)

add_executable(test_cpu_arena test_cpu_arena.cpp)

add_executable(test_memory_plan test_memory_plan.cpp)

# 创建GlobalRegistry测试可执行文件
add_executable(test_global_registry test_global_registry.cpp)

# 创建Downloader测试可执行文件（需要libcurl）
if(TR_USE_LIBCURL)
    add_executable(test_downloader test_downloader.cpp)
    message(STATUS "Base test: test_downloader added (libcurl enabled)")
endif()

# CUDA Arena测试（条件编译）
if(TR_USE_CUDA)
    add_executable(test_cuda_arena test_cuda_arena.cpp)
endif()

# MUSA Arena测试（条件编译）
if(TR_USE_MUSA)
    add_executable(test_musa_arena test_musa_arena.cpp)
endif()

# 设置输出目录
set_target_properties(test_logger PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_exception PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_terminate_handler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_memory_alignment PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_arena PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_cpu_arena PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_memory_plan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_global_registry PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

if(TR_USE_LIBCURL)
    set_target_properties(test_downloader PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
    )
endif()

if(TR_USE_CUDA)
    set_target_properties(test_cuda_arena PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
    )
endif()

if(TR_USE_MUSA)
    set_target_properties(test_musa_arena PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
    )
endif()

# Windows下为控制台程序设置子系统
if(WIN32)
    set_target_properties(test_logger PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_exception PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_terminate_handler PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_memory_alignment PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_arena PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_cpu_arena PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_memory_plan PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_global_registry PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    if(TR_USE_LIBCURL)
        set_target_properties(test_downloader PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()
    if(TR_USE_CUDA)
        set_target_properties(test_cuda_arena PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()
    if(TR_USE_MUSA)
        set_target_properties(test_musa_arena PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
    endif()
endif()

# 编译选项
if(MSVC)
    target_compile_options(test_logger PRIVATE /W4 /utf-8)
    target_compile_options(test_exception PRIVATE /W4 /utf-8)
    target_compile_options(test_terminate_handler PRIVATE /W4 /utf-8)
    target_compile_options(test_memory_alignment PRIVATE /W4 /utf-8)
    target_compile_options(test_arena PRIVATE /W4 /utf-8)
    target_compile_options(test_cpu_arena PRIVATE /W4 /utf-8)
    target_compile_options(test_memory_plan PRIVATE /W4 /utf-8)
    target_compile_options(test_global_registry PRIVATE /W4 /utf-8)
    if(TR_USE_LIBCURL)
        target_compile_options(test_downloader PRIVATE /W4 /utf-8)
    endif()
    if(TR_USE_CUDA)
        target_compile_options(test_cuda_arena PRIVATE /W4 /utf-8)
    endif()
    if(TR_USE_MUSA)
        target_compile_options(test_musa_arena PRIVATE /W4 /utf-8)
    endif()
else()
    target_compile_options(test_logger PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_exception PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_memory_alignment PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_arena PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_cpu_arena PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_memory_plan PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_global_registry PRIVATE -Wall -Wextra -Wpedantic)
    if(TR_USE_LIBCURL)
        target_compile_options(test_downloader PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(TR_USE_CUDA)
        target_compile_options(test_cuda_arena PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(TR_USE_MUSA)
        target_compile_options(test_musa_arena PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# 设置日志级别（Debug模式全开，Release模式只保留WARN/ERROR）
# 注意：TR_WORKSPACE已在主CMakeLists.txt中全局定义，此处无需重复传递
target_compile_definitions(test_logger PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_exception PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_terminate_handler PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_memory_alignment PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_arena PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_cpu_arena PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_memory_plan PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_global_registry PRIVATE TR_LOG_LEVEL=0)
if(TR_USE_LIBCURL)
    target_compile_definitions(test_downloader PRIVATE TR_LOG_LEVEL=0)
endif()
if(TR_USE_CUDA)
    target_compile_definitions(test_cuda_arena PRIVATE TR_LOG_LEVEL=0)
endif()
if(TR_USE_MUSA)
    target_compile_definitions(test_musa_arena PRIVATE TR_LOG_LEVEL=0)
endif()

# 链接renaissance统一库（包含base、data、device及所有依赖）
target_link_libraries(test_logger PRIVATE renaissance)
target_link_libraries(test_exception PRIVATE renaissance)
target_link_libraries(test_terminate_handler PRIVATE renaissance)
target_link_libraries(test_memory_alignment PRIVATE renaissance)
target_link_libraries(test_arena PRIVATE renaissance)
target_link_libraries(test_cpu_arena PRIVATE renaissance)
target_link_libraries(test_memory_plan PRIVATE renaissance)
target_link_libraries(test_global_registry PRIVATE renaissance)
if(TR_USE_LIBCURL)
    target_link_libraries(test_downloader PRIVATE renaissance)
endif()
if(TR_USE_CUDA)
    target_link_libraries(test_cuda_arena PRIVATE renaissance)
endif()
if(TR_USE_MUSA)
    target_link_libraries(test_musa_arena PRIVATE renaissance)
endif()

message(STATUS "TR_WORKSPACE propagated to base tests: ${TR_WORKSPACE}")

message(STATUS "Base component unit tests configured successfully")
