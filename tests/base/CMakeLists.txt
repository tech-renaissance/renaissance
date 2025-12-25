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
    ${CMAKE_SOURCE_DIR}/src/base/memory_arena.cpp
    ${CMAKE_SOURCE_DIR}/src/base/memory_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/base/cpu_arena.cpp
)

# 定义头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# 添加mimalloc支持
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_INCLUDE_DIR)
        include_directories(${TR_MIMALLOC_INCLUDE_DIR})
        message(STATUS "Base tests: Using mimalloc headers from ${TR_MIMALLOC_INCLUDE_DIR}")
    endif()
endif()

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

# 创建内存池测试可执行文件
add_executable(test_memory_alignment
    test_memory_alignment.cpp
    ${BASE_TEST_SOURCES}
)

add_executable(test_arena
    test_arena.cpp
    ${BASE_TEST_SOURCES}
)

add_executable(test_cpu_arena
    test_cpu_arena.cpp
    ${BASE_TEST_SOURCES}
)

add_executable(test_memory_plan
    test_memory_plan.cpp
    ${BASE_TEST_SOURCES}
)

# 创建CUDA Arena测试可执行文件
if(TR_USE_CUDA)
    # CUDA启用时：包含完整源文件（包括CpuArena、CudaArena等）
    add_executable(test_cuda_arena
        test_cuda_arena.cpp
        ${BASE_TEST_SOURCES}
        ${CMAKE_SOURCE_DIR}/src/base/cuda_arena.cpp
    )
    set_target_properties(test_cuda_arena PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
    )
    if(MSVC)
        set_target_properties(test_cuda_arena PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
        target_compile_options(test_cuda_arena PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_arena PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    # 关键：定义TR_USE_CUDA宏，使test_cuda_arena.cpp中的#ifdef TR_USE_CUDA生效
    target_compile_definitions(test_cuda_arena PRIVATE TR_LOG_LEVEL=0)
    target_compile_definitions(test_cuda_arena PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
    target_compile_definitions(test_cuda_arena PRIVATE TR_USE_CUDA)

    # 添加CUDA头文件路径
    if(DEFINED TR_CUDA_PATH)
        target_include_directories(test_cuda_arena PRIVATE ${TR_CUDA_PATH}/include)
    endif()
    if(DEFINED TR_CUDNN_INCLUDE_DIR)
        target_include_directories(test_cuda_arena PRIVATE ${TR_CUDNN_INCLUDE_DIR})
    endif()

    # 链接CUDA库
    if(WIN32)
        target_link_libraries(test_cuda_arena PRIVATE
            ${CUDA_cudart_LIBRARY}
        )
    else()
        target_link_libraries(test_cuda_arena PRIVATE
            ${CUDA_cudart_LIBRARY}
        )
    endif()

    message(STATUS "CudaArena test: CUDA support enabled")
else()
    # CUDA未启用时：只包含必要的源文件（不包括CpuArena等）
    add_executable(test_cuda_arena
        test_cuda_arena.cpp
        ${CMAKE_SOURCE_DIR}/src/base/logger.cpp
        ${CMAKE_SOURCE_DIR}/src/base/tr_exception.cpp
    )
    set_target_properties(test_cuda_arena PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
    )
    if(MSVC)
        set_target_properties(test_cuda_arena PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
        target_compile_options(test_cuda_arena PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_cuda_arena PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_compile_definitions(test_cuda_arena PRIVATE TR_LOG_LEVEL=0)
    target_compile_definitions(test_cuda_arena PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
    message(STATUS "CudaArena test: CUDA support disabled (building stub version)")
endif()

# 设置输出目录
set_target_properties(test_logger PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/base
)

set_target_properties(test_exception PROPERTIES
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

# Windows下为控制台程序设置子系统
if(WIN32)
    set_target_properties(test_logger PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    set_target_properties(test_exception PROPERTIES
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
endif()

# 编译选项
if(MSVC)
    target_compile_options(test_logger PRIVATE /W4 /utf-8)
    target_compile_options(test_exception PRIVATE /W4 /utf-8)
    target_compile_options(test_memory_alignment PRIVATE /W4 /utf-8)
    target_compile_options(test_arena PRIVATE /W4 /utf-8)
    target_compile_options(test_cpu_arena PRIVATE /W4 /utf-8)
    target_compile_options(test_memory_plan PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_logger PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_exception PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_memory_alignment PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_arena PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_cpu_arena PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_memory_plan PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 设置日志级别（Debug模式全开，Release模式只保留WARN/ERROR）
target_compile_definitions(test_logger PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_exception PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_memory_alignment PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_arena PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_cpu_arena PRIVATE TR_LOG_LEVEL=0)
target_compile_definitions(test_memory_plan PRIVATE TR_LOG_LEVEL=0)

# 传递TR_WORKSPACE宏到测试代码
target_compile_definitions(test_logger PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
target_compile_definitions(test_exception PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
target_compile_definitions(test_memory_alignment PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
target_compile_definitions(test_arena PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
target_compile_definitions(test_cpu_arena PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")
target_compile_definitions(test_memory_plan PRIVATE TR_WORKSPACE="${TR_WORKSPACE}")

# 链接mimalloc库
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_LIBRARY_DIR)
        if(WIN32)
            # Windows: 使用完整路径
            target_link_libraries(test_logger PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
            target_link_libraries(test_exception PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
            target_link_libraries(test_memory_alignment PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
            target_link_libraries(test_arena PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
            target_link_libraries(test_cpu_arena PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
            target_link_libraries(test_memory_plan PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
            target_link_libraries(test_cuda_arena PRIVATE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
        else()
            # Linux/其他: 指定库搜索路径 + 库名
            target_link_directories(test_logger PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_directories(test_exception PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_directories(test_memory_alignment PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_directories(test_arena PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_directories(test_cpu_arena PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_directories(test_memory_plan PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_directories(test_cuda_arena PRIVATE ${TR_MIMALLOC_LIBRARY_DIR})

            target_link_libraries(test_logger PRIVATE mimalloc)
            target_link_libraries(test_exception PRIVATE mimalloc)
            target_link_libraries(test_memory_alignment PRIVATE mimalloc)
            target_link_libraries(test_arena PRIVATE mimalloc)
            target_link_libraries(test_cpu_arena PRIVATE mimalloc)
            target_link_libraries(test_memory_plan PRIVATE mimalloc)
            target_link_libraries(test_cuda_arena PRIVATE mimalloc)
        endif()
        message(STATUS "Base tests: Linking mimalloc from ${TR_MIMALLOC_LIBRARY_DIR}")
    endif()
endif()

message(STATUS "TR_WORKSPACE propagated to base tests: ${TR_WORKSPACE}")

message(STATUS "Base component unit tests configured successfully")
