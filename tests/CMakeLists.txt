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

# 为所有单元测试添加CUDA/MUSA头文件路径（因为renaissance.h可能包含它们）
if(TR_USE_CUDA)
    if(DEFINED TR_CUDA_PATH)
        include_directories(${TR_CUDA_PATH}/include)
    endif()
    if(DEFINED TR_CUDNN_INCLUDE_DIR AND NOT TR_CUDNN_INCLUDE_DIR STREQUAL "")
        include_directories(${TR_CUDNN_INCLUDE_DIR})
        file(GLOB CUDNN_VERSION_DIRS "${TR_CUDNN_INCLUDE_DIR}/*")
        foreach(CUDNN_VER_DIR ${CUDNN_VERSION_DIRS})
            if(IS_DIRECTORY ${CUDNN_VER_DIR})
                include_directories(${CUDNN_VER_DIR})
            endif()
        endforeach()
    endif()
endif()

if(TR_USE_MUSA)
    if(DEFINED TR_MUSA_PATH)
        include_directories(${TR_MUSA_PATH}/include)
    endif()
    if(DEFINED TR_MUDNN_INCLUDE_DIR)
        include_directories(${TR_MUDNN_INCLUDE_DIR})
        file(GLOB MUDNN_VERSION_DIRS "${TR_MUDNN_INCLUDE_DIR}/*")
        foreach(MUDNN_VER_DIR ${MUDNN_VERSION_DIRS})
            if(IS_DIRECTORY ${MUDNN_VER_DIR})
                include_directories(${MUDNN_VER_DIR})
            endif()
        endforeach()
    endif()
endif()

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
# 包含data子目录（Shape类测试）
# ============================================================================

# 注意：data测试当前已开启。如果需要暂时关闭，请注释掉下面第3-4行
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)
    add_subdirectory(data)
    message(STATUS "Including data tests")
else()
    message(STATUS "Data tests directory not found")
endif()

# ============================================================================
# 包含device子目录（Device类测试）
# ============================================================================

# 注意：device测试当前已开启
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/device AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/device)
    add_subdirectory(device)
    message(STATUS "Including device tests")
else()
    message(STATUS "Device tests directory not found")
endif()

# ============================================================================
# 包含utils子目录（PythonSession测试）
# ============================================================================

# 注意：utils测试当前已开启
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/utils AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utils)
    add_subdirectory(utils)
    message(STATUS "Including utils tests")
else()
    message(STATUS "Utils tests directory not found")
endif()

# ============================================================================
# 包含integration子目录（集成测试）
# ============================================================================

# 注意：integration测试当前已开启
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/integration AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/integration)
    add_subdirectory(integration)
    message(STATUS "Including integration tests")
else()
    message(STATUS "Integration tests directory not found")
endif()

# ============================================================================
# 输出单元测试信息
# ============================================================================

message(STATUS "=== Unit Tests Configuration Complete ===")