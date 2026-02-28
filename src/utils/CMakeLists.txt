# ============================================================================
# renAIssance Framework Utils Library CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "utils must be included from the parent project")
endif()

message(STATUS "=== Utils Library Configuration ===")

# 创建utils库
add_library(utils STATIC
    profiler.cpp
    initializer.cpp
)

# 添加头文件路径
target_include_directories(utils PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 链接依赖（按照base、data、device的顺序）
target_link_libraries(utils PUBLIC
    base
    data
    device
)

# 根据TR_USE_PYTHON_SESSION宏决定是否添加python_session.cpp
if(TR_USE_PYTHON_SESSION)
    target_sources(utils PRIVATE
        python_session.cpp
    )
    message(STATUS "Utils: PythonSession support enabled")
else()
    message(STATUS "Utils: PythonSession support disabled (not in PC_CUDA/CPU_CLOUD/GPU_CLOUD scene)")
endif()

message(STATUS "Utils library configured")
message(STATUS "=== Utils Library Configuration Complete ===")
