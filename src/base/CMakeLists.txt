# ============================================================================
# renAIssance Framework Base Components CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "src/base must be included from the parent project")
endif()

# 设置基础组件的输出目录
set(BASE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# Framework Core Library
# ============================================================================

# 创建框架核心静态库
add_library(renaissance_base STATIC
    src/base/hello.cpp
)

# 设置包含目录
target_include_directories(renaissance_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 设置输出目录
set_target_properties(renaissance_base PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${BASE_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${BASE_OUTPUT_DIR}
)

# 添加编译定义
target_compile_definitions(renaissance_base PUBLIC
    RENAISSANCE_VERSION_MAJOR=${RENAISSANCE_VERSION_MAJOR}
    RENAISSANCE_VERSION_MINOR=${RENAISSANCE_VERSION_MINOR}
    RENAISSANCE_VERSION_PATCH=${RENAISSANCE_VERSION_PATCH}
)

# 根据CUDA支持添加定义
if(TR_USE_CUDA)
    target_compile_definitions(renaissance_base PUBLIC TR_USE_CUDA)
endif()

if(TR_USE_CUDNN)
    target_compile_definitions(renaissance_base PUBLIC TR_USE_CUDNN)
endif()

if(TR_USE_OPENMP)
    target_compile_definitions(renaissance_base PUBLIC TR_USE_OPENMP)
endif()

# 为MSVC编译器添加特定设置
if(MSVC)
    target_compile_options(renaissance_base PRIVATE /W3)
else()
    target_compile_options(renaissance_base PRIVATE -Wall -Wextra)
endif()

# OpenMP链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(renaissance_base PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Framework Export Configuration
# ============================================================================

# 导出库，供其他目标使用
set(RENAISSANCE_BASE_LIBRARY renaissance_base PARENT_SCOPE)

message(STATUS "Framework base library 'renaissance_base' configured")
message(STATUS "Base library output directory: ${BASE_OUTPUT_DIR}")