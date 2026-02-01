
# 收集数据源文件
set(BASE_SOURCES
    logger.cpp
    tr_exception.cpp
    device_type.cpp
    dtype.cpp
    memory_arena.cpp
    memory_plan.cpp
    cpu_arena.cpp
    rng.cpp
    downloader.cpp
)

# 添加CUDA/MUSA源文件（条件编译）
if(TR_USE_CUDA)
    list(APPEND BASE_SOURCES cuda_arena.cpp)
    message(STATUS "Base library: Adding cuda_arena.cpp (CUDA enabled)")
endif()

if(TR_USE_MUSA)
    list(APPEND BASE_SOURCES musa_arena.cpp)
    message(STATUS "Base library: Adding musa_arena.cpp (MUSA enabled)")
endif()

# 创建数据库静态库
add_library(base STATIC ${BASE_SOURCES})

# 包含头文件目录
target_include_directories(base
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

# 添加mimalloc支持（cpu_arena.cpp需要）
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_INCLUDE_DIR)
        target_include_directories(base PRIVATE ${TR_MIMALLOC_INCLUDE_DIR})
        message(STATUS "Base library: Using mimalloc headers from ${TR_MIMALLOC_INCLUDE_DIR}")
    endif()
endif()

# 添加libcurl支持（downloader.cpp需要）
if(TR_USE_LIBCURL)
    if(DEFINED TR_CURL_INCLUDE_DIR)
        target_include_directories(base PRIVATE ${TR_CURL_INCLUDE_DIR})
        message(STATUS "Base library: Using libcurl headers from ${TR_CURL_INCLUDE_DIR}")
    endif()
endif()

# 添加CUDA支持（cuda_arena.cpp需要）
if(TR_USE_CUDA)
    if(DEFINED TR_CUDA_PATH)
        # CUDA头文件目录
        target_include_directories(base PRIVATE ${TR_CUDA_PATH}/include)
        message(STATUS "Base library: Using CUDA headers from ${TR_CUDA_PATH}/include")
    endif()
endif()

# 添加MUSA支持（musa_arena.cpp需要）
# 使用cmake_paths.cmake中定义的TR_MUSA_PATH宏
if(TR_USE_MUSA)
    if(DEFINED TR_MUSA_PATH)
        # MUSA头文件目录
        target_include_directories(base PRIVATE ${TR_MUSA_PATH}/include)
        message(STATUS "Base library: Using MUSA headers from ${TR_MUSA_PATH}/include")
    endif()
endif()

# 设置编译选项
target_compile_definitions(base PRIVATE
    TR_DATA_VERSION_MAJOR=3
    TR_DATA_VERSION_MINOR=6
    TR_DATA_VERSION_PATCH=0
)

# 设置C++标准
set_target_properties(base PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 设置属性
set_target_properties(base PROPERTIES
    FOLDER "src"
    POSITION_INDEPENDENT_CODE ON
)
