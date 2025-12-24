
# 收集数据源文件
set(BASE_SOURCES
    logger.cpp
    tr_exception.cpp
    device_type.cpp
    dtype.cpp
)

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