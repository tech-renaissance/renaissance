
# 收集数据源文件
set(DATA_SOURCES
    shape.cpp
    storage.cpp
    tensor.cpp
    preprocessor.cpp
    # V3.8.0 新版DataLoader
    imagenet_loader_dts.cpp
    # V3.9.0 MNIST和CIFAR DTS Loader
    mnist_loader_dts.cpp
    cifar_loader_dts.cpp
    # V4.0.0 ImageNet Raw Loader
    imagenet_loader_raw.cpp
    # V4.0.0 MNIST and CIFAR Raw Loader
    mnist_loader_raw.cpp
    cifar_loader_raw.cpp
    # V3.10.0 SampleLoader
    sample_loader.cpp
    # V3.9.0 公共组件
    file_handle.cpp
)

# 创建数据库静态库
add_library(data STATIC ${DATA_SOURCES})

# 包含头文件目录
target_include_directories(data
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

# 设置编译选项
target_compile_definitions(data PRIVATE
    TR_DATA_VERSION_MAJOR=3
    TR_DATA_VERSION_MINOR=6
    TR_DATA_VERSION_PATCH=0
)

# 链接base模块（data依赖base的tr_exception等）
target_link_libraries(data PUBLIC base)

# 链接zlib（DTS数据加载器的CRC校验需要）
find_package(ZLIB)
if(ZLIB_FOUND)
    target_link_libraries(data PUBLIC ZLIB::ZLIB)
    target_compile_definitions(data PUBLIC TR_USE_ZLIB=1)
    message(STATUS "Data library: zlib linked for DTS CRC verification")
else()
    message(WARNING "zlib not found, DTS CRC verification will be disabled")
    target_compile_definitions(data PUBLIC TR_USE_ZLIB=0)
endif()

# 设置C++标准
set_target_properties(data PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 设置属性
set_target_properties(data PROPERTIES
    FOLDER "src"
    POSITION_INDEPENDENT_CODE ON
)