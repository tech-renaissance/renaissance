# Device模块源文件
set(DEVICE_SOURCES
    device.cpp
    device_manager.cpp
    cpu_device.cpp
)

# 创建Device库
add_library(device STATIC ${DEVICE_SOURCES})

# 链接mimalloc（cpu_device.cpp需要）
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_LIBRARY_DIR)
        if(WIN32)
            # Windows: 使用完整路径
            target_link_libraries(device PUBLIC "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
        else()
            # Linux/其他: 指定库搜索路径 + 库名
            target_link_directories(device PUBLIC ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_libraries(device PUBLIC mimalloc)
        endif()
        message(STATUS "Device library: Linking mimalloc from ${TR_MIMALLOC_LIBRARY_DIR}")
    endif()
endif()

# 链接base和data库
target_link_libraries(device PUBLIC
    base
    data
)

target_include_directories(device PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 设置编译选项
target_compile_features(device PUBLIC cxx_std_17)

message(STATUS "Device module configured: ${DEVICE_SOURCES}")
