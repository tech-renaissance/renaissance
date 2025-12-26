# Device模块源文件
set(DEVICE_SOURCES
    device.cpp
    device_manager.cpp
    cpu_device.cpp
)

# CUDA支持
if(TR_USE_CUDA)
    list(APPEND DEVICE_SOURCES cuda_device.cpp)

    # CUDA kernels文件（.cu会被CMake自动识别并使用nvcc编译）
    # 参考 INFO7.md：CMake会自动处理.cu文件的编译
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu)
        list(APPEND DEVICE_SOURCES cuda_kernels.cu)
        message(STATUS "Device library: CUDA kernels (cuda_kernels.cu) added to device library")
    endif()
endif()

# MUSA支持
if(TR_USE_MUSA)
    list(APPEND DEVICE_SOURCES musa_device.cpp)
endif()

# 创建Device库
add_library(device STATIC ${DEVICE_SOURCES})

# 为device库设置编译选项，区分C++和CUDA
# 参考 INFO7.md：使用生成器表达式避免将C++选项传给nvcc
target_compile_options(device PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# 添加头文件路径
target_include_directories(device PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 添加外部库头文件路径（仅用于编译，不链接）
if(TR_USE_MIMALLOC AND DEFINED TR_MIMALLOC_INCLUDE_DIR)
    target_include_directories(device PRIVATE ${TR_MIMALLOC_INCLUDE_DIR})
endif()

if(TR_USE_CUDA)
    if(DEFINED TR_CUDA_PATH)
        target_include_directories(device PRIVATE ${TR_CUDA_PATH}/include)
    endif()
    if(DEFINED TR_CUDNN_INCLUDE_DIR)
        target_include_directories(device PRIVATE ${TR_CUDNN_INCLUDE_DIR})
        file(GLOB CUDNN_VERSION_DIRS "${TR_CUDNN_INCLUDE_DIR}/*")
        foreach(CUDNN_VER_DIR ${CUDNN_VERSION_DIRS})
            if(IS_DIRECTORY ${CUDNN_VER_DIR})
                target_include_directories(device PRIVATE ${CUDNN_VER_DIR})
            endif()
        endforeach()
    endif()
endif()

if(TR_USE_MUSA)
    if(DEFINED TR_MUSA_PATH)
        target_include_directories(device PRIVATE ${TR_MUSA_PATH}/include)
    endif()
    if(DEFINED TR_MUDNN_INCLUDE_DIR)
        target_include_directories(device PRIVATE ${TR_MUDNN_INCLUDE_DIR})
        file(GLOB MUDNN_VERSION_DIRS "${TR_MUDNN_INCLUDE_DIR}/*")
        foreach(MUDNN_VER_DIR ${MUDNN_VERSION_DIRS})
            if(IS_DIRECTORY ${MUDNN_VER_DIR})
                target_include_directories(device PRIVATE ${MUDNN_VER_DIR})
            endif()
        endforeach()
    endif()
endif()


# 设置编译选项
target_compile_features(device PUBLIC cxx_std_17)

message(STATUS "Device module configured: ${DEVICE_SOURCES}")
