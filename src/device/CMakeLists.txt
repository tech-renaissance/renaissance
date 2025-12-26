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

    # MUSA kernels文件 - 使用MUSA编译器单独编译
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/musa_kernels.cu)
        # 查找MUSA
        if(DEFINED TR_MUSA_PATH)
            set(MUSA_ROOT ${TR_MUSA_PATH})
        elseif(DEFINED ENV{MUSA_ROOT})
            set(MUSA_ROOT $ENV{MUSA_ROOT})
        elseif(EXISTS /usr/local/musa)
            set(MUSA_ROOT /usr/local/musa)
        endif()

        if(MUSA_ROOT)
            # 添加MUSA CMake模块路径
            list(APPEND CMAKE_MODULE_PATH ${MUSA_ROOT}/cmake)
            find_package(MUSA REQUIRED)

            # 创建独立的MUSA kernels静态库
            # 注意：所有include路径必须通过OPTIONS传递，使用-Xclang -I路径
            musa_add_library(musa_kernels STATIC musa_kernels.cu
                OPTIONS
                -Xclang -I${CMAKE_SOURCE_DIR}/include
                -Xclang -I${TR_MUSA_PATH}/include
                -Xclang -I/usr/include/c++/13
                -Xclang -I/usr/include/x86_64-linux-gnu/c++/13
            )

            message(STATUS "Device library: MUSA kernels (musa_kernels.cu) compiled as separate library")

            # 将musa_kernels库添加到device库的链接依赖中
            # 注意：不能添加到DEVICE_SOURCES，因为这是一个已编译的库，不是源文件
            set(MUSA_KERNELS_LINK_LIB musa_kernels)
        else()
            message(WARNING "MUSA_ROOT not found, skipping musa_kernels.cu")
        endif()
    endif()
endif()

# 创建Device库
add_library(device STATIC ${DEVICE_SOURCES})

# 为device库设置编译选项，区分C++和CUDA
# 参考 INFO7.md：使用生成器表达式避免将C++选项传给nvcc
if(MSVC)
    target_compile_options(device PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# 添加头文件路径
target_include_directories(device PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 添加外部库头文件路径（仅用于编译，不链接）
# mimalloc
if(TR_USE_MIMALLOC AND DEFINED TR_MIMALLOC_INCLUDE_DIR)
    target_include_directories(device PRIVATE ${TR_MIMALLOC_INCLUDE_DIR})
endif()

# CUDA
if(TR_USE_CUDA AND DEFINED TR_CUDA_PATH)
    target_include_directories(device PRIVATE ${TR_CUDA_PATH}/include)
    # cuDNN头文件（用于CudaDevice）
    if(DEFINED CUDNN_INCLUDE_DIR)
        target_include_directories(device PRIVATE ${CUDNN_INCLUDE_DIR})
        # cuDNN 9.x版本子目录
        file(GLOB CUDNN_VERSION_DIRS "${CUDNN_INCLUDE_DIR}/*")
        foreach(CUDNN_VER_DIR ${CUDNN_VERSION_DIRS})
            if(IS_DIRECTORY ${CUDNN_VER_DIR})
                target_include_directories(device PRIVATE ${CUDNN_VER_DIR})
            endif()
        endforeach()
    endif()
endif()

# MUSA
if(TR_USE_MUSA AND DEFINED TR_MUSA_PATH)
    target_include_directories(device PRIVATE ${TR_MUSA_PATH}/include)
    message(STATUS "Device library: MUSA headers from ${TR_MUSA_PATH}/include")

    # muDNN头文件（用于MusaDevice）
    # 注意：muDNN头文件通常与MUSA Runtime在同一目录
    if(DEFINED TR_MUDNN_INCLUDE_DIR)
        target_include_directories(device PRIVATE ${TR_MUDNN_INCLUDE_DIR})
        message(STATUS "Device library: muDNN headers from ${TR_MUDNN_INCLUDE_DIR}")
    else()
        # 如果没有单独定义TR_MUDNN_INCLUDE_DIR，使用MUSA路径（muDNN通常与MUSA在一起）
        message(STATUS "Device library: muDNN headers assumed to be in ${TR_MUSA_PATH}/include")
    endif()

    # 链接MUSA kernels库（如果已创建）
    if(DEFINED MUSA_KERNELS_LINK_LIB)
        target_link_libraries(device PRIVATE ${MUSA_KERNELS_LINK_LIB})
        message(STATUS "Device library: Linked MUSA kernels library (${MUSA_KERNELS_LINK_LIB})")
    endif()
endif()


# 设置编译选项
target_compile_features(device PUBLIC cxx_std_17)

# 链接base和data模块（device依赖这两个模块）
target_link_libraries(device PUBLIC base data)

message(STATUS "Device module configured: ${DEVICE_SOURCES}")
