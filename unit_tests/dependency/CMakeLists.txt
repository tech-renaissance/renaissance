# ============================================================================
# renAIssance Framework - Dependencies Hello World Tests
# ============================================================================

message(STATUS "=== Configuring Dependencies Hello World Tests ===")

# ============================================================================
# 设置输出目录
# ============================================================================

# 输出目录设置为build目录下的bin/dependency
set(DEPENDENCY_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/dependency)

# ============================================================================
# Hello World测试源文件配置
# ============================================================================

# 聚焦CUDA测试 - 只构建hello_cuda.cpp
set(HELLO_WORLD_SOURCES
    hello_cuda.cpp
)

set(HELLO_WORLD_DEPS
    CUDA
)

set(HELLO_WORLD_MACROS
    TR_USE_CUDA
)

# 收集需要构建的源文件
set(BUILD_SOURCES)
set(BUILD_NAMES)

list(LENGTH HELLO_WORLD_SOURCES SOURCES_LEN)
foreach(idx RANGE 0 ${SOURCES_LEN})
    if(idx EQUAL ${SOURCES_LEN})
        break()
    endif()

    list(GET HELLO_WORLD_SOURCES ${idx} SOURCE_FILE)
    list(GET HELLO_WORLD_DEPS ${idx} DEP_NAME)
    list(GET HELLO_WORLD_MACROS ${idx} REQUIRED_MACRO)

    # 检查宏是否定义
    if(${REQUIRED_MACRO})
        list(APPEND BUILD_SOURCES "${SOURCE_FILE}")
        list(APPEND BUILD_NAMES "${DEP_NAME}")
        message(STATUS "Will build ${DEP_NAME} test: ${SOURCE_FILE}")
    else()
        message(STATUS "Skipping ${DEP_NAME} test: ${REQUIRED_MACRO} is OFF")
    endif()
endforeach()

# ============================================================================
# 使用vcpkg自动查找依赖库
# ============================================================================

# oneDNN (dnnl) - 使用生成的精确路径
if(TR_USE_ONEDNN)
    if(DEFINED TR_ONEDNN_INCLUDE_DIR AND DEFINED TR_ONEDNN_LIBRARY_DIR)
        set(oneDNN_FOUND TRUE)
        set(oneDNN_INCLUDE_DIRS ${TR_ONEDNN_INCLUDE_DIR})
        set(oneDNN_LIBRARIES dnnl)
        set(oneDNN_LIBRARY_DIRS ${TR_ONEDNN_LIBRARY_DIR})
        message(STATUS "Found oneDNN: ${TR_ONEDNN_INCLUDE_DIR}")
    else()
        set(oneDNN_FOUND FALSE)
        message(WARNING "TR_ONEDNN paths not defined")
    endif()
endif()

# XNNPACK - 使用生成的精确路径
if(TR_USE_XNNPACK)
    if(DEFINED TR_XNNPACK_INCLUDE_DIR AND DEFINED TR_XNNPACK_LIBRARY_DIR)
        set(XNNPACK_FOUND TRUE)
        set(XNNPACK_INCLUDE_DIRS ${TR_XNNPACK_INCLUDE_DIR})
        set(XNNPACK_LIBRARIES XNNPACK)
        set(XNNPACK_LIBRARY_DIRS ${TR_XNNPACK_LIBRARY_DIR})
        message(STATUS "Found XNNPACK: ${TR_XNNPACK_INCLUDE_DIR}")
    else()
        set(XNNPACK_FOUND FALSE)
        message(WARNING "TR_XNNPACK paths not defined")
    endif()
endif()

# mimalloc - 使用生成的精确路径
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_INCLUDE_DIR AND DEFINED TR_MIMALLOC_LIBRARY_DIR)
        set(mimalloc_FOUND TRUE)
        set(mimalloc_INCLUDE_DIRS ${TR_MIMALLOC_INCLUDE_DIR})
        set(mimalloc_LIBRARIES mimalloc)
        set(mimalloc_LIBRARY_DIRS ${TR_MIMALLOC_LIBRARY_DIR})
        message(STATUS "Found mimalloc: ${TR_MIMALLOC_INCLUDE_DIR}")
    else()
        set(mimalloc_FOUND FALSE)
        message(WARNING "TR_MIMALLOC paths not defined")
    endif()
endif()

# zlib - 使用生成的精确路径
if(TR_USE_ZLIB)
    if(DEFINED TR_ZLIB_INCLUDE_DIR AND DEFINED TR_ZLIB_LIBRARY_DIR)
        set(ZLIB_FOUND TRUE)
        set(ZLIB_INCLUDE_DIRS ${TR_ZLIB_INCLUDE_DIR})
        set(ZLIB_LIBRARIES zlib)
        set(ZLIB_LIBRARY_DIRS ${TR_ZLIB_LIBRARY_DIR})
        message(STATUS "Found zlib: ${TR_ZLIB_INCLUDE_DIR}")
    else()
        set(ZLIB_FOUND FALSE)
        message(WARNING "TR_ZLIB paths not defined")
    endif()
endif()

# libcurl - 使用生成的精确路径
if(TR_USE_LIBCURL)
    if(DEFINED TR_CURL_INCLUDE_DIR AND DEFINED TR_CURL_LIBRARY_DIR)
        set(CURL_FOUND TRUE)
        set(CURL_INCLUDE_DIRS ${TR_CURL_INCLUDE_DIR})
        set(CURL_LIBRARIES curl)
        set(CURL_LIBRARY_DIRS ${TR_CURL_LIBRARY_DIR})
        message(STATUS "Found libcurl: ${TR_CURL_INCLUDE_DIR}")
    else()
        set(CURL_FOUND FALSE)
        message(WARNING "TR_CURL paths not defined")
    endif()
endif()

# libjpeg-turbo
if(TR_USE_LIBJPEG)
    if(DEFINED TR_JPEG_INCLUDE_DIR AND DEFINED TR_JPEG_LIBRARY_DIR)
        set(JPEG_FOUND TRUE)
        set(JPEG_INCLUDE_DIRS ${TR_JPEG_INCLUDE_DIR})
        set(JPEG_LIBRARIES jpeg)
        set(JPEG_LIBRARY_DIRS ${TR_JPEG_LIBRARY_DIR})
        message(STATUS "Found libjpeg-turbo: ${TR_JPEG_INCLUDE_DIR}")
    else()
        set(JPEG_FOUND FALSE)
        message(WARNING "TR_JPEG paths not defined")
    endif()
endif()

# STB (使用生成的精确路径)
if(TR_USE_STB)
    if(DEFINED TR_STB_INCLUDE_DIR)
        set(STB_INCLUDE_DIR ${TR_STB_INCLUDE_DIR})
        message(STATUS "Found STB headers: ${STB_INCLUDE_DIR}")
    else()
        message(WARNING "TR_STB_INCLUDE_DIR not defined")
    endif()
endif()

# Simd
if(TR_USE_SIMD)
    if(DEFINED TR_SIMD_INCLUDE_DIR AND DEFINED TR_SIMD_LIBRARY_DIR)
        set(SIMD_FOUND TRUE)
        set(SIMD_INCLUDE_DIRS ${TR_SIMD_INCLUDE_DIR})
        set(SIMD_LIBRARIES Simd)
        set(SIMD_LIBRARY_DIRS ${TR_SIMD_LIBRARY_DIR})
        message(STATUS "Found Simd: ${TR_SIMD_INCLUDE_DIR}")
    else()
        set(SIMD_FOUND FALSE)
        message(WARNING "TR_SIMD paths not defined")
    endif()
endif()

# ============================================================================
# 构建Hello World可执行文件
# ============================================================================

list(LENGTH BUILD_SOURCES BUILD_SOURCES_LEN)
foreach(idx RANGE 0 ${BUILD_SOURCES_LEN})
    if(idx EQUAL ${BUILD_SOURCES_LEN})
        break()
    endif()

    list(GET BUILD_SOURCES ${idx} TEST_SOURCE)
    list(GET BUILD_NAMES ${idx} DEP_NAME)

    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    message(STATUS "Creating ${DEP_NAME} test executable: ${TEST_NAME}")

    # 创建可执行文件
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # 设置C++标准
    set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    # 确保使用Release编译模式优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${TEST_NAME} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()

    # MSVC UTF-8编码支持
    if(MSVC)
        target_compile_options(${TEST_NAME} PRIVATE /utf-8)
    endif()

    # 设置输出目录
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${DEPENDENCY_OUTPUT_DIR}
    )

    # 根据测试名称链接相应的库
    if(TEST_NAME MATCHES "cuda")
        if(TR_USE_CUDA)
            # 项目级配置：为测试设置CUDA PATH环境变量（方法二）
            if(WIN32 AND TR_USE_CUDA)
                if(DEFINED TR_CUDA_BIN_DIR AND DEFINED TR_CUDNN_BIN_DIR)
                    # 为可执行文件设置运行时环境变量
                    set_target_properties(${TEST_NAME} PROPERTIES
                        ENVIRONMENT "PATH=${TR_CUDA_BIN_DIR}/x64;${TR_CUDNN_BIN_DIR};$ENV{PATH}"
                    )
                    message(STATUS "  - CUDA PATH added to test environment")
                    message(STATUS "    CUDA Toolkit: ${TR_CUDA_BIN_DIR}/x64")
                    message(STATUS "    cuDNN: ${TR_CUDNN_BIN_DIR}")
                endif()
            endif()
            # 使用配置生成的精确路径
            target_include_directories(${TEST_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../../include
            )

            # 添加cuDNN精确路径（如果已配置）
            if(DEFINED TR_CUDNN_INCLUDE_DIR)
                # cuDNN 9.x有版本子目录，需要同时添加父目录和版本目录
                target_include_directories(${TEST_NAME} PRIVATE ${TR_CUDNN_INCLUDE_DIR})
                # 尝试找到版本子目录
                file(GLOB CUDNN_VERSION_DIRS "${TR_CUDNN_INCLUDE_DIR}/*")
                foreach(CUDNN_VER_DIR ${CUDNN_VERSION_DIRS})
                    if(IS_DIRECTORY ${CUDNN_VER_DIR})
                        target_include_directories(${TEST_NAME} PRIVATE ${CUDNN_VER_DIR})
                        message(STATUS "  - Using cuDNN include path: ${CUDNN_VER_DIR}")
                    endif()
                endforeach()
                message(STATUS "  - Using cuDNN include base path: ${TR_CUDNN_INCLUDE_DIR}")
            endif()

            # 添加CUDA Toolkit include路径
            if(DEFINED TR_CUDA_PATH)
                target_include_directories(${TEST_NAME} PRIVATE ${TR_CUDA_PATH}/include)
                message(STATUS "  - Using CUDA include path: ${TR_CUDA_PATH}/include")
            endif()

            # 链接CUDA基础库（使用库文件路径）
            if(WIN32)
                target_link_libraries(${TEST_NAME} PRIVATE
                    ${CUDA_cudart_LIBRARY}
                    ${CUDA_cublas_LIBRARY}
                )
            else()
                target_link_libraries(${TEST_NAME} PRIVATE
                    ${CUDA_cudart_LIBRARY}
                    ${CUDA_cublas_LIBRARY}
                )
            endif()
            message(STATUS "  - Linked CUDA libraries")

            # 链接cuDNN库
            if(DEFINED TR_CUDNN_LIBRARY_DIR)
                # cuDNN 9.x有版本子目录，需要找到实际的库目录
                set(CUDNN_SEARCH_DIRS ${TR_CUDNN_LIBRARY_DIR})
                # 添加版本子目录
                file(GLOB CUDNN_VERSION_SUBDIRS "${TR_CUDNN_LIBRARY_DIR}/*")
                foreach(CUDNN_VER_DIR ${CUDNN_VERSION_SUBDIRS})
                    if(IS_DIRECTORY ${CUDNN_VER_DIR})
                        list(APPEND CUDNN_SEARCH_DIRS ${CUDNN_VER_DIR})
                        # Windows下可能有x64子目录
                        if(WIN32)
                            file(GLOB CUDNN_ARCH_DIRS "${CUDNN_VER_DIR}/*")
                            foreach(CUDNN_ARCH_DIR ${CUDNN_ARCH_DIRS})
                                if(IS_DIRECTORY ${CUDNN_ARCH_DIR})
                                    list(APPEND CUDNN_SEARCH_DIRS ${CUDNN_ARCH_DIR})
                                endif()
                            endforeach()
                        endif()
                    endif()
                endforeach()

                # Windows下cuDNN库文件名
                if(WIN32)
                    find_library(CUDNN_LIB cudnn
                        PATHS ${CUDNN_SEARCH_DIRS}
                        NO_DEFAULT_PATH
                    )
                    if(CUDNN_LIB)
                        target_link_libraries(${TEST_NAME} PRIVATE ${CUDNN_LIB})
                        message(STATUS "  - Linked cuDNN library: ${CUDNN_LIB}")
                    else()
                        message(WARNING "  - cuDNN library not found in ${CUDNN_SEARCH_DIRS}")
                    endif()
                else()
                    # Linux下cuDNN库
                    find_library(CUDNN_LIB cudnn
                        PATHS ${CUDNN_SEARCH_DIRS}
                        NO_DEFAULT_PATH
                    )
                    if(CUDNN_LIB)
                        target_link_libraries(${TEST_NAME} PRIVATE ${CUDNN_LIB})
                        message(STATUS "  - Linked cuDNN library: ${CUDNN_LIB}")
                    else()
                        message(WARNING "  - cuDNN library not found in ${CUDNN_SEARCH_DIRS}")
                    endif()
                endif()
            else()
                message(WARNING "  - TR_CUDNN_LIBRARY_DIR not defined, skipping cuDNN")
            endif()

            # 关键配置：设置CUDA运行时库为静态（技术觉醒2方案）
            set_property(TARGET ${TEST_NAME} PROPERTY CUDA_RUNTIME_LIBRARY static)
            message(STATUS "  - Using CUDA Static Runtime (no DLL dependency)")

            message(STATUS "  - CUDA test configured successfully")
        else()
            message(WARNING "  - CUDA not available")
        endif()

    elseif(TEST_NAME MATCHES "musa")
        if(TR_USE_MUSA)
            # TODO: MUSA support
            message(WARNING "  - MUSA support not yet implemented")
        endif()

    elseif(TEST_NAME MATCHES "onednn")
        if(oneDNN_FOUND)
            target_link_libraries(${TEST_NAME} PRIVATE ${oneDNN_LIBRARIES})
            message(STATUS "  - Linked oneDNN")
        else()
            message(WARNING "  - oneDNN not available")
        endif()

    elseif(TEST_NAME MATCHES "xnnpack")
        if(XNNPACK_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${XNNPACK_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARIES})
            message(STATUS "  - Linked XNNPACK")
        else()
            message(WARNING "  - XNNPACK not available")
        endif()

    elseif(TEST_NAME MATCHES "mimalloc")
        if(mimalloc_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${mimalloc_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${mimalloc_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${mimalloc_LIBRARIES})
            message(STATUS "  - Linked mimalloc")
        else()
            message(WARNING "  - mimalloc not available")
        endif()

    elseif(TEST_NAME MATCHES "stb")
        if(STB_INCLUDE_DIR)
            target_include_directories(${TEST_NAME} PRIVATE ${STB_INCLUDE_DIR})
            # STB是头文件库，需要定义实现宏
            target_compile_definitions(${TEST_NAME} PRIVATE
                STB_IMAGE_IMPLEMENTATION
                STB_IMAGE_WRITE_IMPLEMENTATION
            )
            message(STATUS "  - Using header-only STB")
        else()
            message(WARNING "  - STB not available")
        endif()

    elseif(TEST_NAME MATCHES "zlib")
        if(ZLIB_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${ZLIB_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${ZLIB_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${ZLIB_LIBRARIES})
            message(STATUS "  - Linked zlib")
        else()
            message(WARNING "  - zlib not available")
        endif()

    elseif(TEST_NAME MATCHES "libcurl")
        if(CURL_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${CURL_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${CURL_LIBRARIES})
            message(STATUS "  - Linked libcurl")
        else()
            message(WARNING "  - libcurl not available")
        endif()

    elseif(TEST_NAME MATCHES "libjpeg")
        if(JPEG_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${JPEG_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${JPEG_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${JPEG_LIBRARIES})
            message(STATUS "  - Linked libjpeg-turbo")
        else()
            message(WARNING "  - libjpeg-turbo not available")
        endif()

    elseif(TEST_NAME MATCHES "nccl")
        if(TR_USE_NCCL)
            # TODO: NCCL support
            message(WARNING "  - NCCL support not yet implemented")
        endif()

    elseif(TEST_NAME MATCHES "simd")
        if(SIMD_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${SIMD_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${SIMD_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${SIMD_LIBRARIES})
            message(STATUS "  - Linked Simd")
        else()
            message(WARNING "  - Simd not available")
        endif()
    endif()

  
    # 编译器警告设置
    if(MSVC)
        target_compile_options(${TEST_NAME} PRIVATE /W4)
    else()
        target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    message(STATUS "  - ${DEP_NAME} test configured successfully")
endforeach()

message(STATUS "=== Dependencies Hello World Tests Configuration Complete ===")
message(STATUS "Test executables will be built in: ${DEPENDENCY_OUTPUT_DIR}")