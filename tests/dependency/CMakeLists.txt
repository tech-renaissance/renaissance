# ============================================================================
# renAIssance Framework - Dependencies Hello World Tests
# ============================================================================

message(STATUS "=== Configuring Dependencies Hello World Tests ===")

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

# ============================================================================
# 设置输出目录
# ============================================================================

# 输出目录设置为build目录下的bin/dependency
set(DEPENDENCY_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests/dependency)

# ============================================================================
# Hello World测试源文件配置 - 场景感知智能编译
# ============================================================================

# 始终编译hello_world.cpp（所有平台都支持）
list(APPEND HELLO_WORLD_SOURCES hello_world.cpp)
list(APPEND HELLO_WORLD_DEPS BASE)
list(APPEND HELLO_WORLD_MACROS ALWAYS)

# 根据GPU支持添加相应测试
if(TR_USE_CUDA)
    list(APPEND HELLO_WORLD_SOURCES hello_cuda.cpp)
    list(APPEND HELLO_WORLD_DEPS CUDA)
    list(APPEND HELLO_WORLD_MACROS TR_USE_CUDA)
    message(STATUS "Will build CUDA test: hello_cuda.cpp")
endif()

# NCCL - 仅在GPU_CLOUD场景支持
if(TR_USE_NCCL)
    list(APPEND HELLO_WORLD_SOURCES hello_nccl.cpp)
    list(APPEND HELLO_WORLD_DEPS NCCL)
    list(APPEND HELLO_WORLD_MACROS TR_USE_NCCL)
    message(STATUS "Will build NCCL test: hello_nccl.cpp")
endif()

# XNNPACK
if(TR_USE_XNNPACK)
    list(APPEND HELLO_WORLD_SOURCES hello_xnnpack.cpp)
    list(APPEND HELLO_WORLD_DEPS XNNPACK)
    list(APPEND HELLO_WORLD_MACROS TR_USE_XNNPACK)
    message(STATUS "Will build XNNPACK test: hello_xnnpack.cpp")
endif()

if(TR_USE_MUSA)
    list(APPEND HELLO_WORLD_SOURCES hello_musa.mu)
    list(APPEND HELLO_WORLD_DEPS MUSA)
    list(APPEND HELLO_WORLD_MACROS TR_USE_MUSA)
    message(STATUS "Will build MUSA test: hello_musa.mu")
endif()

# 全平台支持的依赖测试
if(TR_USE_MIMALLOC)
    list(APPEND HELLO_WORLD_SOURCES hello_mimalloc.cpp)
    list(APPEND HELLO_WORLD_DEPS MIMALLOC)
    list(APPEND HELLO_WORLD_MACROS TR_USE_MIMALLOC)
    message(STATUS "Will build mimalloc test: hello_mimalloc.cpp")
endif()

if(TR_USE_ZLIB)
    list(APPEND HELLO_WORLD_SOURCES hello_zlib.cpp)
    list(APPEND HELLO_WORLD_DEPS ZLIB)
    list(APPEND HELLO_WORLD_MACROS TR_USE_ZLIB)
    message(STATUS "Will build zlib test: hello_zlib.cpp")
endif()

if(TR_USE_LIBCURL)
    list(APPEND HELLO_WORLD_SOURCES hello_libcurl.cpp)
    list(APPEND HELLO_WORLD_DEPS CURL)
    list(APPEND HELLO_WORLD_MACROS TR_USE_LIBCURL)
    message(STATUS "Will build libcurl test: hello_libcurl.cpp")
endif()

if(TR_USE_LIBJPEG)
    list(APPEND HELLO_WORLD_SOURCES hello_libjpeg.cpp)
    list(APPEND HELLO_WORLD_DEPS LIBJPEG)
    list(APPEND HELLO_WORLD_MACROS TR_USE_LIBJPEG)
    message(STATUS "Will build libjpeg-turbo test: hello_libjpeg.cpp")
endif()

# 三个新增的依赖测试
if(TR_USE_ONEDNN)
    list(APPEND HELLO_WORLD_SOURCES hello_onednn.cpp)
    list(APPEND HELLO_WORLD_DEPS ONEDNN)
    list(APPEND HELLO_WORLD_MACROS TR_USE_ONEDNN)
    message(STATUS "Will build oneDNN test: hello_onednn.cpp")
endif()

if(TR_USE_SIMD)
    list(APPEND HELLO_WORLD_SOURCES hello_simd.cpp)
    list(APPEND HELLO_WORLD_DEPS SIMD)
    list(APPEND HELLO_WORLD_MACROS TR_USE_SIMD)
    message(STATUS "Will build Simd test: hello_simd.cpp")
endif()

if(TR_USE_STB)
    list(APPEND HELLO_WORLD_SOURCES hello_stb.cpp)
    list(APPEND HELLO_WORLD_DEPS STB)
    list(APPEND HELLO_WORLD_MACROS TR_USE_STB)
    message(STATUS "Will build STB test: hello_stb.cpp")
endif()

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

    # 检查宏是否定义（ALWAYS总是编译）
    if(${REQUIRED_MACRO} OR REQUIRED_MACRO STREQUAL "ALWAYS")
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
        # Windows下使用mimalloc.dll.lib，Linux下使用mimalloc
        if(WIN32)
            # Windows上vcpkg安装的库文件名是mimalloc.dll.lib
            set(mimalloc_LIBRARIES "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
        else()
            set(mimalloc_LIBRARIES mimalloc)
        endif()
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
        # Windows下使用zlib，Linux下使用z
        if(WIN32)
            set(ZLIB_LIBRARIES zlib)
        else()
            set(ZLIB_LIBRARIES z)
        endif()
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
        # Windows下使用libcurl.lib，Linux下使用curl
        if(WIN32)
            set(CURL_LIBRARIES libcurl)
        else()
            set(CURL_LIBRARIES curl)
        endif()
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
        # libjpeg-turbo的TurboJPEG API在turbojpeg库中
        set(JPEG_LIBRARIES turbojpeg)
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

# NCCL (NVIDIA Collective Communications Library) - 仅在GPU_CLOUD场景
if(TR_USE_NCCL)
    if(DEFINED TR_NCCL_INCLUDE_DIR AND DEFINED TR_NCCL_LIBRARY_DIR)
        set(NCCL_FOUND TRUE)
        set(NCCL_INCLUDE_DIRS ${TR_NCCL_INCLUDE_DIR})
        set(NCCL_LIBRARIES nccl)
        set(NCCL_LIBRARY_DIRS ${TR_NCCL_LIBRARY_DIR})
        message(STATUS "Found NCCL: ${TR_NCCL_INCLUDE_DIR}")
    else()
        set(NCCL_FOUND FALSE)
        message(WARNING "TR_NCCL paths not defined")
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

    # 创建可执行文件 - MUSA需要特殊处理
    set(IS_MUSA_SOURCE FALSE)
    if(TEST_SOURCE MATCHES "\\.mu$")
        set(IS_MUSA_SOURCE TRUE)
        # MUSA文件使用musa_add_executable
        message(STATUS "  - Using MUSA compiler for ${TEST_SOURCE}")
        # 查找MUSA并启用MUSA语言支持
        if(DEFINED TR_MUSA_PATH)
            set(MUSA_ROOT ${TR_MUSA_PATH})
        elseif(DEFINED ENV{MUSA_ROOT})
            set(MUSA_ROOT $ENV{MUSA_ROOT})
        elseif(EXISTS /usr/local/musa)
            set(MUSA_ROOT /usr/local/musa)
        endif()

        if(MUSA_ROOT)
            list(APPEND CMAKE_MODULE_PATH ${MUSA_ROOT}/cmake)
            find_package(MUSA REQUIRED)
            musa_add_executable(${TEST_NAME} ${TEST_SOURCE}
                OPTIONS -Xclang -I/usr/include/c++/13 -Xclang -I/usr/include/x86_64-linux-gnu/c++/13
            )
        else()
            message(FATAL_ERROR "MUSA_ROOT not found for compiling .mu files")
        endif()
    else()
        # 标准C++文件使用add_executable
        add_executable(${TEST_NAME} ${TEST_SOURCE})
    endif()

    # 设置C++标准（仅对C++文件）
    if(NOT IS_MUSA_SOURCE)
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 20)
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
    endif()

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
    if(DEP_NAME STREQUAL "BASE")
        # hello_world.cpp - 不需要任何特殊库
        message(STATUS "  - BASE test configured (no dependencies)")

    elseif(TEST_NAME MATCHES "cuda")
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
            # 查找MUSA包
            if(DEFINED TR_MUSA_PATH)
                set(MUSA_ROOT ${TR_MUSA_PATH})
                message(STATUS "  - Using MUSA root: ${MUSA_ROOT}")
            else()
                # 尝试从环境变量或标准路径查找
                if(DEFINED ENV{MUSA_ROOT})
                    set(MUSA_ROOT $ENV{MUSA_ROOT})
                elseif(DEFINED ENV{MUSA_HOME})
                    set(MUSA_ROOT $ENV{MUSA_HOME})
                else()
                    # Linux默认路径
                    if(EXISTS /usr/local/musa)
                        set(MUSA_ROOT /usr/local/musa)
                    endif()
                endif()

                if(MUSA_ROOT)
                    message(STATUS "  - Found MUSA at: ${MUSA_ROOT}")
                else()
                    message(WARNING "  - MUSA_ROOT not found, trying standard paths")
                endif()
            endif()

            if(MUSA_ROOT)
                # MUSA include目录
                set(MUSA_INCLUDE_DIRS ${MUSA_ROOT}/include)
                target_include_directories(${TEST_NAME} PRIVATE ${MUSA_INCLUDE_DIRS})
                message(STATUS "  - MUSA include: ${MUSA_INCLUDE_DIRS}")

                # MUSA库目录 - 尝试多个可能的目录
                set(MUSA_LIBRARY_DIRS "")
                if(WIN32)
                    set(MUSA_LIBRARY_DIRS ${MUSA_ROOT}/lib/x64)
                else()
                    # Linux下尝试多个目录：lib64, lib
                    if(EXISTS ${MUSA_ROOT}/lib64)
                        set(MUSA_LIBRARY_DIRS ${MUSA_ROOT}/lib64)
                        message(STATUS "  - MUSA library dir: ${MUSA_LIBRARY_DIRS} (found lib64)")
                    elseif(EXISTS ${MUSA_ROOT}/lib)
                        set(MUSA_LIBRARY_DIRS ${MUSA_ROOT}/lib)
                        message(STATUS "  - MUSA library dir: ${MUSA_LIBRARY_DIRS} (found lib)")
                    else()
                        # 如果两个都不存在，先尝试lib64（常见情况）
                        set(MUSA_LIBRARY_DIRS ${MUSA_ROOT}/lib64)
                        message(STATUS "  - MUSA library dir: ${MUSA_LIBRARY_DIRS} (default, may not exist)")
                    endif()
                endif()

                # 链接MUSA运行时库 - 注意：musa_add_executable已经自动链接了MUSA库
                # 所以这里不需要再次手动链接
                message(STATUS "  - MUSA runtime will be linked automatically by musa_add_executable")

                # 添加MUSA CMake模块路径（用于musa_add_executable）
                list(APPEND CMAKE_MODULE_PATH ${MUSA_ROOT}/cmake)

                message(STATUS "  - MUSA test configured successfully")
            else()
                message(WARNING "  - MUSA not available")
            endif()
        else()
            message(WARNING "  - MUSA support disabled (TR_USE_MUSA=OFF)")
        endif()

    elseif(TEST_NAME MATCHES "onednn")
        if(oneDNN_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${oneDNN_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${oneDNN_LIBRARY_DIRS})
            target_link_libraries(${TEST_NAME} PRIVATE ${oneDNN_LIBRARIES})
            # 抑制 oneDNN 库的 execstack 警告（ittptmark64.S.o）
            # PC_MUSA 场景使用 binutils 2.38，不支持 --no-warn-execstack 选项
            if(NOT MSVC AND NOT TR_SCENE_PC_MUSA)
                target_link_options(${TEST_NAME} PRIVATE "LINKER:--no-warn-execstack")
            endif()
            message(STATUS "  - Linked oneDNN")
        else()
            message(WARNING "  - oneDNN not available")
        endif()

    elseif(TEST_NAME MATCHES "xnnpack")
        if(XNNPACK_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${XNNPACK_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARY_DIRS})

            # XNNPACK使用C标准库，需要降低C++标准版本以避免兼容性问题
            set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)

            if(MSVC)
                # Windows上使用WHOLE_ARCHIVE强制包含所有microkernel符号
                target_link_libraries(${TEST_NAME} PRIVATE
                    XNNPACK
                    pthreadpool
                    cpuinfo
                    xnnpack-microkernels-prod
                )

                # MSVC需要使用/WHOLEARCHIVE来强制链接所有符号
                # 否则链接器会优化掉未直接引用的microkernel函数
                target_link_options(${TEST_NAME} PRIVATE
                    "/WHOLEARCHIVE:XNNPACK.lib"
                    "/WHOLEARCHIVE:pthreadpool.lib"
                    "/WHOLEARCHIVE:cpuinfo.lib"
                    "/WHOLEARCHIVE:xnnpack-microkernels-prod.lib"
                )

                # XNNPACK需要_CRT_SECURE_NO_WARNINGS来避免某些C标准库函数的警告
                target_compile_definitions(${TEST_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

                # C:\XNNPACK的库使用动态CRT(/MD)，与项目默认设置一致
                message(STATUS "  - Linked XNNPACK (static with WHOLE_ARCHIVE, /MD CRT, C++17)")
            else()
                # Linux/macOS使用动态库
                target_link_libraries(${TEST_NAME} PRIVATE ${XNNPACK_LIBRARIES})

                # 设置运行时 RPATH 以便找到动态库
                if(UNIX AND NOT APPLE)
                    set_target_properties(${TEST_NAME} PROPERTIES
                        INSTALL_RPATH "${XNNPACK_LIBRARY_DIRS}"
                        BUILD_RPATH "${XNNPACK_LIBRARY_DIRS}"
                    )
                endif()
                message(STATUS "  - Linked XNNPACK (dynamic, C++17)")
            endif()
        else()
            message(WARNING "  - XNNPACK not available")
        endif()

    elseif(TEST_NAME MATCHES "mimalloc")
        if(mimalloc_FOUND)
            target_include_directories(${TEST_NAME} PRIVATE ${mimalloc_INCLUDE_DIRS})
            target_link_directories(${TEST_NAME} PRIVATE ${mimalloc_LIBRARY_DIRS})
            message(STATUS "  - mimalloc libraries: ${mimalloc_LIBRARIES}")
            message(STATUS "  - mimalloc library dirs: ${mimalloc_LIBRARY_DIRS}")
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

            # ========== STB Warning Suppression ==========
            # Windows MSVC: 禁用CRT安全警告(C4996 - sprintf等函数)
            if(MSVC)
                target_compile_definitions(${TEST_NAME} PRIVATE
                    _CRT_SECURE_NO_WARNINGS
                )
                message(STATUS "  - STB: Disabled CRT security warnings (_CRT_SECURE_NO_WARNINGS)")

            # Linux/ARM/RISC-V: 禁用pedantic警告(GCC/Clang扩展)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                # 禁用ISO C++严格标准检查(compound-literals警告)
                target_compile_options(${TEST_NAME} PRIVATE
                    -Wno-pedantic
                    -Wno-missing-field-initializers
                )
                message(STATUS "  - STB: Disabled pedantic warnings (-Wno-pedantic)")

                # MUSA平台: 可选禁用LTO串行编译提示
                if(DEFINED TR_SCENE_PC_MUSA AND TR_SCENE_PC_MUSA)
                    target_compile_options(${TEST_NAME} PRIVATE
                        -Wno-lto-type-mismatch
                    )
                    message(STATUS "  - STB: Disabled LTO warnings for MUSA")
                endif()
            endif()

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
            if(NCCL_FOUND)
                # NCCL 需要 CUDA 支持
                target_include_directories(${TEST_NAME} PRIVATE ${NCCL_INCLUDE_DIRS})
                target_link_directories(${TEST_NAME} PRIVATE ${NCCL_LIBRARY_DIRS})
                target_link_libraries(${TEST_NAME} PRIVATE ${NCCL_LIBRARIES})

                # 添加 CUDA include 路径（NCCL 依赖 CUDA）
                if(DEFINED TR_CUDA_PATH)
                    target_include_directories(${TEST_NAME} PRIVATE ${TR_CUDA_PATH}/include)
                    message(STATUS "  - Added CUDA include for NCCL: ${TR_CUDA_PATH}/include")
                endif()

                # 链接 CUDA runtime（NCCL 依赖 CUDA）
                target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_cudart_LIBRARY})

                message(STATUS "  - Linked NCCL")
            else()
                message(WARNING "  - NCCL not available")
            endif()
        else()
            message(STATUS "  - NCCL support disabled (TR_USE_NCCL=OFF)")
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

    # ========== 编译器警告设置 ==========
    # 注意：STB测试会有特殊处理，这里设置的警告可能在后面被覆盖
    if(MSVC)
        # Windows: 启用/W4高级警告，但STB会单独禁用CRT安全警告
        if(NOT TEST_NAME MATCHES "stb")
            target_compile_options(${TEST_NAME} PRIVATE /W4)
        endif()
    else()
        # Linux/ARM: 启用严格警告，但STB需要禁用pedantic(GCC扩展)
        if(TEST_NAME MATCHES "stb")
            # STB库使用GCC扩展(NEON SIMD)，不能启用pedantic
            target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra)
        else()
            target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()

    message(STATUS "  - ${DEP_NAME} test configured successfully")
endforeach()

message(STATUS "=== Dependencies Hello World Tests Configuration Complete ===")
message(STATUS "Test executables will be built in: ${DEPENDENCY_OUTPUT_DIR}")