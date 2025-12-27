# ============================================================================
# Renaissance统一库 - 包含base、data组件（device已禁用）
# ============================================================================

# 添加子目录：创建base、data、device、utils库
add_subdirectory(base)
add_subdirectory(data)
add_subdirectory(device)
add_subdirectory(utils)

# 创建renaissance统一静态库（INTERFACE库，仅用于聚合）
add_library(renaissance INTERFACE)

# 链接base、data、device、utils库
target_link_libraries(renaissance INTERFACE
    base
    data
    device
    utils
)

# 添加头文件路径
target_include_directories(renaissance INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# 外部库链接 - 统一在这里管理
# ============================================================================

# 添加libcurl支持（用于base模块的downloader）
if(TR_USE_LIBCURL)
    if(DEFINED TR_CURL_INCLUDE_DIR)
        target_include_directories(renaissance INTERFACE ${TR_CURL_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using libcurl headers from ${TR_CURL_INCLUDE_DIR}")
    endif()
    if(DEFINED TR_CURL_LIBRARY_DIR)
        if(WIN32)
            # Windows下直接链接libcurl静态库
            if(EXISTS "${TR_CURL_LIBRARY_DIR}/libcurl.a")
                target_link_libraries(renaissance INTERFACE "${TR_CURL_LIBRARY_DIR}/libcurl.a")
                message(STATUS "Renaissance library: Linking libcurl static library")
            elseif(EXISTS "${TR_CURL_LIBRARY_DIR}/libcurl.lib")
                target_link_libraries(renaissance INTERFACE "${TR_CURL_LIBRARY_DIR}/libcurl.lib")
                message(STATUS "Renaissance library: Linking libcurl import library")
            else()
                message(FATAL_ERROR "Renaissance library: libcurl library not found in ${TR_CURL_LIBRARY_DIR}")
            endif()
        else()
            # Linux/Unix下链接libcurl
            target_link_directories(renaissance INTERFACE ${TR_CURL_LIBRARY_DIR})
            target_link_libraries(renaissance INTERFACE curl)
            message(STATUS "Renaissance library: libcurl linked (for downloader)")
        endif()
    endif()
endif()

# 添加mimalloc支持
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_INCLUDE_DIR)
        target_include_directories(renaissance INTERFACE ${TR_MIMALLOC_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using mimalloc headers from ${TR_MIMALLOC_INCLUDE_DIR}")
    endif()
    if(DEFINED TR_MIMALLOC_LIBRARY_DIR)
        if(WIN32)
            # 根据构建类型选择正确的mimalloc库版本
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                # Debug模式：使用mimalloc-debug.dll.lib（与/MDd兼容）
                if(EXISTS "${TR_MIMALLOC_LIBRARY_DIR}/../debug/lib/mimalloc-debug.dll.lib")
                    target_link_libraries(renaissance INTERFACE "${TR_MIMALLOC_LIBRARY_DIR}/../debug/lib/mimalloc-debug.dll.lib")
                    message(STATUS "Renaissance library: Linking mimalloc debug dynamic library")
                elseif(EXISTS "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    target_link_libraries(renaissance INTERFACE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    message(STATUS "Renaissance library: Linking mimalloc static library (debug fallback)")
                else()
                    target_link_libraries(renaissance INTERFACE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
                    message(WARNING "Renaissance library: Debug mode linking release mimalloc library - runtime mismatch may occur!")
                endif()
            else()
                # Release模式：使用mimalloc.dll.lib（与/MD兼容）
                if(EXISTS "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    target_link_libraries(renaissance INTERFACE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    message(STATUS "Renaissance library: Linking mimalloc static library")
                else()
                    target_link_libraries(renaissance INTERFACE "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
                    message(STATUS "Renaissance library: Linking mimalloc dynamic library")
                endif()
            endif()
        else()
            target_link_directories(renaissance INTERFACE ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_libraries(renaissance INTERFACE mimalloc)
            message(STATUS "Renaissance library: Linking mimalloc from ${TR_MIMALLOC_LIBRARY_DIR}")
        endif()
    endif()
endif()

# 添加CUDA支持（base模块的cuda_arena需要）
if(TR_USE_CUDA)
    if(DEFINED TR_CUDA_PATH)
        target_include_directories(renaissance INTERFACE ${TR_CUDA_PATH}/include)
        message(STATUS "Renaissance library: Using CUDA headers from ${TR_CUDA_PATH}/include")
    endif()
    # 链接CUDA Runtime（用于cuda_arena）
    target_link_libraries(renaissance INTERFACE ${CUDA_cudart_LIBRARY})
    message(STATUS "Renaissance library: CUDA Runtime linked (for cuda_arena)")

    # 链接cuDNN（用于device模块的CudaDevice）
    if(CUDNN_LIBRARY)
        target_link_libraries(renaissance INTERFACE ${CUDNN_LIBRARY})
        message(STATUS "Renaissance library: cuDNN linked (for cuda_device)")
    endif()
endif()

# 添加MUSA支持（base模块的musa_arena需要）
if(TR_USE_MUSA)
    if(DEFINED TR_MUSA_PATH)
        target_include_directories(renaissance INTERFACE ${TR_MUSA_PATH}/include)
        message(STATUS "Renaissance library: Using MUSA headers from ${TR_MUSA_PATH}/include")
    endif()

    # 参照旧版本的方式链接MUSA Runtime
    # 注意：MUSA库在lib目录，不是lib64
    if(WIN32)
        # Windows下：如果存在MUSA库文件，直接链接
        if(DEFINED TR_MUSA_PATH AND EXISTS "${TR_MUSA_PATH}/lib")
            target_link_directories(renaissance INTERFACE "${TR_MUSA_PATH}/lib")
            target_link_libraries(renaissance INTERFACE musa musart)
            message(STATUS "Renaissance library: MUSA Runtime linked (for musa_arena)")
        endif()
    else()
        # Linux下：使用lib目录（不是lib64！）
        if(DEFINED TR_MUSA_PATH)
            if(EXISTS ${TR_MUSA_PATH}/lib)
                target_link_directories(renaissance INTERFACE ${TR_MUSA_PATH}/lib)
                message(STATUS "Renaissance library: MUSA library dir from ${TR_MUSA_PATH}/lib")
            elseif(EXISTS ${TR_MUSA_PATH}/lib64)
                target_link_directories(renaissance INTERFACE ${TR_MUSA_PATH}/lib64)
                message(STATUS "Renaissance library: MUSA library dir from ${TR_MUSA_PATH}/lib64")
            endif()
        endif()
        # 链接MUSA运行时库：musa 和 musart
        target_link_libraries(renaissance INTERFACE musa musart)
        message(STATUS "Renaissance library: MUSA Runtime linked (for musa_arena)")

        # 链接muDNN库（用于MusaDevice）
        if(DEFINED TR_MUDNN_LIBRARY_DIR)
            target_link_directories(renaissance INTERFACE ${TR_MUDNN_LIBRARY_DIR})
            target_link_libraries(renaissance INTERFACE mudnn)
            message(STATUS "Renaissance library: muDNN linked (for musa_device)")
        endif()
    endif()
endif()

message(STATUS "Renaissance unified library created successfully (base + data + device modules)")
