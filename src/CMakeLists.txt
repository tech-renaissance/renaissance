# ============================================================================
# Renaissance统一库 - 无板块依赖，直接收集所有源文件编译为一个库
# ============================================================================

# 收集所有源文件
set(RENAISSANCE_SOURCES)

# ============================================================================
# core 板块 - 核心基础设施（Logger、TRException、GlobalRegistry等）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    core/downloader.cpp
    core/global_registry.cpp
    core/initializer.cpp
    core/logger.cpp
    core/staging_buffer_pool.cpp
    core/staging_param_pool.cpp
    core/rng.cpp
    core/tr_exception.cpp
)

# ============================================================================
# graph 板块 - IR架构（BluePrint、ArchPlan等）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    graph/op_kind.cpp
    graph/arch_plan_expand.cpp
    graph/arch_plan_normalize.cpp
    graph/arch_plan_shape.cpp
    graph/arch_plan_merge.cpp
    graph/arch_plan_format.cpp
    graph/arch_plan_yaml.cpp
)

# ============================================================================
# algo 板块 - 算法配置（Loss、Optimizer、Scheduler等）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    algo/loss.cpp
    algo/optimizer.cpp
    algo/scheduler.cpp
)

# ============================================================================
# tensor 板块 - 张量相关（Tensor、DTensor等）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    tensor/tensor.cpp
)

# ============================================================================
# backend 板块 - 内存池管理（MemoryArena、ArenaKeeper等）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    backend/memory_arena.cpp
    backend/device_context.cpp
    backend/graph_executor.cpp
    backend/op_registry.cpp
    backend/op_stream_policy.cpp
    backend/ops/dtensor/relu_op.cpp
    backend/ops/dtensor/relu_op.cu
    backend/ops/dtensor/identity_op.cpp
    backend/ops/dtensor/identity_op.cu
    backend/ops/dtensor/tanh_op.cpp
    backend/ops/dtensor/tanh_op.cu
    backend/ops/dtensor/silu_op.cpp
    backend/ops/dtensor/silu_op.cu
    backend/ops/dtensor/relu6_op.cpp
    backend/ops/dtensor/relu6_op.cu
    backend/ops/dtensor/leaky_relu_op.cpp
    backend/ops/dtensor/leaky_relu_op.cu
    backend/ops/dtensor/hardswish_op.cpp
    backend/ops/dtensor/hardswish_op.cu
    backend/ops/dtensor/elu_op.cpp
    backend/ops/dtensor/elu_op.cu
    backend/ops/dtensor/sigmoid_op.cpp
    backend/ops/dtensor/sigmoid_op.cu
    backend/ops/dtensor/fc_op.cpp
    backend/ops/dtensor/fc_op.cu
    backend/ops/dtensor/flatten_op.cpp
    backend/ops/dtensor/flatten_op.cu
    backend/ops/dtensor/gap_op.cpp
    backend/ops/dtensor/gap_op.cu
    backend/ops/dtensor/maxpool_op.cpp
    backend/ops/dtensor/dropout_op.cpp
    backend/ops/dtensor/dropout_op.cu
    backend/ops/dtensor/softmax_ce_op.cpp
    backend/ops/dtensor/softmax_ce_op.cu
    backend/ops/dtensor/axpy_op.cpp
    backend/ops/dtensor/axpy_op.cu
    backend/ops/dtensor/conv_op.cpp
    backend/ops/dtensor/conv_op_impl.cpp
    backend/ops/range/h2d_op.cpp
    backend/ops/range/clear_op.cpp
    backend/ops/range/accum_op.cpp
    backend/ops/range/accum_op.cu
    backend/ops/range/copy_op.cpp
    backend/ops/range/cast_op.cpp
    backend/ops/range/cast_op.cu
    backend/ops/range/check_op.cpp
    backend/ops/range/check_op.cu
    backend/ops/range/grad_scaling_op.cpp
    backend/ops/range/grad_scaling_op.cu
    backend/ops/range/allreduce_op.cpp
    backend/ops/range/optimizer_op.cpp
    backend/ops/range/optimizer_op.cu
    backend/ops/dtensor/lars_op.cpp
    backend/ops/dtensor/lars_op.cu
    backend/ops/dtensor/dtensor_copy_op.cpp
    backend/ops/dtensor/adam_bc_op.cpp
    backend/ops/dtensor/adam_bc_op.cu
    backend/infra_kernels.cpp
    backend/infra_kernels.cu
)

# ============================================================================
# memory 板块 - MemoryPlan显存布局引擎（65-Region规范）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    graph/memory_plan.cpp
    graph/captured_graph.cpp
    graph/capture_cuda.cpp
    graph/capture_cpu.cpp
    graph/capture_multi_stream.cpp
    graph/compiler.cpp
    graph/layer_descriptor_registry.cpp
)

# ============================================================================
# task 板块 - 静态图任务管理（TaskBase、SimpleTask）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    task/task_base.cpp
    task/simple_task.cpp
    task/deep_learning_task.cpp
)

# ============================================================================
# data 板块 - 数据加载和预处理（DataLoader、Preprocessor、TransferStation等）
# ============================================================================
list(APPEND RENAISSANCE_SOURCES
    data/center_crop.cpp
    data/cifar_loader_dts.cpp
    data/cifar_loader_raw.cpp
    data/color_jitter.cpp
    data/cpu_advisor.cpp
    data/do_nothing.cpp
    data/transfer_station.cpp
    data/fast_random_resized_crop.cpp
    data/file_handle.cpp
    data/fused_normalization.cpp
    data/normalize.cpp
    data/gaussian_blur.cpp
    data/gaussian_noise.cpp
    data/hardware_topology.cpp
    data/imagenet_loader_dts.cpp
    data/imagenet_loader_raw.cpp
    data/mnist_loader_dts.cpp
    data/mnist_loader_raw.cpp
    data/pad.cpp
    data/preprocess_operation.cpp
    data/preprocessor.cpp
    data/preprocess_worker.cpp
    data/random_autocontrast.cpp
    data/random_crop.cpp
    data/random_erasing.cpp
    data/random_grayscale.cpp
    data/random_horizontal_flip.cpp
    data/random_resized_crop.cpp
    data/random_rotation.cpp
    data/random_scale.cpp
    data/resize.cpp
    data/sample_loader.cpp
)

# ============================================================================
# 后续添加其他板块时的示例：
# ============================================================================
# data 板块
# list(APPEND RENAISSANCE_SOURCES
#     data/dataloader.cpp
#     data/preprocessor.cpp
#     data/transfer_station.cpp
# )

# device 板块
# list(APPEND RENAISSANCE_SOURCES
#     device/cpu_device.cpp
#     device/cpu_tensor.cpp
# )

# model 板块
# list(APPEND RENAISSANCE_SOURCES
#     model/conv2d.cpp
#     model/linear.cpp
# )

# ============================================================================
# 创建Renaissance统一静态库
# ============================================================================
add_library(renaissance STATIC ${RENAISSANCE_SOURCES})

# 设置头文件路径
target_include_directories(renaissance
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 设置C++标准
set_target_properties(renaissance PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 设置属性
set_target_properties(renaissance PROPERTIES
    FOLDER "src"
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# 第三方库依赖管理
# 注：根据 docs/rules.md 规范，所有第三方库链接统一在此处管理
#      src/ 下的子目录 CMakeLists.txt 一律不链接第三方库
# ============================================================================

# 添加mimalloc支持（用于CPU内存池）
if(TR_USE_MIMALLOC)
    if(DEFINED TR_MIMALLOC_INCLUDE_DIR)
        target_include_directories(renaissance PRIVATE ${TR_MIMALLOC_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using mimalloc headers from ${TR_MIMALLOC_INCLUDE_DIR}")
    endif()
    if(DEFINED TR_MIMALLOC_LIBRARY_DIR)
        if(WIN32)
            # Windows下根据构建类型选择库版本
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                if(EXISTS "${TR_MIMALLOC_LIBRARY_DIR}/../debug/lib/mimalloc-debug.dll.lib")
                    target_link_libraries(renaissance PUBLIC "${TR_MIMALLOC_LIBRARY_DIR}/../debug/lib/mimalloc-debug.dll.lib")
                    message(STATUS "Renaissance library: Linking mimalloc debug dynamic library")
                elseif(EXISTS "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    target_link_libraries(renaissance PUBLIC "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    message(STATUS "Renaissance library: Linking mimalloc static library (debug fallback)")
                else()
                    target_link_libraries(renaissance PUBLIC "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
                    message(WARNING "Renaissance library: Debug mode linking release mimalloc library - runtime mismatch may occur!")
                endif()
            else()
                # Release模式
                if(EXISTS "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    target_link_libraries(renaissance PUBLIC "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc-static.lib")
                    message(STATUS "Renaissance library: Linking mimalloc static library")
                else()
                    target_link_libraries(renaissance PUBLIC "${TR_MIMALLOC_LIBRARY_DIR}/mimalloc.dll.lib")
                    message(STATUS "Renaissance library: Linking mimalloc dynamic library")
                endif()
            endif()
        else()
            # Linux/Unix下链接mimalloc（PUBLIC以传播给测试程序）
            target_link_directories(renaissance PUBLIC ${TR_MIMALLOC_LIBRARY_DIR})
            target_link_libraries(renaissance PUBLIC mimalloc)
            message(STATUS "Renaissance library: Linking mimalloc from ${TR_MIMALLOC_LIBRARY_DIR}")
        endif()
    endif()
endif()

# 添加libcurl支持（用于downloader）
if(TR_USE_LIBCURL)
    if(DEFINED TR_CURL_INCLUDE_DIR)
        target_include_directories(renaissance PUBLIC ${TR_CURL_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using libcurl headers from ${TR_CURL_INCLUDE_DIR}")
    endif()
    if(DEFINED TR_CURL_LIBRARY_DIR)
        if(WIN32)
            # Windows下直接链接libcurl静态库
            if(EXISTS "${TR_CURL_LIBRARY_DIR}/libcurl.a")
                target_link_libraries(renaissance PUBLIC "${TR_CURL_LIBRARY_DIR}/libcurl.a")
                message(STATUS "Renaissance library: Linking libcurl static library")
            elseif(EXISTS "${TR_CURL_LIBRARY_DIR}/libcurl.lib")
                target_link_libraries(renaissance PUBLIC "${TR_CURL_LIBRARY_DIR}/libcurl.lib")
                message(STATUS "Renaissance library: Linking libcurl import library")
            else()
                message(FATAL_ERROR "Renaissance library: libcurl library not found in ${TR_CURL_LIBRARY_DIR}")
            endif()
        else()
            # Linux/Unix下链接libcurl（系统库，使用库名方式，PUBLIC传播）
            target_link_libraries(renaissance PUBLIC curl)
            message(STATUS "Renaissance library: libcurl linked (for downloader)")
        endif()
    endif()
endif()

# 添加libarchive支持（用于解压.tar.gz数据集）
if(TR_USE_LIBARCHIVE)
    if(DEFINED TR_LIBARCHIVE_INCLUDE_DIR)
        target_include_directories(renaissance PUBLIC ${TR_LIBARCHIVE_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using libarchive headers from ${TR_LIBARCHIVE_INCLUDE_DIR}")
    endif()
    if(DEFINED TR_LIBARCHIVE_LIBRARY_DIR)
        if(WIN32)
            # Windows下直接链接libarchive静态库
            if(EXISTS "${TR_LIBARCHIVE_LIBRARY_DIR}/archive.lib")
                target_link_libraries(renaissance PUBLIC "${TR_LIBARCHIVE_LIBRARY_DIR}/archive.lib")
                message(STATUS "Renaissance library: Linking libarchive import library")
            elseif(EXISTS "${TR_LIBARCHIVE_LIBRARY_DIR}/libarchive.a")
                target_link_libraries(renaissance PUBLIC "${TR_LIBARCHIVE_LIBRARY_DIR}/libarchive.a")
                message(STATUS "Renaissance library: Linking libarchive static library")
            else()
                message(FATAL_ERROR "Renaissance library: libarchive library not found in ${TR_LIBARCHIVE_LIBRARY_DIR}")
            endif()
        else()
            # Linux/Unix下链接libarchive
            target_link_directories(renaissance PUBLIC ${TR_LIBARCHIVE_LIBRARY_DIR})
            target_link_libraries(renaissance PUBLIC archive)
            message(STATUS "Renaissance library: libarchive linked")
        endif()
    endif()
endif()

# 添加libnuma支持（仅Linux GPU_CLOUD场景，用于NUMA内存绑定）
if(TR_USE_LIBNUMA)
    if(DEFINED TR_NUMA_INCLUDE_DIR)
        target_include_directories(renaissance PUBLIC ${TR_NUMA_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using libnuma headers from ${TR_NUMA_INCLUDE_DIR}")
    endif()
    if(DEFINED TR_NUMA_LIBRARY_DIR)
        if(WIN32)
            message(WARNING "Renaissance library: libnuma is not supported on Windows")
        else()
            target_link_directories(renaissance PUBLIC ${TR_NUMA_LIBRARY_DIR})
            target_link_libraries(renaissance PUBLIC numa)
            message(STATUS "Renaissance library: libnuma linked from ${TR_NUMA_LIBRARY_DIR}")
        endif()
    endif()
endif()

# 添加zlib支持（用于CRC校验）
find_package(ZLIB REQUIRED)
target_include_directories(renaissance PUBLIC ${ZLIB_INCLUDE_DIRS})
target_link_libraries(renaissance PUBLIC ${ZLIB_LIBRARIES})
message(STATUS "Renaissance library: zlib linked (for CRC verification)")

# 添加CUDA支持
if(TR_USE_CUDA)
    # 使用SYSTEM标志以抑制第三方库的警告
    if(DEFINED TR_CUDA_PATH)
        target_include_directories(renaissance SYSTEM PUBLIC ${TR_CUDA_PATH}/include)
        message(STATUS "Renaissance library: Using CUDA headers from ${TR_CUDA_PATH}/include")
    endif()

    # 添加cuDNN头文件支持（cuDNN Frontend依赖cuDNN头文件）
    # 使用SYSTEM标志以抑制第三方库的警告
    if(DEFINED TR_CUDNN_INCLUDE_DIR)
        target_include_directories(renaissance SYSTEM PUBLIC ${TR_CUDNN_INCLUDE_DIR})
        # cuDNN 9.x使用版本化子目录，添加具体版本路径（如果存在）
        if(DEFINED TR_CUDNN_INCLUDE_DIR_VERSIONED)
            target_include_directories(renaissance SYSTEM PUBLIC ${TR_CUDNN_INCLUDE_DIR_VERSIONED})
            message(STATUS "Renaissance library: Using cuDNN headers (versioned) from ${TR_CUDNN_INCLUDE_DIR_VERSIONED}")
        else()
            message(STATUS "Renaissance library: Using cuDNN headers from ${TR_CUDNN_INCLUDE_DIR}")
        endif()
    endif()

    # 添加cuDNN Frontend支持（纯头文件库，仅需包含头文件）
    # 使用SYSTEM标志以抑制第三方库的警告
    if(DEFINED TR_CUDNN_FRONTEND_INCLUDE_DIR)
        target_include_directories(renaissance SYSTEM PUBLIC ${TR_CUDNN_FRONTEND_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using cuDNN Frontend headers from ${TR_CUDNN_FRONTEND_INCLUDE_DIR}")
    elseif(DEFINED TR_CUDNN_FRONTEND_PATH)
        target_include_directories(renaissance SYSTEM PUBLIC ${TR_CUDNN_FRONTEND_PATH}/include)
        message(STATUS "Renaissance library: Using cuDNN Frontend headers from ${TR_CUDNN_FRONTEND_PATH}/include")
    else()
        message(WARNING "TR_CUDNN_FRONTEND_INCLUDE_DIR and TR_CUDNN_FRONTEND_PATH not defined - cuDNN Frontend support may be limited")
    endif()

    # 链接CUDA Runtime（PUBLIC传播）
    target_link_libraries(renaissance PUBLIC ${CUDA_cudart_LIBRARY})
    message(STATUS "Renaissance library: CUDA Runtime linked")

    # 链接cuDNN（PUBLIC传播）
    if(CUDNN_LIBRARY)
        target_link_libraries(renaissance PUBLIC ${CUDNN_LIBRARY})
        message(STATUS "Renaissance library: cuDNN linked")
    endif()

    # 链接cuBLAS（用于 FC AMP BWD 等算子）
    if(CUDA_cublas_LIBRARY)
        target_link_libraries(renaissance PUBLIC ${CUDA_cublas_LIBRARY})
        message(STATUS "Renaissance library: cuBLAS linked")
    endif()

    # 链接NCCL（用于多GPU通信）
    # 使用SYSTEM标志以抑制第三方库的警告
    if(TR_USE_NCCL AND DEFINED TR_NCCL_LIBRARY_DIR)
        if(DEFINED TR_NCCL_INCLUDE_DIR)
            target_include_directories(renaissance SYSTEM PRIVATE ${TR_NCCL_INCLUDE_DIR})
            message(STATUS "Renaissance library: Using NCCL headers from ${TR_NCCL_INCLUDE_DIR}")
        endif()
        # 使用完整路径链接（静态库需要，否则下游无法找到）
        if(WIN32)
            if(EXISTS "${TR_NCCL_LIBRARY_DIR}/nccl.lib")
                target_link_libraries(renaissance PUBLIC "${TR_NCCL_LIBRARY_DIR}/nccl.lib")
            else()
                message(FATAL_ERROR "NCCL library not found in ${TR_NCCL_LIBRARY_DIR}")
            endif()
        else()
            if(EXISTS "${TR_NCCL_LIBRARY_DIR}/libnccl.so")
                target_link_libraries(renaissance PUBLIC "${TR_NCCL_LIBRARY_DIR}/libnccl.so")
            elseif(EXISTS "${TR_NCCL_LIBRARY_DIR}/libnccl_static.a")
                target_link_libraries(renaissance PUBLIC "${TR_NCCL_LIBRARY_DIR}/libnccl_static.a")
            else()
                message(FATAL_ERROR "NCCL library not found in ${TR_NCCL_LIBRARY_DIR}")
            endif()
        endif()
        message(STATUS "Renaissance library: NCCL linked (for multi-GPU communication)")
    endif()

    # 添加CUTLASS支持（header-only库）
    if(DEFINED TR_CUTLASS_INCLUDE_DIR)
        target_include_directories(renaissance PRIVATE ${TR_CUTLASS_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using CUTLASS headers from ${TR_CUTLASS_INCLUDE_DIR}")
    elseif(DEFINED TR_CUTLASS_PATH)
        target_include_directories(renaissance PRIVATE ${TR_CUTLASS_PATH}/include)
        message(STATUS "Renaissance library: Using CUTLASS headers from ${TR_CUTLASS_PATH}/include")
    endif()

    # 添加CUDA C++ Standard Library (libcudacxx) 头文件路径
    # 使用SYSTEM标志以抑制第三方库的警告
    if(DEFINED CUDAToolkit_ROOT)
        target_include_directories(renaissance SYSTEM PRIVATE ${CUDAToolkit_ROOT}/targets/x86_64-linux/include/cccl)
        message(STATUS "Renaissance library: Using CUDA std library (libcudacxx) from ${CUDAToolkit_ROOT}/targets/x86_64-linux/include/cccl")
    elseif(DEFINED TR_CUDA_PATH)
        target_include_directories(renaissance SYSTEM PRIVATE ${TR_CUDA_PATH}/targets/x86_64-linux/include/cccl)
        message(STATUS "Renaissance library: Using CUDA std library (libcudacxx) from ${TR_CUDA_PATH}/targets/x86_64-linux/include/cccl")
    endif()
endif()

# 添加MUSA支持（仅在Linux下可用）
if(TR_USE_MUSA)
    if(DEFINED TR_MUSA_PATH)
        target_include_directories(renaissance PRIVATE ${TR_MUSA_PATH}/include)
        message(STATUS "Renaissance library: Using MUSA headers from ${TR_MUSA_PATH}/include")
    endif()

    # 确定MUSA库目录（优先lib，其次lib64）
    if(DEFINED TR_MUSA_PATH)
        set(MUSA_LIB_DIR "")
        if(EXISTS ${TR_MUSA_PATH}/lib)
            set(MUSA_LIB_DIR ${TR_MUSA_PATH}/lib)
            message(STATUS "Renaissance library: MUSA library dir from ${TR_MUSA_PATH}/lib")
        elseif(EXISTS ${TR_MUSA_PATH}/lib64)
            set(MUSA_LIB_DIR ${TR_MUSA_PATH}/lib64)
            message(STATUS "Renaissance library: MUSA library dir from ${TR_MUSA_PATH}/lib64")
        else()
            message(FATAL_ERROR "MUSA library directory not found in ${TR_MUSA_PATH}")
        endif()

        # 链接MUSA运行时库（完整路径，静态库需要）
        if(EXISTS "${MUSA_LIB_DIR}/libmusa.so")
            target_link_libraries(renaissance PUBLIC "${MUSA_LIB_DIR}/libmusa.so")
        elseif(EXISTS "${MUSA_LIB_DIR}/libmusa.a")
            target_link_libraries(renaissance PUBLIC "${MUSA_LIB_DIR}/libmusa.a")
        else()
            message(FATAL_ERROR "MUSA library not found in ${MUSA_LIB_DIR}")
        endif()

        if(EXISTS "${MUSA_LIB_DIR}/libmusart.so")
            target_link_libraries(renaissance PUBLIC "${MUSA_LIB_DIR}/libmusart.so")
        elseif(EXISTS "${MUSA_LIB_DIR}/libmusart.a")
            target_link_libraries(renaissance PUBLIC "${MUSA_LIB_DIR}/libmusart.a")
        else()
            message(FATAL_ERROR "MUSArt library not found in ${MUSA_LIB_DIR}")
        endif()

        message(STATUS "Renaissance library: MUSA Runtime linked")
    endif()

    # 链接muDNN库（完整路径，静态库需要）
    if(DEFINED TR_MUDNN_LIBRARY_DIR)
        if(EXISTS "${TR_MUDNN_LIBRARY_DIR}/libmudnn.so")
            target_link_libraries(renaissance PUBLIC "${TR_MUDNN_LIBRARY_DIR}/libmudnn.so")
        elseif(EXISTS "${TR_MUDNN_LIBRARY_DIR}/libmudnn.a")
            target_link_libraries(renaissance PUBLIC "${TR_MUDNN_LIBRARY_DIR}/libmudnn.a")
        else()
            message(FATAL_ERROR "muDNN library not found in ${TR_MUDNN_LIBRARY_DIR}")
        endif()
        message(STATUS "Renaissance library: muDNN linked")
    endif()
endif()

# 添加Simd库支持（用于Preprocessor的图像处理）
if(DEFINED TR_SIMD_INCLUDE_DIR AND DEFINED TR_SIMD_LIBRARY_DIR)
    target_include_directories(renaissance PUBLIC ${TR_SIMD_INCLUDE_DIR})
    target_link_directories(renaissance PUBLIC ${TR_SIMD_LIBRARY_DIR})
    target_link_libraries(renaissance PUBLIC Simd)
    message(STATUS "Renaissance library: Simd linked (for Preprocessor image transforms)")
else()
    message(WARNING "TR_SIMD paths not defined - Preprocessor image transforms will not be available")
endif()

# 添加libjpeg-turbo支持（用于Preprocessor的JPEG解码）
if(DEFINED TR_JPEG_INCLUDE_DIR AND DEFINED TR_JPEG_LIBRARY_DIR)
    target_include_directories(renaissance PUBLIC ${TR_JPEG_INCLUDE_DIR})
    target_link_directories(renaissance PUBLIC ${TR_JPEG_LIBRARY_DIR})
    target_link_libraries(renaissance PUBLIC turbojpeg)
    message(STATUS "Renaissance library: turbojpeg linked (for Preprocessor JPEG decode)")
else()
    message(WARNING "TR_JPEG paths not defined - Preprocessor JPEG decode will not be available")
endif()

# 添加STB支持（用于JPEG备用解码）
if(DEFINED TR_STB_INCLUDE_DIR)
    target_include_directories(renaissance PRIVATE ${TR_STB_INCLUDE_DIR})
    target_compile_definitions(renaissance PRIVATE TR_USE_STB=1)
    message(STATUS "Renaissance library: STB headers available (for JPEG fallback decode)")
else()
    message(WARNING "TR_STB_INCLUDE_DIR not defined - STB JPEG fallback will not be available")
endif()

# 添加Eigen3支持（用于CPU kernels向量化优化）
if(TR_USE_EIGEN)
    if(DEFINED TR_EIGEN_INCLUDE_DIR)
        target_include_directories(renaissance PUBLIC ${TR_EIGEN_INCLUDE_DIR})
        message(STATUS "Renaissance library: Using Eigen3 headers from ${TR_EIGEN_INCLUDE_DIR}")
        target_compile_definitions(renaissance PUBLIC TR_USE_EIGEN=1)
    else()
        message(WARNING "TR_USE_EIGEN is ON but TR_EIGEN_INCLUDE_DIR not defined - Eigen3 optimizations will be disabled")
    endif()
endif()

# 添加XNNPACK支持（用于CPU卷积加速）
if(TR_USE_XNNPACK)
    if(DEFINED TR_XNNPACK_INCLUDE_DIR)
        target_include_directories(renaissance PRIVATE ${TR_XNNPACK_INCLUDE_DIR})

        if(DEFINED TR_XNNPACK_LIBRARY_DIR)
            target_link_directories(renaissance PUBLIC ${TR_XNNPACK_LIBRARY_DIR})
            # XNNPACK 的依赖库（pthreadpool、cpuinfo）可能放在子目录中
            if(EXISTS "${TR_XNNPACK_LIBRARY_DIR}/pthreadpool")
                target_link_directories(renaissance PUBLIC "${TR_XNNPACK_LIBRARY_DIR}/pthreadpool")
            endif()
            if(EXISTS "${TR_XNNPACK_LIBRARY_DIR}/cpuinfo")
                target_link_directories(renaissance PUBLIC "${TR_XNNPACK_LIBRARY_DIR}/cpuinfo")
            endif()

            if(MSVC)
                target_link_libraries(renaissance PUBLIC
                    XNNPACK
                    pthreadpool
                    cpuinfo
                    xnnpack-microkernels-prod
                )
                target_link_options(renaissance PUBLIC
                    "/WHOLEARCHIVE:XNNPACK.lib"
                    "/WHOLEARCHIVE:pthreadpool.lib"
                    "/WHOLEARCHIVE:cpuinfo.lib"
                    "/WHOLEARCHIVE:xnnpack-microkernels-prod.lib"
                )
                target_compile_definitions(renaissance PRIVATE _CRT_SECURE_NO_WARNINGS)
                message(STATUS "Renaissance library: XNNPACK linked (MSVC WHOLEARCHIVE) from ${TR_XNNPACK_LIBRARY_DIR}")
            else()
                target_link_libraries(renaissance PUBLIC XNNPACK pthreadpool)
                message(STATUS "Renaissance library: XNNPACK linked (dynamic) from ${TR_XNNPACK_LIBRARY_DIR}")
            endif()
        else()
            message(WARNING "TR_USE_XNNPACK is ON but TR_XNNPACK_LIBRARY_DIR not defined")
        endif()
    else()
        message(WARNING "TR_USE_XNNPACK is ON but TR_XNNPACK_INCLUDE_DIR not defined")
    endif()
endif()

# ============================================================================
# 编译选项
# ============================================================================
if(NOT MSVC)
    target_compile_options(renaissance PRIVATE -Wno-unused-function)
endif()

target_compile_definitions(renaissance PRIVATE
    TR_DATA_VERSION_MAJOR=3
    TR_DATA_VERSION_MINOR=6
    TR_DATA_VERSION_PATCH=0
    TR_LOG_LEVEL=1  # INFO级别，用于调试
)

message(STATUS "Renaissance unified library created successfully")
message(STATUS "  - Core sources: ${RENAISSANCE_SOURCES}")
