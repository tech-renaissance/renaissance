# ============================================================================
# renAIssance Framework Integration Tests CMake Configuration
# ============================================================================

# 检查是否在父项目中
if(NOT PROJECT_NAME STREQUAL "renAIssance")
    message(FATAL_ERROR "integration tests must be included from the parent project")
endif()

message(STATUS "=== Integration Tests Configuration ===")

# ============================================================================
# Python print tensor test
# ============================================================================

if(TR_USE_PYTHON_SESSION)
    add_executable(test_python_print
        test_python_print.cpp
    )

    target_link_libraries(test_python_print
        renaissance
    )

    # 设置输出目录到tests/integration子目录
    set_target_properties(test_python_print PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    )

    add_test(NAME test_python_print COMMAND test_python_print)

    message(STATUS "Python print tensor test configured: test_python_print")
else()
    message(STATUS "Python print tensor test skipped (TR_USE_PYTHON_SESSION not defined)")
endif()

# ============================================================================
# Single GPU async pipeline test
# ============================================================================

if(TR_USE_CUDA)
    add_executable(test_async_pipeline_single_gpu
        test_async_pipeline_single_gpu.cpp
    )

    target_link_libraries(test_async_pipeline_single_gpu
        renaissance
    )

    set_target_properties(test_async_pipeline_single_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    )

    add_test(NAME test_async_pipeline_single_gpu COMMAND test_async_pipeline_single_gpu)

    message(STATUS "Single GPU async pipeline test configured: test_async_pipeline_single_gpu")
endif()

# ============================================================================
# Dual GPU async pipeline test
# ============================================================================

if(TR_USE_CUDA AND TR_USE_NCCL)
    add_executable(test_async_pipeline_dual_gpu
        test_async_pipeline_dual_gpu.cpp
    )

    target_link_libraries(test_async_pipeline_dual_gpu
        renaissance
    )

    set_target_properties(test_async_pipeline_dual_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    )

    add_test(NAME test_async_pipeline_dual_gpu COMMAND test_async_pipeline_dual_gpu)

    message(STATUS "Dual GPU async pipeline test configured: test_async_pipeline_dual_gpu")
endif()

# ============================================================================
# Single MUSA GPU async pipeline test
# ============================================================================

if(TR_USE_MUSA)
    add_executable(test_async_pipeline_single_musa
        test_async_pipeline_single_musa.cpp
    )

    target_link_libraries(test_async_pipeline_single_musa
        renaissance
    )

    set_target_properties(test_async_pipeline_single_musa PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
    )

    add_test(NAME test_async_pipeline_single_musa COMMAND test_async_pipeline_single_musa)

    message(STATUS "Single MUSA GPU async pipeline test configured: test_async_pipeline_single_musa")
endif()

# ============================================================================
# ImageNet Data Loader Test
# ============================================================================

# ========== 临时禁用：test_imagenet_loader使用旧代码，需要更新 ==========
# add_executable(test_imagenet_loader
#     test_imagenet_loader.cpp
# )
#
# target_link_libraries(test_imagenet_loader
#     renaissance
# )
#
# set_target_properties(test_imagenet_loader PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
# )
#
# # 注意：不添加为CTest，因为它需要外部数据集
# message(STATUS "ImageNet data loader test configured: test_imagenet_loader")

# ============================================================================
# Preprocessor V4.0 Performance Test
# ============================================================================

add_executable(test_dataloader_performance
    test_dataloader_performance.cpp
)

target_link_libraries(test_dataloader_performance
    renaissance
)

set_target_properties(test_dataloader_performance PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration
)

# 注意：不添加为CTest，因为它需要外部数据集
message(STATUS "Preprocessor V4.0 performance test configured: test_dataloader_performance")

message(STATUS "=== Integration Tests Configuration Complete ===")
