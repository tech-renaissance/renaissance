# 收集数据测试源文件
set(DATA_TEST_SOURCES
    test_shape.cpp
    test_storage_ownership.cpp
    test_print.cpp
    # V4.0.0 DataLoader测试
    test_performance.cpp
    # V4.0.0 PARTIAL模式测试
    test_partial_mode.cpp
    # V4.1.0 Preprocessor功能测试（decode + crop）
    test_preproc.cpp
    # V4.0.0 FULLY模式测试
    test_fully_mode.cpp
    # V4.0.0 可复现性测试（使用新的Preprocessor）
    test_reproducibility.cpp
    # V4.0.0 跨epoch可复现性测试（FULLY模式）
    test_cross_epoch_reproducibility.cpp
    # V3.8.3 CRC-32验证测试
    test_crc_verification.cpp
    # V3.9.0 MNIST和CIFAR DTS测试
    test_mnist_dts.cpp
    test_cifar_dts.cpp
    # V4.0.0 MNIST和CIFAR RAW测试
    test_mnist_raw.cpp
    test_cifar_raw.cpp
    # V4.0.0 MNIST和CIFAR RAW可复现性测试
    test_mnist_cifar_raw_reproducibility.cpp
    # V4.0.0 共享缓冲区测试
    test_shared_buffers.cpp
    # V1.0.0 ImageNet RAW Loader测试
    test_raw_partial_mode.cpp
    test_raw_fully_mode.cpp
    test_raw_reproducibility.cpp
    test_raw_cross_epoch_reproducibility.cpp
)

# 创建data测试可执行文件
foreach(test_source ${DATA_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})

    # 链接renaissance统一库（包含base、data、device及所有依赖）
    target_link_libraries(${test_name}
        PRIVATE
            renaissance
    )

    # 包含头文件目录
    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    # 设置测试可执行文件的输出目录
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/data
    )

    # 设置属性
    set_target_properties(${test_name} PROPERTIES
        FOLDER "tests/data"
        POSITION_INDEPENDENT_CODE ON
    )

endforeach()

# 打印data测试配置信息
message(STATUS "Data component tests configured successfully")
message(STATUS "Data test sources: ${DATA_TEST_SOURCES}")
message(STATUS "Data test binaries will be in: ${CMAKE_BINARY_DIR}/bin/tests/data")
