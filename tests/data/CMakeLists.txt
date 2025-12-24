# 收集数据测试源文件
set(DATA_TEST_SOURCES
    test_shape.cpp
)

# 创建data测试可执行文件
foreach(test_source ${DATA_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})

    # 链接data库
    target_link_libraries(${test_name}
        PRIVATE
            data
            base
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
