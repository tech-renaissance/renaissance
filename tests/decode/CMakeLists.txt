# =============================================================================
# renAIssance Framework - Decode Performance Tests
# =============================================================================

message(STATUS "=== Configuring Decode Performance Tests ===")

# 设置输出目录
set(DECODE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests/decode)

# 测试源文件列表
set(DECODE_TEST_SOURCES
    b.cpp
    c.cpp
    d.cpp
    e.cpp
)

# =============================================================================
# 查找依赖库：libjpeg-turbo
# =============================================================================

# 查找 libjpeg-turbo（使用 vcpkg 配置的路径）
if(DEFINED TR_JPEG_INCLUDE_DIR AND DEFINED TR_JPEG_LIBRARY_DIR)
    set(JPEG_FOUND TRUE)
    set(JPEG_INCLUDE_DIRS ${TR_JPEG_INCLUDE_DIR})
    set(JPEG_LIBRARIES turbojpeg jpeg)
    set(JPEG_LIBRARY_DIRS ${TR_JPEG_LIBRARY_DIR})
    message(STATUS "Found libjpeg-turbo: ${TR_JPEG_INCLUDE_DIR}")
else()
    set(JPEG_FOUND FALSE)
    message(WARNING "TR_JPEG paths not defined - Decode tests will not be built")
endif()

# =============================================================================
# 构建所有 Decode 测试可执行文件
# =============================================================================

foreach(test_source ${DECODE_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    # 检查依赖是否可用
    if(NOT JPEG_FOUND)
        message(WARNING "Skipping ${test_name}: Required dependency (libjpeg-turbo) not found")
        continue()
    endif()

    # 添加前缀避免与其他测试冲突
    set(target_name "jpeg_decode_${test_name}")

    message(STATUS "Creating decode test executable: ${target_name}")

    # 创建可执行文件
    add_executable(${target_name} ${test_source})

    # 设置 C++17 标准
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)

    # 设置输出目录和输出文件名（去掉前缀，保持简洁）
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${DECODE_OUTPUT_DIR}
        OUTPUT_NAME ${test_name}
        FOLDER "tests/decode"
        POSITION_INDEPENDENT_CODE ON
    )

    # 添加 libjpeg-turbo 的 include 目录和链接
    target_include_directories(${target_name} PRIVATE ${JPEG_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${JPEG_LIBRARY_DIRS})
    target_link_libraries(${target_name} PRIVATE ${JPEG_LIBRARIES})

    # 跨平台编译设置
    if(MSVC)
        # Windows MSVC: 使用 UTF-8 编码
        target_compile_options(${target_name} PRIVATE /utf-8)

        # Windows: 启用优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set_target_properties(${target_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION TRUE
            )
        endif()
    else()
        # Linux/Unix: 启用优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set_target_properties(${target_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION TRUE
            )
        endif()
    endif()

    message(STATUS "  - ${target_name} configured successfully")
endforeach()

message(STATUS "=== Decode Performance Tests Configuration Complete ===")
message(STATUS "Test executables will be built in: ${DECODE_OUTPUT_DIR}")
