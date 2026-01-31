# ============================================================================
# renAIssance Framework - SIMD RandomResizedCrop 性能测试
# ============================================================================

message(STATUS "=== Configuring SIMD Performance Tests ===")

# 设置输出目录
set(SIMD_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests/simd)

# 测试源文件列表
set(SIMD_TEST_SOURCES
    baseline.cpp
    fast_resize.cpp
    fast_crop.cpp
)

# ============================================================================
# 查找依赖库：Simd 和 libjpeg-turbo
# ============================================================================

# 查找 Simd 库（使用 vcpkg 配置的路径）
if(DEFINED TR_SIMD_INCLUDE_DIR AND DEFINED TR_SIMD_LIBRARY_DIR)
    set(SIMD_FOUND TRUE)
    set(SIMD_INCLUDE_DIRS ${TR_SIMD_INCLUDE_DIR})
    set(SIMD_LIBRARIES Simd)
    set(SIMD_LIBRARY_DIRS ${TR_SIMD_LIBRARY_DIR})
    message(STATUS "Found Simd: ${TR_SIMD_INCLUDE_DIR}")
else()
    set(SIMD_FOUND FALSE)
    message(WARNING "TR_SIMD paths not defined - SIMD tests will not be built")
endif()

# 查找 libjpeg-turbo（使用 vcpkg 配置的路径）
if(DEFINED TR_JPEG_INCLUDE_DIR AND DEFINED TR_JPEG_LIBRARY_DIR)
    set(JPEG_FOUND TRUE)
    set(JPEG_INCLUDE_DIRS ${TR_JPEG_INCLUDE_DIR})
    # 使用 turbojpeg 库（TurboJPEG API）
    set(JPEG_LIBRARIES turbojpeg)
    set(JPEG_LIBRARY_DIRS ${TR_JPEG_LIBRARY_DIR})
    message(STATUS "Found libjpeg-turbo: ${TR_JPEG_INCLUDE_DIR}")
else()
    set(JPEG_FOUND FALSE)
    message(WARNING "TR_JPEG paths not defined - SIMD tests will not be built")
endif()

# ============================================================================
# 构建所有 SIMD 测试可执行文件
# ============================================================================

foreach(test_source ${SIMD_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    # 检查依赖是否可用
    if(NOT SIMD_FOUND OR NOT JPEG_FOUND)
        message(WARNING "Skipping ${test_name}: Required dependencies (Simd or libjpeg-turbo) not found")
        continue()
    endif()

    message(STATUS "Creating SIMD test executable: ${test_name}")

    # 创建可执行文件
    add_executable(${test_name} ${test_source})

    # 设置 C++17 标准
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD_REQUIRED ON)

    # 设置输出目录
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${SIMD_OUTPUT_DIR}
        FOLDER "tests/simd"
        POSITION_INDEPENDENT_CODE ON
    )

    # 添加 Simd 库的 include 目录和链接
    target_include_directories(${test_name} PRIVATE ${SIMD_INCLUDE_DIRS})
    target_link_directories(${test_name} PRIVATE ${SIMD_LIBRARY_DIRS})
    target_link_libraries(${test_name} PRIVATE ${SIMD_LIBRARIES})

    # 添加 libjpeg-turbo 的 include 目录和链接
    target_include_directories(${test_name} PRIVATE ${JPEG_INCLUDE_DIRS})
    target_link_directories(${test_name} PRIVATE ${JPEG_LIBRARY_DIRS})

    # f.cpp 和 g.cpp 需要额外的 libjpeg 库
    if(test_name STREQUAL "f" OR test_name STREQUAL "g")
        target_link_libraries(${test_name} PRIVATE ${JPEG_LIBRARIES} jpeg)
    else()
        target_link_libraries(${test_name} PRIVATE ${JPEG_LIBRARIES})
    endif()

    # 跨平台编译设置
    if(MSVC)
        # Windows MSVC: 使用 UTF-8 编码
        target_compile_options(${test_name} PRIVATE /utf-8)

        # Windows: 启用优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set_target_properties(${test_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION TRUE
            )
        endif()
    else()
        # Linux/Unix: 启用优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set_target_properties(${test_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION TRUE
            )
        endif()
    endif()

    message(STATUS "  - ${test_name} configured successfully")
endforeach()

message(STATUS "=== SIMD Performance Tests Configuration Complete ===")
message(STATUS "Test executables will be built in: ${SIMD_OUTPUT_DIR}")
