# ============================================================================
# renAIssance Framework - Loading Performance Tests
# ============================================================================

message(STATUS "=== Configuring Loading Performance Tests ===")

# 设置输出目录
set(LOADING_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/tests/loading)

# 测试源文件列表
set(LOADING_TEST_SOURCES
    c.cpp
    d.cpp
    e.cpp
    e_v4.cpp
)

# ============================================================================
# 构建所有 Loading 测试可执行文件
# ============================================================================

foreach(test_source ${LOADING_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    # 创建可执行文件名
    set(executable_name test_expert_${test_name})

    message(STATUS "Creating Loading test executable: ${executable_name}")

    # 创建可执行文件
    add_executable(${executable_name} ${test_source})

    # 设置 C++17 标准
    set_property(TARGET ${executable_name} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${executable_name} PROPERTY CXX_STANDARD_REQUIRED ON)

    # 设置输出目录
    set_target_properties(${executable_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${LOADING_OUTPUT_DIR}
        FOLDER "tests/loading"
        POSITION_INDEPENDENT_CODE ON
    )

    # 跨平台编译设置
    if(MSVC)
        # Windows MSVC: 使用 UTF-8 编码
        target_compile_options(${executable_name} PRIVATE /utf-8)

        # Windows: 启用优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${executable_name} PRIVATE /O2)
            set_target_properties(${executable_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION TRUE
            )
        else()
            target_compile_options(${executable_name} PRIVATE /W3)
        endif()
    else()
        # Linux/Unix: 链接 pthread
        target_link_libraries(${executable_name} PRIVATE pthread)

        # 启用优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${executable_name} PRIVATE -O3 -march=native)
            set_target_properties(${executable_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION TRUE
            )
        else()
            target_compile_options(${executable_name} PRIVATE -Wall -Wextra -g)
        endif()
    endif()

    message(STATUS "  - ${executable_name} configured successfully")
endforeach()

# ============================================================================
# 测试说明
# ============================================================================
message(STATUS "")
message(STATUS "Loading Performance Tests:")
message(STATUS "  c.cpp - EXPERT_CG: C++ filesystem方案，线程绑定CPU核心")
message(STATUS "  d.cpp - EXPERT_SN: Direct I/O优化，posix_memalign")
message(STATUS "  e.cpp - EXPERT_OP: Direct I/O优化，FILE_FLAG_SEQUENTIAL_SCAN")
message(STATUS "  e_v4.cpp - 当前最优方案：全局洗牌 + 顺序读取（无需Direct I/O）")
message(STATUS "")
message(STATUS "=== Loading Performance Tests Configuration Complete ===")
message(STATUS "Test executables will be built in: ${LOADING_OUTPUT_DIR}")
