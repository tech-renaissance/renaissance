# NUMA测试程序
# 独立于renAIssance框架的NUMA特性测试

cmake_minimum_required(VERSION 3.22)

# 只保留成功的方案：B（原子操作方案）
foreach(FILENAME b)
    set(TARGET_NAME numa_${FILENAME})
    add_executable(${TARGET_NAME} ${FILENAME}.cpp)

    # 设置C++17标准
    target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

    # 添加编译选项
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wextra -O2
        )
    elseif(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            /W4 /O2
        )
    endif()

    # Linux平台需要链接pthread
    if(UNIX AND NOT APPLE)
        target_link_libraries(${TARGET_NAME} PRIVATE pthread)
    endif()
endforeach()
