# ============================================================================
# PW测试模块
# ============================================================================

# 定义test_po可执行文件
add_executable(test_po test_po.cpp)

# 链接renaissance库
target_link_libraries(test_po PRIVATE renaissance)

# 设置输出目录
set_target_properties(test_po PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# 安装目标
install(TARGETS test_po RUNTIME DESTINATION bin/tests/pw)
