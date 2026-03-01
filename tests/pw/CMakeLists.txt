# ============================================================================
# PW测试模块
# ============================================================================

# 定义test_two_po可执行文件
add_executable(test_two_po test_two_po.cpp)

# 链接renaissance库
target_link_libraries(test_two_po PRIVATE renaissance)

# 设置输出目录
set_target_properties(test_two_po PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# Windows下复制所需的DLL
if(WIN32)
    copy_test_dlls(test_two_po)
endif()

# 安装目标
install(TARGETS test_two_po RUNTIME DESTINATION bin/tests/pw)

# 定义test_pw_ultimate可执行文件
add_executable(test_pw_ultimate
    test_pw_ultimate.cpp
    # 额外编译engine_buffer.cpp以启用单元测试模式
    ${CMAKE_SOURCE_DIR}/src/data/engine_buffer.cpp
)

# 链接renaissance库（但data.lib中的engine_buffer.cpp会被上面的覆盖）
target_link_libraries(test_pw_ultimate PRIVATE renaissance)

# 为test_pw_ultimate目标定义宏
target_compile_definitions(test_pw_ultimate PRIVATE
    FOR_ENGINE_BUFFER_UNIT_TESTS_ONLY
)

# 添加data库所需的额外包含目录
target_include_directories(test_pw_ultimate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
)

# 设置输出目录
set_target_properties(test_pw_ultimate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# Windows下复制所需的DLL
if(WIN32)
    copy_test_dlls(test_pw_ultimate)
endif()

# 安装目标
install(TARGETS test_pw_ultimate RUNTIME DESTINATION bin/tests/pw)

# 定义test_engine_crc可执行文件
add_executable(test_engine_crc
    test_engine_crc.cpp
    # 额外编译engine_buffer.cpp以启用CSV功能
    ${CMAKE_SOURCE_DIR}/src/data/engine_buffer.cpp
)

# 链接renaissance库（但data.lib中的engine_buffer.cpp会被上面的覆盖）
target_link_libraries(test_engine_crc PRIVATE renaissance)

# 为test_engine_crc目标定义宏（开启所有三个功能）
target_compile_definitions(test_engine_crc PRIVATE
    VERIFY_ENGINE_BUFFER_CRC
    FOR_ENGINE_BUFFER_UNIT_TESTS_ONLY
    ENGINE_BUFFER_SAVE_FIRST_IMAGE
)

# 添加data库所需的额外包含目录
target_include_directories(test_engine_crc PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
)

# 设置输出目录
set_target_properties(test_engine_crc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# Windows下复制所需的DLL
if(WIN32)
    copy_test_dlls(test_engine_crc)
endif()

# 安装目标
install(TARGETS test_engine_crc RUNTIME DESTINATION bin/tests/pw)

# 定义test_progressive_resolution可执行文件
add_executable(test_progressive_resolution
    test_progressive_resolution.cpp
    # 额外编译engine_buffer.cpp以启用CSV功能和第一图像保存功能
    ${CMAKE_SOURCE_DIR}/src/data/engine_buffer.cpp
)

# 链接renaissance库（但data.lib中的engine_buffer.cpp会被上面的覆盖）
target_link_libraries(test_progressive_resolution PRIVATE renaissance)

# 为test_progressive_resolution目标定义宏（仅开启图像保存和单元测试）
target_compile_definitions(test_progressive_resolution PRIVATE
    FOR_ENGINE_BUFFER_UNIT_TESTS_ONLY
    ENGINE_BUFFER_SAVE_FIRST_IMAGE
)

# 添加data库所需的额外包含目录
target_include_directories(test_progressive_resolution PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
)

# 设置输出目录
set_target_properties(test_progressive_resolution PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# Windows下复制所需的DLL
if(WIN32)
    copy_test_dlls(test_progressive_resolution)
endif()

# 安装目标
install(TARGETS test_progressive_resolution RUNTIME DESTINATION bin/tests/pw)

# 定义test_dataloader_performance可执行文件
add_executable(test_dataloader_performance
    test_dataloader_performance.cpp
    # 重新编译preprocessor.cpp以启用TEST_WITHOUT_PW模式
    ${CMAKE_SOURCE_DIR}/src/data/preprocessor.cpp
)

# 链接renaissance库（但data.lib中的preprocessor.cpp会被上面的覆盖）
target_link_libraries(test_dataloader_performance PRIVATE renaissance)

# 为test_dataloader_performance目标定义TEST_WITHOUT_PW宏
target_compile_definitions(test_dataloader_performance PRIVATE
    TEST_WITHOUT_PW
)

# 添加data库所需的额外包含目录
target_include_directories(test_dataloader_performance PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
)

# 设置输出目录
set_target_properties(test_dataloader_performance PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# Windows下复制所需的DLL
if(WIN32)
    copy_test_dlls(test_dataloader_performance)
endif()

# 安装目标
install(TARGETS test_dataloader_performance RUNTIME DESTINATION bin/tests/pw)

# 定义test_epoch_crc可执行文件
add_executable(test_epoch_crc
    test_epoch_crc.cpp
    # 重新编译preprocessor.cpp以启用TEST_WITHOUT_PW模式
    ${CMAKE_SOURCE_DIR}/src/data/preprocessor.cpp
)

# 链接renaissance库（但data.lib中的preprocessor.cpp会被上面的覆盖）
target_link_libraries(test_epoch_crc PRIVATE renaissance)

# 为test_epoch_crc目标定义TEST_WITHOUT_PW宏
target_compile_definitions(test_epoch_crc PRIVATE
    TEST_WITHOUT_PW
)

# 添加data库所需的额外包含目录
target_include_directories(test_epoch_crc PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
)

# 设置输出目录
set_target_properties(test_epoch_crc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# Windows下复制所需的DLL
if(WIN32)
    copy_test_dlls(test_epoch_crc)
endif()

# 安装目标
install(TARGETS test_epoch_crc RUNTIME DESTINATION bin/tests/pw)
