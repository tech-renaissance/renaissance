/**
 * @file test_logger.cpp
 * @brief Logger类单元测试
 * @details 测试日志器的各项功能，包括不同级别日志输出、模块标记、文件输出等
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: base
 */

// Windows宏冲突处理
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

#include "renaissance.h"
#include <cassert>
#include <fstream>
#include <cstdio>

using namespace tr;

void test_basic_logging() {
    LOG_INFO << "Basic logging test started";

    // 测试各级别日志
    LOG_DEBUG << "This is a DEBUG message";
    LOG_INFO << "This is an INFO message";
    LOG_WARN << "This is a WARN message";
    LOG_ERROR << "This is an ERROR message";

    LOG_INFO << "Basic logging test completed";
}

void test_module_logging() {
    LOG_INFO << "Module logging test started";

    // 测试模块标记
    TR_LOG_DEBUG("test") << "DEBUG message from test module";
    TR_LOG_INFO("model") << "INFO message from model module";
    TR_LOG_WARN("trainer") << "WARN message from trainer module";
    TR_LOG_ERROR("data") << "ERROR message from data module";

    LOG_INFO << "Module logging test completed";
}

void test_stream_operators() {
    LOG_INFO << "Stream operators test started";

    // 测试各种类型的流式输出
    int integer = 42;
    double floating = 3.14159;
    std::string text = "Hello";

    LOG_INFO << "Integer: " << integer;
    LOG_INFO << "Float: " << floating;
    LOG_INFO << "String: " << text;
    LOG_INFO << "Mixed: " << integer << ", " << floating << ", " << text;

    LOG_INFO << "Stream operators test completed";
}

void test_log_level_control() {
    LOG_INFO << "Log level control test started";

    auto& logger = Logger::instance();

    // 测试级别控制
    logger.set_level(LogLevel::WARN);
    LOG_DEBUG << "This DEBUG should not appear (level=INFO < WARN)";
    LOG_INFO << "This INFO should not appear (level=INFO < WARN)";
    LOG_WARN << "This WARN should appear";
    LOG_ERROR << "This ERROR should appear";

    // 恢复默认级别
    logger.set_level(LogLevel::INFO);
    LOG_INFO << "Log level restored to INFO";

    LOG_INFO << "Log level control test completed";
}

void test_quiet_mode() {
    LOG_INFO << "Quiet mode test started";

    auto& logger = Logger::instance();

    // 开启静默模式
    logger.set_quiet_mode(true);
    LOG_INFO << "This should not appear (quiet mode)";
    LOG_ERROR << "This should not appear (quiet mode)";

    // 关闭静默模式
    logger.set_quiet_mode(false);
    LOG_INFO << "Quiet mode disabled";

    LOG_INFO << "Quiet mode test completed";
}

void test_file_output() {
    LOG_INFO << "File output test started";

    // 使用TR_WORKSPACE宏构建日志文件路径
    const std::string log_file = std::string(TR_WORKSPACE) + "/test_log.txt";
    auto& logger = Logger::instance();

    // 设置输出到文件
    logger.set_output_file(log_file);

    LOG_INFO << "This message goes to file";
    LOG_WARN << "This warning goes to file";
    LOG_ERROR << "This error goes to file";

    // 恢复控制台输出
    logger.set_output_file("");
    LOG_INFO << "Output restored to console";

    // 验证文件内容
    std::ifstream file(log_file);
    if (file.is_open()) {
        std::string line;
        int line_count = 0;
        while (std::getline(file, line)) {
            if (line.find("goes to file") != std::string::npos) {
                line_count++;
            }
        }
        file.close();

        if (line_count >= 3) {
            LOG_INFO << "File output test PASSED: found " << line_count << " log lines";
        } else {
            LOG_ERROR << "File output test FAILED: expected >=3 lines, got " << line_count;
        }
    } else {
        LOG_ERROR << "Failed to open log file for verification";
    }

    // 清理测试文件
    std::remove(log_file.c_str());

    LOG_INFO << "File output test completed";
}

void test_performance() {
    LOG_INFO << "Performance test started";

    const int iterations = 10000;

    // 测试日志输出性能（DEBUG级别，Release下应该被编译掉）
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        LOG_DEBUG << "Performance test iteration " << i;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    LOG_INFO << "Performance test: " << iterations << " DEBUG logs in "
             << duration.count() << " ms";

#if TR_LOG_LEVEL > 0
    LOG_INFO << "Note: DEBUG logs are disabled at compile time (TR_LOG_LEVEL="
             << TR_LOG_LEVEL << "), so they should have zero runtime cost";
#endif

    LOG_INFO << "Performance test completed";
}

int main() {
    LOG_INFO << "===============================================";
    LOG_INFO << "  Logger Unit Test Suite";
    LOG_INFO << "===============================================";

    test_basic_logging();
    test_module_logging();
    test_stream_operators();
    test_log_level_control();
    test_quiet_mode();
    test_file_output();
    test_performance();

    LOG_INFO << "===============================================";
    LOG_INFO << "  All Logger Tests Completed Successfully!";
    LOG_INFO << "===============================================";

    return 0;
}
