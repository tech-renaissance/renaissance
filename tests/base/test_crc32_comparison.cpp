/**
 * @file test_crc32_comparison.cpp
 * @brief 对比 C++ zlib 和 Python zlib 的 CRC32 计算结果
 * @details 测试同一个文件的 CRC32 值，确保两者一致
 * @version 1.0.0
 * @date 2026-02-05
 * @author 技术觉醒团队
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdint>
#include <zlib.h>

/**
 * @brief 计算文件的CRC32值（使用zlib）
 * @param filepath 文件路径
 * @return CRC32值（8位16进制大写字符串）
 */
std::string calculate_file_crc32(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file: " << filepath << std::endl;
        return "";
    }

    // 读取文件内容
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    // 计算 CRC32（与 Python 相同的方式）
    uLong crc = crc32(0L, Z_NULL, 0);  // 初始化
    crc = crc32(crc, buffer.data(), static_cast<uInt>(file_size));

    // 转换为 8 位 16 进制大写字符串
    char crc_str[16];
    snprintf(crc_str, sizeof(crc_str), "%08X", static_cast<uint32_t>(crc));

    return std::string(crc_str);
}

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <file_path>\n\n"
              << "Example:\n"
              << "  " << program_name << " T:/Dataset/imagenet/train/n01440764/n01440764_10026.JPEG\n\n"
              << "Output:\n"
              << "  File: <file_path>\n"
              << "  Size: <file_size> bytes\n"
              << "  CRC32: <8-digit-hex>\n\n"
              << "Note: Compare this with Python's zlib.crc32() result.\n"
              << "      Python command: python -c \"import zlib; print(f'{zlib.crc32(open('file').read()) & 0xffffffff:08X}')\"\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filepath = argv[1];

    std::cout << "========================================================================\n";
    std::cout << "CRC32 Comparison Test: C++ (zlib) vs Python (zlib)\n";
    std::cout << "========================================================================\n\n";

    // 计算 CRC32
    std::string crc32 = calculate_file_crc32(filepath);

    if (crc32.empty()) {
        std::cerr << "Error: Failed to calculate CRC32\n";
        return 1;
    }

    // 获取文件大小
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Cannot get file size\n";
        return 1;
    }
    size_t file_size = file.tellg();

    // 打印结果
    std::cout << "File:   " << filepath << "\n";
    std::cout << "Size:   " << file_size << " bytes\n";
    std::cout << "CRC32:  " << crc32 << "\n\n";

    std::cout << "========================================================================\n";
    std::cout << "C++ Result: " << crc32 << "\n\n";
    std::cout << "To verify with Python, run:\n";
    std::cout << "  python -c \"import zlib; data = open('" << filepath << "', 'rb').read(); ";
    std::cout << "print(f'{zlib.crc32(data) & 0xffffffff:08X}')\"\n";
    std::cout << "========================================================================\n";

    return 0;
}
