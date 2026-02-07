#include <iostream>
#include <zlib.h>
#include <iomanip>
#include <fstream>

int main() {
    // 测试数据
    const char* test_data = "Hello, World!";
    
    // C++ zlib 的 crc32 用法
    uLong crc = crc32(0L, Z_NULL, 0);  // 初始化
    crc = crc32(crc, (const Bytef*)test_data, strlen(test_data));
    
    std::cout << "C++ CRC32: " << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << crc << std::endl;
    
    return 0;
}
