#include <iostream>
#include <string>
#include <zlib.h>

int main() {
    // 原始数据
    const std::string original_data = "Hello, zlib! This is a test string for compression.";
    
    // 打印原始数据
    std::cout << "Original data: " << original_data << std::endl;
    std::cout << "Original size: " << original_data.size() << " bytes" << std::endl;

    // 计算压缩缓冲区大小 (通常为原始大小的1.1倍+12字节)
    uLongf compressed_size = static_cast<uLongf>(compressBound(static_cast<uLong>(original_data.size())));
    Bytef* compressed_data = new Bytef[compressed_size];

    // 压缩数据
    int compress_result = compress(compressed_data, &compressed_size,
                                 reinterpret_cast<const Bytef*>(original_data.c_str()),
                                 static_cast<uLong>(original_data.size()));

    // 检查压缩是否成功
    if (compress_result != Z_OK) {
        std::cerr << "Compression failed with error code: " << compress_result << std::endl;
        delete[] compressed_data;
        return 1;
    }

    // 打印压缩结果
    std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << (double)compressed_size/original_data.size()*100 << "%" << std::endl;

    // 准备解压缩缓冲区
    uLongf uncompressed_size = static_cast<uLongf>(original_data.size());
    Bytef* uncompressed_data = new Bytef[uncompressed_size];
    
    // 解压缩数据
    int uncompress_result = uncompress(uncompressed_data, &uncompressed_size, 
                                      compressed_data, compressed_size);
    
    // 检查解压缩是否成功
    if (uncompress_result != Z_OK) {
        std::cerr << "Decompression failed with error code: " << uncompress_result << std::endl;
        delete[] compressed_data;
        delete[] uncompressed_data;
        return 1;
    }
    
    // 打印解压缩结果
    std::string uncompressed_str((char*)uncompressed_data, uncompressed_size);
    std::cout << "Decompressed data: " << uncompressed_str << std::endl;
    
    // 验证数据是否一致
    if (original_data == uncompressed_str) {
        std::cout << "Verification successful: Original and decompressed data match!" << std::endl;
    } else {
        std::cout << "Verification failed: Data mismatch!" << std::endl;
    }
    
    // 释放内存
    delete[] compressed_data;
    delete[] uncompressed_data;
    
    return 0;
}
