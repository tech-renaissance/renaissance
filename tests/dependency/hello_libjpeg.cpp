#include <iostream>
#include <vector>
#include <turbojpeg.h>  // libjpeg-turbo 的主头文件

int main() {
    // ==================== 初始化 ====================
    tjhandle compressor = nullptr;
    tjhandle decompressor = nullptr;

    // 图像宽高和通道数（RGB3通道）
    const int width = 4;
    const int height = 4;
    const int channels = 3; // RGB

    // ==================== 构造原始图像数据 ====================
    // 我们生成一个4x4的纯红色图像
    std::vector<unsigned char> rgbBuf(width * height * channels);
    for (int i = 0; i < width * height; ++i) {
        rgbBuf[i * 3 + 0] = 255; // R
        rgbBuf[i * 3 + 1] = 0;   // G
        rgbBuf[i * 3 + 2] = 0;   // B
    }

    // ==================== 压缩成 JPEG ====================
    unsigned char* jpegBuf = nullptr;  // 注意这个由 libjpeg-turbo 分配
    unsigned long jpegSize = 0;        // 输出大小（字节）
    int jpegQual = 90;                 // JPEG质量(1-100)

    compressor = tjInitCompress();     // 创建压缩句柄
    if (!compressor) {
        std::cerr << "Compressor init failed!" << std::endl;
        return 1;
    }

    // 执行压缩
    if (tjCompress2(compressor,
                    rgbBuf.data(),
                    width,
                    0,      // pitch=0表示默认 width * #channels
                    height,
                    TJPF_RGB, // 输入格式
                    &jpegBuf,
                    &jpegSize,
                    TJSAMP_444, // 采样方式
                    jpegQual,
                    TJFLAG_FASTDCT) != 0) {
        std::cerr << "Compression failed: " << tjGetErrorStr() << std::endl;
        tjDestroy(compressor);
        return 1;
    }

    std::cout << "JPEG compression done. Size = " << jpegSize << " bytes" << std::endl;

    // ==================== 解压缩回RGB ====================
    decompressor = tjInitDecompress();
    if (!decompressor) {
        std::cerr << "Decompressor init failed!" << std::endl;
        tjFree(jpegBuf);
        tjDestroy(compressor);
        return 1;
    }

    int jpegWidth = 0, jpegHeight = 0, jpegSubsamp = 0, jpegColorspace = 0;
    if (tjDecompressHeader3(decompressor, jpegBuf, jpegSize,
                            &jpegWidth, &jpegHeight, &jpegSubsamp, &jpegColorspace) != 0) {
        std::cerr << "Decompress header failed: " << tjGetErrorStr() << std::endl;
        tjFree(jpegBuf);
        tjDestroy(compressor);
        tjDestroy(decompressor);
        return 1;
    }

    std::vector<unsigned char> rgbOut(jpegWidth * jpegHeight * channels);

    if (tjDecompress2(decompressor,
                      jpegBuf,
                      jpegSize,
                      rgbOut.data(),
                      jpegWidth,
                      0,  // pitch
                      jpegHeight,
                      TJPF_RGB,
                      TJFLAG_FASTDCT) != 0) {
        std::cerr << "Decompression failed: " << tjGetErrorStr() << std::endl;
        tjFree(jpegBuf);
        tjDestroy(compressor);
        tjDestroy(decompressor);
        return 1;
    }

    std::cout << "JPEG decompression done. Image info:" << std::endl;
    std::cout << "Width = " << jpegWidth << ", Height = " << jpegHeight << std::endl;
    std::cout << "First pixel RGB = ("
              << (int)rgbOut[0] << ", "
              << (int)rgbOut[1] << ", "
              << (int)rgbOut[2] << ")" << std::endl;

    // ==================== 释放资源 ====================
    tjFree(jpegBuf);
    tjDestroy(compressor);
    tjDestroy(decompressor);

    std::cout << "All done. Hello libjpeg-turbo!" << std::endl;
    return 0;
}
