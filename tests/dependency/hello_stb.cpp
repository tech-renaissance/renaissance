#include <iostream>
#include <vector>
#include <string>

// ==== stb 单头文件库 ====
// STB_IMAGE_IMPLEMENTATION 和 STB_IMAGE_WRITE_IMPLEMENTATION
// 已在 CMakeLists.txt 中通过 target_compile_definitions 定义

#include "stb_image.h"
#include "stb_image_write.h"

// STB_IMAGE_RESIZE2_IMPLEMENTATION 需要在这里定义
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb_image_resize2.h"

// ==== 主程序 ====
int main() {
    // 输入和输出图像路径（使用项目根目录）
    const std::string inputPath  = TR_PROJECT_ROOT "/docs/logo.jpg";
    const std::string outputPath = TR_WORKSPACE "/output.jpg";

    // 图像基本信息
    int width, height, channels;

    // 用 stbi_load 读入图像，stb_image 会自动判断格式
    unsigned char* inputImage = stbi_load(inputPath.c_str(), &width, &height, &channels, 0);
    if (!inputImage) {
        std::cerr << "Failed to load image: " << inputPath << std::endl;
        return 1;
    }

    std::cout << "Loaded image: " << width << "x" << height << " channels=" << channels << std::endl;

    // 设置输出图像尺寸（这里缩小为一半大小）
    int newWidth  = width  / 2;
    int newHeight = height / 2;

    // 分配空间保存输出图像
    std::vector<unsigned char> outputImage(newWidth * newHeight * channels);

    // 使用 stb_image_resize2 进行缩放
    // 参数：输入指针、输入宽高、输出指针、输出宽高、通道数（非空通道也要传，比如 RGB=3, RGBA=4）
    stbir_resize_uint8_linear(
        inputImage,   width,  height, 0,   // 输入参数
        outputImage.data(), newWidth, newHeight, 0, // 输出参数
        (stbir_pixel_layout)channels       // 通道排列，直接传通道数即可
    );

    std::cout << "Resize success: " << newWidth << "x" << newHeight << std::endl;

    // 将缩放后的图像保存为 JPG 文件
    int writeOK = stbi_write_jpg(outputPath.c_str(), newWidth, newHeight, channels, outputImage.data(), 90);
    if (writeOK == 0) {
        std::cerr << "Failed to write output image." << std::endl;
        stbi_image_free(inputImage);
        return 1;
    }

    std::cout << "Output saved to: " << outputPath << std::endl;

    // 释放输入图像内存
    stbi_image_free(inputImage);

    std::cout << "Done." << std::endl;
    return 0;
}
