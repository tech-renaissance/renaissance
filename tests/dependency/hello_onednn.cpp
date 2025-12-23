#include <iostream>
#include <vector>
#include <stdexcept>
#include "dnnl.hpp"

using namespace dnnl;

int main() {
    try {
        std::cout << "=== oneDNN Hello World ===" << std::endl;

        // 1. 创建执行引擎（使用CPU）
        engine eng(engine::kind::cpu, 0);
        std::cout << "Engine created successfully (CPU)" << std::endl;

        // 2. 创建执行流
        stream s(eng);
        std::cout << "Stream created successfully" << std::endl;

        // 3. 定义张量维度：1个批次，3个通道，4x4的图像
        const int N = 1;  // 批次大小
        const int C = 3;  // 通道数
        const int H = 4;  // 高度
        const int W = 4;  // 宽度

        // 4. 创建内存描述符（NCHW格式）
        memory::dims src_dims = {N, C, H, W};
        memory::desc src_md(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        std::cout << "Memory descriptor created: " << N << "x" << C << "x" << H << "x" << W << std::endl;

        // 5. 分配内存并初始化数据
        memory src_mem(src_md, eng);
        float* src_data = static_cast<float*>(src_mem.get_data_handle());

        // 用简单的递增值填充数据
        for (int i = 0; i < N * C * H * W; ++i) {
            src_data[i] = static_cast<float>(i);
        }
        std::cout << "Source memory allocated and initialized" << std::endl;

        // 6. 创建ReLU激活函数（简化版，直接使用memory）
        memory::desc dst_md = src_md;
        memory dst_mem(dst_md, eng);
        std::cout << "Destination memory allocated" << std::endl;

        // 7. 执行ReLU操作 (f(x) = max(0, x))
        float* dst_data = static_cast<float*>(dst_mem.get_data_handle());
        for (int i = 0; i < N * C * H * W; ++i) {
            dst_data[i] = std::max(0.0f, src_data[i]);
        }
        std::cout << "ReLU operation executed successfully" << std::endl;

        // 8. 验证结果（打印前10个输入和输出值）
        std::cout << "\n=== Verification (first 10 elements) ===" << std::endl;
        std::cout << "Index\tInput\tOutput (ReLU)" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << i << "\t" << src_data[i] << "\t" << dst_data[i] << std::endl;
        }

        // 9. 简单验证逻辑正确性
        bool success = true;
        for (int i = 0; i < N * C * H * W; ++i) {
            float expected = (src_data[i] > 0.0f) ? src_data[i] : 0.0f;
            if (dst_data[i] != expected) {
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "\n=== SUCCESS: All results are correct! ===" << std::endl;
            std::cout << "oneDNN library is working correctly!" << std::endl;
        } else {
            std::cout << "\n=== ERROR: Results verification failed! ===" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        // 捕获所有标准异常
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        // 捕获其他异常
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}
