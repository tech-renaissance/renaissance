/**
 * @file test_cuda_cast_perf.cpp
 * @brief CUDA数据类型转换性能测试
 * @version 3.6.18
 * @date 2026-01-03
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace tr;

/**
 * @brief 计时辅助类
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief 打印吞吐量
 */
void print_throughput(const std::string& name, size_t num_elements, double time_ms) {
    double throughput = static_cast<double>(num_elements) / (time_ms / 1000.0);
    double giga_throughput = throughput / 1e9;

    std::cout << "  " << std::left << std::setw(25) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << time_ms << " ms | "
              << std::setw(10) << std::setprecision(2)
              << giga_throughput << " G elems/s" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Cast Performance Test (V3.6.18)" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        auto& cpu = get_cpu();
        auto& cuda = DeviceManager::instance().cuda(0);

        // 测试配置：256 x 1024 x 1024 x 1 = 268,435,456 元素
        const int N = 256;
        const int H = 1024;
        const int W = 1024;
        const int C = 1;
        Shape shape{N, H, W, C};
        const size_t num_elements = N * H * W * C;

        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Shape: ["
                  << N << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        std::cout << "  Total elements: " << num_elements << " ("
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(num_elements) / 1e6 << " M)" << std::endl;

        // 1. 创建FP32张量（CPU上创建，然后传输到GPU）
        std::cout << "\n[1/9] Creating FP32 tensor (randn)..." << std::endl;
        Tensor fp32_cpu = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);
        Tensor fp32_gpu = cuda.empty(shape, DType::FP32);
        cpu.transfer_into(fp32_cpu, fp32_gpu);
        std::cout << "  Created: " << fp32_gpu.nbytes() / (1024.0 * 1024.0)
                  << " MB on GPU" << std::endl;

        // 2. 创建BF16张量
        std::cout << "\n[2/9] Creating BF16 tensor (zeros)..." << std::endl;
        Tensor bf16_gpu = cuda.zeros(shape, DType::BF16);
        std::cout << "  Created: " << bf16_gpu.nbytes() / (1024.0 * 1024.0)
                  << " MB on GPU" << std::endl;

        // 3. FP32 -> BF16
        std::cout << "\n[3/9] Testing FP32 -> BF16..." << std::endl;
        Timer timer;
        cuda.cast_into(fp32_gpu, bf16_gpu);
        cuda.synchronize();
        double time_fp32_to_bf16 = timer.elapsed_ms();
        print_throughput("FP32 -> BF16", num_elements, time_fp32_to_bf16);

        // 4. BF16 -> FP32
        std::cout << "\n[4/9] Testing BF16 -> FP32..." << std::endl;
        timer.reset();
        cuda.cast_into(bf16_gpu, fp32_gpu);
        cuda.synchronize();
        double time_bf16_to_fp32 = timer.elapsed_ms();
        print_throughput("BF16 -> FP32", num_elements, time_bf16_to_fp32);

        // 5. 创建INT32张量（CPU上创建，然后传输到GPU）
        std::cout << "\n[5/9] Creating INT32 tensor (randint)..." << std::endl;
        Tensor int32_cpu = cpu.randint(shape, -1000000, 1000000, DType::INT32);
        Tensor int32_gpu = cuda.empty(shape, DType::INT32);
        cpu.transfer_into(int32_cpu, int32_gpu);
        std::cout << "  Created: " << int32_gpu.nbytes() / (1024.0 * 1024.0)
                  << " MB on GPU" << std::endl;

        // 6. INT32 -> FP32
        std::cout << "\n[6/9] Testing INT32 -> FP32..." << std::endl;
        timer.reset();
        cuda.cast_into(int32_gpu, fp32_gpu);
        cuda.synchronize();
        double time_int32_to_fp32 = timer.elapsed_ms();
        print_throughput("INT32 -> FP32", num_elements, time_int32_to_fp32);

        // 7. FP32 -> INT32
        std::cout << "\n[7/9] Testing FP32 -> INT32..." << std::endl;
        timer.reset();
        cuda.cast_into(fp32_gpu, int32_gpu);
        cuda.synchronize();
        double time_fp32_to_int32 = timer.elapsed_ms();
        print_throughput("FP32 -> INT32", num_elements, time_fp32_to_int32);

        // 8. 创建INT8张量
        std::cout << "\n[8/9] Creating INT8 tensor (zeros)..." << std::endl;
        Tensor int8_gpu = cuda.zeros(shape, DType::INT8);
        std::cout << "  Created: " << int8_gpu.nbytes() / (1024.0 * 1024.0)
                  << " MB on GPU" << std::endl;

        // 9. INT32 -> INT8
        std::cout << "\n[9/9] Testing INT32 -> INT8..." << std::endl;
        timer.reset();
        cuda.cast_into(int32_gpu, int8_gpu);
        cuda.synchronize();
        double time_int32_to_int8 = timer.elapsed_ms();
        print_throughput("INT32 -> INT8", num_elements, time_int32_to_int8);

        // 10. INT8 -> INT32
        std::cout << "\n[10/9] Testing INT8 -> INT32..." << std::endl;
        timer.reset();
        cuda.cast_into(int8_gpu, int32_gpu);
        cuda.synchronize();
        double time_int8_to_int32 = timer.elapsed_ms();
        print_throughput("INT8 -> INT32", num_elements, time_int8_to_int32);

        // 11. INT8 -> FP32
        std::cout << "\n[11/9] Testing INT8 -> FP32..." << std::endl;
        timer.reset();
        cuda.cast_into(int8_gpu, fp32_gpu);
        cuda.synchronize();
        double time_int8_to_fp32 = timer.elapsed_ms();
        print_throughput("INT8 -> FP32", num_elements, time_int8_to_fp32);

        // 打印汇总
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Performance Summary (CUDA)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::left << std::setw(25) << "Conversion"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(18) << "Throughput" << std::endl;
        std::cout << std::string(55, '-') << std::endl;

        print_throughput("FP32 -> BF16", num_elements, time_fp32_to_bf16);
        print_throughput("BF16 -> FP32", num_elements, time_bf16_to_fp32);
        print_throughput("FP32 -> INT32", num_elements, time_fp32_to_int32);
        print_throughput("INT32 -> FP32", num_elements, time_int32_to_fp32);
        print_throughput("INT32 -> INT8", num_elements, time_int32_to_int8);
        print_throughput("INT8 -> INT32", num_elements, time_int8_to_int32);
        print_throughput("INT8 -> FP32", num_elements, time_int8_to_fp32);

        // 计算总吞吐量
        double total_time_ms = time_fp32_to_bf16 + time_bf16_to_fp32 +
                               time_fp32_to_int32 + time_int32_to_fp32 +
                               time_int32_to_int8 + time_int8_to_int32 +
                               time_int8_to_fp32;
        double total_throughput = (7.0 * static_cast<double>(num_elements)) / (total_time_ms / 1000.0);
        double avg_giga_throughput = total_throughput / 1e9 / 7.0;

        std::cout << std::string(55, '-') << std::endl;
        std::cout << "  Average throughput: "
                  << std::fixed << std::setprecision(2)
                  << avg_giga_throughput << " G elems/s" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const TRException& e) {
        LOG_ERROR << e.message();
        return 1;
    } catch (const std::exception& e) {
        LOG_ERROR << "Exception caught: " << e.what();
        return 1;
    }

    return 0;
}
