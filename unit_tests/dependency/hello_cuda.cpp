/**
 * @file test_gemm_e.cpp
 * @brief Matrix Multiplication Solution E - cuDNN 1x1 Convolution Implementation
 * @details Using cuDNN's 1x1 convolution with automatic algorithm finding for GEMM
 * Implements matrix multiplication: C(4096, 4096) = A(4096, 8192) * B(8192, 4096)
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: CUDA, cuDNN
 * @note 所属系列: tests
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>
#include "renaissance.h"

/**
 * @brief CUDA API调用检查宏
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string msg = "CUDA Error at " + std::string(__FILE__) + ":" + \
                         std::to_string(__LINE__) + ": " + \
                         cudaGetErrorString(err); \
        throw std::runtime_error(msg); \
    } \
} while (0)

/**
 * @brief cuDNN API调用检查宏
 */
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::string msg = "cuDNN Error at " + std::string(__FILE__) + ":" + \
                         std::to_string(__LINE__) + ": " + \
                         cudnnGetErrorString(status); \
        throw std::runtime_error(msg); \
    } \
} while (0)

/**
 * @brief 方案E的矩阵乘法实现类（使用cuDNN 1x1卷积）
 */
class SolutionEGEMM {
private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;

    // 矩阵维度
    const int M_;  // 对应batch size
    const int K_;  // 对应input channels
    const int N_;  // 对应output channels

    // 设备内存指针
    float* d_A_;  // 输入张量
    float* d_B_;  // 卷积核（权重）
    float* d_C_;  // 输出张量

    // 工作空间
    void* d_workspace_;
    size_t workspace_size_;
    cudnnConvolutionFwdAlgo_t best_algo_;

    // 内存大小（字节）
    size_t size_A_;
    size_t size_B_;
    size_t size_C_;

public:
    /**
     * @brief 构造函数
     */
    SolutionEGEMM(int M, int K, int N)
        : M_(M), K_(K), N_(N), d_A_(nullptr), d_B_(nullptr), d_C_(nullptr),
          d_workspace_(nullptr), workspace_size_(0) {

        // 计算内存大小
        size_A_ = static_cast<size_t>(M) * K * sizeof(float);
        size_B_ = static_cast<size_t>(K) * N * sizeof(float);
        size_C_ = static_cast<size_t>(M) * N * sizeof(float);

        // 初始化CUDA
        CUDA_CHECK(cudaSetDevice(0));

        // 创建cuDNN句柄
        CUDNN_CHECK(cudnnCreate(&cudnn_handle_));

        // 创建描述符
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));

        // 设置输入张量描述符 (NCHW格式: N=batch, C=channels, H=1, W=1)
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               M_, K_, 1, 1));

        // 设置滤波器（权重）描述符 (NCHW格式: N=output_channels, C=input_channels, H=1, W=1)
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_,
                                             CUDNN_DATA_FLOAT,
                                             CUDNN_TENSOR_NCHW,
                                             N_, K_, 1, 1));

        // 设置卷积描述符 (1x1卷积，无padding，stride=1)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                   0, 0,    // padding
                                                   1, 1,    // stride
                                                   1, 1,    // dilation
                                                   CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));

        // 设置输出张量描述符
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               M_, N_, 1, 1));

        // 查找最优算法（方案E的核心特性）
        findBestAlgorithm();

        // 分配工作空间
        allocateWorkspace();

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_A_, size_A_));
        CUDA_CHECK(cudaMalloc(&d_B_, size_B_));
        CUDA_CHECK(cudaMalloc(&d_C_, size_C_));

        std::cout << "Solution E: cuDNN 1x1 Convolution GEMM initialized" << std::endl;
        std::cout << "Matrix A: " << M << "x" << K << " (" << size_A_ / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "Matrix B: " << K << "x" << N << " (" << size_B_ / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "Matrix C: " << M << "x" << N << " (" << size_C_ / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "Workspace: " << workspace_size_ / 1024.0 / 1024.0 << " MB" << std::endl;
        std::cout << "Best algorithm: " << getAlgorithmName(best_algo_) << std::endl;
    }

    /**
     * @brief 析构函数
     */
    ~SolutionEGEMM() {
        // 修复：避免析构函数中的虚函数调用
        // 直接使用 cudaError_t 检查，不调用宏中的虚函数
        if (d_A_) {
            cudaError_t err = cudaFree(d_A_);
            (void)err; // 静态分析时抑制未使用变量警告
        }
        if (d_B_) {
            cudaError_t err = cudaFree(d_B_);
            (void)err;
        }
        if (d_C_) {
            cudaError_t err = cudaFree(d_C_);
            (void)err;
        }
        if (d_workspace_) {
            cudaError_t err = cudaFree(d_workspace_);
            (void)err;
        }
        if (input_desc_) {
            cudnnStatus_t err = cudnnDestroyTensorDescriptor(input_desc_);
            (void)err;
        }
        if (output_desc_) {
            cudnnStatus_t err = cudnnDestroyTensorDescriptor(output_desc_);
            (void)err;
        }
        if (filter_desc_) {
            cudnnStatus_t err = cudnnDestroyFilterDescriptor(filter_desc_);
            (void)err;
        }
        if (conv_desc_) {
            cudnnStatus_t err = cudnnDestroyConvolutionDescriptor(conv_desc_);
            (void)err;
        }
        if (cudnn_handle_) {
            cudnnStatus_t err = cudnnDestroy(cudnn_handle_);
            (void)err;
        }
    }

    /**
     * @brief 查找最优卷积算法
     */
    void findBestAlgorithm() {
        cudnnConvolutionFwdAlgoPerf_t perf_result;
        int returned_algo_count = 0;

        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
            cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
            1, &returned_algo_count, &perf_result));

        if (returned_algo_count == 0) {
            throw std::runtime_error("Failed to find any convolution algorithm");
        }

        best_algo_ = perf_result.algo;
        std::cout << "Algorithm search completed - Best: " << getAlgorithmName(best_algo_)
                  << " (Time: " << perf_result.time << " ms)" << std::endl;
    }

    /**
     * @brief 分配工作空间
     */
    void allocateWorkspace() {
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
            best_algo_, &workspace_size_));

        if (workspace_size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_size_));
        }
    }

    /**
     * @brief 在GPU上初始化矩阵数据
     */
    void initializeDeviceData(float val_A, float val_B) {
        // 创建主机数据并拷贝到设备（更简单的方法）
        std::vector<float> h_A(M_ * K_, val_A);
        std::vector<float> h_B(K_ * N_, val_B);

        // 拷贝数据到设备
        CUDA_CHECK(cudaMemcpy(d_A_, h_A.data(), size_A_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_, h_B.data(), size_B_, cudaMemcpyHostToDevice));

        std::cout << "Device data initialized: A=" << val_A << ", B=" << val_B << std::endl;
    }

    /**
     * @brief 执行矩阵乘法（使用1x1卷积）
     */
    void multiply() {
        float alpha = 1.0f;
        float beta = 0.0f;

        // 使用cuDNN 1x1卷积执行矩阵乘法
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn_handle_, &alpha,
            input_desc_, d_A_,
            filter_desc_, d_B_,
            conv_desc_, best_algo_, d_workspace_, workspace_size_,
            &beta, output_desc_, d_C_));
    }

    /**
     * @brief 执行性能测试
     */
    double runPerformanceTest(int iterations = 20) {
        std::cout << "Running performance test with " << iterations << " iterations..." << std::endl;

        // 热身运行
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 3; i++) {
            multiply();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 开始计时
        CUDA_CHECK(cudaEventRecord(start));

        // 执行多次迭代
        for (int i = 0; i < iterations; i++) {
            multiply();
        }

        // 结束计时
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // 计算时间
        float elapsed_time_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

        // 清理事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        double avg_time_ms = elapsed_time_ms / iterations;
        std::cout << "Average multiplication time: " << avg_time_ms << " ms" << std::endl;

        // 计算GFLOPS
        double flops = 2.0 * M_ * N_ * K_;  // GEMM浮点运算次数
        double gflops = flops / (avg_time_ms * 1e6);  // GFLOPS
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        return avg_time_ms;
    }

    /**
     * @brief 验证结果并获取第一个元素
     */
    bool validateResult(float expected_value, float tolerance = 1e-4f) {
        std::vector<float> h_C(M_ * N_);

        // 拷贝结果回主机
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C_, size_C_, cudaMemcpyDeviceToHost));

        // 检查第一个元素
        float first_element = h_C[0];
        float rel_error = std::abs(first_element - expected_value) / expected_value;

        std::cout << "First element validation:" << std::endl;
        std::cout << "Expected: " << expected_value << std::endl;
        std::cout << "Actual: " << first_element << std::endl;
        std::cout << "Relative error: " << rel_error << std::endl;

        bool success = rel_error < tolerance;
        if (success) {
            std::cout << "Validation PASSED!" << std::endl;
        } else {
            std::cout << "Validation FAILED!" << std::endl;
        }

        return success;
    }

    /**
     * @brief 获取算法名称
     */
    const char* getAlgorithmName(cudnnConvolutionFwdAlgo_t algo) {
        switch (algo) {
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "IMPLICIT_GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "IMPLICIT_PRECOMP_GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "DIRECT";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "FFT";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "FFT_TILING";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "WINOGRAD";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "WINOGRAD_NONFUSED";
            default: return "UNKNOWN";
        }
    }

    /**
     * @brief 获取内存使用量（MB）
     */
    double getMemoryUsageMB() const {
        return (size_A_ + size_B_ + size_C_ + workspace_size_) / (1024.0 * 1024.0);
    }

    // Getter方法
    int getM() const { return M_; }
    int getK() const { return K_; }
    int getN() const { return N_; }
};

/**
 * @brief 运行方案E的矩阵乘法测试
 */
bool runSolutionETest() {
    std::cout << "=== Solution E: cuDNN 1x1 Convolution GEMM ===" << std::endl;
    std::cout << "Matrix multiplication: C(4096, 4096) = A(4096, 8192) * B(8192, 4096)" << std::endl;
    std::cout << "Features: Automatic algorithm finding, high-performance 1x1 convolution" << std::endl;
    std::cout << std::endl;

    try {
        // 测试参数
        const int M = 4096;  // 对应batch size
        const int K = 8192;  // 对应input channels
        const int N = 4096;  // 对应output channels

        // 创建矩阵乘法对象
        SolutionEGEMM gemm(M, K, N);

        std::cout << "Total memory usage: " << gemm.getMemoryUsageMB() << " MB" << std::endl;
        std::cout << std::endl;

        // 初始化数据
        const float val_A = 0.01f;
        const float val_B = 0.1f;
        gemm.initializeDeviceData(val_A, val_B);

        // 预期结果：C[i,j] = K * val_A * val_B = 8192 * 0.01 * 0.1 = 8.192
        float expected_value = static_cast<float>(K) * val_A * val_B;
        std::cout << "Expected value for each element: " << expected_value << std::endl;
        std::cout << std::endl;

        // 运行性能测试
        double avg_time_ms = gemm.runPerformanceTest();

        // 验证结果
        bool success = gemm.validateResult(expected_value);

        // 输出最终结果
        std::cout << "\n=== Solution E Final Results ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;
        std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;
        std::cout << "Memory usage: " << gemm.getMemoryUsageMB() << " MB" << std::endl;
        std::cout << "First element value: " << expected_value << std::endl;
        std::cout << "Status: " << (success ? "SUCCESS" : "FAILED") << std::endl;

        return success;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    // 设置输出精度
    std::cout << std::fixed << std::setprecision(6);
    std::cerr << std::fixed << std::setprecision(6);

    // 初始化renAIssance框架
    renAIssance::Framework::printWelcome();
    renAIssance::Logger::info("Starting CUDA Matrix Multiplication Test");

    // 演示框架基础功能
    renAIssance::helloWorld();
    renAIssance::demonstrateFramework();

    std::cout << "=== Matrix Multiplication Solution E Test ===" << std::endl;
    std::cout << "Algorithm: cuDNN 1x1 convolution with automatic algorithm finding" << std::endl;
    std::cout << "Implementation: Matrix multiplication as 1x1 convolution" << std::endl;
    std::cout << std::endl;

    bool success = runSolutionETest();

    if (success) {
        std::cout << "\n[SUCCESS] Solution E test completed successfully!" << std::endl;
        renAIssance::Logger::info("renAIssance Framework CUDA test completed successfully!");
        return 0;
    } else {
        std::cout << "\n[FAILURE] Solution E test failed!" << std::endl;
        renAIssance::Logger::error("renAIssance Framework CUDA test failed!");
        return 1;
    }
}