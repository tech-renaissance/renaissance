/**
 * @file test_nccl_perf.cpp
 * @brief 多GPU CUDA Graph性能测试（NCCL AllReduce吞吐量测试）
 * @version 1.0.0
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1, NCCL 2.29, C++17
 * @note 所属系列: graph
 *
 * @note 功能说明：
 *   基于 test_nccl_graph_fusion.cpp 改造的性能测试版本
 *   测试8卡GPU进行NCCL AllReduce的有效吞吐量（不验证数值正确性）
 *
 * @note 测试配置：
 *   - Tensor大小：通过 --size 参数指定（单位MB，默认100MB）
 *   - GPU数量：8卡
 *   - 操作：Square Kernel + NCCL AllReduce (ncclAvg)
 *   - 数据布局：一维张量
 *
 * @note 性能指标计算：
 *   1. 总执行时间：从 GraphLaunch 到 StreamSynchronize
 *   2. AllReduce有效吞吐量 = (数据大小 × (N-1)) / 时间
 *      其中 N=8 为GPU数量
 *      有效数据聚合量 = 数据大小 × (N-1)
 *
 * @note 使用方法：
 *   ./test_nccl_perf --size 100    # 测试100MB数据
 *   ./test_nccl_perf --size 0.5    # 测试0.5MB数据
 *   ./test_nccl_perf               # 默认100MB
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <string>

// ==================== 生产级错误检查宏 ====================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << "\n  File: " << __FILE__                              \
                      << "\n  Line: " << __LINE__                              \
                      << "\n  Call: " #call << std::endl;                      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t res = call;                                               \
        if (res != ncclSuccess) {                                              \
            std::cerr << "NCCL Error: " << ncclGetErrorString(res)             \
                      << "\n  File: " << __FILE__                              \
                      << "\n  Line: " << __LINE__                              \
                      << "\n  Call: " #call << std::endl;                      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ==================== Kernel Wrapper声明（在square_kernel.cu中实现） ====================
extern "C" void launch_square_kernel(const float* a, float* b, int n, cudaStream_t stream);

// ==================== 参数解析 ====================
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --size <MB>    Tensor size in MB (default: 100)" << std::endl;
    std::cout << "                 Examples: --size 100, --size 0.5, --size 2048" << std::endl;
    std::cout << "  --help         Show this help message" << std::endl;
}

double parse_size_arg(const char* arg) {
    try {
        return std::stod(arg);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid size argument '" << arg << "'" << std::endl;
        return -1.0;
    }
}

// ==================== 主程序 ====================
int main(int argc, char* argv[]) {
    constexpr int kNumGpus = 8;
    constexpr double kDefaultSizeMB = 100.0;
    double size_mb = kDefaultSizeMB;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (arg == "--size") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --size requires an argument" << std::endl;
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            size_mb = parse_size_arg(argv[++i]);
            if (size_mb <= 0.0) {
                std::cerr << "Error: Size must be positive" << std::endl;
                return EXIT_FAILURE;
            }
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // 计算元素数量（一维张量）
    const int kElems = static_cast<int>((size_mb / 4.0) * 1024.0 * 1024.0);
    const size_t kBytesPerGPU = kElems * sizeof(float);
    const size_t kEffectiveBytes = kBytesPerGPU * (kNumGpus - 1);

    std::cout << "========================================" << std::endl;
    std::cout << "renAIssance-V4 NCCL Performance Test" << std::endl;
    std::cout << "Version: AllReduce Throughput Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  GPU Count: " << kNumGpus << std::endl;
    std::cout << "  Tensor Shape: [1D] " << kElems << " elements" << std::endl;
    std::cout << "  Tensor Size: " << kElems << " elements (" << kBytesPerGPU / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "  Effective Data (AllReduce): " << kEffectiveBytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==================== 1. 环境检测 ====================
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < kNumGpus) {
        std::cerr << "Fatal Error: Requires " << kNumGpus
                  << " GPUs, but only " << dev_count << " found." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "[OK] Detected " << dev_count << " GPUs" << std::endl;

    // ==================== 2. 资源容器初始化 ====================
    std::vector<cudaStream_t> streams(kNumGpus);
    std::vector<ncclComm_t> comms(kNumGpus);
    std::vector<float*> d_a(kNumGpus, nullptr);
    std::vector<float*> d_b(kNumGpus, nullptr);
    std::vector<cudaGraph_t> graphs(kNumGpus, nullptr);
    std::vector<cudaGraphExec_t> graph_execs(kNumGpus, nullptr);
    std::vector<cudaEvent_t> start_events(kNumGpus, nullptr);
    std::vector<cudaEvent_t> stop_events(kNumGpus, nullptr);

    // ==================== 3. NCCL通信域初始化 ====================
    std::vector<int> dev_ids(kNumGpus);
    std::iota(dev_ids.begin(), dev_ids.end(), 0);  // {0,1,2,...,7}

    NCCL_CHECK(ncclCommInitAll(comms.data(), kNumGpus, dev_ids.data()));
    std::cout << "[1/6] NCCL communicators initialized for " << kNumGpus << " ranks" << std::endl;

    // ==================== 4. 设备资源初始化 ====================
    std::cout << "[2/6] Initializing device resources..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        // 创建非阻塞流
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[g], cudaStreamNonBlocking));

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_a[g], kBytesPerGPU));
        CUDA_CHECK(cudaMalloc(&d_b[g], kBytesPerGPU));

        // 创建CUDA事件（用于计时）
        CUDA_CHECK(cudaEventCreate(&start_events[g]));
        CUDA_CHECK(cudaEventCreate(&stop_events[g]));

        // 初始化数据：填充任意值（性能测试不需要特定值）
        std::vector<float> h_a(kElems, static_cast<float>(g));
        CUDA_CHECK(cudaMemcpyAsync(d_a[g], h_a.data(), kBytesPerGPU,
                                   cudaMemcpyHostToDevice, streams[g]));

        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }
    std::cout << "  [OK] All device memory allocated (" << kBytesPerGPU * kNumGpus / (1024.0 * 1024.0) << " MB total)" << std::endl;

    // ==================== 5. 分阶段CUDA Graph捕获（分离计算和通信）====================
    std::cout << "[3/6] Capturing CUDA Graphs (Separate Compute and Communication)..." << std::endl;

    // 创建两个图：compute_graphs（square kernel）和 allreduce_graphs（NCCL通信）
    std::vector<cudaGraph_t> compute_graphs(kNumGpus, nullptr);
    std::vector<cudaGraphExec_t> compute_execs(kNumGpus, nullptr);

    // ---- 捕获计算图（Square Kernel） ----
    std::cout << "  [Step 1/2] Capturing compute graphs (square kernels)..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamBeginCapture(streams[g], cudaStreamCaptureModeThreadLocal));
        launch_square_kernel(d_a[g], d_b[g], kElems, streams[g]);
        CUDA_CHECK(cudaStreamEndCapture(streams[g], &compute_graphs[g]));
    }
    std::cout << "    -> Compute graphs captured" << std::endl;

    // ---- 捕获通信图（NCCL AllReduce） ----
    std::cout << "  [Step 2/2] Capturing communication graphs (NCCL AllReduce)..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamBeginCapture(streams[g], cudaStreamCaptureModeThreadLocal));
    }

    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        NCCL_CHECK(ncclAllReduce(
            d_b[g], d_b[g], kElems, ncclFloat32, ncclAvg, comms[g], streams[g]
        ));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamEndCapture(streams[g], &graphs[g]));
    }
    std::cout << "    -> Communication graphs captured" << std::endl;

    // ==================== 6. 图实例化 ====================
    std::cout << "[4/6] Instantiating executable graphs..." << std::endl;

    bool instantiation_success = true;

    // 实例化计算图
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        cudaGraphNode_t error_node = nullptr;
        char log_buffer[2048] = {0};

        cudaError_t inst_err = cudaGraphInstantiate(&compute_execs[g], compute_graphs[g], &error_node, log_buffer, sizeof(log_buffer));
        if (inst_err != cudaSuccess) {
            std::cerr << "  [FAIL] GPU " << g << " compute graph instantiation failed!" << std::endl;
            instantiation_success = false;
            break;
        }
    }

    // 实例化通信图
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        cudaGraphNode_t error_node = nullptr;
        char log_buffer[2048] = {0};

        cudaError_t inst_err = cudaGraphInstantiate(&graph_execs[g], graphs[g], &error_node, log_buffer, sizeof(log_buffer));
        if (inst_err != cudaSuccess) {
            std::cerr << "  [FAIL] GPU " << g << " communication graph instantiation failed!" << std::endl;
            instantiation_success = false;
            break;
        }
    }

    if (!instantiation_success) {
        for (int g = 0; g < kNumGpus; ++g) {
            if (compute_graphs[g]) cudaGraphDestroy(compute_graphs[g]);
            if (compute_execs[g]) cudaGraphExecDestroy(compute_execs[g]);
            if (graphs[g]) cudaGraphDestroy(graphs[g]);
            if (graph_execs[g]) cudaGraphExecDestroy(graph_execs[g]);
        }
        return EXIT_FAILURE;
    }
    std::cout << "  [OK] All graphs instantiated successfully" << std::endl;

    // ==================== 7. 性能测试执行（WARMUP + 正式测试）====================
    std::cout << "[5/6] Running performance benchmark..." << std::endl;

    // Warmup迭代（避免冷启动影响）
    std::cout << "  Running warmup iterations (3x)..." << std::endl;
    for (int warmup = 0; warmup < 3; ++warmup) {
        // 执行计算图
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaGraphLaunch(compute_execs[g], streams[g]));
        }
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }
        // 执行通信图
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaGraphLaunch(graph_execs[g], streams[g]));
        }
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }
    }
    std::cout << "    -> Warmup complete" << std::endl;

    // 正式性能测试（只计时 NCCL AllReduce 部分）
    constexpr int kNumIterations = 10;
    std::cout << "  Running benchmark (" << kNumIterations << " iterations)..." << std::endl;

    std::vector<float> gpu_times_ms(kNumGpus, 0.0f);

    for (int iter = 0; iter < kNumIterations; ++iter) {
        // Step 1: 执行计算图（square kernel，不计时）
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaGraphLaunch(compute_execs[g], streams[g]));
        }

        // Step 2: 同步确保计算完成（避免通信计时包含计算时间）
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }

        // Step 3: 记录通信开始时间（只计时 AllReduce）
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventRecord(start_events[g], streams[g]));
        }

        // Step 4: 执行通信图（NCCL AllReduce，计时的部分）
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaGraphLaunch(graph_execs[g], streams[g]));
        }

        // Step 5: 记录通信结束时间
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventRecord(stop_events[g], streams[g]));
        }

        // Step 6: 同步并累计时间
        for (int g = 0; g < kNumGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));

            float iter_time_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[g], stop_events[g]));
            gpu_times_ms[g] += iter_time_ms;
        }
    }
    std::cout << "    -> Benchmark complete (timed AllReduce communication only)" << std::endl;

    // ==================== 8. 性能统计 ====================
    std::cout << "[6/6] Calculating performance metrics..." << std::endl;

    // 找出最慢GPU的时间（作为整体执行时间）
    float max_time_ms = 0.0f;
    for (int g = 0; g < kNumGpus; ++g) {
        float avg_time_ms = gpu_times_ms[g] / kNumIterations;
        if (avg_time_ms > max_time_ms) {
            max_time_ms = avg_time_ms;
        }
    }

    float avg_time_sec = max_time_ms / 1000.0f;

    // 计算通信吞吐量（只针对 AllReduce 通信部分）
    // AllReduce 有效吞吐量 = 有效数据聚合量 / 时间
    // Ring AllReduce 需要传输: 2 * (N-1) / N * 数据大小
    // 但我们这里简化为: 有效数据量 = 单卡数据大小 * (N-1) = 700MB
    double throughput_gbps = (static_cast<double>(kEffectiveBytes) / avg_time_sec) / (1024.0 * 1024.0 * 1024.0);

    // 每卡带宽 = 单卡数据大小 / 时间（注意：AllReduce 期间每卡实际接收 (N-1)/N 的数据）
    double bandwidth_gbps = (static_cast<double>(kBytesPerGPU) * (kNumGpus - 1.0) / kNumGpus / avg_time_sec) / (1024.0 * 1024.0 * 1024.0);

    // ==================== 9. 性能报告 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "PERFORMANCE RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Iterations: " << kNumIterations << std::endl;
    std::cout << "  Tensor Size: " << kElems << " elements (" << kBytesPerGPU / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "  Timed Operation: NCCL AllReduce ONLY (square kernel excluded)" << std::endl;
    std::cout << "  Effective Data: " << kEffectiveBytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "\nTiming Results (AllReduce Communication Only):" << std::endl;
    std::cout << "  Average Execution Time: " << std::fixed << std::setprecision(3) << avg_time_sec * 1000.0f << " ms" << std::endl;
    std::cout << "\nThroughput Metrics:" << std::endl;
    std::cout << "  AllReduce Effective Throughput: " << std::fixed << std::setprecision(2) << throughput_gbps << " GB/s" << std::endl;
    std::cout << "  Per-GPU Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "\nPer-GPU Timing Breakdown:" << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        float avg_time_ms = gpu_times_ms[g] / kNumIterations;
        std::cout << "  GPU " << g << ": " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // ==================== 10. 资源清理 ====================
    std::cout << "\nCleaning up resources..." << std::endl;

    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        if (stop_events[g]) CUDA_CHECK(cudaEventDestroy(stop_events[g]));
        if (start_events[g]) CUDA_CHECK(cudaEventDestroy(start_events[g]));
        if (compute_execs[g]) CUDA_CHECK(cudaGraphExecDestroy(compute_execs[g]));
        if (compute_graphs[g]) CUDA_CHECK(cudaGraphDestroy(compute_graphs[g]));
        if (graph_execs[g]) CUDA_CHECK(cudaGraphExecDestroy(graph_execs[g]));
        if (graphs[g]) CUDA_CHECK(cudaGraphDestroy(graphs[g]));
        if (d_b[g]) CUDA_CHECK(cudaFree(d_b[g]));
        if (d_a[g]) CUDA_CHECK(cudaFree(d_a[g]));
        if (streams[g]) CUDA_CHECK(cudaStreamDestroy(streams[g]));
        if (comms[g]) NCCL_CHECK(ncclCommDestroy(comms[g]));
    }

    std::cout << "All resources cleaned up successfully" << std::endl;
    std::cout << "\n========================================" << std::endl;
    std::cout << "BENCHMARK COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
