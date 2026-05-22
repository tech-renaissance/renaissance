/**
 * @file test_nccl_graph_fusion.cpp
 * @brief 多GPU CUDA Graph基准测试（NCCL AllReduce图内融合 - 分阶段并行捕获方案）
 * @version 1.0.0
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1, NCCL 2.29, C++17
 * @note 所属系列: graph
 *
 * @note 功能说明：
 *   基于SNX专家的"分阶段并行捕获"方案，实现NCCL AllReduce在CUDA Graph内的正确捕获
 *   解决原有串行捕获导致的NCCL握手死锁问题
 *
 * @note 核心策略（参考EXPERT_SNX.md）：
 *   Phase 1: 所有GPU同时进入捕获模式（cudaStreamBeginCapture）
 *   Phase 2: 记录所有计算节点（square_kernel）
 *   Phase 3: 使用ncclGroupStart/End记录NCCL通信节点（确保原子性）
 *   Phase 4: 所有GPU同时结束捕获（cudaStreamEndCapture）
 *
 * @note 关键技术点：
 *   1. ncclGroupStart/End：告知NCCL将AllReduce作为原子事务记录到图中
 *   2. cudaStreamCaptureModeThreadLocal：只捕获当前线程发起的操作
 *   3. 分阶段执行：确保所有Rank在逻辑上"同时"处于关键状态
 *   4. 单线程实现：无需多线程同步，逻辑清晰简洁
 *
 * @note 预期结果：
 *   GPU 0-7 的b张量所有元素均为 17.5f = (0²+1²+...+7²)/8
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <numeric>  // for std::iota

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

// ==================== 主程序 ====================
int main() {
    constexpr int kNumGpus = 8;
    constexpr int kElems = 3 * 2;  // 3x2张量
    constexpr size_t kBytes = kElems * sizeof(float);
    constexpr float kExpected = 17.5f;  // (0²+1²+...+7²)/8 = 140/8

    std::cout << "========================================" << std::endl;
    std::cout << "renAIssance-V4 Multi-GPU Graph Test" << std::endl;
    std::cout << "Version: NCCL In-Graph Fusion (SNX Expert Solution)" << std::endl;
    std::cout << "Strategy: Phased Parallel Capture" << std::endl;
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

    // ==================== 3. NCCL通信域初始化 ====================
    std::vector<int> dev_ids(kNumGpus);
    std::iota(dev_ids.begin(), dev_ids.end(), 0);  // {0,1,2,...,7}

    NCCL_CHECK(ncclCommInitAll(comms.data(), kNumGpus, dev_ids.data()));
    std::cout << "[1/5] NCCL communicators initialized for " << kNumGpus << " ranks" << std::endl;

    // ==================== 4. 设备资源初始化（图捕获前完成）====================
    std::cout << "[2/5] Initializing device resources..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        // 创建非阻塞流（与默认流独立，避免隐式同步）
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[g], cudaStreamNonBlocking));

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_a[g], kBytes));
        CUDA_CHECK(cudaMalloc(&d_b[g], kBytes));

        // 初始化数据：GPU g填充值 g.0f
        std::vector<float> h_a(kElems, static_cast<float>(g));
        CUDA_CHECK(cudaMemcpyAsync(d_a[g], h_a.data(), kBytes,
                                   cudaMemcpyHostToDevice, streams[g]));

        // 同步确保数据就绪（避免H2D拷贝被捕获进图中）
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }
    std::cout << "  [OK] All device memory allocated and data uploaded" << std::endl;

    // ==================== 5. 分阶段CUDA Graph捕获（核心修复区域）====================
    // SNX专家方案的关键：分阶段并行捕获，避免NCCL握手死锁
    std::cout << "[3/5] Capturing CUDA Graphs (Kernel + NCCL Fused)..." << std::endl;

    // --------------------------------------------------
    // 阶段1: 所有GPU同时进入捕获模式
    // 关键：确保NCCL在后续调用时，所有Rank都已"在线"
    // --------------------------------------------------
    std::cout << "  [Phase 1/4] Starting stream capture on all GPUs..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        // ThreadLocal模式：只捕获当前线程发起的操作，避免跨线程依赖
        CUDA_CHECK(cudaStreamBeginCapture(streams[g], cudaStreamCaptureModeThreadLocal));
    }
    std::cout << "    -> All streams in capture mode" << std::endl;

    // --------------------------------------------------
    // 阶段2: 记录所有计算节点
    // --------------------------------------------------
    std::cout << "  [Phase 2/4] Recording compute kernels..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        // 记录平方核函数（b = a²）
        launch_square_kernel(d_a[g], d_b[g], kElems, streams[g]);
    }
    std::cout << "    -> Square kernels recorded" << std::endl;

    // --------------------------------------------------
    // 阶段3: 记录NCCL通信节点（必须用ncclGroup包裹）
    // 关键：ncclGroupStart/End确保所有AllReduce被视为原子事务
    // 这解决了原有串行捕获的死锁问题
    // --------------------------------------------------
    std::cout << "  [Phase 3/4] Recording NCCL AllReduce nodes..." << std::endl;

    // 开启NCCL Group（告知NCCL："请将以下操作作为一个整体记录到图中"）
    NCCL_CHECK(ncclGroupStart());

    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        // 在捕获模式下，ncclAllReduce不会阻塞，而是将通信操作序列化为图节点
        NCCL_CHECK(ncclAllReduce(
            d_b[g],           // sendbuff
            d_b[g],           // recvbuff (in-place)
            kElems,           // count
            ncclFloat32,      // datatype
            ncclAvg,          // reduction operation (average)
            comms[g],         // communicator
            streams[g]        // stream
        ));
    }

    // 结束NCCL Group
    NCCL_CHECK(ncclGroupEnd());
    std::cout << "    -> NCCL AllReduce nodes recorded (Group API)" << std::endl;

    // --------------------------------------------------
    // 阶段4: 所有GPU同时结束捕获
    // --------------------------------------------------
    std::cout << "  [Phase 4/4] Ending stream capture..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamEndCapture(streams[g], &graphs[g]));
    }
    std::cout << "    -> All graphs captured successfully" << std::endl;

    // ==================== 6. 图实例化（生产级错误处理）====================
    std::cout << "[4/5] Instantiating executable graphs..." << std::endl;

    bool instantiation_success = true;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        cudaGraphNode_t error_node = nullptr;
        char log_buffer[2048] = {0};

        // 使用带日志的实例化接口（CUDA 11.0+推荐）
        cudaError_t inst_err = cudaGraphInstantiate(
            &graph_execs[g],    // 输出：可执行图
            graphs[g],          // 输入：图定义
            &error_node,        // 输出：错误节点（如果失败）
            log_buffer,         // 输出：错误日志
            sizeof(log_buffer)  // 日志缓冲区大小
        );

        if (inst_err != cudaSuccess) {
            std::cerr << "  [FAIL] GPU " << g << " instantiation failed!" << std::endl;
            std::cerr << "      Error: " << cudaGetErrorString(inst_err) << std::endl;
            if (log_buffer[0] != '\0') {
                std::cerr << "      Log: " << log_buffer << std::endl;
            }
            instantiation_success = false;
            break;
        }

        std::cout << "  [OK] GPU " << g << " graph instantiated" << std::endl;
    }

    if (!instantiation_success) {
        // 清理已分配的资源
        for (int g = 0; g < kNumGpus; ++g) {
            if (graphs[g]) cudaGraphDestroy(graphs[g]);
            if (graph_execs[g]) cudaGraphExecDestroy(graph_execs[g]);
        }
        return EXIT_FAILURE;
    }

    // ==================== 7. 图执行（单次Launch包含计算+通信）====================
    std::cout << "[5/5] Launching graphs (fused compute + communication)..." << std::endl;

    // 并行启动所有GPU的图
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaGraphLaunch(graph_execs[g], streams[g]));
    }
    std::cout << "  -> All graphs launched" << std::endl;

    // 同步等待所有GPU完成
    std::cout << "  -> Waiting for completion..." << std::endl;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }
    std::cout << "  [OK] All graphs executed successfully" << std::endl;

    // ==================== 8. 结果验证 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Verification Results:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Expected Value: " << kExpected
              << " (Average of 0²+1²+...+7² = 140/8)" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_pass = true;
    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        std::vector<float> h_b(kElems);
        CUDA_CHECK(cudaMemcpy(h_b.data(), d_b[g], kBytes, cudaMemcpyDeviceToHost));

        bool gpu_ok = true;
        std::cout << "GPU " << g << " Result: [";
        for (int i = 0; i < kElems; ++i) {
            std::cout << std::fixed << std::setprecision(1) << h_b[i];
            if (i < kElems - 1) std::cout << ", ";

            // 允许微小的浮点误差
            if (std::fabs(h_b[i] - kExpected) > 1e-4f) {
                gpu_ok = false;
            }
        }
        std::cout << "] " << (gpu_ok ? "PASS" : "FAIL") << std::endl;

        all_pass &= gpu_ok;
    }

    // ==================== 9. 资源清理 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Cleaning up resources..." << std::endl;

    for (int g = 0; g < kNumGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        if (graph_execs[g]) CUDA_CHECK(cudaGraphExecDestroy(graph_execs[g]));
        if (graphs[g]) CUDA_CHECK(cudaGraphDestroy(graphs[g]));
        if (d_b[g]) CUDA_CHECK(cudaFree(d_b[g]));
        if (d_a[g]) CUDA_CHECK(cudaFree(d_a[g]));
        if (streams[g]) CUDA_CHECK(cudaStreamDestroy(streams[g]));
        if (comms[g]) NCCL_CHECK(ncclCommDestroy(comms[g]));
    }

    // ==================== 10. 最终报告 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL RESULT: " << (all_pass ? "ALL TESTS PASSED" : "TEST FAILED") << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Architecture: Single-Stream Fused Graph (SNX Expert Solution)" << std::endl;
    std::cout << "  - Compute: Square Kernel (b = a²)" << std::endl;
    std::cout << "  - Communication: NCCL AllReduce (ncclAvg, in-graph)" << std::endl;
    std::cout << "  - Capture Strategy: Phased Parallel (4 phases)" << std::endl;
    std::cout << "  - Key Technique: ncclGroupStart/End for atomic capture" << std::endl;
    std::cout << "========================================" << std::endl;

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
