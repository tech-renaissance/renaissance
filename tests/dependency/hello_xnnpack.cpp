#include <xnnpack.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>

// ================= 配置区域 =================
const size_t BATCH_SIZE = 512;
const size_t INPUT_CHANNELS = 512;
const size_t OUTPUT_CHANNELS = 512;
const int BENCHMARK_ITERATIONS = 100;
const float EPSILON = 1e-3f;
// ===========================================

int main() {
    // 1. 初始化 XNNPACK
    xnn_status status = xnn_initialize(nullptr);
    if (status != xnn_status_success) {
        std::cerr << "XNNPACK initialization failed!" << std::endl;
        return -1;
    }
    std::cout << "Step 1: XNNPACK initialized successfully." << std::endl;

    // 2. 准备数据
    // FC 层: Input (M x K), Kernel (N x K), Bias (N), Output (M x N)
    size_t input_elements = BATCH_SIZE * INPUT_CHANNELS;
    size_t kernel_elements = OUTPUT_CHANNELS * INPUT_CHANNELS;
    size_t bias_elements = OUTPUT_CHANNELS;
    size_t output_elements = BATCH_SIZE * OUTPUT_CHANNELS;

    std::vector<float> input(input_elements);
    std::vector<float> kernel(kernel_elements);
    std::vector<float> bias(bias_elements);
    std::vector<float> output_xnn(output_elements);

    // 随机填充
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : input) v = dist(rng);
    for (auto& v : kernel) v = dist(rng);
    for (auto& v : bias) v = dist(rng);
    std::cout << "Step 2: Data prepared." << std::endl;

    // =========================================================
    // 3. 构建子图 (Subgraph Construction)
    // =========================================================
    xnn_subgraph_t subgraph = nullptr;
    // 4 个张量节点：Input, Kernel, Bias, Output
    status = xnn_create_subgraph(4, 0, &subgraph);
    if (status != xnn_status_success) {
        std::cerr << "Failed to create subgraph" << std::endl;
        return -1;
    }
    std::cout << "Step 3: Subgraph created." << std::endl;

    // 定义张量的唯一 ID（将由 xnn_define_tensor_value 返回）
    uint32_t input_id = 0;
    uint32_t kernel_id = 0;
    uint32_t bias_id = 0;
    uint32_t output_id = 0;

    // --- 定义 Input Tensor ---
    size_t input_dims[] = {BATCH_SIZE, INPUT_CHANNELS};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        2, input_dims,
        nullptr,
        UINT32_MAX,  // external_id = XNN_INVALID_VALUE_ID
        XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &input_id);  // id_out (返回的 ID)
    if (status != xnn_status_success) {
        std::cerr << "Failed to define Input tensor" << std::endl;
        return -1;
    }

    // --- 定义 Kernel Tensor (权重) ---
    size_t kernel_dims[] = {OUTPUT_CHANNELS, INPUT_CHANNELS};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        2, kernel_dims,
        kernel.data(),
        UINT32_MAX,  // external_id
        0,  // flags
        &kernel_id);  // id_out
    if (status != xnn_status_success) {
        std::cerr << "Failed to define Kernel tensor" << std::endl;
        return -1;
    }

    // --- 定义 Bias Tensor ---
    size_t bias_dims[] = {OUTPUT_CHANNELS};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        1, bias_dims,
        bias.data(),
        UINT32_MAX,  // external_id
        0,  // flags
        &bias_id);  // id_out

    // --- 定义 Output Tensor ---
    size_t output_dims[] = {BATCH_SIZE, OUTPUT_CHANNELS};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        2, output_dims,
        nullptr,
        UINT32_MAX,  // external_id
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
        &output_id);  // id_out

    // --- 定义全连接节点 (Node) ---
    status = xnn_define_fully_connected(
        subgraph,
        -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        input_id,
        kernel_id,
        bias_id,
        output_id,
        0
    );
    if (status != xnn_status_success) {
        std::cerr << "Failed to define FC node" << std::endl;
        return -1;
    }
    std::cout << "Step 4: Operator defined." << std::endl;

    // =========================================================
    // 5. 创建运行时 (Create Runtime)
    // =========================================================
    xnn_runtime_t runtime = nullptr;
    status = xnn_create_runtime_v3(subgraph, nullptr, nullptr, 0, &runtime);
    if (status != xnn_status_success) {
        std::cerr << "Failed to create runtime" << std::endl;
        return -1;
    }
    std::cout << "Step 5: Runtime created." << std::endl;

    // =========================================================
    // 6. 绑定内存并运行 (Setup & Run)
    // =========================================================
    std::vector<xnn_external_value> external_args = {
        {input_id, input.data()},
        {output_id, output_xnn.data()}
    };

    status = xnn_setup_runtime(runtime, external_args.size(), external_args.data());
    if (status != xnn_status_success) {
        std::cerr << "Failed to setup runtime" << std::endl;
        return -1;
    }
    std::cout << "Step 6: Runtime setup." << std::endl;

    // 验证部分 (简化版 - 只验证第一行)
    std::cout << "Step 7: Verifying correctness (first row)..." << std::endl;
    xnn_invoke_runtime(runtime);

    bool pass = true;
    for (size_t out_c = 0; out_c < OUTPUT_CHANNELS; out_c++) {
        float sum = bias[out_c];
        for (size_t in_c = 0; in_c < INPUT_CHANNELS; in_c++) {
            sum += input[in_c] * kernel[out_c * INPUT_CHANNELS + in_c];
        }
        float diff = std::abs(output_xnn[out_c] - sum);
        if (diff > EPSILON && diff / (std::abs(sum) + 1e-6) > EPSILON) {
            std::cerr << "[FAIL] Mismatch at " << out_c
                      << " XNN=" << output_xnn[out_c] << " Ref=" << sum << std::endl;
            pass = false;
            break;
        }
    }
    if (!pass) {
        std::cerr << "Verification FAILED!" << std::endl;
        return -1;
    }
    std::cout << "Step 8: Verification PASSED!" << std::endl;

    // =========================================================
    // 9. 性能压测
    // =========================================================
    std::cout << "Step 9: Benchmarking (" << BENCHMARK_ITERATIONS << " iterations)..." << std::endl;

    // 预热
    for(int i = 0; i < 5; i++) xnn_invoke_runtime(runtime);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        xnn_invoke_runtime(runtime);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double total_ops = 2.0 * BATCH_SIZE * INPUT_CHANNELS * OUTPUT_CHANNELS * BENCHMARK_ITERATIONS;
    double gflops = (total_ops / duration.count()) / 1e9;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  XNNPACK Hello World (Fully Connected)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dimensions      : " << BATCH_SIZE << " x " << INPUT_CHANNELS << " x " << OUTPUT_CHANNELS << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total Time      : " << std::fixed << std::setprecision(4)
              << duration.count() << " s" << std::endl;
    std::cout << "Avg Latency     : " << (duration.count() * 1000 / BENCHMARK_ITERATIONS)
              << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance     : " << std::setprecision(2)
              << gflops << " GFLOPS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Result          : PASS" << std::endl;
    std::cout << "========================================" << std::endl;

    // 清理资源
    xnn_delete_runtime(runtime);
    xnn_delete_subgraph(subgraph);

    return 0;
}
