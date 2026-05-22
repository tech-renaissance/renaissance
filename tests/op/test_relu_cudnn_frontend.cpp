/**
 * @file test_relu_cudnn_frontend.cpp
 * @brief 独立 ReLU 算子性能测试（cuDNN Frontend，OLD 方法）
 * @note 单 GPU，AMP FP16，形状 8x1024x1024x8，预热5次+计时100次
 * @note 无参数解析，全部默认配置
 */

#include "../fp16/ta_v4_common_fp16.hpp"

namespace fe = cudnn_frontend;

class ReLUCuDNNFrontendBenchmark {
private:
    cudnnHandle_t handle_;
    cudaStream_t stream_;

    // 形状参数（与 test_relu 完全一致）
    static constexpr int64_t N = 8;
    static constexpr int64_t H = 1024;
    static constexpr int64_t W = 1024;
    static constexpr int64_t C = 8;

    int64_t total_elements_;

    // GPU 内存
    void* d_x;      // 输入（FWD input / BWD 复用）
    void* d_y;      // FWD 输出
    void* d_mask;   // FWD bitmask
    void* d_dy;     // BWD 上游梯度
    void* d_dx;     // BWD 输出

    // cuDNN Frontend Graphs
    std::shared_ptr<fe::graph::Graph> relu_fwd_graph_;
    std::shared_ptr<fe::graph::Graph> relu_bwd_graph_;

    // FWD Tensor attributes
    std::shared_ptr<fe::graph::Tensor_attributes> fwd_x_;
    std::shared_ptr<fe::graph::Tensor_attributes> fwd_y_;
    std::shared_ptr<fe::graph::Tensor_attributes> fwd_mask_;

    // BWD Tensor attributes
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_dy_;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_y_;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_dx_;

    // Variant Packs
    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_fwd_;
    VariantPack vp_bwd_;

    // CUDA Graph
    cudaGraph_t graph_fwd_ = nullptr;
    cudaGraphExec_t graph_exec_fwd_ = nullptr;
    cudaGraph_t graph_bwd_ = nullptr;
    cudaGraphExec_t graph_exec_bwd_ = nullptr;

public:
    ReLUCuDNNFrontendBenchmark() {
        total_elements_ = N * H * W * C;

        CHECK_CUDNN(cudnnCreate(&handle_));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CHECK_CUDNN(cudnnSetStream(handle_, stream_));

        allocate_memory();
        initialize_data();
        build_relu_fwd_graph();
        build_relu_bwd_graph();
        build_variant_packs();
    }

    ~ReLUCuDNNFrontendBenchmark() {
        if (graph_exec_fwd_) cudaGraphExecDestroy(graph_exec_fwd_);
        if (graph_fwd_) cudaGraphDestroy(graph_fwd_);
        if (graph_exec_bwd_) cudaGraphExecDestroy(graph_exec_bwd_);
        if (graph_bwd_) cudaGraphDestroy(graph_bwd_);

        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));
        CHECK_CUDA(cudaFree(d_mask));
        CHECK_CUDA(cudaFree(d_dy));
        CHECK_CUDA(cudaFree(d_dx));

        CHECK_CUDNN(cudnnDestroy(handle_));
        CHECK_CUDA(cudaStreamDestroy(stream_));
    }

private:
    void allocate_memory() {
        size_t fp16_bytes = total_elements_ * sizeof(__half);
        size_t mask_bytes = total_elements_ * sizeof(uint8_t);

        d_x    = allocate_aligned_gpu_memory(fp16_bytes);
        d_y    = allocate_aligned_gpu_memory(fp16_bytes);
        d_mask = allocate_aligned_gpu_memory(mask_bytes);
        d_dy   = allocate_aligned_gpu_memory(fp16_bytes);
        d_dx   = allocate_aligned_gpu_memory(fp16_bytes);
    }

    void initialize_data() {
        initialize_random_fp16(d_x, total_elements_, 42);
        initialize_random_fp16(d_dy, total_elements_, 43);
    }

    void build_relu_fwd_graph() {
        relu_fwd_graph_ = std::make_shared<fe::graph::Graph>();
        relu_fwd_graph_->set_io_data_type(fe::DataType_t::HALF)
                       .set_intermediate_data_type(fe::DataType_t::FLOAT)
                       .set_compute_data_type(fe::DataType_t::FLOAT);

        // NHWC layout: dim = {N, C, H, W}, stride = {H*W*C, 1, W*C, C}
        std::vector<int64_t> dim    = {N, C, H, W};
        std::vector<int64_t> stride = {H * W * C, 1, W * C, C};

        fwd_x_ = relu_fwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("x")
            .set_dim(dim)
            .set_stride(stride)
            .set_data_type(fe::DataType_t::HALF));

        // ReLU FWD: y = max(0, x)
        auto relu_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto relu_out = relu_fwd_graph_->pointwise(fwd_x_, relu_options);
        fwd_y_ = relu_out;
        fwd_y_->set_output(true).set_data_type(fe::DataType_t::HALF);

        // CMP_GT: mask = x > 0
        float zero = 0.0f;
        auto zero_tensor = relu_fwd_graph_->tensor(zero);

        auto cmp_gt_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::CMP_GT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto bitmask = relu_fwd_graph_->pointwise(fwd_x_, zero_tensor, cmp_gt_options);
        fwd_mask_ = bitmask;
        fwd_mask_->set_output(true).set_data_type(fe::DataType_t::BOOLEAN);

        CHECK_CUDNN_FE(relu_fwd_graph_->validate());
        CHECK_CUDNN_FE(relu_fwd_graph_->build_operation_graph(handle_));
        CHECK_CUDNN_FE(relu_fwd_graph_->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(relu_fwd_graph_->check_support(handle_));
        CHECK_CUDNN_FE(relu_fwd_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
    }

    void build_relu_bwd_graph() {
        relu_bwd_graph_ = std::make_shared<fe::graph::Graph>();
        relu_bwd_graph_->set_io_data_type(fe::DataType_t::HALF)
                       .set_intermediate_data_type(fe::DataType_t::FLOAT)
                       .set_compute_data_type(fe::DataType_t::FLOAT);

        std::vector<int64_t> dim    = {N, C, H, W};
        std::vector<int64_t> stride = {H * W * C, 1, W * C, C};

        bwd_dy_ = relu_bwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("dy")
            .set_dim(dim)
            .set_stride(stride)
            .set_data_type(fe::DataType_t::HALF));

        // ReLU BWD 需要前向输出 y（cuDNN 内部用 y > 0 判断符号）
        bwd_y_ = relu_bwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("y")
            .set_dim(dim)
            .set_stride(stride)
            .set_data_type(fe::DataType_t::HALF));

        auto relu_bwd_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::RELU_BWD)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto dx_out = relu_bwd_graph_->pointwise(bwd_dy_, bwd_y_, relu_bwd_options);
        bwd_dx_ = dx_out;
        bwd_dx_->set_output(true).set_data_type(fe::DataType_t::HALF);

        CHECK_CUDNN_FE(relu_bwd_graph_->validate());
        CHECK_CUDNN_FE(relu_bwd_graph_->build_operation_graph(handle_));
        CHECK_CUDNN_FE(relu_bwd_graph_->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(relu_bwd_graph_->check_support(handle_));
        CHECK_CUDNN_FE(relu_bwd_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
    }

    void build_variant_packs() {
        vp_fwd_.reserve(4);
        vp_fwd_[fwd_x_]   = d_x;
        vp_fwd_[fwd_y_]   = d_y;
        vp_fwd_[fwd_mask_] = d_mask;

        vp_bwd_.reserve(3);
        vp_bwd_[bwd_dy_] = d_dy;
        vp_bwd_[bwd_y_]  = d_y;  // 复用 FWD 输出
        vp_bwd_[bwd_dx_] = d_dx;
    }

    void warmup_and_capture_graphs() {
        // 传统模式预热 5 次
        for (int i = 0; i < 5; ++i) {
            CHECK_CUDNN_FE(relu_fwd_graph_->execute(handle_, vp_fwd_, nullptr));
            CHECK_CUDNN_FE(relu_bwd_graph_->execute(handle_, vp_bwd_, nullptr));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        // 捕获 FWD Graph
        CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        CHECK_CUDNN_FE(relu_fwd_graph_->execute(handle_, vp_fwd_, nullptr));
        CHECK_CUDA(cudaStreamEndCapture(stream_, &graph_fwd_));

        cudaGraphNode_t error_node = nullptr;
        char log_buffer[2048] = {};
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_fwd_, graph_fwd_, &error_node, log_buffer, sizeof(log_buffer)));

        // 捕获 BWD Graph
        CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        CHECK_CUDNN_FE(relu_bwd_graph_->execute(handle_, vp_bwd_, nullptr));
        CHECK_CUDA(cudaStreamEndCapture(stream_, &graph_bwd_));

        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_bwd_, graph_bwd_, &error_node, log_buffer, sizeof(log_buffer)));

        // Graph 预热 5 次
        for (int i = 0; i < 5; ++i) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_));
            CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

public:
    void run() {
        warmup_and_capture_graphs();

        // FWD 计时 100 次
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_));
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

        // BWD 计时 100 次
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_));
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / 100.0;

        double mb = total_elements_ * sizeof(__half) / (1024.0 * 1024.0);

        std::cout << "===== ReLU cuDNN Frontend FP16 (1 GPU) =====\n"
                  << "  Shape: " << N << "x" << H << "x" << W << "x" << C
                  << " (" << mb << " MB FP16)\n"
                  << "  FWD Avg: " << std::fixed << fwd_us << " us/iter\n"
                  << "  BWD Avg: " << std::fixed << bwd_us << " us/iter\n";
    }
};

int main() {
    ReLUCuDNNFrontendBenchmark bench;
    bench.run();
    return 0;
}
