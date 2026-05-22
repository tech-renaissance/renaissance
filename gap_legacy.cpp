/**
 * @file gap.cpp
 * @brief 技术觉醒V4 - Global Average Pool (GAP) FP16 极致性能实现
 * @version 1.0.0
 * @date 2026-04-17
 * @author 技术觉醒团队
 * @note 依赖项: cuDNN Frontend 1.17, CUDA 13.1, cuDNN 9.17
 * @note 所属系列: fp16
 *
 * @note 核心设计原则:
 *   1. 严格单流 stream_comp_1_，彻底消除多流同步开销
 *   2. NHWC 物理布局强制对齐 (256B 起始, 128B 行步幅)
 *   3. FP16 输入输出 + FP32 中间累加/计算，契合 AMP 训练范式
 *   4. 正向/反向独立建图、独立捕获、独立计时
 *   5. 虚张量优化：中间态零显存写回，仅实张量绑定内存
 *   6. Average Pooling 反向为确定性广播，关闭 index 生成节省带宽
 */

// ████████████████████████████████████████████████████████████████████████████
// ████ 引入公共基础设施层（FP16专用，单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
#include "ta_v4_common_fp16.hpp"

// ████████████████████████████████████████████████████████████████████████████
// ████ GAP Backward CUDA Kernel 声明 ████
// ████████████████████████████████████████████████████████████████████████████
// Kernel实现在gap_backward.cu中
extern void gap_backward_kernel(const __half* __restrict__ dy,
                                __half* __restrict__ dx,
                                const int N, const int C, const int H, const int W,
                                const float scale);

namespace fe = cudnn_frontend;

// ████████████████████████████████████████████████████████████████████████████
// ████ GAP Benchmark 核心类 ████
// ████████████████████████████████████████████████████████████████████████████
class GAPBenchmark {
private:
    // ========== 严格单流与句柄 ==========
    cudnnHandle_t handle_comp_1_;
    cudaStream_t stream_comp_1_;

    // 维度与配置
    const int64_t N_, C_, H_, W_;
    const bool use_graph_;
    const fe::HeurMode_t heur_mode_;

    // cuDNN Frontend Graph 对象
    std::shared_ptr<fe::graph::Graph> graph_fwd_;
    std::shared_ptr<fe::graph::Graph> graph_bwd_;

    // Tensor 属性引用 (VariantPack 键值)
    std::shared_ptr<fe::graph::Tensor_attributes> fwd_in_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> fwd_out_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_dy_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_dx_tensor_;

    // GPU 内存指针
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    void* d_grad_output_ = nullptr;
    void* d_grad_input_ = nullptr;

    // Workspace
    void* d_workspace_fwd_ = nullptr;
    void* d_workspace_bwd_ = nullptr;
    size_t ws_fwd_size_ = 0;
    size_t ws_bwd_size_ = 0;

    // CUDA Graph 执行对象 (正/反向完全独立)
    cudaGraph_t graph_raw_fwd_ = nullptr;
    cudaGraphExec_t graph_exec_fwd_ = nullptr;
    cudaGraph_t graph_raw_bwd_ = nullptr;
    cudaGraphExec_t graph_exec_bwd_ = nullptr;
    bool graph_captured_fwd_ = false;
    bool graph_captured_bwd_ = false;

    // 静态 Variant Packs (零堆分配)
    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_fwd_;
    VariantPack vp_bwd_;

    // ========== 对齐与布局计算工具 ==========
    /**
     * @brief 计算 NHWC 物理内存布局的 stride（适配 cuDNN Frontend API）
     *
     * @note 关键概念理解：
     *   1. cuDNN Frontend API 的 set_dim() 参数顺序是固定的 {N, C, H, W}
     *   2. 但实际物理内存布局由 stride 参数决定
     *   3. 通过设置正确的 stride，可以让 cuDNN 正确理解 NHWC 布局的数据
     *
     * @note NHWC 物理内存布局：
     *   数据在内存中按 [n][h][w][c] 顺序排列，对线性索引 idx：
     *   - idx = n * C * H * W + h * W * C + w * C + c
     *   - 这意味着在空间位置 (h,w) 处，所有 C 个通道的数据是连续存储的
     *
     * @note stride 含义（对每个维度 d，stride[d] 表示在该维度上移动 1 步需要跳过的元素数）：
     *   - stride[N] = C * H * W (跳过一个 batch)
     *   - stride[C] = 1         (channel 维度连续，这是 NHWC 的关键特征！)
     *   - stride[H] = C * W     (跳过一行，即 W 个像素，每个像素有 C 个通道)
     *   - stride[W] = C         (跳过一列，即 1 个像素的 C 个通道)
     *
     * @note cuDNN API 约定：
     *   set_dim({N, C, H, W}) 和 set_stride({sN, sC, sH, sW}) 的参数是按位置对应的：
     *   - dim[0]=N, stride[0]=sN (batch 维度)
     *   - dim[1]=C, stride[1]=sC (channel 维度)
     *   - dim[2]=H, stride[2]=sH (height 维度)
     *   - dim[3]=W, stride[3]=sW (width 维度)
     *
     * @note 为什么这样设置是对的：
     *   stride[1]=1 告诉 cuDNN："dim[1] 这个维度（我们称之为 C）在内存中是连续的"
     *   这正是 NHWC 布局的定义！cuDNN 会根据 stride 自动推断数据格式
     *
     * @note 包含 128B 行步幅校验，符合硬件对齐要求 (Requirement #9)
     */
    static inline std::vector<int64_t> make_nhwc_stride(int64_t /*N*/, int64_t C, int64_t H, int64_t W) {
        // NHWC 物理布局的 stride 计算
        int64_t stride_W = C;          // W 维度：跳过 1 个像素的 C 个通道
        int64_t stride_H = C * W;      // H 维度：跳过 1 行（W 个像素，每个 C 个通道）
        int64_t stride_C = 1;          // C 维度：连续存储（NHWC 的关键！）
        int64_t stride_N = C * H * W;  // N 维度：跳过 1 个 batch (C*H*W 个元素)

        // 【Requirement 9】校验行步幅必须是 128B 整数倍
        // 注意：对于 H=1 或 W=1 的特殊情况（如 GAP 输出 1x1），行步幅可能不满足对齐要求
        size_t row_bytes = static_cast<size_t>(W) * static_cast<size_t>(C) * sizeof(__half);
        if (H > 1 && W > 1 && row_bytes % 128 != 0) {
            std::cerr << "[FATAL] Row stride (" << row_bytes << " bytes) is NOT 128-byte aligned! "
                      << "W=" << W << " C=" << C << " violates hardware constraint." << std::endl;
            exit(EXIT_FAILURE);
        }
        return {stride_N, stride_C, stride_H, stride_W};
    }

public:
    explicit GAPBenchmark(const Config& config)
        : N_(config.batch_size), C_(config.in_channels),
          H_(config.input_size), W_(config.input_size),
          use_graph_(config.use_graph),
          heur_mode_(config.get_heur_mode())
    {
        // 1. 创建唯一计算流与句柄
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDNN(cudnnCreate(&handle_comp_1_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_1_, stream_comp_1_));

        // 2. 显存分配与数据初始化 (不计时)
        allocate_memory();
        initialize_data();

        // 3. 构建计算图
        build_forward_graph();
        build_backward_graph();

        // 4. 分配 Workspace
        allocate_workspaces();

        // 5. 绑定静态 Variant Pack
        build_static_variant_packs();

        // 6. 预热期 (触发 Heuristic 选核 + CUDA Graph 捕获，不计时)
        warmup_and_capture();
    }

    ~GAPBenchmark() {
        if (graph_exec_fwd_) CHECK_CUDA(cudaGraphExecDestroy(graph_exec_fwd_));
        // graph_raw_fwd_已在warmup_and_capture中销毁

        if (graph_exec_bwd_) CHECK_CUDA(cudaGraphExecDestroy(graph_exec_bwd_));
        // graph_raw_bwd_已在warmup_and_capture中销毁

        if (d_input_)       CHECK_CUDA(cudaFree(d_input_));
        if (d_output_)      CHECK_CUDA(cudaFree(d_output_));
        if (d_grad_output_) CHECK_CUDA(cudaFree(d_grad_output_));
        if (d_grad_input_)  CHECK_CUDA(cudaFree(d_grad_input_));
        if (d_workspace_fwd_) CHECK_CUDA(cudaFree(d_workspace_fwd_));
        // d_workspace_bwd_未使用

        CHECK_CUDNN(cudnnDestroy(handle_comp_1_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_1_));
    }

    // ==================== 独立正向计时 ====================
    /**
     * @brief 正向传播性能基准测试
     * @param iterations 迭代次数（默认500）
     * @note 包含输入显存读取和输出显存写入时间
     * @return 平均每次迭代时间（毫秒）
     */
    double benchmark_forward(int iterations = 500) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) execute_forward();

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        auto t1 = std::chrono::high_resolution_clock::now();

        double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
        return fwd_ms;
    }

    // ==================== 独立反向计时 ====================
    /**
     * @brief 反向传播性能基准测试
     * @param iterations 迭代次数（默认500）
     * @note 包含输入显存读取和输出显存写入时间
     * @return 平均每次迭代时间（毫秒）
     */
    double benchmark_backward(int iterations = 500) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) execute_backward();

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        auto t1 = std::chrono::high_resolution_clock::now();

        double bwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
        return bwd_ms;
    }

private:
    void allocate_memory() {
        size_t in_el = N_ * C_ * H_ * W_;
        size_t out_el = N_ * C_ * 1 * 1;

        // 【Requirement 9】基地址 256B 对齐
        d_input_        = allocate_aligned_gpu_memory(in_el  * sizeof(__half));
        d_grad_input_   = allocate_aligned_gpu_memory(in_el  * sizeof(__half));
        d_output_       = allocate_aligned_gpu_memory(out_el * sizeof(__half));
        d_grad_output_  = allocate_aligned_gpu_memory(out_el * sizeof(__half));
    }

    void initialize_data() {
        int64_t in_el = N_ * C_ * H_ * W_;
        int64_t out_el = N_ * C_ * 1 * 1;
        // 随机数生成不计时
        initialize_random_fp16(d_input_,        in_el,  42);
        initialize_random_fp16(d_grad_output_,  out_el, 88);
        // 输出缓冲区清零避免脏数据污染
        CHECK_CUDA(cudaMemset(d_output_, 0, out_el * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_grad_input_, 0, in_el * sizeof(__half)));
    }

    /**
     * @brief 构建前向传播计算图
     * @note 使用 Resample (AVERAGE_POOLING) 实现 GAP
     * @note 使用 padding 确保输出为 1x1
     */
    void build_forward_graph() {
        graph_fwd_ = std::make_shared<fe::graph::Graph>();
        // I/O: FP16, 中间计算/累加: FP32 (AMP 最优实践)
        graph_fwd_->set_io_data_type(fe::DataType_t::HALF)
                   .set_intermediate_data_type(fe::DataType_t::FLOAT)
                   .set_compute_data_type(fe::DataType_t::FLOAT);

        // 输入张量 [N, C, H, W] - NHWC 物理布局
        //
        // 重要说明：
        // 1. set_dim({N, C, H, W}) 的参数顺序是 cuDNN API 的约定，不能改变
        //    - dim[0] 对应第一个维度（我们理解为 Batch）
        //    - dim[1] 对应第二个维度（我们理解为 Channel）
        //    - dim[2] 对应第三个维度（我们理解为 Height）
        //    - dim[3] 对应第四个维度（我们理解为 Width）
        //
        // 2. set_stride({sN, sC, sH, sW}) 告诉 cuDNN 每个维度的内存跨度
        //    - stride 通过数值定义实际的物理内存布局
        //    - stride[1]=1 表示 dim[1] 在内存中连续，这正是 NHWC 的特征！
        //
        // 3. 数据排列示例（N=1, C=2, H=2, W=2）：
        //    内存线性索引: [0,1,2,3,4,5,6,7,...]
        //    对应位置:   [(0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1), (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),...]
        //    解读:     [batch=0, h=0, w=0, c=0], [batch=0, h=0, w=0, c=1], [batch=0, h=0, w=1, c=0],...
        //    注意:     在每个空间位置 (h,w)，所有 C 个通道数据是连续的
        //
        // 4. CPU -> GPU 数据拷贝时：
        //    CPU 端需要按 NHWC 顺序准备数据，即使用索引公式：
        //    idx = n * C * H * W + h * W * C + w * C + c
        //    cudaMemcpy 会将这个线性数组按原序拷贝到 GPU，cuDNN 根据 stride 正确理解
        //
        fwd_in_tensor_ = graph_fwd_->tensor(fe::graph::Tensor_attributes()
            .set_name("gap_x")
            .set_dim({N_, C_, H_, W_})
            .set_stride(make_nhwc_stride(N_, C_, H_, W_))
            .set_data_type(fe::DataType_t::HALF));

        // GAP 本质是覆盖全空间维度的 Average Pooling
        // window=[H,W], stride=[H,W] 保证输出为1x1
        fe::graph::Resample_attributes pool_opts;
        pool_opts.set_name("gap_resample")
                 .set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
                 .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
                 .set_window({H_, W_})
                 .set_stride({H_, W_})
                 .set_pre_padding({0, 0})
                 .set_post_padding({0, 0})
                 // Average Pooling 反向为确定性广播，不需要 index
                 // 关闭 index 生成以节省显存带宽（Requirement #5优化）
                 .set_generate_index(false);

        auto resample_outputs = graph_fwd_->resample(fwd_in_tensor_, pool_opts);
        fwd_out_tensor_ = resample_outputs[0];

        // 输出张量：标记为实张量，强制 NHWC 1x1 布局
        //
        // GAP 输出维度为 [N, C, 1, 1]，每个样本的每个通道只有一个标量值
        // 仍然使用 NHWC 格式的 stride 描述：
        //   - stride[N] = C * 1 * 1 = C (每个样本有 C 个输出值)
        //   - stride[C] = 1         (channel 连续)
        //   - stride[H] = C         (H=1，每个 H 对应 C 个通道)
        //   - stride[W] = C         (W=1，每个 W 对应 C 个通道)
        //
        // 物理内存排列（N=2, C=3, H=1, W=1）：
        //   [y(0,0), y(0,1), y(0,2), y(1,0), y(1,1), y(1,2)]
        //   即先输出 batch 0 的所有通道，再输出 batch 1 的所有通道
        //
        fwd_out_tensor_->set_output(true)
            .set_name("gap_y")
            .set_dim({N_, C_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_, 1, 1))
            .set_data_type(fe::DataType_t::HALF);

        CHECK_CUDNN_FE(graph_fwd_->validate());
        CHECK_CUDNN_FE(graph_fwd_->build_operation_graph(handle_comp_1_));
        // GAP层固定使用Heuristic Mode B，不受search_mode参数影响
        CHECK_CUDNN_FE(graph_fwd_->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(graph_fwd_->check_support(handle_comp_1_));
        CHECK_CUDNN_FE(graph_fwd_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        ws_fwd_size_ = graph_fwd_->get_workspace_size();
    }

    /**
     * @brief 构建反向传播计算图
     * @note GAP 反向数学: dx = dy * (1.0 / (H*W))，利用 Pointwise 隐式广播
     * @note 不使用昂贵的 Resample_backward（官方标注 To be supported soon）
     */
    void build_backward_graph() {
        // GAP backward使用自定义CUDA kernel，不使用cuDNN图API
        // 不需要构建graph_bwd_
        graph_bwd_ = nullptr;
        ws_bwd_size_ = 0;
    }

    void allocate_workspaces() {
        if (ws_fwd_size_ > 0) d_workspace_fwd_ = allocate_aligned_gpu_memory(ws_fwd_size_);
        if (ws_bwd_size_ > 0) d_workspace_bwd_ = allocate_aligned_gpu_memory(ws_bwd_size_);
    }

    void build_static_variant_packs() {
        vp_fwd_.reserve(2);
        vp_fwd_[fwd_in_tensor_]  = d_input_;
        vp_fwd_[fwd_out_tensor_] = d_output_;

        vp_bwd_.reserve(2);
        vp_bwd_[bwd_dy_tensor_]  = d_grad_output_;
        vp_bwd_[bwd_dx_tensor_]  = d_grad_input_;
    }

    void execute_forward() {
        if (use_graph_ && graph_captured_fwd_ && graph_exec_fwd_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_comp_1_));
        } else {
            CHECK_CUDNN_FE(graph_fwd_->execute(handle_comp_1_, vp_fwd_, d_workspace_fwd_));
        }
    }

    void execute_backward() {
        if (use_graph_ && graph_captured_bwd_ && graph_exec_bwd_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_comp_1_));
        } else {
            // GAP backward: 调用自定义CUDA kernel
            const int total_elements = N_ * C_ * H_ * W_;
            const int block_size = 256;
            const int grid_size = (total_elements + block_size - 1) / block_size;
            float scale = 1.0f / static_cast<float>(H_ * W_);

            int N = N_, C = C_, H = H_, W = W_;

            void* args[] = {
                &d_grad_output_,
                &d_grad_input_,
                &N,
                &C,
                &H,
                &W,
                &scale
            };

            CHECK_CUDA(cudaLaunchKernel(
                reinterpret_cast<const void*>(gap_backward_kernel),
                grid_size,
                block_size,
                args,
                0,
                stream_comp_1_
            ));
        }
    }

    /**
     * @brief 预热与 CUDA Graph 捕获
     * @note 阶段1：传统执行预热（触发 Heuristic 选核）
     * @note 阶段2：捕获正向 CUDA Graph
     * @note 阶段3：捕获反向 CUDA Graph
     */
    void warmup_and_capture() {
        constexpr int WARMUP_ITERS = 20;

        // 阶段1：传统执行预热 (触发 Heuristic 选核 & JIT)
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            execute_forward();
            execute_backward();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        if (!use_graph_) {
            return;
        }

        // 阶段2：捕获正向 CUDA Graph
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));
        CHECK_CUDNN_FE(graph_fwd_->execute(handle_comp_1_, vp_fwd_, d_workspace_fwd_));
        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_raw_fwd_));

        cudaGraphNode_t err_node_fwd;
        char log_fwd[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_fwd_, graph_raw_fwd_, &err_node_fwd, log_fwd, sizeof(log_fwd)));
        graph_captured_fwd_ = true;
        CHECK_CUDA(cudaGraphDestroy(graph_raw_fwd_)); // 释放原始图，保留 Exec

        // 阶段3：捕获反向 CUDA Graph
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));

        const int total_elements = N_ * C_ * H_ * W_;
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        float scale = 1.0f / static_cast<float>(H_ * W_);

        int N = N_, C = C_, H = H_, W = W_;

        void* args[] = {
            &d_grad_output_,
            &d_grad_input_,
            &N,
            &C,
            &H,
            &W,
            &scale
        };

        CHECK_CUDA(cudaLaunchKernel(
            reinterpret_cast<const void*>(gap_backward_kernel),
            grid_size,
            block_size,
            args,
            0,
            stream_comp_1_
        ));

        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_raw_bwd_));

        cudaGraphNode_t err_node_bwd;
        char log_bwd[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_bwd_, graph_raw_bwd_, &err_node_bwd, log_bwd, sizeof(log_bwd)));
        graph_captured_bwd_ = true;
        CHECK_CUDA(cudaGraphDestroy(graph_raw_bwd_)); // 释放原始图，保留 Exec

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ Main 入口 ████
// ████████████████████████████████████████████████████████████████████████████
int main(int argc, char** argv) {
    Config config = parse_arguments(argc, argv);

    try {
        GAPBenchmark bench(config);

        double fwd_ms = bench.benchmark_forward(500);
        double bwd_ms = bench.benchmark_backward(500);
        double total_ms = fwd_ms + bwd_ms;

        std::cout << "\nTime (Unit: ms):" << std::endl;
        std::cout << "Forward:  " << std::fixed << std::setprecision(5) << fwd_ms << std::endl;
        std::cout << "Backward: " << std::fixed << std::setprecision(5) << bwd_ms << std::endl;
        std::cout << "Total:    " << std::fixed << std::setprecision(5) << total_ms << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
