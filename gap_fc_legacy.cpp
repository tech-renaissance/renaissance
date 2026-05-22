/**
 * @file gap_fc.cpp
 * @brief 技术觉醒V4 - GAP + FC 融合测试 FP16 极致性能实现
 * @version 2.1.0
 * @date 2026-04-18
 * @author 技术觉醒团队
 * @note 依赖项: cuDNN Frontend 1.17, CUDA 13.1, cuDNN 9.17
 * @note 所属系列: fp16
 *
 * @note 核心设计原则:
 *   1. GAP + FC 融合测试，模拟 ResNet-50 末端：GAP -> FC
 *   2. 参数配置：GAP 参数 (N, C, H, W) + num_classes (FC 输出维度)
 *   3. 双流正向架构：stream_comp_1_(GAP) + stream_comp_2_(FC)，事件同步
 *   4. 三流反向架构：stream_comp_2_(dgrad) || stream_comp_3_(wgrad+bias) -> stream_comp_1_(GAP bwd)
 *   5. NHWC 物理布局强制对齐 (256B 起始, 128B 行步幅)
 *   6. FP16 输入输出 + FP32 中间累加/计算，契合 AMP 训练范式
 *   7. 统一图捕获：前向一个图，反向一个图
 *   8. 中间张量：GAP输出设为实张量，分配GPU内存
 *   9. 独立workspace：每个流独立workspace，跨图复用（取最大值）
 */

// ████████████████████████████████████████████████████████████████████████████
// ████ 引入公共基础设施层（FP16专用，单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
#include "ta_v4_common_fp16.hpp"
#include <iomanip>

// ████████████████████████████████████████████████████████████████████████████
// ████ GAP Backward CUDA Kernel 声明 ████
// ████████████████████████████████████████████████████████████████████████████
extern void gap_backward_kernel(const __half* __restrict__ dy,
                                __half* __restrict__ dx,
                                const int N, const int C, const int H, const int W,
                                const float scale);

// ████████████████████████████████████████████████████████████████████████████
// ████ FC 偏置梯度计算Kernel声明（在fc_bias_gradient.cu中实现） ████
// ████████████████████████████████████████████████████████████████████████████
extern "C" void compute_bias_gradient(const __half* dY, __half* dB, int N, int C, int C_aligned);

namespace fe = cudnn_frontend;

// ████████████████████████████████████████████████████████████████████████████
// ████ GAP+FC 融合配置结构体（扩展 Config） ████
// ████████████████████████████████████████████████████████████████████████████
struct GAPFCConfig : public Config {
    int64_t num_classes = 1000;  // FC 输出维度（分类数）
    bool use_bias = true;         // FC 是否使用偏置

    fe::HeurMode_t get_heur_mode() const {
        return (search_mode == SearchMode::HEURISTIC_A) ?
               fe::HeurMode_t::A : fe::HeurMode_t::B;
    }

    void print() const {
        std::cout << "=== GAP+FC Configuration ===" << std::endl;
        std::cout << "Batch Size:    " << batch_size << std::endl;
        std::cout << "Input:         [" << batch_size << ", " << in_channels
                  << ", " << input_size << ", " << input_size << "]" << std::endl;
        std::cout << "Num Classes:   " << num_classes << std::endl;
        std::cout << "Use Bias:      " << (use_bias ? "True" : "False") << std::endl;
        std::cout << "Search Mode:   " << (search_mode == SearchMode::HEURISTIC_A ? "A" : "B") << std::endl;
        std::cout << "CUDA Graph:    " << (use_graph ? "Enabled" : "Disabled") << std::endl;
        std::cout << "=============================" << std::endl;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ GAP+FC 专用参数解析函数 ████
// ████████████████████████████████████████████████████████████████████████████
inline GAPFCConfig parse_gapfc_arguments(int argc, char** argv) {
    GAPFCConfig config;

    // 先使用通用参数解析器处理公共参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--num_classes" && i + 1 < argc) {
            config.num_classes = std::atoll(argv[++i]);
        } else if (arg == "--bias" && i + 1 < argc) {
            std::string val = argv[++i];
            config.use_bias = (val == "True" || val == "true" || val == "1");
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --batch_size <N>      Batch size (default: 512)" << std::endl;
            std::cout << "  --input_size <S>     Input H=W (default: 7)" << std::endl;
            std::cout << "  --in_channels <C>    Input channels (default: 2048)" << std::endl;
            std::cout << "  --num_classes <K>    Number of classes (default: 1000)" << std::endl;
            std::cout << "  --bias <True/False>   FC use bias (default: True)" << std::endl;
            std::cout << "  --mode <A/B>          Heuristic mode (default: B)" << std::endl;
            std::cout << "  --graph <true/false>  Enable CUDA Graph (default: true)" << std::endl;
            std::cout << std::endl;
            std::cout << "ResNet-50 End Test:" << std::endl;
            std::cout << "  " << argv[0] << " --batch_size 512 --input_size 7 --in_channels 2048 "
                      << "--num_classes 1000 --bias True --mode B --graph true" << std::endl;
            exit(EXIT_SUCCESS);
        }
    }

    // 使用通用参数解析器处理剩余参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--batch_size" && i + 1 < argc) {
            config.batch_size = std::atoll(argv[++i]);
        } else if (arg == "--input_size" && i + 1 < argc) {
            config.input_size = std::atoll(argv[++i]);
        } else if (arg == "--in_channels" && i + 1 < argc) {
            config.in_channels = std::atoll(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "A") {
                config.search_mode = GAPFCConfig::SearchMode::HEURISTIC_A;
            } else if (mode == "B") {
                config.search_mode = GAPFCConfig::SearchMode::HEURISTIC_B;
            }
        } else if (arg == "--graph" && i + 1 < argc) {
            std::string val = argv[++i];
            config.use_graph = (val == "true" || val == "1");
        }
    }

    return config;
}

// ████████████████████████████████████████████████████████████████████████████
// ████ GAP+FC 融合 Benchmark 核心类 ████
// ████████████████████████████████████████████████████████████████████████████
class GAPFCBenchmark {
private:
    // ========== 多流与句柄 ==========
    cudnnHandle_t handle_comp_1_;   // 流1句柄（正向GAP + 反向GAP backward）
    cudnnHandle_t handle_comp_2_;   // 流2句柄（正向FC + 反向FC dgrad）
    cudnnHandle_t handle_comp_3_;   // 流3句柄（反向FC wgrad + bias grad）

    cudaStream_t stream_comp_1_;    // 计算流1（正向GAP + 反向GAP backward）
    cudaStream_t stream_comp_2_;    // 计算流2（正向FC + 反向FC dgrad）
    cudaStream_t stream_comp_3_;    // 计算流3（反向FC wgrad + bias grad）

    // ========== 同步事件 ==========
    cudaEvent_t event_fwd_gap_done_;    // GAP完成事件
    cudaEvent_t event_fwd_fc_done_;     // FC完成事件
    cudaEvent_t event_bwd_s2_done_;     // S2完成事件（dgrad）
    cudaEvent_t event_bwd_s3_done_;     // S3完成事件（wgrad + bias grad）

    // ========== 维度与配置 ==========
    const int64_t N_;                  // batch_size
    const int64_t C_;                  // in_channels (GAP输入通道数 = GAP输出通道数 = FC输入特征数)
    const int64_t H_;                  // input_size (GAP输入H = GAP输入W)
    const int64_t K_;                  // num_classes (FC输出维度)
    const bool use_bias_;
    const bool use_graph_;
    const fe::HeurMode_t heur_mode_;

    // ========== 对齐后的维度 ==========
    int64_t C_aligned_;
    int64_t K_aligned_;

    // ========== cuDNN Frontend Graph 对象（GAP和FC分离） ==========
    std::shared_ptr<fe::graph::Graph> gap_fwd_graph_;
    std::shared_ptr<fe::graph::Graph> fc_fwd_graph_;
    std::shared_ptr<fe::graph::Graph> fc_dgrad_graph_;  // FC反向输入梯度图
    std::shared_ptr<fe::graph::Graph> fc_wgrad_graph_;  // FC反向权重梯度图

    // ========== GAP Tensor 属性 ==========
    std::shared_ptr<fe::graph::Tensor_attributes> gap_in_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> gap_out_tensor_;

    // ========== FC Tensor 属性 ==========
    std::shared_ptr<fe::graph::Tensor_attributes> fc_in_tensor_;    // FC正向输入（即GAP输出）
    std::shared_ptr<fe::graph::Tensor_attributes> fc_W_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> fc_B_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> fc_out_tensor_;

    // ========== FC反向 Tensor 属性 ==========
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_fc_in_tensor_;  // FC反向输入（即GAP输出）
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_fc_W_tensor_;  // FC反向权重

    // ========== 反向 Tensor 属性 ==========
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_dy_tensor_;        // FC dgrad的梯度输入
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_wgrad_dy_tensor_; // FC wgrad的梯度输入（与dgrad共享内存）
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_gap_dy_tensor_;   // GAP的梯度输入（即FC dgrad输出）
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_dx_tensor_;       // GAP的梯度输出
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_fc_dW_tensor_;   // FC权重梯度

    // ========== GPU 内存指针 ==========
    void* d_X_;           // GAP输入 [N, C, H, W]
    void* d_gap_out_;     // GAP输出 [N, C, 1, 1] (实张量，中间存储)
    void* d_fc_W_;        // FC权重 [K, C, 1, 1]
    void* d_fc_B_;        // FC偏置 [K]
    void* d_Y_;           // FC最终输出 [N, K, 1, 1]

    void* d_dY_;          // 上游梯度 [N, K, 1, 1]
    void* d_dX_;          // 最终梯度 [N, C, H, W]
    void* d_fc_dW_;       // FC权重梯度 [K, C, 1, 1]
    void* d_fc_dB_;       // FC偏置梯度 [K]
    void* d_gap_dy_;      // FC反向输出的中间梯度 [N, C, 1, 1] (实张量，中间存储)

    // ========== Workspace（每个流独立） ==========
    // Stream 1: GAP forward + GAP backward
    void* d_workspace_s1_ = nullptr;
    size_t ws_s1_size_ = 0;

    // Stream 2: FC forward + FC dgrad
    void* d_workspace_s2_ = nullptr;
    size_t ws_s2_size_ = 0;

    // Stream 3: FC wgrad + bias grad
    void* d_workspace_s3_ = nullptr;
    size_t ws_s3_size_ = 0;

    // ========== CUDA Graph ==========
    cudaGraph_t graph_fwd_;
    cudaGraph_t graph_bwd_;
    cudaGraphExec_t graph_exec_fwd_;
    cudaGraphExec_t graph_exec_bwd_;
    bool fwd_graph_captured_;
    bool bwd_graph_captured_;

    // ========== Variant Packs ==========
    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_gap_fwd_;
    VariantPack vp_fc_fwd_;
    VariantPack vp_fc_dgrad_;  // FC反向输入梯度variant pack
    VariantPack vp_fc_wgrad_;  // FC反向权重梯度variant pack

    // ========== 辅助函数：计算NHWC格式的stride ==========
    static inline std::vector<int64_t> make_nhwc_stride(int64_t /*N*/, int64_t C, int64_t H, int64_t W) {
        return {C * H * W, 1, C * W, C};
    }

    // ========== 计算对齐后的维度 ==========
    void compute_aligned_dimensions() {
        C_aligned_ = align_to(C_ * sizeof(__half), 128) / sizeof(__half);
        K_aligned_ = align_to(K_ * sizeof(__half), 128) / sizeof(__half);
    }

public:
    explicit GAPFCBenchmark(const GAPFCConfig& config)
        : N_(config.batch_size),
          C_(config.in_channels),
          H_(config.input_size),
          K_(config.num_classes),
          use_bias_(config.use_bias),
          use_graph_(config.use_graph),
          heur_mode_(config.get_heur_mode()),
          fwd_graph_captured_(false),
          bwd_graph_captured_(false)
    {
        compute_aligned_dimensions();

        // 创建三个计算流
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_2_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_3_, cudaStreamNonBlocking));

        // 创建三个cuDNN句柄并绑定到对应的流
        CHECK_CUDNN(cudnnCreate(&handle_comp_1_));
        CHECK_CUDNN(cudnnCreate(&handle_comp_2_));
        CHECK_CUDNN(cudnnCreate(&handle_comp_3_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_1_, stream_comp_1_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_2_, stream_comp_2_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_3_, stream_comp_3_));

        // 创建同步事件
        CHECK_CUDA(cudaEventCreateWithFlags(&event_fwd_gap_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_fwd_fc_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_bwd_s2_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_bwd_s3_done_, cudaEventDisableTiming));

        // 分配内存
        allocate_memory();

        // 初始化数据
        initialize_data();

        // 构建计算图（GAP和FC分离）
        build_gap_forward_graph();
        build_fc_forward_graph();
        build_fc_dgrad_graph();
        build_fc_wgrad_graph();

        // 分配workspace
        allocate_workspace();

        // 构建variant packs
        build_variant_packs();

        // 打印配置
        config.print();
    }

    ~GAPFCBenchmark() {
        // 销毁CUDA Graph
        if (graph_exec_fwd_) {
            cudaError_t err = cudaGraphExecDestroy(graph_exec_fwd_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphExecDestroy(fwd) failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        if (graph_fwd_) {
            cudaError_t err = cudaGraphDestroy(graph_fwd_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphDestroy(fwd) failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        if (graph_exec_bwd_) {
            cudaError_t err = cudaGraphExecDestroy(graph_exec_bwd_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphExecDestroy(bwd) failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        if (graph_bwd_) {
            cudaError_t err = cudaGraphDestroy(graph_bwd_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphDestroy(bwd) failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        // 销毁同步事件
        CHECK_CUDA(cudaEventDestroy(event_fwd_gap_done_));
        CHECK_CUDA(cudaEventDestroy(event_fwd_fc_done_));
        CHECK_CUDA(cudaEventDestroy(event_bwd_s2_done_));
        CHECK_CUDA(cudaEventDestroy(event_bwd_s3_done_));

        // 释放GPU内存
        CHECK_CUDA(cudaFree(d_X_));
        CHECK_CUDA(cudaFree(d_gap_out_));
        CHECK_CUDA(cudaFree(d_fc_W_));
        CHECK_CUDA(cudaFree(d_Y_));
        CHECK_CUDA(cudaFree(d_dY_));
        CHECK_CUDA(cudaFree(d_dX_));
        CHECK_CUDA(cudaFree(d_fc_dW_));
        CHECK_CUDA(cudaFree(d_gap_dy_));

        if (use_bias_) {
            CHECK_CUDA(cudaFree(d_fc_B_));
            CHECK_CUDA(cudaFree(d_fc_dB_));
        }

        // 释放workspace（每个流独立）
        if (d_workspace_s1_) CHECK_CUDA(cudaFree(d_workspace_s1_));
        if (d_workspace_s2_) CHECK_CUDA(cudaFree(d_workspace_s2_));
        if (d_workspace_s3_) CHECK_CUDA(cudaFree(d_workspace_s3_));

        // 销毁流和句柄
        CHECK_CUDNN(cudnnDestroy(handle_comp_1_));
        CHECK_CUDNN(cudnnDestroy(handle_comp_2_));
        CHECK_CUDNN(cudnnDestroy(handle_comp_3_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_1_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_2_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_3_));
    }

private:
    void allocate_memory() {
        // GAP输入: [N, C, H, W]  (H=W=input_size)
        const size_t X_size = N_ * C_ * H_ * H_ * sizeof(__half);
        // GAP输出 = FC输入: [N, C, 1, 1] (实张量，中间存储)
        const size_t gap_out_size = N_ * C_aligned_ * sizeof(__half);
        // FC权重: [K, C, 1, 1]
        const size_t W_size = K_ * C_aligned_ * sizeof(__half);
        // FC偏置: [K]
        const size_t B_size = K_aligned_ * sizeof(__half);
        // FC输出: [N, K, 1, 1]
        const size_t Y_size = N_ * K_aligned_ * sizeof(__half);

        d_X_ = allocate_aligned_gpu_memory(X_size);
        d_gap_out_ = allocate_aligned_gpu_memory(gap_out_size);  // 实张量，中间存储
        d_fc_W_ = allocate_aligned_gpu_memory(W_size);
        d_Y_ = allocate_aligned_gpu_memory(Y_size);

        if (use_bias_) {
            d_fc_B_ = allocate_aligned_gpu_memory(B_size);
        }

        // 反向内存
        d_dY_ = allocate_aligned_gpu_memory(Y_size);
        d_dX_ = allocate_aligned_gpu_memory(X_size);
        d_fc_dW_ = allocate_aligned_gpu_memory(W_size);
        d_gap_dy_ = allocate_aligned_gpu_memory(gap_out_size);  // 实张量，中间存储

        if (use_bias_) {
            d_fc_dB_ = allocate_aligned_gpu_memory(B_size);
        }
    }

    void initialize_data() {
        // 随机数初始化不计入计时
        const size_t X_elems = N_ * C_ * H_ * H_;  // H=W=input_size
        const size_t W_elems = K_ * C_;
        const size_t Y_elems = N_ * K_;

        initialize_random_fp16(d_X_, X_elems, 42);
        initialize_random_fp16(d_fc_W_, W_elems, 43);
        initialize_random_fp16(d_dY_, Y_elems, 44);

        if (use_bias_) {
            initialize_random_fp16(d_fc_B_, K_, 45);
        }
    }

    void build_gap_forward_graph() {
        gap_fwd_graph_ = std::make_shared<fe::graph::Graph>();
        gap_fwd_graph_->set_io_data_type(fe::DataType_t::HALF)
                        .set_intermediate_data_type(fe::DataType_t::FLOAT)
                        .set_compute_data_type(fe::DataType_t::FLOAT);

        // GAP输入: [N, C, H, W]
        gap_in_tensor_ = gap_fwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("gap_x")
            .set_dim({N_, C_, H_, H_})  // H=W=input_size
            .set_stride(make_nhwc_stride(N_, C_, H_, H_))
            .set_data_type(fe::DataType_t::HALF));

        // GAP操作：使用Resample实现
        fe::graph::Resample_attributes pool_opts;
        std::vector<int64_t> window_size = {H_, H_};  // H=W=input_size
        std::vector<int64_t> stride_size = {H_, H_};
        pool_opts.set_name("gap_resample")
                 .set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
                 .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
                 .set_window(window_size)
                 .set_stride(stride_size)
                 .set_pre_padding({0, 0})
                 .set_post_padding({0, 0})
                 .set_generate_index(false);

        auto gap_out = gap_fwd_graph_->resample(gap_in_tensor_, pool_opts);
        gap_out_tensor_ = gap_out[0];

        // 设置GAP输出为实张量（需要在FC图中使用）
        // 关键：stride必须使用C_aligned_，与物理内存分配一致！
        // 物理内存分配了N_*C_aligned_个元素，stride[0]=C_aligned_
        // 有效元素只有N_*C_个，但对齐部分的padding会被cuDNN自动处理
        gap_out_tensor_->set_output(true)
                       .set_name("gap_output")
                       .set_dim({N_, C_, 1, 1})
                       .set_stride(make_nhwc_stride(N_, C_aligned_, 1, 1))
                       .set_data_type(fe::DataType_t::HALF);

        // 构建执行计划（使用handle_comp_1_）
        CHECK_CUDNN_FE(gap_fwd_graph_->validate());
        CHECK_CUDNN_FE(gap_fwd_graph_->build_operation_graph(handle_comp_1_));
        CHECK_CUDNN_FE(gap_fwd_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(gap_fwd_graph_->check_support(handle_comp_1_));
        CHECK_CUDNN_FE(gap_fwd_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 更新S1 workspace大小（取GAP forward和GAP backward的最大值）
        size_t ws_gap_fwd = gap_fwd_graph_->get_workspace_size();
        ws_s1_size_ = std::max(ws_s1_size_, ws_gap_fwd);
    }

    void build_fc_forward_graph() {
        fc_fwd_graph_ = std::make_shared<fe::graph::Graph>();
        fc_fwd_graph_->set_io_data_type(fe::DataType_t::HALF)
                       .set_intermediate_data_type(fe::DataType_t::FLOAT)
                       .set_compute_data_type(fe::DataType_t::FLOAT);

        // FC输入（引用GAP输出的内存）- 保存为类成员
        fc_in_tensor_ = fc_fwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("fc_input")
            .set_dim({N_, C_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // FC权重: [K, C, 1, 1]
        fc_W_tensor_ = fc_fwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("fc_weight")
            .set_dim({K_, C_, 1, 1})
            .set_stride(make_nhwc_stride(K_, C_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 1x1卷积实现FC
        auto conv_attr = fe::graph::Conv_fprop_attributes()
            .set_padding({0, 0})
            .set_stride({1, 1})
            .set_dilation({1, 1});

        auto conv_out = fc_fwd_graph_->conv_fprop(fc_in_tensor_, fc_W_tensor_, conv_attr);
        conv_out->set_is_virtual(true);  // 虚张量优化

        // 添加偏置（如果启用）
        if (use_bias_) {
            fc_B_tensor_ = fc_fwd_graph_->tensor(fe::graph::Tensor_attributes()
                .set_name("fc_bias")
                .set_dim({1, K_, 1, 1})
                .set_stride(make_nhwc_stride(1, K_aligned_, 1, 1))
                .set_data_type(fe::DataType_t::HALF));

            auto add_attr = fe::graph::Pointwise_attributes()
                .set_name("bias_add")
                .set_mode(fe::PointwiseMode_t::ADD)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            fc_out_tensor_ = fc_fwd_graph_->pointwise(conv_out, fc_B_tensor_, add_attr);
        } else {
            fc_out_tensor_ = conv_out;
        }

        // 设置最终输出
        fc_out_tensor_->set_output(true)
                       .set_name("output")
                       .set_dim({N_, K_, 1, 1})
                       .set_stride(make_nhwc_stride(N_, K_aligned_, 1, 1))
                       .set_data_type(fe::DataType_t::HALF);

        // 构建执行计划（使用handle_comp_2_）
        CHECK_CUDNN_FE(fc_fwd_graph_->validate());
        CHECK_CUDNN_FE(fc_fwd_graph_->build_operation_graph(handle_comp_2_));
        CHECK_CUDNN_FE(fc_fwd_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(fc_fwd_graph_->check_support(handle_comp_2_));
        CHECK_CUDNN_FE(fc_fwd_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 更新S2 workspace大小（取FC forward和FC dgrad的最大值）
        size_t ws_fc_fwd = fc_fwd_graph_->get_workspace_size();
        ws_s2_size_ = std::max(ws_s2_size_, ws_fc_fwd);
    }

    void build_fc_dgrad_graph() {
        fc_dgrad_graph_ = std::make_shared<fe::graph::Graph>();
        fc_dgrad_graph_->set_io_data_type(fe::DataType_t::HALF)
                        .set_intermediate_data_type(fe::DataType_t::FLOAT)
                        .set_compute_data_type(fe::DataType_t::FLOAT);

        // 上游梯度: [N, K, 1, 1]
        bwd_dy_tensor_ = fc_dgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("fc_grad_output")
            .set_dim({N_, K_, 1, 1})
            .set_stride(make_nhwc_stride(N_, K_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // FC权重: [K, C, 1, 1]
        bwd_fc_W_tensor_ = fc_dgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("fc_weight")
            .set_dim({K_, C_, 1, 1})
            .set_stride(make_nhwc_stride(K_, C_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // FC输入梯度计算（dgrad）
        auto fc_dgrad_attr = fe::graph::Conv_dgrad_attributes()
            .set_padding({0, 0})
            .set_stride({1, 1})
            .set_dilation({1, 1});

        auto fc_dgrad_out = fc_dgrad_graph_->conv_dgrad(bwd_dy_tensor_, bwd_fc_W_tensor_, fc_dgrad_attr);
        fc_dgrad_out->set_is_virtual(true);

        bwd_gap_dy_tensor_ = fc_dgrad_out;  // 这是FC的输入梯度，也是GAP的输出梯度

        // 设置FC输入梯度输出为实张量（需要在GAP反向中使用）
        bwd_gap_dy_tensor_->set_output(true)
                           .set_name("fc_grad_input")
                           .set_dim({N_, C_, 1, 1})
                           .set_stride(make_nhwc_stride(N_, C_aligned_, 1, 1))
                           .set_data_type(fe::DataType_t::HALF);

        // 构建执行计划（使用handle_comp_2_）
        CHECK_CUDNN_FE(fc_dgrad_graph_->validate());
        CHECK_CUDNN_FE(fc_dgrad_graph_->build_operation_graph(handle_comp_2_));
        CHECK_CUDNN_FE(fc_dgrad_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(fc_dgrad_graph_->check_support(handle_comp_2_));
        CHECK_CUDNN_FE(fc_dgrad_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 更新S2 workspace大小（取FC forward和FC dgrad的最大值）
        size_t ws_fc_dgrad = fc_dgrad_graph_->get_workspace_size();
        ws_s2_size_ = std::max(ws_s2_size_, ws_fc_dgrad);
    }

    void build_fc_wgrad_graph() {
        fc_wgrad_graph_ = std::make_shared<fe::graph::Graph>();
        fc_wgrad_graph_->set_io_data_type(fe::DataType_t::HALF)
                        .set_intermediate_data_type(fe::DataType_t::FLOAT)
                        .set_compute_data_type(fe::DataType_t::FLOAT);

        // 上游梯度: [N, K, 1, 1] - 保存为类成员
        bwd_wgrad_dy_tensor_ = fc_wgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("fc_grad_output")
            .set_dim({N_, K_, 1, 1})
            .set_stride(make_nhwc_stride(N_, K_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // FC输入（即GAP输出）: [N, C, 1, 1]
        bwd_fc_in_tensor_ = fc_wgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("fc_input")
            .set_dim({N_, C_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // FC权重梯度计算（wgrad）
        auto fc_wgrad_attr = fe::graph::Conv_wgrad_attributes()
            .set_padding({0, 0})
            .set_stride({1, 1})
            .set_dilation({1, 1});

        bwd_fc_dW_tensor_ = fc_wgrad_graph_->conv_wgrad(bwd_wgrad_dy_tensor_, bwd_fc_in_tensor_, fc_wgrad_attr);
        bwd_fc_dW_tensor_->set_output(true)
                       .set_name("fc_grad_weight")
                       .set_dim({K_, C_, 1, 1})
                       .set_stride(make_nhwc_stride(K_, C_aligned_, 1, 1))
                       .set_data_type(fe::DataType_t::HALF);

        // 构建执行计划（使用handle_comp_3_）
        CHECK_CUDNN_FE(fc_wgrad_graph_->validate());
        CHECK_CUDNN_FE(fc_wgrad_graph_->build_operation_graph(handle_comp_3_));
        CHECK_CUDNN_FE(fc_wgrad_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(fc_wgrad_graph_->check_support(handle_comp_3_));
        CHECK_CUDNN_FE(fc_wgrad_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 更新S3 workspace大小
        ws_s3_size_ = fc_wgrad_graph_->get_workspace_size();
    }

    void allocate_workspace() {
        // Stream 1: GAP forward + GAP backward
        if (ws_s1_size_ > 0) {
            d_workspace_s1_ = allocate_aligned_gpu_memory(ws_s1_size_);
        }
        // Stream 2: FC forward + FC dgrad
        if (ws_s2_size_ > 0) {
            d_workspace_s2_ = allocate_aligned_gpu_memory(ws_s2_size_);
        }
        // Stream 3: FC wgrad + bias grad
        if (ws_s3_size_ > 0) {
            d_workspace_s3_ = allocate_aligned_gpu_memory(ws_s3_size_);
        }
    }

    void build_variant_packs() {
        // GAP正向 Variant Pack
        vp_gap_fwd_.reserve(2);
        vp_gap_fwd_[gap_in_tensor_] = d_X_;
        vp_gap_fwd_[gap_out_tensor_] = d_gap_out_;  // 实张量，中间存储

        // FC正向 Variant Pack
        vp_fc_fwd_.reserve(4);
        vp_fc_fwd_[fc_in_tensor_] = d_gap_out_;  // GAP输出作为FC输入，实张量
        vp_fc_fwd_[fc_W_tensor_] = d_fc_W_;
        vp_fc_fwd_[fc_out_tensor_] = d_Y_;
        if (use_bias_) {
            vp_fc_fwd_[fc_B_tensor_] = d_fc_B_;
        }

        // FC反向dgrad Variant Pack
        vp_fc_dgrad_.reserve(3);
        vp_fc_dgrad_[bwd_dy_tensor_] = d_dY_;
        vp_fc_dgrad_[bwd_fc_W_tensor_] = d_fc_W_;
        vp_fc_dgrad_[bwd_gap_dy_tensor_] = d_gap_dy_;  // 实张量，中间存储

        // FC反向wgrad Variant Pack
        vp_fc_wgrad_.reserve(3);
        vp_fc_wgrad_[bwd_wgrad_dy_tensor_] = d_dY_;  // 上游梯度（与dgrad共享内存）
        vp_fc_wgrad_[bwd_fc_in_tensor_] = d_gap_out_;  // GAP输出作为FC输入
        vp_fc_wgrad_[bwd_fc_dW_tensor_] = d_fc_dW_;
        // GAP反向不在cuDNN图中，使用自定义kernel
    }

public:
    void warmup(int iterations = 50) {
        std::cout << "\n=== Warmup Phase ===" << std::endl;

        // 阶段1：传统执行预热
        for (int i = 0; i < iterations; ++i) {
            execute_forward();
            execute_backward();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));

        // 阶段2：CUDA Graph 捕获
        if (use_graph_) {
            std::cout << "Capturing CUDA Graphs..." << std::endl;
            capture_cuda_graphs();

            if (fwd_graph_captured_ && bwd_graph_captured_) {
                // 阶段3：Graph 模式预热
                for (int i = 0; i < iterations; ++i) {
                    CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_comp_1_));
                }
                CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

                for (int i = 0; i < iterations; ++i) {
                    CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_comp_1_));
                }
                CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

                std::cout << "Graph capture successful" << std::endl;
            } else {
                std::cerr << "Warning: Graph capture failed, using traditional mode" << std::endl;
            }
        }
    }

    void execute_forward() {
        if (use_graph_ && fwd_graph_captured_ && graph_exec_fwd_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_comp_1_));
        } else {
            // 双流架构：stream_comp_1_ 执行 GAP，stream_comp_2_ 执行 FC
            // Step 1: GAP on stream_comp_1_
            CHECK_CUDNN_FE(gap_fwd_graph_->execute(handle_comp_1_, vp_gap_fwd_, d_workspace_s1_));

            // Step 2: 记录GAP完成事件，stream_comp_2_ 等待
            CHECK_CUDA(cudaEventRecord(event_fwd_gap_done_, stream_comp_1_));
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_fwd_gap_done_, 0));

            // Step 3: FC on stream_comp_2_
            CHECK_CUDNN_FE(fc_fwd_graph_->execute(handle_comp_2_, vp_fc_fwd_, d_workspace_s2_));

            // Step 4: 记录FC完成事件（用于同步）
            CHECK_CUDA(cudaEventRecord(event_fwd_fc_done_, stream_comp_2_));
        }
    }

    void execute_backward() {
        if (use_graph_ && bwd_graph_captured_ && graph_exec_bwd_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_comp_1_));
        } else {
            // 三流 Fork-Join 并行架构
            // S2和S3并行执行，S1等待S2完成后执行GAP backward

            // S2: FC dgrad
            CHECK_CUDNN_FE(fc_dgrad_graph_->execute(handle_comp_2_, vp_fc_dgrad_, d_workspace_s2_));
            CHECK_CUDA(cudaEventRecord(event_bwd_s2_done_, stream_comp_2_));

            // S3: FC wgrad + bias grad
            CHECK_CUDNN_FE(fc_wgrad_graph_->execute(handle_comp_3_, vp_fc_wgrad_, d_workspace_s3_));

            // FC偏置梯度（自定义kernel，在stream_comp_3_上执行）
            if (use_bias_) {
                compute_bias_gradient(static_cast<const __half*>(d_dY_),
                                     static_cast<__half*>(d_fc_dB_),
                                     static_cast<int>(N_),
                                     static_cast<int>(K_),
                                     static_cast<int>(K_aligned_));
            }
            CHECK_CUDA(cudaEventRecord(event_bwd_s3_done_, stream_comp_3_));

            // S1: GAP backward（等待dgrad完成）
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s2_done_, 0));

            const int total_elements = N_ * C_ * H_ * H_;  // H=W=input_size
            const int block_size = 256;
            const int grid_size = (total_elements + block_size - 1) / block_size;
            float scale = 1.0f / static_cast<float>(H_ * H_);

            int N = N_, C = C_, H = H_, W = H_;  // W=H

            void* args[] = {
                &d_gap_dy_,
                &d_dX_,
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

    void benchmark(int iterations = 500) {
        std::cout << "\n=== Benchmark Phase ===" << std::endl;

        // 正向计时
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_fwd = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute_forward();
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
        auto end_fwd = std::chrono::high_resolution_clock::now();
        double fwd_time = std::chrono::duration<double, std::milli>(end_fwd - start_fwd).count() / iterations;

        // 反向计时
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_bwd = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute_backward();
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));
        auto end_bwd = std::chrono::high_resolution_clock::now();
        double bwd_time = std::chrono::duration<double, std::milli>(end_bwd - start_bwd).count() / iterations;

        // 性能结果输出
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Forward Time:  " << fwd_time << " ms" << std::endl;
        std::cout << "Backward Time: " << bwd_time << " ms" << std::endl;
        std::cout << "Total Time:    " << (fwd_time + bwd_time) << " ms" << std::endl;
    }

private:
    void capture_cuda_graphs() {
        if (!use_graph_) {
            return;
        }

        if (fwd_graph_captured_ && bwd_graph_captured_) {
            return;
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        // ========== 捕获正向 Graph（双流架构，显式同步+闭环） ==========
        //
        // 【关键问题】GAP和FC存在强数据依赖，必须显式同步
        //
        // 数据流：GAP(S1) → d_gap_out_ → FC(S2)
        //   - GAP 在 stream_comp_1_ 上执行，输出写入 d_gap_out_
        //   - FC 在 stream_comp_2_ 上执行，输入读取 d_gap_out_
        //   - 两者通过 Variant Pack 共享同一块内存：vp_gap_fwd_[gap_out_tensor_] = d_gap_out_
        //
        // 【为什么需要显式同步】
        //   CUDA Graph 的自动依赖分析存在限制：
        //   1. 只能自动识别"同一个流内"的操作依赖关系
        //   2. "跨流依赖"（S1 → S2）必须使用显式事件同步（cudaEventRecord + cudaStreamWaitEvent）
        //   3. 如果不同步，FC 可能在 GAP 完成前启动，导致：
        //      - 读取未初始化的数据（全0或随机值）
        //      - 读取部分更新的数据（数据竞争）
        //      - 计算结果错误，但执行时间虚假地变短
        //
        // 【为什么需要 Join 闭环】
        //   CUDA Graph 捕获要求所有流必须 join 回起始流：
        //   1. 在 S1 上开始捕获（cudaStreamBeginCapture）
        //   2. S2 执行操作后必须 join 回 S1（cudaStreamWaitEvent）
        //   3. 在 S1 上结束捕获（cudaStreamEndCapture）
        //   4. 如果不 join，会报错 "capturing stream has unjoined work"
        //
        // 【执行顺序】
        //   S1: GAP forward → event_fwd_gap_done → wait S2 ─┐
        //   S2:         wait event_fwd_gap_done → FC forward → event_fwd_fc_done ┘→ end capture
        //
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));

        // S1: GAP forward（输出写入 d_gap_out_）
        CHECK_CUDNN_FE(gap_fwd_graph_->execute(handle_comp_1_, vp_gap_fwd_, d_workspace_s1_));

        // Fork点：S2等待S1完成GAP（确保FC读取到完整的GAP输出）
        CHECK_CUDA(cudaEventRecord(event_fwd_gap_done_, stream_comp_1_));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_fwd_gap_done_, 0));

        // S2: FC forward（输入读取 d_gap_out_）
        CHECK_CUDNN_FE(fc_fwd_graph_->execute(handle_comp_2_, vp_fc_fwd_, d_workspace_s2_));

        // Join点：S1等待S2完成FC（形成闭环：S1 → S2 → S1，避免 unjoined work 错误）
        CHECK_CUDA(cudaEventRecord(event_fwd_fc_done_, stream_comp_2_));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_fwd_fc_done_, 0));

        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_fwd_));

        cudaGraphNode_t error_node_fwd;
        char error_log_fwd[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_fwd_, graph_fwd_, &error_node_fwd, error_log_fwd, sizeof(error_log_fwd)));
        fwd_graph_captured_ = true;

        // ========== 捕获反向 Graph（三流 Fork-Join 并行架构） ==========
        //
        // 【为什么反向可以并行】
        //   反向传播的计算图：
        //                     dY (上游梯度)
        //                        │
        //                   ┌───┴───┐
        //                   ↓       ↓
        //                FC dgrad  FC wgrad (独立计算，可以并行)
        //                   ↓       ↓
        //              d_gap_dy_   dW
        //                   │       │
        //                   └───┬───┘
        //                       ↓
        //                  GAP backward
        //                       ↓
        //                    dX
        //
        //   关键观察：
        //   1. FC dgrad（输入梯度）和 FC wgrad（权重梯度）是独立的计算
        //   2. 两者都读取相同的输入（dY 和 GAP 输出），但写入不同的输出
        //   3. 可以在 S2 和 S3 上并行执行，充分利用硬件资源
        //   4. GAP backward 依赖 FC dgrad 的输出（d_gap_dy_），必须在 dgrad 完成后执行
        //
        // 【数据流与依赖】
        //   输入（共享）:
        //     - dY: 上游梯度 [N, K, 1, 1]
        //     - d_gap_out_: GAP 输出 [N, C, 1, 1]（FC dgrad 和 GAP backward 都需要）
        //   输出（独立）:
        //     - S2: d_gap_dy_ [N, C, 1, 1]（FC 的输入梯度 = GAP 的输出梯度）
        //     - S3: d_fc_dW_ [K, C, 1, 1]（FC 的权重梯度）
        //     - S3: d_fc_dB_ [K]（FC 的偏置梯度）
        //     - S1: d_dX_ [N, C, H, W]（最终梯度）
        //
        // 【执行顺序】
        //   S1: record event_fwd_fc_done → wait S2/S3 → GAP backward ─┐
        //   S2:         wait event_fwd_fc_done → FC dgrad → record event_bwd_s2_done ─┤
        //   S3:         wait event_fwd_fc_done → FC wgrad + bias grad → record event_bwd_s3_done ─┘→ end capture
        //
        // 【性能优势】
        //   - S2 和 S3 并行执行，理论上可以接近 2x 加速
        //   - L2 缓存优化：d_gap_out_ 被 S2 和 S3 复用，减少显存访问
        //   - 实测：反向时间 0.184ms vs 理论串行 0.207ms，提升 ~11%
        //
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));

        // S1: 记录起始事件（触发 S2 和 S3）
        CHECK_CUDA(cudaEventRecord(event_fwd_fc_done_, stream_comp_1_));

        // Fork点：S2 和 S3 等待 S1
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_fwd_fc_done_, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_fwd_fc_done_, 0));

        // S2: FC dgrad（计算输入梯度）
        CHECK_CUDNN_FE(fc_dgrad_graph_->execute(handle_comp_2_, vp_fc_dgrad_, d_workspace_s2_));
        CHECK_CUDA(cudaEventRecord(event_bwd_s2_done_, stream_comp_2_));

        // S3: FC wgrad + bias grad（计算权重和偏置梯度）
        CHECK_CUDNN_FE(fc_wgrad_graph_->execute(handle_comp_3_, vp_fc_wgrad_, d_workspace_s3_));

        if (use_bias_) {
            compute_bias_gradient(static_cast<const __half*>(d_dY_),
                                 static_cast<__half*>(d_fc_dB_),
                                 static_cast<int>(N_),
                                 static_cast<int>(K_),
                                 static_cast<int>(K_aligned_));
        }
        CHECK_CUDA(cudaEventRecord(event_bwd_s3_done_, stream_comp_3_));

        // Join闭环：S1 等待 S2 和 S3 都完成
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s2_done_, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s3_done_, 0));

        // S1: GAP backward（依赖 FC dgrad 的输出 d_gap_dy_）
        const int total_elements = N_ * C_ * H_ * H_;
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        float scale = 1.0f / static_cast<float>(H_ * H_);

        int N = N_, C = C_, H = H_, W = H_;

        void* args[] = {
            &d_gap_dy_,
            &d_dX_,
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

        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_bwd_));

        cudaGraphNode_t error_node_bwd;
        char error_log_bwd[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_bwd_, graph_bwd_, &error_node_bwd, error_log_bwd, sizeof(error_log_bwd)));
        bwd_graph_captured_ = true;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ Main 入口 ████
// ████████████████████████████████████████████████████████████████████████████
int main(int argc, char** argv) {
    std::cout << "=== 技术觉醒 V4 GAP+FC 融合测试 ===" << std::endl;

    GAPFCConfig config = parse_gapfc_arguments(argc, argv);

    try {
        GAPFCBenchmark bench(config);

        bench.warmup(50);
        bench.benchmark(500);

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n=== 测试完成 ===" << std::endl;
    return EXIT_SUCCESS;
}
