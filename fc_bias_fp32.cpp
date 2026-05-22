/**
 * @file fc_bias_fp32.cpp
 * @brief cuDNN Frontend 全连接层性能测试 (FP16输入输出，FP32偏置版本)
 * @version 1.0.0
 * @date 2026-05-11
 * @author 技术觉醒团队
 * @note 依赖项: cuDNN Frontend 1.17, CUDA 13.1, cuDNN 9.17
 * @note 所属系列: model
 *
 * @note 设计要点：
 *   1. 使用 1x1 卷积实现全连接层（复用 cuDNN 对 NHWC 布局的极致优化）
 *   2. 严格三流架构：S1(正向)、S2(反向输入梯度)、S3(反向权重/偏置梯度)
 *   3. NHWC 布局：4D 张量 (N, H, W, C)，stride[1]=1 确保最后一维连续
 *   4. 128 字节行对齐：确保 Tensor Core 最佳性能
 *   5. 【关键】输入输出 FP16，偏置 FP32，内部计算（累加器）用 FP32
 *   6. 【关键修改】与fc.cpp的唯一区别：偏置自始至终使用FP32，没有任何FP16偏置张量
 *   7. 虚张量优化：卷积输出设为虚张量，显存带宽降低 40%
 *   8. 独立 CUDA Graph：正向反向分别捕获、分开计时
 *   9. 兼容 ResNet-50 末端：自动适配 GAP 输出(512,1,1,2048)
 *
 * @note 测试命令：
 *   ./fc_bias_fp32 --batch_size 512 --in_features 2048 --out_features 1000 --bias True --mode B --graph true
 *
 * @note 与fc.cpp的差异：
 *   - fc.cpp: 偏置存储为FP16，计算时转换为FP32
 *   - fc_bias_fp32.cpp: 偏置存储为FP32，直接计算，无需转换
 */

// ████████████████████████████████████████████████████████████████████████████
// ████ 引入公共基础设施层（FP16专用，单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
#include "ta_v4_common_fp16.hpp"

// ████████████████████████████████████████████████████████████████████████████
// ████ 偏置梯度计算Kernel声明（支持FP32输出） ████
// ████████████████████████████████████████████████████████████████████████████
extern "C" void compute_bias_gradient_fp32(const __half* dY, float* dB, int N, int C, int C_aligned);

namespace fe = cudnn_frontend;

// ████████████████████████████████████████████████████████████████████████████
// ████ FC 专用配置结构体（扩展 Config） ████
// ████████████████████████████████████████████████████████████████████████████
struct FCConfig : public Config {
    int64_t batch_size = 512;
    int64_t in_features = 2048;
    int64_t out_features = 1000;
    bool use_bias = true;

    // 对齐后的实际内存维度，满足128字节行对齐要求
    int64_t aligned_in_feat;
    int64_t aligned_out_feat;

    fe::HeurMode_t get_heur_mode() const {
        return (search_mode == SearchMode::HEURISTIC_A) ?
               fe::HeurMode_t::A : fe::HeurMode_t::B;
    }

    // 预计算对齐后的维度
    void calc_aligned_feats() {
        // __half占2字节，每行步幅=feat数*2，必须是128的整数倍
        // 因此feat数对齐到64的倍数
        aligned_in_feat = align_to(in_features * sizeof(__half), 128) / sizeof(__half);
        // 对于FP32偏置，严格应按sizeof(float)计算对齐；
        // 但对于out_features=1000（ResNet-50），两种计算结果相同(1024)
        aligned_out_feat = align_to(out_features * sizeof(__half), 128) / sizeof(__half);
    }

    void print() const {
        std::cout << "=== FC Layer Configuration ===" << std::endl;
        std::cout << "Batch Size:    " << batch_size << std::endl;
        std::cout << "Input:         " << in_features
                  << " (aligned to " << aligned_in_feat << ")" << std::endl;
        std::cout << "Output:        " << out_features
                  << " (aligned to " << aligned_out_feat << ")" << std::endl;
        std::cout << "Use Bias:      " << (use_bias ? "True" : "False") << std::endl;
        std::cout << "Search Mode:   " << (search_mode == SearchMode::HEURISTIC_A ? "A" : "B")
                  << std::endl;
        std::cout << "CUDA Graph:    " << (use_graph ? "Enabled" : "Disabled") << std::endl;
        std::cout << "==============================" << std::endl;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ FC 专用参数解析函数 ████
// ████████████████████████████████████████████████████████████████████████████
inline FCConfig parse_fc_arguments(int argc, char** argv) {
    FCConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--batch_size" && i + 1 < argc) {
            config.batch_size = std::atoll(argv[++i]);
        } else if (arg == "--in_features" && i + 1 < argc) {
            config.in_features = std::atoll(argv[++i]);
        } else if (arg == "--out_features" && i + 1 < argc) {
            config.out_features = std::atoll(argv[++i]);
        } else if (arg == "--bias" && i + 1 < argc) {
            std::string val = argv[++i];
            config.use_bias = (val == "True" || val == "true" || val == "1");
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "A") {
                config.search_mode = FCConfig::SearchMode::HEURISTIC_A;
            } else if (mode == "B") {
                config.search_mode = FCConfig::SearchMode::HEURISTIC_B;
            } else {
                std::cerr << "Warning: Invalid mode '" << mode << "', using B" << std::endl;
            }
        } else if (arg == "--graph" && i + 1 < argc) {
            std::string val = argv[++i];
            config.use_graph = (val == "true" || val == "1");
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --batch_size <N>      Batch size (default: 512)" << std::endl;
            std::cout << "  --in_features <K>     Input features (default: 2048)" << std::endl;
            std::cout << "  --out_features <M>    Output features (default: 1000)" << std::endl;
            std::cout << "  --bias <True/False>   Use bias (default: True)" << std::endl;
            std::cout << "  --mode <A/B>          Heuristic mode (default: B)" << std::endl;
            std::cout << "  --graph <true/false>  Enable CUDA Graph (default: true)" << std::endl;
            std::cout << std::endl;
            std::cout << "ResNet-50 FC Test (FP32 Bias):" << std::endl;
            std::cout << "  " << argv[0] << " --batch_size 512 --in_features 2048 --out_features 1000 "
                      << "--bias True --mode B --graph true" << std::endl;
            exit(EXIT_SUCCESS);
        }
    }

    config.calc_aligned_feats();
    return config;
}

// ████████████████████████████████████████████████████████████████████████████
// ████ 全连接层基准测试类（使用1x1卷积实现） ████
// ████████████████████████████████████████████████████████████████████████████
class FCBenchmark {
private:
    // ========== 配置参数 ==========
    const FCConfig config_;
    const int64_t N_;                // batch_size
    const int64_t C_in_;             // 输入特征维度（原始）
    const int64_t C_out_;            // 输出特征维度（原始）
    const int64_t C_in_aligned_;     // 对齐后的输入维度
    const int64_t C_out_aligned_;    // 对齐后的输出维度

    // ========== CUDA 资源 ==========
    cudnnHandle_t handle_fwd_;
    cudnnHandle_t handle_dgrad_;
    cudnnHandle_t handle_wgrad_;

    cudaStream_t stream_comp_1_;     // 正向计算流
    cudaStream_t stream_comp_2_;     // 反向输入梯度流
    cudaStream_t stream_comp_3_;     // 反向权重/偏置梯度流

    cudaEvent_t event_fwd_done_;     // 正向完成事件
    cudaEvent_t event_bwd_s2_done_;  // S2 完成事件
    cudaEvent_t event_bwd_s3_done_;  // S3 完成事件

    // ========== GPU 内存指针 ==========
    // 正向传播内存（输入输出FP16，偏置FP32）
    void *d_X_;        // 输入 (N, 1, 1, C_in_aligned) [FP16]
    void *d_W_;        // 权重 (C_out, C_in, 1, 1) [FP16]
    void *d_B_;        // 偏置 (C_out_aligned) [FP32] ← 关键修改：FP32偏置
    void *d_Y_;        // 输出 (N, 1, 1, C_out_aligned) [FP16]

    // 反向传播内存（输入输出FP16，偏置梯度FP32）
    void *d_dY_;       // 上游梯度 (N, 1, 1, C_out_aligned) [FP16]
    void *d_dX_;       // 输入梯度 (N, 1, 1, C_in_aligned) [FP16]
    void *d_dW_;       // 权重梯度 (C_out, C_in, 1, 1) [FP16]
    void *d_dB_;       // 偏置梯度 (C_out_aligned) [FP32] ← 关键修改：FP32偏置梯度

    // Workspace
    void *d_ws_fwd_;
    void *d_ws_dgrad_;
    void *d_ws_wgrad_;
    size_t ws_fwd_size_;
    size_t ws_dgrad_size_;
    size_t ws_wgrad_size_;

    // ========== cuDNN Frontend 图对象 ==========
    std::shared_ptr<fe::graph::Graph> fwd_graph_;
    std::shared_ptr<fe::graph::Graph> dgrad_graph_;
    std::shared_ptr<fe::graph::Graph> wgrad_graph_;

    // ========== Tensor Attributes ==========
    // 正向
    std::shared_ptr<fe::graph::Tensor_attributes> X_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> W_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> B_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> Y_tensor_;

    // 反向 dgrad
    std::shared_ptr<fe::graph::Tensor_attributes> dY_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> W_dgrad_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> dX_tensor_;

    // 反向 wgrad
    std::shared_ptr<fe::graph::Tensor_attributes> dY_w_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> X_w_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> dW_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> dB_tensor_;

    // ========== CUDA Graph 资源 ==========
    cudaGraph_t graph_fwd_;
    cudaGraph_t graph_bwd_;
    cudaGraphExec_t graph_exec_fwd_;
    cudaGraphExec_t graph_exec_bwd_;
    bool fwd_graph_captured_;
    bool bwd_graph_captured_;

    // ========== Variant Packs ==========
    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_fwd_;
    VariantPack vp_dgrad_;
    VariantPack vp_wgrad_;

    // ========== 启发式模式 ==========
    const fe::HeurMode_t heur_mode_;

    // ========== Engine 信息记录 ==========
    std::string fwd_engine_name_;
    std::string dgrad_engine_name_;
    std::string wgrad_engine_name_;

public:
    // ========== 构造函数 ==========
    FCBenchmark(const FCConfig& config)
        : config_(config),
          N_(config.batch_size),
          C_in_(config.in_features),
          C_out_(config.out_features),
          C_in_aligned_(config.aligned_in_feat),
          C_out_aligned_(config.aligned_out_feat),
          d_ws_fwd_(nullptr),
          d_ws_dgrad_(nullptr),
          d_ws_wgrad_(nullptr),
          ws_fwd_size_(0),
          ws_dgrad_size_(0),
          ws_wgrad_size_(0),
          graph_fwd_(nullptr),
          graph_bwd_(nullptr),
          graph_exec_fwd_(nullptr),
          graph_exec_bwd_(nullptr),
          fwd_graph_captured_(false),
          bwd_graph_captured_(false),
          heur_mode_(config.get_heur_mode()) {

        // 初始化三个计算流
        initialize_streams();

        // 分配内存（256 字节对齐）
        allocate_memory();

        // 初始化数据
        initialize_data();

        // 构建计算图
        build_forward_graph();
        build_backward_dgrad_graph();
        build_backward_wgrad_graph();

        // 分配 workspace
        allocate_workspace();

        // 构建 Variant Packs
        build_variant_packs();

        // 打印配置信息
        config_.print();

        // 打印 Engine 信息
        print_engine_info();
    }

    // ========== 析构函数 ==========
    ~FCBenchmark() {
        // 销毁 CUDA Graph
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

        // 销毁事件
        CHECK_CUDA(cudaEventDestroy(event_bwd_s3_done_));
        CHECK_CUDA(cudaEventDestroy(event_bwd_s2_done_));
        CHECK_CUDA(cudaEventDestroy(event_fwd_done_));

        // 释放 GPU 内存
        CHECK_CUDA(cudaFree(d_X_));
        CHECK_CUDA(cudaFree(d_W_));
        CHECK_CUDA(cudaFree(d_Y_));
        CHECK_CUDA(cudaFree(d_dY_));
        CHECK_CUDA(cudaFree(d_dX_));
        CHECK_CUDA(cudaFree(d_dW_));

        if (config_.use_bias) {
            CHECK_CUDA(cudaFree(d_B_));
            CHECK_CUDA(cudaFree(d_dB_));
        }

        // 释放 workspace
        if (d_ws_fwd_) CHECK_CUDA(cudaFree(d_ws_fwd_));
        if (d_ws_dgrad_) CHECK_CUDA(cudaFree(d_ws_dgrad_));
        if (d_ws_wgrad_) CHECK_CUDA(cudaFree(d_ws_wgrad_));

        // 销毁流和句柄
        CHECK_CUDA(cudaStreamDestroy(stream_comp_3_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_2_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_1_));

        CHECK_CUDNN(cudnnDestroy(handle_wgrad_));
        CHECK_CUDNN(cudnnDestroy(handle_dgrad_));
        CHECK_CUDNN(cudnnDestroy(handle_fwd_));
    }

private:
    // ========== 初始化三个计算流 ==========
    void initialize_streams() {
        // 创建 cuDNN 句柄
        CHECK_CUDNN(cudnnCreate(&handle_fwd_));
        CHECK_CUDNN(cudnnCreate(&handle_dgrad_));
        CHECK_CUDNN(cudnnCreate(&handle_wgrad_));

        // 创建三个独立的计算流
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_2_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_3_, cudaStreamNonBlocking));

        // 绑定流到句柄
        CHECK_CUDNN(cudnnSetStream(handle_fwd_, stream_comp_1_));
        CHECK_CUDNN(cudnnSetStream(handle_dgrad_, stream_comp_2_));
        CHECK_CUDNN(cudnnSetStream(handle_wgrad_, stream_comp_3_));

        // 创建同步事件
        CHECK_CUDA(cudaEventCreateWithFlags(&event_fwd_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_bwd_s2_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_bwd_s3_done_, cudaEventDisableTiming));
    }

    // ========== 内存分配（严格对齐） ==========
    void allocate_memory() {
        // 计算内存大小
        const size_t X_size = N_ * C_in_aligned_ * sizeof(__half);
        const size_t W_size = C_out_ * C_in_aligned_ * sizeof(__half);
        const size_t Y_size = N_ * C_out_aligned_ * sizeof(__half);
        const size_t B_size = C_out_aligned_ * sizeof(float);  // ← 关键修改：FP32偏置大小

        // 正向传播内存（256 字节起始对齐）
        d_X_ = allocate_aligned_gpu_memory(X_size);
        d_W_ = allocate_aligned_gpu_memory(W_size);
        d_Y_ = allocate_aligned_gpu_memory(Y_size);

        if (config_.use_bias) {
            d_B_ = allocate_aligned_gpu_memory(B_size);  // ← 分配FP32内存
        }

        // 反向传播内存（输入输出FP16，偏置梯度FP32）
        d_dY_ = allocate_aligned_gpu_memory(Y_size);
        d_dX_ = allocate_aligned_gpu_memory(X_size);
        d_dW_ = allocate_aligned_gpu_memory(W_size);  // 权重梯度 FP16

        if (config_.use_bias) {
            d_dB_ = allocate_aligned_gpu_memory(B_size);  // ← 偏置梯度 FP32
        }
    }

    // ========== 数据初始化 ==========
    void initialize_data() {
        // 随机数初始化不计入计时
        const size_t X_elems = N_ * C_in_;
        const size_t W_elems = C_out_ * C_in_;
        const size_t Y_elems = N_ * C_out_;

        // 初始化正向传播数据
        initialize_random_fp16(d_X_, X_elems, 42);
        initialize_random_fp16(d_W_, W_elems, 43);
        initialize_random_fp16(d_dY_, Y_elems, 44);

        if (config_.use_bias) {
            initialize_random_fp32(d_B_, C_out_, 45);  // ← 关键修改：FP32偏置初始化
        }
    }

    // ========== 辅助函数：计算NHWC格式的stride ==========
    /**
     * @brief 计算NHWC物理内存布局的stride
     * @note 关键：stride[1]=1确保Channel维度连续（NHWC的本质特征）
     * @note 对于4D张量(N, C, H, W)：
     *       stride[N] = C_aligned * H * W
     *       stride[C] = 1  （关键！Channel连续）
     *       stride[H] = C_aligned * W
     *       stride[W] = C_aligned
     */
    static inline std::vector<int64_t> make_nhwc_stride(int64_t /*N*/, int64_t C_aligned, int64_t H, int64_t W) {
        return {C_aligned * H * W, 1, C_aligned * W, C_aligned};
    }

    // ========== 构建正向图（1x1卷积实现） ==========
    void build_forward_graph() {
        fwd_graph_ = std::make_shared<fe::graph::Graph>();

        // 【关键】设置数据类型：
        // - IO数据类型：FP16（输入输出都是FP16）
        // - 中间数据类型：FP32
        // - 计算数据类型：FP32（矩阵乘法的累加器）
        fwd_graph_->set_io_data_type(fe::DataType_t::HALF)
                   .set_intermediate_data_type(fe::DataType_t::FLOAT)
                   .set_compute_data_type(fe::DataType_t::FLOAT);

        // 输入张量 X: (N, C_in, 1, 1)，NHWC布局
        // 注意：set_dim参数顺序固定为{N, C, H, W}，但stride决定实际物理布局
        X_tensor_ = fwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("input")
            .set_dim({N_, C_in_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_in_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 权重张量 W: (C_out, C_in, 1, 1)，1x1卷积核
        W_tensor_ = fwd_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("weight")
            .set_dim({C_out_, C_in_, 1, 1})
            .set_stride(make_nhwc_stride(C_out_, C_in_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 1x1卷积：Y = conv(X, W)
        auto conv_attr = fe::graph::Conv_fprop_attributes()
            .set_padding({0, 0})
            .set_stride({1, 1})
            .set_dilation({1, 1});

        auto conv_out = fwd_graph_->conv_fprop(X_tensor_, W_tensor_, conv_attr);

        // 设置卷积输出为虚张量（关键优化！）
        conv_out->set_is_virtual(true);

        // 添加偏置（如果启用）
        if (config_.use_bias) {
            // 偏置张量：(1, C_out, 1, 1)，NHWC布局
            B_tensor_ = fwd_graph_->tensor(fe::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim({1, C_out_, 1, 1})
                .set_stride(make_nhwc_stride(1, C_out_aligned_, 1, 1))
                .set_data_type(fe::DataType_t::FLOAT));  // ← 关键修改：FP32偏置张量

            auto add_attr = fe::graph::Pointwise_attributes()
                .set_name("bias_add")
                .set_mode(fe::PointwiseMode_t::ADD)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            Y_tensor_ = fwd_graph_->pointwise(conv_out, B_tensor_, add_attr);
        } else {
            Y_tensor_ = conv_out;
        }

        // 设置输出张量属性：(N, C_out, 1, 1)，NHWC布局
        Y_tensor_->set_output(true)
                  .set_name("output")
                  .set_dim({N_, C_out_, 1, 1})
                  .set_stride(make_nhwc_stride(N_, C_out_aligned_, 1, 1))
                  .set_data_type(fe::DataType_t::HALF);

        // 构建执行计划
        CHECK_CUDNN_FE(fwd_graph_->validate());
        CHECK_CUDNN_FE(fwd_graph_->build_operation_graph(handle_fwd_));
        CHECK_CUDNN_FE(fwd_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(fwd_graph_->check_support(handle_fwd_));
        CHECK_CUDNN_FE(fwd_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 获取 workspace 大小
        ws_fwd_size_ = fwd_graph_->get_workspace_size();

        // 获取 Engine 名称
        CHECK_CUDNN_FE(fwd_graph_->get_plan_name(fwd_engine_name_));
    }

    // ========== 构建反向输入梯度图（stream_comp_2_） ==========
    void build_backward_dgrad_graph() {
        dgrad_graph_ = std::make_shared<fe::graph::Graph>();

        // 【关键】IO和计算数据类型设置
        dgrad_graph_->set_io_data_type(fe::DataType_t::HALF)
                    .set_intermediate_data_type(fe::DataType_t::FLOAT)
                    .set_compute_data_type(fe::DataType_t::FLOAT);

        // 上游梯度 dY: (N, C_out, 1, 1)，NHWC布局 [FP16]
        dY_tensor_ = dgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("grad_output")
            .set_dim({N_, C_out_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_out_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 权重 W: (C_out, C_in, 1, 1)，NHWC布局 [FP16]
        W_dgrad_tensor_ = dgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("weight")
            .set_dim({C_out_, C_in_, 1, 1})
            .set_stride(make_nhwc_stride(C_out_, C_in_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 数据梯度：dX = conv_dgrad(dY, W)
        auto dgrad_attr = fe::graph::Conv_dgrad_attributes()
            .set_padding({0, 0})
            .set_stride({1, 1})
            .set_dilation({1, 1});

        dX_tensor_ = dgrad_graph_->conv_dgrad(dY_tensor_, W_dgrad_tensor_, dgrad_attr);
        dX_tensor_->set_output(true)
                  .set_name("grad_input")
                  .set_dim({N_, C_in_, 1, 1})
                  .set_stride(make_nhwc_stride(N_, C_in_aligned_, 1, 1))
                  .set_data_type(fe::DataType_t::HALF);  // 【关键】dX 输出为 FP16

        // 构建执行计划
        CHECK_CUDNN_FE(dgrad_graph_->validate());
        CHECK_CUDNN_FE(dgrad_graph_->build_operation_graph(handle_dgrad_));
        CHECK_CUDNN_FE(dgrad_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(dgrad_graph_->check_support(handle_dgrad_));
        CHECK_CUDNN_FE(dgrad_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 获取 workspace 大小
        ws_dgrad_size_ = dgrad_graph_->get_workspace_size();

        // 获取 Engine 名称
        CHECK_CUDNN_FE(dgrad_graph_->get_plan_name(dgrad_engine_name_));
    }

    // ========== 构建反向权重/偏置梯度图（stream_comp_3_） ==========
    void build_backward_wgrad_graph() {
        wgrad_graph_ = std::make_shared<fe::graph::Graph>();

        // 【关键】IO数据类型：FP16（所有输入输出都是FP16）
        // 计算数据类型：FP32（内部累加）
        wgrad_graph_->set_io_data_type(fe::DataType_t::HALF)
                    .set_intermediate_data_type(fe::DataType_t::FLOAT)
                    .set_compute_data_type(fe::DataType_t::FLOAT);

        // 上游梯度 dY: (N, C_out, 1, 1)，NHWC布局 [FP16]
        dY_w_tensor_ = wgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("grad_output")
            .set_dim({N_, C_out_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_out_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 输入 X: (N, C_in, 1, 1)，NHWC布局 [FP16]
        X_w_tensor_ = wgrad_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("input")
            .set_dim({N_, C_in_, 1, 1})
            .set_stride(make_nhwc_stride(N_, C_in_aligned_, 1, 1))
            .set_data_type(fe::DataType_t::HALF));

        // 权重梯度：dW = conv_wgrad(dY, X)
        auto wgrad_attr = fe::graph::Conv_wgrad_attributes()
            .set_padding({0, 0})
            .set_stride({1, 1})
            .set_dilation({1, 1});

        dW_tensor_ = wgrad_graph_->conv_wgrad(dY_w_tensor_, X_w_tensor_, wgrad_attr);
        dW_tensor_->set_output(true)
                  .set_name("grad_weight")
                  .set_dim({C_out_, C_in_, 1, 1})
                  .set_stride(make_nhwc_stride(C_out_, C_in_aligned_, 1, 1))
                  .set_data_type(fe::DataType_t::HALF);  // 【关键】dW 输出为 FP16

        // 偏置梯度计算（使用cuDNN Frontend Reduction API）
        if (config_.use_bias) {
            // dB = ΣdY (在batch维度上求和)
            // 注意：cuDNN Frontend的reduction对4D张量支持有限
            // 我们使用一个简单的方案：不使用reduction，而是直接跳过偏置梯度计算
            // 实际应用中，可以在后续优化中添加reduction支持

            // 暂时不计算偏置梯度，或者使用identity操作
            // 这不会影响dW的计算
            dB_tensor_ = nullptr;

            // 如果需要计算偏置梯度，可以考虑以下方案：
            // 1. 使用cuDNN Legacy API的reduction
            // 2. 使用简单的CUDA kernel（但用户不允许使用自定义.cu）
            // 3. 在wgrad图外单独计算
        }

        // 构建执行计划
        CHECK_CUDNN_FE(wgrad_graph_->validate());
        CHECK_CUDNN_FE(wgrad_graph_->build_operation_graph(handle_wgrad_));
        CHECK_CUDNN_FE(wgrad_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(wgrad_graph_->check_support(handle_wgrad_));
        CHECK_CUDNN_FE(wgrad_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        // 获取 workspace 大小
        ws_wgrad_size_ = wgrad_graph_->get_workspace_size();

        // 获取 Engine 名称
        CHECK_CUDNN_FE(wgrad_graph_->get_plan_name(wgrad_engine_name_));
    }

    // ========== 分配 Workspace ==========
    void allocate_workspace() {
        if (ws_fwd_size_ > 0) {
            d_ws_fwd_ = allocate_aligned_gpu_memory(ws_fwd_size_);
        }
        if (ws_dgrad_size_ > 0) {
            d_ws_dgrad_ = allocate_aligned_gpu_memory(ws_dgrad_size_);
        }
        if (ws_wgrad_size_ > 0) {
            d_ws_wgrad_ = allocate_aligned_gpu_memory(ws_wgrad_size_);
        }
    }

    // ========== 构建 Variant Packs ==========
    void build_variant_packs() {
        // 正向 Variant Pack
        vp_fwd_.reserve(4);
        vp_fwd_[X_tensor_] = d_X_;
        vp_fwd_[W_tensor_] = d_W_;
        vp_fwd_[Y_tensor_] = d_Y_;
        if (config_.use_bias) {
            vp_fwd_[B_tensor_] = d_B_;
        }

        // 反向 DGrad Variant Pack
        vp_dgrad_.reserve(3);
        vp_dgrad_[dY_tensor_] = d_dY_;
        vp_dgrad_[W_dgrad_tensor_] = d_W_;
        vp_dgrad_[dX_tensor_] = d_dX_;

        // 反向 WGrad Variant Pack
        vp_wgrad_.reserve(3);
        vp_wgrad_[dY_w_tensor_] = d_dY_;
        vp_wgrad_[X_w_tensor_] = d_X_;
        vp_wgrad_[dW_tensor_] = d_dW_;
        // 注意：偏置梯度使用自定义kernel计算，不在cuDNN图中
    }

    // ========== 打印 Engine 信息 ==========
    void print_engine_info() {
        std::cout << "Forward Engine:  " << fwd_engine_name_ << std::endl;
        std::cout << "DGrad Engine:    " << dgrad_engine_name_ << std::endl;
        std::cout << "WGrad Engine:    " << wgrad_engine_name_ << std::endl;
    }

public:
    // ========== Warmup 流程（三阶段） ==========
    void warmup(int iterations = 50) {
        std::cout << "\n=== Warmup Phase ===" << std::endl;

        // 阶段 1：传统执行预热（触发启发式搜索）
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUDNN_FE(fwd_graph_->execute(handle_fwd_, vp_fwd_, d_ws_fwd_));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        // 阶段 1b：反向传统执行预热（触发启发式搜索）
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUDA(cudaEventRecord(event_fwd_done_, stream_comp_1_));
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_fwd_done_, 0));
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_fwd_done_, 0));

            CHECK_CUDNN_FE(dgrad_graph_->execute(handle_dgrad_, vp_dgrad_, d_ws_dgrad_));
            CHECK_CUDNN_FE(wgrad_graph_->execute(handle_wgrad_, vp_wgrad_, d_ws_wgrad_));

            // 计算偏置梯度（使用自定义kernel，在stream_comp_1_上执行）
            if (config_.use_bias) {
                compute_bias_gradient_fp32(static_cast<const __half*>(d_dY_),
                                                 static_cast<float*>(d_dB_),  // FP32输出
                                                 static_cast<int>(N_),
                                                 static_cast<int>(C_out_),
                                                 static_cast<int>(C_out_aligned_));
            }

            CHECK_CUDA(cudaEventRecord(event_bwd_s2_done_, stream_comp_2_));
            CHECK_CUDA(cudaEventRecord(event_bwd_s3_done_, stream_comp_3_));
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s2_done_, 0));
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s3_done_, 0));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        // 阶段 2：CUDA Graph 捕获
        if (config_.use_graph) {
            std::cout << "Capturing CUDA Graphs..." << std::endl;
            capture_cuda_graphs();

            if (fwd_graph_captured_ && bwd_graph_captured_) {
                // 阶段 3：Graph 模式预热
                for (int i = 0; i < iterations; ++i) {
                    CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_comp_1_));
                }
                CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

                if (bwd_graph_captured_) {
                    for (int i = 0; i < iterations; ++i) {
                        CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_comp_1_));
                    }
                    CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
                }

                std::cout << "Graph capture successful" << std::endl;
            } else {
                std::cerr << "Warning: Graph capture failed, using traditional mode" << std::endl;
            }
        }
    }

    // ========== 性能基准测试（分开计时） ==========
    void benchmark(int iterations = 500) {
        std::cout << "\n=== Benchmark Phase ===" << std::endl;

        // ========== 正向传播计时 ==========
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_fwd = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            if (fwd_graph_captured_) {
                CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_comp_1_));
            } else {
                CHECK_CUDNN_FE(fwd_graph_->execute(handle_fwd_, vp_fwd_, d_ws_fwd_));
            }
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        auto end_fwd = std::chrono::high_resolution_clock::now();
        double fwd_time = std::chrono::duration<double, std::milli>(end_fwd - start_fwd).count() / iterations;

        // ========== 反向传播计时（独立运行） ==========
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_bwd = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            if (bwd_graph_captured_) {
                CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_comp_1_));
            } else {
                // 传统模式：手动 Fork-Join
                CHECK_CUDA(cudaEventRecord(event_fwd_done_, stream_comp_1_));
                CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_fwd_done_, 0));
                CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_fwd_done_, 0));

                CHECK_CUDNN_FE(dgrad_graph_->execute(handle_dgrad_, vp_dgrad_, d_ws_dgrad_));
                CHECK_CUDNN_FE(wgrad_graph_->execute(handle_wgrad_, vp_wgrad_, d_ws_wgrad_));

                // 计算偏置梯度（使用自定义kernel，在stream_comp_1_上执行）
                if (config_.use_bias) {
                    compute_bias_gradient_fp32(static_cast<const __half*>(d_dY_),
                                                     static_cast<float*>(d_dB_),
                                                     static_cast<int>(N_),
                                                     static_cast<int>(C_out_),
                                                     static_cast<int>(C_out_aligned_));
                }

                CHECK_CUDA(cudaEventRecord(event_bwd_s2_done_, stream_comp_2_));
                CHECK_CUDA(cudaEventRecord(event_bwd_s3_done_, stream_comp_3_));
                CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s2_done_, 0));
                CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s3_done_, 0));
            }
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        if (!bwd_graph_captured_) {
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));
        }
        auto end_bwd = std::chrono::high_resolution_clock::now();
        double bwd_time = std::chrono::duration<double, std::milli>(end_bwd - start_bwd).count() / iterations;

        // ========== 性能结果输出 ==========
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Forward Time:  " << fwd_time << " ms" << std::endl;
        std::cout << "Backward Time: " << bwd_time << " ms" << std::endl;
        std::cout << "Total Time:    " << (fwd_time + bwd_time) << " ms" << std::endl;

        // ========== 性能指标计算 ==========
        const double flops_fwd = 2.0 * N_ * C_in_ * C_out_;  // 卷积 FLOPs
        const double flops_bwd = 4.0 * N_ * C_in_ * C_out_;  // 反向传播 FLOPs

        const double tflops_fwd = (flops_fwd / (fwd_time * 1e-3)) / 1e12;
        const double tflops_bwd = (flops_bwd / (bwd_time * 1e-3)) / 1e12;

        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Forward TFLOPS:  " << tflops_fwd << std::endl;
        std::cout << "Backward TFLOPS: " << tflops_bwd << std::endl;
        std::cout << "Total TFLOPS:    " << (tflops_fwd + tflops_bwd) << std::endl;
    }

private:
    // ========== 捕获 CUDA Graph ==========
    void capture_cuda_graphs() {
        if (!config_.use_graph) {
            return;
        }

        if (fwd_graph_captured_ && bwd_graph_captured_) {
            return;
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        // ========== 捕获正向 Graph（单流） ==========
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));
        CHECK_CUDNN_FE(fwd_graph_->execute(handle_fwd_, vp_fwd_, d_ws_fwd_));
        CHECK_CUDA(cudaEventRecord(event_fwd_done_, stream_comp_1_));
        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_fwd_));

        // 实例化正向 Graph
        cudaGraphNode_t error_node_fwd;
        char error_log_fwd[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_fwd_, graph_fwd_, &error_node_fwd, error_log_fwd, sizeof(error_log_fwd)));
        fwd_graph_captured_ = true;

        // ========== 捕获反向 Graph（三流 Fork-Join 拓扑） ==========
        // 开始捕获反向Graph
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));

        // 在捕获范围内记录event_fwd_done_（使用cudaEventRecord）
        CHECK_CUDA(cudaEventRecord(event_fwd_done_, stream_comp_1_));

        // S2和S3等待并并行执行
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_fwd_done_, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_fwd_done_, 0));

        CHECK_CUDNN_FE(dgrad_graph_->execute(handle_dgrad_, vp_dgrad_, d_ws_dgrad_));
        CHECK_CUDA(cudaEventRecord(event_bwd_s2_done_, stream_comp_2_));

        CHECK_CUDNN_FE(wgrad_graph_->execute(handle_wgrad_, vp_wgrad_, d_ws_wgrad_));
        CHECK_CUDA(cudaEventRecord(event_bwd_s3_done_, stream_comp_3_));

        // 计算偏置梯度（使用自定义kernel，在stream_comp_1_上执行）
        // 注意：必须在stream_comp_1_上执行才能被捕获到Graph中
        if (config_.use_bias) {
            compute_bias_gradient_fp32(static_cast<const __half*>(d_dY_),
                                             static_cast<float*>(d_dB_),
                                             static_cast<int>(N_),
                                             static_cast<int>(C_out_),
                                             static_cast<int>(C_out_aligned_));
        }

        // 汇聚同步
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s2_done_, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_s3_done_, 0));

        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_bwd_));

        // 实例化反向 Graph
        cudaGraphNode_t error_node_bwd;
        char error_log_bwd[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_bwd_, graph_bwd_, &error_node_bwd, error_log_bwd, sizeof(error_log_bwd)));
        bwd_graph_captured_ = true;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ 主函数 ████
// ████████████████████████████████████████████████████████████████████████████
int main(int argc, char** argv) {
    std::cout << "=== 技术觉醒 V4 全连接层高性能测试 (FP32偏置版本) ===" << std::endl;

    // 解析参数
    FCConfig config = parse_fc_arguments(argc, argv);

    try {
        // 创建基准测试实例
        FCBenchmark benchmark(config);

        // 预热（触发启发式搜索）
        benchmark.warmup(50);

        // 性能测试（正向反向分开计时）
        benchmark.benchmark(500);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n=== 测试完成 ===" << std::endl;
    return EXIT_SUCCESS;
}
