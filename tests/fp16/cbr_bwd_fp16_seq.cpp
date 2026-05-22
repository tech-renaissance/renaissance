/**
 * @file cbr_bwd_fp16_seq.cpp
 * @brief cuDNN Frontend Conv+BN+ReLU反向融合性能测试V1.2 (FP16版本，串行流执行：DGrad等待WGrad)
 * @version 1.2.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: cuDNN Frontend 1.17, CUDA 13.1, cuDNN 9.17
 * @note 所属系列: cuda
 *
 * @note V1.2 改动（基于V1.1，强制DGrad在WGrad之后执行）：
 *   1. 【修改】反向拓扑：DGrad强制等待WGrad完成（S2等待S3）
 *   2. 【保留】三流架构：S1(BN BWD), S2(DGrad), S3(WGrad)
 *   3. 【保留】所有其他功能和优化策略
 *   4. 【研究用途】对比并行vs串行流的性能差异
 *
 * @note 串行反向拓扑结构：
 *       修改后的反向拓扑（DGrad等待WGrad）：
 *         S1 (Begin) ──→ [BN Backward] ──event_bwd────┬─→ (Wait S2+S3 join) ──→ S1 (End)
 *                                                 │
 *                    ┌────────────────────────────┴────────────────────────────┐
 *                    │                                                            │
 *                    ↓                                                            ↓
 *                   S2 ──→ [DGrad] ──wait S3──┐              S3 ──→ [WGrad] ──event_wgrad_done──┐
 *                                                          │                                                  │
 *                                                          └──────────────→ (DGrad waits WGrad) ←────────────┘
 *
 * @note 关键变化：
 *       - S2 (DGrad) 现在必须等待 S3 (WGrad) 完成
 *       - 不再是并发的 DGrad+WGrad，而是串行的 WGrad→DGrad
 *       - 预期性能：由于失去并行性，应该比原版本慢
 *       - 研究价值：量化流并行对整体性能的贡献
 */

// ████████████████████████████████████████████████████████████████████████████
// ████ 引入公共基础设施层（FP16专用，单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
#include "ta_v4_common_fp16.hpp"

namespace fe = cudnn_frontend;

// ████████████████████████████████████████████████████████████████████████████
// ████ 以下所有定义均已移至ta_v4_common_fp16.hpp（单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
// - 错误检查宏：CHECK_CUDA, CHECK_CUDNN, CHECK_CUDNN_FE
// - 内存工具：align_to, allocate_aligned_gpu_memory
// - 初始化函数：initialize_random_fp16, initialize_random_fp32, initialize_relu_bitmask
// - 配置系统：Config结构体, parse_arguments
// - Experience查询：ExperienceStatus, build_shape_key, match_and_build_plan
// - 平台条件编译：A100/RTX5090 Experience表
// ████████████████████████████████████████████████████████████████████████████

class ConvBNReLUBackwardBenchmark {
private:
    // ========== 流和事件资源 ==========
    cudnnHandle_t handle_comp_1_, handle_comp_2_, handle_comp_3_;
    cudaStream_t stream_comp_1_, stream_comp_2_, stream_comp_3_;
    cudaEvent_t event_comp_1_bwd_done_;
    cudaEvent_t event_bwd_join_s2_;
    cudaEvent_t event_bwd_join_s3_;

    const int64_t N, C, H_in, W_in, K;
    const int64_t R, S;
    const int64_t padding;
    const int64_t stride;
    const int64_t H_out, W_out;

    const Config::SearchMode search_mode_;
    const fe::HeurMode_t heur_mode_;
    bool use_graph_;  // 非const，允许在Graph捕获失败时fallback到传统模式

    // ========== GPU memory pointers (反向) ==========
    // 反向需要的前向输出（模拟已计算）
    void *d_input;               // Conv输入（WGrad需要）
    void *d_output;              // Conv输出，BN输入
    void *d_mean;                // BN mean
    void *d_inv_variance;        // BN inv_variance
    void *d_relu_bitmask;        // ReLU bitmask

    // 反向输入
    void *d_grad_output;         // 上游梯度

    // 反向输出
    void *d_grad_conv_out;       // BN反向输出，Conv反向输入
    void *d_grad_input;          // 输入梯度
    void *d_grad_weight;         // 权重梯度

    // BN参数（scale，反向需要）
    void *d_scale;

    // BN梯度输出
    void *d_dscale, *d_dbias;

    // Conv权重（反向需要）
    void *d_weight;

    // Workspace
    void *d_workspace_comp_1_, *d_workspace_comp_2_, *d_workspace_comp_3_;
    size_t workspace_size_comp_1_bwd_ = 0;
    size_t workspace_size_comp_2_bwd_ = 0;
    size_t workspace_size_comp_3_bwd_ = 0;

    // ========== cuDNN Frontend Graphs (反向) ==========
    std::shared_ptr<fe::graph::Graph> bn_bwd_graph;
    std::shared_ptr<fe::graph::Graph> conv_dgrad_graph;
    std::shared_ptr<fe::graph::Graph> conv_wgrad_graph;

    // ========== Tensor attributes (反向) ==========
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_dy_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_x_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_scale_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_mean_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_inv_var_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_bitmask_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_dx_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_dscale_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_bwd_dbias_tensor;

    std::shared_ptr<fe::graph::Tensor_attributes> conv_dgrad_grad_output_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_dgrad_weight_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_dgrad_grad_input_tensor;

    std::shared_ptr<fe::graph::Tensor_attributes> conv_wgrad_grad_output_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_wgrad_input_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_wgrad_grad_weight_tensor;

    // ========== CUDA Graph资源 (反向) ==========
    cudaGraph_t graph_bwd_ = nullptr;
    cudaGraphExec_t graph_exec_bwd_ = nullptr;
    bool graph_captured_ = false;

    // ========== 静态Variant Packs (反向) ==========
    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_bn_bwd_;
    VariantPack vp_dgrad_;
    VariantPack vp_wgrad_;

    // ========== Experience和Engine信息记录 ==========
    std::vector<std::pair<std::string, ExperienceStatus>> experience_records_;
    std::vector<std::pair<std::string, std::string>> engine_records_;

public:
    ConvBNReLUBackwardBenchmark(const Config& config)
        : N(config.batch_size), C(config.in_channels),
          H_in(config.input_size), W_in(config.input_size),
          K(config.out_channels), R(config.kernel_size), S(config.kernel_size),
          padding(config.get_padding()), stride(config.conv_stride),
          H_out(config.get_output_size()), W_out(config.get_output_size()),
          search_mode_(config.search_mode),
          heur_mode_(config.get_heur_mode()),
          use_graph_(config.use_graph) {

        // ========== 创建流和事件 ==========
        CHECK_CUDNN(cudnnCreate(&handle_comp_1_));
        CHECK_CUDNN(cudnnCreate(&handle_comp_2_));
        CHECK_CUDNN(cudnnCreate(&handle_comp_3_));

        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_2_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_3_, cudaStreamNonBlocking));

        CHECK_CUDNN(cudnnSetStream(handle_comp_1_, stream_comp_1_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_2_, stream_comp_2_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_3_, stream_comp_3_));

        CHECK_CUDA(cudaEventCreateWithFlags(&event_comp_1_bwd_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_bwd_join_s2_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_bwd_join_s3_, cudaEventDisableTiming));

        allocate_memory();
        initialize_data();
        build_bn_backward_graph();
        build_conv_dgrad_graph();
        build_conv_wgrad_graph();

        // ========== 打印Experience和Engine信息 ==========
        std::cout << "Conv Backward Type: BN+DGrad+WGrad (SEQUENTIAL: DGrad waits WGrad)" << std::endl;

        for (const auto& [op_name, exp_status] : experience_records_) {
            if (exp_status == ExperienceStatus::FOUND) {
                std::cout << op_name << " Experience: Found" << std::endl;
            } else if (exp_status == ExperienceStatus::FOUND_BACKUP1) {
                std::cout << op_name << " Experience: Found (Backup1)" << std::endl;
            } else if (exp_status == ExperienceStatus::FOUND_BACKUP2) {
                std::cout << op_name << " Experience: Found (Backup2)" << std::endl;
            }
        }

        std::cout << std::endl;

        for (const auto& [op_name, engine_name] : engine_records_) {
            std::cout << op_name << " Engine: " << engine_name << std::endl;
        }

        allocate_workspace();
        build_static_variant_packs();
        configure_static_l2_policies();
    }

    ~ConvBNReLUBackwardBenchmark() {
        // ========== 销毁Graph资源 ==========
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

        // ========== 销毁事件 ==========
        CHECK_CUDA(cudaEventDestroy(event_bwd_join_s3_));
        CHECK_CUDA(cudaEventDestroy(event_bwd_join_s2_));
        CHECK_CUDA(cudaEventDestroy(event_comp_1_bwd_done_));

        // ========== 销毁GPU内存 ==========
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaFree(d_mean));
        CHECK_CUDA(cudaFree(d_inv_variance));
        CHECK_CUDA(cudaFree(d_relu_bitmask));
        CHECK_CUDA(cudaFree(d_grad_output));
        CHECK_CUDA(cudaFree(d_grad_conv_out));
        CHECK_CUDA(cudaFree(d_grad_input));
        CHECK_CUDA(cudaFree(d_grad_weight));
        CHECK_CUDA(cudaFree(d_dscale));
        CHECK_CUDA(cudaFree(d_dbias));
        CHECK_CUDA(cudaFree(d_scale));
        CHECK_CUDA(cudaFree(d_weight));

        if (d_workspace_comp_1_) CHECK_CUDA(cudaFree(d_workspace_comp_1_));
        if (d_workspace_comp_2_) CHECK_CUDA(cudaFree(d_workspace_comp_2_));
        if (d_workspace_comp_3_) CHECK_CUDA(cudaFree(d_workspace_comp_3_));

        // ========== 销毁流和句柄 ==========
        CHECK_CUDNN(cudnnDestroy(handle_comp_3_));
        CHECK_CUDNN(cudnnDestroy(handle_comp_2_));
        CHECK_CUDNN(cudnnDestroy(handle_comp_1_));

        CHECK_CUDA(cudaStreamDestroy(stream_comp_3_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_2_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_1_));
    }

private:
    void allocate_memory() {
        size_t input_elements = N * C * H_in * W_in;
        size_t output_elements = N * K * H_out * W_out;
        size_t weight_elements = K * C * R * S;
        size_t stats_elements = K;

        size_t input_size = input_elements * sizeof(__half);
        size_t output_size = output_elements * sizeof(__half);
        size_t weight_size = weight_elements * sizeof(__half);
        size_t stats_size = stats_elements * sizeof(float);
        size_t bitmask_size = output_elements;

        // 反向需要的前向输出（模拟已计算）
        d_input = allocate_aligned_gpu_memory(input_size);
        d_output = allocate_aligned_gpu_memory(output_size);
        d_mean = allocate_aligned_gpu_memory(stats_size);
        d_inv_variance = allocate_aligned_gpu_memory(stats_size);
        d_relu_bitmask = allocate_aligned_gpu_memory(bitmask_size);

        // ████████████████████████████████████████████████████████████████████████████
        // ████ 重要说明：此文件为独立的反向传播 Benchmark，不包含正向传播过程 ████
        // ████ ReLU Bitmask 在实际训练中应由前向传播生成（cbr_fwd_fp16.cpp）    ████
        // ████ 此处使用随机数（Bernoulli 0.5）填充仅用于模拟，不代表真实数据     ████
        // ████████████████████████████████████████████████████████████████████████████

        // 反向输入
        d_grad_output = allocate_aligned_gpu_memory(output_size);

        // 反向输出
        d_grad_conv_out = allocate_aligned_gpu_memory(output_size);
        d_grad_input = allocate_aligned_gpu_memory(input_size);
        d_grad_weight = allocate_aligned_gpu_memory(weight_size);
        d_dscale = allocate_aligned_gpu_memory(stats_size);
        d_dbias = allocate_aligned_gpu_memory(stats_size);

        // BN参数
        d_scale = allocate_aligned_gpu_memory(stats_size);

        // Conv权重
        d_weight = allocate_aligned_gpu_memory(weight_size);
    }

    void allocate_workspace() {
        d_workspace_comp_1_ = (workspace_size_comp_1_bwd_ > 0) ?
            allocate_aligned_gpu_memory(workspace_size_comp_1_bwd_) : nullptr;
        d_workspace_comp_2_ = (workspace_size_comp_2_bwd_ > 0) ?
            allocate_aligned_gpu_memory(workspace_size_comp_2_bwd_) : nullptr;
        d_workspace_comp_3_ = (workspace_size_comp_3_bwd_ > 0) ?
            allocate_aligned_gpu_memory(workspace_size_comp_3_bwd_) : nullptr;
    }

    void initialize_data() {
        size_t output_elements = N * K * H_out * W_out;
        size_t input_elements = N * C * H_in * W_in;
        size_t weight_elements = K * C * R * S;
        size_t stats_elements = K;

        // 初始化反向需要的前向输出（模拟已计算）
        initialize_random_fp16(d_input, input_elements, 41);
        initialize_random_fp16(d_output, output_elements, 42);
        initialize_random_fp32(d_mean, stats_elements, 43);
        // EXP.md修复2：inv_variance必须为正值，使用initialize_positive_fp32
        initialize_positive_fp32(d_inv_variance, stats_elements, 0.5f, 5.0f, 44);

        // ████████████████████████████████████████████████████████████████████████████
        // ████ 模拟前向传播生成的 ReLU Bitmask（Bernoulli 0.5分布，50%激活率） ████
        // ████ 注意：真实训练中，此 bitmask 应由 cbr_fwd_fp16.cpp 的前向传播生成   ████
        // ████████████████████████████████████████████████████████████████████████████
        initialize_relu_bitmask(d_relu_bitmask, output_elements);

        // 初始化反向输入（上游梯度）
        initialize_random_fp16(d_grad_output, output_elements, 45);

        // 初始化BN参数
        initialize_random_fp32(d_scale, stats_elements, 46);

        // 初始化Conv权重
        initialize_random_fp16(d_weight, weight_elements, 47);
    }

    void build_static_variant_packs() {
        // ========== 预留容量 ==========
        vp_bn_bwd_.reserve(15);
        vp_dgrad_.reserve(5);
        vp_wgrad_.reserve(5);

        // ========== 反向图1: BN Backward ==========
        vp_bn_bwd_[bn_bwd_dy_tensor] = d_grad_output;
        vp_bn_bwd_[bn_bwd_bitmask_tensor] = d_relu_bitmask;
        vp_bn_bwd_[bn_bwd_x_tensor] = d_output;
        vp_bn_bwd_[bn_bwd_scale_tensor] = d_scale;
        vp_bn_bwd_[bn_bwd_mean_tensor] = d_mean;
        vp_bn_bwd_[bn_bwd_inv_var_tensor] = d_inv_variance;
        vp_bn_bwd_[bn_bwd_dx_tensor] = d_grad_conv_out;
        vp_bn_bwd_[bn_bwd_dscale_tensor] = d_dscale;
        vp_bn_bwd_[bn_bwd_dbias_tensor] = d_dbias;

        // ========== 反向图2: DGrad ==========
        vp_dgrad_[conv_dgrad_grad_output_tensor] = d_grad_conv_out;
        vp_dgrad_[conv_dgrad_weight_tensor] = d_weight;
        vp_dgrad_[conv_dgrad_grad_input_tensor] = d_grad_input;

        // ========== 反向图3: WGrad ==========
        vp_wgrad_[conv_wgrad_grad_output_tensor] = d_grad_conv_out;
        vp_wgrad_[conv_wgrad_input_tensor] = d_input;
        vp_wgrad_[conv_wgrad_grad_weight_tensor] = d_grad_weight;
    }

    void build_bn_backward_graph() {
        bn_bwd_graph = std::make_shared<fe::graph::Graph>();
        bn_bwd_graph->set_io_data_type(fe::DataType_t::HALF)
                    .set_intermediate_data_type(fe::DataType_t::HALF)
                    .set_compute_data_type(fe::DataType_t::FLOAT);

        std::vector<int64_t> x_dim = {N, K, H_out, W_out};
        std::vector<int64_t> x_stride = {int64_t(H_out*W_out*K), int64_t(1), int64_t(W_out)*K, int64_t(K)};
        std::vector<int64_t> param_dim = {1, K, 1, 1};
        std::vector<int64_t> param_stride = {K, 1, K, K};

        bn_bwd_dy_tensor = bn_bwd_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_bwd_dy")
            .set_dim(x_dim)
            .set_stride(x_stride)
            .set_data_type(fe::DataType_t::HALF));

        bn_bwd_x_tensor = bn_bwd_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_bwd_x")
            .set_dim(x_dim)
            .set_stride(x_stride)
            .set_data_type(fe::DataType_t::HALF));

        bn_bwd_scale_tensor = bn_bwd_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_bwd_scale")
            .set_dim(param_dim)
            .set_stride(param_stride)
            .set_data_type(fe::DataType_t::FLOAT));

        bn_bwd_mean_tensor = bn_bwd_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_bwd_mean")
            .set_dim(param_dim)
            .set_stride(param_stride)
            .set_data_type(fe::DataType_t::FLOAT));

        bn_bwd_inv_var_tensor = bn_bwd_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_bwd_inv_var")
            .set_dim(param_dim)
            .set_stride(param_stride)
            .set_data_type(fe::DataType_t::FLOAT));

        // ████████████████████████████████████████████████████████████████████████████
        // ████ ReLU Bitmask Tensor：反向传播用于过滤梯度（dy * bitmask）          ████
        // ████ 此 bitmask 应由前向传播生成，此处使用模拟数据                      ████
        // ████████████████████████████████████████████████████████████████████████████
        bn_bwd_bitmask_tensor = bn_bwd_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_bwd_bitmask")
            .set_dim(x_dim)
            .set_stride(x_stride)
            .set_data_type(fe::DataType_t::BOOLEAN));

        auto mul_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dy_with_mask = bn_bwd_graph->pointwise(bn_bwd_dy_tensor, bn_bwd_bitmask_tensor, mul_options);
        dy_with_mask->set_data_type(fe::DataType_t::HALF);

        auto dbn_options = fe::graph::Batchnorm_backward_attributes()
            .set_saved_mean_and_inv_variance(bn_bwd_mean_tensor, bn_bwd_inv_var_tensor)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto [dx, dscale, dbias] = bn_bwd_graph->batchnorm_backward(
            dy_with_mask, bn_bwd_x_tensor, bn_bwd_scale_tensor, dbn_options);

        bn_bwd_dx_tensor = dx;
        bn_bwd_dscale_tensor = dscale;
        bn_bwd_dbias_tensor = dbias;

        bn_bwd_dx_tensor->set_output(true)
            .set_name("bn_bwd_dx")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out)*W_out*K, int64_t(1), int64_t(W_out)*K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF);

        bn_bwd_dscale_tensor->set_output(true)
            .set_name("bn_bwd_dscale")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT);

        bn_bwd_dbias_tensor->set_output(true)
            .set_name("bn_bwd_dbias")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT);

        CHECK_CUDNN_FE(bn_bwd_graph->validate());
        CHECK_CUDNN_FE(bn_bwd_graph->build_operation_graph(handle_comp_1_));
        CHECK_CUDNN_FE(bn_bwd_graph->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(bn_bwd_graph->check_support(handle_comp_1_));
        CHECK_CUDNN_FE(bn_bwd_graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        workspace_size_comp_1_bwd_ = bn_bwd_graph->get_workspace_size();
    }

    void build_conv_dgrad_graph() {
        conv_dgrad_graph = std::make_shared<fe::graph::Graph>();
        conv_dgrad_graph->set_io_data_type(fe::DataType_t::HALF)
                          .set_intermediate_data_type(fe::DataType_t::HALF)
                          .set_compute_data_type(fe::DataType_t::FLOAT);

        conv_dgrad_grad_output_tensor = conv_dgrad_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("conv_dgrad_grad_output")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out)*W_out*K, int64_t(1), int64_t(W_out)*K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF));

        conv_dgrad_weight_tensor = conv_dgrad_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("conv_dgrad_weight")
            .set_dim({K, C, R, S})
            .set_stride({int64_t(R*S*C), int64_t(1), int64_t(S)*C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        auto dgrad_options = fe::graph::Conv_dgrad_attributes()
            .set_padding({padding, padding})
            .set_stride({stride, stride})
            .set_dilation({1, 1});

        auto dgrad_out = conv_dgrad_graph->conv_dgrad(
            conv_dgrad_grad_output_tensor, conv_dgrad_weight_tensor, dgrad_options);
        conv_dgrad_grad_input_tensor = dgrad_out;

        conv_dgrad_grad_input_tensor->set_output(true)
            .set_name("conv_dgrad_grad_input")
            .set_dim({N, C, H_in, W_in})
            .set_stride({int64_t(H_in)*W_in*C, int64_t(1), int64_t(W_in)*C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF);

        CHECK_CUDNN_FE(conv_dgrad_graph->validate());
        CHECK_CUDNN_FE(conv_dgrad_graph->build_operation_graph(handle_comp_2_));

        // ========== Mode C分支 ==========
        if (search_mode_ == Config::SearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
            std::string key = build_shape_key(
                "conv_dgrad", "fp16",
                N, H_in, W_in, C, K, R, S, stride, padding
            );

            auto exp_rec = ta_v4::experience::lookup(key);

            if (exp_rec != nullptr) {
                CHECK_CUDNN_FE(conv_dgrad_graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}));
                std::vector<int64_t> candidates;
                int64_t count = conv_dgrad_graph->get_execution_plan_count();
                for (int64_t i = 0; i < count; ++i) {
                    candidates.push_back(i);
                }

                auto [exp_status, matched] = match_and_build_plan(
                    conv_dgrad_graph, candidates, exp_rec, handle_comp_2_
                );
                experience_records_.push_back({"DGrad", exp_status});

                if (!matched) {
                    goto fallback_heuristic_dgrad;
                }
            } else {
                goto fallback_heuristic_dgrad;
            }
#else
            goto fallback_heuristic_dgrad;
#endif
        } else {
        fallback_heuristic_dgrad:
            CHECK_CUDNN_FE(conv_dgrad_graph->create_execution_plans({heur_mode_}));
            CHECK_CUDNN_FE(conv_dgrad_graph->check_support(handle_comp_2_));
            CHECK_CUDNN_FE(conv_dgrad_graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        }

        std::string engine_name;
        CHECK_CUDNN_FE(conv_dgrad_graph->get_plan_name(engine_name));
        engine_records_.push_back({"DGrad", engine_name});

        workspace_size_comp_2_bwd_ = conv_dgrad_graph->get_workspace_size();
    }

    void build_conv_wgrad_graph() {
        conv_wgrad_graph = std::make_shared<fe::graph::Graph>();
        conv_wgrad_graph->set_io_data_type(fe::DataType_t::HALF)
                          .set_intermediate_data_type(fe::DataType_t::HALF)
                          .set_compute_data_type(fe::DataType_t::FLOAT);

        conv_wgrad_grad_output_tensor = conv_wgrad_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("conv_wgrad_grad_output")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out)*W_out*K, int64_t(1), int64_t(W_out)*K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF));

        conv_wgrad_input_tensor = conv_wgrad_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("conv_wgrad_input")
            .set_dim({N, C, H_in, W_in})
            .set_stride({int64_t(H_in)*W_in*C, int64_t(1), int64_t(W_in)*C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        auto wgrad_options = fe::graph::Conv_wgrad_attributes()
            .set_padding({padding, padding})
            .set_stride({stride, stride})
            .set_dilation({1, 1});

        auto wgrad_out = conv_wgrad_graph->conv_wgrad(
            conv_wgrad_grad_output_tensor, conv_wgrad_input_tensor, wgrad_options);
        conv_wgrad_grad_weight_tensor = wgrad_out;

        conv_wgrad_grad_weight_tensor->set_output(true)
            .set_name("conv_wgrad_grad_weight")
            .set_dim({K, C, R, S})
            .set_stride({int64_t(R*S*C), int64_t(1), int64_t(S)*C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF);

        CHECK_CUDNN_FE(conv_wgrad_graph->validate());
        CHECK_CUDNN_FE(conv_wgrad_graph->build_operation_graph(handle_comp_3_));

        // ========== Mode C分支 ==========
        if (search_mode_ == Config::SearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
            std::string key = build_shape_key(
                "conv_wgrad", "fp16",
                N, H_in, W_in, C, K, R, S, stride, padding
            );

            auto exp_rec = ta_v4::experience::lookup(key);

            if (exp_rec != nullptr) {
                CHECK_CUDNN_FE(conv_wgrad_graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}));
                std::vector<int64_t> candidates;
                int64_t count = conv_wgrad_graph->get_execution_plan_count();
                for (int64_t i = 0; i < count; ++i) {
                    candidates.push_back(i);
                }

                auto [exp_status, matched] = match_and_build_plan(
                    conv_wgrad_graph, candidates, exp_rec, handle_comp_3_
                );
                experience_records_.push_back({"WGrad", exp_status});

                if (!matched) {
                    goto fallback_heuristic_wgrad;
                }
            } else {
                goto fallback_heuristic_wgrad;
            }
#else
            goto fallback_heuristic_wgrad;
#endif
        } else {
        fallback_heuristic_wgrad:
            CHECK_CUDNN_FE(conv_wgrad_graph->create_execution_plans({heur_mode_}));
            CHECK_CUDNN_FE(conv_wgrad_graph->check_support(handle_comp_3_));
            CHECK_CUDNN_FE(conv_wgrad_graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        }

        std::string engine_name;
        CHECK_CUDNN_FE(conv_wgrad_graph->get_plan_name(engine_name));
        engine_records_.push_back({"WGrad", engine_name});

        workspace_size_comp_3_bwd_ = conv_wgrad_graph->get_workspace_size();
    }

    /**
     * @brief 重置所有流为 Streaming 模式
     */
    void reset_all_l2_policies() {
        cudaStreamAttrValue attr_streaming = {};
        attr_streaming.accessPolicyWindow.base_ptr = nullptr;
        attr_streaming.accessPolicyWindow.num_bytes = 0;
        attr_streaming.accessPolicyWindow.hitRatio = 0.0f;
        attr_streaming.accessPolicyWindow.hitProp = cudaAccessPropertyStreaming;
        attr_streaming.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        CHECK_CUDA(cudaStreamSetAttribute(stream_comp_1_,
            cudaStreamAttributeAccessPolicyWindow, &attr_streaming));
        CHECK_CUDA(cudaStreamSetAttribute(stream_comp_2_,
            cudaStreamAttributeAccessPolicyWindow, &attr_streaming));
        CHECK_CUDA(cudaStreamSetAttribute(stream_comp_3_,
            cudaStreamAttributeAccessPolicyWindow, &attr_streaming));
    }

    /**
     * @brief 配置反向 L2 优化策略（保护 d_grad_conv_out）
     */
    void configure_backward_l2_policies() {
        const size_t bwd_tensor_size =
            static_cast<size_t>(N) * K * H_out * W_out * sizeof(__half);
        constexpr size_t L2_SAFE_THRESHOLD = 10ULL * 1024ULL * 1024ULL;  // 10MB

        reset_all_l2_policies();

        if (bwd_tensor_size <= L2_SAFE_THRESHOLD) {
            cudaStreamAttrValue attr_persist = {};
            attr_persist.accessPolicyWindow.base_ptr = d_grad_conv_out;
            attr_persist.accessPolicyWindow.num_bytes = bwd_tensor_size;
            attr_persist.accessPolicyWindow.hitRatio = 1.0f;
            attr_persist.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr_persist.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            CHECK_CUDA(cudaStreamSetAttribute(stream_comp_2_,
                cudaStreamAttributeAccessPolicyWindow, &attr_persist));
            CHECK_CUDA(cudaStreamSetAttribute(stream_comp_3_,
                cudaStreamAttributeAccessPolicyWindow, &attr_persist));
        }
    }

    /**
     * @brief 静态配置L2缓存策略（与传统模式一致）
     * @note Graph模式下不启用反向L2优化（避免stream-state conflict）
     * @note 传统模式下启用反向L2优化（充分利用硬件预取器）
     */
    void configure_static_l2_policies() {
        const size_t bwd_tensor_size =
            static_cast<size_t>(N) * K * H_out * W_out * sizeof(__half);
        constexpr size_t L2_SAFE_THRESHOLD = 10ULL * 1024ULL * 1024ULL;

        reset_all_l2_policies();

        // 传统模式下启用反向 L2 优化（Graph 模式不启用）
        if (!use_graph_ && bwd_tensor_size <= L2_SAFE_THRESHOLD) {
            cudaStreamAttrValue attr_persist = {};
            attr_persist.accessPolicyWindow.base_ptr = d_grad_conv_out;
            attr_persist.accessPolicyWindow.num_bytes = bwd_tensor_size;
            attr_persist.accessPolicyWindow.hitRatio = 1.0f;
            attr_persist.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr_persist.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            CHECK_CUDA(cudaStreamSetAttribute(stream_comp_2_,
                cudaStreamAttributeAccessPolicyWindow, &attr_persist));
            CHECK_CUDA(cudaStreamSetAttribute(stream_comp_3_,
                cudaStreamAttributeAccessPolicyWindow, &attr_persist));
        }
    }

public:
    /**
     * @brief 捕获CUDA Graph（仅反向）
     */
    void capture_cuda_graphs() {
        if (!use_graph_) {
            return;
        }

        if (graph_captured_) {
            return;
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        // ========== 配置L2策略 ==========
        configure_static_l2_policies();
        CHECK_CUDA(cudaDeviceSynchronize());

        // ========== 捕获反向Graph ==========
        cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeThreadLocal;
        cudaError_t capture_err = cudaStreamBeginCapture(stream_comp_1_, capture_mode);
        if (capture_err != cudaSuccess) {
            std::cerr << "[Backward] Failed to begin capture: "
                      << cudaGetErrorString(capture_err) << std::endl;
            return;
        }

        // === S1: BN Backward ===
        CHECK_CUDNN_FE(bn_bwd_graph->execute(handle_comp_1_, vp_bn_bwd_, d_workspace_comp_1_));
        CHECK_CUDA(cudaEventRecord(event_comp_1_bwd_done_, stream_comp_1_));

        // === Fork点：S2和S3并发等待S1 ===
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_comp_1_bwd_done_, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_comp_1_bwd_done_, 0));

        // === S3: WGrad（先执行） ===
        CHECK_CUDNN_FE(conv_wgrad_graph->execute(handle_comp_3_, vp_wgrad_, d_workspace_comp_3_));
        CHECK_CUDA(cudaEventRecord(event_bwd_join_s3_, stream_comp_3_));

        // === S2: DGrad（等待WGrad完成后执行） ===
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_bwd_join_s3_, 0));  // 关键修改：DGrad等待WGrad
        CHECK_CUDNN_FE(conv_dgrad_graph->execute(handle_comp_2_, vp_dgrad_, d_workspace_comp_2_));
        CHECK_CUDA(cudaEventRecord(event_bwd_join_s2_, stream_comp_2_));

        // === Join闭环：S1等待S2完成 ===
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_join_s2_, 0));

        // === 从S1结束捕获 ===
        capture_err = cudaStreamEndCapture(stream_comp_1_, &graph_bwd_);
        if (capture_err != cudaSuccess) {
            std::cerr << "[Backward] Failed to end capture: "
                      << cudaGetErrorString(capture_err) << std::endl;
            if (capture_err == cudaErrorStreamCaptureUnjoined) {
                std::cerr << "[Backward] Unjoined stream detected! Check topology." << std::endl;
            }
            return;
        }

        if (graph_bwd_ == nullptr) {
            std::cerr << "[Backward] Capture returned null graph!" << std::endl;
            return;
        }

        size_t bwd_nodes = 0;
        CHECK_CUDA(cudaGraphGetNodes(graph_bwd_, nullptr, &bwd_nodes));

        cudaGraphNode_t error_node;
        char log_buffer[2048];
        cudaError_t inst_err = cudaGraphInstantiate(
            &graph_exec_bwd_, graph_bwd_, &error_node, log_buffer, sizeof(log_buffer));

        if (inst_err != cudaSuccess) {
            std::cerr << "[Backward] Instantiation failed: "
                      << cudaGetErrorString(inst_err) << std::endl;
            std::cerr << "[Backward] Error log: " << log_buffer << std::endl;
            cudaGraphDestroy(graph_bwd_);
            graph_bwd_ = nullptr;
            return;
        }

        graph_captured_ = true;
    }

    /**
     * @brief 反向传播（传统模式）
     */
    void execute_backward_traditional() {
        CHECK_CUDNN_FE(bn_bwd_graph->execute(handle_comp_1_, vp_bn_bwd_, d_workspace_comp_1_));
        CHECK_CUDA(cudaEventRecord(event_comp_1_bwd_done_, stream_comp_1_));

        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_comp_1_bwd_done_, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_comp_1_bwd_done_, 0));

        // === S3: WGrad（先执行） ===
        CHECK_CUDNN_FE(conv_wgrad_graph->execute(handle_comp_3_, vp_wgrad_, d_workspace_comp_3_));
        CHECK_CUDA(cudaEventRecord(event_bwd_join_s3_, stream_comp_3_));

        // === S2: DGrad（等待WGrad完成后执行） ===
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_bwd_join_s3_, 0));  // 关键修改：DGrad等待WGrad
        CHECK_CUDNN_FE(conv_dgrad_graph->execute(handle_comp_2_, vp_dgrad_, d_workspace_comp_2_));
        CHECK_CUDA(cudaEventRecord(event_bwd_join_s2_, stream_comp_2_));

        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_bwd_join_s2_, 0));
    }

    /**
     * @brief 反向传播（自动选择Graph或传统模式）
     */
    void execute_backward() {
        if (use_graph_ && graph_captured_ && graph_exec_bwd_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_bwd_, stream_comp_1_));
        } else {
            execute_backward_traditional();
        }
    }

    /**
     * @brief Warmup流程（EXP.md修复4：默认50次迭代）
     * @note 专家共识度：7/8专家支持50次warmup
     * @note 50次warmup + 锁频 = 最优方案
     */
    void warmup(int iterations = 50) {
        // ========== 传统执行预热 ==========
        for (int i = 0; i < iterations; ++i) {
            execute_backward_traditional();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));

        // ========== Graph捕获 ==========
        if (!use_graph_) {
            return;
        }

        try {
            capture_cuda_graphs();

            if (!graph_captured_) {
                std::cerr << "Warning: Graph capture failed, falling back to traditional mode" << std::endl;
                return;
            }

            // ========== Graph模式预热 ==========
            for (int i = 0; i < iterations; ++i) {
                execute_backward();
            }
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        } catch (const std::exception& e) {
            std::cerr << "\nException during graph capture: " << e.what() << std::endl;
            std::cerr << "Falling back to traditional execution mode" << std::endl;
            use_graph_ = false;
            graph_captured_ = false;
        }
    }

    void benchmark(int iterations = 500) {
        // ========== 反向计时 ==========
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute_backward();
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        if (!use_graph_) {
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "\nTime (Unit: ms):" << std::endl;
        std::cout << "Total:    " << total_time << std::endl;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ parse_arguments函数已移至ta_v4_common_fp16.hpp（单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████

int main(int argc, char** argv) {
    Config config = parse_arguments(argc, argv);

    try {
        ConvBNReLUBackwardBenchmark benchmark(config);
        benchmark.warmup();
        benchmark.benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
