/**
 * @file cbr_inf_fp16.cpp
 * @brief cuDNN Frontend Conv+BN+ReLU推理融合性能测试 (FP16版本，仅推理)
 * @version 1.0.0
 * @date 2026-04-13
 * @author 技术觉醒团队
 * @note 依赖项: cuDNN Frontend 1.17, CUDA 13.1, cuDNN 9.17
 * @note 所属系列: cuda
 *
 * @note 设计原则（专家意见B-3）：
 *   1. 与 cbr_fwd_fp16.cpp / cbr_bwd_fp16.cpp 保持张量命名和显存规划兼容
 *   2. 主线采用单流单图：Conv + MUL(eq_scale) + ADD(eq_bias) + ReLU
 *   3. 若单图构建失败，自动降级为单流双图：Conv / BNApply+ReLU
 *   4. 不做BN fold-in到卷积权重
 *   5. 不使用BatchNorm API作为主表达，直接使用Pointwise仿射化BN
 *   6. 使用device端kernel预计算推理态eq_scale/eq_bias
 *
 * @note 推理拓扑（单流单图主线）:
 *       Input + Weight → [Conv Fprop] → Conv_Output
 *                                                ↓
 *                                     eq_scale → [MUL] → Scaled_Output
 *                                                ↓
 *                                      eq_bias → [ADD] → BN_Output
 *                                                ↓
 *                                          [ReLU] → Final_Output
 *
 * @note 相比训练正向的优化:
 *   1. 消除GenStats计算（~10-15%）
 *   2. 消除BN Finalize全局规约（~5-8%）
 *   3. 消除ReLU Bitmask生成（~2-3%）
 *   4. 单流消除Event开销（~1-2%）
 *   预期综合加速: 18-28%
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
// - 初始化函数：initialize_random_fp16, initialize_random_fp32, initialize_scalar_fp32
// - 配置系统：Config结构体, parse_arguments
// - Experience查询：ExperienceStatus, build_shape_key, match_and_build_plan
// - 平台条件编译：A100/RTX5090 Experience表
// ████████████████████████████████████████████████████████████████████████████

// ==================== 推理参数预计算（Host端） ====================
// 注意：B-3专家建议使用device端kernel，但.cpp文件中改用host端计算
// 推理参数预计算只执行一次，host端计算性能影响可忽略

class ConvBNReLUInferenceBenchmark {
private:
    // ========== 资源管理（单流）==========
    cudnnHandle_t handle_;
    cudaStream_t stream_;

    const int64_t N, C, H_in, W_in, K;
    const int64_t R, S;
    const int64_t padding;
    const int64_t stride;
    const int64_t H_out, W_out;
    const float EPSILON = 1e-5f;

    const Config::SearchMode search_mode_;
    const fe::HeurMode_t heur_mode_;
    bool use_graph_;
    bool graph_captured_ = false;

    // true: 单图主线
    // false: 双图fallback
    bool use_single_graph_ = true;

    // ==================== GPU Memory ====================
    // 与训练态兼容的核心张量
    void *d_input, *d_weight, *d_output;
    void *d_bn_relu_output;

    void *d_scale, *d_bias;
    void *d_prev_running_mean, *d_prev_running_var;
    void *d_eq_scale, *d_eq_bias;

    // 保留训练态中间张量分配，保证显存布局兼容
    void *d_sum, *d_sq_sum;
    void *d_mean, *d_inv_variance;
    void *d_next_running_mean, *d_next_running_var;
    void *d_relu_bitmask;

    // ==================== Workspace ====================
    void *d_workspace_main_ = nullptr;
    void *d_workspace_fallback_ = nullptr;
    size_t workspace_size_main_ = 0;
    size_t workspace_size_fallback_ = 0;

    // ==================== Graphs ====================
    std::shared_ptr<fe::graph::Graph> main_graph_;
    std::shared_ptr<fe::graph::Graph> fallback_graph_;

    // 单图所需tensor
    std::shared_ptr<fe::graph::Tensor_attributes> single_input_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> single_weight_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> single_eq_scale_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> single_eq_bias_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> single_output_tensor_;

    // 双图fallback：Conv图
    std::shared_ptr<fe::graph::Tensor_attributes> conv_input_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_weight_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_output_tensor_;

    // 双图fallback：Pointwise图
    std::shared_ptr<fe::graph::Tensor_attributes> pw_x_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> pw_scale_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> pw_bias_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> pw_y_tensor_;

    // ==================== CUDA Graph ====================
    cudaGraph_t graph_inf_ = nullptr;
    cudaGraphExec_t graph_exec_inf_ = nullptr;

    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_main_;
    VariantPack vp_fallback_;

    std::vector<std::pair<std::string, ExperienceStatus>> experience_records_;
    std::vector<std::pair<std::string, std::string>> engine_records_;

public:
    ConvBNReLUInferenceBenchmark(const Config& config)
        : N(config.batch_size), C(config.in_channels),
          H_in(config.input_size), W_in(config.input_size),
          K(config.out_channels), R(config.kernel_size), S(config.kernel_size),
          padding(config.get_padding()), stride(config.conv_stride),
          H_out(config.get_output_size()), W_out(config.get_output_size()),
          search_mode_(config.search_mode),
          heur_mode_(config.get_heur_mode()),
          use_graph_(config.use_graph) {

        CHECK_CUDNN(cudnnCreate(&handle_));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CHECK_CUDNN(cudnnSetStream(handle_, stream_));

        allocate_memory();
        initialize_data();
        // precompute_eq_params() 移至 update_inference_params() 中调用

        build_graphs();

        allocate_workspace();
        build_static_variant_packs();
        configure_static_l2_policies();

        std::cout << "Conv Inference Type: "
                  << (use_single_graph_
                      ? "Single-Graph Conv+Mul(eq_scale)+Add(eq_bias)+ReLU"
                      : "Fallback Two-Graph Conv / BNApply+ReLU")
                  << std::endl;

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
    }

    ~ConvBNReLUInferenceBenchmark() {
        if (graph_exec_inf_) {
            cudaError_t err = cudaGraphExecDestroy(graph_exec_inf_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphExecDestroy(inf) failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        if (graph_inf_) {
            cudaError_t err = cudaGraphDestroy(graph_inf_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphDestroy(inf) failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        if (d_workspace_main_) CHECK_CUDA(cudaFree(d_workspace_main_));
        if (d_workspace_fallback_) CHECK_CUDA(cudaFree(d_workspace_fallback_));

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_weight));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaFree(d_bn_relu_output));

        CHECK_CUDA(cudaFree(d_scale));
        CHECK_CUDA(cudaFree(d_bias));
        CHECK_CUDA(cudaFree(d_prev_running_mean));
        CHECK_CUDA(cudaFree(d_prev_running_var));
        CHECK_CUDA(cudaFree(d_eq_scale));
        CHECK_CUDA(cudaFree(d_eq_bias));

        CHECK_CUDA(cudaFree(d_sum));
        CHECK_CUDA(cudaFree(d_sq_sum));
        CHECK_CUDA(cudaFree(d_mean));
        CHECK_CUDA(cudaFree(d_inv_variance));
        CHECK_CUDA(cudaFree(d_next_running_mean));
        CHECK_CUDA(cudaFree(d_next_running_var));
        CHECK_CUDA(cudaFree(d_relu_bitmask));

        CHECK_CUDNN(cudnnDestroy(handle_));
        CHECK_CUDA(cudaStreamDestroy(stream_));
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
        size_t bitmask_size = output_elements; // 训练态字节mask兼容保留

        d_input = allocate_aligned_gpu_memory(input_size);
        d_weight = allocate_aligned_gpu_memory(weight_size);
        d_output = allocate_aligned_gpu_memory(output_size);
        // ═══════════════════════════════════════════════════════════════════
        // ⚠️ 推理态 d_output 语义说明：
        // ───────────────────────────────────────────────────────────────────────
        // 单图模式（use_single_graph_==true）：
        //   Conv输出作为cuDNN内部虚拟张量直通，d_output仅为显存池占位符，
        //   不包含有效数据，严禁dump或依赖
        //
        // Fallback双图模式（use_single_graph_==false）：
        //   d_output包含真实Conv输出数据，作为图间缓冲
        // ═══════════════════════════════════════════════════════════════════
        d_bn_relu_output = allocate_aligned_gpu_memory(output_size);

        d_scale = allocate_aligned_gpu_memory(stats_size);
        d_bias = allocate_aligned_gpu_memory(stats_size);
        d_prev_running_mean = allocate_aligned_gpu_memory(stats_size);
        d_prev_running_var = allocate_aligned_gpu_memory(stats_size);

        d_eq_scale = allocate_aligned_gpu_memory(stats_size);
        d_eq_bias = allocate_aligned_gpu_memory(stats_size);

        // 保留训练态兼容分配
        d_sum = allocate_aligned_gpu_memory(stats_size);
        d_sq_sum = allocate_aligned_gpu_memory(stats_size);
        d_mean = allocate_aligned_gpu_memory(stats_size);
        d_inv_variance = allocate_aligned_gpu_memory(stats_size);
        d_next_running_mean = allocate_aligned_gpu_memory(stats_size);
        d_next_running_var = allocate_aligned_gpu_memory(stats_size);
        d_relu_bitmask = allocate_aligned_gpu_memory(bitmask_size);
    }

    void allocate_workspace() {
        d_workspace_main_ = (workspace_size_main_ > 0) ? allocate_aligned_gpu_memory(workspace_size_main_) : nullptr;
        if (!use_single_graph_) {
            d_workspace_fallback_ = (workspace_size_fallback_ > 0) ? allocate_aligned_gpu_memory(workspace_size_fallback_) : nullptr;
        }
    }

    void initialize_data() {
        size_t input_elements = N * C * H_in * W_in;
        size_t weight_elements = K * C * R * S;
        size_t stats_elements = K;

        initialize_random_fp16(d_input, input_elements, 42);
        initialize_random_fp16(d_weight, weight_elements, 43);

        initialize_random_fp32(d_scale, stats_elements, 44);
        initialize_random_fp32(d_bias, stats_elements, 45);
        initialize_random_fp32(d_prev_running_mean, stats_elements, 46);
        initialize_positive_fp32(d_prev_running_var, stats_elements, 0.5f, 5.0f, 47);
    }

    void precompute_eq_params() {
        // Host端计算推理态BN等效参数
        // 公式：eq_scale = gamma / sqrt(running_var + epsilon)
        //       eq_bias = beta - running_mean * eq_scale

        std::vector<float> scale(K);
        std::vector<float> bias(K);
        std::vector<float> running_mean(K);
        std::vector<float> running_var(K);

        // 拷贝到host
        CHECK_CUDA(cudaMemcpy(scale.data(), d_scale, K * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(bias.data(), d_bias, K * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(running_mean.data(), d_prev_running_mean, K * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(running_var.data(), d_prev_running_var, K * sizeof(float), cudaMemcpyDeviceToHost));

        // 计算eq_scale和eq_bias
        std::vector<float> eq_scale(K);
        std::vector<float> eq_bias(K);
        for (int64_t i = 0; i < K; ++i) {
            float inv_std = 1.0f / std::sqrt(running_var[i] + EPSILON);
            eq_scale[i] = scale[i] * inv_std;
            eq_bias[i] = bias[i] - running_mean[i] * eq_scale[i];
        }

        // 拷贝回device
        CHECK_CUDA(cudaMemcpy(d_eq_scale, eq_scale.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_eq_bias, eq_bias.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    }

    bool try_build_single_graph() {
        main_graph_ = std::make_shared<fe::graph::Graph>();
        main_graph_->set_io_data_type(fe::DataType_t::HALF);
        main_graph_->set_intermediate_data_type(fe::DataType_t::HALF);
        main_graph_->set_compute_data_type(fe::DataType_t::FLOAT);

        single_input_tensor_ = main_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("input")
            .set_dim({N, C, H_in, W_in})
            .set_stride({int64_t(H_in) * W_in * C, int64_t(1), int64_t(W_in) * C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        single_weight_tensor_ = main_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("weight")
            .set_dim({K, C, R, S})
            .set_stride({int64_t(R) * S * C, int64_t(1), int64_t(S) * C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        single_eq_scale_tensor_ = main_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("eq_scale")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        single_eq_bias_tensor_ = main_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("eq_bias")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        fe::graph::Conv_fprop_attributes conv_options;
        conv_options.set_padding({padding, padding})
                    .set_stride({stride, stride})
                    .set_dilation({1, 1});

        auto conv_out = main_graph_->conv_fprop(
            single_input_tensor_, single_weight_tensor_, conv_options);

        conv_out->set_dim({N, K, H_out, W_out})
                .set_stride({int64_t(H_out) * W_out * K, int64_t(1), int64_t(W_out) * K, int64_t(K)})
                .set_data_type(fe::DataType_t::HALF);

        auto mul_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto scaled = main_graph_->pointwise(conv_out, single_eq_scale_tensor_, mul_options);
        scaled->set_data_type(fe::DataType_t::HALF);

        auto add_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto shifted = main_graph_->pointwise(scaled, single_eq_bias_tensor_, add_options);
        shifted->set_data_type(fe::DataType_t::HALF);

        auto relu_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto relu_out = main_graph_->pointwise(shifted, relu_options);

        single_output_tensor_ = relu_out;
        single_output_tensor_->set_output(true)
            .set_name("bn_relu_output")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out) * W_out * K, int64_t(1), int64_t(W_out) * K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF);

        auto st = main_graph_->validate();
        if (st.is_bad()) return false;

        st = main_graph_->build_operation_graph(handle_);
        if (st.is_bad()) return false;

        try {
            if (search_mode_ == Config::SearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
                std::string key = build_shape_key(
                    "conv_bn_relu_inf", "fp16",
                    N, H_in, W_in, C, K, R, S, stride, padding
                );

                auto exp_rec = ta_v4::experience::lookup(key);

                if (exp_rec != nullptr) {
                    CHECK_CUDNN_FE(main_graph_->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}));

                    std::vector<int64_t> candidates;
                    int64_t count = main_graph_->get_execution_plan_count();
                    for (int64_t i = 0; i < count; ++i) {
                        candidates.push_back(i);
                    }

                    auto [exp_status, matched] = match_and_build_plan(
                        main_graph_, candidates, exp_rec, handle_
                    );
                    experience_records_.push_back({"Inference", exp_status});

                    if (!matched) {
                        CHECK_CUDNN_FE(main_graph_->create_execution_plans({heur_mode_}));
                        CHECK_CUDNN_FE(main_graph_->check_support(handle_));
                        CHECK_CUDNN_FE(main_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
                    }
                } else {
                    CHECK_CUDNN_FE(main_graph_->create_execution_plans({heur_mode_}));
                    CHECK_CUDNN_FE(main_graph_->check_support(handle_));
                    CHECK_CUDNN_FE(main_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
                }
#else
                CHECK_CUDNN_FE(main_graph_->create_execution_plans({heur_mode_}));
                CHECK_CUDNN_FE(main_graph_->check_support(handle_));
                CHECK_CUDNN_FE(main_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
#endif
            } else {
                CHECK_CUDNN_FE(main_graph_->create_execution_plans({heur_mode_}));
                CHECK_CUDNN_FE(main_graph_->check_support(handle_));
                CHECK_CUDNN_FE(main_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
            }
        } catch (...) {
            return false;
        }

        std::string engine_name;
        auto name_status = main_graph_->get_plan_name(engine_name);
        if (!name_status.is_bad()) {
            engine_records_.push_back({"Inference", engine_name});
        }

        workspace_size_main_ = main_graph_->get_workspace_size();
        return true;
    }

    void build_fallback_graphs() {
        // Graph 1: Conv
        main_graph_ = std::make_shared<fe::graph::Graph>();
        main_graph_->set_io_data_type(fe::DataType_t::HALF);
        main_graph_->set_intermediate_data_type(fe::DataType_t::FLOAT);
        main_graph_->set_compute_data_type(fe::DataType_t::FLOAT);

        conv_input_tensor_ = main_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("input")
            .set_dim({N, C, H_in, W_in})
            .set_stride({int64_t(H_in) * W_in * C, int64_t(1), int64_t(W_in) * C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        conv_weight_tensor_ = main_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("weight")
            .set_dim({K, C, R, S})
            .set_stride({int64_t(R) * S * C, int64_t(1), int64_t(S) * C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        fe::graph::Conv_fprop_attributes conv_options;
        conv_options.set_padding({padding, padding})
                    .set_stride({stride, stride})
                    .set_dilation({1, 1});

        auto conv_out = main_graph_->conv_fprop(conv_input_tensor_, conv_weight_tensor_, conv_options);
        conv_output_tensor_ = conv_out;
        conv_output_tensor_->set_output(true)
            .set_name("output")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out) * W_out * K, int64_t(1), int64_t(W_out) * K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF);

        CHECK_CUDNN_FE(main_graph_->validate());
        CHECK_CUDNN_FE(main_graph_->build_operation_graph(handle_));
        CHECK_CUDNN_FE(main_graph_->create_execution_plans({heur_mode_}));
        CHECK_CUDNN_FE(main_graph_->check_support(handle_));
        CHECK_CUDNN_FE(main_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        std::string conv_engine_name;
        CHECK_CUDNN_FE(main_graph_->get_plan_name(conv_engine_name));
        engine_records_.push_back({"Conv", conv_engine_name});
        workspace_size_main_ = main_graph_->get_workspace_size();

        // Graph 2: Pointwise
        fallback_graph_ = std::make_shared<fe::graph::Graph>();
        fallback_graph_->set_io_data_type(fe::DataType_t::HALF);
        fallback_graph_->set_intermediate_data_type(fe::DataType_t::HALF);
        fallback_graph_->set_compute_data_type(fe::DataType_t::FLOAT);

        pw_x_tensor_ = fallback_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_relu_x")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out) * W_out * K, int64_t(1), int64_t(W_out) * K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF));

        pw_scale_tensor_ = fallback_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("eq_scale")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        pw_bias_tensor_ = fallback_graph_->tensor(fe::graph::Tensor_attributes()
            .set_name("eq_bias")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        auto mul_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto scaled = fallback_graph_->pointwise(pw_x_tensor_, pw_scale_tensor_, mul_options);
        scaled->set_data_type(fe::DataType_t::HALF);

        auto add_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto shifted = fallback_graph_->pointwise(scaled, pw_bias_tensor_, add_options);
        shifted->set_data_type(fe::DataType_t::HALF);

        auto relu_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto relu_out = fallback_graph_->pointwise(shifted, relu_options);

        pw_y_tensor_ = relu_out;
        pw_y_tensor_->set_output(true)
            .set_name("bn_relu_output")
            .set_dim({N, K, H_out, W_out})
            .set_stride({int64_t(H_out) * W_out * K, int64_t(1), int64_t(W_out) * K, int64_t(K)})
            .set_data_type(fe::DataType_t::HALF);

        CHECK_CUDNN_FE(fallback_graph_->validate());
        CHECK_CUDNN_FE(fallback_graph_->build_operation_graph(handle_));
        CHECK_CUDNN_FE(fallback_graph_->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(fallback_graph_->check_support(handle_));
        CHECK_CUDNN_FE(fallback_graph_->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        std::string pw_engine_name;
        CHECK_CUDNN_FE(fallback_graph_->get_plan_name(pw_engine_name));
        engine_records_.push_back({"Pointwise", pw_engine_name});
        workspace_size_fallback_ = fallback_graph_->get_workspace_size();
    }

    void build_graphs() {
        if (!try_build_single_graph()) {
            use_single_graph_ = false;
            main_graph_.reset();
            fallback_graph_.reset();
            engine_records_.clear();
            experience_records_.clear();
            build_fallback_graphs();
        }
    }

    void build_static_variant_packs() {
        vp_main_.clear();
        vp_fallback_.clear();

        if (use_single_graph_) {
            vp_main_.reserve(8);
            vp_main_[single_input_tensor_] = d_input;
            vp_main_[single_weight_tensor_] = d_weight;
            vp_main_[single_eq_scale_tensor_] = d_eq_scale;
            vp_main_[single_eq_bias_tensor_] = d_eq_bias;
            vp_main_[single_output_tensor_] = d_bn_relu_output;
        } else {
            vp_main_.reserve(5);
            vp_main_[conv_input_tensor_] = d_input;
            vp_main_[conv_weight_tensor_] = d_weight;
            vp_main_[conv_output_tensor_] = d_output;

            vp_fallback_.reserve(5);
            vp_fallback_[pw_x_tensor_] = d_output;
            vp_fallback_[pw_scale_tensor_] = d_eq_scale;
            vp_fallback_[pw_bias_tensor_] = d_eq_bias;
            vp_fallback_[pw_y_tensor_] = d_bn_relu_output;
        }
    }

    void reset_l2_policy() {
        cudaStreamAttrValue attr_streaming = {};
        attr_streaming.accessPolicyWindow.base_ptr = nullptr;
        attr_streaming.accessPolicyWindow.num_bytes = 0;
        attr_streaming.accessPolicyWindow.hitRatio = 0.0f;
        attr_streaming.accessPolicyWindow.hitProp = cudaAccessPropertyStreaming;
        attr_streaming.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        CHECK_CUDA(cudaStreamSetAttribute(
            stream_, cudaStreamAttributeAccessPolicyWindow, &attr_streaming));
    }

    void configure_static_l2_policies() {
        reset_l2_policy();
    }

public:
    /**
     * @brief 从训练态导入BN统计量并更新推理等效参数
     * @param d_trained_mean 训练收敛后的running mean (GPU指针, FP32)
     * @param d_trained_var  训练收敛后的running var (GPU指针, FP32)
     * @note 必须在每个Epoch训练完成、推理开始前调用
     * @note 采用Host端高效计算（K≤2048时<25μs），满足性能需求
     * @note 📌 Benchmark测试桩：当前传入随机初始化值，真实集成时应传入训练收敛后的统计量
     */
    void update_inference_params(void* d_trained_mean, void* d_trained_var) {
        const size_t stats_bytes = K * sizeof(float);

        // D2D异步拷贝（在推理流中排队，保证Graph兼容）
        CHECK_CUDA(cudaMemcpyAsync(d_prev_running_mean, d_trained_mean, stats_bytes,
                                   cudaMemcpyDeviceToDevice, stream_));
        CHECK_CUDA(cudaMemcpyAsync(d_prev_running_var, d_trained_var, stats_bytes,
                                   cudaMemcpyDeviceToDevice, stream_));

        // 流同步（低频操作，可接受）
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        // 复用现有Host端计算逻辑
        precompute_eq_params();
    }

    void capture_cuda_graphs() {
        if (!use_graph_ || graph_captured_) {
            return;
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        configure_static_l2_policies();
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeThreadLocal;
        cudaError_t capture_err = cudaStreamBeginCapture(stream_, capture_mode);
        if (capture_err != cudaSuccess) {
            std::cerr << "[Inference] Failed to begin capture: "
                      << cudaGetErrorString(capture_err) << std::endl;
            return;
        }

        CHECK_CUDNN_FE(main_graph_->execute(handle_, vp_main_, d_workspace_main_));
        if (!use_single_graph_) {
            CHECK_CUDNN_FE(fallback_graph_->execute(handle_, vp_fallback_, d_workspace_fallback_));
        }

        capture_err = cudaStreamEndCapture(stream_, &graph_inf_);
        if (capture_err != cudaSuccess) {
            std::cerr << "[Inference] Failed to end capture: "
                      << cudaGetErrorString(capture_err) << std::endl;
            return;
        }

        if (graph_inf_ == nullptr) {
            std::cerr << "[Inference] Capture returned null graph!" << std::endl;
            return;
        }

        cudaGraphNode_t error_node;
        char log_buffer[2048];
        cudaError_t inst_err = cudaGraphInstantiate(
            &graph_exec_inf_, graph_inf_, &error_node, log_buffer, sizeof(log_buffer));

        if (inst_err != cudaSuccess) {
            std::cerr << "[Inference] Instantiation failed: "
                      << cudaGetErrorString(inst_err) << std::endl;
            std::cerr << "[Inference] Error log: " << log_buffer << std::endl;
            cudaGraphDestroy(graph_inf_);
            graph_inf_ = nullptr;
            return;
        }

        graph_captured_ = true;
    }

    void execute_inference_traditional() {
        CHECK_CUDNN_FE(main_graph_->execute(handle_, vp_main_, d_workspace_main_));
        if (!use_single_graph_) {
            CHECK_CUDNN_FE(fallback_graph_->execute(handle_, vp_fallback_, d_workspace_fallback_));
        }
    }

    void execute_inference() {
        if (use_graph_ && graph_captured_ && graph_exec_inf_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_inf_, stream_));
        } else {
            execute_inference_traditional();
        }
    }

    void warmup(int iterations = 50) {
        // 📌 Benchmark测试桩：使用随机初始化的running stats进行参数预计算
        // 真实框架集成时，应从训练模块传递收敛后的统计量：
        //   update_inference_params(trained_mean, trained_var);
        update_inference_params(d_prev_running_mean, d_prev_running_var);

        for (int i = 0; i < iterations; ++i) {
            execute_inference_traditional();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        if (!use_graph_) {
            return;
        }

        try {
            capture_cuda_graphs();

            if (!graph_captured_) {
                std::cerr << "Warning: Graph capture failed, falling back to traditional mode" << std::endl;
                return;
            }

            for (int i = 0; i < iterations; ++i) {
                execute_inference();
            }
            CHECK_CUDA(cudaStreamSynchronize(stream_));

        } catch (const std::exception& e) {
            std::cerr << "\nException during graph capture: " << e.what() << std::endl;
            std::cerr << "Falling back to traditional execution mode" << std::endl;
            use_graph_ = false;
            graph_captured_ = false;
        }
    }

    void benchmark(int iterations = 500) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute_inference();
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_));

        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "\nTime (Unit: ms):" << std::endl;
        std::cout << "Total:    " << total_time << std::endl;
    }
};

int main(int argc, char** argv) {
    Config config = parse_arguments(argc, argv);

    try {
        ConvBNReLUInferenceBenchmark benchmark(config);
        benchmark.warmup();
        benchmark.benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
