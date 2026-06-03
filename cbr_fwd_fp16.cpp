/**
 * @file test_cbr_fp16_fwd_only.cpp
 * @brief cuDNN Frontend Conv+BN+ReLU前向融合性能测试V9.5 (FP16版本，仅前向，支持Mode C穷举式搜索)
 * @version 9.5.0
 * @date 2026-04-13
 * @author 技术觉醒团队
 * @note 依赖项: cuDNN Frontend 1.17, CUDA 13.1, cuDNN 9.17
 * @note 所属系列: cuda
 *
 * @note V9.5 改动（基于V9.4，引入ta_v4_common_fp16.hpp公共基础设施）：
 *   1. 【引入】ta_v4_common_fp16.hpp作为单一真理源
 *   2. 【删除】所有在ta_v4_common_fp16.hpp中已有的定义
 *   3. 【保留】前向传播核心实现（Conv+GenStats→BNFinalize→BN+ReLU）
 *   4. 【保留】三流架构和L2缓存优化策略
 *   5. 【保留】CUDA Graph模式支持
 *
 * @note 拓扑结构（Fork-Join前向拓扑）：
 *       前向拓扑：
 *         S1 (Begin) ──┬─→ [Conv+GenStats] ──event1──┬─→ (Wait event_fwd_join) ──→ S1 (End)
 *                      │                               │
 *                      └─→ S2 ──[BNFinalize] ──event2──┴─→ S3 ──[BN Apply+ReLU] ──event_fwd_join──┘
 *
 * @note 关键优化:
 *       - CUDA Graph消除CPU Launch开销（约20-50μs/次）
 *       - 静态Variant Packs避免rehash和堆分配
 *       - 严格拓扑闭环确保Graph捕获成功
 *       - L2策略优化缓存一致性
 *       - 预期性能提升：1.2-2.0%
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

class ConvGenStatsBNFinalizePointwiseBenchmark {
private:
    // ========== V6.1原有成员（完全保留） ==========
    cudnnHandle_t handle_comp_1_, handle_comp_2_, handle_comp_3_;
    cudaStream_t stream_comp_1_, stream_comp_2_, stream_comp_3_;
    cudaEvent_t event_comp_1_done_, event_comp_2_done_;

    const int64_t N, C, H_in, W_in, K;
    const int64_t R, S;
    const int64_t padding;
    const int64_t stride;
    const int64_t H_out, W_out;
    const float EPSILON = 1e-5f;
    const float MOMENTUM = 0.1f;

    // ████ 修改：保存搜索模式 ████
    const Config::SearchMode search_mode_;
    const fe::HeurMode_t heur_mode_;
    // ████████████████████████████████████████████████████████████████████████

    // GPU memory pointers (前向)
    void *d_input, *d_weight, *d_output;
    void *d_sum, *d_sq_sum;
    void *d_scale, *d_bias;
    void *d_prev_running_mean, *d_prev_running_var;
    void *d_eq_scale, *d_eq_bias;
    void *d_mean, *d_inv_variance;
    void *d_next_running_mean, *d_next_running_var;
    void *d_bn_relu_output;
    void *d_relu_bitmask;

    // Workspace
    void *d_workspace_comp_1_, *d_workspace_comp_2_, *d_workspace_comp_3_;
    size_t workspace_size_comp_1_fwd_ = 0;
    size_t workspace_size_comp_2_fwd_ = 0;
    size_t workspace_size_comp_3_fwd_ = 0;

    // cuDNN Frontend Graphs
    std::shared_ptr<fe::graph::Graph> conv_genstats_graph;
    std::shared_ptr<fe::graph::Graph> bn_finalize_graph;
    std::shared_ptr<fe::graph::Graph> bn_relu_graph;

    // Tensor attributes
    std::shared_ptr<fe::graph::Tensor_attributes> conv_input_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_weight_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> conv_output_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> sum_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> sq_sum_tensor;

    std::shared_ptr<fe::graph::Tensor_attributes> bn_finalize_scale_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_finalize_bias_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_sum_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_sq_sum_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> epsilon_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> accum_count_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> prev_running_mean_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> prev_running_var_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> momentum_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> eq_scale_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> eq_bias_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> finalize_mean_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> finalize_inv_var_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> next_running_mean_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> next_running_var_tensor;

    std::shared_ptr<fe::graph::Tensor_attributes> bn_relu_x_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_relu_scale_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_relu_bias_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_relu_y_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_relu_bitmask_tensor;

    // ========== V8.0新增：CUDA Graph资源 ==========
    cudaGraph_t graph_fwd_ = nullptr;
    cudaGraphExec_t graph_exec_fwd_ = nullptr;
    bool use_graph_ = false;
    bool graph_captured_ = false;

    // V8.0：Fork-Join汇聚事件（预分配，捕获时不能动态创建）
    cudaEvent_t event_fwd_join_;       // 前向：S3→S1汇聚

    // V8.0：静态Variant Packs（预留容量，避免rehash）
    using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;
    VariantPack vp_conv_genstats_;
    VariantPack vp_bn_finalize_;
    VariantPack vp_bn_relu_;

    // ████████████████████████████████████████████████████████████████████████████
    // V9.2 新增：Experience和Engine信息记录（用于统一打印）
    // ████████████████████████████████████████████████████████████████████████████
    std::vector<std::pair<std::string, ExperienceStatus>> experience_records_;
    std::vector<std::pair<std::string, std::string>> engine_records_;

public:
    // ████████████████████████████████████████████████████████████████████████
    // ████ 修改：构造函数，保存search_mode ████
    // ████████████████████████████████████████████████████████████████████████
    ConvGenStatsBNFinalizePointwiseBenchmark(const Config& config)
        : N(config.batch_size), C(config.in_channels),
          H_in(config.input_size), W_in(config.input_size),
          K(config.out_channels), R(config.kernel_size), S(config.kernel_size),
          padding(config.get_padding()), stride(config.conv_stride),
          H_out(config.get_output_size()), W_out(config.get_output_size()),
          search_mode_(config.search_mode),
          heur_mode_(config.get_heur_mode()),
          use_graph_(config.use_graph) {  // ████ V9.0新增：从config读取Graph模式开关 ████
    // ████████████████████████████████████████████████████████████████████████
    // ████████████████████████████████████████████████████████████████████████

        // ========== V6.1原有初始化 ==========
        CHECK_CUDNN(cudnnCreate(&handle_comp_1_));
        CHECK_CUDNN(cudnnCreate(&handle_comp_2_));
        CHECK_CUDNN(cudnnCreate(&handle_comp_3_));

        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_2_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_3_, cudaStreamNonBlocking));

        CHECK_CUDNN(cudnnSetStream(handle_comp_1_, stream_comp_1_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_2_, stream_comp_2_));
        CHECK_CUDNN(cudnnSetStream(handle_comp_3_, stream_comp_3_));

        CHECK_CUDA(cudaEventCreateWithFlags(&event_comp_1_done_, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_comp_2_done_, cudaEventDisableTiming));

        // ========== V8.0：预创建汇聚事件 ==========
        // 修复：仅在使用CUDA Graph模式时创建event_fwd_join_，避免传统模式下的资源浪费
        if (use_graph_) {
            CHECK_CUDA(cudaEventCreateWithFlags(&event_fwd_join_, cudaEventDisableTiming));
        } else {
            event_fwd_join_ = nullptr;
        }

        allocate_memory();
        initialize_data();
        build_conv_genstats_graph();
        build_bn_finalize_graph();
        build_bn_relu_graph();

        // ████████████████████████████████████████████████████████████████████████████
        // V9.2 新增：统一打印Experience和Engine信息
        // ████████████████████████████████████████████████████████████████████████████
        std::cout << "Conv Forward Type: Conv+GenStats" << std::endl;

        // 打印Experience信息（仅Mode C）
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

        // 打印Engine信息
        for (const auto& [op_name, engine_name] : engine_records_) {
            std::cout << op_name << " Engine: " << engine_name << std::endl;
        }

        allocate_workspace();

        // ========== V8.0：构建静态Variant Packs ==========
        build_static_variant_packs();

        // ████████████████████████████████████████████████████████████████████████
        // ████ 修复：传统模式需要配置静态L2策略 ████
        // ████████████████████████████████████████████████████████████████████████
        if (!use_graph_) {
            configure_static_l2_policies();  // 传统模式：静态L2配置（仅前向优化）
        }
        // ████████████████████████████████████████████████████████████████████████
    }

    ~ConvGenStatsBNFinalizePointwiseBenchmark() {
        // V8.0：先销毁Graph资源（在销毁流/事件之前）
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

        // V8.0：销毁汇聚事件
        // 修复：仅在使用CUDA Graph模式时销毁event_fwd_join_
        if (event_fwd_join_) {
            CHECK_CUDA(cudaEventDestroy(event_fwd_join_));
        }

        // V6.1原有的资源销毁
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_weight));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaFree(d_sum));
        CHECK_CUDA(cudaFree(d_sq_sum));
        CHECK_CUDA(cudaFree(d_scale));
        CHECK_CUDA(cudaFree(d_bias));
        CHECK_CUDA(cudaFree(d_prev_running_mean));
        CHECK_CUDA(cudaFree(d_prev_running_var));
        CHECK_CUDA(cudaFree(d_eq_scale));
        CHECK_CUDA(cudaFree(d_eq_bias));
        CHECK_CUDA(cudaFree(d_mean));
        CHECK_CUDA(cudaFree(d_inv_variance));
        CHECK_CUDA(cudaFree(d_next_running_mean));
        CHECK_CUDA(cudaFree(d_next_running_var));
        CHECK_CUDA(cudaFree(d_bn_relu_output));
        CHECK_CUDA(cudaFree(d_relu_bitmask));

        if (d_workspace_comp_1_) CHECK_CUDA(cudaFree(d_workspace_comp_1_));
        if (d_workspace_comp_2_) CHECK_CUDA(cudaFree(d_workspace_comp_2_));
        if (d_workspace_comp_3_) CHECK_CUDA(cudaFree(d_workspace_comp_3_));

        CHECK_CUDNN(cudnnDestroy(handle_comp_3_));
        CHECK_CUDNN(cudnnDestroy(handle_comp_2_));
        CHECK_CUDNN(cudnnDestroy(handle_comp_1_));

        CHECK_CUDA(cudaEventDestroy(event_comp_2_done_));
        CHECK_CUDA(cudaEventDestroy(event_comp_1_done_));

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

        d_input = allocate_aligned_gpu_memory(input_size);
        d_weight = allocate_aligned_gpu_memory(weight_size);
        d_output = allocate_aligned_gpu_memory(output_size);
        d_bn_relu_output = allocate_aligned_gpu_memory(output_size);

        d_sum = allocate_aligned_gpu_memory(stats_size);
        d_sq_sum = allocate_aligned_gpu_memory(stats_size);

        d_scale = allocate_aligned_gpu_memory(stats_size);
        d_bias = allocate_aligned_gpu_memory(stats_size);

        d_prev_running_mean = allocate_aligned_gpu_memory(stats_size);
        d_prev_running_var = allocate_aligned_gpu_memory(stats_size);

        d_eq_scale = allocate_aligned_gpu_memory(stats_size);
        d_eq_bias = allocate_aligned_gpu_memory(stats_size);
        d_mean = allocate_aligned_gpu_memory(stats_size);
        d_inv_variance = allocate_aligned_gpu_memory(stats_size);
        d_next_running_mean = allocate_aligned_gpu_memory(stats_size);
        d_next_running_var = allocate_aligned_gpu_memory(stats_size);

        d_relu_bitmask = allocate_aligned_gpu_memory(bitmask_size);
    }

    void allocate_workspace() {
        d_workspace_comp_1_ = (workspace_size_comp_1_fwd_ > 0) ? allocate_aligned_gpu_memory(workspace_size_comp_1_fwd_) : nullptr;
        d_workspace_comp_2_ = (workspace_size_comp_2_fwd_ > 0) ? allocate_aligned_gpu_memory(workspace_size_comp_2_fwd_) : nullptr;
        d_workspace_comp_3_ = (workspace_size_comp_3_fwd_ > 0) ? allocate_aligned_gpu_memory(workspace_size_comp_3_fwd_) : nullptr;
    }

    void initialize_data() {
        size_t input_elements = N * C * H_in * W_in;
        size_t weight_elements = K * C * R * S;
        size_t stats_elements = K;

        initialize_random_fp16(d_input, input_elements, 42);
        initialize_random_fp16(d_weight, weight_elements, 43);

        initialize_random_fp32(d_scale, stats_elements, 44);
        initialize_random_fp32(d_bias, stats_elements, 45);

        std::vector<float> zeros(stats_elements, 0.0f);
        CHECK_CUDA(cudaMemcpy(d_prev_running_mean, zeros.data(), stats_elements * sizeof(float), cudaMemcpyHostToDevice));
        // 修复：running_var初始化为[0.8, 1.2]区间，避免极端的inv_std
        initialize_positive_fp32(d_prev_running_var, stats_elements, 0.8f, 1.2f, 999);
    }

    /**
     * @brief 构建静态Variant Packs（真正的零动态分配）
     * @note 预留容量避免rehash，确保捕获期间无任何堆操作
     */
    void build_static_variant_packs() {
        // ========== 预留容量（关键！避免rehash触发堆分配） ==========
        vp_conv_genstats_.reserve(10);
        vp_bn_finalize_.reserve(20);
        vp_bn_relu_.reserve(10);

        // ========== 前向图1: Conv+GenStats ==========
        vp_conv_genstats_[conv_input_tensor] = d_input;
        vp_conv_genstats_[conv_weight_tensor] = d_weight;
        vp_conv_genstats_[conv_output_tensor] = d_output;
        vp_conv_genstats_[sum_tensor] = d_sum;
        vp_conv_genstats_[sq_sum_tensor] = d_sq_sum;

        // ========== 前向图2: BNFinalize ==========
        vp_bn_finalize_[bn_sum_tensor] = d_sum;
        vp_bn_finalize_[bn_sq_sum_tensor] = d_sq_sum;
        vp_bn_finalize_[bn_finalize_scale_tensor] = d_scale;
        vp_bn_finalize_[bn_finalize_bias_tensor] = d_bias;
        vp_bn_finalize_[prev_running_mean_tensor] = d_prev_running_mean;
        vp_bn_finalize_[prev_running_var_tensor] = d_prev_running_var;
        vp_bn_finalize_[eq_scale_tensor] = d_eq_scale;
        vp_bn_finalize_[eq_bias_tensor] = d_eq_bias;
        vp_bn_finalize_[finalize_mean_tensor] = d_mean;
        vp_bn_finalize_[finalize_inv_var_tensor] = d_inv_variance;
        vp_bn_finalize_[next_running_mean_tensor] = d_next_running_mean;
        vp_bn_finalize_[next_running_var_tensor] = d_next_running_var;

        // ========== 前向图3: BN Apply+ReLU ==========
        vp_bn_relu_[bn_relu_x_tensor] = d_output;
        vp_bn_relu_[bn_relu_scale_tensor] = d_eq_scale;
        vp_bn_relu_[bn_relu_bias_tensor] = d_eq_bias;
        vp_bn_relu_[bn_relu_y_tensor] = d_bn_relu_output;
        vp_bn_relu_[bn_relu_bitmask_tensor] = d_relu_bitmask;
    }

    void build_conv_genstats_graph() {
        conv_genstats_graph = std::make_shared<fe::graph::Graph>();
        conv_genstats_graph->set_io_data_type(fe::DataType_t::HALF);
        conv_genstats_graph->set_intermediate_data_type(fe::DataType_t::FLOAT);
        conv_genstats_graph->set_compute_data_type(fe::DataType_t::FLOAT);

        conv_input_tensor = conv_genstats_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("input")
            .set_dim({N, C, H_in, W_in})
            .set_stride({int64_t(H_in)*W_in*C, int64_t(1), int64_t(W_in)*C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        conv_weight_tensor = conv_genstats_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("weight")
            .set_dim({K, C, R, S})
            .set_stride({int64_t(R)*S*C, int64_t(1), int64_t(S)*C, int64_t(C)})
            .set_data_type(fe::DataType_t::HALF));

        fe::graph::Conv_fprop_attributes conv_options;
        conv_options.set_padding({padding, padding})
                    .set_stride({stride, stride})
                    .set_dilation({1, 1});

        auto conv_output = conv_genstats_graph->conv_fprop(conv_input_tensor, conv_weight_tensor, conv_options);

        conv_output_tensor = conv_output;
        conv_output_tensor->set_output(true)
                    .set_dim({N, K, H_out, W_out})
                    .set_stride({int64_t(H_out)*W_out*K, int64_t(1), int64_t(W_out)*K, int64_t(K)})
                    .set_data_type(fe::DataType_t::HALF);

        fe::graph::Genstats_attributes genstats_attrs;
        genstats_attrs.set_name("genstats")
                     .set_compute_data_type(fe::DataType_t::FLOAT);

        auto genstats_outputs = conv_genstats_graph->genstats(conv_output, genstats_attrs);
        sum_tensor = genstats_outputs[0];
        sq_sum_tensor = genstats_outputs[1];

        sum_tensor->set_output(true)
                 .set_dim({1, K, 1, 1})
                 .set_stride({K, 1, K, K})
                 .set_data_type(fe::DataType_t::FLOAT);

        sq_sum_tensor->set_output(true)
                   .set_dim({1, K, 1, 1})
                   .set_stride({K, 1, K, K})
                   .set_data_type(fe::DataType_t::FLOAT);

        CHECK_CUDNN_FE(conv_genstats_graph->validate());
        CHECK_CUDNN_FE(conv_genstats_graph->build_operation_graph(handle_comp_1_));

        // ========== 新增：Mode C分支 ==========
        if (search_mode_ == Config::SearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
            // 1. 生成Shape Key
            std::string key = build_shape_key(
                "conv_genstats", "fp16",
                N, H_in, W_in, C, K, R, S, stride, padding
            );

            // 2. 查询Experience表
            auto exp_rec = ta_v4::experience::lookup(key);

            if (exp_rec != nullptr) {
                // 3. 获取候选Plan
                CHECK_CUDNN_FE(conv_genstats_graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}));
                std::vector<int64_t> candidates;
                int64_t count = conv_genstats_graph->get_execution_plan_count();
                for (int64_t i = 0; i < count; ++i) {
                    candidates.push_back(i);
                }

                // 4. 三级Fallback Plan匹配
                auto [exp_status, matched] = match_and_build_plan(
                    conv_genstats_graph, candidates, exp_rec, handle_comp_1_
                );
                experience_records_.push_back({"Forward", exp_status});

                if (!matched) {
                    goto fallback_heuristic_conv_genstats;
                }
            } else {
                goto fallback_heuristic_conv_genstats;
            }
#else
            // 非A100/5090平台，直接fallback
            goto fallback_heuristic_conv_genstats;
#endif
        } else {
            // Mode A/B走原有逻辑
        fallback_heuristic_conv_genstats:
            CHECK_CUDNN_FE(conv_genstats_graph->create_execution_plans({heur_mode_}));
            CHECK_CUDNN_FE(conv_genstats_graph->check_support(handle_comp_1_));
            CHECK_CUDNN_FE(conv_genstats_graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        }
        // ========== Mode C分支结束 ==========

        std::string engine_name;
        CHECK_CUDNN_FE(conv_genstats_graph->get_plan_name(engine_name));
        engine_records_.push_back({"Forward", engine_name});

        workspace_size_comp_1_fwd_ = conv_genstats_graph->get_workspace_size();
    }
    // ████████████████████████████████████████████████████████████████████████████

    void build_bn_finalize_graph() {
        bn_finalize_graph = std::make_shared<fe::graph::Graph>();
        bn_finalize_graph->set_io_data_type(fe::DataType_t::FLOAT);
        bn_finalize_graph->set_intermediate_data_type(fe::DataType_t::FLOAT);
        bn_finalize_graph->set_compute_data_type(fe::DataType_t::FLOAT);

        bn_finalize_scale_tensor = bn_finalize_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("scale")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        bn_finalize_bias_tensor = bn_finalize_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bias")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        bn_sum_tensor = bn_finalize_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("sum")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        bn_sq_sum_tensor = bn_finalize_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("sq_sum")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        float epsilon_val = EPSILON;
        float momentum_val = MOMENTUM;
        // BN Finalize需要实际参与累加的元素总数：N × H_out × W_out
        int64_t accum_count_val = N * H_out * W_out;

        epsilon_tensor = bn_finalize_graph->tensor(epsilon_val);
        accum_count_tensor = bn_finalize_graph->tensor(accum_count_val);
        momentum_tensor = bn_finalize_graph->tensor(momentum_val);

        prev_running_mean_tensor = bn_finalize_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("prev_running_mean")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        prev_running_var_tensor = bn_finalize_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("prev_running_var")
            .set_dim({1, K, 1, 1})
            .set_stride({K, 1, K, K})
            .set_data_type(fe::DataType_t::FLOAT));

        fe::graph::BN_finalize_attributes bn_finalize_attrs;
        bn_finalize_attrs.set_name("bn_finalize")
                        .set_compute_data_type(fe::DataType_t::FLOAT)
                        .set_previous_running_stats(prev_running_mean_tensor,
                                                     prev_running_var_tensor,
                                                     momentum_tensor);

        auto bn_finalize_outputs = bn_finalize_graph->bn_finalize(
            bn_sum_tensor,
            bn_sq_sum_tensor,
            bn_finalize_scale_tensor,
            bn_finalize_bias_tensor,
            epsilon_tensor,
            accum_count_tensor,
            bn_finalize_attrs
        );

        eq_scale_tensor = bn_finalize_outputs[0];
        eq_bias_tensor = bn_finalize_outputs[1];
        finalize_mean_tensor = bn_finalize_outputs[2];
        finalize_inv_var_tensor = bn_finalize_outputs[3];
        next_running_mean_tensor = bn_finalize_outputs[4];
        next_running_var_tensor = bn_finalize_outputs[5];

        eq_scale_tensor->set_output(true)
                       .set_dim({1, K, 1, 1})
                       .set_stride({K, 1, K, K})
                       .set_data_type(fe::DataType_t::FLOAT);

        eq_bias_tensor->set_output(true)
                     .set_dim({1, K, 1, 1})
                     .set_stride({K, 1, K, K})
                     .set_data_type(fe::DataType_t::FLOAT);

        finalize_mean_tensor->set_output(true)
                  .set_dim({1, K, 1, 1})
                  .set_stride({K, 1, K, K})
                  .set_data_type(fe::DataType_t::FLOAT);

        finalize_inv_var_tensor->set_output(true)
                          .set_dim({1, K, 1, 1})
                          .set_stride({K, 1, K, K})
                          .set_data_type(fe::DataType_t::FLOAT);

        next_running_mean_tensor->set_output(true)
                               .set_dim({1, K, 1, 1})
                               .set_stride({K, 1, K, K})
                               .set_data_type(fe::DataType_t::FLOAT);

        next_running_var_tensor->set_output(true)
                             .set_dim({1, K, 1, 1})
                             .set_stride({K, 1, K, K})
                             .set_data_type(fe::DataType_t::FLOAT);

        CHECK_CUDNN_FE(bn_finalize_graph->validate());
        CHECK_CUDNN_FE(bn_finalize_graph->build_operation_graph(handle_comp_2_));
        // BN层固定使用Heuristic Mode B，不受search_mode参数影响
        CHECK_CUDNN_FE(bn_finalize_graph->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(bn_finalize_graph->check_support(handle_comp_2_));
        CHECK_CUDNN_FE(bn_finalize_graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        std::string engine_name;
        CHECK_CUDNN_FE(bn_finalize_graph->get_plan_name(engine_name));

        workspace_size_comp_2_fwd_ = bn_finalize_graph->get_workspace_size();
    }

    void build_bn_relu_graph() {
        bn_relu_graph = std::make_shared<fe::graph::Graph>();
        bn_relu_graph->set_io_data_type(fe::DataType_t::HALF)
                   .set_intermediate_data_type(fe::DataType_t::HALF)
                   .set_compute_data_type(fe::DataType_t::FLOAT);

        std::vector<int64_t> x_dim = {N, K, H_out, W_out};
        std::vector<int64_t> x_stride = {int64_t(H_out*W_out*K), int64_t(1), int64_t(W_out)*K, int64_t(K)};

        bn_relu_x_tensor = bn_relu_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bn_relu_x")
            .set_dim(x_dim)
            .set_stride(x_stride)
            .set_data_type(fe::DataType_t::HALF));

        std::vector<int64_t> param_dim = {1, K, 1, 1};
        std::vector<int64_t> param_stride = {K, 1, K, K};

        bn_relu_scale_tensor = bn_relu_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("eq_scale")
            .set_dim(param_dim)
            .set_stride(param_stride)
            .set_data_type(fe::DataType_t::FLOAT));

        bn_relu_bias_tensor = bn_relu_graph->tensor(fe::graph::Tensor_attributes()
            .set_name("eq_bias")
            .set_dim(param_dim)
            .set_stride(param_stride)
            .set_data_type(fe::DataType_t::FLOAT));

        auto mul_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto scaled = bn_relu_graph->pointwise(bn_relu_x_tensor, bn_relu_scale_tensor, mul_options);

        auto add_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto bn_applied = bn_relu_graph->pointwise(scaled, bn_relu_bias_tensor, add_options);

        auto relu_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto relu_output = bn_relu_graph->pointwise(bn_applied, relu_options);
        bn_relu_y_tensor = relu_output;

        float zero = 0.0f;
        auto zero_tensor = bn_relu_graph->tensor(zero);

        auto cmp_gt_options = fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::CMP_GT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto bitmask = bn_relu_graph->pointwise(bn_applied, zero_tensor, cmp_gt_options);
        bn_relu_bitmask_tensor = bitmask;

        bn_relu_y_tensor->set_output(true);
        bn_relu_y_tensor->set_data_type(fe::DataType_t::HALF);

        bn_relu_bitmask_tensor->set_output(true);
        bn_relu_bitmask_tensor->set_data_type(fe::DataType_t::BOOLEAN);

        CHECK_CUDNN_FE(bn_relu_graph->validate());
        CHECK_CUDNN_FE(bn_relu_graph->build_operation_graph(handle_comp_3_));
        // BN层固定使用Heuristic Mode B，不受search_mode参数影响
        CHECK_CUDNN_FE(bn_relu_graph->create_execution_plans({fe::HeurMode_t::B}));
        CHECK_CUDNN_FE(bn_relu_graph->check_support(handle_comp_3_));
        CHECK_CUDNN_FE(bn_relu_graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        std::string engine_name;
        CHECK_CUDNN_FE(bn_relu_graph->get_plan_name(engine_name));

        workspace_size_comp_3_fwd_ = bn_relu_graph->get_workspace_size();
    }

    /**
     * @brief 静态配置L2缓存策略（仅在捕获前调用一次）
     * @note 核心策略：
     *   1. 前向专属：仅优化Forward（stream_comp_3_读取d_output）
     *   2. 智能门控：仅当张量<=10MB时启用Persist
     *   3. 显式初始化：所有字段显式赋值，杜绝未定义行为
     *
     * @note 预期效果：
     *   - 小张量(7x7/14x14)：保留+2-3%收益
     *   - 大张量(56x56)：消除-49%劣化 -> 提升+5-8%
     *   - Graph版本：消除热路径API污染，净收益+5-10%
     */
    /**
     * @brief 重置所有流为 Streaming 模式（清除 L2 Persist 策略）
     * @note 用于 Graph 捕获前的状态重置，确保每次捕获从干净状态开始
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
     * @brief 配置前向 L2 优化策略（保护 d_output）
     * @note 仅在 Graph 捕获前向阶段调用
     *   - 保护对象：d_output（BN Apply+ReLU 在 stream_comp_3_ 上读取）
     *   - 保护流：stream_comp_3_
     *   - 智能门控：10MB 阈值
     */
    void configure_forward_l2_policies() {
        const size_t fwd_tensor_size =
            static_cast<size_t>(N) * K * H_out * W_out * sizeof(__half);
        constexpr size_t L2_SAFE_THRESHOLD = 10ULL * 1024ULL * 1024ULL;  // 10MB

        reset_all_l2_policies();  // 先重置

        if (fwd_tensor_size <= L2_SAFE_THRESHOLD) {
            cudaStreamAttrValue attr_persist = {};
            attr_persist.accessPolicyWindow.base_ptr = d_output;
            attr_persist.accessPolicyWindow.num_bytes = fwd_tensor_size;
            attr_persist.accessPolicyWindow.hitRatio = 1.0f;
            attr_persist.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr_persist.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            CHECK_CUDA(cudaStreamSetAttribute(stream_comp_3_,
                cudaStreamAttributeAccessPolicyWindow, &attr_persist));
        }
    }

    /**
     * @brief 静态配置L2缓存策略（仅在捕获前调用一次）
     * @note 核心策略：
     *   1. 前向专属：仅优化Forward（stream_comp_3_读取d_output）
     *   2. 智能门控：仅当张量<=10MB时启用Persist
     *   3. 显式初始化：所有字段显式赋值，杜绝未定义行为
     *
     * @note 预期效果：
     *   - 小张量(7x7/14x14)：保留+2-3%收益
     *   - 大张量(56x56)：消除-49%劣化 -> 提升+5-8%
     *   - Graph版本：消除热路径API污染，净收益+5-10%
     *
     * @note FP16架构特殊性：
     *   - 前向：三流架构（S1:Conv+GenStats → S2:BNFinalize → S3:BN Apply+ReLU）
     *   - 前向数据流：d_output在stream_comp_3_被读取
     */
    void configure_static_l2_policies() {
        const size_t fwd_tensor_size =
            static_cast<size_t>(N) * K * H_out * W_out * sizeof(__half);
        constexpr size_t L2_SAFE_THRESHOLD = 10ULL * 1024ULL * 1024ULL;  // 10MB智能门控阈值

        // Step 1: 所有流重置为Streaming（显式初始化所有字段）
        reset_all_l2_policies();

        // Step 2: 智能门控 - 条件启用Forward Persist（仅小张量）
        if (fwd_tensor_size <= L2_SAFE_THRESHOLD) {
            cudaStreamAttrValue attr_persist = {};
            attr_persist.accessPolicyWindow.base_ptr = d_output;
            attr_persist.accessPolicyWindow.num_bytes = fwd_tensor_size;
            attr_persist.accessPolicyWindow.hitRatio = 1.0f;
            attr_persist.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr_persist.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            CHECK_CUDA(cudaStreamSetAttribute(stream_comp_3_,
                cudaStreamAttributeAccessPolicyWindow, &attr_persist));
        }
    }

public:
    /**
     * @brief 捕获CUDA Graph（仅前向，完美拓扑闭环）
     * @note 零动态分配，完善的错误检测和日志
     */
    void capture_cuda_graphs() {
        // ========== V9.2 修复：检查是否启用CUDA Graph模式 ==========
        if (!use_graph_) {
            return;
        }
        // ====================================================================

        if (graph_captured_) {
            return;
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        // ==================== 在捕获前配置L2策略，固化Stream状态 ====================
        configure_static_l2_policies();
        CHECK_CUDA(cudaDeviceSynchronize());

        // ==================== 捕获前向Graph ====================

        // 使用ThreadLocal模式，避免捕获到其他线程的操作（更安全）
        cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeThreadLocal;
        cudaError_t capture_err = cudaStreamBeginCapture(stream_comp_1_, capture_mode);
        if (capture_err != cudaSuccess) {
            std::cerr << "[Forward] Failed to begin capture: "
                      << cudaGetErrorString(capture_err) << std::endl;
            return;
        }

        // === S1: Conv+GenStats ===
        CHECK_CUDNN_FE(conv_genstats_graph->execute(handle_comp_1_, vp_conv_genstats_, d_workspace_comp_1_));
        CHECK_CUDA(cudaEventRecord(event_comp_1_done_, stream_comp_1_));
        // === S2: BNFinalize（等待S1） ===
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_comp_1_done_, 0));
        CHECK_CUDNN_FE(bn_finalize_graph->execute(handle_comp_2_, vp_bn_finalize_, d_workspace_comp_2_));
        CHECK_CUDA(cudaEventRecord(event_comp_2_done_, stream_comp_2_));

        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_comp_2_done_, 0));
        CHECK_CUDNN_FE(bn_relu_graph->execute(handle_comp_3_, vp_bn_relu_, d_workspace_comp_3_));
        CHECK_CUDA(cudaEventRecord(event_fwd_join_, stream_comp_3_));
        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_fwd_join_, 0));

        // ████████████████████████████████████████████████████████████████████████████
        // ████ 修复：将BN Running Stats同步直接捕获进Graph（零CPU开销）          ████
        // ████ PLAN.md问题2：Graph模式下需要在迭代间同步running statistics        ████
        // ████████████████████████████████████████████████████████████████████████████
        const size_t stats_bytes = K * sizeof(float);
        CHECK_CUDA(cudaMemcpyAsync(
            d_prev_running_mean, d_next_running_mean,
            stats_bytes, cudaMemcpyDeviceToDevice, stream_comp_1_));
        CHECK_CUDA(cudaMemcpyAsync(
            d_prev_running_var, d_next_running_var,
            stats_bytes, cudaMemcpyDeviceToDevice, stream_comp_1_));
        // ████████████████████████████████████████████████████████████████████████████

        // === 从S1结束捕获（拓扑闭环完成） ===
        capture_err = cudaStreamEndCapture(stream_comp_1_, &graph_fwd_);
        if (capture_err != cudaSuccess) {
            std::cerr << "[Forward] Failed to end capture: "
                      << cudaGetErrorString(capture_err) << std::endl;
            if (capture_err == cudaErrorStreamCaptureUnjoined) {
                std::cerr << "[Forward] Unjoined stream detected! Check topology." << std::endl;
            }
            return;
        }

        if (graph_fwd_ == nullptr) {
            std::cerr << "[Forward] Capture returned null graph!" << std::endl;
            return;
        }

        size_t fwd_nodes = 0;
        CHECK_CUDA(cudaGraphGetNodes(graph_fwd_, nullptr, &fwd_nodes));

        // === 实例化Graph ===
        cudaGraphNode_t error_node;
        char log_buffer[2048];
        cudaError_t inst_err = cudaGraphInstantiate(
            &graph_exec_fwd_, graph_fwd_, &error_node, log_buffer, sizeof(log_buffer));

        if (inst_err != cudaSuccess) {
            std::cerr << "[Forward] Instantiation failed: "
                      << cudaGetErrorString(inst_err) << std::endl;
            std::cerr << "[Forward] Error log: " << log_buffer << std::endl;
            cudaGraphDestroy(graph_fwd_);
            graph_fwd_ = nullptr;
            return;
        }

        // ==================== 标记捕获完成 ====================
        graph_captured_ = true;
        use_graph_ = true;
    }

    /**
     * @brief 原始前向传播（保留作为Fallback和Warmup）
     * @note L2策略已在捕获前静态配置，这里不再动态设置
     */
    void execute_fprop_traditional() {
        CHECK_CUDNN_FE(conv_genstats_graph->execute(handle_comp_1_, vp_conv_genstats_, d_workspace_comp_1_));
        CHECK_CUDA(cudaEventRecord(event_comp_1_done_, stream_comp_1_));

        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_2_, event_comp_1_done_, 0));
        CHECK_CUDNN_FE(bn_finalize_graph->execute(handle_comp_2_, vp_bn_finalize_, d_workspace_comp_2_));
        CHECK_CUDA(cudaEventRecord(event_comp_2_done_, stream_comp_2_));

        CHECK_CUDA(cudaStreamWaitEvent(stream_comp_3_, event_comp_2_done_, 0));
        CHECK_CUDNN_FE(bn_relu_graph->execute(handle_comp_3_, vp_bn_relu_, d_workspace_comp_3_));
    }

    /**
     * @brief 执行前向传播（自动选择Graph或原始路径）
     * @note L2策略已在捕获前静态配置，热路径零API调用
     */
    void execute_fprop() {
        if (use_graph_ && graph_captured_ && graph_exec_fwd_) {
            // Graph模式：直接Launch，Stream属性已在捕获前固化
            CHECK_CUDA(cudaGraphLaunch(graph_exec_fwd_, stream_comp_1_));
        } else {
            // Native模式：使用traditional路径（L2已在静态配置中设置）
            execute_fprop_traditional();
        }
    }

    /**
     * @brief 同步BN Running Statistics（专家一致推荐方案）
     * @note 核心原理：
     *   1. 保持Graph地址绑定不变（d_prev和d_next固定）
     *   2. 每轮迭代间用D2D拷贝同步next→prev
     *   3. ResNet-50最大通道2048 → stats仅16KB
     *   4. A100上D2D异步拷贝耗时<0.5μs，完全被Graph Launch掩盖
     *
     * @note 修复致命缺陷：否则running statistics永远停留在初始值0，模型无法收敛
     */
    void sync_running_stats_async() {
        const size_t stats_bytes = K * sizeof(float);

        // ████████████████████████████████████████████████████████████████████████████
        // ████ FIX2.md P0-2修复：传统模式下强制等待S2完成写入                          ████
        // ████ 理由：消除非Graph模式的读写竞态，确保d_next已被BN Finalize完整更新       ████
        // ████ Graph模式不受影响：依赖关系已在capture时固化                          ████
        // ████████████████████████████████████████████████████████████████████████████
        if (!use_graph_) {
            CHECK_CUDA(cudaStreamWaitEvent(stream_comp_1_, event_comp_2_done_, 0));
        }

        // 在stream_comp_1_上异步执行，确保与前向图串行依赖
        CHECK_CUDA(cudaMemcpyAsync(
            d_prev_running_mean,  // 目标：下一轮的输入
            d_next_running_mean,  // 源：本轮的输出
            stats_bytes,
            cudaMemcpyDeviceToDevice,
            stream_comp_1_
        ));

        CHECK_CUDA(cudaMemcpyAsync(
            d_prev_running_var,
            d_next_running_var,
            stats_bytes,
            cudaMemcpyDeviceToDevice,
            stream_comp_1_
        ));
    }

    /**
     * @brief Warmup流程（三阶段：传统预热→Graph捕获→Graph预热）
     * @note 默认50次迭代（EXP.md修复4：专家共识度7/8）
     * @note 50次warmup + 锁频 = 最优方案
     */
    void warmup(int iterations = 50) {
        // ========== 阶段1：传统执行预热 ==========
        for (int i = 0; i < iterations; ++i) {
            execute_fprop_traditional();
            sync_running_stats_async();  // 修复：同步BN running stats
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));

        // ========== 阶段2：Graph捕获 ==========
        // ========== V9.2 修复：区分用户选择传统模式 vs 捕获失败 ==========
        if (!use_graph_) {
            // 用户主动选择传统模式，直接跳过Graph捕获
            return;
        }
        // ========================================================================

        try {
            capture_cuda_graphs();

            if (!graph_captured_) {
                std::cerr << "Warning: Graph capture failed, falling back to traditional mode" << std::endl;
                return;
            }

            // ========== 阶段3：Graph模式预热 ==========
            for (int i = 0; i < iterations; ++i) {
                execute_fprop();
                sync_running_stats_async();  // 修复：同步BN running stats
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

        // ========== 前向计时 ==========
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute_fprop();
            // ████████████████████████████████████████████████████████████████████████████
            // ████ 修复：非Graph模式需要显式同步BN Running Stats                        ████
            // ████ PLAN.md问题2：传统模式下需要在迭代间手动同步running statistics      ████
            // ████ Graph模式已在capture_cuda_graphs()中处理，此处仅需处理传统模式       ████
            // ████████████████████████████████████████████████████████████████████████████
            if (!use_graph_) {
                sync_running_stats_async();
            }
            // ████████████████████████████████████████████████████████████████████████████
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        if (!use_graph_) {
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_2_));
            CHECK_CUDA(cudaStreamSynchronize(stream_comp_3_));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double fwd_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "\nTime (Unit: ms):" << std::endl;
        std::cout << "Total:    " << fwd_time << std::endl;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ parse_arguments函数已移至ta_v4_common_fp16.hpp（单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████

int main(int argc, char** argv) {
    Config config = parse_arguments(argc, argv);

    try {
        ConvGenStatsBNFinalizePointwiseBenchmark benchmark(config);
        benchmark.warmup();  // 使用默认的20次迭代
        benchmark.benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
