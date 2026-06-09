/**
 * @file task_base.cpp
 * @brief TaskBase 核心实现 —— dry run 引擎
 * @version 4.20.2
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 设计决策：compile() 要求 phase == MEMORY_LOCKED（显式契约）
 */

#include "renaissance/task/task_base.h"
#include "renaissance/task/deep_learning_task.h"
#include "renaissance/core/initializer.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/logger.h"
#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/backend/memory_arena.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/backend/graph_executor.h"
#include "renaissance/backend/op_registry.h"
#include <thread>
#include <exception>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

extern "C" cudaError_t launch_tr_fill_fp32_kernel(
    float* __restrict__ dst, float value, int64_t n, cudaStream_t stream);
extern "C" cudaError_t launch_tr_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset,
    float mean, float std, float* out, cudaStream_t stream);
#endif

#ifdef TR_USE_NCCL
#include <nccl.h>
#endif

namespace tr {
void launch_tr_fill_fp32_kernel_cpu(float* __restrict dst, float value, int64_t n);

namespace {
    const char* dtype_to_string(DType dtype) {
        switch (dtype) {
            case DType::FP32: return "fp32";
            case DType::FP16: return "fp16";
            case DType::INT32: return "int32";
            case DType::INT8: return "int8";
            default: return "?";
        }
    }

    const std::unordered_map<Region, const char*>& region_name_map() {
        static std::unordered_map<Region, const char*> m = {
            // B-Series: BN统计量
            {Region::B_PREV_MEAN, "B_PREV_MEAN"},
            {Region::B_PREV_VAR, "B_PREV_VAR"},
            {Region::B_NEXT_MEAN, "B_NEXT_MEAN"},
            {Region::B_NEXT_VAR, "B_NEXT_VAR"},
            // W-Series: 主模型权重
            {Region::W_EQ_BIAS, "W_EQ_BIAS"},
            {Region::W_EQ_SCALE, "W_EQ_SCALE"},
            {Region::W_BN_BIAS, "W_BN_BIAS"},
            {Region::W_BN_WEIGHT, "W_BN_WEIGHT"},
            {Region::W_FC_BIAS, "W_FC_BIAS"},
            {Region::W_FC_WEIGHT, "W_FC_WEIGHT"},
            {Region::W_FIRST_CONV, "W_FIRST_CONV"},
            {Region::W_DEEP_CONV, "W_DEEP_CONV"},
            // E-Series: EMA权重
            {Region::E_BN_BIAS, "E_BN_BIAS"},
            {Region::E_BN_WEIGHT, "E_BN_WEIGHT"},
            {Region::E_FC_BIAS, "E_FC_BIAS"},
            {Region::E_FC_WEIGHT, "E_FC_WEIGHT"},
            {Region::E_FIRST_CONV, "E_FIRST_CONV"},
            {Region::E_DEEP_CONV, "E_DEEP_CONV"},
            {Region::E_FC_WEIGHT_FP16, "E_FC_WEIGHT_FP16"},
            {Region::E_FIRST_CONV_FP16, "E_FIRST_CONV_FP16"},
            {Region::E_DEEP_CONV_FP16, "E_DEEP_CONV_FP16"},
            // A-Series: AMP FP16权重
            {Region::A_FC_WEIGHT, "A_FC_WEIGHT"},
            {Region::A_FIRST_CONV, "A_FIRST_CONV"},
            {Region::A_DEEP_CONV, "A_DEEP_CONV"},
            // G-Series: 梯度
            {Region::G_BN_BIAS, "G_BN_BIAS"},
            {Region::G_BN_WEIGHT, "G_BN_WEIGHT"},
            {Region::G_FC_BIAS, "G_FC_BIAS"},
            {Region::G_FC_WEIGHT, "G_FC_WEIGHT"},
            {Region::G_FIRST_CONV, "G_FIRST_CONV"},
            {Region::G_DEEP_CONV, "G_DEEP_CONV"},
            {Region::G_FC_WEIGHT_FP16, "G_FC_WEIGHT_FP16"},
            {Region::G_FIRST_CONV_FP16, "G_FIRST_CONV_FP16"},
            {Region::G_DEEP_CONV_FP16, "G_DEEP_CONV_FP16"},
            // M-Series: 一阶动量
            {Region::M_BN_BIAS, "M_BN_BIAS"},
            {Region::M_BN_WEIGHT, "M_BN_WEIGHT"},
            {Region::M_FC_BIAS, "M_FC_BIAS"},
            {Region::M_FC_WEIGHT, "M_FC_WEIGHT"},
            {Region::M_FIRST_CONV, "M_FIRST_CONV"},
            {Region::M_DEEP_CONV, "M_DEEP_CONV"},
            // V-Series: 二阶动量
            {Region::V_BN_BIAS, "V_BN_BIAS"},
            {Region::V_BN_WEIGHT, "V_BN_WEIGHT"},
            {Region::V_FC_BIAS, "V_FC_BIAS"},
            {Region::V_FC_WEIGHT, "V_FC_WEIGHT"},
            {Region::V_FIRST_CONV, "V_FIRST_CONV"},
            {Region::V_DEEP_CONV, "V_DEEP_CONV"},
            // N-Series: LARS范数
            {Region::N_FC_WEIGHT, "N_FC_WEIGHT"},
            {Region::N_FIRST_CONV, "N_FIRST_CONV"},
            {Region::N_DEEP_CONV, "N_DEEP_CONV"},
            // I-Series: 输入缓冲区
            {Region::I_A_LABEL, "I_A_LABEL"},
            {Region::I_A_DATA, "I_A_DATA"},
            {Region::I_B_LABEL, "I_B_LABEL"},
            {Region::I_B_DATA, "I_B_DATA"},
            // F-Series: 特征图与梯度槽
            {Region::F_FEATURE_FP32, "F_FEATURE_FP32"},
            {Region::F_GRAD_SLOT_FP32, "F_GRAD_SLOT_FP32"},
            {Region::F_FEATURE_FP16, "F_FEATURE_FP16"},
            {Region::F_GRAD_SLOT_FP16, "F_GRAD_SLOT_FP16"},
            // S-Series: 标量与掩码
            {Region::S_SCALAR_FP32, "S_SCALAR_FP32"},
            {Region::S_SCALAR_FP16, "S_SCALAR_FP16"},
            {Region::S_SCALAR_INT32, "S_SCALAR_INT32"},
            {Region::S_SCALAR_INT8, "S_SCALAR_INT8"},
            {Region::S_MASK, "S_MASK"},
            // T-Series: 临时张量
            {Region::T_TEMP_FP32, "T_TEMP_FP32"},
            {Region::T_TEMP_FP16, "T_TEMP_FP16"},
            {Region::T_TEMP_INT32, "T_TEMP_INT32"},
            {Region::T_TEMP_INT8, "T_TEMP_INT8"}
        };
        return m;
    }

    const char* region_to_string(Region r) {
        auto& m = region_name_map();
        auto it = m.find(r);
        return it != m.end() ? it->second : "?";
    }
}

// ========== Backend PIMPL ==========
struct TaskBase::Backend {
    std::vector<std::unique_ptr<DeviceContext>> contexts;
};

DeviceContext& TaskBase::context(int rank) const {
    TR_CHECK(rank >= 0 && rank < num_gpus_, IndexError,
             "Rank " << rank << " out of range [0, " << num_gpus_ << ")");
    TR_CHECK(backend_ != nullptr && rank < static_cast<int>(backend_->contexts.size()),
             RuntimeError, "Backend context not initialized for rank " << rank);
    return *backend_->contexts[rank];
}

// ========== 生命周期 ==========
TaskBase::TaskBase() {
    memory_plan_.config_.need_mask = plan_config_.need_mask;
}
TaskBase::~TaskBase() = default;

// ========== PLANNING 阶段 ==========
DTensor TaskBase::alloc(const Shape& shape, DType dtype, Region region) {
    check_phase(Phase::PLANNING, "alloc");

    DTensor dt = memory_plan_.alloc(shape, dtype, region);

    InitConfig config = initializer_.derive(region);
    dt.init_config = config;
    memory_plan_.set_init_config(dt.id, config);

    return dt;
}

// alloc(shape, dtype) 默认Region推断已禁用——必须显式指定Region
// 原因：AMP下F_FEATURE_FP32被is_condition_enabled禁用，默认推断会导致slot_bytes=0

DTensor TaskBase::alloc_scalar(DType dtype) {
    check_phase(Phase::PLANNING, "alloc_scalar");
    return memory_plan_.alloc_scalar(dtype);
}

void TaskBase::finalize_memory() {
    check_phase(Phase::PLANNING, "finalize_memory");
    memory_plan_.finalize();
    phase_ = Phase::MEMORY_LOCKED;
}

// ========== MEMORY_LOCKED 阶段 ==========
void TaskBase::add_graph(const std::string& name, ComputationGraph graph,
                         StreamKind stream) {
    check_phase(Phase::MEMORY_LOCKED, "add_graph");
    named_graphs_.emplace(name, GraphEntry{std::move(graph), stream});
}

// ========== 编译管线 ==========
void TaskBase::compile() {
    debug_mode_ = false;
    compile_impl(false);
}

void TaskBase::compile_for_dry_run() {
    debug_mode_ = true;
    compile_impl(true);
}

void TaskBase::compile_impl(bool debug_mode) {
    (void)debug_mode;  // debug_mode_ 已在 compile()/compile_for_dry_run() 中设置

    // 确保算子已注册
    extern void register_default_ops();
    register_default_ops();

    compile_freeze_global();
    compile_invoke_on_prepare();
    compile_verify_memory_locked();

    if (debug_mode_) {
        compile_debug_print_and_return();
        return;
    }

    compile_alloc_hardware();

    if (is_simple_task()) {
        // SimpleTask：不走 GraphAtlas / pre_capture 深度学习流水线，
        // 直接对每个命名图独立捕获（保持 GraphEntry.stream 语义）
        compile_capture_simple();
    } else {
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {

            GraphAtlas atlas = dl->build_graph_atlas();

            std::vector<DeviceContext*> ctx_ptrs;
            for (auto& ctx : backend_->contexts) {
                ctx_ptrs.push_back(ctx.get());
            }

            for (int rank = 0; rank < num_gpus_; ++rank) {
                backend_->contexts[rank]->set_rank(rank);
                backend_->contexts[rank]->set_memory_plan(active_memory_plan_);
            }

            captured_result_ = pre_capture(atlas, ctx_ptrs);

            for (int rank = 0; rank < num_gpus_; ++rank) {
                backend_->contexts[rank]->set_memory_plan(active_memory_plan_);
            }

            dl->build_exec_table();
            // 绑定 lr DTensor ID，供 run_train_epoch_gpu() 使用
            dl->lr_dtensor_id_ = active_memory_plan_->lr_id();
            // compile_mark_compiled() 延迟到 if/else 块之后无条件调用
        } else {
            GraphAtlas atlas = build_simple_atlas(name_to_gid_);

            std::vector<DeviceContext*> ctx_ptrs;
            for (auto& ctx : backend_->contexts) {
                ctx_ptrs.push_back(ctx.get());
            }

            captured_result_ = pre_capture(atlas, ctx_ptrs);
        }
    }

    compile_mark_compiled();

    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        // ===== 关键：init_all() 之前全局清零 Arena，清除 warmup 残留 =====
        if (GlobalRegistry::instance().using_gpu()) {
            #ifdef TR_USE_CUDA
            for (int rank = 0; rank < num_gpus_; ++rank) {
                cudaSetDevice(backend_->contexts[rank]->device_id());
                cudaMemset(ArenaKeeper::instance().ptr_at(rank, 0), 0,
                           active_memory_plan_->total_bytes());
                cudaDeviceSynchronize();
            }
            #endif
        } else {
            for (int rank = 0; rank < num_gpus_; ++rank) {
                std::memset(ArenaKeeper::instance().ptr_at(rank, 0), 0,
                            active_memory_plan_->total_bytes());
            }
        }
        // ================================================================

        init_all();

        // ===== 为所有 variant memory plan 执行 init_all() =====
        // CPU 模式同样需要：不同变体（如 last-batch 变体）的 DTensor offset 可能不同，
        // 必须单独初始化，否则切换 MemoryPlan 后会读到垃圾值。
        dl->init_all_variant_memory_plans();
        // ======================================================

        // ===== init_all() 之后：为所有 variant 写入运行时标量（batch_size / bn_epsilon 等）=====
        // CPU/GPU 通用路径：确保所有变体的标量都被正确初始化。
        dl->init_variant_scalars();
        // =========================================================

        #ifdef TR_USE_CUDA
        if (GlobalRegistry::instance().using_gpu()) {
            // ========== 诊断：init_all() 后立即读取 scaling / lr ==========
            const auto& b = dl->active_memory_plan_->baseline();
            for (int rank = 0; rank < num_gpus_; ++rank) {
                DeviceContext& ctx = context(rank);
                float scaling_val = 0.0f, lr_val = 0.0f;
                cudaSetDevice(ctx.device_id());
                cudaMemcpy(&scaling_val, ctx.ptr_at(b.scaling), sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lr_val, ctx.ptr_at(b.lr), sizeof(float), cudaMemcpyDeviceToHost);
                // fprintf(stderr, "[INIT-CHECK] rank=%d scaling=%f (id=%d, offset=%llu, ptr=%p) lr=%f (id=%d)\n",
                //        rank, scaling_val, b.scaling,
                //        (unsigned long long)active_memory_plan_->get_dtensor(b.scaling).offset(),
                //        (void*)ctx.ptr_at(b.scaling),
                //        lr_val, b.lr);
                // fflush(stderr);
            }
            // ==============================================================
            dl->lr_pinned_.resize(num_gpus_);
            for (int rank = 0; rank < num_gpus_; ++rank) {
                cudaError_t err = cudaMallocHost(&dl->lr_pinned_[rank], sizeof(float));
                TR_CHECK(err == cudaSuccess, DeviceError,
                         "cudaMallocHost failed for lr_pinned_ rank " << rank);
            }
        }
        #endif

        // ========== 诊断打印：ArchPlan / MemoryPlan / ComputationGraph ==========
        std::cout << "\n========== COMPILE DIAGNOSTICS ==========" << std::endl;
        std::cout << "--- ArchPlan ---" << std::endl;
        std::cout << dl->arch_plan_.to_string() << std::endl;
        std::cout << "--- MemoryPlan ---" << std::endl;
        if (dl->active_memory_plan_) {
            std::cout << dl->active_memory_plan_->dump_layout() << std::endl;
        }
        std::cout << "--- Train ComputationGraph ---" << std::endl;
        if (dl->train_cg_) {
            std::cout << dl->train_cg_->debug_dump(true) << std::endl;
        }
        std::cout << "--- Inference ComputationGraph ---" << std::endl;
        if (dl->infer_cg_) {
            std::cout << dl->infer_cg_->debug_dump(true) << std::endl;
        }
        std::cout << "=========================================\n" << std::endl;
    }
}

void TaskBase::compile_capture_simple() {

    // ====== 自动检测 DTENSOR 节点并分配 StagingParamPool ======
    // GPU/CPU 公共路径：compile_capture_simple 无论设备模式均需分配
    {
        auto& reg = GlobalRegistry::instance();
        if (!reg.has_staging_params()) {
            bool need_params = false;
            for (const auto& [name, entry] : named_graphs_) {
                for (const auto& node : entry.graph.linear_nodes()) {
                    if (node.kind == GraphNode::Kind::RANGE &&
                        node.range_op == RangeOp::RANGE_H2D_COPY_DTENSOR) {
                        need_params = true;
                        break;
                    }
                }
            }
            if (need_params) {
                reg.allocate_staging_params(256);
            }
        }
    }
    // ============================================================

    // ====== 自动检测 A/B 传输节点并分配 StagingBufferPool ======
    // GPU/CPU 公共路径：两种模式下都需要 staging_memory_ptr 访问能力
    {
        auto& reg = GlobalRegistry::instance();
        if (!reg.has_staging_memory()) {
            bool need_staging = false;
            for (const auto& [name, entry] : named_graphs_) {
                for (const auto& node : entry.graph.linear_nodes()) {
                    if (node.kind != GraphNode::Kind::RANGE) continue;
                    if (node.range_op == RangeOp::RANGE_H2D_COPY_A ||
                        node.range_op == RangeOp::RANGE_H2D_COPY_B) {
                        need_staging = true;
                        break;
                    }
                }
            }

            if (need_staging) {
                int batch_size = reg.get_local_batch_size();
                int resolution = reg.max_sample_resolution();
                int channels = reg.num_color_channels();
                bool using_amp = reg.using_amp();

                int effective_c = (using_amp && channels == 3) ? 4 : channels;
                DType data_dtype = using_amp ? DType::FP16 : DType::FP32;

                size_t label_slot = DistributedTensor::compute_slot_bytes(
                    Shape(batch_size, 1, 1, 1), DType::INT32, Region::I_A_LABEL);
                size_t data_slot = DistributedTensor::compute_slot_bytes(
                    Shape(batch_size, resolution, resolution, effective_c),
                    data_dtype, Region::I_A_DATA);
                size_t per_zone = label_slot + data_slot;

                reg.allocate_staging_memory(per_zone * 2);
            }
        }
    }
    // ============================================================

#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        // TR_LOG_INFO("task") << "[DBG] compile_capture_simple: enter";
        cudaGetLastError();
        cudaDeviceSynchronize();

        // TR_LOG_INFO("task") << "[DBG] compile_capture_simple: start warmup loop, num_gpus=" << num_gpus_;
        for (int rank = 0; rank < num_gpus_; ++rank) {
            // TR_LOG_INFO("task") << "[DBG] warmup rank " << rank << " begin";
            DeviceContext& warm_ctx = *backend_->contexts[rank];
            int gpu_id = warm_ctx.device_id();
            // TR_LOG_INFO("task") << "[DBG] warmup rank " << rank << " gpu_id=" << gpu_id;
            cudaSetDevice(gpu_id);
            cudaGetLastError();
            cudaDeviceSynchronize();

            warm_ctx.set_rank(rank);
            warm_ctx.set_memory_plan(&memory_plan_);

            for (const auto& [name, entry] : named_graphs_) {
                for (const auto& node : entry.graph.linear_nodes()) {
                    if (node.kind != GraphNode::Kind::COMPUTE) continue;
                    if (!require_warmup(node.compute_op)) continue;
                    // TR_LOG_INFO("task") << "[DBG] warmup op: " << static_cast<int>(node.compute_op);
                    warmup_single_cudnn_op(node, memory_plan_, warm_ctx);
                    // TR_LOG_INFO("task") << "[DBG] warmup op done";
                }
            }
            cudaDeviceSynchronize();
            cudaGetLastError();
            // TR_LOG_INFO("task") << "[DBG] warmup rank " << rank << " end";
        }
        // TR_LOG_INFO("task") << "[DBG] compile_capture_simple: warmup done";
    }
#endif

    // TR_LOG_INFO("task") << "[DBG] compile_capture_simple: starting capture phase, num_graphs="
    //                     << named_graphs_.size() << " num_gpus=" << num_gpus_;
    int graph_index = 0;
    for (auto& [name, entry] : named_graphs_) {
        // TR_LOG_INFO("task") << "[DBG] capture graph '" << name << "' graph_index=" << graph_index;
        CapturedGraph cg;

        if (entry.graph.has_nccl_ops() && context(0).is_gpu()) {
            // TR_LOG_INFO("task") << "[DBG] NCCL graph detected, coordinated multi-rank capture";

            GraphId gid = GraphId::SIMPLE_TASK_GRAPH;
            cg.reserve_ranks(num_gpus_);

#ifdef TR_USE_CUDA
#ifdef TR_USE_NCCL
            // =====================================================================
            // Phase 0（新增）: 预创建所有事件（对齐 capture_cuda.cpp 模式）
            // =====================================================================
            std::vector<cudaStream_t> cap_streams(num_gpus_);
            std::vector<MultiStreamCaptureState> states(num_gpus_);
            for (int r = 0; r < num_gpus_; ++r) {
                DeviceContext& dc = *backend_->contexts[r];
                cudaSetDevice(dc.device_id());
                cudaDeviceSynchronize();

                dc.set_rank(r);
                dc.set_memory_plan(&memory_plan_);
                cap_streams[r] = static_cast<cudaStream_t>(dc.stream(entry.stream));

                states[r].primary_stream = cap_streams[r];
                states[r].get_or_register(cap_streams[r]);
                states[r].get_or_register(
                    static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_1)));
                states[r].get_or_register(
                    static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_2)));
                states[r].get_or_register(
                    static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_3)));
                states[r].get_or_register(
                    static_cast<cudaStream_t>(dc.stream(StreamKind::UPDATE)));

                for (int i = 0; i < states[r].num_active; ++i) {
                    if (states[r].streams[i].last_done_event) {
                        cudaEventDestroy(states[r].streams[i].last_done_event);
                    }
                    cudaEventCreateWithFlags(
                        &states[r].streams[i].last_done_event,
                        cudaEventDisableTiming);
                }
            }

            // =====================================================================
            // Phase 1: BeginCapture on ALL ranks simultaneously
            // =====================================================================
            for (int r = 0; r < num_gpus_; ++r) {
                cudaSetDevice(backend_->contexts[r]->device_id());
                cudaError_t cap_err = cudaStreamBeginCapture(
                    cap_streams[r], cudaStreamCaptureModeThreadLocal);
                if (cap_err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaStreamBeginCapture failed for rank "
                                    << r << ": " << cudaGetErrorString(cap_err));
                }
            }

            // Phase 2: ncclGroupStart → replay all ranks → ncclGroupEnd
            bool capture_committed = false;
            try {
                ncclGroupStart();
                for (int r = 0; r < num_gpus_; ++r) {
                    DeviceContext& dc = *backend_->contexts[r];
                    cudaSetDevice(dc.device_id());

                    MultiStreamCaptureState& state = states[r];

                    if (state.num_active > 1) {
                        cudaEventRecord(state.streams[0].last_done_event,
                                       state.primary_stream);
                        for (int i = 1; i < state.num_active; ++i) {
                            cudaStreamWaitEvent(state.streams[i].stream,
                                               state.streams[0].last_done_event, 0);
                        }
                    }

                    const auto& nodes = (entry.graph.linear_nodes().empty()
                                         ? entry.graph.nodes(gid)
                                         : entry.graph.linear_nodes());
                    if (!nodes.empty()) {
                        for (size_t ni = 0; ni < nodes.size(); ++ni) {
                            const auto& node = nodes[ni];
                            if (node.kind != GraphNode::Kind::RANGE) continue;
                            auto& range_entry = g_range_op_table[
                                static_cast<size_t>(node.range_op)];
                            if (range_entry.launch_cuda) {
                                range_entry.launch_cuda(
                                    node, memory_plan_, dc, state);
                            }
                        }
                    }
                }
                ncclGroupEnd();
                capture_committed = true;
            } catch (...) {
                ncclGroupEnd();
                for (int r = 0; r < num_gpus_; ++r) {
                    cudaSetDevice(backend_->contexts[r]->device_id());
                    cudaGraph_t dummy = nullptr;
                    cudaError_t end_err = cudaStreamEndCapture(
                        cap_streams[r], &dummy);
                    if (dummy) cudaGraphDestroy(dummy);
                    (void)end_err;
                }
                for (int r = 0; r < num_gpus_; ++r) {
                    cudaSetDevice(backend_->contexts[r]->device_id());
                    states[r].cleanup_all_events();
                }
                throw;
            }

            // Phase 3a: EndCapture on ALL ranks
            std::vector<cudaGraph_t> captured_graphs(num_gpus_, nullptr);
            for (int r = 0; r < num_gpus_; ++r) {
                cudaSetDevice(backend_->contexts[r]->device_id());
                cudaError_t end_err = cudaStreamEndCapture(
                    cap_streams[r], &captured_graphs[r]);
                if (end_err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaStreamEndCapture failed for rank "
                                    << r << ": " << cudaGetErrorString(end_err));
                }
            }

            // Phase 3b: Instantiate on ALL ranks
            for (int r = 0; r < num_gpus_; ++r) {
                cudaSetDevice(backend_->contexts[r]->device_id());

                cudaGraphExec_t exec = nullptr;
                cudaGraphNode_t error_node = nullptr;
                char log_buf[2048] = {};
                cudaError_t inst_err = cudaGraphInstantiate(
                    &exec, captured_graphs[r],
                    &error_node, log_buf, sizeof(log_buf));
                if (inst_err != cudaSuccess) {
                    std::string info;
                    if (log_buf[0] != '\0')
                        info = std::string(" log='") + log_buf + "'";
                    TR_DEVICE_ERROR("cudaGraphInstantiate failed for rank "
                                    << r << ": "
                                    << cudaGetErrorString(inst_err) << info);
                }

                cg.set_rank_exec(r, exec);
                cudaGraphDestroy(captured_graphs[r]);
            }

            // =====================================================================
            // Phase 4（新增）: 所有 Instantiate 完成后销毁事件
            // =====================================================================
            for (int r = 0; r < num_gpus_; ++r) {
                cudaSetDevice(backend_->contexts[r]->device_id());
                states[r].cleanup_all_events();
            }

            cg.set_is_cuda(true);
#else
            TR_NOT_IMPLEMENTED("TR_USE_NCCL not defined, NCCL graph not supported");
#endif // TR_USE_NCCL
#endif // TR_USE_CUDA
        } else {
            cg.reserve_ranks(num_gpus_);

            for (int rank = 0; rank < num_gpus_; ++rank) {
                // TR_LOG_INFO("task") << "[DBG] capture rank " << rank << " begin";
                DeviceContext& ctx = *backend_->contexts[rank];
#ifdef TR_USE_CUDA
                if (ctx.is_gpu()) {
                    cudaError_t err = cudaSetDevice(ctx.device_id());
                    if (err != cudaSuccess) {
                        TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                        << ": " << cudaGetErrorString(err));
                    }
                }
#endif
                ctx.set_rank(rank);
                ctx.set_memory_plan(&memory_plan_);

                GraphId gid = GraphId::SIMPLE_TASK_GRAPH;

                // TR_LOG_INFO("task") << "[DBG] calling CapturedGraph::capture...";
                auto captured = CapturedGraph::capture(
                    entry.graph, memory_plan_, gid, entry.stream, ctx);
                // TR_LOG_INFO("task") << "[DBG] CapturedGraph::capture done, is_cuda=" << captured.is_cuda();

                if (rank == 0) {
                    // TR_LOG_INFO("task") << "[DBG] set_metadata_from...";
                    cg.set_metadata_from(captured);
                    // TR_LOG_INFO("task") << "[DBG] move_cpu_ops_from...";
                    cg.move_cpu_ops_from(captured);
                    // TR_LOG_INFO("task") << "[DBG] rank0 metadata done";
                }
                // TR_LOG_INFO("task") << "[DBG] check is_cuda=" << captured.is_cuda();
                if (captured.is_cuda()) {
                    // TR_LOG_INFO("task") << "[DBG] release_rank_exec...";
                    auto exec = captured.release_rank_exec(0);
                    // TR_LOG_INFO("task") << "[DBG] release_rank_exec done, exec=" << exec;
                    cg.set_rank_exec(rank, exec);
                    // TR_LOG_INFO("task") << "[DBG] set_rank_exec done";
                }
                // TR_LOG_INFO("task") << "[DBG] capture rank " << rank << " end";
            }
        }

        // TR_LOG_INFO("task") << "[DBG] emplace simple_captured_graphs_ ...";
        simple_captured_graphs_.emplace(name, std::move(cg));
        // TR_LOG_INFO("task") << "[DBG] emplace done";
        graph_index++;  // 递增图索引，为下一个图分配GraphId
    }

    // ── CPU Dry-Run：对所有 CPU 图执行一次 launch，触发 workspace 扩容 ──
    for (auto& [name, cg] : simple_captured_graphs_) {
        if (cg.is_cuda()) continue;
        cg.launch(0, nullptr);
    }

    // TR_LOG_INFO("task") << "[DBG] compile_capture_simple: all done";
}

void TaskBase::compile_freeze_global() {
    auto& reg = GlobalRegistry::instance();
    if (reg.using_gpu()) {
        num_gpus_ = static_cast<int>(reg.gpu_ids().size());
        TR_CHECK(num_gpus_ > 0 && num_gpus_ <= 8, ValueError,
                 "Invalid GPU count: " << num_gpus_);
    } else {
        num_gpus_ = 1;
    }
}

void TaskBase::compile_invoke_on_prepare() {
    on_prepare();
}

void TaskBase::compile_verify_memory_locked() {
    if (phase_ == Phase::MEMORY_LOCKED) {
        return;
    }
    if (phase_ == Phase::PLANNING) {
        if (!memory_plan_.is_finalized()) {
            finalize_memory();
        } else {
            phase_ = Phase::MEMORY_LOCKED;
        }
        return;
    }
    TR_VALUE_ERROR("compile_verify_memory_locked: unexpected phase "
                   << static_cast<int>(phase_));
}

void TaskBase::compile_debug_print_and_return() {
    active_memory_plan_->validate();

    std::cout << "\n========================================\n";
    std::cout << "=== Phase A: MemoryPlan & ComputationGraph ===\n";
    std::cout << active_memory_plan_->dump_layout() << "\n";

    for (const auto& dt : active_memory_plan_->dtensors()) {
        std::cout << "DTensor id=" << dt.id
                  << ": " << dt.shape.to_string()
                  << " " << dtype_to_string(dt.dtype)
                  << "  " << region_to_string(dt.region)
                  << "  offset=" << dt.offset()
                  << "  slot_bytes=" << dt.slot_bytes()
                  << "  init=" << to_string(dt.init_config.kind);
        if (dt.init_config.kind != InitKind::NONE && dt.init_config.kind != InitKind::ZEROS) {
            std::cout << "(" << dt.init_config.scale << ")";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== ComputationGraph: " << named_graphs_.size() << " graph(s) ===\n";
    for (const auto& [name, entry] : named_graphs_) {
        std::cout << "=== Graph: " << name << " ===\n"
                  << entry.graph.debug_dump() << "\n";
    }

    // SimpleTask 没有图集概念，debug 模式下跳过 Atlas 构建
    if (!is_simple_task()) {
        // Phase A: 构建 GraphAtlas（P_ULTIMATE.md 三阶段设计）
        std::cout << "\n=== Phase A: GraphAtlas ===\n";
        GraphAtlas atlas;
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            atlas = dl->build_graph_atlas();
        } else {
            atlas = build_simple_atlas(name_to_gid_);
        }
        captured_result_.atlas = std::move(atlas);
        std::cout << "Generated GraphAtlas with " << named_graphs_.size() << " graph(s)\n";
        std::cout << "=== Phase A Complete ===\n";
    } else {
        std::cout << "\n=== SimpleTask: skip GraphAtlas ===\n";
    }

    // 为debug模式生成空的captured_result（不执行实际捕获）
    captured_result_.total_slots = 0;
    captured_result_.captured = 0;
    captured_result_.reused = 0;

    std::cout << "\n=== Dry Run Complete (No Hardware) ===\n";
    std::cout << "========================================\n";

    phase_ = Phase::COMPILED;
}

GraphAtlas TaskBase::build_simple_atlas(
    std::unordered_map<std::string, GraphId>& name_to_gid) {
    TR_CHECK(!is_simple_task(), NotImplementedError,
             "SimpleTask does not support GraphAtlas");
    name_to_gid.clear();
    GraphAtlas atlas;

    std::vector<std::string> sorted_names;
    for (const auto& [name, _] : named_graphs_)
        sorted_names.push_back(name);
    std::sort(sorted_names.begin(), sorted_names.end());

    // SimpleTask 运行时按 name 查找，不走 Atlas 索引。
    // Atlas 仅在 debug/dry-run 模式下用于打印信息，所有 graph 共用 SIMPLE_TASK_GRAPH。
    for (const auto& name : sorted_names) {
        const auto& entry = named_graphs_[name];

        GraphId gid = GraphId::SIMPLE_TASK_GRAPH;
        auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
        sl.cg = &entry.graph;
        sl.mp = &memory_plan_;
        sl.stream_kind = entry.stream;

        if (!memory_plan_.dtensors().empty()) {
            const auto& dt = memory_plan_.dtensors()[0];
            sl.shape_id = ShapeId{dt.shape.n(), dt.shape.h(),
                                   dt.shape.w(), dt.shape.c()};
        } else {
            sl.shape_id = kShapeInvariant;
        }

        name_to_gid[name] = gid;

        std::cout << "Atlas slot (v=0, gid=" << static_cast<int>(gid)
                  << ") filled: " << name
                  << " shape=" << sl.shape_id.to_string() << "\n";
    }

    return atlas;
}

void TaskBase::compile_mark_compiled() {
    phase_ = Phase::COMPILED;
}

// ========== 执行 ==========
void TaskBase::run(const std::string& name) {
    run_impl(name, false);
}

void TaskBase::dry_run(const std::string& name) {
    run_impl(name, true);
}

void TaskBase::run_impl(const std::string& name, bool dry_run) {
    check_run_eligibility(dry_run);

    auto it = named_graphs_.find(name);
    TR_CHECK(it != named_graphs_.end(), ValueError,
             "Graph not found: " << name);

    // SimpleTask 路径：不走 Atlas，直接从独立捕获存储中启动
    if (is_simple_task()) {
        auto cap_it = simple_captured_graphs_.find(name);
        TR_CHECK(cap_it != simple_captured_graphs_.end(), ValueError,
                 "Graph not captured: " << name);

        const GraphEntry& entry = it->second;

        if (dry_run) {
            std::cout << "[DRY RUN] run graph '" << name << "':\n"
                      << "  SimpleTask direct capture (dry run — no real CUDA launch)\n";
            cap_it->second.launch(0, nullptr);
            return;
        }

        for (int rank = 0; rank < num_gpus_; ++rank) {
            DeviceContext& ctx = *backend_->contexts[rank];
#ifdef TR_USE_CUDA
            if (ctx.is_gpu()) {
                cudaError_t err = cudaSetDevice(ctx.device_id());
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }
            }
#endif
            void* stream = ctx.stream(entry.stream);
            cap_it->second.launch(rank, stream);
        }

        for (int rank = 0; rank < num_gpus_; ++rank) {
            backend_->contexts[rank]->synchronize_stream(entry.stream);
        }
        return;
    }

    // 深度学习路径（GraphAtlas / pre_capture）
    if (dry_run) {
        auto gid_it = name_to_gid_.find(name);
        TR_CHECK(gid_it != name_to_gid_.end(), ValueError,
                 "GraphId not found for: " << name);

        int32_t idx = captured_result_.atlas.index(0, gid_it->second);
        TR_CHECK(idx >= 0 && static_cast<size_t>(idx) < captured_result_.graphs.size(),
                 IndexError, "CapturedGraph index " << idx
                 << " out of range for graph '" << name << "'");

        std::cout << "[DRY RUN] run graph '" << name << "':\n"
                  << "  Phase C: CapturedGraph[" << idx << "]"
                  << " gid=" << static_cast<int>(captured_result_.graphs[idx].key().gid)
                  << " shape=" << captured_result_.graphs[idx].key().shape.to_string()
                  << " (dry run — no real CUDA launch)\n";
        captured_result_.graphs[idx].launch(0, nullptr);
        return;
    }

    auto gid_it = name_to_gid_.find(name);
    TR_CHECK(gid_it != name_to_gid_.end(), ValueError,
             "GraphId not found for: " << name);

    int32_t idx = captured_result_.atlas.index(0, gid_it->second);
    TR_CHECK(idx >= 0 && static_cast<size_t>(idx) < captured_result_.graphs.size(),
             IndexError, "CapturedGraph index " << idx
             << " out of range for graph '" << name << "'");

    const GraphEntry& entry = it->second;

    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
#ifdef TR_USE_CUDA
        if (ctx.is_gpu()) {
            cudaError_t err = cudaSetDevice(ctx.device_id());
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                << ": " << cudaGetErrorString(err));
            }
        }
#endif
        void* stream = ctx.stream(entry.stream);
        captured_result_.graphs[idx].launch(rank, stream);
    }

    for (auto& ctx_ptr : backend_->contexts) {
        ctx_ptr->synchronize_all();
    }
}

void TaskBase::run(const std::string& a, const std::string& b) {
    check_run_eligibility(false);

    auto it_a = named_graphs_.find(a);
    auto it_b = named_graphs_.find(b);
    TR_CHECK(it_a != named_graphs_.end(), ValueError, "Graph not found: " << a);
    TR_CHECK(it_b != named_graphs_.end(), ValueError, "Graph not found: " << b);

    const GraphEntry& entry_a = it_a->second;
    const GraphEntry& entry_b = it_b->second;

    // SimpleTask 路径：直接从独立捕获存储中启动
    if (is_simple_task()) {
        auto cap_a = simple_captured_graphs_.find(a);
        auto cap_b = simple_captured_graphs_.find(b);
        TR_CHECK(cap_a != simple_captured_graphs_.end(), ValueError,
                 "Graph not captured: " << a);
        TR_CHECK(cap_b != simple_captured_graphs_.end(), ValueError,
                 "Graph not captured: " << b);

        for (int rank = 0; rank < num_gpus_; ++rank) {
            DeviceContext& ctx = *backend_->contexts[rank];
#ifdef TR_USE_CUDA
            if (ctx.is_gpu()) {
                cudaError_t err = cudaSetDevice(ctx.device_id());
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }
            }
#endif
            void* stream_a = ctx.stream(entry_a.stream);
            void* stream_b = ctx.stream(entry_b.stream);
            cap_a->second.launch(rank, stream_a);
            cap_b->second.launch(rank, stream_b);
        }

        for (int rank = 0; rank < num_gpus_; ++rank) {
            auto& ctx = *backend_->contexts[rank];
            ctx.synchronize_stream(entry_a.stream);
            ctx.synchronize_stream(entry_b.stream);
        }
        return;
    }

    // 深度学习路径（GraphAtlas / pre_capture）
    auto gid_a = name_to_gid_.find(a);
    auto gid_b = name_to_gid_.find(b);
    TR_CHECK(gid_a != name_to_gid_.end(), ValueError, "GraphId not found for: " << a);
    TR_CHECK(gid_b != name_to_gid_.end(), ValueError, "GraphId not found for: " << b);

    int32_t idx_a = captured_result_.atlas.index(0, gid_a->second);
    int32_t idx_b = captured_result_.atlas.index(0, gid_b->second);
    TR_CHECK(idx_a >= 0 && static_cast<size_t>(idx_a) < captured_result_.graphs.size(),
             IndexError, "CapturedGraph index out of range for: " << a);
    TR_CHECK(idx_b >= 0 && static_cast<size_t>(idx_b) < captured_result_.graphs.size(),
             IndexError, "CapturedGraph index out of range for: " << b);

    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
#ifdef TR_USE_CUDA
        if (ctx.is_gpu()) {
            cudaError_t err = cudaSetDevice(ctx.device_id());
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                << ": " << cudaGetErrorString(err));
            }
        }
#endif
        void* stream_a = ctx.stream(entry_a.stream);
        void* stream_b = ctx.stream(entry_b.stream);
        captured_result_.graphs[idx_a].launch(rank, stream_a);
        captured_result_.graphs[idx_b].launch(rank, stream_b);
    }

    for (auto& ctx_ptr : backend_->contexts) {
        ctx_ptr->synchronize_all();
    }
}

void TaskBase::check_run_eligibility(bool dry_run) const {
    TR_CHECK(phase_ == Phase::COMPILED, ValueError,
             "Must call compile() before run (current phase: "
             << static_cast<int>(phase_) << ")");

    if (!dry_run && !backend_) {
        TR_VALUE_ERROR("Backend not initialized. Use compile(false) instead of compile(true)");
    }
}

// 紧凑性检查辅助：DTensor 与 Tensor 布局一致时可直接 memcpy，否则需重排
static void check_compact_or_throw(const DTensor& dt, const char* op) {
    if (!dt.is_compact()) {
        TR_NOT_IMPLEMENTED(std::string(op) +
            " for non-compact DTensor not yet implemented (region=" +
            region_to_string(dt.region) + ", padded_c=" +
            std::to_string(dt.padded_c()) + ")");
    }
}

// ========== 数据传输 ==========
void TaskBase::transfer_to_rank(const Tensor& host, const DTensor& dt, int rank) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip transfer_to_rank " << dt.id
                  << " -> rank " << rank
                  << (dt.is_compact() ? " (compact)" : " (NON-compact)")
                  << "\n";
        return;
    }

    // 参数验证
    TR_CHECK(host.dtype() == dt.dtype, ValueError,
             "DType mismatch: host vs dtensor");
    TR_CHECK(host.numel() >= dt.shape.numel(), ValueError,
             "Host tensor too small: host.numel()=" << host.numel()
             << ", dtensor.numel()=" << dt.shape.numel());
    TR_CHECK(rank >= 0 && rank < num_gpus_, IndexError,
             "Rank " << rank << " out of range [0, " << num_gpus_ << ")");

    // 获取目标设备指针（必须用 memory_plan_ 中 finalize 后的最新 offset，
    // 传入的 dt 可能是 PLANNING 阶段的旧副本，其 offset 已失效）
    auto& reg = GlobalRegistry::instance();
    const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);
    void* dst = ArenaKeeper::instance().ptr_at(rank, static_cast<size_t>(live_dt.offset()));

    // 计算有效字节数（n×h×w×c，不包括padding）
    const uint64_t valid_bytes = dt.nbytes();  // 有效元素的字节数

    if (reg.using_gpu()) {
#ifdef TR_USE_CUDA
        // GPU模式：区分是否需要布局转换
        if (dt.is_compact()) {
            // 紧凑DTensor：直接传输有效字节数
            int physical_gpu_id = reg.gpu_ids()[rank];
            cudaError_t err = cudaSetDevice(physical_gpu_id);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                << ": " << cudaGetErrorString(err));
            }

            // 同步H2D传输
            err = cudaMemcpy(dst, host.data<void>(), valid_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaMemcpy H2D to rank " << rank
                                << " failed: " << cudaGetErrorString(err));
            }
        } else {
            // 非紧凑DTensor：需要缓冲区进行重排中转
            // 创建包含padding的临时缓冲区
            const int64_t C_padded = dt.w_stride_cuda();  // padding后的C通道数
            const uint64_t padded_bytes = dt.n() * dt.h() * dt.w() * C_padded * DistributedTensor::dsize(dt.dtype);

            // 创建临时缓冲区（包含padding）
            Tensor temp_buffer(Shape{static_cast<int>(dt.n()), static_cast<int>(dt.h()),
                                        static_cast<int>(dt.w()), static_cast<int>(C_padded)},
                             dt.dtype);

            // 执行紧凑→非紧凑重排：从host复制数据到temp_buffer
            const size_t elem_size = DistributedTensor::dsize(dt.dtype);
            const size_t src_row_bytes = dt.c() * elem_size;
            const size_t dst_row_bytes = C_padded * elem_size;
            const uint8_t* src = static_cast<const uint8_t*>(host.data<void>());
            uint8_t* dst_buffer = static_cast<uint8_t*>(temp_buffer.data<void>());

            for (int n = 0; n < dt.n(); ++n) {
                for (int h = 0; h < dt.h(); ++h) {
                    for (int w = 0; w < dt.w(); ++w) {
                        size_t src_off = (((n * dt.h() + h) * dt.w() + w) * dt.c()) * elem_size;
                        size_t dst_off = (((n * dt.h() + h) * dt.w() + w) * C_padded) * elem_size;
                        std::memcpy(dst_buffer + dst_off, src + src_off, src_row_bytes);
                    }
                }
            }

            // 传输包含padding的数据到GPU
            int physical_gpu_id = reg.gpu_ids()[rank];
            cudaError_t err = cudaSetDevice(physical_gpu_id);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                << ": " << cudaGetErrorString(err));
            }

            err = cudaMemcpy(dst, temp_buffer.data<void>(), padded_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaMemcpy H2D (padded) to rank " << rank
                                << " failed: " << cudaGetErrorString(err));
            }
        }
#else
        (void)dst; (void)valid_bytes;
        TR_DEVICE_ERROR("GPU mode requested but TR_USE_CUDA is not defined");
#endif
    } else {
        // CPU模式：直接memcpy有效字节数
        if (dt.is_compact()) {
            // 紧凑DTensor：直接memcpy
            std::memcpy(dst, host.data<void>(), valid_bytes);
        } else {
            // 非紧凑DTensor：需要重排
            const int64_t C_padded = dt.w_stride_cuda();
            const size_t elem_size = DistributedTensor::dsize(dt.dtype);
            const size_t src_row_bytes = dt.c() * elem_size;
            const uint8_t* src = static_cast<const uint8_t*>(host.data<void>());
            uint8_t* dst_buffer = static_cast<uint8_t*>(dst);

            for (int n = 0; n < dt.n(); ++n) {
                for (int h = 0; h < dt.h(); ++h) {
                    for (int w = 0; w < dt.w(); ++w) {
                        size_t src_off = (((n * dt.h() + h) * dt.w() + w) * dt.c()) * elem_size;
                        size_t dst_off = (((n * dt.h() + h) * dt.w() + w) * C_padded) * elem_size;
                        std::memcpy(dst_buffer + dst_off, src + src_off, src_row_bytes);
                    }
                }
            }
        }
    }

    TR_LOG_DEBUG("task") << "transfer_to_rank completed: dt.id=" << dt.id
                         << " rank=" << rank
                         << " valid_bytes=" << valid_bytes
                         << (dt.is_compact() ? " compact" : " NON-compact");
}

void TaskBase::broadcast_from_rank0(const DTensor& dt) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip broadcast_from_rank0 " << dt.id
                  << (dt.is_compact() ? " (compact)" : " (NON-compact)")
                  << "\n";
        return;
    }
    check_compact_or_throw(dt, "broadcast_from_rank0");

    if (num_gpus_ <= 1) {
        return;
    }

#ifdef TR_USE_CUDA
#ifdef TR_USE_NCCL
    auto& reg = GlobalRegistry::instance();

    ncclDataType_t nccl_type;
    switch (dt.dtype) {
        case DType::FP32: nccl_type = ncclFloat32; break;
        case DType::FP16: nccl_type = ncclFloat16; break;
        case DType::INT32: nccl_type = ncclInt32; break;
        case DType::INT8:  nccl_type = ncclInt8; break;
        default:
            TR_TYPE_ERROR("broadcast_from_rank0 unsupported dtype: "
                         << static_cast<int>(dt.dtype));
    }

    ncclResult_t group_result = ncclGroupStart();
    if (group_result != ncclSuccess) {
        TR_DEVICE_ERROR("ncclGroupStart failed: " << ncclGetErrorString(group_result));
    }

    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaError_t err = cudaSetDevice(reg.gpu_ids()[rank]);
        if (err != cudaSuccess) {
            ncclGroupEnd();
            TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                            << ": " << cudaGetErrorString(err));
        }

        void* ptr = backend_->contexts[rank]->ptr_at(dt.id);
        cudaStream_t update_stream = static_cast<cudaStream_t>(
            backend_->contexts[rank]->stream(StreamKind::UPDATE));

        size_t bytes_to_broadcast = dt.slot_bytes();

        size_t nccl_count = bytes_to_broadcast;
        switch (dt.dtype) {
            case DType::FP32:  nccl_count /= 4; break;
            case DType::FP16:  nccl_count /= 2; break;
            case DType::INT32: nccl_count /= 4; break;
            case DType::INT8:  nccl_count /= 1; break;
            default: break;
        }

        ncclBroadcast(
            ptr,
            ptr,
            nccl_count,
            nccl_type,
            0,
            static_cast<ncclComm_t>(backend_->contexts[rank]->nccl_comm()),
            update_stream);
    }

    group_result = ncclGroupEnd();
    if (group_result != ncclSuccess) {
        TR_DEVICE_ERROR("ncclGroupEnd failed: " << ncclGetErrorString(group_result));
    }

    for (int rank = 0; rank < num_gpus_; ++rank) {
        backend_->contexts[rank]->synchronize_stream(StreamKind::UPDATE);
    }

    TR_LOG_DEBUG("task") << "broadcast_from_rank0 — completed NCCL broadcast dt.id="
                         << dt.id << " to " << num_gpus_ << " ranks";
#else
    TR_NOT_IMPLEMENTED("NCCL broadcast requested but TR_USE_NCCL is not defined");
#endif
#else
    (void)dt;
#endif
}

void TaskBase::transfer(const Tensor& host, const DTensor& dt) {
    // 广播到所有rank
    for (int rank = 0; rank < num_gpus_; ++rank) {
        transfer_to_rank(host, dt, rank);
    }
}

void TaskBase::transfer(const std::vector<Tensor>& hosts, const DTensor& dt) {
    TR_CHECK(hosts.size() == static_cast<size_t>(num_gpus_), ValueError,
             "Number of host tensors (" << hosts.size()
             << ") must match number of GPUs (" << num_gpus_ << ")");

    // 每个rank传输对应的host tensor
    for (int rank = 0; rank < num_gpus_; ++rank) {
        transfer_to_rank(hosts[rank], dt, rank);
    }
}

void TaskBase::fill(const DTensor& dt, float value) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip fill " << dt.id
                  << (dt.is_compact() ? " (compact)" : " (NON-compact)")
                  << "\n";
        return;
    }

    auto& reg = GlobalRegistry::instance();
    const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);
    int64_t n = live_dt.numel();

    if (reg.using_gpu()) {
#ifdef TR_USE_CUDA
        for (int rank = 0; rank < num_gpus_; ++rank) {
            void* dst = ArenaKeeper::instance().ptr_at(
                rank, static_cast<size_t>(live_dt.offset()));
            cudaSetDevice(reg.gpu_ids()[rank]);
            cudaStream_t comp = static_cast<cudaStream_t>(backend_->contexts[rank]->stream(StreamKind::COMP_1));
            cudaError_t err = launch_tr_fill_fp32_kernel(
                static_cast<float*>(dst), value, n, comp);
            TR_CHECK(err == cudaSuccess, DeviceError,
                     "fill kernel launch failed: rank=" << rank
                     << " error=" << cudaGetErrorString(err));
            cudaStreamSynchronize(comp);
        }
#endif
    } else {
        for (int rank = 0; rank < num_gpus_; ++rank) {
            void* dst = ArenaKeeper::instance().ptr_at(
                rank, static_cast<size_t>(live_dt.offset()));
            launch_tr_fill_fp32_kernel_cpu(static_cast<float*>(dst), value, n);
        }
    }
}

void TaskBase::zero(const DTensor& dt) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip zero " << dt.id
                  << (dt.is_compact() ? " (compact)" : " (NON-compact)")
                  << "\n";
        return;
    }

    auto& reg = GlobalRegistry::instance();
    const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);
    size_t num_bytes = static_cast<size_t>(live_dt.nbytes());

    if (reg.using_gpu()) {
#ifdef TR_USE_CUDA
        for (int rank = 0; rank < num_gpus_; ++rank) {
            void* dst = ArenaKeeper::instance().ptr_at(
                rank, static_cast<size_t>(live_dt.offset()));
            cudaSetDevice(reg.gpu_ids()[rank]);
            cudaStream_t comp = static_cast<cudaStream_t>(backend_->contexts[rank]->stream(StreamKind::COMP_1));
            cudaError_t err = cudaMemsetAsync(dst, 0, num_bytes, comp);
            TR_CHECK(err == cudaSuccess, DeviceError,
                     "zero cudaMemsetAsync failed: rank=" << rank
                     << " error=" << cudaGetErrorString(err));
            cudaStreamSynchronize(comp);
        }
#endif
    } else {
        for (int rank = 0; rank < num_gpus_; ++rank) {
            void* dst = ArenaKeeper::instance().ptr_at(
                rank, static_cast<size_t>(live_dt.offset()));
            std::memset(dst, 0, num_bytes);
        }
    }
}

void TaskBase::randn(const DTensor& dt, uint64_t seed) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip randn " << dt.id
                  << (dt.is_compact() ? " (compact)" : " (NON-compact)")
                  << "\n";
        return;
    }

    auto& reg = GlobalRegistry::instance();
    const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);
    int n = static_cast<int>(live_dt.numel());

    if (reg.using_gpu()) {
#ifdef TR_USE_CUDA
        for (int rank = 0; rank < num_gpus_; ++rank) {
            void* dst = ArenaKeeper::instance().ptr_at(
                rank, static_cast<size_t>(live_dt.offset()));
            cudaSetDevice(reg.gpu_ids()[rank]);
            cudaStream_t comp = static_cast<cudaStream_t>(backend_->contexts[rank]->stream(StreamKind::COMP_1));
            cudaError_t err = launch_tr_philox_normal_float_kernel(
                n, seed, 0, 0.f, 1.f, static_cast<float*>(dst), comp);
            TR_CHECK(err == cudaSuccess, DeviceError,
                     "randn kernel launch failed: rank=" << rank
                     << " error=" << cudaGetErrorString(err));
            cudaStreamSynchronize(comp);
        }
        return;
#endif
    }
    TR_NOT_IMPLEMENTED("randn CPU not yet implemented");
}

// ========== 初始化（COMPILED 阶段） ==========

void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");

    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) {
        return;
    }

    if (debug_mode_) {
        bool overridden = (cfg.kind != InitKind::NONE);
        std::cout << "[DRY RUN] init DTensor id=" << dtensor.id
                  << " kind=" << to_string(config.kind)
                  << " scale=" << config.scale
                  << " region=" << region_to_string(dtensor.region)
                  << " shape=" << dtensor.shape.to_string();
        if (overridden) {
            std::cout << " [OVERRIDE]";
        }
        std::cout << "\n";
        return;
    }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);

    Tensor host(live_dt.shape, live_dt.dtype);
    Initializer::apply_to_tensor(host, live_dt.shape, config);

    transfer_to_rank(host, live_dt, 0);
    if (num_gpus_ > 1) {
        broadcast_from_rank0(live_dt);
    }
}

void TaskBase::init_all() {
    check_phase(Phase::COMPILED, "init_all");

    if (debug_mode_) {
        std::cout << "[DRY RUN] init_all: " << active_memory_plan_->dtensors().size()
                  << " DTensors total\n";
    }

    // 第 1 步：标准初始化全部 DTensor（按各自的 init_config）
    // LOG_INFO << "[DEBUG] init_all: dtensors count=" << active_memory_plan_->dtensors().size();
    for (size_t i = 0; i < active_memory_plan_->dtensors().size(); ++i) {
        const auto& dt = active_memory_plan_->dtensors()[i];
        // LOG_INFO << "[DEBUG] dtensor[" << i << "] id=" << dt.id
        //          << " region=" << static_cast<int>(dt.region)
        //          << " shape=" << dt.shape.to_string()
        //          << " init=" << static_cast<int>(dt.init_config.kind);
    }
    for (const auto& dtensor : active_memory_plan_->dtensors()) {
        init(dtensor);
    }

    // 第 2 步：ZERO_GAMMA 策略 —— BN3 权重覆盖为 CONSTANTS(0.0)
    // NOTE: ZERO_GAMMA 当前未完全验证（BasicBlock 的 bn2 被错误标记为 BN3 等
    // 潜在问题），暂时禁用，待后续修复后再启用。
    if (initializer_.is_zero_gamma()) {
        TR_NOT_IMPLEMENTED("Zero Gamma is currently NOT supported.");
    }

    // 第 3 步：初始化 per-RANK dropout seed
    if (active_memory_plan_->baseline().dropout_seed >= 0) {
        uint64_t global_seed = get_default_generator().seed();
        for (int rank = 0; rank < num_gpus_; ++rank) {
            uint64_t z = global_seed + static_cast<uint64_t>(rank) + 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            uint64_t rank_seed = z ^ (z >> 31);

            int32_t seed_data[2] = {
                static_cast<int32_t>(rank_seed & 0xFFFFFFFFULL),
                static_cast<int32_t>(rank_seed >> 32)
            };

            Tensor host_seed(Shape{1, 1, 1, 2}, DType::INT32);
            memcpy(host_seed.data(), seed_data, sizeof(seed_data));
            transfer_to_rank(host_seed, active_memory_plan_->get_dtensor(
                active_memory_plan_->baseline().dropout_seed), rank);
        }
    }
}

Tensor TaskBase::fetch_from_rank(const DTensor& dt, int rank) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip fetch_from_rank " << dt.id
                  << " <- rank " << rank
                  << (dt.is_compact() ? " (compact)" : " (NON-compact)")
                  << "\n";
        return Tensor(dt.shape, dt.dtype);
    }

    // 参数验证
    TR_CHECK(dt.valid(), ValueError, "DTensor is invalid");
    TR_CHECK(rank >= 0 && rank < num_gpus_, IndexError,
             "Rank " << rank << " out of range [0, " << num_gpus_ << ")");

    // 获取源设备指针（必须用 memory_plan_ 中 finalize 后的最新 offset，
    // 传入的 dt 可能是 PLANNING 阶段的旧副本，其 offset 已失效）
    auto& reg = GlobalRegistry::instance();
    const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);
    const void* src = ArenaKeeper::instance().ptr_at(rank, static_cast<size_t>(live_dt.offset()));

    // 计算有效字节数（n×h×w×c，不包括padding）
    const uint64_t valid_bytes = dt.nbytes();

    if (reg.using_gpu()) {
#ifdef TR_USE_CUDA
        // GPU模式：区分是否需要布局转换
        Tensor result;

        if (dt.is_compact()) {
            // 紧凑DTensor：创建结果tensor，直接传输
            result = Tensor(dt.shape, dt.dtype);

            int physical_gpu_id = reg.gpu_ids()[rank];
            cudaError_t err = cudaSetDevice(physical_gpu_id);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                << ": " << cudaGetErrorString(err));
            }

            // 同步D2H传输
            err = cudaMemcpy(result.data<void>(), src, valid_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaMemcpy D2H from rank " << rank
                                << " failed: " << cudaGetErrorString(err));
            }
        } else {
            // 非紧凑DTensor：需要缓冲区进行重排中转
            const int64_t C_padded = dt.w_stride_cuda();
            const uint64_t padded_bytes = dt.n() * dt.h() * dt.w() * C_padded * DistributedTensor::dsize(dt.dtype);

            // 创建包含padding的临时缓冲区
            Tensor temp_buffer(Shape{static_cast<int>(dt.n()), static_cast<int>(dt.h()),
                                        static_cast<int>(dt.w()), static_cast<int>(C_padded)}, dt.dtype);

            int physical_gpu_id = reg.gpu_ids()[rank];
            cudaError_t err = cudaSetDevice(physical_gpu_id);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                << ": " << cudaGetErrorString(err));
            }

            // 从GPU传输包含padding的数据
            err = cudaMemcpy(temp_buffer.data<void>(), src, padded_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaMemcpy D2H (padded) from rank " << rank
                                << " failed: " << cudaGetErrorString(err));
            }

            // 创建紧凑的结果tensor
            result = Tensor(dt.shape, dt.dtype);

            // 执行非紧凑→紧凑重排
            const size_t elem_size = DistributedTensor::dsize(dt.dtype);
            const size_t src_row_bytes = C_padded * elem_size;
            const size_t dst_row_bytes = dt.c() * elem_size;
            const uint8_t* src_buffer = static_cast<const uint8_t*>(temp_buffer.data<void>());
            uint8_t* dst = static_cast<uint8_t*>(result.data<void>());

            for (int n = 0; n < dt.n(); ++n) {
                for (int h = 0; h < dt.h(); ++h) {
                    for (int w = 0; w < dt.w(); ++w) {
                        // 计算非紧凑布局的偏移（包含padding）
                        size_t src_off = (((n * dt.h() + h) * dt.w() + w) * C_padded) * elem_size;
                        // 计算紧凑布局的偏移
                        size_t dst_off = (((n * dt.h() + h) * dt.w() + w) * dt.c()) * elem_size;
                        std::memcpy(dst + dst_off, src_buffer + src_off, dst_row_bytes);
                    }
                }
            }
        }

        TR_LOG_DEBUG("task") << "fetch_from_rank completed: dt.id=" << dt.id
                             << " rank=" << rank
                             << " valid_bytes=" << valid_bytes
                             << (dt.is_compact() ? " compact" : " NON-compact");
        return result;
#else
        (void)src; (void)valid_bytes;
        TR_DEVICE_ERROR("GPU mode requested but TR_USE_CUDA is not defined");
        return Tensor(dt.shape, dt.dtype);
#endif
    } else {
        // CPU模式：直接memcpy有效字节数
        Tensor result(dt.shape, dt.dtype);

        if (dt.is_compact()) {
            // 紧凑DTensor：直接memcpy
            std::memcpy(result.data<void>(), src, valid_bytes);
        } else {
            // 非紧凑DTensor：需要重排
            const int64_t C_padded = dt.w_stride_cuda();
            const size_t elem_size = DistributedTensor::dsize(dt.dtype);
            const size_t dst_row_bytes = dt.c() * elem_size;
            const uint8_t* src_buffer = static_cast<const uint8_t*>(src);
            uint8_t* dst = static_cast<uint8_t*>(result.data<void>());

            for (int n = 0; n < dt.n(); ++n) {
                for (int h = 0; h < dt.h(); ++h) {
                    for (int w = 0; w < dt.w(); ++w) {
                        size_t src_off = (((n * dt.h() + h) * dt.w() + w) * C_padded) * elem_size;
                        size_t dst_off = (((n * dt.h() + h) * dt.w() + w) * dt.c()) * elem_size;
                        std::memcpy(dst + dst_off, src_buffer + src_off, dst_row_bytes);
                    }
                }
            }
        }

        TR_LOG_DEBUG("task") << "fetch_from_rank completed (CPU): dt.id=" << dt.id
                             << " rank=" << rank
                             << " valid_bytes=" << valid_bytes;
        return result;
    }
}

Tensor TaskBase::fetch(const DTensor& dt) {
    return fetch_from_rank(dt, 0);
}

// ========== 状态校验 ==========
void TaskBase::check_phase(Phase expected, const char* op) const {
    TR_CHECK(phase_ == expected, ValueError,
             "Operation '" << op << "' not allowed in phase "
             << static_cast<int>(phase_));
}

void TaskBase::create_temp_buffer_for_layout_conversion(Tensor&, const Tensor&, const DTensor&) {
    TR_NOT_IMPLEMENTED("Layout conversion not yet implemented");
}

// ========== 其余编译步骤（P_ULTIMATE.md 三阶段设计） ==========

void TaskBase::compile_alloc_hardware() {
    backend_ = std::make_unique<Backend>();

    auto& reg = GlobalRegistry::instance();
    size_t total_bytes = active_memory_plan_->total_bytes();

    if (reg.using_gpu()) {
        const auto& gpu_ids = reg.gpu_ids();

        // 初始化 ArenaKeeper：为每个 GPU 分配显存池
        ArenaKeeper::instance().initialize(true, gpu_ids, total_bytes);

        backend_->contexts.reserve(gpu_ids.size());
        for (int gpu_id : gpu_ids) {
            backend_->contexts.emplace_back(std::make_unique<DeviceContext>(gpu_id));
        }

        #ifdef TR_USE_NCCL
        // TODO: 当前默认单卡不初始化 NCCL（仅 gpu_ids.size() > 1）。
        // 若需测试 NCCL 初始化后对算子性能的影响，启用 CMake option TR_NCCL_SINGLE_GPU=ON
        // 即可让单卡也初始化 NCCL。注意：NCCL 初始化后会占用 GPU 资源（显存、SM、后台线程等），
        // 可能导致部分算子变慢。若发现某算子在多卡场景下性能明显不如单卡，根因很可能就是
        // 多卡初始化 NCCL 引入了资源竞争，拖慢了计算 kernel。此问题留待后续解决。
#if defined(TR_NCCL_SINGLE_GPU)
        if (gpu_ids.size() >= 1) {
#else
        if (gpu_ids.size() > 1) {
#endif
            std::vector<ncclComm_t> comms(gpu_ids.size());
            ncclResult_t nccl_result = ncclCommInitAll(
                comms.data(),
                static_cast<int>(gpu_ids.size()),
                gpu_ids.data());
            if (nccl_result != ncclSuccess) {
                TR_DEVICE_ERROR("ncclCommInitAll failed: " << ncclGetErrorString(nccl_result));
            }
            for (size_t i = 0; i < gpu_ids.size(); ++i) {
                backend_->contexts[i]->set_nccl_comm(comms[i]);
            }
            // TR_LOG_INFO("task") << "NCCL initialized for " << gpu_ids.size() << " GPUs";
        }
#endif

        // TR_LOG_INFO("task") << "Allocated " << gpu_ids.size()
        //                     << " GPU device context(s), "
        //                     << (total_bytes / (1024.0 * 1024.0)) << " MB each";
    } else {
        // CPU 模式：分配 mimalloc 内存池
        ArenaKeeper::instance().initialize(false, {0}, total_bytes);

        backend_->contexts.emplace_back(std::make_unique<DeviceContext>(-1));
        TR_LOG_INFO("task") << "Allocated CPU device context, "
                            << (total_bytes / (1024.0 * 1024.0)) << " MB";
    }
}

void TaskBase::compile_warmup_graphs() {
    LOG_INFO << "TaskBase::compile — warming up captured graphs";

    // Phase B4: 已在 pre_capture 中完成，这里仅做记录
    LOG_INFO << "TaskBase::compile — graph warmup completed (Phase B4)";
}

} // namespace tr
