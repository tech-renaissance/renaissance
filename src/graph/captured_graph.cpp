/**
 * @file captured_graph.cpp
 * @brief CapturedGraph实现——双后端图捕获与运行
 * @version 4.21.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 基于P_ULTIMATE.md设计，实现双后端CapturedGraph
 */

#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/graph_executor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <exception>

namespace tr {

// ============================================================================
// 前向声明：CPU捕获函数（在capture_cpu.cpp中实现）
// ============================================================================
void capture_cpu(const ComputationGraph& cg,
                 const MemoryPlan& mp,
                 GraphId gid,
                 const DeviceContext& ctx,
                 std::vector<CpuOp>& cpu_ops);

// 前向声明：pre_capture 内部使用的单rank捕获辅助函数
void capture_all_for_rank(PreCaptureResult& result,
                           const std::vector<CapturedGraph::Key>& key_by_idx,
                           const std::unordered_map<CapturedGraph::Key, const MemoryPlan*,
                                                    CapturedGraph::KeyHash>& key_to_mp,
                           DeviceContext& ctx,
                           int rank);

// ============================================================================
// capture() 实现 —— 双后端统一入口
// ============================================================================

CapturedGraph CapturedGraph::capture(const ComputationGraph& cg,
                                      const MemoryPlan& mp,
                                      GraphId gid,
                                      ShapeId shape_id,
                                      StreamKind stream_kind,
                                      const DeviceContext& ctx) {
    CapturedGraph result;
    result.key_ = Key{&cg, gid, shape_id};

    if (ctx.is_gpu()) {
        result.is_cuda_ = true;
        result.per_rank_execs_.resize(1, nullptr);
#ifdef TR_USE_CUDA
        capture_cuda(cg, mp, gid, stream_kind, ctx, result);
#endif
    } else {
        result.is_cuda_ = false;
        capture_cpu(cg, mp, gid, ctx, result.cpu_ops_);
    }

    return result;
}

CapturedGraph CapturedGraph::capture(const ComputationGraph& cg,
                                      const MemoryPlan& mp,
                                      GraphId gid,
                                      StreamKind stream,
                                      const DeviceContext& ctx) {
    return capture(cg, mp, gid, kShapeInvariant, stream, ctx);
}

CapturedGraph::~CapturedGraph() {
    if (is_cuda()) {
#ifdef TR_USE_CUDA
        for (auto exec : per_rank_execs_) {
            if (exec) {
                cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec));
            }
        }
#endif
    } else {
        for (auto& op : cpu_ops_) {
            if (op.ctx) {
                delete static_cast<CpuOpContext*>(op.ctx);
            }
        }
    }
}

void CapturedGraph::set_rank_exec(int rank, NativeGraph exec) {
    if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()) {
        per_rank_execs_[rank] = exec;
    }
}

NativeGraph CapturedGraph::release_rank_exec(int rank) {
    if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()) {
        NativeGraph exec = per_rank_execs_[rank];
        per_rank_execs_[rank] = nullptr;
        return exec;
    }
    return nullptr;
}

std::string CapturedGraph::debug_dump() const {
    std::ostringstream oss;
    oss << "CapturedGraph identity:\n"
        << "  cg_ptr=" << key_.cg << "\n"
        << "  gid=" << static_cast<int>(key_.gid) << "\n"
        << "  shape_id=" << key_.shape.to_string() << "\n"
        << "  backend=" << (is_cuda_ ? "CUDA" : "CPU") << "\n";

    if (is_cuda_) {
        oss << "  num_ranks=" << per_rank_execs_.size() << "\n";
        for (size_t i = 0; i < per_rank_execs_.size(); ++i) {
            oss << "  rank_" << i << "_exec=" << per_rank_execs_[i] << "\n";
        }
    } else {
        oss << "  cpu_ops=" << cpu_ops_.size() << "\n";
    }

    if (key_.cg) {
        oss << "  graph_nodes=" << key_.cg->total_node_count() << "\n";
    }

    if ((is_cuda_ && per_rank_execs_.empty()) ||
        (!is_cuda_ && cpu_ops_.empty())) {
        oss << "  status=PLACEHOLDER (no real capture yet)\n";
    }

    return oss.str();
}

void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()) {
            NativeGraph exec = per_rank_execs_[rank];
            if (exec) {
                cudaError_t err = cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec),
                                                  static_cast<cudaStream_t>(stream));
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaGraphLaunch failed: " << cudaGetErrorString(err));
                }
            } else {
                TR_DEVICE_ERROR("cudaGraphLaunch: exec is nullptr for graph");
            }
        }
#endif
    } else {
        (void)rank;
        (void)stream;
        for (const auto& op : cpu_ops_) {
            if (op.fn) {
                op.fn(static_cast<CpuOpContext*>(op.ctx));
            }
        }
    }
}

PreCaptureResult pre_capture(const GraphAtlas& compile_atlas,
                              const std::vector<DeviceContext*>& contexts) {
    PreCaptureResult result;
    result.atlas = compile_atlas;

    const int num_ranks = static_cast<int>(contexts.size());

    // =====================================================================
    // Phase B1: 去重
    // =====================================================================
    std::unordered_map<CapturedGraph::Key, int32_t, CapturedGraph::KeyHash> seen;
    std::unordered_map<CapturedGraph::Key, const MemoryPlan*, CapturedGraph::KeyHash> key_to_mp;

    for (size_t vi = 0; vi < GraphAtlas::kMaxVariants; ++vi) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            auto& slot = result.atlas.slot(vi, gi);
            if (!slot.cg || !slot.mp) continue;

            CapturedGraph::Key key{slot.cg, static_cast<GraphId>(gi), slot.shape_id};
            key_to_mp[key] = slot.mp;

            auto it = seen.find(key);
            if (it != seen.end()) {
                slot.captured_idx = it->second;
                ++result.reused;
            } else {
                slot.captured_idx = static_cast<int32_t>(seen.size());
                seen[key] = slot.captured_idx;
                ++result.captured;
            }
            ++result.total_slots;
        }
    }

    const int32_t K = static_cast<int32_t>(seen.size());
    result.graphs.resize(K);
    for (auto& cg : result.graphs) {
        cg.reserve_ranks(num_ranks);
    }

    std::vector<CapturedGraph::Key> key_by_idx(K);
    for (const auto& [key, idx] : seen) {
        key_by_idx[idx] = key;
    }

    // =====================================================================
    // Phase B2: cuDNN预热 — 每个 rank 独立串行预热
    // cuDNN Frontend 的 kernel/plan cache 是 per-device 的，
    // 必须每个 rank 独立填充，确保捕获阶段只有纯 execute() → cache命中。
    // 串行执行避免 cuDNN 全局锁竞争。
    // =====================================================================
    const bool is_cuda = (contexts[0]->is_gpu());
    if (is_cuda && K > 0) {
#ifdef TR_USE_CUDA
        for (int rank = 0; rank < num_ranks; ++rank) {
            int gpu_id = contexts[rank]->device_id();
            cudaSetDevice(gpu_id);
            contexts[rank]->set_rank(rank);

            TR_LOG_INFO("graph") << "[B2] cuDNN warmup on rank " << rank
                                 << " (GPU " << gpu_id << ") for "
                                 << K << " unique graphs";

            for (int32_t k = 0; k < K; ++k) {
                const auto& key = key_by_idx[k];
                if (!key.cg) continue;

                auto it = key_to_mp.find(key);
                TR_CHECK(it != key_to_mp.end() && it->second != nullptr, ValueError,
                         "No MemoryPlan for key[" << k << "]");
                const MemoryPlan* mp = it->second;
                contexts[rank]->set_memory_plan(mp);

                int node_idx = 0;
                for (const auto& node : key.cg->nodes(key.gid)) {
                    if (node.kind != GraphNode::Kind::COMPUTE) { ++node_idx; continue; }
                    if (!require_warmup(node.compute_op)) { ++node_idx; continue; }
                    TR_LOG_INFO("graph") << "[B2-DBG] warmup k=" << k
                                         << " gid=" << static_cast<int>(key.gid)
                                         << " node=" << node_idx
                                         << " op=" << static_cast<int>(node.compute_op);
                    warmup_single_cudnn_op(node, *mp, *contexts[rank]);
                    ++node_idx;
                }
            }
            cudaDeviceSynchronize();
        }
        TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed on all "
                             << num_ranks << " ranks";
#endif
    }

    // =====================================================================
    // Phase B3: 捕获（0串行 + 1~N-1并行）
    // =====================================================================
    std::vector<std::exception_ptr> exc(num_ranks);

    // Rank 0 在主线程串行捕获
    try {
        TR_LOG_INFO("graph") << "[B3] Capturing rank 0 in main thread";
        ::tr::capture_all_for_rank(result, key_by_idx, key_to_mp, *contexts[0], 0);
    } catch (...) {
        exc[0] = std::current_exception();
    }

    // Rank 1~N-1 并行捕获
    if (num_ranks > 1) {
        TR_LOG_INFO("graph") << "[B3] Capturing ranks 1~" << (num_ranks-1)
                             << " in parallel threads";
        std::vector<std::thread> threads;
        for (int r = 1; r < num_ranks; ++r) {
            threads.emplace_back([&, r]() {
                try {
                    ::tr::capture_all_for_rank(result, key_by_idx, key_to_mp,
                                         *contexts[r], r);
                } catch (...) {
                    exc[r] = std::current_exception();
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    // 异常聚合：任意rank失败都要终止
    for (int r = 0; r < num_ranks; ++r) {
        if (exc[r]) {
            std::rethrow_exception(exc[r]);
        }
    }

    // =====================================================================
    // Phase B4: warmup launch（仅Rank 0）
    // =====================================================================
    if (is_cuda && K > 0) {
#ifdef TR_USE_CUDA
        TR_LOG_INFO("graph") << "[B4] Warmup launching " << K << " graphs on rank 0";
        cudaSetDevice(contexts[0]->device_id());

        for (const auto& cg : result.graphs) {
            StreamKind sk = StreamKind::COMP_1;
            for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
                const auto& sl = result.atlas.slot(0, gi);
                if (static_cast<GraphId>(gi) == cg.key().gid && sl.cg == cg.key().cg) {
                    sk = sl.stream_kind;
                    break;
                }
            }
            void* stream = contexts[0]->stream(sk);
            cg.launch(0, stream);
        }

        contexts[0]->synchronize_all();
        TR_LOG_INFO("graph") << "[B4] Warmup launch completed";
#endif
    }

    TR_LOG_INFO("graph") << "[PRE_CAPTURE] Summary: " << result.total_slots
                         << " slots, " << result.captured
                         << " captured, " << result.reused << " reused, "
                         << K << " unique graphs, " << num_ranks << " ranks";

    return result;
}

// ============================================================================
// 内部辅助函数：单rank捕获
// ============================================================================

void capture_all_for_rank(PreCaptureResult& result,
                           const std::vector<CapturedGraph::Key>& key_by_idx,
                           const std::unordered_map<CapturedGraph::Key, const MemoryPlan*,
                                                    CapturedGraph::KeyHash>& key_to_mp,
                           DeviceContext& ctx,
                           int rank) {
    ctx.set_rank(rank);

#ifdef TR_USE_CUDA
    if (ctx.is_gpu()) {
        cudaSetDevice(ctx.device_id());
    }
#endif

    const int32_t K = static_cast<int32_t>(key_by_idx.size());
    TR_LOG_DEBUG("graph") << "[CAPTURE] Rank " << rank << ": capturing "
                          << K << " graphs";

    for (int32_t k = 0; k < K; ++k) {
        const auto& key = key_by_idx[k];
        auto it = key_to_mp.find(key);
        TR_CHECK(it != key_to_mp.end() && it->second != nullptr, RuntimeError,
                 "No MemoryPlan found for key");

        const MemoryPlan* mp = it->second;
        ctx.set_memory_plan(mp);

        StreamKind stream_kind = StreamKind::COMP_1;
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            const auto& sl = result.atlas.slot(0, gi);
            if (static_cast<GraphId>(gi) == key.gid && sl.cg == key.cg) {
                stream_kind = sl.stream_kind;
                break;
            }
        }

        auto cg = CapturedGraph::capture(*key.cg, *mp, key.gid, key.shape, stream_kind, ctx);

        // rank 0 负责初始化 result.graphs[k] 的完整元数据
        if (rank == 0) {
            result.graphs[k].set_metadata_from(cg);
            result.graphs[k].move_cpu_ops_from(cg);
        }

        // CUDA 路径：转移 exec 句柄
        if (cg.is_cuda()) {
            result.graphs[k].set_rank_exec(rank, cg.release_rank_exec(0));
        }

        TR_LOG_DEBUG("graph") << "[CAPTURE] Rank " << rank << ": graph " 
                              << static_cast<int>(key.gid) << " captured";
    }
}

} // namespace tr