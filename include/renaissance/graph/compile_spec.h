/**
 * @file compile_spec.h
 * @brief 编译参数集 —— 取代 Config 类名，描述单个编译变体的参数
 * @version 4.20.2
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/core/types.h, renaissance/graph/shape_id.h
 * @note 所属系列: graph
 * @note IMPORTANT.md 禁止任何名为 Config 的类名，统一使用 CompileSpec
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/core/global_registry.h"

namespace tr {

/**
 * @brief 单个编译变体的参数集
 *
 * 只含形状相关字段（决定 DTensor 的 shape/dtype 和 ComputationGraph 的拓扑）。
 * MemoryPlan 布局相关字段（bn_folded / use_lars / has_ema 等）在 PlanConfig 中，
 * 由 MemoryPlan::configure() 单独控制。
 *
 * 6 变体定义：
 *   variant 0: train_base         — train_res_begin × standard_batch
 *   variant 1: train_last         — train_res_begin × last_batch
 *   variant 2: train_lowres       — train_res_end   × standard_batch
 *   variant 3: train_lowres_last  — train_res_end   × last_batch
 *   variant 4: val_base           — val_res         × standard_batch
 *   variant 5: val_last           — val_res         × val_last_batch
 */
struct CompileSpec {
    bool amp_enabled = false;       ///< 混合精度（影响 DTensor::dtype）
    int  max_sample_resolution = 0; ///< MemoryPlan 最大槽位预留
    int  actual_resolution     = 0; ///< 该变体的形状推导分辨率
    int  batch_size            = 0; ///< 批次大小
    int  num_color_channels    = 0; ///< 颜色通道数
    bool freeze_first_layer = false; ///< 运行时标志，不影响 MemoryPlan/ComputationGraph

    /**
     * @brief 从 GlobalRegistry 构造 CompileSpec
     *
     * 读取全局配置中的 resolution / batch_size / amp 等参数，
     * 自动填充 CompileSpec 各字段。
     */
    static CompileSpec from_global_registry() {
        auto& gr = GlobalRegistry::instance();
        CompileSpec spec;
        spec.amp_enabled          = gr.using_amp();
        spec.max_sample_resolution = gr.max_sample_resolution();
        spec.actual_resolution     = gr.train_sample_resolution_begin();
        spec.batch_size            = gr.get_local_batch_size();
        spec.num_color_channels    = gr.num_color_channels();
        return spec;
    }

    /**
     * @brief 转换为 ShapeId 四元组
     *
     * ShapeId = { batch_size, actual_resolution, actual_resolution, num_color_channels }
     * 用于 Phase B 的 CapturedGraph 去重键。
     */
    ShapeId get_shape_id() const noexcept {
        return ShapeId{ batch_size, actual_resolution,
                        actual_resolution, num_color_channels };
    }
};

} // namespace tr