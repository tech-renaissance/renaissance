/**
 * @file cpu_advisor.h
 * @brief CPU分配顾问（NUMA感知的核心绑定算法）
 * @version 1.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 * @note 所属系列: data
 * @details
 * 参考tests/bind/a.cpp的Scheduler类
 * - 基于物理核的Round Robin分配
 * - 避免超线程争用（优先用主线程）
 * - NUMA本地化（绑定到GPU所在NUMA节点）
 */

#pragma once

#include "renaissance/data/hardware_topology.h"
#include <map>

// ============================================================================
// 仅在GPU_CLOUD场景下启用CPU分配顾问
// ============================================================================
#if defined(TR_SCENE_GPU_CLOUD)

namespace tr {

/**
 * @class CPUAdvisor
 * @brief NUMA感知的CPU分配顾问
 * @details
 * 核心算法（参考a.cpp第279-312行）：
 * 1. 跨物理核Round Robin分配
 * 2. 超线程深度控制（先用主线程，再用超线程）
 * 3. NUMA本地化（优先使用目标NUMA节点的CPU）
 */
class CPUAdvisor {
public:
    explicit CPUAdvisor(const HardwareTopology& topo);

    // 禁止拷贝
    CPUAdvisor(const CPUAdvisor&) = delete;
    CPUAdvisor& operator=(const CPUAdvisor&) = delete;

    /**
     * @brief 为任务选择最佳CPU核心
     * @param target_node 目标NUMA节点
     * @return 分配的逻辑CPU ID
     * @details
     * 算法流程：
     * 1. 获取该节点的请求计数
     * 2. 物理核Round Robin: request_idx % num_phys
     * 3. 超线程深度: request_idx / num_phys
     * 4. 选择逻辑线程: ht_depth % logic_cpus.size()
     *
     * 这样确保：
     * - 优先填满所有物理核的主线程
     * - 其次使用超线程
     * - 避免多个worker争用同一个物理核
     */
    int pick_best_core(int target_node);

    /**
     * @brief 重置某个NUMA节点的使用计数
     * @param node_id NUMA节点ID
     */
    void reset_node_counter(int node_id);

private:
    const HardwareTopology& topo_;
    std::map<int, int> node_usage_counter_;  // node_id -> 请求计数
};

} // namespace tr

#endif // TR_SCENE_GPU_CLOUD
