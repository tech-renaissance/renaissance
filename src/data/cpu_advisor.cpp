/**
 * @file cpu_advisor.cpp
 * @brief CPU分配顾问实现
 * @version 1.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 */

// ============================================================================
// 仅在GPU_CLOUD场景下编译CPU分配顾问
// ============================================================================
#if defined(TR_SCENE_GPU_CLOUD)

#include "renaissance/data/cpu_advisor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

namespace tr {

CPUAdvisor::CPUAdvisor(const HardwareTopology& topo)
    : topo_(topo)
{
    LOG_INFO << "CPUAdvisor initialized with "
             << topo_.get_num_nodes() << " NUMA nodes, "
             << topo_.get_num_cpus() << " CPUs, "
             << topo_.get_gpus().size() << " GPUs";
}

int CPUAdvisor::pick_best_core(int target_node) {
    // 验证目标节点存在
    if (topo_.get_nodes().find(target_node) == topo_.get_nodes().end()) {
        LOG_WARN << "CPUAdvisor: Target NUMA node " << target_node << " not found, "
                 << "falling back to node 0";
        target_node = topo_.get_nodes().begin()->first;
    }

    const auto& node = topo_.get_nodes().at(target_node);
    const auto& phys_cores = node.physical_cores;

    if (phys_cores.empty()) {
        TR_THROW(ValueError, "CPUAdvisor: No physical cores found in NUMA node " << target_node);
    }

    // 获取该节点的请求计数
    int request_idx = node_usage_counter_[target_node]++;

    // 核心算法（参考a.cpp第294-311行）：
    // 1. Round Robin跨物理核
    int num_phys = static_cast<int>(phys_cores.size());
    int phys_idx = request_idx % num_phys;

    // 2. 计算超线程深度
    int ht_depth = request_idx / num_phys;

    const auto& target_phys_core = phys_cores[phys_idx];

    // 3. 选择逻辑线程
    // 如果超线程深度超过该物理核的逻辑线程数，循环使用
    int logic_idx = ht_depth % static_cast<int>(target_phys_core.logic_cpus.size());

    int assigned_cpu = target_phys_core.logic_cpus[logic_idx];

    LOG_DEBUG << "CPUAdvisor: node=" << target_node
              << ", request_idx=" << request_idx
              << ", phys_idx=" << phys_idx
              << ", ht_depth=" << ht_depth
              << ", assigned_cpu=" << assigned_cpu;

    return assigned_cpu;
}

void CPUAdvisor::reset_node_counter(int node_id) {
    node_usage_counter_[node_id] = 0;
    LOG_DEBUG << "CPUAdvisor: Reset counter for NUMA node " << node_id;
}

} // namespace tr

#endif // TR_SCENE_GPU_CLOUD
