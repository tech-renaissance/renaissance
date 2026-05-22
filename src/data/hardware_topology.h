/**
 * @file hardware_topology.h
 * @brief 硬件拓扑发现和CPU绑核规划
 * @version 1.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 * @note 所属系列: data
 * @note 完整移植自 tests/bind/a.cpp，用 TR_SCENE_GPU_CLOUD 宏保护
 */

#pragma once

#if defined(TR_SCENE_GPU_CLOUD)

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include "renaissance/core/logger.h"

namespace tr {

// ============================================================================
// SysUtils - 字符串和文件处理工具类
// ============================================================================

class SysUtils {
public:
    /**
     * @brief 读取单行文件并去除空白字符
     */
    static std::string read_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return "";
        std::string line;
        std::getline(file, line);
        return line;
    }

    /**
     * @brief 解析CPU列表字符串（如 "0-3,7,10-12"）
     */
    static std::vector<int> parse_cpu_list(const std::string& list_str) {
        std::vector<int> cpus;
        if (list_str.empty()) return cpus;

        std::stringstream ss(list_str);
        std::string segment;

        while (std::getline(ss, segment, ',')) {
            size_t dash = segment.find('-');
            if (dash != std::string::npos) {
                try {
                    int start = std::stoi(segment.substr(0, dash));
                    int end = std::stoi(segment.substr(dash + 1));
                    for (int i = start; i <= end; ++i) cpus.push_back(i);
                } catch (...) {}
            } else {
                try {
                    if (!segment.empty()) cpus.push_back(std::stoi(segment));
                } catch (...) {}
            }
        }
        std::sort(cpus.begin(), cpus.end());
        return cpus;
    }

    /**
     * @brief 将vector转换为压缩字符串格式（用于打印）
     */
    static std::string compress_cpu_list(const std::vector<int>& cpus) {
        if (cpus.empty()) return "None";
        std::stringstream ss;
        int range_start = cpus[0];
        int range_end = cpus[0];
        bool first = true;

        auto flush_range = [&](int start, int end) {
            if (!first) ss << ",";
            if (start == end) ss << start;
            else ss << start << "-" << end;
            first = false;
        };

        for (size_t i = 1; i < cpus.size(); ++i) {
            if (cpus[i] == range_end + 1) {
                range_end = cpus[i];
            } else {
                flush_range(range_start, range_end);
                range_start = cpus[i];
                range_end = cpus[i];
            }
        }
        flush_range(range_start, range_end);
        return ss.str();
    }
};

// ============================================================================
// 核心结构体：硬件拓扑
// ============================================================================

/**
 * @brief 物理核心信息
 */
struct PhysicalCore {
    int core_id;             ///< 物理核心ID
    std::vector<int> logic_cpus; ///< 该核心的逻辑线程（如超线程）
};

/**
 * @brief NUMA节点信息
 */
struct NumaNode {
    int id;
    std::vector<int> cpu_list;     ///< 该节点的所有逻辑CPU
    std::vector<PhysicalCore> physical_cores; ///< 物理核心视图
    std::vector<int> gpu_ids;      ///< 该节点的GPU ID列表
};

/**
 * @brief GPU设备信息
 */
struct GpuDevice {
    int id;
    int pci_bus;
    int pci_domain;
    int numa_node;
    std::string pci_dbdf;         ///< Domain:Bus:Device.Function
    std::vector<int> affinity_cpus; ///< 本地亲和性CPU列表
};

// ============================================================================
// HardwareTopology - 硬件拓扑发现类
// ============================================================================

/**
 * @brief 硬件拓扑发现和管理
 */
class HardwareTopology {
private:
    std::map<int, NumaNode> nodes_; ///< node_id -> Node
    std::vector<GpuDevice> gpus_;
    int num_logical_cpus_ = 0;

public:
    HardwareTopology();

    // Getters
    const std::vector<GpuDevice>& get_gpus() const { return gpus_; }
    const std::map<int, NumaNode>& get_nodes() const { return nodes_; }
    int get_num_gpus() const { return gpus_.size(); }
    int get_num_nodes() const { return nodes_.size(); }
    int get_num_cpus() const { return num_logical_cpus_; }

private:
    /**
     * @brief 探测GPU信息
     */
    void discover_gpus();

    /**
     * @brief 探测NUMA和CPU物理/逻辑架构
     */
    void discover_numa_and_cpus();

    /**
     * @brief 构建物理核心拓扑
     */
    void build_physical_core_view(NumaNode& node);

    /**
     * @brief 单节点回退
     */
    void setup_single_node_fallback();
};

// ============================================================================
// CpuBindingPlanner - CPU绑核规划类
// ============================================================================

/**
 * @brief CPU绑核规划器（原Scheduler类，避免命名冲突）
 */
class CpuBindingPlanner {
private:
    HardwareTopology& topo_;
    std::map<int, int> node_usage_counter_; ///< NUMA节点的使用计数器

public:
    CpuBindingPlanner(HardwareTopology& topo) : topo_(topo) {}

    /**
     * @brief 为目标NUMA节点上的任务分配最佳CPU核心
     * @param target_node 目标NUMA节点
     * @return 分配的CPU核心ID
     */
    int pick_best_core(int target_node) {
        if (topo_.get_nodes().find(target_node) == topo_.get_nodes().end()) {
            // 目标节点不存在，回退到第一个节点
            target_node = topo_.get_nodes().begin()->first;
        }

        const auto& node = topo_.get_nodes().at(target_node);
        const auto& phys_cores = node.physical_cores;

        if (phys_cores.empty()) return -1;

        // 获取该节点的请求索引
        int request_idx = node_usage_counter_[target_node]++;

        // 算法目标：优先填满所有物理核心的主线程，再填超线程
        int num_phys = phys_cores.size();

        // 确定物理核心索引（Round Robin）
        int phys_idx = request_idx % num_phys;

        // 确定超线程深度
        int ht_depth = request_idx / num_phys;

        const auto& target_phys_core = phys_cores[phys_idx];

        // 如果请求深度超过该物理核心的逻辑线程数，回绕
        int logic_idx = ht_depth % target_phys_core.logic_cpus.size();

        return target_phys_core.logic_cpus[logic_idx];
    }
};

} // namespace tr

#endif  // TR_SCENE_GPU_CLOUD
