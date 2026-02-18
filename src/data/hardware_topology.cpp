/**
 * @file hardware_topology.cpp
 * @brief 硬件拓扑发现和CPU绑核规划实现
 * @version 1.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 */

#if defined(TR_SCENE_GPU_CLOUD)

#include "renaissance/data/hardware_topology.h"
#include "renaissance/base/logger.h"

#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tr {

HardwareTopology::HardwareTopology() {
#ifdef TR_USE_CUDA
    discover_gpus();
#endif
    discover_numa_and_cpus();
}

#ifdef TR_USE_CUDA

void HardwareTopology::discover_gpus() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        LOG_WARN << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
        return;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            LOG_WARN << "cudaGetDeviceProperties failed for GPU " << i;
            continue;
        }

        GpuDevice gpu;
        gpu.id = i;
        gpu.pci_bus = prop.pciBusID;
        gpu.pci_domain = prop.pciDomainID;

        char dbdf[32];
        snprintf(dbdf, sizeof(dbdf), "%04x:%02x:%02x.0",
                 prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        gpu.pci_dbdf = std::string(dbdf);

        std::string path_base = "/sys/bus/pci/devices/" + gpu.pci_dbdf;

        std::string numa_str = SysUtils::read_file(path_base + "/numa_node");
        try {
            int node = std::stoi(numa_str);
            gpu.numa_node = (node < 0) ? 0 : node;
        } catch (...) {
            gpu.numa_node = 0;
        }

        std::string local_cpus = SysUtils::read_file(path_base + "/local_cpulist");
        gpu.affinity_cpus = SysUtils::parse_cpu_list(local_cpus);

        gpus_.push_back(gpu);
    }
}

#endif

void HardwareTopology::discover_numa_and_cpus() {
    num_logical_cpus_ = sysconf(_SC_NPROCESSORS_ONLN);

    DIR* dir = opendir("/sys/devices/system/node/");
    if (!dir) {
        setup_single_node_fallback();
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind("node", 0) == 0 && std::isdigit(name[4])) {
            int node_id = std::stoi(name.substr(4));

            NumaNode node;
            node.id = node_id;

            std::string cpulist = SysUtils::read_file("/sys/devices/system/node/" + name + "/cpulist");
            node.cpu_list = SysUtils::parse_cpu_list(cpulist);

            build_physical_core_view(node);

            nodes_[node_id] = node;
        }
    }
    closedir(dir);

    for (const auto& gpu : gpus_) {
        if (nodes_.count(gpu.numa_node)) {
            nodes_[gpu.numa_node].gpu_ids.push_back(gpu.id);
        } else {
            if (!nodes_.empty()) {
                nodes_.begin()->second.gpu_ids.push_back(gpu.id);
            }
        }
    }
}

void HardwareTopology::build_physical_core_view(NumaNode& node) {
    std::map<int, std::vector<int>> core_map;

    for (int cpu_id : node.cpu_list) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + "/topology/core_id";
        std::string core_val = SysUtils::read_file(path);

        int phy_id = -1;
        if (!core_val.empty()) {
            try { phy_id = std::stoi(core_val); } catch(...) { phy_id = cpu_id; }
        } else {
            phy_id = cpu_id;
        }

        core_map[phy_id].push_back(cpu_id);
    }

    for (auto& pair : core_map) {
        PhysicalCore pc;
        pc.core_id = pair.first;
        pc.logic_cpus = pair.second;
        std::sort(pc.logic_cpus.begin(), pc.logic_cpus.end());
        node.physical_cores.push_back(pc);
    }
}

void HardwareTopology::setup_single_node_fallback() {
    NumaNode node;
    node.id = 0;
    for (int i = 0; i < num_logical_cpus_; ++i) node.cpu_list.push_back(i);
    build_physical_core_view(node);
    nodes_[0] = node;
}

} // namespace tr

#endif
