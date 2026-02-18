/*
 * Expert Solution F - SNX
 * NUMA-Aware GPU Thread Binding System
 *
 * Compile: g++ -std=c++17 -O3 -o bind_f f.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lpthread
 * Or: nvcc -std=c++17 -O3 -o bind_f f.cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <cstring>
#include <cassert>

// CUDA
#include <cuda_runtime.h>

// Linux system calls
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>

// ============================================================================
// Utility Functions
// ============================================================================

namespace Utils {

// Trim whitespace from string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

// Read single-line file
std::string readFileLine(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::string line;
    std::getline(file, line);
    return trim(line);
}

// Parse CPU list format (e.g., "0-3,7,10-12")
std::vector<int> parseCpuList(const std::string& str) {
    std::vector<int> result;
    if (str.empty()) return result;

    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) continue;

        size_t dash = token.find('-');
        if (dash != std::string::npos) {
            try {
                int start = std::stoi(token.substr(0, dash));
                int end = std::stoi(token.substr(dash + 1));
                for (int i = start; i <= end; ++i) {
                    result.push_back(i);
                }
            } catch (...) {
                std::cerr << "Warning: Failed to parse range '" << token << "'" << std::endl;
            }
        } else {
            try {
                result.push_back(std::stoi(token));
            } catch (...) {
                std::cerr << "Warning: Failed to parse number '" << token << "'" << std::endl;
            }
        }
    }

    // Sort and deduplicate
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

// Check if power of two
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Format CPU list as compressed string
std::string formatCpuList(const std::vector<int>& cpus) {
    if (cpus.empty()) return "";

    std::vector<int> sorted = cpus;
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream oss;
    int range_start = sorted[0];
    int range_end = sorted[0];

    for (size_t i = 1; i <= sorted.size(); ++i) {
        if (i < sorted.size() && sorted[i] == range_end + 1) {
            range_end = sorted[i];
        } else {
            if (range_start == range_end) {
                oss << range_start;
            } else if (range_end == range_start + 1) {
                oss << range_start << "," << range_end;
            } else {
                oss << range_start << "-" << range_end;
            }
            if (i < sorted.size()) {
                oss << ",";
                range_start = sorted[i];
                range_end = sorted[i];
            }
        }
    }

    return oss.str();
}

} // namespace Utils

// ============================================================================
// CPU Core Information Structure
// ============================================================================

struct CpuCoreInfo {
    int logical_id;      // Logical CPU ID (OS-visible ID)
    int physical_id;     // Physical core ID (core_id)
    int socket_id;       // Socket ID (physical CPU slot)
    int numa_node;       // NUMA node ID

    CpuCoreInfo() : logical_id(-1), physical_id(-1), socket_id(-1), numa_node(-1) {}
};

// ============================================================================
// NUMA Node Information
// ============================================================================

struct NumaNodeInfo {
    int node_id;
    std::vector<int> all_cpus;           // All logical CPUs
    std::vector<int> physical_cores;     // Physical cores (one representative per core)
    std::set<int> gpu_ids;               // GPUs on this node

    NumaNodeInfo() : node_id(-1) {}
};

// ============================================================================
// GPU Information
// ============================================================================

struct GpuInfo {
    int gpu_id;
    int numa_node;
    int pci_bus;
    int pci_domain;
    std::string pci_address;
    std::vector<int> affinity_cpus;      // Affine CPU list

    GpuInfo() : gpu_id(-1), numa_node(-1), pci_bus(-1), pci_domain(-1) {}
};

// ============================================================================
// System Topology Detector
// ============================================================================

class SystemTopology {
private:
    std::vector<NumaNodeInfo> numa_nodes_;
    std::vector<GpuInfo> gpus_;
    std::map<int, CpuCoreInfo> cpu_info_map_;  // logical_id -> CpuCoreInfo
    int total_cpus_;

public:
    SystemTopology() : total_cpus_(0) {
        discoverCpuTopology();
        discoverNumaNodes();
        discoverGpuTopology();
    }

    // Get system information
    int getNumCpus() const { return total_cpus_; }
    int getNumNumaNodes() const { return numa_nodes_.size(); }
    int getNumGpus() const { return gpus_.size(); }

    const std::vector<GpuInfo>& getGpus() const { return gpus_; }
    const std::vector<NumaNodeInfo>& getNumaNodes() const { return numa_nodes_; }

    const GpuInfo* getGpuInfo(int gpu_id) const {
        for (const auto& gpu : gpus_) {
            if (gpu.gpu_id == gpu_id) return &gpu;
        }
        return nullptr;
    }

    const NumaNodeInfo* getNumaNodeInfo(int node_id) const {
        for (const auto& node : numa_nodes_) {
            if (node.node_id == node_id) return &node;
        }
        return nullptr;
    }

private:
    // Detect CPU topology
    void discoverCpuTopology() {
        total_cpus_ = sysconf(_SC_NPROCESSORS_ONLN);

        for (int cpu = 0; cpu < total_cpus_; ++cpu) {
            CpuCoreInfo info;
            info.logical_id = cpu;

            // Read core_id
            std::string core_id_str = Utils::readFileLine(
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id");
            if (!core_id_str.empty()) {
                try {
                    info.physical_id = std::stoi(core_id_str);
                } catch (...) {
                    info.physical_id = cpu;
                }
            } else {
                info.physical_id = cpu;
            }

            // Read physical_package_id (socket)
            std::string socket_str = Utils::readFileLine(
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/physical_package_id");
            if (!socket_str.empty()) {
                try {
                    info.socket_id = std::stoi(socket_str);
                } catch (...) {
                    info.socket_id = 0;
                }
            } else {
                info.socket_id = 0;
            }

            cpu_info_map_[cpu] = info;
        }
    }

    // Detect NUMA nodes
    void discoverNumaNodes() {
        std::set<int> detected_nodes;

        // Method 1: Detect from /sys/devices/system/node/
        DIR* dir = opendir("/sys/devices/system/node");
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name.substr(0, 4) == "node" && name.length() > 4) {
                    try {
                        int node_id = std::stoi(name.substr(4));
                        detected_nodes.insert(node_id);
                    } catch (...) {}
                }
            }
            closedir(dir);
        }

        // If no NUMA nodes detected, create default
        if (detected_nodes.empty()) {
            detected_nodes.insert(0);
        }

        // Build NUMA node information
        for (int node_id : detected_nodes) {
            NumaNodeInfo node;
            node.node_id = node_id;

            // Read CPU list for this node
            std::string cpulist_path = "/sys/devices/system/node/node" +
                                       std::to_string(node_id) + "/cpulist";
            std::string cpulist = Utils::readFileLine(cpulist_path);

            if (!cpulist.empty()) {
                node.all_cpus = Utils::parseCpuList(cpulist);
            } else if (node_id == 0) {
                // Single node system, include all CPUs
                for (int i = 0; i < total_cpus_; ++i) {
                    node.all_cpus.push_back(i);
                }
            }

            // Update NUMA info in cpu_info_map_
            for (int cpu : node.all_cpus) {
                if (cpu_info_map_.find(cpu) != cpu_info_map_.end()) {
                    cpu_info_map_[cpu].numa_node = node_id;
                }
            }

            // Identify physical cores (select one logical CPU per physical core)
            extractPhysicalCores(node);

            numa_nodes_.push_back(node);
        }

        // Sort
        std::sort(numa_nodes_.begin(), numa_nodes_.end(),
                  [](const NumaNodeInfo& a, const NumaNodeInfo& b) {
                      return a.node_id < b.node_id;
                  });
    }

    // Extract physical cores (avoid hyper-threading interference)
    void extractPhysicalCores(NumaNodeInfo& node) {
        // Use (socket_id, physical_id) as unique identifier
        std::map<std::pair<int, int>, int> core_map;  // (socket, core) -> first logical_id

        for (int cpu : node.all_cpus) {
            if (cpu_info_map_.find(cpu) == cpu_info_map_.end()) continue;

            const auto& info = cpu_info_map_[cpu];
            auto key = std::make_pair(info.socket_id, info.physical_id);

            if (core_map.find(key) == core_map.end()) {
                core_map[key] = cpu;
            } else {
                // Select smaller logical_id as representative
                if (cpu < core_map[key]) {
                    core_map[key] = cpu;
                }
            }
        }

        // Extract physical core list
        for (const auto& pair : core_map) {
            node.physical_cores.push_back(pair.second);
        }

        // Sort
        std::sort(node.physical_cores.begin(), node.physical_cores.end());
    }

    // Detect GPU topology
    void discoverGpuTopology() {
        int gpu_count = 0;
        cudaError_t err = cudaGetDeviceCount(&gpu_count);

        if (err != cudaSuccess) {
            std::cerr << "Warning: CUDA error when getting device count: "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }

        for (int i = 0; i < gpu_count; ++i) {
            GpuInfo info;
            info.gpu_id = i;

            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, i);
            if (err != cudaSuccess) {
                std::cerr << "Warning: Cannot get properties for GPU " << i << std::endl;
                continue;
            }

            info.pci_bus = prop.pciBusID;
            info.pci_domain = prop.pciDomainID;

            // Build PCI address
            char pci_addr[32];
            snprintf(pci_addr, sizeof(pci_addr), "%04x:%02x:00.0",
                     prop.pciDomainID, prop.pciBusID);
            info.pci_address = pci_addr;

            // Read NUMA node
            std::string numa_path = "/sys/bus/pci/devices/" + info.pci_address + "/numa_node";
            std::string numa_str = Utils::readFileLine(numa_path);

            if (!numa_str.empty()) {
                try {
                    info.numa_node = std::stoi(numa_str);
                    if (info.numa_node < 0) info.numa_node = 0;
                } catch (...) {
                    info.numa_node = 0;
                }
            } else {
                info.numa_node = 0;
            }

            // Read CPU affinity
            std::string affinity_path = "/sys/bus/pci/devices/" + info.pci_address + "/local_cpulist";
            std::string affinity_str = Utils::readFileLine(affinity_path);

            if (!affinity_str.empty()) {
                info.affinity_cpus = Utils::parseCpuList(affinity_str);
            } else {
                // Fallback: use all CPUs in this NUMA node
                const auto* node = getNumaNodeInfo(info.numa_node);
                if (node) {
                    info.affinity_cpus = node->all_cpus;
                }
            }

            gpus_.push_back(info);

            // Update GPU list for NUMA nodes
            for (auto& node : numa_nodes_) {
                if (node.node_id == info.numa_node) {
                    node.gpu_ids.insert(info.gpu_id);
                    break;
                }
            }
        }
    }
};

// ============================================================================
// Worker Assignment Structure
// ============================================================================

struct WorkerAssignment {
    int worker_id;
    int gpu_id;
    int numa_node;
    int cpu_core;
    bool is_physical_core;  // Whether bound to independent physical core
};

// ============================================================================
// Worker Assignment Planner
// ============================================================================

class WorkerPlanner {
private:
    const SystemTopology& topology_;
    std::vector<int> selected_gpus_;
    int num_workers_;
    std::vector<WorkerAssignment> assignments_;

    // Core allocation state per NUMA node
    struct NodeAllocationState {
        std::vector<int> available_physical_cores;
        std::vector<int> available_logical_cores;
        size_t physical_index;
        size_t logical_index;

        NodeAllocationState() : physical_index(0), logical_index(0) {}
    };

    std::map<int, NodeAllocationState> node_states_;

public:
    WorkerPlanner(const SystemTopology& topology)
        : topology_(topology), num_workers_(0) {}

    // Validate and compute allocation plan
    bool plan(const std::vector<int>& gpu_ids, int num_workers) {
        // 1. Validate GPU selection
        if (!validateGpuSelection(gpu_ids)) {
            return false;
        }

        // 2. Validate worker count
        if (!validateWorkerCount(num_workers, gpu_ids.size())) {
            return false;
        }

        selected_gpus_ = gpu_ids;
        num_workers_ = num_workers;

        // 3. Initialize allocation states
        initializeAllocationStates();

        // 4. Compute assignments
        computeAssignments();

        return true;
    }

    // Print allocation plan
    void printPlan() const {
        std::cout << "\n" << std::string(75, '-') << "\n";
        std::cout << std::left << std::setw(10) << "WorkerID"
                  << std::setw(10) << "RealGPU"
                  << std::setw(10) << "NUMA"
                  << std::setw(12) << "BindCore"
                  << "Note" << std::endl;
        std::cout << std::string(75, '-') << "\n";

        for (const auto& a : assignments_) {
            std::cout << std::left << std::setw(10) << a.worker_id
                      << std::setw(10) << a.gpu_id
                      << std::setw(10) << a.numa_node
                      << std::setw(12) << a.cpu_core
                      << "  (GPU " << a.gpu_id << " Local)" << std::endl;
        }
        std::cout << std::string(75, '-') << "\n";
    }

private:
    bool validateGpuSelection(const std::vector<int>& gpu_ids) {
        if (gpu_ids.empty()) {
            std::cerr << "Error: No GPUs selected\n";
            return false;
        }

        // Check if power of 2
        if (!Utils::isPowerOfTwo(gpu_ids.size())) {
            std::cerr << "Error: Number of GPUs must be a power of 2, got "
                      << gpu_ids.size() << "\n";
            return false;
        }

        // Check maximum
        if (gpu_ids.size() > 16) {
            std::cerr << "Error: Maximum 16 GPUs supported, got "
                      << gpu_ids.size() << "\n";
            return false;
        }

        // Check if exceeds available GPUs
        if (gpu_ids.size() > static_cast<size_t>(topology_.getNumGpus())) {
            std::cerr << "Error: Requested " << gpu_ids.size()
                      << " GPUs but only " << topology_.getNumGpus()
                      << " available\n";
            return false;
        }

        // Check GPU ID validity and duplicates
        std::set<int> unique_ids;
        for (int gpu_id : gpu_ids) {
            if (gpu_id < 0 || gpu_id >= topology_.getNumGpus()) {
                std::cerr << "Error: Invalid GPU ID " << gpu_id
                          << " (valid range: 0-" << topology_.getNumGpus() - 1 << ")\n";
                return false;
            }
            if (unique_ids.count(gpu_id)) {
                std::cerr << "Error: Duplicate GPU ID " << gpu_id << "\n";
                return false;
            }
            unique_ids.insert(gpu_id);
        }

        return true;
    }

    bool validateWorkerCount(int num_workers, size_t num_gpus) {
        if (num_workers <= 0) {
            std::cerr << "Error: Number of workers must be positive\n";
            return false;
        }

        if (num_workers % num_gpus != 0) {
            std::cerr << "Error: Number of workers (" << num_workers
                      << ") must be a multiple of number of GPUs (" << num_gpus << ")\n";
            return false;
        }

        return true;
    }

    void initializeAllocationStates() {
        node_states_.clear();

        for (const auto& node : topology_.getNumaNodes()) {
            NodeAllocationState state;
            state.available_physical_cores = node.physical_cores;
            state.available_logical_cores = node.all_cpus;
            node_states_[node.node_id] = state;
        }
    }

    void computeAssignments() {
        assignments_.clear();
        assignments_.reserve(num_workers_);

        int num_gpus = selected_gpus_.size();

        // Strided allocation strategy: gpu_id = worker_id % num_gpus
        for (int worker_id = 0; worker_id < num_workers_; ++worker_id) {
            WorkerAssignment assignment;
            assignment.worker_id = worker_id;

            // 1. Determine GPU for this worker
            assignment.gpu_id = selected_gpus_[worker_id % num_gpus];

            // 2. Determine NUMA node
            const auto* gpu_info = topology_.getGpuInfo(assignment.gpu_id);
            if (gpu_info) {
                assignment.numa_node = gpu_info->numa_node;
            } else {
                assignment.numa_node = 0;  // Fallback
            }

            // 3. Allocate optimal CPU core
            allocateCpuCore(assignment);

            assignments_.push_back(assignment);
        }
    }

    void allocateCpuCore(WorkerAssignment& assignment) {
        auto it = node_states_.find(assignment.numa_node);
        if (it == node_states_.end()) {
            // Fallback: allocate any CPU
            assignment.cpu_core = 0;
            assignment.is_physical_core = false;
            return;
        }

        NodeAllocationState& state = it->second;

        // Prioritize physical cores
        if (state.physical_index < state.available_physical_cores.size()) {
            assignment.cpu_core = state.available_physical_cores[state.physical_index];
            assignment.is_physical_core = true;
            state.physical_index++;
        } else {
            // Physical cores exhausted, use logical cores (might be hyper-threading)
            if (state.logical_index < state.available_logical_cores.size()) {
                assignment.cpu_core = state.available_logical_cores[state.logical_index];
                assignment.is_physical_core = false;
                state.logical_index++;
            } else {
                // All exhausted, round-robin reuse
                size_t idx = state.logical_index % state.available_logical_cores.size();
                assignment.cpu_core = state.available_logical_cores[idx];
                assignment.is_physical_core = false;
                state.logical_index++;
            }
        }
    }
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [gpu_ids] [num_workers]\n";
    std::cout << "  gpu_ids    : Comma-separated GPU IDs (e.g., 0,1,2,3)\n";
    std::cout << "  num_workers : Total number of worker threads\n";
    std::cout << "\nIf no arguments provided, will enter interactive mode.\n";
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n========================================\n";
    std::cout << "   NUMA-Aware GPU Worker Binder\n";
    std::cout << "            (Expert Solution F)\n";
    std::cout << "========================================\n";

    // 1. Detect system topology
    SystemTopology topology;

    if (topology.getNumGpus() == 0) {
        std::cerr << "\nError: No CUDA-capable GPUs detected\n";
        std::cerr << "Please ensure CUDA drivers are properly installed.\n";
        return 1;
    }

    // Print topology information
    std::cout << "\nDetected " << topology.getNumCpus() << " CPU cores, "
              << topology.getNumNumaNodes() << " NUMA nodes, "
              << topology.getNumGpus() << " GPUs.\n";

    for (const auto& gpu : topology.getGpus()) {
        std::cout << "GPU " << std::setw(2) << gpu.gpu_id
                  << " | NUMA Node: " << std::setw(2) << gpu.numa_node
                  << " | Affinity CPUs: " << Utils::formatCpuList(gpu.affinity_cpus)
                  << std::endl;
    }
    std::cout << std::endl;

    // 2. Get user input
    std::vector<int> selected_gpus;
    int num_workers = 0;

    if (argc == 3) {
        // Command line mode
        selected_gpus = Utils::parseCpuList(argv[1]);
        try {
            num_workers = std::stoi(argv[2]);
        } catch (...) {
            std::cerr << "Error: Invalid worker count\n";
            print_usage(argv[0]);
            return 1;
        }
    } else {
        // Interactive mode
        std::cout << "Enter GPU IDs to use (comma-separated, e.g., 0,1,2,3): ";
        std::string gpu_input;
        std::getline(std::cin, gpu_input);
        selected_gpus = Utils::parseCpuList(gpu_input);

        std::cout << "Enter total number of worker threads: ";
        std::cin >> num_workers;
    }

    // 3. Compute allocation plan
    WorkerPlanner planner(topology);
    if (!planner.plan(selected_gpus, num_workers)) {
        std::cerr << "\nError: Failed to create allocation plan\n";
        return 1;
    }

    // 4. Print plan
    planner.printPlan();

    std::cout << "\n[Implementation Guide]\n";
    std::cout << "1. Thread binding: pthread_setaffinity_np()\n";
    std::cout << "2. Workspace allocation: aligned_alloc() + memset() for first-touch\n";
    std::cout << "3. Pinned memory: cudaHostAlloc() allocates on GPU's NUMA node\n\n";

    std::cout << "Planning completed successfully!\n\n";

    return 0;
}
