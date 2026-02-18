/*
 * Expert Solution A - GMX
 * NUMA-Aware GPU Worker Binding System
 *
 * Compile: g++ -std=c++17 -O3 -o bind_a a.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lpthread
 * Or: nvcc -std=c++17 -O3 -o bind_a a.cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <cmath>

// CUDA Runtime
#include <cuda_runtime.h>

// Linux System Headers
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>

// ============================================================
// Utility Class: String and File Processing
// ============================================================

class SysUtils {
public:
    // Read single-line file and trim whitespace
    static std::string read_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return "";
        std::string line;
        std::getline(file, line);
        return line;
    }

    // Parse CPU list string (e.g., "0-3,7,10-12")
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

    // Convert vector to compressed string format (for printing)
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

// ============================================================
// Core Structures: Hardware Topology
// ============================================================

struct PhysicalCore {
    int core_id;             // Physical core ID
    std::vector<int> logic_cpus; // Logical threads for this core (e.g., hyper-threading)
};

struct NumaNode {
    int id;
    std::vector<int> cpu_list; // All logical CPUs in this node
    // Physical core view: for intelligent allocation algorithm
    std::vector<PhysicalCore> physical_cores;
    std::vector<int> gpu_ids;  // GPU IDs in this node
};

struct GpuDevice {
    int id;
    int pci_bus;
    int pci_domain;
    int numa_node;
    std::string pci_dbdf; // Domain:Bus:Device.Function
    std::vector<int> affinity_cpus;
};

class HardwareTopology {
private:
    std::map<int, NumaNode> nodes_; // node_id -> Node
    std::vector<GpuDevice> gpus_;
    int num_logical_cpus_ = 0;

public:
    HardwareTopology() {
        discover_gpus();
        discover_numa_and_cpus();
    }

    // 1. Detect GPU information
    void discover_gpus() {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            GpuDevice gpu;
            gpu.id = i;
            gpu.pci_bus = prop.pciBusID;
            gpu.pci_domain = prop.pciDomainID;

            // Build SysFS path to find NUMA node
            // Format: 0000:01:00.0 (Domain:Bus:Device.Function)
            char dbdf[32];
            snprintf(dbdf, sizeof(dbdf), "%04x:%02x:%02x.0",
                     prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            gpu.pci_dbdf = std::string(dbdf);

            std::string path_base = "/sys/bus/pci/devices/" + gpu.pci_dbdf;

            // Read NUMA node
            std::string numa_str = SysUtils::read_file(path_base + "/numa_node");
            try {
                int node = std::stoi(numa_str);
                gpu.numa_node = (node < 0) ? 0 : node; // Handle single node or unconfigured
            } catch (...) {
                gpu.numa_node = 0;
            }

            // Read local affinity CPUs
            std::string local_cpus = SysUtils::read_file(path_base + "/local_cpulist");
            gpu.affinity_cpus = SysUtils::parse_cpu_list(local_cpus);

            gpus_.push_back(gpu);
        }
    }

    // 2. Detect NUMA and CPU physical/logical architecture
    void discover_numa_and_cpus() {
        num_logical_cpus_ = sysconf(_SC_NPROCESSORS_ONLN);

        // Traverse possible NUMA nodes (typically Linux exposes in /sys/devices/system/node/nodeX)
        // For safety, we scan the directory
        DIR* dir = opendir("/sys/devices/system/node/");
        if (!dir) {
            // Exception fallback: assume single node
            setup_single_node_fallback();
            return;
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name.rfind("node", 0) == 0 && std::isdigit(name[4])) { // starts with "node" followed by digit
                int node_id = std::stoi(name.substr(4));

                NumaNode node;
                node.id = node_id;

                // Read CPU list for this node
                std::string cpulist = SysUtils::read_file("/sys/devices/system/node/" + name + "/cpulist");
                node.cpu_list = SysUtils::parse_cpu_list(cpulist);

                // Build physical core view (handle hyper-threading)
                build_physical_core_view(node);

                nodes_[node_id] = node;
            }
        }
        closedir(dir);

        // Register GPUs to corresponding NUMA nodes
        for (const auto& gpu : gpus_) {
            if (nodes_.count(gpu.numa_node)) {
                nodes_[gpu.numa_node].gpu_ids.push_back(gpu.id);
            } else {
                // If GPU's reported node doesn't exist (e.g., -1 mapped to 0 but system only has node 1?? rare), fallback to first node
                if (!nodes_.empty()) nodes_.begin()->second.gpu_ids.push_back(gpu.id);
            }
        }
    }

    // Helper: Parse physical core topology
    void build_physical_core_view(NumaNode& node) {
        std::map<int, std::vector<int>> core_map; // physical_id -> [logical_ids]

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
            // Sort logical CPUs, typically smaller ID is primary thread, larger is hyper-thread
            std::sort(pc.logic_cpus.begin(), pc.logic_cpus.end());
            node.physical_cores.push_back(pc);
        }
    }

    void setup_single_node_fallback() {
        NumaNode node;
        node.id = 0;
        for (int i=0; i<num_logical_cpus_; ++i) node.cpu_list.push_back(i);
        build_physical_core_view(node);
        nodes_[0] = node;
    }

    // Getters
    const std::vector<GpuDevice>& get_gpus() const { return gpus_; }
    const std::map<int, NumaNode>& get_nodes() const { return nodes_; }
    int get_num_gpus() const { return gpus_.size(); }
    int get_num_nodes() const { return nodes_.size(); }
    int get_num_cpus() const { return num_logical_cpus_; }
};

// ============================================================
// Strategy Class: Allocation Algorithm
// ============================================================

class Scheduler {
    HardwareTopology& topo_;
    // Record allocation count per NUMA node for Round-Robin algorithm
    std::map<int, int> node_usage_counter_;

public:
    Scheduler(HardwareTopology& topo) : topo_(topo) {}

    // Core algorithm: Assign best CPU for a task on target_node
    int pick_best_core(int target_node) {
        if (topo_.get_nodes().find(target_node) == topo_.get_nodes().end()) {
            // If target node doesn't exist (e.g., GPU on node 1 but machine only has node 0), fallback to node 0
            target_node = topo_.get_nodes().begin()->first;
        }

        const auto& node = topo_.get_nodes().at(target_node);
        const auto& phys_cores = node.physical_cores;

        if (phys_cores.empty()) return -1; // Should not happen

        // Get which request this is for the node
        int request_idx = node_usage_counter_[target_node]++;

        // Algorithm goal: Prioritize filling all physical cores' first thread, then second thread...
        // This avoids two workers squeezing on same physical core until necessary.

        int num_phys = phys_cores.size();

        // Determine physical core index (Round Robin across physical cores)
        int phys_idx = request_idx % num_phys;

        // Determine hyper-threading level (Depth)
        // E.g.: if requests 0-15, depth=0 (primary thread)
        //        if requests 16-31, depth=1 (hyper-thread)
        int ht_depth = request_idx / num_phys;

        const auto& target_phys_core = phys_cores[phys_idx];

        // If requested depth exceeds logical threads this physical core has (e.g., only single thread), wrap around
        int logic_idx = ht_depth % target_phys_core.logic_cpus.size();

        return target_phys_core.logic_cpus[logic_idx];
    }
};

// ============================================================
// Main Program Logic
// ============================================================

bool is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [gpu_ids] [num_workers]\n";
    std::cout << "  gpu_ids    : Comma-separated GPU IDs (e.g., 0,1,2,3)\n";
    std::cout << "  num_workers : Total number of worker threads\n";
    std::cout << "\nIf no arguments provided, will enter interactive mode.\n";
}

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "   NUMA-Aware GPU Worker Binder\n";
    std::cout << "            (Expert Solution A)\n";
    std::cout << "========================================\n\n";

    // 1. Detect hardware
    std::cout << "[Phase 1] Detecting Hardware Topology...\n";
    HardwareTopology topo;

    if (topo.get_num_gpus() == 0) {
        std::cerr << "Error: No CUDA GPUs detected!\n";
        return -1;
    }

    // 2. Print basic information (Requirement 1 & 2)
    std::cout << "Detected " << topo.get_num_cpus() << " CPU cores, "
              << topo.get_num_nodes() << " NUMA nodes, "
              << topo.get_num_gpus() << " GPUs.\n\n";

    for (const auto& gpu : topo.get_gpus()) {
        std::cout << "GPU " << std::setw(2) << gpu.id
                  << " | NUMA Node: " << std::setw(2) << gpu.numa_node
                  << " | Affinity CPUs: " << SysUtils::compress_cpu_list(gpu.affinity_cpus)
                  << std::endl;
    }
    std::cout << std::endl;

    // 3. User input (Requirement 3)
    std::cout << "[Phase 2] Configuration\n";
    std::vector<int> user_gpu_ids;
    int total_workers = 0;

    if (argc == 3) {
        // Command line mode
        user_gpu_ids = SysUtils::parse_cpu_list(argv[1]);
        try {
            total_workers = std::stoi(argv[2]);
        } catch (...) {
            std::cerr << "Error: Invalid worker count\n";
            print_usage(argv[0]);
            return -1;
        }
    } else {
        // Interactive mode
        std::cout << "Enter GPU IDs to use (comma separated, e.g., '0,1,2,3'): ";
        std::string line;
        std::getline(std::cin, line);
        std::stringstream ss(line);
        int val;
        std::set<int> unique_gpus;
        while (ss >> val) {
            if (val >= 0 && val < topo.get_num_gpus()) unique_gpus.insert(val);
        }
        user_gpu_ids.assign(unique_gpus.begin(), unique_gpus.end());
        std::sort(user_gpu_ids.begin(), user_gpu_ids.end());

        std::cout << "Enter total worker threads: ";
        std::cin >> total_workers;
    }

    int n_working_gpu = user_gpu_ids.size();

    // Validation: GPU count must be power of 2 and < 16
    if (n_working_gpu == 0 || n_working_gpu >= 16 || !is_power_of_two(n_working_gpu)) {
        std::cerr << "Error: Selected GPU count (" << n_working_gpu
                  << ") must be a power of 2 and < 16 (and > 0).\n";
        return -1;
    }

    // Validation: Worker count must be multiple of GPU count
    if (total_workers <= 0 || total_workers % n_working_gpu != 0) {
        std::cerr << "Error: Total workers must be a multiple of active GPU count.\n";
        return -1;
    }

    // 4. Generate and print allocation plan (Requirement 3 & 4)
    std::cout << "\n[Phase 3] Binding Plan Generation\n";
    std::cout << "Strategy:\n";
    std::cout << "  1. Strided GPU Mapping (worker % n_gpus)\n";
    std::cout << "  2. Physical Core Priority (Avoid Hyper-Thread sharing if possible)\n";
    std::cout << "  3. NUMA Locality (Thread pinned to GPU's NUMA node)\n\n";

    std::cout << std::string(75, '-') << "\n";
    std::cout << std::left << std::setw(10) << "WorkerID"
              << std::setw(10) << "RealGPU"
              << std::setw(10) << "NUMA"
              << std::setw(12) << "BindCore"
              << "Note" << std::endl;
    std::cout << std::string(75, '-') << "\n";

    Scheduler scheduler(topo);

    // Build quick lookup: GPU ID -> NUMA Node
    std::map<int, int> gpu_to_node;
    for (const auto& g : topo.get_gpus()) gpu_to_node[g.id] = g.numa_node;

    for (int w = 0; w < total_workers; ++w) {
        // A. Determine which GPU this Worker is responsible for
        // Algorithm: Strided allocation, ensures load balancing
        int virt_idx = w % n_working_gpu;
        int real_gpu = user_gpu_ids[virt_idx];

        // B. Determine target NUMA node
        int target_node = gpu_to_node[real_gpu];

        // C. Allocate optimal CPU core
        int assigned_core = scheduler.pick_best_core(target_node);

        std::cout << std::left << std::setw(10) << w
                  << std::setw(10) << real_gpu
                  << std::setw(10) << target_node
                  << std::setw(12) << assigned_core
                  << "  (GPU " << real_gpu << " Local)" << std::endl;
    }
    std::cout << std::string(75, '-') << "\n";

    std::cout << "\n[Implementation Guide: How to apply this in your worker thread]\n";
    std::cout << "1. In 'preprocess_worker(int worker_id)':\n";
    std::cout << "   - Look up 'assigned_core' from table above.\n";
    std::cout << "   - Call: \n";
    std::cout << "       cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(assigned_core, &cpuset);\n";
    std::cout << "       pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);\n";
    std::cout << "2. Immediately allocate workspace:\n";
    std::cout << "   - void* ptr = aligned_alloc(4096, size);\n";
    std::cout << "   - memset(ptr, 0, size); // <--- Triggers Linux First-Touch Policy\n";
    std::cout << "   - Now 'ptr' is physically resident on NUMA Node " << gpu_to_node[user_gpu_ids[0]] << " (if tied to GPU " << user_gpu_ids[0] << ").\n";
    std::cout << "3. Pinned Memory:\n";
    std::cout << "   - CUDA driver will allocate pinned memory on GPU's NUMA node automatically.\n\n";

    return 0;
}
