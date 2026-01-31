// EXPERT_OP_V4 - Direct I/O优化方案 + 全局随机洗牌
// 基于e.cpp，在准备阶段对所有文件进行一次全局随机洗牌

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <filesystem>
#include <random>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <dirent.h>
    #include <sched.h>
    #include <errno.h>
#endif

namespace fs = std::filesystem;

constexpr size_t KB = 1024ULL;
constexpr size_t MB = 1024ULL * KB;
constexpr size_t GB = 1024ULL * MB;
constexpr size_t ALIGNMENT_16KB = 16 * KB;
constexpr size_t ALIGNMENT_4KB = 4 * KB;
constexpr size_t SECTOR_SIZE = 4 * KB;

struct FileInfo {
    std::string path;
    size_t size;
};

struct ThreadContext {
    int thread_id;
    int num_threads;
    char* buffer;
    size_t buffer_size;
    size_t remaining;
    std::vector<FileInfo>* files;  // 指向分配给该线程的文件列表
    bool oom;
    std::mutex* print_mutex;
};

std::mutex g_print_mutex;

inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

char* allocate_numa_buffer(size_t size, int thread_id, int num_threads) {
    (void)thread_id;
    (void)num_threads;
    char* buffer = nullptr;

#ifdef _WIN32
    buffer = static_cast<char*>(VirtualAlloc(
        nullptr,
        size,
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE
    ));
#else
    if (posix_memalign(reinterpret_cast<void**>(&buffer), ALIGNMENT_4KB, size) != 0) {
        buffer = nullptr;
    }
#endif

    if (buffer) {
        const size_t page_size = 4096;
        for (size_t i = 0; i < size; i += page_size) {
            buffer[i] = 0;
        }
    }

    return buffer;
}

void free_numa_buffer(char* buffer, size_t size) {
    if (!buffer) return;

#ifdef _WIN32
    VirtualFree(buffer, 0, MEM_RELEASE);
#else
    (void)size;
    free(buffer);
#endif
}

void bind_thread_to_core(int thread_id, int num_threads) {
#ifdef _WIN32
    DWORD_PTR mask = 1ULL << (thread_id % 64);
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    (void)num_threads;
    int num_cpus = std::thread::hardware_concurrency();
    if (num_cpus > 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id % num_cpus, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#endif
}

#ifdef _WIN32
bool fast_read_file(const std::string& path, char* dest, size_t file_size, size_t buffer_capacity) {
    size_t aligned_size = align_up(file_size, SECTOR_SIZE);

    if (aligned_size > buffer_capacity) {
        return false;
    }

    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    DWORD bytes_read = 0;
    BOOL success = ReadFile(hFile, dest, static_cast<DWORD>(file_size), &bytes_read, nullptr);
    CloseHandle(hFile);
    return success && (bytes_read == file_size);
}
#else
bool fast_read_file(const std::string& path, char* dest, size_t file_size, size_t buffer_capacity) {
    size_t aligned_size = align_up(file_size, SECTOR_SIZE);

    if (aligned_size > buffer_capacity) {
        return false;
    }

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }

    size_t total_read = 0;
    while (total_read < file_size) {
        ssize_t bytes_read = read(fd, dest + total_read, file_size - total_read);

        if (bytes_read <= 0) {
            if (bytes_read < 0 && errno == EINTR) continue;
            break;
        }

        total_read += bytes_read;
    }

    close(fd);
    return total_read == file_size;
}
#endif

void worker_thread(ThreadContext* ctx) {
    bind_thread_to_core(ctx->thread_id, ctx->num_threads);

    char* current_pos = ctx->buffer;
    size_t remaining = ctx->buffer_size;
    ctx->oom = false;

    // 🔧 阶段4：处理已随机洗牌的文件列表
    for (const auto& file_info : *ctx->files) {
        size_t aligned_file_size = align_up(file_info.size, ALIGNMENT_16KB);

        size_t current_offset = current_pos - ctx->buffer;
        size_t aligned_offset = align_up(current_offset, ALIGNMENT_16KB);
        size_t padding = aligned_offset - current_offset;

        size_t total_needed = padding + aligned_file_size;

        if (total_needed > remaining) {
            ctx->oom = true;
            {
                std::lock_guard<std::mutex> lock(*ctx->print_mutex);
                std::cout << "Thread " << ctx->thread_id << " OOM!" << std::endl;
            }
            ctx->remaining = remaining;
            return;
        }

        current_pos += padding;
        remaining -= padding;

        if (!fast_read_file(file_info.path, current_pos, file_info.size, remaining)) {
            continue;
        }

        current_pos += aligned_file_size;
        remaining -= aligned_file_size;
    }

    ctx->remaining = remaining;

    {
        std::lock_guard<std::mutex> lock(*ctx->print_mutex);
        std::cout << "Thread " << ctx->thread_id << " finished loading, remaining memory "
                  << ctx->remaining << " Bytes." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Program started..." << std::endl;

    std::string base_path;
    int num_workers = 0;
    unsigned int shuffle_seed = 42;  // 默认随机种子

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            base_path = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            shuffle_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        }
    }

    if (base_path.empty()) {
        std::cerr << "Error: --path is required" << std::endl;
        std::cerr << "Usage: " << argv[0] << " --path <directory> --workers <1|2|4|8|16> [--seed <N>]" << std::endl;
        return 1;
    }

    size_t buffer_size = (256ULL * GB) / num_workers;

    // ========================================
    // 探测阶段（不计时）：收集所有文件
    // ========================================
    std::vector<fs::path> folders;
    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_directory()) {
            folders.push_back(entry.path());
        }
    }
    std::sort(folders.begin(), folders.end());

    // 收集所有文件到单一列表
    std::vector<FileInfo> all_files;
    for (const auto& folder : folders) {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                all_files.push_back({entry.path().string(), entry.file_size()});
            }
        }
    }

    std::cout << "Collected " << all_files.size() << " files." << std::endl;

    // ========================================
    // 🔧 阶段4：全局随机洗牌（准备阶段）
    // ========================================
    std::cout << "Shuffling files with seed: " << shuffle_seed << "..." << std::endl;

    std::mt19937 rng(shuffle_seed);
    std::shuffle(all_files.begin(), all_files.end(), rng);

    std::cout << "Shuffle completed." << std::endl;

    // ========================================
    // 分配洗牌后的文件给各个线程
    // ========================================
    std::vector<std::vector<FileInfo>> per_thread_files(num_workers);
    std::vector<char*> buffers(num_workers);
    std::vector<ThreadContext> contexts(num_workers);

    // 平均分配文件给线程
    size_t files_per_thread = (all_files.size() + num_workers - 1) / num_workers;
    for (int i = 0; i < num_workers; ++i) {
        size_t start_idx = i * files_per_thread;
        size_t end_idx = std::min(start_idx + files_per_thread, all_files.size());

        for (size_t j = start_idx; j < end_idx; ++j) {
            per_thread_files[i].push_back(all_files[j]);
        }
    }

    // 分配缓冲区
    for (int i = 0; i < num_workers; ++i) {
        buffers[i] = allocate_numa_buffer(buffer_size, i, num_workers);
        if (!buffers[i]) {
            std::cerr << "Failed to allocate buffer for thread " << i << std::endl;
            for (int j = 0; j < i; ++j) {
                free_numa_buffer(buffers[j], buffer_size);
            }
            return 1;
        }

        contexts[i].thread_id = i;
        contexts[i].num_threads = num_workers;
        contexts[i].buffer = buffers[i];
        contexts[i].buffer_size = buffer_size;
        contexts[i].remaining = buffer_size;
        contexts[i].files = &per_thread_files[i];
        contexts[i].oom = false;
        contexts[i].print_mutex = &g_print_mutex;
    }

    std::cout << "Loading..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(num_workers);

    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(worker_thread, &contexts[i]);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "All thread joined, total time: " << duration.count() << " ms." << std::endl;

    // 释放内存
    for (int i = 0; i < num_workers; ++i) {
        free_numa_buffer(buffers[i], buffer_size);
    }

    return 0;
}
