// EXPERT_SN - Direct I/O + NUMA库方案
// 使用O_DIRECT和numa_alloc_onnode，需要链接-lnuma（Linux）

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <atomic>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <errno.h>
#endif

namespace fs = std::filesystem;

constexpr size_t ALIGNMENT = 16 * 1024;
constexpr size_t TOTAL_MEMORY = 256ULL * 1024 * 1024 * 1024;

struct FileInfo {
    fs::path path;
    size_t size;
};

struct ThreadTask {
    int thread_id;
    std::vector<std::vector<FileInfo>> folders;
    char* buffer;
    size_t buffer_size;
    std::atomic<size_t>* remaining_bytes;
    std::atomic<bool>* oom_flag;
};

inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void* allocate_numa_memory(size_t size, int /* thread_id */) {
#ifdef _WIN32
    return _aligned_malloc(size, ALIGNMENT);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, size) == 0) {
        return ptr;
    }
    return nullptr;
#endif
}

void free_numa_memory(void* ptr, size_t size) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    (void)size;
    free(ptr);
#endif
}

#ifdef _WIN32
bool read_file_fast(const fs::path& filepath, char* buffer, size_t& offset, size_t buffer_size) {
    HANDLE hFile = CreateFileA(filepath.string().c_str(), GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);

    if (hFile == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        return false;
    }

    size_t aligned_offset = align_up(offset, ALIGNMENT);
    if (aligned_offset + fileSize.QuadPart > buffer_size) {
        CloseHandle(hFile);
        return false;
    }

    DWORD bytesRead;
    if (!ReadFile(hFile, buffer + aligned_offset, fileSize.QuadPart, &bytesRead, nullptr)) {
        CloseHandle(hFile);
        return false;
    }

    CloseHandle(hFile);
    offset = aligned_offset + bytesRead;
    return true;
}
#else
bool read_file_fast(const fs::path& filepath, char* buffer, size_t& offset, size_t buffer_size) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }

    size_t file_size = st.st_size;
    size_t aligned_offset = align_up(offset, ALIGNMENT);

    if (aligned_offset + file_size > buffer_size) {
        close(fd);
        return false;
    }

    size_t total_read = 0;
    while (total_read < file_size) {
        ssize_t n = read(fd, buffer + aligned_offset + total_read, file_size - total_read);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        total_read += n;
    }

    close(fd);
    offset = aligned_offset + total_read;
    return total_read == file_size;
}
#endif

void worker_thread(ThreadTask task) {
    size_t current_offset = 0;

    for (const auto& folder_files : task.folders) {
        for (const auto& file_info : folder_files) {
            if (!read_file_fast(file_info.path, task.buffer, current_offset, task.buffer_size)) {
                if (current_offset >= task.buffer_size ||
                    align_up(current_offset, ALIGNMENT) + file_info.size > task.buffer_size) {
                    std::cout << "Thread " << task.thread_id << " OOM!" << std::endl;
                    task.oom_flag->store(true);
                    task.remaining_bytes->store(0);
                    return;
                }
            }
        }
    }

    size_t remaining = task.buffer_size - align_up(current_offset, ALIGNMENT);
    task.remaining_bytes->store(remaining);
    std::cout << "Thread " << task.thread_id << " finished loading, remaining memory "
              << remaining << " Bytes." << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Program started..." << std::endl;

    std::string path;
    int workers = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            path = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            workers = std::stoi(argv[++i]);
        }
    }

    if (path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --path <path> --workers <num>" << std::endl;
        return 1;
    }

    size_t buffer_per_thread = TOTAL_MEMORY / workers;

    // 探测阶段
    std::vector<fs::path> folders;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_directory()) {
            folders.push_back(entry.path());
        }
    }
    std::sort(folders.begin(), folders.end());

    std::vector<std::vector<FileInfo>> folder_files(folders.size());
    for (size_t i = 0; i < folders.size(); i++) {
        for (const auto& entry : fs::directory_iterator(folders[i])) {
            if (entry.is_regular_file()) {
                folder_files[i].push_back({entry.path(), entry.file_size()});
            }
        }
    }

    // 分配文件夹
    size_t folders_per_thread = folders.size() / workers;
    size_t extra_folders = folders.size() % workers;

    std::vector<std::thread> threads;
    std::vector<char*> buffers(workers);
    std::vector<std::atomic<size_t>> remaining_bytes(workers);
    std::vector<std::atomic<bool>> oom_flags(workers);

    // 分配缓冲区
    for (int i = 0; i < workers; i++) {
        buffers[i] = static_cast<char*>(allocate_numa_memory(buffer_per_thread, i));
        if (!buffers[i]) {
            std::cerr << "Failed to allocate memory for thread " << i << std::endl;
            return 1;
        }
        std::memset(buffers[i], 0, buffer_per_thread);
        remaining_bytes[i].store(0);
        oom_flags[i].store(false);
    }

    std::cout << "Loading..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 启动线程
    size_t folder_index = 0;
    for (int i = 0; i < workers; i++) {
        ThreadTask task;
        task.thread_id = i;
        task.buffer = buffers[i];
        task.buffer_size = buffer_per_thread;
        task.remaining_bytes = &remaining_bytes[i];
        task.oom_flag = &oom_flags[i];

        size_t num_folders = folders_per_thread + (static_cast<size_t>(i) < extra_folders ? 1 : 0);
        for (size_t j = 0; j < num_folders && folder_index < folder_files.size(); j++) {
            task.folders.push_back(folder_files[folder_index++]);
        }

        threads.emplace_back(worker_thread, std::move(task));
    }

    // 等待所有线程
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "All thread joined, total time: " << duration.count() << " ms." << std::endl;

    // 释放内存
    for (int i = 0; i < workers; i++) {
        free_numa_memory(buffers[i], buffer_per_thread);
    }

    return 0;
}
