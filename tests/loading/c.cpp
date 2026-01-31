// EXPERT_CG - C++ Filesystem方案
// 使用std::filesystem，线程绑定CPU核心

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstring>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sched.h>
#endif

namespace fs = std::filesystem;

struct Args {
    std::string basePath;
    int workers = 1;
};

struct FileInfo {
    fs::path path;
    size_t size;
};

struct ThreadTask {
    int id;
    size_t bufferSize;
    void* buffer;
    std::vector<FileInfo> files;  // 改为存储文件信息而不是文件夹
};

Args parseArgs(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc)
            args.basePath = argv[++i];
        else if (arg == "--workers" && i + 1 < argc)
            args.workers = std::stoi(argv[++i]);
    }
    return args;
}

void bindThreadToCore(int threadId) {
#ifdef _WIN32
    DWORD_PTR mask = (DWORD_PTR)1 << (threadId % std::thread::hardware_concurrency());
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(threadId % std::thread::hardware_concurrency(), &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

static constexpr size_t ALIGNMENT = 16 * 1024;

size_t readFileBinary(const fs::path& file, char* dest, size_t maxBytes) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(file.string().c_str(), GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;

    DWORD bytesRead = 0;
    DWORD total = 0;
    char buffer[ALIGNMENT];
    while (true) {
        BOOL ok = ReadFile(hFile, buffer, sizeof(buffer), &bytesRead, NULL);
        if (!ok || bytesRead == 0) break;
        if (total + bytesRead > maxBytes) break;
        std::memcpy(dest + total, buffer, bytesRead);
        total += bytesRead;
    }
    CloseHandle(hFile);
    return total;
#else
    int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    size_t total = 0;
    char buffer[ALIGNMENT];
    ssize_t r;
    while ((r = read(fd, buffer, sizeof(buffer))) > 0) {
        if (total + (size_t)r > maxBytes) break;
        std::memcpy(dest + total, buffer, (size_t)r);
        total += (size_t)r;
    }
    close(fd);
    return total;
#endif
}

void worker(ThreadTask task, std::atomic<int>& doneCount, std::mutex& printMutex) {
    bindThreadToCore(task.id);
    char* ptr = reinterpret_cast<char*>(task.buffer);
    size_t used = 0;
    bool oom = false;

    // 计时阶段：只负责读取文件，不进行任何文件系统操作
    for (const auto& file_info : task.files) {
        size_t fileSize = file_info.size;
        if (fileSize == 0) continue;

        size_t start = (used + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

        if (start + fileSize > task.bufferSize) {
            oom = true;
            break;
        }

        size_t actuallyRead = readFileBinary(file_info.path, ptr + start, fileSize);
        if (actuallyRead == fileSize) {
            used = start + actuallyRead;
        }
    }

    {
        std::lock_guard<std::mutex> lk(printMutex);
        if (oom) {
            std::cout << "Thread " << task.id << " OOM!" << std::endl;
        } else {
            std::cout << "Thread " << task.id << " finished loading, remaining memory "
                      << (task.bufferSize - used) << " Bytes." << std::endl;
        }
    }

    doneCount.fetch_add(1);
}

int main(int argc, char* argv[]) {
    std::cout << "Program started..." << std::endl;

    Args args = parseArgs(argc, argv);

    if (args.basePath.empty()) {
        std::cerr << "Usage: " << argv[0] << " --path <dir> --workers <1|2|4|8|16>" << std::endl;
        return 1;
    }

    // 探测阶段（不计时）：扫描所有文件并收集元数据
    std::vector<fs::path> subDirs;
    for (const auto& entry : fs::directory_iterator(args.basePath)) {
        if (entry.is_directory())
            subDirs.push_back(entry.path());
    }
    std::sort(subDirs.begin(), subDirs.end());

    // 收集所有文件信息（路径和大小）
    std::vector<std::vector<FileInfo>> all_files;
    for (const auto& dir : subDirs) {
        std::vector<FileInfo> files_in_dir;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                size_t sz = fs::file_size(entry);
                files_in_dir.push_back({entry.path(), sz});
            }
        }
        all_files.push_back(files_in_dir);
    }

    int numThreads = args.workers;
    uint64_t totalMem = 256ULL * 1024 * 1024 * 1024;
    uint64_t perThreadMem = totalMem / numThreads;

    // 分配任务
    std::vector<ThreadTask> tasks(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        tasks[i].id = i;
        tasks[i].bufferSize = perThreadMem;
#ifdef _WIN32
        tasks[i].buffer = _aligned_malloc(perThreadMem, ALIGNMENT);
#else
        tasks[i].buffer = std::aligned_alloc(ALIGNMENT, perThreadMem);
#endif
        if (tasks[i].buffer) {
            std::memset(tasks[i].buffer, 0, perThreadMem);
        }
    }

    // 将文件分配给各个线程
    size_t totalFolders = all_files.size();
    size_t per = (totalFolders + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        size_t start = t * per;
        size_t end = (std::min)((t + 1) * per, totalFolders);
        for (size_t j = start; j < end; ++j) {
            // 将所有文件添加到该线程
            tasks[t].files.insert(tasks[t].files.end(), all_files[j].begin(), all_files[j].end());
        }
    }

    std::atomic<int> doneCount(0);
    std::mutex printMutex;
    std::cout << "Loading..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i)
        threads.emplace_back(worker, tasks[i], std::ref(doneCount), std::ref(printMutex));

    for (auto& th : threads) th.join();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "All thread joined, total time: " << ms << " ms." << std::endl;

    // 释放内存
    for (auto& t : tasks) {
#ifdef _WIN32
        _aligned_free(t.buffer);
#else
        std::free(t.buffer);
#endif
    }

    return 0;
}
