#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <atomic>

extern "C" {
#include <turbojpeg.h>
}

struct Config {
    std::string path;
    int workers = 4;
    int iter = 100;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) cfg.path = argv[++i];
        else if (arg == "--workers" && i + 1 < argc) cfg.workers = std::stoi(argv[++i]);
        else if (arg == "--iter" && i + 1 < argc) cfg.iter = std::stoi(argv[++i]);
    }
    return cfg;
}

std::vector<unsigned char> load_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Failed to open: " + path);
    auto sz = ifs.tellg();
    std::vector<unsigned char> data(sz);
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

struct ThreadResult {
    double ms_per_iter = 0.0;
};

void decode_worker(int id, const unsigned char* jpeg_data, size_t jpeg_size,
                   int iter, std::atomic<bool>& start_flag, ThreadResult& result) {
    tjhandle handle = tjInitDecompress();
    if (!handle) return;
    std::vector<unsigned char> local_jpeg(jpeg_data, jpeg_data + jpeg_size);
    std::vector<unsigned char> decode_buffer(16 * 1024 * 1024);
    int width, height, subsamp, colorspace;
    tjDecompressHeader3(handle, local_jpeg.data(), local_jpeg.size(),
                       &width, &height, &subsamp, &colorspace);
    tjDecompress2(handle, local_jpeg.data(), local_jpeg.size(),
                 decode_buffer.data(), width, 0, height, TJPF_RGB,
                 TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
    while (!start_flag.load(std::memory_order_acquire)) std::this_thread::yield();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; i++) {
        tjDecompress2(handle, local_jpeg.data(), local_jpeg.size(),
                     decode_buffer.data(), width, 0, height, TJPF_RGB,
                     TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
    }
    auto t1 = std::chrono::steady_clock::now();
    result.ms_per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iter;
    tjDestroy(handle);
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    if (cfg.path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --path <jpg> [--workers N] [--iter M]\n";
        return 1;
    }
    std::cout << "Configuration: Image=" << cfg.path << ", Workers=" << cfg.workers << ", Iter=" << cfg.iter << "\n";
    auto img = load_file(cfg.path);
    std::cout << "JPEG size: " << img.size() << " bytes\n";
    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(cfg.workers);
    std::atomic<bool> start_flag{false};
    auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.workers; i++) {
        threads.emplace_back(decode_worker, i, img.data(), img.size(),
                             cfg.iter, std::ref(start_flag), std::ref(results[i]));
    }
    start_flag.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
    auto t_end = std::chrono::steady_clock::now();
    double total = 0.0;
    for (auto& r : results) total += r.ms_per_iter;
    double avg = total / cfg.workers;
    double total_elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "\nResults:\n  Total time: " << total_elapsed << " ms\n";
    std::cout << "  Total decodes: " << (cfg.workers * cfg.iter) << "\n";
    std::cout << "  Average time per decode: " << avg << " ms\n";
    std::cout << "  Throughput: " << ((cfg.workers * cfg.iter) / total_elapsed * 1000.0) << " decodes/sec\n";
    return 0;
}
