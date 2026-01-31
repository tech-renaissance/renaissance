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

std::vector<unsigned char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Cannot read: " + path);
    return buffer;
}

struct ThreadResult {
    double ms_per_iter = 0.0;
};

void decode_worker(int id, const unsigned char* jpeg_data, size_t jpeg_size,
                  int iter, std::atomic<bool>& start_flag, ThreadResult& result) {
    std::vector<unsigned char> decode_buffer(16 * 1024 * 1024);
    std::fill(decode_buffer.begin(), decode_buffer.end(), 0);
    tjhandle handle = tjInitDecompress();
    if (!handle) return;
    std::vector<unsigned char> local_jpeg(jpeg_data, jpeg_data + jpeg_size);
    int width, height, subsamp, colorspace;
    if (tjDecompressHeader3(handle, local_jpeg.data(), local_jpeg.size(),
                           &width, &height, &subsamp, &colorspace) != 0) {
        tjDestroy(handle);
        return;
    }
    size_t required_size = width * height * 3;
    if (decode_buffer.size() < required_size) decode_buffer.resize(required_size);
    tjDecompress2(handle, local_jpeg.data(), local_jpeg.size(),
                 decode_buffer.data(), width, 0, height,
                 TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
    while (!start_flag.load(std::memory_order_acquire)) std::this_thread::yield();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; i++) {
        int ret = tjDecompress2(handle, local_jpeg.data(), local_jpeg.size(),
                               decode_buffer.data(), width, 0, height, TJPF_RGB,
                               TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
        if (ret != 0) break;
    }
    auto t1 = std::chrono::steady_clock::now();
    result.ms_per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iter;
    tjDestroy(handle);
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    if (cfg.path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --path <image.jpg> [--workers N] [--iter M]\n";
        return 1;
    }
    std::cout << "Configuration: Image=" << cfg.path << ", Workers=" << cfg.workers << ", Iter=" << cfg.iter << "\n";
    std::vector<unsigned char> jpeg_data;
    try {
        jpeg_data = readFile(cfg.path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "JPEG size: " << jpeg_data.size() << " bytes\n";
    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(cfg.workers);
    std::atomic<bool> start_flag{false};
    auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.workers; i++) {
        threads.emplace_back(decode_worker, i, jpeg_data.data(), jpeg_data.size(),
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
