#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>

extern "C" {
#include <jpeglib.h>
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

std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open: " + path);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Failed to read: " + path);
    return buffer;
}

struct ThreadResult {
    double ms_per_iter = 0.0;
};

void worker_thread(const uint8_t* jpeg_data, size_t jpeg_size, int iterations,
                  int thread_id, ThreadResult& result) {
    constexpr size_t OUTPUT_BUFFER_SIZE = 16 * 1024 * 1024;
    std::vector<uint8_t> output_buffer(OUTPUT_BUFFER_SIZE, 0);
    std::vector<uint8_t*> row_ptrs;
    row_ptrs.reserve(16384);
    double total_time_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        jpeg_decompress_struct cinfo;
        jpeg_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, jpeg_data, jpeg_size);
        jpeg_read_header(&cinfo, TRUE);
        cinfo.do_fancy_upsampling = FALSE;
        cinfo.do_block_smoothing = FALSE;
        cinfo.dct_method = JDCT_IFAST;
        cinfo.out_color_space = JCS_RGB;
        jpeg_start_decompress(&cinfo);
        const size_t row_stride = cinfo.output_width * cinfo.output_components;
        const size_t needed_size = row_stride * cinfo.output_height;
        if (needed_size > OUTPUT_BUFFER_SIZE) {
            jpeg_abort_decompress(&cinfo);
            break;
        }
        row_ptrs.resize(cinfo.output_height);
        for (size_t j = 0; j < cinfo.output_height; ++j) {
            row_ptrs[j] = reinterpret_cast<uint8_t*>(&output_buffer[j * row_stride]);
        }
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, row_ptrs.data() + cinfo.output_scanline,
                               cinfo.output_height - cinfo.output_scanline);
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        auto end = std::chrono::high_resolution_clock::now();
        total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    result.ms_per_iter = total_time_ms / iterations;
}

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);
    if (cfg.path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --path <image.jpg> --workers <num> --iter <num>\n";
        return 1;
    }
    std::cout << "Configuration: Image=" << cfg.path << ", Workers=" << cfg.workers << ", Iter=" << cfg.iter << "\n";
    std::vector<uint8_t> jpeg_data;
    try {
        jpeg_data = load_file(cfg.path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "JPEG size: " << jpeg_data.size() << " bytes\n";
    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(cfg.workers);
    auto global_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.workers; ++i) {
        threads.emplace_back(worker_thread, jpeg_data.data(), jpeg_data.size(),
                            cfg.iter, i, std::ref(results[i]));
    }
    for (auto& t : threads) t.join();
    auto global_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> global_duration = global_end - global_start;
    double total_decodes = cfg.workers * cfg.iter;
    double avg_time = 0.0;
    for (const auto& r : results) avg_time += r.ms_per_iter;
    avg_time /= cfg.workers;
    double throughput = (total_decodes / global_duration.count()) * 1000.0;
    std::cout << "\nResults:\n  Total time: " << global_duration.count() << " ms\n";
    std::cout << "  Total decodes: " << total_decodes << "\n";
    std::cout << "  Average time per decode: " << avg_time << " ms\n";
    std::cout << "  Throughput: " << static_cast<int>(throughput) << " decodes/sec\n";
    return 0;
}
