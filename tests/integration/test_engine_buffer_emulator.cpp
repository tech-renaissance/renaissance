/**
 * @file test_engine_buffer_emulator.cpp
 * @brief EngineBuffer多线程写入测试（延迟终止版本 - OP方案）
 * @details Worker 不知道总样本数，只在获取失败时调用 no_more_samples()
 */

#include "renaissance.h"
#include "renaissance/base/global_registry.h"
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <sstream>
#include <queue>

using namespace tr;

// =============================================================================
// 测试配置
// =============================================================================

static constexpr int LOCAL_BATCH_SIZE = 8;
static constexpr int NUM_WORKERS = 3;
static constexpr int RESOLUTION = 224;
static constexpr int NUM_CHANNELS = 3;
static constexpr size_t SAMPLE_BYTES = RESOLUTION * RESOLUTION * NUM_CHANNELS;
static constexpr int TOTAL_SAMPLES = 100;

// Worker 分配：Worker 0 -> 34, Worker 1 -> 33, Worker 2 -> 33
static const int WORKER_SAMPLE_LIMITS[NUM_WORKERS] = {34, 33, 33};

static std::string output_log;
static std::mutex output_mutex;

#define LOG_OUTPUT(stream) \
    do { \
        std::lock_guard<std::mutex> lock(output_mutex); \
        std::ostringstream oss; \
        oss << stream; \
        output_log += oss.str(); \
    } while(0)

// =============================================================================
// 模拟数据源（Worker 不知道总数，只能尝试获取）
// =============================================================================

class DataSource {
public:
    DataSource(int worker_id, int num_workers, int total_samples)
        : worker_id_(worker_id), num_workers_(num_workers) {
        // 预计算该 Worker 负责的所有样本序号
        for (int seq = worker_id; seq < total_samples; seq += num_workers) {
            pending_samples_.push(seq);
        }
    }

    /**
     * @brief 尝试获取下一个样本
     * @param[out] global_seq 全局序号
     * @return true=获取成功, false=没有更多样本
     *
     * 关键：Worker 只有在调用此方法返回 false 时，才知道样本耗尽
     */
    bool try_get_next(int& global_seq) {
        if (pending_samples_.empty()) {
            return false;
        }
        global_seq = pending_samples_.front();
        pending_samples_.pop();
        return true;
    }

private:
    int worker_id_;
    int num_workers_;
    std::queue<int> pending_samples_;
};

// =============================================================================
// Worker 线程函数
// =============================================================================

void worker_thread(int worker_id, EngineBuffer& buffer) {
    DataSource source(worker_id, NUM_WORKERS, TOTAL_SAMPLES);

    std::vector<uint8_t> data(SAMPLE_BYTES, static_cast<uint8_t>(worker_id + 1));
    int32_t label = static_cast<int32_t>(worker_id + 1);

    int samples_written = 0;
    int global_seq;

    // 主循环：尝试获取样本 -> 写入 -> 重复
    while (source.try_get_next(global_seq)) {
        int batch_id = global_seq / LOCAL_BATCH_SIZE;
        int position = global_seq % LOCAL_BATCH_SIZE;

        // 1. 申请写入位置
        uint8_t* write_ptr = buffer.request_write_slot(position, batch_id, label);

        if (write_ptr == nullptr) {
            // Buffer 已停止（不应发生，除非有 bug）
            LOG_OUTPUT("[Thread " << worker_id << "] Buffer stopped unexpectedly!\n");
            break;
        }

        // 2. 写入数据
        std::memcpy(write_ptr, data.data(), SAMPLE_BYTES);

        LOG_OUTPUT("[Thread " << worker_id << "] Wrote sample " << samples_written
                  << " (value=" << (worker_id + 1) << ") to position " << position
                  << " in batch " << batch_id << "\n");

        // 3. 通知写入完成
        buffer.notify_sample_written();

        ++samples_written;
    }

    // 样本耗尽，调用 no_more_samples()
    LOG_OUTPUT("[Thread " << worker_id << "] Finished\n");
    buffer.no_more_samples(worker_id);
}

// =============================================================================
// 主函数
// =============================================================================

int main() {
    std::cout << "\n========================================\n"
              << "EngineBuffer Multi-Thread Test\n"
              << "========================================\n\n";

    std::cout << "Configuration:\n"
              << "  Batch size: " << LOCAL_BATCH_SIZE << "\n"
              << "  Workers: " << NUM_WORKERS << "\n"
              << "  Total samples: " << TOTAL_SAMPLES << "\n"
              << "  Sample bytes: " << SAMPLE_BYTES << " ("
              << RESOLUTION << "x" << RESOLUTION << "x" << NUM_CHANNELS << ")\n";

    for (int i = 0; i < NUM_WORKERS; ++i) {
        std::cout << "  Worker " << i << ": " << WORKER_SAMPLE_LIMITS[i] << " samples (value=" << (i+1) << ")\n";
    }
    std::cout << "\n";

    // 配置 GlobalRegistry
    auto& registry = GlobalRegistry::instance();
    registry.set_using_gpu(false);
    registry.set_world_size(1);
    registry.set_batch_size(LOCAL_BATCH_SIZE);
    registry.set_max_resolution(RESOLUTION);
    registry.set_num_color_channels(NUM_CHANNELS);
    registry.set_num_load_workers(1);
    registry.set_num_preproc_workers(NUM_WORKERS);

    // 创建 EngineBuffer
    EngineBuffer buffer;
    buffer.configure(LOCAL_BATCH_SIZE, SAMPLE_BYTES, SAMPLE_BYTES, NUM_WORKERS, 0);
    buffer.update_phase(true, RESOLUTION, NUM_CHANNELS);

    // 启动 Worker 线程
    std::cout << "Starting worker threads...\n\n";

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        threads.emplace_back(worker_thread, i, std::ref(buffer));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 输出日志
    std::cout << output_log;

    std::cout << "\n========================================\n"
              << "All worker threads completed\n"
              << "========================================\n\n";

    // 验证结果
    size_t total_transferred = buffer.total_samples_transferred();

    std::cout << "Statistics:\n"
              << "  Total samples written: " << TOTAL_SAMPLES << "\n"
              << "  Total samples transferred: " << total_transferred << "\n"
              << "  Current buffer ID: " << buffer.current_buffer_id() << "\n\n";

    if (total_transferred == static_cast<size_t>(TOTAL_SAMPLES)) {
        std::cout << "[PASS] All samples transferred successfully!\n";
        int expected_batches = (TOTAL_SAMPLES + LOCAL_BATCH_SIZE - 1) / LOCAL_BATCH_SIZE;
        std::cout << "  Expected batches: " << expected_batches << "\n";
        std::cout << "  Final batch size: " << (TOTAL_SAMPLES % LOCAL_BATCH_SIZE)
                  << " (if " << (TOTAL_SAMPLES % LOCAL_BATCH_SIZE) << " != 0 else "
                  << LOCAL_BATCH_SIZE << ")\n\n";
    } else {
        std::cout << "[FAIL] Expected " << TOTAL_SAMPLES
                  << ", got " << total_transferred << "\n";
        return 1;
    }

    std::cout << "========================================\n"
              << "Test completed successfully!\n"
              << "========================================\n\n";

    return 0;
}
