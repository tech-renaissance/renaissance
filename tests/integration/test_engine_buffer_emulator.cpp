/**
 * @file test_engine_buffer_emulator.cpp
 * @brief EngineBuffer多线程写入测试
 * @details 模拟3个PW线程并发写入EngineBuffer
 * @version 2.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include "renaissance/base/global_registry.h"
#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <sstream>
#include <iomanip>

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

// 线程0处理34个样本，线程1和2各处理33个样本
static constexpr int WORKER0_SAMPLES = 34;
static constexpr int WORKER1_SAMPLES = 33;
static constexpr int WORKER2_SAMPLES = 33;

// 全局输出日志（线程安全）
static std::string output_log;
static std::mutex output_mutex;

// =============================================================================
// 辅助宏：线程安全的输出
// =============================================================================

#define LOG_OUTPUT(stream) \
    do { \
        std::lock_guard<std::mutex> lock(output_mutex); \
        std::ostringstream oss; \
        oss << stream; \
        output_log += oss.str(); \
    } while(0)

// =============================================================================
// 模拟PW线程的工作函数
// =============================================================================

/**
 * @brief 模拟PW线程写入EngineBuffer
 * @param worker_id 线程ID（0, 1, 2）
 * @param num_samples 该线程需要处理的样本数
 * @param value 写入的数据值（线程0写1，线程1写2，线程2写3）
 * @param buffer EngineBuffer实例
 * @param total_samples 总样本数
 *
 * 关键逻辑：
 * 1. 每个worker维护自己的样本计数 worker_sample_count
 * 2. 全局序号 = worker_id + worker_sample_count * NUM_WORKERS
 * 3. batch_id = 全局序号 / LOCAL_BATCH_SIZE
 * 4. position = 全局序号 % LOCAL_BATCH_SIZE
 * 5. write_at()内部会检查batch_id并自动等待（如果batch还未准备好）
 */
void worker_thread(int worker_id, int num_samples, int value,
                   EngineBuffer& buffer, int total_samples) {

    // 准备数据（填充value）
    std::vector<uint8_t> data(SAMPLE_BYTES, static_cast<uint8_t>(value));
    int32_t label = static_cast<int32_t>(value);

    // 该worker的样本计数（从0开始，每个worker独立计数）
    int worker_sample_count = 0;

    for (int i = 0; i < num_samples; ++i, ++worker_sample_count) {
        // 计算全局序号
        int global_seq = worker_id + worker_sample_count * NUM_WORKERS;

        // 计算batch_id和position
        int batch_id = global_seq / LOCAL_BATCH_SIZE;
        int position = global_seq % LOCAL_BATCH_SIZE;

        // 是否是该worker的最后一个样本
        bool is_last_sample = (i == num_samples - 1);

        // 写入EngineBuffer（write_at内部会检查batch_id并自动等待）
        buffer.write_at(position, batch_id, label, data.data(), SAMPLE_BYTES);

        // 打印写入信息
        LOG_OUTPUT("[Thread " << worker_id << "] Wrote sample " << i
                  << " (value=" << value << ") to position " << position
                  << " in batch " << batch_id << "\n");

        // 通知写入完成
        bool triggered = buffer.notify_sample_written(global_seq, is_last_sample, total_samples);

        // 如果触发了传输，打印传输信息
        if (triggered) {
            LOG_OUTPUT("[Thread " << worker_id << "] Triggered async transfer\n");
        }
    }

    LOG_OUTPUT("[Thread " << worker_id << "] Finished\n");
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
              << RESOLUTION << "x" << RESOLUTION << "x" << NUM_CHANNELS << ")\n"
              << "  Worker 0: " << WORKER0_SAMPLES << " samples (value=1)\n"
              << "  Worker 1: " << WORKER1_SAMPLES << " samples (value=2)\n"
              << "  Worker 2: " << WORKER2_SAMPLES << " samples (value=3)\n\n";

    // =========================================================================
    // 步骤1: 配置GlobalRegistry
    // =========================================================================

    auto& registry = GlobalRegistry::instance();

    registry.set_world_size(1);  // world_size = 1
    registry.set_batch_size(LOCAL_BATCH_SIZE);
    registry.set_max_resolution(RESOLUTION);
    registry.set_num_color_channels(NUM_CHANNELS);
    registry.set_num_load_workers(1);
    registry.set_num_preproc_workers(NUM_WORKERS);

    // =========================================================================
    // 步骤2: 创建EngineBuffer
    // =========================================================================

    EngineBuffer buffer;

    // 配置buffer
    buffer.configure(
        LOCAL_BATCH_SIZE,          // local_batch_size
        SAMPLE_BYTES,              // max_train_sample_bytes
        SAMPLE_BYTES,              // max_val_sample_bytes
        NUM_WORKERS,               // num_workers_per_engine
        0                          // engine_id
    );

    // 更新phase（使用train模式）
    buffer.update_phase(true, RESOLUTION, NUM_CHANNELS);

    // =========================================================================
    // 步骤3: 创建3个worker线程
    // =========================================================================

    std::cout << "Starting worker threads...\n\n";

    std::thread t0(worker_thread, 0, WORKER0_SAMPLES, 1, std::ref(buffer), TOTAL_SAMPLES);
    std::thread t1(worker_thread, 1, WORKER1_SAMPLES, 2, std::ref(buffer), TOTAL_SAMPLES);
    std::thread t2(worker_thread, 2, WORKER2_SAMPLES, 3, std::ref(buffer), TOTAL_SAMPLES);

    // =========================================================================
    // 步骤4: 等待所有线程完成
    // =========================================================================

    t0.join();
    t1.join();
    t2.join();

    // =========================================================================
    // 步骤4: 打印所有线程的输出
    // =========================================================================

    std::cout << output_log;

    std::cout << "\n========================================\n"
              << "All worker threads completed\n"
              << "========================================\n\n";

    // =========================================================================
    // 步骤5: 验证结果
    // =========================================================================

    size_t total_transferred = buffer.total_samples_transferred();

    std::cout << "Statistics:\n"
              << "  Total samples written: " << TOTAL_SAMPLES << "\n"
              << "  Total samples transferred: " << total_transferred << "\n"
              << "  Current buffer ID: " << buffer.current_buffer_id() << "\n\n";

    // 验证
    if (total_transferred == TOTAL_SAMPLES) {
        std::cout << "[PASS] All samples transferred successfully!\n";
    } else {
        std::cout << "[FAIL] Transfer count mismatch! Expected: "
                  << TOTAL_SAMPLES << ", Got: " << total_transferred << "\n";
        return 1;
    }

    // 计算期望的batch数
    int expected_batches = (TOTAL_SAMPLES + LOCAL_BATCH_SIZE - 1) / LOCAL_BATCH_SIZE;
    std::cout << "  Expected batches: " << expected_batches << "\n";
    std::cout << "  Final batch size: " << (TOTAL_SAMPLES % LOCAL_BATCH_SIZE)
              << " (if " << (TOTAL_SAMPLES % LOCAL_BATCH_SIZE) << " != 0 else "
              << LOCAL_BATCH_SIZE << ")\n\n";

    std::cout << "========================================\n"
              << "Test completed successfully!\n"
              << "========================================\n\n";

    return 0;
}
