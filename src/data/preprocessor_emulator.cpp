/**
 * @file preprocessor_emulator.cpp
 * @brief 预处理器模拟器实现
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 */

#include "renaissance/data/preprocessor_emulator.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <chrono>
#include <algorithm>

namespace tr {
namespace data {

// =============================================================================
// PreprocessorEmulator 实现
// =============================================================================

PreprocessorEmulator::PreprocessorEmulator(DataLoaderBase* loader, int num_workers, int simulate_ms)
    : loader_(loader),
      num_workers_(clamp_to_power_of_two(num_workers, 64)),
      simulate_ms_(simulate_ms) {

    if (loader_ == nullptr) {
        TR_VALUE_ERROR("DataLoader pointer cannot be null");
    }

    LOG_INFO << "PreprocessorEmulator created: workers=" << num_workers_
             << ", simulate_ms=" << simulate_ms_;
}

PreprocessorEmulator::~PreprocessorEmulator() {
    join();
}

int PreprocessorEmulator::clamp_to_power_of_two(int n, int max_val) {
    // 钳制到最大值
    if (n > max_val) {
        LOG_WARN << "num_workers " << n << " exceeds maximum " << max_val
                 << ", clamping to " << max_val;
        n = max_val;
    }

    // 向下取2的幂
    int power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }

    if (power != n) {
        n = power;  // 不报WARNING
    }

    return std::max(1, n);
}

void PreprocessorEmulator::start() {
    if (loader_ == nullptr) {
        TR_VALUE_ERROR("DataLoader not set");
    }

    if (!workers_.empty()) {
        LOG_WARN << "Workers already started";
        return;
    }

    LOG_INFO << "Starting " << num_workers_ << " preprocessor workers...";

    stop_flag_.store(false, std::memory_order_relaxed);
    workers_.reserve(num_workers_);

    for (int i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&PreprocessorEmulator::worker_thread, this, i);
    }

    LOG_INFO << "Preprocessor workers started";
}

void PreprocessorEmulator::join() {
    if (workers_.empty()) {
        return;
    }

    LOG_INFO << "Waiting for preprocessor workers to finish...";

    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }

    workers_.clear();

    // 输出统计信息
    size_t total = get_total_processed();
    LOG_INFO << "All preprocessor workers finished. Total samples processed: " << total;
}

void PreprocessorEmulator::worker_thread(int worker_id) {
    LOG_DEBUG << "Preprocessor worker " << worker_id << " started";

    SampleView view;
    size_t local_count = 0;  // 该worker处理的样本数

    while (!stop_flag_.load(std::memory_order_relaxed)) {
        // 从DataLoader获取样本
        if (!loader_->next_sample(worker_id, view)) {
            // Epoch结束
            break;
        }

        // 模拟预处理时间
        if (simulate_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(simulate_ms_));
        }

        // 统计标签
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            label_counts_[view.label]++;
        }

        // 保存指定图片（如果配置了）
        {
            std::lock_guard<std::mutex> lock(save_mutex_);
            if (!save_done_.load() &&
                worker_id == save_worker_id_ &&
                local_count == static_cast<size_t>(save_sample_idx_)) {

                LOG_INFO << "Saving sample from worker " << worker_id
                         << ", sample #" << local_count
                         << ", label=" << view.label
                         << ", size=" << view.size
                         << " to " << save_path_;

                std::ofstream out(save_path_, std::ios::binary);
                out.write(reinterpret_cast<const char*>(view.data), view.size);

                if (out.good()) {
                    LOG_INFO << "Sample saved successfully";
                } else {
                    LOG_ERROR << "Failed to save sample to " << save_path_;
                }

                save_done_.store(true, std::memory_order_release);
            }
        }

        // 更新计数
        ++local_count;
        total_processed_.fetch_add(1, std::memory_order_relaxed);
    }

    LOG_INFO << "Preprocessor worker " << worker_id << " finished: "
              << local_count << " samples processed";
}

std::map<int32_t, size_t> PreprocessorEmulator::get_label_counts() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return label_counts_;
}

size_t PreprocessorEmulator::get_total_processed() const {
    return total_processed_.load(std::memory_order_acquire);
}

void PreprocessorEmulator::save_sample_image(int worker_id, int sample_idx,
                                             const std::string& output_path) {
    std::lock_guard<std::mutex> lock(save_mutex_);

    save_worker_id_ = worker_id;
    save_sample_idx_ = sample_idx;
    save_path_ = output_path;
    save_done_.store(false, std::memory_order_release);

    LOG_INFO << "Configured to save: worker " << worker_id
             << ", sample #" << sample_idx
             << " -> " << output_path;
}

} // namespace data
} // namespace tr
