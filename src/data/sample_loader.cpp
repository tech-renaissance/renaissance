/**
 * @file sample_loader.cpp
 * @brief 通用样本加载器实现（用于部署场景）
 * @version V3.10.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#include "renaissance/data/sample_loader.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

#include <fstream>
#include <cstring>
#include <mimalloc.h>
#include <functional>

namespace tr {

// =============================================================================
// 单例
// =============================================================================

SampleLoader& SampleLoader::getInstance() {
    static SampleLoader instance;
    return instance;
}

SampleLoader::~SampleLoader() {
    // 释放所有剩余的 buffer
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!buffer_queue_.empty()) {
        BufferInfo& buffer = buffer_queue_.front();
        if (buffer.data) {
            mi_free(buffer.data);
        }
        buffer_queue_.pop();
    }
}

// =============================================================================
// 配置接口
// =============================================================================

void SampleLoader::configure_memory_pool(size_t memory_pool_size_mb) {
    TR_CHECK(memory_pool_size_mb > 0, ValueError,
             "memory_pool_size_mb must be positive, got " << memory_pool_size_mb);

    // 初始化成员变量
    memory_pool_size_bytes_ = memory_pool_size_mb * 1024 * 1024;
    current_memory_usage_ = 0;
    end_called_ = false;
    configured_ = true;
    global_seq_.store(0, std::memory_order_relaxed);

    LOG_INFO << "SampleLoader configured: memory_pool=" << memory_pool_size_mb << " MB";
}

void SampleLoader::end() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    end_called_ = true;
    LOG_INFO << "SampleLoader end() called";
}

// =============================================================================
// 加载接口
// =============================================================================

void SampleLoader::load_jpeg_file(const std::string& file_path) {
    TR_CHECK(configured_, ValueError, "SampleLoader not configured yet");

    // 打开文件
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_WARN << "Failed to open file: " << file_path
                 << ". Skipping this file. Error: " << strerror(errno);
        return;
    }

    // 获取文件大小
    std::streamsize file_size = file.tellg();
    if (file_size <= 0) {
        LOG_WARN << "Invalid file size: " << file_path
                 << ". Size: " << file_size << " bytes. Skipping.";
        return;
    }

    file.seekg(0, std::ios::beg);

    // 检查内存池是否足够
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this, file_size]() {
        return current_memory_usage_ + static_cast<size_t>(file_size) <= memory_pool_size_bytes_;
    });

    // 从 mimalloc 分配内存
    uint8_t* data = static_cast<uint8_t*>(mi_malloc(static_cast<size_t>(file_size)));
    if (!data) {
        LOG_ERROR << "Failed to allocate " << file_size << " bytes from mimalloc";
        return;
    }

    // 读取文件内容
    if (!file.read(reinterpret_cast<char*>(data), file_size)) {
        LOG_WARN << "Failed to read file: " << file_path
                 << ". Skipping this file.";
        mi_free(data);
        return;
    }

    // 创建 BufferInfo
    BufferInfo buffer;
    buffer.data = data;
    buffer.num_samples = 1;  // JPEG 文件 = 1 个样本
    buffer.sample_idx = 0;
    buffer.buffer_size = static_cast<size_t>(file_size);
    buffer.sample_sizes.push_back(static_cast<size_t>(file_size));
    buffer.is_jpeg_buffer = true;

    // 加入队列
    buffer_queue_.push(buffer);
    current_memory_usage_ += buffer.buffer_size;

    lock.unlock();

    LOG_DEBUG << "Loaded JPEG file: " << file_path
              << ", size=" << file_size << " bytes";
}

// =============================================================================
// DataLoader 基类接口实现
// =============================================================================

bool SampleLoader::get_next_sample(int preproc_worker_id, int32_t& label,
                                   const uint8_t*& data_ptr, size_t& data_size) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    // 如果队列为空，返回 false
    if (buffer_queue_.empty()) {
        return false;
    }

    BufferInfo& buffer = buffer_queue_.front();

    // 计算当前样本位置
    size_t sample_pos = buffer.sample_idx;

    if (sample_pos >= buffer.num_samples) {
        // 当前 buffer 已读完
        return false;
    }

    // 定位样本数据
    if (buffer.is_jpeg_buffer) {
        // JPEG 数据：需要计算偏移
        size_t offset = 0;
        for (size_t i = 0; i < sample_pos; ++i) {
            offset += buffer.sample_sizes[i];
        }

        data_ptr = buffer.data + offset;
        data_size = buffer.sample_sizes[sample_pos];
    } else {
        // NHWC float 数据：等长数据
        size_t sample_size = buffer.buffer_size / buffer.num_samples;
        data_ptr = buffer.data + sample_pos * sample_size;
        data_size = sample_size;
    }

    // 标签固定为 0
    label = 0;

    // 递增样本索引
    buffer.sample_idx++;

    // 如果 buffer 读完，释放内存并出队
    if (buffer.sample_idx >= buffer.num_samples) {
        mi_free(buffer.data);
        current_memory_usage_ -= buffer.buffer_size;
        buffer_queue_.pop();
        queue_cv_.notify_all();  // 通知可能等待的用户线程

        // 递增全局序列号（用于静态分配）
        global_seq_.fetch_add(1, std::memory_order_relaxed);
    }

    return true;
}

bool SampleLoader::has_more_buffers() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    // 如果队列为空，返回 false（无论是否调用 end()）
    // 这样可以避免 Preprocessor 无限循环
    return !buffer_queue_.empty();
}

// =============================================================================
// 数据集下载
// =============================================================================

void SampleLoader::download(const std::string& save_path) {
    (void)save_path;  // Unused parameter
    TR_NOT_IMPLEMENTED("SampleLoader does not support download(). "
                      "Please load JPEG files manually using load_jpeg_file().");
}

} // namespace tr
