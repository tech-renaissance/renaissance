##### 5.2 构造函数实现

```cpp
/**
 * @file preprocess_worker.cpp
 * @brief 预处理工作器实现
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 */

#include "renaissance/data/preprocess_worker.h"
#include "renaissance/data/engine_buffer.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/global_registry.h"
#include "renaissance/base/philox.h"
#include <cstring>
#include <algorithm>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

namespace tr {

PreprocessWorker::PreprocessWorker(
    const Config& config,
    const std::vector<std::unique_ptr<transforms::PreprocessOperation>>& train_ops,
    const std::vector<std::unique_ptr<transforms::PreprocessOperation>>& val_ops
)
    : config_(config)
    , rng_(0)  // ✅ 修正（V2.1）：临时seed，延迟初始化
{
    LOG_DEBUG << "Creating PW " << config_.worker_id
              << ", engine=" << config_.engine_id
              << ", pid=" << config_.pid_in_engine;
    
    // ==================== 1. 分配Workshop（不立即First-Touch）====================
    // ✅ 修正（V2.1）：分离分配与First-Touch
    allocate_workshop();
    
    // ==================== 2. 克隆PO列表 ====================
    train_ops_.reserve(train_ops.size());
    for (const auto& op : train_ops) {
        train_ops_.push_back(op->clone());
    }
    
    val_ops_.reserve(val_ops.size());
    for (const auto& op : val_ops) {
        val_ops_.push_back(op->clone());
    }
    
    LOG_DEBUG << "PW " << config_.worker_id << " cloned POs: "
              << train_ops_.size() << " train, "
              << val_ops_.size() << " val";
    
    // ==================== 3. 初始化TurboJPEG ====================
    tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
    TR_CHECK(tj_handle_ != nullptr, MemoryError,
             "Failed to initialize TurboJPEG for PW " << config_.worker_id);
    
    // 设置优化标志
    tj3Set(tj_handle_, TJPARAM_FASTDCT, 1);
    tj3Set(tj_handle_, TJPARAM_FASTUPSAMPLE, 1);
    
    LOG_DEBUG << "PW " << config_.worker_id << " TurboJPEG initialized";
    
    // ==================== 4. 初始化S区管理数组 ====================
    // ✅ 修正（V2.1）：每个S区完全独立管理
    s_shuffle_indices_.resize(config_.num_region_s);
    s_read_indices_.resize(config_.num_region_s, 0);
    num_samples_per_s_.resize(config_.num_region_s, 0);
    
    if (config_.num_region_s > 0) {
        // 精确计算S区标签数组的预留大小
        size_t num_train = GlobalRegistry::instance().num_train_samples();
        size_t samples_per_engine = (num_train + config_.world_size - 1) / config_.world_size;
        size_t samples_per_worker = (samples_per_engine + config_.num_workers_per_engine - 1) 
                                   / config_.num_workers_per_engine;
        
        // 预留10%余量，避免边界情况扩容
        size_t reserve_size = samples_per_worker + samples_per_worker / 10;
        
        region_s_labels_.reserve(reserve_size);
        
        LOG_DEBUG << "PW " << config_.worker_id << " S区标签预留: " << reserve_size;
    }
    
    // ==================== 5. 初始化C区标签数组 ====================
    if (config_.region_c_size > 0) {
        size_t num_val = GlobalRegistry::instance().num_val_samples();
        size_t samples_per_engine = (num_val + config_.world_size - 1) / config_.world_size;
        size_t samples_per_worker = (samples_per_engine + config_.num_workers_per_engine - 1) 
                                   / config_.num_workers_per_engine;
        
        size_t reserve_size = samples_per_worker + samples_per_worker / 10;
        region_c_labels_.reserve(reserve_size);
        
        LOG_DEBUG << "PW " << config_.worker_id << " C区标签预留: " << reserve_size;
    }
    
    LOG_INFO << "PW " << config_.worker_id << " created"
             << ", workshop=" << (workshop_size_ / (1024.0*1024.0)) << " MB";
}

PreprocessWorker::~PreprocessWorker() {
    // 释放Workshop
    free_workshop();
    
    // 释放TurboJPEG句柄
    if (tj_handle_) {
        tj3Destroy(tj_handle_);
        tj_handle_ = nullptr;
    }
    
    LOG_DEBUG << "PW " << config_.worker_id << " destroyed";
}

```

---

##### 5.3 allocate_workshop（内存分配）

```cpp
void PreprocessWorker::allocate_workshop() {
    // 辅助对齐函数
    auto align_64 = [](size_t s) -> size_t {
        return (s + 63) & ~size_t(63);
    };
    
    auto align_4k = [](size_t s) -> size_t {
        return (s + 4095) & ~size_t(4095);
    };
    
    // ==================== 计算总大小（所有区对齐）====================
    workshop_size_ = align_64(config_.region_d_size) +
                     align_64(config_.region_ab_size) * 2 +
                     align_64(config_.region_t_size);
    
    // S区对齐到4KB页边界（NUMA优化）
    for (int i = 0; i < config_.num_region_s; ++i) {
        workshop_size_ += align_4k(config_.region_s_size);
    }
    
    workshop_size_ += align_64(config_.region_c_size);
    
    // 总大小对齐到4KB页
    workshop_size_ = align_4k(workshop_size_);
    
    LOG_DEBUG << "PW " << config_.worker_id << " allocating workshop: " 
              << (workshop_size_ / (1024.0*1024.0)) << " MB";
    
    // ==================== 分配内存 ====================
#ifdef _WIN32
    workshop_ = static_cast<uint8_t*>(
        VirtualAlloc(NULL, workshop_size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    );
    TR_CHECK(workshop_ != nullptr, MemoryError,
             "VirtualAlloc failed for PW " << config_.worker_id
             << "\n  Size: " << (workshop_size_ / (1024.0*1024.0)) << " MB"
             << "\n  Error: " << GetLastError());
#else
    constexpr size_t PAGE_SIZE = 4096;
    int ret = posix_memalign(
        reinterpret_cast<void**>(&workshop_),
        PAGE_SIZE,
        workshop_size_
    );
    TR_CHECK(ret == 0 && workshop_ != nullptr, MemoryError,
             "posix_memalign failed for PW " << config_.worker_id
             << "\n  Size: " << (workshop_size_ / (1024.0*1024.0)) << " MB"
             << "\n  Return code: " << ret);
#endif
    
    // ✅ 修正（V2.1）：移除立即memset，延迟到ensure_first_touch()
    // std::memset(workshop_, 0, workshop_size_);  // ← 删除
    
    LOG_DEBUG << "PW " << config_.worker_id << " workshop allocated at "
              << static_cast<void*>(workshop_)
              << ", First-Touch deferred";
    
    // ==================== 划分各区（64字节对齐）====================
    uint8_t* ptr = workshop_;
    
    region_d_ = ptr;
    LOG_DEBUG << "PW " << config_.worker_id << " D区: " << static_cast<void*>(ptr)
              << ", size=" << (config_.region_d_size / 1024) << " KB";
    ptr += align_64(config_.region_d_size);
    
    region_a_ = ptr;
    LOG_DEBUG << "PW " << config_.worker_id << " A区: " << static_cast<void*>(ptr)
              << ", size=" << (config_.region_ab_size / 1024) << " KB";
    ptr += align_64(config_.region_ab_size);
    
    region_b_ = ptr;
    LOG_DEBUG << "PW " << config_.worker_id << " B区: " << static_cast<void*>(ptr)
              << ", size=" << (config_.region_ab_size / 1024) << " KB";
    ptr += align_64(config_.region_ab_size);
    
    if (config_.region_t_size > 0) {
        region_t_ = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " T区: " << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_t_size / 1024) << " KB";
        ptr += align_64(config_.region_t_size);
    }
    
    // S区对齐到4KB页边界
    region_s_.resize(config_.num_region_s);
    for (int i = 0; i < config_.num_region_s; ++i) {
        region_s_[i] = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " S" << (i+1) << "区: " 
                  << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_s_size / (1024.0*1024.0)) << " MB";
        ptr += align_4k(config_.region_s_size);
    }
    
    if (config_.region_c_size > 0) {
        region_c_ = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " C区: " << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_c_size / (1024.0*1024.0)) << " MB";
        ptr += align_64(config_.region_c_size);
    }
    
    // 验证指针未越界
    TR_CHECK(ptr <= workshop_ + workshop_size_, MemoryError,
             "Workshop allocation overflow: " << (ptr - workshop_) 
             << " > " << workshop_size_);
}

void PreprocessWorker::free_workshop() {
    if (workshop_) {
#ifdef _WIN32
        VirtualFree(workshop_, 0, MEM_RELEASE);
#else
        free(workshop_);
#endif
        workshop_ = nullptr;
        
        LOG_DEBUG << "PW " << config_.worker_id << " workshop freed";
    }
}

```

---

##### 5.4 ensure_first_touch（延迟First-Touch）

```cpp
void PreprocessWorker::ensure_first_touch() {
    if (!workshop_touched_) {
        // ✅ 关键（V2.1）：在worker线程绑核后首次触发
        // 确保内存分配在本地NUMA节点
        std::memset(workshop_, 0, workshop_size_);
        workshop_touched_ = true;
        
        LOG_DEBUG << "PW " << config_.worker_id << " First-Touch completed, "
                  << "workshop at " << static_cast<void*>(workshop_);
    }
}

```

---

##### 5.5 ensure_rng_initialized（延迟Generator初始化）

```cpp
void PreprocessWorker::ensure_rng_initialized() {
    if (!rng_initialized_) {
        // ✅ 修正（V2.1）：在首次work()时初始化，使用用户设置的seed
        uint64_t base_seed = get_default_generator().seed();
        uint64_t worker_seed = base_seed ^ (static_cast<uint64_t>(config_.worker_id) << 16);
        rng_.set_seed(worker_seed);
        rng_initialized_ = true;
        
        LOG_DEBUG << "PW " << config_.worker_id << " RNG initialized with seed=" << worker_seed;
    }
}

```

---

##### 5.6 update_parameters（参数更新）

```cpp
void PreprocessWorker::update_parameters(const PreprocessWorkerParameter& param) {
    param_ = param;
    
    // ✅ 修正（V2.1）：每个Phase开始时重置样本计数器
    // 采纳B1+B5意见4
    total_samples_processed_ = 0;
    
    // ✅ 修正（V2.1）：验证阶段开始时重置C区读取索引
    // 采纳B3+B5意见5
    if (!param_.is_train) {
        c_read_idx_ = 0;
        LOG_DEBUG << "PW " << config_.worker_id << " reset C区读取索引";
    }
    
    LOG_DEBUG << "PW " << config_.worker_id << " parameters updated: "
              << (param_.is_train ? "TRAIN" : "VAL")
              << ", total_samples_processed_ reset to 0";
    
    // ==================== 更新PO的output_size（渐进式分辨率）====================
    int target_resolution = param_.is_train ? param_.current_train_resolution
                                             : param_.current_val_resolution;
    
    const auto& ops = param_.is_train ? train_ops_ : val_ops_;
    for (const auto& op : ops) {
        if (op->is_crop() || op->is_resize()) {
            op->set_output_size(target_resolution);
        }
    }
}

```

---

##### 5.7 shuffle_s_region（S区洗牌）

```cpp
void PreprocessWorker::shuffle_s_region(int s_region_idx, int epoch_id) {
    TR_CHECK(s_region_idx >= 0 && s_region_idx < config_.num_region_s, ValueError,
             "Invalid s_region_idx: " << s_region_idx);
    
    // ✅ 修正（V2.1）：使用独立的样本计数
    // 采纳B1+B3+B5意见2
    size_t num_samples = num_samples_per_s_[s_region_idx];
    
    if (num_samples == 0) {
        LOG_DEBUG << "PW " << config_.worker_id << " S" << s_region_idx 
                  << " is empty, skip shuffle";
        return;
    }
    
    // ✅ 修正（V2.1）：使用独立的索引数组
    auto& indices = s_shuffle_indices_[s_region_idx];
    
    if (indices.size() != num_samples) {
        indices.resize(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            indices[i] = i;
        }
    }
    
    // ==================== Fisher-Yates洗牌 ====================
    // ✅ 修正（V2.1）：seed只使用epoch_id，移除s_region_idx
    // 采纳B4+B5意见7
    uint64_t base_seed = get_default_generator().seed();
    uint64_t shuffle_seed = base_seed ^
        (static_cast<uint64_t>(epoch_id) << 32);
    
    uint32_t n = static_cast<uint32_t>(num_samples);
    for (uint32_t i = n - 1; i > 0; --i) {
        uint32_t r[4];
        detail::philox_generate_4x32(shuffle_seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(indices[i], indices[j]);
    }
    
    // ✅ 修正：重置独立的读取索引
    s_read_indices_[s_region_idx] = 0;
    
    LOG_DEBUG << "PW " << config_.worker_id << " shuffled S" << s_region_idx
              << ": " << n << " samples, seed=" << shuffle_seed;
}

```

---

##### 5.8 work（核心方法）

```cpp
bool PreprocessWorker::work(
    int32_t label,
    const uint8_t* data_ptr,
    size_t data_size,
    EngineBuffer& engine_buffer
) {
    // ✅ 修正（V2.1）：首次调用时确保First-Touch已执行
    // 采纳B5意见1
    ensure_first_touch();
    
    // ✅ 修正（V2.1）：首次调用时确保RNG已初始化
    // 采纳B3意见8
    ensure_rng_initialized();
    
    const bool is_train = param_.is_train;
    const bool is_busy = param_.is_busy_epoch;
    const int res = is_train ? param_.current_train_resolution 
                             : param_.current_val_resolution;
    const size_t sample_bytes = res * res * config_.num_color_channels;
    const size_t sample_stride = transforms::PreprocessOperation::calculate_stride(
        res, config_.num_color_channels);
    
    // ==================== 路径1: Lazy Epoch（从S区读取）====================
    if (is_train && !is_busy && config_.sdmp_factor > 1) {
        int s_idx = param_.active_s_region_idx;
        
        // ✅ 修正（V2.1）：使用独立的s_read_indices_和num_samples_per_s_
        // 采纳B1+B3+B5意见2
        if (s_read_indices_[s_idx] >= num_samples_per_s_[s_idx]) {
            return false;  // 此S区已读完
        }
        
        size_t actual_idx = s_shuffle_indices_[s_idx][s_read_indices_[s_idx]];
        s_read_indices_[s_idx]++;
        
        const uint8_t* s_ptr = region_s_[s_idx] + actual_idx * sample_bytes;
        int32_t s_label = region_s_labels_[actual_idx];
        
        // ✅ 修正（V2.1）：计算批次ID并传递
        // 采纳B1+B2+B5意见1
        int batch_id = static_cast<int>(total_samples_processed_ / config_.local_batch_size);
        
        // 写入EngineBuffer
        auto [position, valid] = calculate_write_position();
        if (!valid) return false;
        
        engine_buffer.write_at(position, batch_id, s_label, s_ptr, sample_bytes);
        engine_buffer.notify_sample_written();
        
        total_samples_processed_++;
        return true;
    }
    
    // ==================== 路径2: CPVS非首次（从C区读取）====================
    if (!is_train && !param_.is_first_val && config_.using_cpvs) {
        if (c_read_idx_ >= num_samples_in_c_) {
            return false;  // C区已读完
        }
        
        const uint8_t* c_ptr = region_c_ + c_read_idx_ * sample_bytes;
        int32_t c_label = region_c_labels_[c_read_idx_];
        c_read_idx_++;
        
        int batch_id = static_cast<int>(total_samples_processed_ / config_.local_batch_size);
        
        auto [position, valid] = calculate_write_position();
        if (!valid) return false;
        
        engine_buffer.write_at(position, batch_id, c_label, c_ptr, sample_bytes);
        engine_buffer.notify_sample_written();
        
        total_samples_processed_++;
        return true;
    }
    
    // ==================== 路径3: Busy或首次（需预处理）====================
    
    const auto& ops = is_train ? train_ops_ : val_ops_;
    
    // ==================== 3.1 获取图像尺寸 ====================
    int32_t image_width = config_.raw_image_width;
    int32_t image_height = config_.raw_image_height;
    
    if (config_.dataset_type == DatasetType::imagenet) {
        // ImageNet：必须先读JPEG头
        if (tj3DecompressHeader(tj_handle_, data_ptr, data_size) != 0) {
            LOG_WARN << "PW " << config_.worker_id << " failed to read JPEG header, skip sample";
            return true;  // 跳过损坏样本（符合PyTorch标准）
        }
        
        image_width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
        image_height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);
    }
    
    // ==================== 3.2 获取解码策略 ====================
    transforms::DecodeStrategy strategy;
    const uint8_t* initial_ptr = nullptr;
    int32_t initial_w = 0, initial_h = 0;
    size_t initial_stride = 0;
    
    if (!ops.empty()) {
        strategy = ops[0]->get_decode_strategy(
            image_width, image_height,
            config_.sdmp_factor, &rng_
        );
    }
    
    // ==================== 3.3 解码（如果需要）====================
    if (strategy.need_decode) {
        if (strategy.use_partial) {
            // 局部解码
            if (!decode_partial(data_ptr, data_size, strategy, initial_stride)) {
                LOG_WARN << "PW " << config_.worker_id << " partial decode failed, skip sample";
                return true;
            }
            
            // 计算精确裁剪区域的起始指针（相对于解码窗口）
            initial_ptr = region_d_ + strategy.crop_y * initial_stride 
                        + strategy.crop_x * config_.num_color_channels;
            initial_w = strategy.crop_w;
            initial_h = strategy.crop_h;
            
        } else {
            // 完整解码
            if (!decode_full(data_ptr, data_size, initial_w, initial_h, initial_stride)) {
                LOG_WARN << "PW " << config_.worker_id << " full decode failed, skip sample";
                return true;
            }
            initial_ptr = region_d_;
        }
    } else {
        // 非ImageNet：直接使用输入
        initial_ptr = data_ptr;
        initial_w = image_width;
        initial_h = image_height;
        initial_stride = image_width * config_.num_color_channels;
    }
    
    // ==================== 3.4 SDMP循环：多次预处理 ====================
    // ✅ 文档澄清（V2.1）：每次循环独立调用should_flip()
    // 采纳B2+B5意见3
    int num_loops = is_train ? config_.sdmp_factor : 1;
    
    for (int loop = 0; loop < num_loops; ++loop) {
        uint8_t* target_ptr;
        int target_s_idx = -1;
        
        if (loop < num_loops - 1) {
            // 前(sdmp_factor-1)次 → S区
            target_s_idx = loop;
            target_ptr = region_s_[target_s_idx] + num_samples_in_s_ * sample_bytes;
            
            LOG_DEBUG << "PW " << config_.worker_id << " SDMP loop " << loop
                      << " -> S" << target_s_idx << "[" << num_samples_in_s_ << "]";
        } else {
            // 最后一次 → 临时使用A区（后面复制到EngineBuffer）
            target_ptr = region_a_;
            
            LOG_DEBUG << "PW " << config_.worker_id << " SDMP loop " << loop
                      << " -> A区（最终输出）";
        }
        
        // 执行PO链
        // ✅ 注意：每次循环都会独立调用should_flip()，确保随机性独立
        int32_t out_w, out_h;
        execute_po_chain(
            initial_ptr, initial_w, initial_h, initial_stride,
            target_ptr, sample_stride,
            ops, out_w, out_h
        );
        
        // ✅ 修正（V2.1）：标签管理优化
        if (target_s_idx >= 0) {
            // 仅在首次存入S区时保存标签（同一原始样本的标签相同）
            if (loop == 0) {
                region_s_labels_.push_back(label);
                
                // ✅ 修正：所有S区的样本数同步递增
                for (int i = 0; i < config_.num_region_s; ++i) {
                    num_samples_per_s_[i]++;
                }
            }
        }
    }
    
    // ✅ 统一更新num_samples_in_s_（所有S区应相同）
    if (is_train && config_.sdmp_factor > 1) {
        num_samples_in_s_++;
        
        // Debug验证：所有S区样本数应相同
#ifndef NDEBUG
        for (int i = 0; i < config_.num_region_s; ++i) {
            TR_CHECK(num_samples_per_s_[i] == num_samples_in_s_, ValueError,
                     "S区样本数不一致: S" << i << " has " << num_samples_per_s_[i]
                     << ", expected " << num_samples_in_s_);
        }
#endif
    }
    
    // ==================== 3.5 CPVS首次：写C区 ====================
    if (!is_train && param_.is_first_val && config_.using_cpvs) {
        std::memcpy(region_c_ + num_samples_in_c_ * sample_bytes,
                   region_a_, sample_bytes);
        region_c_labels_.push_back(label);
        num_samples_in_c_++;
        
        LOG_DEBUG << "PW " << config_.worker_id << " saved to C区[" 
                  << (num_samples_in_c_ - 1) << "]";
    }
    
    // ==================== 3.6 写入EngineBuffer ====================
    // ✅ 修正（V2.1）：计算批次ID并传递
    // 采纳B1+B2+B5意见1
    int batch_id = static_cast<int>(total_samples_processed_ / config_.local_batch_size);
    
    auto [position, valid] = calculate_write_position();
    if (!valid) return false;
    
    engine_buffer.write_at(position, batch_id, label, region_a_, sample_bytes);
    engine_buffer.notify_sample_written();
    
    total_samples_processed_++;
    return true;
}

```

---

##### 5.9 execute_po_chain（PO链执行）

```cpp
void PreprocessWorker::execute_po_chain(
    const uint8_t* initial_ptr,
    int32_t initial_width,
    int32_t initial_height,
    size_t initial_stride,
    uint8_t* final_output_ptr,
    size_t final_stride,
    const std::vector<std::unique_ptr<transforms::PreprocessOperation>>& ops,
    int32_t& final_width,
    int32_t& final_height
) {
    if (ops.empty()) {
        // 无操作：直接复制
        final_width = initial_width;
        final_height = initial_height;
        
        size_t row_bytes = initial_width * config_.num_color_channels;
        for (int y = 0; y < initial_height; ++y) {
            std::memcpy(final_output_ptr + y * final_stride,
                       initial_ptr + y * initial_stride,
                       row_bytes);
        }
        return;
    }
    
    // ==================== 检测RandomHorizontalFlip ====================
    bool has_flip = false;
    bool should_flip_flag = false;
    transforms::RandomHorizontalFlip* flip_op = nullptr;
    size_t num_ops_before_flip = ops.size();
    
    if (ops.back()->is_random_horizontal_flip()) {
        has_flip = true;
        flip_op = static_cast<transforms::RandomHorizontalFlip*>(ops.back().get());
        num_ops_before_flip--;
        
        // ✅ 关键：只调用一次should_flip()
        should_flip_flag = flip_op->should_flip(&rng_);
        
        LOG_DEBUG << "PW " << config_.worker_id << " Flip decision: " 
                  << (should_flip_flag ? "YES" : "NO");
    }
    
    // ==================== 执行除Flip外的所有操作 ====================
    const uint8_t* current_in = initial_ptr;
    int32_t current_w = initial_width;
    int32_t current_h = initial_height;
    size_t current_stride = initial_stride;
    
    bool use_a = true;  // AB区乒乓标志
    
    for (size_t i = 0; i < num_ops_before_flip; ++i) {
        uint8_t* current_out;
        size_t out_stride;
        
        if (i == num_ops_before_flip - 1) {
            // ==================== 最后一个操作（Flip之前）====================
            if (!has_flip || !should_flip_flag) {
                // 情况1：无Flip或不需要Flip → 直接输出到final
                current_out = final_output_ptr;
                out_stride = final_stride;
            } else {
                // 情况2：需要Flip → 输出到AB区
                current_out = use_a ? region_a_ : region_b_;
                
                int next_size = ops[i]->get_output_size();
                if (next_size == 0) next_size = res;
                out_stride = transforms::PreprocessOperation::calculate_stride(
                    next_size, config_.num_color_channels);
            }
        } else {
            // ==================== 中间操作 → AB区乒乓 ====================
            current_out = use_a ? region_a_ : region_b_;
            
            int next_size = ops[i]->get_output_size();
            if (next_size == 0) next_size = res;
            out_stride = transforms::PreprocessOperation::calculate_stride(
                next_size, config_.num_color_channels);
        }
        
        // 执行当前操作
        int32_t out_w, out_h;
        ops[i]->execute(
            current_in, current_w, current_h, current_stride,
            current_out, out_w, out_h, out_stride,
            &rng_
        );
        
        // 更新输入为当前输出（下一轮迭代使用）
        current_in = current_out;
        current_w = out_w;
        current_h = out_h;
        current_stride = out_stride;
        
        // 切换AB区
        use_a = !use_a;
    }
    
    // ==================== 执行Flip（如果需要）====================
    if (has_flip && should_flip_flag) {
        flip_op->execute(
            current_in, current_w, current_h, current_stride,
            final_output_ptr, final_width, final_height, final_stride,
            nullptr  // ✅ Flip的execute不使用RNG
        );
    } else {
        // 不需要Flip或无Flip操作，直接使用最后输出
        final_width = current_w;
        final_height = current_h;
    }
}

```

---

##### 5.10 decode_full（完整解码）

```cpp
bool PreprocessWorker::decode_full(
    const uint8_t* jpeg_data,
    size_t jpeg_size,
    int32_t& width,
    int32_t& height,
    size_t& stride
) {
    // TurboJPEG 3.x API已在work()中调用过Header，这里直接读取
    width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
    height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);
    
    // 计算stride（64字节对齐）
    stride = transforms::PreprocessOperation::calculate_stride(
        width, config_.num_color_channels);
    
    // 验证D区容量
    size_t required_size = stride * height;
    
    if (required_size > config_.region_d_size) {
        // ✅ 按【七】的要求：不扩大D区，跳过超大样本
        LOG_WARN << "PW " << config_.worker_id << " image too large for D区: "
                 << "need " << (required_size / (1024.0*1024.0)) << " MB, "
                 << "have " << (config_.region_d_size / (1024.0*1024.0)) << " MB"
                 << ", skip sample";
        return false;
    }
    
    // 解码
    if (tj3Decompress8(tj_handle_, jpeg_data, jpeg_size,
                      region_d_, static_cast<int>(stride), TJPF_RGB) != 0) {
        LOG_ERROR << "PW " << config_.worker_id << " tj3Decompress8 failed: "
                  << tj3GetErrorStr(tj_handle_);
        return false;
    }
    
    return true;
}

```

---

##### 5.11 decode_partial（局部解码）

```cpp
bool PreprocessWorker::decode_partial(
    const uint8_t* jpeg_data,
    size_t jpeg_size,
    const transforms::DecodeStrategy& strategy,
    size_t& stride
) {
    // 设置裁剪区域（MCU对齐）
    tjregion crop_region;
    crop_region.x = strategy.decode_x;
    crop_region.y = strategy.decode_y;
    crop_region.w = strategy.decode_w;
    crop_region.h = strategy.decode_h;
    
    if (tj3SetCroppingRegion(tj_handle_, crop_region) != 0) {
        LOG_ERROR << "PW " << config_.worker_id << " tj3SetCroppingRegion failed: "
                  << tj3GetErrorStr(tj_handle_);
        return false;
    }
    
    // stride基于decode_w计算（不是crop_w）
    stride = transforms::PreprocessOperation::calculate_stride(
        strategy.decode_w, config_.num_color_channels);
    
    // 验证D区容量
    size_t required_size = stride * strategy.decode_h;
    
    if (required_size > config_.region_d_size) {
        LOG_WARN << "PW " << config_.worker_id << " decode region too large, skip sample";
        return false;
    }
    
    // 解码（只解码裁剪区域）
    if (tj3Decompress8(tj_handle_, jpeg_data, jpeg_size,
                      region_d_, static_cast<int>(stride), TJPF_RGB) != 0) {
        LOG_ERROR << "PW " << config_.worker_id << " tj3Decompress8 (partial) failed: "
                  << tj3GetErrorStr(tj_handle_);
        return false;
    }
    
    return true;
}

```

---

##### 5.12 calculate_write_position（位置计算）

```cpp
std::pair<int, bool> PreprocessWorker::calculate_write_position() {
    // 核心公式：position = (n * M + j) % B
    // n: 该PW的全局样本计数（从0开始，跨batch连续）
    // M: num_workers_per_engine
    // j: pid_in_engine（固定）
    // B: local_batch_size
    
    const int M = config_.num_workers_per_engine;
    const int j = config_.pid_in_engine;
    const int B = config_.local_batch_size;
    
    // n使用total_samples_processed_（在work()末尾递增）
    size_t n = total_samples_processed_;
    
    int position = static_cast<int>((n * M + j) % B);
    
    // Debug模式边界检查
#ifndef NDEBUG
    TR_CHECK(position >= 0 && position < B, ValueError,
             "Position out of range: " << position << ", batch_size=" << B);
#endif
    
    return {position, true};
}

```

---

#### 【六、EngineBuffer 设计】

##### 6.1 抽象基类

```cpp
/**
 * @file engine_buffer.h
 * @brief Engine缓冲区抽象基类
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 * @note 所属系列: data
 * 
 * 修正记录（V2.1）：
 * - 采纳B1+B2+B5意见1：write_at增加batch_id参数
 * - 增加批次边界保护机制
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace tr {

/**
 * @class EngineBuffer
 * @brief Engine缓冲区（双缓冲+零竞争写入+批次边界保护）
 * 
 * 设计原则：
 * 1. 静态位置分配：写入位置由PW计算，EngineBuffer不检查冲突
 * 2. 原子计数触发：最后一个写完的PW自动触发传输
 * 3. 双缓冲流水：写入和传输并行
 * 4. ✅ 批次边界保护（V2.1新增）：防止快Worker覆盖慢Worker数据
 * 
 * 零竞争证明：
 * 设PW₁的(n₁, j₁), PW₂的(n₂, j₂)
 * 写入位置公式: pos = (n*M + j) % B
 * 其中M=num_workers_per_engine, j=pid (0~M-1)
 * 
 * 因为不同PW的j值不同，且n是各自独立的计数器，
 * 所以(n₁*M + j₁) ≠ (n₂*M + j₂) (mod B)，
 * 即不同PW永远不会写入同一位置 ∎
 * 
 * 批次边界保护（V2.1新增）：
 * - 引入逻辑Batch ID：batch_id = total_samples / local_batch_size
 * - 快Worker如果跑到下一个batch，必须等待当前batch传输完成
 * - 防止覆盖尚未传输的数据
 */
class EngineBuffer {
public:
    virtual ~EngineBuffer() = default;
    
    // =========================================================================
    // 配置接口
    // =========================================================================
    
    /**
     * @brief 配置缓冲区参数
     */
    virtual void configure(
        int local_batch_size,
        size_t max_train_sample_bytes,
        size_t max_val_sample_bytes,
        int num_workers_per_engine
    ) = 0;
    
    /**
     * @brief 更新phase参数
     */
    virtual void update_phase(
        bool is_train,
        int current_resolution,
        int num_color_channels
    ) = 0;
    
    // =========================================================================
    // 写入接口（零竞争 + 批次保护）
    // =========================================================================
    
    /**
     * @brief 写入指定位置（带批次保护）
     * @param position Batch内位置（0 ~ local_batch_size-1）
     * @param batch_id 逻辑Batch ID（防止快Worker覆盖慢Worker）
     * @param label 标签
     * @param data_ptr 数据指针
     * @param data_size 数据大小
     * 
     * @note ✅ V2.1修正：增加batch_id参数，实现批次边界保护
     * @note position由PW计算，EngineBuffer不检查冲突
     * @note 此方法在batch_id匹配时无锁，否则阻塞等待
     */
    virtual void write_at(
        int position,
        int batch_id,      // ✅ 新增（V2.1）
        int32_t label,
        const uint8_t* data_ptr,
        size_t data_size
    ) = 0;
    
    /**
     * @brief 通知一个样本写入完成
     * @return true=触发了传输, false=等待其他PW
     * 
     * @note 使用原子计数器，最后一个PW自动触发传输
     */
    virtual bool notify_sample_written() = 0;
    
    /**
     * @brief 等待当前buffer可写
     * @param timeout_ms 超时时间（毫秒）
     * @return true=可写, false=超时
     */
    virtual bool wait_writable(int timeout_ms = 5000) = 0;
    
    // =========================================================================
    // 传输控制（子类实现）
    // =========================================================================
    
    virtual void trigger_async_transfer() = 0;
    virtual bool is_transfer_complete() const = 0;
    
    // =========================================================================
    // 状态查询
    // =========================================================================
    
    virtual size_t total_samples_transferred() const = 0;
    virtual int current_buffer_id() const = 0;
};

} // namespace tr
```

---

