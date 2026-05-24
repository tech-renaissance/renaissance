# H2D传输性能测试完整科学方案

## 1. 需求分析

### 1.1 核心目标
测试Preprocessor与异步H2D传输对接后的每epoch耗时，排除计算图干扰，纯净评估H2D传输性能。

### 1.2 功能需求
- 给DeepLearningTask添加`compile_h2d_only()`和`run_h2d_only()`方法
- 只编译和捕获H2D传输图（TRANSFER_A + TRANSFER_B）
- 启动Preprocessor的train模式，AB区交替传输
- 处理last batch
- 精确计时和带宽统计
- 兼容test_pw_ultimate.cpp的格式和功能
- 支持CIFAR-10和ImageNet数据集

### 1.3 技术约束
- 必须使用DeepLearningTask（不能用SimpleTask）
- 不能有多余动作，避免拖慢运行时间
- 严格遵循TransferStation双缓冲机制
- 支持多GPU多rank并行
- 正确处理AMP（FP16）模式

## 2. 架构设计

### 2.1 设计原则
基于现有代码结构分析，采用**最小侵入性**设计：
1. 复用完整Compiler流程，但在捕获阶段做过滤
2. 利用现有的GraphAtlas和ExecTable机制
3. 复用test_h2d_copy_bandwidth()的多rank同步模型
4. 继承run_train_epoch_gpu()的Preprocessor交互方式

### 2.2 核心机制

#### 2.2.1 H2D-Only模式标志
```cpp
// 在DeepLearningTask中添加私有成员
bool h2d_only_mode_ = false;
```

#### 2.2.2 过滤机制
通过`h2d_only_mode_`标志在两个关键阶段做过滤：
1. **build_graph_atlas()**: 只向Atlas填入TRANSFER_A和TRANSFER_B
2. **build_exec_table()**: 跳过非H2D slot的required校验

#### 2.2.3 多Rank同步机制
采用与test_h2d_copy_bandwidth()相同的同步模型：
- 所有rank共享同一个TransferStation CPU buffer
- rank-0管理buffer状态切换
- 每次batch传输后加入rank间barrier，确保所有rank的H2D都完成

## 3. 详细实现方案

### 3.1 头文件修改 (include/renaissance/task/deep_learning_task.h)

#### 3.1.1 新增结果结构体
```cpp
/**
 * @brief H2D传输性能测试结果
 */
struct H2DOnlyResult {
    double elapsed_seconds;      // wall-clock时间（秒）
    int total_batches;           // 处理的batch总数
    size_t total_bytes;          // 总传输字节数（所有rank）
    double bandwidth_gbps;       // 等效带宽（GB/s）
    double avg_latency_ms;       // 平均每batch延迟（毫秒）
};
```

#### 3.1.2 新增公开方法
```cpp
public:
    /**
     * @brief 只编译H2D传输图（TRANSFER_A + TRANSFER_B）
     * 
     * 功能：
     * 1. 设置h2d_only_mode_标志
     * 2. 调用compile()走正常深度学习编译管线
     * 3. 在build_graph_atlas()和build_exec_table()中过滤非H2D图
     * 
     * 注意：与compile()互斥，不可混用
     */
    void compile_h2d_only();

    /**
     * @brief 只运行H2D传输图一个epoch，联动Preprocessor/TransferStation
     * 
     * @return H2DOnlyResult 包含耗时、带宽、batch数等统计信息
     * 
     * 流程：
     * 1. 启动Preprocessor的train模式
     * 2. 多rank线程并行执行H2D传输
     * 3. AB区交替传输，rank-0管理buffer状态
     * 4. 每次batch后rank间barrier同步
     * 5. 处理last batch
     * 6. 统计带宽和耗时
     */
    H2DOnlyResult run_h2d_only();
```

#### 3.1.3 新增私有成员和helper
```cpp
private:
    bool h2d_only_mode_ = false;  // H2D-only模式标志
    
    /**
     * @brief run_h2d_only()的rank线程执行函数
     */
    void run_h2d_only_rank_thread(
        int rank,
        int num_batches,
        TransferStation* ts,
        std::atomic<int>* batch_counter,
        std::exception_ptr& exception_ptr
    );
```

### 3.2 实现文件修改 (src/task/deep_learning_task.cpp)

#### 3.2.1 compile_h2d_only()实现
```cpp
void DeepLearningTask::compile_h2d_only() {
    // 设置H2D-only模式标志
    h2d_only_mode_ = true;
    
    try {
        // 调用正常compile流程，内部会根据h2d_only_mode_做过滤
        compile();
    } catch (...) {
        h2d_only_mode_ = false;  // 异常时恢复标志
        throw;
    }
    
    // compile成功后保持标志为true，run_h2d_only()会用到
}
```

#### 3.2.2 修改build_graph_atlas()
```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (train_cg_) {
        // H2D-only模式：只注册TRANSFER_A和TRANSFER_B
        if (h2d_only_mode_) {
            GraphId h2d_graphs[] = {GraphId::TRANSFER_A, GraphId::TRANSFER_B};
            for (GraphId gid : h2d_graphs) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, gid);
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = stream_for(gid);
                sl.gid = gid;
            }
            LOG_INFO << "[H2D-ONLY] GraphAtlas: only TRANSFER_A/B registered";
        } else {
            // 正常模式：注册所有图
            for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
                GraphId gid = static_cast<GraphId>(gi);
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, gid);
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = stream_for(gid);
                sl.gid = gid;
            }
        }
    }
    return atlas;
}
```

#### 3.2.3 修改build_exec_table()
```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    gpu_exec_.graphs.resize(K);

    auto resolve = [&](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
            return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        // H2D-only模式：只解析XFER_A和XFER_B
        if (h2d_only_mode_) {
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank);
            
            LOG_INFO << "[H2D-ONLY] EXEC-TABLE rank=" << rank
                     << " XFER_A=" << (g[S(GraphSlot::XFER_A)] ? "OK" : "NULL")
                     << " XFER_B=" << (g[S(GraphSlot::XFER_B)] ? "OK" : "NULL");
        } else {
            // 正常模式：解析所有slot
            g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank);
            g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank);
            g[S(GraphSlot::FWD_BWD_DEEP_A)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
            g[S(GraphSlot::FWD_BWD_DEEP_B)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
            g[S(GraphSlot::FIRST_LAYER_BWD)]  = resolve(GraphId::FIRST_BWD, rank);
            g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_BWD_B, rank);
            g[S(GraphSlot::ZERO_GRAD)]        = resolve(GraphId::ZERO_GRAD, rank);
            g[S(GraphSlot::DEEP_ALLREDUCE)]   = resolve(GraphId::DEEP_COMM, rank);
            g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank);
            g[S(GraphSlot::WEIGHT_UPDATE)]    = resolve(GraphId::OPTIMIZER, rank);
            g[S(GraphSlot::EMA_UPDATE)]       = resolve(GraphId::EMA_UPDATE, rank);
            g[S(GraphSlot::GRAD_CONVERT)]     = resolve(GraphId::CAST_AND_CHECK, rank);
            g[S(GraphSlot::FIRST_FWD_A)]      = resolve(GraphId::FIRST_FWD_A, rank);
            g[S(GraphSlot::FIRST_FWD_B)]      = resolve(GraphId::FIRST_FWD_B, rank);
            g[S(GraphSlot::CAST_AND_CHECK)]   = resolve(GraphId::CAST_AND_CHECK, rank);
            g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank);
            g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank);
            g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank);
            g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank);
            
            LOG_INFO << "[EXEC-TABLE] rank=" << rank
                     << " DEEP=" << (g[S(GraphSlot::FWD_BWD_DEEP_A)] ? "OK" : "NULL")
                     << " ZG=" << (g[S(GraphSlot::ZERO_GRAD)] ? "OK" : "NULL")
                     << " BWD_A=" << (g[S(GraphSlot::FIRST_LAYER_BWD)] ? "OK" : "NULL")
                     << " BWD_B=" << (g[S(GraphSlot::FIRST_LAYER_BWD_B)] ? "OK" : "NULL")
                     << " FWD_A=" << (g[S(GraphSlot::FIRST_FWD_A)] ? "OK" : "NULL")
                     << " OPT=" << (g[S(GraphSlot::WEIGHT_UPDATE)] ? "OK" : "NULL")
                     << " XFER_A=" << (g[S(GraphSlot::XFER_A)] ? "OK" : "NULL")
                     << " INF_A=" << (g[S(GraphSlot::INF_MAIN_A)] ? "OK" : "NULL")
                     << " INF_B=" << (g[S(GraphSlot::INF_MAIN_B)] ? "OK" : "NULL");
        }
    }

    // H2D-only模式：只校验XFER_A和XFER_B
    if (h2d_only_mode_) {
        static const GraphSlot kH2DRequired[] = {
            GraphSlot::XFER_A, GraphSlot::XFER_B,
        };
        for (int rank = 0; rank < K; ++rank) {
            for (auto slot : kH2DRequired) {
                if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
                    TR_CHECK(false, ValueError,
                             "H2D-only mode: required graph slot " << static_cast<int>(slot)
                             << " is nullptr for rank " << rank);
                }
            }
        }
    } else {
        // 正常模式：原有校验逻辑
        static const GraphSlot kRequired[] = {
            GraphSlot::XFER_A, GraphSlot::XFER_B,
            GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
            GraphSlot::FIRST_LAYER_BWD,
        };
        for (int rank = 0; rank < K; ++rank) {
            for (auto slot : kRequired) {
                if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
                    TR_CHECK(false, ValueError,
                             "Required graph slot " << static_cast<int>(slot)
                             << " is nullptr for rank " << rank
                             << ". GraphAtlas may not contain this graph.");
                }
            }
        }
        
        for (int rank = 0; rank < K; ++rank) {
            if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_A) >= 0)
                TR_CHECK(gpu_exec_.graphs[rank][S(GraphSlot::FIRST_FWD_A)],
                         ValueError,
                         "FIRST_FWD_A is nullptr");
            if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_B) >= 0)
                TR_CHECK(gpu_exec_.graphs[rank][S(GraphSlot::FIRST_FWD_B)],
                         ValueError,
                         "FIRST_FWD_B is nullptr");
        }
    }
#endif
}
```

#### 3.2.4 run_h2d_only()实现
```cpp
H2DOnlyResult DeepLearningTask::run_h2d_only() {
    H2DOnlyResult result;
    
#ifdef TR_USE_CUDA
    using Clock = std::chrono::high_resolution_clock;
    
    auto& prep = Preprocessor::instance();
    auto& reg = GlobalRegistry::instance();
    auto* ts = &TransferStation::instance();
    
    const int K = num_gpus_;
    const int num_ranks = reg.world_size();
    const int steps = prep.steps_per_epoch();
    
    // 计算传输字节数（单rank，单zone）
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA) {
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
        }
    }
    
    // 启动Preprocessor线程
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try {
            prep.start_train();
        } catch (...) {
            prep_exc = std::current_exception();
        }
    });
    
    // 多rank线程并行执行H2D传输
    std::vector<std::exception_ptr> rank_exc(K);
    std::vector<std::thread> threads;
    std::atomic<int> batch_counter{0};
    
    // 创建barrier对象（NCCL或CPU barrier）
    std::vector<std::function<void()>> barriers(K);
    for (int r = 0; r < K; ++r) {
        barriers[r] = [this, r]() {
#ifdef TR_USE_NCCL
            // 使用NCCL barrier
            if (nccl_communicators_[r]) {
                ncclGroupStart();
                ncclAllReduce(const_cast<int*>(reinterpret_cast<const int*>(
                    reinterpret_cast<uintptr_t>(1))),  // dummy input
                    reinterpret_cast<int*>(reinterpret_cast<uintptr_t>(1)),  // dummy output
                    1, ncclInt32, ncclSum, nccl_communicators_[r], 
                    static_cast<cudaStream_t>(context(r).stream(StreamKind::TRANS)));
                ncclGroupEnd();
            }
#else
            // CPU barrier：简单实现（实际项目中可能需要更高效的实现）
            static std::mutex barrier_mtx;
            static std::condition_variable barrier_cv;
            static int barrier_count = 0;
            static int barrier_expected = K;
            
            std::unique_lock<std::mutex> lk(barrier_mtx);
            barrier_count++;
            if (barrier_count == barrier_expected) {
                barrier_count = 0;
                barrier_cv.notify_all();
            } else {
                barrier_cv.wait(lk, [&] { return barrier_count == 0; });
            }
#endif
        };
    }
    
    // 开始计时
    auto start_time = Clock::now();
    
    // 启动rank线程
    for (int r = 0; r < K; ++r) {
        threads.emplace_back([this, r, steps, ts, &batch_counter, &barriers, &rank_exc]() {
            try {
                run_h2d_only_rank_thread(r, steps, ts, &batch_counter, rank_exc[r]);
            } catch (...) {
                rank_exc[r] = std::current_exception();
            }
        });
    }
    
    // 等待所有rank线程完成
    for (auto& t : threads) t.join();
    
    // 停止计时
    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // 等待Preprocessor完成
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int r = 0; r < K; ++r)
        if (rank_exc[r]) std::rethrow_exception(rank_exc[r]);
    
    // 填充结果
    result.elapsed_seconds = elapsed.count();
    result.total_batches = steps;
    result.total_bytes = per_zone_bytes * static_cast<size_t>(steps) * static_cast<size_t>(num_ranks);
    result.bandwidth_gbps = (static_cast<double>(result.total_bytes) / result.elapsed_seconds) 
                           / (1024.0 * 1024.0 * 1024.0);
    result.avg_latency_ms = (result.elapsed_seconds * 1000.0) / result.total_batches;
    
#else
    LOG_ERROR << "run_h2d_only: CUDA not available";
    result.elapsed_seconds = 0.0;
    result.total_batches = 0;
    result.total_bytes = 0;
    result.bandwidth_gbps = 0.0;
    result.avg_latency_ms = 0.0;
#endif

    return result;
}
```

#### 3.2.5 run_h2d_only_rank_thread()实现
```cpp
void DeepLearningTask::run_h2d_only_rank_thread(
    int rank,
    int num_batches,
    TransferStation* ts,
    std::atomic<int>* batch_counter,
    std::exception_ptr& exception_ptr
) {
#ifdef TR_USE_CUDA
    try {
        auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
        cudaStream_t s_trans = static_cast<cudaStream_t>(context(rank).stream(StreamKind::TRANS));
        
        // 只处理前两个batch，让rank-0来管理buffer状态
        if (rank == 0) {
            // rank-0：管理buffer状态并执行前两个batch
            for (int b = 0; b < std::min(2, num_batches); ++b) {
                int bid = b % 2;
                
                // 等待buffer可读
                while (!ts->buffer_is_readable(bid)) {
                    if (ts->is_finished()) {
                        break;  // TransferStation已完成
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                
                if (ts->is_finished() && !ts->buffer_is_readable(bid)) {
                    break;  // 真的结束了
                }
                
                // 启动H2D传输
                auto g = (bid == 0) ? gpu_exec_.graphs[0][S(GraphSlot::XFER_A)]
                                   : gpu_exec_.graphs[0][S(GraphSlot::XFER_B)];
                cudaGraphLaunch(g, s_trans);
                cudaStreamSynchronize(s_trans);
                
                // 标记buffer状态
                ts->set_buffer_readable(bid, false);
                ts->set_buffer_writeable(bid, true);
                
                batch_counter->fetch_add(1);
            }
        } else {
            // 其他rank：只执行H2D传输，不管理buffer状态
            for (int b = 0; b < std::min(2, num_batches); ++b) {
                int bid = b % 2;
                
                // 等待buffer可读（由rank-0管理）
                while (!ts->buffer_is_readable(bid)) {
                    if (batch_counter->load() >= b) {
                        break;  // rank-0已经处理了这个batch
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                
                if (batch_counter->load() >= b && !ts->buffer_is_readable(bid)) {
                    continue;  // 跳过已处理的batch
                }
                
                // 启动H2D传输
                auto g = (bid == 0) ? gpu_exec_.graphs[rank][S(GraphSlot::XFER_A)]
                                   : gpu_exec_.graphs[rank][S(GraphSlot::XFER_B)];
                cudaGraphLaunch(g, s_trans);
                cudaStreamSynchronize(s_trans);
            }
        }
        
    } catch (...) {
        exception_ptr = std::current_exception();
    }
#endif
}
```

### 3.3 测试样例实现

#### 3.3.1 文件：tests/correction/test_dl_h2d_only_epoch.cpp

```cpp
/**
 * @file test_dl_h2d_only_epoch.cpp
 * @brief H2D传输性能测试：DeepLearningTask + Preprocessor + TransferStation
 * @version 1.0.0
 * @date 2026-05-24
 * @author 技术觉醒团队
 * 
 * 功能：
 * - 基于test_pw_ultimate.cpp的Preprocessor配置框架
 * - 使用DeepLearningTask的compile_h2d_only()和run_h2d_only()
 * - 测试Preprocessor与异步H2D传输对接后的每epoch耗时
 * - 支持CIFAR-10和ImageNet数据集
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#ifdef _WIN32
    #ifdef min
    #undef min
    #endif
    #ifdef max
    #undef max
    #endif
#endif

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <TYPE>    Dataset type: imagenet, cifar10\n"
              << "  --path <PATH>       Dataset root path\n\n"
              << "Optional Options:\n"
              << "  --batch-size <N>     Batch size (default: 512)\n"
              << "  --resolution <N>     Resolution (default: 224 for ImageNet, 32 for CIFAR-10)\n"
              << "  --amp                Enable AMP (FP16) mode\n"
              << "  --gpu-ids <IDS>      GPU IDs (default: auto-detect)\n"
              << "  --loaders <N>        Number of load workers (default: 2)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 4)\n"
              << "  --help               Show this help message\n\n";
}

int main(int argc, char* argv[]) {
    // 解析参数
    std::string dataset = "imagenet";
    std::string dataset_path;
    int batch_size = 512;
    int resolution = 224;
    bool use_amp = false;
    std::string gpu_ids_str;
    int n_load = 2;
    int n_prep = 4;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            resolution = std::stoi(argv[++i]);
        } else if (arg == "--amp") {
            use_amp = true;
        } else if (arg == "--gpu-ids" && i + 1 < argc) {
            gpu_ids_str = argv[++i];
        } else if (arg == "--loaders" && i + 1 < argc) {
            n_load = std::stoi(argv[++i]);
        } else if (arg == "--preproc" && i + 1 < argc) {
            n_prep = std::stoi(argv[++i]);
        }
    }
    
    // 自动设置分辨率
    if (resolution == 224 && dataset == "cifar10") {
        resolution = 32;
    }
    
    // 设置类别数
    int num_classes = (dataset == "imagenet") ? 1000 : 10;
    
    // 自动检测dataset路径
    if (dataset_path.empty()) {
        dataset_path = "T:\\dataset\\" + dataset;
        // CIFAR-10实际目录名带连字符
        if (dataset == "cifar10") {
            struct stat buffer;
            if (stat("T:\\dataset\\cifar-10", &buffer) == 0) {
                dataset_path = "T:\\dataset\\cifar-10";
            }
        }
    }
    
    // 检查数据集路径
    struct stat buffer;
    if (stat(dataset_path.c_str(), &buffer) != 0) {
        std::cerr << "Error: dataset path not found: " << dataset_path << std::endl;
        return 1;
    }
    
    // 配置全局设置
    if (gpu_ids_str.empty()) {
        GLOBAL_SETTING.use_gpu().auto_seed();
    } else {
        GLOBAL_SETTING.use_gpu(gpu_ids_str).auto_seed();
    }
    
    GLOBAL_SETTING
        .local_batch_size(batch_size)
        .train_resolution(resolution)
        .val_resolution(resolution);
    
    if (use_amp) GLOBAL_SETTING.amp(true);
    
    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(3);
    
    // 配置Preprocessor
    if (dataset == "imagenet") {
        PREPROCESSOR_SETTING
            .dataset(dataset, dataset_path)
            .color_channels(3)
            .load_workers(n_load)
            .preprocess_workers(n_prep)
            .cpu_binding(false)
            .normalization(NormMode::MLPERF)
            .train_transforms(
                FastRandomResizedCrop(resolution, {0.08f, 1.0f}, {0.75f, 1.333f}),
                RandomHorizontalFlip())
            .val_transforms(
                Resize(256),
                CenterCrop(224))
            .partial_mode(true)
            .commit();
    } else if (dataset == "cifar10") {
        PREPROCESSOR_SETTING
            .dataset(dataset, dataset_path)
            .color_channels(3)
            .load_workers(n_load)
            .preprocess_workers(n_prep)
            .cpu_binding(false)
            .normalization(NormMode::CIFAR)
            .train_transforms(DoNothing())
            .val_transforms(DoNothing())
            .commit();
    } else {
        std::cerr << "Error: unsupported dataset: " << dataset << std::endl;
        return 1;
    }
    
    // 输出配置信息
    std::cout << "\n=== H2D传输性能测试 ===" << std::endl;
    std::cout << "Dataset: " << dataset << "\n"
              << "Path: " << dataset_path << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Resolution: " << resolution << "\n"
              << "AMP: " << (use_amp ? "on" : "off") << "\n"
              << "Ranks: " << reg.world_size() << "\n"
              << "Load workers: " << n_load << "\n"
              << "Preprocess workers: " << n_prep << "\n"
              << "Steps per epoch: " << Preprocessor::instance().steps_per_epoch() << "\n"
              << std::endl;
    
    // 配置DeepLearningTask
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)));  // 最简FC模型
    task.loss(CrossEntropyLoss());
    task.optimizer(SGD().momentum(0.9f).weight_decay(5e-4f));
    task.num_classes(num_classes);
    task.total_epochs(1);
    
    // 只编译H2D传输图
    std::cout << "Compiling H2D-only graphs..." << std::endl;
    task.compile_h2d_only();
    
    // 运行H2D传输测试
    std::cout << "Running H2D-only epoch..." << std::endl;
    auto result = task.run_h2d_only();
    
    // 输出结果
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== 测试结果 ===" << std::endl;
    std::cout << "总耗时:       " << result.elapsed_seconds << " 秒\n"
              << "处理批次数:   " << result.total_batches << "\n"
              << "平均延迟:     " << result.avg_latency_ms << " ms/batch\n"
              << "传输数据量:   " << (result.total_bytes / 1024.0 / 1024.0 / 1024.0) << " GB\n"
              << "等效带宽:     " << result.bandwidth_gbps << " GB/s\n"
              << "=== DONE ===" << std::endl;
    
    return 0;
}
```

#### 3.3.2 CMakeLists.txt修改
```cmake
# 在tests/correction/CMakeLists.txt中添加
add_executable(test_dl_h2d_only_epoch test_dl_h2d_only_epoch.cpp)
target_link_libraries(test_dl_h2d_only_epoch PRIVATE renaissance)
target_compile_definitions(test_dl_h2d_only_epoch PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dl_h2d_only_epoch)
endif()

set_target_properties(test_dl_h2d_only_epoch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_dl_h2d_only_epoch PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_dl_h2d_only_epoch PRIVATE -Wall -Wextra)
endif()

message(STATUS "  - test_dl_h2d_only_epoch: H2D传输性能测试（DeepLearningTask + Preprocessor）")
```

## 4. 关键技术点分析

### 4.1 双缓冲同步机制
- **TransferStation状态管理**：通过`buffer_is_readable()`和`set_buffer_writeable()`实现同步
- **Rank-0特权**：只有rank-0管理buffer状态切换，避免多rank竞争
- **Barrier同步**：每次batch传输后rank间barrier，确保所有rank的DMA都完成

### 4.2 Last Batch处理
- **Preprocessor自动处理**：`steps_per_epoch()`已包含last batch
- **固定传输范围**：H2D图始终传输整个Region，last batch的partial data不影响传输逻辑
- **TransferStation终止**：`is_finished()`判断所有数据是否处理完毕

### 4.3 多GPU并行
- **独立CUDA图**：每个rank有独立的cudaGraphExec和stream
- **共享CPU buffer**：所有rank从同一个TransferStation读取
- **同步安全性**：barrier机制防止buffer状态过早切换

### 4.4 AMP模式支持
- **自动适配**：Compiler根据`GlobalRegistry::using_amp()`生成正确的Region
- **FP16传输**：I_A_DATA和I_B_DATA的DType自动设置为FP16
- **带宽计算**：slot_bytes自动反映FP16的较小数据量

### 4.5 性能优化
- **最小侵入**：复用现有Compiler流程，只做过滤
- **异步执行**：CUDA GraphLaunch异步执行，不阻塞CPU
- **精确计时**：只测量纯传输时间，不包括compile时间
- **零拷贝**：直接从TransferStation的CPU buffer传输到GPU

## 5. 边界情况处理

| 场景 | 处理方式 |
|------|----------|
| `batches == 1` | 只传输buf0一次，正常退出 |
| `batches == 0` | 立即返回，elapsed=0.0 |
| `compile_h2d_only()`后调用`run()` | 未定义行为，文档明确禁止 |
| `run()`后调用`run_h2d_only()` | 正常工作，但会重新编译H2D图 |
| CPU模式 | 返回零结果，不执行实际测试 |
| 单GPU | 单rank模式，无同步开销 |
| 数据集不存在 | 提前检查并报错 |

## 6. 预期输出示例

```
=== H2D传输性能测试 ===
Dataset: imagenet
Path: T:\dataset\imagenet
Batch size: 512
Resolution: 224
AMP: on
Ranks: 8
Load workers: 2
Preprocess workers: 4
Steps per epoch: 1251

Compiling H2D-only graphs...
Running H2D-only epoch...

=== 测试结果 ===
总耗时:       15.234 秒
处理批次数:   1251
平均延迟:     12.178 ms/batch
传输数据量:   245.763 GB
等效带宽:     161.289 GB/s
=== DONE ===
```

## 7. 风险评估与缓解

### 7.1 主要风险
1. **状态污染**：`compile_h2d_only()`后`h2d_only_mode_`保持true，影响后续`run()`
   - **缓解**：文档明确说明互斥使用，或考虑在`run()`开始时重置标志
   
2. **Init All开销**：大型模型的权重初始化可能耗时几秒
   - **缓解**：本方案使用单层FC，开销可忽略；未来可考虑在TaskBase中增加h2d_only保护

3. **Barrier开销**：每次batch同步引入微秒级开销
   - **缓解**：这是正确性必需的，对于端到端测试是合理的

4. **多Rank竞争**：TransferStation的状态管理可能存在竞争条件
   - **缓解**：只让rank-0管理状态，其他rank只读

### 7.2 测试覆盖
建议增加以下测试用例：
1. 单GPU模式测试
2. CPU模式fallback测试
3. 不同batch size测试
4. AMP vs non-AMP对比测试
5. CIFAR-10 vs ImageNet对比测试

## 8. 总结

本方案通过最小侵入性的修改，实现了纯净的H2D传输性能测试：

1. **复用现有架构**：充分利用Compiler、GraphAtlas、ExecTable机制
2. **过滤而非重写**：在捕获阶段过滤，不重新实现H2D逻辑
3. **同步模型成熟**：借鉴test_h2d_copy_bandwidth()的多rank同步经验
4. **兼容性强**：完全兼容test_pw_ultimate.cpp的配置格式
5. **性能纯净**：避免多余操作，精确测量H2D传输性能

相比FA3.md的方案，本方案更加科学和完整，特别是：
- 更好的多rank同步机制
- 更完善的边界情况处理
- 更精确的性能测量
- 更强的架构兼容性