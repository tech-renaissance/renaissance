



# 【小伙伴S】

  SimpleTask CUDA Graph 运行逻辑优化方案

  基于 BEST_S.md 和 BEST_V2.md 的原则，我提出以下针对 SimpleTask 的具体优化方案：

  一、问题诊断

  当前 SimpleTask 的 run() 方法存在以下性能瓶颈：

  1. 循环内重复查找：每次 run() 都进行 named_graphs_.find() 和 simple_captured_graphs_.find()
  2. 重复 cudaSetDevice：每次调用都设置设备，即使在单 GPU 场景
  3. 使用 cudaDeviceSynchronize：全设备同步开销大（~150-250 μs）
  4. 多线程开销：在性能测试循环中反复创建/销毁线程

  二、优化方案

  方案 A：新增 run_iter() 方法

  为 SimpleTask 添加专门的迭代执行方法，消除循环内的冗余操作：

  // 在 simple_task.h 中添加新接口
  class SimpleTask : public TaskBase {
  public:
      // ... 现有接口 ...

      /**
       * @brief 迭代执行单个图（优化版本，用于性能测试）
       * @param name 图名称
       * @param iterations 迭代次数
       *
       * @note 特点：循环外一次性查找，循环内只执行图+流同步
       *       适用于性能测试场景，避免重复查找和同步开销
       */
      void run_iter(const std::string& name, int iterations);
    
      /**
       * @brief 迭代执行双图并行（优化版本）
       * @param a 第一个图名称
       * @param b 第二个图名称
       * @param iterations 迭代次数
       */
      void run_iter(const std::string& a, const std::string& b, int iterations);
  };

  方案 B：优化现有 run() 方法

  保持 API 兼容性，优化现有实现：

  // 在 task_base.cpp 中优化 run_impl()
  void TaskBase::run_impl(const std::string& name, bool dry_run) {
      check_run_eligibility(dry_run);

      auto it = named_graphs_.find(name);
      TR_CHECK(it != named_graphs_.end(), ValueError, "Graph not found: " << name);
    
      if (is_simple_task()) {
          auto cap_it = simple_captured_graphs_.find(name);
          TR_CHECK(cap_it != simple_captured_graphs_.end(), ValueError,
                   "Graph not captured: " << name);
    
          const GraphEntry& entry = it->second;
    
          if (dry_run) {
              // ... dry run 逻辑 ...
              return;
          }
    
          // 优化1：移除循环内的 cudaSetDevice（在线程初始化时设置一次）
          // 优化2：将同步改为流级别同步
          for (int rank = 0; rank < num_gpus_; ++rank) {
              DeviceContext& ctx = *backend_->contexts[rank];
  #ifdef TR_USE_CUDA
              // 只在第一次调用时设置设备
              static thread_local int last_device = -1;
              if (ctx.is_gpu() && ctx.device_id() != last_device) {
                  cudaError_t err = cudaSetDevice(ctx.device_id());
                  if (err != cudaSuccess) {
                      TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                     << ": " << cudaGetErrorString(err));
                  }
                  last_device = ctx.device_id();
              }
  #endif
              void* stream = ctx.stream(entry.stream);
              cap_it->second.launch(rank, stream);
          }

          // 优化3：使用流同步替代设备同步
          for (int rank = 0; rank < num_gpus_; ++rank) {
              backend_->contexts[rank]->synchronize_stream(entry.stream);
          }
          return;
      }
    
      // ... DeepLearningTask 路径保持不变 ...
  }

  方案 C：run_iter() 完整实现

  // 在 simple_task.cpp 中实现
  void SimpleTask::run_iter(const std::string& name, int iterations) {
      TR_CHECK(iterations > 0, ValueError, "Iterations must be positive: " << iterations);
      TR_CHECK(phase_ == Phase::COMPILED, ValueError,
               "Task not compiled. Call compile() before run_iter()");

      // 循环外：一次性查找所有资源
      auto it = named_graphs_.find(name);
      TR_CHECK(it != named_graphs_.end(), ValueError, "Graph not found: " << name);
    
      auto cap_it = simple_captured_graphs_.find(name);
      TR_CHECK(cap_it != simple_captured_graphs_.end(), ValueError,
               "Graph not captured: " << name);
    
      const GraphEntry& entry = it->second;
      const CapturedGraph& cg = cap_it->second;
    
      // 预解析每个 rank 的执行资源
      struct RankExecInfo {
          int rank;
          void* stream;
          const CapturedGraph* graph;
      };
    
      std::vector<RankExecInfo> exec_infos;
      exec_infos.reserve(num_gpus_);
    
      for (int rank = 0; rank < num_gpus_; ++rank) {
          DeviceContext& ctx = *backend_->contexts[rank];

  #ifdef TR_USE_CUDA
          // 每个线程只设置一次设备
          if (ctx.is_gpu()) {
              cudaError_t err = cudaSetDevice(ctx.device_id());
              if (err != cudaSuccess) {
                  TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                 << ": " << cudaGetErrorString(err));
              }
          }
  #endif
          exec_infos.push_back({
              rank,
              ctx.stream(entry.stream),
              &cg
          });
      }

      // 循环内：纯粹的图执行 + 流同步
      for (int i = 0; i < iterations; ++i) {
          // 启动所有 rank 的图
          for (const auto& info : exec_infos) {
              info.graph->launch(info.rank, info.stream);
          }
    
          // 同步所有流（比设备同步快得多）
          for (const auto& info : exec_infos) {
              DeviceContext& ctx = *backend_->contexts[info.rank];
              ctx.synchronize_stream(entry.stream);
          }
      }
  }

  void SimpleTask::run_iter(const std::string& a, const std::string& b, int iterations) {
      TR_CHECK(iterations > 0, ValueError, "Iterations must be positive: " << iterations);
      TR_CHECK(phase_ == Phase::COMPILED, ValueError,
               "Task not compiled. Call compile() before run_iter()");

      // 循环外：一次性查找所有资源
      auto it_a = named_graphs_.find(a);
      auto it_b = named_graphs_.find(b);
      TR_CHECK(it_a != named_graphs_.end(), ValueError, "Graph not found: " << a);
      TR_CHECK(it_b != named_graphs_.end(), ValueError, "Graph not found: " << b);
    
      auto cap_a = simple_captured_graphs_.find(a);
      auto cap_b = simple_captured_graphs_.find(b);
      TR_CHECK(cap_a != simple_captured_graphs_.end(), ValueError, "Graph not captured: " << a);
      TR_CHECK(cap_b != simple_captured_graphs_.end(), ValueError, "Graph not captured: " << b);
    
      const GraphEntry& entry_a = it_a->second;
      const GraphEntry& entry_b = it_b->second;
      const CapturedGraph& cg_a = cap_a->second;
      const CapturedGraph& cg_b = cap_b->second;
    
      // 预解析执行资源
      struct DualRankExecInfo {
          int rank;
          void* stream_a;
          void* stream_b;
          const CapturedGraph* graph_a;
          const CapturedGraph* graph_b;
      };
    
      std::vector<DualRankExecInfo> exec_infos;
      exec_infos.reserve(num_gpus_);
    
      for (int rank = 0; rank < num_gpus_; ++rank) {
          DeviceContext& ctx = *backend_->contexts[rank];

  #ifdef TR_USE_CUDA
          if (ctx.is_gpu()) {
              cudaError_t err = cudaSetDevice(ctx.device_id());
              if (err != cudaSuccess) {
                  TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                 << ": " << cudaGetErrorString(err));
              }
          }
  #endif
          exec_infos.push_back({
              rank,
              ctx.stream(entry_a.stream),
              ctx.stream(entry_b.stream),
              &cg_a,
              &cg_b
          });
      }

      // 循环内：双图并行执行
      for (int i = 0; i < iterations; ++i) {
          // 启动双图
          for (const auto& info : exec_infos) {
              info.graph_a->launch(info.rank, info.stream_a);
              info.graph_b->launch(info.rank, info.stream_b);
          }
    
          // 同步两个流
          for (const auto& info : exec_infos) {
              DeviceContext& ctx = *backend_->contexts[info.rank];
              ctx.synchronize_stream(entry_a.stream);
              ctx.synchronize_stream(entry_b.stream);
          }
      }
  }

  三、性能测试代码修改

  // 在性能测试中使用新的 run_iter() 方法
  // 旧的低效方式：
  // for (int i = 0; i < 100; ++i) {
  //     task.run("relu");
  //     cudaDeviceSynchronize(); // 额外的同步开销
  // }

  // 新的高效方式：
  task.run("relu"); // 预热
  auto t0 = std::chrono::high_resolution_clock::now();
  task.run_iter("relu", 100); // 优化的迭代执行
  auto t1 = std::chrono::high_resolution_clock::now();

  四、预期性能提升

  根据 BEST_S.md 的分析：

  1. 消除查找开销：从 100 次 find() 操作减少到 1 次（~99% 减少）
  2. 消除冗余 cudaSetDevice：从 1000 次减少到 1 次（~99.9% 减少）
  3. 流同步替代设备同步：从 cudaDeviceSynchronize() 改为 cudaStreamSynchronize()（~50-100 μs 节省）
  4. 总体提升：预计 BWD 性能从 ~1,617μs 提升到 ~1,240μs（约 23% 提升）

  五、实施建议

  1. 优先级高：先实施方案 A（新增 run_iter()），风险低，收益明显
  2. 保持兼容性：保留现有 run() 方法，逐步迁移
  3. 测试验证：在 ReLU 性能测试中验证提升效果
  4. 后续优化：考虑在 DeviceContext 中移除 synchronize_all() 内的冗余 cudaSetDevice

  这个方案完全符合 BEST_V2.md 的设计原则，特别针对 SimpleTask 的使用场景（算子性能测试）进行了优化。



# 【小伙伴K】



  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  SimpleTask CUDA Graph 执行优化方案
  ══════════════════════════════════
  一、当前性能退化点诊断

退化点               当前代码位置             具体表现                                     估计开销

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1   全设备同步           task_base.cpp:500–502    synchronize_all() → cudaDeviceSynchronize(   ~150–250 μs（最严重）
                                                     )
   2   重复 cudaSetDevice   task_base.cpp:488–494    每次 run() 每个 rank 都 set                  ~5–10 μs
   3   重复哈希查找         task_base.cpp:466, 472   named_graphs_.find + simple_captured_graph   ~2–5 μs
                                                     s_.find
   4   双重 setdevice       device_context.cpp:125   synchronize_all() 内部又 set 一次            ~5 μs
   5   多线程展开在循环内   测试代码层面             每次 task.run("relu") 重复线程创建/销毁      框架级开销
  ▌ BEST_S.md 明确指出："我想到的一个我们的性能退化的原因就是，我们在 SimpleTask 的'重复运行同一个算子测性能'的测试里
  ▌ 面，把多线程的展开和 join 放进了循环里。"
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  二、优化方案（分三阶段实施）
  阶段 1：同步方式热修复（零架构改动，立即可实施）
  目标：将 cudaDeviceSynchronize 降级为 cudaStreamSynchronize，消除全设备同步开销。
  改动 A：单图 run(name) 改用 stream 级同步
  // 文件：src/task/task_base.cpp 第 500–502 行
  // ===== 旧 =====
  for (auto& ctx_ptr : backend_->contexts) {
      ctx_ptr->synchronize_all();  // ← cudaDeviceSynchronize
  }

  // ===== 新 =====
  for (int rank = 0; rank < num_gpus_; ++rank) {
      backend_->contexts[rank]->synchronize_stream(entry.stream);
  }
  改动 B：双图 run(a, b) 改用双 stream 同步
  // 文件：src/task/task_base.cpp 第 594–596 行
  // ===== 旧 =====
  for (auto& ctx_ptr : backend_->contexts) {
      ctx_ptr->synchronize_all();
  }

  // ===== 新 =====
  for (int rank = 0; rank < num_gpus_; ++rank) {
      auto& ctx = *backend_->contexts[rank];
      ctx.synchronize_stream(entry_a.stream);
      ctx.synchronize_stream(entry_b.stream);
  }
  改动 C：删除 synchronize_all() 内部冗余 cudaSetDevice
  // 文件：src/backend/device_context.cpp 第 125 行
  // ===== 旧 =====
  cudaError_t err = cudaSetDevice(device_id_);  // ← 多此一举

  // ===== 新 =====
  // 直接删除该行
  预期效果：BWD 从 ~1,484 μs → ~1,280 μs；FWD 从 ~1,309 μs → ~1,250 μs。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  阶段 2：新增 run_iter() 方法（核心优化）
  目标：为"重复运行同一个算子测性能"场景提供专用 API，将 查找、setdevice、多线程展开 全部移到循环外。
  2.1 接口设计
  // 文件：include/renaissance/task/task_base.h
  // 在 protected 区域新增（SimpleTask 通过 using 提升为 public）

  /**
   * @brief SimpleTask 专用：高性能迭代执行单图
   * @param name 图名称
   * @param iterations 迭代次数
   * @note 循环外完成所有查找和线程展开，循环内只做 cudaGraphLaunch + sync
      */
    void run_iter(const std::string& name, int iterations);

  /**
   * @brief SimpleTask 专用：高性能迭代执行双图并行
   * @param a 第一个图名称
   * @param b 第二个图名称
   * @param iterations 迭代次数
      */
    void run_iter(const std::string& a, const std::string& b, int iterations);
    2.2 实现设计（单图版）
    // 文件：src/task/task_base.cpp
    void TaskBase::run_iter(const std::string& name, int iterations) {
      check_run_eligibility(false);

      // --- 循环外：一次性解析所有指针 ---
      auto it = named_graphs_.find(name);
      TR_CHECK(it != named_graphs_.end(), ValueError, "Graph not found: " << name);

      auto cap_it = simple_captured_graphs_.find(name);
      TR_CHECK(cap_it != simple_captured_graphs_.end(), ValueError,
               "Graph not captured: " << name);

      const GraphEntry& entry = it->second;
      const CapturedGraph& cg = cap_it->second;

      // 预组装每个 rank 的执行上下文
      struct RankExec {
          int rank;
          int device_id;           // 物理 GPU ID
          void* stream;            // 对应 entry.stream 的流指针
          NativeGraph exec;        // 预解析的 cudaGraphExec_t（void*）
      };
      std::vector<RankExec> execs;
      execs.reserve(num_gpus_);

      for (int rank = 0; rank < num_gpus_; ++rank) {
          DeviceContext& ctx = *backend_->contexts[rank];
          RankExec re;
          re.rank = rank;
          re.device_id = ctx.device_id();
          re.stream = ctx.stream(entry.stream);
          re.exec = cg.native_exec(rank);   // CapturedGraph 已提供 native_exec()
          TR_CHECK(re.exec != nullptr, RuntimeError,
                   "Graph not instantiated for rank " << rank);
          execs.push_back(re);
      }

  #ifdef TR_USE_CUDA
      // --- 多线程展开（epoch 级，循环外）---
      std::vector<std::thread> threads;
      std::exception_ptr exc = nullptr;
      std::mutex exc_mtx;

      for (int rank = 0; rank < num_gpus_; ++rank) {
          threads.emplace_back([&, rank]() {
              try {
                  const auto& e = execs[rank];
                  cudaError_t err = cudaSetDevice(e.device_id);
                  if (err != cudaSuccess) {
                      TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                      << ": " << cudaGetErrorString(err));
                  }
    
                  cudaGraphExec_t exec = static_cast<cudaGraphExec_t>(e.exec);
                  cudaStream_t stream = static_cast<cudaStream_t>(e.stream);
    
                  // --- 循环内：零查找、零 setdevice、只 launch + sync ---
                  for (int i = 0; i < iterations; ++i) {
                      cudaGraphLaunch(exec, stream);
                      cudaStreamSynchronize(stream);
                  }
              } catch (...) {
                  std::lock_guard<std::mutex> lock(exc_mtx);
                  if (!exc) exc = std::current_exception();
              }
          });
      }
    
      for (auto& t : threads) t.join();
      if (exc) std::rethrow_exception(exc);
  #else
      // CPU 模式：串行执行
      for (int i = 0; i < iterations; ++i) {
          for (auto& e : execs) {
              cg.launch(e.rank, e.stream);
          }
      }
  #endif
  }
  2.3 关键设计决策
   决策                            理由
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   epoch 级多线程                  与 DeepLearningTask run_train_epoch_gpu() 保持一致；线程在循环外展开、循环后 join
   每线程 cudaSetDevice 一次       符合 BEST_S.md 原则："每个线程只需要进行一次 setdevice"
   预解析 native_exec()            CapturedGraph 已提供 native_exec(rank)，在循环外转为 cudaGraphExec_t
   零查找、零 setdevice 在循环内   循环内只有 cudaGraphLaunch + cudaStreamSynchronize
   流级同步                        cudaStreamSynchronize 替代 cudaDeviceSynchronize，避免等待无关流
  2.4 双图版 run_iter(a, b, iterations)
  逻辑与单图版相同，预组装两份 RankExec，循环内先后 launch 两个图，然后同步两个 stream。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  阶段 3：测试代码适配
  // ===== 旧（性能退化）=====
  for (int i = 0; i < 100; ++i) {
      task.run("relu");           // 每次：查找 + setdevice ×N + deviceSync
      cudaDeviceSynchronize();    // 测试代码自己又 sync 一次！
  }

  // ===== 新（最优路径）=====
  task.run("relu");               // 预热 1 次（含 compile + capture）
  auto t0 = std::chrono::high_resolution_clock::now();
  task.run_iter("relu", 100);     // 100 次纯热路径
  auto t1 = std::chrono::high_resolution_clock::now();
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  三、SimpleTask 头文件暴露
  // 文件：include/renaissance/task/simple_task.h
  // 在 public 区域新增 using 声明：

  /**
   * @brief 高性能迭代执行单图（循环外展开多线程，循环内只做 launch + sync）
      */
    using TaskBase::run_iter;
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    四、预期效果汇总
      改动                    FWD 预期                      BWD 预期
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      阶段 1（stream sync）   1,309 → ~1,250 μs             1,484 → ~1,280 μs
      阶段 2（run_iter）      ~1,250 → ~1,230 μs            ~1,280 → ~1,240 μs
      合计 vs cuDNN           ~1,230 μs（cuDNN 1,225 μs）   ~1,240 μs（cuDNN 1,227 μs）
    核心结论：BWD 慢 257 μs 的大头不是 INT8 vs BOOLEAN，而是 cudaDeviceSynchronize 等全设备同步 + 循环内重复查找/setdevice
    。把同步粒度从 device 降到 stream，并将线程展开移到循环外，性能就能追平 cuDNN。
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    五、与 DeepLearningTask 的对比
      维度         DeepLearningTask                                                      SimpleTask
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      图索引       GraphSlot 枚举 + GpuExecTable                                         字符串 name → native_exec(rank)
      多线程粒度   epoch 级                                                              run_iter() 调用级
      同步策略     三计算流 + 更新流 + 传输流分别同步                                    单图/双图按各自 entry.stream 同步
      核心原则     一致：compile 阶段完成实例化，run 阶段零查找、零 setdevice、流级同
                步
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    是否需要我立即按阶段 1 和阶段 2 实施代码改动并编译验证？



# 【小伙伴D】

## SimpleTask CUDA Graph 执行逻辑优化 —— 综合方案
### 一、当前 SimpleTask 热路径诊断
以 task.run("relu") 单次调用为例（ task_base.cpp:463-504 ）：

```
① named_graphs_.find("relu")              ← 字符串哈希
② simple_captured_graphs_.find("relu")    ← 又一次字符串哈希
③ for each rank:
     cudaSetDevice(ctx.device_id())        ← 每次 run 都 set
     ctx.stream(entry.stream)              ← switch+数组
     cg.launch(rank, stream)               ← 边界检查 + 虚调用
④ for each ctx:
     synchronize_all()                     ← cudaSetDevice + cudaDeviceSynchronize
```
假设 test_relu.cpp 调用 500 次 task.run("relu") ：

操作 调用次数 分类 named_graphs_.find() 500 哈希查找 simple_captured_graphs_.find() 500 哈希查找 cudaSetDevice 1000（500×2） 冗余 CUDA API cudaDeviceSynchronize 500 全设备阻塞 cudaGraphLaunch 500 唯一有效

BEST.md 原文定位 ：
 "我想到的一个我们的性能退化的原因就是，我们在SimpleTask的'重复运行同一个算子测性能'的测试里面，把多线程的展开和join放进了循环里。只要提供了一个run_iter方法来专门测单算子性能，不就可以避免这个问题了？"
### 二、方案设计 2.1 放在哪里
对比三个候选意见：

- S 提议放 TaskBase — 污染基类，不合理
- K 提议放 TaskBase — 同上
- D 提议放 SimpleTask — ✅ 正确，遵循 BEST_V2 "不污染 TaskBase" 原则
放在 SimpleTask （ simple_task.h 内 #ifdef TR_USE_CUDA 块），因为：

- simple_captured_graphs_ 是 protected ，子类可直接访问
- context(rank) 是 protected 访问器，无需碰 backend_
- named_graphs_ 是 protected 2.2 两个重载
```
#ifdef TR_USE_CUDA
void run_iter(const std::string& name, int iterations);        // 单图
void run_iter(const std::string& a, const std::string& b,      // 双图并行
              int iterations);
#endif
``` 2.3 单图 run_iter(name, iterations) 实现
```
void SimpleTask::run_iter(const std::string& name, int iterations) {
    // ── 循环外：一次性解析所有指针（O(N) → O(1)）──
    auto cap_it = simple_captured_graphs_.find(name);
    TR_CHECK(cap_it != simple_captured_graphs_.end(), ValueError,
             "Graph not captured: " << name);
    auto g_it = named_graphs_.find(name);
    TR_CHECK(g_it != named_graphs_.end(), ValueError,
             "Graph not registered: " << name);

    CapturedGraph& cg = cap_it->second;
    StreamKind sk = g_it->second.stream;
    
    const int K = num_gpus_;
    
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);
    
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, &cg, rank, sk, iterations, &exc]() {
            try {
                DeviceContext& ctx = context(rank);
                cudaError_t err = cudaSetDevice(ctx.device_id());   // ★ 每个线程只 
                set 一次
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }
    
                cudaGraphExec_t exec = static_cast<cudaGraphExec_t>(
                    cg.native_exec(rank));
                cudaStream_t stream = static_cast<cudaStream_t>(
                    ctx.stream(sk));
    
                // ── 循环内：零查找、零 setdevice、零 deviceSync ──
                for (int i = 0; i < iterations; ++i) {
                    cudaGraphLaunch(exec, stream);        // ★ 纯粹的两行
                    cudaStreamSynchronize(stream);        //
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    
    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);
}
``` 2.4 双图 run_iter(a, b, iterations) 实现
```
void SimpleTask::run_iter(const std::string& a, const std::string& b, int 
iterations) {
    auto cap_a = simple_captured_graphs_.find(a);
    auto cap_b = simple_captured_graphs_.find(b);
    auto g_a = named_graphs_.find(a);
    auto g_b = named_graphs_.find(b);
    // ... TR_CHECK ...
    CapturedGraph& cg_a = cap_a->second;
    CapturedGraph& cg_b = cap_b->second;
    StreamKind sk_a = g_a->second.stream;
    StreamKind sk_b = g_b->second.stream;

    const int K = num_gpus_;
    
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);
    
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, &cg_a, &cg_b, rank, sk_a, sk_b, iterations, &
        exc]() {
            try {
                DeviceContext& ctx = context(rank);
                cudaSetDevice(ctx.device_id());
    
                cudaGraphExec_t exec_a = static_cast<cudaGraphExec_t>(cg_a.
                native_exec(rank));
                cudaGraphExec_t exec_b = static_cast<cudaGraphExec_t>(cg_b.
                native_exec(rank));
                cudaStream_t s_a = static_cast<cudaStream_t>(ctx.stream(sk_a));
                cudaStream_t s_b = static_cast<cudaStream_t>(ctx.stream(sk_b));
    
                for (int i = 0; i < iterations; ++i) {
                    cudaGraphLaunch(exec_a, s_a);         // 双图并行发射
                    cudaGraphLaunch(exec_b, s_b);
                    cudaStreamSynchronize(s_a);           // 并行完一起同步
                    cudaStreamSynchronize(s_b);
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    
    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);
}
```
### 三、test_relu.cpp 调用方式变更
旧 （500 次 run ，500 次全部冗余操作）：

```
for (int i = 0; i < 500; ++i) {
    task.run("relu");
    cudaDeviceSynchronize();  // 额外的双重同步
}

```
新 （1 次解析，500 次纯粹 launch+sync）：

```
task.run("relu");                  // 预热 1 次
auto t0 = std::chrono::high_resolution_clock::now();
task.run_iter("relu", 500);        // 500 次纯热路径
auto t1 = std::chrono::high_resolution_clock::now();
```
### 四、文件改动清单
文件 改动 行数 include/renaissance/task/simple_task.h 新增 #include <cuda_runtime.h> （ #ifdef TR_USE_CUDA 条件编译）、2 个 run_iter() 方法声明 + 实现 ~65 行 tests/op/test_relu.cpp 计时循环改为 task.run_iter("relu", 500) 3 行

不修改 ：

- task_base.h / task_base.cpp — 零污染
- captured_graph.h — native_exec() 已在 BEST_V2 实施中添加
- device_context.cpp — 新路径绕过 synchronize_all() ，无需修改
### 五、开销对比
操作（×500 次） task.run() 旧 task.run_iter() 新 节省 named_graphs_.find() 500 1 499 simple_captured_graphs_.find() 500 1 499 cudaSetDevice （单卡） 1000 1 999 cudaDeviceSynchronize 500 0 500 cudaStreamSynchronize 0 500 合理新增 cudaGraphLaunch 500 500 不变 多线程 create/join 0（隐式每 run） 1 次 —

### 六、为什么 SimpleTask 与 DeepLearningTask 优化策略不同
维度 DeepLearningTask SimpleTask 线程模型 epoch 级 create/join 整个 run_iter() 周期一次 create/join 图索引 GraphSlot 编译期枚举 图名直接解析到 CapturedGraph& ，然后 native_exec() 执行表 GpuExecTable [rank][GraphSlot] lambda 捕获的栈变量 同步粒度 多流精确同步（三计算流+传输流+更新流） 单流或双流同步 调度器 rank 0 per-batch scheduler.step() 无 核心目标 训练吞吐最大化 算子性能测量的 Host 开销最小化

### 七、与 BEST.md 原文对齐
BEST.md 原则 本方案 "run只是根据图名来找到图然后跑" ✅ run_iter() 循环外只找一次 "解哈希都不放进循环里" ✅ 循环内零哈希 "每个rank一个线程" ✅ per-rank 线程，iterations 内不重新创建 "直接进行循环" ✅ cudaGraphLaunch + cudaStreamSynchronize 两行 "cudaStream_t 传值而非传指针" ✅ 直接按值传递 "不要在run里面实例化" ✅ compile 阶段已完成实例化

需要我实施这个方案吗？