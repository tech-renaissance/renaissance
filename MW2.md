# 【小伙伴K】

• 已对照修复后的 UTK_FINAL.md 与代码库做第三次交叉审查。方案文件已大幅完善（旧符号删除清单、per-rank 测试值、allocate_staging_params
  空 gpu_ids 处理、R3 风险更新均已到位），但发现了 1 个严重运行时问题 和 4 个中低等级问题：
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一、严重问题
  1. 测试代码缺少 reg.train_resolution() → StagingBufferPool 自动分配崩溃
    位置：test_h2d_copy_a.cpp、test_h2d_copy_b.cpp、perf_h2d_copy_a.cpp
    代码事实：三个测试文件均只设置了：
    reg.local_batch_size(batch_size);
    reg.set_num_color_channels(channels);
    但 compile_capture_simple 自动分配 StagingBufferPool 时（UTK_FINAL.md §3.4）：
    int resolution = reg.current_resolution_train();
    // ...
    size_t data_raw = static_cast<size_t>(batch_size) * resolution
                    * resolution * effective_c * elem_size;
    current_resolution_train() 初始值为 -1（global_registry.h:1075：alterable_current_resolution_train_{-1}）。
    后果：size_t * (-1) 运算中 -1 被提升为 size_t（0xFFFFFFFFFFFFFFFF），data_raw 变为极大值，allocate_staging_memory 尝试分配数 TB 内
    存 → 立即崩溃或分配失败。
    修复：三个测试文件在 compile() 前均需加入：
    reg.train_resolution(resolution);  // 与测试使用的 resolution 变量一致
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    二、中等问题
  2. has_staging_params / staging_params_ptr / staging_params_bytes 实现缺失
    位置：UTK_FINAL.md §2.3
    文档只给出了 allocate_staging_params 的实现，但未给出其余三个方法的实现。虽然逻辑 trivial，但文档应补全以确保实现者不遗漏：
    bool GlobalRegistry::has_staging_params() const {
      return s_params_state.pool != nullptr;
    }

  void* GlobalRegistry::staging_params_ptr(int rank) const {
      if (!s_params_state.pool) {
          TR_RUNTIME_ERROR("StagingParamPool not allocated");
      }
      return s_params_state.pool->ptr(rank);
  }

  size_t GlobalRegistry::staging_params_bytes() const {
      return s_params_state.bytes_per_rank;
  }
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3. CMakeLists.txt 未提及新增 staging_param_pool.cpp
    位置：UTK_FINAL.md §6.1 F3/F4
    新增 src/core/staging_param_pool.cpp 需要注册到 src/CMakeLists.txt（或对应编译目标），否则链接时找不到符号。
    修复：在 src/CMakeLists.txt 的源文件列表中加入：
    src/core/staging_param_pool.cpp
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. 期望输出数值未同步更新
    位置：UTK_FINAL.md §3.4 L842
    // 期望输出仍写：
    StagingParamPool[rank][0] = 0.123456
    但测试代码已改为 per-rank 不同值（0.001f * (rank + 1)），0.123456 这个具体值已不匹配。
    修复：
  - 每个 rank 的标量 DTensor[0] 精确匹配 StagingParamPool[rank][0]
  + 每个 rank 的标量 DTensor[0] 精确匹配 StagingParamPool[rank][0] = 0.001×(rank+1)
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    三、低风险/建议
  5. h2d_op.cpp 缺少显式 #include
    新 kernel 代码使用了 GlobalRegistry::instance()，但当前 h2d_op.cpp 没有直接 include renaissance/core/global_registry.h（虽然可能间
    接可用）。建议在 F1 的修改说明中加入：
    #include "renaissance/core/global_registry.h"
    #include "renaissance/core/staging_param_pool.h"  // 若 h2d_op.cpp 直接使用
  6. dual_graph.cpp 需在代码层面标注 BROKEN
    如果 fill_transfer_buffer 声明被删除，dual_graph.cpp:132 编译即失败。仅文档建议"标注为 BROKEN"不够，需在 dual_graph.cpp 中实际注释
    掉相关行并加 TODO，否则编译断裂。
  7. A/B CPU kernel 仍为 zero-fill（已知限制）
    launch_range_h2d_copy_cpu 在 UTK_FINAL.md 中仍是 zero-fill。--cpu 模式下 A/B 测试无法验证数据内容（transfer_to_rank 写入的数据会被
    zero-fill 覆盖）。这不是新问题，但建议文档中注明 A/B 测试 --cpu 模式仅用于零填充基准，或待后续统一修复 CPU 路径。
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    四、总结
      问题                                等级        修复位置                修复代价
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      缺少 train_resolution()         🔴 运行时崩溃   3 个测试文件          每文件 +1 行
      has_staging_params 等实现缺失    🟡 实现遗漏    global_registry.cpp      +12 行
      CMakeLists.txt 未更新            🟡 链接错误    src/CMakeLists.txt       +1 行
      期望输出数值过时                🟡 文档不一致   UTK_FINAL.md L842       改 1 行
      include 缺失                     🟢 编译依赖    h2d_op.cpp               +1 行

  最紧急修复：三个测试文件的 reg.train_resolution(resolution)。没有这一行，A/B 测试在 GPU 模式下会直接因内存分配失败而崩溃。



# 【用户补充】

按照我们的惯例，测试样例最好GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution)，一起设，设为一样的值。

