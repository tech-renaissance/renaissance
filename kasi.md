# 【十七、当前重点问题】

各位专家。

我们团队开发的自研深度学习框架来到了一个关键的节点。ImageNet数据集的加载器功能完全实现、性能优越（FULLY模式10s内加载完成ImageNet，PARTIAL模式5s内），完整性通过了验证。我们通过在不同平台上多次运行训练集、验证集的加载，并计算CRC32值，与原始数据集逐文件计算的CRC32值进行排序后比对，确认加载的结果是完整的、正确的。这证明了我们的加载逻辑（包括buffer切换、样本的多线程分配、地址偏移量的传递等等）是正确的。

但是，碰到的问题是进行多个epoch的连续加载时，会偶发卡死的情况。卡死的情况具有以下特点：

1、随机性。偶发，并不是每次都卡死。并且卡死在第几个epoch完全不确定。

2、硬件相关。不同的服务器平台上卡死的概率明显不同（疑似是性能较差、运行较慢的服务器会更倾向于卡死）。

3、覆盖所有类型。不管是FULLY还是PARTIAL加载，不管是RAW加载器还是DTS加载器，都会偶发卡死。

4、逻辑正确性。只要不发生卡死，不管在第几个epoch验证CRC，都会得到完全正确的结果。

5、线程数相关。我们有大于等于1个加载线程（load worker）和大于等于1个预处理线程（preproc worker）。其中，发生卡死的概率与预处理线程数高度相关。预处理线程为96时大概率卡死（但也不是必定），而预处理线程为1时，卡死的概率会巨幅下降，接近于0（但并不是0，因为还是会有极小概率卡死）。

7、卡死后无法恢复。一旦发生卡死，等多久也不会恢复，而且没有任何报错。

根据以上信息，我们初步判断卡死的根源是某种竞态条件，或者说是NUMA架构下的多线程同步超时。而且，发生在预处理线程读取阶段的可能性要大于发生在加载阶段的可能性（因为卡死概率与预处理线程数成正相关）。

我们此前设计加载器的早期，也曾碰到过卡死的情况，当时很明显是多线程同步超时。我们当时采用了静态分配+每个buffer加载完成后join的方式来避免此问题（即，在多线程开始前，预先分配好每个线程的加载任务）。每个线程预先知道自己的加载对象和加载目的地，并且加载完成后就结束，那么就不存在竞争。但是，如果加上预处理阶段，预处理阶段又有其他的线程，就未必没有问题了。

我们认为，不大可能是加载线程导致的卡死，一个重要的证据就是，卡死会发生在FULLY模式。FULLY模式下，加载线程几乎只有第一个epoch会工作，进入到第2个及以后的epoch，由于数据已经在内存里了，加载线程就只需要在epoch之初进行一次洗牌，不需要多线程。又因为我们的测试样例把shuffle都关了，所以，FULLY模式的第2个epoch及以后，预处理线程根本就不做任何事情，状态也几乎没变化。尽管如此FULLY还是会在第2个epoch以后卡死，这说明很可能不是加载线程的问题。

当然，也有可能是其他问题导致的卡死，而不一定是我们猜测的预处理线程的竞态条件。

目前我们只在NUMA架构的Linux服务器上进行过测试，无法确定其他平台是否存在同样的问题。因此也无法判断一定是NUMA架构带来的问题。

以下是我们团队的初步报告和分析，不一定对，仅供参考。

## 卡死问题分析

**日期**: 2026-02-07 **问题**: ImageNet数据加载器在多epoch运行时随机卡死 **状态**: 仍未解决

## 一、问题症状

### 1.1 核心现象

- **卡死位置**: 完全不可预测，可能发生在epoch 2~100+的任何位置
- **卡死模式**: FULLY模式和PARTIAL模式都会卡死
- **机器依赖性**: 与机器情况密切相关，不同机器卡死概率不同
- **并发敏感度**: 与preproc线程数强相关

### 1.2 关键线索

| preproc数量 | 卡死概率         | 说明                             |
| ----------- | ---------------- | -------------------------------- |
| 1           | 接近0（但不是0） | 单线程几乎不触发，但仍有极小概率 |
| 96          | 很高             | 高并发情况下频繁触发             |



## 二、已识别的竞态条件

### 2.1 第一个竞态：`current_ready_buffer_seq`的非原子访问

**位置**:

- `imagenet_loader_dts.h:211`
- `imagenet_loader_raw.h:229`

**问题**:

```
// 原始代码
size_t current_ready_buffer_seq = 0;  // 非原子类型

// Worker线程读取（96个并发）
size_t seq = current_set_->current_ready_buffer_seq;

// 主线程写入
current_set_->current_ready_buffer_seq = next_seq;
```

**竞态窗口**:

```
T1: Worker A 读取 current_ready_buffer_seq = 0
T2: 主线程读取并计算 next = 1
T3: 主线程写入 current_ready_buffer_seq = 1
T4: Worker B 读取 current_ready_buffer_seq = 1
T5: Worker A 使用旧值0访问 buffer_metas[0]  ← 可能越界/错误
```

**修复方案**:

```
// 修改为原子类型
std::atomic<size_t> current_ready_buffer_seq{0};

// 所有读取使用
size_t seq = current_set_->current_ready_buffer_seq.load(std::memory_order_acquire);

// 所有写入使用
current_set_->current_ready_buffer_seq.store(next, std::memory_order_release);
```

**修复效果**: 有一定效果，但未完全解决问题

### 2.2 第二个竞态：TOCTOU（Time-of-Check to Time-of-Use）

**问题核心**: Worker持有buffer引用期间，主线程可能修改buffer内容

**FULLY模式竞态时序**:

```
Worker线程A (处理buffer 0的最后一个样本):
  T1: current_buffer_seq = 0
  T2: buffer_meta = &buffer_metas[0]  ← 保存引用
  T3: 检查 sample 9999 在 [0, 10000) 范围内 ✓
  T4: 进入 while(!buffer_meta.ready->load())
      - buffer_metas[0].ready 已是true，立即通过

  【其他96个Worker都完成了】

  主线程:
    wait_workers_complete_buffer() 返回
    load_next_buffer() 开始:
      - current_ready_buffer_seq = 1
      - 可能修改 buffer_metas[0] 的内容

  T5: Worker A 执行 location = buffer_meta.shuffled_locations[9999]
      - 但 buffer_metas[0] 已被修改
      - location 可能是错误值 → 越界/崩溃
```

**PARTIAL模式竞态时序**:

```
Worker线程A (处理buffer_A的最后一个样本):
  T1: ready_buffer = &buffer_A  ← 保存指针
  T2: 检查样本在范围内 ✓
  T3: 进入 while(ready_buffer->state != READY)

  【其他96个Worker都完成了】

  主线程:
    wait_workers_complete_buffer() 返回
    load_next_buffer() 开始:
      - current_set_->ready_buffer = &buffer_B
      - **重新加载buffer_A的数据**
      - 修改 buffer_A.shuffled_locations
      - 修改 buffer_A.slot_metas

  T4: Worker A 继续执行
      - 使用旧指针 ready_buffer (仍指向buffer_A)
      - 但 buffer_A 内容已被修改
      - 访问破坏的数据 → 卡死
```

**修复方案**: 双重检查 + 局部拷贝

```
// 1. 拷贝关键数据到局部变量
size_t buffer_start = buffer_meta.load_start_offset;
size_t buffer_end = buffer_start + buffer_meta.total_samples;

// 2. 范围检查
if (global_sample_idx < buffer_start || global_sample_idx >= buffer_end) {
    return false;
}

// 3. 等待ready
while (!buffer_meta.ready->load()) {
    sleep(100us);
}

// 4. 【关键】再次验证buffer序号未变
size_t verify_seq = current_ready_buffer_seq.load();
if (verify_seq != current_buffer_seq) {
    return false;  // buffer已切换，重试
}

// 5. 【关键】拷贝vector到局部变量
std::vector<uint32_t> local_shuffled_locations = buffer_meta.shuffled_locations;
std::vector<SlotMeta> local_slot_metas = buffer_meta.slot_metas;

// 6. 后续只使用局部拷贝
```

**修复效果**: 卡死从早期epoch延迟到第86个epoch，说明有效但未根治

## 三、疑似未解决的竞态条件

### 3.1 Preprocessor的线程同步机制

**位置**: `preprocessor.cpp`

**架构**:

```
// 主线程
do {
    notify_workers_new_buffer();      // 原子递增 current_buffer_seq_
    wait_workers_complete_buffer();   // 等待 workers_finished_ == num_workers
    loader.load_next_buffer();        // 切换buffer
} while (has_more_buffers());

// Worker线程（持久化）
while (!stop_flag_) {
    int last_seen = current_buffer_seq_.load();
    while (current_buffer_seq_.load() == last_seen) {
        sleep(10us);  // 等待唤醒
    }

    while (loader.get_next_sample(...)) {
        // 处理样本
    }

    workers_finished_.fetch_add(1);  // 标记完成
}
```

**疑似问题**:

1. **JOIN完成的判断可能不准确**:

   ```
   wait_workers_complete_buffer() {
   ```

   ```
       while (workers_finished_.load() < num_workers) {
   ```

   ```
           sleep(10us);
   ```

   ```
       }
   ```

   ```
   }
   ```

   - 96个Worker都返回false后，`workers_finished_` == 96
   - 但此时可能仍有Worker在执行`get_next_sample()`的最后几行代码
   - 主线程立即调用`load_next_buffer()`，修改共享状态

2. **唤醒信号的时序问题**:

   ```
   // Worker
   ```

   ```
   while (current_buffer_seq_.load() == last_seen) {
   ```

   ```
       sleep(10us);
   ```

   ```
   }
   ```

   - Worker检测到`current_buffer_seq_`变化后退出等待
   - 但此时主线程可能还未准备好新buffer
   - Worker可能访问到半初始化的状态

3. **`workers_finished_`的计数时机**:

   ```
   while (loader.get_next_sample(...)) {
   ```

   ```
       // 处理
   ```

   ```
   }
   ```

   ```
   workers_finished_.fetch_add(1);  // ← 在这里计数
   ```

   - `get_next_sample()`返回false后，Worker递增计数
   - 但返回false的Worker可能：
     - 刚刚读取了`current_ready_buffer_seq`
     - 正准备访问`buffer_metas[current_ready_buffer_seq]`
     - 还没有完全"完成"当前buffer的处理

4. **Worker的退出和重启**:

   - Worker在返回false后，会进入下一轮等待
   - 但此时`global_seq`已经递增
   - 下次调用`get_next_sample()`时，计算的`global_sample_idx`可能跨过buffer边界
   - 可能导致某些样本被跳过或重复处理

### 3.2 FULLY模式的buffer切换逻辑

**观察**:

- FULLY模式的数据已在内存中
- 后续epoch只需要shuffle，不需要重新加载
- 但代码仍然使用了`current_ready_buffer_seq`和`load_next_buffer()`

**疑问**:

- FULLY模式是否需要buffer切换机制？
- 能否在epoch开始时一次性设置所有buffer为ready？
- 当前的切换机制是否引入了不必要的竞态？

### 3.3 NUMA架构的特殊问题

**可能的放大效应**:

1. **内存一致性的延迟**:

   - 跨NUMA节点的内存访问延迟更高（几十ns vs 几ns）
   - 缓存失效协议的延迟更大
   - `atomic.store()`的写入可能不会立即对其他核心可见

2. **原子操作的内存序**:

   ```
   current_ready_buffer_seq.store(next, std::memory_order_release);
   ```

   ```
   // ...
   ```

   ```
   size_t seq = current_ready_buffer_seq.load(std::memory_order_acquire);
   ```

   - `acquire/release`语义保证 happens-before 关系
   - 但在NUMA架构下，实际的可见性延迟可能更长
   - 是否需要更强的`seq_cst`内存序？

3. **缓存行伪共享**:

   - `workers_finished_`可能与其他高频访问的变量在同一缓存行
   - 96个Worker频繁递增可能导致缓存行抖动

## 四、待验证的假设

### 4.1 假设1：Preprocessor的JOIN时机问题

**描述**: `wait_workers_complete_buffer()`返回时，并非所有Worker都真正停止访问当前buffer

**验证方法**:

- 在`load_next_buffer()`前后添加日志
- 检查是否有Worker仍在访问旧buffer
- 使用`thread sanitizer`或类似的工具检测数据竞争

### 4.2 假设2：`get_next_sample()`的返回值语义不清

**描述**: `get_next_sample()`返回false时，Worker可能仍持有buffer的引用

**当前行为**:

```
while (loader.get_next_sample(...)) {
    // 处理
}
// 返回后，Worker递增 workers_finished_
```

**问题**:

- 最后一次调用`get_next_sample()`返回false
- 但Worker可能刚刚读取了`current_ready_buffer_seq`
- 还没有检查样本是否在范围内
- 主线程已经开始切换buffer

### 4.3 假设3：FULLY模式的过度设计

**描述**: FULLY模式不需要动态buffer切换，当前的切换机制引入了竞态

**简化方案**:

```
// FULLY模式：在begin_epoch时一次性设置
if (epoch_id > 0) {
    // 所有buffer已在内存，直接标记为ready
    for (auto& meta : buffer_metas) {
        meta.ready->store(true);
    }
    current_ready_buffer_seq.store(0);

    // 后续不再调用load_next_buffer()
}
```

### 4.4 假设4：NUMA架构的内存屏障问题

**描述**: 当前的原子操作内存序不足以保证NUMA架构下的可见性

**验证方法**:

- 尝试使用`std::memory_order_seq_cst`
- 在关键位置添加`std::atomic_thread_fence()`
- 使用硬件特定的内存屏障指令

## 五、已尝试的修复及其效果

### 5.1 修复1：原子化`current_ready_buffer_seq`

**修改**:

- 将`size_t`改为`std::atomic<size_t>`
- 所有访问使用`load(memory_order_acquire)`和`store(memory_order_release)`

**效果**: 有一定效果，卡死概率降低，但仍会卡死

**结论**: 必要但不充分，还有其他竞态条件

### 5.2 修复2：双重检查+局部拷贝

**修改**:

- 在等待buffer ready后，再次验证`current_ready_buffer_seq`未变
- 拷贝`shuffled_locations`和`slot_metas`到局部vector
- 后续只使用局部拷贝

**效果**: 有一定效果，卡死概率降低，但仍会卡死

**结论**: 有效但未根治，说明还有其他竞态窗口

## 六、建议的后续调试方向

### 6.1 短期：添加更多日志和断言

1. **在每个关键操作前后添加日志**:

   ```
   LOG_DEBUG << "Worker " << id << " reading current_ready_buffer_seq";
   ```

   ```
   size_t seq = current_ready_buffer_seq.load();
   ```

   ```
   LOG_DEBUG << "Worker " << id << " got seq=" << seq;
   ```

   ```
   
   ```

   ```
   // ... 使用seq ...
   ```

   ```
   
   ```

   ```
   LOG_DEBUG << "Worker " << id << " verifying seq";
   ```

   ```
   size_t verify = current_ready_buffer_seq.load();
   ```

   ```
   if (verify != seq) {
   ```

   ```
       LOG_ERROR << "Worker " << id << " detected seq change!";
   ```

   ```
   }
   ```

2. **添加断言检查**:

   ```
   // 在load_next_buffer()开始时
   ```

   ```
   for (int i = 0; i < num_preproc_workers_; i++) {
   ```

   ```
       TR_CHECK(worker_states_[i].global_seq >= expected_min,
   ```

   ```
                LogicError, "Worker " << i << " state inconsistent");
   ```

   ```
   }
   ```

3. **使用Thread Sanitizer**:

   ```
   ninja -C build clean
   ```

   ```
   CC=clang CXX=clang++ cmake -B build -DRENAISSANCE_USE_TSAN=ON
   ```

   ```
   ninja -C build
   ```

   ```
   bin/test_epoch_crc --dataset imagenet --format dts --mode fully --phase train --epoch 85
   ```

### 6.2 中期：重构Preprocessor的同步机制

**方案A：使用条件变量替代忙等待**

```
// 主线程
{
    std::unique_lock<std::mutex> lock(mutex_);
    buffer_ready_cv_.wait(lock, [&] {
        return workers_finished_.load() == num_workers;
    });
    loader.load_next_buffer();
    workers_finished_.store(0);
    buffer_ready_cv_.notify_all();
}

// Worker线程
{
    std::unique_lock<std::mutex> lock(mutex_);
    buffer_ready_cv_.wait(lock, [&] {
        return current_buffer_seq_.load() > last_seen;
    });
}
```

**方案B：引入epoch barrier**

```
// 在每个buffer结束时
{
    // 1. 主线程设置barrier
    epoch_barrier_.arrive_and_wait();

    // 2. 主线程切换buffer
    loader.load_next_buffer();

    // 3. 主线程通知Worker继续
    start_signal_.store(true);
}
```

### 6.3 长期：重新设计FULLY模式

**核心思想**: FULLY模式的数据已在内存，不需要动态切换

**方案**: 在epoch开始时一次性初始化，之后静态访问

```
void begin_epoch(int epoch_id, bool is_train) {
    if (mode == FULLY && epoch_id > 0) {
        // 1. shuffle full_shuffled_locations
        shuffle_full_dataset();

        // 2. 所有buffer标记为ready
        for (auto& meta : buffer_metas) {
            meta.ready->store(true);
        }

        // 3. 后续不再调用load_next_buffer()
        // Worker直接根据global_sample_idx计算buffer序号
        size_t buffer_idx = global_sample_idx / samples_per_buffer;
        const auto& meta = buffer_metas[buffer_idx];
        // ...
    }
}
```

## 七、我们的问题（包括但不限于）

1. **Preprocessor的JOIN机制**:
   - 当前的`wait_workers_complete_buffer()`是否正确？
   - 如何确保所有Worker都真正停止访问当前buffer后才切换？
2. **原子操作的内存序**:
   - NUMA架构下，`acquire/release`是否足够？
   - 是否需要使用`seq_cst`或添加内存屏障？
3. **FULLY模式的设计**:
   - 是否应该移除FULLY模式的动态buffer切换？
   - 有没有更简单的设计？
4. **调试工具**:
   - 有没有工具可以检测NUMA架构下的竞态条件？
   - Thread Sanitizer是否适用于这种情况？
5. **性能和正确性的权衡**:
   - 如果使用互斥锁或条件变量，性能损失有多大？
   - 有没有零开销的同步方案？