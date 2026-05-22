# 多流捕获指南

## 一、核心原理

CUDA Graph 多流捕获只有一条规则：**单流 `BeginCapture`，同线程其他流自动入图；跨流依赖必须用 `cudaEventRecord` + `cudaStreamWaitEvent` 显式建立**。

```cpp
cudaStreamBeginCapture(stream_1, cudaStreamCaptureModeThreadLocal);

// stream_1 上执行
cbr1_conv_genstats_graph->execute(handle_1, ...);
cudaEventRecord(ev1, stream_1);

// stream_2 上执行 — 自动入图，因为同线程
cudaStreamWaitEvent(stream_2, ev1, 0);
cbr1_bn_finalize_graph->execute(handle_2, ...);
cudaEventRecord(ev2, stream_2);

// stream_3 上执行 — 也自动入图
cudaStreamWaitEvent(stream_3, ev1, 0);
cudaStreamWaitEvent(stream_3, ev2, 0);
cbr1_bn_relu_graph->execute(handle_3, ...);

cudaStreamEndCapture(stream_1, &graph);
```

ThreadLocal 模式只捕获当前线程的操作，多线程环境下互不干扰。捕获块内所有 `cudaEventRecord` / `cudaStreamWaitEvent` 被 CUDA Graph 固化为内部拓扑边，运行时 `cudaGraphLaunch(exec, stream_1)` 一次调用即可重建全部多流并发。CPU 无需介入流调度。

---

## 二、三流架构

一个 CBR 组件（Conv+BN+ReLU）拆成三段，投到三条持久流上：

```
stream_1: Conv 1x1 + GenStats          (计算密集型)
stream_2: BN Finalize                  (规约操作，只等 S1 的 sum/sq_sum)
stream_3: BN Apply + ReLU              (点wise操作，等 S1 的 conv_out + S2 的 eq_scale)
```

**拆三而不拆二的原因**：S1 的 `sum/sq_sum`（几十 KB）远小于 `conv_out`（几十~几百 MB）。GenStats 归约完成即可触发 S2，无需等整个 Conv kernel 算完。三流分离利用这种"端粒先行、体积后到"的特征使 Conv 和 BN Finalize 重叠。

三条流需要用 `cudaStreamNonBlocking` 创建，避免 legacy default stream 的全局 barrier。每条流独立的 `cudnnHandle_t`（cuDNN 不保证多流安全），独立的 workspace（并发执行不能共享）。

```cpp
cudaStreamCreateWithFlags(&stream_1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream_2, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream_3, cudaStreamNonBlocking);

cudnnCreate(&handle_1); cudnnSetStream(handle_1, stream_1);
cudnnCreate(&handle_2); cudnnSetStream(handle_2, stream_2);
cudnnCreate(&handle_3); cudnnSetStream(handle_3, stream_3);

d_ws_1 = max(ws_cbr1_conv, ws_cbr2_conv, ws_cbar3_conv);    // 每条流一个 workspace
d_ws_2 = max(ws_cbr1_bn,   ws_cbr2_bn,   ws_cbar3_bn);
d_ws_3 = max(ws_cbr1_relu, ws_cbr2_relu, ws_cbar3_relu);
```

---

## 三、跨流同步

### 3.1 组件内 Fork-Join

每个 CBR 内部，S3 同时依赖于 S1 和 S2 的输出：

```
stream_1: ──[S1]──┬────────────────────
                  │ ev1
stream_2: ────────┼──[S2]──┬───────────
                  │         │ ev2
stream_3: ────────┼─────────┼──[S3]────  （等 S1 且 S2）
                  └─────────┘
```

```cpp
// S1 record → S2 wait → S2 record → S3 wait both
cudaEventRecord(ev1, stream_1);
cudaStreamWaitEvent(stream_2, ev1, 0);
// ... execute S2 ...
cudaEventRecord(ev2, stream_2);
cudaStreamWaitEvent(stream_3, ev1, 0);
cudaStreamWaitEvent(stream_3, ev2, 0);
// ... execute S3 ...
```

### 3.2 跨组件屏障

CBR1 的 S3（stream_3 写入 `d_relu_1_out`）→ CBR2 的 S1（stream_1 读取 `d_relu_1_out`）。**不在同一条流上，CUDA 不会自动保证顺序**。不显式同步 = 静默数据竞争（训练结果随机漂移）。

```cpp
// CBR1 S3 完成后，CBR2 的三条流全部等待
cudaEventRecord(ev_cbr1_done, stream_3);
cudaStreamWaitEvent(stream_1, ev_cbr1_done, 0);
cudaStreamWaitEvent(stream_2, ev_cbr1_done, 0);
cudaStreamWaitEvent(stream_3, ev_cbr1_done, 0);
// 现在 CBR2 安全启动
```

性能代价约 1-2%，但这是数学正确性的必要条件。

---

## 四、Running Stats 入图

BN 的 Running Statistics 滚动更新（`prev ← next`）是训练每轮必须做的。传统模式每轮额外调 CPU 函数发起 D2D DMA；Graph 模式下直接把 `cudaMemcpyAsync` 写进捕获块——`cudaGraphLaunch` 一次覆盖"计算 + stats 同步"。

```cpp
// 写在 cudaStreamEndCapture 之前
const size_t sz = C_out * sizeof(float);
cudaMemcpyAsync(d_prev_mean, d_next_mean, sz, cudaMemcpyDeviceToDevice, stream_1);
cudaMemcpyAsync(d_prev_var,  d_next_var,  sz, cudaMemcpyDeviceToDevice, stream_1);
```

Graph 模式运行时无需再调 `sync_running_stats()`。

---

## 五、完整流程

```
Phase 0（创建）:
  streams[3] = cudaStreamNonBlocking × 3
  handles[3] = cudnnCreate + cudnnSetStream
  workspaces[3] = 每条流上最大 ws

Phase 1（传统预热）:
  for 50 iters: execute_traditional() + sync_stats()
  cudaDeviceSynchronize()           ← 确保 cuDNN cache 填满

Phase 2（捕获）:
  cudaDeviceSynchronize()           ← 起始状态干净
  configure_l2_policy()             ← 固化 Stream 的 L2 属性
  cudaDeviceSynchronize()
  cudaStreamBeginCapture(stream_1, ThreadLocal)
  ├─ 全部 CBR S1/S2/S3 + 跨流事件 + 跨组件屏障 + stats DMA
  └─ cudaStreamEndCapture(stream_1, &graph)
  cudaGraphInstantiate(&exec, graph, &err_node, log, 2048)
  cudaGraphDestroy(graph)

Phase 3（运行时）:
  cudaGraphLaunch(exec, stream_1)   ← 一次调用，全部完成
```

---

## 六、规则清单

- **单流 begin / end**：`BeginCapture` 和 `EndCapture` 必须在同一条流上
- **ThreadLocal 模式**：只捕获当前线程的 kernel，不干扰其他线程
- **同线程自动入图**：begin 后同线程内所有流的操作全部自动入图
- **跨流必须显式同步**：`cudaEventRecord` + `cudaStreamWaitEvent` 是唯一跨流依赖表达方式
- **跨组件必须屏障**：N+1 层读 N 层输出时，三条流全部等待 N 层完成事件
- **Graph 内无阻塞 API**：不能有 `cudaStreamSynchronize` 或 `cudaDeviceSynchronize`
- **捕获前 cuDNN 预热**：`build_plans` 引擎搜索必须在传统模式完成，捕获只录已稳定的 API 调用
- **stats 入图**：`cudaMemcpyAsync` 写入捕获块，消除运行期 CPU 开销
- **独立 handle + workspace**：每条流独占，cuDNN 不保证多流安全
- **非阻塞流**：`cudaStreamNonBlocking` 避免 legacy default stream barrier