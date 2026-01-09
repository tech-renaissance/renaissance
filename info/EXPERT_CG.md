你们的 NCCL 通信带宽只有 **≈4 GB/s**，比 PCIe 4.0 x16（理论 32 GB/s、实测 22–26 GB/s）低出一个数量级，这在单机 2 GPU 的情况下非常明显。  
下面我分三层说明定位思路与可立即验证的优化点。

---

### 一、根因分析（基于你们代码）

#### ✅ 1. NCCL 调用虽然“正确”，却 **没有真正产生并行通信**

在 `CudaDevice::allreduce_gradient()` 与 `broadcast_param()` 内：

```cpp
ncclAllReduce(..., comm_stream_);
cudaEventRecord(comm_ready_, comm_stream_);
```

→ 每个 GPU 都是在主机同步地调用 `ncclGroupStart()` / `End()`，而**各 GPU 的通信流都在同一个进程里按顺序启动**。  
这会导致 NCCL 内部在 GPU 0 通信时 GPU 1 仍处于等待状态，无法形成真正的对等 DMA。  
典型症状就是 3 ~ 5 GB/s 的伪串行带宽。

**验证：**

将测试样例里的双 GPU 通信改成真正并行的线程／进程式触发：

```cpp
std::thread t0([&] { gpu0.allreduce_gradient(g0); });
std::thread t1([&] { gpu1.allreduce_gradient(g1); });
t0.join(); t1.join();
```

只改这两行即可。如果带宽立刻升到 20+ GB/s → 问题确认为串行发起。

---

#### ✅ 2. NCCL “组启动”位置放错

在测试样例中：

```cpp
ncclGroupStart();
gpu0.allreduce_gradient(gpu_0);
gpu1.allreduce_gradient(gpu_1);
ncclGroupEnd();
```

但 `allreduce_gradient()` 内部又执行了 `cudaStreamWaitEvent()`、`cudaSetDevice()`、`ncclAllReduce()` 等多次切 GPU context。  
这会强迫驱动在两张卡之间不断切换上下文，破坏 NCCL 的 group 优化。

**正确做法：**

把 `ncclGroupStart/End` 与实际 `ncclAllReduce` 放在同一函数体里（即 `DeviceManager::allreduce_all` 级别），  
在循环里分别设 device 后再调用 `ncclAllReduce`，不要在外层套 group。

或改成：

```cpp
// 不在 Device 内部封装 group
gpu0.allreduce_gradient(..., group=false);
gpu1.allreduce_gradient(..., group=false);
```

然后在外面真正并行触发。

---

#### ✅ 3. 不必要的 `cudaStreamSynchronize()` 阻塞

你们的析构或同步函数中多处有：

```cpp
cudaStreamSynchronize(comm_stream_);
cudaStreamSynchronize(transfer_stream_);
```

在测试时执行完通信后立即又同步，会把 CPU 阻塞在总线 IO 完成上，从而让定时统计包括了 **同步等待时间**。  
NCCL 本身的通信早已异步完成，只是等你 `sync()` 才返回。

**改法（测试时）：**

- 改用事件计时，仅包裹 NCCL 调用，不包含同步。
- 或只 `cudaEventElapsedTime` 在 comm_stream 上测通信纯时延。

---

#### ✅ 4. 环境变量未手动设置，NCCL 可能退化为 P2P=off 模式

在 Windows / PCIe 服务器下如果没显式打开 P2P 支持，NCCL 会自动退化为通过 host 中转（GDRDMA 退化路径），  
带宽恰好就是你看到的 3 ~ 5 GB/s。

**检查：**

```bash
$ nvidia-smi topo -m
```

看两 GPU 之间是否为 `PHB`（仅 PCIe 桥）并且 `P2P` 为 `N`。  
如果是，请开以下环境变量后再测：

```bash
export NCCL_P2P_LEVEL=SYS        ## 允许系统级PCIe P2P
export NCCL_P2P_NET_DISABLE=1    ## 只走GPU直连，不走socket
export NCCL_DEBUG=INFO           ## 观察是否出现 P2P path enable
```

若依旧无效，再强制：

```bash
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=SYS
```

---

#### ✅ 5. 内核启动规模太大，通信实际只传 4 GB/s = 单方向拷贝速率 / 2

你的测试是 **AllReduce(sum)**：每个方向需要 2 × 数据量（发送 + 接收），再除以 2 GPU。  
计算带宽时应除以 2 × GPU数。例如：

```cpp
double effective_bytes = 2 * size_gb;  // 双向通信
double throughput = effective_bytes / (time_s);
```

按真实公式算可能已 8~10 GB/s，只是除数错误导致“看似 4 GB/s”。

---

#### ✅ 6. 测试参数：用 1 次大 2 GB 通信会触发 GPUDirect 分页缓冲

建议改为多次迭代统计平均：

```cpp
for(int i=0;i<20;++i){
    gpu0.allreduce_gradient(...);
    gpu1.allreduce_gradient(...);
}
cudaDeviceSynchronize();
```

前几次 NCCL 会分配内部缓冲（malloc/pin），导致首批慢。

---

### 二、建议的修改要点（最少侵入）

| 问题 | 修改方式 | 预期提升 |
|------|-----------|-----------|
| 串行启动（单线程） | 每个 GPU 启动独立线程或进程执行通信 | +5~6× |
| group 嵌套错误 | 把 ncclGroupStart/End 移到同一上下文 | +2× |
| P2P 未启用 | 设置 NCCL_P2P_LEVEL=SYS | +3–4× |
| 含同步计时 | 用事件测 GPU 端时间 | 修正偏差 |
| 单次测试 | 多次 warm‑up + 平均 | 稳定结果 |

---

### 三、带改测试示例（最小重写）

```cpp
#ifdef TR_USE_NCCL
#include <thread>

std::cout << "[NCCL Concurrent AllReduce Test]" << std::endl;
manager.setup_nccl(2);

auto run = [&](int rank){
    auto& gpu = manager.cuda(rank);
    for(int i=0;i<20;++i){
        gpu.mark_compute_done();
        gpu.allreduce_gradient(rank==0?g0:g1);
    }
    gpu.sync(TR_COMM_STREAM);
};

auto t0 = std::thread(run,0);
auto t1 = std::thread(run,1);
t0.join(); t1.join();
#endif
```

然后用 `cudaEvent` 测时间：

```cpp
cudaEvent_t s,e;
cudaEventCreate(&s); cudaEventCreate(&e);
cudaEventRecord(s, gpu0.get_comm_stream());
// 调用AllReduce
cudaEventRecord(e, gpu0.get_comm_stream());
cudaEventSynchronize(e);
float ms; cudaEventElapsedTime(&ms,s,e);
```

---

### 四、性能预期

| 环境 | 启用项 | 期望带宽 |
|------|---------|-----------|
| PCIe 4.0 x16 双 5090 | P2P+并行启动 | **22–28 GB/s** |
| 无 P2P（Host copy） | | 3–5 GB/s（即当前） |

---

### 五、额外建议

1. **初始化顺序优化**  
   在 `setup_nccl()` 里调用 `cudaSetDevice` 与 `ncclCommInitRank` 要匹配进程/线程上下文，否则 NCCL 内部隐式同步。

2. **流优先级**  
   计算与通信流都设最高优先级，仅传输流低优先级；目前你们的 streamPriorityRange 逻辑已对，但可检测 `cudaDeviceGetStreamPriorityRange` 返回顺序（有的驱动 least>greatest）。

3. **PCIe BAR1 大小**  
   若 BAR1 默认 256 MB，会限制 P2P chunk 大小，可在 `nvidia-smi -q | grep BAR1` 确认，必要时调整驱动参数开启 Resizable BAR。

---

#### 总结一句话

> **症状匹配“单线程 + 无 P2P” 回退路径。**  
> 把 NCCL 并行化执行、确保 P2P 通道生效后，单机双 5090 AllReduce/Broadcast 带宽应当超过 20 GB/s。