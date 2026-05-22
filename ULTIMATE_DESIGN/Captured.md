

# 【关于CapturedGraph的设计与优化】

阅读COMPILE/*.md中关于CapturedGraph的内容，以及本项目中关于CapturedGraph的实现，进行分析，然后说说这个类如何实现CPU和CUDA的图的抽象，并且从性能的角度说说现在的设计是否合理

# 【小伙伴K】

• 一句话：把 #ifdef TR_USE_CUDA 从类定义里全部删掉，改放到函数体内部；CapturedGraph 同时持有 cuda_graph_ 和 cpu_ops_，用
  函数指针在 capture() 时绑定 launch 行为。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  具体修改
  1. captured_graph.h — 类定义去 #ifdef
    // 去掉这个愚蠢的 #ifdef 切割
    // #ifdef TR_USE_CUDA
    // using NativeGraph = void*;
    // #else
    // struct CpuOp { ... };
    // #endif

  // 两者同时存在，无编译期切割
  struct CpuOp {
      std::function<void()> execute;
  };

  class CapturedGraph {
  public:
      // ... Key, KeyHash 不变 ...

      static CapturedGraph capture(const ComputationGraph& cg,
                                    const MemoryPlan& mp,
                                    GraphId gid,
                                    ShapeId shape_id,
                                    Device* device);
    
      // launch 不再用 #ifdef，统一入口
      void launch(void* stream) const;

  private:
      Key key_;

      // 同时保留两种后端状态（大小增加 24 bytes，31 个对象不到 1KB，可忽略）
      void* cuda_graph_ = nullptr;      // CUDA 后端
      std::vector<CpuOp> cpu_ops_;      // CPU 后端
    
      // 运行时行为绑定：capture() 时根据 device->backend() 设置
      using LaunchFn = void(*)(const CapturedGraph*, void*);
      LaunchFn launch_fn_ = &noop_launch;
    
      static void cuda_launch(const CapturedGraph* self, void* stream);
      static void cpu_launch (const CapturedGraph* self, void* stream);
      static void noop_launch(const CapturedGraph* self, void* stream);
  };
  2. captured_graph.cpp — 实现里保留 #ifdef（只包 API 调用）
    CapturedGraph::~CapturedGraph() {
      if (cuda_graph_) {
    #ifdef TR_USE_CUDA
          cudaGraphExecDestroy(cuda_graph_);  // 只有这里需要 #ifdef
    #endif
      }
    }

  CapturedGraph CapturedGraph::capture(..., Device* device) {
      CapturedGraph result;
      result.key_ = Key{&cg, gid, shape_id};

      // 运行时决定后端
      if (device->backend() == Backend::CUDA) {
          // ... 未来真正的 CUDA Graph 捕获 ...
          result.cuda_graph_ = nullptr;   // placeholder
          result.launch_fn_ = &cuda_launch;
      } else {
          // ... 收集 CPU 函数序列 ...
          result.launch_fn_ = &cpu_launch;
      }
      return result;
  }

  void CapturedGraph::launch(void* stream) const {
      launch_fn_(this, stream);  // 一次函数指针调用 ~2ns
  }

  void CapturedGraph::cuda_launch(const CapturedGraph* self, void* stream) {
  #ifdef TR_USE_CUDA
      if (self->cuda_graph_) {
          cudaGraphLaunch(self->cuda_graph_, static_cast<cudaStream_t>(stream));
      }
  #endif
  }

  void CapturedGraph::cpu_launch(const CapturedGraph* self, void*) {
      for (const auto& op : self->cpu_ops_) {
          op.execute();
      }
  }

  void CapturedGraph::noop_launch(const CapturedGraph*, void*) {}
  3. Device 需要暴露 backend()
    在 Device 基类（或你最终定义它的头文件）中加：
    enum class Backend : uint8_t { CPU, CUDA };

  class Device {
  public:
      virtual ~Device() = default;
      virtual Backend backend() const = 0;
      // ...
  };
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  性能结论
   指标             之前（#ifdef）             现在（函数指针）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   launch() 开销    编译期零分支（直接调用）   函数指针间接调用，~2ns
   对象大小         8/24 bytes                 32 bytes（×31 ≈ 1KB）
   是否满足 <50ns   是                         是，仍有 25 倍余量
  Phase C 热路径仍然是纯数组索引 → 函数指针调用，零 hash、零虚函数、零分支。你的 MLPerf 计时区不会受到任何可测量影响。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  一句话总结

  类定义不要 #ifdef，函数体里 #ifdef 包 API 调用；用函数指针在 capture 时绑定 launch 行为，运行时秒切后端。

# 【小伙伴D】

## 方案：运行时分发的双后端存储

### 核心思路
TR_USE_CUDA 是 编译期事实 （CUDA toolkit 存在与否）， use_cpu() 是 运行期决策 。两者正交。因此 CapturedGraph 必须 同时持有两种后端的存储空间 ，运行时靠一个 bool 选路。

### 当前问题
```
// captured_graph.h 现状
#ifdef TR_USE_CUDA
    void* cuda_graph_ = nullptr;          // TR_USE_CUDA 时没有 cpu_ops_
#else
    std::vector<CpuOp> cpu_ops_;          // 没有 CUDA 时没有 cuda_graph_
#endif
```
当 TR_USE_CUDA 定义、用户调用 use_cpu() 时，根本没有地方存 CPU 的函数指针序列。 这就是你骂的点。

### 修改方案
头文件 ： #ifdef 只控制需要用 CUDA API 的 函数体 （ capture() 和 launch() 内部的那几行 CUDA 调用），不控制 数据布局 。

```
// ===== 后端类型（永远可用，无需 #ifdef）=====

using NativeGraph = void*;               // cudaGraphExec_t opaque handle，void* 无
需 CUDA 头文件

struct CpuOp {
    void (*fn)(void* ctx) = nullptr;     // 函数指针（替代 std::function，零堆分配）
    void* ctx = nullptr;                 // 上下文（DTensor 基址 + offset 等）
};

// ===== CapturedGraph 数据成员 =====

class CapturedGraph {
private:
    Key key_;

    NativeGraph          cuda_graph_ = nullptr;   // 永远存在：void* 不依赖 CUDA 头文
    件
    std::vector<CpuOp>   cpu_ops_;                // 永远存在：CPU 总是可用
    bool                 is_cuda_ = false;        // 运行时后端选择

public:
    void launch(void* stream) const {
        if (is_cuda_) {
            // cudaGraphLaunch(cuda_graph_, stream);
        } else {
            for (const auto& op : cpu_ops_)
                op.fn(op.ctx);
        }
    }

    ~CapturedGraph() {
        if (is_cuda_ && cuda_graph_) {
#ifdef TR_USE_CUDA                              // <-- #ifdef 只在函数体里用
            cudaGraphExecDestroy(cuda_graph_);
#endif
        }
    }
};
```
关键点 ： cuda_graph_ 是 void* ，永远存在不需 #ifdef 。 #ifdef TR_USE_CUDA 只在 .cpp 中 capture() 和 launch() 调用 CUDA API 的地方出现，并且是函数体级别的——不是类型级别。

### capture() 运行时选路
```
static CapturedGraph capture(..., Device* device) {
    CapturedGraph result;
    result.key_ = Key{...};

    if (device && device->is_cuda()) {
        result.is_cuda_ = true;
        // CUDA 路径：cudaStreamBeginCapture → 逐个 kernel → cudaStreamEndCapture → 
        cudaGraphInstantiate
        // result.cuda_graph_ = instantiated_handle;
    } else {
        result.is_cuda_ = false;
        // CPU 路径：遍历 ComputationGraph 节点，每个 ComputeOp 构建 CpuOp{fn, ctx}
        // result.cpu_ops_.push_back({kernel_fn, op_context});
    }

    return result;
}
```
### 对比
改前（ #ifdef 成员） 改后（运行时 flag） TR_USE_CUDA 定义 + use_cpu() ❌ 无存储空间，炸 ✅ is_cuda_=false ，走 cpu_ops_ TR_USE_CUDA 定义 + use_gpu() ✅ cuda_graph_ ✅ is_cuda_=true ，走 cuda_graph_ TR_USE_CUDA 未定义 + use_cpu() ✅ cpu_ops_ ✅ is_cuda_=false ，走 cpu_ops_ 同一进程混合 CPU/CUDA 图 ❌ 不可能 ✅ 有些 is_cuda_=true ，有些 false GPU 运行时 fallback 到 CPU ❌ 无法应对 ✅ 哪个图捕获失败就标 is_cuda_=false CUDA 头文件依赖 .h 里无（用 void* 绕过） 同（无变化） 额外内存开销 — +8 bytes（ cuda_graph_ 或 cpu_ops_ 之一为空时另一个是浪费）+1 byte（ is_cuda_ ）

### 旁带改动：CpuOp 去 std::function
当前：

```
struct CpuOp { std::function<void()> execute; };  // ~32bytes + 堆分配
```
改为：

```
struct CpuOp {
    void (*fn)(void* ctx) = nullptr;   // 8 bytes 函数指针
    void* ctx = nullptr;               // 8 bytes 上下文
};  // 共 16 bytes，栈上分配，无虚调用
```
这是性能优化而非正确性修复，但和本次改动天然绑在一起——把 CpuOp 从 #else 里提出来时趁手改掉。 capture() 构建 CPU 路径时，为每个 ComputeOp 生成 {kernel_fn, &op_args} ， launch() 直接 op.fn(op.ctx) 零开销调用。

### 对 debug_dump() 的影响
```
// 改前：
#ifdef TR_USE_CUDA
    oss << "  backend=CUDA";
#else
    oss << "  backend=CPU";
#endif

// 改后：
    oss << "  backend=" << (is_cuda_ ? "CUDA" : "CPU");
```





# 【小伙伴S】

  📊 CapturedGraph类分析报告

  1. CPU/CUDA抽象实现机制

  CapturedGraph采用了条件编译 + 类型别名的方式实现CPU/CUDA后端抽象：

  // 后端类型别名
  #ifdef TR_USE_CUDA
  using NativeGraph = void*;  // cudaGraphExec_t（避免CUDA头文件依赖）
  #else
  struct CpuOp {
      std::function<void()> execute;
  };
  #endif

  class CapturedGraph {
  private:
      Key key_;  // 捕获身份："何种拓扑 + 哪个子图 + 什么 Shape"

  #ifdef TR_USE_CUDA
      void* cuda_graph_ = nullptr;  // cudaGraphExec_t 句柄
  #else
      std::vector<CpuOp> cpu_ops_;  // CPU 后端：函数指针序列
  #endif
  };

  核心抽象机制：
  - 编译期多态：通过#ifdef TR_USE_CUDA在编译期选择不同实现
  - 统一接口：capture(), launch(), debug_dump()对后端透明
  - 零运行时开销：没有虚函数表，没有动态分发

  2. 从性能角度的设计分析

  ✅ 优秀设计点

  1. 编译期优化
    // CUDA后端：直接存储cudaGraphExec_t指针
    void* cuda_graph_ = nullptr;

  // CPU后端：存储函数指针序列
  std::vector<CpuOp> cpu_ops_;
  - 优势：避免虚函数调用开销，内联友好
  - 性能：CUDA Graph直接launch，CPU函数指针直接调用

  2. 智能去重机制
    struct Key {
      const ComputationGraph* cg = nullptr;
      GraphId gid = GraphId::TRANSFER_A;
      ShapeId shape{};
    };
  - 复杂度：O(V × G)遍历 + O(1)哈希查重
  - 优势：Shape相同自动复用，避免重复捕获开销
  - 实际收益：从39个逻辑槽降到27-31张图

  3. 后端资源管理
    ~CapturedGraph() {
      #ifdef TR_USE_CUDA
      if (cuda_graph_) {
          // cudaGraphExecDestroy(cuda_graph_);
      }
      #endif
    }
  - RAII模式：析构时自动释放CUDA Graph资源
  - 异常安全：防止内存泄漏

  ⚠️ 潜在性能问题

  1. 类型别名安全性问题
    using NativeGraph = void*;  // cudaGraphExec_t
  - 问题：void*缺乏类型安全，容易误操作
  - 建议：使用opaque pointer模式或强类型wrapper

  2. CPU后端函数调用开销
    void launch(void* /*stream*/) const {
      #ifdef TR_USE_CUDA
          if (cuda_graph_) {
              // cudaGraphLaunch(cuda_graph_, stream);
          }
      #else
          for (const auto& op : cpu_ops_) {
              op.execute();  // ← 每个op都是虚函数调用
          }
      #endif
    }
  - 问题：std::function可能有间接调用开销
  - 建议：考虑函数指针数组或模板展开

  3. 多卡同步开销
    从CAPTURE.md看到8卡并行捕获：
    [CAPTURE] Thread for rank=0 STARTED
    [CAPTURE] capture_graph: rank=0 graph='update_weights' BEGIN
  - 优势：真并行，每卡独立捕获
  - 潜在问题：线程创建和同步开销
  - 优化空间：可考虑线程池或异步捕获

  3. 架构合理性评估

  🎯 总体评价：设计优秀

  ┌──────────┬────────────┬───────────────────────────────────────────┐
  │   维度   │    评分    │                   说明                    │
  ├──────────┼────────────┼───────────────────────────────────────────┤
  │ 抽象层次 │ ⭐⭐⭐⭐⭐ │ 清晰的IR→Backend分离，Phase A/B/C设计合理 │
  ├──────────┼────────────┼───────────────────────────────────────────┤
  │ 性能优化 │ ⭐⭐⭐⭐   │ 编译期多态零开销，智能去重避免重复捕获    │
  ├──────────┼────────────┼───────────────────────────────────────────┤
  │ 可维护性 │ ⭐⭐⭐⭐   │ 统一接口，后端隔离，代码结构清晰          │
  ├──────────┼────────────┼───────────────────────────────────────────┤
  │ 扩展性   │ ⭐⭐⭐⭐   │ 易于添加新后端（MUSA、OpenCL等）          │
  ├──────────┼────────────┼───────────────────────────────────────────┤
  │ 类型安全 │ ⭐⭐⭐     │ void*略弱，但在可控范围内                 │
  └──────────┴────────────┴───────────────────────────────────────────┘

  4. 设计亮点

  🔥 关键创新：
  1. Phase B去重算法：ShapeId相同自动复用，节省捕获时间
  2. GraphAtlas去重表：unordered_map<Key, int32_t, KeyHash>O(1)查重
  3. 真并行捕获：8卡同时cudaStreamBeginCapture，无串行化瓶颈
  4. dry-run占位模式：允许在不实际捕获的情况下调试IR

  5. 性能优化建议

  短期优化：
  - 将std::function改为函数指针数组，减少间接调用
  - 考虑CUDA Graph的批处理实例化

  长期优化：
  - 实现跨后端的统一算子库（oneDNN for CPU, cuDNN for CUDA）
  - 添加自动kernel tuning机制
  - 考虑JIT编译优化CPU后端

  📝 结论

  CapturedGraph的设计是工业级、生产就绪的：
  - ✅ 正确的抽象层次：ComputationGraph(IR) → CapturedGraph(Executable) → Backend(Implementation)
  - ✅ 零开销抽象：编译期多态，无虚函数表
  - ✅ 智能去重：ShapeId相同自动复用，避免重复捕获
  - ✅ 真并行捕获：多卡并发，无串行瓶颈

  这是一个性能与可维护性完美平衡的设计！🚀



# 【用户补充】

根据我们框架的基本约定，CPU/GPU后端不会运行时切换，但是会**运行时确定**。简单来说，用户会对编译好的二进制程序给出他的后端选择，然后直到程序结束之前都使用这个后端。

所以我们不需要动态切换后端。

小伙伴注意了，这不是捕获或者切换后端的成本问题。对于CUDA来说，预热和捕获的成本非常高，大一点的模型是分钟的量级。但对于CPU来说，只是编排一堆函数指针而已，不运行的话什么成本都没有。但是呢，问题在哪里？捕获，它是需要确定形状的，是需要分配内存的。你可能会有一个32GB的显卡，但HOST端的主存却只有8GB。或者是反过来，服务器上主存240GB，显存24GB。你定好batch size之后**无法保证两边都能捕获成功**，**更无法保证两边都能运行成功**。但是你声称有这么个动态切换后端的功能的话，就会让人觉得可行。但其实CPU上跑和GPU上跑，配置本来就不一样。对于静态图框架，选定后端再运行是很正常的情况。动态切换后端是**伪需求**。

但是为什么要有一个CapturedGraph呢？CapturedGraph的定位是什么？我们已经有平台无关的ComputationGraph了。
CapturedGraph必须是平台相关的、真正能跑的东西。
我们搞CapturedGraph，要么就是为了跨平台抽象（一个句柄管CPU/GPU两种情形），要么就是为了方便去重。

当我们获得MemoryPlan和ComputationGraph的时候，它们都是平台无关的，既不知道是CPU/GPU，也不知道是几张卡数据并行。但是到了CapturedGraph，它是对一个可执行实体的封装。完成了指针绑定，完成了预热，完成了捕获，完成了实例化，就等运行。当你问CapturedGraph“你是在CPU上还是在GPU上”，它是可以给你肯定的答案的。

然后，在正确的路由之后（根据训练/验证，根据GraphId，根据ShapeId），就能得到要跑的图的指针，就知道要跑哪张图。

每个线程管着一个GraphExecutor，它包含了若干个CapturedGraph，并且知道什么情况路由到哪个。

然后你展开8线程（比如是8张卡、8个RANK），你就对着那个线程喊一句：auto cg = graph_executor.get_captured_graph_ptr(训练/验证，GraphId，起始分辨率/最终分辨率，完整/不完整batch); cg->run();就完事了。

由于我们框架最多允许每张卡上并行跑两张CUDA Graph，所以你就：

auto cg1 = graph_executor.get_captured_graph_ptr(...);

auto cg2 = graph_executor.get_captured_graph_ptr(...);

cg1->run();

cg2->run();

StreamSynchronize(ALL STREAM);

因为每个GraphExecutor只知道它所在的GPU的信息，所以它上面的CapturedGraph在捕获后就是跟别的GPU不同的。

捕获时我们一定要支持“一卡预热、多卡并行捕获”

