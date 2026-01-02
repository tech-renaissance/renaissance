# 【十、方案重要更新与补充】

**如果与上面的设计相冲突，请以本部分内容为准。**

## （一）关于C++头文件和源文件的标准开头规则

### 头文件（.h）标准开头规则

头文件开头必须包含以下信息，使用Doxygen格式注释块（以`/** ... */`包裹）。规则简单明确，直接复制粘贴即可使用：

```java
/**
 * @file 文件名.h
 * @brief 模块中文名称
 * @details 模块功能的详细描述（1-2句话，说明核心职责）
 * @version X.XX.XX
 * @date YYYY-MM-DD
 * @author 作者名（固定写"技术觉醒团队"）
 * @note 依赖项: 依赖1, 依赖2, ...（仅提及外部非核心依赖，不包括核心文件）
 * @note 所属系列: 模块系列（如data, model, trainer, 可以新加别的系列）
 */
```

**说明**：

- **`@file`**：文件名（如`tensor.h`），必须与实际文件名一致。
- **`@brief`**：模块中文名称（如“张量类声明”）。
- **`@details`**：详细描述（如“持有Storage句柄以及张量的形状、类型、偏移等信息”），1-2句话，便于文档生成。
- **`@version`**：版本号格式`X.XX.XX`（如`1.00.00`），主版本、次版本、修订版本各两位。
- **`@date`**：发布日期格式`YYYY-MM-DD`（如`2025-10-23`）。
- **`@author`**：可选字段，默认为“技术觉醒团队”。
- **`@note 依赖项`**：仅列出外部非核心依赖（如`protobuf`），用逗号分隔。不包括默认核心文件（如`tech_renaissance.h`），以简化解析。
- **`@note 所属系列`**：模块所属系列（如`data`, `model`, `trainer`），与文件组织结构一致，便于CMake脚本按系列生成编译选项。

**示例**：

```java
/**
 * @file tensor.h
 * @brief 张量类声明
 * @details 持有Storage句柄以及张量的形状、类型、偏移等信息，支持设备无关操作，维度不得超过4维
 * @version 1.00.00
 * @date 2025-10-23
 * @author 技术觉醒团队
 * @note 依赖项: protobuf
 * @note 所属系列: data
 */
```

### 源文件（.cpp 或 .cu）标准开头规则

源文件开头与头文件类似，但侧重实现细节。依赖项可包括相关头文件（非核心），便于脚本解析：

```java
/**
 * @file 文件名.cpp
 * @brief 模块中文名称实现
 * @details 实现核心逻辑的简要描述（1-2句话）
 * @version X.XX.XX
 * @date YYYY-MM-DD
 * @author 作者名（固定写"技术觉醒团队"）
 * @note 依赖项: 依赖1, 依赖2, ...（仅外部非核心依赖或相关头文件）
 * @note 所属系列: 模块系列（如data, model, trainer, 可以新加别的系列）
 */
```

**说明**：

- **`@brief`**：添加“实现”后缀（如“张量类实现”）。
- **`@details`**：描述实现重点（如“设备无关操作实现”）。
- **`@note 依赖项`**：可包括相关头文件（如`tensor.h`）和外部依赖（如`cuBLAS`），不包括核心文件。
- 其他字段同头文件。

**示例**：

```java
/**
 * @file tensor.cpp
 * @brief 张量类实现
 * @details 实现设备无关操作，维度不得超过4维
 * @version 1.00.00
 * @date 2025-10-23
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, cuBLAS
 * @note 所属系列: data
 */
```



## （二）关于cuDNN

CUDNN_ROOT的位置在C:\Program Files\NVIDIA\CUDNN\v9.17



## （三）关于Tensor生命周期管理

### **生命周期管理**：当Tensor对象的实际使用周期远短于其作用域时，是否需要提供显式释放内存的机制？

**核心观点**：

- 生命周期管理：坚持RAII原则，无需显式释放
- **不应引入显式释放方法**（如 `del(t1)` 或 `t1.release()`），应完全依赖C++ RAII（资源获取即初始化）机制和智能指针管理
- Tensor通过`std::shared_ptr<Storage>`管理内存，当最后一个引用被销毁时自动触发后端的内存释放
- **方案：无需显式释放方法，依靠RAII与作用域控制**

**主要理由**：

1. **显式释放是陷阱，违背现代C++哲学**，会破坏RAII模型，容易引入悬空指针和二次释放等严重bug
2. 违背框架"避免用户编程错误"的设计初衷
3. 在设计中采纳的 `Tensor` → `std::shared_ptr<Storage>` → `Backend` 句柄模式，其核心优势就是通过 `shared_ptr` 实现了**自动、安全**的内存生命周期管理。当最后一个指向 `Storage` 的 `Tensor` 或其他 `shared_ptr` 析构时，其关联的后端内存会被自动释放。我们应当强化而非破坏这一设计
4. 静态图框架可通过内存池机制实现更高效的内存复用

**建议做法**：

1. **利用C++作用域控制**：将Tensor声明限制在独立代码块内

   ```cpp
   {
       Tensor t1 = create_tensor();
       // 使用t1
   }  // t1自动析构，释放内存
   ```

2. **重置句柄**：`Tensor` 本质是个句柄。我们可以通过将其重置为一个空的 `Tensor` 来提前释放其对 `Storage` 的引用。

   ```cpp
   tr::Tensor t1 = create_some_tensor();
   // ... 使用t1 ...
   
   t1 = tr::Tensor(); // 将t1重置为空张量，效果等同于离开作用域
   
   // ... 后续代码 ...
   ```

3. **核心答案：静态图与内存池才是终极解决方案**：
   框架是**静态图**，这一点至关重要。在静态图模式下，整个计算图的内存需求可以在执行前被**完整分析**。这意味着：

   *   **内存池化**：框架可以预先分配一个或多个大的内存块（Memory Pool），而不是为每个 `Tensor` 单独申请。
   *   **内存复用**：`t1` 占用的内存，在它完成其在计算图中的角色后，内存池管理器可以立即将其标记为“可复用”，并分配给后续计算中的某个新 `Tensor`，例如 `t2`。
   *   因此，即使 `t1` 这个C++对象还“活着”，它指向的物理内存可能早已被“腾笼换鸟”。您对 `padding` 的精妙设计（“上一层直接往12×12的全为0的张量里写入10×10的结果”）已经体现了这种内存复用思想。



## （四）关于API文档和设计方案说明

为了方便开发者和AI了解API格式、用法和我们的设计理由，我们在docs目录下为几乎每一个类都写了一个.md文件的API文档。如果不明白函数怎么用或者为什么要那样设计，可以查阅docs目录下相应的.md文件。



## （五）关于卷积层和全连接层的Bias

卷积层和全连接层需不需要实现Bias呢？我们的观点是——不需要。理由是，Bias虽能略微提升表达能力，但在有BN层的情况下，Bias相当于完全是多余的。



## （六）关于AI编译器和深度学习加速器

2025年10月30日21:23:32，现在正式开始规划开发与我们自研深度学习框架配套的AI编译器和深度学习加速器。这不是项目扩张，恰恰相反，是项目分割。原本我们是打算实现一个拥有很强功能的深度学习框架，但现在我们把它的AI编译和硬件加速（后端）的功能分离出来，形成单独的“产品”，这样在开发体量相同的情况下，能够拿出3个独立的产品，一来功能划分更为明确，方便管理，二来显得我们的链条更加完善，甚至形成一个小“生态”，为后续项目的扩展打好基础。我们想要演示的功能依然不会太复杂，只是把链条划分为3段。

关于AI编译器和深度学习加速器的名称设想我已有基本方案，中文名称和英文名称都有。

AI编译器就叫“算力场”（Computing Power Field）。

基于FPGA的深度学习加速器就叫“FI-0420”（Flaming Icarus 0420）。叫Flaming Icarus是因为我总是用Icarus来写Verilog HDL。0420是一个重要的日期，是我以前在研究所启动深度学习加速器项目（但最终未能完成）的日期，很有纪念意义。此外，巧妙的是，“FI”也可以解释为“Fast Inferencer”或“FPGA Inferencer”，表示我们的项目是用FPGA做加速推理的。

2025年10月30日21:38:21，AI编译器的GitHub项目名确定为“cpf-compiler”！今后我们叫它CPF编译器。

2025年10月30日21:51:09，经过专家团队论证，深度学习加速器的GitHub项目名称确定为“fi-0420”！今后我们叫它FI加速器。



## （七）关于指定输出张量位置的运算函数

### **第一问：指定输出位置的做法是否科学、必要？**

**答案是：非常科学，极其必要。**

这不是临时的变通方案，而是通往高性能、低开销架构的**必经之路**。理由如下：

1.  **性能核心：避免内存分配开销**
    在深度学习的计算密集型场景中，尤其是在GPU上，频繁地分配和释放内存（`cudaMalloc`/`cudaFree`）是巨大的性能杀手。能够复用已分配的内存缓冲区，是所有高性能计算库（包括cuDNN、cuBLAS）的共同实践。

2.  **契合静态图的终极目标：内存池**
    文档中明确框架是**静态图**，并规划了统一的内存池。静态图的最大优势之一就是可以在执行前**预先分析整个计算图的内存使用**。届时，框架会预分配一个或多个巨大的内存块（Memory Pool），然后将不同算子的输出指向这个内存池的不同位置。现在设计的“指定输出位置”的函数，正是实现内存池化和内存复用的**原子操作**。今天的`relu_into(a, b)`，在未来就是`relu_into(a, memory_pool.get_slice(...))`，API完全兼容。

3.  **赋予上层调度灵活性**
    将内存管理权交给调用者（无论是用户，还是未来的`Task`或图执行器），使得更复杂的内存优化（如“padding优化”）成为可能。这完全符合“以开发者为中心”和追求极致性能的理念。

4.  **业界标准实践**
    PyTorch（`torch.relu(a, out=b)`）、NumPy（`np.add(a, b, out=c)`）、OpenCV等几乎所有主流计算库都提供了这种`out`参数形式的函数，这证明了其在实践中的价值和必要性。

### **第二问：哪种函数名和形式最合理？**

众多AI共同决策的最终选择是：`op_into(in, out)`。实际的C++形式将类似于`void relu_into(const Tensor& input, Tensor& output);`

理由如下：

-   **清晰无歧义**：通过`_into`后缀，明确告知用户这是一个“有指定输出”的版本，完美区分了返回新张量的`op()`和原地操作的`op_inplace()`。
-   **命名系统性**：形成了`op`、`op_inplace`、`op_into/op_out`三位一体的命名体系，非常规整。
-   **符合直觉**：`in, out`的参数顺序遵循了自然的数据流，可读性强。
-   **可扩展性强**：对于二元操作，`add_into(a, b, out)`的语义清晰明了。

至于后缀应该是`out`还是`into`，我们最终选定`into`，理由是：`into`这个词在英文中带有强烈的“进入”、“放入”的动态含义。`relu_into(input, output)`可以自然地读作“put the result of relu(input) **into** output”，这在语感上比`relu_out`更生动和贴切。

**最终，我们把后端类的张量运算函数分为三大类型：return型（返回结果张量）、inplace型（原地写入结果）、into型（将结果写入指定张量）。**



## （八）关于技术觉醒2的性能测试

我们前面说到，技术觉醒2在其已实现的关键运算上，性能超越PyTorch。以下是我们此前测试的具体结果。

**基本运算对比**：

| Device | Operation | PyTorch Performance (GFLOPS) | TR Performance (GFLOPS) | Percentage  |
| :----: | :-------: | :--------------------------: | :---------------------: | :---------: |
|  CPU   |    MM     |            907.97            |         123.18          |   13.57%    |
|  CPU   |  CONV3X3  |            907.59            |         346.38          |   38.16%    |
|  CPU   |  CONV1X1  |            591.42            |         165.05          |   27.91%    |
|  CPU   | TCONV3X3  |            942.28            |         204.78          |   21.73%    |
|  CUDA  |    MM     |           6604.40            |         6678.33         | **101.12%** |
|  CUDA  |  CONV3X3  |           8394.59            |        11896.71         | **141.72%** |
|  CUDA  |  CONV1X1  |           5781.71            |         6602.31         | **114.19%** |
|  CUDA  | TCONV3X3  |           8420.02            |        13418.89         | **159.37%** |

全部的GFLOPS数据都是独立的10次测试取平均值的结果。

**训练性能对比**：

#### （1）Intel Core i9 + Windows

| 优化器 | PyTorch  | Tech Renaissance | Speed Up |
| :----: | :------: | :--------------: | :------: |
|  SGD   | 108.40 s |     60.85 s      |  1.78×   |
|  Adam  | 112.00 s |     67.90 s      |  1.65×   |
| AdamW  | 114.30 s |     67.95 s      |  1.68×   |

测试条件：Intel Core i9-14900HX，内存32.0 GB，Windows 11专业版，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。

#### （2）Intel Xeon + Ubuntu

| 优化器 | PyTorch  | Tech Renaissance | Speed Up |
| :----: | :------: | :--------------: | :------: |
|  SGD   | 177.30 s |     79.85 s      |  2.22×   |
|  Adam  | 180.60 s |     97.15 s      |  1.86×   |
| AdamW  | 181.50 s |     97.10 s      |  1.87×   |

测试条件：Xeon Platinum 8369B，内存60.0 GB，Ubuntu 24.04 LTS，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。

**测试准确率对比**：

#### （1）Intel Core i9 + Windows

| 优化器 | PyTorch | Tech Renaissance |  Diff  |
| :----: | :-----: | :--------------: | :----: |
|  SGD   | 98.29%  |      98.34%      | 0.06%  |
|  Adam  | 98.07%  |      98.09%      | 0.02%  |
| AdamW  | 98.07%  |      98.04%      | -0.03% |

测试条件：Intel Core i9-14900HX，内存32.0 GB，Windows 11专业版，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。所有数据都是20次独立测试的结果取平均值。

#### （2）Intel Xeon + Ubuntu

| 优化器 | PyTorch | Tech Renaissance | Diff  |
| :----: | :-----: | :--------------: | :---: |
|  SGD   | 98.26%  |      98.36%      | 0.09% |
|  Adam  | 98.06%  |      98.07%      | 0.01% |
| AdamW  | 98.05%  |      98.07%      | 0.02% |

测试条件：Xeon Platinum 8369B，内存60.0 GB，Ubuntu 24.04 LTS，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。所有数据都是20次独立测试的结果取平均值。

上述测试结果给了我们极大的信心：技术觉醒2框架尚且能够在性能上超越PyTorch，那么架构、工具进一步优化设计后的技术觉醒3框架就更有希望实现优越的性能。



## （九）关于技术觉醒2的头文件包含关系与前向声明情况

我们的各个类之间存在依赖关系，但又要避免循环包含的情况，因此需要合理安排包含关系和前向声明。截至2025年11月5日20:03:47，技术觉醒2框架的张量-后端系统稳定运作，其中的头文件包含关系与前向声明情况如下表所示（注："包含的头文件"只列出列表最左侧一列提到的头文件）：

|       文件名        |      类名      |                   包含的头文件                    |  包含的前向声明  |
| :-----------------: | :------------: | :-----------------------------------------------: | :--------------: |
|       shape.h       |     Shape      |                        无                         |        无        |
|      shape.cpp      |     Shape      |                      shape.h                      |        无        |
|      offset.h       |     Offset     |                        无                         |        无        |
|     offset.cpp      |     Offset     |                     offset.h                      |        无        |
|      device.h       |     Device     |                        无                         |        无        |
|     device.cpp      |     Device     |                     device.h                      |        无        |
|      storage.h      |    Storage     |                     device.h                      |        无        |
|     storage.cpp     |    Storage     |                     storage.h                     |        无        |
|      tensor.h       |     Tensor     |                 shape.h, device.h                 | Storage, Backend |
|     tensor.cpp      |     Tensor     | tensor.h, storage.h, backend.h, backend_manager.h |        无        |
|      backend.h      |    Backend     |                     device.h                      |  Tensor, Shape   |
|    cpu_backend.h    |   CpuBackend   |           backend.h, shape.h, offset.h            |        无        |
|   cpu_backend.cpp   |   CpuBackend   |              cpu_backend.h, tensor.h              |        无        |
|   cuda_backend.h    |  CudaBackend   |                     backend.h                     |        无        |
|  cuda_backend.cpp   |  CudaBackend   |        cuda_backend.h, tensor.h, storage.h        |        无        |
|  backend_manager.h  | BackendManager |                     device.h                      |     Backend      |
| backend_manager.cpp | BackendManager | backend_manager.h, cpu_backend.h, cuda_backend.h  |        无        |

注意：技术觉醒2框架的“后端类”相当于技术觉醒3框架的“器件类”。



## （十）关于技术觉醒2的后端解耦的进一步说明

一个被现代框架广泛采纳且清晰解耦的建议做法：**基于句柄的内存管理模式**。核心思想是：**张量本身不直接拥有设备内存，而是持有一个指向内存的“句柄”或“存储对象”。所有与内存相关的操作（分配、释放、拷贝）都通过后端接口来完成。**我们需要定义三个核心的C++类来优雅地解决这个问题：

**a) `Storage` (内存存储类)**

- **职责**：封装一块原始的、连续的内存。它是一个纯粹的、与设备无关的数据容器。
- 成员：
  - `void* data_ptr`: 指向原始内存的指针。
  - `size_t size`: 内存大小（字节）。
  - `std::shared_ptr<void> data_holder`: 一个智能指针，用于管理内存的生命周期。这是实现RAII（资源获取即初始化）和自动内存释放的关键。当最后一个指向这块内存的`Storage`对象被销毁时，`data_holder`的删除器会调用后端接口来释放内存。
  - `Device device`: 一个枚举或结构体，标明这块内存位于哪个设备上（如 `Device::CPU`, `Device::CUDA(0)`）。
- **关键点**：`Storage`本身不知道如何分配或释放内存。它只是被动地持有一个指针和管理其生命周期。

**b) `Backend` (后端抽象基类)**

- **职责**：定义所有与硬件相关的操作的纯虚函数接口。这是实现多态的关键。
- 核心接口：
  - `virtual std::shared_ptr<void> allocate(size_t size) const = 0;`: 分配内存。返回一个`shared_ptr`，其自定义删除器会调用`deallocate`。
  - `virtual void deallocate(void* ptr) const = 0;`: 释放内存。
  - `virtual void copy(void* dst, const void* src, size_t size, const Device& dst_device, const Device& src_device) const = 0;`: 内存拷贝。这个函数需要能处理CPU->CPU, CPU->CUDA, CUDA->CPU, CUDA->CUDA等多种情况。
  - `virtual void* get_data_ptr(const std::shared_ptr<void>& holder) const = 0;`: 从智能指针中获取原始指针（某些后端可能需要特殊处理）。
  - 以及其他算子接口，如 `virtual void conv_forward(...)` 等。

**c) `Tensor` (张量类)**

- **职责**：持有张量的元数据和一个指向`Storage`的句柄。它是用户直接交互的对象。
- 成员：
  - `std::shared_ptr<Storage> storage`: 指向`Storage`对象的共享指针。多个`Tensor`可以共享同一个`Storage`（例如，切片操作`y = x[0]`）。
  - `Shape shape`: 张量的形状 (e.g., {N, C, H, W})。
  - `Dtype dtype`: 数据类型 (e.g., `float`, `int8_t`)。
  - `size_t offset`: 在共享的`Storage`中的偏移量（用于支持切片等视图操作）。
  - `Strides strides`: 步长（同样用于视图操作）。

为什么说这个设计好？

1. **彻底解耦**：`Tensor`类完全不知道什么是`cudaMalloc`或`new float[]`。它只关心形状、类型和一个`Storage`句柄。这使得`Tensor`类的代码极其干净，与任何具体硬件无关。
2. 完美契合“可重构”：当想增加一个新的后端（如摩尔线程GPU），只需要：
   - 创建一个`MttS80Backend`类，继承`Backend`。
   - 实现所有的纯虚函数（`allocate`, `deallocate`, `copy`, `conv_forward`等）。
   - 在后端工厂/管理器中注册它。
   - `Tensor`和`Storage`的代码一行都不用改！
3. **内存安全**：通过`std::shared_ptr`和自定义删除器，内存管理变得自动化且安全。无论发生什么（包括异常），只要`Storage`对象被正确销毁，它持有的内存就会被对应的`Backend`释放，有效防止内存泄漏。
4. **视图操作高效**：由于`Tensor`持有`Storage`的共享指针，切片等操作（如`y = x[:, 0]`）只需创建一个新的`Tensor`对象，指向同一个`Storage`，但更新`shape`、`offset`和`strides`即可，无需任何数据拷贝，效率极高。
5. 清晰的责任划分：
   - `Tensor`：**是什么**（形状、类型）。
   - `Storage`：**在哪里**（设备）和**数据本体**。
   - `Backend`：**怎么做**（如何操作特定设备上的数据）。

这个设计虽然引入了一点间接性（通过句柄），但它带来的可维护性、可扩展性和健壮性，对于构建一个长期支持多平台的深度学习框架来说，是完全值得的。



## （十一）关于技术觉醒2的极简API

2025年11月25日，在实现了Task类之后，技术觉醒2框架的API大幅简化，用户体验巨幅上升。

在我们的最简单的演示样例中，仅仅**27行代码**（包括空行）就实现了**功能完整、语义清晰、输出丰富、性能强大**的MLP训练，而相同的功能在PyTorch大约需要5倍的代码量来实现。不仅如此，这个示例程序的运行速度还比PyTorch快1.7倍以上！

下面贴出我们完整的示例代码：

```c++
#include "tech_renaissance.h"

using namespace tr;

int main() {
    auto backend = BackendManager::get_cpu_backend();
    auto mnist = MnistDataset(backend, std::string(WORKSPACE_PATH) + "/../../MNIST/tsr/");
    auto model = Model::create("MLP",
        std::make_shared<Flatten>(),
        std::make_shared<Linear>(784, 512),
        std::make_shared<Tanh>(),
        std::make_shared<Linear>(512, 256),
        std::make_shared<Tanh>(),
        std::make_shared<Linear>(256, 10)
    );
    auto loss_fn = CrossEntropyLoss();
    auto optimizer = SGD(0.1f);
    auto scheduler = ConstantLR(0.1f);
    auto trainer = Trainer(model, loss_fn, optimizer, scheduler);
    auto task = Task(model, mnist, trainer);
    TaskConfig cfg;
    cfg.num_epochs = 20;
    cfg.batch_size = 128;
    task.config(cfg);
    task.run();
    return 0;
}
```

部分输出如下：

```shell
R:\tech-renaissance\build\clion-mingw-release\bin\tests\test_task_sgd.exe
[2025-11-25 07:06:21.181] [INFO] [TR] Initializing backend manager...
[2025-11-25 07:06:21.182] [INFO] [TR] CPU backend initialized
[2025-11-25 07:06:21.182] [INFO] [TR] CPU backend registered successfully
[2025-11-25 07:06:21.182] [INFO] [TR] CUDA backend not compiled (TR_USE_CUDA=OFF)
[2025-11-25 07:06:21.182] [INFO] [TR] Backend manager initialization completed


[INFO] Model Information:
  - Model: [Provided]
  - Device: CPU
[INFO] Dataset Information:
  - Training samples: 60000
  - Test samples: 10000
[INFO] Training Configuration:
  - Epochs: 20
  - Batch size: 128
Loading MNIST dataset...
Loading MNIST train data...
[2025-11-25 07:06:21.186] [INFO] [TR] Importing tensor from R:/tech-renaissance/workspace/../../MNIST/tsr/train_images.t
sr
[2025-11-25 07:06:21.238] [INFO] [TR] Tensor imported successfully from R:/tech-renaissance/workspace/../../MNIST/tsr/tr
ain_images.tsr
[2025-11-25 07:06:21.239] [INFO] [TR] Importing tensor from R:/tech-renaissance/workspace/../../MNIST/tsr/train_labels.t
sr
[2025-11-25 07:06:21.241] [INFO] [TR] Tensor imported successfully from R:/tech-renaissance/workspace/../../MNIST/tsr/tr
ain_labels.tsr
Original image shape: (60000,1,28,28)
Original label shape: (60000)
Loading MNIST t10k data...
[2025-11-25 07:06:22.248] [INFO] [TR] Importing tensor from R:/tech-renaissance/workspace/../../MNIST/tsr/test_images.ts
r
[2025-11-25 07:06:22.260] [INFO] [TR] Tensor imported successfully from R:/tech-renaissance/workspace/../../MNIST/tsr/te
st_images.tsr
[2025-11-25 07:06:22.282] [INFO] [TR] Importing tensor from R:/tech-renaissance/workspace/../../MNIST/tsr/test_labels.ts
r
[2025-11-25 07:06:22.282] [INFO] [TR] Tensor imported successfully from R:/tech-renaissance/workspace/../../MNIST/tsr/te
st_labels.tsr
Original image shape: (10000,1,28,28)
Original label shape: (10000)
[OK] MNIST dataset loaded successfully!
=== MNIST Dataset Information ===
Training samples: 60000
Test samples: 10000
Input shape: (1,28,28)
Output shape: (10)
Mean: 0.1307, Std: 0.3081
================================

--- Epoch 1/20 ---
Learning Rate: 0.1
Batch [  0/469] Loss: 2.3304    Accuracy: 7.81%
Batch [100/469] Loss: 0.4858    Accuracy: 85.94%
Batch [200/469] Loss: 0.1977    Accuracy: 96.09%
Batch [300/469] Loss: 0.3699    Accuracy: 89.06%
Batch [400/469] Loss: 0.2753    Accuracy: 93.75%
Epoch 1 - Train Loss: 0.3671    Train Accuracy: 89.82%
Epoch 1 - Test Loss:  0.2160    Test Accuracy:  93.86%

New best test accuracy achieved: 93.86%
New best model has been saved to: trained_model.mdl
Best test accuracy so far: 93.86% (Epoch 1)

--- Epoch 2/20 ---
Learning Rate: 0.100000
Batch [  0/469] Loss: 0.2616    Accuracy: 90.62%
Batch [100/469] Loss: 0.2226    Accuracy: 93.75%
Batch [200/469] Loss: 0.1062    Accuracy: 96.09%
Batch [300/469] Loss: 0.1498    Accuracy: 95.31%
Batch [400/469] Loss: 0.0900    Accuracy: 97.66%
Epoch 2 - Train Loss: 0.1871    Train Accuracy: 94.53%
Epoch 2 - Test Loss:  0.1554    Test Accuracy:  95.69%

New best test accuracy achieved: 95.69%
New best model has been saved to: trained_model.mdl
Best test accuracy so far: 95.69% (Epoch 2)

...

--- Epoch 20/20 ---
Learning Rate: 0.100000
Batch [  0/469] Loss: 0.0017    Accuracy: 100.00%
Batch [100/469] Loss: 0.0020    Accuracy: 100.00%
Batch [200/469] Loss: 0.0026    Accuracy: 100.00%
Batch [300/469] Loss: 0.0025    Accuracy: 100.00%
Batch [400/469] Loss: 0.0010    Accuracy: 100.00%
Epoch 20 - Train Loss: 0.0032   Train Accuracy: 99.99%
Epoch 20 - Test Loss:  0.0596   Test Accuracy:  98.28%
Best test accuracy so far: 98.39% (Epoch 17)


Task Completed!
Total Train Time: 56.881 s      Total Test Time: 4.563 s
Total Time: 61.473 s

Best test accuracy: 98.39%

Logs saved to: training.txt

进程已结束，退出代码为 0

```



## （十二）关于视图

View（视图）是技术觉醒框架中的核心特性，允许以零拷贝的方式重新解释张量的形状。View操作不会复制任何数据，而是创建一个新的张量对象，共享原始张量的底层存储空间，但具有不同的形状和步长信息。

### 为什么需要View？

#### 传统方法的痛点

```cpp
// 传统方法：需要复制数据
Tensor original = cpu_backend->zeros(Shape(2, 3, 4), DType::FP32);
Tensor reshaped = cpu_backend->empty(Shape(6, 4), DType::FP32);
cpu_backend->copy_into(original, reshaped);  // 数据复制，内存翻倍，性能损失
```

#### View方法的优势

```cpp
// View方法：零拷贝
Tensor original = cpu_backend->zeros(Shape(2, 3, 4), DType::FP32);
Tensor reshaped = cpu_backend->view(original, Shape(6, 4));  // 无数据复制，内存不变
```

### 核心概念

#### 1. 什么是View？

View是原始张量的一个"分身"，具有以下特性：

- **共享存储**: 与原始张量共享相同的内存空间
- **不同形状**: 可以有不同的形状解释
- **不同步长**: 根据新形状重新计算内存访问步长
- **零拷贝**: 创建过程不涉及任何数据复制

#### 2. 内存共享模型

```
原始张量: Storage (共享存储区域)
            ↑
     ┌────┴────┐
     │         │
Tensor A   Tensor B (View)
Shape(2,3,4)  Shape(6,4)
```



## （十三）关于高性能算子库的选型

我们的技术觉醒2框架在CPU上是采用Eigen3来实现加速的。但是Eigen3对NHWC的支持不到位，我们有必要采用新一代的高性能算子库。

经过调研，我们确信在x86 CPU上，oneDNN是不二之选。

而对于ARM上的推理，大量的信息指向XNNPACK。它是ARM上INT8推理的顶流，很好地弥补了oneDNN在ARM INT8推理上的缺陷。

对于有些推荐说的OpenVINO，它本身可视为半个深度学习框架，用在我们的框架中不太合适。

至于CUDA和MUSA，我们毫无疑问选择官方准备好的cuDNN和muDNN。



## （十四）关于ARM上的功能定义

XNNPACK对BF16的支持非常有限，我们在ARM上首先直接禁用这种数据类型，也就禁用AMP。

此外，在端侧基本上没有训练需求，我们也不推荐在端侧进行训练，为了节省时间人力，我们不开发ARM上的训练功能。

所以ARM上需要支持的只有**FP32推理**和**INT8推理**。

除非ARM连接GPU，然后用GPU实现训练，但这种情形在我们的项目中暂不考虑。我们假定训练只在PC和数据中心进行，而ImageNet的训练只在数据中心进行，并假定PC和数据中心的CPU都是x86架构。

如果使用了未实现的算子，结果当然是报NotImplementedError。但如果不使用，其他功能都可以正常运作。因此在ARM上削减训练功能对我们的框架来说实现起来并不复杂。



## （十五）关于技术觉醒2的技术文档

开发技术觉醒2框架时，我们在项目文件夹的docs文件夹下，为每一个类和每一个重要功能留下了宝贵的设计说明，都是markdown格式，不但介绍了API，也介绍了设计思路和原理，这些很多都可以用作技术觉醒3的开发参考。

但使用这些文档的时候要特别注意：**第一，技术觉醒2的“后端”指的是技术觉醒3的“器件”；第二，技术觉醒2的维度顺序是NCHW，而技术觉醒3的维度顺序是NHWC，这在涉及Shape类和Tensor类的地方差异很明显；第三，技术觉醒2完全没考虑BF16，但对于技术觉醒3，这是必须支持的一个数据类型。**



## （十六）关于内存池/显存池

内存池在技术觉醒2中被绕过了，但在技术觉醒3中，这可能是一个重要的优化方向，甚至是整个框架的地基。

经过多方论证，我们认为应该实现**器件级的内存池**，但暂不考虑跨器件管理。如果不实现内存池，相当于放弃了静态图框架的最大红利。专家认为实现内存池可以再获得10%~50%的端到端性能提升，这在深度学习框架性能竞争当中是决定性的。

此外，我们认为，**深度学习算法**（Tensor）和**图像预处理**（Preprocessor）各需要一个内存池，而且这两个内存池性质有所不同。Preprocessor 不需要、也不应该复用 Tensor / Storage 那一套内存池；但它必须有自己的、专用的内存复用机制。Tensor内存池是“计算图级”的结构性优化；Preprocessor内存池是“流水线级”的工程性优化。Tensor内存池的Tensor的生命周期极其固定、形状极其固定；而Preprocessor内存池的ImageTemp（或其他类型对象）的生命周期、形状变化较大。



## （十七）关于双版本

前文提到，我们在开源问题上有所保留，因此准备做一个闭源的完整版和一个开源的阉割版。闭源版本开发验证完成后，再将其裁剪部分功能，改为开源版本。

关于开源和闭源两个版本的称呼，此前有过争议。刚开始我们打算“技术觉醒2开源、技术觉醒3闭源”，但这显然会带来问题，因为技术觉醒2已经是旧版本，架构都还是基于NCHW，跟技术觉醒3完全不兼容，而且我们很明确是先开发闭源再开发开源，显得技术觉醒3的开发是发生在技术觉醒2之前，这样会引起混乱。后来我们提出“闭源的叫3F，开源的叫3O”，但这个名称也会让人疑惑。

2025年12月16日17:23:33，我们提出的最新方案是：闭源的叫**“技术觉醒（renAIssance）”**或**“技术觉醒完整版（renAIssance-full）”**，开源的叫**“技术觉醒轻量版（renAIssance-lite）”**，显得更专业而明确，不在名称上强调开源闭源，而是用full和lite来表明一个是另一个的轻量化版本。

此外，我们没有必要强调这是第3代，因为前2代根本就没有发布。此后对于我们开发的框架，除非特别声明，“技术觉醒”一般指“技术觉醒3”。

当然，版本号依然继续沿用，从现在起开发的版本都是3.X.X。



## （十八）关于预处理瓶颈

我们在PyTorch上训练ResNet-50时，双GPU训练速度优化到极致大概是300s/epoch。这时我们发现CPU的各个核的使用率都达到或接近100%，运行的线程都是data_worker，也就是图像预处理。据我们了解，PyTorch的图像预处理默认都是在CPU上进行的。另一方面，GPU的显存和功耗都还有闲余。这可能说明，CPU上执行图像预处理的速度是性能瓶颈。我们提出图像预处理的gpu offload方案，也就是把一部分预处理任务交给GPU分担。
在我们的演示方案中，训练过程的预处理主要是RandomResizedCrop、RandomHorizontalFlip、TrivialAugmentWide、ToTensor、Normalize、RandomErasing（对于MNIST还有个RandomAffine），验证过程的预处理主要是Resize、CenterCrop、ToTensor、Normalize。我们认为，TrivialAugmentWide属于过于复杂的运算，交给GPU实现会非常困难。而RandomResizedCrop和Resize、CenterCrop最好还是由CPU执行，这样我们就能得到比较小的、尺寸标准化的图像。这样一来，GPU上可以执行的预处理操作就是RandomHorizontalFlip、RandomErasing、ToTensor、Normalize。
ToTensor操作会把8bit的像素通道变成32bit的浮点数，体积增大4倍，这个操作如果在CPU上完成，不但给CPU增加负荷，而且在拷贝到显存的过程中也容易碰到带宽瓶颈。所以ToTensor必须在GPU完成，这种操作对GPU来说应该没有难度。作为输入图像，通道数较少，因此占用的显存也应该不多。在我们的训练方案中，显存开销大约20GB，如果使用的GPU是A10，那就还有4GB的空余，完全足以容纳一个batch的输入图像和它预处理的中间产物（batch size是384，那么32位浮点数大约占220M，8位整数大约占55M）。
我们相信ToTensor操作之前就是把数据传输到显存的最佳时机。
至于预处理的顺序，尽管PyTorch的预处理顺序是人为定义的，但通常那有一个最佳顺序。我们预先把预处理的顺序定好，有利于进一步优化性能。
此外，由于验证集的预处理操作不具有随机性，它完全可以预先执行完成然后缓存下来。我们的思路是把Resize和CenterCrop预先做完，存储到硬盘中（内存较大的情况下甚至可以全部存到内存中，只需8GB，但这样通常不太划算，因为验证所占的时间比重很小）。

上述思路现在也只是一个猜想，但我们认为应该提供这样一个offload的选项，在实践中测试这样是否能带来收益。

另外，如有可能，我们希望框架能动态给出建议，自动在运行中识别当前超参数配置下的性能瓶颈。



## （十九）关于数据集自动下载

对于我们演示所需要的三个关键数据集：MNIST、CIFAR-10、ImageNet2012，我们提供MNIST和CIFAR-10的自动下载，从我们自己的官网服务器。但对于ImageNet2012，考虑到版权问题和实现难度问题，就不提供下载了，深度学习爱好者会自己想办法获取这个数据集，这不应该是框架的核心功能。



## （二十）关于对ARM和RISC-V的支持

由于我们设计开发深度学习框架的初衷之一就是方便AI芯片的研发，因此对于ARM和RISC-V架构CPU的支持必不可少。ARM以其高能效的优势，现在已经在云服务崭露头角。但是据了解，现在ARM+GPU的组合在国内各大云服务提供商尚未公开发售。考虑到运算后端的支持情况（比如oneDNN、XNNPACK等）以及开发的人力物力时间成本，我们暂时只考虑在ARM和RISC-V上实现FP32和INT8推理，**在近期暂不打算实现训练**，也不打算实现ARM+GPU组合的应用。

另外，我们默认ARM和RISC-V运行的系统是Linux。

对于ARM，我们的测试平台是树莓派5；对于RISC-V，我们的测试平台是香橙派RV2。



## （二十一）关于EMA原生支持

我们决定不原生支持。根据我们的试验和AI的建议和网上的经验，EMA给ResNet-50带来的收益本来就很小（Top-1 0.1~0.3%），而且前提通常是长训练。对于我们的演示速度为主的框架，没有必要为了追求一点精度而去实现对EMA的原生支持，浪费时间精力。



## （二十二）关于最终演示

极致速度和极致精度都用双GPU演示。以RTX 5090为主，A10为辅。

以前技术觉醒2用的是cuDNN8.9.7。

但技术觉醒3一定要用cuDNN9，而不是cuDNN8.9.7。RTX 5090 引入了新的 Tensor Core 指令集（特别是针对 FP8 和 BF16 的优化）。cuDNN 8.9.7 发布时，Blackwell 架构还不存在，它只能用通用的 CUDA Core 路径或者旧的 Tensor Core 策略来跑，效率极低。此外，cuDNN9的算法搜索更为强大。

我们注意到，PyTorch 2.9.1用的cuDNN版本已经是9.10.2.21。如果用cuDNN8.9.7写深度学习框架去跟PyTorch竞争，没有半点取胜的希望。

我们最后决定使用CUDA 13.0.0和cuDNN9.17.0。

最终演示我们采用六步走：Baseline（FP32单卡训练，100epoch）→AMP（BF16）→多GPU→超参数调整（大batch size、短epoch、高学习率、OneCycle）→预解码图像+内存映射（类似于FFCV）→Progressive Resizing。

训练我们只演示RTX 5090上的ImageNet训练（FP32/AMP）和x86 CPU上的MNIST训练。推理我们演示：NVIDIA RTX 5090 FP32/BF16/INT8、Intel Core i9-14900HX FP32/INT8（笔记本电脑单核性能王）、AMD EPYC 9965 FP32/INT8（阿里云叫EPYC 9T95，数据中心多核性能怪兽）、摩尔线程MTT S80 FP32、树莓派5 FP32/INT8、香橙派RV2 FP32/INT8、FPGA INT8。

需要说明的是，CPU上我们不演示BF16了，即便oneDNN支持这种数据类型。因为当前绝大多数消费级CPU不支持AVX-512，使用BF16会fallback成FP32来运算，是没有性能加成的，这种演示毫无意义。

列成表来看就是：

### 训练器件

|         训练器件          | FP32 | AMP  |
| :-----------------------: | :--: | :--: |
|    **NVIDIA RTX 5090**    |  √   |  √   |
| **Intel Core i9-14900HX** |  √   |      |
|     **AMD EPYC 9965**     |  √   |      |
|        **MTT S80**        |  √   |      |



### 推理器件

|         推理器件          | FP32 | INT8 | BF16 |
| :-----------------------: | :--: | :--: | :--: |
|    **NVIDIA RTX 5090**    |  √   |  √   |  √   |
|        **MTT S80**        |  √   |      |      |
| **Intel Core i9-14900HX** |  √   |  √   |      |
|     **AMD EPYC 9965**     |  √   |  √   |      |
|    **ARM（树莓派5）**     |  √   |  √   |      |
|  **RISC-V（香橙派RV2）**  |  √   |  √   |      |
|    **FPGA（FI-0420）**    |      |  √   |      |



### 其他演示

1. 在RTX 5090上演示.yaml配置的ResNet-50训练和MobileNetV2训练
2. 在RTX 5090上演示QAT
3. 在Intel Core i9-14900HX上演示PTQ
4. 在RTX 5090上演示ImageNet预训练的ResNet-50通过迁移学习识别CIFAR-10（代码和.yaml配置两种写法）



## （二十三）关于依赖项

我们把依赖项按六大场景来区分：

|      工具/库      |  最低版本   | CPU_CLOUD | GPU_CLOUD | PC_CUDA | PC_MUSA | EDGE_ARM | EDGE_RISCV |
| :---------------: | :---------: | :-------: | :-------: | :-----: | :-----: | :------: | :--------: |
|       vcpkg       |             |     √     |     √     |    √    |    √    |    √     |     √      |
|       CMake       |    3.22     |     √     |     √     |    √    |    √    |    √     |     √      |
|       Ninja       |    1.10     |     √     |     √     |    √    |    √    |    √     |     √      |
|       MSVC        | 14.44.35207 |           |           |    √    |         |          |            |
|        g++        |    13.0     |     √     |     √     |         |    √    |    √     |     √      |
|       clang       |    14.0     |           |           |         |    √    |          |            |
|    **oneDNN**     |     3.7     |     √     |     √     |    √    |    √    |          |            |
|    **XNNPACK**    | 2024-08-20  |     √     |     √     |    √    |    √    |    √     |     √      |
|       CUDA        |    13.0     |           |     √     |    √    |         |          |            |
|       cuDNN       |    9.17     |           |     √     |    √    |         |          |            |
|       NCCL        |    2.28     |           |     √     |         |         |          |            |
|       DALI        |             |           |     ⭕️     |         |         |          |            |
|    MUSA (mcc)     |     3.1     |           |           |         |    √    |          |            |
|      Python       |    3.12     |     √     |     √     |    √    |    √    |    √     |     √      |
|       Numpy       |     2.3     |     √     |     √     |    √    |    √    |    √     |     √      |
|     **zlib**      |     1.3     |     √     |     √     |    √    |    √    |    √     |     √      |
|      libcurl      |     7.0     |     √     |     √     |    √    |    √    |    √     |     √      |
| **libjpeg-turbo** |     3.1     |     √     |     √     |    √    |    √    |    √     |     √      |
|   **mimalloc**    |     2.2     |     √     |     √     |    √    |    √    |    √     |     √      |
|      **stb**      | 2024-07-29  |     √     |     √     |    √    |    √    |    √     |     √      |
|     **simd**      |     6.2     |     √     |     √     |    √    |    √    |    √     |            |

（注：“⭕️”表示非必须）

我们对场景做了一些限制，因此可以这样来判断：

1. 首先判断**CPU架构**，如果是ARM，则直接认为是**EDGE_ARM**；如果是RISC-V，则直接认为是**EDGE_RISCV**；如果是x86，则认为是**CPU_CLOUD、GPU_CLOUD、PC_CUDA或PC_MUSA**四大场景之一；如果是其他，则报错。
2. 然后问用户是否使用GPU（Y/N），如果N那就是没有GPU或即使有也不使用GPU，视为**CPU_CLOUD**场景；否则就是**PC_CUDA、PC_MUSA或GPU_CLOUD**三者之一。
3. 然后问用户的GPU是NVIDIA的（N卡）还是摩尔线程的（M卡）。N卡对应**PC_CUDA或GPU_CLOUD**二者之一，M卡对应**PC_MUSA**。
4. 然后判断**操作系统**。对于**EDGE_ARM、EDGE_RISCV和PC_MUSA**，操作系统必须是Linux，否则报错，说ARM架构、RISC-V架构和M卡只支持Linux。**CPU_CLOUD**可以对应Windows或Linux。除此之外，对于使用N卡的两种场景，如果是Windows，则视为**PC_CUDA**；如果是Linux，则视为**GPU_CLOUD**。
5. 然后，对于用户说有显卡的场景，检查GPU硬件识别情况/驱动安装情况。根据显卡种类的不同，分别执行nvidia-smi或mthreads-gmi检测，执行nvcc --version或mcc --version查看版本，确认驱动和Toolkit是否装好，确定GPU型号、数量。这里我们默认GPU数量超过1的场景只能是**GPU_CLOUD**，其他场景即使检测到多块GPU也只用第一块。
6. 然后按照上述六大场景的情况检查依赖，对于Python解释器，因为可能有多个版本以及虚拟环境，在检测到以后会问是不是使用这个，不是的话请用户输入；而其他的则是检测到就直接使用最高版本，没检测到就让用户输入。
7. 值得注意的是，对于**PC_CUDA**场景，要使用MSVC编译器，而不能使用g++，其他场景一律只使用g++。
8. 我们会用到OpenMP，但是这个已经集成到编译器里了，不需要额外安装。
9. 我们在检测完场景之后，要产生一些编译选项/宏，以便后面在CMake和C++代码中分场景处理。
10. 这里特别说明一下，截至2025年12月21日，NVIDIA的DALI和NCCL都只支持Linux平台，不支持Windows。
11. 最后需要验证所有规定了版本号的依赖项的版本。必须大于等于最低版本。但对于MSVC，如果检测到了14.44.35207版本就用14.44.35207版本；如果没有检测到，而是检测到更高的版本，就使用更高的版本；如果只检测到更低的版本，就报错。原因是，MSVC的后续版本的前向兼容可能存在问题。vcpkg安装的包可以通过`vcpkg list {package}`来查阅。需要注意的是，Windows下通常用vcpkg来安装libcurl，而Linux通常用apt，所以对于Linux来说，libcurl可能不在vcpkg里，可能要用`curl --version`来查；而对于Windows来说，libcurl可能不在系统命令里，可能要用`.\vcpkg list curl:x64-windows`来查。如果找不到相应的依赖、然后请用户输入时用户使用Enter跳过，则视为配置失败，等待用户安装（CUDA和cuDNN除外，缺少了也可以视为CPU_CLOUD场景，但是需要向用户确认是否不打算用GPU）；如果找到了相应的依赖项但是版本不符合要求，则视为配置失败，向用户告知我们的最低要求版本，并给出安装建议。配置失败不会生成任何的文件。
12. 依赖配置应该产生以下编译宏（因为是全局定义，统一加“TR_”前缀，防止CMake交叉编译或编译Python模块时的命名空间冲突）：

```cmake
### 场景宏（互斥，只能有一个为1）
TR_SCENE_CPU_CLOUD
TR_SCENE_GPU_CLOUD
TR_SCENE_PC_CUDA
TR_SCENE_PC_MUSA
TR_SCENE_EDGE_ARM
TR_SCENE_EDGE_RISCV

### 功能宏
TR_USE_CUDA          ### 是否使用CUDA/cuDNN
TR_USE_MUSA          ### 是否使用MUSA/muDNN
TR_USE_ONEDNN        ### 是否使用oneDNN（x86专用）
TR_USE_XNNPACK       ### 是否使用XNNPACK
TR_USE_DALI          ### 是否使用NVIDIA DALI（可选）
TR_USE_NCCL          ### 是否使用NCCL多卡通信
TR_USE_SIMD          ### 是否使用Simd库
TR_USE_STB           ### 是否使用STB库
TR_USE_MULTI_GPU     ### 是否启用多GPU支持
TR_NUM_GPUS          ### 检测到的GPU数量

### 边缘设备标识
TR_IS_EDGE_DEVICE    ### EDGE_ARM 或 EDGE_RISCV 时为 1
```



## （二十四）关于PC-MUSA

2025年12月18日，我们成功组装了一台带摩尔线程显卡的台式机，并且安装了操作系统，配置了环境，安装了依赖，成功运行了第一个MUSA程序。具体的配置如下：

|      CPU型号      |      Intel Core i5-14600KF      |
| :---------------: | :-----------------------------: |
|   核数与线程数    |           14核20线程            |
|       内存        |              32 GB              |
|     显卡型号      |             MTT S80             |
|       显存        |              16 GB              |
|        SSD        |             400 GB              |
|     操作系统      |        Ubuntu 22.04 LTS         |
|     内核版本      |       5.15.0-105-generic        |
|   显卡驱动版本    |             v3.0.0              |
|     SDK文件名     | musasdk_v4.0.1_Intel_Ubuntu.zip |
| MUSA Toolkits版本 |              4.0.1              |
|     muDNN版本     |              2.8.1              |
|      mcc版本      |              3.1.0              |
|      g++版本      |             13.1.0              |



具体信息：

```shell
tech-renaissance@MUSA-PC:~$ mthreads-gmi
Fri Dec 19 03:35:38 2025
---------------------------------------------------------------
    mthreads-gmi:2.0.0           Driver Version:3.0.0
---------------------------------------------------------------
ID   Name           |PCIe                |%GPU  Mem            
     Device Type    |Pcie Lane Width     |Temp  MPC Capable    
                                         |      ECC Mode       
+-------------------------------------------------------------+
0    MTT S80        |00000000:01:00.0    |0%    662MiB(16384MiB)
     Physical       |16x(16x)            |41C   YES            
                                         |      N/A            
---------------------------------------------------------------

---------------------------------------------------------------
Processes:
ID   PID       Process name                         GPU Memory
                                                         Usage
+-------------------------------------------------------------+
0    1123      /usr/lib/xorg/Xorg                       134MiB
0    1361      /usr/sbin/unity-greeter                    1MiB
---------------------------------------------------------------

```



## （二十五）关于内存池/显存池

此前最为困扰我们的一个大问题就是内存池的实现。以前的思路是根据需要实现内存池。但事实上，内存池的功能单一而明确，从需求的层面来讲，区别只有大小是否固定、是否支持多线程。其余都是算法的优劣问题。既然如此，选择性能良好的开源内存池就是一个很好的思路。经过多方论证，确实可以直接找开源实现。我们最后选定微软的**mimalloc**，因为它综合性能很强、低碎片、易于集成到C++项目，更重要的是，它支持多个平台，包括我们的RISC-V。通过vcpkg我们已经轻松完成了安装。因此，选用mimalloc，适当进行封装，我们的内存池问题就迎刃而解。

相应的，在NVIDIA GPU上我们需要显存池。有人提出显存池的实现采用NVIDIA的RMM（RAPIDS Memory Manager），遗憾的是，Windows不支持。考虑到跨平台的必要性，我们决定统一使用CUDA内置的**cudaMallocAsync**。



## （二十六）关于模型的延迟构建

我们认为，定义模型的时候，并不一定需要立刻为Tensor分配内存。因为这个时候没有任何张量计算。而且，所需的内存是不确定的。模型定义好后，在训练和测试之前必须先调用一个model.compile()方法来构建模型，执行图优化和模型大小计算和初始化。在调用这个方法之前，修改模型结构都不涉及张量的构建和张量storage内存的分配。初步认为，对模型第一次调用这个方法的时候就是执行图优化的最佳时机。调用compile()方法之前，可以add_module；执行这个函数之后就不可以了，因为模型结构和所需内存都固定了下来。或者必须重新再执行一次。

 这里的思路从“定义即运行（Eager Mode）”转向了标准的“静态图构建（Define-and-Run）”。这是实现图优化、算子融合和内存复用的前提。

**必须明确的点**：

- `compile()` **必须接收** `input_shape` 参数，因为静态图需要固定形状才能做内存规划
- 调用 `compile()` 后模型应该**锁定**，不能再 `add_module`
- 如果需要改变输入形状，必须先 `reset()` 再重新 `compile()`



## （二十七）关于图像预处理后端

Simd库在RISC-V上的安装失败，让我们重新规划了图像预处理后端。因为在EDGE_ARM和EDGE_RISCV场景，我们只做FP32和INT8推理，所涉及的图像预处理非常简单（只是图像解码、缩放、裁剪），因此“轻量级fallback”是专家的一致意见。也就是，在EDGE_RISCV场景，就只使用stb库进行预处理，不用Simd。而对于GPU_CLOUD场景，我们采用GPU offload，借助显卡加速图像预处理，需要用到DALI作为图像预处理后端。而对于其他所有场景，全都是Simd库作为图像预处理后端。不同后端实现统一接口。

专家组的总体意见是：采纳“**统一接口，后端实现**”的分层解耦架构。

1.  **架构设计**：定义一套统一的图像预处理抽象接口，上层业务代码调用此统一接口，无需关心底层实现。

2.  **平台策略**：
    *   **高性能平台 (x86/ARM/GPU)**：维持现有策略，继续使用Simd库或NVIDIA DALI以保证最高性能。
    *   **RISC-V平台**：实现一个轻量级回退（Fallback）后端。该后端利用 `stb_image_resize2.h` (单头文件库) 完成核心的缩放操作，并结合已有的 `libjpeg-turbo` 依赖进行JPEG解码。其他如裁剪、归一化等简单操作可手写实现。

3.  **构建方式**：在CMake中通过平台检测，自动为RISC-V编译并链接Fallback后端，为其他平台链接Simd/DALI后端。

此方案在不牺牲主流平台性能的前提下，以最小开发成本和零新增复杂依赖的方式，优雅地解决了RISC-V的兼容性问题，完全符合MVP原则及未来的可扩展性。

此外，stb库主要是负责缩放，但解码依然全平台统一用libjpeg-turbo，因为用 `stb_image.h` 进行解码速度极慢。`libjpeg-turbo` 在各个平台都有很好的SIMD优化，必须利用起来。据了解stb_image_resize2与libjpeg-turbo几乎可以实现无缝衔接。不过，据了解libjpeg-turbo解码出来是RGB或BGR，而Simd处理BGR、stb_image_resize2处理RGB，这个地方的具体实现可能需要注意一下。

最后再补充一点，那就是DALI属于图像预处理的极致优化，DALI本身有很多依赖，而且对硬件平台的兼容性未知，我们应该把它放在最后作为扩展来实现。在此之前，GPU_CLOUD场景下的图像预处理依然跟其他场景一样使用Simd。



## （二十八）关于MUSA后端开发

GPU训练是本框架的一个重点任务，而摩尔线程GPU的支持也是我们的一块大蛋糕。但是，由于工具集的不完善，MUSA上的程序开发几乎就是一个未知区域。可能碰到的难题和可能需要花费的时间完全未知。比如，关于我们此前讨论的显存池，我们就发现MUSA根本没有提供显存池实现。此外，社区的不完善和编程资源的缺乏，也会让我们的人工编程和AI编程都陷入困境。

但是我们提出的一个重要思路是：使用摩尔线程官方给的**Musify工具**。这个工具据说可以实现CUDA代码到MUSA代码的直接转换，使用简单。虽然不一定能够实现100%正确转换，MUSA提供的功能也显然不如CUDA多，但是这可以帮我们解决基本代码写不出来或者调不通的问题。我们可以先写出CUDA代码实现某个功能，用Musify转化为MUSA代码，然后再借助AI解读它是如何实现这个功能的，这样就可以逐步学到撰写MUSA代码实现功能的方法。

举例来说，MUSA没有提供显存池，但我们知道实现显存池的大致原理。我们可以先用CUDA手写显存池，先在NVIDIA GPU上运行、调优，然后转化为MUSA代码，在摩尔线程GPU上跑通之后，再看能否继续优化。

积累了足够多的样例之后，我们可以找到规律，也可以优中选优，开发会越来越顺畅。



## （二十九）关于BF16类

我们的观点是，不要在C++引入单独的BF16类（即不将其封装为一个包含完整运算符重载的 C++ 类，而是采用 `enum` + 工具函数的方式）。

这一决策是“技术债务最小”的方案，其背后的深层原因非常具有说服力：

### 1. 避免与未来 C++ 标准或第三方库冲突（核心理由）

文档中明确提到“避免未来冲突”。这是一个非常有远见的考虑。

- **现状**：C++23 标准已经引入了 `<stdfloat>`，很多编译器和硬件厂商（如 NVIDIA 的 `__nv_bfloat16`）也有自己的实现。
- **风险**：如果我们现在实现了一个名为 `class BF16` 的类，未来在整合这些原生支持时，会出现命名空间冲突、隐式类型转换歧义等棘手问题。
- **优势**：使用 `enum class DType` 仅仅作为一个标签，实际数据存储用 `uint16_t` 或 `void*`，计算时转换为 `float`，这样可以完美避开类型系统的“打架”。

### 2. 降低编译开销与代码复杂度

深度学习框架往往大量使用模板。如果 BF16 是一个复杂的类：

- **编译膨胀**：每个算子都需要针对 `BF16` 类进行模板实例化，会显著增加编译时间及二进制体积。
- **样板代码**：为了让 `BF16` 类像原生类型一样工作，你需要为它实现 `+`, `-`, `*`, `/`, `+=`, `==`, `!=` 以及与 `float`, `int` 混合运算的所有重载。这是巨大的开发量且容易出错。
- **方案优势**：文档采用的“内联转换工具函数”方案，仅需实现 `fp32_to_bf16` 和 `bf16_to_fp32` 两个核心函数，逻辑极简，且由编译器优化，性能无损。

### 3. 符合 AI 框架的“批量处理”特性

在深度学习框架中，我们几乎从不单独操作一个标量（Scalar）的 BF16 数值。

- **实际场景**：我们操作的是 `Tensor`，是成千上万个数据的集合。算子（Operator）内部通常利用 SIMD 指令（如 AVX-512 或 Tensor Cores）批量处理数据。
- **无用功**：在 C++ 层面封装一个完美的单个 `BF16` 对象，对于底层的向量化计算几乎没有帮助，反而可能因为对象的构造/析构引入微小的运行时开销（如果不小心的话）。

### 4. 灵活性与跨平台适配

- **硬件差异**：并非所有硬件都原生支持 BF16 计算。
- **软模拟**：在不支持 BF16 的 CPU 上，标准的做法就是：`Load BF16 -> Convert to FP32 -> Compute -> Convert back to BF16 -> Store`。
- 您的设计正好符合这种流程。如果不引入独立类，我们可以更自由地在不同后端（Backend）上决定如何解释这块内存，而不需要被一个 C++ 类的行为束缚。

### 总结

**不引入独立 BF16 类**是典型的“奥卡姆剃刀”原则体现——**如无必要，勿增实体**。对于一个追求“轻量级、可重构”的框架（Tech-Renaissance），这是目前性价比最高的选择。



## （三十）关于MUSA上支持的运算

经过我们的调研和实践，我们发现MUSA上的运算支持远没有我们预期的完善。muDNN所支持的运算有限，很多需要手写核函数。我们在试验BF16的简单加法运算时，多次尝试仍未能达到理想的性能。鉴于MUSA不是我们重点演示的平台，出于人力物力开发效率考虑，我们决定**在MUSA上完全不支持BF16、INT32和INT8运算**，只支持FP32。不过，MUSA上依然要支持完整的ResNet-50的FP32单卡训练和推理。



## （三十一）关于内存对齐

虽然前面提到过多种内存对齐方案，但我们的最终实现是：**256字节（256B）**。关于对齐的做法，所有AI的一致意见是：**保持256B，不要改为64B**。



## （三十二）关于器件类的指针/引用

器件管理器类应该返回器件的指针还是引用？全部专家一致认为应该使用**引用**。在我们的框架内几乎不会有任何风险，而且还有显著收益。



## （三十三）关于学习率调度器

对于余弦退火，有必要同时提供epoch-wise和step-wise两种选项。



## （三十四）关于BN层

即便使用AMP，BN层的参数也必须是FP32。



# 【十一、开发日志（实时更新）】

## （一）技术觉醒1

- 2025年10月23日18:17:30，成功建立项目文件夹tech_renaissance
- 2025年10月23日18:17:48，成功在项目文件夹下建立third_party文件夹，并安装protobuf
- 2025年10月23日00:21:37，成功安装NVIDIA GPU驱动、CUDA、cuDNN、Visual Studio、CLion、Git
- 2025年10月24日00:22:24，基本实现异常类和日志器类，完成GitHub首次推送
- 2025年10月24日00:41:00，优化 `Logger` 与 `TRException` 的代码结构与注释规范，完善异常与日志安全机制
- 2025年10月24日00:45:30，修正日志等级比较逻辑、改进跨平台时间戳、安全文件写入
- 2025年10月24日00:47:00，完成异常类和日志器类四个文件优化，符合开发准则
- 2025年10月24日01:20:04，安装了Doxygen（1.14.0）和Graphviz（14.0.2）
- 2025年10月24日02:40:10，修复字符编码问题 - 解决了用户提示的(936)个编码错误
  - 在CMakeLists.txt中添加了/utf-8编译选项
  - 成功解决了UTF-8文件被GBK解析的问题
- 解决CMake链接问题 - 修复了export.lib链接错误
  - 问题原因：CMake INTERFACE库配置导致自动推导错误的库名
  - 解决方案：使用$<TARGET_OBJECTS:>生成器表达式和正确的库链接方式
- 完成后端系统编译和测试 - 所有核心功能验证通过
  - 后端库（backend.lib）编译成功
  - 数据库（data.lib）编译成功
  - 导出库（tech_renaissance_export.lib）编译成功
  - 所有基础测试通过（logger、exception、simple、backend_fixed）
- 解决了问题“CUDA工具集检测失败”。专家诊断：使用MinGW的CMake而不是MSVC的CMake。解决方案：使用正确的CMake路径 T:/Softwares/CMake/bin/cmake.exe
- 解决了问题“CMake INTERFACE库链接问题”。专家诊断：90%是CMake在展开INTERFACE库时把tech_renaissance_export截断成了export.lib。根本原因：在src/backend/CMakeLists.txt第45行使用了裸字符串export而不是目标名tech_renaissance_export
- 字符编码问题修复：通过在 CMakeLists.txt 中添加 /utf-8 编译选项，解决了 936 个 UTF-8 编码错误
- CMake 链接问题修复：修正了 INTERFACE 库目标名称从 export 到 tech_renaissance_export，解决了链接器找不到 export.lib 的问题
- CUDA 环境配置简化：移除了复杂的环境变量要求，现在使用标准的 PATH 和 LD_LIBRARY_PATH 进行 CUDA 库检测
- 2025年10月24日03:39:34，修复了CUDA、CUDNN、CUBLAS路径问题
  - 现在用户只需要按照标准流程安装CUDA和cuDNN：
    1. 安装CUDA Toolkit - 自动添加到PATH
    2. 安装cuDNN - 可选择性地添加到PATH/LD_LIBRARY_PATH
    3. 运行项目 - 无需任何额外配置！
- test_cuda_backend.cpp 编译错误修复：用工作正常的测试内容替换了有问题的 CUDA 测试代码
- 2025年10月24日04:59:42，解决了cuDNN动态链接库的问题，现已确认CUDA、cuDNN、cuBLAS全部可用
- 2025年10月24日19:51:40，成功在CUDA后端实现3×3卷积运算原型（unit_tests/test_cuda_conv.cpp）
- 2025年10月24日23:21:32，成功在CUDA后端实现矩阵乘法运算原型（unit_tests/test_cuda_gemm.cpp）
- 2025年10月25日03:21:27，成功在CPU后端用Eigen实现了3×3卷积运算原型（unit_tests/test_cpu_conv.cpp）
- 2025年10月25日04:57:57，成功在CPU后端用Eigen实现了矩阵乘法运算原型
- 2025年10月25日08:11:30，成功实现了Shape类
- 2025年10月26日02:17:10，成功实现了Tensor类
- 2025年10月26日05:39:01，成功实现了四大后端类（后端管理器、后端基类、CPU后端类、CUDA后端类）
- 2025年10月26日06:32:40，成功上传GitHub并发布预览版（V1.11.05）
- 2025年10月26日07:50:58，完成PyBind11安装
- 2025年10月26日17:24:17，完成优化目录结构
- 2025年10月26日21:53:19，完成数据类和后端类的优化，实现后端管理器的Meyers单例，优化 `CudaBackend::fill` 的性能
- 2025年10月27日04:17:38，实现了tech_renaissance框架与PyTorch的实时通信，实现了后台调用任意Python脚本，优化了4维张量打印效果，与pytorch对齐
- 2025年10月27日05:24:15，修复了与PyTorch实时通信的bug
- 2025年10月28日23:00:00，实现TSR（Tech Renaissance）文件格式规范，完成张量导入导出功能（V1.18.02）
  - 定义64字节标准头部结构，支持FP32和INT8数据类型
  - 实现CPU后端export_tensor/import_tensor接口
  - 提供便捷宏EXPORT_TENSOR/IMPORT_TENSOR
  - 完整支持0-4维张量的跨平台数据交换
- 2025年10月28日23:00:00，优化PyTorchSession通信机制，统一工作区目录管理（V1.18.02）
  - 实现WORKSPACE_PATH宏，统一临时文件路径管理
  - CPU后端注册时自动创建workspace目录
  - 修复Python脚本路径计算问题，确保跨进程通信一致性
  - 完善会话ID生成和清理机制
- 2025年10月28日23:00:00，完善CPU后端数据访问接口，增强张量操作便利性（V1.18.02）
  - 新增get_scalar_float、get_scalar_int32、get_scalar_int8数据访问接口
  - 支持从张量中安全提取标量值
  - 完善异常处理机制，提供类型安全的数据访问
- 2025年10月28日23:00:00，更新项目文档体系，确保与实际实现同步（V1.18.02）
  - 更新所有API文档至V1.18.01版本
  - 创建PyTorch通信机制详细文档
  - 修正TSR格式规范文档，完善实现指南
  - 优化Tensor API文档，反映当前架构设计
  - **全面更新核心设计文档tech_renaissance_prompt.md**
- 2025年10月28日23:00:00，增强测试覆盖率，完善跨进程功能验证（V1.18.02）
  - 完善PyTorchSession测试，验证跨进程通信稳定性
  - 增加TSR文件格式测试，覆盖所有维度和数据类型组合
  - 优化Python脚本路径处理，支持多平台部署
  - 完善设备兼容性测试，确保CPU/CUDA后端稳定运行
- 2025年10月28日23:00:00，优化内存管理，实现64字节内存对齐（V1.18.02）
  - CPU后端内存分配采用64字节对齐，优化SIMD性能
  - 完善RAII内存管理机制，确保资源安全释放
  - 增强Storage类的shared_ptr管理，支持多张量安全共享
- 2025年10月28日23:00:00，完善项目配置和构建系统（V1.18.02）
  - 优化CMake配置，简化依赖管理
  - 移除对spdlog和protobuf的依赖，坚持轻量级设计
  - 完善跨平台编译支持，确保Windows/Linux兼容性
- 2025年10月29日09:41:30，优化与PyTorch的通信机制
- 2025年10月30日22:23:03，在GitHub和Gitee上分别创建了AI编译器和深度学习加速器的仓库，并更新了自研深度学习框架的仓库路径
- 2025年10月31日00:43:17，V1.24.2，修复了Logger类初始化的重大bug，并更新了PythonSession类的功能和文档
- 2025年10月31日13:41:53，完成了CPU后端单目运算符的朴素实现
- 2025年10月31日16:21:03，完成了CPU后端单目运算符的Eigen实现
- 2025年10月31日20:49:07，完成了CUDA后端的is_close方法的实现
- 2025年10月31日20:49:20，完成了CPU后端和CUDA后端的copy、copy_into方法的实现
- 2025年10月31日21:42:16，完成了CPU后端的转置的实现
- 2025年11月2日18:38:38，实现了对INT32的支持
- 2025年11月3日01:31:20，完成了张量的内存测试，提供了本框架推荐的null_tensor()方法来让用户按需销毁大张量
- 2025年11月4日06:35:53，完成了CUDA后端类的FP32矩阵乘法、卷积、转置卷积、1×1卷积的优化，性能均超越PyTorch
- 2025年11月16日16:41:20，CPU后端类实现了交叉熵损失函数，并与PyTorch结果对齐
- 2025年11月16日16:41:58，后端基类改为可实例化，但其所有方法均默认报NotImplementedError，待后端派生类改写
- 2025年11月17日19:38:37，实现了Module类（以及Linear、Tanh、Flatten子类）、Model类和Scheduler类（以及6个学习率调度器子类）
- 2025年11月19日17:52:42，实现了以下优化：
  - Linear层权重转置缓存：3.75倍性能提升
  - Model零拷贝前向传播：7.5倍logits访问提升
  - 智能参数缓存机制：8倍参数访问提升
  - StateManager架构：统一优化器状态管理
- 2025年11月22日00:30:00，实现了Trainer类，完成了首个集成测试：CPU后端上训练MLP

## （二）技术觉醒2

- 2025年11月23日12:10:38，实现了跨平台（在Ubuntu 24.04上测试通过），实现了依赖自动配置功能（configure.py）
- 2025年11月24日07:07:11，修复Adam和AdamW重大bug，完成MLP性能测试，取得了出色的成绩
- 2025年11月25日00:02:08，实现了Task类，实现了极简API（27行代码完成完整演示）

## （三）技术觉醒3

- 2025年12月13日13:52:01，自研深度学习框架英文名称改为：**renAIssance**（以前是tech-renaissance）。中文名称依然是**“技术觉醒”**
- 2025年12月13日16:54:33，在Gitee上创建私有仓库（https://gitee.com/tech-renaissance/renaissance），标志着技术觉醒3的正式开始开发！
- 2025年12月13日23:59:59，完成了技术觉醒3的新版设计文档，明确了与技术觉醒2的不同
- 2025年12月16日16:54:23，完成了AMD CPU服务器、Intel Core i3笔记本、GPU服务器、树莓派5的环境配置和备份
- 2025年12月18日01:49:14，在PyTorch上实现了单卡、双卡速度最优的ResNet-50训练，可用于对标
- 2025年12月18日02:04:56，完成了GPU选型调研，最后的演示将用两个RTX 5090
- 2025年12月18日21:11:53，成功配置MUSA主机，安装所有环境（包括摩尔线程显卡驱动、MUSA Toolkit、muDNN），成功编译运行第一个MUSA程序
- 2025年12月19日23:41:19，通过VS Code + Claude成功实现远程AI开发
- 2025年12月19日23:42:00，编写脚本实现了项目的一键备份和一键远程同步
- 2025年12月21日05:07:12，实现了依赖项的自动配置，覆盖PC-MUSA、PC-CUDA、CPU云、GPU云、嵌入式五大场景，横跨Windows和Linux两大系统，兼顾x86、ARM、RISC-V三种CPU架构。但是，只剩一个依赖没装好，就是RISC-V上的Simd库。
- 2025年12月21日18:39:39，提出了model.compile()的思路，得到了众多专家的支持，这是静态图优化的关键转折点。模型延迟构建、内存池实现、图像预处理后端的决策都得到了专家的高度认可，合称技术觉醒3的**“三大决策”**。
- 2025年12月22日04:18:32，全平台依赖安装完毕、全平台自动配置实现！
- 2025年12月23日03:07:45，实现了一键编译脚本和HELLO CUDA、HELLO MUSA、HELLO WORLD、以及libcurl、mimalloc、libjpeg-turbo、zlib的测试样例在支持的平台上全部测试通过！
- 2025年12月23日23:26:25，simd、stb、onednn、xnnpack、nccl的测试样例在支持的平台上全部测试通过！
- 2025年12月24日05:40:42，V3.5.5，实现了Logger类（日志器类）和TRException类（异常类），实现了项目从Window平台到其他所有测试平台的一键分发，实现了全平台一键自动重新配置、一键自动重新编译，实现了测试样例的编译开关，大幅提高测试效率
- 2025年12月24日19:49:10，确定了内存池、器件类、数据类的设计方案。
- 2025年12月25日00:50:00，实现了DType、DeviceType、Shape类。
- 2025年12月25日20:08:49，安装好了CUDA转MUSA工具musify。
- 2025年12月25日20:52:11，实现了所有的内存池和显存池（包括CUDA显存池和MUSA显存池），全平台测试通过。
- 2025年12月26日00:05:06，实现了Storage类和Tensor类，全平台测试通过。
- 2025年12月26日02:39:41，实现了Device类、DeviceManager类和CpuDevice类，全平台测试通过。
- 2025年12月26日03:43:01，修复了全局宏的bug。
- 2025年12月26日10:50:00，修复了链接bug，大幅优化CMakeLists.txt的链接逻辑。
- 2025年12月26日13:00:00，实现了CudaDevice和MusaDevice类。
- 2025年12月27日02:53:00，V3.6.7，根据专家意见进行了修复和优化，全平台测试通过。更新了全部文档。这个版本确保我们的框架拥有了最稳健的底层。
- 2025年12月27日09:45:00，V3.6.8，CPU、CUDA、MUSA上实现了具备可复现性的伪随机数发生器和随机张量创建方法。
- 2025年12月27日19:41:00，V3.6.9，优化打印效果，实现了张量的TSR V3格式导入导出，提供了np.ndarray和torch.tensor类型的TSR V3格式导入导出脚本。
- 2025年12月27日21:06:00，V3.6.10，实现了张量相等的判断。
- 2025年12月27日23:59:12，V3.6.11，实现了PythonSession类。
- 2025年12月28日02:29:00，V3.6.12，实现了Downloader类。
- 2025年12月29日00:35:11，提出了《最终方案》，大幅强化框架的架构，有助于大幅提升框架在ResNet-50训练时的性能。方案得到专家一致高度评价。
- 2025年12月29日02:50:23，在PyTorch上进一步优化了ResNet-50的演示参数方案，将有利于演示快速训练（最新成果是100个epoch内取得77.99%的验证准确率，每个epoch平均用时仅288s）。
- 2025年12月30日23:32:22，完成了有关MLPerf的调研，提出了30个epoch内演示75.9%验证准确率的最终演示方案。
- 2025年12月31日14:46:51，在PyTorch上进一步优化了ResNet-50的双卡快速训练方案，在不使用复杂图像预处理的情况下，28个epoch即可达到75.9%的Top-1验证准确率。
- 2026年1月1日06:42:55，在PyTorch上实现了ResNet-50的八卡快速训练方案，29个epoch即可达到75.9%的Top-1验证准确率。这将是我们最终演示中采用的算法方案。
- 2026年1月1日14:20:03，V3.6.13，张量打印效果对齐pytorch。
- 2026年1月1日21:55:51，V3.6.14，优化了Logger类和异常类，完成了整个项目代码的风格迁移。
- 2026年1月1日23:38:22，V3.6.15，优化了所有device类的张量创建，实现了CPU、CUDA、MUSA上的张量复制方法。
- 2026年1月2日01:22:02，V3.6.16，实现了Host-to-Device和Device-to-Host同步传输。
- 2026年1月2日08:24:11，V3.6.17，实现了All-Reduce和Parameter Broadcast。

