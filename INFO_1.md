# 【十三、探讨：数据集.dts文件高速读取方案及DataLoader类的实现】

**（注：本部分内容介绍的.dts文件正式实现的情况，如果与前文冲突，应以本部分为准。）**

## **（一）基本情况**

各位专家。我们已经通过Python实现了ImageNet、MNIST、CIFAR-10、CIFAR-100这四个常用数据集的自动下载、解压、重排、整合，最终输出我们的自研深度学习框架适配的数据集.dts二进制文件。

下面简单介绍情况。

开源数据集的格式不统一，并且文件可能是分散的，比如ImageNet解压后就是分散的JPEG。这样不利于高速读取。我们之前有过多次测试和探讨，数据加载就是ResNet-50训练的最大瓶颈之一。我们的服务器的云盘，排除掉cache的影响的话，在之前的测试中只有1000MB/s的读取速度。单文件的顺序读显然比分散文件的随机读要快很多，我们需要把数据整合起来，这就是最基本的动机。ImageNet是JPEG格式，我们研究认为应该保持它为JPEG进行整合，而不是整合为解码后的像素值，因为那样体积会大很多，读取速度只会更慢。借助多线程的libjpeg-turbo解码，读取后再从JPEG中解码信息将不是一个大问题。事实上我们参考了FFCV的方案，他们默认也是存储解码前的数据。我们的方案已经是高度优化过的，主要是体现在对齐方式、压缩方式的方面。

我们的每个数据集都是整合成两个文件：训练集和测试集。

需要说明的是，我们的演示是采用ImageNet，所以会优先实现ImageNet的读取。而其他几个数据集，存储的是RAW的像素字节流，与ImageNet的JPEG有很大不同。其他几个数据集也不需要分块存储。所以下面我们重点探讨ImageNet。

我们的ImageNet采用LV0~LV3这四种不同的压缩级别。其中LV0就是完全无压缩，直接读取原JPEG字节流，然后重排、对齐后保存。LV1是仅缩放，LV2是缩放+裁剪，LV3是缩放+裁剪+降低JPEG质量。据统计，ImageNet的图像宽高中位数就是500×375，而我们设定的阈值是短边400，这可以很好地保护绝大部分图像的数据，仅适当压缩了小部分超规图片。

我们采用分块存放的方式，ImageNet每个块（BLOCK）的大小为**16MB**。这样设定的原因是，ImageNet的原图最大是15MB，我们要避免一个图跨两个BLOCK的情况，这样才能真正实现BLOCK与BLOCK之间的解耦，实现随机、并行的高速读取。每个BLOCK包含必要的元信息，就是BLOCK魔数、该BLOCK的图片数、BLOCK编号、每张图片的偏移量、每张图片的有效字节数、每张图片的标签。LV0的BLOCK元信息大小是4KB，而LV1~LV3的是16KB。这样设定的原因是，LV1~LV3每个BLOCK能容纳更多的图片。我们的每张图片还采取了**64字节**的对齐。

由于ImageNet原图的大小参差不齐，限定了BLOCK大小16MB、图片对齐64B之后，就涉及到一个浪费空间、文件体积增大的问题。这里我们设计了一个**贪心填充算法**，往每个16MB的BLOCK空间里填充大小合适的图片，尽量确保最大限度地利用存储空间、压缩文件体积。最后效果出人意料的好，就测试集而言，L0V压缩后，即便添加了图片对齐、BLOCK对齐、元信息、文件头，文件总体积也**只增大了0.13%**。而对于LV1、LV2、LV3，压缩率分别是**47%、47%、30%**。这样一来，我们就很好地控制了文件的大小，然后在此基础上将可以实现数倍的读取加速。

我们很荣幸地说，我们的导出是成功的，数据在文件中的排列位置已经得到了验证。

顺带说一句，我们的ImageNet的.dts文件的文件头也是16MB，为的是BLOCK对齐。但文件头只有前144个字节是有用的，所以我们打开一个.dts文件时，首先应该是读取那前144个字节，了解文件基本情况。

在那之后，我们就可以随机读取之后的每一个16MB的BLOCK。毫无疑问我们的文件大小就是16MB的整数倍。

我们预想是有两种加载模式，一种是全量加载，一种是部分加载。全量加载的话，只要内存足够，我们就申请一块超大内存，把整个数据集一次性读取进去，之后的每个epoch就不再需要访问硬盘了。而部分加载，我们就需要在内存里准备若干个BLOCK的空间，然后，反复利用。我们的设想是，部分加载的时候，内存采用循环队列，队列里是若干个16MB的空间，每个刚好放一个BLOCK。DataLoader加载BLOCK数据，从前往后填充，而后面的负责预处理的Preprocessor模块也是从前往后读取。在循环队列写满后，DataLoader又再从头覆盖写入。这样就大幅节省了内存空间。至于这个循环队列所占内存应该有多大，我们的初步想法是按DataLoader的workers数来计算。workers就是并行读取硬盘BLOCK数据的线程，它们的工作是互不干扰的，每个线程被指定了应该读文件的哪一个BLOCK，也被指定了应该读取到内存的哪个位置。假如workers数是N，那么理论最小需要的空间应该是2N×16MB，相当于双缓冲。为了让读取更顺畅，避免读写的等待，我们的方案是默认4N×16MB。更大似乎也没有意义。这样一来，即使我们开16个线程来读取，所需的空间也只是1GB而已，即使是嵌入式系统也可以提供这样的空间。

但是，考虑到一些用户可能不想使用.dts文件，或者在公平的测试对比中我们不允许使用.dts，所以我们还是得提供原始数据集的读取功能。我们的初步想法是，把原始数据集的读取功能放在一个DataLoader类，而对于.dts文件的高速读取功能，则放在它的子类，叫FastDataLoader类。我没太想好这两个类要怎么设计，但是，毫无疑问的是，DataLoader类肯定要负责维护一块可复用的大内存，来容纳读取的数据。然后，它需要暴露每个样本的偏移量、size、标签信息。并且很重要一点，就是它要能够标明哪些位置是已经写入完成可以被读取的，而哪些位置是已经被读取完可以写入的。最后一点就是随机性的实现。为了避免固定顺序导致过拟合，训练集样本排列的随机性是必须的。验证集我们默认不随机。对于训练集，我们其实引入了“三重随机”。第一重，是我们在制作.dts文件时，已经进行过shuffle；第二重，是我们从.dts文件读取BLOCK时，可以随机跳到任意一个没读取过的BLOCK；第三重，则是数据已经加载到内存里之后，Preprocessor的worker可以随机地选择未被处理过的样本来处理。有了这三重随机，随机性显然是非常足够了，理论上任何一个样本都可以出现在任何一个batch。而且按照我们的预估，这三重随机的成本极其低廉：第一重随机是已经实现了的，不影响运行时的耗时；第二重随机是粗粒度，只需要重排几百个BLOCK的顺序，而且对于第一个epoch甚至可以不随机；第三重随机是窗口，其实是对某一个范围内的样本进行随机选择——三重随机的成本加起来可能依然小于直接对1,281,167个训练集样本随机重排！

我们在这里讨论，最主要就是讨论类的设计、最快读取方式、多线程的实现、以及如何减少锁竞争。现阶段我们的方案就是决定如何设计实现DataLoader/FastDataLoader，让它们在实现最快速度读取的基础上，给后续的Preprocessor暴露合适的API。我们初步预想所要达到的效果就是，假如有P个Preprocessor的workers要访问DataLoader（规定P在1~64之间，并且是2的幂），DataLoader能给它们提供抽象，第i个Preprocess worker能够直接从DataLoader中拿到样本的偏移量、有效字节数、标签这三大信息，不重复也不遗漏，如果还没有加载完成，就暂时等待。

这里再次强调，DataLoader一定要注重多线程，因为它既要多线程读取不同BLOCK，又要实现循环队列，又要提供接口给后面的多线程预处理的Preprocess workers。DataLoader显然必须是全局单例类。dataloader worker在获取可写入的内存的指针的时候、preprocess worker在获取三大信息的时候，都是有可能引起锁竞争的时候。反倒是在决定读取哪个BLOCK的时候几乎可以完全避免锁竞争——因为这个可以直接预先分配好谁读取哪一块。

上面都是背景情况。这里我们进一步明确我们的需求：**我们需要的是可用于读取原始ImageNet数据集（train和val两个目录，下面各有1000个子目录，里面都是JPEG，子目录的名称代表不同类别）的DataLoader类和用于读取imagenet_train.dts、imagenet_val.dts的FastDataLoader类（需要支持LV0~LV3）。要求一定一定都要支持多线程读取（我们限定加载线程数在1~16之间，并且必须是2的幂），一定一定要为后续的多线程读取的Preprocess workers提供安全的、高效的API。**

注意，我们的数据集导出是使用的python（为了方便运用各种工具，也为了公开透明地展示我们对数据集做了什么），但我们的框架依然是C++，运行时也是C++读取数据集。

在读取的速度方面，我们是做过实验的，16线程可以达到10GB/s~16GB/s的超高读取速度，理论上加载完成我们的LV3压缩的ImageNet数据集**只需要不到5秒**！这是一个非常诱人的愿景，但是实现起来能不能达到这个效果又是一个问题。

## **（二）补充信息：跨平台高速读取实验的代码及经验总结**

我们的项目要求能在Windows、Linux两大操作系统下运行，并且要求能用在PC、云端、嵌入式等场景，因此跨平台很重要。我们在Windows、Linux两大操作系统都做过实验，以下是我们试出来的最快速度读取的代码，仅供参考。

```c++
// 跨平台最优版本: 多线程pread/ReadFile (原生API)
// 支持1/2/4/8/16线程，使用4MB缓冲区
// 核心技术：
// - 每线程独立文件句柄/描述符
// - 4MB缓冲区适配CPU L3缓存
// - 移除系统提示，让OS自动管理I/O策略
// - Windows: VirtualAlloc大页内存
// - Linux: posix_memalign对齐内存

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <string>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
#endif

class NativeLoader {
private:
    std::string filename;
    size_t file_size;
    char* buffer;

    // 4MB缓冲区 - 适配CPU L3缓存，最佳性能
    static constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;
    static constexpr size_t ALIGNMENT = 4096;

public:
    NativeLoader(const std::string& fname) : filename(fname), buffer(nullptr) {
        file_size = getFileSize();
    }

    ~NativeLoader() {
        if (buffer) {
#ifdef _WIN32
            VirtualFree(buffer, 0, MEM_RELEASE);
#else
            free(buffer);
#endif
        }
    }

    size_t getFileSize() {
#ifdef _WIN32
        HANDLE hFile = CreateFileA(filename.c_str(), GENERIC_READ,
                                    FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("无法打开文件获取大小");
        }
        LARGE_INTEGER size;
        GetFileSizeEx(hFile, &size);
        CloseHandle(hFile);
        return static_cast<size_t>(size.QuadPart);
#else
        struct stat st;
        if (stat(filename.c_str(), &st) != 0) {
            throw std::runtime_error("无法获取文件大小");
        }
        return st.st_size;
#endif
    }

    double load(int num_threads) {
        // 分配对齐内存
#ifdef _WIN32
        // Windows: 使用VirtualAlloc分配大页内存
        buffer = (char*)VirtualAlloc(nullptr, file_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!buffer) {
            throw std::runtime_error("内存分配失败");
        }
#else
        // Linux: 使用posix_memalign分配对齐内存
        if (posix_memalign((void**)&buffer, ALIGNMENT, file_size) != 0) {
            throw std::runtime_error("内存分配失败");
        }
#endif

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        size_t chunk_size = file_size / num_threads;

        for (int i = 0; i < num_threads; ++i) {
            size_t start_offset = i * chunk_size;
            size_t end_offset = (i == num_threads - 1) ? file_size : (i + 1) * chunk_size;
            size_t size_to_read = end_offset - start_offset;

#ifdef _WIN32
            // Windows实现：使用原生ReadFile API，无系统提示标志
            // 移除FILE_FLAG_SEQUENTIAL_SCAN让Windows智能预读算法自动优化
            threads.emplace_back([this, start_offset, size_to_read]() {
                HANDLE hFile = CreateFileA(filename.c_str(), GENERIC_READ,
                                            FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                            FILE_ATTRIBUTE_NORMAL,  // 使用默认属性，无特殊提示
                                            NULL);
                if (hFile == INVALID_HANDLE_VALUE) {
                    return;
                }

                LARGE_INTEGER offset;
                offset.QuadPart = start_offset;
                SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);

                size_t remaining = size_to_read;
                char* dst = buffer + start_offset;

                while (remaining > 0) {
                    DWORD to_read = static_cast<DWORD>((remaining < BUFFER_SIZE) ? remaining : BUFFER_SIZE);
                    DWORD bytes_read = 0;

                    if (!ReadFile(hFile, dst, to_read, &bytes_read, NULL)) {
                        break;
                    }

                    dst += bytes_read;
                    remaining -= bytes_read;

                    if (bytes_read == 0) break;
                }

                CloseHandle(hFile);
            });
#else
            // Linux实现：使用pread系统调用，无系统提示
            // 移除posix_fadvise让Linux内核自动管理预读策略
            threads.emplace_back([this, start_offset, size_to_read]() {
                int fd = open(filename.c_str(), O_RDONLY);
                if (fd < 0) {
                    return;
                }

                // 不使用posix_fadvise - 让内核自动优化

                size_t remaining = size_to_read;
                char* dst = buffer + start_offset;
                size_t offset = start_offset;

                while (remaining > 0) {
                    size_t to_read = (remaining < BUFFER_SIZE) ? remaining : BUFFER_SIZE;
                    ssize_t bytes_read = pread(fd, dst, to_read, offset);

                    if (bytes_read <= 0) break;

                    dst += bytes_read;
                    offset += bytes_read;
                    remaining -= bytes_read;
                }

                close(fd);
            });
#endif
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    size_t getSize() const { return file_size; }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filepath> <threads>" << std::endl;
        std::cerr << "Supported thread counts: 1, 2, 4, 8, 16" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int numThreads = std::stoi(argv[2]);

    // 支持1/2/4/8/16线程
    if (numThreads != 1 && numThreads != 2 && numThreads != 4 &&
        numThreads != 8 && numThreads != 16) {
        std::cerr << "Error: Thread count must be 1, 2, 4, 8, or 16" << std::endl;
        return 1;
    }

    try {
        NativeLoader loader(filename);

        std::cout << "========================================" << std::endl;
        std::cout << "Method 2 - Native API (Cross-Platform)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "File: " << filename << std::endl;
        std::cout << "Size: " << loader.getSize() << " bytes ("
                  << std::fixed << std::setprecision(2)
                  << (loader.getSize() / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
        std::cout << "Threads: " << numThreads << std::endl;
        std::cout << "Buffer: 4MB (Optimized for CPU L3 Cache)" << std::endl;
        std::cout << "System Hints: None (OS Auto-Optimized)" << std::endl;
#ifdef _WIN32
        std::cout << "Platform: Windows (ReadFile API)" << std::endl;
#else
        std::cout << "Platform: Linux (pread API)" << std::endl;
#endif
        std::cout << "========================================" << std::endl;

        double time_ms = loader.load(numThreads);

        std::cout << "Load time: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
        std::cout << "Speed: " << std::fixed << std::setprecision(2)
                  << (loader.getSize() / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0)
                  << " GB/s" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

```

以下是我们团队的测试经验总结：

### 📌 核心结论

经过大量实验验证，我们找到了**跨平台最快的大文件加载方案**：

- **Windows**: 17.1 GB/s (16线程) / 16.1 GB/s (8线程)
- **Linux**: 11.0 GB/s (8线程)
- **提升**: 比标准fstream快2-3倍

**代码文件**: `method2_native_unified.cpp`

---

### 🎯 优化思路（关键发现）

### 1. 反直觉的发现：简化即正义

**常见误区**：更大的缓冲区 + 更多的系统提示 = 更快
**实验结果**：❌ 错误！

| 配置         | 缓冲区  | 系统提示                                   | 实测速度      | 结论          |
| ------------ | ------- | ------------------------------------------ | ------------- | ------------- |
| 过度优化版   | 16MB    | FILE_FLAG_SEQUENTIAL_SCAN<br>posix_fadvise | 3.04 GB/s     | ❌ 慢          |
| **最优版本** | **4MB** | **无**                                     | **16.1 GB/s** | ✅ **快4.4倍** |

**核心洞察**：

- 现代OS的I/O调度器已经很智能
- 手动干预往往适得其反
- **让系统自动管理优于强制提示**

---

### 2. 为什么4MB缓冲区最优？

#### CPU缓存层次结构

```
L1缓存:  32-64 KB    (1-2 ns延迟)  ⚡⚡⚡⚡⚡
L2缓存:  256-512 KB  (3-5 ns延迟)  ⚡⚡⚡⚡
L3缓存:  8-32 MB     (10-15 ns延迟) ⚡⚡⚡
主内存:  8-64 GB     (50-100 ns延迟) ⚡
```

#### 计算分析

**16MB缓冲区方案**：

- 8线程 × 16MB = 128MB工作集
- 远超L3缓存容量（8-32MB）
- 频繁的L3缓存未命中 → 访问主内存
- 内存带宽成为瓶颈

**4MB缓冲区方案** ✅：

- 8线程 × 4MB = 32MB工作集
- 完美适配L3缓存大小
- 每个线程的工作集驻留在L3缓存中
- 缓存命中率高 → 吞吐量大幅提升

**性能提升**: 4.4倍

---

### 3. 为什么不用系统提示标志？

#### Windows的FILE_FLAG_SEQUENTIAL_SCAN

**使用该标志时**：

```
Windows激进预读 → 128KB-1MB预读窗口
                ↓
         占用大量系统缓存
                ↓
    缓存压力过大，数据被驱逐
                ↓
         缓存未命中增加
                ↓
        性能下降（3.04 GB/s）
```

**不使用该标志时** ✅：

```
Windows智能预读算法 → 根据实际访问模式动态调整
                     ↓
              8个并发读取展现真实模式
                     ↓
              系统自动优化策略
                     ↓
           更少的过度预读
                     ↓
         缓存命中率提升
                     ↓
        性能飙升（16.1 GB/s）
```

#### Linux的posix_fadvise

**同样适用**：移除`posix_fadvise(POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED)`

- 让Linux内核自动管理预读策略
- 内核的智能算法比手动提示更优

---

### 4. 原生API vs 标准库

#### fstream的额外开销

```
用户代码
  ↓
std::ifstream::read()    ← 虚拟函数调用开销
  ↓
std::filebuf::xsgetn()   ← 内部缓冲管理开销
  ↓
底层系统调用             ← 额外一层抽象
  ↓
ReadFile() / pread()     ← 真正的I/O操作
```

#### 原生API的直通路径

```
用户代码
  ↓
ReadFile() / pread()     ← 直接系统调用
  ↓
磁盘I/O
```

**实测对比**（8线程）：

- **原生API**: 16.1 GB/s ✅
- **fstream**: 6.43 GB/s（慢150%）❌

---

### 5. 独立文件句柄的优势

#### 共享句柄方案（需要锁）

```cpp
// 需要互斥锁保护
mutex.lock();
SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);
ReadFile(hFile, dst, to_read, &bytes_read, NULL);
mutex.unlock();

// 8个线程竞争同一锁 → 严重的锁竞争
// 大部分时间花在等待锁上
```

#### 独立句柄方案（零锁竞争）✅

```cpp
// 每个线程独立打开文件
HANDLE hFile = CreateFileA(filename.c_str(), ...);
SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);
ReadFile(hFile, dst, to_read, &bytes_read, NULL);
CloseHandle(hFile);

// 零锁竞争，完全并行I/O
// Windows文件系统支持高效并发I/O
```

---

6. 内存分配优化

**Windows**: `VirtualAlloc` vs `new`

- ✅ 绕过CRT堆管理器
- ✅ 自动4KB页面对齐（优化DMA传输）
- ✅ 延迟提交（MEM_RESERVE + MEM_COMMIT）
- ✅ 大块内存分配更高效

**Linux**: `posix_memalign` vs `malloc`

- ✅ 4KB字节对齐
- ✅ 优化系统缓存访问

---

### 🏆 最快方案配置

核心配置清单

| 参数             | 最优值                                       | 说明                                      |
| ---------------- | -------------------------------------------- | ----------------------------------------- |
| **缓冲区大小**   | 4MB                                          | 适配CPU L3缓存                            |
| **线程数**       | 8线程                                        | 性能/资源最佳平衡<br>16线程略快但收益递减 |
| **系统提示标志** | 无                                           | 让OS自动优化                              |
| **API选择**      | ReadFile (Win)<br>pread (Linux)              | 原生系统API                               |
| **文件句柄**     | 每线程独立                                   | 零锁竞争                                  |
| **内存分配**     | VirtualAlloc (Win)<br>posix_memalign (Linux) | 页面对齐                                  |

---

### 📊 性能测试数据

Windows平台（实测）

| 线程数 | 加载时间    | 速度           | 加速比    | 推荐度     |
| ------ | ----------- | -------------- | --------- | ---------- |
| 1      | 1747 ms     | 3.60 GB/s      | 1.00×     | ❌          |
| 2      | 930 ms      | 6.75 GB/s      | 1.88×     | ❌          |
| 4      | 636 ms      | 9.88 GB/s      | 2.75×     | 可用       |
| **8**  | **~391 ms** | **~16.1 GB/s** | **4.47×** | ✅✅✅ 推荐   |
| **16** | **~368 ms** | **~17.1 GB/s** | **4.75×** | ✅ 极致性能 |

**注意**：

- 第一次运行是冷启动（~2秒，从磁盘读取）
- 后续运行利用系统缓存（~390ms，从内存读取）
- 16线程仅比8线程快6%，但线程开销更大

Linux平台（历史数据）

| 线程数 | 加载时间   | 速度           | 加速比    |
| ------ | ---------- | -------------- | --------- |
| 1      | 3372 ms    | 1.86 GB/s      | 1.00×     |
| 2      | 1846 ms    | 3.40 GB/s      | 1.83×     |
| 4      | 993 ms     | 6.32 GB/s      | 3.40×     |
| **8**  | **569 ms** | **11.04 GB/s** | **5.93×** |

---

### 🚀 快速开始（编译与运行）

Windows编译运行

**步骤1：打开VS Developer Command Prompt**

- 开始菜单 → Visual Studio 2022 → Developer Command Prompt for VS 2022

**步骤2：编译**

```cmd
cd /d T:\dataset\imagenet
cl /EHsc /std:c++17 /O2 /MD method2_native_unified.cpp /Fe:method2_unified.exe
```

**编译选项说明**：

- `/EHsc` - C++异常处理
- `/std:c++17` - C++17标准
- `/O2` - 最大速度优化
- `/MD` - 动态链接多线程CRT

**步骤3：运行**

```cmd
# 推荐：8线程（性能/资源最佳平衡）
.\method2_unified.exe imagenet_val.dts 8

# 极致性能：16线程（略快，但开销更大）
.\method2_unified.exe imagenet_val.dts 16

# 其他测试
.\method2_unified.exe imagenet_val.dts 4
.\method2_unified.exe imagenet_val.dts 2
.\method2_unified.exe imagenet_val.dts 1
```

**获取稳定数据**：

```cmd
# 连续运行5次，取后4次平均值
for /L %i in (1,1,5) do method2_unified.exe imagenet_val.dts 8
```

---

Linux编译运行

**步骤1：编译**

```bash
cd /path/to/imagenet
g++ -std=c++17 -O3 -march=native -pthread method2_native_unified.cpp -o method2_unified
```

**编译选项说明**：

- `-std=c++17` - C++17标准
- `-O3` - 最高优化级别
- `-march=native` - 针对当前CPU优化
- `-pthread` - 多线程支持

**步骤2：运行**

```bash
# 推荐：8线程
./method2_unified imagenet_val.dts 8

# 其他线程数
./method2_unified imagenet_val.dts 4
./method2_unified imagenet_val.dts 2
./method2_unified imagenet_val.dts 1
```

**获取稳定数据**：

```bash
# 连续运行5次，取后4次平均值
for i in {1..5}; do ./method2_unified imagenet_val.dts 8; done
```

---

### ⚡️ 性能优化建议

✅ DO（应该做的）

1. **使用4MB缓冲区** - 完美适配L3缓存
2. **使用原生系统API** - ReadFile/pread，比fstream快2-3倍
3. **每线程独立文件句柄** - 零锁竞争，完全并行
4. **使用VirtualAlloc/posix_memalign** - 页面对齐内存
5. **让OS自动优化** - 移除FILE_FLAG_SEQUENTIAL_SCAN等提示
6. **8线程作为默认配置** - 性价比最高

❌ DON'T（不应该做的）

1. **不要使用16MB缓冲区** - 超过L3缓存，降低性能
2. **不要使用系统提示标志** - 让OS自动优化更好
3. **不要使用fstream** - 抽象层开销大，慢150%
4. **不要共享文件句柄** - 需要锁，降低并发性能
5. **不要过度使用线程** - 16线程收益递减
6. **不要忽略冷启动** - 第一次运行慢是正常的

---

### 🔬 性能对比总结

不同方法对比（8线程，Windows）

| 方法                      | 速度          | 相对性能 | 说明       |
| ------------------------- | ------------- | -------- | ---------- |
| **Method 2 (Native API)** | **16.1 GB/s** | **100%** | ✅ 最优方案 |
| fstream (标准库)          | 6.43 GB/s     | 40%      | 慢150%     |
| mmap + memcpy             | 5.35 GB/s     | 33%      | 慢200%     |

跨平台对比

| 平台        | 8线程速度     | 相对性能 | 优势                             |
| ----------- | ------------- | -------- | -------------------------------- |
| **Windows** | **16.1 GB/s** | **100%** | ✅ 绝对速度快                     |
| **Linux**   | **11.0 GB/s** | **68%**  | ✅ 线程扩展性好（5.93× vs 4.47×） |

---

### 💡 关键经验总结

1. 反直觉的优化

- **更小的缓冲区反而更快**（4MB > 16MB）
- **移除系统提示反而更快**（无标志 > FILE_FLAG_SEQUENTIAL_SCAN）
- **简化代码往往比复杂优化更有效**

2. 理解硬件特性

- **CPU缓存层次**是关键考虑因素
- **L3缓存大小**决定了最优缓冲区配置
- 缓存命中率比内存带宽更重要

3. 相信现代OS

- **Windows/Linux的I/O调度器已经高度优化**
- 手动干预往往适得其反
- 让系统自动管理优于强制提示

4. 原生API的力量

- **绕过C++抽象层**，直达系统调用
- 减少函数调用开销
- 更精细的控制，更好的性能

5. 并行设计的艺术

- **独立文件句柄** > 共享句柄+锁
- 零锁竞争 = 完全并行
- 8线程是大多数场景的甜点

---

### 🎯 适用场景

✅ 强烈推荐

- 大文件顺序读取（>1GB）
- 需要快速加载到内存的场景
- 对I/O性能敏感的应用
- 数据库、机器学习、科学计算

❌ 不推荐

- 小文件随机读取（用mmap更好）
- 需要频繁修改文件
- 跨平台代码优先（考虑fstream牺牲部分性能）
- 内存极受限的嵌入式系统

---

### 📚 相关文件

- **代码实现**: `method2_native_unified.cpp` - 跨平台最优版本
- **详细实验报告**: `EXP.md` - 完整的实验过程和技术分析
- **编译指南**: `UNIFIED_BUILD.md` - 完整的编译运行文档
- **历史基准**: `benchmark_analysis.md` - 三种方法的性能对比
- **跨平台分析**: `benchmark_analysis_linux_vs_windows.md` - Windows vs Linux

---

*最终更新：2026-01-06*
*Windows最佳性能：17.29 GB/s (16线程)*
*Linux最佳性能：11.04 GB/s (8线程)*
*推荐配置：8线程 + 4MB缓冲区 + 原生API + 无系统提示*

## **（三）补充信息：.dts格式的文件头和BLOCK元信息**

简单起见，我们直接贴出我们的导出和读取.dts文件的python代码：

```python
BLOCK_SIZE = 16 * 1024 * 1024  # 16 MB
USING_PIC_ALIGNMENT = True
PIC_ALIGNMENT_BYTES = 64  # Bytes
FILE_MAGIC = b'.DTS'
DTS_VERSION = [3, 0, 0, 0]
IMAGENET_HEADER_SIZE = 16 * 1024 * 1024  # 16 MB
CIFAR_MNIST_HEADER_SIZE = 256  # 256 Bytes

# LV0 参数
HEADER_SIZE_LV0 = 4 * 1024  # 4 KB, 256 elements
MINIMUM_TRAIN_BLOCKS_LV0 = 8700
MINIMUM_VAL_BLOCKS_LV0 = 400
MARGINAL_FILL_ITERATIONS = 32
BLOCK_MAGIC_LV0 = b'LV0B'

def make_imagenet_file_header(file_path, train):
    if train:
        file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'train_file_name_list.pkl'))
        info_list = load_list(os.path.join(os.path.abspath(output_path), 'train_info_list.pkl'))
        blocks = load_list(os.path.join(output_path, 'train_blocks.pkl'))
    else:
        file_name_list = load_list(os.path.join(os.path.abspath(output_path), 'val_file_name_list.pkl'))
        info_list = load_list(os.path.join(os.path.abspath(output_path), 'val_info_list.pkl'))
        blocks = load_list(os.path.join(output_path, 'val_blocks.pkl'))

    stream = BytesIO()
    dataset_type = b'IMAGENET'
    is_training_set = 1 if train else 0
    val_set_prep = 1 if preprocess_val_set else 0
    num_classes = 1000
    tensor_layout = b'NHWC'
    image_width, image_height = 0, 0  # Default
    num_channels = 3
    color_channel_type = b' RGB'
    num_samples = len(file_name_list)
    num_volumes = 1
    volume_id = 0
    num_blocks = len(blocks)
    total_bytes = num_blocks * BLOCK_SIZE + IMAGENET_HEADER_SIZE
    block_bytes = num_blocks * BLOCK_SIZE
    if compress_level == 0:
        block_header_size = HEADER_SIZE_LV0
    else:
        block_header_size = HEADER_SIZE_LV123
    pic_alignment = PIC_ALIGNMENT_BYTES if USING_PIC_ALIGNMENT else 0
    maximum_pic_area = 31667328 if train else 18248230
    max_pic_per_block = max([len(s) for s in blocks])
    compression_ratio = total_bytes / sum([s[1] for s in info_list])
    normalize_mean = [0.0, 0.0, 0.0]
    normalize_std = [0.0, 0.0, 0.0]
    crc_code = get_partial_crc32(file_path, IMAGENET_HEADER_SIZE, return_str=False)

    # 144 Bytes
    data = struct.pack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI',
                       FILE_MAGIC,  # 4s, FILE_MAGIC = b'.DTS'
                       DTS_VERSION[0],  # B, 3
                       DTS_VERSION[1],  # B, 0
                       DTS_VERSION[2],  # B, 0
                       DTS_VERSION[3],  # B, 0
                       dataset_type,    # 8s, 'IMAGENET'
                       is_training_set, # I
                       compress_level,  # I
                       val_set_prep,    # I, 通常是0
                       num_classes,     # I
                       tensor_layout,   # 4s
                       image_width,     # I
                       image_height,    # I
                       num_channels,    # I
                       color_channel_type,  # 4s
                       num_samples,     # I
                       num_volumes,     # I
                       volume_id,       # I
                       num_blocks,      # I
                       num_blocks,      # I
                       total_bytes,       # Q
                       IMAGENET_HEADER_SIZE,      # I
                       block_bytes,      # Q
                       BLOCK_SIZE,       # I
                       block_header_size, # I
                       pic_alignment,    # I
                       maximum_pic_area, # I
                       max_pic_per_block,  # I
                       compression_ratio,  # f
                       normalize_mean[0],  # f, 正则化参数，这个暂时不使用
                       normalize_mean[1],  # f, 正则化参数，这个暂时不使用
                       normalize_mean[2],  # f, 正则化参数，这个暂时不使用
                       normalize_std[0],  # f, 正则化参数，这个暂时不使用
                       normalize_std[1],  # f, 正则化参数，这个暂时不使用
                       normalize_std[2],  # f, 正则化参数，这个暂时不使用
                       crc_code,          # I
                       )
    stream.write(data)
    stream.write(b'\x00' * (IMAGENET_HEADER_SIZE - 144))
    return stream.getvalue()


def load_dts_header_info(file_path):
    """ 读取文件头，显示数据集信息 """
    header_size = 144
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DTS file not found: {file_path}")
    with open(file_path, 'rb') as f:
        header_data = f.read(256)
        if len(header_data) < 256:
            raise ValueError(f"Invalid DTS file.")
        header = struct.unpack('<4sBBBB8sIIII4sIII4sIIIIIQIQIIIIIfffffffI', header_data[:header_size])
        magic, version_0, version_1, version_2, _, \
        dataset_type, is_training_set, compress_level_read, val_set_prep, num_classes, \
        tensor_layout, image_width, image_height, num_channels, color_channel_type, \
        num_samples, num_volumes, volume_id, total_blocks, num_blocks, \
        total_bytes, header_bytes, block_bytes, bytes_per_block, block_header_size, \
        pic_alignment, maximum_pic_area, max_pic_per_block, compression_ratio, \
        normalize_mean_0, normalize_mean_1, normalize_mean_2, \
        normalize_std_0, normalize_std_1, normalize_std_2, crc_code = header
        print('Magic:\t\t\t', f'{magic}')
        print('Version:\t\t', f'{version_0:d}.{version_1:d}.{version_2:d}')
        print('Dataset type:\t\t', f'{dataset_type}')
        print('Is training set:\t', 'True' if is_training_set else 'False')
        print('Compression level:\t', f'{compress_level_read}')
        print('Val set preprocess:\t', 'True' if val_set_prep else 'False')
        print('Number of classes:\t', f'{num_classes}')
        print('Tensor layout:\t\t', f'{tensor_layout}')
        print('Image width:\t\t', f'{image_width}')
        print('Image height:\t\t', f'{image_height}')
        print('Number of channels:\t', f'{num_channels}')
        print('Color channel type:\t', f'{color_channel_type[1:]}')
        print('Number of samples:\t', f'{num_samples}')
        print('Number of volumes:\t', f'{num_volumes}')
        print('Volume ID:\t\t', f'{volume_id}')
        print('Total blocks:\t\t', f'{total_blocks}')
        print('Blocks in this file:\t', f'{num_blocks}')
        print('Total bytes:\t\t', f'{total_bytes}')
        print('Header bytes:\t\t', f'{header_bytes}')
        print('Block bytes:\t\t', f'{block_bytes}')
        print('Bytes per block:\t', f'{bytes_per_block}')
        print('Block header size:\t', f'{block_header_size}')
        print('Picture alignment:\t', f'{pic_alignment}')
        print('Max picture area:\t', f'{maximum_pic_area}')
        print('Max num of pic per block: ', f'{max_pic_per_block}')
        print('Compression ratio:\t', f'{100 * compression_ratio:.2f}')
        print('Normalization mean R:\t', f'{normalize_mean_0:.2f}')
        print('Normalization mean G:\t', f'{normalize_mean_1:.2f}')
        print('Normalization mean B:\t', f'{normalize_mean_2:.2f}')
        print('Normalization std R:\t', f'{normalize_std_0:.2f}')
        print('Normalization std G:\t', f'{normalize_std_1:.2f}')
        print('Normalization std B:\t', f'{normalize_std_2:.2f}')
        print('CRC-32 code:\t\t', f'{crc_code & 0xFFFFFFFF:08x}')

```

可以看到我们的整数基本都是用int32_t存储的，因为文件头大小固定为16MB，空间绰绰有余，不需要在这里节省空间。另外，LV0~LV3的文件头的格式、大小都是一样的。

而每个BLOCK的元信息，情况大致如下：

```python
# LV0 参数
HEADER_SIZE_LV0 = 4 * 1024  # 4 KB, 256 elements
MINIMUM_TRAIN_BLOCKS_LV0 = 8700
MINIMUM_VAL_BLOCKS_LV0 = 400
MARGINAL_FILL_ITERATIONS = 32
BLOCK_MAGIC_LV0 = b'LV0B'

# LV1 ~ LV3 参数
HEADER_SIZE_LV123 = 16 * 1024  # 16 KB, 1024 elements
BLOCK_MAGIC_LV1 = b'LV1B'
BLOCK_MAGIC_LV2 = b'LV2B'
BLOCK_MAGIC_LV3 = b'LV3B'

fmt = f'<{len(offset_list)}I'
header_data = struct.pack('<4sII', block_magic, block_id, num_pics)
block_header_steam.write(header_data)
block_header_steam.seek(0, SEEK_END)
block_header_steam.write(struct.pack(fmt, *offset_list))
block_header_steam.seek(0, SEEK_END)
block_header_steam.write(struct.pack(fmt, *size_list))
block_header_steam.seek(0, SEEK_END)
block_header_steam.write(struct.pack(fmt, *label_list))
block_header_steam.seek(0, SEEK_END)
padding_header = HEADER_SIZE_LV123 - 12 - 3 * num_pics * 4
block_header_steam.write(b'\x00' * padding_header)

block_header_steam.seek(0, SEEK_END)
block_stream.seek(0)
block_header_steam.write(block_stream.read())
```

这里再次强调，LV0和LV1~3的BLOCK HEADER（元信息） 的空间是不一样的，LV0因为图片较大、总数较少，元信息所占的空间只有4KB；而LV1~3的元信息所占空间则是16KB。在元信息之后，才是每张图片的数据。每张图片都是64B对齐的。元信息在写完block_magic, block_id, num_pics之后，依次是offset、size、label（所有图片的offset连在一起，size和label也是）。offset_list里的数据是偏移量，指的是每张图片的第一个字节相对于该BLOCK的起始位置的偏移量；size_list里的数据是图片的实际字节数，不包含对齐的padding；而label是标签，0~999之间的整数。

## **（四）补充信息：随机数发生器**

我们已经实现了理论上具备多线程随机可复现性的随机数发生器。但是如何在数据读取中做到真正的随机可复现，也要看怎么使用。

### philox.h

```c++
/**
 * @file philox.h
 * @brief Philox4x32-10 伪随机数生成算法（CPU/GPU通用）
 * @details Counter-Based RNG，支持并行可复现生成
 *          参考：Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: 无（纯C++标准库 + CUDA兼容）
 * @note 所属系列: base
 */

#pragma once

#include <cstdint>
#include <array>
#include <cmath>

// CUDA/MUSA兼容性宏
#if defined(__CUDA_ARCH__)
    #define TR_HOST_DEVICE __host__ __device__
    #define TR_DEVICE __device__
    #define TR_FORCEINLINE __forceinline__
#elif defined(__MUSA_ARCH__)
    #define TR_HOST_DEVICE __host__ __device__
    #define TR_DEVICE __device__
    #define TR_FORCEINLINE __forceinline__
#else
    #define TR_HOST_DEVICE
    #define TR_DEVICE
    #define TR_FORCEINLINE inline
#endif

namespace tr {
namespace detail {

// =============================================================================
// Philox4x32-10 常量（CPU/GPU通用）
// =============================================================================

constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9u;  // golden ratio
constexpr uint32_t PHILOX_W32_1   = 0xBB67AE85u;  // sqrt(3)-1

// =============================================================================
// 核心算法（CPU/GPU通用）
// =============================================================================

/**
 * @brief 32位乘法的高位结果（CPU/GPU通用）
 * @param a 乘数1
 * @param b 乘数2
 * @param lo 输出：低32位
 * @return 高32位
 */
TR_HOST_DEVICE TR_FORCEINLINE
uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* lo) {
#ifdef __CUDA_ARCH__
    // GPU：使用PTX内联汇编（最优性能）
    uint32_t hi;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(a), "r"(b));
    *lo = a * b;
    return hi;
#else
    // CPU：使用64位乘法
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *lo = static_cast<uint32_t>(product);
    return static_cast<uint32_t>(product >> 32);
#endif
}

/**
 * @brief Philox4x32 单轮函数
 * @param ctr0,ctr1,ctr2,ctr3 计数器（输入输出）
 * @param key0,key1 密钥
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox4x32_round(
    uint32_t* ctr0, uint32_t* ctr1, uint32_t* ctr2, uint32_t* ctr3,
    uint32_t key0, uint32_t key1
) {
    uint32_t lo0, lo1;
    uint32_t hi0 = mulhilo32(PHILOX_M4x32_0, *ctr0, &lo0);
    uint32_t hi1 = mulhilo32(PHILOX_M4x32_1, *ctr2, &lo1);

    uint32_t new_ctr0 = hi1 ^ *ctr1 ^ key0;
    uint32_t new_ctr1 = lo1;
    uint32_t new_ctr2 = hi0 ^ *ctr3 ^ key1;
    uint32_t new_ctr3 = lo0;

    *ctr0 = new_ctr0;
    *ctr1 = new_ctr1;
    *ctr2 = new_ctr2;
    *ctr3 = new_ctr3;
}

/**
 * @brief Philox4x32-10 核心函数（10轮迭代）
 * @param ctr0,ctr1,ctr2,ctr3 计数器（输入输出）
 * @param key0,key1 密钥
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox4x32_10(
    uint32_t* ctr0, uint32_t* ctr1, uint32_t* ctr2, uint32_t* ctr3,
    uint32_t key0, uint32_t key1
) {
    // 10轮迭代（展开以优化）
#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (int round = 0; round < 10; ++round) {
        philox4x32_round(ctr0, ctr1, ctr2, ctr3, key0, key1);
        key0 += PHILOX_W32_0;
        key1 += PHILOX_W32_1;
    }
}

/**
 * @brief 从seed和offset生成4个uint32随机数
 * @param seed 64位种子
 * @param offset 64位偏移量
 * @param out 输出数组[4]
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox_generate_4x32(uint64_t seed, uint64_t offset, uint32_t* out) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    uint32_t ctr0 = static_cast<uint32_t>(offset);
    uint32_t ctr1 = static_cast<uint32_t>(offset >> 32);
    uint32_t ctr2 = 0;
    uint32_t ctr3 = 0;

    philox4x32_10(&ctr0, &ctr1, &ctr2, &ctr3, key0, key1);

    out[0] = ctr0;
    out[1] = ctr1;
    out[2] = ctr2;
    out[3] = ctr3;
}

/**
 * @brief 生成[0, 1)范围的float（CPU/GPU通用）
 * @param seed 种子
 * @param offset 偏移量
 * @return [0, 1)范围的随机浮点数
 */
TR_HOST_DEVICE TR_FORCEINLINE
float philox_uniform_float(uint64_t seed, uint64_t offset) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);

    // 标准方法：取高23位作为尾数
    constexpr float scale = 1.0f / 16777216.0f;  // 2^-24
    return static_cast<float>(r[0] >> 8) * scale;
}

/**
 * @brief Box-Muller变换生成标准正态分布（CPU/GPU通用）
 * @param seed 种子
 * @param offset 偏移量
 * @param out0 第一个正态随机数
 * @param out1 第二个正态随机数
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox_normal_pair(uint64_t seed, uint64_t offset, float* out0, float* out1) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);

    // 转换为(0, 1)范围（避免log(0)）
    constexpr float scale = 1.0f / 16777216.0f;
    float u1 = (static_cast<float>((r[0] >> 8) | 1)) * scale;  // (0, 1]
    float u2 = static_cast<float>(r[1] >> 8) * scale;          // [0, 1)

    // Box-Muller变换
    constexpr float two_pi = 6.283185307179586f;

#ifdef __CUDA_ARCH__
    float radius = sqrtf(-2.0f * logf(u1));
    float theta = two_pi * u2;
    float sin_theta, cos_theta;
    sincosf(theta, &sin_theta, &cos_theta);  // CUDA优化版本
    *out0 = radius * cos_theta;
    *out1 = radius * sin_theta;
#else
    float radius = std::sqrt(-2.0f * std::log(u1));
    float theta = two_pi * u2;
    *out0 = radius * std::cos(theta);
    *out1 = radius * std::sin(theta);
#endif
}

/**
 * @brief 生成单个uint64随机数
 * @param seed 种子
 * @param offset 偏移量
 * @return uint64随机数
 */
TR_HOST_DEVICE TR_FORCEINLINE
uint64_t philox_uint64(uint64_t seed, uint64_t offset) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);
    return (static_cast<uint64_t>(r[0]) << 32) | r[1];
}

} // namespace detail
} // namespace tr

#undef TR_HOST_DEVICE
#undef TR_DEVICE
#undef TR_FORCEINLINE

```



### rng.h

```c++
/**
 * @file rng.h
 * @brief 随机数生成器类声明
 * @details 基于Philox4x32-10的Counter-Based RNG
 *          核心特性：多线程可复现、高性能、跨平台
 *          使用Pimpl模式避免在头文件中包含<atomic>
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: base
 */

#pragma once

#include <cstdint>
#include <memory>  // for std::unique_ptr

namespace tr {

/**
 * @class Generator
 * @brief 伪随机数生成器（基于Philox4x32-10）
 *
 * 核心设计：
 * - 状态仅由 seed 和 offset 决定
 * - 使用原子操作预留偏移量区间，支持无锁并行
 * - 相同(seed, offset)永远产生相同随机数序列
 *
 * 内存布局：使用Pimpl模式，避免在头文件中暴露std::atomic
 * - 原因：MUSA SDK的musa/std/atomic与标准库<atomic>存在命名空间冲突
 * - 解决方案：通过前置声明将std::atomic隐藏在.cpp实现文件中
 * - 效果：MUSA环境下编译renaissance.h时不会包含<atomic>
 *
 * 典型用法：
 * @code
 * // 全局设置
 * tr::manual_seed(42);
 *
 * // 使用默认生成器
 * tr::cpu_rand_normal_float(ptr, count, 0.0f, 1.0f);
 *
 * // 使用独立生成器（推荐用于多线程数据加载）
 * tr::Generator gen(1234);
 * tr::cpu_rand_normal_float(ptr, count, 0.0f, 1.0f, gen);
 * @endcode
 */
class Generator {
public:
    /**
     * @brief 构造函数
     * @param seed 随机种子（默认0）
     */
    explicit Generator(uint64_t seed = 0) noexcept;

    /**
     * @brief 析构函数
     */
    ~Generator();

    // 禁止拷贝（原子成员不可拷贝）
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // 允许移动
    Generator(Generator&& other) noexcept;
    Generator& operator=(Generator&& other) noexcept;

    // =========================================================================
    // 种子管理
    // =========================================================================

    /**
     * @brief 设置种子（重置offset为0）
     * @param seed 新种子
     * @note 线程安全，但会短暂阻塞其他操作
     */
    void set_seed(uint64_t seed);

    /**
     * @brief 获取当前种子
     * @return 种子值
     */
    uint64_t seed() const noexcept;

    // =========================================================================
    // 状态管理（用于Checkpoint）
    // =========================================================================

    /**
     * @brief 获取完整状态
     * @return {seed, offset}
     */
    std::pair<uint64_t, uint64_t> get_state() const;

    /**
     * @brief 设置完整状态
     * @param seed 种子
     * @param offset 偏移量
     */
    void set_state(uint64_t seed, uint64_t offset);

    // =========================================================================
    // 偏移量管理（核心方法）
    // =========================================================================

    /**
     * @brief 原子预留偏移量区间
     * @param count 需要的随机数个数
     * @return 本次预留的起始offset
     *
     * 这是实现多线程可复现的关键：
     * - 调用者获得 [返回值, 返回值+count) 区间的独占使用权
     * - 即便多线程并发调用，每个调用获得的区间互不重叠
     * - 相同的调用序列产生相同的区间分配
     */
    uint64_t next_offset(uint64_t count);

    /**
     * @brief 获取当前偏移量（不修改状态）
     * @return 当前offset
     */
    uint64_t current_offset() const noexcept;

private:
    class Impl;  // Pimpl: 前向声明实现类
    std::unique_ptr<Impl> impl_;  // 指向实现的智能指针
};

// =============================================================================
// 全局函数
// =============================================================================

/**
 * @brief 获取默认全局生成器
 * @return Generator引用
 * @note 线程安全（Meyers单例）
 */
Generator& get_default_generator();

/**
 * @brief 设置全局随机种子
 * @param seed 种子值
 *
 * 等价于 get_default_generator().set_seed(seed)
 */
void manual_seed(uint64_t seed);

// =============================================================================
// CPU随机数生成函数（核心API）
// =============================================================================

/**
 * @brief 生成N个随机uint64整数
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param gen 生成器引用
 *
 * 多线程安全且可复现
 */
void cpu_rand_uint64(uint64_t* ptr, size_t count, Generator& gen);

/**
 * @brief 生成N个伯努利分布的INT8（0或1）
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param prob_one "1"的概率，范围[0, 1]
 * @param gen 生成器引用
 */
void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one, Generator& gen);

/**
 * @brief 生成N个均匀分布的INT8
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param low 最小值（包含）
 * @param high 最大值（包含）
 * @param gen 生成器引用
 */
void cpu_rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high, Generator& gen);

/**
 * @brief 生成N个伯努利分布的INT32（0或1）
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param prob_one "1"的概率，范围[0, 1]
 * @param gen 生成器引用
 */
void cpu_rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one, Generator& gen);

/**
 * @brief 生成N个均匀分布的INT32
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param low 最小值（包含）
 * @param high 最大值（不包含）
 * @param gen 生成器引用
 * @note 范围为 [low, high)，与Python randint语义一致
 */
void cpu_rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high, Generator& gen);

/**
 * @brief 生成N个均匀分布的FP32
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param low 最小值（包含）
 * @param high 最大值（不包含）
 * @param gen 生成器引用
 */
void cpu_rand_uniform_float(float* ptr, size_t count, float low, float high, Generator& gen);

/**
 * @brief 生成N个正态分布的FP32
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param mean 均值
 * @param std 标准差
 * @param gen 生成器引用
 */
void cpu_rand_normal_float(float* ptr, size_t count, float mean, float std, Generator& gen);

// =============================================================================
// 便捷函数（使用默认生成器）
// =============================================================================

inline void cpu_rand_uint64(uint64_t* ptr, size_t count) {
    cpu_rand_uint64(ptr, count, get_default_generator());
}

inline void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one) {
    cpu_rand_bernoulli_int8(ptr, count, prob_one, get_default_generator());
}

inline void cpu_rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high) {
    cpu_rand_uniform_int8(ptr, count, low, high, get_default_generator());
}

inline void cpu_rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one) {
    cpu_rand_bernoulli_int32(ptr, count, prob_one, get_default_generator());
}

inline void cpu_rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high) {
    cpu_rand_uniform_int32(ptr, count, low, high, get_default_generator());
}

inline void cpu_rand_uniform_float(float* ptr, size_t count, float low = 0.0f, float high = 1.0f) {
    cpu_rand_uniform_float(ptr, count, low, high, get_default_generator());
}

inline void cpu_rand_normal_float(float* ptr, size_t count, float mean = 0.0f, float std = 1.0f) {
    cpu_rand_normal_float(ptr, count, mean, std, get_default_generator());
}

} // namespace tr

```



### rng.cpp

```c++
/**
 * @file rng.cpp
 * @brief 随机数生成器实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: rng.h, philox.h
 * @note 所属系列: base
 */

#include "renaissance/base/rng.h"
#include "renaissance/base/philox.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <mutex>

// OpenMP支持检测
#if defined(_OPENMP)
    #include <omp.h>
    #define TR_USE_OPENMP 1
#else
    #define TR_USE_OPENMP 0
#endif

namespace tr {

// =============================================================================
// Generator::Impl 定义（Pimpl实现类）
// =============================================================================
//
// 重要说明：为什么使用Pimpl模式？
//
// 问题描述：
// - MUSA SDK的<atomic>实现(musa/std/atomic)与C++标准库的<atomic>存在命名空间冲突
// - 如果在头文件中包含<atomic>，MUSA环境编译时会触发大量编译错误
// - 即使通过条件编译(#ifdef TR_USE_MUSA)也无法完全解决，因为包含顺序影响
//
// 解决方案：
// - 在头文件(rng.h)中只前置声明"class Impl"，使用std::unique_ptr<Impl>
// - 在实现文件(rng.cpp)中定义Impl类，此时包含<atomic>是安全的
// - MUSA环境编译renaissance.h时看不到<atomic>，避免冲突
// - CPU/CUDA环境正常编译，不受影响
//
// 性能影响：
// - 增加一次指针解引用(indirection)，但现代CPU的分支预测可缓解
// - unique_ptr开销极小(8字节指针)，相比原来直接成员增加可忽略
// - 关键路径(next_offset)仍使用原子操作，性能不受影响

class Generator::Impl {
public:
    uint64_t seed_;
    std::atomic<uint64_t> offset_;
    std::mutex mutex_;

    explicit Impl(uint64_t seed) noexcept
        : seed_(seed)
        , offset_(0)
    {
    }
};

// =============================================================================
// Generator 实现
// =============================================================================

Generator::Generator(uint64_t seed) noexcept
    : impl_(new Impl(seed))
{
}

Generator::~Generator() = default;

Generator::Generator(Generator&& other) noexcept
    : impl_(std::move(other.impl_))
{
    // 移动后源对象的impl_自动置为nullptr
}

Generator& Generator::operator=(Generator&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        // 源对象的impl_自动置为nullptr
    }
    return *this;
}

void Generator::set_seed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->seed_ = seed;
    impl_->offset_.store(0, std::memory_order_release);
    LOG_DEBUG << "Generator seed set to " << seed;
}

uint64_t Generator::seed() const noexcept {
    return impl_->seed_;
}

std::pair<uint64_t, uint64_t> Generator::get_state() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return {impl_->seed_, impl_->offset_.load(std::memory_order_acquire)};
}

void Generator::set_state(uint64_t seed, uint64_t offset) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->seed_ = seed;
    impl_->offset_.store(offset, std::memory_order_release);
}

uint64_t Generator::next_offset(uint64_t count) {
    // 原子加法，返回旧值
    // 这是实现并发安全的关键：每次调用获得独占的区间
    return impl_->offset_.fetch_add(count, std::memory_order_relaxed);
}

uint64_t Generator::current_offset() const noexcept {
    return impl_->offset_.load(std::memory_order_relaxed);
}

// =============================================================================
// 全局函数实现
// =============================================================================

Generator& get_default_generator() {
    // Meyers单例，线程安全
    static Generator instance(0);
    return instance;
}

void manual_seed(uint64_t seed) {
    get_default_generator().set_seed(seed);
    LOG_INFO << "Global random seed set to " << seed;
}

// =============================================================================
// CPU随机数生成函数实现
// =============================================================================

namespace {

/**
 * @brief 计算并行任务的线程数
 */
inline int get_num_threads(size_t count) {
#if TR_USE_OPENMP
    // 小任务不值得并行
    if (count < 4096) return 1;

    int max_threads = omp_get_max_threads();
    // 每个线程至少处理1024个元素
    int ideal_threads = static_cast<int>((count + 1023) / 1024);
    return std::min(max_threads, ideal_threads);
#else
    (void)count;
    return 1;
#endif
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// uint64 生成
// -----------------------------------------------------------------------------

void cpu_rand_uint64(uint64_t* ptr, size_t count, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uint64");
    }

    // 预留偏移量区间
    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        ptr[i] = detail::philox_uint64(seed, base_offset + i);
    }
}

// -----------------------------------------------------------------------------
// INT8 伯努利分布
// -----------------------------------------------------------------------------

void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_bernoulli_int8");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_VALUE_ERROR("prob_one must be in [0, 1], got " << prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    // 将概率转换为阈值（uint32范围）
    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        ptr[i] = (r[0] < threshold) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// INT8 均匀分布
// -----------------------------------------------------------------------------

void cpu_rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uniform_int8");
    }
    if (low > high) {
        TR_VALUE_ERROR("low (" << static_cast<int>(low)
                 << ") must be <= high (" << static_cast<int>(high) << ")");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    // 计算范围
    uint32_t range = static_cast<uint32_t>(high - low) + 1;

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        // 使用模运算映射到[0, range)，然后加low
        uint32_t val = r[0] % range;
        ptr[i] = static_cast<int8_t>(low + static_cast<int8_t>(val));
    }
}

// -----------------------------------------------------------------------------
// INT32 伯努利分布
// -----------------------------------------------------------------------------

void cpu_rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_bernoulli_int32");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_VALUE_ERROR("prob_one must be in [0, 1], got " << prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        ptr[i] = (r[0] < threshold) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// INT32 均匀分布
// -----------------------------------------------------------------------------

void cpu_rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uniform_int32");
    }
    if (low > high) {
        TR_VALUE_ERROR("low (" << low << ") must be <= high (" << high << ")");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    // 使用64位计算避免溢出
    // 注意：范围是 [low, high)（左闭右开），与Python randint语义一致
    uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        // 组合两个32位数获得更大的范围
        uint64_t combined = (static_cast<uint64_t>(r[0]) << 32) | r[1];
        uint64_t val = combined % range;
        ptr[i] = static_cast<int32_t>(low + static_cast<int64_t>(val));
    }
}

// -----------------------------------------------------------------------------
// FP32 均匀分布
// -----------------------------------------------------------------------------

void cpu_rand_uniform_float(float* ptr, size_t count, float low, float high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uniform_float");
    }
    if (low > high) {
        TR_VALUE_ERROR("low (" << low << ") must be <= high (" << high << ")");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    float scale = high - low;

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        float u = detail::philox_uniform_float(seed, base_offset + i);
        ptr[i] = low + u * scale;
    }
}

// -----------------------------------------------------------------------------
// FP32 正态分布
// -----------------------------------------------------------------------------

void cpu_rand_normal_float(float* ptr, size_t count, float mean, float std, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_normal_float");
    }
    if (std < 0.0f) {
        TR_VALUE_ERROR("std must be >= 0, got " << std);
    }

    // Box-Muller每次生成2个数，所以消耗的offset是(count + 1) / 2
    uint64_t pairs_needed = (count + 1) / 2;
    uint64_t base_offset = gen.next_offset(pairs_needed);
    uint64_t seed = gen.seed();

    int num_threads = get_num_threads(count);

    // 处理成对的元素
    int64_t pair_count = static_cast<int64_t>(count / 2);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < pair_count; ++i) {
        float n0, n1;
        detail::philox_normal_pair(seed, base_offset + i, &n0, &n1);
        ptr[i * 2] = mean + std * n0;
        ptr[i * 2 + 1] = mean + std * n1;
    }

    // 处理最后一个奇数元素（如果有）
    if (count % 2 == 1) {
        float n0, n1;
        detail::philox_normal_pair(seed, base_offset + pair_count, &n0, &n1);
        ptr[count - 1] = mean + std * n0;
        // n1被丢弃，但这是确定性的
    }
}

} // namespace tr

```



## **（五）小结**

怎么说呢，在成功导出完数据集并且验证了数据排布正确之后，我们觉得完成了一件大事。但这只是万里长征第一步。我们已经在数据格式上建立了初步的优势，后面一定要把这个优势利用好，在读取速度上要达到一流水准。我们的LV0~LV3的格式高度统一，几乎可以用完全相同的方式去读取。BLOCK规格都是16MB，分块读取一定要利用好多线程。我们需要更具体的方案，来实现最高性能读取、并且合理避免锁竞争、更要避免读取到错误的数据、避免读取到未完成写入的数据、更要避免遗漏任何数据。另外，我们也要避免多余的内存复制。循环队列的方式可以在部分加载的模式下有效节省内存空间，但多线程场景下的读写顺序需要精细安排——比如，我们绝对不要出现“一次性硬性分配32个读取指针给32个worker，必须等待32个worker都读取完成才分配下一组32个指针”的低效情况。

下一步，正式实现DataLoader、FastDataLoader类，让我们的深度学习框架正式扬帆起航！


