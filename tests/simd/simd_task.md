# 【顶级谜题：最快的RandomResizedCrop】



我们是一个做自研深度学习框架的团队。项目基于C++17。

我们已经实现了超高速度的数据集加载。

但是，现在还需要一个RandomResizedCrop。我们指定使用的库是**ermig 1979的Simd库**。

我们后期会使用多线程，但是为了试出单线程的最佳方案，我们现在先**强制不允许使用多线程**。只有在单线程取得最快速度后，我们转为多线程才会最快。

## 一、什么是RandomResizedCrop

以下是PyTorch官网的解释：

torchvision.transforms.RandomResizedCrop(*size*, *scale=(0.08, 1.0)*, *ratio=(0.75, 1.3333333333333333)*, *interpolation=InterpolationMode.BILINEAR*, *antialias: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = True*)

Crop a random portion of image and resize it to a given size.

If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions

A crop of the original image is made: the crop has a random area (H * W) and a random aspect ratio. This crop is finally resized to the given size. This is popularly used to train the Inception networks.

- Parameters:

  **size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* *sequence*) –expected output size of the crop, for each edge. If size is an int instead of sequence like (h, w), a square output size `(size, size)` is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).NoteIn torchscript mode size as single int is not supported, use a sequence of length 1: `[size, ]`.**scale** (*tuple of python:float*) – Specifies the lower and upper bounds for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.**ratio** (*tuple of python:float*) – lower and upper bounds for the random aspect ratio of the crop, before resizing.**interpolation** (*InterpolationMode*) – Desired interpolation enum defined by `torchvision.transforms.InterpolationMode`. Default is `InterpolationMode.BILINEAR`. If input is Tensor, only `InterpolationMode.NEAREST`, `InterpolationMode.NEAREST_EXACT`, `InterpolationMode.BILINEAR` and `InterpolationMode.BICUBIC` are supported. The corresponding Pillow integer constants, e.g. `PIL.Image.BILINEAR` are accepted as well.**antialias** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) –Whether to apply antialiasing. It only affects **tensors** with bilinear or bicubic modes and it is ignored otherwise: on PIL images, antialiasing is always applied on bilinear or bicubic modes; on other modes (for PIL images and tensors), antialiasing makes no sense and this parameter is ignored. Possible values are:`True` (default): will apply antialiasing for bilinear or bicubic modes. Other mode aren’t affected. This is probably what you want to use.`False`: will not apply antialiasing for tensors on any mode. PIL images are still antialiased on bilinear or bicubic modes, because PIL doesn’t support no antialias.`None`: equivalent to `False` for tensors and `True` for PIL images. This value exists for legacy reasons and you probably don’t want to use it unless you really know what you are doing.The default value changed from `None` to `True` in v0.17, for the PIL and Tensor backends to be consistent.



## 二、我们的要求（重点！）

用C++写一个test_crop.cpp文件，并给出编译和运行它的命令。

这个文件的作用是这样的：

运行命令中提供一个--path参数，指向一个input.jpg文件。

文件打开这个jpg文件，然后用libjpeg-turbo解码，放到内存中某个地方。

这个文件实现了一个RandomResizedCrop(void* input_ptr, void* output_ptr, int total_bytes, int output_width, int input_width, int input_height, float scale_min, float_scale_max, float ratio_min, float ratio_max)——这个函数签名只是个参考，可以改。

默认scale_min是0.08，scale_max是1.0。ratio_min是0.75，ratio_max是1.33333。缩放算法固定使用双线性。

输出的图像的长宽也要可变（但是固定为正方形），默认224×224。

这个程序把JPEG加载到内存中以后，一直保留着，然后对它连续做1000次RandomResizedCrop，并把结果输出到指定的内存指针位置。为了防止编译器优化掉，我们设置一个int8_t变量result，每次都对第一个元素求按位异或，最后打印这个变量result。

统计1000次RandomResizedCrop的总用时，然后除以1000，最后打印这个时间（以毫秒为单位打印）。

### 要求：

必须使用Simd库。

测试平台是Windows或Linux，代码要能够跨平台。CPU支持AVX2，不支持AVX512。

使用C++17。

计时不包括加载jpg文件和解码，也不包括最终的打印，而只包括那1000次RandomResizedCrop。

随机数发生器现在统一使用rand，后期我们会自行修改。

可以使用mimalloc，或其他更先进的内存分配策略，但不要引入更多其他依赖。并且，全部功能必须在这一个cpp文件里面实现，不能增加其他文件。

必须是单线程。

注意设置好内存对齐策略。

必须给出完整的编译链接命令。编译器假定是g++。

这个程序一定要极度优化，达到最快的运行速度。

返回完整的可运行的代码。



## 三、我们征集到的方案

三个方案都很糟糕，速度很慢。我们希望有办法可以优化。

### 方案A

```c++
/**
 * @file a.cpp
 * @brief Expert CG's RandomResizedCrop Implementation using Simd C API
 * @details 使用 Simd C API 实现基本的 RandomResizedCrop
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <iostream>
#include <algorithm>

#include <Simd/SimdLib.h>
#include <turbojpeg.h>

// ============================================================================
// 跨平台对齐内存分配
// ============================================================================
static inline void* aligned_malloc(size_t size, size_t alignment = 64) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size)) ptr = nullptr;
    return ptr;
#endif
}

static inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// 图像数据结构
// ============================================================================
struct ImageData {
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

// ============================================================================
// JPEG 解码（使用 libjpeg-turbo）
// ============================================================================
bool decode_jpeg(const std::string& path, ImageData& img) {
    tjhandle tj = tjInitDecompress();
    if (!tj) {
        return false;
    }

    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        return false;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char* jpeg_buf = (unsigned char*)malloc(size);
    fread(jpeg_buf, 1, size, file);
    fclose(file);

    int subsamp, colorspace;
    if (tjDecompressHeader3(tj, jpeg_buf, size, &img.width, &img.height, &subsamp, &colorspace)) {
        free(jpeg_buf);
        tjDestroy(tj);
        return false;
    }

    img.stride = img.width * 3;
    img.data = (uint8_t*)aligned_malloc(img.height * img.stride, 64);
    if (tjDecompress2(tj, jpeg_buf, size, img.data, img.width, 0, img.height, TJPF_RGB, TJFLAG_FASTDCT)) {
        aligned_free(img.data);
        img.data = nullptr;
        free(jpeg_buf);
        tjDestroy(tj);
        return false;
    }

    free(jpeg_buf);
    tjDestroy(tj);
    return true;
}

// ============================================================================
// 裁剪矩形结构
// ============================================================================
struct CropRect {
    int x, y, w, h;
};

// ============================================================================
// 随机裁剪参数生成
// ============================================================================
static inline CropRect get_random_crop(int img_w, int img_h,
                                       float scale_min, float scale_max,
                                       float ratio_min, float ratio_max) {
    const float area = static_cast<float>(img_w * img_h);
    for (int i = 0; i < 10; ++i) {
        float target_area = area * (scale_min + (scale_max - scale_min) * (rand() / (RAND_MAX + 1.0f)));
        float aspect = ratio_min + (ratio_max - ratio_min) * (rand() / (RAND_MAX + 1.0f));
        int w = static_cast<int>(std::round(std::sqrt(target_area * aspect)));
        int h = static_cast<int>(std::round(std::sqrt(target_area / aspect)));
        if (w <= img_w && h <= img_h)
            return {rand() % (img_w - w + 1), rand() % (img_h - h + 1), w, h};
    }
    int size = std::min(img_w, img_h);
    return {(img_w - size) / 2, (img_h - size) / 2, size, size};
}

// ============================================================================
// RandomResizedCrop 核心函数
// ============================================================================
void RandomResizedCrop(const uint8_t* input_ptr, int input_w, int input_h, int input_stride,
                       uint8_t* output_ptr, int output_size,
                       float scale_min = 0.08f, float scale_max = 1.0f,
                       float ratio_min = 0.75f, float ratio_max = 1.3333f)
{
    CropRect rect = get_random_crop(input_w, input_h, scale_min, scale_max, ratio_min, ratio_max);

    // 使用 Simd C API 的 Resize 函数
    // SimdResize 需要源和目标的指针和尺寸
    const uint8_t* src_data = input_ptr + rect.y * input_stride + rect.x * 3;

    // 创建 resizer
    void* resizer = SimdResizerInit(rect.w, rect.h, output_size, output_size, 3,
                                     SimdResizeChannelByte, SimdResizeMethodBilinear);

    if (resizer) {
        // 执行 resize
        SimdResizerRun(resizer, src_data, input_stride, output_ptr, output_size * 3);
        // 释放 resizer
        SimdRelease(resizer);
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    std::string path = "";
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--path=", 7) == 0)
            path = argv[i] + 7;
    }
    if (path.empty()) {
        return 1;
    }

    ImageData img;
    if (!decode_jpeg(path, img)) {
        return 1;
    }

    const int output_size = 224;
    size_t out_bytes = output_size * output_size * 3;
    uint8_t* output = (uint8_t*)aligned_malloc(out_bytes, 64);

    srand(1234);

    // warm up (not timed)
    RandomResizedCrop(img.data, img.width, img.height, img.stride, output, output_size);

    // timing start
    auto t0 = std::chrono::high_resolution_clock::now();
    int8_t result = 0;
    for (int i = 0; i < 1000; ++i) {
        RandomResizedCrop(img.data, img.width, img.height, img.stride, output, output_size);
        result ^= output[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;

    std::cout << ms << std::endl;
    std::cout << (int)result << std::endl;

    aligned_free(output);
    aligned_free(img.data);
    return 0;
}

```



### 方案B

```c++
/**
 * @file b.cpp
 * @brief Expert GLM's RandomResizedCrop Implementation using Simd C API
 * @details 使用 Simd C API 实现零拷贝裁剪和 Resizer 缓存
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <Simd/SimdLib.h>
#include <turbojpeg.h>

// 参数常量
constexpr int OUTPUT_SIZE = 224;
constexpr float SCALE_MIN = 0.08f;
constexpr float SCALE_MAX = 1.0f;
constexpr float RATIO_MIN = 0.75f;
constexpr float RATIO_MAX = 1.3333333f;
constexpr int ALIGNMENT = 64;
constexpr int NUM_ITERATIONS = 1000;

/**
 * @brief 计算对齐后的大小
 */
inline size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief 读取文件到vector
 */
std::vector<uint8_t> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::vector<uint8_t>();
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return std::vector<uint8_t>();
    }
    return buffer;
}

/**
 * @brief 使用 turbojpeg 解码 JPEG
 */
bool decode_jpeg(const std::vector<uint8_t>& jpeg_data, uint8_t*& data,
                 int& width, int& height, size_t& stride) {
    tjhandle tjInstance = tjInitDecompress();
    if (!tjInstance) {
        return false;
    }

    int jpegSubsamp, jpegColorspace;
    if (tjDecompressHeader3(tjInstance, jpeg_data.data(), jpeg_data.size(),
                            &width, &height, &jpegSubsamp, &jpegColorspace) != 0) {
        tjDestroy(tjInstance);
        return false;
    }

    stride = align_size(width * 3, ALIGNMENT);
    size_t buffer_size = stride * height;

#ifdef _MSC_VER
    data = static_cast<uint8_t*>(_aligned_malloc(buffer_size, ALIGNMENT));
#else
    posix_memalign((void**)&data, ALIGNMENT, buffer_size);
#endif

    if (tjDecompress2(tjInstance, jpeg_data.data(), jpeg_data.size(), data,
                      width, static_cast<int>(stride), height,
                      TJPF_RGB, TJFLAG_FASTDCT) != 0) {
#ifdef _MSC_VER
        _aligned_free(data);
#else
        free(data);
#endif
        data = nullptr;
        tjDestroy(tjInstance);
        return false;
    }

    tjDestroy(tjInstance);
    return true;
}

/**
 * @brief 随机浮点数生成
 */
inline float random_float(float min, float max) {
    return min + (max - min) * (rand() / (float)RAND_MAX);
}

/**
 * @brief 裁剪参数结构
 */
struct CropParams {
    int x, y, w, h;
};

/**
 * @brief 生成随机裁剪参数
 */
CropParams generate_random_crop(int src_w, int src_h) {
    float scale = random_float(SCALE_MIN, SCALE_MAX);
    float ratio = random_float(RATIO_MIN, RATIO_MAX);

    int area = static_cast<int>(src_w * src_h * scale);
    int w = static_cast<int>(std::sqrt(static_cast<float>(area) * ratio));
    int h = static_cast<int>(std::sqrt(static_cast<float>(area) / ratio));

    for (int attempt = 0; attempt < 10; ++attempt) {
        if (w <= src_w && h <= src_h && w > 0 && h > 0) break;
        ratio = random_float(RATIO_MIN, RATIO_MAX);
        w = static_cast<int>(std::sqrt(static_cast<float>(area) * ratio));
        h = static_cast<int>(std::sqrt(static_cast<float>(area) / ratio));
    }

    if (w > src_w) w = src_w;
    if (h > src_h) h = src_h;
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    int x = rand() % (src_w - w + 1);
    int y = rand() % (src_h - h + 1);

    return {x, y, w, h};
}

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    if (argc < 3 || strcmp(argv[1], "--path") != 0) {
        return 1;
    }
    const char* input_path = argv[2];

    // 1. 加载并解码 JPEG
    std::vector<uint8_t> jpeg_buffer = read_file(input_path);
    if (jpeg_buffer.empty()) {
        return 1;
    }

    uint8_t* src_data = nullptr;
    int src_width, src_height;
    size_t src_stride;

    if (!decode_jpeg(jpeg_buffer, src_data, src_width, src_height, src_stride)) {
        return 1;
    }

    // 2. 预分配输出内存
    const size_t dst_stride = align_size(OUTPUT_SIZE * 3, ALIGNMENT);
    const size_t dst_size = dst_stride * OUTPUT_SIZE;

#ifdef _MSC_VER
    uint8_t* dst_data = static_cast<uint8_t*>(_aligned_malloc(dst_size, ALIGNMENT));
#else
    uint8_t* dst_data = nullptr;
    posix_memalign((void**)&dst_data, ALIGNMENT, dst_size);
#endif

    // 3. 预计算随机参数
    std::vector<CropParams> crop_params_list;
    crop_params_list.reserve(NUM_ITERATIONS);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        crop_params_list.push_back(generate_random_crop(src_width, src_height));
    }

    // 4. 预热
    {
        CropParams p = crop_params_list[0];
        void* resizer = SimdResizerInit(p.w, p.h, OUTPUT_SIZE, OUTPUT_SIZE, 3,
                                        SimdResizeChannelByte, SimdResizeMethodBilinear);
        if (resizer) {
            const uint8_t* crop_src = src_data + p.y * src_stride + p.x * 3;
            SimdResizerRun(resizer, crop_src, src_stride, dst_data, dst_stride);
            SimdRelease(resizer);
        }
    }

    // 5. 开始计时循环
    int8_t result = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto& p = crop_params_list[i];

        void* resizer = SimdResizerInit(p.w, p.h, OUTPUT_SIZE, OUTPUT_SIZE, 3,
                                        SimdResizeChannelByte, SimdResizeMethodBilinear);
        if (resizer) {
            const uint8_t* crop_src = src_data + p.y * src_stride + p.x * 3;
            SimdResizerRun(resizer, crop_src, src_stride, dst_data, dst_stride);
            SimdRelease(resizer);
        }

        result ^= dst_data[0];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double avg_ms = duration.count() / 1000.0 / NUM_ITERATIONS;

    std::cout << avg_ms << std::endl;
    std::cout << static_cast<int>(result) << std::endl;

    // 清理内存
#ifdef _MSC_VER
    _aligned_free(src_data);
    _aligned_free(dst_data);
#else
    free(src_data);
    free(dst_data);
#endif

    return 0;
}

```



### 方案C

```c++
/**
 * @file d.cpp
 * @brief Expert OP's RandomResizedCrop Implementation using Simd C API
 * @details 使用 Simd C API (SimdResizerInit/Run)，实现预分配工作区和Resizer缓存机制
 *          支持零拷贝裁剪、64字节内存对齐、turbojpeg解码
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>
#include <iostream>

// Simd 库头文件
#include <Simd/SimdLib.h>

// libjpeg-turbo 头文件
#include <turbojpeg.h>

// ============================================================================
// 跨平台对齐内存分配
// ============================================================================
#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

// ============================================================================
// 常量定义
// ============================================================================
constexpr size_t SIMD_ALIGNMENT = 64;  // AVX2 需要 32 字节，预留 64 字节以备 AVX-512
constexpr int NUM_CHANNELS = 3;        // RGB/BGR
constexpr int MAX_ATTEMPTS = 10;       // RandomResizedCrop 最大尝试次数

// ============================================================================
// 内存对齐辅助函数
// ============================================================================
inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline size_t calculate_aligned_stride(int width, int channels, size_t alignment = SIMD_ALIGNMENT) {
    return align_up(static_cast<size_t>(width) * channels, alignment);
}

// ============================================================================
// 预分配的工作区管理器 - 避免运行时内存分配
// ============================================================================
class PreallocatedWorkspace {
private:
    uint8_t* crop_buffer_;      // 裁剪后的临时缓冲区
    size_t crop_buffer_size_;
    size_t crop_stride_;

    // Resizer 缓存 - 避免重复创建
    void* cached_resizer_;
    size_t cached_src_w_, cached_src_h_;
    size_t cached_dst_w_, cached_dst_h_;

public:
    PreallocatedWorkspace(int max_input_width, int max_input_height,
                          int output_size) {
        // 预分配最大可能的裁剪缓冲区
        crop_stride_ = calculate_aligned_stride(max_input_width, NUM_CHANNELS);
        crop_buffer_size_ = crop_stride_ * max_input_height;
        crop_buffer_ = static_cast<uint8_t*>(
            ALIGNED_ALLOC(SIMD_ALIGNMENT, crop_buffer_size_));

        if (!crop_buffer_) {
            exit(1);
        }

        // 初始化 resizer 缓存
        cached_resizer_ = nullptr;
        cached_src_w_ = cached_src_h_ = 0;
        cached_dst_w_ = cached_dst_h_ = 0;
    }

    ~PreallocatedWorkspace() {
        if (crop_buffer_) {
            ALIGNED_FREE(crop_buffer_);
        }
        if (cached_resizer_) {
            SimdRelease(cached_resizer_);
        }
    }

    // 禁止拷贝
    PreallocatedWorkspace(const PreallocatedWorkspace&) = delete;
    PreallocatedWorkspace& operator=(const PreallocatedWorkspace&) = delete;

    uint8_t* get_crop_buffer() { return crop_buffer_; }
    size_t get_crop_stride() const { return crop_stride_; }

    // 获取或创建 Resizer（缓存机制）
    void* get_resizer(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h) {
        // 如果尺寸匹配，复用缓存的 resizer
        if (cached_resizer_ &&
            cached_src_w_ == src_w && cached_src_h_ == src_h &&
            cached_dst_w_ == dst_w && cached_dst_h_ == dst_h) {
            return cached_resizer_;
        }

        // 释放旧的 resizer
        if (cached_resizer_) {
            SimdRelease(cached_resizer_);
            cached_resizer_ = nullptr;
        }

        // 创建新的 resizer
        cached_resizer_ = SimdResizerInit(
            src_w, src_h, dst_w, dst_h,
            NUM_CHANNELS,
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );

        cached_src_w_ = src_w;
        cached_src_h_ = src_h;
        cached_dst_w_ = dst_w;
        cached_dst_h_ = dst_h;

        return cached_resizer_;
    }
};

// ============================================================================
// RandomResizedCrop 参数结构
// ============================================================================
struct RandomResizedCropParams {
    float scale_min = 0.08f;
    float scale_max = 1.0f;
    float ratio_min = 0.75f;
    float ratio_max = 1.3333333f;
    int output_size = 224;  // 输出正方形边长
};

// ============================================================================
// 快速随机数生成器封装
// ============================================================================
class FastRandom {
private:
    unsigned int seed_;

public:
    explicit FastRandom(unsigned int seed = 42) : seed_(seed) {
        srand(seed);
    }

    // 生成 [0, 1) 的浮点数
    float uniform() {
        return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    }

    // 生成 [min, max) 的浮点数
    float uniform(float min_val, float max_val) {
        return min_val + uniform() * (max_val - min_val);
    }
};

// ============================================================================
// RandomResizedCrop 核心实现
// ============================================================================
class RandomResizedCrop {
private:
    PreallocatedWorkspace& workspace_;
    RandomResizedCropParams params_;
    FastRandom rng_;

    // 输出缓冲区信息
    size_t output_stride_;

public:
    RandomResizedCrop(PreallocatedWorkspace& workspace,
                      const RandomResizedCropParams& params = {})
        : workspace_(workspace)
        , params_(params)
        , rng_(42)
    {
        output_stride_ = calculate_aligned_stride(params_.output_size, NUM_CHANNELS);
    }

    // 计算裁剪区域 - 返回是否成功
    bool get_crop_params(int input_width, int input_height,
                         int& crop_x, int& crop_y,
                         int& crop_w, int& crop_h)
    {
        const float area = static_cast<float>(input_width * input_height);
        const float log_ratio_min = std::log(params_.ratio_min);
        const float log_ratio_max = std::log(params_.ratio_max);

        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            // 随机选择目标面积
            const float target_area = area * rng_.uniform(params_.scale_min, params_.scale_max);

            // 随机选择宽高比（对数均匀分布）
            const float aspect_ratio = std::exp(rng_.uniform(log_ratio_min, log_ratio_max));

            // 计算裁剪尺寸
            int w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
            int h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));

            // 检查是否在有效范围内
            if (w > 0 && w <= input_width && h > 0 && h <= input_height) {
                // 随机选择裁剪起始位置
                crop_x = static_cast<int>(rng_.uniform() * (input_width - w + 1));
                crop_y = static_cast<int>(rng_.uniform() * (input_height - h + 1));
                crop_w = w;
                crop_h = h;
                return true;
            }
        }

        // 回退策略：中心裁剪
        const float in_ratio = static_cast<float>(input_width) / input_height;
        if (in_ratio < params_.ratio_min) {
            crop_w = input_width;
            crop_h = static_cast<int>(std::round(input_width / params_.ratio_min));
        } else if (in_ratio > params_.ratio_max) {
            crop_h = input_height;
            crop_w = static_cast<int>(std::round(input_height * params_.ratio_max));
        } else {
            crop_w = input_width;
            crop_h = input_height;
        }
        crop_x = (input_width - crop_w) / 2;
        crop_y = (input_height - crop_h) / 2;
        return true;
    }

    // 执行 RandomResizedCrop
    // input_ptr: 输入图像数据（RGB/BGR，已对齐）
    // output_ptr: 输出图像数据（需预分配，已对齐）
    // input_width, input_height: 输入图像尺寸
    // input_stride: 输入图像行步长
    void operator()(const uint8_t* input_ptr,
                    uint8_t* output_ptr,
                    int input_width,
                    int input_height,
                    size_t input_stride)
    {
        // Step 1: 计算裁剪参数
        int crop_x, crop_y, crop_w, crop_h;
        get_crop_params(input_width, input_height, crop_x, crop_y, crop_w, crop_h);

        // Step 2: 获取裁剪区域的起始指针
        const uint8_t* crop_src = input_ptr + crop_y * input_stride + crop_x * NUM_CHANNELS;

        // Step 3: 获取或创建 Resizer
        void* resizer = workspace_.get_resizer(
            crop_w, crop_h,
            params_.output_size, params_.output_size
        );

        // Step 4: 直接 Resize（从裁剪区域到输出）
        // Simd 的 Resizer 支持非连续内存（通过 stride 参数）
        SimdResizerRun(
            resizer,
            crop_src, input_stride,      // 源：裁剪区域（使用原始 stride）
            output_ptr, output_stride_   // 目标：输出缓冲区
        );
    }

    size_t get_output_stride() const { return output_stride_; }
    int get_output_size() const { return params_.output_size; }
};

// ============================================================================
// JPEG 解码器（使用 libjpeg-turbo）
// ============================================================================
class JpegDecoder {
private:
    tjhandle handle_;
    uint8_t* decoded_buffer_;
    int width_, height_;
    size_t stride_;
    size_t buffer_size_;

public:
    JpegDecoder() : handle_(nullptr), decoded_buffer_(nullptr),
                    width_(0), height_(0), stride_(0), buffer_size_(0) {
        handle_ = tjInitDecompress();
        if (!handle_) {
            exit(1);
        }
    }

    ~JpegDecoder() {
        if (decoded_buffer_) {
            ALIGNED_FREE(decoded_buffer_);
        }
        if (handle_) {
            tjDestroy(handle_);
        }
    }

    // 禁止拷贝
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;

    bool decode(const char* filepath) {
        // 读取文件
        FILE* fp = fopen(filepath, "rb");
        if (!fp) {
            return false;
        }

        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        uint8_t* jpeg_buffer = static_cast<uint8_t*>(malloc(file_size));
        if (!jpeg_buffer) {
            fclose(fp);
            return false;
        }

        size_t read_size = fread(jpeg_buffer, 1, file_size, fp);
        fclose(fp);

        if (static_cast<long>(read_size) != file_size) {
            free(jpeg_buffer);
            return false;
        }

        // 获取 JPEG 信息
        int subsamp, colorspace;
        if (tjDecompressHeader3(handle_, jpeg_buffer, file_size,
                                &width_, &height_, &subsamp, &colorspace) != 0) {
            free(jpeg_buffer);
            return false;
        }

        // 计算对齐的 stride 并分配缓冲区
        stride_ = calculate_aligned_stride(width_, NUM_CHANNELS);
        buffer_size_ = stride_ * height_;

        if (decoded_buffer_) {
            ALIGNED_FREE(decoded_buffer_);
        }
        decoded_buffer_ = static_cast<uint8_t*>(
            ALIGNED_ALLOC(SIMD_ALIGNMENT, buffer_size_));

        if (!decoded_buffer_) {
            free(jpeg_buffer);
            return false;
        }

        // 解码 JPEG（使用 RGB 格式）
        if (tjDecompress2(handle_, jpeg_buffer, file_size,
                          decoded_buffer_, width_, static_cast<int>(stride_), height_,
                          TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC) != 0) {
            free(jpeg_buffer);
            return false;
        }

        free(jpeg_buffer);

        return true;
    }

    const uint8_t* data() const { return decoded_buffer_; }
    int width() const { return width_; }
    int height() const { return height_; }
    size_t stride() const { return stride_; }
};

// ============================================================================
// 命令行参数解析
// ============================================================================
struct Args {
    std::string path;
    int output_size = 224;
    int iterations = 1000;

    bool parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--path" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--size" && i + 1 < argc) {
                output_size = std::atoi(argv[++i]);
            } else if (arg == "--iter" && i + 1 < argc) {
                iterations = std::atoi(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                printf("Usage: %s --path <input.jpg> [--size <output_size>] [--iter <iterations>]\n",
                       argv[0]);
                printf("  --path   Path to input JPEG file (required)\n");
                printf("  --size   Output size (default: 224)\n");
                printf("  --iter   Number of iterations (default: 1000)\n");
                return false;
            }
        }

        if (path.empty()) {
            fprintf(stderr, "Error: --path is required\n");
            return false;
        }
        return true;
    }
};

// ============================================================================
// 主程序
// ============================================================================
int main(int argc, char* argv[]) {
    // 解析命令行参数
    Args args;
    if (!args.parse(argc, argv)) {
        return 1;
    }

    // Step 1: 解码 JPEG
    JpegDecoder decoder;
    if (!decoder.decode(args.path.c_str())) {
        return 1;
    }

    // Step 2: 创建预分配工作区
    PreallocatedWorkspace workspace(decoder.width(), decoder.height(), args.output_size);

    // Step 3: 创建 RandomResizedCrop 处理器
    RandomResizedCropParams params;
    params.output_size = args.output_size;
    RandomResizedCrop cropper(workspace, params);

    // Step 4: 预分配输出缓冲区
    size_t output_stride = cropper.get_output_stride();
    size_t output_buffer_size = output_stride * args.output_size;
    uint8_t* output_buffer = static_cast<uint8_t*>(
        ALIGNED_ALLOC(SIMD_ALIGNMENT, output_buffer_size));

    if (!output_buffer) {
        return 1;
    }

    // Step 5: 预热（避免首次运行的额外开销）
    for (int i = 0; i < 10; ++i) {
        cropper(decoder.data(), output_buffer,
                decoder.width(), decoder.height(), decoder.stride());
    }

    // Step 6: 基准测试
    int8_t result = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < args.iterations; ++i) {
        cropper(decoder.data(), output_buffer,
                decoder.width(), decoder.height(), decoder.stride());

        result ^= static_cast<int8_t>(output_buffer[0]);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Step 7: 计算并输出结果
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double total_ms = duration.count() / 1000.0;
    double avg_ms = total_ms / args.iterations;

    std::cout << avg_ms << std::endl;
    std::cout << static_cast<int>(result) << std::endl;

    // 清理
    ALIGNED_FREE(output_buffer);

    return 0;
}

```





