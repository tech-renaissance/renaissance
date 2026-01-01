# Downloader类使用文档

**版本**: V3.6.12
**日期**: 2025-12-28
**作者**: 技术觉醒团队
**状态**: ✅ 全平台测试通过

---

## 目录

1. [概述](#概述)
2. [核心功能](#核心功能)
3. [API参考](#api参考)
4. [使用示例](#使用示例)
5. [技术细节](#技术细节)
6. [常见问题](#常见问题)

---

## 概述

`Downloader`类是一个基于libcurl实现的文件下载器，为技术觉醒框架提供HTTP/HTTPS文件下载能力。

### 设计目标

- **双URL支持**: 主URL失败时自动切换到备用URL
- **自动目录管理**: 自动创建目标目录（递归）
- **文件名灵活处理**: 支持自动提取URL中的文件名或自定义重命名
- **覆盖控制**: 可选择跳过已存在文件或强制覆盖
- **详细日志**: 完整的下载过程日志记录
- **异常安全**: 统一的异常处理机制

### 文件位置

| 类 | 头文件 | 源文件 |
|----|--------|---------|
| Downloader | `include/renaissance/base/downloader.h` | `src/base/downloader.cpp` |

### 依赖项

- **libcurl**: HTTP/HTTPS客户端库
- **std::filesystem**: C++17文件系统库
- **Logger**: 框架日志系统
- **TRException**: 框架异常系统

---

## 核心功能

### 1. 双URL自动切换

```cpp
Downloader downloader;
downloader.set_url(
    "https://primary-url.com/file.zip",           // 主URL
    "https://backup-url.com/file.zip"            // 备用URL（可选）
);
```

**工作流程**:
1. 尝试从主URL下载
2. 如果主URL失败，自动切换到备用URL
3. 记录详细的切换日志

### 2. 自动目录创建

```cpp
// 即使目录不存在，也会自动创建
downloader.download_to("path/to/deep/nested/dir", "file.zip");
```

**行为**:
- 递归创建所有父目录
- 如果目录已存在，直接使用
- 创建失败时抛出 `ValueError` 异常

### 3. 文件名处理

#### 方式1：自动提取文件名
```cpp
// 从URL中自动提取文件名: "file.zip"
downloader.set_url("https://example.com/path/to/file.zip");
downloader.download_to("downloads/", "", true);  // file_name为空
```

#### 方式2：自定义文件名
```cpp
// 下载后重命名为 "my_file.zip"
downloader.download_to("downloads/", "my_file.zip", true);
```

### 4. 覆盖控制

```cpp
// cover=false: 文件已存在时跳过下载
bool success = downloader.download_to("downloads/", "", false);
if (downloader.already_exists()) {
    // 文件已存在，未下载
}

// cover=true: 强制覆盖已存在文件
success = downloader.download_to("downloads/", "", true);
```

---

## API参考

### 类定义

```cpp
namespace tr {

class Downloader {
public:
    Downloader();
    ~Downloader();

    // 禁止拷贝和移动
    Downloader(const Downloader&) = delete;
    Downloader& operator=(const Downloader&) = delete;
    Downloader(Downloader&&) = delete;
    Downloader& operator=(Downloader&&) = delete;

    // 核心接口
    void set_url(const std::string& url, const std::string& spare_url = "");
    bool download_to(const std::string& dir_name,
                     const std::string& file_name = "",
                     bool cover = false);
    bool already_exists() const;

private:
    // 私有方法
    std::string extract_filename_from_url(const std::string& url) const;
    bool download_impl(const std::string& url, const std::string& full_path);
    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace tr
```

### 方法详解

#### `set_url()`

```cpp
void set_url(const std::string& url, const std::string& spare_url = "");
```

**参数**:
- `url`: 主下载URL（必填，不能为空）
- `spare_url`: 备用下载URL（可选，为空表示无备用URL）

**异常**:
- `ValueError`: 如果主URL为空

**说明**:
- 可以重复调用以更新URL
- URL必须是HTTP或HTTPS协议
- 自动记录日志

#### `download_to()`

```cpp
bool download_to(const std::string& dir_name,
                 const std::string& file_name = "",
                 bool cover = false);
```

**参数**:
- `dir_name`: 目标目录路径（相对或绝对路径）
- `file_name`: 保存的文件名（空字符串表示从URL自动提取）
- `cover`: 是否覆盖已存在文件（true=覆盖，false=跳过）

**返回值**:
- `true`: 下载成功或文件已存在（且`cover=false`）
- `false`: 下载失败

**异常**:
- `ValueError`: URL未设置、目录创建失败、文件名提取失败

**行为**:
1. 检查URL是否已设置
2. 确定最终文件名
3. 创建目标目录（如不存在）
4. 检查文件是否已存在
5. 尝试从主URL下载
6. 如果主URL失败且有备用URL，尝试备用URL
7. 验证文件是否成功创建

#### `already_exists()`

```cpp
bool already_exists() const;
```

**返回值**:
- `true`: 最后一次`download_to`因文件已存在而跳过下载
- `false`: 文件是新下载的或未调用过`download_to`

**使用场景**:
- 区分"下载成功"和"文件已存在跳过"
- 向用户报告实际行为

---

## 使用示例

### 示例1：基本下载

```cpp
#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    Downloader downloader;

    // 设置URL
    downloader.set_url("https://example.com/data.zip");

    // 下载到当前目录
    bool success = downloader.download_to(".", "data.zip", true);

    if (success) {
        std::cout << "Download completed!\n";
    } else {
        std::cout << "Download failed!\n";
    }

    return 0;
}
```

### 示例2：双URL下载（主+备用）

```cpp
Downloader downloader;

// 主URL: 官方源（可能速度慢或不可用）
// 备用URL: 国内镜像（速度快）
downloader.set_url(
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "https://tech-renaissance.cn/download/mnist/train-images-idx3-ubyte.gz"
);

bool success = downloader.download_to("datasets/", "", false);

if (success) {
    if (downloader.already_exists()) {
        LOG_INFO << "File already exists, skipped";
    } else {
        LOG_INFO << "Download completed successfully";
    }
}
```

### 示例3：自动创建目录+自动提取文件名

```cpp
Downloader downloader;

downloader.set_url("https://example.com/path/to/myfile.zip");

// 自动创建 deep/nested/dir 目录
// 自动从URL提取文件名为 "myfile.zip"
downloader.download_to("deep/nested/dir", "", true);
```

### 示例4：强制覆盖已存在文件

```cpp
Downloader downloader;

downloader.set_url("https://example.com/data.bin");

// 强制覆盖（即使文件已存在）
downloader.download_to("data/", "data.bin", true);
```

### 示例5：跳过已存在文件

```cpp
Downloader downloader;

downloader.set_url("https://example.com/large_file.zip");

// 如果文件已存在，跳过下载（节省带宽）
bool success = downloader.download_to("downloads/", "", false);

if (success && downloader.already_exists()) {
    std::cout << "File already exists, download skipped\n";
}
```

### 示例6：异常处理

```cpp
Downloader downloader;

try {
    downloader.set_url("https://example.com/file.zip");
    bool success = downloader.download_to("downloads/", "", true);

    if (success) {
        LOG_INFO << "Download succeeded";
    } else {
        LOG_WARN << "Download failed (but no exception thrown)";
    }
} catch (const ValueError& e) {
    LOG_ERROR << "ValueError: " << e.what();
} catch (const TRException& e) {
    LOG_ERROR << "Exception: " << e.what();
}
```

---

## 技术细节

### libcurl配置

`Downloader`类使用以下libcurl选项：

| 选项 | 值 | 说明 |
|------|-----|------|
| `CURLOPT_URL` | URL字符串 | 目标URL |
| `CURLOPT_WRITEFUNCTION` | `write_callback` | 写入回调函数 |
| `CURLOPT_FOLLOWLOCATION` | `1L` | 跟随HTTP重定向（最多5次） |
| `CURLOPT_MAXREDIRS` | `5L` | 最大重定向次数 |
| `CURLOPT_CONNECTTIMEOUT` | `30L` | 连接超时30秒 |
| `CURLOPT_TIMEOUT` | `300L` | 总超时5分钟 |
| `CURLOPT_SSL_VERIFYPEER` | `1L` | 验证SSL证书 |
| `CURLOPT_SSL_VERIFYHOST` | `2L` | 验证SSL主机 |
| `CURLOPT_USERAGENT` | `"renAIssance/3.6.12"` | 用户代理标识 |
| `CURLOPT_NOSIGNAL` | `1L` | 禁用信号（多线程安全） |

### 线程安全

- ✅ **全局初始化**: 使用静态变量确保`curl_global_init`只调用一次
- ✅ **信号安全**: `CURLOPT_NOSIGNAL`确保多线程环境安全
- ❌ **实例方法**: 单个`Downloader`对象不是线程安全的（不要多线程共享）
- ✅ **多实例**: 多个线程可以各自创建独立的`Downloader`对象

### 内存管理

- 使用`std::ofstream`进行文件写入（RAII自动关闭）
- libcurl的`CURL*`句柄在函数结束时自动清理
- 下载失败时自动删除部分下载的文件

### 错误处理

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| **网络错误** | DNS解析失败、连接超时 | 记录WARN日志，返回false，尝试备用URL |
| **HTTP错误** | 404、500等HTTP状态码 | 记录WARN日志，返回false，尝试备用URL |
| **文件系统错误** | 目录创建失败、权限不足 | 抛出`ValueError`异常 |
| **参数错误** | URL为空、文件名提取失败 | 抛出`ValueError`异常 |

### 平台兼容性

| 平台 | 编译状态 | 测试状态 |
|------|---------|---------|
| Windows + MSVC | ✅ | ✅ |
| Linux + GCC | ✅ | ✅ |
| Windows + CUDA | ✅ | ✅ |
| Linux + CUDA | ✅ | ✅ |
| Windows + MUSA | ✅ | ✅ |
| Linux + MUSA | ✅ | ✅ |

---

## 常见问题

### Q1: 为什么下载失败时有时抛异常，有时不抛？

**A**: 取决于错误类型：

| 错误类型 | 是否抛异常 | 原因 |
|---------|-----------|------|
| URL未设置 | ✅ 抛出 | 编程错误，应在开发阶段修复 |
| 目录创建失败 | ✅ 抛出 | 文件系统错误，无法继续 |
| 文件名提取失败 | ✅ 抛出 | URL格式错误 |
| 网络错误 | ❌ 不抛 | 可通过备用URL恢复 |
| HTTP 404/500 | ❌ 不抛 | 可通过备用URL恢复 |

### Q2: 如何判断是下载成功还是文件已存在？

**A**: 使用`already_exists()`方法：

```cpp
bool success = downloader.download_to("data/", "", false);
if (success) {
    if (downloader.already_exists()) {
        std::cout << "File already exists, skipped\n";
    } else {
        std::cout << "Downloaded successfully\n";
    }
}
```

### Q3: 备用URL什么时候会使用？

**A**: 当**且仅当**以下条件同时满足时：

1. 主URL下载失败（网络错误或HTTP错误）
2. 备用URL不为空（`spare_url != ""`）

**不会使用备用URL的情况**：
- 主URL下载成功
- URL未设置（抛出异常）
- 目录创建失败（抛出异常）

### Q4: 如何下载大文件（>1GB）？

**A**: `Downloader`类支持任意大小文件：

```cpp
downloader.set_url("https://example.com/large_file.iso");
downloader.download_to("downloads/", "", true);

// libcurl会自动处理大文件下载
// 超时时间设置为5分钟（300秒）
// 对于更大的文件，可以修改源码中的CURLOPT_TIMEOUT值
```

### Q5: 支持断点续传吗？

**A**: 当前版本不支持断点续传。如果下载中断：
- 部分下载的文件会被自动删除
- 需要重新下载完整文件

**未来计划**: 可添加`Range`头支持断点续传。

### Q6: 支持FTP或其他协议吗？

**A**: libcurl支持多种协议（FTP、SFTP、SMTP等），但当前版本只测试了HTTP/HTTPS。

如需使用FTP，可以尝试：
```cpp
downloader.set_url("ftp://example.com/file.zip");
```

### Q7: 如何查看详细的下载日志？

**A**: 设置Logger级别：

```cpp
// Debug模式：查看所有日志
Logger::instance().set_level(LogLevel::DEBUG);

// 或使用环境变量
TR_LOG_LEVEL=0 ./my_program
```

### Q8: 为什么我的编译时报错"找不到libcurl"?

**A**: 检查以下几点：

1. **确认libcurl已安装**: `vcpkg install curl`

2. **检查CMake配置**: `config/cmake_paths.cmake` 中是否有：
   ```
   TR_USE_LIBCURL ON
   TR_CURL_INCLUDE_DIR "path/to/curl/include"
   TR_CURL_LIBRARY_DIR "path/to/curl/lib"
   ```

3. **重新运行configure**: `python configure.py`

4. **重新编译**: `./build.bat` 或 `./build.sh`

---

## 版本历史

### V3.6.12 (2025-12-28)

**初始实现**：
- ✅ 基于libcurl的HTTP/HTTPS下载
- ✅ 主URL + 备用URL自动切换
- ✅ 自动创建目标目录
- ✅ 文件名自动提取或自定义
- ✅ 覆盖控制（`cover`参数）
- ✅ `already_exists()`状态查询
- ✅ 完整的异常处理和日志记录
- ✅ 全平台编译测试通过（Windows/Linux + CUDA/MUSA）

**测试覆盖**：
- ✅ 基本下载功能
- ✅ 双URL切换机制
- ✅ 覆盖控制
- ✅ 文件名自动提取
- ✅ 自动目录创建
- ✅ 异常处理

**性能基准**：

| 文件大小 | 网络 | 耗时 | 吞吐量 |
|---------|------|------|--------|
| 1.6MB (MNIST) | 国内镜像 | ~0.5s | ~3.2MB/s |
| 1.6MB (MNIST) | AWS S3 | ~2-5s | ~0.3-0.8MB/s |

---

**文档版本**: V1.0
**最后更新**: 2025-12-28
**作者**: 技术觉醒团队
