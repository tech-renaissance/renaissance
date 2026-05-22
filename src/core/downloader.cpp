/**
 * @file downloader.cpp
 * @brief 下载器类实现
 * @details 基于libcurl实现的文件下载器，支持主URL和备用URL，自动创建目录，支持文件覆盖控制
 * @version 3.6.12
 * @date 2025-12-28
 * @author 技术觉醒团队
 * @note 依赖项: libcurl, std::filesystem
 * @note 所属系列: base
 */

#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/downloader.h"

#include <curl/curl.h>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <functional>

namespace tr {

namespace fs = std::filesystem;

//==============================================================================
// 构造与析构
//==============================================================================

Downloader::Downloader()
    : url_()
    , spare_url_()
    , file_already_exists_(false)
    , progress_callback_(nullptr)
{
    // 全局初始化libcurl（仅第一次调用时生效）
    static bool curl_initialized = false;
    if (!curl_initialized) {
        curl_global_init(CURL_GLOBAL_ALL);
        curl_initialized = true;
        LOG_INFO << "libcurl initialized";
    }
}

Downloader::~Downloader() {
    // 注意：不调用curl_global_cleanup()，因为可能有其他Downloader实例
}

//==============================================================================
// 公共接口
//==============================================================================

void Downloader::set_url(const std::string& url, const std::string& spare_url) {
    if (url.empty()) {
        TR_VALUE_ERROR("Primary URL cannot be empty");
    }

    url_ = url;
    spare_url_ = spare_url;

    LOG_INFO << "Downloader URL set to: " << url;
    if (!spare_url.empty()) {
        LOG_INFO << "Downloader spare URL set to: " << spare_url;
    }
}

bool Downloader::download_to(const std::string& dir_name,
                              const std::string& file_name,
                              bool cover) {
    // 重置状态
    file_already_exists_ = false;

    // 检查URL是否已设置
    if (url_.empty()) {
        TR_VALUE_ERROR("URL not set. Call set_url() first");
    }

    // 确定文件名
    std::string final_filename = file_name;
    if (final_filename.empty()) {
        final_filename = extract_filename_from_url(url_);
    }

    // 构建完整路径
    fs::path dir_path(dir_name);
    fs::path full_path = dir_path / final_filename;

    // 规范化路径
    try {
        full_path = fs::absolute(full_path);
        dir_path = full_path.parent_path();
    } catch (const fs::filesystem_error& e) {
        TR_VALUE_ERROR("Failed to resolve path: " << dir_name << "/" << final_filename
                 << ". Error: " << e.what());
    }

    // 创建目录（如果不存在）
    try {
        if (!fs::exists(dir_path)) {
            fs::create_directories(dir_path);
            LOG_INFO << "Created directory: " << dir_path.string();
        }
    } catch (const fs::filesystem_error& e) {
        TR_VALUE_ERROR("Failed to create directory: " << dir_path.string()
                 << ". Error: " << e.what());
    }

    // 检查文件是否已存在
    if (fs::exists(full_path)) {
        if (!cover) {
            file_already_exists_ = true;
            LOG_INFO << "File already exists, skipping download: " << full_path.string();
            return true;
        } else {
            LOG_INFO << "File exists, will overwrite: " << full_path.string();
            fs::remove(full_path);
        }
    }

    // 尝试从主URL下载
    LOG_INFO << "Starting download from: " << url_;
    LOG_INFO << "Target path: " << full_path.string();

    bool success = download_impl(url_, full_path.string());

    // 如果主URL失败且存在备用URL，尝试备用URL
    if (!success && !spare_url_.empty()) {
        LOG_WARN << "Primary URL failed, trying spare URL: " << spare_url_;
        success = download_impl(spare_url_, full_path.string());
    }

    if (success) {
        // 验证文件是否真的创建成功
        if (fs::exists(full_path)) {
            uintmax_t file_size = fs::file_size(full_path);
            LOG_INFO << "Download completed: " << full_path.string()
                     << " (size: " << file_size << " bytes)";
            return true;
        } else {
            TR_VALUE_ERROR("Download reported success but file not found: " << full_path.string());
            return false;
        }
    } else {
        TR_VALUE_ERROR("All download attempts failed for: " << url_);
        return false;
    }
}

bool Downloader::already_exists() const {
    return file_already_exists_;
}

void Downloader::set_progress_callback(std::function<void(size_t, size_t, int)> callback) {
    progress_callback_ = callback;
}

//==============================================================================
// 私有方法
//==============================================================================

std::string Downloader::extract_filename_from_url(const std::string& url) const {
    // 查找最后一个 '/'
    size_t last_slash = url.find_last_of('/');

    // 查找最后一个 '?'（查询参数开始）
    size_t last_question = url.find_last_of('?');

    // 确定文件名的结束位置
    size_t end_pos = (last_question != std::string::npos) ? last_question : std::string::npos;

    // 提取文件名
    if (last_slash != std::string::npos) {
        std::string filename = url.substr(last_slash + 1, end_pos - last_slash - 1);
        if (!filename.empty()) {
            return filename;
        }
    }

    TR_VALUE_ERROR("Cannot extract filename from URL: " << url
             << ". URL does not contain a valid filename component");
}

bool Downloader::download_impl(const std::string& url, const std::string& full_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        TR_VALUE_ERROR("Failed to initialize libcurl easy handle");
        return false;
    }

    // 打开文件
    std::ofstream outfile(full_path, std::ios::binary);
    if (!outfile.is_open()) {
        TR_VALUE_ERROR("Failed to open file for writing: " << full_path);
        curl_easy_cleanup(curl);
        return false;
    }

    // 准备进度数据
    ProgressData progress_data;
    progress_data.user_callback = &progress_callback_;

    // 配置libcurl选项
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, Downloader::write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outfile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);       // 跟随重定向
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);            // 最多5次重定向
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);      // 连接超时30秒
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);            // 总超时5分钟
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);       // 验证SSL证书
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);       // 验证SSL主机
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "renAIssance/3.6.12");  // 用户代理
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);           // 启用进度回调
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, Downloader::progress_callback);  // 进度回调
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress_data);  // 进度数据

    // 禁用信号（多线程安全）
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    // 执行下载
    CURLcode res = curl_easy_perform(curl);

    // 关闭文件
    outfile.close();

    // 检查HTTP响应码
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    // 清理
    curl_easy_cleanup(curl);

    // 处理结果
    if (res != CURLE_OK) {
        LOG_WARN << "Download failed: " << curl_easy_strerror(res);
        // 删除可能部分下载的文件
        if (fs::exists(full_path)) {
            fs::remove(full_path);
        }
        return false;
    }

    if (http_code >= 400) {
        LOG_WARN << "HTTP error: " << http_code;
        // 删除错误文件
        if (fs::exists(full_path)) {
            fs::remove(full_path);
        }
        return false;
    }

    return true;
}

size_t Downloader::write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::ofstream* outfile = static_cast<std::ofstream*>(userp);

    if (outfile->is_open()) {
        outfile->write(static_cast<const char*>(contents), total_size);
        if (outfile->good()) {
            return total_size;
        }
    }

    return 0;  // 写入失败
}

int Downloader::progress_callback(void* clientp,
                                   curl_off_t dltotal,
                                   curl_off_t dlnow,
                                   curl_off_t ultotal,
                                   curl_off_t ulnow) {
    (void)ultotal;  // Unused
    (void)ulnow;     // Unused

    ProgressData* progress = static_cast<ProgressData*>(clientp);

    // 如果不知道总大小，不显示进度
    if (dltotal <= 0) {
        return 0;  // 继续下载
    }

    // 计算进度百分比
    int percent = static_cast<int>((dlnow * 100) / dltotal);

    // 如果用户设置了自定义回调
    if (progress->user_callback && *progress->user_callback) {
        (*progress->user_callback)(static_cast<size_t>(dlnow),
                                   static_cast<size_t>(dltotal),
                                   percent);
    } else {
        // 默认：每10%打印一次进度
        if (percent >= progress->last_reported_percent + 10 ||
            percent == 100) {
            LOG_INFO << "Downloading: " << percent << "%"
                     << " (" << dlnow << " / " << dltotal << " bytes)";
            progress->last_reported_percent = percent;
        }
    }

    return 0;  // 继续下载
}

} // namespace tr
