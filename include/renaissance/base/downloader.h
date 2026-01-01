/**
 * @file downloader.h
 * @brief 下载器类声明
 * @details 基于libcurl实现的文件下载器，支持主URL和备用URL，自动创建目录，支持文件覆盖控制
 * @version 3.6.12
 * @date 2025-12-28
 * @author 技术觉醒团队
 * @note 依赖项: libcurl
 * @note 所属系列: base
 */

#pragma once

#include <string>
#include <filesystem>

namespace tr {

/**
 * @class Downloader
 * @brief 基于libcurl的文件下载器
 *
 * @details
 * 核心特性：
 * - 支持主URL和备用URL自动切换
 * - 自动创建目标目录（递归）
 * - 支持文件重命名
 * - 支持文件覆盖控制
 * - 详细的错误日志记录
 *
 * 使用示例：
 * @code
 * Downloader downloader;
 * downloader.set_url("http://example.com/file.zip", "http://backup.com/file.zip");
 * bool success = downloader.download_to("downloads/", "myfile.zip", false);
 * if (success) {
 *     if (downloader.already_exists()) {
 *         LOG_INFO << "File already exists, skipped download";
 *     } else {
 *         LOG_INFO << "Download completed";
 *     }
 * }
 * @endcode
 */
class Downloader {
public:
    /**
     * @brief 构造函数
     */
    Downloader();

    /**
     * @brief 析构函数
     */
    ~Downloader();

    // 禁止拷贝和移动
    Downloader(const Downloader&) = delete;
    Downloader& operator=(const Downloader&) = delete;
    Downloader(Downloader&&) = delete;
    Downloader& operator=(Downloader&&) = delete;

    /**
     * @brief 设置下载URL（主URL和可选备用URL）
     *
     * @param url 主下载URL
     * @param spare_url 备用下载URL（当主URL失败时使用），空字符串表示无备用URL
     *
     * @note
     * - URL必须指向有效的HTTP/HTTPS地址
     * - 备用URL可以为空字符串，表示无备用
     * - 重复调用会覆盖之前的URL设置
     */
    void set_url(const std::string& url, const std::string& spare_url = "");

    /**
     * @brief 下载文件到指定目录
     *
     * @param dir_name 目标目录路径（相对或绝对路径）
     * @param file_name 保存的文件名（空字符串表示使用URL中的原始文件名）
     * @param cover 是否覆盖已存在的文件（true=覆盖，false=跳过）
     *
     * @return true表示成功（下载完成或文件已存在且cover=false），false表示失败
     *
     * @note
     * - 目录不存在时会自动创建（包括父目录）
     * - file_name为空时，从URL中提取文件名
     * - 文件已存在且cover=false时，返回true，already_exists()返回true
     * - 下载失败时会尝试备用URL（如果设置了）
     * - 所有操作会通过Logger记录
     *
     * @throws ValueError 如果未设置URL或URL为空
     * @throws ValueError 如果无法从URL提取文件名且file_name为空
     */
    bool download_to(const std::string& dir_name,
                     const std::string& file_name = "",
                     bool cover = false);

    /**
     * @brief 查询最后一次download_to是否因文件已存在而跳过
     *
     * @return true表示文件已存在且cover=false，跳过下载；false表示已下载或其他情况
     *
     * @note
     * - 仅在download_to返回true时有意义
     * - 每次调用download_to会重置此状态
     */
    bool already_exists() const;

private:
    std::string url_;               ///< 主下载URL
    std::string spare_url_;         ///< 备用下载URL
    bool file_already_exists_;      ///< 标记：文件是否已存在而跳过下载

    /**
     * @brief 从URL中提取文件名
     *
     * @param url URL地址
     * @return 提取的文件名
     *
     * @throws ValueError 如果URL中没有文件名部分
     *
     * @details
     * 示例：
     * - "http://example.com/path/to/file.zip" -> "file.zip"
     * - "http://example.com/file.txt" -> "file.txt"
     * - "http://example.com/" -> 抛出异常
     */
    std::string extract_filename_from_url(const std::string& url) const;

    /**
     * @brief 执行实际下载操作
     *
     * @param url 下载URL
     * @param full_path 完整的本地文件路径
     * @return true表示成功，false表示失败
     *
     * @details
     * - 使用libcurl下载文件
     * - 失败时记录详细错误信息到Logger
     */
    bool download_impl(const std::string& url, const std::string& full_path);

    /**
     * @brief libcurl写入回调函数（静态方法）
     *
     * @param contents 接收到的数据指针
     * @param size 每个元素的大小
     * @param nmemb 元素数量
     * @param userp 用户指针（指向std::ofstream）
     * @return 实际写入的字节数
     *
     * @details
     * - libcurl接收到数据时会调用此函数
     * - 将数据写入到ofstream中
     */
    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace tr
