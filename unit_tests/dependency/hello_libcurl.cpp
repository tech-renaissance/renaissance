#include <iostream>
#include <string>
#include <curl/curl.h> // 必须包含 libcurl 的头文件

/**
 * 回调函数：用于处理 libcurl 接收到的数据。
 * 当服务器返回数据时，libcurl 会多次调用此函数。
 *
 * @param contents 指向接收到的数据的指针
 * @param size     数据块的大小 (总是 1)
 * @param nmemb    数据块的数量 (总字节数 = size * nmemb)
 * @param userp    用户自定义数据的指针 (在我们这里，它指向我们传递的 std::string)
 * @return         返回处理的字节数，如果与传递的字节数不作为，libcurl 会认为出错
 */
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t totalSize = size * nmemb;
    std::string *str = (std::string *)userp;
    
    // 将接收到的数据追加到我们的 string 中
    str->append((char*)contents, totalSize);
    
    return totalSize;
}

int main() {
    // 1. 全局初始化 (通常在程序开始时做一次)
    CURLcode globalRes = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (globalRes != CURLE_OK) {
        std::cerr << "Global init failed." << std::endl;
        return 1;
    }

    // 2. 初始化 easy handle (用于单个连接会话)
    CURL *curl = curl_easy_init();
    std::string readBuffer; // 用于存储下载的内容

    if(curl) {
        std::cout << "Initializing curl..." << std::endl;

        // 3. 设置选项 (Configuration)
        
        // 设置目标 URL (这里使用 example.com 作为测试)
        curl_easy_setopt(curl, CURLOPT_URL, "http://example.com");

        // 设置写数据的回调函数 (否则 curl 默认会打印到 stdout)
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);

        // 设置传给回调函数的 "userp" 指针 (即我们要写入的 string 变量地址)
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        // 设置超时时间 (可选，防止网络卡死，单位为秒)
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

        // 如果你是请求 HTTPS，且在开发环境没有配置好证书路径，
        // 可以取消注释下面两行来跳过 SSL 验证 (生产环境不建议!)
        // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        // 4. 执行请求
        std::cout << "Performing HTTP GET request..." << std::endl;
        CURLcode res = curl_easy_perform(curl);

        // 5. 检查执行结果
        if(res != CURLE_OK) {
            // 请求失败，打印错误信息 (英文)
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            // 请求成功
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            
            std::cout << "Request successful!" << std::endl;
            std::cout << "HTTP Response Code: " << response_code << std::endl;
            std::cout << "Data received: " << readBuffer.length() << " bytes" << std::endl;
            
            // 打印前100个字符作为预览
            std::cout << "Content Preview:\n" << "--------------------------------\n";
            if (readBuffer.length() > 0) {
                 // 简单截断打印，防止太长刷屏
                std::cout << readBuffer.substr(0, 100) << (readBuffer.length() > 100 ? "..." : "") << std::endl;
            }
            std::cout << "--------------------------------" << std::endl;
        }

        // 6. 清理 easy handle
        curl_easy_cleanup(curl);
    } else {
        std::cerr << "Failed to initialize curl easy handle." << std::endl;
    }

    // 7. 全局清理
    curl_global_cleanup();

    return 0;
}