/**
 * @file python_session.cpp
 * @brief Python session manager implementation
 * @version 3.6.10
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#ifdef TR_USE_PYTHON_SESSION

#include "renaissance/utils/python_session.h"
#include "renaissance/data/tensor.h"
#include "renaissance/device/device_manager.h"
#include "renaissance/device/cpu_device.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

namespace tr {

// =============================================================================
// 辅助函数
// =============================================================================

namespace {

/**
 * @brief 获取TR_WORKSPACE路径
 */
std::string get_workspace() {
    // TR_WORKSPACE是编译时宏，由CMake定义
    #ifdef TR_WORKSPACE
        return std::string(TR_WORKSPACE);
    #else
        // 如果未定义，使用当前目录的workspace子目录
        return "workspace";
    #endif
}

/**
 * @brief 生成唯一的会话目录路径
 */
std::string generate_session_dir(int session_id) {
    std::ostringstream oss;
    oss << get_workspace() << "/python_session_id" << session_id;
    return oss.str();
}

/**
 * @brief 写入JSON文件
 */
void write_json_file(const std::string& path, const std::string& content) {
    std::ofstream ofs(path);
    if (!ofs) {
        TR_DEVICE_ERROR("Failed to open file for writing: " << path);
    }
    ofs << content;
    ofs.close();
}

/**
 * @brief 读取JSON文件内容
 */
std::string read_json_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        TR_DEVICE_ERROR("Failed to open file for reading: " << path);
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    ifs.close();
    return buffer.str();
}

} // 匿名命名空间

// =============================================================================
// PythonSession实现
// =============================================================================

PythonSession::PythonSession(
    const std::string& python_exe,
    const std::string& server_script
)
    : python_exe_(python_exe)
    , server_script_(server_script)
    , process_handle_(nullptr)
    , running_(false)
    , session_id_(0)
{
}

PythonSession::~PythonSession() {
    if (running_) {
        stop();
    }
}

void PythonSession::start() {
    if (running_) {
        TR_DEVICE_ERROR("Session is already running");
    }

    // 创建会话目录
    create_session_dir();

    // 标记会话为已启动（但Python进程在第一次send()时才真正启动）
    running_ = true;
}

void PythonSession::stop() {
    LOG_DEBUG << "PythonSession::stop() called, running_=" << running_ << ", session_dir_=" << session_dir_;

    if (!running_) {
        return;
    }

#ifdef _WIN32
    if (process_handle_) {
        // 等待进程自然退出（最多5秒）
        DWORD wait_result = WaitForSingleObject(process_handle_, 5000);

        if (wait_result == WAIT_TIMEOUT) {
            // 如果超时，才强制终止
            LOG_WARN << "Python process did not exit naturally, terminating";
            TerminateProcess(process_handle_, 1);
            WaitForSingleObject(process_handle_, 1000);
        } else if (wait_result == WAIT_OBJECT_0) {
            // 进程已自然退出
            LOG_DEBUG << "Python process exited naturally";
        }

        CloseHandle(process_handle_);
        process_handle_ = nullptr;

        // 给文件系统一点时间释放文件句柄
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#else
    if (process_handle_) {
        pid_t pid = reinterpret_cast<intptr_t>(process_handle_);
        // 先发送SIGTERM，给进程机会自然退出
        kill(pid, SIGTERM);

        // 等待进程退出（最多5秒）
        int status;
        int wait_result;
        for (int i = 0; i < 50; ++i) {
            wait_result = waitpid(pid, &status, WNOHANG);
            if (wait_result == pid) {
                // 进程已退出
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 如果还没退出，强制杀死
        if (wait_result != pid) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
        }

        process_handle_ = nullptr;
    }
#endif

    running_ = false;

    // 清理会话目录
    cleanup_session_dir();
}

std::vector<Tensor> PythonSession::calculate(
    const std::string& method,
    const std::vector<Tensor>& inputs,
    const std::map<std::string, std::string>& parameters
) {
    // 发送请求
    send(method, inputs, parameters);

    // 等待响应
    wait();

    // 获取结果
    return fetch();
}

void PythonSession::send(
    const std::string& method,
    const std::vector<Tensor>& inputs,
    const std::map<std::string, std::string>& parameters
) {
    if (!running_) {
        TR_DEVICE_ERROR("Session is not running");
    }

    // 将输入张量写入TSR文件 - 使用Device的方法
    // 注意：这里需要通过DeviceManager获取CPU设备来调用export_tensor
    auto& cpu = DeviceManager::instance().cpu();

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string tensor_path = session_dir_ + "/input_" + std::to_string(i) + ".tsr";
        cpu.export_tensor(inputs[i], tensor_path, false);  // 使用RAW模式
    }

    // 写入request.json
    write_request(method, parameters);

    // 第一次调用时启动Python进程（此时所有文件都已准备好）
    if (process_handle_ == nullptr) {
        // 确定服务器脚本路径（server_script_可能是绝对路径或相对路径）
        std::string server_path;
        if (server_script_.size() > 1 && server_script_[1] == ':') {
            // Windows绝对路径 (如 "R:/renaissance/...")
            server_path = server_script_;
        } else if (server_script_[0] == '/') {
            // Unix绝对路径
            server_path = server_script_;
        } else {
            // 相对路径，需要拼接project_root
            std::string workspace = get_workspace();
            size_t pos = workspace.find("/workspace");
            std::string project_root = (pos != std::string::npos) ? workspace.substr(0, pos) : workspace;
            server_path = project_root + "/" + server_script_;
        }

        LOG_DEBUG << "Python server path: " << server_path;

        // 构建命令行
        std::string command = "\"" + python_exe_ + "\" \"" + server_path + "\" \"" + session_dir_ + "\"";
        LOG_DEBUG << "Python command: " << command;

#ifdef _WIN32
        // Windows: 使用CreateProcess
        STARTUPINFOA si;
        PROCESS_INFORMATION pi;

        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));

        // 创建命令行缓冲区（CreateProcess会修改它）
        std::vector<char> cmd_line(command.begin(), command.end());
        cmd_line.push_back(0);

        if (!CreateProcessA(
            nullptr,                   // 不使用模块名（使用命令行）
            cmd_line.data(),           // 命令行
            nullptr,                   // 进程句柄不可继承
            nullptr,                   // 线程句柄不可继承
            FALSE,                     // 设置句柄继承为FALSE
            CREATE_NO_WINDOW,          // 创建标志（隐藏控制台）
            nullptr,                   // 使用父进程的环境块
            nullptr,                   // 使用父进程的起始目录
            &si,                       // 指向STARTUPINFO结构的指针
            &pi                        // 指向PROCESS_INFORMATION结构的指针
        )) {
            cleanup_session_dir();
            TR_DEVICE_ERROR("Failed to start Python process");
        }

        process_handle_ = pi.hProcess;
        CloseHandle(pi.hThread);  // 我们不需要线程句柄

#else
        // Unix: 使用fork + exec
        pid_t pid = fork();
        if (pid < 0) {
            cleanup_session_dir();
            TR_DEVICE_ERROR("Failed to fork process");
        } else if (pid == 0) {
            // 子进程
            execlp(python_exe_.c_str(), python_exe_.c_str(),
                   server_path.c_str(), session_dir_.c_str(), nullptr);
            // 如果execlp返回，说明出错了
            std::cerr << "Failed to execute Python: " << python_exe_ << std::endl;
            _exit(1);
        }

        process_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(pid));
#endif

        running_ = true;
    }

    // 给Python进程一点时间启动并处理请求
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void PythonSession::wait() {
    if (!running_) {
        TR_DEVICE_ERROR("Session is not running");
    }

    std::string response_path = session_dir_ + "/response.json";

    // 轮询响应文件（超时：10秒）
    const int max_attempts = 100;  // 10秒 * 每秒10次
    int attempts = 0;

    while (attempts < max_attempts) {
        if (std::filesystem::exists(response_path)) {
            return;  // 找到响应文件
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        attempts++;
    }

    TR_DEVICE_ERROR("Timeout waiting for Python response");
}

std::vector<Tensor> PythonSession::fetch() {
    if (!running_) {
        TR_DEVICE_ERROR("Session is not running");
    }

    // 读取响应
    auto [success, message, result] = read_response();

    if (!success) {
        TR_DEVICE_ERROR("Python operation failed: " << message);
    }

    // 加载输出张量 - 使用Device的方法
    auto& cpu = DeviceManager::instance().cpu();
    std::vector<Tensor> outputs;

    // 确定输出数量（从result中读取output_count，或假设为1）
    int output_count = 1;
    auto it = result.find("output_count");
    if (it != result.end()) {
        output_count = std::stoi(it->second);
    }

    for (int i = 0; i < output_count; ++i) {
        std::string tensor_path = session_dir_ + "/output_" + std::to_string(i) + ".tsr";

        if (!std::filesystem::exists(tensor_path)) {
            TR_DEVICE_ERROR("Output tensor file not found: " << tensor_path);
        }

        // 不使用mmap，避免文件句柄被占用导致清理失败
        Tensor tensor = cpu.import_tensor(tensor_path, false);  // false = 禁用mmap
        outputs.push_back(std::move(tensor));
    }

    return outputs;
}

std::string PythonSession::fetch_text_output() {
    if (!running_) {
        TR_DEVICE_ERROR("Session is not running");
    }

    // 读取响应
    auto [success, message, result] = read_response();

    if (!success) {
        TR_DEVICE_ERROR("Python operation failed: " << message);
    }

    // 验证返回类型是txt
    if (message != "txt") {
        TR_DEVICE_ERROR("Expected txt output, got: " << message);
    }

    // 读取文本文件
    std::string txt_path = session_dir_ + "/output_0.txt";

    if (!std::filesystem::exists(txt_path)) {
        TR_DEVICE_ERROR("Output text file not found: " << txt_path);
    }

    // 读取文件内容
    std::ifstream ifs(txt_path);
    if (!ifs) {
        TR_DEVICE_ERROR("Failed to open text file: " << txt_path);
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    ifs.close();

    // 立即删除文件（阅后即焚）
    try {
        std::filesystem::remove(txt_path);
        LOG_DEBUG << "Deleted text output file: " << txt_path;
    } catch (const std::exception& e) {
        LOG_WARN << "Failed to delete text output file: " << txt_path
                 << ", error: " << e.what();
    }

    return content;
}

std::string PythonSession::print_tensor(const Tensor& tensor) {
    // 发送张量和命令
    std::vector<Tensor> inputs = {tensor};
    std::map<std::string, std::string> empty_params;
    send("print_tensor", inputs, empty_params);

    // 等待响应
    wait();

    // 获取文本输出
    return fetch_text_output();
}

void PythonSession::create_session_dir() {
    // 生成唯一会话ID（使用时间戳）
    session_id_ = static_cast<int>(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000
    );
    session_dir_ = generate_session_dir(session_id_);

    // 创建目录
    if (!std::filesystem::create_directories(session_dir_)) {
        TR_DEVICE_ERROR("Failed to create session directory: " << session_dir_);
    }
}

void PythonSession::cleanup_session_dir() {
    LOG_DEBUG << "cleanup_session_dir() called, session_dir_=" << session_dir_;

    if (!session_dir_.empty()) {
        try {
            namespace fs = std::filesystem;
            if (fs::exists(session_dir_)) {
                fs::remove_all(session_dir_);
                LOG_DEBUG << "Successfully cleaned up session directory: " << session_dir_;
            }
        } catch (const std::exception& e) {
            LOG_WARN << "Failed to cleanup session directory: " << session_dir_
                     << ", error: " << e.what();
        }

        session_dir_.clear();
    }
}

void PythonSession::write_request(
    const std::string& method,
    const std::map<std::string, std::string>& parameters
) {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"method\": \"" << method << "\",\n";
    oss << "  \"parameters\": {\n";

    bool first = true;
    for (const auto& [key, value] : parameters) {
        if (!first) oss << ",\n";
        oss << "    \"" << key << "\": \"" << value << "\"";
        first = false;
    }

    oss << "\n  }\n";
    oss << "}\n";

    std::string request_path = session_dir_ + "/request.json";
    write_json_file(request_path, oss.str());
}

std::tuple<bool, std::string, std::map<std::string, std::string>>
PythonSession::read_response() {
    std::string response_path = session_dir_ + "/response.json";
    std::string content = read_json_file(response_path);

    // 解析JSON（简单解析我们的格式）
    bool success = false;
    std::string message;
    std::map<std::string, std::string> result;

    // 查找success字段
    size_t success_pos = content.find("\"success\"");
    if (success_pos != std::string::npos) {
        size_t true_pos = content.find("true", success_pos);
        size_t false_pos = content.find("false", success_pos);

        if (true_pos != std::string::npos && (false_pos == std::string::npos || true_pos < false_pos)) {
            success = true;
        }
    }

    // 查找message字段
    size_t message_pos = content.find("\"message\"");
    if (message_pos != std::string::npos) {
        size_t quote_start = content.find("\"", message_pos + 10);
        if (quote_start != std::string::npos) {
            size_t quote_end = content.find("\"", quote_start + 1);
            if (quote_end != std::string::npos) {
                message = content.substr(quote_start + 1, quote_end - quote_start - 1);
            }
        }
    }

    return {success, message, result};
}

} // namespace tr

#endif // TR_USE_PYTHON_SESSION
