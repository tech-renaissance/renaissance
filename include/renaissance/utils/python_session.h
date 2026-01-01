/**
 * @file python_session.h
 * @brief Python session manager for C++/Python interoperability
 * @version 3.6.10
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#pragma once

#ifdef TR_USE_PYTHON_SESSION

#include <string>
#include <vector>
#include <map>
#include <tuple>

namespace tr {

// 前向声明
class Tensor;

/**
 * @class PythonSession
 * @brief Manages communication with a Python server process via temporary files
 *
 * The session creates a dedicated directory in TR_WORKSPACE/python_session_idXXXX/
 * and communicates with Python via:
 * - JSON files for commands and responses
 * - TSR V3 format files for tensor data exchange
 *
 * Usage example:
 * @code
 * PythonSession session("python3", "python/tests/test_python_server.py");
 * session.start();
 *
 * // Prepare input tensors
 * Tensor a = gpu.randn({2, 3, 4, 5}, DType::FP32);
 * Tensor b = gpu.randn({2, 3, 4, 5}, DType::FP32);
 *
 * // Send to Python and compute
 * std::vector<Tensor> inputs = {a, b};
 * auto outputs = session.calculate("add", inputs);
 *
 * // Verify result
 * Tensor c = gpu.add(a, b);
 * assert(gpu.is_close(c, outputs[0]));
 *
 * session.stop();
 * @endcode
 */
class PythonSession {
public:
    /**
     * @brief Construct a Python session
     *
     * @param python_exe Path to Python executable (e.g., "python", "python3")
     *                    Default: TR_PYTHON_EXECUTABLE (from CMake) or "python"
     * @param server_script Path to Python server script (relative to project root)
     */
    PythonSession(const std::string& python_exe =
#ifdef TR_PYTHON_EXECUTABLE
                    TR_PYTHON_EXECUTABLE,
#else
                    "python",
#endif
                  const std::string& server_script =
#ifdef TR_PROJECT_ROOT
                      TR_PROJECT_ROOT "/python/scripts/default_python_server.py"
#else
                      "python/scripts/default_python_server.py"
#endif
                  );

    /**
     * @brief Destructor - automatically stops the session
     */
    ~PythonSession();

    // Disable copy and move (process handle management is complex)
    PythonSession(const PythonSession&) = delete;
    PythonSession& operator=(const PythonSession&) = delete;
    PythonSession(PythonSession&&) = delete;
    PythonSession& operator=(PythonSession&&) = delete;

    /**
     * @brief Start the Python server process
     *
     * Creates session directory and spawns Python process.
     * The process stays alive and waits for requests.
     *
     * @throw RuntimeError if process fails to start
     */
    void start();

    /**
     * @brief Stop the Python server process
     *
     * Terminates the process and cleans up session directory.
     */
    void stop();

    /**
     * @brief Send a command to Python and wait for result
     *
     * This is a convenience method that combines send(), wait(), and fetch().
     *
     * @param method Method name to invoke (e.g., "add", "matmul")
     * @param inputs Input tensors
     * @param parameters Optional parameters (currently empty for most operations)
     *
     * @return Vector of output tensors
     * @throw RuntimeError if operation fails
     */
    std::vector<Tensor> calculate(
        const std::string& method,
        const std::vector<Tensor>& inputs,
        const std::map<std::string, std::string>& parameters = {}
    );

    /**
     * @brief Check if the session is running
     *
     * @return true if Python process is alive
     */
    bool is_running() const { return running_; }

    /**
     * @brief Print tensor using PyTorch and return the printed text
     *
     * Sends a tensor to Python server, which prints it using PyTorch's format,
     * then returns the printed text as a string.
     *
     * @param tensor Input tensor to print
     * @return PyTorch's printed representation of the tensor
     * @throw RuntimeError if operation fails
     */
    std::string print_tensor(const Tensor& tensor);

    /**
     * @brief Get the session directory path
     *
     * @return Full path to the session directory
     */
    const std::string& session_dir() const { return session_dir_; }

private:
    /**
     * @brief Send tensors and command to Python
     *
     * Writes input tensors to input_0.tsr, input_1.tsr, ...
     * and writes request.json with method and parameters.
     *
     * @param method Method name
     * @param inputs Input tensors
     * @param parameters Optional parameters
     */
    void send(
        const std::string& method,
        const std::vector<Tensor>& inputs,
        const std::map<std::string, std::string>& parameters
    );

    /**
     * @brief Wait for Python to finish processing
     *
     * Polls for response.json file until timeout.
     *
     * @throw RuntimeError if timeout occurs
     */
    void wait();

    /**
     * @brief Fetch output tensors from Python
     *
     * Reads output_0.tsr, output_1.tsr, ... based on response.json
     *
     * @return Vector of output tensors
     */
    std::vector<Tensor> fetch();

    /**
     * @brief Fetch text output from Python
     *
     * Reads output_0.txt file and returns its content.
     * The file is deleted immediately after reading (read-and-delete).
     *
     * @return Text content from Python
     * @throw RuntimeError if file not found or read fails
     */
    std::string fetch_text_output();

    /**
     * @brief Create session directory
     */
    void create_session_dir();

    /**
     * @brief Clean up session directory
     */
    void cleanup_session_dir();

    /**
     * @brief Write request.json file
     */
    void write_request(
        const std::string& method,
        const std::map<std::string, std::string>& parameters
    );

    /**
     * @brief Read and parse response.json file
     *
     * @return Tuple of (success, message, result_dict)
     */
    std::tuple<bool, std::string, std::map<std::string, std::string>> read_response();

    std::string python_exe_;      ///< Python executable path
    std::string server_script_;   ///< Server script path
    std::string session_dir_;     ///< Session directory path
    void* process_handle_;        ///< Platform-specific process handle
    bool running_;                ///< Whether the process is running
    int session_id_;              ///< Unique session ID
};

} // namespace tr

#endif // TR_USE_PYTHON_SESSION
