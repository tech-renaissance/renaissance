# PythonSession C++/Python互操作机制

**版本**: V3.6.10
**日期**: 2025-12-27
**作者**: 技术觉醒团队
**状态**: ✅ 已实现并测试通过

---

## 目录

1. [概述](#概述)
2. [设计目标](#设计目标)
3. [架构设计](#架构设计)
4. [工作流程](#工作流程)
5. [文件协议](#文件协议)
6. [使用示例](#使用示例)
7. [实现细节](#实现细节)
8. [Python服务器开发](#python服务器开发)
9. [故障排查](#故障排查)

---

## 概述

`PythonSession`是技术觉醒框架提供的C++/Python互操作机制，允许C++代码调用Python进行张量计算，主要用于：

1. **开发调试**：与PyTorch结果对齐，验证C++实现的正确性
2. **功能对比**：对比不同框架的计算结果
3. **灵活扩展**：利用Python生态进行原型验证

### 核心特性

✅ **进程隔离** - C++和Python运行在独立进程中，互不影响
✅ **文件通信** -通过临时文件和JSON协议通信，无需复杂IPC
✅ **张量交换** - 使用TSR V3格式进行张量数据的导入导出
✅ **跨平台** - 支持Windows和Linux，自动适配进程管理
✅ **易用性** - 简洁的API，3行代码即可完成Python调用

---

## 设计目标

### 为什么选择文件通信而非IPC？

| 方案 | 优点 | 缺点 | 选择理由 |
|------|------|------|---------|
| **文件通信** | 简单、跨平台、易调试 | 速度较慢 | ✅ **开发阶段首选** - 简单可靠 |
| **命名管道** | 速度快 | 平台相关、复杂 | ❌ 增加开发负担 |
| **共享内存** | 最快 | 极其复杂、易出错 | ❌ 过度设计 |
| **HTTP/REST** | 标准化 | 依赖网络栈 | ❌ 增加依赖 |

**结论**：文件通信完全满足开发调试需求，且实现简单、易于维护。

---

## 架构设计

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         C++进程                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Tensor a    │  │  Tensor b    │  │Tensor c/cpp  │      │
│  │  (2,3,4,5)   │  │  (2,3,4,5)   │  │  (2,3,4,5)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └────────┬────────┘                 │               │
│                  │                          │               │
│         ┌────────▼────────┐                │               │
│         │ PythonSession   │                │               │
│         │  .calculate()   │                │               │
│         └────────┬────────┘                │               │
│                  │                         │               │
└──────────────────┼─────────────────────────┼───────────────┘
                   │                         │
          ┌────────▼─────────┐      ┌────────▼───────┐
          │  Session Dir     │      │ 比较:         │
          │  /workspace/     │      │ is_close()    │
          │  python_session_ │      │      ✓         │
          │  idXXXX/         │      │               │
          └────────┬─────────┘      └───────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│input_0   │ │input_1   │ │request   │
│  .tsr    │ │  .tsr    │ │  .json   │
└──────────┘ └──────────┘ └──────────┘
      ▲            ▲            ▲
      │            │            │
      └────────────┼────────────┘
                   │
          ┌────────▼─────────┐
          │   Python进程     │
          │                 │
          │  server.py      │
          │  test_python_   │
          │  server.py      │
          │                 │
          │  1. 读取请求    │
          │  2. 加载张量    │
          │  3. PyTorch计算 │
          │  4. 保存结果    │
          │                 │
          └────────┬─────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│output_0  │ │output_1  │ │response  │
│  .tsr    │ │  .tsr    │ │  .json   │
└──────────┘ └──────────┘ └──────────┘
```

### 会话目录结构

```
workspace/
└── python_session_id123456/          # 会话目录（唯一ID）
    ├── input_0.tsr                    # 输入张量0
    ├── input_1.tsr                    # 输入张量1
    ├── request.json                   # 请求文件
    ├── response.json                  # 响应文件
    ├── output_0.tsr                   # 输出张量0
    └── output_1.tsr                   # 输出张量1
```

---

## 工作流程

### 完整流程图

```
C++端                                    Python端
  │                                        │
  │ 1. 创建PythonSession                   │
  │    session.start()                    │
  │    └── 创建会话目录                    │
  │                                        │
  │ 2. session.calculate("add", {a, b})   │
  │    │                                  │
  │    ├── 2.1 send()                     │
  │    │    ├── 导出 input_0.tsr          │◄────────────┐
  │    │    ├── 导出 input_1.tsr          │             │
  │    │    └── 写入 request.json         │             │
  │    │                                  │             │
  │    ├── 2.2 启动Python进程             │             │
  │    │   fork/exec → server.py          │             │
  │    │                                  │             │
  │    ├── 2.3 wait()                     │             │
  │    │    └── 轮询 response.json        │
  │    │        │                         │             │
  │    │         ◄─────────────────────────┘             │
  │    │         │                                     │
  │    │         │  3. Python处理                        │
  │    │         │  ├── 读取 request.json               │
  │    │         │  ├── 加载 input_*.tsr                │
  │    │         │  ├── 执行 main_logic()               │
  │    │         │  │   └── PyTorch计算                  │
  │    │         │  └── 保存 output_*.tsr               │
  │    │         │  └── 写入 response.json              │
  │    │         │                                     │
  │    │    ────┼───────────────────────────────────── │
  │    │         │                                     │
  │    ├── 2.4 fetch()                     │             │
  │    │    ├── 读取 response.json          │             │
  │    │    └── 加载 output_*.tsr           │             │
  │    │                                  │
  │ 3. 返回结果                            │
  │    std::vector<Tensor>                 │
  │                                        │
  │ 4. session.stop()                     │
  │    └── 终止Python进程                  │
  │       清理会话目录                     │
```

### 详细步骤说明

#### 步骤1：初始化（C++端）

```cpp
PythonSession session;  // 使用默认参数
session.start();
```

**内部操作**：
1. 生成唯一的会话ID（基于时间戳）
2. 创建会话目录：`workspace/python_session_idXXXX/`
3. 设置`running_ = true`

#### 步骤2：发送请求（C++端）

```cpp
session.send("add", {a, b}, {});
```

**内部操作**：
1. **导出输入张量**：调用`cpu.export_tensor()`保存为`input_0.tsr`, `input_1.tsr`
2. **写入请求文件**：
   ```json
   {
     "method": "add",
     "parameters": {}
   }
   ```
3. **启动Python进程**（首次调用时）：
   - Windows: `CreateProcess()`启动
   - Linux: `fork() + execlp()`启动
   - 命令行：`python server.py <session_dir>`
4. 等待100ms让Python进程启动

#### 步骤3：处理请求（Python端）

Python服务器执行：
1. **读取请求**：解析`request.json`
2. **加载张量**：通过`tsr_io.import_tensor()`加载`input_0.tsr`, `input_1.tsr`
3. **执行计算**：调用`main_logic("add", {})`方法
   - PyTorch执行：`result = tensor_a + tensor_b`
4. **保存结果**：通过`tsr_io.export_tensor()`保存为`output_0.tsr`
5. **写入响应**：
   ```json
   {
     "success": true,
     "message": "Addition completed successfully",
     "result": {
       "output_count": 1
     }
   }
   ```

#### 步骤4：等待响应（C++端）

```cpp
session.wait();
```

**内部操作**：
- 轮询`response.json`文件（最多10秒）
- 每100ms检查一次
- 超时则抛出异常

#### 步骤5：获取结果（C++端）

```cpp
auto outputs = session.fetch();
```

**内部操作**：
1. 读取`response.json`，检查`success`字段
2. 读取`output_count`字段
3. 加载`output_0.tsr`, `output_1.tsr`, ...
4. 返回`std::vector<Tensor>`

#### 步骤6：清理（C++端）

```cpp
session.stop();
```

**内部操作**：
1. **终止进程**：
   - 先发送`SIGTERM`（Windows：等待5秒）
   - 超时则强制`SIGKILL`（Windows：`TerminateProcess`）
2. **清理目录**：删除整个会话目录
3. 重置状态：`running_ = false`

---

## 文件协议

### request.json格式

```json
{
  "method": "add",           // 方法名
  "parameters": {            // 方法参数（可选）
    "param1": "value1",
    "param2": "value2"
  }
}
```

### response.json格式

**成功响应**：
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "result": {
    "output_count": 2,       // 输出张量数量
    "extra_info": "..."      // 额外信息（可选）
  }
}
```

**失败响应**：
```json
{
  "success": false,
  "message": "Addition failed: shape mismatch",
  "result": {}
}
```

### TSR文件格式

所有张量使用TSR V3格式（详见`docs/tsr_format.md`）：
- **C++导出**：`cpu.export_tensor(tensor, "input_0.tsr", false)`
- **Python导入**：`tensor = import_tensor("input_0.tsr")`
- **支持数据类型**：FP32, BF16, INT32, INT8
- **支持维度**：1D, 2D, 3D, 4D

---

## 使用示例

### 基础用法

```cpp
#include "renaissance.h"

using namespace tr;

int main() {
    // 1. 创建会话（使用默认参数）
    PythonSession session;
    session.start();

    // 2. 准备输入张量
    auto& cpu = DeviceManager::instance().cpu();
    Tensor a = cpu.randn({2, 3, 4, 5}, 0.0f, 1.0f, DType::FP32);
    Tensor b = cpu.randn({2, 3, 4, 5}, 0.0f, 1.0f, DType::FP32);

    // 3. 调用Python计算
    auto outputs = session.calculate("add", {a, b});

    // 4. 验证结果
    Tensor cpp_result = cpu.add(a, b);
    assert(cpu.is_close(outputs[0], cpp_result));

    // 5. 清理
    session.stop();

    return 0;
}
```

### 便捷方法

```cpp
// 使用calculate()一步完成发送、等待、获取
auto outputs = session.calculate("add", {a, b});

// 等价于：
session.send("add", {a, b});
session.wait();
auto outputs = session.fetch();
```

### 自定义参数

```cpp
std::map<std::string, std::string> params;
params["alpha"] = "0.5";
params["beta"] = "0.8";

auto outputs = session.calculate("custom_op", {a}, params);
```

### 自定义Python脚本

```cpp
// 使用自定义服务器脚本
PythonSession session(
    TR_PYTHON_EXECUTABLE,           // Python解释器路径（默认）
    "python/my_custom_server.py"    // 自定义脚本
);
```

---

## 实现细节

### 进程管理

#### Windows实现

```cpp
// 创建进程
STARTUPINFOA si;
PROCESS_INFORMATION pi;
CreateProcessA(
    nullptr,              // 不使用模块名
    cmd_line.data(),      // 命令行
    nullptr,              // 进程句柄不可继承
    nullptr,              // 线程句柄不可继承
    FALSE,                // 不继承句柄
    CREATE_NO_WINDOW,     // 隐藏控制台
    nullptr,              // 使用父进程环境
    nullptr,              // 使用父进程目录
    &si, &pi
);
process_handle_ = pi.hProcess;

// 终止进程（优雅退出）
DWORD wait_result = WaitForSingleObject(process_handle_, 5000);
if (wait_result == WAIT_TIMEOUT) {
    TerminateProcess(process_handle_, 1);  // 强制终止
}
CloseHandle(process_handle_);
```

#### Linux实现

```cpp
// 创建进程
pid_t pid = fork();
if (pid == 0) {
    // 子进程
    execlp(python_exe_.c_str(), python_exe_.c_str(),
           server_path.c_str(), session_dir_.c_str(), nullptr);
    _exit(1);  // execlp失败则退出
}
process_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(pid));

// 终止进程（优雅退出）
pid_t pid = reinterpret_cast<intptr_t>(process_handle_);
kill(pid, SIGTERM);  // 发送终止信号

// 等待最多5秒
for (int i = 0; i < 50; ++i) {
    int status;
    if (waitpid(pid, &status, WNOHANG) == pid) {
        break;  // 已退出
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// 强制杀死
if (wait_result != pid) {
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
}
```

### 会话ID生成

```cpp
session_id_ = static_cast<int>(
    std::chrono::system_clock::now().time_since_epoch().count() % 1000000
);
session_dir_ = generate_session_dir(session_id_);
```

**示例**：`python_session_id123456`

### 延迟启动策略

Python进程并非在`start()`时启动，而是在首次`send()`时启动：

**原因**：确保所有文件（输入张量、request.json）在Python启动前已准备好。

```cpp
void send(...) {
    // ... 写入文件 ...

    // 首次调用时启动Python
    if (process_handle_ == nullptr) {
        start_python_process();
    }
}
```

### JSON解析

实现了一个轻量级的手动解析器（避免依赖第三方库）：

```cpp
std::tuple<bool, std::string, std::map<std::string, std::string>>
read_response() {
    std::string content = read_json_file(response_path);

    // 解析success字段
    size_t success_pos = content.find("\"success\"");
    if (success_pos != std::string::npos) {
        size_t true_pos = content.find("true", success_pos);
        size_t false_pos = content.find("false", success_pos);

        if (true_pos != std::string::npos &&
            (false_pos == std::string::npos || true_pos < false_pos)) {
            success = true;
        }
    }

    // 解析message字段（类似方法）
    // ...
}
```

---

## Python服务器开发

### 服务器基类（RenaissanceServer）

位置：`python/scripts/server.py`

**核心方法**：

```python
class RenaissanceServer:
    def __init__(self, session_dir: str):
        """初始化服务器"""

    def run(self):
        """主入口：读取请求→处理→写入响应"""

    def main_logic(self, method: str, parameters: dict) -> tuple:
        """
        处理命令（子类必须实现）

        Returns:
            (success: bool, message: str, result: dict)
        """
        raise NotImplementedError()

    def load_input_tensor(self, index: int):
        """加载输入张量 input_X.tsr"""

    def save_output_tensor(self, tensor, index: int):
        """保存输出张量 output_X.tsr"""
```

### 实现自定义服务器

**示例1：简单加法**（test_python_server.py）

```python
from server import RenaissanceServer
import torch

class PythonServerExample(RenaissanceServer):
    def main_logic(self, method: str, parameters: dict) -> tuple:
        if method == "add":
            return self._handle_add()
        else:
            return False, f"Unknown method: {method}", {}

    def _handle_add(self) -> tuple:
        # 加载输入
        tensor_a = self.load_input_tensor(0)
        tensor_b = self.load_input_tensor(1)

        # PyTorch计算
        result_tensor = tensor_a + tensor_b

        # 保存输出
        self.save_output_tensor(result_tensor, 0)

        return True, "Addition completed successfully", {
            "output_count": 1
        }
```

**示例2：矩阵乘法**

```python
def main_logic(self, method: str, parameters: dict) -> tuple:
    if method == "matmul":
        a = self.load_input_tensor(0)
        b = self.load_input_tensor(1)

        # 矩阵乘法
        result = torch.matmul(a, b)

        self.save_output_tensor(result, 0)
        return True, "Matrix multiplication completed", {
            "output_count": 1
        }
```

**示例3：带参数的操作**

```python
def main_logic(self, method: str, parameters: dict) -> tuple:
    if method == "scale":
        a = self.load_input_tensor(0)

        # 从parameters获取参数
        alpha = float(parameters.get("alpha", "1.0"))

        result = a * alpha

        self.save_output_tensor(result, 0)
        return True, f"Scaling completed (alpha={alpha})", {
            "output_count": 1
        }
```

### 调用带参数的Python服务器

```cpp
std::map<std::string, std::string> params;
params["alpha"] = "2.5";
auto outputs = session.calculate("scale", {a}, params);
```

### 服务器测试

**直接运行服务器**（用于调试）：

```bash
python python/tests/test_python_server.py /path/to/session_dir
```

这会：
1. 读取`session_dir/request.json`
2. 执行对应的`main_logic()`
3. 写入`session_dir/response.json`
4. 退出

---

## 故障排查

### 问题1：Python进程启动失败

**症状**：
```
Failed to execute Python: python
```

**原因**：
- 使用了硬编码的"python"命令
- 系统上Python解释器路径不同

**解决方案**：
```cpp
// 错误写法
PythonSession session("python", ...);

// 正确写法（使用默认参数）
PythonSession session;  // 自动使用TR_PYTHON_EXECUTABLE宏
```

**检查配置**：
```bash
# 查看CMake配置的Python路径
grep TR_PYTHON_EXECUTABLE build/build.ninja
```

### 问题2：超时等待响应

**症状**：
```
PythonSession::wait: Timeout waiting for Python response
```

**原因**：
1. Python脚本执行出错
2. PyTorch未安装
3. TSR导入失败

**调试步骤**：

**步骤1**：手动运行Python服务器
```bash
# 创建测试会话目录
mkdir -p /tmp/test_session

# 创建测试请求
echo '{"method":"add","parameters":{}}' > /tmp/test_session/request.json

# 创建测试张量（使用已有TSR文件）
cp /path/to/input_0.tsr /tmp/test_session/
cp /path/to/input_1.tsr /tmp/test_session/

# 手动运行服务器
python python/tests/test_python_server.py /tmp/test_session

# 查看输出和错误
cat /tmp/test_session/response.json
```

**步骤2**：检查Python依赖
```bash
# 检查PyTorch
python -c "import torch; print(torch.__version__)"

# 检查TSR I/O
python -c "from tsr_io import import_tensor; print('OK')"
```

**步骤3**：启用Python调试输出
在`test_python_server.py`中添加：
```python
def main_logic(self, method: str, parameters: dict) -> tuple:
    print(f"[DEBUG] Received method: {method}", file=sys.stderr)
    print(f"[DEBUG] Parameters: {parameters}", file=sys.stderr)
    # ...
```

### 问题3：会话目录清理失败

**症状**：
```
Failed to cleanup session directory
```

**原因**：
- Python进程仍在运行，文件句柄未释放
- Windows下文件仍被mmap占用

**解决方案**：

已在代码中实现：
1. 等待进程自然退出（最多5秒）
2. Windows下额外等待100ms让文件系统释放句柄
3. 使用禁用mmap的import（`import_tensor(path, false)`）

如果仍有问题，手动清理：
```bash
# Linux
rm -rf workspace/python_session_*

# Windows
rmdir /s /q workspace\python_session_*
```

### 问题4：张量数据不匹配

**症状**：
```
FAIL: Python and C++ results differ
```

**原因**：
1. 随机种子不同
2. 浮点精度差异
3. 算法实现不同

**调试步骤**：

```cpp
// 设置相同的随机种子
tr::manual_seed(42);
Tensor a = cpu.randn({2, 3, 4, 5}, 0.0f, 1.0f);
// Python端也要设置相同的种子

// 使用合适的容差
bool match = cpu.is_close(python_result, cpp_result, 1e-5f);
// BF16使用更大的容差
bool match = cpu.is_close(python_result, cpp_result, 1e-3f);
```

### 问题5：编译错误：找不到PythonSession

**症状**：
```
error: 'PythonSession' was not declared
```

**原因**：
- `TR_USE_PYTHON_SESSION`宏未定义

**解决方案**：

检查CMake配置：
```bash
grep TR_USE_PYTHON_SESSION config/cmake_paths.cmake
```

应该看到：
```cmake
set(TR_USE_PYTHON_SESSION ON)
add_compile_definitions(TR_USE_PYTHON_SESSION=1)
```

---

## 性能考虑

### 开销分析

| 操作 | 耗时 | 备注 |
|------|------|------|
| 启动Python进程 | ~100ms | 仅首次调用 |
| 导出TSR文件 | ~10ms | 取决于张量大小 |
| PyTorch计算 | ~5ms | 取决于操作复杂度 |
| 导入TSR文件 | ~10ms | 取决于张量大小 |
| **总计** | **~125ms** | 首次调用 |
| 后续调用 | **~25ms** | 无需重启进程 |

### 优化建议

1. **批量操作**：一次性发送多个张量
   ```cpp
   // 慢：多次调用
   auto r1 = session.calculate("add", {a1, b1});
   auto r2 = session.calculate("add", {a2, b2});

   // 快：扩展服务器支持批量
   session.calculate("add_batch", {a1, b1, a2, b2});
   ```

2. **复用会话**：避免频繁创建/销毁
   ```cpp
   // 慢
   for (int i = 0; i < 100; ++i) {
       PythonSession session;
       session.start();
       session.calculate(...);
       session.stop();
   }

   // 快
   PythonSession session;
   session.start();
   for (int i = 0; i < 100; ++i) {
       session.calculate(...);
   }
   session.stop();
   ```

3. **禁用mmap**：开发阶段使用文件I/O（默认已禁用）
   ```cpp
   tensor = cpu.import_tensor(path, false);  // false = 禁用mmap
   ```

---

## 最佳实践

### 1. 使用默认参数

```cpp
// ✅ 推荐
PythonSession session;

// ❌ 不推荐（硬编码）
PythonSession session("python", "python/tests/test_python_server.py");
```

### 2. 异常处理

```cpp
try {
    PythonSession session;
    session.start();
    auto outputs = session.calculate("add", {a, b});
    session.stop();
} catch (const TRException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // session会自动清理
}
```

### 3. RAII模式

利用析构函数自动清理：

```cpp
{
    PythonSession session;
    session.start();
    // ... 使用 ...
    // 自动调用stop()
}  // 析构自动清理
```

### 4. 服务器实现

```python
# ✅ 推荐：继承基类
class MyServer(RenaissanceServer):
    def main_logic(self, method, parameters):
        # 实现逻辑
        pass

# ❌ 不推荐：直接修改server.py
```

---

## 版本历史

### V3.6.10 (2025-12-27)

- **新增**：完整的PythonSession机制
- **新增**：C++/Python张量互操作
- **新增**：进程管理（Windows/Linux）
- **新增**：TSR V3格式集成
- **测试**：test_python_session.cpp测试通过
- **文档**：本说明文档

---

## 参考资料

- [TSR文件格式规范](tsr_format.md)
- [Device类文档](device.md)
- [CMake配置说明](../alpha_build.md)

---

**文档维护**：技术觉醒团队
**最后更新**：2025-12-27
