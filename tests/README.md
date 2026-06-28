# 测试

本目录包含 Renaissance 的单元测试、集成测试与性能基准。所有测试均为独立可执行文件，统一 `#include <renaissance.h>`。

## 运行测试

构建全部测试：

```bash
cd build
cmake --build . -j
```

测试可执行文件输出到 `build/bin/tests/<subdir>/`。当前项目未启用 CTest，需直接运行：

```bash
./bin/tests/correction/test_gap
./bin/tests/op/test_relu
```

Windows 环境下运行 GPU 测试前，需先注入 CUDA/cuDNN 路径：

```powershell
.\setup_cuda_env.bat .\bin\tests\<subdir>\<test>.exe
```

## 目录结构

| 目录 | 覆盖范围 |
|------|---------|
| `op/` | 单算子正确性测试（Conv、BN、ReLU、Pooling、激活函数等） |
| `fp16/` | FP16 半精度融合算子测试（CBR 等，需 CUDA） |
| `bn/` | BatchNorm 专项测试与 LeNet/MNIST 训练验证 |
| `graph/` | 计算图基础测试（AXPY、MLP、双图并行等） |
| `correction/` | 数值正确性验证（优化器更新、算子精度、归一化、H2D/D2D 拷贝等） |
| `integration/` | 端到端模型集成测试（VGG16BN、NiN、MLP 组合等） |
| `perf/` | 性能微基准（CBR、cast、copy、dropout、softmax CE 等） |
| `top/` | 顶层模型测试（ResNet-50 checksum、ArchPlan 验证） |
| `ref/` | 参考实现与对照实验 |

## 添加新测试

1. 在对应子目录下创建 `test_<name>.cpp`。
2. 在子目录 `CMakeLists.txt` 中用 `add_executable` 注册。
3. 若测试依赖 GPU，调用 `setup_gpu_runtime_env(<target>)` 设置运行时 PATH。
