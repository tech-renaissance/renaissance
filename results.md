# tests/op/test_relu.cpp的测试结果

## 测试样例tests/op/test_relu.py

```python
#!/usr/bin/env python3
"""
ReLU 算子数学正确性验证 — PyTorch 参考数据生成
输出: TSR 张量 + YAML 配置 → workspace/
"""

import torch
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr

def generate(op_type: str, dtype_str: str, shape: tuple, seed: int, ws: str):
    N, H, W, C = shape
    is_fp16 = (dtype_str == "FP16")
    torch_dtype = torch.float16 if is_fp16 else torch.float32
    np_dtype    = np.float16    if is_fp16 else np.float32

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cudnn.fp32_precision = 'ieee'
    except AttributeError:
        pass

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(ws, exist_ok=True)

    # 文件名后缀：同时区分dtype和op_type，避免6个测试相互覆盖
    # 格式: {op_type}_{dtype后缀}.tsr
    dtype_suffix = "amp" if is_fp16 else "fp32"

    if op_type == 'fwd':
        X = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
        Y = torch.relu(X)
        mask = (X > 0).to(torch.int8)

        save_tsr(os.path.join(ws, f'input_fwd_{dtype_suffix}.tsr'),
                 [X.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'output_ref_fwd_{dtype_suffix}.tsr'),
                 [Y.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'mask_ref_fwd_{dtype_suffix}.tsr'),
                 [mask.cpu().numpy().astype(np.int8)], compress=False)

        file_desc = (
            f"  input: input_fwd_{dtype_suffix}.tsr\n"
            f"  output: output_ref_fwd_{dtype_suffix}.tsr\n"
            f"  mask: mask_ref_fwd_{dtype_suffix}.tsr\n"
        )
    else:
        X = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
        mask = (X > 0).to(torch.int8)

        torch.manual_seed(seed + 1000)
        dY = torch.randn(N, H, W, C, dtype=torch_dtype) * 2.0
        dX = dY * mask.float().to(torch_dtype)

        save_tsr(os.path.join(ws, f'input_bwd_{dtype_suffix}.tsr'),
                 [dY.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'output_ref_bwd_{dtype_suffix}.tsr'),
                 [dX.cpu().numpy().astype(np_dtype)], compress=False)
        save_tsr(os.path.join(ws, f'mask_ref_bwd_{dtype_suffix}.tsr'),
                 [mask.cpu().numpy().astype(np.int8)], compress=False)

        file_desc = (
            f"  input: input_bwd_{dtype_suffix}.tsr\n"
            f"  output: output_ref_bwd_{dtype_suffix}.tsr\n"
            f"  mask: mask_ref_bwd_{dtype_suffix}.tsr\n"
        )

    yaml = (
        f"op: relu\n"
        f"op_type: {op_type}\n"
        f"dtype: {dtype_str}\n"
        f"shape: [{N}, {H}, {W}, {C}]\n"
        f"seed: {seed}\n"
        f"mse_threshold: {'1e-3' if is_fp16 else '1e-5'}\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"ReLU {op_type} {dtype_str} reference generated in: {ws}")

def main():
    parser = argparse.ArgumentParser(description='ReLU test data generator')
    parser.add_argument('--op_type', required=True, choices=['fwd', 'bwd'])
    parser.add_argument('--dtype',   default='FP32', choices=['FP32', 'FP16'])
    parser.add_argument('--shape',   default='8,1024,1024,8',   # 67M elements = 256MB FP32 / 128MB FP16
                        help='NHWC shape (comma-separated, e.g. 1,1024,1,1)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()

    dims = tuple(int(x) for x in args.shape.split(','))
    if len(dims) != 4:
        print(f"Invalid shape '{args.shape}': must be N,H,W,C (4D)")
        sys.exit(1)

    generate(args.op_type, args.dtype, dims, args.seed, args.workspace)

if __name__ == '__main__':
    main()
```



## 测试样例tests/op/test_relu.cpp

```c++
/**
 * @file test_relu.cpp
 * @brief ReLU算子数学正确性验证测试
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 设计原则: 4种ComputeOp × 2后端 = 8组合, 2种CPU AMP无效(TR_TYPE_ERROR), 6种有效
 * @note 参照: tests/graph/axpy.cpp (GPU) + axpy_cpu.cpp (CPU) 的顶层写法
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>

using namespace tr;

// ============================== 工具函数 ==============================

// IEEE 754 half-precision to float（内联，Tensor 无公开 half 转换 API）
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign << 31;
        } else {
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FF;
            exponent = 1 + (127 - 15);
            f = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    union { uint32_t u; float fl; } uf;
    uf.u = f;
    return uf.fl;
}

double compute_mse(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError,
             "MSE shape mismatch");
    int64_t n = a.numel();
    double sum = 0.0;

    if (a.dtype() == DType::FP32) {
        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
            sum += d * d;
        }
    } else if (a.dtype() == DType::FP16) {
        const uint16_t* pa = a.data<uint16_t>();
        const uint16_t* pb = b.data<uint16_t>();
        for (int64_t i = 0; i < n; ++i) {
            double d = static_cast<double>(fp16_to_f32(pa[i]))
                     - static_cast<double>(fp16_to_f32(pb[i]));
            sum += d * d;
        }
    } else {
        TR_THROW(ValueError, "Unsupported dtype for MSE: "
                 << static_cast<int>(a.dtype()));
    }
    return sum / n;
}


struct TestConfig {
    bool use_cpu = false;
    bool use_amp = false;
    bool no_pytorch = false;  // 跳过PyTorch数据生成，直接使用已有文件
    std::string op_type = "fwd";
    std::string shape_str = "8,1024,1024,8";  // 256MB FP32 / 128MB FP16
    int seed = 42;
};

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") c.use_cpu = true;
        else if (a == "--gpu") c.use_cpu = false;
        else if (a == "--amp") c.use_amp = true;
        else if (a == "--no-pytorch") c.no_pytorch = true;
        else if (a == "--op_type" && i + 1 < argc) c.op_type = argv[++i];
        else if (a == "--shape"   && i + 1 < argc) c.shape_str = argv[++i];
        else if (a == "--seed"    && i + 1 < argc) c.seed = std::stoi(argv[++i]);
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --cpu           Force CPU mode\n"
                << "  --gpu           Force GPU mode (default if available)\n"
                << "  --amp           Use FP16 precision (GPU only)\n"
                << "  --no-pytorch    Skip PyTorch data generation, use existing files\n"
                << "  --op_type fwd|bwd  Direction (default: fwd)\n"
                << "  --shape N,H,W,C    Tensor shape (default: 1,1024,1,1)\n"
                << "  --seed N        Random seed (default: 42)\n"
                << "  --help          Show this message\n";
            std::exit(0);
        }
    }
    return c;
}


// ============================== 主测试流程 ==============================

int main(int argc, char** argv) {
    // ── Step 1: 解析 CLI ──
    auto cfg = parse_cli(argc, argv);

    // ── Step 2: 配置 GlobalRegistry ──
    if (cfg.use_cpu) {
        GLOBAL_SETTING.use_cpu().auto_seed();
    } else {
        int n = GlobalRegistry::get_visible_gpu_count();
        TR_CHECK(n > 0, DeviceError, "No visible GPU. Use --cpu for CPU mode.");
        std::string ids;
        for (int i = 0; i < n; ++i) {
            if (i) ids += ",";
            ids += std::to_string(i);
        }
        GLOBAL_SETTING.use_gpu(ids.c_str()).auto_seed();
    }
    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // 派生参数（仅用于Python调用）
    std::string dt_str = cfg.use_amp ? "FP16" : "FP32";

    // ── Step 3: 调用 Python 生成参考数据（如果需要）──
    std::string ws = std::string(TR_WORKSPACE) + "/relu_test_data";
    if (!cfg.no_pytorch) {
        std::ostringstream py;
        py << "python " << std::string(TR_PROJECT_ROOT) << "/tests/op/test_relu.py"
           << " --op_type " << cfg.op_type
           << " --dtype "   << dt_str
           << " --shape "   << cfg.shape_str
           << " --seed "    << cfg.seed
           << " --workspace \"" << ws << "\"";
        std::cout << "Generating reference data: " << py.str() << std::endl;
        TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
                 "Python failed. Is PyTorch installed? Use --no-pytorch if files already exist. Command: " << py.str());
    } else {
        std::cout << "Using existing reference data (skip PyTorch generation)" << std::endl;
    }

    // ── Step 4: 读取 YAML ──
    std::string ypath = ws + "/config.yaml";
    std::ifstream yf(ypath);
    TR_CHECK(yf.is_open(), FileNotFoundError, ypath);
    std::string ys((std::istreambuf_iterator<char>(yf)),
                    std::istreambuf_iterator<char>());
    auto yaml = fkyaml::node::deserialize(ys);

    auto& sh = yaml["shape"];
    Shape shape{sh[0].get_value<int>(), sh[1].get_value<int>(),
                sh[2].get_value<int>(), sh[3].get_value<int>()};

    // 根据当前测试配置确定dtype（不再从YAML读取，避免配置污染）
    DType dtype = cfg.use_amp ? DType::FP16 : DType::FP32;
    float mse_thr = cfg.use_amp ? 1e-3f : 1e-5f;
    std::string dtype_str = cfg.use_amp ? "FP16" : "FP32";

    // 构造文件名：同时区分op_type和dtype，避免6个测试相互覆盖
    // 格式: {op_type}_{dtype后缀}.tsr
    std::string dtype_suffix = (dtype == DType::FP16) ? "amp" : "fp32";
    std::string fn_input  = "input_"  + cfg.op_type + "_" + dtype_suffix + ".tsr";
    std::string fn_output = "output_ref_" + cfg.op_type + "_" + dtype_suffix + ".tsr";
    std::string fn_mask   = "mask_ref_"   + cfg.op_type + "_" + dtype_suffix + ".tsr";

    // ── Step 5: 加载 TSR ──
    Tensor h_input  = Tensor::load_tensor(ws + "/" + fn_input);
    Tensor h_ref    = Tensor::load_tensor(ws + "/" + fn_output);
    Tensor h_mask   = Tensor::load_tensor(ws + "/" + fn_mask);

    // ── Step 6: SimpleTask + DTensor 分配 ──
    SimpleTask task;
    DTensor d_x    = task.alloc(shape, dtype);        // fwd: X, bwd: dY
    DTensor d_y    = task.alloc(shape, dtype);        // fwd: Y, bwd: dX
    DTensor d_mask = task.alloc(shape, DType::INT8);  // mask
    task.finalize_memory();

    // ── Step 7: 构建单算子 ComputationGraph ──
    // 同一 ComputeOp 枚举，CPU/GPU 自动分发
    ComputeOp op = (cfg.op_type == "fwd")
        ? (cfg.use_amp ? ComputeOp::RELU_AMP_FWD  : ComputeOp::RELU_FP32_FWD)
        : (cfg.use_amp ? ComputeOp::RELU_AMP_BWD  : ComputeOp::RELU_FP32_BWD);

    ComputationGraph g;
    if (cfg.op_type == "fwd") {
        g.append(op, {d_x.id}, {d_y.id, d_mask.id});
    } else {
        g.append(op, {d_x.id, d_mask.id}, {d_y.id});
    }
    task.add_graph("relu", std::move(g), StreamKind::COMP_1);

    // ── Step 8: 编译 + H2D ──
    task.compile();
    {
        task.transfer_to_rank(h_input, d_x, 0);
        if (cfg.op_type == "bwd") {
            task.transfer_to_rank(h_mask, d_mask, 0);
        }
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            if (cfg.op_type == "bwd") {
                task.broadcast_from_rank0(d_mask);
            }
        }
    }

    // ── Step 9: 预热 + 100 次计时 ──
    for (int i = 0; i < 5; ++i) task.run("relu");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) task.run("relu");
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count()
                  / 100.0;

    // ── Step 10: 逐 Rank D2H + MSE ──
    bool all_pass = true;
    double max_mse = 0.0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_out = task.fetch_from_rank(d_y, rank);
        double mse = compute_mse(h_out, h_ref);
        max_mse = (mse > max_mse) ? mse : max_mse;

        std::cout << "  Rank " << rank << " MSE = " << std::scientific
                  << mse << " (threshold: " << mse_thr << ")";
        if (mse > mse_thr) {
            std::cout << " FAIL";
            all_pass = false;
        }
        std::cout << std::endl;
    }

    std::cout << "\n===== ReLU " << cfg.op_type << " [" << dt_str
              << "] " << (cfg.use_cpu ? "CPU" : "GPU")
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Avg:   " << std::fixed << avg_us << " us/iter\n"
              << "  MaxMSE: " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}
```



## Windows结果（RTX 4060）

```shell
PS R:\renaissance\build\windows-msvc-release\bin\tests\op> .\test_relu.exe --cpu --op_type fwd
[2026-05-17 01:46:58.947] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to false
Device: CPU
[2026-05-17 01:46:58.948] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x00027473CCC7AC48
Generating reference data: python R:/renaissance/tests/op/test_relu.py --op_type fwd --dtype FP32 --shape 8,1024,1024,8 --seed 42 --workspace "R:/renaissance/workspace/relu_test_data"
C:\Python314\Lib\site-packages\torch\backends\__init__.py:42: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\Context.cpp:85.)
  return self.getter()
ReLU fwd FP32 reference generated in: R:/renaissance/workspace/relu_test_data
[2026-05-17 01:47:03.505] [INFO ] [task] Allocated CPU device context, 640.001 MB
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU fwd [FP32] CPU (1 rank(s)): PASS =====
  Avg:   49482.201000 us/iter
  MaxMSE: 0.000000E+00
PS R:\renaissance\build\windows-msvc-release\bin\tests\op> .\test_relu.exe --cpu --op_type bwd
[2026-05-17 01:47:08.921] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to false
Device: CPU
[2026-05-17 01:47:08.921] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x000274761F30D23C
Generating reference data: python R:/renaissance/tests/op/test_relu.py --op_type bwd --dtype FP32 --shape 8,1024,1024,8 --seed 42 --workspace "R:/renaissance/workspace/relu_test_data"
C:\Python314\Lib\site-packages\torch\backends\__init__.py:42: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\Context.cpp:85.)
  return self.getter()
ReLU bwd FP32 reference generated in: R:/renaissance/workspace/relu_test_data
[2026-05-17 01:47:12.528] [INFO ] [task] Allocated CPU device context, 640.001 MB
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU bwd [FP32] CPU (1 rank(s)): PASS =====
  Avg:   178367.112000 us/iter
  MaxMSE: 0.000000E+00
PS R:\renaissance\build\windows-msvc-release\bin\tests\op> .\test_relu.exe --gpu --op_type fwd
[2026-05-17 01:47:31.498] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:47:31.498] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0]
Device: GPU [0]
[2026-05-17 01:47:31.499] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x0002747B60EADDE4
Generating reference data: python R:/renaissance/tests/op/test_relu.py --op_type fwd --dtype FP32 --shape 8,1024,1024,8 --seed 42 --workspace "R:/renaissance/workspace/relu_test_data"
C:\Python314\Lib\site-packages\torch\backends\__init__.py:42: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\Context.cpp:85.)
  return self.getter()
ReLU fwd FP32 reference generated in: R:/renaissance/workspace/relu_test_data
[2026-05-17 01:47:34.885] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:47:34.885] [INFO ] [task] Allocated 1 GPU device context(s), 640.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU fwd [FP32] GPU (1 rank(s)): PASS =====
  Avg:   2874.728000 us/iter
  MaxMSE: 0.000000E+00
PS R:\renaissance\build\windows-msvc-release\bin\tests\op> .\test_relu.exe --gpu --op_type bwd
[2026-05-17 01:47:35.383] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:47:35.384] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0]
Device: GPU [0]
[2026-05-17 01:47:35.384] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x0002747C487FDC18
Generating reference data: python R:/renaissance/tests/op/test_relu.py --op_type bwd --dtype FP32 --shape 8,1024,1024,8 --seed 42 --workspace "R:/renaissance/workspace/relu_test_data"
C:\Python314\Lib\site-packages\torch\backends\__init__.py:42: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\Context.cpp:85.)
  return self.getter()
ReLU bwd FP32 reference generated in: R:/renaissance/workspace/relu_test_data
[2026-05-17 01:47:39.044] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:47:39.045] [INFO ] [task] Allocated 1 GPU device context(s), 640.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU bwd [FP32] GPU (1 rank(s)): PASS =====
  Avg:   2680.027000 us/iter
  MaxMSE: 0.000000E+00
PS R:\renaissance\build\windows-msvc-release\bin\tests\op> .\test_relu.exe --gpu --op_type fwd --amp
[2026-05-17 01:47:39.534] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:47:39.535] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0]
Device: GPU [0]
[2026-05-17 01:47:39.535] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x0002747D3FE6C340
Generating reference data: python R:/renaissance/tests/op/test_relu.py --op_type fwd --dtype FP16 --shape 8,1024,1024,8 --seed 42 --workspace "R:/renaissance/workspace/relu_test_data"
C:\Python314\Lib\site-packages\torch\backends\__init__.py:42: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\Context.cpp:85.)
  return self.getter()
ReLU fwd FP16 reference generated in: R:/renaissance/workspace/relu_test_data
[2026-05-17 01:47:43.569] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:47:43.569] [INFO ] [task] Allocated 1 GPU device context(s), 384.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-03)

===== ReLU fwd [FP16] GPU (1 rank(s)): PASS =====
  Avg:   2667.794000 us/iter
  MaxMSE: 0.000000E+00
PS R:\renaissance\build\windows-msvc-release\bin\tests\op> .\test_relu.exe --gpu --op_type bwd --amp
[2026-05-17 01:47:44.256] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:47:44.256] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0]
Device: GPU [0]
[2026-05-17 01:47:44.256] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x0002747E595561A0
Generating reference data: python R:/renaissance/tests/op/test_relu.py --op_type bwd --dtype FP16 --shape 8,1024,1024,8 --seed 42 --workspace "R:/renaissance/workspace/relu_test_data"
C:\Python314\Lib\site-packages\torch\backends\__init__.py:42: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\Context.cpp:85.)
  return self.getter()
ReLU bwd FP16 reference generated in: R:/renaissance/workspace/relu_test_data
[2026-05-17 01:47:49.886] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:47:49.887] [INFO ] [task] Allocated 1 GPU device context(s), 384.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-03)

===== ReLU bwd [FP16] GPU (1 rank(s)): PASS =====
  Avg:   1390.950000 us/iter
  MaxMSE: 0.000000E+00
```

## Linux结果（A100×8）

```shell
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu --no-pytorch --cpu --op_type fwd
[2026-05-17 01:55:54.958] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to false
Device: CPU
[2026-05-17 01:55:54.958] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B01D60EDFEBB28
Using existing reference data (skip PyTorch generation)
[2026-05-17 01:55:57.177] [INFO ] [task] Allocated CPU device context, 640.001 MB
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU fwd [FP32] CPU (1 rank(s)): PASS =====
  Avg:   44036.630990 us/iter
  MaxMSE: 0.000000E+00
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu --no-pytorch --cpu --op_type bwd
[2026-05-17 01:56:02.812] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to false
Device: CPU
[2026-05-17 01:56:02.812] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B01D62C2265629
Using existing reference data (skip PyTorch generation)
[2026-05-17 01:56:04.964] [INFO ] [task] Allocated CPU device context, 640.001 MB
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU bwd [FP32] CPU (1 rank(s)): PASS =====
  Avg:   31948.431340 us/iter
  MaxMSE: 0.000000E+00
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu --no-pytorch --gpu --op_type fwd
[2026-05-17 01:56:10.857] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:56:10.857] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]
[2026-05-17 01:56:10.857] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B01D64A1A9FF3D
Using existing reference data (skip PyTorch generation)
[2026-05-17 01:56:12.661] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.670] [INFO ] [backend] DeviceContext 1: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.678] [INFO ] [backend] DeviceContext 2: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.687] [INFO ] [backend] DeviceContext 3: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.696] [INFO ] [backend] DeviceContext 4: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.704] [INFO ] [backend] DeviceContext 5: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.713] [INFO ] [backend] DeviceContext 6: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:12.721] [INFO ] [backend] DeviceContext 7: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:13.955] [INFO ] [task] NCCL initialized for 8 GPUs
[2026-05-17 01:56:13.955] [INFO ] [task] Allocated 8 GPU device context(s), 640.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 1 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 2 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 3 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 4 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 5 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 6 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 7 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU fwd [FP32] GPU (8 rank(s)): PASS =====
  Avg:   540.398230 us/iter
  MaxMSE: 0.000000E+00
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu --no-pytorch --gpu --op_type bwd
[2026-05-17 01:56:18.854] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:56:18.854] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]
[2026-05-17 01:56:18.854] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B01D667E4EAE17
Using existing reference data (skip PyTorch generation)
[2026-05-17 01:56:20.637] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.646] [INFO ] [backend] DeviceContext 1: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.654] [INFO ] [backend] DeviceContext 2: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.663] [INFO ] [backend] DeviceContext 3: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.671] [INFO ] [backend] DeviceContext 4: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.680] [INFO ] [backend] DeviceContext 5: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.689] [INFO ] [backend] DeviceContext 6: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:20.697] [INFO ] [backend] DeviceContext 7: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:21.918] [INFO ] [task] NCCL initialized for 8 GPUs
[2026-05-17 01:56:21.918] [INFO ] [task] Allocated 8 GPU device context(s), 640.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 1 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 2 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 3 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 4 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 5 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 6 MSE = 0.000000E+00 (threshold: 1.000000E-05)
  Rank 7 MSE = 0.000000E+00 (threshold: 1.000000E-05)

===== ReLU bwd [FP32] GPU (8 rank(s)): PASS =====
  Avg:   717.445990 us/iter
  MaxMSE: 0.000000E+00
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu --no-pytorch --gpu --op_type fwd --amp
[2026-05-17 01:56:26.904] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:56:26.904] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]
[2026-05-17 01:56:26.904] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B01D685E28A635
Using existing reference data (skip PyTorch generation)
[2026-05-17 01:56:28.440] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.448] [INFO ] [backend] DeviceContext 1: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.457] [INFO ] [backend] DeviceContext 2: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.466] [INFO ] [backend] DeviceContext 3: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.474] [INFO ] [backend] DeviceContext 4: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.483] [INFO ] [backend] DeviceContext 5: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.491] [INFO ] [backend] DeviceContext 6: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:28.500] [INFO ] [backend] DeviceContext 7: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:29.720] [INFO ] [task] NCCL initialized for 8 GPUs
[2026-05-17 01:56:29.720] [INFO ] [task] Allocated 8 GPU device context(s), 384.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 1 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 2 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 3 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 4 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 5 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 6 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 7 MSE = 0.000000E+00 (threshold: 1.000000E-03)

===== ReLU fwd [FP16] GPU (8 rank(s)): PASS =====
  Avg:   1012.329700 us/iter
  MaxMSE: 0.000000E+00
root@2af6a1cc17a3:~/epfs/R/renaissance/build/bin/tests/op# ./test_relu --no-pytorch --gpu --op_type bwd --amp
[2026-05-17 01:56:36.763] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-17 01:56:36.763] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]
[2026-05-17 01:56:36.763] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
Random Seed: 0x18B01D6AA9CAD971
Using existing reference data (skip PyTorch generation)
[2026-05-17 01:56:38.294] [INFO ] [backend] DeviceContext 0: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.302] [INFO ] [backend] DeviceContext 1: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.311] [INFO ] [backend] DeviceContext 2: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.320] [INFO ] [backend] DeviceContext 3: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.328] [INFO ] [backend] DeviceContext 4: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.337] [INFO ] [backend] DeviceContext 5: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.345] [INFO ] [backend] DeviceContext 6: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:38.354] [INFO ] [backend] DeviceContext 7: created 5 non-blocking streams + per-stream cuDNN handles
[2026-05-17 01:56:39.605] [INFO ] [task] NCCL initialized for 8 GPUs
[2026-05-17 01:56:39.605] [INFO ] [task] Allocated 8 GPU device context(s), 384.001 MB each
  Rank 0 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 1 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 2 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 3 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 4 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 5 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 6 MSE = 0.000000E+00 (threshold: 1.000000E-03)
  Rank 7 MSE = 0.000000E+00 (threshold: 1.000000E-03)

===== ReLU bwd [FP16] GPU (8 rank(s)): PASS =====
  Avg:   475.254290 us/iter
  MaxMSE: 0.000000E+00

```

