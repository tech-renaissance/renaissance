/**
 * @file test_relu.cpp
 * @brief ReLU算子数学正确性验证测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/op
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
#include <cstring>

using namespace tr;

// ============================== 工具函数 ==============================

// IEEE 754 half-precision to float（内联，Tensor 无公开 half 转换 API）
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            float zero = 0.0f;
            uint32_t f = sign << 31;
            std::memcpy(&zero, &f, sizeof(zero));
            return zero;
        }
        float result = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
        return sign ? -result : result;
    }

    uint32_t f;
    if (exponent == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
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
        else {
            std::cerr << "ERROR: Unknown or invalid command-line argument: " << a << "\n";
            std::cerr << "Use --help to see valid options.\n";
            std::exit(1);
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
    DTensor d_x    = task.alloc(shape, dtype, Region::F_FEATURE_FP32);        // fwd: X, bwd: dY
    DTensor d_y    = task.alloc(shape, dtype, Region::F_FEATURE_FP32);        // fwd: Y, bwd: dX
    DTensor d_mask = task.alloc(shape, DType::INT8, Region::S_MASK);  // mask
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

    // ── Step 9: H2D 已同步完成，只跑 compute 图 ──
    // 统一使用 run_iter：循环外查找 + 多线程展开，循环内只做 launch + sync
    task.run_iter("relu", 5);   // 预热
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("relu", 100); // 100 次纯热路径
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