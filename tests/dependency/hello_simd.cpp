#include <iostream>
#include <vector>
#include <chrono>   // 用于计时

// 包含Simd库的核心头文件
#include <Simd/SimdLib.h>

int main() {
    // --- 1. 准备工作：定义数组和参数 ---

    // 定义数组的大小，为了能明显看到SIMD的效果，建议使用较大的数
    const size_t N = 1024 * 1024;

    // 创建两个源数组，并用一些初始值填充
    // std::vector 是一个很好的选择，它保证了内存的连续性
    std::vector<float> a(N, 1.0f); // 填充 1.0f
    std::vector<float> b(N, 2.0f); // 填充 2.0f

    // 创建一个目标数组，用于存储结果
    std::vector<float> c_scalar(N); // 用于存储标量版本的结果
    std::vector<float> c_simd(N);   // 用于存储SIMD版本的结果

    std::cout << "Simd Library Hello World Example" << std::endl;
    std::cout << "Array size: " << N << " elements" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // --- 2. 标量版本实现 ---

    auto start_scalar = std::chrono::high_resolution_clock::now();

    // 传统的C++循环，一个一个地相加
    for (size_t i = 0; i < N; ++i) {
        c_scalar[i] = a[i] + b[i];
    }

    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_scalar = end_scalar - start_scalar;

    std::cout << "Scalar version finished." << std::endl;
    std::cout << "Time taken: " << duration_scalar.count() << " ms" << std::endl;
    std::cout << "Result (first 5 elements): ";
    for(int i = 0; i < 5; ++i) std::cout << c_scalar[i] << " ";
    std::cout << std::endl << std::endl;


    // --- 3. SIMD版本实现 ---

    auto start_simd = std::chrono::high_resolution_clock::now();

    // 使用Simd库的NeuralAddVector函数
    // 将b向量加到a向量上，结果存入c_simd
    SimdNeuralAddVector(b.data(), N, c_simd.data());
    // 然后再加上a向量
    SimdNeuralAddVector(a.data(), N, c_simd.data());

    auto end_simd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_simd = end_simd - start_simd;

    std::cout << "Simd version finished." << std::endl;
    std::cout << "Time taken: " << duration_simd.count() << " ms" << std::endl;
    std::cout << "Result (first 5 elements): ";
    for(int i = 0; i < 5; ++i) std::cout << c_simd[i] << " ";
    std::cout << std::endl << std::endl;


    // --- 4. 验证结果 ---

    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        // 由于浮点数可能存在精度差异，这里不直接用 == 比较
        if (std::abs(c_scalar[i] - c_simd[i]) > 1e-6) {
            correct = false;
            break;
        }
    }

    std::cout << "------------------------------------" << std::endl;
    if (correct) {
        std::cout << "Verification: PASSED! The SIMD result is correct." << std::endl;
        if (duration_scalar.count() > 0) {
             std::cout << "Speedup: " << (duration_scalar.count() / duration_simd.count()) << "x" << std::endl;
        }
    } else {
        std::cout << "Verification: FAILED! The SIMD result is incorrect." << std::endl;
    }

    return 0;
}
