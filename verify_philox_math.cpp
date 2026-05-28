#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>

// 测试Philox实现的正确性
void test_philox_math() {
    // 模拟Philox的实现
    uint32_t test_values[] = {0, 16777215, 33554431, 50331647};  // 测试用例

    std::cout << "Philox Uniform Float Implementation Test\n";
    std::cout << "==========================================\n\n";

    std::cout << "Implementation: (r[0] >> 8) * (1.0f / 16777216.0f)\n";
    std::cout << "Where r[0] is a 32-bit random integer\n\n";

    for (int i = 0; i < 4; i++) {
        uint32_t r0 = test_values[i];

        // Philox实现
        float result = static_cast<float>(r0 >> 8) * (1.0f / 16777216.0f);

        // 理论分析
        uint32_t shifted = r0 >> 8;  // 取高24位
        double expected_min = 0.0;
        double expected_max = static_cast<double>((1 << 24) - 1) / 16777216.0;  // (2^24-1)/2^24 ≈ 0.99999994

        std::cout << "Test " << i << ": r0 = " << r0 << " (0x" << std::hex << r0 << std::dec << ")\n";
        std::cout << "  shifted = r0 >> 8 = " << shifted << " (高24位)\n";
        std::cout << "  result = " << std::fixed << std::setprecision(10) << result << "\n";
        std::cout << "  Expected range: [" << expected_min << ", " << expected_max << "]\n";
        std::cout << "  Status: " << (result >= 0.0f && result < 1.0f ? "✓ PASS" : "✗ FAIL") << "\n\n";
    }

    // 关键分析
    std::cout << "\nMathematical Analysis:\n";
    std::cout << "====================\n";
    std::cout << "r[0] >> 8 gives a 24-bit integer in range [0, 2^24-1]\n";
    std::cout << "Multiplying by (1/2^24) gives range [0, (2^24-1)/2^24]\n";
    std::cout << "This is approximately [0, 0.99999994]\n";
    std::cout << "For practical purposes: [0, 1) range ✓\n\n";

    std::cout << "Testing extreme cases:\n";
    uint32_t min_val = 0;
    uint32_t max_val = 0xFFFFFFFF;

    float min_result = static_cast<float>(min_val >> 8) * (1.0f / 16777216.0f);
    float max_result = static_cast<float>(max_val >> 8) * (1.0f / 16777216.0f);

    std::cout << "r0 = 0x00000000: " << min_result << " (minimum)\n";
    std::cout << "r0 = 0xFFFFFFFF: " << max_result << " (maximum)\n";
    std::cout << "Both in [0, 1) range: " << ((min_result >= 0.0f && max_result < 1.0f) ? "✓" : "✗") << "\n";
}

int main() {
    test_philox_math();
    return 0;
}