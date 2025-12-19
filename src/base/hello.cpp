/**
 * @file hello.cpp
 * @brief renAIssance Framework Basic Hello World Implementation
 * @details 技术觉醒框架基础功能实现
 * @version 3.0.4
 * @date 2025-12-20
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <vector>

namespace renAIssance {

/**
 * @brief Hello World function that demonstrates framework usage
 */
void helloWorld() {
    Logger::info("Hello World from renAIssance Framework!");

    // 创建一个简单的张量
    Tensor tensor("hello_tensor");
    tensor.print();

    // 打印框架信息
    Framework::printWelcome();
}

/**
 * @brief Demonstrate basic framework functionality
 */
void demonstrateFramework() {
    Logger::info("Demonstrating renAIssance Framework capabilities...");

    // 显示框架版本
    Logger::info("Framework Version: " + Framework::getVersion());
    Logger::info("Framework Name: " + Framework::getName());

    // 创建多个张量进行演示
    std::vector<Tensor> tensors;
    tensors.emplace_back("input_tensor");
    tensors.emplace_back("weight_tensor");
    tensors.emplace_back("output_tensor");

    Logger::info("Created " + std::to_string(tensors.size()) + " tensors:");
    for (const auto& tensor : tensors) {
        tensor.print();
    }

    Logger::info("Framework demonstration completed successfully!");
}

/**
 * @brief Simple matrix operations placeholder
 */
class SimpleMatrix {
private:
    std::vector<float> data_;
    int rows_;
    int cols_;

public:
    SimpleMatrix(int rows, int cols, float value = 0.0f)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {
        Logger::info("Created matrix " + std::to_string(rows_) + "x" + std::to_string(cols_));
    }

    int getRows() const { return rows_; }
    int getCols() const { return cols_; }

    void printInfo() const {
        std::cout << "[" << RENAISSANCE_NAME << "] Matrix("
                  << rows_ << "x" << cols_ << ")" << std::endl;
    }
};

/**
 * @brief Test basic functionality
 */
void testBasicFunctionality() {
    Logger::info("Testing basic framework functionality...");

    // 测试张量
    Tensor t1("test_tensor");
    t1.setName("renamed_tensor");
    t1.print();

    // 测试简单矩阵
    SimpleMatrix mat(3, 4, 1.0f);
    mat.printInfo();

    Logger::info("Basic functionality test completed!");
}

} // namespace renAIssance