#include <iostream>
#include <mimalloc.h>

int main() {
    // 使用mimalloc分配内存
    int* numbers = static_cast<int*>(mi_malloc(10 * sizeof(int)));
    
    // 检查内存分配是否成功
    if (numbers == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }
    
    // 初始化数组
    for (int i = 0; i < 10; ++i) {
        numbers[i] = i + 1;
    }
    
    // 打印数组内容
    std::cout << "Hello from mimalloc!" << std::endl;
    std::cout << "Numbers: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
    
    // 使用mimalloc释放内存
    mi_free(numbers);
    
    return 0;
}
