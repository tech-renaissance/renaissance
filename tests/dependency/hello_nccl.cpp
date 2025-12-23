#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

// 定义一个检查 CUDA 调用返回值的宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 定义一个检查 NCCL 调用返回值的宏
#define NCCL_CHECK(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL Error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    // 1. 初始化 - 获取设备数量并检查
    int nGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nGpus));

    if (nGpus < 2) {
        std::cerr << "Error: This test requires at least 2 GPUs." << std::endl;
        return 1;
    }
    std::cout << "Found " << nGpus << " GPUs. Using GPU 0 and GPU 1 for this test." << std::endl;

    const int num_gpus_to_use = 2;
    const int data_size = 1024 * 1024; // 每个GPU上处理1M个浮点数

    // 为每个GPU分配 NCCL 通信器和 CUDA 流
    ncclComm_t comms[num_gpus_to_use];
    cudaStream_t streams[num_gpus_to_use];
    int devices[num_gpus_to_use] = {0, 1};

    // 为每个GPU分配设备端内存指针
    float* d_send_buffers[num_gpus_to_use];
    float* d_recv_buffers[num_gpus_to_use];

    // 2. 数据准备 - 在每个GPU上分配内存和初始化数据
    std::cout << "Preparing data on each GPU..." << std::endl;
    for (int i = 0; i < num_gpus_to_use; ++i) {
        int dev_id = devices[i];
        CUDA_CHECK(cudaSetDevice(dev_id));

        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        
        // 分配发送和接收缓冲区
        CUDA_CHECK(cudaMalloc(&d_send_buffers[i], sizeof(float) * data_size));
        CUDA_CHECK(cudaMalloc(&d_recv_buffers[i], sizeof(float) * data_size));

        // 在Host端准备数据
        std::vector<float> h_buffer(data_size);
        // GPU 0 的数据是 1.0f, GPU 1 的数据是 2.0f
        float value_to_fill = (float)(i + 1);
        std::fill(h_buffer.begin(), h_buffer.end(), value_to_fill);
        
        // 将Host数据异步拷贝到Device发送缓冲区
        CUDA_CHECK(cudaMemcpyAsync(d_send_buffers[i], h_buffer.data(), sizeof(float) * data_size, cudaMemcpyHostToDevice, streams[i]));

        // 初始化接收缓冲区为0 (可选，但好习惯)
        CUDA_CHECK(cudaMemsetAsync(d_recv_buffers[i], 0, sizeof(float) * data_size, streams[i]));
        
        std::cout << "  - GPU " << dev_id << " send buffer initialized with: " << value_to_fill << std::endl;
    }

    // 3. 初始化 NCCL 通信器
    std::cout << "Initializing NCCL communicators..." << std::endl;
    NCCL_CHECK(ncclCommInitAll(comms, num_gpus_to_use, devices));

    // 4. 执行 AllReduce 操作
    std::cout << "Performing ncclAllReduce(ncclSum)..." << std::endl;
    
    // 使用 ncclGroupStart/End 可以将多个通信操作组合在一起，以获得更好的性能
    // 对于单个操作，这不是必需的，但是一个好习惯
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus_to_use; ++i) {
        NCCL_CHECK(ncclAllReduce(
            (const void*)d_send_buffers[i], 
            (void*)d_recv_buffers[i], 
            data_size, 
            ncclFloat, 
            ncclSum, 
            comms[i], 
            streams[i]
        ));
    }
    NCCL_CHECK(ncclGroupEnd());
    
    // 5. 等待所有操作完成并验证结果
    std::cout << "Synchronizing streams and verifying results..." << std::endl;
    std::vector<float> h_result_buffer(data_size);

    // 同步GPU 0的流，并从GPU 0拷贝结果回Host
    CUDA_CHECK(cudaSetDevice(devices[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaMemcpy(h_result_buffer.data(), d_recv_buffers[0], sizeof(float) * data_size, cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool success = true;
    float expected_value = 3.0f; // 1.0f + 2.0f
    for (int i = 0; i < data_size; ++i) {
        if (std::abs(h_result_buffer[i] - expected_value) > 1e-5) {
            std::cerr << "Verification FAILED! at index " << i 
                      << ", expected " << expected_value 
                      << " but got " << h_result_buffer[i] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "✅  SUCCESS! NCCL AllReduce works correctly." << std::endl;
        std::cout << "First element on GPU 0 after AllReduce is: " << h_result_buffer[0] << std::endl;
        std::cout << "----------------------------------------\n" << std::endl;
    } else {
         std::cout << "\n----------------------------------------\n";
         std::cout << "❌ FAILED! Something went wrong with NCCL communication." << std::endl;
         std::cout << "----------------------------------------\n" << std::endl;
    }

    // 6. 清理资源
    std::cout << "Cleaning up resources..." << std::endl;
    for(int i = 0; i < num_gpus_to_use; ++i) {
        CUDA_CHECK(cudaSetDevice(devices[i]));
        CUDA_CHECK(cudaFree(d_send_buffers[i]));
        CUDA_CHECK(cudaFree(d_recv_buffers[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }
    
    std::cout << "Done." << std::endl;
    return 0;
}
