/**
 * @file staging_debug.cpp
 * @brief NUMA绑核策略和StagingBufferPool分配调试测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/top
 * @note 本测试只到 .commit()，触发 StagingBufferPool + TransferStation 配置日志。
 *       PW创建日志在完整训练任务（resnet50_c）的 GPU_CLOUD 场景中可见。
 */

#include "renaissance.h"

using namespace tr;

int main() {
#ifdef _WIN32
    const std::string dataset_path = "T:\\dataset\\imagenet";
#else
    const std::string dataset_path = "/root/epfs/dataset/imagenet";
#endif

    GLOBAL_SETTING
        .use_gpu()  // 自动使用所有可用GPU
        .auto_seed()
        .local_batch_size(128)
        .train_resolution(224)
        .val_resolution(224)
        .amp(true);

    PREPROCESSOR_SETTING
        .dataset("imagenet", dataset_path)
        .load_workers(2)
        .preprocess_workers(8)  // 修改为8个worker，确保能被8个GPU整除
        .train_transforms(
            FastRandomResizedCrop(224, {0.08f, 1.0f}, {0.75f, 1.333f}),
            RandomHorizontalFlip())
        .val_transforms(
            Resize(256),
            CenterCrop(224))
        .normalization(NormMode::MLPERF)
        .commit();

    return 0;
}