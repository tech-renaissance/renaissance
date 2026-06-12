模型：VGG16BN
数据集：ImageNet
硬件：单机八卡，DDP，A100×8（显存80GB版）
总epoch数：100
warmup epoch数：5
准确率目标：Top-1超过73.36%
batch size：128×8=1024
峰值LR：0.4
优化器：SGD with momentum，不开Nesterov
Momentum：0.9
Weight Decay：1e-4
学习率调度器：余弦退火（5 epoch的warmup，0.01→0.4，eta_min=1e-6）
Label Smoothing：0.1
RandomErasing：p=0.5, scale=(0.02, 0.33)
训练数据增强：RandomResizedCrop(224，scale 0.08~1.0)+RandomHorizontalFlip+ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)+RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
验证数据正确：Resize(256)+CenterCrop(224)
Normalize：ImageNet标准（[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]）
模型内Dropout：p=0.5
早停：不开启
模型EMA/SEMA：不开启
TF32：开启
输出：train loss、val loss、top-1、top-5、学习率
BN层跨卡同步（SyncBN）：开启
TR4框架已经确保BN参数的Weight Decay为0