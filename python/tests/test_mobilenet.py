#!/usr/bin/env python3
"""
@file test_mobilenet.py
@brief 测试从torch和torchvision加载MobileNet模型并打印结构
@version V3.14.1
@date 2026-02-25
@author 技术觉醒团队
@note 依赖项: torch, torchvision
"""

import torch
import torchvision.models as models


def print_model_structure(model, model_name):
    """打印模型结构信息"""
    print("=" * 80)
    print(f"{model_name} Structure:")
    print("=" * 80)
    print(model)

    print("\n" + "=" * 80)
    print(f"{model_name} Detailed Layer Information:")
    print("=" * 80)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只打印叶子节点
            print(f"{name}: {module}")

    print("\n" + "=" * 80)
    print(f"{model_name} Parameters Statistics:")
    print("=" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def main():
    """主函数"""
    print("Loading MobileNet models from PyTorch...\n")

    # ==================== MobileNet V1 ====================
    print("\n" + "=" * 80)
    print("Loading MobileNet V1...")
    print("=" * 80)
    mobilenet_v1 = models.mobilenet_v3_small(pretrained=False)
    print_model_structure(mobilenet_v1, "MobileNet V3 Small")

    # ==================== MobileNet V2 ====================
    print("\n\n" + "=" * 80)
    print("Loading MobileNet V2...")
    print("=" * 80)
    mobilenet_v2 = models.mobilenet_v2(pretrained=False)
    print_model_structure(mobilenet_v2, "MobileNet V2")


if __name__ == "__main__":
    main()
