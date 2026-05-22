#!/usr/bin/env python3
"""
CLOSED.py PolynomialLR 学习率计算脚本
完全按照CLOSED.py第149-194行的逻辑实现
用于与renAIssance C++实现进行逐行对比
"""

def polynomial_lr_get_lr(last_epoch, total_steps, warmup_steps, base_lr, start_lr, power):
    """
    CLOSED.py的PolynomialLR.get_lr()的忠实复现

    Args:
        last_epoch: 当前步数索引
        total_steps: 总训练步数
        warmup_steps: warmup步数
        base_lr: 峰值学习率
        start_lr: warmup起始学习率
        power: 多项式幂次

    Returns:
        当前学习率
    """
    if last_epoch < warmup_steps:
        # === Warmup 阶段：线性增长 ===
        if warmup_steps > 0:
            alpha = last_epoch / warmup_steps
        else:
            alpha = 1.0
        return start_lr + (base_lr - start_lr) * alpha
    else:
        # === Polynomial Decay 阶段 ===
        current_decay_step = last_epoch - warmup_steps
        total_decay_steps = total_steps - warmup_steps

        if total_decay_steps > 0:
            progress = current_decay_step / total_decay_steps
            progress = min(progress, 1.0)  # 防止超调
        else:
            progress = 1.0

        # 标准 polynomial decay: lr = base_lr × (1 - progress)^power
        scale = (1.0 - progress) ** power
        return base_lr * scale


def main():
    """主函数：按照CLOSED.py第354-366行的MLPerf标准配置"""

    # === MLPerf标准配置（CLOSED.py第19-26行）===
    REFERENCE_LR = 6.300
    REFERENCE_PER_GPU_BATCH = 256
    WARMUP_EPOCHS = 3
    NUM_EPOCHS = 34
    PER_GPU_BATCH = 512
    WARMUP_START_LR = 0.0
    power = 2.0

    # 线性缩放学习率（CLOSED.py第227行）
    poly_base_lr = REFERENCE_LR * (PER_GPU_BATCH / REFERENCE_PER_GPU_BATCH)

    # 假设每个epoch有100个update（batch）
    updates_per_epoch = 100
    total_updates = NUM_EPOCHS * updates_per_epoch
    warmup_updates = WARMUP_EPOCHS * updates_per_epoch

    print("=" * 80)
    print("CLOSED.py PolynomialLR 学习率计算")
    print("=" * 80)
    print(f"配置参数:")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  WARMUP_EPOCHS: {WARMUP_EPOCHS}")
    print(f"  PER_GPU_BATCH: {PER_GPU_BATCH}")
    print(f"  poly_base_lr: {poly_base_lr} (线性缩放后)")
    print(f"  WARMUP_START_LR: {WARMUP_START_LR}")
    print(f"  power: {power}")
    print(f"  updates_per_epoch: {updates_per_epoch}")
    print(f"  total_updates: {total_updates}")
    print(f"  warmup_updates: {warmup_updates}")
    print("=" * 80)

    # 选择关键测试点
    test_steps = [
        0,          # warmup起点
        50,         # warmup中点
        150,        # warmup后期
        299,        # warmup最后一步
        300,        # decay起点
        850,        # decay中点
        1200,       # decay后期
        1699,       # decay最后一步
        1700,       # 超出范围测试
        2000,       # 更大超出范围
    ]

    print(f"\n{'last_epoch':<12} {'阶段':<10} {'学习率':<15} {'计算说明'}")
    print("-" * 80)

    for last_epoch in test_steps:
        lr = polynomial_lr_get_lr(
            last_epoch, total_updates, warmup_updates,
            poly_base_lr, WARMUP_START_LR, power
        )

        # 确定阶段和计算说明
        if last_epoch < warmup_updates:
            phase = "Warmup"
            if warmup_updates > 0:
                alpha = last_epoch / warmup_updates
                explanation = f"α={alpha:.4f}, lr={WARMUP_START_LR}+({poly_base_lr}-{WARMUP_START_LR})×α"
            else:
                explanation = "warmup_steps=0, 直接返回base_lr"
        else:
            phase = "Decay"
            current_decay_step = last_epoch - warmup_updates
            total_decay_steps = total_updates - warmup_updates
            progress = current_decay_step / total_decay_steps
            progress = min(progress, 1.0)
            scale = (1.0 - progress) ** power
            explanation = f"progress={progress:.4f}, scale={scale:.6f}, lr={poly_base_lr}×scale"

        print(f"{last_epoch:<12} {phase:<10} {lr:<15.8f} {explanation}")

    print("=" * 80)
    print("数学验证点:")
    print("-" * 80)

    # 特殊验证点
    points = [
        (0, "warmup起点", f"应该等于WARMUP_START_LR={WARMUP_START_LR}"),
        (299, "warmup终点前", f"应该略小于base_lr={poly_base_lr}"),
        (300, "decay起点", f"应该精确等于base_lr={poly_base_lr}"),
        (850, "decay中点", "计算公式验证"),
        (1699, "decay终点", "应该接近0但不为0"),
    ]

    for step, name, expected in points:
        lr = polynomial_lr_get_lr(
            step, total_updates, warmup_updates,
            poly_base_lr, WARMUP_START_LR, power
        )
        print(f"Step {step:4d} ({name}): lr={lr:.10f}, {expected}")

    print("=" * 80)


if __name__ == "__main__":
    main()