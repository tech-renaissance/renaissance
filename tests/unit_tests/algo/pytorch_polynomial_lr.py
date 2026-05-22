#!/usr/bin/env python3
"""
完全按照CLOSED.py的PolynomialLR实现和使用方式
第149-194行：PolynomialLR类定义
第354-366行：使用方式
"""

import torch
from torch.optim.lr_scheduler import LRScheduler

class PolynomialLR(LRScheduler):
    """Polynomial learning rate decay with warmup.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        base_lr: Peak learning rate after warmup.
        start_lr: Starting learning rate for warmup (default: 0.0).
        power: Polynomial power (default: 2.0).
        last_epoch: The index of last epoch (default: -1).
    """
    def __init__(self, optimizer, total_steps, warmup_steps, base_lr, start_lr=0.0, power=2.0, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.start_lr = start_lr
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # === Warmup 阶段：线性增长 ===
            # 处理 warmup_steps = 0 的边缘情况
            if self.warmup_steps > 0:
                alpha = self.last_epoch / self.warmup_steps
            else:
                alpha = 1.0
            return [self.start_lr + (self.base_lr - self.start_lr) * alpha
                    for _ in self.base_lrs]
        else:
            # === Polynomial Decay 阶段 ===
            # 【关键修复】去掉 +1 偏移，确保进度从 0 开始计数
            current_decay_step = self.last_epoch - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps

            # 边界保护：防止除以零
            if total_decay_steps > 0:
                progress = current_decay_step / total_decay_steps
                progress = min(progress, 1.0)  # 防止超调
            else:
                progress = 1.0

            # 标准 polynomial decay: lr = base_lr × (1 - progress)^power
            scale = (1.0 - progress) ** self.power
            return [self.base_lr * scale for _ in self.base_lrs]


def main():
    """完全按照CLOSED.py第354-366行的使用方式"""

    # === CLOSED.py第19-26行的MLPerf配置 ===
    REFERENCE_LR = 6.300
    REFERENCE_PER_GPU_BATCH = 256
    WARMUP_EPOCHS = 3
    NUM_EPOCHS = 34
    PER_GPU_BATCH = 512
    WARMUP_START_LR = 0.0

    # === CLOSED.py第227行：线性缩放学习率 ===
    poly_base_lr = REFERENCE_LR * (PER_GPU_BATCH / REFERENCE_PER_GPU_BATCH)

    # === CLOSED.py第355-357行：计算步数 ===
    # 假设每个epoch有100个update（模拟len(train_loader)）
    updates_per_epoch = 100
    total_updates = NUM_EPOCHS * updates_per_epoch
    warmup_updates = WARMUP_EPOCHS * updates_per_epoch

    print("Configuration (from CLOSED.py):")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  WARMUP_EPOCHS: {WARMUP_EPOCHS}")
    print(f"  poly_base_lr: {poly_base_lr}")
    print(f"  total_updates: {total_updates}")
    print(f"  warmup_updates: {warmup_updates}")

    # === CLOSED.py第359-366行：创建scheduler ===
    # 创建虚拟optimizer（PyTorch scheduler需要optimizer）
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=poly_base_lr)

    scheduler = PolynomialLR(
        optimizer=optimizer,
        total_steps=total_updates,
        warmup_steps=warmup_updates,
        base_lr=poly_base_lr,
        start_lr=WARMUP_START_LR,
        power=2.0
    )

    # === 输出每个batch的学习率（静态函数式查询，与C++ get_lr_by_batch语义一致）===
    total_decay_steps = total_updates - warmup_updates

    with open("pytorch_lr_output.txt", "w") as f:
        f.write("PyTorch (CLOSED.py) PolynomialLR Learning Rate Output\n")
        f.write("=" * 80 + "\n")
        f.write(f"Configuration: total_updates={total_updates}, warmup_updates={warmup_updates}, "
                f"base_lr={poly_base_lr}, start_lr={WARMUP_START_LR}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'batch_id':<10} {'phase':<10} {'lr':<15}\n")
        f.write("-" * 80 + "\n")

        print("Computing learning rates for each batch...")

        for batch_id in range(total_updates + 100):
            effective_step = batch_id
            if effective_step < 0:
                effective_step = 0
            if effective_step >= total_updates:
                effective_step = total_updates

            if effective_step < warmup_updates:
                phase = "Warmup"
                if warmup_updates > 0:
                    progress = effective_step / warmup_updates
                else:
                    progress = 1.0
                lr = WARMUP_START_LR + (poly_base_lr - WARMUP_START_LR) * progress
            else:
                phase = "Decay"
                decay_step = effective_step - warmup_updates
                if total_decay_steps > 0:
                    progress = decay_step / total_decay_steps
                    if progress > 1.0:
                        progress = 1.0
                else:
                    progress = 1.0
                lr = poly_base_lr * ((1.0 - progress) ** 2.0)

            f.write(f"{batch_id:<10} {phase:<10} {lr:<15.10f}\n")

            if batch_id % 100 == 0:
                print(f"  Processed batch {batch_id}")

    print(f"\nPyTorch (CLOSED.py) learning rates written to pytorch_lr_output.txt")
    print(f"Total batches: {total_updates + 100}")


if __name__ == "__main__":
    main()