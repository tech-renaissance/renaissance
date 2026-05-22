#!/usr/bin/env python3
"""
test_mnist_mlp_3.py
PyTorch 对照实现：MNIST 3-Layer MLP (784-512-256-10)
与 TR4 tests/top/mnist_mlp_3.cpp 保持参数一一对应
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ------------------------------------------------------------------
#  Global settings
# ------------------------------------------------------------------
torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20

DATASET_PATH = r"T:\dataset\mnist" if os.name == "nt" else "/root/epfs/dataset/mnist"

# ------------------------------------------------------------------
#  Data pipeline
# ------------------------------------------------------------------
# TR4 NormMode::MNIST => mean=0.1307, std=0.3081
transform = transforms.Compose([
    transforms.ToTensor(),                       # [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
])

train_dataset = datasets.MNIST(
    root=DATASET_PATH, train=True, download=True, transform=transform
)
val_dataset = datasets.MNIST(
    root=DATASET_PATH, train=False, download=True, transform=transform
)

# TR4 的 preprocess_workers 是"总预处理线程数"；
# PyTorch 的 num_workers 是"每张 GPU 的预处理线程数"。
# 换算关系：TR4 preprocess_workers = PyTorch num_workers × world_size。
#
# 经验数据（128 核 CPU + 8×GPU）：
#   PyTorch num_workers = 16（每张卡）→ TR4 preprocess_workers = 128（总计）。
#
# 本例 world_size = 1（单卡），理论对应 preprocess_workers = 16，
# 但 Windows 用 spawn（非 fork），worker 越多越慢；MNIST 数据集极小，
# 故设为 4 即可饱和单卡，与 TR4 版 preprocess_workers = 4 保持一致。
NUM_WORKERS = 4

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

# ------------------------------------------------------------------
#  Model definition: 784 -> 512 -> 256 -> 10
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 10, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (N, 1, 28, 28) -> flatten -> (N, 784)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)          # CrossEntropyLoss 内含 Softmax
        return x

model = MLP().to(DEVICE)

# ------------------------------------------------------------------
#  Training task configuration
# ------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=False,
    dampening=0.0,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# ------------------------------------------------------------------
#  Run training
# ------------------------------------------------------------------
if __name__ == '__main__':
    best_top1 = 0.0
    best_epoch = -1

    # 计时开始
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Epoch计时开始
        epoch_t0 = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        val_loss /= len(val_loader.dataset)
        top1 = correct / len(val_loader.dataset)

        # Epoch计时结束
        epoch_t1 = time.time()
        epoch_time = epoch_t1 - epoch_t0

        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Top-1: {top1 * 100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.3f}s"
        )

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch
            torch.save(model.state_dict(), "mnist_mlp_3.pt")

        scheduler.step()

    # 总计时结束
    t1 = time.time()
    elapsed = t1 - t0

    # ------------------------------------------------------------------
    #  Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 53)
    print(" PyTorch MNIST 3-Layer MLP (784-512-256-10)")
    print("-" * 53)
    print(f" Best Val Top-1:    {best_top1 * 100:.3f} %")
    print(f" Best Epoch:        {best_epoch}")
    print(f" Total Time:        {elapsed:.3f} s")
    print("=" * 53)
