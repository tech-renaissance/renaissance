import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ------------------------------------------------------------------
# 命令行参数
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PyTorch MNIST MLP benchmark")
parser.add_argument("--cpu", action="store_true", help="Run on CPU")
parser.add_argument("--gpu", action="store_true", help="Run on GPU (FP32)")
parser.add_argument("--amp", action="store_true", help="Run on GPU with AMP (FP16)")
args = parser.parse_args()

# ------------------------------------------------------------------
# 模式选择
# ------------------------------------------------------------------
if args.amp:
    mode = "AMP"
    device = torch.device("cuda")
    use_amp = True
elif args.gpu:
    mode = "GPU"
    device = torch.device("cuda")
    use_amp = False
else:
    mode = "CPU"
    device = torch.device("cpu")
    use_amp = False

print(f"Mode: {mode}  Device: {device}")

# ------------------------------------------------------------------
# TF32（与 Renaissance .use_tf32(true) 对齐）
# ------------------------------------------------------------------
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# ------------------------------------------------------------------
# 超参数（与 Renaissance test_dl_full 对齐）
# ------------------------------------------------------------------
torch.manual_seed(42)
batch_size = 128
epochs = 4
lr = 0.1
momentum = 0.9

# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST("T:/dataset/mnist", train=True, download=False, transform=transform)
val_set   = datasets.MNIST("T:/dataset/mnist", train=False, download=False, transform=transform)

pin_mem = (device.type == "cuda")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  pin_memory=pin_mem)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

# ------------------------------------------------------------------
# 模型定义
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10, bias=True)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP().to(device)

# torch.compile 加速（PyTorch 2.0+）
# CPU 模式下 Windows 常因缺少 omp.h 导致编译失败，故仅在 GPU 启用
if hasattr(torch, "compile") and device.type == "cuda":
    print("Using torch.compile (max-autotune) ...")
    model = torch.compile(model, mode="max-autotune")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                      weight_decay=0.0, nesterov=False, dampening=0.0)

# AMP GradScaler
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# ------------------------------------------------------------------
# 训练循环
# ------------------------------------------------------------------
t0 = time.perf_counter()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    output = model(data)
                    val_loss += criterion(output, target).item()
            else:
                output = model(data)
                val_loss += criterion(output, target).item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_val_loss = val_loss / len(val_loader)
    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  val_acc={acc:.2f}%")

t1 = time.perf_counter()
print(f"\nFinal Val Accuracy: {acc:.2f}%")
print(f"Training time ({epochs} epochs): {t1 - t0:.3f}s")
