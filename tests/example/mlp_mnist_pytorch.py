import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class UltimateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Match TR4: Kaiming Uniform (FAN_IN) for ReLU
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                # Match TR4: bias is initialized to ZEROS (initializer.cpp is_bias_region -> ZEROS)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


WARMUP_EPOCHS = 5
TOTAL_EPOCHS  = 100
ETA_MIN       = 1e-6
BASE_LR       = 0.001


def warmup_cosine_lambda(epoch):
    """
    匹配 C++ CosineAnnealingLR(base_lr=0.001, eta_min=1e-6, warmup=5) 的 step-by-epoch 行为。
    C++ 在 epoch 模式下，最后一个 epoch 精确降到 eta_min，因此衰减区间总长度 = total_epochs - warmup_epochs - 1。
    """
    if epoch <= WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS          # epoch 0 -> 0.0, ..., epoch 5 -> 1.0
    decay_total = max(1, TOTAL_EPOCHS - WARMUP_EPOCHS - 1)
    progress = (epoch - WARMUP_EPOCHS) / decay_total
    progress = min(progress, 1.0)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (ETA_MIN + (BASE_LR - ETA_MIN) * cos_val) / BASE_LR


if __name__ == '__main__':
    # 固定使用 GPU + AMP + torch.compile
    mode = "AMP"
    device = torch.device("cuda")

    print("\n========== COMPILE INFORMATION ==========", flush=True)
    print(f"  Device:     {device}", flush=True)
    print(f"  Mode:       {mode}", flush=True)
    print(f"  Network:    784->1024->512->256->10", flush=True)
    print(f"  Activation: ReLU", flush=True)
    print(f"  Optimizer:  AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)", flush=True)
    print(f"  Scheduler:  CosineAnnealing + Warmup(5)", flush=True)
    print("=========================================", flush=True)

    # 启用 TF32 与高精度矩阵运算，提升 AMP 训练速度
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    torch.manual_seed(123)
    batch_size = 200
    epochs = 100

    # -----------------------------------------------------------------------
    # Transforms: match the C++ example as closely as possible
    # Order: Pad -> RandomCrop -> RandomRotation -> RandomAffine(scale)
    #        -> ToTensor -> Normalize -> RandomErasing
    # -----------------------------------------------------------------------
    train_transform_list = [
        transforms.Pad((2, 2, 2, 2), fill=0),
        transforms.RandomCrop(28),
        transforms.RandomRotation(20, fill=0),          # [对齐 CPP] C++ RandomRotation(20.0f, 0)
        transforms.RandomAffine(0, scale=(0.8, 1.2)),   # [对齐 CPP] C++ RandomScale(0.8f, 1.2f)
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.5, value=0),       # [对齐 CPP] C++ RandomErasing(0.5f)
    ]
    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # [对齐 CPP] 使用与 C++ 示例相同的数据集路径：项目根目录下的 data/mnist
    project_root = Path(__file__).resolve().parents[2]
    mnist_root = project_root / "data" / "mnist"

    train_set = datasets.MNIST(str(mnist_root), train=True,  download=True, transform=train_transform)
    val_set   = datasets.MNIST(str(mnist_root), train=False, download=True, transform=val_transform)

    pin_mem = True
    # [对齐 CPP] num_workers=8 对齐 C++ preprocess_workers(8)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_mem, num_workers=8, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin_mem, num_workers=8, persistent_workers=True)

    model = UltimateMLP().to(device)

    # [对齐 CPP] torch.compile(max-autotune) 对齐 C++ 图级优化/算子融合
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # [对齐 CPP] bias 等一维参数不施加 weight_decay，与 C++ 底层 WEIGHT/BIAS 分离 kernel 行为一致
    weight_params, bias_like_params = [], []
    for param in model.parameters():
        if param.ndim <= 1:
            bias_like_params.append(param)
        else:
            weight_params.append(param)
    optimizer = optim.AdamW([
        {'params': weight_params, 'weight_decay': 1e-4},
        {'params': bias_like_params, 'weight_decay': 0.0},
    ], lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    scaler = torch.amp.GradScaler("cuda")

    # ====================================================================
    # WARMUP: 触发 torch.compile(max-autotune) 编译，隔离编译耗时
    # 流程：compile → dummy batch(触发编译) → sync → 打印耗时 →
    #       re-seed → re-init → re-compile → 正式开始计时
    # ====================================================================
    if hasattr(torch, "compile"):
        print("\n--- Warmup: triggering max-autotune compilation ---", flush=True)
        tw0 = time.perf_counter()

        dummy_data  = torch.randn(batch_size, 1, 28, 28, device=device)
        dummy_label = torch.randint(0, 10, (batch_size,), device=device)

        model.train()
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(dummy_data)
            l = criterion(out, dummy_label)
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = model(dummy_data)

        torch.cuda.synchronize()
        tw1 = time.perf_counter()
        print(f"    warmup done in {tw1 - tw0:.3f}s", flush=True)

        # Re-initialize everything so the timed run is pure training cost
        # [对齐 CPP] re-seed + re-init + re-compile，与 C++ task.compile() 后全新起点一致
        torch.manual_seed(123)
        model = UltimateMLP().to(device)
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")

        # [对齐 CPP] 重新初始化优化器，bias 等一维参数不施加 weight_decay
        weight_params, bias_like_params = [], []
        for param in model.parameters():
            if param.ndim <= 1:
                bias_like_params.append(param)
            else:
                weight_params.append(param)
        optimizer = optim.AdamW([
            {'params': weight_params, 'weight_decay': 1e-4},
            {'params': bias_like_params, 'weight_decay': 0.0},
        ], lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
        scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.0
    best_epoch = 0
    # [对齐 CPP] 计时前显式同步 CUDA，确保无未完成 kernel 干扰计时
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        ep_t0 = time.perf_counter()

        model.train()
        train_loss = 0.0

        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    output = model(data)
                    val_loss += criterion(output, target).item()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = 100.0 * correct / total

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        ep_t1 = time.perf_counter()

        print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
              f"val_top1={acc:.2f}%  lr={current_lr:.6f}  time={ep_t1 - ep_t0:.1f}s", flush=True)

        scheduler.step()

    # [对齐 CPP] 计时后显式同步 CUDA，确保所有 kernel 已完成再取时间
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print("\n========== TRAINING RESULT ==========", flush=True)
    print(f"  Best Top-1:      {best_acc:.2f}%", flush=True)
    print(f"  Best Epoch:      {best_epoch}", flush=True)
    print(f"  Total Time:      {t1 - t0:.2f} s", flush=True)
    print(f"  Time per Epoch:  {(t1 - t0) / epochs:.2f} s", flush=True)
    print("=====================================", flush=True)
