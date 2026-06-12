import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 NIN benchmark")
parser.add_argument("--cpu", action="store_true", help="Run on CPU (FP32)")
parser.add_argument("--gpu", action="store_true", help="Run on GPU (FP32)")
parser.add_argument("--amp", action="store_true", help="Run on GPU with AMP (FP16)")


class NIN(nn.Module):
    def __init__(self):
        super().__init__()
        # NIN for CIFAR-10: 3 mlpconv + GAP (no FC)
        # All conv layers have bias=False (match TR framework)
        self.features = nn.Sequential(
            # mlpconv 1
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(p=0.5),

            # mlpconv 2
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(p=0.5),

            # mlpconv 3 (no ReLU after final conv, no dropout per paper)
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0, bias=False),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self._init_weights()

    def _init_weights(self):
        """Match TR: Kaiming Uniform (FAN_IN) for all conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')

    def forward(self, x):
        x = self.features(x)
        return x


WARMUP_EPOCHS = 10
TOTAL_EPOCHS  = 200
ETA_MIN       = 1e-6
BASE_LR       = 0.001


def warmup_cosine_lambda(epoch):
    """
    Match TR CosineAnnealingLR + warmup(10) behaviour.
    """
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (TOTAL_EPOCHS - WARMUP_EPOCHS)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (ETA_MIN + (BASE_LR - ETA_MIN) * cos_val) / BASE_LR


if __name__ == '__main__':
    args = parser.parse_args()

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

    print("===========================================", flush=True)
    print(" PyTorch CIFAR-10 NIN Benchmark", flush=True)
    print("===========================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print(f" Device:     {device}", flush=True)
    print(f" Network:    NIN (3 mlpconv + GAP)", flush=True)
    print(f"   mlpconv1: conv(192,5) -> ReLU -> conv(192,1) -> ReLU -> conv(192,1) -> ReLU", flush=True)
    print(f"             -> MaxPool(3,2) -> Dropout(0.5)", flush=True)
    print(f"   mlpconv2: conv(192,5) -> ReLU -> conv(192,1) -> ReLU -> conv(192,1) -> ReLU", flush=True)
    print(f"             -> MaxPool(3,2) -> Dropout(0.5)", flush=True)
    print(f"   mlpconv3: conv(192,3) -> ReLU -> conv(192,1) -> ReLU -> conv(10,1)", flush=True)
    print(f"             -> GAP", flush=True)
    print(f" Conv bias:  false", flush=True)
    print(f" Loss:       CrossEntropy (label_smoothing=0.05)", flush=True)
    print(f" Optimizer:  AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)", flush=True)
    print(f" Scheduler:  CosineAnnealing + Warmup(10)", flush=True)
    print(f" Augment:    Pad(4) -> RandomCrop(32) -> HFlip -> RandomErasing(0.2)", flush=True)
    print("===========================================", flush=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    torch.manual_seed(123)
    batch_size = 128
    epochs = TOTAL_EPOCHS

    # CIFAR-10 normalization (match TR NormMode::CIFAR)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # T:\dataset\cifar-10 only has binary format (cifar-10-batches-bin) for CifarLoaderRaw.
    # torchvision.datasets.CIFAR10 needs Python pickle format (cifar-10-batches-py).
    # Download the Python version to T:/dataset so both formats coexist.
    data_root = "T:/dataset"
    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_mem, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=pin_mem, num_workers=4, persistent_workers=True)

    model = NIN().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # AdamW with weight_decay=1e-4 (applied to all parameters)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_acc = 0.0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        ep_t0 = time.perf_counter()

        model.train()
        train_loss = 0.0

        for data, target in train_loader:
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

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

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

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        ep_t1 = time.perf_counter()

        print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
              f"val_top1={acc:.2f}%  lr={current_lr:.6f}  time={ep_t1 - ep_t0:.1f}s", flush=True)

        scheduler.step()

    t1 = time.perf_counter()
    print("\n===========================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print(f" Best Top-1: {best_acc:.2f}%", flush=True)
    print(f" Best Epoch: {best_epoch}", flush=True)
    print(f" Total Time: {t1 - t0:.2f} s", flush=True)
    print(f" Time/Epoch: {(t1 - t0) / epochs:.2f} s", flush=True)
    print("===========================================", flush=True)

    if best_acc >= 91.19:
        print("EXCELLENT! Accuracy >= 91.19% (paper target)", flush=True)
    elif best_acc >= 85.0:
        print("GOOD! Accuracy >= 85.0%", flush=True)
    elif best_acc >= 60.0:
        print("ACCEPTABLE (smoke test). Accuracy >= 60.0%", flush=True)
    else:
        print("NEEDS INVESTIGATION. Accuracy < 60.0%", flush=True)
    print("===========================================", flush=True)
