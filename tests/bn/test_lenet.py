import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# RandomAutocontrast compatibility (torchvision >= 0.10)
# ---------------------------------------------------------------------------
try:
    _RandomAutocontrast = transforms.RandomAutocontrast
    _HAS_AUTOCONTRAST = True
except AttributeError:
    _HAS_AUTOCONTRAST = False

parser = argparse.ArgumentParser(description="PyTorch MNIST LeNet-5 benchmark")
parser.add_argument("--cpu", action="store_true", help="Run on CPU (FP32)")
parser.add_argument("--gpu", action="store_true", help="Run on GPU (FP32)")
parser.add_argument("--amp", action="store_true", help="Run on GPU with AMP (FP16)")
parser.add_argument("--activation", default="tanh",
                    help="Activation function (relu, tanh)")


class LeNet5(nn.Module):
    def __init__(self, activation="tanh"):
        super().__init__()

        # 选择激活函数：仅支持 ReLU 和 Tanh
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}. Supported: relu, tanh")

        # 网络结构（与 Renaissance test_lenet.cpp 对齐）：
        # channel_padding (implicit in NCHW) -> conv(8,5,1,2) -> act -> avgpool(2,2,0)
        #   -> conv(16,5,1,0) -> act -> avgpool(2,2,0)
        #   -> fc(120) -> act -> fc(88) -> act -> fc(10)
        self.features = nn.Sequential(
            # C1: 8通道 5x5 卷积, padding=2 -> 28x28x8
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2, bias=False),
            act,
            # S2: 2x2 AvgPool, stride=2 -> 14x14x8
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # C3: 16通道 5x5 卷积, padding=0 -> 10x10x16
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0, bias=False),
            act,
            # S4: 2x2 AvgPool, stride=2 -> 5x5x16
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            # Flatten 5x5x16 = 400
            nn.Flatten(),
            # fc(120), 有bias
            nn.Linear(400, 120, bias=True),
            act,
            # fc(88), 有bias（8的倍数，兼容AMP模式）
            nn.Linear(120, 88, bias=True),
            act,
            # 输出: 10单元, 有bias
            nn.Linear(88, 10, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        """Match TR4: Kaiming Uniform (FAN_IN) for all linear/conv layers"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                # Match TR4: bias is initialized to ZEROS
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


WARMUP_EPOCHS = 5
TOTAL_EPOCHS  = 100
ETA_MIN       = 1e-6
BASE_LR       = 0.001


def warmup_cosine_lambda(epoch):
    """
    Match TR4 CosineAnnealingLR + warmup(5) behaviour exactly.
    Epoch 1 (epoch=0): lr = 0
    Epoch 2 (epoch=1): lr = 0.0002
    ...
    Epoch 6 (epoch=5): lr = 0.001  (base_lr reached)
    Epoch 7+          : cosine annealing down to eta_min
    """
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS          # 0 -> 0.0, 1 -> 0.2, ..., 4 -> 0.8
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
    print(" PyTorch MNIST LeNet-5 Benchmark", flush=True)
    print("===========================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print(f" Device:     {device}", flush=True)
    print(f" Network:    LeNet-5 (2 conv + 3 fc)", flush=True)
    print(f"   C1: conv(8,5,1,2) -> activation -> avgpool(2,2,0)", flush=True)
    print(f"   C3: conv(16,5,1,0) -> activation -> avgpool(2,2,0)", flush=True)
    print(f"   Flatten(400) -> fc(120) -> activation -> fc(88) -> activation -> fc(10)", flush=True)
    print(f" Conv bias:  false", flush=True)
    print(f" FC bias:    true", flush=True)
    print(f" Activation: {args.activation}", flush=True)
    print(f" Optimizer:  AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)", flush=True)
    print(f" Scheduler:  CosineAnnealing + Warmup(5)", flush=True)
    if not _HAS_AUTOCONTRAST:
        print(" Warning:    RandomAutocontrast not available (torchvision < 0.10)", flush=True)
    print("===========================================", flush=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    torch.manual_seed(123)
    batch_size = 128
    epochs = 100

    # -----------------------------------------------------------------------
    # Transforms: match TR4 as closely as possible
    # TR4 order: Pad -> RandomRotation -> RandomScale -> RandomCrop
    #            -> RandomAutocontrast -> (ToTensor + Normalize + RandomErasing)
    # -----------------------------------------------------------------------
    train_transform_list = [
        transforms.Pad((2, 2, 2, 2), fill=0),
        transforms.RandomRotation(15, fill=0),
        transforms.RandomAffine(0, scale=(0.9, 1.1)),
        transforms.RandomCrop(28),
    ]
    if _HAS_AUTOCONTRAST:
        train_transform_list.append(transforms.RandomAutocontrast(p=0.5))
    train_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.5),
    ]
    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST("T:/dataset/mnist", train=True,  download=False, transform=train_transform)
    val_set   = datasets.MNIST("T:/dataset/mnist", train=False, download=False, transform=val_transform)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_mem, num_workers=8, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin_mem, num_workers=8, persistent_workers=True)

    model = LeNet5(activation=args.activation).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 参数分组：weight 用 AdamW (wd=1e-4)，bias/1D 参数用 Adam (wd=0.0)
    # 匹配 TR4 行为：BN Weight、BN Bias、FC Bias 固定 WD=0
    weight_params = []
    bias_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or len(param.shape) == 1:
            bias_params.append(param)
        else:
            weight_params.append(param)

    param_groups = [
        {'params': weight_params, 'weight_decay': 1e-4},
        {'params': bias_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(param_groups, lr=BASE_LR,
                            betas=(0.9, 0.999), eps=1e-8)
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

    # 打印评估结果
    if best_acc >= 98.5:
        print("EXCELLENT! Accuracy >= 98.5%", flush=True)
        print("Target achieved: Goal met!", flush=True)
    elif best_acc >= 98.0:
        print("GOOD! Accuracy >= 98.0%", flush=True)
        print("Performance: Above baseline", flush=True)
    elif best_acc >= 97.5:
        print("ACCEPTABLE. Accuracy >= 97.5%", flush=True)
        print("Consider: Increasing epochs or adjusting hyperparameters", flush=True)
    else:
        print("NEEDS IMPROVEMENT. Accuracy < 97.5%", flush=True)
        print("Recommendations:", flush=True)
        print("  1. Increase training epochs to 150+", flush=True)
        print("  2. Adjust weight_decay (try 5e-4 or 5e-5)", flush=True)
        print("  3. Increase network width to fc(1152, true)", flush=True)
    print("===========================================", flush=True)
