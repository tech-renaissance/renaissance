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

parser = argparse.ArgumentParser(description="PyTorch MNIST mini-LeNet-BN benchmark")
parser.add_argument("--cpu", action="store_true", help="Run on CPU (FP32)")
parser.add_argument("--gpu", action="store_true", help="Run on GPU (FP32)")
parser.add_argument("--amp", action="store_true", help="Run on GPU with AMP (FP16)")
parser.add_argument("--activation", default="relu",
                    help="Activation function (relu, tanh, silu, relu6, leaky_relu, hardswish, elu, sigmoid)")


# ---------------------------------------------------------------------------
# Unbiased Batch Normalization (variance uses N-1 denominator)
# ---------------------------------------------------------------------------
class UnbiasedBatchNorm2d(nn.Module):
    """
    BatchNorm2d with unbiased variance estimation.
    Training:   var = sum((x - mean)^2) / (N*H*W - 1)   [Bessel's correction]
    Running:    running_var updated with the unbiased var
    Inference:  y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # Compute batch mean over N, H, W
            mean = x.mean(dim=[0, 2, 3], keepdim=True)          # [1, C, 1, 1]
            # Compute unbiased variance over N, H, W (N-1 denominator)
            var = x.var(dim=[0, 2, 3], unbiased=True, keepdim=True)  # [1, C, 1, 1]

            # Update running stats
            with torch.no_grad():
                mean_s = mean.squeeze()
                var_s = var.squeeze()
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_s
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_s
                self.num_batches_tracked += 1

            # Normalize using unbiased variance
            inv_std = 1.0 / torch.sqrt(var + self.eps)
            x_norm = (x - mean) * inv_std
        else:
            inv_std = 1.0 / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
            x_norm = (x - self.running_mean.view(1, -1, 1, 1)) * inv_std

        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class UnbiasedBatchNorm1d(nn.Module):
    """
    BatchNorm1d with unbiased variance estimation.
    Training:   var = sum((x - mean)^2) / (N - 1)   [Bessel's correction]
    Running:    running_var updated with the unbiased var
    Inference:  y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # Compute batch mean over N
            mean = x.mean(dim=0, keepdim=True)           # [1, C]
            # Compute unbiased variance over N (N-1 denominator)
            var = x.var(dim=0, unbiased=True, keepdim=True)  # [1, C]

            # Update running stats
            with torch.no_grad():
                mean_s = mean.squeeze()
                var_s = var.squeeze()
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_s
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_s
                self.num_batches_tracked += 1

            # Normalize using unbiased variance
            inv_std = 1.0 / torch.sqrt(var + self.eps)
            x_norm = (x - mean) * inv_std
        else:
            inv_std = 1.0 / torch.sqrt(self.running_var.view(1, -1) + self.eps)
            x_norm = (x - self.running_mean.view(1, -1)) * inv_std

        return x_norm * self.weight.view(1, -1) + self.bias.view(1, -1)


class MiniLeNetBN(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "silu":
            act = nn.SiLU()
        elif activation == "relu6":
            act = nn.ReLU6()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU(0.01)
        elif activation == "hardswish":
            act = nn.Hardswish()
        elif activation == "elu":
            act = nn.ELU()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Match C++ mini_lenet_bn:
        # conv(8,3,1,1) -> bn -> act -> flatten -> fc(400) -> bn -> act
        # -> fc(120) -> bn -> act -> fc(84) -> bn -> act -> fc(10)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            UnbiasedBatchNorm2d(8),
            act,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 8*28*28 = 6272
            nn.Linear(6272, 400, bias=True),
            UnbiasedBatchNorm1d(400),
            act,
            nn.Linear(400, 120, bias=True),
            UnbiasedBatchNorm1d(120),
            act,
            nn.Linear(120, 84, bias=True),
            UnbiasedBatchNorm1d(84),
            act,
            nn.Linear(84, 10, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        """Match TR4: Kaiming Uniform (FAN_IN) for all linear/conv layers"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (UnbiasedBatchNorm1d, UnbiasedBatchNorm2d)):
                nn.init.ones_(m.weight)
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

    print("=====================================", flush=True)
    print(" PyTorch MNIST mini-LeNet-BN (BN)", flush=True)
    print("=====================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print(f" Device:     {device}", flush=True)
    print(f" Network:    conv(8,3,1,1)->bn -> 400->bn -> 120->bn -> 84->bn -> 10", flush=True)
    print(f" Activation: {args.activation}", flush=True)
    print(f" Optimizer:  AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)", flush=True)
    print(f" Scheduler:  CosineAnnealing + Warmup(5)", flush=True)
    print(f" Loss:       CrossEntropy (label_smoothing=0.1)", flush=True)
    if not _HAS_AUTOCONTRAST:
        print(" Warning:    RandomAutocontrast not available (torchvision < 0.10)", flush=True)
    print("=====================================", flush=True)

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
                              pin_memory=pin_mem, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin_mem, num_workers=0)

    model = MiniLeNetBN(activation=args.activation).to(device)

    if hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Match C++: AdamW applies weight_decay=1e-4 to ALL parameters uniformly
    # (no param-group splitting; bias/BN gamma/beta decay at same rate as weights)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ====================================================================
    # WARMUP  (GPU / AMP only)
    # ====================================================================
    if hasattr(torch, "compile") and device.type == "cuda":
        print("\n--- Warmup: triggering max-autotune compilation ---", flush=True)
        tw0 = time.perf_counter()

        dummy_data  = torch.randn(batch_size, 1, 28, 28, device=device)
        dummy_label = torch.randint(0, 10, (batch_size,), device=device)

        model.train()
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(dummy_data)
                l = criterion(out, dummy_label)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(dummy_data)
            l = criterion(out, dummy_label)
            l.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(dummy_data)
            else:
                _ = model(dummy_data)

        torch.cuda.synchronize()
        tw1 = time.perf_counter()
        print(f"    warmup done in {tw1 - tw0:.3f}s", flush=True)

        # Re-initialize everything so the timed run is pure training cost
        torch.manual_seed(123)
        model = MiniLeNetBN(activation=args.activation).to(device)
        model = torch.compile(model, mode="max-autotune")

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
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
        print("--- Re-initialized.  Timed 100-epoch run begins. ---\n", flush=True)

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

        # Match C++ output format:
        #      1 |   2.657882 |   2.799003 |      9.66% |     0.000000 |    25.5s
        print(f"{epoch+1:6d} | {avg_train_loss:10.6f} | {avg_val_loss:10.6f} | {acc:9.2f}% | {current_lr:12.6f} | {ep_t1 - ep_t0:6.1f}s", flush=True)

        scheduler.step()

    t1 = time.perf_counter()
    print("\n=====================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print("=====================================", flush=True)
    print(f" Best Top-1: {best_acc:.2f}%", flush=True)
    print(f" Best Epoch: {best_epoch}", flush=True)
    print(f" Total Time: {t1 - t0:.2f} s", flush=True)
    print(f" Time/Epoch: {(t1 - t0) / epochs:.2f} s", flush=True)
    print("=====================================", flush=True)

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
    print("=====================================", flush=True)
