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

parser = argparse.ArgumentParser(description="PyTorch MNIST MLP Ultimate benchmark")
parser.add_argument("--cpu", action="store_true", help="Run on CPU (FP32)")
parser.add_argument("--gpu", action="store_true", help="Run on GPU (FP32)")
parser.add_argument("--amp", action="store_true", help="Run on GPU with AMP (FP16)")


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
    print(" PyTorch MNIST MLP Ultimate Benchmark", flush=True)
    print("===========================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print(f" Device:     {device}", flush=True)
    print(f" Network:    784->1024->512->256->10", flush=True)
    print(f" Activation: ReLU", flush=True)
    print(f" Optimizer:  AdamW (wd=1e-4)", flush=True)
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

    model = UltimateMLP().to(device)

    if hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ====================================================================
    # WARMUP  (GPU / AMP only)
    #
    # Purpose: trigger torch.compile(max-autotune) so compilation finishes
    #          BEFORE the timed 100-epoch loop starts.
    # After warmup: re-seed + re-init.  torch.compile FX-graph cache makes
    #               the second compilation near-instant.
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
        torch.manual_seed(42)
        model = UltimateMLP().to(device)
        model = torch.compile(model, mode="max-autotune")
        optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
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
