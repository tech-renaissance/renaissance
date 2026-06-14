import argparse
import math
import os
import time

# 限制 OpenMP/MKL 线程，防止 128 核被 8 进程 + DataLoader workers 抢爆
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Hyperparameters (aligned with test_nin.cpp)
# ---------------------------------------------------------------------------
TOTAL_EPOCHS = 200
GLOBAL_BATCH_SIZE = 128  # global batch size matches single-GPU C++ side
BASE_LR = 0.001
ETA_MIN = 1e-6
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16
PREFETCH_FACTOR = 8
EARLY_STOP_ACC = 92.0


# ---------------------------------------------------------------------------
# Distributed setup helpers
# ---------------------------------------------------------------------------
def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    if is_distributed():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_local_rank():
    return int(os.environ["LOCAL_RANK"]) if is_distributed() else 0


def reduce_tensor(tensor):
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt
    return tensor


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Warmup + CosineAnnealing scheduler
# ---------------------------------------------------------------------------
def warmup_cosine_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (TOTAL_EPOCHS - WARMUP_EPOCHS)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (ETA_MIN + (BASE_LR - ETA_MIN) * cos_val) / BASE_LR


# ---------------------------------------------------------------------------
# NIN (Network in Network) for CIFAR-10
# Conv bias: false | Activation: ReLU | GAP output
# ---------------------------------------------------------------------------
class NIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # mlpconv 1
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [对齐 CPP] maxpool(3,2,1)
            nn.Dropout(p=0.5),

            # mlpconv 2
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [对齐 CPP] maxpool(3,2,1)
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
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in")

    def forward(self, x):
        return self.features(x)


# ---------------------------------------------------------------------------
# Parameter groups: weight decay for weights, no wd for 1D params
# ---------------------------------------------------------------------------
def build_param_groups(model):
    weight_params = []
    no_wd_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1:
            no_wd_params.append(param)
        else:
            weight_params.append(param)
    return [
        {"params": weight_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 NIN benchmark (DDP + torch.compile)")
parser.add_argument("--cpu", action="store_true", help="Run on CPU (FP32, single-process only)")
parser.add_argument("--gpu", action="store_true", help="Run on GPU (FP32)")
parser.add_argument("--amp", action="store_true", help="Run on GPU with AMP (FP16)")


def main():
    args = parser.parse_args()

    setup_distributed()
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # Mode selection ---------------------------------------------------------
    if args.amp:
        mode = "AMP"
        use_amp = True
        if world_size > 1 or is_distributed():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    elif args.gpu:
        mode = "GPU"
        use_amp = False
        if world_size > 1 or is_distributed():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        if is_distributed():
            raise RuntimeError("DDP mode only supports --gpu or --amp")
        mode = "CPU"
        device = torch.device("cpu")
        use_amp = False

    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # 主进程线程限制，配合 OMP_NUM_THREADS 防止 CPU oversubscription
    torch.set_num_threads(2)

    # Seed: per-rank offset so different ranks see different augmentation
    torch.manual_seed(123)

    local_batch_size = GLOBAL_BATCH_SIZE // world_size
    if GLOBAL_BATCH_SIZE % world_size != 0:
        raise ValueError(f"GLOBAL_BATCH_SIZE {GLOBAL_BATCH_SIZE} must be divisible by world_size {world_size}")

    # Data pipeline ----------------------------------------------------------
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        transforms.RandomErasing(p=0.2, value=0),  # [对齐 CPP] fill=0
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    if os.name == "nt":
        data_root = "T:/dataset"
    else:
        data_root = "/root/epfs/dataset"

    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)

    pin_mem = device.type == "cuda"
    if is_distributed():
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set, batch_size=local_batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=pin_mem,
        pin_memory_device=f"cuda:{local_rank}" if pin_mem else "",
        num_workers=NUM_WORKERS,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
        multiprocessing_context="spawn",
    )
    val_loader = DataLoader(
        val_set, batch_size=local_batch_size, sampler=val_sampler,
        shuffle=False,
        pin_memory=pin_mem,
        pin_memory_device=f"cuda:{local_rank}" if pin_mem else "",
        num_workers=NUM_WORKERS,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
        multiprocessing_context="spawn",
    )

    # Model, loss, optimizer, scheduler --------------------------------------
    model = NIN().to(device).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    param_groups = build_param_groups(model)
    optimizer = optim.AdamW(
        param_groups, lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, fused=True
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # torch.compile warmup: trigger compilation before timed run -------------
    if hasattr(torch, "compile") and device.type == "cuda":
        log("\n--- Warmup: triggering torch.compile(max-autotune) ---")
        tw0 = time.perf_counter()

        dummy_data = torch.randn(local_batch_size, 3, 32, 32, device=device)
        dummy_data = dummy_data.to(memory_format=torch.channels_last)
        dummy_label = torch.randint(0, 10, (local_batch_size,), device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
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
        barrier()
        tw1 = time.perf_counter()
        log(f"    warmup done in {tw1 - tw0:.3f}s")

        # Re-initialize so the timed run starts from a clean state
        torch.manual_seed(123)
        model = NIN().to(device).to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="max-autotune", dynamic=False)

        param_groups = build_param_groups(model)
        optimizer = optim.AdamW(
            param_groups, lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, fused=True
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
        log("--- Re-initialized. Timed run begins. ---\n")

    # DDP wrapper ------------------------------------------------------------
    if is_distributed():
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            static_graph=True,
            gradient_as_bucket_view=True,
        )

    # Banner -----------------------------------------------------------------
    log("===========================================")
    log(" PyTorch CIFAR-10 NIN Benchmark")
    log("===========================================")
    log(f" Mode:       {mode}")
    log(f" Device:     {device}")
    if is_distributed():
        log(f" DDP:        rank={rank}, local_rank={local_rank}, world_size={world_size}")
    log(f" Network:    NIN (3 mlpconv + GAP)")
    log(f"   mlpconv1: conv(192,5) -> ReLU -> conv(192,1) -> ReLU -> conv(192,1) -> ReLU")
    log(f"             -> MaxPool(3,2,1) -> Dropout(0.5)")
    log(f"   mlpconv2: conv(192,5) -> ReLU -> conv(192,1) -> ReLU -> conv(192,1) -> ReLU")
    log(f"             -> MaxPool(3,2,1) -> Dropout(0.5)")
    log(f"   mlpconv3: conv(192,3) -> ReLU -> conv(192,1) -> ReLU -> conv(10,1)")
    log(f"             -> GAP")
    log(f" Conv bias:  false")
    log(f" Loss:       CrossEntropy (label_smoothing=0.05)")
    log(f" Optimizer:  AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)")
    log(f" Scheduler:  CosineAnnealing + Warmup({WARMUP_EPOCHS})")
    log(f" Batch:      global={GLOBAL_BATCH_SIZE}, local={local_batch_size} per GPU")
    log(f" Augment:    Pad(4) -> RandomCrop(32) -> HFlip -> RandomErasing(0.2)")
    log(f" Compile:    {'torch.compile(max-autotune)' if hasattr(torch, 'compile') and device.type == 'cuda' else 'disabled'}")
    log("===========================================")

    # Training loop ----------------------------------------------------------
    best_acc = 0.0
    best_epoch = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    barrier()
    t0 = time.perf_counter()

    for epoch in range(TOTAL_EPOCHS):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ep_t0 = time.perf_counter()
        model.train()

        train_loss = torch.tensor(0.0, device=device)
        train_total = torch.tensor(0.0, device=device)

        for data, target in train_loader:
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

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

            train_loss += loss * data.size(0)
            train_total += data.size(0)

        train_loss = reduce_tensor(train_loss) / reduce_tensor(train_total)

        # Validation
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        val_total = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                target = target.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)

                val_loss += loss * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum()
                val_total += data.size(0)

        val_loss = reduce_tensor(val_loss) / reduce_tensor(val_total)
        correct = reduce_tensor(correct)
        val_total = reduce_tensor(val_total)
        acc = 100.0 * correct / val_total
        acc_val = acc.item()

        if acc_val > best_acc:
            best_acc = acc_val
            best_epoch = epoch + 1

        ep_t1 = time.perf_counter()

        log(f"Epoch {epoch+1:3d}: train_loss={train_loss.item():.6f}  val_loss={val_loss.item():.6f}  "
            f"val_top1={acc_val:.2f}%  lr={current_lr:.6f}  time={ep_t1 - ep_t0:.1f}s")

        if acc_val >= EARLY_STOP_ACC:
            log(f"Early stop at epoch {epoch+1}: val_top1={acc_val:.2f}% >= {EARLY_STOP_ACC}%")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    barrier()
    t1 = time.perf_counter()

    log("\n===========================================")
    log(f" Mode:       {mode}")
    if is_distributed():
        log(f" DDP:        world_size={world_size}, local_batch={local_batch_size}")
    log(f" Best Top-1: {best_acc:.2f}%")
    log(f" Best Epoch: {best_epoch}")
    log(f" Total Time: {t1 - t0:.2f} s")
    log(f" Time/Epoch: {(t1 - t0) / TOTAL_EPOCHS:.2f} s")
    log("===========================================")

    if best_acc >= 91.19:
        log("EXCELLENT! Accuracy >= 91.19% (paper target)")
    elif best_acc >= 85.0:
        log("GOOD! Accuracy >= 85.0%")
    elif best_acc >= 60.0:
        log("ACCEPTABLE (smoke test). Accuracy >= 60.0%")
    else:
        log("NEEDS INVESTIGATION. Accuracy < 60.0%")
    log("===========================================")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
