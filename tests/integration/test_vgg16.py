import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Hyperparameters (aligned with test_vgg16.cpp)
# ---------------------------------------------------------------------------
TOTAL_EPOCHS = 100
LOCAL_BATCH_SIZE = 256       # per-GPU, aligned with test_vgg16.cpp local_batch_size(256)
BASE_LR = 0.001
ETA_MIN = 1e-6
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 16

# ---------------------------------------------------------------------------
# DDP setup (aligned with main.py)
# ---------------------------------------------------------------------------
def setup():
    dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


# ---------------------------------------------------------------------------
# Warmup + CosineAnnealing scheduler (aligned with test_vgg16.cpp)
# Epoch 1 (epoch=0): lr = 0
# Epoch 6 (epoch=5): lr = base_lr
# Epoch 7+         : cosine annealing down to eta_min
# ---------------------------------------------------------------------------
def warmup_cosine_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (TOTAL_EPOCHS - WARMUP_EPOCHS)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (ETA_MIN + (BASE_LR - ETA_MIN) * cos_val) / BASE_LR


# ---------------------------------------------------------------------------
# VGG-16 without BN (aligned with test_vgg16.cpp)
# Conv bias: false | FC bias: true | Activation: ReLU
# ---------------------------------------------------------------------------
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14 -> 7
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        """Match TR4: Kaiming Uniform (FAN_IN) for conv and fc, bias = zeros"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    is_main = (rank == 0)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # TF32 (aligned with test_vgg16.cpp use_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # Seed
    torch.manual_seed(123 + rank)

    # Data pipeline (aligned with test_vgg16.cpp)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.25, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if os.name == 'nt':
        data_dir = r"T:\dataset\imagenet"
    else:
        data_dir = "/root/epfs/dataset/imagenet"

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(
        train_set, batch_size=LOCAL_BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_set, batch_size=LOCAL_BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    # Model (eager mode, no torch.compile, no AMP)
    model = VGG16().to(device).to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Parameter grouping: weight_decay for weights, no wd for bias
    # (aligned with test_vgg16.cpp AdamW behaviour)
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
        {'params': weight_params, 'weight_decay': WEIGHT_DECAY},
        {'params': bias_params, 'weight_decay': 0.0}
    ]

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)

    optimizer = optim.AdamW(param_groups, lr=BASE_LR,
                            betas=(0.9, 0.999), eps=1e-8)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    if is_main:
        log("")
        log("=====================================")
        log(" PyTorch ImageNet VGG-16 Test (DDP)")
        log("=====================================")
        log(f" Mode:       GPU [FP32] (Eager)")
        log(f" Device:     cuda:{local_rank} (world_size={world_size})")
        log(f" Network:    VGG-16 (no BN)")
        log(f"   conv(64)x2  -> maxpool -> 112x112")
        log(f"   conv(128)x2 -> maxpool -> 56x56")
        log(f"   conv(256)x3 -> maxpool -> 28x28")
        log(f"   conv(512)x3 -> maxpool -> 14x14")
        log(f"   conv(512)x3 -> maxpool -> 7x7")
        log(f"   flatten -> fc(4096) -> ReLU -> fc(4096) -> ReLU -> fc(1000)")
        log(f" Conv bias:  false | FC bias: true")
        log(f" Activation: ReLU (fixed)")
        log(f" Optimizer:  AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)")
        log(f" Scheduler:  CosineAnnealing + Warmup(5)")
        log(f" Augmentation: RandomResizedCrop + HFlip")
        log(f" Training:   {TOTAL_EPOCHS} epochs, local_batch_size={LOCAL_BATCH_SIZE} per GPU")
        log(f" Global batch: {LOCAL_BATCH_SIZE * world_size}")
        log("=====================================")

    best_acc = 0.0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(TOTAL_EPOCHS):
        ep_t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()

        train_loss = torch.tensor(0.0, device=device)
        train_total = torch.tensor(0.0, device=device)

        for data, target in train_loader:
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_total += data.size(0)

        # Aggregate train stats across all ranks
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

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                val_total += data.size(0)

        # Aggregate val stats across all ranks
        val_loss = reduce_tensor(val_loss) / reduce_tensor(val_total)
        correct = reduce_tensor(correct)
        val_total = reduce_tensor(val_total)
        acc = 100.0 * correct / val_total
        acc_val = acc.item()

        if acc_val > best_acc:
            best_acc = acc_val
            best_epoch = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        ep_t1 = time.perf_counter()

        if is_main:
            log(f"Epoch {epoch+1:3d}: train_loss={train_loss.item():.6f}  "
                f"val_loss={val_loss.item():.6f}  val_top1={acc_val:.2f}%  "
                f"lr={current_lr:.6f}  time={ep_t1 - ep_t0:.1f}s")

        scheduler.step()

        # Early stop (aligned with test_vgg16.cpp early_stop_by_top1(0.70f))
        if acc_val >= 70.0:
            if is_main:
                log(f"Early stop triggered at epoch {epoch+1}: Top-1 >= 70.0%")
            break

    t1 = time.perf_counter()

    if is_main:
        log("")
        log("=====================================")
        log(" Mode:       GPU [FP32] (Eager, DDP)")
        log("=====================================")
        log(f" Best Top-1: {best_acc:.2f}%")
        log(f" Best Epoch: {best_epoch}")
        log(f" Total Time: {t1 - t0:.2f} s")
        log(f" Time/Epoch: {(t1 - t0) / (epoch + 1):.2f} s")
        log("=====================================")

        if best_acc >= 70.0:
            log("Target achieved: Top-1 >= 70.0%")
            log("Goal met!")
        elif best_acc >= 65.0:
            log("Acceptable: Top-1 >= 65.0%")
            log("Performance: Above baseline")
        else:
            log("Needs improvement: Top-1 < 65.0%")
            log("Recommendations:")
            log("  1. Increase training epochs to 150+")
            log("  2. Adjust weight_decay (try 5e-4)")
            log("  3. Consider adding BatchNorm for stability")
        log("=====================================")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
