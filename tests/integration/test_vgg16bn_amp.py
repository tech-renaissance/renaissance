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
# Hyperparameters (aligned with VGG_RECIPE.md)
# ---------------------------------------------------------------------------
TOTAL_EPOCHS     = 100
LOCAL_BATCH_SIZE = 128        # per-GPU, global = 128*8 = 1024
PEAK_LR          = 0.2
WARMUP_LR_START  = 0.01
ETA_MIN          = 1e-6
WARMUP_EPOCHS    = 10
WEIGHT_DECAY     = 1e-4
MOMENTUM         = 0.9
LABEL_SMOOTHING  = 0.1
NUM_WORKERS      = 16

# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------
def setup():
    dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


# ---------------------------------------------------------------------------
# Warmup + CosineAnnealing scheduler
# ---------------------------------------------------------------------------
def warmup_cosine_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        alpha = epoch / max(WARMUP_EPOCHS - 1, 1)
        return (WARMUP_LR_START + (PEAK_LR - WARMUP_LR_START) * alpha) / PEAK_LR
    progress = (epoch - WARMUP_EPOCHS) / max(TOTAL_EPOCHS - WARMUP_EPOCHS - 1, 1)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (ETA_MIN + (PEAK_LR - ETA_MIN) * cos_val) / PEAK_LR


# ---------------------------------------------------------------------------
# VGG-16 with BatchNorm (VGG16BN)
# ---------------------------------------------------------------------------
class VGG16BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14 -> 7
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming Uniform (FAN_IN) for conv and fc; bias = zeros; BN weight = 1, bias = 0"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # Seed
    torch.manual_seed(123 + rank)

    # Data pipeline
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
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
        persistent_workers=True, prefetch_factor=8
    )
    val_loader = DataLoader(
        val_set, batch_size=LOCAL_BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=8
    )

    # Model (with SyncBN)
    model = VGG16BN().to(device).to(memory_format=torch.channels_last)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # [NEW] torch.compile with max-autotune
    model = torch.compile(model, mode="max-autotune")

    # Parameter grouping
    weight_params = []
    no_wd_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1:
            no_wd_params.append(param)
        else:
            weight_params.append(param)

    param_groups = [
        {'params': weight_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_wd_params, 'weight_decay': 0.0}
    ]

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)

    optimizer = optim.SGD(param_groups, lr=PEAK_LR,
                          momentum=MOMENTUM, nesterov=False)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    # [NEW] AMP: GradScaler for automatic mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # ====================================================================
    # WARMUP: trigger torch.compile(max-autotune) BEFORE timed loop
    #
    # 目的: 在计时的 100-epoch 循环开始之前完成所有编译。
    # 方法: 用 dummy 数据跑一次完整的 train fwd+bwd 和 eval fwd，
    #       触发 max-autotune 的所有 kernel 编译和 autotune。
    # 之后: 重新 seed + 重新初始化模型/优化器/scaler。
    #       torch.compile 的 FX graph 缓存使第二次编译近乎瞬时。
    # ====================================================================
    if is_main:
        log("")
        log("--- Warmup: triggering max-autotune compilation ---")
    dist.barrier()
    tw0 = time.perf_counter()

    dummy_data  = torch.randn(LOCAL_BATCH_SIZE, 3, 224, 224, device=device)
    dummy_data  = dummy_data.to(memory_format=torch.channels_last)
    dummy_label = torch.randint(0, 1000, (LOCAL_BATCH_SIZE,), device=device)

    # Train warmup
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        out = model(dummy_data)
        l = criterion(out, dummy_label)
    scaler.scale(l).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()

    # Eval warmup
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(dummy_data)

    torch.cuda.synchronize()
    dist.barrier()
    tw1 = time.perf_counter()
    if is_main:
        log(f"    warmup done in {tw1 - tw0:.3f}s")
        log("--- Re-initializing for timed run ---")

    # Re-initialize everything so the timed run is pure training cost
    torch.manual_seed(123 + rank)
    model = VGG16BN().to(device).to(memory_format=torch.channels_last)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.compile(model, mode="max-autotune")  # 第二次编译，缓存命中，近乎瞬时

    weight_params = []
    no_wd_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1:
            no_wd_params.append(param)
        else:
            weight_params.append(param)

    param_groups = [
        {'params': weight_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_wd_params, 'weight_decay': 0.0}
    ]

    optimizer = optim.SGD(param_groups, lr=PEAK_LR,
                          momentum=MOMENTUM, nesterov=False)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    scaler = torch.amp.GradScaler('cuda')

    if is_main:
        log("--- Timed 100-epoch run begins. ---")
        log("")

    if is_main:
        log("")
        log("=====================================")
        log(" PyTorch ImageNet VGG16BN Test (AMP + torch.compile)")
        log("=====================================")
        log(f" Mode:       GPU [AMP] (Eager, SyncBN, torch.compile max-autotune)")
        log(f" Device:     cuda:{local_rank} (world_size={world_size})")
        log(f" Network:    VGG-16-BN")
        log(f"   conv(64)x2+BN  -> maxpool -> 112x112")
        log(f"   conv(128)x2+BN -> maxpool -> 56x56")
        log(f"   conv(256)x3+BN -> maxpool -> 28x28")
        log(f"   conv(512)x3+BN -> maxpool -> 14x14")
        log(f"   conv(512)x3+BN -> maxpool -> 7x7")
        log(f"   flatten -> fc(4096) -> ReLU -> Dropout(0.5)")
        log(f"           -> fc(4096) -> ReLU -> Dropout(0.5) -> fc(1000)")
        log(f" Conv bias:  false | FC bias: true")
        log(f" Activation: ReLU (fixed)")
        log(f" Optimizer:  SGD (momentum=0.9, nesterov=False)")
        log(f" Scheduler:  CosineAnnealing + Warmup(10)")
        log(f" LR:         {WARMUP_LR_START} -> {PEAK_LR} (warmup) -> {ETA_MIN} (cosine)")
        log(f" Weight Decay: {WEIGHT_DECAY} (BN & bias excluded)")
        log(f" Augmentation: RRC(0.08~1.0) + HFlip + ColorJitter(0.2) + RandomErasing(0.25)")
        log(f" Training:   {TOTAL_EPOCHS} epochs, local_batch_size={LOCAL_BATCH_SIZE} per GPU")
        log(f" Global batch: {LOCAL_BATCH_SIZE * world_size}")
        log("=====================================")

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(TOTAL_EPOCHS):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        ep_t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()

        train_loss = torch.tensor(0.0, device=device)
        train_total = torch.tensor(0.0, device=device)

        for data, target in train_loader:
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # [NEW] AMP autocast for forward pass
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)

            # [NEW] AMP: scale loss, backward, unscale, clip, step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach() * data.size(0)
            train_total += data.size(0)

        # Aggregate train stats across all ranks
        train_loss = reduce_tensor(train_loss) / reduce_tensor(train_total)

        # Validation
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        correct1 = torch.tensor(0.0, device=device)
        correct5 = torch.tensor(0.0, device=device)
        val_total = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                target = target.to(device, non_blocking=True)

                # [NEW] AMP autocast for validation
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)

                val_loss += loss.detach() * data.size(0)

                # Top-1
                _, pred1 = output.topk(1, dim=1)
                correct1 += pred1.eq(target.view(-1, 1)).sum().item()

                # Top-5
                _, pred5 = output.topk(5, dim=1)
                correct5 += pred5.eq(target.view(-1, 1)).sum().item()

                val_total += data.size(0)

        # Aggregate val stats across all ranks
        val_loss = reduce_tensor(val_loss) / reduce_tensor(val_total)
        correct1 = reduce_tensor(correct1)
        correct5 = reduce_tensor(correct5)
        val_total = reduce_tensor(val_total)
        top1 = 100.0 * correct1 / val_total
        top5 = 100.0 * correct5 / val_total
        top1_val = top1.item()
        top5_val = top5.item()

        if top1_val > best_top1:
            best_top1 = top1_val
            best_top5 = top5_val
            best_epoch = epoch + 1

        ep_t1 = time.perf_counter()

        if is_main:
            log(f"Epoch {epoch+1:3d}: train_loss={train_loss.item():.6f}  "
                f"val_loss={val_loss.item():.6f}  top1={top1_val:.2f}%  top5={top5_val:.2f}%  "
                f"lr={current_lr:.6f}  time={ep_t1 - ep_t0:.1f}s")

    t1 = time.perf_counter()

    if is_main:
        log("")
        log("=====================================")
        log(" Mode:       GPU [AMP] (torch.compile max-autotune, DDP, SyncBN)")
        log("=====================================")
        log(f" Best Top-1: {best_top1:.2f}%")
        log(f" Best Top-5: {best_top5:.2f}%")
        log(f" Best Epoch: {best_epoch}")
        log(f" Total Time: {t1 - t0:.2f} s")
        log(f" Time/Epoch: {(t1 - t0) / TOTAL_EPOCHS:.2f} s")
        log("=====================================")

        if best_top1 >= 73.36:
            log("Target achieved: Top-1 >= 73.36%")
            log("Goal met!")
        elif best_top1 >= 65.0:
            log("Acceptable: Top-1 >= 65.0%")
            log("Performance: Above baseline")
        else:
            log("Needs improvement: Top-1 < 65.0%")
            log("Recommendations:")
            log("  1. Increase training epochs to 150+")
            log("  2. Adjust weight_decay")
            log("  3. Check SyncBN / gradient flow")
        log("=====================================")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()