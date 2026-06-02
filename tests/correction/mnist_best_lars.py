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

parser = argparse.ArgumentParser(description="PyTorch MNIST MLP Ultimate benchmark (eager mode, no compile)")
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
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


WARMUP_EPOCHS = 5
TOTAL_EPOCHS  = 100
ETA_MIN       = 1e-6
BASE_LR       = 0.5  # 提高学习率，LARS通常需要更高学习率


def warmup_cosine_lambda(epoch):
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
    print(" PyTorch MNIST MLP Ultimate (LARS vs TR)", flush=True)
    print("===========================================", flush=True)
    print(f" Mode:       {mode}", flush=True)
    print(f" Device:     {device}", flush=True)
    print(f" Network:    784->1024->512->256->10", flush=True)
    print(f" Activation: ReLU", flush=True)
    print(f" Optimizer:  LARS (m=0.9, wd=5e-5, tc=0.001, eps=0.0)", flush=True)
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
                              pin_memory=pin_mem, num_workers=4, persistent_workers=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin_mem, num_workers=4, persistent_workers=False)

    model = UltimateMLP().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # MLPerf Closed Division LARS配置
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-5
    LARS_TRUST_COEF = 0.001
    LARS_EPS = 0.0

    # 参数分组：权重使用LARS，BN/bias使用标准SGD
    weight_params = []
    bn_bias_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # BN层参数或bias参数使用标准SGD（trust_coefficient=0）
        if 'bn' in name or 'bias' in name or len(param.shape) == 1:
            bn_bias_params.append(param)
        else:
            weight_params.append(param)

    param_groups = [
        {
            'params': weight_params,
            'weight_decay': WEIGHT_DECAY,
            'trust_coefficient': LARS_TRUST_COEF
        },
        {
            'params': bn_bias_params,
            'weight_decay': 0.0,
            'trust_coefficient': 0.0
        }
    ]

    # 使用CLOSED.py的LARS实现
    class LARS(optim.Optimizer):
        def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=0.0,
                     trust_coefficient=0.001, eps=1e-8, nesterov=False):
            defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                           trust_coefficient=trust_coefficient, eps=eps, nesterov=nesterov)
            super(LARS, self).__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                trust_coefficient = group['trust_coefficient']
                eps = group['eps']
                lr = group['lr']
                nesterov = group['nesterov']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data

                    # Step 1: 计算Trust Ratio（使用原始梯度）
                    if trust_coefficient > 0:
                        param_norm = torch.norm(p.data)
                        grad_norm = torch.norm(grad)

                        if param_norm > 1e-12 and grad_norm > 1e-12:
                            # 标准LARS公式
                            trust_ratio = trust_coefficient * param_norm / (
                                grad_norm + weight_decay * param_norm + eps
                            )
                            trust_ratio = min(trust_ratio, 100.0)
                        else:
                            trust_ratio = 1.0
                    else:
                        trust_ratio = 1.0

                    # Step 2: 施加Weight Decay
                    if weight_decay != 0:
                        grad = grad.add(p.data, alpha=weight_decay)

                    # Step 3: Momentum更新
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach() * (lr * trust_ratio)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=lr * trust_ratio)

                    # Step 4: Nesterov
                    if nesterov:
                        update = grad.mul(lr * trust_ratio).add(buf, alpha=momentum)
                    else:
                        update = buf

                    # Step 5: 应用更新
                    p.data.add_(update, alpha=-1.0)

            return loss

    optimizer = LARS(param_groups, lr=BASE_LR, momentum=MOMENTUM,
                     weight_decay=0.0, trust_coefficient=LARS_TRUST_COEF,
                     eps=LARS_EPS, nesterov=False)
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