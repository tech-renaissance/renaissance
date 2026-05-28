import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from multiprocessing import freeze_support

torch.manual_seed(42)

batch_size = 200
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与 TR 对齐：关闭 TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST("T:/dataset/mnist", train=True, download=False, transform=transform)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512, bias=True)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(256, 10, bias=True)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.tanh1(self.fc1(x))
        x = self.tanh2(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0, nesterov=False, dampening=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

    # Warmup
    print("Warmup...")
    torch.cuda.synchronize()
    with torch.no_grad():
        dummy = torch.randn(batch_size, 1, 28, 28, device=device)
        for _ in range(5):
            _ = model(dummy)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            current_lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"  batch {batch_idx}: lr={current_lr:.4f}  loss={loss.item():.4f}")
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")

    t1 = time.perf_counter()
    torch.cuda.synchronize()

    print(f"\nTraining time (3 epochs train only, eager): {t1 - t0:.3f}s")

if __name__ == '__main__':
    freeze_support()
    main()
