import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class ODEFunc(nn.Module):
    def __init__(self, gamma):
        super(ODEFunc, self).__init__()
        self.gamma = gamma  # γ 参数，可以学习

    def forward(self, t, y):
        # 定义常微分方程：dy/dt = -y + sin^2(y + γ)
        dy_dt = -y + torch.sin(y + self.gamma) ** 2
        return dy_dt


class NeuralMemoryODE(nn.Module):
    gamma: nn.Parameter

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, device: torch.device
    ):
        super(NeuralMemoryODE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # 可训练的控制参数 γ
        self.fc3 = nn.Linear(input_size, hidden_size)  # 输入到吸引子计算的映射
        self.pre_norm = nn.LayerNorm(input_size)
        self.post_norm = nn.LayerNorm(hidden_size)

        self.y = torch.randn(hidden_size).to(device)  # 状态 y 的初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(self.pre_norm(x))  # 输入到隐藏层
        y = self.post_norm(y)  # 归一化隐藏层输出

        for i in range(0, self.hidden_size, 64):
            y = self.fc1(self.pre_norm(x))  # 输入到隐藏层
            y = self.post_norm(y)
            end = min(i + 64, self.hidden_size)
            y[:, i:end] = self.update_state_with_attractor(y[:, i:end], x)

        y = torch.relu(y)  # 激活函数
        out = self.fc2(y)  # 最后通过输出层得到预测结果
        return out

    # 使用全局吸引子更新状态
    def update_state_with_attractor(self, y, x):
        attractor = self.solve_attractor(y, x)  # 求解吸引子
        # return -y + torch.pow(torch.sin(attractor), 2) + self.gamma * y
        return -y + torch.pow(torch.sin(attractor + self.gamma), 2)

    # 吸引子的计算
    def solve_attractor(self, y, x):
        # 使用权重 W(1) 和输入 x 映射到与 y 相同的维度
        x_mapped = self.fc3(x)  # 将输入 x 映射到与 y 相同的维度
        return y + torch.sin(x_mapped)  # 使用非线性激活函数


def test(
    model: NeuralMemoryODE, test_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train(
    model: NeuralMemoryODE,
    train_loader: DataLoader,
    optimizer: optim.AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels = data
        inputs: torch.Tensor = inputs.view(inputs.size(0), -1)  # 展平输入图像为一维向量

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs: torch.Tensor = model(inputs)  # 前向传播

        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100 * correct / total


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_set = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_set = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 设置模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralMemoryODE(3 * 32 * 32, 512, 10, device).to(
    device
)  # CIFAR-10图像的大小为 32x32，3个通道
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(20):
    loss, accuracy = train(model, train_loader, optimizer, criterion, device)
    test_accuracy = test(model, test_loader, device)
    print(
        f"Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}%, Test Accuracy: {test_accuracy}%"
    )
