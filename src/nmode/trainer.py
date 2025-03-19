import tqdm
import torch
import torch.nn as nn

from typing import Tuple
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from nmode.core import NeuralMemoryODE
from nmode.loader import get_cifar10_loaders, get_mnist_loaders


class MNISTTrainer:
    """Trainer for MNIST dataset."""

    input_dim: int = 28 * 28
    hidden_dim: int = 128
    output_dim: int = 10

    def __init__(
        self,
        device: torch.device,
    ):
        self.model = NeuralMemoryODE(
            self.hidden_dim,
            self.output_dim,
            encoder=nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
            ),
        ).to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=3,
        )

    def train(self, epochs: int = 10, batch_size: int = 256):
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)
            self.scheduler.step(test_loss)

            print(
                f"Epoch {epoch + 1}/{epochs}:\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%\n"
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%"
            )

    def train_epoch(self, loader: DataLoader[MNIST]) -> Tuple[float, float]:
        loader_len = len(loader)
        dataset_len = len(loader.dataset)  # type: ignore

        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        with torch.amp.autocast_mode.autocast(self.device.type) and tqdm.tqdm(
            loader, desc="Training", leave=False
        ) as p_bar:
            for x, y in p_bar:
                x: torch.Tensor
                y: torch.Tensor
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()

                total_samples += x.size(0)

                p_bar.set_postfix(
                    loss=f"{total_loss / total_samples:.4f}",
                    acc=f"{(correct / total_samples) * 100:.2f}%",
                )

        return total_loss / loader_len, correct / dataset_len

    def evaluate(self, loader: DataLoader[MNIST]) -> Tuple[float, float]:
        loader_len = len(loader)
        dataset_len = len(loader.dataset)  # type: ignore

        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0

        with (
            torch.no_grad(),
            tqdm.tqdm(loader, desc="Evaluating", leave=False) as p_bar,
        ):
            for x, y in p_bar:
                x: torch.Tensor
                y: torch.Tensor
                x, y = x.to(self.device), y.to(self.device)
                outputs: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.criterion(outputs, y)

                total_loss += float(loss.item())
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()

                total_samples += x.size(0)

                p_bar.set_postfix(
                    {
                        "loss": f"{total_loss / total_samples:.4f}",
                        "acc": f"{correct / total_samples:.2%}",
                    }
                )

        return total_loss / loader_len, correct / dataset_len


class CIFAR10Trainer:
    """Trainer for CIFAR10 dataset."""

    hidden_dim: int = 3 * 32 * 32
    output_dim: int = 10

    def __init__(self, device: torch.device):
        self.model = NeuralMemoryODE(
            self.hidden_dim,
            self.output_dim,
            encoder=nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
            ),
        ).to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )

        self.grad_clip = 1.0

    def train(self, epochs: int = 120, batch_size: int = 256):
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)
            self.scheduler.step()

            print(
                f"Epoch {epoch + 1}/{epochs}:\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%\n"
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%"
            )

    def train_epoch(self, loader: DataLoader):
        loader_len = len(loader)
        dataset_len = len(loader.dataset)  # type: ignore

        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.amp.autocast_mode.autocast(self.device.type) and tqdm.tqdm(
            loader, desc="Training", leave=False
        ) as p_bar:
            for x, y in p_bar:
                x: torch.Tensor
                y: torch.Tensor
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                outputs: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.criterion(outputs, y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_samples += x.size(0)

                p_bar.set_postfix(
                    loss=f"{total_loss / total_samples:.4f}",
                    acc=f"{correct / total_samples:.2%}",
                )

        self.scheduler.step()
        return total_loss / loader_len, correct / dataset_len

    def evaluate(self, loader: DataLoader):
        loader_len = len(loader)
        dataset_len = len(loader.dataset)  # type: ignore

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with (
            torch.no_grad(),
            tqdm.tqdm(loader, desc="Evaluating", leave=False) as p_bar,
        ):
            for x, y in p_bar:
                x: torch.Tensor
                y: torch.Tensor
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast_mode.autocast(self.device.type):
                    outputs: torch.Tensor = self.model(x)
                    loss: torch.Tensor = self.criterion(outputs, y)

                # 修复损失值累加方式
                total_loss += loss.item() * x.size(0)
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_samples += x.size(0)

                p_bar.set_postfix(
                    {
                        "loss": f"{total_loss / total_samples:.4f}",
                        "acc": f"{correct / total_samples:.2%}",
                    }
                )

        return total_loss / loader_len, correct / dataset_len
