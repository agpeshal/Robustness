import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.attackers import Attacker


class AdversarialTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        attacker: Attacker,
        loss: nn.Module,
        epochs: int,
        eval_interval: int,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        """Trainer for adversarial training on the entire dataset.

        Args:
            model (nn.Module): Classifier to train
            train_loader (DataLoader): Train dataloader
            test_loader (DataLoader): Test dataloader
            attacker (Attacker): White box adversarial attacker
            loss (nn.Module): Loss function for classification
            epochs (int): Total training epochs
            eval_interval (int): Epoch interval for eval run during training.
            optimizer (Optimizer): Optimizer to update model weights
            device (torch.device): Training device. CUDA or CPU.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.attacker = attacker
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def train(self) -> None:
        """Train model for specified epochs with evaluation after fixed intervals."""
        for i in tqdm(range(self.epochs), desc="Training ..."):
            self._train_epoch()

            if i % self.eval_interval == 0:
                self.eval()

    def eval(self) -> None:
        """Eval current model accuracy on clean samples."""
        self.model.eval()
        correct, total = 0, 0
        for images, labels in tqdm(iter(self.test_loader), desc="Evaluating ..."):
            images = images.to(self.device)
            labels = labels.to(self.device)
            predictions = self._predict(images).detach()
            correct += (labels == predictions).sum().cpu().numpy()
            total += len(labels)
        accuracy = correct / total * 100.0
        print(f"Test accuracy: {accuracy:.2f}")

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.model(images).data, dim=1)

    def _train_epoch(self) -> None:
        self.model.train()
        for images, labels in iter(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            adv_images = self.attacker.attack(images, labels)
            self.optimizer.zero_grad()
            outputs = self.model(adv_images)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
