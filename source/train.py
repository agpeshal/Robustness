import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from adversarial_trainer import AdversarialTrainer
from attackers import PGD
from net.cnn_classifier import CNNClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--epochs', type=int, default=64, help="training epochs")
    parser.add_argument('--eps', type=float, default=0.3, help="max adversarial norm")
    parser.add_argument('--alpha', type=float, default=2/255, help="max per step noise")
    parser.add_argument('--steps', type=int, default=10, help="number of PGD steps")
    parser.add_argument('--eval_interval', type=int, default=5, help="interval for inference")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args.bs
    train_loader = DataLoader(datasets.CIFAR10(
        '../datasets/cifar10',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    ), shuffle=True, batch_size=batch_size)

    test_loader = DataLoader(datasets.CIFAR10(
        '../datasets/cifar10',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    ), shuffle=False, batch_size=batch_size)


    model = CNNClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    attacker = PGD(model, steps=args.steps, alpha=args.alpha, eps=args.eps, device=device)
    loss = nn.CrossEntropyLoss()

    trainer = AdversarialTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        attacker=attacker,
        loss=loss,
        epochs=args.epochs,
        optimizer=optimizer,
        eval_interval=args.eval_interval,
        device=device,
    )

    trainer.train()


if __name__ == '__main__':
    main()
