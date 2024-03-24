from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(
    batch_size: int = 1, train: bool = True, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        datasets.CIFAR10(
            "../../datasets/cifar10",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        ),
        shuffle=shuffle,
        batch_size=batch_size,
    )
