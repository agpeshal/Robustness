
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataset(split):

    return datasets.MNIST(
        '../datasets/mnist',
        train=split=='train',
        transform=transforms.ToTensor()
    )


def get_cifar10_dataset(split):

    return datasets.CIFAR10(
        '../datasets/cifar10',
        train=split=='train',
        transform=transforms.ToTensor()
    )


def get_dataloader(dataset_name, split, batch_size=1, shuffle=False):

    if dataset_name == 'mnist':
        dataset = get_mnist_dataset(split)
    
    elif dataset_name == 'cifar10':
        dataset = get_cifar10_dataset(split)
    
    else:
        raise NotImplementedError
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
