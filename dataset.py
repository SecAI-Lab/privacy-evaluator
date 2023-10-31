from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch


@dataclass
class TempData:
    target_trainloader: DataLoader
    target_valloader: DataLoader
    shadow_trainloader: DataLoader
    shadow_valloader: DataLoader


def get_cifar(n_classes, batch_size):
    transform1 = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if n_classes == 100:
        trainset = CIFAR100(root='./data', train=True,
                            download=True, transform=transform1)

        testset = CIFAR100(root='./data', train=False,
                           download=True, transform=transform1)
    else:
        trainset = CIFAR10(root='./data', train=True,
                           download=True, transform=transform1)

        testset = CIFAR10(root='./data', train=False,
                          download=True, transform=transform1)

    dataset = testset + trainset
    target_train, target_val, shadow_train, shadow_val = torch.utils.data.random_split(
        dataset, [25000, 25000, 5000, 5000]
    )

    target_trainloader = DataLoader(
        target_train, batch_size=batch_size, shuffle=True, num_workers=2)
    target_valloader = DataLoader(
        target_val, batch_size=batch_size, shuffle=False, num_workers=2)
    shadow_trainloader = DataLoader(
        shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
    shadow_valloader = DataLoader(
        shadow_val, batch_size=batch_size, shuffle=False, num_workers=2)

    return TempData(
        target_trainloader=target_trainloader,
        target_valloader=target_valloader,
        shadow_trainloader=shadow_trainloader,
        shadow_valloader=shadow_valloader
    )
