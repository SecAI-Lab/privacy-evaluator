from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch

from _utils.data import TData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_torch_cifar10(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform)

    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform)

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=2)

    return TData(
        train_data=train_loader,
        test_data=test_loader
    )


def torch_predict(model, data_loader):
    logits = []
    labels = []
    for x, y in tqdm(data_loader):
        inp = x.to(device)
        pred = model(inp)

        logits.append(pred.cpu().detach())
        labels.append(y)

    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)
