import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

from attacks.config import aconf
from _utils.data import TData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ds_to_numpy(trainset, testset):
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets
    x = np.concatenate([x_train, x_test]).astype(np.float32) / 255
    y = np.concatenate([y_train, y_test]).astype(np.int32).squeeze()

    return x, y


def group_data(data, label):
    gr_data = []
    for i, j in zip(data, label):
        gr_data.append([i, j])
    return gr_data


def load_torch_cifar10():
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

    x, y = ds_to_numpy(trainset, testset)

    return TData(
        train_data=trainset.data,
        test_data=testset.data,
        train_labels=trainset.targets,
        test_labels=testset.targets,
        x_concat=x,
        y_concat=y
    )


def torch_predict(model, dataset):
    logits = []
    data_loader = DataLoader(dataset, batch_size=aconf['batch_size'])
    model.eval()
    for x in tqdm(data_loader):
        x = x.to(device)
        if x.shape[0] > 3:
            x = torch.transpose(x, 1, -1)
        pred = model(x)
        pred = pred.cpu().detach().numpy().copy()
        logits.append(pred)
    logits = np.array(logits)
    return np.concatenate(logits.copy(), axis=0)


def torch_train(model, dataset, checkpoint_path=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=aconf['lr'])

    train_loader = DataLoader(group_data(
        dataset.train_data, dataset.train_labels), batch_size=aconf['batch_size'], shuffle=True)

    for _ in range(aconf['epochs']):
        train_loss = 0
        train_acc = 0
        for x, y in tqdm(train_loader):
            x = torch.transpose(x, 1, -1).to(device)
            y = y.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            _, preds = torch.max(pred, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_acc += torch.sum(preds == y.data)

        print('train acc: {:.4f}, loss: {:.4f}'.format(
            train_acc / len(train_loader.dataset), train_loss / len(train_loader.dataset)))

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
