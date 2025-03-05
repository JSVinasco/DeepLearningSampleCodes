#!/usr/bin/env python3

"""
Some code snippets from the following tutorial

https://www.youtube.com/watch?v=qaDe0qQZ5AQ&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=8

with my personal comments
"""
# imports
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transfoms
import torchvision


# Create a do nothing module to replace parts
# of the original architecture
class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self,).__init__()

    def forward(self, x):
        return x


def check_accuracy(loader, model, device):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct} / {num_samples} with accuracy {acc}")
        model.train()

        return acc


def cnn_nout_features(nin: int, k: int, p: int, s: int) -> int:
    """
    Calculate the number of features size
    based on the inpute parameters

    :param: nin: number of input features : int
    :param: k: convulution kernel size: int
    :param: p: convolition padding size
    :param: s: convolution stride size
    :return: nout: number of output features
    """

    nout = ((nin + (2 * p) - k) / (s)) + 1

    return nout


def save_checkpoint(state: dict, filename="my_checkpoint.pt") -> None:
    print("--> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer) -> None:
    print("--> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def main():
    # model = NN(784, 10)
    # x = torch.randn(64, 784)
    # print(model(x).shape)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    inchannels = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5
    load_model = False
    # Load Data
    train_datasets = datasets.CIFAR10(
        root="dataset/", train=True, transform=transfoms.ToTensor(), download=True
    )

    train_dataloader = DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
    )

    test_datasets = datasets.CIFAR10(
        root="dataset/", train=False, transform=transfoms.ToTensor(), download=True
    )

    test_dataloader = DataLoader(
        dataset=test_datasets,
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize the network
    model = torchvision.models.vgg16(pretrained=True).to(device)
    print(model)

    # Modify elements in the network
    model.avgpool = Identity()
    model.classifier = nn.Linear(512, 10)

    # replace classifier one by one
    # for i in range(1,7):
    #     model.classifier[i] = Identity()

    model.to(device)
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load Model
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pt"),
                        model,
                        optimizer)

    # Train Network
    for epoch in tqdm(range(num_epochs)):
        losses = []

        # if epoch % 3 == 0:

        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     save_checkpoint(checkpoint)

        for batch_idx, (data, targets) in enumerate(train_dataloader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        # Check accuracy
        check_accuracy(train_dataloader, model, device)
        check_accuracy(test_dataloader, model, device)


if __name__ == "__main__":
    main()
