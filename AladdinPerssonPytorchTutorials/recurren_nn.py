#!/usr/bin/env python3
"""
Some code snippets from the following tutorial

https://www.youtube.com/watch?v=Gl2WXLIMvKA&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=5

with my personal comments
"""
#!/usr/bin/env python3
"""
Some code snippets from the following tutorial

https://www.youtube.com/watch?v=Jy4wM2X21u0&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=3

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


# Create a RNN
class RNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 sequence_lenght: int,
                 device: str,
                 )->None:
        super(RNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          num_layers,
                          batch_first=True)
        # N x time_sqeq x features
        self.fc = nn.Linear(hidden_size*sequence_lenght, num_classes)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0),
            self.hidden_size).to(self.device)

        # Forward Prop
        out, _ = self.rnn(x, h0)

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


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
            x = x.squeeze(1).to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = float(num_correct)/float(num_samples)*100
        print(f"Got {num_correct} / {num_samples} with accuracy {acc}")
        model.train()

        return acc

def main():
    # model = NN(784, 10)
    # x = torch.randn(64, 784)
    # print(model(x).shape)


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Hyper parameters
    input_size = 28
    sequence_lenght = 28
    num_layers = 2
    hidden_size = 256
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 2

    # Load Data
    train_datasets = datasets.MNIST(
        root="dataset/",
        train=True,
        transform=transfoms.ToTensor(),
        download=True)

    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  )

    test_datasets = datasets.MNIST(
        root="dataset/",
        train=False,
        transform=transfoms.ToTensor(),
        download=True)

    test_dataloader = DataLoader(dataset=test_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  )

    # Initialize the network
    model = RNN(input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                sequence_lenght=sequence_lenght,
                device=device).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            # Get data to cuda if possible
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)
            # print(data.shape)

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
