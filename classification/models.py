import argparse
import torch
from torch import nn, optim

class EntMin(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 5),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 26 * 26, self.num_classes)
        )
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        return self.sm(self.net(x))

def test():
    # get data
    from datasets import CIFAR10
    from torch.utils.data import DataLoader
    data = CIFAR10()
    data_loader = DataLoader(
        data, batch_size=4, shuffle=True, num_workers=6, drop_last=True, pin_memory=True
    )
    for img, _ in data_loader:
        model = EntMin()
        y = model(img)
        print(y.shape)
        break

if __name__ == '__main__':
    test()
