from torch import nn


class Simple(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 3, 3, 1)
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        raise NotImplementedError()
