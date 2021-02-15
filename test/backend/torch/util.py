import torch as th
from torch import nn


class SimpleProvider:
    def __init__(self):
        self.a = th.tensor([1, 2, 3], dtype=th.float)
        self.b = th.tensor([4], dtype=th.float)
        self.criteria = th.nn.MSELoss()
        self.model = SimpleModel()
        self.opt = th.optim.Adam(params=self.model.parameters())

    def train_one_iteration(self):
        self.opt.zero_grad()
        y_ = self.model(self.a)
        loss = self.criteria(y_, self.b)
        loss.backward()
        self.opt.step()


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)
