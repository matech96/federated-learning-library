from typing import Callable

import torch as th
from torch import nn

from fll.backend.torchbackend import TorchBackendFactory, binary_accuracy


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


class SimpleDataSet(th.utils.data.Dataset):
    def __getitem__(self, index):
        return index

    def __len__(self):
        return 10


class SignModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()

        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.fc(x)


class SignModelComplex(nn.Module):
    def __init__(self, n_features: int = 10):
        super().__init__()

        self.fc0 = nn.Linear(n_features, n_features)
        self.batchnorm = nn.BatchNorm1d(n_features)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc1(x)


class SignDataSet(th.utils.data.Dataset):
    def __init__(self, length: int = 100, n_features: int = 10):
        self.length = length
        self.x = th.randn(length, n_features)
        self.y = (th.sum(self.x, 1) > 0).to(th.float32)

    def __getitem__(self, item):
        return self.x[item, ], self.y[[item], ]

    def __len__(self):
        return self.length


class SignProvider:
    def __init__(self, length: int = 10, n_features: int = 10, is_complex_model: bool = False,
                 opt_class: Callable[..., th.optim.Optimizer] = th.optim.SGD):
        ds = SignDataSet(length=length, n_features=n_features)
        dl = th.utils.data.DataLoader(ds, batch_size=2)
        model = SignModelComplex(n_features) if is_complex_model else SignModel(n_features)
        opt = opt_class(lr=0.1, params=model.parameters())
        loss = th.nn.BCEWithLogitsLoss()
        acc = binary_accuracy

        self.dl = TorchBackendFactory.create_data_loader(dl)
        self.model = TorchBackendFactory.create_model(model)
        self.loss = TorchBackendFactory.create_loss(loss, 'bce')
        self.opt = TorchBackendFactory.create_opt(opt)
        self.metrics = [TorchBackendFactory.create_metric(acc, "Accuracy")]
