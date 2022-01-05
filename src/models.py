import torch
import torch.nn as nn


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_func, normalize):
        super(FCBlock, self).__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if normalize else None
        self.af = activation_func

    def forward(self, x):
        y = self.fc(x)
        if self.bn:
            y = self.bn(y)
        return self.af(y)


class Generator(nn.Module):
    def __init__(self, blocks, normalize):
        super(Generator, self).__init__()
        self.net = nn.Sequential(*[
            FCBlock(b['in_features'], b['out_features'], b['activation_func'], normalize)
            for b in blocks
        ])

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        z = self.net(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, blocks, normalize, p):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(p)
        self.net = nn.Sequential(*[
            FCBlock(b['in_features'], b['out_features'], b['activation_func'], normalize)
            for b in blocks
        ])

    def forward(self, x, y):
        x_ = self.dropout(x)
        z = torch.cat([x_, y], dim=1)
        z = self.net(z)
        return z
