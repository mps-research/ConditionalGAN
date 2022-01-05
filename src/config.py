import torch.nn as nn
from ray import tune


large_netG_normalized = {
    'normalize': True,
    'blocks': [
        {'in_features': 110, 'out_features': 256, 'activation_func': nn.ReLU()},
        {'in_features': 256, 'out_features': 512, 'activation_func': nn.ReLU()},
        {'in_features': 512, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()},
    ]
}


large_netD = {
    'normalize': False,
    'p': 0.5,
    'blocks': [
        {'in_features': 794, 'out_features': 512, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 512, 'out_features': 256, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 256, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 64, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 64, 'out_features': 1, 'activation_func': nn.Sigmoid()},
    ]
}


netGs = {
    'large_netG_normalized': large_netG_normalized,
}

netDs = {
    'large_netD': large_netD
}


config = {
    'netG': tune.grid_search(list(netGs.keys())),
    'netD': tune.grid_search(list(netDs.keys())),
    'lrG': tune.grid_search([4e-3, 4e-4]),
    'lrD': tune.grid_search([4e-3, 4e-4]),
    'batch_size': 128,
    'n_epochs': 200,
}