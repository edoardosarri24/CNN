import torch.nn as nn
from utilityFunction import split_vector
from modelBlock import *

class MLP(nn.Module):
    def __init__(self, layers:list[int], res:bool=False):
        super(MLP, self).__init__()
        modules = []
        if (res):
            layers = split_vector(layers, 3)
            for i in range(len(layers)-1):
                modules.append(Residual_Block(layers[i], 'linear'))
            if len(layers[-1])==2:
                modules.append(nn.Linear(layers[-1][0], layers[-1][1]))
            else:
                modules.append(nn.Sequential(nn.Linear(layers[-1][0], layers[-1][1]),
                                        nn.ReLU(),
                                        nn.Linear(layers[-1][1], layers[-1][2])))
        else:
            layers = split_vector(layers, 2)
            for i in range(len(layers)-1):
                modules.append(nn.Linear(layers[i][0], layers[i][1]))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(layers[-1][0],layers[-1][1]))
        self.modul = nn.Sequential(nn.Flatten(), *modules)
    def forward(self, x):
        return self.modul(x)

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.modul = nn.Sequential(
            Convolutional_block([3, 8, 16]),
            nn.AdaptiveMaxPool2d(32),
            Convolutional_block([16, 32, 64]),
            nn.AdaptiveMaxPool2d(4),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            MLP([64*4*4, 256, 64, 10])
        )
    def forward(self, x):
        return self.modul(x)

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.modul = nn.Sequential(
            Convolutional_block([3, 8, 16]),
            nn.AdaptiveMaxPool2d(64),
            Convolutional_block([16, 32, 64]),
            nn.AdaptiveMaxPool2d(16),
            Convolutional_block([64, 64, 128, 256]),
            nn.AdaptiveMaxPool2d(4),
            MLP([256*4*4, 1024, 128, 10])
        )
    def forward(self, x):
        return self.modul(x)
    
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.modul = nn.Sequential(
            Convolutional_block([3, 8, 8, 8]),
            nn.AdaptiveMaxPool2d(64),
            Convolutional_block([8, 8, 16, 16, 16, 16, 32]),
            nn.AdaptiveMaxPool2d(32),
            Convolutional_block([32, 32, 32, 32, 32, 32, 64]),
            nn.AdaptiveMaxPool2d(16),
            Convolutional_block([64, 64, 64]),
            nn.AdaptiveMaxPool2d(8),
            Convolutional_block([64, 64, 128, 128]),
            nn.AdaptiveMaxPool2d(4),
            Convolutional_block([128, 256, 256]),
            MLP([256*4*4, 2048, 256, 10])
        )
    def forward(self, x):
        return self.modul(x)
    
class CNN2_res(nn.Module):
    def __init__(self):
        super(CNN2_res, self).__init__()
        self.modul = nn.Sequential(
            Residual_Block([3, 8, 16], layer_type='conv'),
            nn.AdaptiveMaxPool2d(64),
            Residual_Block([16, 32, 64], layer_type='conv'),
            nn.AdaptiveMaxPool2d(16),
            Residual_Block([64, 64, 128, 256], layer_type='conv'),
            nn.AdaptiveMaxPool2d(4),
            MLP([256*4*4, 1024, 128, 10])
        )
    def forward(self, x):
        return self.modul(x)
    
class CNN3_res(nn.Module):
    def __init__(self):
        super(CNN3_res, self).__init__()
        self.modul = nn.Sequential(
            Residual_Block([3, 8, 8, 8], layer_type='conv'),
            nn.AdaptiveMaxPool2d(64),
            Residual_Block([8, 8, 16, 16, 16, 16, 32], layer_type='conv'),
            nn.AdaptiveMaxPool2d(32),
            Residual_Block([32, 32, 32, 32, 32, 32, 64], layer_type='conv'),
            nn.AdaptiveMaxPool2d(16),
            Residual_Block([64, 64, 64], layer_type='conv'),
            nn.AdaptiveMaxPool2d(8),
            Residual_Block([64, 64, 128, 128], layer_type='conv'),
            nn.AdaptiveMaxPool2d(4),
            Residual_Block([128, 256, 256], layer_type='conv'),
            MLP([256*4*4, 2048, 256, 10])
        )
    def forward(self, x):
        return self.modul(x)
    
class CNN_deep_res(nn.Module):
    def __init__(self):
        super(CNN_deep_res, self).__init__()
        self.modul = nn.Sequential(
            Residual_Block([3, 8, 8, 8], layer_type='conv'),
            nn.AdaptiveMaxPool2d(64),
            Residual_Block([8, 16, 16, 16, 16], layer_type='conv'),
            Residual_Block([16, 16, 16, 16, 32, 32], layer_type='conv'),
            Residual_Block([32, 32, 32], layer_type='conv'),
            Residual_Block([32, 32, 32 , 32], layer_type='conv'),
            nn.AdaptiveMaxPool2d(32),
            Residual_Block([32, 32, 32], layer_type='conv'),
            Residual_Block([32, 32, 32, 32, 32, 32, 32], layer_type='conv'),
            nn.AdaptiveMaxPool2d(16),
            Residual_Block([32, 32, 32], layer_type='conv'),
            Residual_Block([32, 32, 32, 64, 64, 64], layer_type='conv'),
            Residual_Block([64, 64, 64], layer_type='conv'),
            nn.AdaptiveMaxPool2d(8),
            Residual_Block([64, 64, 64], layer_type='conv'),
            Residual_Block([64, 64, 128, 128], layer_type='conv'),
            nn.AdaptiveMaxPool2d(4),
            Residual_Block([128, 256], layer_type='conv'),
            Residual_Block([256, 256, 256], layer_type='conv'),
            MLP([256*4*4, 2048, 256, 10])
        )
    def forward(self, x):
        return self.modul(x)