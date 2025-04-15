import torch.nn as nn

class Convolutional_block(nn.Module):
    def __init__(self, dim:list[int]):
        super(Convolutional_block, self).__init__()
        modules = []
        for i in range(len(dim)-1):
            modules.append(nn.Conv2d(dim[i], dim[i+1], 3, padding=1))
            modules.append(nn.ReLU())
        self.modul = nn.Sequential(*modules)
    def forward(self, x):
        return self.modul(x)
    
class Residual_Block(nn.Module):
    def __init__(self, dim:list[int], layer_type:str):
        super(Residual_Block, self).__init__()
        self.layer = layer_type
        modules = []
        if self.layer == 'conv':
            for i in range(len(dim)-2):
                modules.append(nn.Conv2d(dim[i], dim[i+1], 3, padding=1))
                modules.append(nn.ReLU())
            modules.append(nn.Conv2d(dim[-2], dim[-1], 3, padding=1))
            self.module = nn.Sequential(*modules)
            self.identity = nn.Identity() if (dim[0] == dim[-1]) else nn.Conv2d(dim[0], dim[-1], 1)
        elif self.layer == 'linear':
            for i in range(len(dim)-2):
                modules.append(nn.Linear(dim[i], dim[i+1]))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(dim[-2], dim[-1]))
            self.module = nn.Sequential(*modules)
            self.identity = nn.Identity() if (dim[0] == dim[-1]) else nn.Linear(dim[0], dim[-1])
        else:
            print(f'ERROR: {layer_type} non permesso')
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = self.identity(x)
        x = self.module(x)
        x += identity
        return self.relu(x)