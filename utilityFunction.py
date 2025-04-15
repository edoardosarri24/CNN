import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

def count_layers(model:nn.Module) -> int:
    return sum(1 for layer in model.modules() if isinstance(layer, (nn.Linear, nn.Conv2d, nn.AdaptiveMaxPool2d)))

def split_vector(vec:list[int], size:int) -> list[list[int]]:
    result = [vec[i:i+size] for i in range(0, len(vec)-1, size-1)]
    if len(result[-1]) == 1:
        result[-2].append(result.pop()[0])
    return result

def get_dataloader(set, batch_size:int):
    return DataLoader(set,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4,
                      pin_memory=True)

def cloneResnet(model:nn.Module):
    clone = resnet18(weights=None)
    clone.fc = nn.Linear(512, 10)
    clone.load_state_dict(model.state_dict())
    clone.fc = nn.Linear(512,100)
    for param in clone.parameters():
        param.requires_grad = False
    for param in clone.fc.parameters():
        param.requires_grad = True
    for param in clone.layer4.parameters():
        param.requires_grad = True
    for param in clone.layer3.parameters():
        param.requires_grad = True
    return clone