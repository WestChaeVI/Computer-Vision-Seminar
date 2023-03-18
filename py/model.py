import torch
from torchvision import models # model 라이브러리
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu' # device 배정
torch.manual_seed(42)
if device == 'cuda':
  torch.cuda.manual_seed_all(42)

def ResNet50():

    resnet_50 = models.resnet50(pretrained = True).to(device)
    # fine tuning
    resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 3).to(device)

    model = resnet_50.to(device)

    return model
