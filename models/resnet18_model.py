#ResNet18_Model.py
from torchvision.models import resnet18,ResNet18_Weights
import torch.nn as nn

def get_resnet18(num_classes = 6):
  model = resnet18(weights = ResNet18_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features,num_classes)
  return model

  