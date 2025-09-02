##EfficientNet_B0_Model.py
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch

def get_efficientnet_b0(num_classes=6):
    """
    Returns EfficientNet-B0 model with modified classifier head.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_efficientnet_tuned = model_efficientnet_tuned.to(device)
    
    # freezing all base layers
    for param in model_efficientnet.classifier.parameters():
        param.requires_grad = False
    # unfreezing classification head
    for param in model_efficientnet_tuned.classifier.parameters():
        param.requires_grad = True
        
    
    return model