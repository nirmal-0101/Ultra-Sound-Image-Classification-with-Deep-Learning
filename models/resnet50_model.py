#resNet50_Model.py
from torchvision.models import resnet50,ResNet50_Weights
import torch.nn as nn

def get_resnet50(num_classes = 6):
  """
  Returns a pretrained Resnet50 model with a modified final
  classifier layer for evaluating it's performance in this task
  """
  # loading pretrained model 
  resnet50_model = resnet50(weights = ResNet50_Weights.DEFAULT)

  # modifying the final layer
  resnet50_model.fc  = nn.Sequential(
    nn.Linear(resnet50_model.fc.in_features,128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128,num_classes)
  )

  return resnet50_model
