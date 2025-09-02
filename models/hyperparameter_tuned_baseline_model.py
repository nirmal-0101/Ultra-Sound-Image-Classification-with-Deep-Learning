#hyperparameter_tuned_baseline_model.py
# hyperparameter_tuned_baseline_model.py
import torch.nn as nn

class HyperParameterBaselineCNN(nn.Module):
    """
    Hyperparameter tuned Baseline Model 
    """
    def __init__(self, num_classes, kernel_size=3, dropout_rate=0.5):
        super(HyperParameterBaselineCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64 * 14 * 14, 128),  # Adjust based on input image size
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
