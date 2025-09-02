#evaluate.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from data.data_loader import get_dataloaders
from models.EfficientNet_B0_Model import get_efficientnet_b0 
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
model_path = "/content/drive/MyDrive/Deep_Learning_UltrasoundImageClassification_Final_Project/models/Model_Efficientnet_finetuned.pth"
images_path = "/content/drive/MyDrive/Deep_Learning_UltrasoundImageClassification_Final_Project/Images"

# loading data using dataloader
loader_train, loader_valid, loader_test = get_dataloaders(
    train_data, valid_data, test_data,
    images_path,
    transform_train=transform_valid,
    transform_valid=transform_valid,
    batch_size=64,
    num_workers=2
)

#loading the model
num_classes = 6
model_efficientnet_tuned = get_efficientnet_b0(num_classes=6)
model_efficientnet.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_efficientnet.classifier[1].in_features,num_classes)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_efficientnet_tuned = model_efficientnet_tuned.to(device)

# setting the model to evaluation 
model_efficientnet_tuned.eval()

y_correct_efficientnet_tuned = []
y_predicted_efficientnet_tuned= []

with torch.no_grad():
  for input_images,labels in loader_test:
    input_images = input_images.to(device)
    labels = labels.to(device)
    outputs = model_efficientnet_tuned(input_images)
    _,predicted = torch.max(outputs,1)
    y_correct_efficientnet_tuned.extend(labels.cpu().numpy())
    y_predicted_efficientnet_tuned.extend(predicted.cpu().numpy())
    
# getting the name of classes
class_names = list(class_label_map.keys())

# displaying confusion matrix
cm = confusion_matrix(y_correct_efficientnet_tuned,y_predicted_efficientnet_tuned)
confusion_matrix_output = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
confusion_matrix_output.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.title("Confusion Matrix of Tuned EfficientNet b0 model  on Test set")
plt.savefig("/content/drive/MyDrive/Deep_Learning_UltrasoundImageClassification_Final_Project/results/Tuned EfficientNet b0_Model_Confusion_Matrix.png")
plt.show()

# classification report
tuned_efficientnet_results = classification_report(y_correct_efficientnet_tuned,y_predicted_efficientnet_tuned,target_names=class_names)

with open("/content/drive/MyDrive/Deep_Learning_UltrasoundImageClassification_Final_Project/results/Tuned EfficientNet b0_Model_Classification_Report.txt","w") as file:
  file.write(tuned_efficientnet_results)
  print(tuned_efficientnet_results)




