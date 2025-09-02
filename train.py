#train.py
import torch
import torch.nn as nn
import torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

best_valid_accuracy = 0.0
for epoch in range(num_epochs):
  # setting the  model for training using train()
  hyperparameter_tuned_model.train()
  current_loss = 0.0
  correct_training = 0.0
  total_training = 0.0
  print(f"\nEpoch:{epoch+1}/{num_epochs}")
  for images,labels in tqdm(loader_train,desc= "Training",leave= False ):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = hyperparameter_tuned_model(images)
    loss = loss_criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    current_loss += loss.item()

    _,predicted = torch.max(outputs,1)
    total_training += labels.size(0)
    correct_training += (predicted == labels).sum().item()

  tuned_train_average_loss = current_loss/len(loader_train)
  tuned_train_accuracy = 100*correct_training/total_training
  hyperparameter_tuned_train_loss.append(tuned_train_average_loss)
  hyperparameter_tuned_training_accuracies.append(tuned_train_accuracy)

  # tracking validatioin

  # setting the  model to evaluation phase using eval()
  hyperparameter_tuned_model.eval()
  validation_loss = 0.0
  correct_validation = 0.0
  total_validation = 0.0

  with torch.no_grad():
    for valid_images,valid_labels in tqdm(loader_valid,desc= "Validating",leave=False):
      valid_images = valid_images.to(device)
      valid_labels = valid_labels.to(device)
      valid_outputs = hyperparameter_tuned_model(valid_images)
      loss = loss_criterion(valid_outputs,valid_labels)
      validation_loss += loss.item()

      _,valid_predicted = torch.max(valid_outputs,1)
      total_validation += valid_labels.size(0)
      correct_validation += (valid_predicted == valid_labels).sum().item()

  # calculating the validation loss and accuracy
  tuned_valid_average_loss = validation_loss/len(loader_valid)
  tuned_valid_accuracy = 100* correct_validation/total_validation
  hyperparameter_tuned_valid_loss.append(tuned_valid_average_loss)
  hyperparameter_tuned_validation_accuracies.append(tuned_valid_accuracy)

  # saving the best model

  if tuned_valid_accuracy > best_valid_accuracy:
    best_valid_accuracy = tuned_valid_accuracy
    torch.save(hyperparameter_tuned_model.state_dict(),"/content/drive/MyDrive/Deep_Learning_UltrasoundImageClassification_Final_Project/models/hyperparameter_tuned_model.pth")
    print("Hyperparameter tuned Baseline Model saved")

  # printing loss and accuracy of training and validation phase
  print(f"Training Loss: {tuned_train_average_loss:.4f} | Training Accuracy: {tuned_train_accuracy:.2f}% |"
        f"Validation Loss:{tuned_valid_average_loss:.4f} | Validation Accuracy: {tuned_valid_accuracy:.2f}%")


  # saving train log
  tuned_model_log_data_path = "/content/drive/MyDrive/Deep_Learning_UltrasoundImageClassification_Final_Project/logs/hyperparameter_tuned_train.log"
  with open(tuned_model_log_data_path,"a")as log_file:
    log_file.write(
        f"Epoch: {epoch+1}/{num_epochs} - "
        f"Training Loss: {tuned_train_average_loss:.4f} | Training Accuracy: {tuned_train_accuracy:.2f}% |"
        f"Validation Loss:{tuned_valid_average_loss:.4f} | Validation Accuracy: {tuned_valid_accuracy:.2f}%\n"
    )




