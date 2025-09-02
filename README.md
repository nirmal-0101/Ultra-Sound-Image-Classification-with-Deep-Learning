# Deep Learning-Based Classification of Maternal-Fetal
  Ultrasound Planes using Convolutional Neural
  Networks and Transfer Learning

The aim of this project is to classify maternal-fetal ultrasound images into 6 anatomical planes such as Fetal Brain, Fetal Abdomen, Fetal Thorax, Fetal Femur, MATERNAL CERVIX and OTHER. 

DATASET:
The dataset was used from publicly available source. 
Dataset source: https://zenodo.org/records/3904280
The dataset consists of an Image folder containing 12,400 images. 
It also contains .csv data file and .xlsx file with images labeled into 6 categories as mentioned before.

## DATA PREPROCESSING:
- The csv data file was loaded and have undergone some preprocessing techniques including removing white spaces using strip() method. 
- The csv data had no missing values present. 
- The 'Train' column in the csv file mentioned the patient's data taken for training and testing. (Train ==1 : Train data, Train == 0 : Test data)

- Preprocessing methods such as resizing, normalization was done for the images before feeding it to the model.

## SPLITTING OF DATA:

- The dataset was splitted for training, validation and testing. 
- Firstly, the dataset was splitted into train and test in which 80% were for training and the 20% for testing. The train data was again splitted to train and validation. So that 20% test data was untouched until the final evaluation process.

## MODELS USED

1. BaselineCNN : 
A simple baseline CNN model built from scratch

2. HyperparameterTunedCNN: 
   Tuned the Baseline model with the best hyperparameters found using gridsearch

3. ResNet18: 
   Pretrained model used for transfer learning

4. Resnet50:
   
   Pretrained resnet model with more deeper layers 

5. EfficientNet-b0 : 

   This lightweight model is used as a part of transfer learning.
   
6. Tuned EfficientNet-b0 model:
    EfficientNet-b0 model was tuned to enhance the performance.


PROJECT FOLDER STRUCTURE:
1. Data: 
   - dataset.py
   - data_loader.py

2. models: 
   - baseline_model.py
   - hyperparameter_tuned_baseline_model.py
   - ResNet18_Model.py
   - ResNet50_Model.py
   -EfficientNet_B0_Model.py

3. figs:
   - Sample Training images
   - Baseline model loss and accuracy curve
   - Baseline model architecture
   - Hyperparameter tuned model loss and accuracy curve
   - ResNet18 model loss and accuracy curve
   - Resnet50 model loss and accuracy curve
   - Efficientnet model loss and accuracy curve

4. logs:
   - baselinemodel train log
   - efficientnet train log
   - resnet18 train log

5. results:
   - baseline model classification report
   - baseline model confusion matrix
   - baseline model test image prediction
   - efficientnet b0 model classification report
   - efficientnet b0 model confusion matrix
   - efficientnet b0 model predictions on test images
   - Hyperparameter model predictions on test image
   - hyperparameter model confusion matrix
   - hyperparameter model classification report
   - resnet18 and resnet50 model's classification report, confusion matrix and predictions on test image 
   


## Main files:


- `train.py` – script for training selected models
- `eval.py` – script to evaluate the models
- `models/` – saved model
- `data/` – contains custom dataset class and dataloader
- `figs/` – training plots (loss curve,accuracy curve etc)and sample training images
- `results/` – classification report, confusion matrix and predictions made by the model
- `requirements.txt` – Python package requirements

  
   

   
    




 

