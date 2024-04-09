import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import pandas as pd
import PIL
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torchvision.models as models
import time
import joblib
import sklearn

#Dataset prep fpr base model
IMAGE_SIZE = 224
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.images_folder = "/kaggle/input/kinetosis-utkf/dataset/images_utk"
        self.normalize = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        file = self.df.iloc[index]
        filename = file.imageId
        age = torch.tensor((file.age)/80)
        im = cv2.imread(os.path.join(self.images_folder, filename))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im,(224,224))
        im = torch.tensor(im).permute(2,0,1)
        im = self.normalize(im/255.)
        
        return im.float().to(device), age.to(device)

# split the dataset into training and validation dataset
print("starting...")
df = pd.read_csv("/kaggle/input/kinetosis-utkf/dataset/csv/dataset_UTK1.csv")
dataset = CustomDataset(df)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#test_loader = DataLoader(dataset, batch_size = 32)
test_loader = DataLoader(dataset)

#model 1 load
def load_model1():
    # define modelpath for this ensemble
    model1path = '/kaggle/input/model-age/model1.pth'
    # load model from model path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_model1 = torch.load(model1path, map_location=device)
    model1 = models.vgg16(pretrained=True)
    for param in model1.parameters():
        param.requires_grad = False
    # overwriting in the average pool layer
    model1.avgpool = nn.Sequential(
                nn.Conv2d(512,512, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten()
            )
        # custom classifier for age classification   
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            # intermediate layer calculation
            self.intermediate = nn.Sequential(
                nn.Linear(2048,512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128,64),
                nn.ReLU(),
            )
            # age_classifier 
            self.age_classifier = nn.Sequential(nn.Linear(64,1),
                                                nn.Sigmoid())
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            return age.to(device)
    model1.classifier = ageGenderClassifier().to(device)   
    model1 = model1.to(device)
    model1.load_state_dict(checkpoint_model1['model'])
    return model1

#model 2 load
def load_model2():
    # define modelpath for this ensemble
    model2path = '/kaggle/input/model-age/model2.pth'
    # load model from model path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_model2 = torch.load(model2path, map_location=device)
    model2 = models.resnet34(pretrained = True)
    for param in model2.parameters():
        param.requires_grad = False
    # overwriting in the average pool layer
    model2.avgpool = nn.Sequential(
                nn.Conv2d(512,512, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten()
            )
    # custom classifier for age classification   
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            # intermediate layer calculation
            self.intermediate = nn.Sequential(
                nn.Linear(2048,512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128,64),
                nn.ReLU(),
            )
            # age_classifier 
            self.age_classifier = nn.Sequential(nn.Linear(64,1),
                                                nn.Sigmoid())
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            return age.to(device)
    model2.fc = ageGenderClassifier().to(device)
    model2 = model2.to(device)
    model2.load_state_dict(checkpoint_model2['model'])
    return model2

#model 3 load    
def load_model3():
    # define modelpath for this ensemble
    model3path = '/kaggle/input/model-age/model3.pth'
    # load model from model path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_model3 = torch.load(model3path, map_location=device)
    model3 = models.efficientnet_b0(pretrained=True)
    for param in model3.parameters():
        param.requires_grad = False
    # overwriting in the average pool layer
    model3.avgpool = nn.Sequential(
                nn.Conv2d(1280,512, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten()
            )
    # custom classifier for age classification   
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()

            # intermediate layer calculation
            self.intermediate = nn.Sequential(
                nn.Linear(2048,512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128,64),
                nn.ReLU(),
            )
            # age_classifier 
            self.age_classifier = nn.Sequential(nn.Linear(64,1),
                                                nn.Sigmoid())
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            return age.to(device)
    model3.classifier = ageGenderClassifier().to(device)
    model3 = model3.to(device)
    model3.load_state_dict(checkpoint_model3['model'])
    return model3

#model 4 load
def load_model4():
    # define modelpath for this ensemble
    model4path = '/kaggle/input/model-age/model4.pth'
    # load model from model path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_model4 = torch.load(model4path, map_location=device)
    model4 = models.densenet121(pretrained = True)
    for param in model4.parameters():
        param.requires_grad = False
    # custom classifier for age classification   
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            # intermediate layer calculation
            self.intermediate = nn.Sequential(
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128,64),
                nn.ReLU(),
            )
            # age_classifier 
            self.age_classifier = nn.Sequential(nn.Linear(64,1),
                                                nn.Sigmoid())
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            return age.to(device)

    model4.classifier = ageGenderClassifier().to(device)
    model4 = model4.to(device)
    model4.load_state_dict(checkpoint_model4['model'])
    return model4

#Create the Base Model1 Predictions

# Function to get predictions and labels from a model1
def get_model1_predictions(model, dataloader):
    predictions = []
    actual_ages = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,age = data
            im = im.to(device)
            age = age.to(device)
            pred_age = model(im)
            pred_age = pred_age.to('cpu').detach().numpy()
            pred_age = pred_age.flatten()
            age = age.to('cpu').detach().numpy()
            predictions.append(pred_age)
            actual_ages.append(age)
    return predictions, actual_ages


model1_base = load_model1()
model1_predictions, model1_actual_ages = get_model1_predictions(model1_base, test_loader)
model1_predictions = [x for row in model1_predictions for x in row]
model1_actual_ages = [x for row in model1_actual_ages for x in row]


# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model1_Predictions': preds * 100, 'Model1_Actual_Age': ages * 100}
        for preds, ages in zip(model1_predictions, model1_actual_ages)]

df_model1_predictions = pd.DataFrame(data)
df_model1_predictions.to_csv('./model1_dataframe.csv')

# Display the DataFrame
print(df_model1_predictions)

#Create the Base Model2 Predictions

# Function to get predictions and labels from a model2
def get_model2_predictions(model, dataloader):
    predictions = []
    actual_ages = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,age = data
            im = im.to(device)
            age = age.to(device)
            pred_age = model(im)
            pred_age = pred_age.to('cpu').detach().numpy()
            pred_age = pred_age.flatten()
            age = age.to('cpu').detach().numpy()
            predictions.append(pred_age)
            actual_ages.append(age)
    return predictions, actual_ages


model2_base = load_model2()
model2_predictions, model2_actual_ages = get_model2_predictions(model2_base, test_loader)
model2_predictions = [x for row in model2_predictions for x in row]
model2_actual_ages = [x for row in model2_actual_ages for x in row]

# Flatten the model1_predictions list of lists
#flattened_model2_predictions = [item for sublist in model2_predictions for item in sublist]

# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model2_Predictions': preds * 100, 'Model2_Actual_Age': ages * 100}
        for preds, ages in zip(model2_predictions, model2_actual_ages)]

df_model2_predictions = pd.DataFrame(data)
df_model2_predictions.to_csv('./model2_dataframe.csv')


# Display the DataFrame
print(df_model2_predictions)

#Create the Base Model3 Predictions

# Function to get predictions and labels from a model3
def get_model3_predictions(model, dataloader):
    predictions = []
    actual_ages = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,age = data
            im = im.to(device)
            age = age.to(device)
            pred_age = model(im)
            pred_age = pred_age.to('cpu').detach().numpy()
            pred_age = pred_age.flatten()
            age = age.to('cpu').detach().numpy()
            predictions.append(pred_age)
            actual_ages.append(age)
    return predictions, actual_ages


model3_base = load_model3()
model3_predictions, model3_actual_ages = get_model3_predictions(model3_base, test_loader)
model3_predictions = [x for row in model3_predictions for x in row]
model3_actual_ages = [x for row in model3_actual_ages for x in row]

# Flatten the model1_predictions list of lists
#flattened_model3_predictions = [item for sublist in model3_predictions for item in sublist]
# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model3_Predictions': preds * 100, 'Model3_Actual_Age': ages * 100}
        for preds, ages in zip(model3_predictions, model3_actual_ages)]

df_model3_predictions = pd.DataFrame(data)
df_model3_predictions.to_csv('./model3_dataframe.csv')

# Display the DataFrame
print(df_model3_predictions)

#Create the Base Model4 Predictions

# Function to get predictions and labels from a model4
def get_model4_predictions(model, dataloader):
    predictions = []
    actual_ages = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,age = data
            im = im.to(device)
            age = age.to(device)
            pred_age = model(im)
            pred_age = pred_age.to('cpu').detach().numpy()
            pred_age = pred_age.flatten()
            age = age.to('cpu').detach().numpy()
            predictions.append(pred_age)
            actual_ages.append(age)
    return predictions, actual_ages


model4_base = load_model4()
model4_predictions, model4_actual_ages = get_model4_predictions(model4_base, test_loader)
model4_predictions = [x for row in model4_predictions for x in row]
model4_actual_ages = [x for row in model4_actual_ages for x in row]
# Flatten the model1_predictions list of lists
#flattened_model4_predictions = [item for sublist in model4_predictions for item in sublist]

# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model4_Predictions': preds * 100, 'Model4_Actual_Age': ages * 100}
        for preds, ages in zip(model4_predictions, model4_actual_ages)]

df_model4_predictions = pd.DataFrame(data)
df_model4_predictions.to_csv('./model4_dataframe.csv')

# Display the DataFrame
print(df_model4_predictions)

#dataset for metamodel-Decision tree

# Concatenate the DataFrames along columns (axis=1)
final_df = pd.concat([
    df_model1_predictions['Model1_Actual_Age'],
    df_model1_predictions['Model1_Predictions'],
    df_model2_predictions['Model2_Predictions'],
    df_model3_predictions['Model3_Predictions'],
    df_model4_predictions['Model4_Predictions']
], axis=1)

# Rename the columns
final_df.columns = ['True_Age_Label', 'Model1_Predictions', 'Model2_Predictions', 'Model3_Predictions', 'Model4_Predictions']
final_df.to_csv('./final_dataframe.csv')
# Display the final DataFrame
print(final_df)

# create meta model - pretrained decision tree and train them
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
print(sklearn.__version__)
decision_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=2)

# Assign the predictions to x and the true labels to y
x = final_df[['Model1_Predictions', 'Model2_Predictions', 'Model3_Predictions', 'Model4_Predictions']]
y = final_df['True_Age_Label']

# Split the data into training and evaluation sets (80% train, 20% eval)
x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2, random_state=42)

#print("Unique values in y_train:", y_train.unique())
#print("Unique values in y_eval:", y_eval.unique())


# Train the decision tree regression on the training data
decision_tree.fit(x_train, y_train)
# Save the trained age meta model to a file
model_path = '/kaggle/working/age_meta_model.pkl'
joblib.dump(decision_tree, model_path)
print(f"Trained age meta model saved to {model_path}")

# Evaluate the decision tree regression on the evaluation data
eval_predictions = decision_tree.predict(x_eval)
mae = mean_absolute_error(y_eval, eval_predictions)
print("MAE on Evaluation Data:", mae)

