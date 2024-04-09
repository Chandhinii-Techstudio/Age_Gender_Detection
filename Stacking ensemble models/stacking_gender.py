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

IMAGE_SIZE = 224
class CustomDatasetGen(torch.utils.data.Dataset):
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
        gender = torch.tensor(file.gender).view(-1)
        im = cv2.imread(os.path.join(self.images_folder, filename))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im,(224,224))
        im = torch.tensor(im).permute(2,0,1)
        im = self.normalize(im/255.)
        
        return im.float().to(device), gender.float().to(device)

# split the dataset into training and validation dataset
print("starting...")
df = pd.read_csv("/kaggle/input/kinetosis-utkf/dataset/csv/dataset_UTK1.csv")
dataset = CustomDatasetGen(df)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#test_loader = DataLoader(dataset, batch_size = 32)
test_loader = DataLoader(dataset)

#model 1 load
def load_model1():
    # define modelpath for this ensemble
    model1path = '/kaggle/input/model-gen/model1.pth'
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
        # custom classifier for gender classification   
    class genderClassifier(nn.Module):
        def __init__(self):
            super(genderClassifier, self).__init__()

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
            # gender_classifier
            self.gender_classifier = nn.Sequential(
                                                nn.Linear(64,1),
                                                nn.Sigmoid())

        def forward(self, x):
            x = self.intermediate(x)
            gender = self.gender_classifier(x)
            return gender.to(device)
    model1.classifier = genderClassifier().to(device)   
    model1 = model1.to(device)
    model1.load_state_dict(checkpoint_model1['model'])
    return model1

#model 2 load
def load_model2():
    # define modelpath for this ensemble
    model2path = '/kaggle/input/model-gen/model2.pth'
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
    # custom classifier for gender classification   
    class genderClassifier(nn.Module):
        def __init__(self):
            super(genderClassifier, self).__init__()

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
            # gender_classifier
            self.gender_classifier = nn.Sequential(
                                                nn.Linear(64,1),
                                                nn.Sigmoid())

        def forward(self, x):
            x = self.intermediate(x)
            gender = self.gender_classifier(x)
            return gender.to(device)
    model2.fc = genderClassifier().to(device)
    model2 = model2.to(device)
    model2.load_state_dict(checkpoint_model2['model'])
    return model2

#model 3 load    
def load_model3():
    # define modelpath for this ensemble
    model3path = '/kaggle/input/model-gen/model3.pth'
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
    # custom classifier for gender classification   
    class genderClassifier(nn.Module):
        def __init__(self):
            super(genderClassifier, self).__init__()

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
            # gender_classifier 
            self.gender_classifier = nn.Sequential(nn.Linear(64,1),
                                                nn.Sigmoid())
        def forward(self, x):
            x = self.intermediate(x)
            gender = self.gender_classifier(x)
            return gender.to(device)
    model3.classifier = genderClassifier().to(device)
    model3 = model3.to(device)
    model3.load_state_dict(checkpoint_model3['model'])
    return model3

#model 4 load
def load_model4():
    # define modelpath for this ensemble
    model4path = '/kaggle/input/model-gen/model4.pth'
    # load model from model path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_model4 = torch.load(model4path, map_location=device)
    model4 = models.densenet121(pretrained = True)
    for param in model4.parameters():
        param.requires_grad = False
   # custom classifier for gender classification   
    class genderClassifier(nn.Module):
        def __init__(self):
            super(genderClassifier, self).__init__()

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
            # gender_classifier
            self.gender_classifier = nn.Sequential(
                                                nn.Linear(64,1),
                                                nn.Sigmoid())
        def forward(self, x):
                x = self.intermediate(x)
                gender = self.gender_classifier(x)
                return gender.to(device)

    model4.classifier = genderClassifier().to(device)
    model4 = model4.to(device)
    model4.load_state_dict(checkpoint_model4['model'])
    return model4

#Create the Base Model1 Predictions

# Function to get predictions and labels from a model1
def get_model1_predictions(model, dataloader):
    predictions = []
    actual_gen = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,gen= data
            im = im.to(device)
            gen = gen.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
            gen = gen.to('cpu').detach().numpy()
            predictions.append(pred_gen)
            actual_gen.append(gen)
    return predictions, actual_gen


model1_base = load_model1()
model1_predictions, model1_actual_gen = get_model1_predictions(model1_base, test_loader)
model1_predictions = [x for row in model1_predictions for x in row]
model1_actual_gen = [x for row in model1_actual_gen for x in row]


# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model1_Predictions': 1 if preds > 0.6 else 0, 'Model1_Actual_Gender': int(gen[0])}
        for preds, gen in zip(model1_predictions, model1_actual_gen)]

df_model1_predictions = pd.DataFrame(data)
df_model1_predictions.to_csv('./model1_dataframe.csv')

# Display the DataFrame
print(df_model1_predictions)

#Create the Base Model2 Predictions

# Function to get predictions and labels from a model2
def get_model2_predictions(model, dataloader):
    predictions = []
    actual_gen = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,gen= data
            im = im.to(device)
            gen = gen.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
            gen = gen.to('cpu').detach().numpy()
            predictions.append(pred_gen)
            actual_gen.append(gen)
    return predictions, actual_gen


model2_base = load_model2()
model2_predictions, model2_actual_gen = get_model2_predictions(model2_base, test_loader)
model2_predictions = [x for row in model2_predictions for x in row]
model2_actual_gen = [x for row in model2_actual_gen for x in row]


# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model2_Predictions':  1 if preds > 0.6 else 0, 'Model2_Actual_Gender': int(gen[0])}
        for preds, gen in zip(model2_predictions, model2_actual_gen)]

df_model2_predictions = pd.DataFrame(data)
df_model2_predictions.to_csv('./model2_dataframe.csv')

# Display the DataFrame
print(df_model2_predictions)

#Create the Base Model3 Predictions

# Function to get predictions and labels from a model3
def get_model3_predictions(model, dataloader):
    predictions = []
    actual_gen = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,gen= data
            im = im.to(device)
            gen = gen.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
            gen = gen.to('cpu').detach().numpy()
            predictions.append(pred_gen)
            actual_gen.append(gen)
    return predictions, actual_gen


model3_base = load_model3()
model3_predictions, model3_actual_gen = get_model3_predictions(model3_base, test_loader)
model3_predictions = [x for row in model3_predictions for x in row]
model3_actual_gen = [x for row in model3_actual_gen for x in row]


# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model3_Predictions': 1 if preds > 0.6 else 0, 'Model3_Actual_Gender': int(gen[0])}
        for preds, gen in zip(model3_predictions, model3_actual_gen)]

df_model3_predictions = pd.DataFrame(data)
df_model3_predictions.to_csv('./model3_dataframe.csv')

# Display the DataFrame
print(df_model3_predictions)

#Create the Base Model4 Predictions

# Function to get predictions and labels from a model4
def get_model4_predictions(model, dataloader):
    predictions = []
    actual_gen = []
    model.eval()
    
    with torch.no_grad():
        for ix, data in enumerate(dataloader):
            im,gen= data
            im = im.to(device)
            gen = gen.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
            gen = gen.to('cpu').detach().numpy()
            predictions.append(pred_gen)
            actual_gen.append(gen)
    return predictions, actual_gen


model4_base = load_model4()
model4_predictions, model4_actual_gen = get_model4_predictions(model4_base, test_loader)
model4_predictions = [x for row in model4_predictions for x in row]
model4_actual_gen = [x for row in model4_actual_gen for x in row]


# Convert the numpy arrays to pandas DataFrame
# making each numpy array data to be represent as a row in the DataFrame
data = [{'Model4_Predictions': 1 if preds > 0.6 else 0, 'Model4_Actual_Gender': int(gen[0])}
        for preds, gen in zip(model4_predictions, model4_actual_gen)]

df_model4_predictions = pd.DataFrame(data)
df_model4_predictions.to_csv('./model4_dataframe.csv')

# Display the DataFrame
print(df_model4_predictions)

#dataset for metamodel-Decision tree

# Concatenate the DataFrames along columns (axis=1)
final_df = pd.concat([
    df_model1_predictions['Model1_Actual_Gender'],
    df_model1_predictions['Model1_Predictions'],
    df_model2_predictions['Model2_Predictions'],
    df_model3_predictions['Model3_Predictions'],
    df_model4_predictions['Model4_Predictions']
], axis=1)

# Rename the columns
final_df.columns = ['True_Gender_Label', 'Model1_Predictions', 'Model2_Predictions', 'Model3_Predictions', 'Model4_Predictions']
final_df.to_csv('./final_dataframe.csv')
# Display the final DataFrame
print(final_df)

# create meta model - pretrained decision tree and train them
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
print(sklearn.__version__)
decision_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)

# Assign the predictions to x and the true labels to y
x = final_df[['Model1_Predictions', 'Model2_Predictions', 'Model3_Predictions', 'Model4_Predictions']]
y = final_df['True_Gender_Label']

# Split the data into training and evaluation sets (80% train, 20% eval)
x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.4, random_state=42)

#print("Unique values in y_train:", y_train.unique())
#print("Unique values in y_eval:", y_eval.unique())


# Train the decision tree regression on the training data
decision_tree.fit(x_train, y_train)
# Save the trained age meta model to a file
model_path = '/kaggle/working/gender_meta_model.pkl'
joblib.dump(decision_tree, model_path)
print(f"Trained age meta model saved to {model_path}")

# Evaluate the decision tree regression on the evaluation data
eval_predictions = decision_tree.predict(x_eval)
accuracy = accuracy_score(y_eval, eval_predictions)
print("Accuracy on Evaluation Data:", accuracy)