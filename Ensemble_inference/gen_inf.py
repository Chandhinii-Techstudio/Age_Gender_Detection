import numpy as np
import torch
import torch.nn as nn
import os
import cv2
from torchvision import models
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.utils import save_img
from tensorflow.keras.utils import img_to_array
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from torchvision import transforms
    

device =  'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN()

def crop_to_bounding_box(img, y, x, height, width):
    return img[ y:y+height, x:x+width]

def gen_inference():    
    #model 1 load
    def load_model1():
        # define modelpath for this ensemble
        model1path = '/app/Kinetosis/gender/model1.pth'
        # load model from model path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_model1 = torch.load(model1path, map_location=device)
        model1 = models.vgg16(pretrained=False)
        # overwriting in the average pool layer
        model1.avgpool = nn.Sequential(
                    nn.Conv2d(512,512, kernel_size=3),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Flatten()
                )
            # custom classifier for gender classification   
        class genderClassifier1(nn.Module):
            def __init__(self):
                super(genderClassifier1, self).__init__()

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
        model1.classifier = genderClassifier1().to(device)   
        model1 = model1.to(device)
        model1.load_state_dict(checkpoint_model1['model'])
        return model1

    #model 2 load
    def load_model2():
        # define modelpath for this ensemble
        model2path = '/app/Kinetosis/gender/model2.pth'
        # load model from model path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_model2 = torch.load(model2path, map_location=device)
        model2 = models.resnet34(pretrained = False)
        # overwriting in the average pool layer
        model2.avgpool = nn.Sequential(
                    nn.Conv2d(512,512, kernel_size=3),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Flatten()
                )
        # custom classifier for gender classification   
        class genderClassifier2(nn.Module):
            def __init__(self):
                super(genderClassifier2, self).__init__()

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
        model2.fc = genderClassifier2().to(device)
        model2 = model2.to(device)
        model2.load_state_dict(checkpoint_model2['model'])
        return model2
        
    #model 3 load    
    def load_model3():
        # define modelpath for this ensemble
        model3path = '/app/Kinetosis/gender/model3.pth'
        # load model from model path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_model3 = torch.load(model3path, map_location=device)
        model3 = models.efficientnet_b0(pretrained=False)
        # overwriting in the average pool layer
        model3.avgpool = nn.Sequential(
                    nn.Conv2d(1280,512, kernel_size=3),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Flatten()
                )
        # custom classifier for gender classification   
        class genderClassifier3(nn.Module):
            def __init__(self):
                super(genderClassifier3, self).__init__()

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
        model3.classifier = genderClassifier3().to(device)
        model3 = model3.to(device)
        model3.load_state_dict(checkpoint_model3['model'])
        return model3        

    #model 4 load
    def load_model4():
        # define modelpath for this ensemble
        model4path = '/app/Kinetosis/gender/model4.pth'
        # load model from model path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_model4 = torch.load(model4path, map_location=device)
        model4 = models.densenet121(pretrained = False)
    # custom classifier for gender classification   
        class genderClassifier4(nn.Module):
            def __init__(self):
                super(genderClassifier4, self).__init__()

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

        model4.classifier = genderClassifier4().to(device)
        model4 = model4.to(device)
        model4.load_state_dict(checkpoint_model4['model'])
        return model4    


    # test image 1
    img = cv2.imread("/app/Kinetosis/InfImage/mock.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_faces = detector.detect_faces(img)
    for face in detected_faces:
        x , y  = max(face['box'][0],0),max(face['box'][1],0)
        width, height = min(face['box'][2],img.shape[1]-x),min(face['box'][3],img.shape[0]-y)
        # crop the face
        face_img = tf.image.crop_to_bounding_box(img, y, x,height,width)

        # resize image
        face_img = tf.image.resize(face_img, (120,120), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        face_img = tf.dtypes.cast(face_img, tf.int32)
        # save image in numpy array
        img_array = img_to_array(face_img)
        imagePath = '/app/Kinetosis/InfImage/mock_cropped1.jpg'
        save_img(imagePath , img_array)

    # Define mean and std deviation values for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_age = cv2.imread("/app/Kinetosis/InfImage/mock_cropped1.jpg")
    im_age = cv2.cvtColor(im_age, cv2.COLOR_BGR2RGB)
    #im_age =  np.array(im_age)
    im_age = cv2.resize(im_age,(224,224))
    im_age = torch.tensor(im_age).permute(2,0,1)
    # Normalize the image using the specified mean and std deviation
    normalize = transforms.Normalize(mean=mean, std=std)
    im_age = normalize(im_age/255.)
    im_age = im_age.float().to(device)
    #im_age = im_age[None].float()
    
    #Create the Base Model1 Predictions

    # Function to get predictions and labels from a model1
    def get_model1_predictions(model, im):
        model.eval()
        
        with torch.no_grad():
            im = im.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
        return pred_gen


    model1_base = load_model1()
    model1_predictions = get_model1_predictions(model1_base, im_age)


    # Convert the numpy arrays to pandas DataFrame
    # making each numpy array data to be represent as a row in the DataFrame
    data1 = [{'Model1_Predictions': 1 if model1_predictions > 0.5 else 0}]

    df_model1_predictions = pd.DataFrame(data1)
    #df_model1_predictions.to_csv('./model1_dataframe.csv')

    # Display the DataFrame
    print(df_model1_predictions)   


    #Create the Base Model2 Predictions

    # Function to get predictions and labels from a model2
    def get_model2_predictions(model, im):
        model.eval()
        
        with torch.no_grad():
            im = im.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
        return pred_gen


    model2_base = load_model2()
    model2_predictions = get_model2_predictions(model2_base, im_age)



    # Convert the numpy arrays to pandas DataFrame
    # making each numpy array data to be represent as a row in the DataFrame
    data2 = [{'Model2_Predictions':  1 if model2_predictions > 0.5 else 0}]

    df_model2_predictions = pd.DataFrame(data2)
    #df_model2_predictions.to_csv('./model2_dataframe.csv')

    # Display the DataFrame
    print(df_model2_predictions) 


    #Create the Base Model3 Predictions

    # Function to get predictions and labels from a model3
    def get_model3_predictions(model, im):
        model.eval()
        
        with torch.no_grad():
            im = im.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
        return pred_gen


    model3_base = load_model3()
    model3_predictions = get_model3_predictions(model3_base, im_age)


    # Convert the numpy arrays to pandas DataFrame
    # making each numpy array data to be represent as a row in the DataFrame
    data3 = [{'Model3_Predictions': 1 if model3_predictions > 0.5 else 0}]

    df_model3_predictions = pd.DataFrame(data3)
    #df_model3_predictions.to_csv('./model3_dataframe.csv')

    # Display the DataFrame
    print(df_model3_predictions)


    #Create the Base Model4 Predictions

    # Function to get predictions and labels from a model4
    def get_model4_predictions(model, im):
        model.eval()
        
        with torch.no_grad():
            im = im.to(device)
            pred_gen  = model(im)
            pred_gen = pred_gen.to('cpu').detach().numpy()
            pred_gen = pred_gen.flatten()
        return pred_gen


    model4_base = load_model4()
    model4_predictions = get_model4_predictions(model4_base, im_age)



    # Convert the numpy arrays to pandas DataFrame
    # making each numpy array data to be represent as a row in the DataFrame
    data4 = [{'Model4_Predictions': 1 if preds > 0.5 else 0}]

    df_model4_predictions = pd.DataFrame(data4)
    #df_model4_predictions.to_csv('./model4_dataframe.csv')

    # Display the DataFrame
    print(df_model4_predictions)

    #dataset for metamodel-Decision tree

    # Concatenate the DataFrames along columns (axis=1)
    final_df = pd.concat([
        df_model1_predictions['Model1_Predictions'],
        df_model2_predictions['Model2_Predictions'],
        df_model3_predictions['Model3_Predictions'],
        df_model4_predictions['Model4_Predictions']
    ], axis=1)

    # Rename the columns
    final_df.columns = ['Model1_Predictions', 'Model2_Predictions', 'Model3_Predictions', 'Model4_Predictions']
    #final_df.to_csv('./final_dataframe.csv')
    # Display the final DataFrame
    print(final_df)

    # Assign the predictions to x and the true labels to y
    x = final_df[['Model1_Predictions', 'Model2_Predictions', 'Model3_Predictions', 'Model4_Predictions']]
    

    # load model
    loaded_model = joblib.load("/app/Kinetosis/metamodel/gender_meta_model.pkl")

    # Evaluate the decision tree on the evaluation data
    gender_predictions = loaded_model.predict(x)
    gen_pred = np.where(gender_predictions > 0.5, 'Female', 'Male')
    print('Predicted gender', gen_pred)

    return  gen_pred