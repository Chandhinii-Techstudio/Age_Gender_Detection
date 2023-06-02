import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import mtcnn
from mtcnn.mtcnn import MTCNN
print(mtcnn.__version__)
from torchvision import models
#from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.utils import save_img
from tensorflow.keras.utils import img_to_array

device =  'cpu'
detector = MTCNN()

def crop_to_bounding_box(img, y, x, height, width):
    return img[ y:y+height, x:x+width]

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
        # age_classifier and gender_classifier
        self.age_classifier = nn.Sequential(nn.Linear(64,1),
                                            nn.Sigmoid())
    def forward(self, x):
        x = self.intermediate(x)
        age = self.age_classifier(x)
        return age.to(device)

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

def inference():
    # Load the model
    checkpoint1 = torch.load("/home/input/bestmodelgenresnet/model_best_checkpoint_gen.pth", map_location=torch.device('cpu'))
    checkpoint2 = torch.load("/home/input/bestagemodelresnet/model_best_checkpoint_modified_batch32.pth", map_location=torch.device('cpu'))

    model1 = models.resnet34(pretrained = True)
    for param in model1.parameters():
            param.requires_grad = False
    # overwriting in the average pool layer
    model1.avgpool = nn.Sequential(
                    nn.Conv2d(512,512, kernel_size=3),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Flatten()
                )
        
    for param in model1.classifier.parameters():
            param.requires_grad = True
    # redefine the classifier, optimizer
    model1.fc = genderClassifier().to(device)
    optimizer = torch.optim.Adam(model1.parameters(),lr= 1e-4)
    # laod the saved model parameters and put the model in evaluation mode
    model1.load_state_dict(checkpoint1['model'], strict = False)
    model1.eval()

    model2 = models.resnet34(pretrained = True)
    for param2 in model2.parameters():
            param2.requires_grad = False
    # overwriting in the average pool layer
    model2.avgpool = nn.Sequential(
                    nn.Conv2d(512,512, kernel_size=3),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Flatten()
                )
        
    for param2 in model2.classifier.parameters():
            param2.requires_grad = True
    # redefine the classifier, optimizer
    model2.fc = ageGenderClassifier().to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(),lr= 1e-4)
    # laod the saved model parameters and put the model in evaluation mode
    model2.load_state_dict(checkpoint2['model'], strict = False)
    model2.eval()

    # test image 1
    img = cv2.imread("/home/input/kinet-inf/InfImage/mock.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_faces = detector.detect_faces(img)
    for face in detected_faces:
        x , y  = max(face['box'][0],0),max(face['box'][1],0)
        width, height = min(face['box'][2],img.shape[1]-x),min(face['box'][3],img.shape[0]-y)
        # crop the face
        face_img = tf.image.crop_to_bounding_box(img, y, x,height,width)

        # resize image
        face_img = tf.image.resize(face_img, (200,200), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        face_img = tf.dtypes.cast(face_img, tf.int32)
        # save image in numpy array
        img_array = img_to_array(face_img)
        imagePath = '/home/working/mock_final.jpg'
        save_img(imagePath , img_array)
                

    im = cv2.imread("/home/working/mock_final.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im =  np.array(im)
    im = cv2.resize(im,(224,224)) / 255
    im = torch.tensor(im).permute(2,0,1).to(device)
    age= model2(im[None].float())
    pred_age = age.to('cpu').detach().numpy()
    pred_age = int(pred_age[0][0]* 79)

    gender = model1(im[None].float())
    pred_gender = gender.to('cpu').detach().numpy()
    gen = np.where(pred_gender[0][0]>0.5, 'Female','Male')

    print('predicted gender:',np.where(pred_gender[0][0]>0.5, \
                                    'Female','Male'))

    return gen, pred_age