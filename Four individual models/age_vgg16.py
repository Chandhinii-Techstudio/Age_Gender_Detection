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


IMAGE_SIZE = 224
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.images_folder = "/home/chandhinii/kinetosis/dataset/images_utk/"
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
df = pd.read_csv("/home/chandhinii/kinetosis/dataset/csv/dataset_UTK1.csv")
splits = np.random.randn(len(df)) < 0.8
train_set = df[splits]
val_set = df[~splits]

train_dataset = CustomDataset(train_set)
val_dataset = CustomDataset(val_set)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(val_dataset, batch_size=32)
a,b, = next(iter(train_loader))

# get VGG16 pretrained model
import torchvision.models as models
model = models.vgg16(pretrained = True)

for param in model.parameters():
        param.requires_grad = False
    
# overwriting in the average pool layer
model.avgpool = nn.Sequential(
                nn.Conv2d(512,512, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten()
            )
# custom classifier for age and gender classification   
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

# function to save the best accuracy model parameters
def save_checkpoint(model, epoch, optimizer, best_acc):
    print("saving model")
    state = {
        'epoch': epoch +1,
        'model': model.state_dict(),
        'best_test_loss':best_test_loss,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, '/home/chandhinii/kinetosis/model_best_checkpoint_modified_batch32.pth')

# define the parameter for training and validation
model.classifier = ageGenderClassifier().to(device)
model = model.to(device)
age_criterion = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr= 1e-4)


val_age_maes = []
train_losses = []
val_losses = []

n_epochs = 100
best_test_loss = 100
best_mae = 1000
start = time.time()
for epoch in range(n_epochs):
    epoch_train_loss, epoch_test_loss = 0, 0
    val_age_mae,  ctr = 0, 0
    _n = len(train_loader)
    for ix, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        im,age = data
        im = im.to(device)
        age = age.to(device)
        pred_age = model(im)
        age_loss = age_criterion(pred_age.squeeze(),age)
        age_loss.backward()
        optimizer.step()
        epoch_train_loss += age_loss.item()
        
    for ix, data in enumerate(test_loader):
        model.eval()
        im,age= data
        im = im.to(device)
        age = age.to(device)
        with torch.no_grad():
            pred_age = model(im)
        age_loss = age_criterion(pred_age.squeeze(),age)
        age_mae = torch.abs(age - pred_age).float().sum()
        
        val_age_mae += age_mae
        epoch_test_loss += age_loss.item()
        ctr += len(data[0])
        
    val_age_mae /= ctr
    epoch_train_loss /= len(train_loader)
    epoch_test_loss /= len(test_loader)
    
    elapsed = time.time()-start
    #best_test_loss = min(best_test_loss, epoch_test_loss)
    with open('/home/chandhinii/age_modified_batch32.log', 'a') as f:
        f.write('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(\
                        epoch+1, n_epochs, time.time()-start, \
                        (n_epochs-epoch)*(elapsed/(epoch+1))))
        info = f'''Epoch: {epoch+1}
                    \tTrain Loss: {epoch_train_loss:.3f}
                    \tTest Loss:{epoch_test_loss:.3f}
                    \tBest Test Loss: {best_test_loss:.4f}'''
        info += f'\nAge MAE:: \
                                        {val_age_mae:.2f}\n'
        f.write(info)

    if (epoch_test_loss < best_test_loss):
        best_test_loss = epoch_test_loss
        save_checkpoint(model, epoch, optimizer, best_test_loss)
        
    val_age_maes.append(val_age_mae)

im = cv2.imread("/home/chandhinii/kinetosis/test/file_5.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(224,224)) / 255
im = torch.tensor(im).permute(2,0,1).to(device)
im = train_dataset.normalize(im)
age = model(im[None].float())
pred_age = age.to('cpu').detach().numpy()
print(' Predicted age_5', int(pred_age[0][0]* 79))

im = cv2.imread("/home/chandhinii/kinetosis/test/file_131.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(224,224)) / 255
im = torch.tensor(im).permute(2,0,1).to(device)
im = train_dataset.normalize(im)
age = model(im[None].float())
pred_age = age.to('cpu').detach().numpy()
print(' Predicted age_131', int(pred_age[0][0]* 79))


im = cv2.imread("/home/chandhinii/kinetosis/test/file_23.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(224,224)) / 255
im = torch.tensor(im).permute(2,0,1).to(device)
im = train_dataset.normalize(im)
age = model(im[None].float())
pred_age = age.to('cpu').detach().numpy()
print(' Predicted age_23', int(pred_age[0][0]* 79))
