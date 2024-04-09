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
splits = np.random.randn(len(df)) < 0.8
train_set = df[splits]
val_set = df[~splits]

train_dataset = CustomDatasetGen(train_set)
val_dataset = CustomDatasetGen(val_set)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(val_dataset, batch_size=32)
a,b, = next(iter(train_loader))

# get resnet34 pretrained model
import torchvision.models as models
model = models.resnet34(pretrained = True)

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

# function to save the best accuracy model parameters
def save_checkpoint(model, epoch, optimizer, best_acc):
    print("saving model")
    state = {
        'epoch': epoch +1,
        'model': model.state_dict(),
        'best_accuracy':best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, '/kaggle/working/model_best_checkpoint_gen.pth')

# define the parameter for training and validation
model.fc = genderClassifier().to(device)
model = model.to(device)
gender_criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr= 1e-4)


val_gender_accuracies = []
train_losses = []
val_losses = []

n_epochs = 80
best_test_loss = 1000
best_acc = 0
start = time.time()
for epoch in range(n_epochs):
    epoch_train_loss, epoch_test_loss = 0, 0
    val_gender_acc, ctr = 0, 0
    _n = len(train_loader)
    for ix, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        im,gen = data
        im = im.to(device)
        gen = gen.to(device)
        pred_gen = model(im)
        gen_loss = gender_criterion(pred_gen, gen)
        gen_loss.backward()
        optimizer.step()
        epoch_train_loss += gen_loss.item()
        
    for ix, data in enumerate(test_loader):
        model.eval()
        im,gen= data
        im = im.to(device)
        gen = gen.to(device)
        with torch.no_grad():
            pred_gen  = model(im)
        gen_loss = gender_criterion(pred_gen ,gen)
        total_loss = gen_loss
        pred_gender = (pred_gen > 0.5)
        gender_acc = (pred_gender == gen).float().sum()
        
        epoch_test_loss += total_loss.item()
        val_gender_acc += gender_acc
        ctr += len(data[0])
        
    val_gender_acc /= ctr
    epoch_train_loss /= len(train_loader)
    epoch_test_loss /= len(test_loader)
    
    elapsed = time.time()-start
    best_test_loss = min(best_test_loss, epoch_test_loss)
    with open('/kaggle/working/gender_batch32.log', 'a') as f:
        f.write('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(\
                        epoch+1, n_epochs, time.time()-start, \
                        (n_epochs-epoch)*(elapsed/(epoch+1))))
        info = f'''Epoch: {epoch+1}
                    \tTrain Loss: {epoch_train_loss:.3f}
                    \tTest Loss:{epoch_test_loss:.3f}
                    \tBest Test Loss: {best_test_loss:.4f}'''
        info += f'\nGender Accuracy: {val_gender_acc*100:.2f}\n'
        f.write(info)

    if (val_gender_acc > best_acc):
        best_acc = val_gender_acc
        save_checkpoint(model, epoch, optimizer, best_acc)
        
    
    val_gender_accuracies.append(val_gender_acc)

im = cv2.imread("/kaggle/input/kinet-test/test/file_5.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(224,224)) / 255
im = torch.tensor(im).permute(2,0,1).to(device)
im = train_dataset.normalize(im)
gender = model(im[None].float())
pred_gender = gender.to('cpu').detach().numpy()


print('predicted gender_5:',np.where(pred_gender[0][0]>0.5, \
                                   'Female','Male'))

im = cv2.imread("/kaggle/input/kinet-test/test/file_1.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(224,224)) / 255
im = torch.tensor(im).permute(2,0,1).to(device)
im = train_dataset.normalize(im)
gender = model(im[None].float())
pred_gender = gender.to('cpu').detach().numpy()



print('predicted gender_1:',np.where(pred_gender[0][0] >0.5, \
                                   'Female','Male'))


im = cv2.imread("/kaggle/input/kinet-test/test/file_23.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(224,224)) / 255
im = torch.tensor(im).permute(2,0,1).to(device)
im = train_dataset.normalize(im)
gender = model(im[None].float())
pred_gender = gender.to('cpu').detach().numpy()


print('predicted gender_23:',np.where(pred_gender[0][0] >0.5, \
                                   'Female','Male'))