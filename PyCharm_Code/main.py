# Sources used in development of this file:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.cs.toronto.edu/~kriz/cifar.html
# https://stanfordmlgroup.github.io/competitions/mrnet/
# https://blog.paperspace.com/fighting-coronavirus-with-ai-building-covid-19-classifier/
# Programmer involved : Fong Sun Joses Chan
# Creation Date 23/11/2021
# Information:
# Uses CIFAR10 Dataset to prototype a 2D Convolutional Neural Network
# Interesting notes:
# How much CT scans are there of healthy patients without any diseases? Does this limit effectiveness?

#Dicom compressed image workaround providedd by:
#https://github.com/pydicom/pylibjpeg

from tkinter.ttk import LabeledScale
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
import seaborn as sns
import pandas
import torch.utils.data
import pydicom
import glob
import re
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pydicom.pixel_data_handlers import apply_voi_lut
from torchvision.models import vgg19_bn
from scipy.ndimage import zoom

#define parameters
displaySampleSize = 2
sampleSizeCovidPositive = 10
sampleSizeCovidNegative = 10
random.seed(0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# file locations
covid_files_path = 'C:\\Users\\Shadow\\Documents\\GitHub\\CNN_COVID_19\\PyCharm_Code\\CT_Scans\\Covid_Positive\\'
non_covid_files_path = 'C:\\Users\\Shadow\\Documents\\GitHub\\CNN_COVID_19\\PyCharm_Code\\CT_Scans\\Covid_Negative\\'

# Read in covid and non covid file locations
# covid_files      = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
# covid_images    =  [cv2.imread(x) for x in random.sample(covid_files, displaySampleSize)]
# non_covid_files      = [os.path.join(non_covid_files_path, x) for x in os.listdir(non_covid_files_path)]
# non_covid_images    =  [cv2.imread(x) for x in random.sample(non_covid_files, displaySampleSize)]
covidPatients = os.listdir(covid_files_path)
nonCovidPatients = os.listdir(non_covid_files_path)
patientsSum = covidPatients + nonCovidPatients

covidPositiveDf = pandas.DataFrame(covidPatients)
covidPositiveDf.insert(1, "label", 1, True)
print(covidPositiveDf.head())

covidNegativeDf = pandas.DataFrame(nonCovidPatients)
covidNegativeDf.insert(1, "label", 0, True)
print(covidNegativeDf.head())

patientDf = pandas.concat([covidPositiveDf,covidNegativeDf])
patientDf = patientDf.astype({'label':'int'})
print(patientDf.head())

chosenDf = patientDf.head(3)

print("Total Patient Count")
print(len(chosenDf))

       
# Source Code for resizer and SIZ
#https://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynb

# Resize 2D slices using bicubic interpolation to common CT size
def rs_img(img):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    return img

# Spline interpolated zoom (SIZ)
def change_depth_siz(img):
    desired_depth = 64
    current_depth = img.shape[-1]
    print("Current Depth: ", current_depth)
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (1, 1, depth_factor), mode='nearest')
    print("New Depth", img.shape[-1])
    return img_new


for index, i in chosenDf.iterrows():
    label = i["label"]
    imageSlice = []

    if(label==1):
        path = covid_files_path + chosenDf.iat[index,0]

    if(label==0):
        path = non_covid_files_path + chosenDf.iat[index,0]

    # read in slices
    slices = [pydicom.read_file(path+"/"+s) for s in os.listdir(path)]
    # sort slices by their image position in comparison to other slices 
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices), slices[0].Rows, slices[0].Columns)
    
    # collect CT scan data as a pixel array, resize then store
    for i in range(len(slices)):
        imageSlice.append(rs_img(slices[i].pixel_array))
    print("Resize x and y complete")

    # Apply sampling technique
    imageSlice = change_depth_siz(np.asanyarray(imageSlice))

    print("Raw Pixel Volume Size: ", imageSlice)

    imageSlice=None

trainingData = np.array()
print(trainingData.shape)
 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        self.sequentialLayers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    #     self.sequentialLayers = nn.Sequential(
    #     # Convolution and Pooling layer
    #     nn.Conv3d(),
    #     nn.MaxPool3d(),
    #     nn.BatchNorm3d(),
    #     nn.Conv3d(),
    #     nn.MaxPool3d(),
    #     nn.BatchNorm3d(),
    #     nn.Conv3d(),
    #     nn.MaxPool3d(),
    #     nn.BatchNorm3d(),
    #     nn.Conv3d(),
    #     nn.MaxPool3d(),
    #     nn.BatchNorm3d(),
    #     nn.Conv3d(),
    #     nn.MaxPool3d(),
    #     nn.BatchNorm3d(),
    #     nn.Flatten(),
    #     #Full connected layer
    #     nn.ReLU(),
    #     nn.Dropout3d,
    #     nn.ReLU(),
    #     nn.Dropout3d
    #    ) 

    # feed foward function
    def forward(self, x):
        # flatten for input layer
        x = self.flatten(x)
        # create non normalised predictions
        logits = self.sequentialLayers(x)
        # return non normalised predictions
        return logits

# model = NeuralNetwork().to(device)
# print(model)

# print(f"Model structure: {model}\n\n")

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")






# batch_size = 4

# # prepare training data
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# # prepare testing data
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# class CovidCTDataset(torch.utils.data.Dataset):
#     # Intialisation variables and 
#     def __init__(self, root_dir, classes, covid_files, non_covid_files, transform=None):
#         self.root_dir = root_dir
#         self.classes = classes
#         self.files_path = [non_covid_files, covid_files]
#         self.image_list = []

#         # read the files from data split text files
#         covid_files = read_txt(covid_files)
#         non_covid_files = read_txt(non_covid_files)

#         # combine the positive and negative files into a cummulative files list
#         for cls_index in range(len(self.classes)):
            
#             class_files = [[os.path.join(self.root_dir, self.classes[cls_index], x), cls_index] \
#                             for x in read_txt(self.files_path[cls_index])]
#             self.image_list += class_files
                
#         self.transform = transform
#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         path = self.image_list[idx][0]
        
#         # Read the image
#         image = Image.open(path).convert('RGB')
        
#         # Apply transforms
#         if self.transform:
#             image = self.transform(image)

#         label = int(self.image_list[idx][1])

#         data = {'img':   image,
#                 'label': label,
#                 'paths' : path}

#         return data