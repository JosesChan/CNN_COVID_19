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

#define parameters
displaySampleSize = 2
sampleSizeCovidPositive = 10
sampleSizeCovidNegative = 10
random.seed(0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# file locations
covid_files_path = 'C:\\Users\\Shadow\\Documents\\GitHub\\CNN_COVID_19\\PyCharm_Code\\CT_Scans\\Covid_Positive'
non_covid_files_path = 'C:\\Users\\Shadow\\Documents\\GitHub\\CNN_COVID_19\\PyCharm_Code\\CT_Scans\\Covid_Negative'

# Read in covid and non covid file locations
# covid_files      = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
# covid_images    =  [cv2.imread(x) for x in random.sample(covid_files, displaySampleSize)]
# non_covid_files      = [os.path.join(non_covid_files_path, x) for x in os.listdir(non_covid_files_path)]
# non_covid_images    =  [cv2.imread(x) for x in random.sample(non_covid_files, displaySampleSize)]
covidPatients = os.listdir(covid_files_path)
nonCovidPatients = os.listdir(non_covid_files_path)

covidPositiveDf = pandas.DataFrame(covidPatients)
covidPositiveDf.insert(1, "label", 1, True)
print(covidPositiveDf.head())
print("End")

covidNegativeDf = pandas.DataFrame(nonCovidPatients)
covidNegativeDf.insert(1, "label", 0, True)
print(covidNegativeDf.head())
print("End")

# for i in covidPatients[:1]:
#      label = covidPositiveDf.get

# # read in slices
# slices = [pydicom.read_file(non_covid_files_path+"/"+s) for s in os.listdir(non_covid_files_path)]
# # sort slices by their image position in comparison to other slices
# slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
# print(len(slices),label)

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






# for patient in patients[:1]:
#     label = labels_df.get_value(patient, 'cancer')
#     path = data_dir + patient
#     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]

# # Display ct covid positive
# plt.figure(figsize=(20,10))
# for i, image in enumerate(covid_images):
#     plt.subplot(len(covid_images) / displaySampleSize + 1, displaySampleSize, i + 1)
#     plt.imshow(image)
#     plt.show()
# print("Check if cuda is available")
# print(torch.cuda.is_available())


# # Display ct covid negative
# plt.figure(figsize=(20,10))
# for i, image in enumerate(non_covid_images):
#     plt.subplot(len(non_covid_images) / displaySampleSize + 1, displaySampleSize, i + 1)
#     plt.imshow(image)
#     plt.show()

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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