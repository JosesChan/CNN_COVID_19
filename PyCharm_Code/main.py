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
from sklearn.model_selection import train_test_split


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

chosenDf = patientDf.head(5)

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
    desired_depth = 128
    current_depth = img.shape[0]
    print("Current Depth: ", current_depth)
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (depth_factor, 1, 1), mode='nearest')
    print("New Depth", img_new.shape[0])
    print(img_new.shape)
    return img_new

processedCTScans = []
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
    processedCTScans.append(imageSlice)

    plt.imshow(imageSlice[0])
    plt.show()
    imageSlice=None

print((np.asanyarray(processedCTScans)).shape)
 
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

model = NeuralNetwork().to(device)
print(model)

logits = model(processedCTScans)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


