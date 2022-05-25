# Sources used in development of this file:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.cs.toronto.edu/~kriz/cifar.html
# https://stanfordmlgroup.github.io/competitions/mrnet/
# https://blog.paperspace.com/fighting-coronavirus-with-ai-building-covid-19-classifier/
# Programmer involved : Fong Sun Joses Chan
# Creation Date 23/11/2021
# Information:
# Used CIFAR10 Dataset to prototype a 2D Convolutional Neural Network
# Interesting notes:
# How much CT scans are there of healthy patients without any diseases? Does this limit effectiveness?

#Dicom compressed image workaround provided by:
#https://github.com/pydicom/pylibjpeg

from itertools import count
from tkinter.ttk import LabeledScale
from collections import deque
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
import pandas
import torch.utils.data
import pydicom
import sklearn
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pydicom.pixel_data_handlers import apply_voi_lut
from torchvision.models import vgg19_bn
from scipy.ndimage import zoom
from torch.autograd import Variable
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# file locations
covid_files_path = 'C:\\Users\\Shadow\\Documents\\GitHub\\CNN_COVID_19\\PyCharm_Code\\CT_Scans\\Covid_Positive\\'
non_covid_files_path = 'C:\\Users\\Shadow\\Documents\\GitHub\\CNN_COVID_19\\PyCharm_Code\\CT_Scans\\Covid_Negative\\'

# Read in covid and non covid file locations
covidPatients = os.listdir(covid_files_path)
nonCovidPatients = os.listdir(non_covid_files_path)
patientsSum = covidPatients + nonCovidPatients

covidPositiveDf = pandas.DataFrame(covidPatients)
covidPositiveDf.insert(1, "label", 1, True)
print(covidPositiveDf.head())

covidNegativeDf = pandas.DataFrame(nonCovidPatients)
covidNegativeDf.insert(1, "label", 0, True)
print(covidNegativeDf.head())

patientDf = pandas.concat([covidPositiveDf.head(3),covidNegativeDf.head(3)])
patientDf = patientDf.astype({'label':'int'})
print(patientDf.head(6))

chosenDf = patientDf.head(6)
chosenDf.insert(2, "prediction", None, allow_duplicates=True)
chosenDf.insert(3, "dataID", None, allow_duplicates=False)
chosenDf.insert(4, "data", None, allow_duplicates=False)
print("Total Patient Count")
print(len(chosenDf))

       
# Source Code for resizer and SIZ
#https://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynb

# Resize 2D slices using bicubic interpolation to common CT size
def rs_img(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    return img

# Spline interpolated zoom (SIZ)
def change_depth_siz(img):
    desired_depth = 64
    current_depth = img.shape[0]
    print("Current Depth: ", current_depth)
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (depth_factor, 1, 1), mode='nearest')
    print("New Depth: ", img_new.shape[0])
    print(img_new.shape)
    return img_new

yImages = []
def loadProcessImage(dataset, processedCTScans):
    for i in range(len(dataset)):
        print("Index location: ", i)
        dataset.iat[i, 3]
        label = None
        label = dataset.iat[i, 1]
        imageSlice = []
        path = ""

        if(label==0):
            path = non_covid_files_path + dataset.iat[i,0]

        if(label==1):
            path = covid_files_path + dataset.iat[i,0]

        print(path)

        # read in slices
        slices = [pydicom.read_file(path+"/"+s) for s in os.listdir(path)]
        # sort slices by their image position in comparison to other slices 
        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        print(len(slices), slices[0].Rows, slices[0].Columns)
        
        # collect CT scan data as a pixel array, resize then store
        for imageCount in range(len(slices)):
            imageSlice.append(rs_img(slices[imageCount].pixel_array))
        print("Resize x and y complete")

        # Apply sampling technique
        imageSlice = change_depth_siz(np.asanyarray(imageSlice))
        imageSlice = torch.tensor(imageSlice)
        print(imageSlice.size())
        processedCTScans.append(imageSlice)
        dataset.iat[i, 4] = imageSlice
        dataset.iat[i, 3] = i
        # plt.imshow(imageSlice[0])
        # plt.show()
        imageSlice=None
    print(np.asanyarray(processedCTScans).shape)
 
class cnnSimple(nn.Module):
    def __init__(self):
        super(cnnSimple, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

class cnnNeuralNet(nn.Module):
    def __init__(self):
        super(cnnNeuralNet, self).__init__()

        self.sequentialLayers = nn.Sequential(
        # Convolution and Pooling Layer 1
        nn.Conv3d(in_channels = 3, out_channels = 64,kernel_size = (3,3,3)),
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
        nn.BatchNorm3d(),

        # Convolution and Pooling Layer 2
        nn.Conv3d(in_channels = 64, out_channels = 64, Kernel_size=(3, 3, 3)),
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
        nn.BatchNorm3d(),

        # Convolution and Pooling Layer 3
        nn.Conv3d(in_channels = 64, out_channels = 64, Kernel_size=(3, 3, 3)),
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
        nn.BatchNorm3d(),

        # Convolution and Pooling Layer 4        
        nn.Conv3d(in_channels = 64, out_channels = 128, Kernel_size=(3, 3, 3)),
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
        nn.BatchNorm3d(),

        # Convolution and Pooling Layer 5
        nn.Conv3d(in_channels = 64, out_channels = 256, Kernel_size=(3, 3, 3)),
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
        nn.BatchNorm3d(),
        
        #Flatten
        nn.Flatten(),
        #Full connected layer
        nn.ReLU(),
        nn.Dropout3d(0.4),
        nn.ReLU(),
        nn.Dropout3d(0.4)
       ) 

    # feed foward function
    def forward(self, x):
        out = self.Model(x)
        return out

loadProcessImage(chosenDf, yImages)

batchSize = 64
epochs = 60

xTrainingDf, xTesting, yTrainingDf, yTesting  = train_test_split(chosenDf["data"], chosenDf["label"], train_size = 0.6, test_size=0.4,random_state=42)

xTraining, xVal, yTraining, yVal  = train_test_split(xTrainingDf, yTrainingDf, train_size = 0.8, test_size=0.2,random_state=42)


train_x = torch.stack(xTraining.tolist())
val_x = torch.stack(xVal.tolist())
test_x = torch.stack(xTesting.tolist())
train_y = torch.as_tensor(yTraining.to_numpy())
val_y = torch.as_tensor(yVal.to_numpy())
test_y = torch.as_tensor(yTesting.to_numpy())

print("Training Data:\n", train_x)
print("Testing Data:\n", test_x)
print("Y Train:\n", train_y)
print("Y Test:\n", test_y)

train = torch.utils.data.TensorDataset(train_x,train_y)
validation = torch.utils.data.TensorDataset(val_x,val_y)
test = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train, batch_size = batchSize, drop_last=False, shuffle = False)
validation_loader = torch.utils.data.DataLoader(train, batch_size = batchSize, drop_last=False, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batchSize, drop_last=False, shuffle = False)

# calculate steps per epoch for training and validation set
trainSteps = len(train_loader.dataset) // batchSize
valSteps = len(validation_loader.dataset) // batchSize

#-------------------------------------------------Training and Testing the model-------------------------------------------------#

# Source code
# https://blog.paperspace.com/fighting-coronavirus-with-ai-building-covid-19-classifier/
# https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = torch.nn.functional.relu(self.bn1(self.conv1(input)))      
        output = torch.nn.functional.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = torch.nn.functional.relu(self.bn4(self.conv4(output)))     
        output = torch.nn.functional.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output



class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = nn.Conv3d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)