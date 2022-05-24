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

#Dicom compressed image workaround provided by:
#https://github.com/pydicom/pylibjpeg

from itertools import count
from tkinter.ttk import LabeledScale
from collections import deque
import torch
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
from torchvision import datasets, transforms
from pydicom.pixel_data_handlers import apply_voi_lut
from torchvision.models import vgg19_bn
from scipy.ndimage import zoom
from torch.autograd import Variable
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

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

patientDf = pandas.concat([covidPositiveDf.head(5),covidNegativeDf.head(5)])
patientDf = patientDf.astype({'label':'int'})
print(patientDf.head(10))

chosenDf = patientDf.head(10)
chosenDf.insert(2, "prediction", None, allow_duplicates=True)
chosenDf.insert(3, "dataID", None, allow_duplicates=False)
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
    print("New Depth", img_new.shape[0])
    print(img_new.shape)
    return img_new

processedCTScans = []
for i in range(len(chosenDf)):
    print("Index location: ", i)
    chosenDf.iat[i, 3]
    label = None
    label = chosenDf.iat[i, 1]
    imageSlice = []
    path = ""

    if(label==0):
        path = non_covid_files_path + chosenDf.iat[i,0]

    if(label==1):
        path = covid_files_path + chosenDf.iat[i,0]

    print(path)

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
    # plt.imshow(imageSlice[0])
    # plt.show()
    imageSlice=None

print((np.asanyarray(processedCTScans)).shape)
 
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


batchSize = 64
epochs = 60

trainingDf, testingDf = torch.utils.data.random_split(chosenDf,[7,3],torch.Generator().manual_seed(42))

trainLoader = torch.utils.data.DataLoader(trainingDf, batch_size = batchSize, drop_last=False, shuffle = False)
testLoader = torch.utils.data.DataLoader(testingDf, batch_size = batchSize, drop_last=False, shuffle = False)

#-------------------------------------------------Training and Testing the model-------------------------------------------------#

# Source code
# https://blog.paperspace.com/fighting-coronavirus-with-ai-building-covid-19-classifier/

def compute_metrics(model, test_loader, plot_roc_curve = False):
    
    model.eval()
    
    val_loss = 0
    val_correct = 0
    
    criterion = nn.CrossEntropyLoss()
    
    score_list   = torch.Tensor([]).to(device)
    pred_list    = torch.Tensor([]).to(device).long()
    target_list  = torch.Tensor([]).to(device).long()
    path_list    = []
    
    for index, i in test_loader.iterrows():
        # Convert image data into single channel data
        image, target = processedCTScans[i], i['label'].to(device)

        # Compute the loss
        with torch.no_grad():
            output = model(image)
        
        # Log loss
        val_loss += criterion(output, target.long()).item()
        
        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.long().view_as(pred)).sum().item()
        
        # Bookkeeping 
        score_list   = torch.cat([score_list, nn.Softmax(dim = 1)(output)[:,1].squeeze()])
        pred_list    = torch.cat([pred_list, pred.squeeze()])
        target_list  = torch.cat([target_list, target.squeeze()])
        
    classification_metrics = classification_report(target_list.tolist(), pred_list.tolist(), target_names = ['CT_NonCOVID', 'CT_COVID'],output_dict= True)
    
    # sensitivity is the recall of the positive class
    sensitivity = classification_metrics['CT_COVID']['recall']
    
    # specificity is the recall of the negative class 
    specificity = classification_metrics['CT_NonCOVID']['recall']
    
    # accuracy
    accuracy = classification_metrics['accuracy']
    
    # confusion matrix
    conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())
    
    # roc score
    roc_score = roc_auc_score(target_list.tolist(), score_list.tolist())
    
    # plot the roc curve
    if plot_roc_curve:
        fpr, tpr, _ = roc_curve(target_list.tolist(), score_list.tolist())
        plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
        plt.legend(loc = 'best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        
    
    # put together values
    metrics_dict = {"Accuracy": accuracy,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "Roc_score"  : roc_score, 
                    "Confusion Matrix": conf_matrix,
                    "Validation Loss": val_loss / len(test_loader),
                    "score_list":  score_list.tolist(),
                    "pred_list": pred_list.tolist(),
                    "target_list": target_list.tolist(),
                    "paths": path_list}
    
    
    return metrics_dict

class EarlyStopping(object):
    def __init__(self, patience = 8):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.previous_loss = int(1e8)
        self.previous_accuracy = 0
        self.init = False
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0
        self.best_running_accuracy = 0
        self.best_running_loss = int(1e7)
    
    def add_data(self, model, loss, accuracy):
        
        # compute moving average
        if not self.init:
            running_loss = loss
            running_accuracy = accuracy 
            self.init = True
        
        else:
            running_loss = 0.2 * loss + 0.8 * self.previous_loss
            running_accuracy = 0.2 * accuracy + 0.8 * self.previous_accuracy
        
        # check if running accuracy has improved beyond the best running accuracy recorded so far
        if running_accuracy < self.best_running_accuracy:
            self.accuracy_decrease_iters += 1
        else:
            self.best_running_accuracy = running_accuracy
            self.accuracy_decrease_iters = 0
        
        # check if the running loss has decreased from the best running loss recorded so far
        if running_loss > self.best_running_loss:
            self.loss_increase_iters += 1
        else:
            self.best_running_loss = running_loss
            self.loss_increase_iters = 0
        
        # log the current accuracy and loss
        self.previous_accuracy = running_accuracy
        self.previous_loss = running_loss        
        
    
    def stop(self):
        
        # compute thresholds
        accuracy_threshold = self.accuracy_decrease_iters > self.patience
        loss_threshold = self.loss_increase_iters > self.patience
        
        
        # return codes corresponding to exhuaustion of patience for either accuracy or loss 
        # or both of them
        if accuracy_threshold and loss_threshold:
            return 1
        
        if accuracy_threshold:
            return 2
        
        if loss_threshold:
            return 3
        
        
        return 0
    
    def reset(self):
        # reset
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0
    
early_stopper = EarlyStopping(patience = 5)

model = cnnNeuralNet()
learning_rate = 0.01
momentumValue = 0.9
optimizer = nn.Sigmoid(model.parameters(), lr = learning_rate, momentum=momentumValue)

best_model = model
best_val_score = 0

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):

    model.train()    
    train_loss = 0
    train_correct = 0
    
    for iter_num, data in enumerate(trainLoader):
        image, target = data['img'].to(device), data['label'].to(device)     

        # Compute the loss
        output = model(image)
        loss = criterion(output, target.long()) / 8
        
        # Log loss
        train_loss += loss.item()
        loss.backward()

        # Perform gradient udpate
        if iter_num % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
            

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
        
    
    # Compute and print the performance metrics
    metrics_dict = compute_metrics(model, trainLoader)
    print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch,iter_num))
    print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
    print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
    print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
    print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
    print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
    print("------------------------------------------------------------------------------")
    
    # Save the model with best validation accuracy
    if metrics_dict['Accuracy'] > best_val_score:
        torch.save(model, "/storage/best_model.pkl")
        best_val_score = metrics_dict['Accuracy']
    
    
    # print the metrics for training data for the epoch
    print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, train_loss/len(trainLoader.dataset), train_correct, len(trainLoader.dataset),
        100.0 * train_correct / len(trainLoader.dataset)))
    
    # log the accuracy and losses in tensorboard
    # writer.add_scalars( "Losses", {'Train loss': train_loss / len(trainLoader), 'Validation_loss': metrics_dict["Validation Loss"]},epoch)
    # writer.add_scalars( "Accuracies", {"Train Accuracy": 100.0 * train_correct / len(trainLoader.dataset),"Valid Accuracy": 100.0 * metrics_dict["Accuracy"]}, epoch)

    # Add data to the EarlyStopper object
    early_stopper.add_data(model, metrics_dict['Validation Loss'], metrics_dict['Accuracy'])
    
    # If both accuracy and loss are not improving, stop the training
    if early_stopper.stop() == 1:
        break
    
    # if only loss is not improving, lower the learning rate
    if early_stopper.stop() == 3:
        for param_group in optimizer.param_groups:
            learning_rate *= 0.1
            param_group['lr'] = learning_rate
            print('Updating the learning rate to {}'.format(learning_rate))
            early_stopper.reset()


model = torch.load("/storage/pretrained_covid_model.pkl" )

metrics_dict = compute_metrics(model, testLoader, plot_roc_curve = True)
print('------------------- Test Performance --------------------------------------')
print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
print("------------------------------------------------------------------------------")
    