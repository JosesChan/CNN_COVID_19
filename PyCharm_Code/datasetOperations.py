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

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import random

covid_files_path = 'CT_Scans/CNN_COVID_19/'
covid_files      = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
covid_images    =  [cv2.imread(x) for x in random.sample(covid_files, 5)]

# Display ct images
plt.figure(figsize=(20,10))

columns = 5
for i, image in enumerate(covid_images):
    plt.subplot(len(covid_images) / columns + 1, columns, i + 1)
    plt.imshow(image)


print(torch.cuda.is_available())


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# prepare training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# prepare testing data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


