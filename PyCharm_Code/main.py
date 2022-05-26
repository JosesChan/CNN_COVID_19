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
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
import pandas
import pydicom
import sklearn
import scipy
import h5py
from pydicom.pixel_data_handlers import apply_voi_lut
from pydicom.pixel_data_handlers.util import apply_modality_lut
from scipy.ndimage import zoom, rotate
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import time

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

patientDf = pandas.concat([covidPositiveDf.sample(frac=0.20, replace=True, random_state=42),
covidNegativeDf.sample(frac=0.20, replace=True, random_state=42)])
patientDf = patientDf.astype({'label':'int'})
print(patientDf.head(6))

chosenDf = patientDf
chosenDf.insert(2, "prediction", None, allow_duplicates=True)
chosenDf.insert(3, "dataID", None, allow_duplicates=False)
chosenDf.insert(4, "data", None, allow_duplicates=False)
print("Total Patient Count")
print(len(chosenDf))

	   
# Source Code for resizer and SIZ
# https://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynbhttps://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynb
# Source Code for augmentation
# https://towardsdatascience.com/simple-3d-mri-classification-ranked-bronze-on-kaggle-87edfdef018a
# Resize 2D slices using bicubic interpolation to common CT size
def rs_img(img):
	img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
	return img

def normalise(data):
	max = 400
	min = -1000
	data[data > max] = max
	data[data < min] = min
	data = (data - min) / (max - min)
	data = data.astype("float32")
	return data

# Spline interpolation
def change_depth_siz(img):
	desired_depth = 64
	current_depth = img.shape[0]
	print("Current Depth: ", current_depth)
	depth = current_depth / desired_depth
	depthFactor = 1 / depth
	img_new = zoom(img, (depthFactor, 1, 1), mode='nearest')
	print("New Depth: ", img_new.shape[0])
	print(img_new.shape)
	return img_new

def resize_data(img):
	# Set the desired depth
	desired_depth = 64
	desired_width = 128
	desired_height = 128
	# Get current depth
	current_depth = img.shape[-1]
	current_width = img.shape[0]
	current_height = img.shape[1]
	# Compute depth factor
	depth = current_depth / desired_depth
	width = current_width / desired_width
	height = current_height / desired_height
	depth_factor = 1 / depth
	width_factor = 1 / width
	height_factor = 1 / height
	# Rotate
	# img = scipy.ndimage.rotate(img, 90, reshape=False)
	# Resize across z-axis
	# img = change_depth_siz(img)
	img = scipy.ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
	return img

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
			imageNormalised = normalise(slices[imageCount].pixel_array)
			imageVOI = apply_voi_lut(slices[imageCount].pixel_array, slices[imageCount]).astype(np.float32)
			imageSlice.append((imageNormalised))
		print("Resize x and y complete")

		# Apply sampling technique
		imageSlice = resize_data(np.asanyarray(imageSlice))
		# print(imageSlice.shape)
		processedCTScans.append(imageSlice)
		dataset.iat[i, 4] = imageSlice
		dataset.iat[i, 3] = i
		# plt.imshow(imageSlice[0])
		# plt.show()
		imageSlice=None
	print(np.asanyarray(processedCTScans).shape)
 
loadProcessImage(chosenDf, yImages)


xTraining, xTesting, yTraining, yTesting  = train_test_split(chosenDf["data"], chosenDf["label"], train_size = 0.7, test_size=0.3,random_state=42)


train_x = np.asanyarray(xTraining.tolist())
test_x = np.asanyarray(xTesting.tolist())
train_y = np.asanyarray(yTraining.tolist())
test_y = np.asanyarray(yTesting.tolist())

print("Training Data:\n", train_x.shape)
print("Testing Data:\n", test_x.shape)
print("Y Train:\n", train_y)
print("Y Test:\n", test_y)

# dataloaders for the model
train_loader = tf.data.Dataset.from_tensor_slices((train_x, train_y))
validation_loader = tf.data.Dataset.from_tensor_slices((test_x, test_y))

batch_size = 2

@tf.function
def rotate(volume):
	def scipy_rotate(volume):
		# angles that will be chosen randomly to rotate CT images
		angles = [-20, -10, -5, 5, 10, 20]
		angle = random.choice(angles)
		volume = scipy.ndimage.rotate(volume, angle, reshape=False)
		volume[volume < 0] = 0
		volume[volume > 1] = 1
		return volume
	augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
	return augmented_volume

# Add rotation dimension to training data
def train_preprocessing(volume, label):
	volume = rotate(volume)
	volume = tf.expand_dims(volume, axis=3)
	return volume, label

# Add rotation dimension to testing data
def validation_preprocessing(volume, label):
	volume = tf.expand_dims(volume, axis=3)
	return volume, label


# Changes values during training.
train_dataset = (
	train_loader.shuffle(len(train_x))
	.map(train_preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

# rescale
validation_dataset = (
	validation_loader.shuffle(len(test_x))
	.map(validation_preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")

#-------------------------------------------------Training and Testing the model-------------------------------------------------#

# Source code
# https://blog.paperspace.com/fighting-coronavirus-with-ai-building-covid-19-classifier/
# https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
# https://keras.io/examples/vision/3D_image_classification/#loading-data-and-preprocessing


def get_model(width=128, height=128, depth=64):

	inputs = keras.Input((width, height, depth, 1))

	x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.GlobalAveragePooling3D()(x)
	x = layers.Dense(units=512, activation="relu")(x)
	x = layers.Dropout(0.3)(x)

	outputs = layers.Dense(units=1, activation="sigmoid")(x)

	# Define the model.
	model = keras.Model(inputs, outputs, name="3dcnn")
	return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
	loss="binary_crossentropy",
	optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
	metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
	"3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
	train_dataset,
	validation_data=validation_dataset,
	epochs=epochs,
	shuffle=True,
	verbose=2,
	callbacks=[checkpoint_cb, early_stopping_cb],
)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
	ax[i].plot(model.history.history[metric])
	ax[i].plot(model.history.history["val_" + metric])
	ax[i].set_title("Model {}".format(metric))
	ax[i].set_xlabel("epochs")
	ax[i].set_ylabel(metric)
	ax[i].legend(["train", "val"])
plt.show()
	
# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(test_x)
# scores = [1 - prediction[0], prediction[0]]

predictionList=[]
# class_names = ["normal", "abnormal"]
for predIndex in range(len(prediction)):
	if(prediction[predIndex]<0.5):
		predictionList.append(0)
	else:
		predictionList.append(1)
print(prediction)
print(predictionList)

print(classification_report(test_y, predictionList, target_names=["covidNegative","covidPositive"]))


