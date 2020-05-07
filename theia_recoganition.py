# -*- coding: utf-8 -*-
"""Theia Recoganition.ipynb""""



from matplotlib import pyplot
from matplotlib.image import imread
from PIL import Image
import cv2
import os 
from numpy import *
import numpy as np
import random
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# plot photos of custom dataset

# define location of dataset
folder = '/content/Dataset/'
# plot first few imagesi
i = 0
for i in range(1,9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	i +=1
	# define filename
	filename = folder  + str(i) + '.jpg'
  # load image pixels
	image = cv2.imread(filename , 0)
  # shape of the image
	image.shape
	# plot raw pixel data
	pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()
# image.format

# Creating new dataset with resized images
#definiing new path(optional) and old path for processing
dataset = '/content/dataset/'
folder = '/content/Dataset/'
ls = os.listdir(folder)
ls.sort()
nosample = len(ls)
print(nosample)
for file in ls:
  #since name is string 
  s = file.split('.')[0]
  #opening image using pillow pacakage
  image = Image.open(folder + file)
  #resize the image for faster processing
  imag = image.resize((28,28))
  #check if dir exits
  if not os.path.exists(dataset):
    # savind data in JPEG
    imag.save(dataset + s  , "JPEG")
#printing Length
print(len(os.listdir(dataset)))
print(file)

#ploting images of resized dataset
# define location of dataset
folder = '/content/Dataset/'
dataset = '/content/dataset/'
ls = os.listdir(dataset)
ls.sort()
# plot first few images
for i in range(102,109):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = dataset  + str(i) 
  # load image pixels
	image = cv2.imread(filename , 0)
	# n=np.array
	# plot raw pixel data
	pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

#intitialize  a list for label
label = []
dataset = '/content/dataset/'
ls= os.listdir(dataset)
ls.sort()
#intitialize list for traing data
training_data = []
#looping the image dataset
for img in ls:
  image = Image.open(dataset + img)
  # converting the filename to integer for labeling
  h = int(img)
  # defining the condition for labeling
  if  h <= 135:
    # label 1 
    label =  1
  else:
    #label 2
    label = 2
  # convert the image to n dim array for processing
  new_array = np.array(image, 'f')
  #append the array and label to single list
  training_data.append([new_array , label])

#printing the types of list
print(type(training_data[0][0]))
print(type(training_data[0][]))

#listing the training data
for a in training_data[0:5]:
  #printing the labels
  print(a[1])
print(len(training_data))

#intialise new list for features and target varaible
X = []
Y = []

#shuffle the training data  
random.shuffle(training_data)

#define the features and labels in training data
for feature,labels in training_data:
  # print(feature.shape)
  # print(labels)
  # break
  #list of image array
  X.append(feature)
  #list of labels
  Y.append(labels)
# print(X)
print(len(feature))

#convert the list into array since CNN does operaion on arrays and reshape with respect model 
# -1-for all samples, 28,28 -defines width and height , 1- channel number since image n greyscale
x = np.array(X).reshape(-1,28,28,1)
#convert the list into array 
y = np.array(Y)

# create a final dataset with images and labels
# concatenate features and labels
data = [ x, y] 
#type of data
print(type(data))
# print shapes of features and labels
print(data[0].shape)
print(data[1].shape)

#define 2 variables for features and labels (optional)
(X,y) = (data[0],data[1])
print(type(y))

# split the final training dataset to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# print shapes of train and test data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#convert integers to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize to range 0-1
X_train /= 255
X_test /= 255

#printing sample  list
print('Train Shape' ,X_train.shape)
print(X_train.shape[0], 'Train samples')
print(X_test.shape[1], 'Test Samples')



# no_classes=5
# print("Shape after one-hot encoding: ", type(y_train))

#converting to binary class matrices
# one-hot encoding using keras' numpy-related utilities
print("Shape before one-hot encoding: ", y_train.shape)

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print("Shape after one-hot encoding: ", Y_train.shape)

# print(type(Y_train))

# i = 100
# pyplot.imshow(X_train[i,0], interpolation='nearest')
# print("lavbel",Y_train[i:])

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

# output layer
model.add(Dense(3, activation='softmax'))
	
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training the model for 10 epochs
model.fit(X_train, Y_train, epochs=10, batch_size=32,  validation_data=(X_test, Y_test), verbose=0)

# evaluate model
score = model.evaluate(X_test, Y_test,  verbose=0)
#print score and accuracy
print("test Score", score[0])
print("test accutacy", score[1])

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
# def load_image(filename):
	# load the image
img = Image.open('/content/34.jpg')
img = img.resize((28,28))
	# convert to array
img = img_to_array(img)
	# reshape into a single sample with 1 channel
img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
img = img.astype('float32')
img = img / 255.0
	# return img

# load an image and predict the class
# def run_example():
	# load the image
	# img = load_image()
	# load model
	# model = load_model('final_model.h5')
	# predict the class
digit = model.predict_classes(img)
print(digit[0])

