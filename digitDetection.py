# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:25:45 2020

@author: suhail
"""
import numpy as np
import mnist #get dataset
import matplotlib.pyplot as plt #for plotting
from keras.models import Sequential #ANN architecture
from keras.layers import Dense #The layers in the ANN
from keras.utils import to_categorical

#Importing the dataset
train_images=mnist.train_images() #training data images
train_labels=mnist.train_labels() #training data label

test_images=mnist.test_images() #test data images
test_labels=mnist.test_labels() #test data labels

#Feature scaling normalizing the pixel value from [0-255] to [-0.5 to 0.5]
train_images=(train_images/255)-0.5
test_images=(test_images/255)-0.5

#Flatten the images. Flatten each 28x28 image into a 784(28^2) dimensional vector to pass into the neural network
train_images=train_images.reshape((-1,784))
test_images=test_images.reshape((-1,784))

#size
#train_images= (60000,784)
#test_images= (10000,784)

#Build the model
# 3 layers, 2 layer with 64 neurons and relu function and 1 layer with 10 neurons and softmax function
model=Sequential()
model.add(Dense(64,activation='relu',input_dim=784))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

#Compile the model
#The loss function measures how well the model did in training and then tries to improvise it using optimizer
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Train the model
model.fit(train_images,to_categorical(train_labels),epochs=5,batch_size=32)

#Evaluate the model
model.evaluate(test_images,to_categorical(test_labels))

#Predictions the first 5 test images
predictions=model.predict(test_images[:5])
print(predictions)
print(np.argmax(predictions,axis=1))
print(test_labels[:5])

#plotting
for i in range(5):
    first_image=test_images[i]
    first_image=np.array(first_image,dtype='float')
    pixels=first_image.reshape((28,28))
    plt.imshow(pixels,cmap='gray')
    plt.show()