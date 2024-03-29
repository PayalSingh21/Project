# -*- coding: utf-8 -*-
"""Project_week2_imp_model10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1huQXmhU0PU-1RNmhvD6CbQzTF_qeYzNB
"""

#Import the libraries 

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

data = np.load('modelnet10.npz')

#Finding out the array names

np_Array=np.load('modelnet10.npz')
print(np_Array.files)

X_train, Y_train = shuffle(data['X_train'], data['y_train'])
X_test, Y_test = shuffle(data['X_test'], data['y_test'])

#Define the model structure.
#We will create a simple architecture with 2 convolutional layers, one dense hidden layer and an output layer

from keras import backend as K
K.clear_session()
model = Sequential()
model.add(Reshape((30, 30, 30, 1), input_shape=(30, 30, 30)))
model.add(Conv3D(input_shape=(30, 30, 30, 1), filters=30, kernel_size=(5,5,5), strides=(2, 2, 2)))
model.add(Conv3D(32, kernel_size=(3, 3, 3),activation='relu',input_shape=(30,30,30, 1)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(Activation(LeakyReLU(alpha=0.1)))
#model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(rate=0.3))
model.add(Conv3D(filters=32, kernel_size=(3,3,3)))
model.add(Activation(LeakyReLU(alpha=0.1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None))
model.add(Dropout(rate=0.4))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
#model.load_weights("modelnet10.npz")
print (model.summary())

print(X_train.shape)
print(Y_train.shape)

from keras.utils import to_categorical
Y_train = to_categorical(Y_train)

print(Y_train.shape)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=560, epochs=50, verbose=2,
          validation_split=0.2, shuffle=True)

Y_test_pred = np.argmax(model.predict(X_test), axis=1)
print('Test accuracy: {:.3f}'.format(accuracy_score(Y_test, Y_test_pred)))

conf = confusion_matrix(Y_test, Y_test_pred)
avg_per_class_acc = np.mean(np.diagonal(conf) / np.sum(conf, axis=1))
print('Confusion matrix:\n{}'.format(conf))
print('Average per-class accuracy: {:.3f}'.format(avg_per_class_acc))