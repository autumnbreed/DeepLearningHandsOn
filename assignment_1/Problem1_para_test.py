# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:47:17 2019

@author: autpx
"""

import csv
import copy
from io import StringIO
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.lines as lin
import matplotlib.patches as pat
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os

import keras
keras.__version__
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape((50000, 32 * 32 * 3))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 32 * 32 * 3))
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


from keras import models
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras import optimizers


fnetwork = models.Sequential()
fnetwork.add(layers.Dense(512, activation='relu', input_shape=(3072,)))
fnetwork.add(layers.Dense(10, activation='softmax'))

fnetwork.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

fnetwork.fit(x_train, y_train, epochs=20, batch_size=128)

test_loss, test_acc = fnetwork.evaluate(x_test, y_test)

print('test_acc:', test_acc)

def para_test(bsize, nlay, deep, lr, ActFun, DropRate, ep):
    model = models.Sequential()
    model.add(layers.Dense(nlay, activation=ActFun, input_shape=(3072,)))
    model.add(Dropout(DropRate))
    if deep:
        model.add(layers.Dense(nlay, activation=ActFun))
        model.add(Dropout(DropRate))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    sgd = optimizers.SGD(lr=lr)
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=ep, batch_size = bsize)
    return (model)

m1 = para_test(bsize=128, nlay=1024, deep=False, lr=0.05, ActFun='relu', DropRate=0.25, ep=500)
test_loss, test_acc = m1.evaluate(x_test, y_test)
print('test_acc:', test_acc)

history = m1.history
history_dict = history.history

acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'go', label='Training loss')
# b is for "solid blue line"
#plt.plot(epochs, val_acc, 'g', label='Validation loss')
plt.title('Training acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()













