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

mb = para_test(bsize=128, nlay=512, deep=False, lr=0.05, ActFun='relu', DropRate=0.25, ep=40)
m1 = para_test(bsize=128, nlay=512, deep=False, lr=0.05, ActFun='relu', DropRate=0.25, ep=10)
m2 = para_test(bsize=1000, nlay=512, deep=False, lr=0.05, ActFun='relu', DropRate=0.25, ep=40)
m3 = para_test(bsize=128, nlay=64, deep=False, lr=0.05, ActFun='relu', DropRate=0.25, ep=40)
m4 = para_test(bsize=128, nlay=512, deep=True, lr=0.05, ActFun='relu', DropRate=0.25, ep=40)
m5 = para_test(bsize=128, nlay=512, deep=False, lr=0.05, ActFun='relu', DropRate=0.75, ep=40)
m6 = para_test(bsize=128, nlay=512, deep=False, lr=0.05, ActFun='sigmoid', DropRate=0.25, ep=40)
m7 = para_test(bsize=128, nlay=512, deep=False, lr=0.5, ActFun='relu', DropRate=0.25, ep=40)
#m_optimal

def show_r(model):
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('test_acc:', test_acc)

    history = model.history
    #history_dict = history.history

    acc = history.history['acc']
    print('train acc:', acc[-1])
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

    plt.plot(epochs, acc, 'go', label='Training acc')
    # b is for "solid blue line"
    #plt.plot(epochs, val_acc, 'g', label='Validation loss')
    plt.title('Training acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()

    plt.show()

show_r(mb)
show_r(m1)
show_r(m2)
show_r(m3)
show_r(m4)
show_r(m5)
show_r(m6)
show_r(m7)


