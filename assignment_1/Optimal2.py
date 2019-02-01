from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers

import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model01 = Sequential()
model01.add(Conv2D(64, (4, 4),strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01), input_shape=x_train.shape[1:]))
model01.add(Activation('relu'))
model01.add(Conv2D(64, (4, 4),strides=1,kernel_regularizer=regularizers.l2(0.01)))
model01.add(Activation('relu'))
model01.add(MaxPooling2D(pool_size=(2, 2)))
model01.add(Dropout(0.25))

model01.add(Conv2D(64, (4, 4),strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01)))
model01.add(Activation('relu'))
model01.add(Conv2D(64, (4, 4),strides=1,kernel_regularizer=regularizers.l2(0.01)))
model01.add(Activation('relu'))
model01.add(MaxPooling2D(pool_size=(2, 2)))
model01.add(Dropout(0.25))

model01.add(Conv2D(64, (4, 4),strides=1, padding='same',kernel_regularizer=regularizers.l2(0.01)))
model01.add(Activation('relu'))
model01.add(Conv2D(64, (4, 4),strides=1,kernel_regularizer=regularizers.l2(0.01)))
model01.add(Activation('relu'))
model01.add(MaxPooling2D(pool_size=(2, 2)))
model01.add(Dropout(0.25))

model01.add(Flatten())
model01.add(Dense(512))
model01.add(Activation('relu'))
model01.add(Dropout(0.5))
model01.add(Dense(num_classes))
model01.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model01.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model01.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

scores = model01.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

history = model01.history
history_dict = history.history
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'go', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()


