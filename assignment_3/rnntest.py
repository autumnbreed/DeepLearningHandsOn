# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:17:18 2019

@author: autpx
"""
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import os
import collections
import json
import random


import keras
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers


from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.layers import Dense

from keras.datasets import imdb
from keras.preprocessing import sequence

#######################################################
max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

######################## model
model1 = Sequential()
model1.add(Embedding(max_features, 32))
model1.add(SimpleRNN(32))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history1 = model1.fit(input_train, y_train,
                    epochs=10,
                    batch_size=1024,
                    validation_split=0.2)

acc = history1.history['acc']
val_acc = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_f = model1.predict(input_test)
ylist = test_f.tolist()
ypred = [y[0] for y in ylist]
ytrue = y_test.tolist()
ypred = [1 if y >= 0.5 else 0 for y in ypred]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytrue, ypred)

#np.set_printoptions(precision=2)
#cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#print(cm_normalized)
from plot_cm import plot_confusion_matrix
plot_confusion_matrix(cm,
                      normalize    = False,
                      target_names = ['po', 'ng'],
                      title        = "Confusion Matrix")

#######################################################
data1 =[]
def load (filename):
    with open(filename, 'r') as f:
        load_dict = json.load(f)
        for key, v in load_dict['text'].items():
            if load_dict['sentiment'][key].lower() == "positive":p = 0
            else:
                if load_dict['sentiment'][key].lower() == 'neutral':p = 1
                else: p = 2
            data1.append([v, p])
    return 0


load('./Team1_Google.json')
load('./Team2_Amazon.json')
load('./Team3_Facebook.json')
load('./Team4_Netflix.json')
load('./Team5_Microsoft.json')
load('./Team6_Tesla.json')
load('./Team7_Walmart.json')
load('./Team8_Kroger.json')
load('./Team9_GoldmanSachs.json')
load('./Team11_Boeing.json')
load('./Team12_Chevron.json')

random.shuffle(data1)
xdata = [v[0] for v in data1]
ydata = [v[1] for v in data1]

yc = keras.utils.to_categorical(ydata)


from keras_preprocessing import text
from keras.preprocessing.text import Tokenizer
text_to_word_sequence = text.text_to_word_sequence
one_hot = text.one_hot
hashing_trick = text.hashing_trick
tokenizer_from_json = text.tokenizer_from_json

t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(xdata)

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

#encoded_docs = t.texts_to_matrix(docs, mode='count') #mode='binary'
encoded_docs = t.texts_to_sequences(xdata)
# sequences_to_texts

max_f = 6090  # number of words to consider as features
maxl = 400  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
xc = sequence.pad_sequences(encoded_docs, maxlen=maxl)
xc_t = xc[:1200,]
yc_t = yc[:1200,]
xc_v = xc[1201:,]
yc_v = yc[1201:,]
#yyyy = yc_v.tolist()
print('train shape:', xc_t.shape)
print('test shape:', xc_v.shape)


model2 = Sequential()
model2.add(Embedding(max_f, 32))
model2.add(SimpleRNN(32,return_sequences=True))
model2.add(SimpleRNN(32,return_sequences=True))
model2.add(SimpleRNN(32))
model2.add(Dense(3, activation='softmax'))
model2.summary()

model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history2 = model2.fit(xc_t, yc_t,
                    epochs=10,
                    batch_size=64)


acc = history2.history['acc']
loss = history2.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')

v_f = model2.predict(xc_v)
#print(np.max(v_f,axis=1))
vflist = v_f.tolist()

ypred = [bb.index(max(bb)) for bb in vflist]
ytrue = ydata[1201:]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytrue, ypred)

#np.set_printoptions(precision=2)
#cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#print(cm_normalized)

from plot_cm import plot_confusion_matrix
plot_confusion_matrix(cm,
                      normalize    = False,
                      target_names = ['po', 'n', 'ng'],
                      title        = "Confusion Matrix")