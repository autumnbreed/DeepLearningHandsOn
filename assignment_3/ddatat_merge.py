# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:32:37 2019

@author: autpx
"""

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import os
import random
import tarfile

import collections








def read_imdb(folder='train'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('E:/Workspace.win/Pywork/untitled folder/docs/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')


# stored as json


    


# Data Preprocessing


def get_tokenized_imdb(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)

vocab = get_vocab_imdb(train_data)
'# Words in vocab:', len(vocab)



def read_12(folder='./'):
    data = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'rb') as f:
            one_load = json.load(f)
            for key, v in one_load['text'].items():
                #v.decode('utf-8')
                if one_load['sentiment'][key] == 'positive':
                    la = 1
                if one_load['sentiment'][key] == 'neutral':
                    la = 2
                else: 
                    la = 3
                data.append([v, la])
    random.shuffle(data)
    return data

data12 = read_12('E:/Workspace.win/Pywork/untitled folder/docs/doc12/')

    
    
    

