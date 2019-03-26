# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:20:50 2019

@author: autpx
"""

from tpot import TPOTClassifier
import numpy as np
import pandas as pd

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,random_state=42, verbosity=2)

merged = pd.read_csv('./book1.csv')
merged['score'] = merged.apply(lambda x: (x['score'] +1)/2, axis=1)
merged['col3'] = merged['scores'] * merged['magnitude']
merged['l'] = ydata

merged = merged.iloc[:,2:]

yy = np.array(ydata)
datac = np.mat(merged)
#datac = np.concatenate((merged, yy),axis =1)
random.shuffle(datac)


xc_t = datac[:,:-1][:1200,]
xc_v = datac[:,:-1][1201:,]
yt = [x[0] for x in datac[:,-1][:1200,].astype(np.int32).tolist()]
yv = [x[0] for x in datac[:,-1][1201:,].astype(np.int32).tolist()]


pipeline_optimizer.fit(xc_t, yt)

print(pipeline_optimizer.score(xc_v, yv))

pipeline_optimizer.export('./tpot_exported_pipeline.py')



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=6, max_features=0.9000000000000001, min_samples_leaf=1, min_samples_split=13, n_estimators=100, subsample=0.6000000000000001)
)
    
    
training_features, training_target = xc_t, yt

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(xc_v)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yv, results)

#np.set_printoptions(precision=2)
#cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#print(cm_normalized)

from plot_cm import plot_confusion_matrix
plot_confusion_matrix(cm,
                      normalize    = False,
                      target_names = ['po', 'n', 'ng'],
                      title        = "Confusion Matrix")




