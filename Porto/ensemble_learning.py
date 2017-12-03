#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:21:29 2017

@author: sudhir
SOURCE: https://www.toptal.com/machine-learning/ensemble-methods-machine-learning
"""

#Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Read data set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

calc_col = [c for c in train.columns if 'calc' in c ]

X= train.drop(['id','target']+calc_col,axis=1)
y = train['target']
x_test = test.drop(['id']+calc_col,axis=1)

#VOTING AND AVERAING BASED ENSEMBLE METHOD
algorithms = [LogisticRegression(class_weight='balanced'),
              DecisionTreeClassifier(class_weight='balanced'),
              RandomForestClassifier(n_estimators=200,class_weight='balanced')]

predictions = pd.DataFrame()

for i, algorithm in enumerate(algorithms):
    predictions[i] = algorithm.fit(X,y).predict_proba(x_test)[:,1]
    
predictions.sum() 

final_prediction = predictions.mean(axis=0)

#Wieghted averaging
for row_number in range(0,len(predictions)-1):
    final_prediction.append(predictions[row_number]*[0.3,0.6,0.3])
    
#Stacking
base_algorithms = [LogisticRegression(class_weight='balanced'),
              DecisionTreeClassifier(class_weight='balanced'),
              RandomForestClassifier(n_estimators=200,class_weight='balanced')] 
stacking_train_dataset = pd.DataFrame()
stacking_test_dataset = pd.DataFrame()

for i,base_algo in enumerate(base_algorithms):
    stacking_train_dataset[i] = base_algo.fit(X,y).predict(X)
    stacking_test_dataset[i] = base_algo.predict(x_test)

stack_final_pred = LogisticRegression(class_weight='balanced').fit(stacking_train_dataset,y).predict_proba(stacking_test_dataset)[:,1]
    
#Predict for unseen data set
submit = pd.DataFrame({'id':test['id'],'target':stack_final_pred})
submit.to_csv('stack_porto.csv.gz',index=False,compression='gzip')
   