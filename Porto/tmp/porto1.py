#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:15:44 2017

@author: sudhir
"""

#Import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

#Import data set
train = pd.read_csv('train.csv',na_values=-1)
test = pd.read_csv('test.csv',na_values=-1)

# Explore data set
print('Number rows and columns:',train.shape)
print('Number rows and columns:',test.shape)

train.head(3).T

train.info()

#Traget varaiable
plt.figure(figsize=(10,8))
sns.countplot(train['target'],palette='rainbow')
plt.xlabel('Target')
#variable in imbalanced
train.isnull().sum() # No missing values
"""Values of -1 indicate that the feature was missing from the observation. 
The target columns signifies whether or not a claim was filed for that policy holder."""

sns.heatmap(train.isnull())

#split data set
X_train = train.drop('target',axis=1)
y_train = train['target']
xtr,xvl,ytr,yvl = train_test_split(X_train,y_train,test_size=0.3,random_state=192)

#Modelling
lr = LogisticRegression(max_iter=1000,verbose = 1,)
lr.fit(xtr,ytr)

y_lr = lr.predict(xvl)

confusion_matrix(yvl,y_lr)

#Predict for unsen data set
y_pred = lr.predict_proba(test)
submit = pd.DataFrame({'id':test['id'],'target':y_pred[:,0]})
submit.to_csv('lr_porto.csv',index=False) 
