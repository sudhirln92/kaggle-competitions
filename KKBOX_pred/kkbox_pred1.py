#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:58:16 2017

@author: sudhir
"""
#Import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import datetime as dt
seed = 129

#import Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('sample_submission_zero.csv')
members = pd.read_csv('members.csv',parse_dates=['registration_init_time','expiration_date'])
train.head()
test.head()
members.head()
members.info()

train = train.merge(members,on='msno',how='left')
test = test.merge(members,on='msno',how='left')

#Date feature 
def date_feature(df):
    
    col = ['registration_init_time','expiration_date']
    var = ['reg','exp']
    df['duration'] = (df[col[1]] - df[col[0]]).dt.days 
    
    for i ,j in zip(col,var):
        df[j+'_day'] = df[i].dt.day
        df[j+'_weekday'] = df[i].dt.weekday
        df[j +'_week'] = df[i].dt.week
        df[j+'_month'] = df[i].dt.month
        df[j+'_year'] =df[i].dt.year
       
date_feature(train)
date_feature(test)
train['registration_init_time'].dt.year


#Missing value
train.isnull().sum()
test.isnull().sum()

def missing(df):
    col = df.columns
    for i in col:
        df[i].fillna(df[i].mode()[0],inplace=True)

missing(train)
missing(test)

#train['registration_init_time'].fillna(0,inplace=True)
#test['registration_init_time'].fillna(0,inplace=True)

#Encoder
le = LabelEncoder()
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])

plt.figure(figsize=(8,6))
sns.countplot(train['is_churn'])

#imbalanced data set

#split data set
X = train.drop(['msno','is_churn','registration_init_time','expiration_date'],axis=1)
y = train['is_churn']
x_test = test.drop(['msno','is_churn','registration_init_time','expiration_date'],axis=1)

# model
kf = StratifiedKFold(n_splits=2,shuffle=True,random_state=seed)
pred_test_full =0
cv_score =[]

for train_index,test_index in kf.split(X,y):
    print('KFold',kf.n_splits)
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr =LogisticRegression()
    lr.fit(xtr,ytr)
    cv_score.append(roc_auc_score(yvl,lr.predict(xvl)))
    pred_test = lr.predict_proba(x_test)[:,1]
    pred_test_full +=pred_test
    
print(cv_score)
print('\nMean accuracy',np.mean(cv_score))
confusion_matrix(yvl,lr.predict(xvl))

# Predict data 
y_pred = pred_test_full/2
submit = pd.DataFrame({'msno':test['msno'],'is_churn':y_pred})
#submit.to_csv('kk_pred.csv',index=False)
submit.to_csv('kk_pred.csv.gz',index=False,compression='gzip')


