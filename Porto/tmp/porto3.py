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
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,train_test_split
import lightgbm as lgb
import xgboost as xgb
seed =45

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

#CORELATION PLOT
cor = train.corr()
plt.figure(figsize=(12,10))
sns.heatmap(cor,square =True,)

""" ps_calc value as 0 relation with remaining varialble"""
ps_cal = train.columns[train.columns.str.startswith('ps_calc')] 
train1 = train.drop(ps_cal,axis =1)
test1 = test.drop(ps_cal,axis=1)

#Missing value is data set
train.isnull().sum() # No missing values
"""Values of -1 indicate that the feature was missing from the observation. 
The target columns signifies whether or not a claim was filed for that policy holder."""

k= pd.DataFrame()
k['train']=train.isnull().sum()
k['test'] = test.isnull().sum()
k
def missing_value(df):
    col = df.columns
    for i in col:
        if df[i].isnull().sum()>0:
            df[i].fillna(df[i].mode()[0],inplace=True)

missing_value(train)
missing_value(test)
"""
def uniq(df):
    col = df.columns
    for i in col:
        print('\n Unique value of "{}" is "{}" '.format(i,df[i].nunique()))
        print(df[i].unique())
uniq(train)

def category_type(df):
    col = df.columns
    for i in col:
        if df[i].nunique()<=104:
            df[i] = df[i].astype('category')
category_type(train)
category_type(test) """

#split data set
X = train1.drop(['target','id'],axis=1)
y = train1['target']
x_test = test1.drop('id',axis=1)
#xtr,xvl,ytr,yvl = train_test_split(X,y,test_size=0.3,random_state=seed)

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score


#XGBOOST

def XGB(train,valid,y_train,y_valid,test,eta=0.1,num_rounds=100,max_depth=10,
        sample=0.7,col_sample=0.8,early_stop=20):
    
    param ={
            'objective':'binary:logistic',
            'eta':eta,
            'max_depth':max_depth,
            'metric':'gini',
            #'num_boost_rounds':num_rounds,
            'colsample_bytree':col_sample,
            'subsample':sample,
            'min_child_weight':10.0,
            'gamma':1,
            'nthread':-1
            }
    pslt = list(param.items())
    d_train = xgb.DMatrix(train,label=y_train)
    d_valid = xgb.DMatrix(valid,label=y_valid)
    d_test = xgb.DMatrix(test)
    watchlist = [(d_train,'train'),(d_valid,'valid')]
    model = xgb.train(pslt, d_train, num_rounds,watchlist,feval=gini_xgb,maximize=True,
                      early_stopping_rounds=early_stop,verbose_eval=10)
    #pred_valid = model.predict(y_valid,ntree_limit=model.best_ntree_limit)
    pred = model.predict(d_test,ntree_limit=model.best_ntree_limit)
    return pred, model
    
#Kfold
kf = KFold(n_splits =5,random_state=seed,shuffle=True)
pred_test_full=0
for train_index,test_index in kf.split(X):
    xtr,xvl = X.loc[train_index], X.loc[test_index]
    ytr,yvl = y.loc[train_index], y.loc[test_index]
    
    pred_test,model = XGB(xtr,xvl,ytr,yvl,x_test,eta=0.02,num_rounds=500,max_depth=10)
    pred_test_full +=pred_test
#Predict for unsen data set
y_pred = rf.predict_proba(x_test)
submit = pd.DataFrame({'id':test['id'],'target':y_pred[:,1]})

y_pred = pred_test_full/5
submit = pd.DataFrame({'id':test['id'],'target':y_pred})
submit.to_csv('lr_porto.csv.gz',index=False,compression='gzip') 