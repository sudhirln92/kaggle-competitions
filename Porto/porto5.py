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
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
import lightgbm as lgb
from sklearn.utils import resample
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
sns.heatmap(cor,square =True,cmap='coolwarm',)

""" ps_calc value as 0 relation with remaining varialble"""
ps_cal = train.columns[train.columns.str.startswith('ps_calc')] 
train = train.drop(ps_cal,axis =1)
test = test.drop(ps_cal,axis=1)

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
category_type(test)

# Up sample
train['target'].value_counts()
df_majority = train[train['target'] == 0]
df_minority = train[train['target'] == 1]

df_minority_upsampled = resample(df_minority,replace=True,n_samples=573518,
                                 random_state =seed)
train_reup = pd.concat([df_majority,df_minority_upsampled])
train_reup['target'].value_counts()
# One hot encoding
def OHE(df):
    cat_col = [col for col in df.columns if '_cat' in col]
    bin_col = [col for col in df.columns if '_bin' in col]
    
    c2,c3 = [],{}
    print('Binary feature',len(bin_col))
    for c in  bin_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
            print(c3,c2)
    print('Categorical feature',len(cat_col))
    for c in cat_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)

    df = df.drop(bin_col,axis=1)
    #df = df.drop(cat_col,axis=1)
    print(df.shape)
    return df

train1 = OHE(train_reup)
test1 = OHE(test)

"""
def category_type(df):
    col = df.columns
    for i in col:
        if df[i].nunique()<=104:
            df[i] = df[i].astype('category')
category_type(train)
category_type(test)

"""

#split data set
X = train.drop(['target','id'],axis=1)
y = train['target']
x_test = test.drop('id',axis=1)
xtr,xvl,ytr,yvl = train_test_split(X,y,test_size=0.3,random_state=seed)

#Modelling
lr = LogisticRegression(max_iter=1000,verbose = 1,)
lr.fit(xtr,ytr)

y_lr = lr.predict(xvl)

confusion_matrix(yvl,y_lr)
accuracy_score(yvl,y_lr)
roc_auc_score(yvl,lr.predict_proba(xvl)[:,1])
#roc_curve(yvl,lr.predict_proba(xvl)[:,1]).plot()
#RF
rf = RandomForestClassifier(n_estimators=500,max_depth=10,max_leaf_nodes=100,verbose=0)
rf.fit(xtr,ytr)
y_rf = rf.predict(xvl)

confusion_matrix(yvl,y_rf)
accuracy_score(yvl,y_rf)
roc_auc_score(yvl,rf.predict_proba(xvl)[:,1])

# Light GBM model
def runLGB(train,valid,y_train,y_valid,test,eta=0.5,num_rounds=10,early_stopping_rounds=50,sample=0.7,
           col_sample=0.8,max_depth=7):
    
    param = {
            'objective':'binary',
            'boosting':'gbdt',
            'learning_rate':eta,
            #'metric':'gini',
            'metric':'auc',
            'bagging_fraction':sample,
            'bagging_freq':5,
            'bagging_seed':seed,
            'num_leaves':120,
            'feature_fraction':col_sample,
            'verbose':10,            
            'min_child_weight':10,
            'max_depth':max_depth,
            'nthread':-1           
            }
   
    lgtrain = lgb.Dataset(train,label=y_train)
    lgvalid = lgb.Dataset(valid,label=y_valid)
    
    model=lgb.train(param,lgtrain,num_rounds,valid_sets=lgvalid,
              early_stopping_rounds=early_stopping_rounds)
    #lg_pred = model.predict(xvl,num_iteration=model.best_iteration)
    pred = model.predict(test,num_iteration=model.best_iteration)
    
    return pred,model

#Kfold
kf = StratifiedKFold(n_splits =5,random_state=seed,shuffle=True)
pred_test_full=0
for train_index,test_index in kf.split(X,y):
    xtr,xvl = X.loc[train_index], X.loc[test_index]
    ytr,yvl = y.loc[train_index], y.loc[test_index]
    
    pred_test,model = runLGB(xtr,xvl,ytr,yvl,x_test,eta=0.01,num_rounds=1000,max_depth=10)
    pred_test_full +=pred_test

#Predict for unsen data set

y_pred = rf.predict_proba(x_test)
submit = pd.DataFrame({'id':test['id'],'target':y_pred[:,1]})

y_pred_lgb = pred_test_full/5
y_pred = (y_pred_lgb+y_pred_xgb) /2
submit = pd.DataFrame({'id':test['id'],'target':y_pred})
submit.to_csv('lr_porto.csv.gz',index=False,compression='gzip') 

