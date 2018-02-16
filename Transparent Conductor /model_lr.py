#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 08:03:00 2018

@author: sudhir
"""
# =============================================================================
# Load packages
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
import gc

%matplotlib inline
seed=2300 

# Feature models
from feature import feature_engineering,OHE

# =============================================================================
# Read data set
# =============================================================================
path = 'file/'
#path = '../input/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
print('Number of rows and columns in train data set:',train.shape)
print('Number of rows and columns in test data  set:',test.shape)

# =============================================================================
# Features
# =============================================================================
train1 = feature_engineering(train)
test1 = feature_engineering(test)

col = ['spacegroup','number_of_total_atoms']
train1,test1 = OHE(train1,test1,col)
gc.collect()

def rmsle(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5
# =============================================================================
# MOdel
# =============================================================================
col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train1.drop(['id']+col,axis=1)
y = train1[col]
x_test = test1.drop(['id']+col,axis=1)

kf = KFold(n_splits=5,random_state=seed,shuffle=True)
cv_score =[]
pred_test_full_1 = np.zeros((x_test.shape[0],kf.n_splits))
pred_test_full_2 = np.zeros((x_test.shape[0],kf.n_splits))
lr = LinearRegression()

#
for i, (train_index, valid_index) in enumerate(kf.split(X)):
    print('{} of Kfold {}'.format(i+1,kf.n_splits))
    xtrain, xvalid = X.loc[train_index], X.loc[valid_index]
    ytrain, yvalid = y.loc[train_index], y.loc[valid_index]
    
    ##Building model for ',col[0]
    lr.fit(xtrain,ytrain[col[0]])
    pred_test_full_1[:,i] = lr.predict(x_test)
    score = lr.score(xvalid,yvalid[col[0]])
    cv_score.append(score)
    print('R square for {} is {''} :'.format(col[0],score))
    
    ##Building model for ',col[1]
    lr.fit(xtrain,ytrain[col[1]])
    pred_test_full_2[:,i] = lr.predict(x_test)
    score = lr.score(xvalid,yvalid[col[1]])
    print('R square for {} is {}:'.format(col[1],score))
    cv_score.append(score)

print(cv_score)
np.mean(cv_score)

k = lr.predict(xvalid)
log_loss(yvalid,k)
# =============================================================================
# Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100,random_state=1)
rf.fit(X,y.iloc[:,0])
l = rf.predict(x_test)

k['0'] = pd.DataFrame(l) 
y_pred = k
y_pred = np.array(y_pred)
l = np.array(y.iloc[:,0])
log_loss(l,k)



# =============================================================================
# Ensemble
# =============================================================================
class create_ensemble(object):
    def __init__(self,n_splits,base_models):
        self.n_splits = n_splits
        self.base_models = base_models
    def predict(self,X,y,T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        
        fold = list(KFold(n_splits = self.n_splits,random_state = seed,shuffle = True).split(X))
        
        S_train = np.zeros((X.shape[0],len(self.base_models)))
        S_test = np.zeros((T.shape[0],len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0],self.n_splits))
            
            for j , (train_index,valid_index) in enumerate(fold):
                Xtrain,Xvalid = X[train_index], X[valid_index]
                ytrain,yvalid = y[train_index], y[valid_index]
                
                clf.fit(Xtrain,ytrain)
                valid_pred = clf.predict(Xvalid)
                S_train[valid_index,i] = valid_pred
                S_test_i[:,j] = clf.predict(T)
            print('\n Training rmsle for model {} : {}'.format(i,rmsle(y,S_train[:,i])))
            S_test[:,i] = S_test_i.mean(axis=1) 
        return S_train,S_test

# =============================================================================
# Ligth gbm
# =============================================================================
import lightgbm as lgb

lg_param = {}
lg_param['learning_rate'] = 0.01
lg_param['n_estimators'] = 500
lg_param['max_depth'] = 4
lg_param['num_leaves'] = 255
lg_param['boosting_type'] = 'gbdt'
lg_param['subsample'] = 0.95
lg_param['colsample_bytree'] = 0.9
lg_param['reg_alpha'] = 0.1
lg_param['reg_lambda'] = 0.1
lg_param['metric'] = ['mse']


lg_param1 = {}
lg_param1['learning_rate'] = 0.1
lg_param1['n_estimators'] = 1000
lg_param1['max_depth'] = 4
lg_param1['num_leaves'] = 255
lg_param1['boosting_type'] = 'gbdt'
lg_param1['subsample'] = 0.95
lg_param1['colsample_bytree'] = 0.9
lg_param1['reg_alpha'] = 0.1
lg_param1['reg_lambda'] = 0.1
lg_param1['metric'] = ['mse']

pred_lg = np.zeros((x_test.shape[0],len(col)))
dtrain = lgb.Dataset(X, label=y[col[0]])
dtest = lgb.Dataset(x_test)
lg_model = lgb.LGBMRegressor(**lg_param)
lg_model1 = lgb.LGBMRegressor(**lg_param1)

gs_param = {'learning_rate':[0.01,0.03,0.001,0.003,],
            'num_leaves':[12,120,200,255,320],
            'subsample':[0.5,0.6,0.7,0.8,0.85,0.9],
            'n_estimators':[50,100,150,200,300,500,600,1000],
            'num_leaves':[60,120,255,512]
            }

cv_regressor = GridSearchCV(lg_model,gs_param,cv= 2,verbose=10)
cv_regressor.fit(X,y)
cv_regressor.best_estimator_

lg_model.fit(dtrain,y,eval_metric='log_loss',)
lg_model = lgb.train(lg_param,dtrain,num_boost_round=2000,verbose_eval=True,)
                     #early_stopping_rounds=100,)


yred = lg_model.predict(x_test)
pred_lg[:,0] = yred
y_pred = pred_lg

stack = create_ensemble(n_splits=5, base_models= [lg_model, lg_model1])
col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train1.drop(['id']+col,axis=1)
y = train1[col[0]]
T = test1.drop(['id']+col,axis=1)
lgb_train, lgb_train = stack.predict(X,y,T)

# =============================================================================
# submission
# =============================================================================
y_pred = np.zeros((x_test.shape[0],len(col)))
y_pred[:,0],y_pred[:,1] = pred_test_full_1.mean(axis=1), pred_test_full_2.mean(axis=1)
y_pred[y_pred <= 0] = 1e-5
submit = pd.DataFrame({'id':test['id'],'formation_energy_ev_natom':y_pred[:,0],'bandgap_energy_ev':y_pred[:,1]})
submit.to_csv('lr_conductor.csv',index=False)

lgb.plot_importance(lg_model)
