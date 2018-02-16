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

from sklearn.linear_model import LinearRegression,Ridge 
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel,VarianceThreshold
import gc

#%matplotlib
seed=42

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cgb

# Feature models
from feature import feature_engineering,OHE

from geomertic_xyz import xyz
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

train1,test1 = xyz(train,test,path,seed)

train1 = feature_engineering(train1)
test1 = feature_engineering(test1)

col = ['spacegroup','number_of_total_atoms']
train1,test1 = OHE(train1,test1,col)
gc.collect()


vector = np.vstack((train1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values,
          test1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values))                                              

pca = PCA().fit(vector)
train1['vector_pca0'] = pca.transform(train1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values)[:,0]
test1['vector_pca0'] = pca.transform(test1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values)[:,0]
train1['vector_pca1'] = pca.transform(train1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values)[:1,]
test1['vector_pca1'] = pca.transform(test1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values)[:,1]
train1['vector_pca1'] = pca.transform(train1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values)[:,2]
test1['vector_pca1'] = pca.transform(test1[['lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang']].values)[:,2]



def rmsle(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

# =============================================================================
# fature selection 
# =============================================================================

col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train1.drop(['id']+col,axis=1)
y = np.log(train1[col[0]]+1)
T = test1.drop(['id']+col,axis=1)

selector = VarianceThreshold(threshold=.01)
selector.fit(X) # Fit to train without id and target variables

f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements

v = X.columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))
selected_feat = X.columns.drop(v)

#update 
X = X[selected_feat]
y = np.log(train1[col[0]]+1) 
T = T[selected_feat]


gc.collect()
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
gc.collect()
# =============================================================================
# Ligth gbm
# =============================================================================

lg_param = {}
lg_param['learning_rate'] = 0.01
lg_param['n_estimators'] = 800
lg_param['max_depth'] = 5
lg_param['num_leaves'] = 2**5
lg_param['boosting_type'] = 'gbdt'
lg_param['feature_fraction'] = 0.85
lg_param['bagging_fraction'] = 0.8
lg_param['min_child_samples'] = 30
lg_param['lambda_l1'] = 0.1
lg_param['lambda_l2'] = 0.3
lg_param['metric'] = ['rmse']
lg_param['seed'] = seed

lg_param1 = {}
lg_param1['learning_rate'] = 0.03
lg_param1['n_estimators'] = 600
lg_param1['max_depth'] = 5
lg_param1['num_leaves'] = 2**5
lg_param1['boosting_type'] = 'gbdt'
lg_param1['feature_fraction'] = 0.9
lg_param1['bagging_fraction'] = 0.6
lg_param1['min_child_samples'] = 100
lg_param1['lambda_l1'] = 0.1
lg_param1['lambda_l2'] = 0.5
lg_param1['metric'] = ['rmse']
lg_param1['seed'] = seed


lg_model = lgb.LGBMRegressor(**lg_param)
lg_model1 = lgb.LGBMRegressor(**lg_param1)


"""gs_param = {#'learning_rate':[0.01,0.03,0.001,0.003,0.005],
            #'max_depth':[4,5,7,9,10,12],
            #'num_leaves':[4,10,20,30,50,70,80,200,500,],
            #'feature_fraction':[0.7,0.8,0.85,0.9],
            #'n_estimators':[400,500,600,800,1000,1200],
            #'min_child_samples':[20,30,50,80,100,200]
            #'bagging_fraction':[0.5,0.6,0.7,0.8,0.85,0.9],
            #'lambda_l1':[0.1,0.3,0.5,0.8,1,1.5],
            #'lambda_l2':[0.1,0.3,0.5,0.8,1,1.5],
            
            }

cv_regressor = GridSearchCV(lg_model,gs_param, cv=3, verbose=1)
cv_regressor.fit(X,y)
cv_regressor.best_params_"""
gc.collect()

# =============================================================================
# #XGB
# =============================================================================
xg_param = {}
xg_param['max_depth'] = 4
xg_param['learning_rate'] = 0.03
xg_param['n_estimators'] = 800
xg_param['subsample'] = 0.9
xg_param['colsample_bytree'] = 0.7
xg_param['reg_alpha'] = 0.5
xg_param['reg_lambda'] = 0.1
xg_param['min_child_weight'] = 20
xg_param['seed'] = seed

xg_param1 = {} # for col[1]
xg_param1['max_depth'] = 4
xg_param1['learning_rate'] = 0.003
xg_param1['n_estimators'] = 1500
xg_param1['subsample'] = 0.8
xg_param1['colsample_bytree'] = 0.95
xg_param1['reg_alpha'] = 0.1
xg_param1['reg_lambda'] = 0.1
xg_param1['min_child_weight'] = 20
xg_param1['seed'] = seed

xg_model = xgb.XGBRegressor(**xg_param)
xg_model1 = xgb.XGBRegressor(**xg_param1)

"""
gs_param = {'learning_rate':[0.01,0.03,0.001,0.003,0.005],
            #'max_depth':[4,5,7,9,10],
            #'subsample':[0.7,0.8,0.85,0.9],
            #'n_estimators':[500,800,1000,1200,1500],
            #'min_child_weight':[1,5,20,30,50,80,100,200],
            #'colsample_bytree':[0.5,0.6,0.7,0.8,0.85,0.9],
            #'reg_alpha':[0.1,0.3,0.5,0.8,1,1.5],
            #'reg_lambda':[0.1,0.3,0.5,0.8,1,1.5],
            
            }

cv_regressor = GridSearchCV(xg_model,gs_param,cv= 5,verbose=10)
cv_regressor.fit(X,y)
cv_regressor.best_params_"""
gc.collect()


# =============================================================================
# # Catboost
# =============================================================================

cat_param = {
        'iterations':2000,
        'depth':4,
        'learning_rate':0.03,
        'loss_function':'RMSE',
        'eval_metric':'RMSE',
        'od_type':'Iter', 
        'random_seed':seed,
        'logging_level':'Silent'
        }
cat_param1 = {
        'iterations':2000,
        'depth':4,
        'learning_rate':0.03,
        'loss_function':'RMSE',
        'eval_metric':'RMSE',
        'od_type':'Iter', 
        'random_seed':seed,
        'logging_level':'Silent',
        }

cat_model = cgb.CatBoostRegressor(**cat_param)
cat_model1 = cgb.CatBoostRegressor(**cat_param1)

"""col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train1.drop(['id']+col,axis=1)
y = np.log(train1[col[0]]+1)
T = test1.drop(['id']+col,axis=1)
gs_param = {'learning_rate':[0.01,0.03,0.001,0.003,],
            'depth':[4,5,7,9,10],
            #'subsample':[0.7,0.8,0.85,0.9],
            #'n_estimators':[500,800,1000,1200,1500],
            #'min_child_weight':[1,5,20,30,50,80,100,200],
            #'colsample_bytree':[0.5,0.6,0.7,0.8,0.85,0.9],
            #'reg_alpha':[0.1,0.3,0.5,0.8,1,1.5],
            #'reg_lambda':[0.1,0.3,0.5,0.8,1,1.5],
            
            }

cv_regressor = GridSearchCV(cat_model,gs_param,cv= 5,verbose=10)
cv_regressor.fit(X,y)
cv_regressor.best_params_"""


# =============================================================================
# Stacking 
# =============================================================================

# col0 fomation ev
print('Stacking for column:',col[0])
base_models = [lg_model,xg_model,cat_model]
stack = create_ensemble(n_splits=5, base_models= base_models)

stack_train, stack_test = stack.predict(X,y,T)

#col1 bandgap ev
print('Stacking for column:',col[1])
y = np.log(train1[col[1]]+1)
base_models = [lg_model1,xg_model1,cat_model1]
stack = create_ensemble(n_splits=5, base_models= base_models)
stack_train_1, stack_test_1 = stack.predict(X,y,T)

# Model co relation
pd.DataFrame(stack_test_1).corr()
gc.collect()

lgb.plot_importance(lg_model,max_num_features=30)
# =============================================================================
# ## stage2 lr
# =============================================================================

lr = LinearRegression()
y = np.log(train1[col[0]]+1)
lr.fit(stack_train,y)
pred_test_full_1 = lr.predict(stack_test)

y = np.log(train1[col[1]]+1)
lr.fit(stack_train_1,y)
pred_test_full_2 = lr.predict(stack_test_1)

y_pred = np.zeros((test.shape[0],len(col)))
y_pred[:,0],y_pred[:,1] = np.exp(pred_test_full_1)-1, np.exp(pred_test_full_2)-1

# =============================================================================
# submission
# =============================================================================

submit = pd.DataFrame({'id':test['id'],'formation_energy_ev_natom':y_pred[:,0],'bandgap_energy_ev':y_pred[:,1]})
submit.to_csv('stack_conductor.csv',index=False)

