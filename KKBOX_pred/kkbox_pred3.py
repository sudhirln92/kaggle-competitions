#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:54:25 2017

@author: sudhir
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve,log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import datetime as dt
from xgboost import XGBClassifier 
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb

%matplotlib inline
seed = 129

# =============================================================================
# Import Dataset
# =============================================================================
#path = '../input/'
path = ''
train = pd.read_csv(path+'train_v2.csv',dtype={'is_churn':np.int8})
test = pd.read_csv(path+'sample_submission_v2.csv',dtype={'is_churn':np.int8})
members = pd.read_csv(path+'members_v3.csv',parse_dates=['registration_init_time'],dtype={'city':np.int8,'bd':np.int8,
                                                                                         'registered_via':np.int8})
transactions = pd.read_csv(path+'transactions_v2.csv',parse_dates=['transaction_date','membership_expire_date'],
                          dtype={'payment_method_id':np.int8,'payment_plan_days':np.int8,'plan_list_price':np.int8,
                                'actual_amount_paid':np.int8,'is_auto_renew':np.int8,'is_cancel':np.int8})

user_log = pd.read_csv(path+'user_logs_v2.csv',parse_dates=['date'],dtype={'num_25':np.int16,'num_50':np.int16,
                                    'num_75':np.int16,'num_985':np.int16,'num_100':np.int16,'num_unq':np.int16})
# =============================================================================
# Cagnge datatype
# =============================================================================

def dtype_compression(df):
    col = list(df.select_dtype(include=['int']).columns)    
    for c in col:
        if (np.max(df[c]) < 2**8//2) & (np.min(df[c]) >= -2**8//2): 
            df[c] = df[c].astype(np.int8)
        elif (np.max(df[c]) < 2**16//2) & (np.min(df[c]) >=- 2**16//2):
            df[c] = df[c].astype(np.int16)
        elif (np.max(df[c]) < 2**32//2) & (np.max(df[c]) >= -2**32//2):
            df[c] = df[c].astype(np.int32)
        else:
            df[c] = df[c].astype(np.int64)
        
# =============================================================================
# Explore data set
# =============================================================================

print('Number of rows  & columns',train.shape)
train.head()

print('Number of rows  & columns',test.shape)
test.head()

print('Number of rows  & columns',members.shape)
members.head()

print('Number of rows & columns',transactions.shape)
transactions.head()

print('Number of rows & columns',user_log.shape)
user_log.head()


# =============================================================================
# Merge data set
# =============================================================================

train = pd.merge(train,members,on='msno',how='left')
test = pd.merge(test,members,on='msno',how='left')
train = pd.merge(train,transactions,how='left',on='msno',left_index=True, right_index=True)
test = pd.merge(test,transactions,how='left',on='msno',left_index=True, right_index=True,)
train = pd.merge(train,user_log,how='left',on='msno',left_index=True, right_index=True)
test = pd.merge(test,user_log,how='left',on='msno',left_index=True, right_index=True)

del members,transactions,user_log
print('Number of rows & columns',train.shape)
print('Number of rows & columns',test.shape)

# =============================================================================
# #Pickle
# =============================================================================
train.to_pickle('train_m.pkl')
test.to_pickle('test_m.pkl')
train = pd.read_pickle('train_m.pkl')
test = pd.read_pickle('test_m.pkl')

# =============================================================================
# Date feature
# =============================================================================

train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].describe()
train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].isnull().sum()

train['registration_init_time'] = train['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))
test['registration_init_time'] = test['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))


def date_feature(df):
    
    col = ['registration_init_time' ,'transaction_date','membership_expire_date','date']
    var = ['reg','trans','mem_exp','user']
    #print(col)
    for i, c in enumerate(col):
        
        df[var[i]+'_day'] = df[c].dt.day.astype('uint8')
        df[var[i]+'_weekday'] = df[c].dt.weekday.astype('uint8')        
        df[var[i]+'_month'] = df[c].dt.month.astype('uint8') 
        df[var[i]+'_year'] =df[c].dt.year.astype('uint16') 
        for j in range(i+1,len(col)):
            df[var[i]+'_dif_'+var[j]] = (df[col[i]]-df[col[j]]).dt.days

date_feature(train)
date_feature(test)

# =============================================================================
# # Data analysis 
# =============================================================================
train.columns

train.info()

train.head().T
train.describe().T

# =============================================================================
# Co relation
# =============================================================================

cor = train.corr()
plt.figure(figsize=(12,12))
sns.heatmap(cor,cmap='Set1',annot=False)
plt.xticks(rotation=45)
plt.show()

drop_col = ['user_month','user_year']
train = train.drop(drop_col,axis=1)
test = test.drop(drop_col,axis=1)
# =============================================================================
# Missing value
# =============================================================================
train.isnull().sum()

col = [ 'city', 'bd', 'gender', 'registered_via']
def missing(df,columns):
    col = columns
    for i in col:
        df[i].fillna(df[i].mode()[0],inplace=True)

missing(train,col)
missing(test,col)

def unique_value(df):
    col = df.columns
    for i in col:
        print('Number of unique value in {} is {}'.format(i,df[i].nunique()))

unique_value(train)

# =============================================================================
#  Encoder
# =============================================================================
le = LabelEncoder()
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])

# =============================================================================
# Mean and Median range
# =============================================================================
def mean_median(df):
    df = pd.DataFrame(df)
    unwanted = ['msno','is_churn','registration_init_time','transaction_date','membership_expire_date','date']
    
    dcol = [c for c in df.columns if df[c].nunique()>2]
    dcol = [c for c in df.columns if c not in unwanted]
    d_medain = df[dcol].median(axis=0)
    d_mean = df[dcol].mean(axis=0)
    
    #Add mean and median to data set having more than 2 category
    for c in dcol:
        df[c+'_median_range'] = (df[c].values > d_medain[c]).astype(np.int8)
        df[c+'_mean_range'] = (df[c].values > d_mean[c]).astype(np.int8)
    return df

train = mean_median(train)
test = mean_median(test)

# =============================================================================
# #frequency encoding
# =============================================================================
def freq_encoding(cols, train_df, test_df):
    result_traindf = pd.DataFrame()
    result_testdf = pd.DataFrame()
    
    for col in cols:
        print(" ",col)
        col_freq = col+'_freq'
        freq = train_df[col].value_counts()
        freq = pd.DataFrame(freq)
        freq.reset_index(inplace=True)
        freq.columns = [[col,col_freq]]
        
        # merge this 'freq' dataframe with train
        temp_train_df = pd.merge(train_df[[col]],freq,how='left',on=col)
        temp_train_df.drop([col],axis=1,inplace=True)
        
        # merge this 'freq' data frame with test data 
        temp_test_df = pd.merge(test_df[[col]],freq, how='left', on=col)
        temp_test_df.drop([col], axis=1, inplace=True)
        
        #if certain level of freq is not observed in test dataset will assign 0
        temp_test_df.fillna(0,inplace=True)
        temp_test_df[col_freq] = temp_test_df[col_freq].astype(np.int32)
        
        if result_traindf.shape[0] ==0:
            result_traindf = temp_train_df
            result_testdf = temp_test_df
        else:
            result_traindf = pd.concat([result_traindf,temp_train_df], axis=1)
            result_testdf = pd.concat([result_testdf,temp_test_df], axis=1)
        
    return result_traindf,result_testdf


freq_col = [c for c in train.columns if (2<train[c].nunique()<=586)]
train_freq,test_freq = freq_encoding(freq_col,train,test)

train = pd.concat([train,train_freq],axis=1)
test = pd.concat([test,test_freq],axis=1)
del train_freq,test_freq
# =============================================================================
#  # One Hot Encoding
# =============================================================================
def OHE(df1,df2,col):
    
    df = pd.concat([df1,df2],ignore_index=True)
    c1,c2=[],{}
    print('Number of categorical features',len(col))
    for c in col:
        if df[c].nunique()>2:                                
            c1.append(c)
            c2[c] = 'ohe_'+c
    df = pd.get_dummies(df,columns=c1,prefix=c2,drop_first=True)
    df1 = df.loc[:df1.shape[0]-1]
    df2 = df.loc[df1.shape[0]:]
    print('Train',df1.shape)
    print('Test',df2.shape)
    return df1,df2

col = ['city','gender','registered_via']
train1,test1 = OHE(train,test,col)

# =============================================================================
# #Pickle
# =============================================================================
train1.to_pickle('train_m1.pkl')
test1.to_pickle('test_m1.pkl')
train1 = pd.read_pickle('train_m1.pkl')
test1 = pd.read_pickle('test_m1.pkl')

# =============================================================================
# # Split data set
# =============================================================================

unwanted = ['msno','is_churn','registration_init_time','transaction_date','membership_expire_date','date']

X = train1.drop(unwanted,axis=1)
y = train1['is_churn'].astype('category')
x_test = test1.drop(unwanted,axis=1)

# =============================================================================
# #Ensemble
# =============================================================================
class Create_ensemble(object):
    def __init__(self, n_splits, base_models):
        self.n_splits = n_splits
        self.base_models = base_models
    
    def predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
    
        fold = list(StratifiedKFold(n_splits=self.n_splits,random_state=seed,shuffle=True).split(X,y))

        S_train = np.zeros((X.shape[0],len(self.base_models)))
        S_test = np.zeros((T.shape[0],len(self.base_models)))

        for i,clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0],self.n_splits))

            for j, (train_index,valid_index) in enumerate(fold):
                Xtrain, Xvalid = X[train_index], X[valid_index]
                ytrain, yvalid = y[train_index], y[valid_index]

                clf.fit(Xtrain, ytrain)
                valid_pred = clf.predict_proba(Xvalid)[:,1]            
                S_train[valid_index,i] = valid_pred
                S_test_i[:,j] = clf.predict_proba(T)[:,1]

            print('\n Training Log loss for model {} : {}'.format(i,log_loss(y,S_train[:,i])))
            S_test[:,i] = S_test_i.mean(axis=1)

        return S_train, S_test


#parameters
lg_params ={}
lg_params['learning_rate'] = 0.002
lg_params['max_depth'] = -1
lg_params['subsample'] = 0.8
lg_params['colsample_bytree'] =0.8
lg_params['n_estimators'] = 1000
lg_params['max_bin'] = 252
lg_params['min_child_weight'] = 512
lg_params['silent'] = False
lg_params['metric'] = 'binary_logloss'
lg_params['num_leaves'] = 100

lg_params1 ={}
lg_params1['learning_rate'] = 0.03
lg_params1['max_depth'] = -1
lg_params1['subsample'] = 0.9
lg_params1['colsample_bytree'] =0.9
lg_params1['n_estimators'] = 500
lg_params1['max_bin'] = 252
lg_params1['min_child_weight'] = 252
lg_params1['silent'] = False
lg_params1['metric'] = 'binary_logloss'
lg_params1['num_leaves'] = 50

xg_params = {}
xg_params['learning_rate'] = 0.3


model_lg = LGBMClassifier(**lg_params)
model_lg1 = LGBMClassifier(**lg_params1)
lgb_stack = Create_ensemble(n_splits=3,base_models=[model_lg,model_lg1])

lg_train_prd,lg_test_prd = lgb_stack.predict(X,y,x_test)

model_lg1.fit(X,y)
lg_pred = model_lg1.predict_proba(lg_test)

# =============================================================================
#  ## Hyper parameter tuning
# =============================================================================

#log_reg = LogisticRegression(class_weight='balanced')
param = {'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,3]}
rs_cv = RandomizedSearchCV(estimator=log_reg,param_distributions=param,random_state=seed)
rs_cv.fit(X,y)
print('Best parameter :{} Best score :{}'.format(rs_cv.best_params_,rs_cv.best_score_))

lg_params['mertic'] = 'binary_logloss'
lg_params['num_leaves'] = 200

# =============================================================================
# 
# =============================================================================
cat_model = CatBoostClassifier(iterations=500,
                               depth=7,
                               learning_rate=0.1,
                               loss_function='Logloss',
                              )

cat_model.fit(lg_train_prd,y)
y_pred = cat_model.predict_proba(lg_test_prd)[:,1]

# =============================================================================
# Predict for unseen data set
# =============================================================================

#y_pred = pred_test_full/3
submit = pd.DataFrame({'msno':test['msno'],'is_churn':y_pred})
#submit.to_csv('kk_pred.csv',index=False)
submit.to_csv('kk_pred.csv.gz',index=False,compression='gzip')


# # Thank you for visiting