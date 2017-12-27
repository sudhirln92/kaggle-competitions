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
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import datetime as dt
from xgboost import XGBClassifier 
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

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
    var = ['reg','trans','mem_exp','user_']
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

# Data analysis 
train.columns

train.info()

train.head().T
train.describe().T
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
# #Pickle
# =============================================================================
train.to_pickle('train_m1.pkl')
test.to_pickle('test_m1.pkl')
train = pd.read_pickle('train_m1.pkl')
test = pd.read_pickle('test_m1.pkl')

# =============================================================================
# Mean and Median range
# =============================================================================
def mean_median(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if df[c].nunique()>2]
    #dcol = dcol.drop('is_churn')
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
#  Encoder
# =============================================================================
le = LabelEncoder()
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])

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
# Co relation
# =============================================================================

cor = train.corr()
plt.figure(figsize=(16,12))
sns.heatmap(cor,cmap='binary',annot=False)
plt.xticks(rotation=45)
plt.show()

# =============================================================================
# #TF IDF
# =============================================================================
tf_idf = TfidfVectorizer(max_features=5,use_idf=True,lowercase=False,stop_words=None,analyzer='char',
                        ngram_range=(1, 3))

X_tf = tf_idf.fit_transform(train['msno']).toarray()
X_test_tf = tf_idf.fit_transform(test['msno']).toarray()

train1 = pd.concat([train1,pd.DataFrame(X_tf)],axis=1)
test1 = pd.concat([test1,pd.DataFrame(X_test_tf)],axis=1)

# =============================================================================
# # Split data set
# =============================================================================

unwanted = ['msno','is_churn','registration_init_time','transaction_date','membership_expire_date','date']

X = train1.drop(unwanted,axis=1)
y = train1['is_churn'].astype('category')
x_test = test1.drop(unwanted,axis=1)


# =============================================================================
#  ## Hyper parameter tuning
# =============================================================================

log_reg = LogisticRegression(class_weight='balanced')
param = {'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,3]}
rs_cv = RandomizedSearchCV(estimator=log_reg,param_distributions=param,random_state=seed)
rs_cv.fit(X,y)
print('Best parameter :{} Best score :{}'.format(rs_cv.best_params_,rs_cv.best_score_))


# =============================================================================
# 
# =============================================================================
cat_model = CatBoostClassifier(iterations=1000,
                               depth=7,
                               learning_rate=0.1,
                               loss_function='Logloss',
                              )

cat_model.fit(X,y)

y_pred = cat_model.predict_proba(x_test)[:,1]
y_pred

# =============================================================================
# # # XGBoost
# =============================================================================
def runXGB(xtrain,xvalid,ytrain,yvalid,xtest,eta=0.1,num_rounds=100):
    params = {
        'objective':'binary:logistic',        
        'max_depth':7,
        'learning_rate':eta,
        'eval_metric':'logloss',
        'min_child_weight':5,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'seed':seed,
        #'reg_lambda':0.1,
        #'reg_alpha':0.1,
        #'scale_pos_weight':1,
        'n_thread':-1
    }
    
    dtrain = xgb.DMatrix(xtrain,label=ytrain)
    dvalid = xgb.DMatrix(xvalid,label=yvalid)
    dtest = xgb.DMatrix(xtest)
    watchlist = [(dtrain,'train'),(dvalid,'test')]
    
    model = xgb.train(params,dtrain,num_rounds,watchlist,early_stopping_rounds=20,verbose_eval=30)
    pred = model.predict(dvalid,ntree_limit=model.best_ntree_limit)
    pred_test = model.predict(dtest,ntree_limit=model.best_ntree_limit)
    return pred_test,model

# =============================================================================
# KFold
# =============================================================================
kf = StratifiedKFold(n_splits=3,random_state=seed)
pred_test_full =0
cv_score = []
i=1
for train_index,test_index in kf.split(X,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y.loc[train_index],y.loc[test_index]
        
    pred_test,xg_model = runXGB(xtr,xvl,ytr,yvl,x_test,num_rounds=1000,eta=0.3)    
    pred_test_full += pred_test
    cv_score.append(xg_model.best_score)
    i+=1

# =============================================================================
# Model validation
# =============================================================================
print(cv_score)
print('Mean cv score',np.mean(cv_score))


fig,ax = plt.subplots(figsize=(14,8))
xgb.plot_importance(xg_model,ax=ax,height=0.8,color='r')
#plt.tight_layout()
plt.show()

# =============================================================================
# Predict for unseen data set
# =============================================================================

y_pred = pred_test_full/3
submit = pd.DataFrame({'msno':test['msno'],'is_churn':y_pred})
#submit.to_csv('kk_pred.csv',index=False)
submit.to_csv('kk_pred.csv.gz',index=False,compression='gzip')


# # Thank you for visiting
