#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:38:47 2018

@author: sudhir
"""
# =============================================================================
# Load packages
# =============================================================================
import pandas as pd
import numpy as np
from datetime import timedelta,date
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import gc
# =============================================================================
# Read data set
# =============================================================================
dtypes = {'id':'int64','item_nbr':'int32','store_nbr':'int8','onpromotion':bool}
path = 'file/'
#path = '../input/'
train = pd.read_csv(path+'train.csv',usecols=[1,2,3,4,5], dtype = dtypes, 
                    parse_dates = ['date'])#,skiprows = range(1,66458909)) # skip dates before 2017-01-01
train.loc[train['unit_sales']<0,'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] = train['unit_sales'].apply(pd.np.log1p)

test = pd.read_csv(path+'test.csv',dtype = dtypes,usecols= [0,1,2,3,4],
                   parse_dates = ['date'], nrows=None)
test = test.set_index(['store_nbr','item_nbr','date'])
items = pd.read_csv(path+'items.csv').set_index('item_nbr')

df_2017 = train[train['date'].isin(pd.date_range('2013-01-1',periods = 7*11))].copy()
del train
gc.collect()
# ============================================================================
# 
# =============================================================================
promo_2017_train = df_2017.set_index(
        ['store_nbr','item_nbr','date'])[['onpromotion']].unstack(
                level =-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = test[['onpromotion']].unstack().fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017 = pd.concat([promo_2017_train,promo_2017_test],axis=1)
del promo_2017_test,promo_2017_train

df_2017 = df_2017.set_index(
        ['store_nbr','item_nbr','date'])[['unit_sales']].unstack(
                level=-1).fillna(0)
df_2017.columns =df_2017.columns.get_level_values(1)

def get_timespan(df,dt, minus, periods):
    return df[
            pd.date_range(dt-timedelta(days=minus),periods=periods)]

def prepare_dataset(t2017,is_train=True):
    X = pd.DataFrame({
            'mean_3_2017': get_timespan(df_2017, t2017, 3,3).mean(axis=1).values,
            'mean_7_2107': get_timespan(df_2017, t2017, 7,7).mean(axis=1).values,
            'mean_14_2017': get_timespan(df_2017, t2017, 14,14).mean(axis=1).values,
            'promo_14_2017': get_timespan(df_2017, t2017, 14,14).sum(axis=1).values
            })
    for i in range(16):
        X['promo_{}'.format(i)] = promo_2017[
                t2017 + timedelta(days=i)].values.astype(np.int8)
    if is_train:
        y = df_2017[pd.date_range(t2017,periods=16)],values
        return X,y
    return X

print('Preparing dataset...')
t2107= date(2017,6,21)
X_l, y_l = [], []

for i in range(4):
    delta = timedelta(7*i)
    X_tmp, y_tmp = prepare_dataset(t2017 + delta)
    y_l.append(y_tmp)
    X_l.append(X_tmp)

X_train = pd.concat(X_l, axis =0)
y_train = np.concatenate(y_l, axis =0)
del X_l, y_l
X_val,y_val = prepare_dataset(2107,7,26)
x_test = prepare_dataset(2017,8,16, is_train= False)

print('Training and predicting model..')

param = {
        'num_leaves': 2**5 -1,
        'objective': 'regression_l2',
        'max_depth': 8,
        'min_data_in_leaf':50,                                                                                                                                                                                                  
        'learning_rate': 0.05,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,                                                                                                                                                                                                                                                                                             
        'metric': 'l2',
        'num_threads': 4
        }

MAX_ROUNDS = 1000
val_pred = []
cate_pred = []
for i in range(16):         
    print('='*50)
    print('Step %d'%(i+1))
    print('=' *50)
    dtrain = lgb.Dataset(X_train, label= y_train[:,i],categorical_feature = cate_vars,
                         weight = pd.concat([items['perishable']] * 4) * 0.25 + 1)
    dval = lgb.Dataset(X_val, label = y_val[:,i],reference = dtrain,
                       weight =  items['pershable'] *0.25 + 1)
    bst = lgb.train(param, dtrain, num_boost_rounds = MAX_ROUNDS,
                   valid_set = [dtrain,dval],early_stopping_rounds= 50,
                   verbose_eval = 50)
    print('\n'.join('%s: %.2f' %x) for x in sorted(
            zip(X_train.columns, bst.feature_importance('gain')),
            key = lambda x: x[1], reverse =True
            ))
    
    val_pred.append(bst.predict(X_val, num_iteration = bst.best_iteration 
                                or MAX_ROUNDS))
    test_pred.append(bst.predict(X_test, num_iteration = bst.best_iteratin or
                                 MAX_ROUNDS))
print('valid mse:',mean_squared_error(y_val, np.array(val_pred).transpose()))
# =============================================================================
# Submmision
# =============================================================================
print('Making submission..')
y_test = np.array(test_pred).transpose()
df_pred = pd.DataFrame(y_test, index = df_2107.index,
        columns = pd.date_range('2017-08-16',perieds =16)).stack(
                ).to_frame('unit_sales')
df_pred.index.set_names(['store_nbr','item_nbr','date'],inplace = True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb.csv', float_format='%.4f', index=None)

