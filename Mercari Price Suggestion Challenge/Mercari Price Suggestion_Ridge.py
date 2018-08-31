na#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:22:10 2018

@author: sudhir
"""
# =============================================================================
# Import packages
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import time

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from scipy.sparse import hstack, csc_matrix
from nltk.corpus import stopwords
import re
import gc
seed = 321

# =============================================================================
# Read data set
# =============================================================================
start_time = time.time()
#path ='../input/'
path = 'file/'
train = pd.read_csv(path+'train.tsv',sep='\t',nrows=None)
test = pd.read_csv(path+'test.tsv',sep='\t',nrows=None)
print('Number of rows and columns in train data set is :',train.shape)
print('Number of rows and columns in teset data set is :',test.shape)

NUM_BRANDS = 4500
NUM_CATEGORIES = 1290
NAME_MIN_Df = 10
MAX_FEAT_DESCP = 50000

nrow_train = train.shape[0]
y = np.log1p(train['price'])
submit = pd.DataFrame( {'test_id':test['test_id']})
df = pd.concat([train,test])

# =============================================================================
# Evaluvation mertic
# =============================================================================
def rmsle(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

# =============================================================================
# data analysis
# =============================================================================
train.head()
test.head()
train.describe(include='all').T
test.describe(include='all').T

k = pd.DataFrame()
k['train'] = train.isnull().sum()
k['test'] = test.isnull().sum() ;k

# =============================================================================
# Feature engineering
# =============================================================================

def missing_data(d):
    d['brand_name'].fillna(value='missing',inplace = True )
    d['item_description'].fillna(value='missing',inplace = True )
    d['general_cat'].fillna(value='missing',inplace = True )
    d['subcat_1'].fillna(value='missing',inplace = True )
    d['subcat_2'].fillna(value='missing',inplace = True )


def split_cat(text):
    try:
        return text.split('/')
    except:
        return ('No Label','No Label','No Label')

def cutting_data(d):
    #Cutting data set
    pop_brands = d['brand_name'].value_counts().loc[lambda x: x.index !='missing'].index[:NUM_BRANDS]
    d.loc[~d['brand_name'].isin(pop_brands),'brand_name'] = 'missing'
    pop_category = df['general_cat'].value_counts().loc[lambda x: x.index !='missing'].index[:NUM_CATEGORIES]
    pop_category = df['subcat_1'].value_counts().loc[lambda x: x.index !='missing'].index[:NUM_CATEGORIES]
    pop_category = df['subcat_2'].value_counts().loc[lambda x: x.index !='missing'].index[:NUM_CATEGORIES]
    d.loc[~d['general_cat'].isin(pop_category),'general_cat'] = 'missing'
    d.loc[~d['subcat_1'].isin(pop_category),'subcat_1'] = 'missing'
    d.loc[~d['subcat_2'].isin(pop_category),'subcat_2'] = 'missing'

def category_variable(d):
    # Convert to categorical variable
    d['brand_name'] = d['brand_name'].astype('category')
    d['item_condition_id'] = d['item_condition_id'].astype('category')
    d['general_cat'] = d['general_cat'].astype('category')
    d['subcat_1'] = d['subcat_1'].astype('category')
    d['subcat_1'] = d['subcat_1'].astype('category')


df['general_cat'], df['subcat_1'], df['subcat_2'] = \
    zip(*df['category_name'].apply(lambda x: split_cat(x)))
print("[{}] Finished split category".format(time.time()-start_time))


missing_data(df)
print('[{}] Finshed handling missing value '.format(time.time()-start_time))

cutting_data(df)
print('[{}] Fininshed cutting'.format(time.time()-start_time))

category_variable(df)
print('[{}] Finished converting to category'.format(time.time()-start_time))

cv = CountVectorizer(min_df=NAME_MIN_Df)
X_name = cv.fit_transform(df['name'])
print('[{}] Finished count vector name'.format(time.time()-start_time))

cv = CountVectorizer(min_df=NAME_MIN_Df)
X_general = cv.fit_transform(df['general_cat'])
X_subcat_1 = cv.fit_transform(df['subcat_1'])
X_subcat_2 = cv.fit_transform(df['subcat_2'])
print('[{}] Finished count category name'.format(time.time()-start_time))

tv = TfidfVectorizer(max_features=MAX_FEAT_DESCP,  stop_words='english',
                         lowercase=True,analyzer='word', dtype=np.float32,
                        ngram_range=(1,3))
X_desciption = tv.fit_transform(df['item_description'])
print('[{}] Finished TFIDF vector name'.format(time.time()-start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(df['brand_name'])
print('[{}] Finished label binarizer brand name'.format(time.time()-start_time))

X_dummies = csc_matrix(pd.get_dummies(df[['item_condition_id','shipping']],
                                      sparse=True).values)
print("[{}] Finished to dummies on 'item_condition_id','shipping'".format(time.time()-start_time))

sparse_df = hstack((X_brand, X_general, X_subcat_1, X_subcat_2, X_desciption, X_name,X_dummies)).tocsr()
print("[{}] Finished to sparse".format(time.time()-start_time))


X = sparse_df[:nrow_train]
X_test = sparse_df[nrow_train:]
# =============================================================================
# MOdel
# =============================================================================
rdg_model = Ridge(solver='sag',fit_intercept=True, random_state = seed)
rdg_model.fit(X,y)
print("[{}] Finished to train Ridge".format(time.time()-start_time))

pred = rdg_model.predict(X_test)
print("[{}] Finished to predict Ridge".format(time.time()-start_time))

# =============================================================================
# Submission
# =============================================================================

#pred = np.abs(pred)
submit = pd.DataFrame({'test_id':test['test_id'],'price':np.expm1(pred)})
#submit.to_csv('mercari.csv.gz',index=False,compression='gzip')
submit.to_csv('mercari.csv',index=False)

submit.head()