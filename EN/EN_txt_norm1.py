#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:21:40 2017

@author: sudhir
"""
#Import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import word_tokenize
#Load data set
train = pd.read_csv('en_train.csv')
test = pd.read_csv('en_test.csv')

#Data exploration 
train.head()
test.head()

train.isnull().sum()
test.isnull().sum()

train[train['token_id']==train['token_id'].isnull()]

print('Number of rows & columns in train data set:',train.shape)
print('Number of rows & columns in train data set:',test.shape)

train['class'].unique()
train['class'].nunique()

plt.figure(figsize=(16,8))
sns.countplot(train['class'])
plt.yscale('log')
plt.xlabel('Class')
plt.ylabel('Number of word in a class')
plt.xticks(rotation=45)


label = list(train['class'].unique())

#let's explore each class categories  
for i in label:
    print('\n\nClass label:',i)
    print('\n number of class label:',train[train['class']==i].count()['class'])
    print(train[train['class']==i].head())
    print(len(train[train['class']==i]))
    

#let's explore token_id
train['token_id'].unique()
train['token_id'].nunique()

plt.figure(figsize=(12,10))
sns.countplot(train['token_id'])
plt.yscale('log')
plt.xlabel('Token ID')
plt.ylabel('Number of occurance')

train.group_by(['class'])
train['token_id'].count()
train['sentence_id'].nunique()

train['before'].values

# Bag of word
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

train.dropna(inplace=True)
t= train.iloc[0:100000]
#cve = CountVectorizer()
tfi = TfidfVectorizer(min_df=5,lowercase=True,ngram_range=(1,3),stop_words='english',
                      strip_accents='unicode',analyzer='word',token_pattern=r'\w+',
                      use_idf=True,smooth_idf=True,sublinear_tf=True).fit(t['before'].apply(str))
train_tfi = tfi.transform(t['before'].apply(str))

svd = TruncatedSVD(n_components=25,n_iter=25,random_state=10).fit(train_tfi)
b= svd.transform(train_tfi)   
c= t.join(train_tfi)


#data set split
c= train['class']
c=c.iloc[0:100000]
y_train = pd.factorize(c) 
xtr,xvl,ytr,yvl = train_test_split(b,y_train[0],test_size=0.3,random_state=1)

clf = lgb.LGBMClassifier()


clf.fit(xtr,ytr,eval_set=[(xvl,yvl)],early_stopping_rounds=10,)

pred = clf.predict(xvl)

y = pd.Series(pred).apply(lambda x: y_train[1][x])

x_valid = [ [chr(x +ord('a')) for x in y ] for y in xvl]

max_size = 20000

k=train_tfi.getcol