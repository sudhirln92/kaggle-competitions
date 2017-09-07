#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:59:43 2017

@author: sudhir
"""

#https://www.kaggle.com/c/web-traffic-time-series-forecasting/kernels

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re #Regural expression for 
import fbprophet as Prophet

#Import data set
train = pd.read_csv('train_1.csv')
train = train.fillna(0)

#Reduce size of data set
for col in train.columns[1:]:
     train[col] = pd.to_numeric(train[col],downcast='integer')

train.head()
train.info()

#Find out language in URL
def findlanguage(url):
    res = re.search('[a-z][a-z].wikipedia.org',url)
    if res:
        return res[0][0:2]
    return 'na'
train['lang'] = train['Page'].map(findlanguage)

#barplot
train.groupby(['lang'])['Page'].count().plot(kind='bar')
plt.xlabel('Language')
#Seperater language
'''There are 7 languages plus the media pages. The languages used here are: 
English, Japanese, German, French, Chinese, Russian, and Spanish '''
lang_set ={}
lang_set['en'] = train[train['lang']=='en'].iloc[:,0:-1]
lang_set['ja'] = train[train['lang']=='ja'].iloc[:,0:-1]
lang_set['de'] = train[train['lang']=='de'].iloc[:,0:-1]
lang_set['es'] = train[train['lang']=='es'].iloc[:,0:-1]
lang_set['fr'] = train[train['lang']=='fr'].iloc[:,0:-1]
lang_set['na'] = train[train['lang']=='na'].iloc[:,0:-1]
lang_set['ru'] = train[train['lang']=='ru'].iloc[:,0:-1]
lang_set['zh'] = train[train['lang']=='zh'].iloc[:,0:-1]

lang_set['en'][0:2]

#temp = pd.DataFrame({'date':train.columns[1:],'mean':train.mean(axis=0)})
#train.mean()
