#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:24:17 2017

@author: sudhir
"""
#NLP

#Importing dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset
import os
os.chdir('/home/sudhir/R/cancer')
train = pd.read_csv('training_variants')
test = pd.read_csv('test_variants')

trainx = pd.read_csv('training_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])
testx = pd.read_csv('test_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how = 'left', on = 'ID').fillna('')
test = pd.merge(test, testx, how = 'left', on = 'ID').fillna('')

train.Gene.unique().shape
train.Variation.unique().shape

#Data Exploration
cnt_srs = trainx['Text'].value_counts()

train['Gene'].unique()
train['Variation'].unique()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.show()

#cleaning of data
trainx.head()
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
 corpus = []
 for i in range(0,3321):
    review = re.sub('[^a-zA-Z0-9.]',' ',  trainx['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

 corpus1 = []
 for k in range(0,3321):
    review = re.sub('[^a-zA-Z0-9.]',' ',  train['Variation'][k])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
	min_df=1, max_features=1600, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english'
).fit(trainx)

X_train = tfidf.transform(trainx['Text']).toarray()
print(X_train)

#Creating Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

y = train.iloc[:,3]

a= [{'thier is gun in my poket'}]
measurements = [
...     {'city': 'Dubai', 'temperature': 33.},
...     {'city': 'London', 'temperature': 12.},
...     {'city': 'San Fransisco', 'temperature': 18.},
... ]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec.fit_transform(measurements).toarray()

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)