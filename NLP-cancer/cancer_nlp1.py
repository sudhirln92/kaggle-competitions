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
os.chdir('D:\Py-R\cancer')
train = pd.read_csv('training_variants')
test = pd.read_csv('test_variants')

trainx = pd.read_csv('training_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])
testx = pd.read_csv('test_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how = 'left', on = 'ID').fillna('')
test = pd.merge(test, testx, how = 'left', on = 'ID').fillna('')

train.Gene.unique().value_counts()
train.Variation.unique().value_counts()

#Data Exploration
cnt_srs = trainx['Text'].value_counts()

train['Gene'].unique()
train['Variation'].unique().value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.show()

#cleaning of data
trainx.head()
#nltk.download() 

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
	min_df=1, max_features=1600, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english'
).fit(train)

X_train = tfidf.transform(trainx['Text']).toarray()
print(X_train)

X_test = tfidf.transform(testx['Text']).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = train['Class']


# Preparing model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=5000)
classifier.fit(X_train,y_train)

#Predict
y_pred=classifier.predict(X_test)

#np.mean(y_pred==y_train)
#Converting to categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y_pred=le.fit_transform(y_pred)
y_pred=y_pred.reshape(-1, 1)
onh = OneHotEncoder(categorical_features = [0])
y_pred =onh.fit_transform(y_pred).toarray()

submit = pd.DataFrame(test.ID)
submit = submit.join(pd.DataFrame(y_pred))
submit.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']

submit.to_csv('submission.csv', index=False)
