#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:03:31 2017

@author: sudhir
"""

#Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb 
seed = 4353

#Load data set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Explore data set
print('Number of rows and columns in data set',train.shape)
train.head()

print('Number of rows and columns in data set',test.shape)
test.head()

#Authors
train['author'].value_counts()

plt.figure(figsize=(14,5))
sns.countplot(train['author'],)
plt.xlabel('Author')
plt.title('Target variable distribution')

# Text cleaning
# Remove unwanted punctvation mark
review = re.sub('[^A-Za-z0-9]'," ",train['text'][0]) 
print(review)

#split sentence into word
review = word_tokenize(train['text'][0]) 
print('Word Tokenize:\n',review)

review = [word for word in str(train['text'][0]).lower().split() if  word not in set(stopwords.words('english'))]
print('\nRemoval of Stopwords:\n',review)

review = [word for word in str(train['text'][0]).lower().split() if  word in set(stopwords.words('english'))]
print('\nStopwords in the sentence:\n',review)

ps = PorterStemmer()
review = [ps.stem(word) for word in str(train['text'][0]).lower().split()]
print('\nStemming of word:\n',review)

# Function for text cleaning
def clean_text(df):
    ps = PorterStemmer()
    corpus = []
    for i in range(0, df.shape[0]):        
        review = re.sub('[^A-Za-z0-9]'," ",df['text'][i])
        review = word_tokenize(review)        
        review = [word for word in review if word.lower() not in set(stopwords.words('english'))]
        review = [ps.stem(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)
    
    return corpus

corp_train = clean_text(train)
corp_test = clean_text(test)
train['clean_text'] = corp_train
test['clean_text'] = corp_test
del corp_train,corp_test

# determine length of text
def text_len(df):
    #i = ['text']
    df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
    df['num_uniq_words'] = df['text'].apply(lambda x: len(set(str(x).split())))
    df['num_chars'] = df['text'].apply(lambda x: len(str(x)))
    df['num_stopwords'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in set(stopwords.words('english'))]))
    df['num_punctuations'] = df['text'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))
    df['num_words_upper'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df['num_words_title'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df['mean_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

text_len(train)
text_len(test)

# Bag of words
cv =CountVectorizer(max_features=5000,ngram_range=(1,3),dtype=np.int8,stop_words='english')
X_cv = cv.fit_transform(train['clean_text']).toarray()
X_test_cv = cv.fit_transform(test['clean_text']).toarray()

#TfIdf 
tfidf = TfidfVectorizer(max_features=5000,dtype=np.float32,analyzer='word',
                        ngram_range=(1, 3),use_idf=True, smooth_idf=True, 
                        sublinear_tf=True,stop_words='english',tokenizer=word_tokenize)
X_tf = tfidf.fit_transform(train['clean_text']).toarray()
X_test_tf = tfidf.fit_transform(test['clean_text']).toarray()

#Encoder
author_name = {'EAP':0,'HPL':1,'MWS':2}
y = train['author'].map(author_name) 

#filter data set
unwanted = ['text','id','clean_text']
#X= train.drop(unwanted+['author'],axis=1)
#X_test = test.drop(unwanted,axis=1)

#merge
X = np.concatenate((X_tf,X_cv, train.drop(unwanted+['author'],axis=1).values),axis=1)
X_test = np.concatenate((X_test_cv,X_test_tf,test.drop(unwanted,axis=1).values),axis=1)

del X_tf,X_test_tf,X_cv,X_test_cv

#split dataset
#xtr,xvl,ytr,yvl = train_test_split(X,y,test_size=0.3,random_state=seed)

#model
#MultinomialNB
mNB = MultinomialNB()

kf = KFold(n_splits=10,shuffle=True,random_state=seed)
pred_test_full = 0
cv_score = []
i=1
for train_index,test_index in kf.split(X):
    print('{} of KFlod {}'.format(i,kf.n_splits))    
    xtr,xvl = X[train_index], X[test_index]
    ytr,yvl = y[train_index], y[test_index]
    
    mNB.fit(xtr,ytr)
    y_mNB = mNB.predict(xvl)
    cv_score.append(log_loss(yvl,mNB.predict_proba(xvl)))
    print('confusion matrix:\n',confusion_matrix(yvl,y_mNB))
    pred_test_full += mNB.predict_proba(X_test)
    i+=1
#roc_auc_score(yvl,mNB.predict_proba(xvl)[:,1]) # not for multi class
print('accuracy score',cv_score)
print('Mean accuracy score',np.mean(cv_score))
del xtr,ytr,xvl,yvl

# submit prediction for unseen dataset
#y_pred = mNB.predict_proba(X_test)
y_pred = pred_test_full/10
submit = pd.DataFrame(test['id'])
submit = submit.join(pd.DataFrame(y_pred))
submit.columns = ['id','EAP','HPL','MWS'] 
submit.to_csv('spooky_pred.csv.gz',index=False,compression='gzip')
#submit.to_csv('spooky_pred.csv',index=False)
