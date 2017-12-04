
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:24:17 2017

@author: sudhir
"""
#NLP

#Importing library
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# =============================================================================
# #Importing Dataset
# =============================================================================
#import os
#os.chdir('D:\Py-R\cancer')
train = pd.read_csv('training_variants')
test = pd.read_csv('test_variants')

trainx = pd.read_csv('training_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])
testx = pd.read_csv('test_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how = 'left', on = 'ID').fillna('')
test = pd.merge(test, testx, how = 'left', on = 'ID').fillna('')

# =============================================================================
# #Data Exploration
# =============================================================================
#cnt_srs = trainx['Text'].sum()
train.Gene.nunique()
train['Gene'].unique()

k = train.groupby('Gene')['Gene'].count()

plt.figure(figsize=(12,6))
plt.hist(k, bins=150,log=True)
plt.xlabel('Number of times Gene appared')
plt.ylabel('Log of count')
plt.show()

# =============================================================================
# #count Gene
# =============================================================================
from collections import Counter
plt.figure(figsize=(12,10))
sns.countplot((train['Gene']))
plt.xticks()
genecount = Counter(train['Gene'])
print(genecount,'\n',len(genecount))

train.Variation.nunique()
train['Variation'].unique()

k = train.groupby('Variation')['Variation'].count()

plt.figure(figsize=(12,6))
sns.distplot(k)

# =============================================================================
# #cleaning of data
# =============================================================================
def cleantext(train,):
    corpus = []
    for i in range(0,train.shape[0]):
        review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]"," ",train['Text'][i])
        review = review.lower().split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    
    return corpus

corp_train = cleantext(train)
corp_test = cleantext(test)

#corp_train = pd.read_csv('corp_train.csv')
#corp_test = pd.read_csv('corp_test.csv')
#corp_train = pd.DataFrame(corp_train,columns=['Text'])
#corp_test = pd.DataFrame(corp_test,columns=['Text'])
#corp_train.to_csv('corp_train.csv',index=False)
#corp_test.to_csv('corp_test.csv',index=False)

# =============================================================================
# # Determine lenght of text
# =============================================================================
def textlen(train):
    k = train['Text'].apply(lambda x: len(str(x).split()))
    l = train['Text'].apply(lambda x: len(str(x)))
    return k, l

train['Text_no_word'], train['Text_no_char'] = textlen(corp_train)
test['Text_no_word'], test['Text_no_char'] = textlen(corp_test)

#
for i in range(10):
    print('\n Doc', str(i))
    stopcheck = Counter(corp_train[i].split())
    print(stopcheck.most_common()[:10])

# =============================================================================
# # Bag of word
# =============================================================================
tfidf = TfidfVectorizer(
	min_df=1, max_features=1600, strip_accents='unicode',lowercase =False,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english')

train1['Text'] = corp_train
X_train = tfidf.fit_transform(train1['Text']).toarray()
print(X_train)
X_test = tfidf.fit_transform(test['Text']).toarray()

cve = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=None)  
X=cve.fit_transform(train1['Text']).toarray()
test1 = cve.fit_transform(corp_test).toarray()

# =============================================================================
# #Converting to categorical variable
# =============================================================================
def encoding(df,col):
    le = LabelEncoder()
    for i in col:
        df[i] = le.fit_transform(df[i])
train.columns
col = ['Gene', 'Variation', 'Class']
encoding(train,col)
encoding(test,['Gene', 'Variation'])

#onh = OneHotEncoder(sparse=False)
#temp = onh.fit_transform(train['Class'])
#y_train=pd.get_dummies(y_train)
X_train = pd.DataFrame(X_train)
X_train = X_train.join(train[['Gene', 'Variation', 'Text_no_word','Text_no_char']]) 
X_test = pd.DataFrame(X_test)
X_test = X_test.join(test[['Gene', 'Variation', 'Text_no_word','Text_no_char']])
#a= pd.get_dummies(X_train['Gene'])

# =============================================================================
# # Feature Scaling
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = train['Class']

# =============================================================================
# #Modeling
# #Naive clf
# nbc = GaussianNB()
# =============================================================================
nbc.fit(X_train,y_train)
y_nbc = nbc.predict_proba(X_test)

# =============================================================================
# # Random forest classifier
# =============================================================================
rfc = RandomForestClassifier(n_estimators=5,max_depth=8,min_samples_split=2)
rfc.fit(X_train,y_train)
y_rfc=rfc.predict_proba(X_test)

# =============================================================================
# #Light gbm 
# =============================================================================

def runLgb(Xtr,Xvl,ytr,yvl,test,num_rounds=10,max_depth=10,eta=0.5,subsample=0.8,
           colsample=0.8,min_child_weight=1,early_stopping_rounds=50,seeds_val=2017):
    
    param = {'objective':'naive_bayes',
             'learning_rate':eta,
             'metric':'mlogloss',
             'max_depth':max_depth,
             'min_child_weight':min_child_weight,
             'bagging_fraction':subsample,
             'feature_fraction':colsample,
             'bagging_seed':seeds_val,
             'verbose':10,
             'nthread':-1}
    lgtrain = lgb.Dataset(Xtr,label=ytr)
    lgval = lgb.Dataset(Xvl,label=yvl)
    model = lgb.train(param,lgtrain,num_rounds,valid_set=lgval,
                      early_stopping_rounds=early_stopping_rounds,verbose_eval=20)
    pred_val = model.predict_proba(Xvl,num_iteration = model.best_iteration)
    pred_test = model.predict_proba(test,num_iteration=model.best_iteration)
    return pred_test,pred_val,model

# =============================================================================
# #k-fold corss validate model
# =============================================================================
kf = KFold(n_splits=10,random_state=111,shuffle=True)
cv_score = []
pred_test_full=0

for train_index,test_index in kf.split(X_train):
    Xtr,Xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train[train_index],y_train[test_index]
    
    pred_test,pred_val,model = runLgb(Xtr,Xvl,ytr,yvl,X_test,num_rounds=10,max_depth=3,
                            eta=0.8,)
    pred_test_full +=pred_test
    cv_score.append(np.sqrt(mean_squared_error(yvl,pred_val)))

# =============================================================================
# #Predict
# =============================================================================
y=pd.DataFrame(y_rfc)
#np.mean(y_pred==y_train)

submit = pd.DataFrame(test.ID)
submit = submit.join(pd.DataFrame(y_pred))
submit.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']

submit.to_csv('submission.csv', index=False)
