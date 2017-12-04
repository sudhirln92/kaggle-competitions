#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:24:17 2017

@author: sudhir
"""
#NLP

# =============================================================================
# #Importing library
# =============================================================================
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# =============================================================================
# Read data set
# =============================================================================
train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/stage2_test_variants.csv')

trainx = pd.read_csv('../input/training_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/stage2_test_text.csv',sep = '\|\|', engine= 'python', header=None, 
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

#count Gene
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

"""
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
corp_train = pd.read_csv('corp_train.csv')
corp_test = pd.read_csv('corp_test2.csv')
#corp_train = pd.DataFrame(corp_train,columns=['Text'])
#corp_test = pd.DataFrame(corp_test,columns=['Text'])
#corp_train.to_csv('corp_train.csv',index=False)
#corp_test.to_csv('corp_test2.csv',index=False)

train1=train
test1=test
train1['Text'] = corp_train
test1['Text'] = corp_test """

# =============================================================================
# # Determine lenght of text
# =============================================================================
def textlen(train):
    k = train['Text'].apply(lambda x: len(str(x).split()))
    l = train['Text'].apply(lambda x: len(str(x)))
    return k, l

train['Text_no_word'], train['Text_no_char'] = textlen(train)
test['Text_no_word'], test['Text_no_char'] = textlen(test)

# =============================================================================
# # Bag of word
# =============================================================================
tfidf = TfidfVectorizer(
	min_df=1, max_features=1600, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english')
X_train = tfidf.fit_transform(train['Text']).toarray()
print(X_train)
X_test = tfidf.fit_transform(test['Text']).toarray()

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

#onh = OneHotEncoder(categorical_features=X_train['Gene'])
#K = onh.fit_transform(X_train).toarray()
#y_train=pd.get_dummies(y_train)
X_train = pd.DataFrame(X_train)
X_train = X_train.join(train[['Gene', 'Variation', 'Text_no_word','Text_no_char']]) 
X_test = pd.DataFrame(X_test)
X_test = X_test.join(test[['Gene', 'Variation', 'Text_no_word','Text_no_char']])

# =============================================================================
# # Feature Scaling
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = train['Class']

# =============================================================================
# #split data set
# =============================================================================
xtr,xvl,ytr,yvl = train_test_split(X_train,y_train,test_size=0.3,random_state=10)

# =============================================================================
# #Modeling
# #Naive clf
# =============================================================================
nbc = GaussianNB()
#mnb = MultinomialNB()
nbc.fit(xtr,ytr)
#mnb.fit(xtr,ytr)
y_nbcP = nbc.predict(xvl)

y_nbc = nbc.predict_proba(X_test)
#y_mnb = mnb.predict_proba(X_test)
print(confusion_matrix(yvl,nbc.predict(xvl)))

# =============================================================================
# # Random forest classifier
# =============================================================================
"""
rfc = RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=4)
rfc.fit(xtr,ytr)
confusion_matrix(yvl,rfc.predict(xvl))

y_rfc=rfc.predict_proba(X_test) """

# =============================================================================
# #Light gbm 
# =============================================================================

def runLgb(Xtr,Xvl,ytr,yvl,test,num_rounds=10,max_depth=10,eta=0.5,subsample=0.8,
           colsample=0.8,min_child_weight=1,early_stopping_rounds=50,seeds_val=2017):
    
    param = {'task': 'train',
             'boosting_type': 'gbdt',
             'objective':'multiclass',
             'num_class':9,
             'learning_rate':eta,
             'metric':{'multi_logloss'},
             'max_depth':max_depth,
             #'min_child_weight':min_child_weight,
             'bagging_fraction':subsample,
             'feature_fraction':colsample,
             'bagging_seed':seeds_val,
             'num_iterations': num_rounds, 
             'num_leaves': 95,           
             'min_data_in_leaf': 60, 
             'lambda_l1': 1.0,
             'verbose':10,
             'nthread':-1}
    lgtrain = lgb.Dataset(Xtr,label=ytr)
    lgval = lgb.Dataset(Xvl,label=yvl)
    model = lgb.train(param,lgtrain,num_rounds,valid_sets=lgval,
                      early_stopping_rounds=early_stopping_rounds,verbose_eval=20)
    pred_val = model.predict(Xvl,num_iteration = model.best_iteration)
    pred_test = model.predict(test,num_iteration=model.best_iteration)
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
                            eta=0.02,)
    pred_test_full +=pred_test
    #cv_score.append(np.sqrt(mean_squared_error(yvl,pred_val)))
pred_test = pred_test_full/10
# =============================================================================
# #Result submission
# =============================================================================
submit = pd.DataFrame(test.ID)
submit = submit.join(pd.DataFrame(pred_test))
submit.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
submit.to_csv('submission.csv', index=False)