
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:24:17 2017

@author: sudhir
"""
#NLP

# =============================================================================
# #Importing dataset
# =============================================================================
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

train.Gene.unique()
train.Variation.unique()

# =============================================================================
# #Data Exploration
# =============================================================================
cnt_srs = trainx['Text'].sum()

train['Gene'].unique()
train['Variation'].unigue()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.show()

# =============================================================================
# #cleaning of data
# =============================================================================
trainx.head()
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus1 = []
for i in range(0,3321):
    review = re.sub('[^a-zA-Z]', ' ', train['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)
 
# =============================================================================
# TF-IDF
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
	min_df=1, max_features=1600, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english'
).fit(train)

X_train = tfidf.transform(trainx['Text']).toarray()
print(X_train)

X_test = tfidf.transform(testx['Text']).toarray()

# =============================================================================
# # Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = train['Class']

# =============================================================================
# #Converting to categorical variable
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y_train=le.fit_transform(y_train)
#onh = OneHotEncoder(categorical_features = [0])
#y_train =onh.fit_transform(y_train).toarray()

# =============================================================================
# # Preparing model
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=5000)
classifier.fit(X_train,y_train)

# =============================================================================
# #Predict
# =============================================================================
y_pred=classifier.predict_proba(X_test)
y=pd.DataFrame(y_pred)
#np.mean(y_pred==y_train)

submit = pd.DataFrame(test.ID)
submit = submit.join(pd.DataFrame(y_pred))
submit.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']

submit.to_csv('submission.csv', index=False)
