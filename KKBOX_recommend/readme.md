# WSDM - KKBox's Music Recommendation Challenge
Can you build the best music recommendation system?

https://www.kaggle.com/c/kkbox-music-recommendation-challenge

KKBOX is Asia’s leading music streaming service, holding the world’s most comprehensive Asia-Pop music library with over 30 million tracks. 


# Data set
train,test,songs,members, song extra info data set.

# Feature engineering
The missing value present in data is replaced with mode repective feature. The diffferent date feature is extracted from data. 
Encoding Data: label encoding, One hot encoding

# Model
Logistic regression, Light GBM

# Model evaluation  
The evaluation metric for this competition is AUC.
The model is cross validate using Stratified KFold algorithm.

 

