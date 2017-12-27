# WSDM - KKBox's Churn Prediction Challenge 

Can you predict when subscribers will churn?
https://www.kaggle.com/c/kkbox-churn-prediction-challenge

KKBOX is Asia’s leading music streaming service, holding the world’s most comprehensive Asia-Pop music library with over 30 million tracks. In this  task is to build an algorithm that predicts whether a user will churn after their subscription expires. Specifically, we want to see if a user make a new service subscription transaction within 30 days after their current membership expiration date.As a music streaming service provider, KKBox has members subscribe to their service. When the subscription is about to expire, the user can choose to renew, or cancel the service. They also have the option to auto-renew but can still cancel their membership any time. 


# Data set
Their are 5 dataset such as train, members, transcations, user logs, sample_submission. The date set is realized two times in the compitetion.
The sample submission contains used id(msno) and is_churn, based on this user id(msno) test data set is created. The given diffrent data is merged on user id of train, test. The churn in data set is a depedent variable, which is imbalanced.

# Feature engineering
The base dataset consist of 23 features. The missing value present in data is replaced with mode repective feature. Their are 4 date feature present in data set, diffferent date feature is extracted from data. 
Encoding Data: label encoding, One hot encoding, Frequency encoding, Mean and median range encoding.

# Model
Logistic regression, Light GBM, Xgboost,Catboost
Ensemble modeling is performed, base model is 2 Lightgbm and next level stack is catboost

# Model evaluation  
The evaluation metric for this competition is Log Loss.
The model is cross validate using Stratified KFold algorithm.

 

