# Nomad2018 Predicting Transparent Conductors
Predict the key properties of novel transparent semiconductors
https://www.kaggle.com/c/nomad2018-predict-transparent-conductors

The diffrent properties of Aluminum,Gallium,Indium is given in data set. In order to reduce electric transmission loss,discovery of new transparent conductor alloy is important. The transparent conductor having characteristic good conductivity and have a low absorption.

The aim is to prediction of two target properties: the formation energy (which is an indication of the stability of a new material) and the bandgap energy (which is an indication of the potential for transparency over the visible range) to facilitate the discovery of new transparent conductors

The task for this competition is to predict two target properties:

    Formation energy (an important indicator of the stability of a material)
    Bandgap energy (an important property for optoelectronic applications)


# Data set
train,test,geomeric property data set are provided by the host.

# Feature engineering
Basic element property feature is extracted from geomerty detail file, PCA transformation is performed.


#Hyperparameter Tuning
Grid search meathodis used for parameter tuning with 3 fold cross validation. Different parameter tuning is done for two diffirent target variable.

# Model
The target variable is transformed to log. 

Stack layer 1
1. LightGBM
2. XgBoost
3. CatBoost

Stack layer 2
4. Linear Regression

# Submision 


