# Nomad2018 Predicting Transparent Conductors
Predict the key properties of novel transparent semiconductors
[Compitation link](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

The diffrent properties of Aluminum,Gallium,Indium is given in data set. In order to reduce electric transmission loss,discovery of new transparent conductor alloy is important. The transparent conductor having characteristic good conductivity and have a low absorption.

The aim is to prediction of two target properties: the formation energy (which is an indication of the stability of a new material) and the bandgap energy (which is an indication of the potential for transparency over the visible range) to facilitate the discovery of new transparent conductors

## Data set
Submissions are now evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.



## Evaluvation metrics
The root mean square log error (RMSLE) is used as evaluvation metric to analyze performance of model. 

RMSLE = \sqrt {\frac{1}{n}\sum_{1}^{m}(log(y'+1) +log(y+1))}

m  is the total number of observations 
y' is your prediction
y is the actual value 
log(x) is the natural logarithm of x


## Feature engineering
Some descriptive statistic features and one hot encoding on categorical variables are added.

## Model
Intial analysis is made using logistic regression, later gradient boosting methods are used to build model. The model is evaluated using 5 fold cross validation stragergy. 

## Public Kernel
[Simple logistic regression - Wisdom](/home/sudhir/Git/Kaggle_competition/Normad 2018 Transperant condoctor/readme.md)


## Out come of project
This project help to analysze property of atoms using exploratory data analysis.
