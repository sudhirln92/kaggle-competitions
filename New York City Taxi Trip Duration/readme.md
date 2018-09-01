# New York City Taxi Trip Duration
Share code and data to improve ride time predictions.In this competition, Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City. 
[Compitation link](https://www.kaggle.com/c/nyc-taxi-trip-duration)

## Data set
The data set contains 14.5lakhs observations and 11 variable.

## Evaluvation metrics
The root mean square log error (RMSLE) is used as evaluvation metric to analyze performance of model. 

RMSLE = $\sqrt {\frac{1}{n}\sum_{1}^{m}(log(y'+1) +log(y+1))}$

m  is the total number of observations 
y' is your prediction
y is the actual value 
log(x) is the natural logarithm of x


## Feature engineering
The new features are extracted from variables, outlier data points were removed, Exploratory data analysis is performed. Clustering algorithm is applied to location features in dataset.  descriptive statistic features and one hot encoding on categorical variables are added.

## Model
Intial analysis is made using linear regression, later gradient boosting methods are used to build model. The model is evaluated using 5 fold cross validation stragergy.

## Out come of project
This project help to analysze property of atoms using exploratory data analysis.
