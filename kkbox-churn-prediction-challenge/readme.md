# WSDM - KKBox's Churn Prediction Challenge
Predict the key properties of novel transparent semiconductors

| Project Name | Type  | Public Kernel | Metric |
| ------ | ------ | ------ | ------ | 
|[WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)| Classification |[Simple logistic regression - Wisdom](https://www.kaggle.com/sudhirnl7/simple-logistic-regression-wisdom)| log loss |

In this competition you’re tasked to build an algorithm that predicts whether a user will churn after their subscription expires.

## Data set
Submissions are now evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

## Evaluvation metrics
The evaluation metric for this competition is Log Loss

## Feature engineering
Some descriptive statistic features and one hot encoding on categorical variables are added.

## Model
Intial analysis is made using logistic regression, later gradient boosting methods are used to build model. The model is evaluated using 5 fold cross validation stragergy.

## Out come of project
This project help to analysze property of atoms using exploratory data analysis.
