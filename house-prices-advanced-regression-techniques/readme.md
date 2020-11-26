# House Prices: Advanced Regression Techniques
Predict sales prices and practice feature engineering, RFs, and gradient boosting

| Project Name | Type  | Public Kernel | Metric |
| ------ | ------ | ------ | ------ | 
|[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)| Regression |[House price analysis, Ridge Regression](https://www.kaggle.com/sudhirnl7/house-price-analysis-ridge-regression)| RMSE |

It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

## Data set
train.csv, test.csv


## Evaluvation metrics
The root mean square log error (RMSLE) is used as evaluvation metric to analyze performance of model. Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

RMSLE = $\sqrt {\frac{1}{n}\sum_{1}^{m}(log(y'+1) +log(y+1))}$

m  is the total number of observations 
y' is your prediction
y is the actual value 
log(x) is the natural logarithm of x


## Feature engineering
LabelBinarizer,One hot encoding.

## Public Kernel


## Model
The ridge (l2 regularized linear regression) methods is used to build model.

## Out come of project
This project help to analysze property of atoms using exploratory data analysis.
