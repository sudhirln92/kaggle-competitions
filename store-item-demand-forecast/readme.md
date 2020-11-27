# Store Item Demand Forecasting Challenge
Predict 3 months of item sales at different stores.

| Project Name | Type  | Public Kernel | Metric | Date |
| ------ | ------ | ------ | ------ | ------ | 
|[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)| Time Series | [Store Item Demand Forecast-EDA-GBM Model](https://www.kaggle.com/sudhirnl7/tore-item-demand-forecast-eda-gbm-model) | SMAPE | Jun-2018 |


You are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

# Evaluation metric
Submissions are evaluated on SMAPE between forecasts and actual values. We define SMAPE = 0 when the actual and predicted values are both 0.
SMAPE: Symmetric mean absolute percentage error

## Data set Description
The objective of this competition is to predict 3 months of item-level sales data at different store locations.
* train.csv - Training data
* test.csv - Test data (Note: the Public/Private split is time based)

### Data fields
* date - Date of the sale data. There are no holiday effects or store closures.
* store - Store ID
* item - Item ID
* sales - Number of items sold at a particular store on a particular date.

## Exploratory Data analysis


## Feature Engineering
One hot encoding is performed on categorical variable. Date time features are create. Based on diffirent datetime, store, item  new feature such as Mean, median, min, max, sum, std,  are  aggregate on sales add as new column to the datset.

## Modeling
Lightgbm algorithms are used. 

## Model validation
The model is 5 times cross validated using Kfold validation strategy, the average of predicted target variable is submitted. 

## Out come of project:
This project gives a good opportunity to explore data set. Create new feature from data  set.
