# Sberbank Russian Housing Market
Can you predict realty price fluctuations in Russia’s volatile economy?
Housing costs demand a significant investment from both consumers and developers. And when it comes to planning a budget—whether personal or corporate—the last thing anyone needs is uncertainty about one of their biggets expenses. Sberbank, Russia’s oldest and largest bank, helps their customers by making predictions about realty prices so renters, developers, and lenders are more confident when they sign a lease or purchase a building.

The aim of this competition is to predict the sale price of each property.

# Data set Description
The training data is from August 2011 to June 2015, and the test set is from July 2015 to May 2016. The dataset also includes information about overall conditions in Russia's economy and finance sector. Train dataset consist of 30.5K * 292 observaation and test dataset consist of 7662*291 observation.

## Feature Engineering
Missing value in dataset is replace by mean and mode.The feature with 0 zero variance is remove from the dataset.

## Modeling
Linear regression is used to build model. Random forest algorithm is used to build predictive model

## Model validation
Submissions are evaluated on the RMSLE between their predicted prices and the actual data. The target variable, called price_doc in the training set, is the sale price of each property.

## Out come of project:
This project give good opportunity to explore data set. Create new feature from data  set.
