# Nomad2018 Predicting Transparent Conductors
Predict the key properties of novel transparent semiconductors

| Project Name | Type  | Public Kernel | Metric | Date |
| ------ | ------ | ------ | ------ | ------ | 
|[Nomad2018 Predicting Transparent Conductors](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors)| Regression | [1. Simple Electron Volt predictor](https://www.kaggle.com/sudhirnl7/simple-electron-volt-predictor), [2. Simple ANN](https://www.kaggle.com/sudhirnl7/simple-ann) | RMSLE |Dec-2017 |

The diffrent properties of Aluminum,Gallium,Indium is given in data set. In order to reduce electric transmission loss,discovery of new transparent conductor alloy is important. The transparent conductor having characteristic good conductivity and have a low absorption.

The aim is to prediction of two target properties: the formation energy (which is an indication of the stability of a new material) and the bandgap energy (which is an indication of the potential for transparency over the visible range) to facilitate the discovery of new transparent conductors.

## Data set
train,test,geomeric property data set are provided by the host.
The task for this competition is to predict two target properties:

    Formation energy (an important indicator of the stability of a material)
    Bandgap energy (an important property for optoelectronic applications)


## Evaluvation metrics
The root mean square log error (RMSLE) is used as evaluvation metric to analyze performance of model. 

RMSLE = $\sqrt {\frac{1}{n}\sum_{1}^{m}(log(y'+1) +log(y+1))}$

m  is the total number of observations 
y' is your prediction
y is the actual value 
log(x) is the natural logarithm of x


## Feature engineering
Some of domain feature such as volumn, lattice angle in radians,atomic density, descriptive statistic features and one hot encoding on categorical variables are added.

## Public kernel
* [Simple Electron Volt predictor](https://www.kaggle.com/sudhirnl7/simple-electron-volt-predictor)
* [Simple ANN](https://www.kaggle.com/sudhirnl7/simple-ann)

## Model
Intial analysis is made using linear regression, later gradient boosting methods are used to build model. The model is evaluated using 5 fold cross validation stragergy.

## Out come of project
This project help to analysze property of atoms using exploratory data analysis.
