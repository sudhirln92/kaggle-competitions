# Porto Seguro’s Safe Driver Prediction					Monday 04 December 2017 03:58:30 PM IST

# Porto Seguro Safe Driver Prediction 
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
	Porto Sergo Is Largest Auto And Homeowner Insurance Company. The aim of the project is to predict probability that a driver will intiate an auto insurance claim next yesr. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more driver. 

# Data set Description
Porto Seguro provided close to 600k and 900k observation of train and test dataset respectively. They were 57 feature anonymized in order to protect company trade secrets, but we were given bit informaation about  The train and test data set contains feature with similar grouping are tagged with (e.g., ind, reg, car, cat, calc, bin). Values of  -1 indicate that the feature was missing from the observation. The target column in data set is whether or not claim was filed for that policy holder. The target variable is quite unbalanced, with only  %4 of  policyholders in training data filing claim within the year.

# Exploratory Data analysis
The feature name contains tag ind,car, contains category data type, there are few of them also numeric type. The feature contains bin tag is binary feature, feature having tag calc  is zero correlation with other variables, it has been drop from data set. 
About 2.5% of values are missing in total in each of the train and test data sets

# Feature Extraction
One hot encoding is performed on categorical variable. 
Mean and median of all variable is determined and add as new column in data set. 

# Modeling
Logistic regression,Xgboost,Lightgbm algorithms are used

# Data validation
The model is evaluated on Gini coefficient, a popular measure in insurance industry which quantifies how well-ranked predicted probabilitie are relative to actual class labels. We are familiar with ROC AUC metric, it turns out simple relationship of gini is 2 * AUC -1 with ROC AUC  . The model is 5 times cross validated using Straigfied Kfold validation strategy, the average of predicted target variable is submitted.
 
# Out come of project:
This project give good opportunity to explore data set. Create new feature from data  set.    
I have submitted prediction more than 70 to kaggle wesite. My best gini score 0.282 on public LB and 0.287 on private LB. Overall my standing at end of competition is top 39%. 
