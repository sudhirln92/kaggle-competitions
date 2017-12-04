# Porto Seguro’s Safe Driver Prediction					Monday 04 December 2017 03:58:30 PM IST

# Porto Seguro Safe Driver Prediction 
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
	Porto Sergo Is Largest Auto And Homeowner Insurance Company.
The Aim Of This Competition Is To Predict Probability That A Driver Will Intiate An Auto Insurance Claim Next Year. A More Accurate Prediction Will Allow Them To Further Tailor Their Prices, And Hopefully Make Auto Insurance Coverage More Accessible To More Drivers.


# Data set Description
The train and test data set contains feature with similar grouping are tagged with (e.g., ind, reg, car, cat, calc, bin). Values of  -1 indicate that the feature was missing from the observation. The target column in data set is whether or not claim was filed for that policy holder.

The train,test data set contains 595212 and 892816 observations, 59 feature.  The labels of data set were change by hosting team.

# Exploratory Data analysis
The feature name contains tag ind,car, contains category data type, there are few of them also numeric type. The feature contains bin tag is binary feature, feature having tag calc  is zero correlation with other variables, it has been drop from data set. 
About 2.5% of values are missing in total in eacho of the train and test data sets

# Feature Extraction
One hot encoding is performed on categorical variable. 
Mean and median of all variable is determined and add as new column in data set. 

# Modeling
Logistic regression,Xgboost,Lightgbm algorithms are used

# Data validation
The model is evaluated on AUC, gini metric.
 
# Out come of project:
This project give good opportunity to explore data set. Create new feature from data  set.    
I have submitted prediction more than 70 to kaggle wesite. My best gini score 0.282 on public LB and 0.287 on private LB. Overall my standing at end of competition is top 39%. 
