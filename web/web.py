#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:11:04 2017

@author: sudhir
https://www.kaggle.com/c/web-traffic-time-series-forecasting
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re #Regural expression for 
from fbprophet import Prophet

N=60 
i = 1800 #one sample time series to train
#Import data set
all_data = pd.read_csv('train_1.csv').T
all_data.head()
key = pd.read_csv('key_1.csv')

#Handle NA
train, test = all_data.iloc[0:-N,:], all_data.iloc[-N:,:]
train_cleaned = train.T.fillna(method='ffill').T
test_cleaned = test.T.fillna(method='ffill').T

data = train_cleaned.iloc[1:,i].to_frame()
data.columns = ['visit']
data['mean'] = pd.Series.rolling(data['visit'],50,min_periods=1).mean()
#std_mult = 1.5
#data.ix[np.abs(data.visit-data.visit.mean())>=(std_mult*data.visits.std()),'visit'] = data.ix[np.abs(data.visit-data.visit.mean())>=(std_mult*data.visit.std()),'mean']
#data['visit'][np.abs(data['visit']-data['visit'].mean()) >= (std_mult*data['visit'].std())] =data['mean'][np.abs(data['visit']-data['visit'].mean()) >= (std_mult*data['visit'].std())]

print(data.tail())

#prophet label
X = pd.DataFrame(index = range(0,len(data)))
X['ds'] = data.index 
X['y'] = data['visit'].values.astype('float')
X.tail()

#Prophet
m = Prophet(yearly_seasonality=True)
m.fit(X)

# Python
future = m.make_future_dataframe(periods=N)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast);
