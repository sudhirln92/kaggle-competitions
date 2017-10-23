#New York City Taxi Trip Duration
#https://www.kaggle.com/c/nyc-taxi-trip-duration
#Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics



#Import data
train_main = pd.read_csv('train.csv',parse_dates=[2,3],dtype={'id':np.object,'vender_id':np.uint8,'passenger_count':np.uint8,
                                                        'pickup_longitude':np.float32,'pickup_latitude':np.float32,
                                                        'dropoff_longitude':np.float32,'dropoff_latitude':np.float32,
                                                        'store_and_fwd_flag':'category','trip_duration':np.int32})
test_main = pd.read_csv('test.csv',parse_dates=[2],dtype={'id':np.object,'vender_id':np.uint8,'passenger_count':np.uint8,
                                                        'pickup_longitude':np.float32,'pickup_latitude':np.float32,
                                                        'dropoff_longitude':np.float32,'dropoff_latitude':np.float32,
                                                        'store_and_fwd_flag':'category'})
print('Number of rows & columns in Train & test', train_main.shape,test_main.shape)

train_fr1 = pd.read_csv('fastest_routes_train_part_1.csv')
train_fr2 = pd.read_csv('fastest_routes_train_part_2.csv')
train_frmain = pd.concat([train_fr1,train_fr2])
train_fr =train_frmain[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
train = pd.merge(train_main,train_fr,on='id',how='left')
print('Number of rows & columns in Train',train.shape)

test_frmain = pd.read_csv('fastest_routes_test.csv')
test_fr = test_frmain[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
test = pd.merge(test_main,test_fr,on='id',how='left')
print('Number of rows & columns in Test ',test.shape)

del train_fr1 , train_fr2 ,train_frmain,train_fr,test_fr,test_frmain,train_main,test_main
#weather = pd.read_csv('weather_data_nyc_centralpark_2016.csv')
#accident = pd.read_csv('accidents_2016.csv')

#'../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv'
#'../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv'
#../input/nyc2016holidays/NYC_2016Holidays.csv
train.head()
test.info()

#missing values
train.isnull().sum()
train.dropna(inplace =True)

# # Data exploratio
#Date Time exploration
def datetime_con(data):
    data['pu_dayofYr'] = data['pickup_datetime'].dt.dayofyear
    data['pu_month'] = data['pickup_datetime'].dt.month
    data['pu_hour'] = data['pickup_datetime'].dt.hour
    data['pu_wday']  = data['pickup_datetime'].dt.dayofweek
    data['pu_minute'] = data['pickup_datetime'].dt.minute
    data['is_weekend'] = (data['pickup_datetime'].dt.dayofweek //4==1).astype(int)
    data['is_satsun'] = ((data['pickup_datetime'].dt.dayofweek ==5) | (data['pickup_datetime'].dt.dayofweek == 6)).astype(int)
    data['date'] = data['pickup_datetime'].dt.date
    #data['pu_wkofyear'] = data['pickup_datetime'].dt.weekofyear
    #data['is_weekend'] = data

datetime_con(train)
datetime_con(test)
train.head()

#temp = pd.merge(train,weather,on='date',how='left')
#temp.head()

#Check for latitude logitude bound
print('Latitude bound: {} to {}'.format(max(train['pickup_latitude'].min(),train['dropoff_latitude'].min()),
                                          max( train['pickup_latitude'].max(), train['dropoff_latitude'].max())))

print('Longitude bound: {} to {}'.format(max(train['pickup_longitude'].min(), train['dropoff_longitude'].min()),
                                        max(train['pickup_longitude'].max(),train['dropoff_longitude'].max())))

#Check for passenger count
print('Passenger:{} to {}'.format(train['passenger_count'].min(),train['passenger_count'].max()))

#Check for trip duration
print('Trip duration in seconds: {} to{}'.format(train['trip_duration'].min(), train['trip_duration'].max()))

#Check for datetime
print('Date and time: {} to {}'.format(train['pickup_datetime'].min(),train['dropoff_datetime'].max()))

plt.figure(figsize=(10,5))
sns.distplot(train['trip_duration'])

#apply log on traget variable
#train['trip_duration'] = np.log10(train)
plt.figure(figsize=(14,5))
sns.distplot(np.log10(train['trip_duration']))

# As trip duration is  then location 
#Remove outlier in dataset

def outlier(df,col):
    lowq , highq =1,99
    count,a,b =0,0,0
    for i in col:
        lowdp, highdp = np.percentile(train[i],[lowq , highq])
        a = df.shape[0]
        df_temp = df[df[i]<highdp]
        df_temp = df_temp[df_temp[i]>lowdp]
        df = df_temp
        b = df.shape[0]
        print('Removed outlier in {} column'.format(i))
        count += a-b
    print('Number of outlier data removed',count)

train.columns
#col = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude',
 #      'trip_duration',]
col =['trip_duration',]
outlier(train,col)

train.shape 

#Determine distance 
#https://rosettacode.org/wiki/Haversine_formula
def haversine(df,columns):
    lat1, lon1, lat2, lon2 = columns
      
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(df[lat2] - df[lat1])
    dLon = np.radians(df[lon2] - df[lon1])
    lat1 = np.radians(df[lat1])
    lat2 = np.radians(df[lat2])
    
    
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
 
    return R * c

#Ecludian distance 
def ecludi(df,columns):
    lat1, lon1, lat2, lon2 = columns
    d = np.sqrt((df[lat2]-df[lat1])**2 + (df[lon2]-df[lon1])**2)
    return d

# Bearing distance
def bearing(df,columns):
    lat1,lon1,lat2,lon2  = columns
    
    #R = 6372.8 #Earth radius in kilometers
    dlon = np.radians(df[lon2]-df[lon2])
    lat1,lon1,lat2,lon2 = map(np.radians , (df[lat1],df[lon1],df[lat2],df[lon2]))
    y = np.cos(lat2)*np.sin(dlon)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    b = np.degrees(np.arctan2(y,x))
    return b

cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']

train['haversine_distances'] = haversine(train,cols)
test['haversine_distances'] = haversine(test,cols)

train['eclud_distances'] = ecludi(train,cols)
test['eclud_distances'] = ecludi(test,cols)

train['bearing_angle'] = bearing(train,cols)
test['bearing_angle'] = bearing(test,cols)
#distances = train[cols].apply(lambda x: haversine(x),axis = 1)
#train['haversine_distances'] = distances.copy()
train.head()


#Exploratory Data Analysis
train['haversine_distances'].describe()

sns.distplot(train['trip_duration'])

sns.distplot(train['total_distance'])
plt.figure(figsize=(10,5))
sns.distplot(train['pu_dayofYr'])
plt.xlabel('Day of week')

sns.distplot(train['bearing_angle'])

plt.figure(figsize=(10,5))
sns.countplot(x='pu_month',data=train)
plt.xlabel('Month')
plt.title('Trip count per month')


plt.figure(figsize=(14,5))
sns.countplot(x='pu_hour',data=train)
plt.xlabel('Hour: 0-23')
plt.title('Number of trip per hour ')


plt.figure(figsize=(12,3))
sns.countplot(x='pu_wday',data=train)
plt.xlabel('Day of week')


plt.figure(figsize=(14,3))
train.groupby('pu_hour').count()['id'].plot()
plt.xlabel('Hour: 0-23')


plt.figure(figsize=(14,3))
train.groupby('pu_month').count()['id'].plot()
plt.xlabel('Month')


plt.figure(figsize=(14,3))
train.groupby('pu_wday').count()['id'].plot()
plt.xlabel('Day of week')


sns.countplot(train['store_and_fwd_flag'])


#Pickup heatmap month vs hour
monthVShour= train.groupby(['pu_month','pu_hour']).count()['passenger_count'].unstack()

plt.figure(figsize=(16,3))
sns.heatmap(monthVShour,cmap='copper_r',linewidths=.01)
plt.xlabel('Hour, 0-23')
plt.ylabel('Month')
plt.title('Pickup heatmap, Hour Vs Month')


# Pickup Heatmap, Day of week Vs Hour
wdayVshour = train.groupby(['pu_wday','pu_hour']).count()['trip_duration'].unstack()

plt.figure(figsize=(16,3))
sns.heatmap(wdayVshour,cmap='copper_r',linewidths=0.01)
plt.xlabel('Hour: 0-23')
plt.ylabel('Month')
plt.title('Pickup Heatmap, Day of week Vs Hour')


#Pick up Heatmap, Month vs Day of week
monthVswday = train.groupby(['pu_month','pu_wday']).count()['id'].unstack()

plt.figure(figsize=(16,3))
sns.heatmap(monthVswday,linewidths=.005,cmap='copper_r')

plt.title('Pick up Heatmap, Month vs Day of week')
plt.xlabel('Month')
plt.ylabel('Day of week')


sns.pointplot(x='pu_hour',y='pu_month',data=train,hue='vendor_id')
plt.xlabel('Hour')


# Time series forcast
plt.figure(figsize=(14,5))
ts = pd.Series(np.array(train['trip_duration']),index=train['pickup_datetime'])
ts.resample('D').sum().plot()


plt.figure(figsize=(14,5))
ts2 = pd.Series(np.array(train['trip_duration']),index=train['dropoff_datetime'])
ts2.resample('D').sum().plot(color='red')


plt.figure(figsize=(14,5))
ts.resample('12H').count().plot()


plt.figure(figsize=(16,8))
train.resample('D',on='pickup_datetime').max()['pickup_longitude'].plot()
train.resample('D',on='pickup_datetime').max()['dropoff_longitude'].plot(color='red')

#Scatter plot 
train['pickup_latitude'].min()
train['pickup_latitude'].max()
train['pickup_longitude'].min()
train['pickup_longitude'].max()

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,figsize=(12,8))

ax[0].scatter(train['pickup_longitude'].values, train['pickup_latitude'].values,
            color='blue',alpha=0.1,s=1)
ax[1].scatter(train['dropoff_longitude'].values,train['dropoff_latitude'].values,
                  color='red',s=1,alpha=0.1)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
ax[0].set_xlabel('Longitude')
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')


    #Lets make cluster
    def cluster(df,k):
        pickup = ['pickup_longitude','pickup_latitude']
        dropoff = ['dropoff_longitude','dropoff_latitude']
        kmeans_pick = KMeans(n_clusters=k,n_init=1)
        kmeans_pick.fit(df[pickup])
        cluster_pick = kmeans_pick.labels_
        df['label_pick'] = cluster_pick.tolist()
        df['label_drop'] = kmeans_pick.predict(df[dropoff])
        centroid_pickups = pd.DataFrame(kmeans_pick.cluster_centers_, columns = ['centroid_pick_long', 'centroid_pick_lat'])
        centroid_dropoff = pd.DataFrame(kmeans_pick.cluster_centers_, columns = ['centroid_drop_long', 'centroid_drop_lat'])
        centroid_pickups['label_pick'] = centroid_pickups.index
        centroid_dropoff['label_drop'] = centroid_dropoff.index
        
        df = pd.merge(df, centroid_pickups, how='left', on=['label_pick'])
        df = pd.merge(df, centroid_dropoff, how='left', on=['label_drop'])
        return df
    
    train = cluster(train,100)
    test = cluster(test,100)
train.head()

#PCA tranformation for ratation 
coord = np.vstack((train[['pickup_latitude','pickup_longitude']].values,
                  train[['dropoff_latitude','dropoff_longitude']].values,
                  test[['pickup_latitude','pickup_longitude']].values,
                  test[['dropoff_latitude','dropoff_longitude']].values,
                  ))
pca = PCA().fit(coord)
train['pick_pca0'] = pca.transform(train[['pickup_latitude','pickup_longitude']])[:,0]
train['pick_pca1'] = pca.transform(train[['pickup_latitude','pickup_longitude']])[:,1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude','dropoff_longitude']])[:,0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude','dropoff_longitude']])[:,1]

test['pick_pca0'] = pca.transform(test[['pickup_latitude','pickup_longitude']])[:,0]
test['pick_pca1'] = pca.transform(test[['pickup_latitude','pickup_longitude']])[:,1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude','dropoff_longitude']])[:,0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude','dropoff_longitude']])[:,1]

#Map
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot,iplot, init_notebook_mode
init_notebook_mode(connected=True)


tc = train.corr()
plt.figure(figsize=(14,10))
sns.heatmap(tc,annot=True,cmap='coolwarm')
plt.title("Correlation plot")

#Data preprocesing
def encode(data):
    le = LabelEncoder()
    data['store_and_fwd_flag'] = le.fit_transform(data['store_and_fwd_flag'])
encode(train)
encode(test)

# # Feature selection
train.columns

Xcol = [ 'vendor_id','passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 
        'dropoff_latitude', 'total_distance','total_travel_time','number_of_steps','store_and_fwd_flag','pu_dayofYr', 
        'pu_month','pu_hour', 'pu_wday','pu_minute','is_weekend','is_satsun','haversine_distances',
        'bearing_angle','label_pick', 'label_drop', 'centroid_pick_long','centroid_pick_lat', 'centroid_drop_long', 'centroid_drop_lat'
       ,'pick_pca0', 'pick_pca1', 'dropoff_pca0', 'dropoff_pca1']
X_test = test[Xcol].values

"""
X_train = train[Xcol]
y_train = train['trip_duration']

from sklearn.cross_validation import train_test_split

TX_train,TX_test,Ty_train,Ty_test = train_test_split(X_train,np.log(y_train+1),test_size=0.3,random_state=0)
Ty_train.head()


# Model building
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(TX_train,(Ty_train))

y_pred = model.predict(TX_test)
y_pred = (y_pred)
print(y_pred[0:5])

print('R**2:',model.score(TX_test,Ty_test))
cross_val_score(model,X,y,cv=kf,n_jobs=1) """

"""#Building optimal model using backward elimination
Xcol = ['vendor_id','passenger_count','total_distance','total_travel_time','number_of_steps',
        'pu_hour', 'pu_wday','is_weekend','is_satsun','haversine_distances','label_pick', 'label_drop',
        'dropoff_pca0']
import statsmodels.formula.api as sm

X = np.append(arr=np.ones((X.shape[0],1)).astype(int),values= X,axis=1)
X_opt = X
model_ols = sm.OLS(endog=(y),exog=X_opt).fit()
model_ols.summary()

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import confusion_matrix
rf  = RandomForestRegressor(n_estimators=10,verbose=True)
rf.fit(Tx_train,np.log(Ty_train+1))
rf.feature_importances_
features = pd.Series(rf.feature_importances_,Xcol).sort_values(ascending=False)
features.plot(kind='bar')

y_pred = np.log(rf.predict(Tx_test))-1

print('RMSLE:',rmsle(Ty_test,y_pred))

sns.distplot(y_pred)



#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(loss='ls',learning_rate=0.3, max_depth=6,
                                n_estimators=10,random_state=10,
                                verbose=1)
gbm.fit(TX_train,np.log(Ty_train+1))

features = pd.Series( gbm.feature_importances_ , Xcol).sort_values(ascending=False)
features.plot(kind='bar')
y_pred = np.exp(gbm.predict(TX_test))-1
print('RMSLE for GBM:',rmsle(Ty_test,y_pred))

#pred = np.exp(gbm.predict(X_test))-1"""

#Light GBM
def  runLGB(train_X,train_y, val_X,val_y, test_X, eta =0.05, max_depth =5, 
            min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, 
            early_stopping_rounds = 100, seeds_val=2017):
    
    params = {'objective':'regression', 
              'metric':'l2_root',
              'learning_rate':eta,
              'min_child_weight':min_child_weight,
              'bagging_fraction':subsample,
              'bagging_seed':seeds_val,
              'feature_fraction':colsample,
              'verbosity':0,              
              'max_depth':max_depth,
              #'reg_lambda':2
              'nthread':-1} 
    lgtrain = lgb.Dataset(train_X,label= train_y)
    lgval =lgb.Dataset(val_X,label=val_y)
    model = lgb.train(params, lgtrain, num_rounds, valid_sets=lgval,
                      early_stopping_rounds=early_stopping_rounds, 
                      verbose_eval=20)
    pred_val = model.predict(val_X,num_iteration=model.best_iteration)
    pred_test = model.predict(test_X,num_iteration=model.best_iteration)
    return pred_val,pred_test,model

#K fold validation
X = train[Xcol].values
y = np.log(train['trip_duration'].values+1)

kf =KFold(n_splits=5, random_state=5,shuffle=True)
cv_scores = []
pred_test_full =0
pred_val_full = np.zeros(train.shape[0])

for train_index ,test_index in kf.split(X):
    print('TRAIN:',train_index,'TEST:',test_index)
    TX_train,TX_test = X[train_index],X[test_index]
    Ty_train,Ty_test = y[train_index],y[test_index]
    pred_val,pred_test, model = runLGB(TX_train,Ty_train,TX_test,Ty_test,X_test,eta=0.1,
                                       num_rounds=10000,max_depth=8)
    pred_val_full[test_index] = pred_val
    pred_test_full +=pred_test
    cv_scores.append(np.sqrt(metrics.mean_squared_error(Ty_test, pred_val)))

print(cv_scores)
print('Mean cv score:',np.mean(cv_scores))
lgb.plot_importance(model)
pred_test_full = pred_test_full/5
pred_lgb = np.exp(pred_test_full)-1
submit = pd.DataFrame({'id':test['id'],'trip_duration':pred_lgb})
submit.to_csv('nyc_predict.csv',index=False)
submit.head()
# RMSLE 
def rmsle(y_train,y_pred):
    return np.sqrt(np.mean((np.log(y_train+1)-np.log(y_pred+1))**2))


#Xgboost


def runXGB(train_X,train_y,val_x,val_y, test_X, num_rounds=1000, eta=0.3,max_depth=5,
           min_child_weight=1,subsample=0.8,colsample=0.7,
           early_stopping_rounds=50,seeds_val=2017):
    
    params = {'objective':'reg:linear',
              'booster':'gbtree',
              'eta':eta,
              'subsample':subsample,
              'colsamaple_bytree':colsample,
              'max_depth':max_depth,
              'min_child_weight':min_child_weight,
              'eval_metric':'rmse',
              #'early_stoping_rounds':early_stoping_rounds,
              'seeds':seeds_val,
              #verbose_eval=20
              'silent':1,
              'lambda':2,
              'nthread':-1 }
    pslt = list(params.items())
    xgtrain = xgb.DMatrix(train_X,label=train_y)
    xgvalid = xgb.DMatrix(val_x,label=val_y)
    xgtest = xgb.DMatrix(test_X)
    watchlist = [(xgtrain,'train'),(xgvalid,'test')]
    model = xgb.train(pslt,xgtrain, num_rounds,watchlist,verbose_eval=10,early_stopping_rounds=early_stopping_rounds)
    pred_val = model.predict(xgvalid,ntree_limit=model.best_ntree_limit)
    pred_test = model.predict(xgtest,ntree_limit=model.best_ntree_limit)
    
    return pred_val, pred_test

# K flod and model
kf1 = KFold(n_splits=5,shuffle=True,random_state=2017)

cv_scores1 = []                                                                                                                                                                                                 
pred_test_full1 = 0
pred_val_full1 = np.zeros(train.shape[0])

for train_index1,test_index1 in kf1.split(X):
    train_X1, train_y1 = X[train_index1], y[train_index1]
    val_X1 , val_y1 = X[test_index1],y[test_index1]
    pred_val1,pred_test1 = runXGB(train_X1,train_y1,val_X1,val_y1,test[Xcol],
                                         num_rounds=50,eta=0.3,max_depth=5)
    pred_val_full1[test_index1] = pred_val1
    pred_test_full1 +=pred_test1
    cv_scores1.append(np.sqrt(metrics.mean_squared_error(val_y1,pred_val1)))

print(cv_scores1)
print('Mean cv score :',np.mean(cv_scores1))

pred_test_full1 = pred_test_full1/5
pred_xgb = np.exp(pred_test_full1)-1

submit1 = pd.DataFrame({'id':test['id'],'trip_duration':pred_xgb})
submit1.to_csv('nyc_predict1.csv',index=False)

submit1.head()

#XGBOOST
pred_val1,pred_test1 = runXGB(TX_train,Ty_train,TX_test,Ty_test,test[Xcol],
                                         num_rounds=50,eta=0.3,max_depth=8)

pred_xgb = np.exp(pred_test1)-1    


#Ensemble
ensemble = pred_xgb*0.4+pred_lgb*0.6
submit1 = pd.DataFrame({'id':test['id'],'trip_duration':ensemble})
submit1.to_csv('nyc_predict1.csv',index=False)

submit1.head()

#Grid SEARCH
from sklearn.grid_search import GridSearchCV

cv_params = {'max_depth':[5,7,9],'min_child_weight':[1,3,5]}
ind_params = {'learning_rate':0.3,'n_estimators':100,'seed':0,'subsample':0.8,
              'colsample_bytree':0.8,'objective':'reg:linear'}
model = lgb.LGBMRegressor(colsample_bytree=1, learning_rate=0.1,
       max_bin=255, max_depth=1, min_child_samples=10, min_child_weight=5,
       min_split_gain=0, n_estimators=100, nthread=-1,
       objective='reg:linear', reg_alpha=0, reg_lambda=0, seed=10,
       silent=True, subsample=1 )
gdsearch = GridSearchCV(model,cv_params,
                             scoring='roc_auc',cv=5,n_jobs=1,verbose=1)
gdsearch.fit(TX_train,Ty_train)

optimized_GBM.grid_scores_

#2 trail
cv_params = {'learnig_rate':[0.3,0.1,0.01],'subsample':[0.7,0.8,0.9]}
ind_params = {'n_estimator':1000,} 

import ...

if __name__=='__main__':
    cv_params = {'max_depth':[5,7,9],'min_child_weight':[1,3,5]}
    ind_params = {'learning_rate':0.3,'n_estimators':10,'seed':0,'subsample':0.8,
              'colsample_bytree':0.8,'objective':'reg:linear'}
    optimized_GBM = GridSearchCV(lgb.LGBMRegressor(objective='reg:linear',learning_rate=0.1),cv_params,
                             scoring='accuracy',cv=5,n_jobs=1,verbose=1)
    optimized_GBM.fit(X,y)

    
