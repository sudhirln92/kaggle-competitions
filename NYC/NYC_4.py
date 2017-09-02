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
col = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude',
       'trip_duration',]
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
plt.figure(figsize=(14,5))
sns.heatmap(tc,annot=True,cmap='coolwarm')
plt.title("Correlation plot")

#Data preprocesing
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
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
X_test = test[Xcol]
X_train = train[Xcol]
y_train = train['trip_duration']

from sklearn.cross_validation import train_test_split

TX_train,TX_test,Ty_train,Ty_test = train_test_split(X_train,np.log(y_train+1),test_size=0.3,random_state=0)
Ty_train.head()

"""
#K fold validation
from sklearn.model_selection import KFold, cross_val_score
X = train[Xcol].values
y = (train['trip_duration'].values)

kf =KFold(n_splits=10, random_state=10)
kf.get_n_splits(X)
print(kf)

for train_index ,test_index in kf.split(X):
    print('TRAIN:',train_index,'TEST:',test_index)
    TX_train,TX_test = X[train_index],X[test_index]
    Ty_train,Ty_test = y[train_index],y[test_index]

# RMSLE 
def rmsle(y_train,y_pred):
    return np.sqrt(np.mean((np.log(y_train+1)-np.log(y_pred+1))**2))
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

sns.distplot(y_pred)"""

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

#pred = np.exp(gbm.predict(X_test))-1

#Xgboost
import xgboost as xgb

dtrain = xgb.DMatrix(TX_train,label = Ty_train)
dvalid = xgb.DMatrix(TX_test,label = Ty_test)
dtest = xgb.DMatrix(test[Xcol])
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_pars = {'min_child_weight': 1, 'eta': 0.3, 'colsample_bytree': 0.9, 
            'max_depth': 12,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
xgb_model = xgb.train(xgb_pars, dtrain, 100, watchlist, early_stopping_rounds=3,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % xgb_model.best_score)
xgb.plot_importance(xgb_model)
# Submit solution
#pred = np.exp(model.predict(X_test))-1
pred = np.exp(xgb_model.predict(dtest))-1
submit = pd.DataFrame({'id':test['id'],'trip_duration':pred})
submit.to_csv('nyc_predict.csv',index=False)
submit.head()

