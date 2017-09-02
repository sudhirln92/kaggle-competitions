
#Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans

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

#'../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv'
#'../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv'
#../input/nyc2016holidays/NYC_2016Holidays.csv


train.head()

test.info()

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
    #data['pu_wkofyear'] = data['pickup_datetime'].dt.weekofyear
    #data['is_weekend'] = data

datetime_con(train)
datetime_con(test)
train.head()

plt.figure(figsize=(10,5))
sns.distplot(train['trip_duration'])

#apply log on traget variable
#train['trip_duration'] = np.log10(train)
plt.figure(figsize=(14,5))
sns.distplot(np.log10(train['trip_duration']))


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


cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']

train['haversine_distances'] = haversine(train,cols)
test['haversine_distances'] = haversine(test,cols)

train['eclud_distances'] = ecludi(train,cols)
test['eclud_distances'] = ecludi(test,cols)
#distances = train[cols].apply(lambda x: haversine(x),axis = 1)
#train['haversine_distances'] = distances.copy()
train.head()


train['haversine_distances'].describe()

#Remove outlier
lowq , highq =1,99

#Remove outlier data haversine_distances and Trip duration
lowdp, highdp = np.percentile(train['haversine_distances'],[lowq , highq])
print('Quartile of Haversine Distance: {} to {}'.format(lowdp, highdp))

train_temp = train[train['haversine_distances'] < highdp]
train_temp = train_temp[train_temp['haversine_distances'] > lowdp]

lowdp, highdp = np.percentile(train['trip_duration'],[lowq , highq])
print('Quartile of Trip duration: {} to {}'.format(lowdp, highdp))

train_temp = train_temp[train_temp['trip_duration'] < highdp]
train_temp = train_temp[train_temp['trip_duration'] > lowdp]

print('Number of outlier data removed: {}'.format( train.shape[0]-train_temp.shape[0]))
train = train_temp
train.shape


sns.distplot(train['trip_duration'])


plt.figure(figsize=(10,5))
sns.distplot(train['pu_dayofYr'])
plt.xlabel('Day of week')


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


def assign_cluster(df, k):
    """function to assign clusters """
    df_pick = df[['pickup_longitude','pickup_latitude']]
    df_drop = df[['dropoff_longitude','dropoff_latitude']]
    #df = df.dropna()
    init = np.array([[ -73.98737616,   40.72981533],
       [-121.93328857,   37.38933945],
       [ -73.78423222,   40.64711269],
       [ -73.9546417 ,   40.77377538],
       [ -66.84140269,   36.64537175],
       [ -73.87040541,   40.77016484],
       [ -73.97316185,   40.75814346],
       [ -73.98861094,   40.7527791 ],
       [ -72.80966949,   51.88108444],
       [ -76.99779701,   38.47370625],
       [ -73.96975298,   40.69089596],
       [ -74.00816622,   40.71414939],
       [ -66.97216034,   44.37194443],
       [ -61.33552933,   37.85105133],
       [ -73.98001393,   40.7783577 ],
       [ -72.00626526,   43.20296402],
       [ -73.07618713,   35.03469086],
       [ -73.95759366,   40.80316361],
       [ -79.20167796,   41.04752096],
       [ -74.00106031,   40.73867723]])
    k_means_pick = KMeans(n_clusters=k, init=init, n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick = k_means_pick.labels_
    df['label_pick'] = clust_pick.tolist()
    df['label_drop'] = k_means_pick.predict(df_drop)
    return df, k_means_pick


train_cl, k_means = assign_cluster(train, 20)  # make it 100 when extracting features 
centroid_pickups = pd.DataFrame(k_means.cluster_centers_, columns = ['centroid_pick_long', 'centroid_pick_lat'])
centroid_dropoff = pd.DataFrame(k_means.cluster_centers_, columns = ['centroid_drop_long', 'centroid_drop_lat'])
centroid_pickups['label_pick'] = centroid_pickups.index
centroid_dropoff['label_drop'] = centroid_dropoff.index
#centroid_pickups.head()
train_cl = pd.merge(train_cl, centroid_pickups, how='left', on=['label_pick'])
train_cl = pd.merge(train_cl, centroid_dropoff, how='left', on=['label_drop'])


test_cl, kmeans = assign_cluster(test,20)
centroid_pickups = pd.DataFrame(k_means.cluster_centers_,columns=['centroid_pick_long','centroid_pick_lat'])
centroid_dropoff = pd.DataFrame(k_means.cluster_centers_,columns=['centroid_drop_long','centroid_drop_lat'])
centroid_pickups['label_pick'] = centroid_pickups.index
centroid_dropoff['label_drop'] = centroid_dropoff.index
test_cl = pd.merge(test_cl, centroid_pickups, how='left', on=['label_pick'])
test_cl = pd.merge(test_cl, centroid_dropoff, how='left', on=['label_drop'])


train = train_cl
test  = test_cl
train.head()


#Map
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot,iplot, init_notebook_mode
init_notebook_mode(connected=True)


tc = train.corr()
plt.figure(figsize=(14,5))
sns.heatmap(tc,annot=True,cmap='coolwarm')
plt.title("Correlation plot")


from sklearn.preprocessing import OneHotEncoder , LabelEncoder
def encode(data):
    le = LabelEncoder()
    data['store_and_fwd_flag'] = le.fit_transform(data['store_and_fwd_flag'])
encode(train)
encode(test)


# # Feature selection

train.columns


Xcol = [ 'passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 
        'dropoff_latitude', 'total_distance','total_travel_time','number_of_steps','store_and_fwd_flag','pu_dayofYr', 
        'pu_month','pu_hour', 'pu_wday','pu_minute','is_weekend','is_satsun','haversine_distances','eclud_distances',
       'label_pick', 'label_drop', 'centroid_pick_long','centroid_pick_lat', 'centroid_drop_long', 'centroid_drop_lat']
X_train = train[Xcol]
y_train = train['trip_duration']

X_test = test[Xcol]

from sklearn.cross_validation import train_test_split

Tx_train,Tx_test,Ty_train,Ty_test = train_test_split(X_train,y_train,test_size=0.3,random_state=0)
Ty_train.head()


# RMSLE 
def rmsle(y_train,y_pred):
    return np.sqrt(np.mean((np.log(Ty_test+1)-np.log(y_pred+1))**2))
# Model building

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Tx_train,(Ty_train))


y_pred = model.predict(Tx_test)
y_pred = (y_pred)
print(y_pred[0:5])

print('R**2:',model.score(Tx_test,Ty_test))


# Transform log base 10 value to linear
#y_pred = 10 **y_pred
#y_pred[0:5]


#Model Evaluvation
SS_Residual = sum((Ty_test-y_pred)**2)
SS_Total = sum((Ty_test-np.mean(Ty_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
print ('R**2',r_squared)

adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print('Adjusted R**2',adjusted_r_squared)

plt.scatter(Ty_test,y_pred,c=('r','b'))

#Building optimal model using backward elimination
import statsmodels.formula.api as sm

X = np.append(arr=np.ones((X_train.shape[0],1)).astype(int),values= X_train.values,axis=1)
X_opt = X
model_ols = sm.OLS(endog=(y_train.values),exog=X_opt).fit()
model_ols.summary()
#model_ols.rsquared
# NO need of backward elimination

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
gbm = GradientBoostingRegressor(loss='ls',learning_rate=0.5,
                                n_estimators=10,random_state=10,
                                verbose=1)
gbm.fit(Tx_train,Ty_train)

# Submit solution
#pred = np.exp(model.predict(X_test))-1
pred = np.sqrt(rf.predict(X_test)**2)
submit = pd.DataFrame({'id':test['id'],'trip_duration':pred})
submit.to_csv('nyc_predict.csv',index=False)
submit.head()

