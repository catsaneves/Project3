# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:23:53 2022

@author: Sofia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn import  metrics

from sklearn import  linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor

import pickle

import math

excel_file = pd.ExcelFile('bp-stats-review-2021-all-data.xlsx')
#print(excel_file.sheet_names)

#Note: in order to ease the comparision between the different variables being studied we will express everything in units of TWh
#1 Exajoule [EJ] = 277.777 777 777 78 Terawatt hour [TWh]

#-------------------------------------DATA FOR CONSUMPTIONS-------------------------------------

oil_reserves = pd.read_excel(excel_file, sheet_name=9,header=[2],index_col=0)
oil_reserves = oil_reserves[oil_reserves.columns.drop(list(oil_reserves.filter(regex='Unnamed:')))]
oil_reserves.dropna(axis = 0, how = 'all', inplace = True)
#print(oil_reserves)
oil_reserves=oil_reserves.loc['Total World']
oil_reserves=oil_reserves[0:len(oil_reserves)-3]
#print(oil_reserves)

gas_reserves = pd.read_excel(excel_file, sheet_name=28,header=[2],index_col=0)
gas_reserves = gas_reserves[gas_reserves.columns.drop(list(gas_reserves.filter(regex='Unnamed:')))]
gas_reserves.dropna(axis = 0, how = 'all', inplace = True)
#print(gas_reserves)
gas_reserves=gas_reserves.loc['Total World']
gas_reserves=gas_reserves[0:len(gas_reserves)-3]
#print(gas_reserves)

cobalt_reserve = pd.read_excel(excel_file, sheet_name=72,header=[3],index_col=0)
cobalt_reserve = cobalt_reserve[cobalt_reserve.columns.drop(list(cobalt_reserve.filter(regex='Unnamed:')))]
cobalt_reserve.dropna(axis = 0, how = 'all', inplace = True)
#print(cobalt_reserve)
cobalt_reserve=cobalt_reserve.loc['Total World']
cobalt_reserve=cobalt_reserve[0:len(cobalt_reserve)-6]
#print('printing cobalt reserves...',cobalt_reserve)

lithium_reserves = pd.read_excel(excel_file, sheet_name=73,header=[2],index_col=0)
lithium_reserves = lithium_reserves[lithium_reserves.columns.drop(list(lithium_reserves.filter(regex='Unnamed:')))]
lithium_reserves.dropna(axis = 0, how = 'all', inplace = True)
#print(lithium_reserves)
lithium_reserves=lithium_reserves.loc['Total World']
lithium_reserves=lithium_reserves[0:len(lithium_reserves)-6]
#print(lithium_reserves)

graphite_reserves = pd.read_excel(excel_file, sheet_name=74,header=[2],index_col=0)
graphite_reserves = graphite_reserves[graphite_reserves.columns.drop(list(graphite_reserves.filter(regex='Unnamed:')))]
graphite_reserves.dropna(axis = 0, how = 'all', inplace = True)
#print(graphite_reserves)
graphite_reserves=graphite_reserves.loc['Total World']
graphite_reserves=graphite_reserves[0:len(graphite_reserves)-6]
#print(graphite_reserves)

raremetals_reserves = pd.read_excel(excel_file, sheet_name=75,header=[2],index_col=0)
raremetals_reserves = raremetals_reserves[raremetals_reserves.columns.drop(list(raremetals_reserves.filter(regex='Unnamed:')))]
raremetals_reserves.dropna(axis = 0, how = 'all', inplace = True)
#print(raremetals_reserves)
raremetals_reserves=raremetals_reserves.loc['Total World']
raremetals_reserves=raremetals_reserves[0:len(raremetals_reserves)-6]
#print(raremetals_reserves)

#-------------------------------------DATA FOR CONSUMPTIONS-------------------------------------
#all data is in Twh
#we should first find if there's any empty dataframe and disregard it
#print(oil_reserves)
#print(gas_reserves)
#print(cobalt_reserve)
#print(lithium_reserves)
#print(graphite_reserves)
#print(raremetals_reserves)


if (oil_reserves == 0).all():
    print('this dataframe is empty - oil')
if (gas_reserves == 0).all():
    print('this dataframe is empty - gas')
if (cobalt_reserve == 0).all():
    print('this dataframe is empty - cobalt')
if (lithium_reserves == 0).all():
    print('this dataframe is empty - lithium')
if (graphite_reserves == 0).all():
    print('this dataframe is empty - graphite')
if (raremetals_reserves == 0).all():
    print('this dataframe is empty - rare metals')
    
#we will now exclude the nuclear dataframe

#now let's make a dataframe with all the types of energy

print(type(oil_reserves)) #all of the data is a pandas.core.series.Series

oil_reserves=pd.DataFrame(oil_reserves)
oil_reserves.rename(columns={'Total World':'Oil'},inplace=True)
oil_reserves['year']=oil_reserves.index
oil_reserves.set_index('year',inplace=True)
gas_reserves=pd.DataFrame(gas_reserves)
gas_reserves.rename(columns={'Total World':'Gas'},inplace=True)
gas_reserves['year']=gas_reserves.index
gas_reserves.set_index('year',inplace=True)
cobalt_reserve=pd.DataFrame(cobalt_reserve)
cobalt_reserve.rename(columns={'Total World':'Cobalt'},inplace=True)
cobalt_reserve['year']=cobalt_reserve.index
cobalt_reserve.set_index('year',inplace=True)
lithium_reserves=pd.DataFrame(lithium_reserves)
lithium_reserves.rename(columns={'Total World':'Lithium'},inplace=True)
lithium_reserves['year']=lithium_reserves.index
lithium_reserves.set_index('year',inplace=True)
graphite_reserves=pd.DataFrame(graphite_reserves)
graphite_reserves.rename(columns={'Total World':'Graphite'},inplace=True)
graphite_reserves['year']=graphite_reserves.index
graphite_reserves.set_index('year',inplace=True)
raremetals_reserves=pd.DataFrame(raremetals_reserves)
raremetals_reserves.rename(columns={'Total World':'Rare Metals'},inplace=True)
raremetals_reserves['year']=raremetals_reserves.index
raremetals_reserves.set_index('year',inplace=True)

data_reserves=pd.merge(oil_reserves,gas_reserves, on='year',how='outer')
data_reserves=pd.merge(data_reserves,cobalt_reserve, on='year',how='outer')
data_reserves=pd.merge(data_reserves,lithium_reserves, on='year',how='outer')
data_reserves=pd.merge(data_reserves,graphite_reserves, on='year',how='outer')
data_reserves=pd.merge(data_reserves,raremetals_reserves, on='year',how='outer')
print(data_reserves)

data_reserves['year']=data_reserves.index
        
Z=data_reserves.values
X=Z[:,6] #time
Y_oil=Z[:,0] 
Y_gas=Z[:,1]
Y_cobalt=Z[:,2] 
Y_lithium=Z[:,3]
Y_graphite=Z[:,4] 
Y_rare=Z[:,5]

to_be_delete=[]
for i in range(len(Y_cobalt)):
    if math.isnan(Y_cobalt[i]):
        #print('NaN found')
        to_be_delete.append(i)
Y_cobalt=np.delete(Y_cobalt, to_be_delete)
X_cobalt=np.delete(X, to_be_delete)
to_be_delete=[]
for i in range(len(Y_lithium)):
    if math.isnan(Y_lithium[i]):
        #print('NaN found')
        to_be_delete.append(i)
Y_lithium=np.delete(Y_lithium, to_be_delete)
X_lithium=np.delete(X, to_be_delete)
to_be_delete=[]
for i in range(len(Y_graphite)):
    if math.isnan(Y_graphite[i]):
        #print('NaN found')
        to_be_delete.append(i)
Y_graphite=np.delete(Y_graphite, to_be_delete)
X_graphite=np.delete(X, to_be_delete)
to_be_delete=[]
for i in range(len(Y_rare)):
    if math.isnan(Y_rare[i]):
        #print('NaN found')
        to_be_delete.append(i)
Y_rare=np.delete(Y_rare, to_be_delete)
X_rare=np.delete(X, to_be_delete)
print('the values for X are',X)


X_oil_train, X_oil_test, y_oil_train, y_oil_test = train_test_split(X,Y_oil)
X_gas_train, X_gas_test, y_gas_train, y_gas_test = train_test_split(X,Y_gas)
X_cobalt_train, X_cobalt_test, y_cobalt_train, y_cobalt_test = train_test_split(X_cobalt,Y_cobalt)
X_lithium_train, X_lithium_test, y_lithium_train, y_lithium_test = train_test_split(X_lithium,Y_lithium)
X_graphite_train, X_graphite_test, y_graphite_train, y_graphite_test = train_test_split(X_graphite,Y_graphite)
X_rare_train, X_rare_test, y_rare_train, y_rare_test = train_test_split(X_rare,Y_rare)

## autoregressive
X_train=[X_oil_train,X_gas_train,X_cobalt_train,X_lithium_train,X_graphite_train,X_rare_train]
X_test=[X_oil_test,X_gas_test,X_cobalt_test,X_lithium_test,X_graphite_test,X_rare_test]
y_train=[y_oil_train,y_gas_train,y_cobalt_train,y_lithium_train,y_graphite_train,y_rare_train]
y_test=[y_oil_test,y_gas_test,y_cobalt_test,y_lithium_test,y_graphite_test,y_rare_test]
labels=['oil','gas','cobalt','lithium','graphite','rare_metals']

#print(y_energy_train)
#print(y_train[0].reshape(-1, 1))

data=data_reserves
data['year'] = pd.to_datetime(data['year'], format='%Y')
data = data.set_index('year')
data = data.asfreq('YS')
data = data.sort_index()
print(data.head())

path=''
## linear regression
print('linear regressor')
Y_predict_LR=[]
MAE_LR=[]
MSE_LR=[]
RMSE_LR=[]
cvRMSE_LR=[]
LR=[]
for i in range(2):
    print('doing linear regression models ',labels[i])
    regr = linear_model.LinearRegression()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1))
    Y_predict_LR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_LR.append(metrics.mean_absolute_error(y_test[i],Y_predict_LR[i]))
    MSE_LR.append(metrics.mean_squared_error(y_test[i],Y_predict_LR[i]))
    RMSE_LR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_LR[i])))
    cvRMSE_LR.append(RMSE_LR[i]/np.mean(y_test[i]))
    LR.append(regr)
#    with open(path+'models_consumption/LR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing linear regression models ',labels[i])
    regr = linear_model.LinearRegression()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1))
    Y_predict_LR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_LR.append(metrics.mean_absolute_error(y_test[i],Y_predict_LR[i]))
    MSE_LR.append(metrics.mean_squared_error(y_test[i],Y_predict_LR[i]))
    RMSE_LR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_LR[i])))
    cvRMSE_LR.append(RMSE_LR[i]/np.mean(y_test[i]))
    LR.append(regr)
#    with open(path+'models_consumption/LR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)

 
"""        
regr_energy = linear_model.LinearRegression()
regr_energy.fit(X_energy_train,y_energy_train)
#y_pred_LR_central = regr_central.predict(X_central_test)
with open(path+'models/model_LR_central', 'wb') as files:
    pickle.dump(regr_central, files)
"""

print('Support vector regressor')
Y_predict_SVR=[]
MAE_SVR=[]
MSE_SVR=[]
RMSE_SVR=[]
cvRMSE_SVR=[]
SVRm=[]
for i in range(len(X_train)-4):
    print('doing support vector regression models ',labels[i])
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X_train_ss = ss_X.fit_transform(X_train[i].reshape(-1, 1))
    y_train_ss = ss_y.fit_transform(y_train[i].reshape(-1, 1))
    regr = SVR(kernel='rbf')
    regr.fit(X_train_ss,y_train_ss)
    Y_predict_SVR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_SVR.append(metrics.mean_absolute_error(y_test[i],Y_predict_SVR[i]))
    MSE_SVR.append(metrics.mean_squared_error(y_test[i],Y_predict_SVR[i]))
    RMSE_SVR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_SVR[i])))
    cvRMSE_SVR.append(RMSE_LR[i]/np.mean(y_test[i]))
    SVRm.append(regr)
#    with open(path+'models_consumption/SVR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing support vector regression models ',labels[i])
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X_train_ss = ss_X.fit_transform(X_train[i].reshape(-1, 1))
    y_train_ss = ss_y.fit_transform(y_train[i].reshape(-1, 1))
    regr = SVR(kernel='rbf')
    regr.fit(X_train_ss,y_train_ss)
    Y_predict_SVR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_SVR.append(metrics.mean_absolute_error(y_test[i],Y_predict_SVR[i]))
    MSE_SVR.append(metrics.mean_squared_error(y_test[i],Y_predict_SVR[i]))
    RMSE_SVR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_SVR[i])))
    cvRMSE_SVR.append(RMSE_LR[i]/np.mean(y_test[i]))
    SVRm.append(regr)
#    with open(path+'models_consumption/SVR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
    
"""    
## Support vector regressor
ss_central_X = StandardScaler()
ss_central_y = StandardScaler()
X_central_train_ss = ss_central_X.fit_transform(X_central_train)
y_central_train_ss = ss_central_y.fit_transform(y_central_train.reshape(-1,1))
regr_central = SVR(kernel='rbf')
regr_central.fit(X_central_train_ss,y_central_train_ss)
with open(path+'models/model_SVR_central', 'wb') as files:
    pickle.dump(regr_central, files)
"""

print('Decision tree regressor')
Y_predict_DTR=[]
MAE_DTR=[]
MSE_DTR=[]
RMSE_DTR=[]
cvRMSE_DTR=[]
DTR=[]
for i in range(len(X_train)-4):
    print('doing decision tree regression models ',labels[i])
    regr = DecisionTreeRegressor()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1))
    Y_predict_DTR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_DTR.append(metrics.mean_absolute_error(y_test[i],Y_predict_DTR[i]))
    MSE_DTR.append(metrics.mean_squared_error(y_test[i],Y_predict_DTR[i]))
    RMSE_DTR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_DTR[i])))
    cvRMSE_DTR.append(RMSE_DTR[i]/np.mean(y_test[i]))
    DTR.append(regr)
#    with open(path+'models_consumption/DTR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing decision tree regression models ',labels[i])
    regr = DecisionTreeRegressor()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1))
    Y_predict_DTR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_DTR.append(metrics.mean_absolute_error(y_test[i],Y_predict_DTR[i]))
    MSE_DTR.append(metrics.mean_squared_error(y_test[i],Y_predict_DTR[i]))
    RMSE_DTR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_DTR[i])))
    cvRMSE_DTR.append(RMSE_DTR[i]/np.mean(y_test[i]))
    DTR.append(regr)
#    with open(path+'models_consumption/DTR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
    
"""
## Decision tree regressor
DT_regr_model_central = DecisionTreeRegressor()
DT_regr_model_central.fit(X_central_train, y_central_train)
with open(path+'models/model_DTR_central', 'wb') as files:
    pickle.dump(DT_regr_model_central, files)
"""

print('Random Forest regressor')
Y_predict_RFR=[]
MAE_RFR=[]
MSE_RFR=[]
RMSE_RFR=[]
cvRMSE_RFR=[]
RFR=[]
parameters = {'bootstrap': True,'min_samples_leaf': 3,'n_estimators': 200, 'min_samples_split': 15,'max_features': 'sqrt','max_depth': 20,'max_leaf_nodes': None}
for i in range(len(X_train)-4):
    print('doing random forest regression models ',labels[i])
    regr = RandomForestRegressor(**parameters)
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_RFR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_RFR.append(metrics.mean_absolute_error(y_test[i],Y_predict_RFR[i]))
    MSE_RFR.append(metrics.mean_squared_error(y_test[i],Y_predict_RFR[i]))
    RMSE_RFR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_RFR[i])))
    cvRMSE_RFR.append(RMSE_RFR[i]/np.mean(y_test[i]))
    RFR.append(regr)
#    with open(path+'models_consumption/RFR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing random forest regression models ',labels[i])
    regr = RandomForestRegressor(**parameters)
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_RFR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_RFR.append(metrics.mean_absolute_error(y_test[i],Y_predict_RFR[i]))
    MSE_RFR.append(metrics.mean_squared_error(y_test[i],Y_predict_RFR[i]))
    RMSE_RFR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_RFR[i])))
    cvRMSE_RFR.append(RMSE_RFR[i]/np.mean(y_test[i]))
    RFR.append(regr)
#    with open(path+'models_consumption/RFR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)

"""
## Random Forest regressor
parameters = {'bootstrap': True,'min_samples_leaf': 3,'n_estimators': 200, 'min_samples_split': 15,'max_features': 'sqrt','max_depth': 20,'max_leaf_nodes': None}
RF_model_central = RandomForestRegressor(**parameters)
RF_model_central.fit(X_central_train, y_central_train)
with open(path+'models/model_RFR_central', 'wb') as files:
    pickle.dump(RF_model_central, files)
"""

print('Gradient boosting regressor')
Y_predict_GBR=[]
MAE_GBR=[]
MSE_GBR=[]
RMSE_GBR=[]
cvRMSE_GBR=[]
GBR=[]
for i in range(len(X_train)-4):
    print('doing gradient boosting regression models ',labels[i])
    regr = GradientBoostingRegressor()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_GBR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_GBR.append(metrics.mean_absolute_error(y_test[i],Y_predict_GBR[i]))
    MSE_GBR.append(metrics.mean_squared_error(y_test[i],Y_predict_GBR[i]))
    RMSE_GBR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_GBR[i])))
    cvRMSE_GBR.append(RMSE_GBR[i]/np.mean(y_test[i]))
    GBR.append(regr)
#    with open(path+'models_consumption/GBR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing gradient boosting regression models ',labels[i])
    regr = GradientBoostingRegressor()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_GBR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_GBR.append(metrics.mean_absolute_error(y_test[i],Y_predict_GBR[i]))
    MSE_GBR.append(metrics.mean_squared_error(y_test[i],Y_predict_GBR[i]))
    RMSE_GBR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_GBR[i])))
    cvRMSE_GBR.append(RMSE_GBR[i]/np.mean(y_test[i]))
    GBR.append(regr)
#    with open(path+'models_consumption/GBR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)

"""
## Gradient boosting regressor
GB_model_central = GradientBoostingRegressor()
GB_model_central.fit(X_central_train, y_central_train)
with open(path+'models/model_GB_central', 'wb') as files:
    pickle.dump(GB_model_central, files)
"""

print('Bootstraping regressor')
Y_predict_BSR=[]
MAE_BSR=[]
MSE_BSR=[]
RMSE_BSR=[]
cvRMSE_BSR=[]
BSR=[]
for i in range(len(X_train)-4):
    print('doing bootstraping regression models ',labels[i])
    regr = BaggingRegressor()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_BSR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_BSR.append(metrics.mean_absolute_error(y_test[i],Y_predict_BSR[i]))
    MSE_BSR.append(metrics.mean_squared_error(y_test[i],Y_predict_BSR[i]))
    RMSE_BSR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_BSR[i])))
    cvRMSE_BSR.append(RMSE_BSR[i]/np.mean(y_test[i]))
    BSR.append(regr)
#    with open(path+'models_consumption/BSR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing bootstraping regression models ',labels[i])
    regr = BaggingRegressor()
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_BSR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_BSR.append(metrics.mean_absolute_error(y_test[i],Y_predict_BSR[i]))
    MSE_BSR.append(metrics.mean_squared_error(y_test[i],Y_predict_BSR[i]))
    RMSE_BSR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_BSR[i])))
    cvRMSE_BSR.append(RMSE_BSR[i]/np.mean(y_test[i]))
    BSR.append(regr)
#    with open(path+'models_consumption/BSR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)

"""
## Bootstraping regressor
BT_model_central = BaggingRegressor()
BT_model_central.fit(X_central_train, y_central_train)
with open(path+'models/model_BTR_central', 'wb') as files:
    pickle.dump(BT_model_central, files)
"""

print('Neural Networks regressor')
Y_predict_NNR=[]
MAE_NNR=[]
MSE_NNR=[]
RMSE_NNR=[]
cvRMSE_NNR=[]
NNR=[]
for i in range(len(X_train)-4):
    print('doing neural networks regression models ',labels[i])
    regr = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_NNR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_NNR.append(metrics.mean_absolute_error(y_test[i],Y_predict_NNR[i]))
    MSE_NNR.append(metrics.mean_squared_error(y_test[i],Y_predict_NNR[i]))
    RMSE_NNR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_NNR[i])))
    cvRMSE_NNR.append(RMSE_NNR[i]/np.mean(y_test[i]))
    NNR.append(regr)
#    with open(path+'models_consumption/NNR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)
for i in range(4):
    i=i+2
    print('doing neural networks regression models ',labels[i])
    regr = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
    regr.fit(X_train[i].reshape(-1, 1),y_train[i].reshape(-1, 1).ravel())
    Y_predict_NNR.append(regr.predict(X_test[i].reshape(-1, 1)))
    MAE_NNR.append(metrics.mean_absolute_error(y_test[i],Y_predict_NNR[i]))
    MSE_NNR.append(metrics.mean_squared_error(y_test[i],Y_predict_NNR[i]))
    RMSE_NNR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_predict_NNR[i])))
    cvRMSE_NNR.append(RMSE_NNR[i]/np.mean(y_test[i]))
    NNR.append(regr)
#    with open(path+'models_consumption/NNR_'+labels[i], 'wb') as files:
#        pickle.dump(regr, files)

"""
## Neural Networks regressor
NN_model_central = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model_central.fit(X_central_train,y_central_train)
with open(path+'models/model_NNR_central', 'wb') as files:
    pickle.dump(NN_model_central, files)
"""

#now let's compare the errors

labels_graph = ['Mean Absolute Error']
for i in range(len(MAE_LR)):   
    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)
    #ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
    x = np.arange(len(labels_graph))  # the label locations
    ax.set_ylabel('Error')
    ax.set_title('Comparison')
    ax.set_xticks(x, labels_graph)
    ax.legend()
    
    width = 0.15  # the width of the bars
    space=fig.get_size_inches()*fig.dpi/14
    space=1.7/22
    rects1 = ax.bar(x - 6*space, MAE_LR[i], width, label="Linear regressor")
    rects2 = ax.bar(x - 4*space, MAE_SVR[i], width, label="Support vector regressor")
    rects3 = ax.bar(x - 2*space, MAE_DTR[i], width, label="Decision tree regressor")
    rects4 = ax.bar(x, MAE_RFR[i], width, label="Random forest regressor")
    rects5 = ax.bar(x + 2*space, MAE_GBR[i], width, label="Gradient boosting regressor")
    rects6 = ax.bar(x + 4*space, MAE_BSR[i], width, label="Bootstraping regressor")
    rects7 = ax.bar(x + 6*space, MAE_NNR[i], width, label="Neural network regressor")
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)
    ax.bar_label(rects7, padding=3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.legend()
    
    fig.tight_layout()
    #plt.show()
    fig.savefig('models_resources/MAE/prediciton_'+labels[i]+'.png')
    
labels_graph = ['Mean Squared Error']
for i in range(len(MSE_LR)):   
    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)
    #ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
    x = np.arange(len(labels_graph))  # the label locations
    ax.set_ylabel('Error')
    ax.set_title('Comparison')
    ax.set_xticks(x, labels_graph)
    ax.legend()
    
    width = 0.15  # the width of the bars
    space=fig.get_size_inches()*fig.dpi/14
    space=1.7/22
    rects1 = ax.bar(x - 6*space, MSE_LR[i], width, label="Linear regressor")
    rects2 = ax.bar(x - 4*space, MSE_SVR[i], width, label="Support vector regressor")
    rects3 = ax.bar(x - 2*space, MSE_DTR[i], width, label="Decision tree regressor")
    rects4 = ax.bar(x, MSE_RFR[i], width, label="Random forest regressor")
    rects5 = ax.bar(x + 2*space, MSE_GBR[i], width, label="Gradient boosting regressor")
    rects6 = ax.bar(x + 4*space, MSE_BSR[i], width, label="Bootstraping regressor")
    rects7 = ax.bar(x + 6*space, MSE_NNR[i], width, label="Neural network regressor")
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)
    ax.bar_label(rects7, padding=3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.legend()
    
    fig.tight_layout()
    #plt.show()
    fig.savefig('models_resources/MSE/prediciton_'+labels[i]+'.png')
    
labels_graph = ['Root Mean Squared Error']
for i in range(len(RMSE_LR)):   
    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)
    #ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
    x = np.arange(len(labels_graph))  # the label locations
    ax.set_ylabel('Error')
    ax.set_title('Comparison')
    ax.set_xticks(x, labels_graph)
    ax.legend()
    
    width = 0.15  # the width of the bars
    space=fig.get_size_inches()*fig.dpi/14
    space=1.7/22
    rects1 = ax.bar(x - 6*space, RMSE_LR[i], width, label="Linear regressor")
    rects2 = ax.bar(x - 4*space, RMSE_SVR[i], width, label="Support vector regressor")
    rects3 = ax.bar(x - 2*space, RMSE_DTR[i], width, label="Decision tree regressor")
    rects4 = ax.bar(x, RMSE_RFR[i], width, label="Random forest regressor")
    rects5 = ax.bar(x + 2*space, RMSE_GBR[i], width, label="Gradient boosting regressor")
    rects6 = ax.bar(x + 4*space, RMSE_BSR[i], width, label="Bootstraping regressor")
    rects7 = ax.bar(x + 6*space, RMSE_NNR[i], width, label="Neural network regressor")
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)
    ax.bar_label(rects7, padding=3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.legend()
    
    fig.tight_layout()
    #plt.show()
    fig.savefig('models_resources/RMSE/prediciton_'+labels[i]+'.png')
    
labels_graph = ['Coefficient of Variation Mean Square Error']
for i in range(len(cvRMSE_LR)):   
    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)
    #ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
    x = np.arange(len(labels_graph))  # the label locations
    ax.set_ylabel('Error')
    ax.set_title('Comparison')
    ax.set_xticks(x, labels_graph)
    ax.legend()
    
    width = 0.15  # the width of the bars
    space=fig.get_size_inches()*fig.dpi/14
    space=1.7/22
    rects1 = ax.bar(x - 6*space, cvRMSE_LR[i], width, label="Linear regressor")
    rects2 = ax.bar(x - 4*space, cvRMSE_SVR[i], width, label="Support vector regressor")
    rects3 = ax.bar(x - 2*space, cvRMSE_DTR[i], width, label="Decision tree regressor")
    rects4 = ax.bar(x, cvRMSE_RFR[i], width, label="Random forest regressor")
    rects5 = ax.bar(x + 2*space, cvRMSE_GBR[i], width, label="Gradient boosting regressor")
    rects6 = ax.bar(x + 4*space, cvRMSE_BSR[i], width, label="Bootstraping regressor")
    rects7 = ax.bar(x + 6*space, cvRMSE_NNR[i], width, label="Neural network regressor")
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)
    ax.bar_label(rects7, padding=3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.legend()
    
    fig.tight_layout()
    #plt.show()
    fig.savefig('models_resources/cvRMSE/prediciton_'+labels[i]+'.png')
    
#since we have 4 different measures of performance for each energy, 
#we are going to choose the regressor that shows the best performance while considering each error
#in the case of a tie, we are going to pick the one that reveals the higher improvement

min_MAE_index=[]
for i in range(len(MAE_LR)): 
    MAE=[]
    MAE.append(MAE_LR[i])
    MAE.append(MAE_SVR[i])
    MAE.append(MAE_DTR[i])
    MAE.append(MAE_RFR[i])
    MAE.append(MAE_GBR[i])
    MAE.append(MAE_BSR[i])
    MAE.append(MAE_NNR[i])
    #print('minimum found ',min(MAE))
    for j in range(len(MAE)):
        if MAE[j]==min(MAE):
            min_MAE_index.append(j)
            break
            
#print(min_MAE_index)

min_MSE_index=[]
for i in range(len(MSE_LR)): 
    MSE=[]
    MSE.append(MSE_LR[i])
    MSE.append(MSE_SVR[i])
    MSE.append(MSE_DTR[i])
    MSE.append(MSE_RFR[i])
    MSE.append(MSE_GBR[i])
    MSE.append(MSE_BSR[i])
    MSE.append(MSE_NNR[i])
    #print('minimum found ',min(MSE))
    for j in range(len(MSE)):
        if MSE[j]==min(MSE):
            min_MSE_index.append(j)
            break

min_RMSE_index=[]
for i in range(len(RMSE_LR)): 
    RMSE=[]
    RMSE.append(RMSE_LR[i])
    RMSE.append(RMSE_SVR[i])
    RMSE.append(RMSE_DTR[i])
    RMSE.append(RMSE_RFR[i])
    RMSE.append(RMSE_GBR[i])
    RMSE.append(RMSE_BSR[i])
    RMSE.append(RMSE_NNR[i])
    #print('minimum found ',min(RMSE))
    for j in range(len(RMSE)):
        if RMSE[j]==min(RMSE):
            min_RMSE_index.append(j)
            break

min_cvRMSE_index=[]
for i in range(len(cvRMSE_LR)): 
    cvRMSE=[]
    cvRMSE.append(cvRMSE_LR[i])
    cvRMSE.append(cvRMSE_SVR[i])
    cvRMSE.append(cvRMSE_DTR[i])
    cvRMSE.append(cvRMSE_RFR[i])
    cvRMSE.append(cvRMSE_GBR[i])
    cvRMSE.append(cvRMSE_BSR[i])
    cvRMSE.append(cvRMSE_NNR[i])
    #print('minimum found ',min(cvRMSE))
    for j in range(len(cvRMSE)):
        if cvRMSE[j]==min(cvRMSE):
            min_cvRMSE_index.append(j)
            break

regressor=['LR','SVR','DTR','RFR','GBR','BSR','NNR']
print(min_MAE_index)
print(min_MSE_index)
print(min_RMSE_index)
print(min_cvRMSE_index)
all_the_same=[]
not_the_same=[]
for i in range(len(min_MAE_index)):
    if all(x == min_MAE_index[i] for x in (min_MAE_index[i], min_MSE_index[i], min_RMSE_index[i], min_cvRMSE_index[i])):
        print('they are all the same for ',i)
        all_the_same.append(i)
        #now that we know they are the same, we are going to import the correct regressor
        if min_MAE_index[i]==0:
            regr=LR[i]
            print('for ',labels[i],' the regressor used was ',regressor[0])
        elif min_MAE_index[i]==1:
            regr=SVRm[i]
            print('for ',labels[i],' the regressor used was ',regressor[1])
        elif min_MAE_index[i]==2:
            regr=DTR[i]
            print('for ',labels[i],' the regressor used was ',regressor[2])
        elif min_MAE_index[i]==3:
            regr=RFR[i]
            print('for ',labels[i],' the regressor used was ',regressor[3])
        elif min_MAE_index[i]==4:
            regr=GBR[i]
            print('for ',labels[i],' the regressor used was ',regressor[4])
        elif min_MAE_index[i]==5:
            regr=BSR[i]
            print('for ',labels[i],' the regressor used was ',regressor[5])
        elif min_MAE_index[i]==6:
            regr=NNR[i]
            print('for ',labels[i],' the regressor used was ',regressor[6])
        with open(path+'models_resources/'+labels[i], 'wb') as files:
            pickle.dump(regr, files)
    else:
        decision=[]
        decision.append(min_MAE_index[i])
        decision.append(min_MSE_index[i])
        decision.append(min_RMSE_index[i])
        decision.append(min_cvRMSE_index[i])
        print(decision)
        
        repeat=[]
        index_repeat=[]
        count=1
        flag_repeat=0
        for z in range(4):
            #print('FOR Z = ',z)
            for j in range(3-z):
                #print('for ',z,' and ',j)
                if flag_repeat==0:
                    if decision[z]==decision[j+z+1]:
                        count=count+1
                    flag_count=1
                    flag_repeat=1
                    #print('here at repeat=0, for ',decision[z],' the count was ',count)
                elif flag_repeat==1:
                    flag_repeat_ignore=0
                    for previous in index_repeat:
                        if decision[j+z+1]==previous:
                            flag_repeat_ignore=1
                            #print('here at repeat=1, ignore=1, this was already counted for')
                    if flag_repeat_ignore==0:
                        if decision[z]==decision[j+z+1]:
                            count=count+1
                        flag_count=1
                        #print('here at repeat=1, for ',decision[z],' the count was ',count)
            if flag_count==1:
                repeat.append(count)
                index_repeat.append(decision[z])
            count=1
            flag_count=0
        print('the repeated numbers were for',labels[i],'were',index_repeat,' and each repeated itself',repeat)
        
        max_repeat=max(repeat)
        regres=[]
        for t in range(len(repeat)):
            if repeat[t]==max_repeat:
                regres.append(index_repeat[t])
        
        #no draw
        if len(regres)==1:
            if regres[0]==0:
                regr=LR[i]
                print('for ',labels[i],' the regressor used was ',regressor[0])
            elif regres[0]==1:
                regr=SVRm[i]
                print('for ',labels[i],' the regressor used was ',regressor[1])
            elif regres[0]==2:
                regr=DTR[i]
                print('for ',labels[i],' the regressor used was ',regressor[2])
            elif regres[0]==3:
                regr=RFR[i]
                print('for ',labels[i],' the regressor used was ',regressor[3])
            elif regres[0]==4:
                regr=GBR[i]
                print('for ',labels[i],' the regressor used was ',regressor[4])
            elif regres[0]==5:
                regr=BSR[i]
                print('for ',labels[i],' the regressor used was ',regressor[5])
            elif regres[0]==6:
                regr=NNR[i]
                print('for ',labels[i],' the regressor used was ',regressor[6])
            with open(path+'models_resources/'+labels[i], 'wb') as files:
                pickle.dump(regr, files)
                
            #there's draw
            #in the case of a draw, we will pick the regressor with the highest improvement
        elif len(regres)==1:
            #first, let's being by seeing which regressor got second:
            
            min_compare=[]
            #min_MAE_index_2=[]
            MAE=[]
            MAE.append(MAE_LR[i])
            MAE.append(MAE_SVR[i])
            MAE.append(MAE_DTR[i])
            MAE.append(MAE_RFR[i])
            MAE.append(MAE_GBR[i])
            MAE.append(MAE_BSR[i])
            MAE.append(MAE_NNR[i])
            MAE.remove(decision[0])
            print(MAE)
            print(decision[0])
            #for j in range(len(MAE)):
            #    if MAE[j]==min(MAE):
            #        min_MAE_index_2.append(j)
            #        break
            min_compare.append(min(MAE))
        
            MSE=[]
            MSE.append(MSE_LR[i])
            MSE.append(MSE_SVR[i])
            MSE.append(MSE_DTR[i])
            MSE.append(MSE_RFR[i])
            MSE.append(MSE_GBR[i])
            MSE.append(MSE_BSR[i])
            MSE.append(MSE_NNR[i])
            MSE.remove(decision[1])
            #print('minimum found ',min(MSE))
            min_compare.append(min(MSE))

            min_RMSE_index_2=[]
            RMSE=[]
            RMSE.append(RMSE_LR[i])
            RMSE.append(RMSE_SVR[i])
            RMSE.append(RMSE_DTR[i])
            RMSE.append(RMSE_RFR[i])
            RMSE.append(RMSE_GBR[i])
            RMSE.append(RMSE_BSR[i])
            RMSE.append(RMSE_NNR[i])
            RMSE.remove(decision[2])
            #print('minimum found ',min(RMSE))
            min_compare.append(min(RMSE))
                
            min_cvRMSE_index_2=[]
            cvRMSE=[]
            cvRMSE.append(cvRMSE_LR[i])
            cvRMSE.append(cvRMSE_SVR[i])
            cvRMSE.append(cvRMSE_DTR[i])
            cvRMSE.append(cvRMSE_RFR[i])
            cvRMSE.append(cvRMSE_GBR[i])
            cvRMSE.append(cvRMSE_BSR[i])
            cvRMSE.append(cvRMSE_NNR[i])
            cvRMSE.remove(decision[3])
            #print('minimum found ',min(cvRMSE))
            min_compare.append(min(cvRMSE)) 
                
            for u in range(len(min_compare)):
                min_compare[u]=(abs(min_compare[u]-decision[u]))/(decision[u])
             
            #best_decision=0
            for z in range(len(min_compare)):
                if min_compare[z]==min_compare:
                    best_decision=z
                
            new_reg=index_repeat[best_decision]
            if new_reg==0:
                regr=LR[i]
                print('for ',labels[i],' the regressor used was ',regressor[0])
            elif new_reg==1:
                regr=SVRm[i]
                print('for ',labels[i],' the regressor used was ',regressor[1])
            elif new_reg==2:
                regr=DTR[i]
                print('for ',labels[i],' the regressor used was ',regressor[2])
            elif new_reg==3:
                regr=RFR[i]
                print('for ',labels[i],' the regressor used was ',regressor[3])
            elif new_reg==4:
                regr=GBR[i]
                print('for ',labels[i],' the regressor used was ',regressor[4])
            elif new_reg==5:
                regr=BSR[i]
                print('for ',labels[i],' the regressor used was ',regressor[5])
            elif new_reg==6:
                regr=NNR[i]
                print('for ',labels[i],' the regressor used was ',regressor[6])
            with open(path+'models_resources/'+labels[i], 'wb') as files:
                pickle.dump(regr, files)
            
            
        