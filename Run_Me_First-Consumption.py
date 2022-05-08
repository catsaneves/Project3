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
primary_energy_consumptions = pd.read_excel(excel_file, sheet_name=1,header=[2],index_col=0)
primary_energy_consumptions = primary_energy_consumptions[primary_energy_consumptions.columns.drop(list(primary_energy_consumptions.filter(regex='Unnamed:')))]
primary_energy_consumptions.dropna(axis = 0, how = 'all', inplace = True)
#print(primary_energy_consumptions.dtypes)
not_ignore_col=[]
for i in range(len(primary_energy_consumptions.columns)-3):
    not_ignore_col.append(primary_energy_consumptions.columns[i])
#print(not_ignore_col)
#print(primary_energy_consumptions[~primary_energy_consumptions.columns.isin(ignore_col)])
primary_energy_consumptions[not_ignore_col]=primary_energy_consumptions[not_ignore_col].multiply(277.77777777778)
#print(primary_energy_consumptions[primary_energy_consumptions.columns[len(primary_energy_consumptions.columns)-2]])
primary_energy_consumptions_portugal=primary_energy_consumptions.loc['Portugal']
#print(primary_energy_consumptions_portugal)

primary_energy_consumptions_by_fuel = pd.read_excel(excel_file, sheet_name=2,header=[2],index_col=0)
primary_energy_consumptions_by_fuel = primary_energy_consumptions_by_fuel[primary_energy_consumptions_by_fuel.columns.drop(list(primary_energy_consumptions_by_fuel.filter(regex='Unnamed:')))]
primary_energy_consumptions_by_fuel.dropna(axis = 0, how = 'all', inplace = True)
primary_energy_consumptions_by_fuel=primary_energy_consumptions_by_fuel.multiply(277.77777777778)
#print(primary_energy_consumptions_by_fuel)
primary_energy_consumptions_by_fuel_portugal=primary_energy_consumptions_by_fuel.loc['Portugal']
#print(primary_energy_consumptions_by_fuel_portugal)

primary_energy_consumptions_per_capita = pd.read_excel(excel_file, sheet_name=3,header=[2],index_col=0)
primary_energy_consumptions_per_capita = primary_energy_consumptions_per_capita[primary_energy_consumptions_per_capita.columns.drop(list(primary_energy_consumptions_per_capita.filter(regex='Unnamed:')))]
primary_energy_consumptions_per_capita.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(primary_energy_consumptions_per_capita.columns)-2):
    not_ignore_col.append(primary_energy_consumptions_per_capita.columns[i])
primary_energy_consumptions_per_capita[not_ignore_col]=primary_energy_consumptions_per_capita[not_ignore_col].multiply(277.77777777778)
#print(primary_energy_consumptions_per_capita)
primary_energy_consumptions_per_capita_portugal=primary_energy_consumptions_per_capita.loc['Portugal']
#print(primary_energy_consumptions_per_capita_portugal)


oil_consumptions = pd.read_excel(excel_file, sheet_name=17,header=[2],index_col=0)
oil_consumptions = oil_consumptions[oil_consumptions.columns.drop(list(oil_consumptions.filter(regex='Unnamed:')))]
oil_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(oil_consumptions.columns)-3):
    not_ignore_col.append(oil_consumptions.columns[i])
oil_consumptions[not_ignore_col] = oil_consumptions[not_ignore_col].multiply(277.77777777778)
#print(oil_consumptions)
oil_consumptions_portugal=oil_consumptions.loc['Portugal']
#print(oil_consumptions_portugal)

gas_consumptions = pd.read_excel(excel_file, sheet_name=34,header=[2],index_col=0)
gas_consumptions = gas_consumptions[gas_consumptions.columns.drop(list(gas_consumptions.filter(regex='Unnamed:')))]
gas_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(gas_consumptions.columns)-3):
    not_ignore_col.append(gas_consumptions.columns[i])
gas_consumptions[not_ignore_col] = gas_consumptions[not_ignore_col].multiply(277.77777777778)
#print(gas_consumptions)
gas_consumptions_portugal=gas_consumptions.loc['Portugal']
#print(gas_consumptions_portugal)

coal_consumptions = pd.read_excel(excel_file, sheet_name=44,header=[2],index_col=0)
coal_consumptions = coal_consumptions[coal_consumptions.columns.drop(list(coal_consumptions.filter(regex='Unnamed:')))]
coal_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(coal_consumptions.columns)-3):
    not_ignore_col.append(coal_consumptions.columns[i])
coal_consumptions[not_ignore_col] = coal_consumptions[not_ignore_col].multiply(277.77777777778)
#print(coal_consumptions)
coal_consumptions_portugal=coal_consumptions.loc['Portugal']
#print(coal_consumptions_portugal)

nuclear_consumptions = pd.read_excel(excel_file, sheet_name=49,header=[2],index_col=0)
nuclear_consumptions = nuclear_consumptions[nuclear_consumptions.columns.drop(list(nuclear_consumptions.filter(regex='Unnamed:')))]
nuclear_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(nuclear_consumptions.columns)-3):
    not_ignore_col.append(nuclear_consumptions.columns[i])
nuclear_consumptions[not_ignore_col] = nuclear_consumptions[not_ignore_col].multiply(277.77777777778)
#print(nuclear_consumptions)
nuclear_consumptions_portugal=nuclear_consumptions.loc['Portugal']
#print(nuclear_consumptions_portugal)

hydro_consumptions = pd.read_excel(excel_file, sheet_name=51,header=[2],index_col=0)
hydro_consumptions = hydro_consumptions[hydro_consumptions.columns.drop(list(hydro_consumptions.filter(regex='Unnamed:')))]
hydro_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(hydro_consumptions.columns)-3):
    not_ignore_col.append(hydro_consumptions.columns[i])
hydro_consumptions[not_ignore_col] = hydro_consumptions[not_ignore_col].multiply(277.77777777778)
#print(hydro_consumptions)
hydro_consumptions_portugal=hydro_consumptions.loc['Portugal']
#print(hydro_consumptions_portugal)

renewable_consumptions = pd.read_excel(excel_file, sheet_name=52,header=[2],index_col=0)
renewable_consumptions = renewable_consumptions[renewable_consumptions.columns.drop(list(renewable_consumptions.filter(regex='Unnamed:')))]
renewable_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(renewable_consumptions.columns)-3):
    not_ignore_col.append(renewable_consumptions.columns[i])
renewable_consumptions[not_ignore_col] = renewable_consumptions[not_ignore_col].multiply(277.77777777778)
#print(renewable_consumptions)
renewable_consumptions_portugal=renewable_consumptions.loc['Portugal']
#print(renewable_consumptions_portugal)

solar_consumptions = pd.read_excel(excel_file, sheet_name=57,header=[2],index_col=0)
solar_consumptions = solar_consumptions[solar_consumptions.columns.drop(list(solar_consumptions.filter(regex='Unnamed:')))]
solar_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(solar_consumptions.columns)-3):
    not_ignore_col.append(solar_consumptions.columns[i])
solar_consumptions[not_ignore_col] = solar_consumptions[not_ignore_col].multiply(277.77777777778)
#print(solar_consumptions)
solar_consumptions_portugal=solar_consumptions.loc['Portugal']
#print(solar_consumptions_portugal)

wind_consumptions = pd.read_excel(excel_file, sheet_name=59,header=[2],index_col=0)
wind_consumptions = wind_consumptions[wind_consumptions.columns.drop(list(wind_consumptions.filter(regex='Unnamed:')))]
wind_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(wind_consumptions.columns)-3):
    not_ignore_col.append(wind_consumptions.columns[i])
wind_consumptions[not_ignore_col] = wind_consumptions[not_ignore_col].multiply(277.77777777778)
#print(wind_consumptions)
wind_consumptions_portugal=wind_consumptions.loc['Portugal']
#print(wind_consumptions_portugal)

geo_biomass_consumptions = pd.read_excel(excel_file, sheet_name=61,header=[2],index_col=0)
geo_biomass_consumptions = geo_biomass_consumptions[geo_biomass_consumptions.columns.drop(list(geo_biomass_consumptions.filter(regex='Unnamed:')))]
geo_biomass_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(geo_biomass_consumptions.columns)-3):
    not_ignore_col.append(geo_biomass_consumptions.columns[i])
geo_biomass_consumptions[not_ignore_col] = geo_biomass_consumptions[not_ignore_col].multiply(277.77777777778)
#print(geo_biomass_consumptions)
geo_biomass_consumptions_portugal=geo_biomass_consumptions.loc['Portugal']
#print(geo_biomass_consumptions_portugal)

biofuels_consumptions = pd.read_excel(excel_file, sheet_name=65,header=[2],index_col=0)
biofuels_consumptions = biofuels_consumptions[biofuels_consumptions.columns.drop(list(biofuels_consumptions.filter(regex='Unnamed:')))]
biofuels_consumptions.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(biofuels_consumptions.columns)-3):
    not_ignore_col.append(biofuels_consumptions.columns[i])
biofuels_consumptions[not_ignore_col] = biofuels_consumptions[not_ignore_col].multiply(0.277777778)
#print(biofuels_consumptions)
biofuels_consumptions_portugal=biofuels_consumptions.loc['Portugal']
#print(biofuels_consumptions_portugal)

#-------------------------------------DATA FOR CONSUMPTIONS-------------------------------------
#all data is in Twh
#we should first find if there's any empty dataframe and disregard it
#print(primary_energy_consumptions_portugal)
#print(oil_consumptions_portugal)
#print(gas_consumptions_portugal)
#print(coal_consumptions_portugal)
#print(nuclear_consumptions_portugal)
#print(hydro_consumptions_portugal)
#print(renewable_consumptions_portugal)
#print(solar_consumptions_portugal)
#print(wind_consumptions_portugal)
#print(geo_biomass_consumptions_portugal)
#print(biofuels_consumptions_portugal)

if (primary_energy_consumptions_portugal == 0).all():
    print('this dataframe is empty - energy')
if (oil_consumptions_portugal == 0).all():
    print('this dataframe is empty - oil')
if (gas_consumptions_portugal == 0).all():
    print('this dataframe is empty - gas')
if (coal_consumptions_portugal == 0).all():
    print('this dataframe is empty - coal')
if (nuclear_consumptions_portugal == 0).all():
    print('this dataframe is empty - nuclear')
if (hydro_consumptions_portugal == 0).all():
    print('this dataframe is empty - hydro')
if (renewable_consumptions_portugal == 0).all():
    print('this dataframe is empty - renewables')
if (solar_consumptions_portugal == 0).all():
    print('this dataframe is empty - solar')
if (wind_consumptions_portugal == 0).all():
    print('this dataframe is empty - wind')
if (geo_biomass_consumptions_portugal == 0).all():
    print('this dataframe is empty - geo and biomass')
if (biofuels_consumptions_portugal == 0).all():
    print('this dataframe is empty - biofuels')
    
#we will now exclude the nuclear dataframe

#now let's make a dataframe with all the types of energy

print(type(primary_energy_consumptions_portugal)) #all of the data is a pandas.core.series.Series

energy_Portugal=pd.DataFrame(primary_energy_consumptions_portugal)
energy_Portugal.rename(columns={'Portugal':'Energy [TWh]'},inplace=True)
energy_Portugal['year']=energy_Portugal.index
energy_Portugal.set_index('year',inplace=True)
oil_Portugal=pd.DataFrame(oil_consumptions_portugal)
oil_Portugal.rename(columns={'Portugal':'Oil consumption [TWh]'},inplace=True)
oil_Portugal['year']=oil_Portugal.index
oil_Portugal.set_index('year',inplace=True)
gas_Portugal=pd.DataFrame(gas_consumptions_portugal)
gas_Portugal.rename(columns={'Portugal':'Gas consumption [TWh]'},inplace=True)
gas_Portugal['year']=gas_Portugal.index
gas_Portugal.set_index('year',inplace=True)
coal_Portugal=pd.DataFrame(coal_consumptions_portugal)
coal_Portugal.rename(columns={'Portugal':'Coal consumption [TWh]'},inplace=True)
coal_Portugal['year']=coal_Portugal.index
coal_Portugal.set_index('year',inplace=True)
hydro_Portugal=pd.DataFrame(hydro_consumptions_portugal)
hydro_Portugal.rename(columns={'Portugal':'Hydro consumption [TWh]'},inplace=True)
hydro_Portugal['year']=hydro_Portugal.index
hydro_Portugal.set_index('year',inplace=True)
renewables_Portugal=pd.DataFrame(renewable_consumptions_portugal)
renewables_Portugal.rename(columns={'Portugal':'Renewables consumption [TWh]'},inplace=True)
renewables_Portugal['year']=renewables_Portugal.index
renewables_Portugal.set_index('year',inplace=True)
solar_Portugal=pd.DataFrame(solar_consumptions_portugal)
solar_Portugal.rename(columns={'Portugal':'Solar consumption [TWh]'},inplace=True)
solar_Portugal['year']=solar_Portugal.index
solar_Portugal.set_index('year',inplace=True)
wind_Portugal=pd.DataFrame(wind_consumptions_portugal)
wind_Portugal.rename(columns={'Portugal':'Wind consumption [TWh]'},inplace=True)
wind_Portugal['year']=wind_Portugal.index
wind_Portugal.set_index('year',inplace=True)
geo_biomass_Portugal=pd.DataFrame(geo_biomass_consumptions_portugal)
geo_biomass_Portugal.rename(columns={'Portugal':'Geo_biomass consumption [TWh]'},inplace=True)
geo_biomass_Portugal['year']=geo_biomass_Portugal.index
geo_biomass_Portugal.set_index('year',inplace=True)
biofuels_Portugal=pd.DataFrame(biofuels_consumptions_portugal)
biofuels_Portugal.rename(columns={'Portugal':'Bio_fuels consumption [TWh]'},inplace=True)
biofuels_Portugal['year']=biofuels_Portugal.index
biofuels_Portugal.set_index('year',inplace=True)

data_Portugal=pd.merge(energy_Portugal,oil_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,gas_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,coal_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,hydro_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,renewables_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,solar_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,wind_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,geo_biomass_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,biofuels_Portugal, on='year',how='outer')
col_number=len(data_Portugal)
data_Portugal.drop(data_Portugal.index[col_number-3:col_number], axis=0, inplace=True) #remove the last rows which correspond to %'s
print(data_Portugal)


data_Portugal['year']=data_Portugal.index

#let's check for nan values
for col in data_Portugal.columns:
    if data_Portugal[col].isnull().values.any():
        print('the column ',col,'has ', data_Portugal[col].isnull().sum(),'nan values')
        
Z=data_Portugal.values
X=Z[:,10] #time
Y_energy=Z[:,0] 
Y_oil=Z[:,1] 
Y_gas=Z[:,2]
Y_coal=Z[:,3] 
Y_hydro=Z[:,4]
Y_renewables=Z[:,5] 
Y_solar=Z[:,6]
Y_wind=Z[:,7]
Y_geo_biomass=Z[:,8] 
Y_biofuels=Z[:,9] 
#print(Y_biofuels)
to_be_delete=[]
for i in range(len(Y_biofuels)):
    if math.isnan(Y_biofuels[i]):
        #print('NaN found')
        to_be_delete.append(i)
Y_biofuels=np.delete(Y_biofuels, to_be_delete)
X_biofuels=np.delete(X, to_be_delete)
print('the values for X are',X)

X_energy_train, X_energy_test, y_energy_train, y_energy_test = train_test_split(X,Y_energy)
X_oil_train, X_oil_test, y_oil_train, y_oil_test = train_test_split(X,Y_oil)
X_gas_train, X_gas_test, y_gas_train, y_gas_test = train_test_split(X,Y_gas)
X_coal_train, X_coal_test, y_coal_train, y_coal_test = train_test_split(X,Y_coal)
X_hydro_train, X_hydro_test, y_hydro_train, y_hydro_test = train_test_split(X,Y_hydro)
X_renewables_train, X_renewables_test, y_renewables_train, y_renewables_test = train_test_split(X,Y_renewables)
X_solar_train, X_solar_test, y_solar_train, y_solar_test = train_test_split(X,Y_solar)
X_wind_train, X_wind_test, y_wind_train, y_wind_test = train_test_split(X,Y_wind)
X_geo_biomass_train, X_geo_biomass_test, y_geo_biomass_train, y_geo_biomass_test = train_test_split(X,Y_geo_biomass)
X_biofuels_train, X_biofuels_test, y_biofuels_train, y_biofuels_test = train_test_split(X_biofuels,Y_biofuels)

## autoregressive
X_train=[X_energy_train,X_oil_train,X_gas_train,X_coal_train,X_hydro_train,X_renewables_train,X_solar_train,X_wind_train,X_geo_biomass_train,X_biofuels_train]
X_test=[X_energy_test,X_oil_test,X_gas_test,X_coal_test,X_hydro_test,X_renewables_test,X_solar_test,X_wind_test,X_geo_biomass_test,X_biofuels_test]
y_train=[y_energy_train,y_oil_train,y_gas_train,y_coal_train,y_hydro_train,y_renewables_train,y_solar_train,y_wind_train,y_geo_biomass_train,y_biofuels_train]
y_test=[y_energy_test,y_oil_test,y_gas_test,y_coal_test,y_hydro_test,y_renewables_test,y_solar_test,y_wind_test,y_geo_biomass_test,y_biofuels_test]
labels=['energy','oil','gas','coal','hydro','renewables','solar','wind','geo_biomass','biofuels']

#print(y_energy_train)
#print(y_train[0].reshape(-1, 1))

data=data_Portugal
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
for i in range(len(X_train)-1):
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

print('doing linear regression models biofuels')
regr = linear_model.LinearRegression()
i=9
regr.fit(X_biofuels_train.reshape(-1, 1),y_biofuels_train.reshape(-1, 1))
Y_biofuels_predict_LR=regr.predict(X_biofuels_test.reshape(-1, 1))
MAE_LR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_LR))
MSE_LR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_LR))
RMSE_LR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_LR)))
cvRMSE_LR.append(RMSE_LR[i]/np.mean(y_test[i]))
LR.append(regr)
#with open(path+'models_consumption/LR_biofuels', 'wb') as files:
 #   pickle.dump(regr, files)
 
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
for i in range(len(X_train)-1):
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
        
print('doing support vector regression models biofuels')
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train_ss = ss_X.fit_transform(X_biofuels_train.reshape(-1, 1))
y_train_ss = ss_y.fit_transform(y_biofuels_train.reshape(-1, 1))
regr = SVR(kernel='rbf')
i=9
regr.fit(X_train_ss,y_train_ss)
Y_biofuels_predict_SVR=regr.predict(X_biofuels_test.reshape(-1, 1))
MAE_SVR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_SVR))
MSE_SVR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_SVR))
RMSE_SVR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_SVR)))
cvRMSE_SVR.append(RMSE_SVR[i]/np.mean(y_test[i]))
SVRm.append(regr)
#with open(path+'models_consumption/SVR_biofuels', 'wb') as files:
#    pickle.dump(regr, files)
    
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
for i in range(len(X_train)-1):
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

print('doing decision tree regression models biofuels')
regr = DecisionTreeRegressor()
regr.fit(X_biofuels_train.reshape(-1, 1),y_biofuels_train.reshape(-1, 1))
Y_biofuels_predict_DTR=regr.predict(X_biofuels_test.reshape(-1, 1))
i=9
MAE_DTR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_DTR))
MSE_DTR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_DTR))
RMSE_DTR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_DTR)))
cvRMSE_DTR.append(RMSE_DTR[i]/np.mean(y_test[i]))
DTR.append(regr)
#with open(path+'models_consumption/DTR_biofuels', 'wb') as files:
#    pickle.dump(regr, files)
    
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
for i in range(len(X_train)-1):
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

print('doing random forest regression models biofuels')
regr = RandomForestRegressor(**parameters)
i=9
regr.fit(X_biofuels_train.reshape(-1, 1),y_biofuels_train.reshape(-1, 1).ravel())
Y_biofuels_predict_RFR=regr.predict(X_biofuels_test.reshape(-1, 1))
MAE_RFR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_RFR))
MSE_RFR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_RFR))
RMSE_RFR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_RFR)))
cvRMSE_RFR.append(RMSE_RFR[i]/np.mean(y_test[i]))
RFR.append(regr)
#with open(path+'models_consumption/RFR_biofuels', 'wb') as files:
#    pickle.dump(regr, files)

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
for i in range(len(X_train)-1):
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

print('doing gradient boosting regression models biofuels')
regr = GradientBoostingRegressor()
regr.fit(X_biofuels_train.reshape(-1, 1),y_biofuels_train.reshape(-1, 1).ravel())
Y_biofuels_predict_GBR=regr.predict(X_biofuels_test.reshape(-1, 1))
i=9
MAE_GBR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_GBR))
MSE_GBR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_GBR))
RMSE_GBR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_GBR)))
cvRMSE_GBR.append(RMSE_GBR[i]/np.mean(y_test[i]))
GBR.append(regr)
#with open(path+'models_consumption/GBR_biofuels', 'wb') as files:
#    pickle.dump(regr, files)


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
for i in range(len(X_train)-1):
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

print('doing bootstraping regression models biofuels')
regr = BaggingRegressor()
regr.fit(X_biofuels_train.reshape(-1, 1),y_biofuels_train.reshape(-1, 1).ravel())
Y_biofuels_predict_BSR=regr.predict(X_biofuels_test.reshape(-1, 1))
i=9
MAE_BSR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_BSR))
MSE_BSR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_BSR))
RMSE_BSR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_BSR)))
cvRMSE_BSR.append(RMSE_BSR[i]/np.mean(y_test[i]))
BSR.append(regr)
#with open(path+'models_consumption/BSR_biofuels', 'wb') as files:
#    pickle.dump(regr, files)

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
for i in range(len(X_train)-1):
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

print('doing neural networks regression models biofuels')
regr = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
regr.fit(X_biofuels_train.reshape(-1, 1),y_biofuels_train.reshape(-1, 1).ravel())
Y_biofuels_predict_NNR=regr.predict(X_biofuels_test.reshape(-1, 1))
i=9
MAE_NNR.append(metrics.mean_absolute_error(y_test[i],Y_biofuels_predict_NNR))
MSE_NNR.append(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_NNR))
RMSE_NNR.append(np.sqrt(metrics.mean_squared_error(y_test[i],Y_biofuels_predict_NNR)))
cvRMSE_NNR.append(RMSE_NNR[i]/np.mean(y_test[i]))
NNR.append(regr)
#with open(path+'models_consumption/NNR_biofuels', 'wb') as files:
#    pickle.dump(regr, files)

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
    fig.savefig('models_consumption/MAE/prediciton_'+labels[i]+'.png')
    
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
    fig.savefig('models_consumption/MSE/prediciton_'+labels[i]+'.png')
    
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
    fig.savefig('models_consumption/RMSE/prediciton_'+labels[i]+'.png')
    
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
    fig.savefig('models_consumption/cvRMSE/prediciton_'+labels[i]+'.png')
    
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
        with open(path+'models_consumption/'+labels[i], 'wb') as files:
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
            with open(path+'models_consumption/'+labels[i], 'wb') as files:
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
            with open(path+'models_consumption/'+labels[i], 'wb') as files:
                pickle.dump(regr, files)
            
            
        