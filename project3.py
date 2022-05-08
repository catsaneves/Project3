"""
Energy Services - Project 3

Diana Bernardo 90384
Sofia Costa 90426
Catarina Neves 91036
"""

# Import libraries
import dash
import dash_bootstrap_components as dbc
#import dash_html_components as html
from dash import html
#import dash_core_components as dcc
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import pickle
import dash_daq as daq

import plotly.express as px
import json

import numpy as np

#import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import math
import plotly.graph_objects as go
from sklearn import  metrics

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
df_pe_pc=primary_energy_consumptions_per_capita[not_ignore_col].T
df_pe_pc = df_pe_pc.reset_index()

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
df_oc=oil_consumptions[not_ignore_col].T
df_oc=df_oc.reset_index()
oil_consumptions_per_capita=pd.read_csv('per-capita-oil-consumption-kWh.csv')


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
gas_consumptions_per_capita=pd.read_csv('per-capita-gas-consumption-kWh.csv')

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
coal_consumptions_per_capita=pd.read_csv('per-capita-coal-consumption-kWh.csv')


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
nuclear_consumptions_per_capita=pd.read_csv('per-capita-nuclear-consumption-kWh.csv')


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
hydro_consumptions_per_capita=pd.read_csv('per-capita-hydro-consumption-kWh.csv')


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
renewable_consumptions_per_capita=pd.read_csv('per-capita-renewables-consumption-kWh.csv')


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
solar_consumptions_per_capita=pd.read_csv('per-capita-solar-consumption-kWh.csv')


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
wind_consumptions_per_capita=pd.read_csv('per-capita-wind-consumption-kWh.csv')


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

#-------------------------------------DATA FOR PRODUCTION-------------------------------------

electricity_production = pd.read_excel(excel_file, sheet_name=66,header=[2],index_col=0)
electricity_production = electricity_production[electricity_production.columns.drop(list(electricity_production.filter(regex='Unnamed:')))]
electricity_production.dropna(axis = 0, how = 'all', inplace = True)
#print(electricity_production)
electricity_production_portugal=electricity_production.loc['Portugal']
electricity_production_portugal=electricity_production_portugal[0:len(electricity_production_portugal)-3]
#print(electricity_production_portugal)
electricity_generation_per_capita=pd.read_csv('per-capita-electricity-generation.csv')

oil_production = pd.read_excel(excel_file, sheet_name=67,header=[2],index_col=0)
oil_production = oil_production[oil_production.columns.drop(list(oil_production.filter(regex='Unnamed:')))]
oil_production.dropna(axis = 0, how = 'all', inplace = True)
oil_generation_per_capita=pd.read_csv('oil-electricity-per-capita-generation-kWh.csv')


gas_production = pd.read_excel(excel_file, sheet_name=31,header=[2],index_col=0)
gas_production = gas_production[gas_production.columns.drop(list(gas_production.filter(regex='Unnamed:')))]
gas_production.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(gas_production.columns)-3):
    not_ignore_col.append(gas_production.columns[i])
gas_production[not_ignore_col] = gas_production[not_ignore_col].multiply(277.77777777778)
gas_generation_per_capita=pd.read_csv('gas-electricity-per-capita-generation-kWh.csv')


coal_production = pd.read_excel(excel_file, sheet_name=43,header=[2],index_col=0)
not_ignore_col=[]
for i in range(len(coal_production.columns)-3):
    not_ignore_col.append(coal_production.columns[i])
coal_production = coal_production[coal_production.columns.drop(list(coal_production.filter(regex='Unnamed:')))]
coal_production.dropna(axis = 0, how = 'all', inplace = True)
coal_production[not_ignore_col] = coal_production[not_ignore_col].multiply(277.77777777778)
coal_generation_per_capita=pd.read_csv('coal-electricity-per-capita-generation-kWh.csv')


nuclear_production = pd.read_excel(excel_file, sheet_name=48,header=[2],index_col=0)
nuclear_production = nuclear_production[nuclear_production.columns.drop(list(nuclear_production.filter(regex='Unnamed:')))]
nuclear_production.dropna(axis = 0, how = 'all', inplace = True)
nuclear_generation_per_capita=pd.read_csv('nuclear-electricity-per-capita-generation-kWh.csv')


"""
electricity_production_by_fuel= pd.read_excel(excel_file, sheet_name=67,header=[2],index_col=0)
electricity_production_by_fuel= electricity_production_by_fuel[electricity_production_by_fuel.columns.drop(list(electricity_production_by_fuel.filter(regex='Unnamed:')))]
electricity_production_by_fuel.dropna(axis = 0, how = 'all', inplace = True)
#print(electricity_production_by_fuel)
"""

hydro_production = pd.read_excel(excel_file, sheet_name=50,header=[2],index_col=0)
hydro_production = hydro_production[hydro_production.columns.drop(list(hydro_production.filter(regex='Unnamed:')))]
hydro_production.dropna(axis = 0, how = 'all', inplace = True)
#print(hydro_production)
hydro_production_portugal=hydro_production.loc['Portugal']
hydro_production_portugal=hydro_production_portugal[0:len(hydro_production_portugal)-3]
#print(electricity_production_portugal)
#print(hydro_production_portugal)
hydro_generation_per_capita=pd.read_csv('hydro-electricity-per-capita-generation-kWh.csv')


renewable_production = pd.read_excel(excel_file, sheet_name=54,header=[2],index_col=0)
renewable_production = renewable_production[renewable_production.columns.drop(list(renewable_production.filter(regex='Unnamed:')))]
renewable_production.dropna(axis = 0, how = 'all', inplace = True)
#print(renewable_production)
renewable_production_portugal=renewable_production.loc['Portugal']
renewable_production_portugal=renewable_production_portugal[0:len(renewable_production_portugal)-3]
#print(renewable_production_portugal)
renewable_generation_per_capita=pd.read_csv('renewable-electricity-per-capita-generation-kWh.csv')


solar_production = pd.read_excel(excel_file, sheet_name=56,header=[2],index_col=0)
solar_production = solar_production[solar_production.columns.drop(list(solar_production.filter(regex='Unnamed:')))]
solar_production.dropna(axis = 0, how = 'all', inplace = True)
#print(solar_production)
solar_production_portugal=solar_production.loc['Portugal']
solar_production_portugal=solar_production_portugal[0:len(solar_production_portugal)-3]
#print(solar_production_portugal)
solar_generation_per_capita=pd.read_csv('solar-electricity-per-capita-generation-kWh.csv')


wind_production = pd.read_excel(excel_file, sheet_name=58,header=[2],index_col=0)
wind_production = wind_production[wind_production.columns.drop(list(wind_production.filter(regex='Unnamed:')))]
wind_production.dropna(axis = 0, how = 'all', inplace = True)
#print(wind_production)
wind_production_portugal=wind_production.loc['Portugal']
wind_production_portugal=wind_production_portugal[0:len(wind_production_portugal)-3]
#print(wind_production_portugal)
wind_generation_per_capita=pd.read_csv('wind-electricity-per-capita-generation-kWh.csv')


geo_biomass_production = pd.read_excel(excel_file, sheet_name=60,header=[2],index_col=0)
geo_biomass_production = geo_biomass_production[geo_biomass_production.columns.drop(list(geo_biomass_production.filter(regex='Unnamed:')))]
geo_biomass_production.dropna(axis = 0, how = 'all', inplace = True)
#print(geo_biomass_production)
geo_biomass_production_portugal=geo_biomass_production.loc['Portugal']
geo_biomass_production_portugal=geo_biomass_production_portugal[0:len(geo_biomass_production_portugal)-3]
#print(geo_biomass_production_portugal)

biofuels_production = pd.read_excel(excel_file, sheet_name=63,header=[2],index_col=0)
biofuels_production = biofuels_production[biofuels_production.columns.drop(list(biofuels_production.filter(regex='Unnamed:')))]
biofuels_production.dropna(axis = 0, how = 'all', inplace = True)
not_ignore_col=[]
for i in range(len(biofuels_production.columns)-3):
    not_ignore_col.append(biofuels_production.columns[i])
biofuels_production[not_ignore_col] = biofuels_production[not_ignore_col].multiply(0.277777778)
#print(biofuels_production)
biofuels_production_portugal=biofuels_production.loc['Portugal']
biofuels_production_portugal=biofuels_production_portugal[0:len(biofuels_production_portugal)-3]
#print(biofuels_production_portugal)


#---------------------------------------DATA FOR PRICING---------------------------------------
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

df_hydro=pd.read_csv('hydro_production_clean.csv')
df_nuclear=pd.read_csv('nuclear_production_clean.csv')
df_biofuels=pd.read_csv('biofuels_production_clean.csv')
df_coal=pd.read_csv('coal_production_clean.csv')
df_gas=pd.read_csv('gas_production_clean.csv')
df_geo_biomass=pd.read_csv('geo_biomass_production_clean.csv')
df_renewable=pd.read_csv('renewable_production_clean.csv')
df_solar=pd.read_csv('solar_production_clean.csv')
df_wind=pd.read_csv('wind_production_clean.csv')
df_electricity=pd.read_csv('electricity_production_clean.csv')
df_oil=pd.read_csv('oil_production_clean.csv')

df_hydro_c=pd.read_csv('hydro_consumption_clean.csv')
df_nuclear_c=pd.read_csv('nuclear_consumption_clean.csv')
df_biofuels_c=pd.read_csv('biofuels_consumption_clean.csv')
df_coal_c=pd.read_csv('coal_consumption_clean.csv')
df_gas_c=pd.read_csv('gas_consumption_clean.csv')
df_geo_biomass_c=pd.read_csv('geo_biomass_consumption_clean.csv')
df_renewable_c=pd.read_csv('renewable_consumption_clean.csv')
df_solar_c=pd.read_csv('solar_consumption_clean.csv')
df_wind_c=pd.read_csv('wind_consumption_clean.csv')
df_primary_c=pd.read_csv('primary_consumption_clean.csv')
df_oil_c=pd.read_csv('oil_consumption_clean.csv')

#------------------------ENERGY TRANSITION--------------------------------
production_fossil_nuclear_renewable=pd.read_csv('production-fossil-nuclear-renewables.csv')
df_elec_cons_transition=pd.read_csv('elec-cons-transition.csv')
df_d_primary_en_transition=pd.read_csv('direct-penergy-transition.csv')
df_primary_en_transition=pd.read_csv('penergy-transition.csv')

geoJSONFile = 'continents.json'
with open(geoJSONFile) as response:
    continents = json.load(response)
    

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

if (electricity_production_portugal == 0).all():
    print('this dataframe is empty - energy')
if (hydro_production_portugal == 0).all():
    print('this dataframe is empty - hydro')
if (renewable_production_portugal == 0).all():
    print('this dataframe is empty - renewables')
if (solar_production_portugal == 0).all():
    print('this dataframe is empty - solar')
if (wind_production_portugal == 0).all():
    print('this dataframe is empty - wind')
if (geo_biomass_production_portugal == 0).all():
    print('this dataframe is empty - geo_biomass')
if (biofuels_production_portugal == 0).all():
    print('this dataframe is empty - biofuels')

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

#-------------------------------------DATA FOR CONSUMPTIONS-------------------------------------
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
renewable_Portugal=pd.DataFrame(renewable_consumptions_portugal)
renewable_Portugal.rename(columns={'Portugal':'Renewable consumption [TWh]'},inplace=True)
renewable_Portugal['year']=renewable_Portugal.index
renewable_Portugal.set_index('year',inplace=True)
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
data_Portugal=pd.merge(data_Portugal,renewable_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,solar_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,wind_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,geo_biomass_Portugal, on='year',how='outer')
data_Portugal=pd.merge(data_Portugal,biofuels_Portugal, on='year',how='outer')
col_number=len(data_Portugal)
data_Portugal.drop(data_Portugal.index[col_number-3:col_number], axis=0, inplace=True) #remove the last rows which correspond to %'s
#print(data_Portugal)

data_Portugal['year']=data_Portugal.index


to_be_delete=[]
for i in range(len(data_Portugal['Bio_fuels consumption [TWh]'])):
    #print('NaN found',data_Portugal.iloc[i,8])
    if math.isnan(data_Portugal.iloc[i,9]):
        print('NaN found',data_Portugal.iloc[i,9])
        to_be_delete.append(i)
   
print(data_Portugal)

models=[]
predictors=[]
path=''
labels=['energy','oil','gas','coal','hydro','renewables','solar','wind','geo_biomass','biofuels']
for label in labels:
    with open(path+'models_consumption/'+label ,'rb') as f:
        models.append(pickle.load(f))

data=data_Portugal
data['year'] = pd.to_datetime(data['year'], format='%Y')
data = data.set_index('year')
data = data.asfreq('YS')
data = data.sort_index()
print(data.head())

for i in range(len(models)-1):
    models[i].fit(np.array(data.index).reshape(-1, 1),data[data.columns[i]])
    upper_limit=len(data[data.columns[i]])-5
    error_forecaster=[]
    for j in range(19):
        forecaster = ForecasterAutoreg(regressor = models[i],lags=j+5)
        forecaster.fit(y=data.iloc[0:upper_limit,i])
        predict_temp=forecaster.predict(steps=5)
        real_temp=data.iloc[upper_limit:upper_limit+5,i]
        error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
    #print(error_forecaster)
    min_error=min(error_forecaster)
    #print(min_error)
    for z in range(len(error_forecaster)):
        if error_forecaster[z]==min_error:
            lag_val=z+5
            print('minimum value for ',lag_val)
    forecaster = ForecasterAutoreg(regressor = models[i],lags=lag_val)
    forecaster.fit(y=data[data.columns[i]])
    predictors.append(forecaster)

i=9
down_limit=len(to_be_delete)
upper_limit=len(data[data.columns[i]])
#regr_forecaster = models[i]
models[i].fit(np.array(data.index[down_limit:upper_limit]).reshape(-1, 1),data.iloc[down_limit:upper_limit,i])
error_forecaster=[]
for j in range(5):
    forecaster = ForecasterAutoreg(regressor = models[i],lags=j+5)
    forecaster.fit(y=data.iloc[down_limit:upper_limit-5,i])
    predict_temp=forecaster.predict(steps=5)
    real_temp=data.iloc[upper_limit-5:upper_limit,i]
    error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
#print(error_forecaster)
min_error=min(error_forecaster)
#print(min_error)
for z in range(len(error_forecaster)):
    if error_forecaster[z]==min_error:
        lag_val=z+5
        print('for ',labels[z],' the minimum error was found for ',lag_val,' lags')
#regr_forecaster.fit(np.array(data.index[down_limit:upper_limit]).reshape(-1, 1),data.iloc[down_limit:upper_limit,i])
forecaster = ForecasterAutoreg(regressor = models[i],lags = lag_val)
forecaster.fit(y=data.iloc[down_limit:upper_limit,i])
predictors.append(forecaster)

highlight=[]
other=[]
count=0
#y2=predictors[0].predict(steps=4)
energy_select_1=[0,1,2,3,4]
energy_select_2=[5,6,7,8,9]
time=[]
year_slider=[2021,2030]
for t in range(10):
    time.append(t)
if len(energy_select_1)!=0:
    for i in energy_select_1:
        indexi=int(i)
        for j in time:
            indexj=int(j)
            highlight.append({'year':year_slider[0]+indexj,'label':labels[indexi],'predict':(predictors[indexi].predict(steps=10))[indexj]})
        count=count+1
        for s in range(11):
            other.append({'year':2010+s,'label':labels[indexi],'real':data_Portugal.iat[s+45,indexi]})
if len(energy_select_2)!=0:
    for i in energy_select_2:
        indexi=int(i)
        for j in time:
            indexj=int(j)
            highlight.append({'year':year_slider[0]+indexj,'label':labels[indexi],'predict':(predictors[indexi].predict(steps=10))[indexj]})
        count=count+1
        for s in range(11):
            other.append({'year':2010+s,'label':labels[indexi],'real':data_Portugal.iat[s+45,indexi]})
   
df_highlight_temp = pd.DataFrame(highlight)
df_others_temp = pd.DataFrame(other)
column_names=['year']
for type_label in df_highlight_temp["label"].unique():
    column_names.append(str(type_label))
df_highlight = pd.DataFrame(columns = column_names)
df_others = pd.DataFrame(columns = column_names)  

for group in df_others_temp['year'].unique():
    #print('group: ', group)
    data_temp = df_others_temp[df_others_temp['year'] == group]
    other_temp=[group]
    count=0
    for type_label in data_temp['label']:
        other_temp.append(data_temp.iat[count,2])
        #print('inserting from ',type_label,' the value ',data.iat[count,2])
        count=count+1
    df_others=pd.concat([df_others,pd.DataFrame([other_temp],columns = column_names)],ignore_index=True)
       
for group in df_highlight_temp['year'].unique():
    data_temp = df_highlight_temp[df_highlight_temp['year'] == group]
    highlight_temp=[group]
    count=0
    for type_label in data_temp["label"]:
        highlight_temp.append(data_temp.iat[count,2])
        #print('inserting in',data.iat[count,2])
        count=count+1
    df_highlight=pd.concat([df_highlight,pd.DataFrame([highlight_temp],columns = column_names)], ignore_index=True)
       
print('printing highlights...')
print(df_highlight)
print('printing dataframe others...')
print(df_others)

df_highlight['%_renewables']=df_highlight['renewables']/df_highlight['energy']*100


#-------------------------------------DATA FOR PRODUCTION-------------------------------------

#print(type(electricity_production_portugal)) #all of the data is a pandas.core.series.Series

energy_Portugal=pd.DataFrame(electricity_production_portugal)
energy_Portugal.rename(columns={'Portugal':'Energy production [TWh]'},inplace=True)
energy_Portugal['year']=energy_Portugal.index
energy_Portugal.set_index('year',inplace=True)
hydro_Portugal=pd.DataFrame(hydro_production_portugal)
hydro_Portugal.rename(columns={'Portugal':'Hydro production [TWh]'},inplace=True)
hydro_Portugal['year']=hydro_Portugal.index
hydro_Portugal.set_index('year',inplace=True)
renewables_Portugal=pd.DataFrame(renewable_production_portugal)
renewables_Portugal.rename(columns={'Portugal':'Renewables production [TWh]'},inplace=True)
renewables_Portugal['year']=renewables_Portugal.index
renewables_Portugal.set_index('year',inplace=True)
solar_Portugal=pd.DataFrame(solar_production_portugal)
solar_Portugal.rename(columns={'Portugal':'Solar production [TWh]'},inplace=True)
solar_Portugal['year']=solar_Portugal.index
solar_Portugal.set_index('year',inplace=True)
wind_Portugal=pd.DataFrame(wind_production_portugal)
wind_Portugal.rename(columns={'Portugal':'Wind production [TWh]'},inplace=True)
wind_Portugal['year']=wind_Portugal.index
wind_Portugal.set_index('year',inplace=True)
geo_biomass_Portugal=pd.DataFrame(geo_biomass_production_portugal)
geo_biomass_Portugal.rename(columns={'Portugal':'Geo_biomass production [TWh]'},inplace=True)
geo_biomass_Portugal['year']=geo_biomass_Portugal.index
geo_biomass_Portugal.set_index('year',inplace=True)
biofuels_Portugal=pd.DataFrame(biofuels_production_portugal)
biofuels_Portugal.rename(columns={'Portugal':'Bio_fuels production [TWh]'},inplace=True)
biofuels_Portugal['year']=biofuels_Portugal.index
biofuels_Portugal.set_index('year',inplace=True)

data_Portugal_production=pd.merge(energy_Portugal,hydro_Portugal, on='year',how='right')
data_Portugal_production=pd.merge(data_Portugal_production,renewables_Portugal, on='year',how='outer')
data_Portugal_production=pd.merge(data_Portugal_production,solar_Portugal, on='year',how='outer')
data_Portugal_production=pd.merge(data_Portugal_production,wind_Portugal, on='year',how='outer')
data_Portugal_production=pd.merge(data_Portugal_production,geo_biomass_Portugal, on='year',how='outer')
data_Portugal_production=pd.merge(data_Portugal_production,biofuels_Portugal, on='year',how='outer')
#col_number=len(data_Portugal_production)
#data_Portugal.drop(data_Portugal.index[col_number-3:col_number], axis=0, inplace=True) #remove the last rows which correspond to %'s
#print(data_Portugal_production)

data_Portugal_production['year']=data_Portugal_production.index

to_be_delete_energy=[]
for i in range(len(data_Portugal_production['Energy production [TWh]'])):
    #print('NaN found',data_Portugal_production.iloc[i,8])
    if math.isnan(data_Portugal_production.iloc[i,0]):
        print('NaN found',data_Portugal_production.iloc[i,0])
        to_be_delete_energy.append(i)

to_be_delete=[]
for i in range(len(data_Portugal_production['Bio_fuels production [TWh]'])):
    #print('NaN found',data_Portugal_production.iloc[i,8])
    if math.isnan(data_Portugal_production.iloc[i,6]):
        print('NaN found',data_Portugal_production.iloc[i,6])
        to_be_delete.append(i)
   
print(data_Portugal_production)

models_production=[]
predictors_production=[]
path=''
labels_production=['energy','hydro','renewables','solar','wind','geo_biomass','biofuels']
for label in labels_production:
    with open(path+'models_production/'+label ,'rb') as f:
        models_production.append(pickle.load(f))

data_production=data_Portugal_production
data_production['year'] = pd.to_datetime(data_production['year'], format='%Y')
data_production = data_production.set_index('year')
data_production = data_production.asfreq('YS')
data_production = data_production.sort_index()
print(data_production.head())

i=0
down_limit=len(to_be_delete_energy)
upper_limit=len(data_production[data_production.columns[i]])
models_production[i].fit(np.array(data_production.index[down_limit:upper_limit]).reshape(-1, 1),data_production.iloc[down_limit:upper_limit,i])
error_forecaster=[]
for j in range(5):
    forecaster = ForecasterAutoreg(regressor = models_production[i],lags=j+5)
    forecaster.fit(y=data_production.iloc[down_limit:upper_limit-5,i])
    predict_temp=forecaster.predict(steps=5)
    real_temp=data_production.iloc[upper_limit-5:upper_limit,i]
    error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
#print(error_forecaster)
min_error=min(error_forecaster)
#print(min_error)
for z in range(len(error_forecaster)):
    if error_forecaster[z]==min_error:
        lag_val=z+5
        print('for ',labels[z],' the minimum error was found for ',lag_val,' lags')
#regr_forecaster.fit(np.array(data.index[down_limit:upper_limit]).reshape(-1, 1),data.iloc[down_limit:upper_limit,i])
forecaster = ForecasterAutoreg(regressor = models_production[i],lags = lag_val)
forecaster.fit(y=data_production.iloc[down_limit:upper_limit,i])
predictors_production.append(forecaster)

for i in range(len(models_production)-2):
    i=i+1
    models_production[i].fit(np.array(data_production.index).reshape(-1, 1),data_production[data_production.columns[i]])
    upper_limit=len(data_production[data_production.columns[i]])-5
    error_forecaster=[]
    for j in range(19):
        forecaster = ForecasterAutoreg(regressor = models_production[i],lags=j+5)
        forecaster.fit(y=data_production.iloc[0:upper_limit,i])
        predict_temp=forecaster.predict(steps=5)
        real_temp=data_production.iloc[upper_limit:upper_limit+5,i]
        error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
    #print(error_forecaster)
    min_error=min(error_forecaster)
    #print(min_error)
    for z in range(len(error_forecaster)):
        if error_forecaster[z]==min_error:
            lag_val=z+5
            print('minimum value for ',lag_val)
    forecaster = ForecasterAutoreg(regressor = models_production[i],lags=lag_val)
    forecaster.fit(y=data_production[data_production.columns[i]])
    predictors_production.append(forecaster)

i=6
down_limit=len(to_be_delete)
upper_limit=len(data_production[data_production.columns[i]])
models_production[i].fit(np.array(data_production.index[down_limit:upper_limit]).reshape(-1, 1),data_production.iloc[down_limit:upper_limit,i])
error_forecaster=[]
for j in range(5):
    forecaster = ForecasterAutoreg(regressor = models_production[i],lags=j+5)
    forecaster.fit(y=data_production.iloc[down_limit:upper_limit-5,i])
    predict_temp=forecaster.predict(steps=5)
    real_temp=data_production.iloc[upper_limit-5:upper_limit,i]
    error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
#print(error_forecaster)
min_error=min(error_forecaster)
#print(min_error)
for z in range(len(error_forecaster)):
    if error_forecaster[z]==min_error:
        lag_val=z+5
        print('for ',labels[z],' the minimum error was found for ',lag_val,' lags')
#regr_forecaster.fit(np.array(data.index[down_limit:upper_limit]).reshape(-1, 1),data.iloc[down_limit:upper_limit,i])
forecaster = ForecasterAutoreg(regressor = models_production[i],lags = lag_val)
forecaster.fit(y=data_production.iloc[down_limit:upper_limit,i])
predictors_production.append(forecaster)

highlight_production=[]
other_production=[]
count=0
#y2=predictors[0].predict(steps=4)
energy_select_1=[0,1,2]
energy_select_2=[3,4,5,6]
time=[]
year_slider=[2021,2030]
for t in range(10):
    time.append(t)
if len(energy_select_1)!=0:
    for i in energy_select_1:
        indexi=int(i)
        for j in time:
            indexj=int(j)
            highlight_production.append({'year':year_slider[0]+indexj,'label':labels_production[indexi],'predict':(predictors_production[indexi].predict(steps=10))[indexj]})
        count=count+1
        for s in range(11):
            other_production.append({'year':2010+s,'label':labels_production[indexi],'real':data_Portugal_production.iat[s+45,indexi]})
if len(energy_select_2)!=0:
    for i in energy_select_2:
        indexi=int(i)
        for j in time:
            indexj=int(j)
            highlight_production.append({'year':year_slider[0]+indexj,'label':labels_production[indexi],'predict':(predictors_production[indexi].predict(steps=10))[indexj]})
        count=count+1
        for s in range(11):
            other_production.append({'year':2010+s,'label':labels_production[indexi],'real':data_Portugal_production.iat[s+45,indexi]})
   
df_highlight_temp = pd.DataFrame(highlight_production)
df_others_temp = pd.DataFrame(other_production)
column_names=['year']
for type_label in df_highlight_temp["label"].unique():
    column_names.append(str(type_label))
df_highlight_production = pd.DataFrame(columns = column_names)
df_others_production = pd.DataFrame(columns = column_names)  

for group in df_others_temp['year'].unique():
    data_temp = df_others_temp[df_others_temp['year'] == group]
    other_temp=[group]
    count=0
    for type_label in data_temp['label']:
        other_temp.append(data_temp.iat[count,2])
        count=count+1
    df_others_production=pd.concat([df_others_production,pd.DataFrame([other_temp],columns = column_names)],ignore_index=True)
       
for group in df_highlight_temp['year'].unique():
    data_temp = df_highlight_temp[df_highlight_temp['year'] == group]
    highlight_temp=[group]
    count=0
    for type_label in data_temp["label"]:
        highlight_temp.append(data_temp.iat[count,2])
        count=count+1
    df_highlight_production=pd.concat([df_highlight_production,pd.DataFrame([highlight_temp],columns = column_names)], ignore_index=True)
       
print('printing highlights for production...')
print(df_highlight_production)
print('printing dataframe others for production...')
print(df_others_production)

df_highlight_production['%_renewables']=df_highlight_production['renewables']/df_highlight_production['energy']*100

#-------------------------------------DATA FOR RESOURCES-------------------------------------
#now let's make a dataframe with all the types of energy

#print(type(oil_reserves)) #all of the data is a pandas.core.series.Series

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

data_reserves['year']=data_reserves.index

       
to_be_delete=[]
for i in range(len(data_reserves['Cobalt'])):
    if math.isnan(data_reserves.iloc[i,2]):
        #print('NaN found')
        to_be_delete.append(i)
   
print(data_reserves)

models=[]
predictors=[]
path=''
labels=['oil','gas','cobalt','lithium','graphite','rare_metals']
for label in labels:
    with open(path+'models_resources/'+label ,'rb') as f:
        models.append(pickle.load(f))

data=data_reserves
data['year'] = pd.to_datetime(data['year'], format='%Y')
data = data.set_index('year')
data = data.asfreq('YS')
data = data.sort_index()
print(data.head())

for i in range(len(models)-4):
    models[i].fit(np.array(data.index).reshape(-1, 1),data[data.columns[i]])
    upper_limit=len(data[data.columns[i]])-5
    error_forecaster=[]
    for j in range(10):
        forecaster = ForecasterAutoreg(regressor = models[i],lags=j+5)
        forecaster.fit(y=data.iloc[0:upper_limit,i])
        predict_temp=forecaster.predict(steps=5)
        real_temp=data.iloc[upper_limit:upper_limit+5,i]
        error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
    #print(error_forecaster)
    min_error=min(error_forecaster)
    #print(min_error)
    for z in range(len(error_forecaster)):
        if error_forecaster[z]==min_error:
            lag_val=z+5
            print('minimum value for ',lag_val)
    forecaster = ForecasterAutoreg(regressor = models[i],lags=lag_val)
    forecaster.fit(y=data[data.columns[i]])
    predictors.append(forecaster)
    
down_limit=len(to_be_delete)
for i in range(4):
    i=i+2
    models[i].fit(np.array(data.index[down_limit:upper_limit]).reshape(-1, 1),data.iloc[down_limit:upper_limit,i])
    upper_limit=len(data[data.columns[i]])-5
    error_forecaster=[]
    for j in range(3):
        forecaster = ForecasterAutoreg(regressor = models[i],lags=j+5)
        forecaster.fit(y=data.iloc[down_limit:upper_limit-5,i])
        predict_temp=forecaster.predict(steps=5)
        real_temp=data.iloc[upper_limit-5:upper_limit,i]
        error_forecaster.append(metrics.mean_squared_error(real_temp,predict_temp)) 
    #print(error_forecaster)
    min_error=min(error_forecaster)
    #print(min_error)
    for z in range(len(error_forecaster)):
        if error_forecaster[z]==min_error:
            lag_val=z+5
            print('minimum value for ',lag_val)
    forecaster = ForecasterAutoreg(regressor = models[i],lags=lag_val)
    forecaster.fit(y=data.iloc[down_limit:upper_limit,i])
    predictors.append(forecaster)
    
highlight=[]
other=[]
count=0
#y2=predictors[0].predict(steps=4)
energy_select_1=[0,1,2]
energy_select_2=[3,4,5]
time=[]
year_slider=[2021,2030]
for t in range(10):
    time.append(t)
if len(energy_select_1)!=0:
    for i in energy_select_1:
        indexi=int(i)
        for j in time:
            indexj=int(j)
            highlight.append({'year':year_slider[0]+indexj,'label':labels[indexi],'predict':(predictors[indexi].predict(steps=10))[indexj]})
        count=count+1
        for s in range(11):
            other.append({'year':2010+s,'label':labels[indexi],'real':data_Portugal.iat[s+45,indexi]})
if len(energy_select_2)!=0:
    for i in energy_select_2:
        indexi=int(i)
        for j in time:
            indexj=int(j)
            highlight.append({'year':year_slider[0]+indexj,'label':labels[indexi],'predict':(predictors[indexi].predict(steps=10))[indexj]})
        count=count+1
        for s in range(11):
            other.append({'year':2010+s,'label':labels[indexi],'real':data_Portugal.iat[s+45,indexi]})
   
df_highlight_temp = pd.DataFrame(highlight)
df_others_temp = pd.DataFrame(other)
column_names=['year']
for type_label in df_highlight_temp["label"].unique():
    column_names.append(str(type_label))
df_highlight_resources = pd.DataFrame(columns = column_names)
df_others_resources = pd.DataFrame(columns = column_names)  

for group in df_others_temp['year'].unique():
    #print('group: ', group)
    data_temp = df_others_temp[df_others_temp['year'] == group]
    other_temp=[group]
    count=0
    for type_label in data_temp['label']:
        other_temp.append(data_temp.iat[count,2])
        #print('inserting from ',type_label,' the value ',data.iat[count,2])
        count=count+1
    df_others_resources=pd.concat([df_others_resources,pd.DataFrame([other_temp],columns = column_names)],ignore_index=True)
       
for group in df_highlight_temp['year'].unique():
    data_temp = df_highlight_temp[df_highlight_temp['year'] == group]
    highlight_temp=[group]
    count=0
    for type_label in data_temp["label"]:
        highlight_temp.append(data_temp.iat[count,2])
        #print('inserting in',data.iat[count,2])
        count=count+1
    df_highlight_resources=pd.concat([df_highlight_resources,pd.DataFrame([highlight_temp],columns = column_names)], ignore_index=True)
       
print('printing highlights...')
print(df_highlight_resources)
print('printing dataframe others...')
print(df_others_resources)

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

all_options_p = {
    9: list(df_electricity.iloc[:0,2:]),
    0: list(df_oil.iloc[:0,2:]),
    1: list(df_gas.iloc[:0,2:]),
    2: list(df_coal.iloc[:0,2:]),
    3: list(df_renewable.iloc[:0,2:]),
    4: list(df_hydro.iloc[:0,2:]),
    5: list(df_solar.iloc[:0,2:]),
    6: list(df_nuclear.iloc[:0,2:]),
    7: list(df_geo_biomass.iloc[:0,2:]),
    8: list(df_biofuels.iloc[:0,2:]),
    10: list(df_wind.iloc[:0,2:])
}

all_options_c = {
    9: list(df_primary_c.iloc[:0,2:]),
    0: list(df_oil_c.iloc[:0,2:]),
    1: list(df_gas_c.iloc[:0,2:]),
    2: list(df_coal_c.iloc[:0,2:]),
    3: list(df_renewable_c.iloc[:0,2:]),
    4: list(df_hydro_c.iloc[:0,2:]),
    5: list(df_solar_c.iloc[:0,2:]),
    6: list(df_nuclear_c.iloc[:0,2:]),
    7: list(df_geo_biomass_c.iloc[:0,2:]),
    8: list(df_biofuels_c.iloc[:0,2:]),
    10: list(df_wind_c.iloc[:0,2:])
}


# Style sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "10rem",
    "padding": "2rem 1rem",
    "background-color": "#391542",
    "fontWeight": "bold",
}


# Style page content
CONTENT_STYLE = {
    "margin": 0,
    "margin-left": "8rem",
    "margin-right": "0rem",
    "marginBottom" : 0,
    "padding": "2rem 1rem",
    'height': '2000px', #'200vh',
    "background-size": "cover",
    "background-color" : "#B880C6",
    "responsive" : True,
    "autosize" : True
}


# Define sidebar
sidebar = html.Div(
    [
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home",style={'fontWeight': 'bold', 'color':'white'}, href="/", active="exact"),
                dbc.NavLink("World Demand",style={'fontWeight': 'bold', 'color':'white'}, href="/demand", active="exact"),
                dbc.NavLink("World Generation",style={'fontWeight': 'bold', 'color':'white'}, href="/generation", active="exact"),
                dbc.NavLink("Demand Forecast",style={'fontWeight': 'bold', 'color':'white'}, href="/forecast1", active="exact"),
                dbc.NavLink("Generation Forecast",style={'fontWeight': 'bold', 'color':'white'}, href="/forecast2", active="exact"),
                dbc.NavLink("Resources Forecast",style={'fontWeight': 'bold', 'color':'white'}, href="/forecast3", active="exact"),
                dbc.NavLink("Energy Transition",style={'fontWeight': 'bold', 'color':'white'}, href="/transition", active="exact"),
                dbc.NavLink("Informations",style={'color':'white'}, href="/informations", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


# Define page content
content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)


# Layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)

def render_page_content(pathname):
    if pathname == "/":
        return [            
                html.Div([html.Img(src=app.get_asset_url('IST_A.png'),
                         style = {'height': '25%', 'width': '25%', 'marginTop': 10})], style={'textAlign': 'right'}),

                html.H1('Energy Worldwide',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
            
                html.Div([html.H3('Explore how the energy demand and generation have changed all over the world in the last 50 years.',
                         style={'font-size': '20px','marginTop': 30,'textAlign':'center', 'color': '#414343', "fontWeight": "bold","margin-left": "4rem"})]),
                
                html.Div([html.H3('Here you can find the energy discriminated by countries and by type (fossil fuels, renewables and nuclear).',
                         style={'font-size': '20px','marginTop': 10,'textAlign':'center', 'color': '#414343', "fontWeight": "bold","margin-left": "4rem"})]), 

                html.Div([html.H3('You can also find the energy demand, generation and resources forecasts for Portugal.',
                         style={'font-size': '20px','marginTop': 10,'textAlign':'center', 'color': '#414343', "fontWeight": "bold","margin-left": "4rem"})]),                                     
                html.Div([html.H3('Lastly, you can find a study of energy transition - how energy production and consumption has been shifting from fossil-based into renewable-based systems.',
                         style={'font-size': '20px','marginTop': 10,'textAlign':'center', 'color': '#414343', "fontWeight": "bold","margin-left": "4rem"})]),                                     
                html.Div([html.H3('Here, you can also find a predicion of this energy transition for Portugal in the next 10 years.',
                         style={'font-size': '20px','marginTop': 10,'textAlign':'center', 'color': '#414343', "fontWeight": "bold","margin-left": "4rem"})]),                                     
                                                                
                ]
    
    if pathname == "/generation":
        return [
            html.Div([
                
                html.H1('Energy Generation Worldwide',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                dcc.Dropdown(
                    id='dropdown1',
                    options=[
                        {'label':'Total','value':9},
                        {'label':'Oil','value':0},
                        {'label':'Gas','value':1},
                        {'label':'Coal','value':2},
                        {'label':'Renewables','value':3},
                        {'label':'Hydroelectric','value':4},
                        {'label':'Solar','value':5},
                        {'label':'Nuclear','value':6},
                        {'label':'Geothermal, Biomass and Other','value':7},
                        {'label':'Biofuels','value':8},
                        {'label':'Wind','value':10}
                        ],
                    value=4
                ),
                        
              #  dcc.Dropdown(
              #      id='dropdown4',
               #     options=[{'label': i, 'value': i} for i in list(df_hydro.iloc[:0,2:58])],
                #    value='2020'
                #),
                
                dcc.Dropdown(id='dropdown4'),

                       
                                                
                ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
    html.Div(id='drop2')
    ]

    elif pathname == "/demand":
        return [
            html.Div([
                
                html.H1('Energy Demand Worldwide',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
               
                dcc.Dropdown(
                    id='dropdown2',
                    options=[
                        {'label':'Total','value':9},
                        {'label':'Oil','value':0},
                        {'label':'Gas','value':1},
                        {'label':'Coal','value':2},
                        {'label':'Renewables','value':3},
                        {'label':'Hydroelectric','value':4},
                        {'label':'Solar','value':5},
                        {'label':'Nuclear','value':6},
                        {'label':'Geothermal, Biomass and Other','value':7},
                        {'label':'Biofuels','value':8},
                        {'label':'Wind','value':10}

                        ],
                    value=4
                ),
                
                        
             #   dcc.Dropdown(
             #       id='dropdown3',
             #       options=[{'label': i, 'value': i} for i in list(df_hydro.iloc[:0,2:58])],
             #       value='2020'
             #   ),
             
                dcc.Dropdown(id='dropdown3'),

                       
                                                
                ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
    html.Div(id='drop3')
    ]
    
    elif pathname=="/transition":
        return[
            html.Div([
                html.H1('Energy Transition',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                html.Label(['Here you can explore how energy production and consumption has been gradually shifting from fossil-based into renewable-based systems'],
                        style={'font-size': '20px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                dcc.Dropdown(
                    id='dropdown_cons_prod',
                    options=[
                        {'label':'Consumption','value':0},
                        {'label':'Production','value':1},
                        {'label':'Consumption Prediction','value':2},
                        {'label':'Production Prediction','value':3},
                        ],
                    value=1
                ),

                dcc.Dropdown(
                    id='dropdown_transition',
                    options=[
                        {'label':'World','value':0},
                        {'label':'Portugal','value':1},
                        ],
                    value=1
                ),
                
                ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
    html.Div(id='transition')
            
            ]
    
    elif pathname == '/forecast1':
        return [
            html.Div([
                html.H1('Energy Demand for Portugal',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                html.Div([
                    html.Div([html.Label('years interval, 20xx'),
                              dcc.RangeSlider(21, 30, 1, value=[21, 22], id='my-range-slider'),
                              daq.ToggleSwitch(id='toggle-theme',label=['Line', 'Area'],value=False),
                              ], style={'width': '45%', 'display': 'inline-block','marginBottom': '1.0 em',"margin-left": 40,'marginTop': 20}),
                    html.Div([
                        html.Div(
                            style={'width':'50%','float':'right'},
                            children=[
                                dcc.Checklist(#className ='checkbox_1',
                                              options=[
                                                  {'label': 'Energy', 'value': '0'},
                                                  {'label': 'Oil', 'value': '1'},
                                                  {'label': 'Gas', 'value': '2'},
                                                  {'label': 'Coal', 'value': '3'},
                                                  {'label': 'Hydroelectricity', 'value': '4'}
                                                  ],
                                              value=['3','4'],
                                              labelStyle = {'display': 'block'},
                                              id='checklist_select_energy_1'
                                              ),
                                ]
                            ),
                        html.Div(
                            style={'width':'50%','float':'right'},
                            children=[
                                dcc.Checklist(#className ='checkbox_1',
                                              options=[
                                                  {'label': 'Renewables', 'value': '5'},
                                                  {'label': 'Solar', 'value': '6'},
                                                  {'label': 'Wind', 'value': '7'},
                                                  {'label': 'Geothermal, Biomass and others', 'value': '8'},
                                                  {'label': 'Biofuels', 'value': '9'}
                                                  ],
                                              value=['7'],
                                              labelStyle = {'display': 'block'},
                                              id='checklist_select_energy_2'
                                              )
                                ]
                            ),
                        
                        #id='checklist_select_energy'
                        #, style={'width': '60%', 'float': 'right', 'display': 'inline-block','marginBottom': '1.0 em',"margin-left": 10,'marginTop': 10,"margin-right":10}
                        #dcc.Graph(id='predict_demand', style={'width': '80%','marginBottom': '1.0 em',"margin-left": "2rem",'marginTop': 20,"margin-right":30}),
                        ],style={'width':'45%','display': 'inline-block','float':'right'})
                    ]),
                dcc.Graph(id='predict_demand', style={'width': '100%','marginBottom': '1.0 em',"margin-left": "2rem",'marginTop': 20,"margin-right":30}),
                ])
            ]
    elif pathname == '/forecast2':
        return [
            html.Div([
                html.H1('Energy Production in Portugal',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                html.Div([
                    html.Div([html.Label('years interval, 20xx'),
                              dcc.RangeSlider(21, 30, 1, value=[21, 22], id='my-range-slider-prod'),
                              daq.ToggleSwitch(id='toggle-theme-prod',label=['Line', 'Area'],value=False),
                              ], style={'width': '45%', 'display': 'inline-block','marginBottom': '1.0 em',"margin-left": 40,'marginTop': 20}),
                    html.Div([
                        html.Div(
                            style={'width':'50%','float':'right'},
                            children=[
                                dcc.Checklist(#className ='checkbox_1',
                                              options=[
                                                  {'label': 'Energy', 'value': '0'},
                                                  {'label': 'Hydroelectricity', 'value': '1'},
                                                  {'label': 'Renewables', 'value': '2'}
                                                  ],
                                              value=['1'],
                                              labelStyle = {'display': 'block'},
                                              id='checklist_select_energy_1-prod'
                                              ),
                                ]
                            ),
                        html.Div(
                            style={'width':'50%','float':'right'},
                            children=[
                                dcc.Checklist(#className ='checkbox_1',
                                              options=[
                                                  {'label': 'Solar', 'value': '3'},
                                                  {'label': 'Wind', 'value': '4'},
                                                  {'label': 'Geothermal, Biomass and others', 'value': '5'},
                                                  {'label': 'Biofuels', 'value': '6'}
                                                  ],
                                              value=['6'],
                                              labelStyle = {'display': 'block'},
                                              id='checklist_select_energy_2-prod'
                                              )
                                ]
                            ),
                        
                        #id='checklist_select_energy'
                        #, style={'width': '60%', 'float': 'right', 'display': 'inline-block','marginBottom': '1.0 em',"margin-left": 10,'marginTop': 10,"margin-right":10}
                        #dcc.Graph(id='predict_demand', style={'width': '80%','marginBottom': '1.0 em',"margin-left": "2rem",'marginTop': 20,"margin-right":30}),
                        ],style={'width':'45%','display': 'inline-block','float':'right'})
                    ]),
                dcc.Graph(id='predict_production', style={'width': '100%','marginBottom': '1.0 em',"margin-left": "2rem",'marginTop': 20,"margin-right":30}),
                ])
            ]
    elif pathname == '/forecast3':
        return [
            html.Div([
                html.H1('Predictions of Resources',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                html.Div([
                    html.Div([html.Label('years interval, 20xx'),
                              dcc.RangeSlider(21, 30, 1, value=[21, 22], id='my-range-slider-res'),
                              daq.ToggleSwitch(id='toggle-theme-res',label=['Line', 'Area'],value=False),
                              ], style={'width': '45%', 'display': 'inline-block','marginBottom': '1.0 em',"margin-left": 40,'marginTop': 20}),
                    html.Div([
                        html.Div(
                            style={'width':'50%','float':'right'},
                            children=[
                                dcc.Checklist(#className ='checkbox_1',
                                              options=[
                                                  {'label': 'Oil', 'value': '0'},
                                                  {'label': 'Gas', 'value': '1'},
                                                  {'label': 'Cobalt', 'value': '2'}
                                                  ],
                                              value=['0'],
                                              labelStyle = {'display': 'block'},
                                              id='checklist_select_energy_1-res'
                                              ),
                                ]
                            ),
                        html.Div(
                            style={'width':'50%','float':'right'},
                            children=[
                                dcc.Checklist(#className ='checkbox_1',
                                              options=[
                                                  {'label': 'Lithium', 'value': '3'},
                                                  {'label': 'Graphite', 'value': '4'},
                                                  {'label': 'Rare Materials', 'value': '5'}
                                                  ],
                                              value=['3'],
                                              labelStyle = {'display': 'block'},
                                              id='checklist_select_energy_2-res'
                                              )
                                ]
                            ),
                        
                        #id='checklist_select_energy'
                        #, style={'width': '60%', 'float': 'right', 'display': 'inline-block','marginBottom': '1.0 em',"margin-left": 10,'marginTop': 10,"margin-right":10}
                        #dcc.Graph(id='predict_demand', style={'width': '80%','marginBottom': '1.0 em',"margin-left": "2rem",'marginTop': 20,"margin-right":30}),
                        ],style={'width':'45%','display': 'inline-block','float':'right'})
                    ]),
                dcc.Graph(id='predict_resources', style={'width': '100%','marginBottom': '1.0 em',"margin-left": "2rem",'marginTop': 20,"margin-right":30}),
                ])
            ]
    
    elif pathname == "/informations":
        return[
            html.Div([
                html.H1('Informations',
                        style={'font-size': '60px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                html.H2('Energy Services Project #3',
                        style={'font-size': '30px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                html.H3('Done by:',
                        style={'font-size': '20px','marginTop': 20,'textAlign':'left', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                html.Label(['Diana Bernardo 90384 - Graphs of World Consumption and Production Per Capita & Energy Transition - Production, Consumption Prediction, Production Prediction'],
                        style={'font-size': '20px','marginTop': 20,'textAlign':'left', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                
                html.Label(['Sofia Costa 90426 - Forecast of Consumption, Production and Resources Portugal'],
                        style={'font-size': '20px','marginTop': 20,'textAlign':'left', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                
                html.Label(['Catarina Neves 91036 - Map of World Consumption and Production & Energy Transition - Consumption'],
                        style={'font-size': '20px','marginTop': 20,'textAlign':'left', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                
                
                html.Label(['with data from BP Statistical Review of World Energy 2021 and Our World in Data'],
                        style={'font-size': '20px','marginTop': 20,'textAlign':'left', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                
                
            ]),
    html.Div(id='transition')
            
            ]
            

    
labels=['energy','oil','gas','coal','hydro','renewables','solar','wind','geo_biomass','biofuels']

@app.callback(Output('dropdown4', 'options'),
    [Input('dropdown1', 'value')])

def set_years_options(selected_energy):
    return [{'label': i, 'value': i} for i in all_options_p[selected_energy]]

@app.callback(
    dash.dependencies.Output('dropdown4', 'value'),
    [dash.dependencies.Input('dropdown4', 'options')])

def set_years_value(available_options):
    return available_options[30]['value']

@app.callback(
    dash.dependencies.Output('drop2', 'children'),
    [dash.dependencies.Input('dropdown4', 'value'),
     dash.dependencies.Input('dropdown1', 'value')])
    
def create_graph_generation(chosen_year,chosen_energy):
    
    if chosen_energy==9:
            return html.Div([
                        html.Label(['Total Primary Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                         figure=px.choropleth(data_frame=df_electricity, locations=df_electricity["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                     height=600,hover_name='name'),
                         ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': electricity_generation_per_capita['Years'], 'y': electricity_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Primary Energy Generation (kWh)'},'title': 'Primary Energy Generation per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_energy==0:
            return html.Div([
                        html.Label(['Oil Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                         figure=px.choropleth(data_frame=df_oil, locations=df_oil["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                     height=600,hover_name='name'),
                         ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': oil_generation_per_capita['Years'], 'y': oil_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Oil Generation (kWh)'},'title': 'Oil Generation per capita (kWh) for some representative countires'}
                                }
                            ),
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_energy==1:
            return html.Div([
                        html.Label(['Gas Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                         figure=px.choropleth(data_frame=df_gas, locations=df_gas["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                     height=600,hover_name='name'),
                         ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': gas_generation_per_capita['Years'], 'y': gas_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Gas Generation (kWh)'},'title': 'Gas Generation per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==2:
            return html.Div([
                        html.Label(['Coal Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_coal, locations=df_coal["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': coal_generation_per_capita['Years'], 'y': coal_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Coal Generation (kWh)'},'title': 'Coal Generation per capita (kWh) for some representative countires'}
                                }
                            ), 

                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==3:
            return html.Div([
                        html.Label(['Total Renewable Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_renewable, locations=df_renewable["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': renewable_generation_per_capita['Years'], 'y': renewable_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Renewables Generation (kWh)'},'title': 'Renewables Generation per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})

    if chosen_energy==4:
            return html.Div([
                        html.Label(['Hydroelectric Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_hydro, locations=df_hydro["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': hydro_generation_per_capita['Years'], 'y': hydro_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Hydroelectric Generation (kWh)'},'title': 'Hydroelectric Generation per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==5:
            return html.Div([
                        html.Label(['Solar Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_solar, locations=df_solar["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': solar_generation_per_capita['Years'], 'y': solar_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Solar Generation (kWh)'},'title': 'Solar Generation per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==6:
            return html.Div([
                        html.Label(['Nuclear Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_nuclear, locations=df_nuclear["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': nuclear_generation_per_capita['Years'], 'y': nuclear_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Nuclear Generation (kWh)'},'title': 'Nuclear Generation per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==7:
            return html.Div([
                        html.Label(['Geothermal, Biomass and Other Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_geo_biomass, locations=df_geo_biomass["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.H3('Countries in gray had no data',
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        
                        html.H3('No per capita data available for this type of energy',
                                style={'font-size': '30px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==8:
            return html.Div([
                        html.Label(['Biofuel Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_biofuels, locations=df_biofuels["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.H3('Countries in gray had no data',
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        
                        html.H3('No per capita data available for this type of energy',
                                style={'font-size': '30px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==10:
            return html.Div([
                        html.Label(['Wind Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map1',
                            figure=px.choropleth(data_frame=df_wind, locations=df_wind["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                            dcc.Graph(
                                figure={
                                    'data': [{'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                             {'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                             {'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                             {'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                             {'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                             {'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                             {'x': wind_generation_per_capita['Years'], 'y': wind_generation_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Wind Generation (kWh)'},'title': 'Wind Generation per capita (kWh) for some representative countires'}
                                    }
                                ), 

                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
                    
        
        
@app.callback(Output('dropdown3', 'options'),
    [Input('dropdown2', 'value')])

def set_years_options_c(selected_energy):
    return [{'label': i, 'value': i} for i in all_options_c[selected_energy]]

@app.callback(
    dash.dependencies.Output('dropdown3', 'value'),
    [dash.dependencies.Input('dropdown3', 'options')])

def set_years_value_c(available_options):
    return available_options[30]['value']

@app.callback(
    dash.dependencies.Output('drop3', 'children'),
    [dash.dependencies.Input('dropdown3', 'value'),
     dash.dependencies.Input('dropdown2', 'value')])
        

def create_graph_demand(chosen_year,chosen_energy):
    if chosen_energy==9:
            return html.Div([
                        html.Label(['Total Primary Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                         figure=px.choropleth(data_frame=df_primary_c, locations=df_primary_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                     height=600,hover_name='name'),
                         ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': df_pe_pc['index'], 'y': df_pe_pc['US'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': df_pe_pc['index'], 'y': df_pe_pc['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': df_pe_pc['index'], 'y': df_pe_pc['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': df_pe_pc['index'], 'y': df_pe_pc['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': df_pe_pc['index'], 'y': df_pe_pc['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': df_pe_pc['index'], 'y': df_pe_pc['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': df_pe_pc['index'], 'y': df_pe_pc['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Primary Energy Consumption (kWh)'},'title': 'Primary Energy Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_energy==0:
            return html.Div([
                        html.Label(['Oil Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                         figure=px.choropleth(data_frame=df_oil_c, locations=df_oil_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                     height=600,hover_name='name'),
                         ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': oil_consumptions_per_capita['Years'], 'y': oil_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Oil Consumption (kWh)'},'title': 'Oil Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_energy==1:
            return html.Div([
                        html.Label(['Gas Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                         figure=px.choropleth(data_frame=df_gas_c, locations=df_gas_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                     height=600,hover_name='name'),
                         ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': gas_consumptions_per_capita['Years'], 'y': gas_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Gas Consumption (kWh)'},'title': 'Gas Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==2:
            return html.Div([
                        html.Label(['Coal Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_coal_c, locations=df_coal_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': coal_consumptions_per_capita['Years'], 'y': coal_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Coal Consumption (kWh)'},'title': 'Coal Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==3:
            return html.Div([
                        html.Label(['Total Renewable Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_renewable_c, locations=df_renewable_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': renewable_consumptions_per_capita['Years'], 'y': renewable_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Renewables Consumption (kWh)'},'title': 'Renewables Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})

    if chosen_energy==4:
            return html.Div([
                        html.Label(['Hydroelectric Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_hydro_c, locations=df_hydro_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': hydro_consumptions_per_capita['Years'], 'y': hydro_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Hydroelectric Consumption (kWh)'},'title': 'Hydroelectric Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==5:
            return html.Div([
                        html.Label(['Solar Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_solar_c, locations=df_solar_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': solar_consumptions_per_capita['Years'], 'y': solar_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Solar Consumption (kWh)'},'title': 'Solar Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})

    if chosen_energy==6:
            return html.Div([
                        html.Label(['Nuclear Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_nuclear_c, locations=df_nuclear_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': nuclear_consumptions_per_capita['Years'], 'y': nuclear_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Nuclear Consumption (kWh)'},'title': 'Nuclear Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==7:
            return html.Div([
                        html.Label(['Geothermal, Biomass and Other Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_geo_biomass_c, locations=df_geo_biomass_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        
                        html.H3('Countries in gray had no data',
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        html.H3('No per capita data available for this type of energy',
                                style={'font-size': '30px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
        
    if chosen_energy==8:
            return html.Div([
                        html.Label(['Biofuel Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_biofuels_c, locations=df_biofuels_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.H3('Countries in gray had no data',
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        
                        html.H3('No per capita data available for this type of energy',
                                style={'font-size': '30px','marginTop': 20,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
                    
    if chosen_energy==10:
            return html.Div([
                        html.Label(['Wind Energy (TWh)'],style={'font-size': '20px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"}),
                        dcc.Graph(
                        id='map',
                            figure=px.choropleth(data_frame=df_wind_c, locations=df_wind_c["name"], locationmode='country names',color_discrete_sequence=['blue'],color=chosen_year,
                                                         height=600,hover_name='name'),
                        ),
                        html.Label(['Countries in gray had no data'],
                                style={'font-size': '16px','marginTop': 10,'textAlign':'center', 'color': 'white', "fontWeight": "bold","margin-left": "3rem"}),
                        
                        dcc.Graph(
                            figure={
                                'data': [{'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['United States'], 'type': 'line', 'name': 'US','line':dict(color='purple')},
                                         {'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['Portugal'], 'type': 'line', 'name': 'Portugal','line':dict(color='red')},
                                         {'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['China'], 'type': 'line', 'name': 'China','line':dict(color='green')},
                                         {'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['United Kingdom'], 'type': 'line', 'name': 'United Kingdom','line':dict(color='blue')},
                                         {'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['Brazil'], 'type': 'line', 'name': 'Brazil','line':dict(color='orange')},
                                         {'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['Australia'], 'type': 'line', 'name': 'Australia','line':dict(color='black')},
                                         {'x': wind_consumptions_per_capita['Years'], 'y': wind_consumptions_per_capita['South Africa'], 'type': 'line', 'name': 'South Africa','line':dict(color='aqua')}],
                                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Wind Consumption (kWh)'},'title': 'Wind Consumption per capita (kWh) for some representative countires'}
                                }
                            ), 
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
                    
        
@app.callback(Output('transition', 'children'),
    [Input('dropdown_transition', 'value'),Input('dropdown_cons_prod','value')])

def create_graph_transition(chosen_region,chosen_cons_pred):
    if chosen_region==0 and chosen_cons_pred==0:
            return html.Div([
                        dcc.Graph(
                figure={
                    'data': [{'x':df_elec_cons_transition['Year'], 'y': df_elec_cons_transition['Ren_share_World'], 'type': 'line', 'name': 'World','line':dict(color='purple')}],
                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Electricity Consumption (%)'},'title': 'World Electricity Consumption From Renewable Sources Over the Years'}
                    }
                ),
                        
                        dcc.Graph(
                figure={
                    'data': [{'x':df_d_primary_en_transition['Year'], 'y': df_d_primary_en_transition['World_ren'], 'type': 'line', 'name': 'Portugal','line':dict(color='purple')}],
                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Direct Primary Energy Consumption (%)'},'title': 'World Direct Primary Energy Consumption From Renewable Sources Over the Years'}
                    }
                ),
                        
                    dcc.Graph(
            figure={
                'data': [{'x':df_primary_en_transition['Year'], 'y': df_primary_en_transition['World_ren'], 'type': 'line', 'name': 'Portugal','line':dict(color='purple')}],
                'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Primary Energy Consumption (%)'},'title': 'World Primary Energy Consumption From Renewable Sources Over the Years'}
                }
            ),
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    
    if chosen_region==1 and chosen_cons_pred==0:
            return html.Div([
                        dcc.Graph(
                figure={
                    'data': [{'x':df_elec_cons_transition['Year'], 'y': df_elec_cons_transition['Ren_share_Portugal'], 'type': 'line', 'name': 'World','line':dict(color='purple')}],
                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Electricity Consumption (%)'},'title': 'Portugal Electricity Consumption From Renewable Sources Over the Years'}
                    }
                ),
                        dcc.Graph(
                figure={
                    'data': [{'x':df_d_primary_en_transition['Year'], 'y': df_d_primary_en_transition['Portugal_ren'], 'type': 'line', 'name': 'Portugal','line':dict(color='purple')}],
                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Direct Primary Energy Consumption (%)'},'title': 'Portugal Direct Primary Energy Consumption From Renewable Sources Over the Years'}
                    }
                ),
                        
                        dcc.Graph(
                figure={
                    'data': [{'x':df_primary_en_transition['Year'], 'y': df_primary_en_transition['Portugal_ren'], 'type': 'line', 'name': 'Portugal','line':dict(color='purple')}],
                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Primary Energy Consumption (%)'},'title': 'Portugal Primary Energy Consumption From Renewable Sources Over the Years'}
                    }
                ),
                    ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_region==0 and chosen_cons_pred==1:
        return html.Div([
                        
            dcc.Graph(
                figure={
                    'data': [{'x': production_fossil_nuclear_renewable['Year'], 'y': production_fossil_nuclear_renewable['World Renewables/Total'], 'type': 'line', 'name': 'US','line':dict(color='purple')}],
                    'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Percentage of Renewables'},'title': 'World Percentage of Renewables in Energy Production'}
                    }
                ),   
                        
            ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_region==1 and chosen_cons_pred==1:
         return html.Div([
                         
             dcc.Graph(
                 figure={
                     'data': [{'x': production_fossil_nuclear_renewable['Year Portugal'], 'y': production_fossil_nuclear_renewable['Portugal Renewables/Total'], 'type': 'line', 'name': 'US','line':dict(color='purple')}],
                     'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Percentage of Renewables'},'title': 'Portugal Percentage of Renewables in Energy Production'}
                     }
                 ),  
                         
             ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_region==0 and chosen_cons_pred==2:
        return html.Div([html.Label(['World Forecast was not done in this project']),],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
    
    if chosen_region==0 and chosen_cons_pred==3:
        return html.Div([html.Label(['World Forecast was not done in this project']),],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})

    if chosen_region==1 and chosen_cons_pred==2:
         return html.Div([
                         
             dcc.Graph(
                 figure={
                     'data': [{'x': df_highlight['year'], 'y': df_highlight['%_renewables'], 'type': 'line', 'line':dict(color='purple')}],
                     'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Percentage of Renewables'},'title': 'Portugal Prediction of Percentage of Renewables in Energy Consumption'}
                     }
                 ),  
                         
             ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
  
    if chosen_region==1 and chosen_cons_pred==3:
         return html.Div([
                         
             dcc.Graph(
                 figure={
                     'data': [{'x': df_highlight_production['year'], 'y': df_highlight_production['%_renewables'], 'type': 'line', 'line':dict(color='purple')}],
                     'layout':{'xaxis': {'title':'Year'},'yaxis': {'title':'Percentage of Renewables'},'title': 'Portugal Prediction of Percentage of Renewables in Energy Production'}
                     }
                 ),                    
             ],style={'font-size': '16px','marginTop': 2,'textAlign':'center', 'color': 'black', "fontWeight": "bold","margin-left": "3rem"})
       
@app.callback(
    Output("predict_demand", "figure"),
    [Input("my-range-slider", "value"),
     Input('checklist_select_energy_1','value'),
     Input('checklist_select_energy_2','value'),
     Input('toggle-theme','value')]
)

def update_graph_demand(year_slider,energy_select_1,energy_select_2,toogle_theme):
    # Shades of gray
    GREY_SCALE=[
       "#1a1a1a",
        "#4d4d4d",
        "#666666",
        "#7f7f7f",
        "#999999",
        "#bfbfbf",
        "#e8e8e8",
       "#fafafa",
       "#839192",
       "#D0D3D4"
    ]
    
        # Colors used to shade countries
    COLOR_SCALE = [
            "#FF7F50",
            "#F5B041",
            "#F7DC6F",
            "#82E0AA",
            "#73C6B6",
            "#5DADE2",
            "#AF7AC5",
            "#EC7063",
            "#EC407A",
            "#D1C4E9"
    ]
    
    marker_style=['circle','square','cross','x','hexagon','hourglass','star','diamond','bowtie','star-square']
   
    t0=int(year_slider[0]-21)
    t1=int(year_slider[1]-21+1)
    data_others=pd.DataFrame(df_others[df_others.columns[0]],columns=[df_others.columns[0]])
    data_highlight=pd.DataFrame(df_highlight.iloc[t0:t1,0],columns=[df_highlight.columns[0]])
    
    if len(energy_select_1)!=0:
        for i in energy_select_1:
            i=int(i)
            data_highlight[df_highlight.columns[i+1]]=df_highlight.iloc[t0:t1,i+1]
            data_others[df_others.columns[i+1]]=df_others[df_others.columns[i+1]]
    if len(energy_select_2)!=0:
        for i in energy_select_2:
            i=int(i)
            data_highlight[df_highlight.columns[i+1]]=df_highlight.iloc[t0:t1,i+1]
            data_others[df_others.columns[i+1]]=df_others[df_others.columns[i+1]]
    if toogle_theme:
        fill_val='tozeroy'
    else:
        fill_val='none'
 
    fig = go.Figure()
    for i in range(len(data_highlight.columns)-1):
        i=int(i)
        fig.add_trace(go.Scatter(x=data_others[data_others.columns[0]], y=data_others[data_others.columns[i+1]], name=data_others.columns[i+1],line=dict(color=GREY_SCALE[i], width=2),marker_symbol=marker_style[i]))
        fig.add_trace(go.Scatter(x=data_highlight[data_highlight.columns[0]], y=data_highlight[data_highlight.columns[i+1]], name=data_highlight.columns[i+1],line=dict(color=COLOR_SCALE[i], width=4),fill=fill_val,marker_symbol=marker_style[i]))
        
    fig.update_layout(title='Consumption',
                   xaxis_title='Year',
                   yaxis_title='Consumption [Twh]')   
    return fig

@app.callback(
    Output("predict_production", "figure"),
    [Input("my-range-slider-prod", "value"),
     Input('checklist_select_energy_1-prod','value'),
     Input('checklist_select_energy_2-prod','value'),
     Input('toggle-theme-prod','value')]
)

def update_graph_production(year_slider,energy_select_1,energy_select_2,toogle_theme):
    # Shades of gray
    GREY_SCALE=[
       "#1a1a1a",
        "#4d4d4d",
        "#666666",
        "#7f7f7f",
        "#999999",
        "#bfbfbf",
        "#e8e8e8",
       "#fafafa",
       "#839192",
       "#D0D3D4"
    ]
    
        # Colors used to shade countries
    COLOR_SCALE = [
            "#FF7F50",
            "#F5B041",
            "#F7DC6F",
            "#82E0AA",
            "#73C6B6",
            "#5DADE2",
            "#AF7AC5",
            "#EC7063",
            "#EC407A",
            "#D1C4E9"
    ]
    
    marker_style=['circle','square','cross','x','hexagon','hourglass','star','diamond','bowtie','star-square']
   
    t0=int(year_slider[0]-21)
    t1=int(year_slider[1]-21+1)
    data_others_production=pd.DataFrame(df_others_production[df_others_production.columns[0]],columns=[df_others_production.columns[0]])
    data_highlight_production=pd.DataFrame(df_highlight_production.iloc[t0:t1,0],columns=[df_highlight_production.columns[0]])
    
    if len(energy_select_1)!=0:
        for i in energy_select_1:
            i=int(i)
            data_highlight_production[df_highlight_production.columns[i+1]]=df_highlight_production.iloc[t0:t1,i+1]
            data_others_production[df_others_production.columns[i+1]]=df_others_production[df_others_production.columns[i+1]]
    if len(energy_select_2)!=0:
        for i in energy_select_2:
            i=int(i)
            data_highlight_production[df_highlight_production.columns[i+1]]=df_highlight_production.iloc[t0:t1,i+1]
            data_others_production[df_others_production.columns[i+1]]=df_others_production[df_others_production.columns[i+1]]
    if toogle_theme:
        fill_val='tozeroy'
    else:
        fill_val='none'
 
    fig = go.Figure()
    for i in range(len(data_highlight_production.columns)-1):
        i=int(i)
        fig.add_trace(go.Scatter(x=data_others_production[data_others_production.columns[0]], y=data_others_production[data_others_production.columns[i+1]], name=data_others_production.columns[i+1],line=dict(color=GREY_SCALE[i], width=2),marker_symbol=marker_style[i]))
        fig.add_trace(go.Scatter(x=data_highlight_production[data_highlight_production.columns[0]], y=data_highlight_production[data_highlight_production.columns[i+1]], name=data_highlight_production.columns[i+1],line=dict(color=COLOR_SCALE[i], width=4),fill=fill_val,marker_symbol=marker_style[i]))
        
    fig.update_layout(title='Production',
                   xaxis_title='Year',
                   yaxis_title='Production [Twh]')   
    return fig
   
@app.callback(
    Output("predict_resources", "figure"),
    [Input("my-range-slider-res", "value"),
     Input('checklist_select_energy_1-res','value'),
     Input('checklist_select_energy_2-res','value'),
     Input('toggle-theme-res','value')]
)

def update_graph_resources(year_slider,energy_select_1,energy_select_2,toogle_theme):
    # Shades of gray
    GREY_SCALE=[
       "#1a1a1a",
        "#4d4d4d",
        "#666666",
        "#7f7f7f",
        "#999999",
        "#bfbfbf",
        "#e8e8e8",
       "#fafafa",
       "#839192",
       "#D0D3D4"
    ]
    
        # Colors used to shade countries
    COLOR_SCALE = [
            "#FF7F50",
            "#F5B041",
            "#F7DC6F",
            "#82E0AA",
            "#73C6B6",
            "#5DADE2",
            "#AF7AC5",
            "#EC7063",
            "#EC407A",
            "#D1C4E9"
    ]
    
    marker_style=['circle','square','cross','x','hexagon','hourglass','star','diamond','bowtie','star-square']
   
    t0=int(year_slider[0]-21)
    t1=int(year_slider[1]-21+1)
    data_others_resources=pd.DataFrame(df_others_resources[df_others_resources.columns[0]],columns=[df_others_resources.columns[0]])
    data_highlight_resources=pd.DataFrame(df_highlight_resources.iloc[t0:t1,0],columns=[df_highlight_resources.columns[0]])
    
    if len(energy_select_1)!=0:
        for i in energy_select_1:
            i=int(i)
            data_highlight_resources[df_highlight_resources.columns[i+1]]=df_highlight_resources.iloc[t0:t1,i+1]
            data_others_resources[df_others_resources.columns[i+1]]=df_others_resources[df_others_resources.columns[i+1]]
    if len(energy_select_2)!=0:
        for i in energy_select_2:
            i=int(i)
            data_highlight_resources[df_highlight_resources.columns[i+1]]=df_highlight_resources.iloc[t0:t1,i+1]
            data_others_resources[df_others_resources.columns[i+1]]=df_others_resources[df_others_resources.columns[i+1]]
    if toogle_theme:
        fill_val='tozeroy'
    else:
        fill_val='none'
 
    fig = go.Figure()
    for i in range(len(data_highlight_resources.columns)-1):
        i=int(i)
        fig.add_trace(go.Scatter(x=data_others_resources[data_others_resources.columns[0]], y=data_others_resources[data_others_resources.columns[i+1]], name=data_others_resources.columns[i+1],line=dict(color=GREY_SCALE[i], width=2),marker_symbol=marker_style[i]))
        fig.add_trace(go.Scatter(x=data_highlight_resources[data_highlight_resources.columns[0]], y=data_highlight_resources[data_highlight_resources.columns[i+1]], name=data_highlight_resources.columns[i+1],line=dict(color=COLOR_SCALE[i], width=4),fill=fill_val,marker_symbol=marker_style[i]))
        
    fig.update_layout(title='Resources',
                   xaxis_title='Year',
                   yaxis_title='Resources')   
    return fig 
if __name__ == '__main__':
    app.run_server(debug=True)
