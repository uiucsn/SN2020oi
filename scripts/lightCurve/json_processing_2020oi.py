import json
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/Users/alexgagliano/Documents/Research/2020oi/data/photometry")

t0 = 58854.2

# Opening JSON file and loading the data
# into the variable data
with open('2020oi_data.json') as json_file:
    data = json.load(json_file)

data = data['2020oi']

# now we will open a file for writing
data_file = open('2020oi_YSEPZ_0222.csv', 'w')

df_arr = []
for emp in data['photometry']:
    for val in emp['data']:
        df_arr.append(pd.DataFrame(val['fields']))

DF = pd.concat(df_arr, ignore_index=True)

DF.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_YSEPZ_0222.csv")

import numpy as np

from astropy.time import Time

allData = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_YSEPZ_0222.csv")
t = Time([str(x) for x in allData['obs_date'].values], format='isot', scale='utc')
allData['MJD'] = t.mjd

allData_late = allData[(allData['MJD']-t0)>190]
#allData_late = allData.copy)
del allData_late['Unnamed: 0']
allData_late.columns

sns.set_context("talk")
#59250-t0
allData_late.loc[allData_late['band'] == 'Sinistro - up', 'band'] = 'u'
allData_late.loc[allData_late['band'] == 'Sinistro - rp', 'band'] = 'r'
allData_late.loc[allData_late['band'] == 'Sinistro - ip', 'band'] = 'i'
allData_late.loc[allData_late['band'] == 'Sinistro - gp', 'band'] = 'g'
allData_late = allData_late[allData_late['band'].isin(['u', 'g', 'r', 'i'])]
allData_late['TELESCOPE'] = 'Sinistro'
allData_late.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_LateTimeUpperLimits.csv",index=False)

plt.figure(figsize=(10,7))
sns.scatterplot(allData_late['MJD']-t0, allData_late['mag'], hue=allData_late['band'])
plt.gca().invert_yaxis()
