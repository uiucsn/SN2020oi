
#new plots for 2020oi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from astropy import wcs
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization import ZScaleInterval
from os import listdir
from os.path import isfile, join
from astropy.time import Time
from astropy.wcs import wcs
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import get_pkg_data_filename
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
import brokenaxes
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib as mpl

Comp = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/FinalComposition.csv")

other = np.ones(len(Comp))
for colname in Comp.columns[Comp.columns != 'Day']:
    other -= Comp[colname]

Comp['Other'] = other

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

DensityModel = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/density_1FOE_CO21_mod_scaled_wBump.dat", delim_whitespace=True)

sns.set_context("talk")
#plt.figure(figsize=(14,14))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


sns.set(font_scale=3.5)
#sns.set_style("white")
sns.set_style('white', {'axes.linewidth': 0.5})
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['ytick.major.size'] = 15

plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 40})
#sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})
#Comp.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/FinalComposition_wOther.csv",index=False)

v1 = 14000 #km/s
v2 = 11000 #km/s
v3 = 8000 #km/s
v4 = 7000 #km/s
v5 = 26000 #km/s

idx1 = find_nearest(DensityModel['velocity(km/s)'], v1)
idx2 = find_nearest(DensityModel['velocity(km/s)'], v2)
idx3 = find_nearest(DensityModel['velocity(km/s)'], v3)
idx4 = find_nearest(DensityModel['velocity(km/s)'], v4)
idx5 = find_nearest(DensityModel['velocity(km/s)'], v5)

plt.figure(figsize=(18,18))
plt.plot(DensityModel['velocity(km/s)'], DensityModel['density(g/cm^3)'], lw=3)
plt.plot(DensityModel['velocity(km/s)'].values[idx1], DensityModel['density(g/cm^3)'].values[idx1], 's', ms=15, c='k')
plt.plot(DensityModel['velocity(km/s)'].values[idx2], DensityModel['density(g/cm^3)'].values[idx2], 's', ms=15, c='k')
plt.plot(DensityModel['velocity(km/s)'].values[idx3], DensityModel['density(g/cm^3)'].values[idx3], 's', ms=15, c='k')
plt.plot(DensityModel['velocity(km/s)'].values[idx4], DensityModel['density(g/cm^3)'].values[idx4], 's', ms=15, c='k')
plt.plot(DensityModel['velocity(km/s)'].values[idx5], DensityModel['density(g/cm^3)'].values[idx5], 's', ms=15, c='k')
plt.yscale("log")
plt.xlabel("Velocity (km/s)")
plt.ylabel("Density (g/cm$^3$)")
#plt.text(6000, 3.e-16, "Day 80", fontsize=40);
plt.ylim((1.e-18, 1.e-14));
plt.xlim((4000, 30000));
plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/DensityModel_wBump.png",dpi=300, bbox_inches='tight')

sns.set(rc={'figure.figsize':(30,20)})

sns.set_context("talk")
#plt.figure(figsize=(14,14))

sns.set(font_scale=2.5)
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 40})
sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

sns.set_style('white', {'axes.linewidth': 0.5})
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['ytick.major.size'] = 15

plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False

plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#sns.set_palette("dark")
cols = sns.color_palette("CMRmap", 10)
cols2 = sns.color_palette("colorblind")
cols_full = np.concatenate([cols, cols2])

Comp.index = Comp['Day']
del Comp['Day']

del Comp['Other']
Comp = Comp[['Fe', 'O', 'Ni', 'He', 'Ne', 'C', 'Na', 'Ar']]
cols = Comp.columns
Comp['Other'] = 1.0
for element in cols:
    Comp['Other'] -= Comp[element]
Comp.loc[Comp['Other']<0., 'Other'] = 0.0
Comp = Comp[Comp.index!=3]
Comp = Comp[Comp.index!=6]

#reorder from most to least abundant:
Comp = Comp[['O', 'Ne', 'He', 'C', 'Ar', 'Na', 'Fe', 'Ni', 'Other']]

#flip for easier reading
Comp.reindex(index=Comp.index[::-1])

if True:
    Comp.replace(0.0, np.nan, inplace=True)
    Comp.dropna(how='all', axis='columns', inplace=True)

    tax = Comp.plot.barh(stacked=True, width=0.4,color=cols_full, legend=True);
    for patch in tax.patches:
        clr = patch.get_facecolor()
        patch.set_edgecolor(clr)
    plt.xlim((-0.01,1.01))
    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/CompositionBars_FePeak_wLegend.png",dpi=200,bbox_inches='tight')
