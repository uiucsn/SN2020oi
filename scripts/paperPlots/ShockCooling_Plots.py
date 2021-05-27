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

#i+5
#r+4
#v+2.5
#g+2
#b
#u'-0.5
#u-2
#w1-5
#m2-5.5
#w2-7

#t0 = 58854.2
t0 = 58854.2#manually shift for alignment
#t0 += 1.8

#bands = np.array(['b','g','m2','u','w2','i','r','up','v','w1'])
#w2 m2 w1 u u' b g v r i
cols = sns.color_palette("colorblind", 10)

cols = np.array(['#927FC5', cols[4], cols[6], cols[0], cols[-1], cols[2], cols[8], cols[1], cols[3], 'tab:red'])
#cols = np.array([cols[3], cols[1], cols[8], cols[2], cols[9], cols[0], cols[4], cols[5], cols[7],cols[6]])
cols = np.flip(cols)

def make_plot():

    SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_finalPhotometry_0133ExtCorr.csv")
#    SN2020oi_old = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_ExtCorrfinalPhotometry_0218.csv")
#    SS_u = SN2020oi_old[SN2020oi_old['FLT']=='up']
#    SS_V = SN2020oi_old[(SN2020oi_old['FLT']=='V') & (SN2020oi_old['TELESCOPE']=='Siding Spring 1m')]
#    SS_b = SN2020oi_old[(SN2020oi_old['FLT']=='B') & (SN2020oi_old['TELESCOPE']=='Siding Spring 1m')]

#    SN2020oi = pd.concat([SN2020oi, SS_u, SS_V, SS_b], ignore_index=True)

    SN2020oi.sort_values(by=['MJD'], inplace=True)
    SN2020oi.loc[SN2020oi['ULIM'] != SN2020oi['ULIM'], 'ULIM'] = 0

    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'GPC1', 'TELESCOPE'] = 'Pan-STARRS1'

    SN2020oi = SN2020oi[SN2020oi['FLT'] != 'orange']
    SN2020oi = SN2020oi[SN2020oi['MAG'] > 13.5]
    SN2020oi = SN2020oi[SN2020oi['FLT'] != 'G']
    SN2020oi['band'] = SN2020oi['FLT']
    SN2020oi.loc[SN2020oi['band'] == 'up', 'band'] = 'u'
    SN2020oi.loc[SN2020oi['band'] == 'gp', 'band'] = 'g'
    SN2020oi.loc[SN2020oi['band'] == 'g-ZTF', 'band'] = 'g'
    SN2020oi.loc[SN2020oi['band'] == 'rp', 'band'] = 'r'
    SN2020oi.loc[SN2020oi['band'] == 'r-ZTF', 'band'] = 'r'
    SN2020oi.loc[SN2020oi['band'] == 'ip', 'band'] = 'i'
    SN2020oi.loc[SN2020oi['band'] == 'zp', 'band'] = 'z'
    SN2020oi.loc[SN2020oi['band'] == 'UVM1', 'band'] = r'$m1$'
    SN2020oi.loc[SN2020oi['band'] == 'UVW1', 'band'] = r'$w1$'
    SN2020oi.loc[SN2020oi['band'] == 'UVW2', 'band'] = r'$w2$'
    SN2020oi.loc[SN2020oi['band'] == 'UVM2', 'band'] = r'$m2$'
    SN2020oi.loc[SN2020oi['band'] == 'B', 'band'] = r'$b$'
    SN2020oi.loc[SN2020oi['band'] == 'U', 'band'] = r'$up$'
    SN2020oi.loc[SN2020oi['band'] == 'V', 'band'] = r'$v$'
    SN2020oi.loc[SN2020oi['band'] == 'u', 'band'] = r'$u$'
    SN2020oi.loc[SN2020oi['band'] == 'g', 'band'] = r'$g$'
    SN2020oi.loc[SN2020oi['band'] == 'r', 'band'] = r'$r$'
    SN2020oi.loc[SN2020oi['band'] == 'i', 'band'] = r'$i$'
    SN2020oi.loc[SN2020oi['band'] == 'z', 'band'] = r'$z$'
    SN2020oi = SN2020oi[SN2020oi['TELESCOPE'] != 'Sinistro']
    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'Direct/2Kx2K', 'TELESCOPE'] = 'Nickel'
    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'Direct/4Kx4K', 'TELESCOPE'] = 'Swope'
    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'Sinistro', 'TELESCOPE'] = 'LCO'
    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'Siding Spring 1m', 'TELESCOPE'] = 'LCO'
    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'P48', 'TELESCOPE'] = 'ZTF'
    SN2020oi.loc[SN2020oi['TELESCOPE'] == 'ZTF-Cam', 'TELESCOPE'] = 'ZTF'

    telescope_dict = {'Nickel':'o', 'P48':".", 'Pan-STARRS1':'^', 'LCO':'s', 'Swift':'*', 'Swope':'D', 'Thacher':'>', 'ZTF':'h', 'Sinistro':'X', 'Synthetic':'d'}

    telescopes = np.unique(SN2020oi['TELESCOPE'].dropna().values)
    telescopes = np.concatenate([telescopes, ['Unk']])

    telescopes = np.concatenate([telescopes, ['Sinistro']])

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
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.right'] = True

    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20), sharey=True)
    ax1.set_ylim((-3, -28.5))

    f.subplots_adjust(wspace=0)
    ax1.set_xlabel("Rest-Frame Days Relative to Explosion",fontsize=30)
    ax2.set_xlabel("Rest-Frame Days Relative to Explosion",fontsize=30)
    ax1.set_ylabel("Absolute Magnitude + Offset",fontsize=30)

    bands = np.array([r'$w2$', '$m2$', '$w1$', '$up$', '$u$',  '$b$', '$g$', '$v$', '$r$', '$i$'])
    bands = np.flip(bands)
    shifts = np.array([-10.5, -8.5, -6, -4, -2, 0., 2., 4., 6., 8.])
    #offset = 0.0
    offset = [1.7, 1.9, 1.95, 1.9]
    #+1.9d, +1.7d, +1.9d and +1.95d for P20, P15, BSG and RSG models

    for i in np.arange(len(bands)):
        temp = SN2020oi[SN2020oi['band'] == bands[i]]
        for j in np.arange(len(telescopes)):
            temp = SN2020oi[SN2020oi['band'] == bands[i]]
            if telescopes[j] == 'Unk':
                temp = temp[temp['TELESCOPE'] != temp['TELESCOPE']]
            else:
                temp = temp[temp['TELESCOPE'] == telescopes[j]]
            temp.loc[temp['ULIM'] == 1, 'MAGERR'] = 0.3
            #print(temp['MAG'])
            if len(temp) > 0:
                if telescope_dict[telescopes[j]] in np.array(['*', '1', '+']):
                    ms = 22
                else:
                    ms = 12
                if bands[i] in [r'$w2$', r'$m2$', r'$w1$']:
                    temp_ulim = temp[temp['ULIM'] == 1]
                    uplims = temp_ulim['ULIM']
                    ax1.errorbar(temp_ulim['MJD']-t0, temp_ulim['MAG']+shifts[i]-31.16,yerr=temp_ulim['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                    ax2.errorbar(temp_ulim['MJD']-t0, temp_ulim['MAG']+shifts[i]-31.16,yerr=temp_ulim['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                    temp_reg = temp[temp['ULIM'] == 0]
                    uplims = temp_reg['ULIM']
                    ax1.errorbar(temp_reg['MJD']-t0, temp_reg['MAG']+shifts[i]-31.16,yerr=temp_reg['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                    ax2.errorbar(temp_reg['MJD']-t0, temp_reg['MAG']+shifts[i]-31.16,yerr=temp_reg['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                else:
                    uplims = temp['ULIM']
                    ax1.errorbar(temp['MJD']-t0, temp['MAG']+shifts[i]-31.16,yerr=temp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                    ax2.errorbar(temp['MJD']-t0, temp['MAG']+shifts[i]-31.16,yerr=temp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                #    if bands[i] in ['$i$', '$B$', '$V$', '$U/u$']:
                        #print(telescope_dict[telescopes[j]])
                        #badtemp = temp[temp['Flag']==1]
                #        uplims = badtemp['ULIM']
                #        ax2.errorbar(badtemp['MJD']-t0, badtemp['MAG']+shifts[i],yerr=badtemp['MAGERR'],  lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='tab:red', ms=ms, mew=3, capsize=5.,capthick=2., alpha=0.5, ecolor=cols[i],mfc=cols[i])
                    #    ax3.errorbar(badtemp['MJD']-t0, badtemp['MAG']+shifts[i],yerr=badtemp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='tab:red', ms=ms, mew=3, capsize=5.,capthick=2.,  alpha=0.5,ecolor=cols[i],mfc=cols[i]) #cols[i]

        tempDF = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/SC_Model_LCs_final/Piro15_SC/Piro15_%s.cat"%bands[i].strip("$"), delim_whitespace=True, header=None, names=['Days', 'Mag'])
        tempDF2 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/SC_Model_LCs_final/SW17_BSG/SW17_BSG_%s.cat"%bands[i].strip("$"), delim_whitespace=True,header=None, names=['Days', 'Mag'])
        tempDF3 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/SC_Model_LCs_final/SW17_RSG/SW17_RSG_%s.cat"%bands[i].strip("$"), delim_whitespace=True,header=None, names=['Days', 'Mag'])
        tempDF4 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/SC_Model_LCs_final/Piro20_SC/Piro20_%s.cat"%bands[i].strip("$"), delim_whitespace=True,header=None, names=['Days', 'Mag'])
        tempDF = tempDF[tempDF['Days'] <= 4]
        tempDF2 = tempDF2[tempDF2['Days'] <= 4]
        tempDF3 = tempDF3[tempDF3['Days'] <= 4]
        tempDF4 = tempDF4[tempDF4['Days'] <= 4]

        if i==0:
            ax1.plot(tempDF['Days']+offset[0], tempDF['Mag']-31.16+shifts[i], ':', lw=3, label="Piro 2015", c=cols[i])
            ax1.plot(tempDF4['Days']+offset[3], tempDF4['Mag']-31.16+shifts[i], ls='dashdot', lw=3, label='Piro 2020', c=cols[i])
            ax2.plot(tempDF2['Days']+offset[1], tempDF2['Mag']-31.16+shifts[i], '--', lw=3, label='SW17 (BSG)', c=cols[i])
            ax2.plot(tempDF3['Days']+offset[2], tempDF3['Mag']-31.16+shifts[i], lw=3, label='SW17 (RSG)', c=cols[i])
        else:
            ax1.plot(tempDF['Days']+offset[0], tempDF['Mag']-31.16+shifts[i], ':', lw=3, c=cols[i])
            ax1.plot(tempDF4['Days']+offset[3], tempDF4['Mag']-31.16+shifts[i], ls='dashdot', lw=3, c=cols[i])
            ax2.plot(tempDF2['Days']+offset[1], tempDF2['Mag']-31.16+shifts[i], '--', lw=3, c=cols[i])
            ax2.plot(tempDF3['Days']+offset[2], tempDF3['Mag']-31.16+shifts[i], lw=3, c=cols[i])

    ax1.set_xlim((0.5, 4.0))
    ax2.set_xlim((0.5, 5))
#    ax2.set_ylim((-24,-5))
    ax1.legend(fontsize=25, borderaxespad=1., loc='upper left')
    ax2.legend(fontsize=25, borderaxespad=1., loc='upper left')

    for ax in [ax1, ax2]:
        ax.text(0.9, 7.5-31.16, r"$i$-10.5", color=cols[0],transform=ax.transData)
        ax.text(0.9, 9.2-31.16, r"$r$-8.5", color=cols[1],transform=ax.transData)
        ax.text(0.9, 11-31.16, r"$V$-6.0", color=cols[2],transform=ax.transData)
        ax.text(0.9, 13-31.16, r"$g$-4.0", color=cols[3],transform=ax.transData)
        ax.text(0.9, 15-31.16, r"$B$-2.0", color=cols[4],transform=ax.transData)
        ax.text(0.9, 17-31.16, r"$u'$", color=cols[5],transform=ax.transData)
        ax.text(0.9, 19-31.16, r"$u$+2.0", color=cols[6],transform=ax.transData)
        ax.text(0.9, 21-31.16, r"$w1$+4.0", color=cols[7],transform=ax.transData)
        ax.text(0.9, 23-31.16, r"$m2$+6.0", color=cols[8],transform=ax.transData)
        ax.text(0.9, 25-31.16, r"$w2$+8.0", color=cols[9],transform=ax.transData)

    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/shockCooling_extCorr.png",dpi=300, bbox_inches='tight')

make_plot()
