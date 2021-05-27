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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

t0 = 58854.2
#t0 = 58855.653

HST = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/HST_allPhotometry.csv",delim_whitespace=True)
HST_phases = HST['MJD'] - t0
HST['ph'] = HST_phases
#for i in HST['ph'].values:
#    print("%.1f"%i)

t = Time(HST['MJD'].values, format='mjd')
t.to_value('datetime')

from astropy.time import Time
times = ['2020-01-27T00:00:00.00', '2020-02-16T00:00:00.00', '2020-01-09T07:17:20.00', '2020-02-13T00:00:00.00', '2020-05-26T14:37:00.00', '2020-01-28T16:50:21.00']
t = Time(times, format='isot')

MJDs = np.array([58857.3,
58864.6,
58868.6,
58870.5,
58872.4,
58876.7,
58879.4,
58880.3,
58880.4,
58892.0,
58995.6])

peak = 58866.1

mpl.rcParams['patch.linewidth'] = 15
Lbol = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/scripts/superbol/superbol_output_2020oi/logL_bb_2020oi_AUBgVriz.txt")
Lbol['ph'] = Lbol['MJD'] - peak

sns.set_context("poster")

#peak - t0

#all data from 2020oi -- updated as of 03/11 with new Siding Spring photometry!
#SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry.csv")
#SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry_wSynthetic.csv")
SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry_wSynthetic_rBand.csv")


cols = sns.color_palette("colorblind", 10)
cols_hex = cols.as_hex()
cols = np.array([cols[3], cols[1], cols[8], cols[2], cols[9], cols[0], cols[4], cols[5], cols[7],cols[6]])

np.unique(SN2020oi['FLT'])

SN2020oi_sub = SN2020oi[SN2020oi['TELESCOPE']!='Sinistro']
#bands_earlyPts = np.array(['gp', 'rp', 'z', 'g-ZTF', 'r-ZTF', 'r', 'u', 'up', 'U', 'V', 'B'])
bands_earlyPts = np.array(['gp', 'rp', 'ip'])
SN20_sub = SN2020oi_sub[SN2020oi_sub['FLT'].isin(['gp','rp','ip', 'B', 'V'])]

SN20_sub.loc[SN20_sub['FLT']=='B','c'] = cols_hex[0]
SN20_sub.loc[SN20_sub['FLT']=='V','c'] = cols_hex[1]
SN20_sub.loc[SN20_sub['FLT']=='gp','c'] = cols_hex[2]
SN20_sub.loc[SN20_sub['FLT']=='rp','c'] = cols_hex[3]
SN20_sub.loc[SN20_sub['FLT']=='ip','c'] = cols_hex[4]
SN20_sub.loc[SN20_sub['FLT']=='B', 'MAG'] -= 0.4

#plt.figure(figsize=(10,7))
#plt.scatter(SN20_sub['MJD']-t0, SN20_sub['MAG'], color=SN20_sub['c'].values, s=35)
#plt.xlim((1.5, 16.5))
#plt.ylim((22, 12))

#legend_elements = [Line2D([0], [0], marker='>', color='w', label='B',
#                        markerfacecolor='w', mec='k',markersize=15, mew=2),#
#                   Line2D([0], [0], marker='D', color='w', label='V',
#                        markerfacecolor='w', mec='k',markersize=15, mew=2),
#                   Line2D([0], [0], marker='*', color='w', label='gp',
#                        markerfacecolor='w', mec='k',markersize=20, mew=2),
#                   Line2D([0], [0], marker='s', color='w', label='rp',
#                        markerfacecolor='w', mec='k',markersize=15, mew=2),
#                   Line2D([0], [0], marker='^', color='w', label='ip',]

#leg1 = plt.gca().legend(fontsize=18, handles=legend_elements, loc='lower right', handletextpad=0.0, borderaxespad=1.5,labelspacing=0.5, bbox_to_anchor=(-0.9,1.0), ncol=4,frameon=True, fancybox=True, edgecolor='k')


np.unique(SN2020oi['TELESCOPE'])
SN2020oi['Flag'] = 0

#GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_LateTime.csv")
GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_withSBO2_Trunc.csv")

f_g = interp1d(GP_results['MJD'], GP_results['gp_LC'], fill_value='extrapolate')
pred_g = f_g(SN2020oi.loc[(SN2020oi['FLT']=='g') & (SN2020oi['ULIM']==0), 'MJD'].values)
bad_g =  SN2020oi.loc[(SN2020oi['FLT']=='g') & (SN2020oi['ULIM']==0), 'MJD'].values[np.abs(pred_g - SN2020oi.loc[(SN2020oi['FLT']=='g'), 'MAG'].values)>0.5]
bad_g = bad_g[bad_g > (t0+10)]
for val in bad_g:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='g')& (SN2020oi['TELESCOPE']!='Synthetic')), 'Flag'] = 1

f_i = interp1d(GP_results['MJD'], GP_results['i_LC'], fill_value='extrapolate')
pred_i = f_i(SN2020oi.loc[(SN2020oi['FLT']=='i') & (SN2020oi['ULIM']==0), 'MJD'].values)
bad_i =  SN2020oi.loc[(SN2020oi['FLT']=='i') & (SN2020oi['ULIM']==0), 'MJD'].values[np.abs(pred_i - SN2020oi.loc[(SN2020oi['FLT']=='i'), 'MAG'].values)>0.5]
bad_i = bad_i[bad_i > (t0+10)]
for val in bad_i:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='i')& (SN2020oi['TELESCOPE']!='Synthetic')), 'Flag'] = 1

f_B = interp1d(GP_results['MJD'], GP_results['B_LC'], fill_value='extrapolate')
pred_B = f_B(SN2020oi.loc[(SN2020oi['FLT']=='B'), 'MJD'].values)
bad_B =  SN2020oi.loc[(SN2020oi['FLT']=='B'), 'MJD'].values[np.abs(pred_B - SN2020oi.loc[(SN2020oi['FLT']=='B'), 'MAG'].values)>0.5]
bad_B = bad_B[bad_B > (t0+10)]
for val in bad_B:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='B')& (SN2020oi['TELESCOPE']!='Synthetic')), 'Flag'] = 1

f_V = interp1d(GP_results['MJD'], GP_results['V_LC'], fill_value='extrapolate')
pred_V = f_V(SN2020oi.loc[(SN2020oi['FLT']=='V'), 'MJD'].values)
bad_V =  SN2020oi.loc[(SN2020oi['FLT']=='V'), 'MJD'].values[np.abs(pred_V - SN2020oi.loc[(SN2020oi['FLT']=='V'), 'MAG'].values)>0.5]
bad_V = bad_V[bad_V > (t0+10)]
for val in bad_V:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='V')& (SN2020oi['TELESCOPE']!='Synthetic')), 'Flag'] = 1

f_U = interp1d(GP_results['MJD'], GP_results['U_LC'], fill_value='extrapolate')
pred_U = f_U(SN2020oi.loc[(SN2020oi['FLT'].isin(['U','u'])), 'MJD'].values)
bad_U =  SN2020oi.loc[(SN2020oi['FLT'].isin(['U','u'])), 'MJD'].values[np.abs(pred_U - SN2020oi.loc[(SN2020oi['FLT'].isin(['U','u'])), 'MAG'].values)>0.5]
bad_U = bad_U[bad_U > (t0+10)]
for val in bad_U:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT'].isin(['U', 'u'])) & (SN2020oi['TELESCOPE']!='Synthetic')), 'Flag'] = 1

SN2020oi[SN2020oi['Flag']==1]
#SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/2020oi_finalPhotometry_0215.csv")
#SN2020oi_nonSwift = SN2020oi[(SN2020oi['FLT'] != 'UVW1')]
#SN2020oi_Swift = SN2020oi[(SN2020oi['FLT'] == 'UVW1')]
#SN2020oi = pd.concat([SN2020oi_nonSwift, SN2020oi_Swift],ignore_index=True)

#allData_late.loc[allData_late['band']=='u', 'band'] =  r'$U/u$'
#allData_late.loc[allData_late['band']=='g', 'band'] =  r'$g$'
#allData_late.loc[allData_late['band']=='r', 'band'] =  r'$r$'
#allData_late.loc[allData_late['band']=='i', 'band'] =  r'$i$'
#allData_late.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_LateTimeUpperLimits.csv",index=False)

SN2020oi.sort_values(by=['MJD'], inplace=True)
SN2020oi.loc[SN2020oi['ULIM'] != SN2020oi['ULIM'], 'ULIM'] = 0

SN2020oi.loc[SN2020oi['TELESCOPE'] == 'GPC1', 'TELESCOPE'] = 'Pan-STARRS1'
np.unique(SN2020oi['TELESCOPE'])
############## PLOT ZERO: A KILLER PLOT OF THE SHOCK-BREAKOUT ###############################
#U, B, V, UVW1, UVM2, g, r, i, z
#SN2020oi['ULIM'].values

#SN2020oi[SN2020oi['ULIM'] != 0]
#SN2020oi = SN2020oi[SN2020oi['FLT'] != 'UVM2']
#SN2020oi = SN2020oi[SN2020oi['FLT'] != 'UVW2']
SN2020oi = SN2020oi[SN2020oi['TELESCOPE'] != 'UVOT']
np.unique(SN2020oi['FLT'])

SN2020oi = SN2020oi[~(~SN2020oi['TELESCOPE'].isin(['Swift', 'Synthetic']) & (SN2020oi['MAGERR'] > 0.1))]


SN2020oi_g = SN2020oi[SN2020oi['FLT'].isin(['g', 'gp', 'g-ZTF'])]
SN2020oi_r = SN2020oi[SN2020oi['FLT'].isin(['r', 'rp', 'r-ZTF'])]
SN2020oi_i = SN2020oi[SN2020oi['FLT'].isin(['i', 'ip'])]



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
SN2020oi.loc[SN2020oi['band'] == 'B', 'band'] = r'$B$'
SN2020oi.loc[SN2020oi['band'] == 'U', 'band'] = r'$U/u$'
SN2020oi.loc[SN2020oi['band'] == 'V', 'band'] = r'$V$'
SN2020oi.loc[SN2020oi['band'] == 'u', 'band'] = r'$U/u$'
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

#removing the late time siding spring data - we just can't get it clean
#SN2020oi = SN2020oi[~(((SN2020oi['MJD'] - t0) > 15) & (SN2020oi['TELESCOPE'] == 'Siding Spring 1m'))]

telescopes = np.unique(SN2020oi['TELESCOPE'].dropna().values)
#telescopes = np.concatenate([telescopes, ['Unk']])
telescopes = np.concatenate([telescopes, ['Sinistro']])

cols = sns.color_palette("colorblind", 10)

cols = np.array([cols[3], cols[1], cols[8], cols[2], cols[9], cols[0], cols[4], cols[5], cols[7],cols[6]])

#bands = np.unique(SN2020oi['band'].values)


def app2abs(x):
    return x - 31.16 #distance modulus associated with 17.1 Mpc

def abs2app(x):
    return x + 31.16

def t0_to_mjd(x):
    return x + t0

def mjd_to_t0(x):
    return x - t0
#convert to absolute from apparent
#M = m - 31.16

SN20oi_U = SN2020oi[SN2020oi['FLT'].isin(['u', 'up'])]
SN20oi_B = SN2020oi[SN2020oi['FLT'] == 'B']
SN20oi_V = SN2020oi[SN2020oi['FLT'] == 'V']
SN20oi_g = SN2020oi[SN2020oi['FLT'].isin(['g', 'gp'])]
SN20oi_r = SN2020oi[SN2020oi['FLT'].isin(['r', 'rp'])]
SN20oi_i = SN2020oi[SN2020oi['FLT'] == 'i']
SN20oi_z = SN2020oi[SN2020oi['FLT'] == 'z']
#'Direct/2Kx2K':'X', 'Direct/4Kx4K':'p',

def make_plot():
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
    allData_late = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_LateTimeUpperLimits.csv")

    #    GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/GPResults_2020oi_UBVgriz_goldStandard_0216.csv")
    #GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_LateTime.csv")
    GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_withSBO2_Trunc.csv")
    GP_results_full = GP_results.copy()
    GP_results = GP_results[GP_results['MJD'] < (t0+62)]
    #set some stylings
    sns.set(font_scale=2.5)
    sns.set_style("white")
    sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 40})
    sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})

    #fig, ax1 = plt.subplots(figsize=(20,14))
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[0.5, 3],"hspace":0.3,"wspace":0.1}, figsize=(20,12))
    ax2 = axs[1]
    ax3 = axs[0]
#    ax6 = axs[2]
#    ax3 = axs[1][0]
#    ax4 = axs[0][0]
#    ax5 = axs[0][2]
#    ax6 = axs[1][2]

    #ax2.get_shared_y_axes().join(ax2, ax3)
#    ax1.get_shared_x_axes().join(ax1, ax2)
#    ax2.get_shared_y_axes().join(ax2, ax6)

#    ax4.axis('off')#
#    ax5.axis('off')

    fig.tight_layout()
#    ax1.invert_yaxis()
    ax2.invert_yaxis()
    telescope_dict = {'Nickel':'o', 'P48':".", 'Pan-STARRS1':'^', 'LCO':'s', 'Swift':'*', 'Swope':'D', 'Thacher':'>', 'ZTF':'h', 'Sinistro':'X', 'Synthetic':'d'}
    #rearrange
    stds = 2.0*np.array([np.nanmean(GP_results['z_std']), np.nanmean(GP_results['i_std']), np.nanmean(GP_results['rp_std']), np.nanmean(GP_results['gp_std']), np.nanmean(GP_results['V_std']), np.nanmean(GP_results['B_std']), np.nanmean(GP_results['U_std'])])
    bands = [r'$w2$', r'$m2$', r'$w1$', r'$U/u$', r'$B$', r'$g$', r'$V$', r'$r$', r'$i$',r'$z$']
    bands = bands[::-1]
    shifts = np.array([-1.8,  -0.5,  1.0,  2.5,  4.,  5.5,  6.5,  8.,  9.5,  11.])

    for i in np.arange(len(bands)):
        temp = SN2020oi[SN2020oi['band'] == bands[i]]
        #create a linear extrapolation of the photometry at late times
        if bands[i] not in [r'$w2$', r'$m2$', r'$w1$']:
            if bands[i] == r'$z$':
                startT = 30
                endT = 40
            else:
                startT = 45
                endT = 60
            interp_df = temp[(temp['MJD'] - t0) > startT]
            interp_df = interp_df[(interp_df['MJD'] - t0) < endT]
            interp_df.loc[interp_df['ULIM'] != interp_df['ULIM'], 'ULIM'] = 0
            interp_df = interp_df[interp_df['ULIM'] == 0]
            #if band == 'z':
            #    interp_df = interp_df[interp_df['TELESCOPE']== 'Thacher']
            #else:
            #    interp_df = interp_df[interp_df['TELESCOPE'] == 'Swope']
#            plt.plot(interp_df['MJD']-t0,interp_df['MAG']+shifts[i], 'o')
            #z = np.polyfit(interp_df['MJD']-t0,interp_df['MAG']+shifts[i], 1)
            #tail = np.poly1d(z)
            #xnew = np.linspace(300, 400, num=50)
            #ax6.fill_between(xnew, tail(xnew)-stds[i], tail(xnew)+stds[i],lw=3, alpha=0.5, color=cols[i])

        for j in np.arange(len(telescopes)):
            temp = SN2020oi[SN2020oi['band'] == bands[i]]
            goodtemp = temp[temp['Flag']==0]
            badtemp = temp[temp['Flag']==1]
            temp = goodtemp.copy()
            temp6 = allData_late[allData_late['band'] == bands[i]]

            if telescopes[j] == 'Unk':
                temp = temp[temp['TELESCOPE'] != temp['TELESCOPE']]
                temp6 = temp6[temp6['TELESCOPE'] != temp6['TELESCOPE']]
                badtemp = temp[temp['Flag']==1000] #dummy just to zero out this frame
            else:
                temp = temp[temp['TELESCOPE'] == telescopes[j]]
                temp6 = temp6[temp6['TELESCOPE'] == telescopes[j]]
                badtemp = badtemp[badtemp['TELESCOPE'] == telescopes[j]]
                badtemp.loc[badtemp['ULIM'] == 1, 'MAGERR'] = 0.5
            temp.loc[temp['ULIM'] == 1, 'MAGERR'] = 0.5
            if len(temp) > 0:
                if telescope_dict[telescopes[j]] in np.array(['*', '1', '+']):
                    ms = 22
                else:
                    ms = 12
                if bands[i] in [r'$w2$', r'$m2$', r'$w1$']:
                    temp_ulim = temp[temp['ULIM'] == 1]
                    #uplims = temp_ulim['ULIM']
                    #ax1.errorbar(temp_ulim['MJD']-t0, temp_ulim['MAG'],yerr=temp_ulim['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i], alpha=0.2)
                    #temp_reg = temp[temp['ULIM'] == 0]
                    #uplims = temp_reg['ULIM']
                    #ax1.errorbar(temp_reg['MJD']-t0, temp_reg['MAG'],yerr=temp_reg['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                else:
                    uplims = temp['ULIM']
                    if telescopes[j] == 'LCO':
                        zorder=1000
                        mew=2
                    else:
                        zorder = 1
                        mew=1
                    ax2.errorbar(temp['MJD']-t0, temp['MAG']+shifts[i],yerr=temp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=mew, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i], zorder=zorder)
                    ax3.errorbar(temp['MJD']-t0, temp['MAG']+shifts[i],yerr=temp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=mew, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i], zorder=zorder)
            if len(badtemp)>0:
                # (bands[i] in ['$i$', '$B$', '$V$', '$U/u$']) &
                #print(telescope_dict[telescopes[j]])
                #badtemp = temp[temp['Flag']==1]
                uplims = badtemp['ULIM']
                ax2.errorbar(badtemp['MJD']-t0, badtemp['MAG']+shifts[i],yerr=badtemp['MAGERR'],  lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='tab:red', ms=ms, mew=3, capsize=5.,capthick=2., alpha=0.5, ecolor=cols[i],mfc=cols[i])
                ax3.errorbar(badtemp['MJD']-t0, badtemp['MAG']+shifts[i],yerr=badtemp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='tab:red', ms=ms, mew=3, capsize=5.,capthick=2.,  alpha=0.5,ecolor=cols[i],mfc=cols[i]) #cols[i]
            #if (telescopes[j] == 'Sinistro'):
            #    ax6.errorbar(temp6['MJD']-t0, temp6['mag']+shifts[i], np.ones(len(temp6))*0.5, lolims=[int(x) for x in np.ones(len(temp6))],  fmt=telescope_dict['LCO'], ms=12, mew=2,  mec='k', capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
    #ax6.invert_yaxis()

    #replot the points in question


    legend_elements = [Line2D([0], [0], marker='>', color='w', label='Thacher',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
                       Line2D([0], [0], marker='D', color='w', label='Swope',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
                       Line2D([0], [0], marker='*', color='w', label='Swift UVOT',
                            markerfacecolor='w', mec='k',markersize=20, mew=2),
                       Line2D([0], [0], marker='s', color='w', label='LCO',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
                       Line2D([0], [0], marker='^', color='w', label='Pan-STARRS 1',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
#                       Line2D([0], [0], marker='.', color='w', label='ZTF',
#                            markerfacecolor='w', mec='k',markersize=15, mew=2),
                       Line2D([0], [0], marker='o', color='w', label='Nickel',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
                       #Line2D([0], [0], marker='X', color='w', label='Direct/2Kx2K',
                    #        markerfacecolor='w', mec='k',markersize=15, mew=2),
                      # Line2D([0], [0], marker='p', color='w', label='Direct/4Kx4K',
                    #        markerfacecolor='w', mec='k',markersize=15, mew=2),
                       Line2D([0], [0], marker='h', color='w', label='ZTF',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
                       Line2D([0], [0], marker='d', color='w', label='Synthetic',
                            markerfacecolor='w', mec='k',markersize=15, mew=2),
#                       Line2D([0], [0], marker='X', color='w', label='Sinistro',
#                            markerfacecolor='w', mec='k',markersize=15, mew=2),
    ]

    legend_elements2 = []
    legend_elements2_swift = []
    for i in np.arange(len(bands)):
        if shifts[i] > 0.:
            tempText = " + " + str(np.abs(shifts[i]))
        else:
            tempText = " - " + str(np.abs(shifts[i]))
        if bands[i] in [r'$w2$', r'$m2$', r'$w1$']:
            legend_elements2_swift.append(Line2D([0], [0], marker='o', color='w', mec='k', mew=1, label=bands[i], markerfacecolor=cols[i], markersize=15))

        else:
            legend_elements2.append(Line2D([0], [0], marker='o', color='w', mec='k', mew=1, label=bands[i] + tempText, markerfacecolor=cols[i], markersize=15))
    ax2.axvspan(58855.6-t0, 58857.4-t0, alpha=0.3, color='gray', label='Shock-breakout')

    specTimes = np.array([58857.3, 58864.6, 58868.6, 58870.5, 58872.4, 58875. , 58876.7,
       58879.4, 58880.3, 58880.4, 58892. , 58895. , 58995.6])
    #specTimes = np.array([58892.5, 58996.10902778, 58865.07993056, 58872.93881944,58879.92064815, 58877.20163194, 58869.13825231, 58871.01646991,
    #58880.93724537, 58857.8037037 ])
    for i in np.arange(len(specTimes)):
        ax2.axvline(x=specTimes[i] - t0, c='#A72608', lw=3.5, ymin=0, ymax=0.05, )

    ax2.set_xlim((-5, 64))
    ax2.set_ylim((27.5, 11.5))

    #absMag = ax1.secondary_yaxis('right', functions=(app2abs, abs2app))
    absMag2 = ax2.secondary_yaxis('right', functions=(app2abs, abs2app))
    absMag2.set_ylabel('Absolute Magnitude + Offset')
    absMag3 = ax3.secondary_yaxis('right', functions=(app2abs, abs2app))
    #absMag3.set_ylabel('Absolute Magnitude + Offset')

    #ax1.tick_params(which='major', axis='y', direction='in', length=16, width=2)
    #ax1.tick_params(which='major', axis='x', direction='in', length=16, width=2)
    #ax1.tick_params(which='minor', axis='y', direction='in', length=8, width=2)
    #ax1.tick_params(which='minor', axis='x', direction='in', length=8, width=2)
    #ax1.minorticks_on()

    ax3.tick_params(which='major', axis='y', direction='in', length=16, width=2)
    ax3.tick_params(which='major', axis='x', direction='in', length=16, width=2)
    ax3.tick_params(which='minor', axis='y', direction='in', length=8, width=2)
    ax3.tick_params(which='minor', axis='x', direction='in', length=8, width=2)
    ax3.minorticks_on()

    ax2.tick_params(which='major', axis='y', direction='in', length=16, width=2)
    ax2.tick_params(which='major', axis='x', direction='in', length=16, width=2)
    ax2.tick_params(which='minor', axis='y', direction='in',length=8, width=2)
    ax2.tick_params(which='minor', axis='x', direction='in', length=8, width=2)
    #ax6.tick_params(which='major', axis='x', direction='in', length=8, width=2)
    #ax6.tick_params(which='minor', axis='x', direction='in', length=8, width=2)

    #ax6.minorticks_on()
    ax2.minorticks_on()

    #absMag.tick_params(which='major', axis='y', direction='in', right=True, length=16, width=2)
    #absMag2.tick_params(which='major', axis='y', direction='in', right=True, length=16, width=2)
    #absMag3.tick_params(which='major', axis='y', direction='in', right=True, length=16, width=2)

#    ax2.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    #absMag.tick_params(which='minor', axis='y', direction='in', right=True, length=8, width=2)
    absMag2.tick_params(which='minor', axis='y', direction='in', right=True, length=8, width=2)
    #absMag3.tick_params(which='minor', axis='y', direction='in', right=True, length=8, width=2)

    #absMag.minorticks_on()
    absMag2.minorticks_on()
    #absMag3.minorticks_on()

    #absMJD = ax1.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    absMJD2 = ax2.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    absMJD3 = ax3.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    #absMJD6 = ax6.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    #absMJD3.set_xlabel('Time (MJD)')
    #absMJD.set_xlabel('Time (MJD)')

    #absMJD.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    #absMJD.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    #absMJD.minorticks_on()

    absMJD2.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    absMJD2.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMJD2.minorticks_on()

    absMJD3.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    absMJD3.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMJD3.minorticks_on()

    #absMJD6.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    #absMJD6.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    #absMJD6.minorticks_on()

    absMJD2.xaxis.set_ticklabels([])
    #absMJD2.yaxis.set_visible
    #ax6.yaxis.set_ticklabels([])
    absMag3.yaxis.set_ticklabels([])
    absMag3.yaxis.set_visible(False)
    #ax2.axes.get_yaxis().set_ticks([])

    # Create the figure
    ax2.set_xlabel("Time From Explosion (days)")
    ax3.set_ylabel("Apparent Magnitude + Offset")
    leg1 = plt.gca().legend(fontsize=18, handles=legend_elements, loc='lower right', handletextpad=0.0, borderaxespad=1.5,labelspacing=0.5, bbox_to_anchor=(0.85,1.0), ncol=4,frameon=True, fancybox=True, edgecolor='k')
    #leg2 = plt.gca().legend(fontsize=26, handles=legend_elements2, loc=(0.86, 0.133), borderaxespad=2.5, labelspacing=3.,frameon=False, handletextpad=0.0)
    #leg3 = ax1.legend(fontsize=26, handles=legend_elements2_swift, loc=(0.86, 0.4), borderaxespad=2.5, labelspacing=2.,frameon=False, handletextpad=0.0)
    #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # Manually add the first legend back
    plt.gca().add_artist(leg1)

    ax2.text(-2, 21.0, r"$Shock \ Breakout$", color='grey', rotation=90, style='italic')

    ax2.text(54, 13.8, r"$z$-1.8", color=cols[0],transform=ax2.transData)
    ax2.text(54, 15.8, r"$i$-0.5", color=cols[1],transform=ax2.transData)
    ax2.text(54, 17.3, r"$r$+1.0", color=cols[2],transform=ax2.transData)
    ax2.text(54.5, 19.1, r"$V$+2.5", color=cols[3],transform=ax2.transData)
    ax2.text(54.5, 20.6, r"$g$+4.0", color=cols[4],transform=ax2.transData)
    ax2.text(54, 22.7, r"$B$+5.5", color=cols[5],transform=ax2.transData)
    ax2.text(55, 24.7, r"$u$+6.5", color=cols[6],transform=ax2.transData)

    #-1.5,  -0.5,  1.0,  2.5,  4.5,  5.5,  6.5,  8.,  9.5,  11.
    ax3.fill_between(GP_results['MJD']-t0, GP_results['z_LC'] - GP_results['z_std']+shifts[0], GP_results['z_LC'] + GP_results['z_std']+shifts[0],lw=3, alpha=0.5, color=cols[0])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['i_LC'] - GP_results['i_std']+shifts[1], GP_results['i_LC'] + GP_results['i_std']+shifts[1],lw=3, alpha=0.5, color=cols[1])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['rp_LC'] - GP_results['rp_std']+shifts[2], GP_results['rp_LC'] + GP_results['rp_std']+shifts[2],lw=3, alpha=0.5, color=cols[2])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['V_LC'] - GP_results['V_std']+shifts[3], GP_results['V_LC'] + GP_results['V_std']+shifts[3],lw=3, alpha=0.5, color=cols[3])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['gp_LC'] - GP_results['gp_std']+shifts[4], GP_results['gp_LC'] + GP_results['gp_std']+shifts[4],lw=3, alpha=0.5, color=cols[4])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['B_LC'] - GP_results['B_std']+shifts[5], GP_results['B_LC'] + GP_results['B_std']+shifts[5],lw=3, alpha=0.5, color=cols[5])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['U_LC'] - GP_results['U_std']+shifts[6], GP_results['U_LC'] + GP_results['U_std']+shifts[6],lw=3, alpha=0.5, color=cols[6])


    ax2.fill_between(GP_results['MJD']-t0, GP_results['z_LC'] - GP_results['z_std']+shifts[0], GP_results['z_LC'] + GP_results['z_std']+shifts[0],lw=3, alpha=0.5, color=cols[0])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['i_LC'] - GP_results['i_std']+shifts[1], GP_results['i_LC'] + GP_results['i_std']+shifts[1],lw=3, alpha=0.5, color=cols[1])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['rp_LC'] - GP_results['rp_std']+shifts[2], GP_results['rp_LC'] + GP_results['rp_std']+shifts[2],lw=3, alpha=0.5, color=cols[2])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['V_LC'] - GP_results['V_std']+shifts[3], GP_results['V_LC'] + GP_results['V_std']+shifts[3],lw=3, alpha=0.5, color=cols[3])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['gp_LC'] - GP_results['gp_std']+shifts[4], GP_results['gp_LC'] + GP_results['gp_std']+shifts[4],lw=3, alpha=0.5, color=cols[4])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['B_LC'] - GP_results['B_std']+shifts[5], GP_results['B_LC'] + GP_results['B_std']+shifts[5],lw=3, alpha=0.5, color=cols[5])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['U_LC'] - GP_results['U_std']+shifts[6], GP_results['U_LC'] + GP_results['U_std']+shifts[6],lw=3, alpha=0.5, color=cols[6])

    #plot the late-time photometry upper limits
    #allData_late = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_LateTimeUpperLimits.csv")
    #col_dict = {'u':cols[3], 'g':cols[6], 'r':cols[7], 'i':cols[8]}

    #for band in ['u', 'g', 'r', 'i']:
    #    temp = allData_late[allData_late['band'] == band]
    #    ax6.errorbar(temp['MJD']-t0, temp['mag'], np.ones(len(temp))*0.5, uplims=[int(x) for x in np.ones(len(temp))], fmt='o', c=col_dict[band])

    #58868-t0
    #58923-t0
    #59005-t0
    #59266-t0
    #when were the HST observations?
    #ax2.axvline(58868-t0, c='k', lw=3, ls=':')
    #ax2.axvline(58923-t0, c='k', lw=3, ls=':')
    #ax2.axvline(59005-t0, c='k', lw=3, ls=':')
    #ax2.axvline(59266-t0, c='k', lw=3, ls=':')

    ax3.set_ylim((24.5, 13.2))

    #ax1.text(57, 18.2, r"$w1$", color=cols[7],transform=ax1.transData)
    #ax1.text(57, 18.7, r"$m2$", color=cols[8],transform=ax1.transData)
    #ax1.text(57, 19.2, r"$w2$", color=cols[9],transform=ax1.transData)

    ax3.set_xlim((1.8, 3.4))

    #absMJD6.set_xticks([58856, 58856])
    absMJD3.set_xticklabels(["", 58857])

    #ax6.tick_params(
    #axis='y',          # changes apply to the y-axis
    #which='both',      # both major and minor ticks are affected
    #left=False,      # ticks along the bottom edge are off
    #labelleft=False)

    #absMJD6.set_xticks([59240, 59250])
    #absMJD6.set_xticklabels([59240, ""])

    #uncertainty in the distance value
    #ax6.vlines(383, 13, 13-0.2173,lw=2, color='k')
    #ax6.vlines(383, 13, 13+0.2415, lw=2, color='k')

    #caps
    #ax6.hlines(13-0.2173, 381,385, lw=2, color='k')
    #ax6.hlines(13+0.2415, 381,385, lw=2, color='k')
    #ax6.text(380, 13.8, r"$\delta \mu$", c='k', fontsize=24)

    #uncertainty in the extinction estimate
    #ax6.vlines(390, 13, 13+0.17/2., lw=2, color='#7377AB')
    #ax6.vlines(390, 13, 13-0.17/2., lw=2, color='#7377AB')
    #caps
    #ax6.hlines(13-0.17/2., 388, 392, lw=2, color='#7377AB')
    #ax6.hlines(13+0.17/2., 388, 392, lw=2, color='#7377AB')
    #ax6.text(387., 13.8, r"$\delta E_{BV}$", c='#7377AB', fontsize=24)

    #uncertainty in the time of explosion
    ax3.hlines(24, 2.1, 2.9, color='k',  lw=2)
    #with caps
    ax3.vlines(2.1, 23.8,24.2, color='k', lw=2)
    ax3.vlines(2.9, 23.8,24.2, color='k', lw=2)
    ax3.text(2.3, 23.8, r"$\delta t_0$", c='k', fontsize=24)

#    ax6.set_xlim((380, 400))  # most of the data

#    ax2.spines['right'].set_visible(False)
    #ax6.spines['left'].set_visible(False)

    #d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    #kwargs = dict(marker=[(1, d), (-1, -d)], markersize=22,
        #          linestyle="none", color='k', mec='k', mew=2, clip_on=False)
    #ax2.plot([1, 1], [0, 1], transform=ax2.transAxes, **kwargs)
    #ax6.plot([0, 0], [0, 1], transform=ax6.transAxes, **kwargs)

    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/2020oi_FullPhotometry_Proposal.png",dpi=300, bbox_inches='tight')

make_plot()
