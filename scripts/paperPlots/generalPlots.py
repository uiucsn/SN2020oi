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

t0 = 58854.2

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
SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry.csv")

np.unique(SN2020oi['TELESCOPE'])
SN2020oi['Flag'] = 0

#GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_LateTime.csv")
GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_withSBO2_Trunc.csv")

f_i = interp1d(GP_results['MJD'], GP_results['i_LC'], fill_value='extrapolate')
pred_i = f_i(SN2020oi.loc[(SN2020oi['FLT']=='i') & (SN2020oi['ULIM']==0), 'MJD'].values)
bad_i =  SN2020oi.loc[(SN2020oi['FLT']=='i') & (SN2020oi['ULIM']==0), 'MJD'].values[np.abs(pred_i - SN2020oi.loc[(SN2020oi['FLT']=='i'), 'MAG'].values)>0.5]
bad_i = bad_i[bad_i > (t0+32)]
for val in bad_i:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='i')), 'Flag'] = 1

f_B = interp1d(GP_results['MJD'], GP_results['B_LC'], fill_value='extrapolate')
pred_B = f_B(SN2020oi.loc[(SN2020oi['FLT']=='B'), 'MJD'].values)
bad_B =  SN2020oi.loc[(SN2020oi['FLT']=='B'), 'MJD'].values[np.abs(pred_B - SN2020oi.loc[(SN2020oi['FLT']=='B'), 'MAG'].values)>0.5]
bad_B = bad_B[bad_B > (t0+32)]
for val in bad_B:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='B')), 'Flag'] = 1

f_V = interp1d(GP_results['MJD'], GP_results['V_LC'], fill_value='extrapolate')
pred_V = f_V(SN2020oi.loc[(SN2020oi['FLT']=='V'), 'MJD'].values)
bad_V =  SN2020oi.loc[(SN2020oi['FLT']=='V'), 'MJD'].values[np.abs(pred_V - SN2020oi.loc[(SN2020oi['FLT']=='V'), 'MAG'].values)>0.5]
bad_V = bad_V[bad_V > (t0+32)]
for val in bad_V:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT']=='V')), 'Flag'] = 1

f_U = interp1d(GP_results['MJD'], GP_results['U_LC'], fill_value='extrapolate')
pred_U = f_U(SN2020oi.loc[(SN2020oi['FLT'].isin(['U','u'])), 'MJD'].values)
bad_U =  SN2020oi.loc[(SN2020oi['FLT'].isin(['U','u'])), 'MJD'].values[np.abs(pred_U - SN2020oi.loc[(SN2020oi['FLT'].isin(['U','u'])), 'MAG'].values)>0.5]
bad_U = bad_U[bad_U > (t0+32)]
for val in bad_U:
    SN2020oi.loc[((SN2020oi['MJD']==val) & (SN2020oi['FLT'].isin(['U', 'u']))), 'Flag'] = 1

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

SN2020oi = SN2020oi[~(~SN2020oi['TELESCOPE'].isin(['Swift']) & (SN2020oi['MAGERR'] > 0.1))]


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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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
    fig, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [1, 3], 'width_ratios':[0.5, 3, 0.5],"hspace":0.3,"wspace":0.1}, figsize=(20,20))
    ax1 = axs[0][1]
    ax2 = axs[1][1]
    ax3 = axs[1][0]
    ax4 = axs[0][0]
    ax5 = axs[0][2]
    ax6 = axs[1][2]

    #ax2.get_shared_y_axes().join(ax2, ax3)
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax2.get_shared_y_axes().join(ax2, ax6)

    ax4.axis('off')
    ax5.axis('off')

    fig.tight_layout()
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    telescope_dict = {'Nickel':'o', 'P48':".", 'Pan-STARRS1':'^', 'LCO':'s', 'Swift':'*', 'Swope':'D', 'Thacher':'>', 'ZTF':'h', 'Sinistro':'X'}
    #rearrange
    stds = 2.0*np.array([np.nanmean(GP_results['z_std']), np.nanmean(GP_results['i_std']), np.nanmean(GP_results['rp_std']), np.nanmean(GP_results['gp_std']), np.nanmean(GP_results['V_std']), np.nanmean(GP_results['B_std']), np.nanmean(GP_results['U_std'])])
    bands = [r'$w2$', r'$m2$', r'$w1$', r'$U/u$', r'$B$', r'$V$', r'$g$',r'$r$', r'$i$',r'$z$']
    bands = bands[::-1]
    shifts = np.array([-1.8,  -0.5,  1.0,  2.5,  4.5,  5.5,  6.5,  8.,  9.5,  11.])

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
            plt.plot(interp_df['MJD']-t0,interp_df['MAG']+shifts[i], 'o')
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
                    uplims = temp_ulim['ULIM']
                    ax1.errorbar(temp_ulim['MJD']-t0, temp_ulim['MAG'],yerr=temp_ulim['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i], alpha=0.2)
                    temp_reg = temp[temp['ULIM'] == 0]
                    uplims = temp_reg['ULIM']
                    ax1.errorbar(temp_reg['MJD']-t0, temp_reg['MAG'],yerr=temp_reg['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                else:
                    uplims = temp['ULIM']
                    ax2.errorbar(temp['MJD']-t0, temp['MAG']+shifts[i],yerr=temp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
                    ax3.errorbar(temp['MJD']-t0, temp['MAG']+shifts[i],yerr=temp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='k', ms=ms, mew=1, capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
            if len(badtemp)>0:
                # (bands[i] in ['$i$', '$B$', '$V$', '$U/u$']) &
                #print(telescope_dict[telescopes[j]])
                #badtemp = temp[temp['Flag']==1]
                uplims = badtemp['ULIM']
                ax2.errorbar(badtemp['MJD']-t0, badtemp['MAG']+shifts[i],yerr=badtemp['MAGERR'],  lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='tab:red', ms=ms, mew=3, capsize=5.,capthick=2., alpha=0.5, ecolor=cols[i],mfc=cols[i])
                ax3.errorbar(badtemp['MJD']-t0, badtemp['MAG']+shifts[i],yerr=badtemp['MAGERR'], lolims=uplims, fmt=telescope_dict[telescopes[j]], label=bands[i], mec='tab:red', ms=ms, mew=3, capsize=5.,capthick=2.,  alpha=0.5,ecolor=cols[i],mfc=cols[i]) #cols[i]
            if (telescopes[j] == 'Sinistro'):
                ax6.errorbar(temp6['MJD']-t0, temp6['mag']+shifts[i], np.ones(len(temp6))*0.5, lolims=[int(x) for x in np.ones(len(temp6))],  fmt=telescope_dict['LCO'], ms=12, mew=1,  mec='k', capsize=5.,capthick=2., ecolor=cols[i],mfc=cols[i])
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

    specTimes = np.array([58892.5, 58996.10902778, 58865.07993056, 58872.93881944,58879.92064815, 58877.20163194, 58869.13825231, 58871.01646991,
    58880.93724537, 58857.8037037 ])
    for i in np.arange(len(specTimes)):
        ax2.axvline(x=specTimes[i] - t0, c='#DB5461', ls='--', lw=0.75)

    ax2.set_xlim((-5, 64))
    ax2.set_ylim((27.5, 11.5))

    absMag = ax1.secondary_yaxis('right', functions=(app2abs, abs2app))
    absMag2 = ax6.secondary_yaxis('right', functions=(app2abs, abs2app))
    absMag2.set_ylabel('Absolute Magnitude + Offset')
    absMag3 = ax3.secondary_yaxis('right', functions=(app2abs, abs2app))
    #absMag3.set_ylabel('Absolute Magnitude + Offset')

    ax1.tick_params(which='major', axis='y', direction='in', length=16, width=2)
    ax1.tick_params(which='major', axis='x', direction='in', length=16, width=2)
    ax1.tick_params(which='minor', axis='y', direction='in', length=8, width=2)
    ax1.tick_params(which='minor', axis='x', direction='in', length=8, width=2)
    ax1.minorticks_on()

    ax3.tick_params(which='major', axis='y', direction='in', length=16, width=2)
    ax3.tick_params(which='major', axis='x', direction='in', length=16, width=2)
    ax3.tick_params(which='minor', axis='y', direction='in', length=8, width=2)
    ax3.tick_params(which='minor', axis='x', direction='in', length=8, width=2)
    ax3.minorticks_on()

    ax2.tick_params(which='major', axis='y', direction='in', length=16, width=2)
    ax2.tick_params(which='major', axis='x', direction='in', length=16, width=2)
    ax2.tick_params(which='minor', axis='y', direction='in',  length=8, width=2)
    ax2.tick_params(which='minor', axis='x', direction='in', length=8, width=2)
    ax6.tick_params(which='major', axis='x', direction='in', length=8, width=2)
    ax6.tick_params(which='minor', axis='x', direction='in', length=8, width=2)

    ax6.minorticks_on()
    ax2.minorticks_on()

    absMag.tick_params(which='major', axis='y', direction='in', right=True, length=16, width=2)
    #absMag2.tick_params(which='major', axis='y', direction='in', right=True, length=16, width=2)
    #absMag3.tick_params(which='major', axis='y', direction='in', right=True, length=16, width=2)

    #ax2.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMag.tick_params(which='minor', axis='y', direction='in', right=True, length=8, width=2)
    #absMag2.tick_params(which='minor', axis='y', direction='in', right=True, length=8, width=2)
    #absMag3.tick_params(which='minor', axis='y', direction='in', right=True, length=8, width=2)

    absMag.minorticks_on()
    #absMag2.minorticks_on()
    #absMag3.minorticks_on()

    absMJD = ax1.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    absMJD2 = ax2.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    absMJD3 = ax3.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    absMJD6 = ax6.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
    #absMJD3.set_xlabel('Time (MJD)')
    absMJD.set_xlabel('Time (MJD)')

    absMJD.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    absMJD.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMJD.minorticks_on()

    absMJD2.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    absMJD2.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMJD2.minorticks_on()

    absMJD3.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    absMJD3.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMJD3.minorticks_on()

    absMJD6.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
    absMJD6.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
    absMJD6.minorticks_on()

    absMJD2.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])
    absMag3.yaxis.set_ticklabels([])
    absMag3.yaxis.set_visible(False)
    #ax2.axes.get_yaxis().set_ticks([])

    # Create the figure
    ax2.set_xlabel("Time From Explosion (days)")
    ax3.set_ylabel("Apparent Magnitude + Offset")
    leg1 = plt.gca().legend(fontsize=18, handles=legend_elements, loc='lower right', handletextpad=0.0, borderaxespad=1.5,labelspacing=0.5, bbox_to_anchor=(-0.9,1.0), ncol=4,frameon=True, fancybox=True, edgecolor='k')
    #leg2 = plt.gca().legend(fontsize=26, handles=legend_elements2, loc=(0.86, 0.133), borderaxespad=2.5, labelspacing=3.,frameon=False, handletextpad=0.0)
    #leg3 = ax1.legend(fontsize=26, handles=legend_elements2_swift, loc=(0.86, 0.4), borderaxespad=2.5, labelspacing=2.,frameon=False, handletextpad=0.0)
    #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # Manually add the first legend back
    plt.gca().add_artist(leg1)

    ax2.text(-2, 21.0, r"$Shock \ Breakout$", color='grey', rotation=90, style='italic')

    ax2.text(54, 13.8, r"$z$-1.8", color=cols[0],transform=ax2.transData)
    ax2.text(54, 15.8, r"$i$-0.5", color=cols[1],transform=ax2.transData)
    ax2.text(54, 17.3, r"$r$+1.0", color=cols[2],transform=ax2.transData)
    ax2.text(54.5, 19.1, r"$g$+2.5", color=cols[3],transform=ax2.transData)
    ax2.text(54.5, 21.0, r"$V$+4.5", color=cols[4],transform=ax2.transData)
    ax2.text(54, 22.7, r"$B$+5.5", color=cols[5],transform=ax2.transData)
    ax2.text(55, 24.7, r"$u'/u$+6.5", color=cols[6],transform=ax2.transData)

    #-1.5,  -0.5,  1.0,  2.5,  4.5,  5.5,  6.5,  8.,  9.5,  11.
    #ax3.fill_between(GP_results['MJD']-t0, GP_results['z_LC'] - GP_results['z_std']+shifts[0], GP_results['z_LC'] + GP_results['z_std']+shifts[0],lw=3, alpha=0.5, color=cols[0])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['i_LC'] - GP_results['i_std']+shifts[1], GP_results['i_LC'] + GP_results['i_std']+shifts[1],lw=3, alpha=0.5, color=cols[1])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['rp_LC'] - GP_results['rp_std']+shifts[2], GP_results['rp_LC'] + GP_results['rp_std']+shifts[2],lw=3, alpha=0.5, color=cols[2])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['gp_LC'] - GP_results['gp_std']+shifts[3], GP_results['gp_LC'] + GP_results['gp_std']+shifts[3],lw=3, alpha=0.5, color=cols[3])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['V_LC'] - GP_results['V_std']+shifts[4], GP_results['V_LC'] + GP_results['V_std']+shifts[4],lw=3, alpha=0.5, color=cols[4])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['B_LC'] - GP_results['B_std']+shifts[5], GP_results['B_LC'] + GP_results['B_std']+shifts[5],lw=3, alpha=0.5, color=cols[5])
    ax3.fill_between(GP_results['MJD']-t0, GP_results['U_LC'] - GP_results['U_std']+shifts[6], GP_results['U_LC'] + GP_results['U_std']+shifts[6],lw=3, alpha=0.5, color=cols[6])


    ax2.fill_between(GP_results['MJD']-t0, GP_results['z_LC'] - GP_results['z_std']+shifts[0], GP_results['z_LC'] + GP_results['z_std']+shifts[0],lw=3, alpha=0.5, color=cols[0])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['i_LC'] - GP_results['i_std']+shifts[1], GP_results['i_LC'] + GP_results['i_std']+shifts[1],lw=3, alpha=0.5, color=cols[1])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['rp_LC'] - GP_results['rp_std']+shifts[2], GP_results['rp_LC'] + GP_results['rp_std']+shifts[2],lw=3, alpha=0.5, color=cols[2])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['gp_LC'] - GP_results['gp_std']+shifts[3], GP_results['gp_LC'] + GP_results['gp_std']+shifts[3],lw=3, alpha=0.5, color=cols[3])
    ax2.fill_between(GP_results['MJD']-t0, GP_results['V_LC'] - GP_results['V_std']+shifts[4], GP_results['V_LC'] + GP_results['V_std']+shifts[4],lw=3, alpha=0.5, color=cols[4])
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

    ax3.set_ylim((24.5, 14))

    ax1.text(57, 18.2, r"$w1$", color=cols[7],transform=ax1.transData)
    ax1.text(57, 18.7, r"$m2$", color=cols[8],transform=ax1.transData)
    ax1.text(57, 19.2, r"$w2$", color=cols[9],transform=ax1.transData)

    ax3.set_xlim((1.8, 3.4))

    #absMJD6.set_xticks([58856, 58856])
    absMJD3.set_xticklabels(["", 58857])

    ax6.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    labelleft=False)

    absMJD6.set_xticks([59240, 59250])
    absMJD6.set_xticklabels([59240, ""])

    #uncertainty in the distance value
    ax6.vlines(383, 13, 13-0.2173,lw=2, color='k')
    ax6.vlines(383, 13, 13+0.2415, lw=2, color='k')

    #caps
    ax6.hlines(13-0.2173, 381,385, lw=2, color='k')
    ax6.hlines(13+0.2415, 381,385, lw=2, color='k')
    ax6.text(380, 13.8, r"$\delta \mu$", c='k', fontsize=24)

    #uncertainty in the extinction estimate
    ax6.vlines(390, 13, 13+0.17/2., lw=2, color='#7377AB')
    ax6.vlines(390, 13, 13-0.17/2., lw=2, color='#7377AB')
    #caps
    ax6.hlines(13-0.17/2., 388, 392, lw=2, color='#7377AB')
    ax6.hlines(13+0.17/2., 388, 392, lw=2, color='#7377AB')
    ax6.text(387., 13.8, r"$\delta E_{BV}$", c='#7377AB', fontsize=24)

    #uncertainty in the time of explosion
    ax3.hlines(24, 2.1, 2.9, color='k',  lw=2)
    #with caps
    ax3.vlines(2.1, 23.8,24.2, color='k', lw=2)
    ax3.vlines(2.9, 23.8,24.2, color='k', lw=2)
    ax3.text(2.3, 23.8, r"$\delta t_0$", c='k', fontsize=24)

    ax6.set_xlim((380, 400))  # most of the data

    ax2.spines['right'].set_visible(False)
    ax6.spines['left'].set_visible(False)

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(1, d), (-1, -d)], markersize=22,
                  linestyle="none", color='k', mec='k', mew=2, clip_on=False)
    ax2.plot([1, 1], [0, 1], transform=ax2.transAxes, **kwargs)
    ax6.plot([0, 0], [0, 1], transform=ax6.transAxes, **kwargs)

    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/2020oi_FullPhotometry_forPaperSBO_Trunc.png",dpi=200, bbox_inches='tight')

make_plot()


#they estimate the absolute V-band peak magnitude of SN 2017ein as −17.47 ± 0.35 mag.
GP_results = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/interpolated_lcs/GPResults_2020oi_LateTime.csv")

#np.nanmin(GP_results['V_LC'].values - 31.16)
#GP_results[(GP_results['V_LC'].values -31.16) == -17.368159825013905] /pm 0.043812


############## PLOT ONE: COMPARING BOLOMETRIC LIGHTCURVE TO PREVIOUS OBSERVATIONS ###############################
peak = 58865.615999999995 #of the bolometric LC

sns.set_context("poster")

np.unique(SN2020oi['FLT'])

SN2020oi_g = SN2020oi[SN2020oi['FLT'].isin(['g', 'gp', 'g-ZTF'])]
SN2020oi_r = SN2020oi[SN2020oi['FLT'].isin(['r', 'rp', 'r-ZTF'])]
#SN2020oi_r = SN2020oi_r[SN2020oi_r['MAGERR'] < 0.05]
SN2020oi_i = SN2020oi[SN2020oi['FLT'].isin(['i', 'ip', 'i-ZTF'])]
#SN2020oi_i = SN2020oi_i[SN2020oi_i['MAGERR'] < 0.03]

SN2020oi_U = SN2020oi[SN2020oi['FLT'] == 'U']
SN2020oi_B  = SN2020oi[SN2020oi['FLT'] == 'B']

#plt.xlim((58856.35, 58857.5))
#plt.plot(SN2020oi_U['MJD'], SN2020oi_U['MAG'])
#plt.plot(SN2020oi_r['MJD'], SN2020oi_r['MAG'])
#plt.plot(SN2020oi_B['MJD'], SN2020oi_B['MAG'])
#plt.gca().invert_yaxis()

#comparison from https://academic.oup.com/mnras/article/457/1/328/989045
Lbol_Ics = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/reference_data/digitized_Ic_bol.txt")
Lbol_Ics_low = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/reference_data/digitized_Ic_bol_low.txt")
Lbol_Ics_high = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/reference_data/digitized_Ic_bol_high.txt")

path = '/Users/alexgagliano/Downloads/2020oi_spectra/'

spectra = [f for f in listdir(path) if isfile(join(path, f))]
dates = []
instr = []
for spectrum in spectra:
    with open(path+spectrum, 'r') as file:
        lines = np.array(file.readlines())
        for line in lines:
            if line.startswith("# OBS_DATE") :
                date = line[12:]
                date = date.strip("\n")
                date_time = date.split(" ")
                date = date_time[0] + "T" + date_time[1]
                date = date.split("+")[0]
                dates.append(date)
        instr.append(spectrum.split("-")[1])

date_dash = np.array([x[5:-9] for x in dates])
t = Time(dates)
np.unique(instr)
specTimes = t.jd  - 2400000
cols = {'Goodman':'#24A5C5', 'FLOYDS':'#7360A7', 'KAST':'#E49273'}
legend = {'FLOYDS':0, 'Goodman':0, 'KAST':0}
specTimes
#58857.8037037 - 58866.1


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(13, 10), dpi=300, facecolor='w', edgecolor='k')
fig.subplots_adjust(wspace=0, hspace=0)
#inset_ax = inset_axes(ax, height="32%",width="15%", loc='upper right',borderpad=0.5) # center, you can check the different codes in plt.legend?
#inset_ax.set_xlim((-9, -5))
#inset_ax.set_ylim((15, 17.5))
#inset_ax.set_xlabel("Phase (days)",fontsize=14)
#inset_ax.set_ylabel("Mag",fontsize=14)
#inset_ax.tick_params(labelsize=12)
#inset_ax.errorbar(SN2020oi_r['MJD']-peak, SN2020oi_r['MAG'], yerr=SN2020oi_r['MAGERR'], marker='o',  c='#A63D40', lw=2)
#inset_ax.errorbar(SN2020oi_i['MJD']-peak, SN2020oi_i['MAG'], yerr=SN2020oi_i['MAGERR'], marker='o',  c='tab:green', lw=2, label='i')
#inset_ax.errorbar(SN2020oi_U['MJD']-peak, SN2020oi_U['MAG'], yerr=SN2020oi_U['MAGERR'], marker='o',  c='#E08E45', lw=2, label='U')
#inset_ax.errorbar(SN2020oi_B['MJD']-peak, SN2020oi_B['MAG'], yerr=SN2020oi_B['MAGERR'], marker='o', c='#7F7CAF', lw=2, label='B')
#inset_ax.invert_yaxis()
ax.set_xlabel("Time From Explosion (days)")
ax.set_ylabel(r"$log$[$L_{bol}$] $log$[erg/s]")
ax.tick_params(axis="both", which="both", right=False, top=False)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

absMJD = ax.secondary_xaxis('top', functions=(t0_to_mjd, mjd_to_t0))
absMJD.set_xlabel('Time (MJD)')

#specTimes - t0
for i in np.arange(len(specTimes)):
    time = specTimes[i]
    spec_inst = instr[i]
    if time < 58920:
        if legend[spec_inst] == 0:
            ax.axvline(x=time-t0, c='gray', label=spec_inst, alpha=0.5)
            legend[spec_inst] = 1
        else:
            ax.axvline(x=time-t0, c='gray', alpha=0.5)
shift = 11

ax.axvspan(58856.2-t0, 58857.5-t0, alpha=0.4, color='red', label='Shock-breakout')
Llow = interp1d(Lbol_Ics_low['phase']+shift,Lbol_Ics_low['log_lbol'], bounds_error=False)
Lhigh = interp1d(Lbol_Ics_high['phase']+shift,Lbol_Ics_high['log_lbol'], bounds_error=False)
Lmid = interp1d(Lbol_Ics['phase']+shift,Lbol_Ics['log_lbol'], bounds_error=False, fill_value='extrapolate')
xnew = Lbol_Ics_low['phase'].values+shift
xnew_mid = np.linspace(0, shift)
#plt.plot(xnew_mid, Lmid(xnew_mid), ls='--')
ax.plot(Lbol_Ics['phase']+shift, Lbol_Ics['log_lbol'], label='Ic Template (Lyman+2016)', c='tab:blue')
ax.fill_between(xnew, Llow(xnew), Lhigh(xnew), where=Lhigh(xnew) >Llow(xnew), facecolor='tab:blue', alpha=0.3)

ax.plot(Lbol['MJD'].values-t0, Lbol['logL'].values, c='#7F7CAF')
ax.fill_between(Lbol['MJD'].values-t0, Lbol['logL'].values-Lbol['logLerr'].values, Lbol['logL'].values+Lbol['logLerr'].values, facecolor='#7F7CAF', alpha=0.6)

ax.set_xlim((0, 39.))
ax.set_ylim(ymin=41.4,ymax=43.)
obs = np.array(['SOAR 01/09', 'LCO 1/16', 'LCO 01/20', 'LCO 01/22',  'LCO 01/24', 'Keck 01/27', 'LCO 01/28', 'LCO 01/31',  'LCO 02/01', 'Lick 02/13', 'Keck 02/16'])
alignment_times = specTimes - t0
alignment_times.sort()
y_txt= 41.8
for i in np.arange(len(alignment_times)-1):
    if (i == 5) or (i==6):
        ax.text(alignment_times[i]-0.8, 41.46, obs[i],fontsize=18, rotation=90, color='grey')
    elif i == 8:
        ax.text(alignment_times[i]-0.8, 42.7, obs[i],fontsize=18, rotation=90, c='grey')
        continue
    #elif i==9:
    else:
        ax.text(alignment_times[i]+0.25, 41.46, obs[i],fontsize=18, rotation=90, color='grey')

ax.tick_params(which='major', axis='x', direction='in', top=True, length=16, width=2)
ax.tick_params(which='minor', axis='x', direction='in', top=True, length=8, width=2)
ax.tick_params(which='major', axis='y', direction='in', top=True, length=16, width=2)
ax.tick_params(which='minor', axis='y', direction='in', top=True, length=8, width=2)
ax.minorticks_on()

plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/2020oi_Lbol_wSpectra_comparisonIc_0215.pdf",dpi=300,bbox_inches='tight')

###### COMPARING SPECTRA
spec_0109 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra_mangled_018/SN2020oi-20200109-goodman.flm_mangled_ext_cor_total_0.18.flm", delimiter='   ')
spec_0116 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra_mangled_018/SN2020oi-20200116-floyds.flm_mangled_ext_cor_total_0.18.flm", delimiter='   ')
spec_0128 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra_mangled_018/SN2020oi-20200127-lris.flm_mangled_ext_cor_total_0.18.flm", delimiter='   ')

template_Ic_min10 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/template_Ics/minus10day_phase.csv", delimiter=', ')
template_Ic_0 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/template_Ics/0day_phase.csv", delimiter=', ')
template_Ic_10 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/template_Ics/10day_phase.csv", delimiter=', ')
spec_0109.columns.values

offset1 = 1
offset2 = 1.5
offset3 = 2

spec_0128['flux_mean'] = spec_0128['flux(erg/s cm2 A)'].rolling(5).mean()
spec_0116['flux_mean'] = spec_0116['flux(erg/s cm2 A)'].rolling(5).mean()
spec_0109['flux_mean'] = spec_0109['flux(erg/s cm2 A)'].rolling(5).mean()

plt.figure(figsize=(20,10))
plt.plot(template_Ic_0['Wavelength(A)'], template_Ic_0['Flux']+offset2, '--', c='tab:blue')
plt.plot(spec_0116['wavelength(A)'], spec_0116['flux_mean']/1.e-14+offset2, c='#993955', lw=2)

plt.plot(template_Ic_10['Wavelength(A)'], template_Ic_10['Flux']+1.55+offset1, '--', c='tab:blue')
plt.plot(spec_0128['wavelength(A)'], spec_0128['flux_mean']/1.e-14+offset1, c='#993955')

plt.ylabel("Relative Flux + Offset")
plt.text(4100,3.5, "0 d +/- 2 d",fontsize=26)
plt.text(4100, 2.2, "10 d +/- 2 d",fontsize=26)
plt.text(5500,3.7, "Template Ic Spectra",fontsize=26,color='tab:blue')
plt.text(5800, 3.5, "SN 2020oi",fontsize=26,color='#993955')
#plt.xlim((4000, 7000))
plt.ylim((1, 4))
plt.xlabel("Rest-Frame Wavelength (A)")
plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/2020oi_Spectra_Comparison_notBold.png",dpi=300)

#The Kast Double Spectrograph is used at the cassegrain focus of the Shane 3-m Telescope at Lick Observatory
#FLOYDS spectrograph at Las Cumbres Observatory
#Goodman spectrograph for the SOAR telescope at CTIO

#plt.text(18, 42.83, "Goodman", c='#24A5C5') #SOAR and date
#plt.text(18, 42.75, "FLOYDS", c='#7360A7') #LCO and date
#plt.text(18, 42.67, "KAST", c='#E49273') #Lick and date
#plt.legend()
#plt.savefig("/Users/alexgagliano/Desktop/2020oi_Lbol_wSpectra.pdf",dpi=300)

############## PLOT TWO: HST PRE-EXPLOSION IMAGES ###############################

from matplotlib.path import Path
verts = [
(0.0, -1.0), # middle, bottom
(0.0, -0.3), # middle, below center
(0.0, +1.0), # middle, top<br />
(0.0, +0.3), # middle, above center
(-1.0, 0.0), # left, middle
(-0.3, 0.0), # before center, middle
(+1.0, 0.0), # right, middle
(+0.3, 0.0), # before center, middle
]

codes = [Path.MOVETO,
Path.LINETO,
Path.MOVETO,
Path.LINETO,
Path.MOVETO,
Path.LINETO,
Path.MOVETO,
Path.LINETO,
]
path = Path(verts, codes)


ra_20oi = 185.728875
dec_20oi = +15.8236

#wfc3.f110w.ref_001.drz.fits
#wfc3.f438w.ref_001.drz.fits
#wfc3.f160w.ref_001.drz.fits
#wfc3.f555w.ref_001.drz.fits
#wfc3.f275w.ref_001.drz.fits
#wfc3.f625w.ref_001.drz.fits
#wfc3.f336w.ref_001.drz.fits
#wfc3.f814w.ref_001.drz.fits

hdu_814_1 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/wfc3.f814w.ref_001.drz.fits', 'readonly')
#hdu_814_2 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/idkv31n8q_flt.fits', 'readonly')
#hdu_814_3 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/idkv31n9q_flt.fits', 'readonly')

hdu_475_1 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/wfc3.f555w.ref_001.drz.fits', 'readonly')
#hdu_475_2 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/idkv31nbq_flt.fits', 'readonly')
#hdu_475_3 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/idkv31ncq_flt.fits', 'readonly')

hdu_160_1 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/wfc3.f160w.ref_001.drz.fits', 'readonly')
#hdu_160_2 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/idkv31niq_flt.fits', 'readonly')
#hdu_160_3 =fits.open('/Users/alexgagliano/Documents/Research/2020oi/data/HST_explosionImaging/idkv31njq_flt.fits', 'readonly')

#hdu_814 = fits.open(get_pkg_data_filename('/Users/alexgagliano/Desktop/HST_M100/M100/idkv31n7q_flt.fits'))[0]
#hdu_475 = fits.open(get_pkg_data_filename('/Users/alexgagliano/Desktop/HST_M100/M100/idkv31ncq_flt.fits'))[0]
#hdu_160 = fits.open(get_pkg_data_filename('/Users/alexgagliano/Desktop/HST_M100/M100/idkv31ngq_flt.fits'))[0]
hdu_475_1[0].data

img_475_1 = hdu_475_1[0].data
#img_475_2 = hdu_475_2[1].data
#img_475_3 = hdu_475_3[1].data

img_814_1 = hdu_814_1[0].data
#img_814_2 = hdu_814_2[1].data
#img_814_3 = hdu_814_3[1].data

img_160_1 = hdu_160_1[0].data
#img_160_2 = hdu_160_2[1].data
#img_160_3 = hdu_160_3[1].data

#img_475 = np.nanmedian([img_475_1, img_475_2, img_475_3], axis=0)
#img_814 = np.nanmedian([img_814_1, img_814_2, img_814_3], axis=0)
#img_160 = np.nanmedian([img_160_1, img_160_2, img_160_3], axis=0)
img_475 = np.nanmedian([img_475_1], axis=0)
img_814 = np.nanmedian([img_814_1], axis=0)
img_160 = np.nanmedian([img_160_1], axis=0)

hdu_814_1[0].header
w814 = wcs.WCS(hdu_814_1[0].header, hdu_814_1)
w475 = wcs.WCS(hdu_475_1[0].header, hdu_475_1)
w160 = wcs.WCS(hdu_160_1[0].header, hdu_160_1)

x, y = w814.wcs_world2pix(ra_20oi, dec_20oi,1, ra_dec_order=True)

w814.wcs_world2pix(ra_20oi, dec_20oi,1, ra_dec_order=True)
w475.wcs_world2pix(ra_20oi, dec_20oi,1, ra_dec_order=True)
w160.wcs_world2pix(ra_20oi, dec_20oi,1, ra_dec_order=True)

#start of FoR
#185.7356 - 185.735
#15.8205 - 15.819900000000008
#185.735 - 0.0005999999999914962

from astropy.nddata import Cutout2D
from reproject import reproject_interp
from astropy import units as u

array_160, footprint = reproject_interp((img_160, w160),w814, np.shape(img_814))
array_475, footprint = reproject_interp((img_475, w475),w814, np.shape(img_814))
stretch = SqrtStretch() + ZScaleInterval()

r = stretch(array_160,clip=False)
g = stretch(img_814,clip=False)
b = stretch(array_475,clip=False)


size = 700
x_gal, y_gal = w814.wcs_world2pix(185.7288750, 15.8223028,1, ra_dec_order=True)
cutout_r = Cutout2D(r, (x_gal,y_gal), (size, size), wcs=w814)
cutout_g = Cutout2D(g, (x_gal,y_gal), (size, size), wcs=w814)
cutout_b = Cutout2D(b, (x_gal,y_gal), (size, size), wcs=w814)

from scipy import interpolate

rgb_array = []
for data in [cutout_r.data, cutout_g.data, cutout_b.data]:
    backx = np.arange(0,data.shape[1])
    backy = np.arange(0, data.shape[0])
    backxx, backyy = np.meshgrid(backx, backy)
    #mask invalid values
    array = np.ma.masked_invalid(data)
    x1 = backxx[~array.mask]
    y1 = backyy[~array.mask]
    newarr = array[~array.mask]
    rgb_array.append(interpolate.griddata((x1, y1), newarr.ravel(), (backxx, backyy), method='linear'))

rgb_array = np.array(rgb_array)
#plt.imshow(cutout.data, origin='lower')

#uvis : 0.04''/px
px_scale = 0.04 #arcsec/px
width = 10 #arcsec
width_px = width/px_scale

extent = 0.001
ra_origin = 185.732
dec_origin = 15.8181
#ra_20oi = 185.728875
#dec_20oi = +15.8236
x_origin, y_origin = cutout_r.wcs.wcs_world2pix(ra_origin, dec_origin,1, ra_dec_order=True)
x_end1, y_end1 = cutout_r.wcs.wcs_world2pix(ra_origin-extent, dec_origin,1, ra_dec_order=True)
x_end2, y_end2 = cutout_r.wcs.wcs_world2pix(ra_origin, dec_origin-extent,1, ra_dec_order=True)

lo_val, up_val = np.nanpercentile(np.hstack((rgb_array[0].flatten(), rgb_array[1].flatten(), rgb_array[2].flatten())), (5, 99))  # Get the value of lower and upper 0.5% of all pixels
stretch_val = up_val - lo_val
rgb_default = make_lupton_rgb(rgb_array[0],rgb_array[1],rgb_array[2], minimum=lo_val, stretch=stretch_val, Q=1)

scalebar_x = 400
scalebar_y = 50
gal_name_x = 550
gal_name_y = y+170+20

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(111, projection=cutout_r.wcs)
x, y = cutout_r.wcs.wcs_world2pix(ra_20oi, dec_20oi,1, ra_dec_order=True)
plt.imshow(rgb_default)
plt.axis("off")
plt.plot(x,y,marker=path,markersize=80, mew=3, c='red')
plt.text(x-320, y+170, 'SN 2020oi', fontsize=70, color='w')
overlay = plt.gca().get_coords_overlay('fk5')
overlay.grid(color='white', ls='dotted')
plt.arrow(x_origin, y_origin, -(x_end1-x_origin), -(y_end1-y_origin), head_width=10, head_length=10, lw=3, fc='w', ec='w')
plt.arrow(x_origin, y_origin, -(x_end2-x_origin), -(y_end2-y_origin), head_width=10, head_length=10, lw=3, fc='w', ec='w')
plt.text(x_origin-(x_end1-x_origin)-27, y_origin-(y_end1-y_origin)-7, 'E',color='w',fontweight='bold', fontsize=30)
plt.text(x_origin-(x_end2-x_origin)-7, y_origin-(y_end2-y_origin)+15, 'N',color='w',fontweight='bold', fontsize=30)
plt.arrow(scalebar_x, scalebar_y, width_px, 0, head_width=0, head_length=0, lw=3, fc='w', ec='w')
plt.arrow(scalebar_x, scalebar_y, 0, 10, head_width=0, head_length=0, lw=3, fc='w', ec='w')
plt.arrow(scalebar_x, scalebar_y, 0, -10, head_width=0, head_length=0, lw=3, fc='w', ec='w')
plt.arrow(scalebar_x+width_px, scalebar_y, 0, 10, head_width=0, head_length=0, lw=3, fc='w', ec='w')
plt.arrow(scalebar_x+width_px, scalebar_y, 0, -10, head_width=0, head_length=0, lw=3, fc='w', ec='w')
plt.text(scalebar_x+105, scalebar_y+10, r"10$''$",fontsize=40, color='w')
plt.text(gal_name_x, gal_name_y, "M100",fontsize=50, color='w')
plt.text(gal_name_x-25, gal_name_y-35, r"$z=0.0052$",fontsize=40, color='w')
plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/SN2020oi_HST.png", bbox_inches='tight', dpi=300)


############## PLOT THREE: LC FITTING IN DIFFERENT BANDS TO ESTIMATE BRIGHTNESS IN FEBRUARY AND APRIL ###############################
sns.set_context("talk")

iPTF13bvn = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/template_Ics/iPTF13bvn_photometry.csv")
#SN2015bn = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/SN2007gr_photometry.csv")
SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/2020oi_allBands_0921.csv")

proposed_obs = 59246 #02-01-2021
peak+460

proposed_phase = proposed_obs - peak

#SNoi = SN2020oi[SN2020oi['MAGERR'] < .05]
#SNoi = SN2020oi
SNoi_prePeak = SN2020oi[SN2020oi['MJD'] <= peak]
SNoi_prePeak_lowErr = SNoi_prePeak[SNoi_prePeak['MAGERR'] < 1]
SNoi_postPeak = SN2020oi[SN2020oi['MJD'] > peak]
SNoi_postPeak_lowErr = SNoi_postPeak[SNoi_postPeak['MAGERR'] < 0.2]
SNoi = pd.concat([SNoi_prePeak_lowErr, SNoi_postPeak_lowErr],ignore_index=True)

SNoi_g = SNoi[SNoi['FLT'] == 'g']
SNoi_g.sort_values(by=['MJD'], inplace=True)
SNoi_g_model = SNoi_g.rolling(5).mean()
SNoi_g_resids = np.abs(SNoi_g_model['MAG'] - SNoi_g['MAG'])
#SNoi_g_highE = SNoi_g[SNoi_g_resids.values > 0.8]
SNoi_g_highE = SNoi_g[SNoi_g_resids.values > 0.3]
SNoi_errEpochs = SNoi_g_highE.loc[SNoi_g_highE['MJD'] > peak, 'MJD'].values
SNoi_g = SNoi_g[~SNoi_g['MJD'].isin(SNoi_errEpochs)]

#plt.hist(SNoi_g_resids)
#SNoi[SNoi['FLT'] == 'g']
SNoi_r = SNoi[SNoi['FLT'] == 'r']
SNoi_r.sort_values(by=['MJD'], inplace=True)
SNoi_r_model = SNoi_r.rolling(5).mean()
SNoi_r_resids = np.abs(SNoi_r_model['MAG'] - SNoi_r['MAG'])
SNoi_r_highE = SNoi_r[SNoi_r_resids.values > 0.4]
SNoi_errEpochs = SNoi_r_highE.loc[SNoi_r_highE['MJD'] > peak, 'MJD'].values
SNoi_r = SNoi_r[~SNoi_r['MJD'].isin(SNoi_errEpochs)]

SNoi_i = SNoi[SNoi['FLT'] == 'i']
SNoi_i.sort_values(by=['MJD'], inplace=True)
SNoi_i_model = SNoi_i.rolling(5).mean()
SNoi_i_resids = np.abs(SNoi_i_model['MAG'] - SNoi_i['MAG'])
SNoi_i_highE = SNoi_i[SNoi_i_resids.values > 0.3]
SNoi_errEpochs = SNoi_i_highE.loc[SNoi_i_highE['MJD'] > peak, 'MJD'].values
SNoi_i = SNoi_i[~SNoi_i['MJD'].isin(SNoi_errEpochs)]

SNoi_z = SNoi[SNoi['FLT'] == 'z']
SNoi_z.sort_values(by=['MJD'], inplace=True)
SNoi_z_model = SNoi_z.rolling(5).mean()
SNoi_z_resids = np.abs(SNoi_z_model['MAG'] - SNoi_z['MAG'])
SNoi_z_highE = SNoi_z[SNoi_z_resids.values > 0.8]
SNoi_z_highE = SNoi_z[SNoi_z_resids.values > 0.3]
SNoi_errEpochs = SNoi_z_highE.loc[SNoi_z_highE['MJD'] > peak, 'MJD'].values
SNoi_z = SNoi_z[~SNoi_z['MJD'].isin(SNoi_errEpochs)]

SNoi_cut = SNoi_g[SNoi_g['MAG'] < 16.7]
SNoi_cut = SNoi_cut[SNoi_cut['MJD'] > 58890]
SNoi_g = SNoi_g[~SNoi_g.index.isin(SNoi_cut.index)]

#plt.plot(SNoi_i['MJD'], SNoi_i['MAG'], 'o')
#plt.plot(SNoi_i_model['MJD'], SNoi_i_model['MAG'], ':')
#plt.gca().invert_yaxis()

test = SN2020oi.dropna(subset=['TELESCOPE'])
np.unique(test['TELESCOPE'])

#snbn_g = SN2015bn[SN2015bn['band'] == 'g\'']
#snbn_r = SN2015bn[SN2015bn['band'] == 'r\'']
#snbn_i = SN2015bn[SN2015bn['band'] == 'i\'']

ptf_g = iPTF13bvn[iPTF13bvn['band'] == 'g']
ptf_r = iPTF13bvn[iPTF13bvn['band'] == 'r']
ptf_i = iPTF13bvn[iPTF13bvn['band'] == 'i']
ptf_z = iPTF13bvn[iPTF13bvn['band'] == 'z']

#SNoi_g = SNoi[SNoi['FLT'] == 'g']
#SNoi_r = SNoi[SNoi['FLT'] == 'r']
#SNoi_i = SNoi[SNoi['FLT'] == 'i']

#snbn_gmax = snbn_g.iloc[np.where(snbn_g['magnitude'] == np.nanmin(snbn_g['magnitude']))]['time'].values[0]
#snbn_rmax = snbn_r.iloc[np.where(snbn_r['magnitude'] == np.nanmin(snbn_r['magnitude']))]['time'].values[0]
#snbn_imax = snbn_i.iloc[np.where(snbn_i['magnitude'] == np.nanmin(snbn_i['magnitude']))]['time'].values[0]

ptf_gmax = ptf_g.iloc[np.where(ptf_g['magnitude'] == np.nanmin(ptf_g['magnitude']))]['time'].values[0]
ptf_rmax = ptf_r.iloc[np.where(ptf_r['magnitude'] == np.nanmin(ptf_r['magnitude']))]['time'].values[0]
ptf_imax = ptf_i.iloc[np.where(ptf_i['magnitude'] == np.nanmin(ptf_i['magnitude']))]['time'].values[0]
ptf_zmax = ptf_z.iloc[np.where(ptf_z['magnitude'] == np.nanmin(ptf_z['magnitude']))]['time'].values[0]

snoi_gmax = SNoi_g.iloc[np.where(SNoi_g['MAG'] == np.nanmin(SNoi_g['MAG']))]['MJD'].values[0]
snoi_rmax = SNoi_r.iloc[np.where(SNoi_r['MAG'] == np.nanmin(SNoi_r['MAG']))]['MJD'].values[0]
snoi_imax = SNoi_i.iloc[np.where(SNoi_i['MAG'] == np.nanmin(SNoi_i['MAG']))]['MJD'].values[0]
SNoi_zmax = SNoi_z.iloc[np.where(SNoi_z['MAG'] == np.nanmin(SNoi_z['MAG']))]['MJD'].values[0]

match_g = 0.8
match_r = 0.75
match_i = 0.7
match_z = 0.5

offset_g = 0
offset_r = 2
offset_i = 4
offset_z = 6

#best line fits to g, r, and i data after ~25 days and the fit at proposed_phase
p1_g = np.poly1d(np.polyfit(ptf_g['time'][70:] - ptf_gmax, ptf_g['magnitude'][70:]-match_g-offset_g, 1))
p1_r = np.poly1d(np.polyfit(ptf_r['time'][70:] - ptf_gmax, ptf_r['magnitude'][70:]-match_r-offset_r, 1))
p1_i = np.poly1d(np.polyfit(ptf_i['time'][70:] - ptf_gmax, ptf_i['magnitude'][70:]-match_i-offset_i, 1))
p1_z = np.poly1d(np.polyfit(ptf_z['time'][25:] - ptf_gmax, ptf_z['magnitude'][25:]-match_z-offset_z, 1))

g_est = p1_g(proposed_phase)
r_est = p1_r(proposed_phase)
i_est = p1_i(proposed_phase)
z_est = p1_z(proposed_phase)

g_est_norm = g_est + offset_g
r_est_norm = r_est + offset_r
i_est_norm = i_est + offset_i
z_est_norm = z_est + offset_z

#22.7
#day 460 would be the latest to observe it
#460-380

g_est_norm
r_est_norm
i_est_norm
z_est_norm

g_line = p1_g(np.linspace(50, 400, num=50))
r_line = p1_r(np.linspace(50, 400, num=50))
i_line = p1_i(np.linspace(50, 400, num=50))
z_line = p1_z(np.linspace(50, 400, num=50))

g_col = '#246EB9'
r_col = '#4CB944'
i_col = 'tab:orange'
z_col = '#C33C54'

plt.figure(figsize=(10,8))
plt.plot(SNoi_g['MJD'] - peak, SNoi_g['MAG']-offset_g, 'o', mec='k', ms=5, c=g_col,zorder=50, label='g')
plt.plot(SNoi_r['MJD'] - peak, SNoi_r['MAG']-offset_r, 'o', mec='k',ms=5, c=r_col,zorder=50, label='r - %.1f'%offset_r)
plt.plot(SNoi_i['MJD'] - peak, SNoi_i['MAG']-offset_i, 'o',mec='k', ms=5,  c=i_col,zorder=50, label='i - %.1f'%offset_i)
plt.plot(SNoi_z['MJD'] - peak, SNoi_z['MAG']-offset_z, 'o',mec='k', ms=5,  c=z_col,zorder=50, label='z - %.1f'%offset_z)

ptf_pt, = plt.plot(-1000, -1000, 's', ms=5,mew=2, c='k', alpha=0.5, fillstyle='none')
plt.plot(ptf_g['time'] - ptf_gmax, ptf_g['magnitude']-match_g-offset_g, 's', ms=5,mew=2, c=g_col, alpha=0.5, fillstyle='none')
plt.plot(ptf_r['time'] - ptf_rmax, ptf_r['magnitude']-match_r-offset_r, 's', ms=5, mew=2, c=r_col, alpha=0.5, fillstyle='none')
plt.plot(ptf_i['time'] - ptf_imax, ptf_i['magnitude']-match_i-offset_i, 's', ms=5,  mew=2, c=i_col, alpha=0.5, fillstyle='none')
plt.plot(ptf_z['time'] - ptf_imax, ptf_z['magnitude']-match_z-offset_z, 's', ms=5,  mew=2, c=z_col, alpha=0.5, fillstyle='none')

plt.plot(ptf_g['time'][:-5] - ptf_gmax, ptf_g['magnitude'].rolling(1).mean()[:-5]-match_g-offset_g, '--',  c=g_col, alpha=0.5)
plt.plot(ptf_r['time'][:-10] - ptf_rmax, ptf_r['magnitude'].rolling(1).mean()[:-10]-match_r-offset_r, '--',  c=r_col, alpha=0.5)
plt.plot(ptf_i['time'][:-5] - ptf_imax, ptf_i['magnitude'].rolling(1).mean()[:-5]-match_i-offset_i, '--',  c=i_col, alpha=0.5)
plt.plot(ptf_z['time'][:-5] - ptf_imax, ptf_z['magnitude'].rolling(1).mean()[:-5]-match_z-offset_z, '--',  c=z_col, alpha=0.5)

plt.plot(np.linspace(50, 400, num=50), g_line, c=g_col, linestyle='--', alpha=0.5)
plt.plot(np.linspace(50, 400, num=50), r_line, c=r_col,  linestyle='--', alpha=0.5)
plt.plot(np.linspace(50, 400, num=50), i_line, c=i_col,  linestyle='--', alpha=0.5)
plt.plot(np.linspace(50, 400, num=50), z_line, c=z_col,  linestyle='--', alpha=0.5)

#spec_0116['flux_mean'] = spec_0116['flux(erg/s cm2 A)'].rolling(5).mean()
#spec_0109['flux_mean'] = spec_0109['flux(erg/s cm2 A)'].rolling(5).mean()

rect = patches.Rectangle((proposed_phase-20,g_est-9.5),40,10.5,linewidth=2,edgecolor='k',facecolor='gray', alpha=0.4,zorder=10)
plt.gca().add_patch(rect)

plt.xlabel("Phase from Peak (days)")
plt.ylabel("Observed Magnitude")
plt.plot(proposed_phase, g_est, '*', ms=15, c=g_col,zorder=1000)
plt.plot(proposed_phase, r_est, '*', ms=15, c=r_col,zorder=1000)
plt.plot(proposed_phase, i_est, '*', ms=15, c=i_col,zorder=1000)
plt.plot(proposed_phase, z_est, '*', ms=15, c=z_col,zorder=1000)
plt.text(295, 8.3, 'SN2020oi')

legend = plt.legend(loc='center', bbox_to_anchor=(0.8, 0.8),frameon=False)
leg2 = plt.legend([ptf_pt],['iPTF13bvn'], frameon=False, bbox_to_anchor=(0.2, 0.2), loc='center')
plt.gca().tick_params(axis="y",direction="in", right="on",labelleft="on")
plt.gca().tick_params(axis="x",direction="in", top="on",labelleft="on")
plt.gca().tick_params(axis="x",which='minor',direction="in", top="on",labelleft="on")
plt.gca().tick_params(axis="y",which='minor',direction="in", right="on",labelleft="on")
# Manually add the first legend back
plt.text(15, 20.75, "SN Ic Template")
plt.gca().add_artist(legend)
plt.minorticks_on()
#plt.gca().add_artist(leg2)
plt.xlim((-10, 80))
plt.ylim((7, 25))
#legend.get_frame().set_facecolor('none')
#plt.gca().invert_yaxis()
plt.axvline(x=289, alpha=0.5, c='gray')
plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/SN2020oi_brightnessFit.pdf",dpi=300)

SNoi_prePeak[SNoi_prePeak['MAGERR'] < 0.1]

############## PLOT FOUR: DE-REDDENING AND RE-REDDENING 289 DAY SPECTRUM OF IPTF13BVN ###############################

#plt.xlim((-25, 80))
#plt.ylim((10, 18))
# E(B−V)MW=0.045 and host galaxy extinction of E(B−V)host=0.17,
#total E(B_V)~0.21500000000000002

template_spec = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/iptf13_spectrum_d289.txt")
plt.figure(figsize=(16,7))
plt.plot(template_spec['wavelength'], template_spec['flux'])

wave_nm = 0.1*template_spec['wavelength'].values
wave_invmic = 1/(template_spec['wavelength'].values*1.e-4) #wavelength in inverse microns
from extinction import ccm89, apply, remove

#AV = 3.2E(B-V)
#RV = A(V)/E(B-V) =  3.1
#R_V = 3.1
R_V = 3.1
EBV_20oi = 0.18
EBV_ptf = 0.215
AV_20oi = R_V*EBV_20oi
AV_ptf = R_V*EBV_ptf

# "deredden" flux by E(B-V)~0.215, iPTF13bvn value
dereddened_flux = remove(ccm89(wave_invmic, AV_ptf, R_V, unit='invum'), template_spec['flux'].values)

## "redden" flux by E(B-V)~0.18, 2020oi value
reddened_flux = apply(ccm89(wave_invmic, AV_20oi, R_V, unit='invum'), dereddened_flux)

plt.figure(figsize=(20,8))
plt.plot(template_spec['wavelength'], template_spec['flux'], label='orig')
plt.plot(template_spec['wavelength'], dereddened_flux,label='dereddened (true)')
plt.plot(template_spec['wavelength'], reddened_flux, label='reddened (2020oi)')
plt.xlim((6700,8000))
plt.legend(fontsize=16)

new_spec = pd.DataFrame({'wavelength':wave_nm, 'flux':reddened_flux})
new_spec.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/iptf_spectrum_corrected.csv",index=False)


############## PLOT FIVE: COMPARING Nickel mass, ejecta mass, and kinetic energy of the explosion! ###############################
Mni = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/Ek_ov_Mej_vs_MNi_noerr.txt", delimiter=', ')

#Final resultss
#diffusion time t$_d$ (days) : 7.24 $\pm$ 0.09
#Nickel mass M$_{Ni}$ (MSun) : 0.22 $\pm$ 0.002
#t$_0$ (MJD) : 58854.7
#Ejected mass M$_{tot}$ (MSun) : 1.47
#Kinetic energy (FOE) : 2.4

type = 14*['broad-lined']
type[0] = 'dark'
type[1] = 'dark'
#type[2] = 'broad-lined'
type[-7] = 'hypernova'
type[-8] = 'hypernova'
type[-9] = 'hypernova'
type[-10] = 'hypernova'
type[-3] = 'normal'
type[-4] = 'normal'
type[-5] = 'normal'
type[3] = 'normal'
#type[-7] = 'normal'
#type[2] = 'broad-lined'
Mni['type'] = np.array(type)


#hn_c = '#08A4BD'
#norm_c = '#52AA5E'
#bl_c = '#FFBC42'
#dark_c = '#4E148C'
#c_20oi = '#C41E3D'

hn_c = sns.color_palette("colorblind", 10)[0]
norm_c = sns.color_palette("colorblind", 10)[1]
bl_c = sns.color_palette("colorblind", 10)[2]
dark_c = sns.color_palette("colorblind", 10)[3]
c_20oi = sns.color_palette("colorblind", 10)[4]

#proposed_phase

# A meta-analysis of core-collapse supernova 56Ni masses
Ni = [0.07, 0.11]
Mj = [1.41-0.11, 1.41+0.11]
Ek = [1.38-0.19, 1.38+0.19]
#1.3764 $\pm$ 0.19
Mni_20oi = np.mean(Ni)
Ek_20oi = np.mean(Ek) #change later
Mej_20oi = np.mean(Mj)

err_Ni = np.std(Ni)
err_Ek = np.sqrt((np.std(Ek)/Ek[-1])**2 + (np.std(Mj)/Mj[-1])**2)

#2003jd with an ejected mass of Mej= 3.0 ± 1 M⊙ and a kinetic energy of Ek(tot) = 7+3−2× 1051erg.
#MCG -01-59-21
#z= 0.0187
# 81.7 Mpc
Ek_jd = 7
Mej_jd = 3
Mni_jd = 0.36
err_Ni_jd = 0.04
err_Mej_jd = 1
err_Ek_jd = 3
err_Ek_jd = np.sqrt((err_Ek_jd/Ek_jd)**2 + (err_Mej_jd/Mej_jd)**2)

#2009bb 4.1+-1.9 Msun of material was ejected with 0.22 +-0.06 Msun of it being 56Ni. The resulting kinetic energy is 1.8+-0.7x10^52 e
#NGC 3278
#z~0.009937
# 43.1 Mpc
Ek_bb = 18
Mej_bb = 4.1
Mni_bb = 0.22
err_Ni_bb = 0.06
err_Mej_bb = 1.9
err_Ek_bb = 7
err_Ek_bb = np.sqrt((err_Ek_bb/Ek_bb)**2 + (err_Mej_bb/Mej_bb)**2)

plt.figure(figsize=(10,8))
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

sns.set_context("poster",font_scale=1.)

plt.axhline(y=0.155, label='Ic Median ', linestyle=':',alpha=0.5)
plt.axhline(y=0.369, label='Ic-BL Median', linestyle='--',c='gray',alpha=0.5)
plt.errorbar(Ek_20oi/Mej_20oi, Mni_20oi, fmt='*',yerr=err_Ni,xerr=err_Ek, ms=20, markeredgecolor='k', c=c_20oi, label='SN 2020oi')
plt.plot(Mni.loc[Mni['type'] == 'dark', 'Ek/Mej'], Mni.loc[Mni['type'] == 'dark', 'MNi'], '+', ms=15, mew=3, c=dark_c, markeredgecolor=dark_c, label='faint/dark SNe')
plt.plot(Mni.loc[Mni['type'] == 'normal', 'Ek/Mej'], Mni.loc[Mni['type'] == 'normal', 'MNi'], 's', c=norm_c, markeredgecolor='k', label='normal SNe')
plt.plot(Mni.loc[Mni['type'] == 'broad-lined', 'Ek/Mej'], Mni.loc[Mni['type'] == 'broad-lined', 'MNi'], '^', c=bl_c, markeredgecolor='k', label='broad-lined SNe')
plt.plot(Mni.loc[Mni['type'] == 'hypernova', 'Ek/Mej'], Mni.loc[Mni['type'] == 'hypernova', 'MNi'], 'o', ms=8, c=hn_c,  markeredgecolor='k', label='hypernovae associated with GRBs')
plt.errorbar(Ek_jd/Mej_jd, Mni_jd,  marker='s', c=norm_c,  yerr=err_Ni_jd,xerr=err_Ek_jd, markeredgecolor='k')
plt.errorbar(Ek_bb/Mej_bb, Mni_bb, marker='s', c=norm_c,  yerr=err_Ni_bb,xerr=err_Ek_bb, markeredgecolor='k')
#plt.plot(Ek_bb/Mej_bb, Mni_bb, fmt='1')
plt.gca().tick_params(axis="y",direction="in", right="on",labelleft="on")
plt.gca().tick_params(axis="x",direction="in", top="on",labelleft="on")
plt.gca().tick_params(axis="x",which='minor',direction="in", top="on",labelleft="on")
plt.gca().tick_params(axis="y",which='minor',direction="in", right="on",labelleft="on")
# Manually add the first legend back
plt.minorticks_on()
plt.xlabel(r"$E_k/M_{ej}$ ($10^{51}$ erg/$M_{\odot}$)")
plt.ylabel(r"$^{56}$Ni ($M_{\odot}$)")
#plt.text(0.02, 0.0043, "97D",fontsize=14)
#plt.text(0.05, 0.002, "99br",fontsize=14)
#plt.text(0.2, 0.028, "05bf",fontsize=14)
#plt.text(0.48, 0.055, "07Y",fontsize=14)
#plt.text(0.35, 0.1, "13bvn",fontsize=14)
#plt.text(0.4, 0.075, "93J",fontsize=14)
#plt.text(0.75, 0.05, "94I",fontsize=14)
#plt.text(1.1, 0.05, "08D",fontsize=14)
#plt.text(1.3, 0.08, "02ap",fontsize=14)
#plt.text(1.6, .13, "97ef",fontsize=14)
#plt.text(0.7, 0.2, "06aj",fontsize=14)
#plt.text(3, 0.65, "03Iw",fontsize=14)
#plt.text(2.4, 0.45, "98bw",fontsize=14)
#plt.text(5.5, 0.4, "03dh",fontsize=14)
#plt.text(2.2, 0.25, "13jd",fontsize=14)
#plt.text(5.3, 0.2, "09bb",fontsize=14)
#plt.text(0.8, 0.115, "20oi",fontsize=16, fontweight='bold')
plt.xlim((0.01, 10))
plt.ylim((0.001, 1))
plt.legend(loc='lower right', frameon=False,fontsize=16)
plt.xscale("log")
plt.yscale("log")
plt.gca().tick_params(axis='x', which='major', pad=10)
plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/Supernova_ExplosionEnergies_ComparisonLines_newCols.png",dpi=300,bbox_inches='tight')

############## PLOT SIX: Plotting the best-fit spectral fit!! ###############################
specmatch = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/SN2020oi_sim_spec_Day13.txt", header=None, delim_whitespace=True)
spec3 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra_mangled_018/SN2020oi-20200120-floyds.flm_mangled_ext_cor_total_0.18.flm", delimiter='   ')
spec3.columns.values[1]

#spec3['flux'] = spec3['flux']*1.e-20
#He: 0.005#
#    C: 0.00
#    O: 0.625
#    Ne: 0.20
#    Mg: 0.040
#    Si: 0.011
#    S: 0.011
#    Ca: 0.02
#    Ni: 0.014
#    Fe: 0.01
#    Cr: 0.009
#    Ti: 0.009
#    Ar: 0.045

d = 16.8*3.08568e24
Lumdens = 4*np.pi*(d**2)*spec3['  flux(erg / s cm2 A)']

sns.set_context("poster",font_scale=1.)
#sns.set_context("dc")
plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
plt.figure(figsize=(22,10))
plt.plot(spec3['wavelength(A)'], Lumdens, c='#993955')
plt.plot(specmatch[0], specmatch[1], c='tab:blue', linestyle='--')
plt.xlabel("Wavelength (Å)")
plt.text(6000, 7e38, "+13d (20 Jan 2020)", c='#993955',fontweight='bold',fontsize=30)
plt.text(6300, 6.2e38,"Best-fit Model", c='tab:blue',fontweight='bold',fontsize=30)
plt.ylabel(r"Luminosity Density (erg s$^{-1}$ Å$^{-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/SpectralFit_Day13.pdf",dpi=300, bbox_inches='tight')

#####################################################################################################################
from astropy.coordinates import SkyCoord
import astropy.units as u

SNoi_ra = 185.728875
SNoi_dec = +15.8236
M100_ra =185.728463
M100_dec =15.821818

tempCoord = SkyCoord(SNoi_ra*u.deg,SNoi_dec*u.deg, frame='icrs')
sep = SkyCoord(M100_ra*u.deg,M100_dec*u.deg, frame='icrs').separation(tempCoord)
print(sep)

############# how far is 2020oi from its host galaxy? let's find out: #################################################################
#######################################################################################################################################
Uband = iPTF13bvn[iPTF13bvn['band'] == 'U']
plt.plot(Uband['time'], Uband['magnitude'], 'o')

UVOT = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/UVOT_Reductions/temp_from_younger_cluster/2020oi_Swift_3arcsec.csv")
UVOT['MAGERR'] = pd.to_numeric(UVOT['MAGERR'])
UVOT.replace(-999, 0.00, inplace=True)
bb = UVOT[UVOT['FLT'] == 'B']
w2 = UVOT[UVOT['FLT'] == 'UVW2']
uu = UVOT[UVOT['FLT'] == 'U']
vv = UVOT[UVOT['FLT'] == 'V']
w1 = UVOT[UVOT['FLT'] == 'UVW1']
m2 = UVOT[UVOT['FLT'] == 'UVM2']

peak = 58866.1
plt.figure(figsize=(10,8))

uPeak_2020oi = np.nanmin(uu['MAG'])
uPeak_iPTF13bvn = np.nanmin(Uband['magnitude'])
uPT_2020oi = uu['MJD'].values[np.where(uu['MAG'] == uPeak_2020oi)][0]
uPT_iPTF13bvn = Uband['time'].values[np.where(Uband['magnitude'] == uPeak_iPTF13bvn)][0]

plt.errorbar(uu['MJD'].values-uPT_2020oi, uu['MAG'].values, yerr=uu['MAGERR'], uplims=uu['ULIM'], marker='o', ms=4, c='tab:orange', label='U, 2020oi')
plt.errorbar(Uband['time']-uPT_iPTF13bvn, Uband['magnitude']-1.2, yerr=Uband['e_magnitude'], uplims=Uband['upperlimit'], marker='o', ms=4, c='tab:red', label='U, iPTF13bvn',capsize=5)
plt.legend()
plt.xlim(xmin=-8, xmax=20)
plt.xlabel("Phase from Peak (days)",fontsize=16)
plt.ylabel("Magnitude")
plt.gca().invert_yaxis()

spec = pd.read_csv("/Users/alexgagliano/Desktop/iptf_spectrum_corrected_blocked.txt", delimiter='  ', header=None)


######### old code - estimating bolometric luminosity for Luca

Lbol_interp = interp1d(Lbol['MJD'] - t0, Lbol['logL'])
vals = np.array([4, 11, 15, 17, 19, 22])

plt.figure(figsize=(10,7))
for i in np.arange(len(vals)):
    x = np.array([vals[i]])
    plt.plot(x, Lbol_interp(x), 'o', zorder=30, label="log10 Lbol(%i days) = %.3f"% (vals[i], Lbol_interp(vals[i])))
plt.plot(Lbol['MJD'] - t0, Lbol['logL'])
plt.xlabel("Time from t0")
plt.ylabel("log10 Lbol")
plt.legend()
