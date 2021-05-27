import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#phase from peak -- day -11 (MJD 54329)
SN2007gr = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/reference_data/2007gr_2007-08-17_00-00-00_Other_Other_SUSPECT.dat", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
#phase from peak -- day -9
SN1983V = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/reference_data/1983V_1983-11-15_00-00-00_Other_Other_None.ascii", delimiter='\t', header=None, names=['wavelength', 'flux'])
#phase from peak -- day -9
SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_goodman_20200109_norm_ext_cor_total_0_133.csv", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
#phase from peak -- day -6
SN1994I = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/reference_data/SN1994I_1994-04-03_09-21-36_FLWO-1.5m_FAST_CfA-Stripped.flm",delim_whitespace=True, header=None, names=['wavelength', 'flux'])

SN2020oi_sim = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/finalSims_SNpeak/SN2020oi_sim_spec_Day3_final.txt",delim_whitespace=True, header=None, names=['wavelength', 'flux'])

from astropy.time import Time
times = ['2007-08-17T00:00:00.00',
'1983-11-15T00:00:00.00',
'2020-01-09T00:00:00.00',
'1994-04-03T09:21:36.00']
t = Time(times, format='isot')

t.mjd

#array([54329.  , 45653.  , 58857.  , 49445.39])
t0_set = np.array([54325, 45514, 58853., 49438.])

t.mjd - t0_set

sns.set_context("poster")

import matplotlib as mpl

sns.set_context("talk",font_scale=2.5)

#plotting - TARDIS sim
#sns.set_context("poster")
#    sns.set(font_scale=)
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
#plt.title('SN2020oi spectral sequence', fontsize=17)
#plt.plot(w2,l2+1.55E39,'k',label='Day 2')
plt.plot(SN1994I['wavelength'],SN1994I['flux']/np.nanmean(SN1994I['flux']),'k',lw=2, zorder=100, c='k', alpha=1)
plt.plot(SN2020oi['wavelength']/1.0054,SN2020oi['flux']/np.nanmean(SN2020oi['flux'])+1.8,'k',lw=2, zorder=100, c='#A165E2')
plt.plot(SN2020oi_sim['wavelength']/1.0054,SN2020oi_sim['flux']/(3.e52*np.nanmean(SN2020oi['flux']))+1.7,'k',lw=2, zorder=100, c='#00916E', alpha=1)
#plt.plot(SN1983V['wavelength'],SN1983V['flux']/np.nanmean(SN1983V['flux'])+1.9,'k',lw=1.7, zorder=100, c='k', alpha=0.6)
plt.plot(SN2007gr['wavelength'],SN2007gr['flux']/np.nanmean(SN2007gr['flux'])+1,'k',lw=2, zorder=100, c='k', alpha=1)


plt.xlim(3100,9000)
#plt.ylim(-0.06*1E39,2.2*1E39)
plt.tick_params(axis='y',labeltop='off')
plt.tick_params(axis='x', labelbottom='off')
plt.ylim(ymax=4.2)
plt.gca().yaxis.set_ticklabels([])
plt.xlabel(r"Rest-Frame Wavelength (\AA)")
plt.ylabel(r"Normalized $F_{\lambda}$ + Offset")
plt.savefig('/Users/alexgagliano/Documents/Research/2020oi/img/SN2020oi_Day3_Comparison_wSim.png',dpi=300, bbox_inches='tight')
