#spectra plotting for 2020oi
import os
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.chdir("/Users/alexgagliano/Documents/Research/2020oi/data/derived_data/")

#spec_phases relative to time of explosion
#spec_phases = np.array([+3.1, +10.4, +14.4, +16.3, +18.2, +20.8, +22.5, +25.2, +26.1, +26.2, +37.8, +40.8, +141.4])
#t0 = 58854.2
#MJD_spec = spec_phases + t0
#peak = 58866.1
#MJD_spec - peak
#spec_phases.sort()
spec1 = pd.read_csv("./museFits.txt", delim_whitespace=True)
spec1
sns.set_context("poster")
sns.set(font_scale=2.5)
sns.set_style('white', {'axes.linewidth': 0.5})
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['ytick.major.size'] = 15

plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.bottom'] = True#
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def make_plot():
    import matplotlib as mpl
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

    #sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    sns.set_context("talk",font_scale=1.5)
    #sns.set_context("dc")
    plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
    #plt.figure(figsize=(20,20), )
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20,20), sharex=True,gridspec_kw={'height_ratios': [3, 1], "hspace":0.1,"wspace":0.1})
    #plt.ylim((-25, 6))
    ax1.plot(spec1['l_obs'],  3.65e-17*spec1['f_obs']/1.e-17, c='#A165E2',lw=1)
    ax1.plot(spec1['l_obs'],  3.65e-17*spec1['f_syn']/1.e-17, c='#00916E',lw=1)
    resid = spec1['f_obs'] - spec1['f_syn']
    ax2.axhline(y=0, ls='--', c='gray', lw=4)
    ax2.plot(spec1['l_obs'], resid, c='tab:blue', lw=1)

    ax1.tick_params(axis="y",direction="in", right="on",labelleft="on")
    ax1.tick_params(axis="x",direction="in", top="on",labelleft="on")
    ax1.tick_params(axis="x",which='minor',direction="in", top="on",labelleft="on")
    ax1.tick_params(axis="y",which='minor',direction="in", right="on",labelleft="on")

    ax1.minorticks_on()
    ax1.set_xlim((4500, 9500))
    ax1.set_ylim(ymin=1.5,ymax=5.5)
    ax2.set_xlabel(r"Rest-Frame Wavelength (\AA)",fontsize=36)
    ax1.set_ylabel(r"F$_{\lambda}$ ($10^{-17}$ erg cm$^{-1}$ s$^{-1}$ \AA$^{-1}$)",fontsize=36)
    ax2.set_ylabel(r"$F_{\lambda, obs} - F_{\lambda, sim}$",fontsize=36)

    ax1.tick_params(axis="y",direction="in", right="on",labelleft="on")
    ax1.tick_params(axis="x",direction="in", top="on",labelleft="on")
    ax1.tick_params(axis="x",which='minor',direction="in", top="on",labelleft="on")
    ax1.tick_params(axis="y",which='minor',direction="in", right="on",labelleft="on")

    ax2.minorticks_on()
    ax2.set_xlim((4600, 9400))
    #ax1.set_ylim((1.e-17, 1.e-16))
    ax2.set_ylim(ymin=-0.2,ymax=0.6)
    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/MUSEspec.png",dpi=300,bbox_inches='tight')

make_plot()
