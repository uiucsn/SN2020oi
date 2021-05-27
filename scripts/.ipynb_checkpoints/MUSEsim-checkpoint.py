#spectra plotting for 2020oi
import os
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.chdir("/Users/alexgagliano/Documents/Research/2020oi/data/spectra")

#spec_phases relative to time of explosion
#spec_phases = np.array([+3.1, +10.4, +14.4, +16.3, +18.2, +20.8, +22.5, +25.2, +26.1, +26.2, +37.8, +40.8, +141.4])
#t0 = 58854.2
#MJD_spec = spec_phases + t0
#peak = 58866.1
#MJD_spec - peak
#spec_phases.sort()

spec1 = pd.read_csv("./2020oi-Goodman-2020-01-09.csv", skiprows=18)
spec2 = pd.read_csv("./2020oi-FLOYDS-N-2020-01-16.csv", skiprows=18)

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
plt.rcParams['xtick.bottom'] = True
#plt.rcParams['xtick.top'] = True
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
    shift = np.array([1, 3, 5, 7, 9,11, 13, 15,18, 21, 23, 25])
    plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
    plt.figure(figsize=(20,20))
    plt.ylim((-25, 6))
    plt.plot(spec1['wavelength'], spec1['flux_norm']+1.5,label='2020-01-09', c='k',lw=1)
    n = 10
    plt.fill_between(spec1['wavelength'][::n],  spec1['flux_norm'][::n]-0.1+1.5, y2=spec1['flux_norm'][::n]+0.1+1.5,color='#004FFF',alpha=0.4)

    plt.plot(spec2['wavelength'], spec2['flux_norm']-shift[0],lw=1, label='2020-01-16', c='k')
    plt.plot(spec3['wavelength'], spec3['flux_norm']-shift[1],lw=1, label='2020-01-20', c='k')
    plt.plot(spec4['wavelength'], spec4['flux_norm']-shift[2],lw=1, label='2020-01-22', c='k')
    plt.plot(spec5['wavelength'], spec5['flux_norm']-shift[3],lw=1, label='2020-01-24', c='k')
    plt.plot(spec6['wavelength'], spec6['flux_norm']-shift[4],lw=1, label='2020-01-27', c='k')
    plt.plot(spec7['wavelength'], spec7['flux_norm']-shift[5],lw=1, label='2020-01-28', c='k')
    plt.plot(spec8['wavelength'], spec8['flux_norm']-shift[6],lw=1, label='2020-01-31', c='k')
    plt.plot(spec9['wavelength'], spec9['flux_norm']-shift[7],lw=1, label='2020-02-01', c='k')
    plt.plot(spec10['wavelength'], spec10['flux_norm']-shift[8],lw=1, label='2020-02-01', c='k')
    plt.plot(spec11['wavelength'], spec11['flux_norm']-shift[9],lw=1, label='2020-02-13', c='k')
    plt.plot(spec12['wavelength'], spec12['flux_norm']-shift[10],lw=1, label='2020-02-16', c='k')
    plt.plot(spec13['wavelength'], spec13['flux_norm']-shift[11],lw=1, label='2020-05-26', c='k')
    plt.gca().tick_params(axis="y",direction="in", right="on",labelleft="on")
    plt.gca().tick_params(axis="x",direction="in", top="on",labelleft="on")
    plt.gca().tick_params(axis="x",which='minor',direction="in", top="on",labelleft="on")
    plt.gca().tick_params(axis="y",which='minor',direction="in", right="on",labelleft="on")

    plt.minorticks_on()
    plt.xlim((2800, 11700))
    plt.ylim(ymin=-26,ymax=5)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(r"Rest-Frame Wavelength (\AA)",fontsize=36)
    plt.ylabel(r"Normalized $F_{\lambda}$ + Offset",fontsize=36)
    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/FullSequence_noLabels.png",dpi=300,bbox_inches='tight')

make_plot()
