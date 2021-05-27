#spectra plotting for 2020oi
import os
import matplotlib as mpl
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.chdir("/Users/alexgagliano/Documents/Research/2020oi/data/spectra")

#spec_phases relative to time of explosion
spec_phases = np.array([+3.1, +10.4, +14.4, +16.3, +18.2, +20.8, +22.5, +25.2, +26.1, +26.2, +37.8, +40.8, +141.4])
t0 = 58854.2
MJD_spec = spec_phases + t0
peak = 58866.1
MJD_spec - peak

spec_phases.sort()
spec1 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_goodman_20200109_norm_ext_cor_total_0_133.csv", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
spec1pt5_b = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_kast_blue_20200112_norm_ext_cor_total_0_133.csv", names=['wavelength', 'flux', 'fluxerr'], delim_whitespace=True)
spec1pt5_r = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_kast_red_20200112_norm_ext_cor_total_0_133.csv", names=['wavelength', 'flux', 'fluxerr'], delim_whitespace=True)
spec2 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_floyds_20200116_norm_no_ext_cor.csv", header=None,  names=['wavelength', 'flux'], delim_whitespace=True)
spec3 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_floyds_20200120_norm_no_ext_cor.csv", header=None,  names=['wavelength', 'flux'], delim_whitespace=True)
spec4 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_floyds_20200122_norm_no_ext_cor.csv", header=None,  names=['wavelength', 'flux'], delim_whitespace=True)
spec5 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_floyds_20200124_norm_no_ext_cor.csv",  header=None, names=['wavelength', 'flux'], delim_whitespace=True)
spec6 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_lris_20200127_norm_no_ext_cor.csv", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
#spec7 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/2020oi-FLOYDS-S-2020-01-28.csv", skiprows=18)
spec8 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_floyds_20200131_norm_no_ext_cor.csv", header=None,names=['wavelength', 'flux'], delim_whitespace=True)
spec9 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_goodman_20200201_norm_no_ext_cor.csv", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
spec10 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_floyds_20200201_norm_no_ext_cor.csv",header=None, names=['wavelength', 'flux'], delim_whitespace=True)
spec11 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_kast_20200213_norm_ext_cor_total_0_133.csv", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
spec12 = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv/SN2020oi_lris_20200216_norm_no_ext_cor.csv", delim_whitespace=True, header=None, names=['wavelength', 'flux'])
spec13 = pd.read_csv("./2020oi-KAST-2020-05-26.csv", skiprows=18)

spec1['flux_norm'] = spec1['flux']/np.nanmedian(spec1['flux'])
spec1pt5_b['flux_norm'] = spec1pt5_b['flux']/np.nanmedian(spec1pt5_b['flux'])
spec1pt5_r['flux_norm'] = spec1pt5_r['flux']/np.nanmedian(spec1pt5_r['flux'])
spec2['flux_norm'] = spec2['flux']/np.nanmedian(spec2['flux'])
spec3['flux_norm'] = spec3['flux']/np.nanmedian(spec3['flux'])
spec4['flux_norm'] = spec4['flux']/np.nanmedian(spec4['flux'])
spec5['flux_norm'] = spec5['flux']/np.nanmedian(spec5['flux'])
spec6['flux_norm'] = spec6['flux']/np.nanmedian(spec6['flux'])
#spec7['flux_norm'] = spec7['flux']/np.nanmedian(spec7['flux'])
spec8['flux_norm'] = spec8['flux']/np.nanmedian(spec8['flux'])
spec9['flux_norm'] = spec9['flux']/np.nanmedian(spec9['flux'])
spec10['flux_norm'] = spec10['flux']/np.nanmedian(spec10['flux'])
spec11['flux_norm'] = spec11['flux']/np.nanmedian(spec11['flux'])
spec12['flux_norm'] = spec12['flux']/np.nanmedian(spec12['flux'])
spec13['flux_norm'] = spec13['flux']/np.nanmedian(spec13['flux'])

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
#    sns.set_context("talk",font_scale=1.5)
    #sns.set_context("dc")
    shift = np.array([1, 3, 5, 7, 9,11, 13.5, 16,19, 22, 24, 26])
#    plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})
    plt.figure(figsize=(20,20))
#    plt.ylim((-25, 6))
    plt.plot(spec1['wavelength'], spec1['flux_norm']+3,label='2020-01-09', c='k',lw=1)
    n = 2
    plt.fill_between(spec1['wavelength'][::n],  spec1['flux_norm'][::n]-0.1+3, y2=spec1['flux_norm'][::n]+0.1+3,color='#004FFF',alpha=0.4)

    plt.plot(spec1pt5_b['wavelength'], spec1pt5_b['flux_norm']+0.5+1.0, lw=1, c='k')
    plt.plot(spec1pt5_r['wavelength'], spec1pt5_r['flux_norm']-0.5+1.0, lw=1, c='k')

    plt.plot(spec2['wavelength'], spec2['flux_norm']-shift[0],lw=1, label='2020-01-16', c='k')
    plt.plot(spec3['wavelength'], spec3['flux_norm']-shift[1],lw=1, label='2020-01-20', c='k')
    plt.plot(spec4['wavelength'], spec4['flux_norm']-shift[2],lw=1, label='2020-01-22', c='k')
    plt.plot(spec5['wavelength'], spec5['flux_norm']-shift[3],lw=1, label='2020-01-24', c='k')
    plt.plot(spec6['wavelength'], spec6['flux_norm']-shift[4],lw=1, label='2020-01-27', c='k')
    #plt.plot(spec7['wavelength'], spec7['flux_norm']-shift[5],lw=1, label='2020-01-28', c='k')
    plt.plot(spec8['wavelength'], spec8['flux_norm']-shift[5],lw=1, label='2020-01-31', c='k')
    plt.plot(spec9['wavelength'], spec9['flux_norm']-shift[6],lw=1, label='2020-02-01', c='k')
    plt.plot(spec10['wavelength'], spec10['flux_norm']-shift[7],lw=1, label='2020-02-01', c='k')
    plt.plot(spec11['wavelength'], spec11['flux_norm']-shift[8],lw=1, label='2020-02-13', c='k')
    plt.plot(spec12['wavelength'], spec12['flux_norm']-shift[9],lw=1, label='2020-02-16', c='k')
    #plt.plot(spec13['wavelength'], spec13['flux_norm']-shift[11],lw=1, label='2020-05-26', c='k')
    plt.gca().tick_params(axis="y",direction="in", right="on",labelleft="on")
    plt.gca().tick_params(axis="x",direction="in", top="on",labelleft="on")
    plt.gca().tick_params(axis="x",which='minor',direction="in", top="on",labelleft="on")
    plt.gca().tick_params(axis="y",which='minor',direction="in", right="on",labelleft="on")


#    plt.text(4700, 0.9, '[Fe II]',color='#7A0037')#
#    plt.text(5700, 1.1, '[Na I D]',color='#7A0037')#
#    plt.text(8000, 0.45, '[Ca II]',color='#7A0037')
#    plt.text(7500, 0.5, '[O I]',color='#7A0037')

#    plt.text(10500, +3., "+%.2f d" %spec_phases[0], c='#2FA9B1')
#    plt.text(9000, shift[0]+2., "+%.2f d" %spec_phases[1], c='#2FA9B1')
#    plt.text(10700, -shift[1],  "+%.2f d" %spec_phases[2], c='#2FA9B1')
#    plt.text(10700, -shift[2],  "+%.2f d" %spec_phases[3], c='#2FA9B1')
#    plt.text(10700, -shift[3], "+%.2f d" %spec_phases[4], c='#2FA9B1')
#    plt.text(10500, -shift[4],  "+%.2f d" %spec_phases[5], c='#2FA9B1')
#    plt.text(10500, -shift[5],  "+%.2f d" %spec_phases[6], c='#2FA9B1')
#    plt.text(10500, -shift[6],  "+%.2f d" %spec_phases[7], c='#2FA9B1')#
#    plt.text(9000, -shift[7],  "+%.2f d" %spec_phases[8], c='#2FA9B1')#
#    plt.text(10500, -shift[8], "+%.2f d" %spec_phases[9], c='#2FA9B1')
#    plt.text(10500, -shift[9],  "+%.2f d" %spec_phases[10], c='#2FA9B1')
#    plt.text(10500, -shift[10],  "+%.2f d" %spec_phases[11], c='#2FA9B1')#
#    plt.text(10500, -shift[11]+1,  "+%.2f d" %spec_phases[12], c='#2FA9B1')

    #plt.legend()

    #plt.minorticks_on()
    plt.xlim((2800, 11700))
    plt.ylim(ymin=-23,ymax=6)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(r"Rest-Frame Wavelength (\AA)",fontsize=36)
    plt.ylabel(r"Normalized $F_{\lambda}$ + Offset",fontsize=36)
    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/FullSequence_noLabels.png",dpi=300,bbox_inches='tight')

make_plot()
