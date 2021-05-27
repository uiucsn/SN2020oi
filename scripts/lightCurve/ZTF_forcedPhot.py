import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ZTF = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/ZTF/ZTF_LC.csv", delim_whitespace=True)
ZTF

SNT = 3.
SNU = 5.

MJD = ZTF['jd']
mag_list = []
band = []
sigma_mag_list = []
ulim = []
for idx, row in ZTF.iterrows():
    band.append(row['filter'])
    if( row['forcediffimsnr']> SNT ):
    # we have a “confident” detection, compute and plot mag with error bar:
        mag = row['zpdiff']-2.5*np.log10(row['forcediffimflux'])
        mag_list.append(mag)
        sigma_mag= 1.0857 / row['forcediffimsnr']
        sigma_mag_list.append(sigma_mag)
        ulim.append(0)
    else:
    # compute flux upper limit and plot as arrow:
        mag = row['zpdiff']-2.5*np.log10(SNU*row['forcediffimfluxunc'])
        mag_list.append(mag)
        sigma_mag_list.append(np.nan)
        ulim.append(1)


ZTF_derivedPhot = pd.DataFrame({'TELESCOPE':np.array(['P48']*len(MJD)), 'MJD':MJD-2400000.5, 'JD':MJD, 'FLT':band, 'MAG':mag_list, 'MAGERR':sigma_mag_list, 'ULIM':ulim})

ZTF_g = ZTF_derivedPhot[ZTF_derivedPhot['FLT']=='ZTF_g']
ZTF_r = ZTF_derivedPhot[ZTF_derivedPhot['FLT']=='ZTF_r']

#last ZTF non-detection was at MJD ..... in the paper
#
ZTF_r[ZTF_r['MJD']<58856].sort_values(by=['MJD'])
58852.55
58855.54

np.nanmean([58852.55, 58855.54])

plt.figure(figsize=(10,7))
plt.plot(ZTF_g['MJD'], ZTF_g['MAG'], 'o')
plt.plot(ZTF_r['MJD'], ZTF_r['MAG'], 'o')
plt.xlim((58850, 58860))
plt.gca().invert_yaxis()

ZTF_real = ZTF_derivedPhot[ZTF_derivedPhot['ULIM']==0]

#ZTF_real.to_csv('/Users/alexgagliano/Documents/Research/2020oi/data/photometry/ZTFphotometry.csv',index=False)
plt.figure(figsize=(10,7))
sns.scatterplot(ZTF_real['JD']-2458857.50000, ZTF_real['MAG'], hue=ZTF_real['FLT'])
ZTF_real
plt.gca().invert_yaxis()
plt.xlim((-5, 100))

ZTF_real

ZTF_real.loc[ZTF_real['FLT']=='ZTF_r', 'FLT'] = 'r-ZTF'
ZTF_real.loc[ZTF_real['FLT']=='ZTF_g', 'FLT'] = 'g-ZTF'
ZTF_real.loc[ZTF_real['FLT']=='ZTF_i', 'FLT'] = 'i-ZTF'

SN2020oi = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry_wSynthetic_rBand.csv")

SN2020oi_noZTF = SN2020oi[~SN2020oi['FLT'].isin(['g-ZTF', 'r-ZTF', 'i-ZTF'])]

SN2020oi_forcedZTF = pd.concat([SN2020oi_noZTF, ZTF_real],ignore_index=True)
#SN2020oi_forcedZTF.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry_wSynthetic_wForcedZTF_final.csv",index=False)

LCO_u = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oi_u_LCO.phot", header=None, names=['MJD', 'FLT', 'MAG', 'MAGERR'], delim_whitespace=True)

LCO_u['TELESCOPE'] = 'Siding Spring 1m'
LCO_u['ULIM'] = 0.
np.unique(SN2020oi_forcedZTF['INSTRUMENT'].dropna())

SN2020oi_wLCO = pd.concat([SN2020oi_forcedZTF, LCO_u], ignore_index=True)
#SN2020oi_wLCO.to_csv("/Users/alexgagliano/Documents/Research/2020oi/data/photometry/2020oiphotometry_wSynthetic_wForcedZTF_wLCOu_final.csv",index=False)
