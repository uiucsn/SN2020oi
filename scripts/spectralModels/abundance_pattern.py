from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Palatino "


#data
vel = np.array([12700, 10000, 9000, 8000])
C = np.array([0.05, 0.05, 0.02, 0.01])
O = np.array([0.60, 0.65, 0.75, 0.75])
Ne = np.array([0.10, 0.19, 0.20, 0.20])
Mg = np.array([0.006, 0.005, 0.003, 0.003])
Si = np.array([0.006, 0.004, 0.003, 0.003])
S = np.array([0.005, 0.001, 0.005, 0.005])
Ca = np.array([0.0005, 0.0005, 0.0005, 0.001])
Fepeak = np.array([0.0018, 0.0016, 0.0016, 0.0023])

#data - Sauer
velS = np.array([11600, 10800, 8800, 7200])
CS = np.array([0.05, 0.05, 0.08, 0.03])
OS = np.array([0.59, 0.59, 0.58, 0.60])
NeS = np.array([0.26, 0.30, 0.30, 0.35])
MgS = np.array([0.016, 0.015, 0.019, 0.001])
SiS = np.array([0.0017, 0.0024, 0.0017, 0.0006])
SS = np.array([0.00067, 0.0009, 0.002, 0.0009])
#Ca = np.array([0.0005, 0.0005, 0.0005, 0.001])
FepeakS = np.array([0.0063, 0.0068, 0.0017, 0.0027])
import matplotlib as mpl

sns.set_context("talk",font_scale=1.5)

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
plt.rcParams['xtick.top'] = False
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


#Gagliano
def plot():
    plt.figure(figsize=(10,10))

#    cols = sns.color_palette("CMRmap", 10)
    cols2 = sns.color_palette("Dark2")
    cols_full = np.concatenate([cols2])

    plt.plot(vel, C, 's-', label='C', c=cols_full[0])
    plt.plot(vel, O, 's-', label='O', c=cols_full[1])
    plt.plot(vel, Ne, 's-', label='Ne', c=cols_full[2])
    plt.plot(vel, Mg, 's-', label='Mg', c=cols_full[3])
    plt.plot(vel, Si, 's-', label='Si', c=cols_full[4])
    plt.plot(vel, S, 's-', label='S', c=cols_full[5])
    #plt.plot(vel, Ca, 's:', color='purple', label='Ca')
    plt.plot(vel, Fepeak, 's-', label='Fe-peak', c=cols_full[6])
    #Sauer
    plt.plot(velS, CS, 'o:', alpha=0.5, c=cols_full[0])
    plt.plot(velS, OS, 'o:', alpha=0.5, c=cols_full[1])
    plt.plot(velS, NeS, 'o:', alpha=0.5, c=cols_full[2])
    plt.plot(velS, MgS, 'o:', alpha=0.5, c=cols_full[3])
    plt.plot(velS, SiS, 'o:', alpha=0.5, c=cols_full[4])
    plt.plot(velS, SS, 'o:', alpha=0.5, c=cols_full[5])
    plt.plot(velS, FepeakS, 'o:', alpha=0.5, c=cols_full[6])
    #plot
    plt.xscale('linear')
    plt.yscale('log')
    #plt.legend(loc='upper center', bbox_to_anchor=(1.02, 1.0), ncol=1, fancybox=True, fontsize=20)
    plt.xlabel('Velocity (km s$^{-1}$)')
    plt.ylabel('Ejecta mass fraction')
    plt.xlim(6500,13500)
    plt.ylim(ymax=3.e0)
    plt.suptitle('Time from Explosion (days)',fontsize=28, y=0.965)
    plt.text(12700, 3.25, '10.4', ha='center')
    plt.text(10000, 3.25, '14.4', ha='center')
    plt.text(9000, 3.25, '16.3', ha='center')
    plt.text(8000, 3.25, '18.2', ha='center')

    legend_elements = [Line2D([0], [0], marker='s', linestyle='-', color='k', label='SN 2020oi',
                            markerfacecolor='k', mec='k',markersize=10, mew=2),
                       Line2D([0], [0], marker='o', linestyle=':',alpha=0.5, color='k', label='SN 1994I',
                            markerfacecolor='k', mec='k',markersize=10, mew=2)]

    leg1 = plt.gca().legend(fontsize=18, handles=legend_elements, loc='lower right', handletextpad=0.1, borderaxespad=1.5,labelspacing=0.5, bbox_to_anchor=(0.79,0.85), ncol=2,frameon=True, fancybox=True, edgecolor='k')

    plt.savefig("/Users/alexgagliano/Documents/Research/2020oi/img/AbundancePlot_noLabels.png",dpi=200, bbox_inches='tight')

plot()
