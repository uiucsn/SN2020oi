#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:48:21 2020

@author: lucaizzo
"""
import os
folderroot = '/Users/alexgagliano/Documents/Research/2020oi/data/spectra/all_20oi_spectra_final/csv'
os.chdir(folderroot)
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from astropy.convolution import convolve, Box1DKernel
import seaborn as sns

########
#SN2020oi_floyds_20200201_norm_ext_cor_total_0_133.csv
#SN2020oi_lris_20200216_norm_ext_cor_total_0_133.csv

#load observed spectrum
#w2 = np.genfromtxt('../SN2020oi-20200109-goodman.flm_mangled_ext_cor_total_0.18.flm', usecols=0) #need to change!
#f2 = np.genfromtxt('SN2020oi-20200109-goodman.flm_mangled_ext_cor_total_0.18.flm', usecols=1) #need to change!
w11 = np.genfromtxt('SN2020oi_floyds_20200116_norm_ext_cor_total_0_133.csv', usecols=0)
f11 = np.genfromtxt('SN2020oi_floyds_20200116_norm_ext_cor_total_0_133.csv', usecols=1)
w15 = np.genfromtxt('SN2020oi_floyds_20200120_norm_ext_cor_total_0_133.csv', usecols=0)
f15 = np.genfromtxt('SN2020oi_floyds_20200120_norm_ext_cor_total_0_133.csv', usecols=1)
w17 = np.genfromtxt('SN2020oi_floyds_20200122_norm_ext_cor_total_0_133.csv', usecols=0)
f17 = np.genfromtxt('SN2020oi_floyds_20200122_norm_ext_cor_total_0_133.csv', usecols=1)
w19 = np.genfromtxt('SN2020oi_floyds_20200124_norm_ext_cor_total_0_133.csv', usecols=0)
f19 = np.genfromtxt('SN2020oi_floyds_20200124_norm_ext_cor_total_0_133.csv', usecols=1)
w20 = np.genfromtxt('SN2020oi_lris_20200127_norm_ext_cor_total_0_133.csv', usecols=0)
f20 = np.genfromtxt('SN2020oi_lris_20200127_norm_ext_cor_total_0_133.csv', usecols=1)
#SN2020oi_floyds_20200131_norm_ext_cor_total_0_133.csv
w25 = np.genfromtxt('SN2020oi_goodman_20200201_norm_ext_cor_total_0_133.csv', usecols=0)
f25 = np.genfromtxt('SN2020oi_goodman_20200201_norm_ext_cor_total_0_133.csv', usecols=1)#corrected for the Angstrom
#w37 = np.genfromtxt('spectrum_2020_02_13.txt', usecols=0)
#f37 = np.genfromtxt('spectrum_2020_02_13.txt', usecols=1)


d = 16.8*3.08568e24
l2 = 4*np.pi*(d**2)*(f2)
l11 = 4*np.pi*(d**2)*(f11)
l15 = 4*np.pi*(d**2)*(f15)
l17 = 4*np.pi*(d**2)*(f17)
l19 = 4*np.pi*(d**2)*(f19)
l20 = 4*np.pi*(d**2)*(f20)
l25 = 4*np.pi*(d**2)*(f25)
l37 = 4*np.pi*(d**2)*(f37)

#smooth day 9 spectrum
l11 = convolve(l11, Box1DKernel(5))

#load 1994I spectra
w114I_day8 = np.genfromtxt('SN1994I_day8_KAST.txt', usecols=0)
f114I_day8 = np.genfromtxt('SN1994I_day8_KAST.txt', usecols=1)

w114I_day9 = np.genfromtxt('SN1994I_day9_FAST.txt', usecols=0)
f114I_day9 = np.genfromtxt('SN1994I_day9_FAST.txt', usecols=1)

w114I_day13 = np.genfromtxt('SN1994I_day13_Ekar.txt', usecols=0)
f114I_day13 = np.genfromtxt('SN1994I_day13_Ekar.txt', usecols=1)

w114I_day15 = np.genfromtxt('SN1994I_day15_MMT.txt', usecols=0)
f114I_day15 = np.genfromtxt('SN1994I_day15_MMT.txt', usecols=1)

w114I_day24 = np.genfromtxt('SN1994I_day24_KAST.txt', usecols=0)
f114I_day24 = np.genfromtxt('SN1994I_day24_KAST.txt', usecols=1)

#load 2004aw spectra
w04aw_day15 = np.genfromtxt('2004aw_Day15.dat', usecols=0)
f04aw_day15 = np.genfromtxt('2004aw_Day15.dat', usecols=1)

w04aw_day17 = np.genfromtxt('2004aw_Day17.dat', usecols=0)
f04aw_day17 = np.genfromtxt('2004aw_Day17.dat', usecols=1)

w04aw_day27 = np.genfromtxt('2004aw_Day27.dat', usecols=0)
f04aw_day27 = np.genfromtxt('2004aw_Day27.dat', usecols=1)

#luminosities
d94I = 8.82*3.08568e24
l114I_day8 = 4*np.pi*(d94I**2)*f114I_day8
l114I_day9 = 4*np.pi*(d94I**2)*f114I_day9
l114I_day13 = 4*np.pi*(d94I**2)*f114I_day13
l114I_day15 = 4*np.pi*(d94I**2)*f114I_day15
l114I_day24 = 4*np.pi*(d94I**2)*f114I_day24

d04aw = 100.2*3.08568e24
l04aw_day15 = 4*np.pi*(d04aw**2)*f04aw_day15
l04aw_day17 = 4*np.pi*(d04aw**2)*f04aw_day17
l04aw_day27 = 4*np.pi*(d04aw**2)*f04aw_day27

#smooth 2004aw spectra
l04aw_day15s = convolve(l04aw_day15, Box1DKernel(11))
l04aw_day17s = convolve(l04aw_day17, Box1DKernel(11))
l04aw_day27s = convolve(l04aw_day27, Box1DKernel(11))


#plotting
plt.subplots(1, 1, figsize=(12,16))
plt.title('SN2020oi spectral sequence', fontsize=17)
#plt.plot(w2,l2+1.55E39,'k',label='Day 2')
plt.plot(w11,l11+1.20E39,'k',label='Day 9')
plt.plot(w15,l15+0.9E39,'k',label='Day 13')
plt.plot(w17,l17+0.65E39,'k',label='Day 15')
plt.plot(w19,l19+0.45E39,'k',label='Day 17')
plt.plot(w20,l20+0.3E39,'k',label='Day 20')
plt.plot(w25,l25+0.15E39,'k',label='Day 25')
plt.plot(w37,l37,'k',label='Day 37')
#plot outputs
#plt.plot(wo,(lo+0.9E39)/1.1,'r',label='TARDIS')
#plt.plot(wo3FOE_CO21,(lo3FOE_CO21+0.9E39)/1.1,'b',label='TARDIS')
plt.xlim(3000,10500)
plt.ylim(-0.06E39,1.88E39)
plt.xlabel(r'Rest-frame wavelength ($\AA$)',fontsize=15)
plt.ylabel('Luminosity (erg/s)',fontsize=15)
plt.tick_params(axis='y', labelsize=14, labeltop='off')
plt.tick_params(axis='x', labelsize=14, labelbottom='off')
#plt.text(9700, 1.7E39, r"Day 2", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 1.45E39, r"Day 9", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 1.05E39, r"Day 13", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.8E39, r"Day 15", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.57E39, r"Day 17", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.4E39, r"Day 20", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.2E39, r"Day 25", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.08E39, r"Day 35", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
#plt.legend(fontsize=14)
plt.savefig('SN2020oi.pdf')
#plt.close()
#plt.show()

#plotting - check 1994I
plt.figure(figsize=(10,28))
plt.title('SN2020oi spectral sequence', fontsize=17)
#plt.plot(w2,l2+1.55E39,'k',label='Day 2')
plt.plot(w11/1.005,(l11+1.30E39)/1E39,'k',label='2020oi')
plt.plot(w15/1.005,(l15+0.7E39)/1E39,'k')
plt.plot(w17/1.005,(l17+0.35E39)/1E39,'k')
plt.plot(w25/1.005,(l25*1.5+0.05E39)/1E39,'k')
#plot 1994I
plt.plot(w114I_day8/1.001,(l114I_day8*3E-15 + 1.15E39)/1E39 ,'r',label='1994I')
plt.plot(w114I_day13/1.001,(l114I_day13*6 +0.65E39)/1E39,'r')
plt.plot(w114I_day15/1.001,(l114I_day15*2E-15 + 0.3E39)/1E39,'r')
plt.plot(w114I_day24/1.001,(l114I_day24*2E-15)/1E39 ,'r')
#plot 2004aw
#plt.plot(w04aw_day15/1.023,(l04aw_day15s +0.65E39)/1E39,'b',label='2004aw')
#plt.plot(w04aw_day17/1.023,(l04aw_day17s*1E-15 + 0.3E39)/1E39,'b')
#plt.plot(w04aw_day27/1.023,(l04aw_day27s*2E-15) ,'b')
#plot lines
plt.plot([3700, 3700], [1.75, 1.85], '-', lw=1.5, color='k', alpha=0.8)
plt.text(3700, 1.95, r'Ca II', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([4250, 4250], [1.85, 1.95], '-', lw=1.5, color='k', alpha=0.8)
plt.text(4250, 2.05, r'Mg II', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([4800, 4800], [1.75, 1.85], '-', lw=1.5, color='k', alpha=0.8)
plt.text(4800, 1.95, r'Fe II', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([6077, 6077], [1.75, 1.85], '-', lw=1.5, color='k', alpha=0.8)
plt.text(6077, 1.95, r'Si II', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([6320, 6320], [1.85, 1.95], '-', lw=1.5, color='k', alpha=0.8)
plt.text(6320, 2.05, r'C II', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([7033, 7033], [1.05, 1.15], '-', lw=1.5, color='k', alpha=0.8)
plt.text(7033, 1.25, r'C II', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([5630, 5630], [1.25, 1.35], '-', lw=1.5, color='k', alpha=0.8)
plt.text(5630, 1.45, r'Na I', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([7400, 7400], [1.65, 1.75], '-', lw=1.5, color='k', alpha=0.8)
plt.text(7400, 1.85, r'O I', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
plt.plot([8050, 8050], [1.65, 1.75], '-', lw=1.5, color='k', alpha=0.8)
plt.text(8050, 1.85, r'Ca II, O I', fontsize=11, fontweight='bold', color='k', alpha=0.9, ha='center')
#plt.plot(wo,(lo+0.9E39)/1.1,'r',label='TARDIS')
#plt.plot(wo3FOE_CO21,(lo3FOE_CO21+0.9E39)/1.1,'b',label='TARDIS')
plt.xlim(3000,10500)
plt.ylim(-0.06,2.18)
plt.xlabel(r'Rest-frame wavelength ($\AA$)',fontsize=15)
plt.ylabel('Relative flux',fontsize=15)
plt.tick_params(axis='y', labelsize=14, labeltop='off')
plt.tick_params(axis='x', labelsize=14, labelbottom='off')
#plt.text(9700, 1.7E39, r"Day 2", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 1.65, r"Day 11", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 1.05, r"Day 15", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.57, r"Day 17", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.2, r"Day 27", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.legend(fontsize=14)
plt.savefig('SN2020oi_vs_SN1994I.pdf')

#plotting - check 2004aw
plt.figure(figsize=(10,28))
plt.title('SN2020oi spectral sequence', fontsize=17)
plt.plot(w2,l2+1.55E39,'k')
plt.plot(w11/1.005,(l11+1.35E39)/1E39,'k',label='2020oi')
plt.plot(w15/1.005,(l15+0.9E39)/1E39,'k')
plt.plot(w17/1.005,(l17+0.5E39)/1E39,'k')
plt.plot(w25/1.005,(l25*1.5+0.15E39)/1E39,'k')
#plot 1994I
#plt.plot(w114I_day8/1.001,(l114I_day8*3E-15 + 1.15E39)/1E39 ,'r',label='1994I')
#plt.plot(w114I_day13/1.001,(l114I_day13*6 +0.65E39)/1E39,'r')
#plt.plot(w114I_day15/1.001,(l114I_day15*2E-15 + 0.3E39)/1E39,'r')
#plt.plot(w114I_day24/1.001,(l114I_day24*2E-15)/1E39 ,'r')
#plot 2004aw
plt.plot(w04aw_day15/1.023,(l04aw_day15s +0.65E39)/1E39,'b',label='2004aw')
plt.plot(w04aw_day17/1.023,(l04aw_day17s*1E-15 + 0.2E39)/1E39,'b')
plt.plot(w04aw_day27/1.023,(l04aw_day27s*3E-15)/1E25 ,'b')
#plot lines
plt.plot([3700, 3700], [1.75, 1.85], '-', lw=2, color='gray', alpha=0.8)
plt.text(3700, 1.95, r'Ca II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([4250, 4250], [1.85, 1.95], '-', lw=2, color='gray', alpha=0.8)
plt.text(4250, 2.05, r'Mg II, Fe II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([4800, 4800], [1.75, 1.85], '-', lw=2, color='gray', alpha=0.8)
plt.text(4800, 1.95, r'Fe II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([6077, 6077], [1.75, 1.85], '-', lw=2, color='gray', alpha=0.8)
plt.text(6077, 1.95, r'Si II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([6320, 6320], [1.85, 1.95], '-', lw=2, color='gray', alpha=0.8)
plt.text(6320, 2.05, r'C II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([7033, 7033], [1.2, 1.3], '-', lw=2, color='gray', alpha=0.8)
plt.text(7033, 1.37, r'C II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([7400, 7400], [1.65, 1.75], '-', lw=2, color='gray', alpha=0.8)
plt.text(7400, 1.85, r'O I', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
plt.plot([8030, 8030], [1.65, 1.75], '-', lw=2, color='gray', alpha=0.8)
plt.text(8030, 1.85, r'Ca II', fontsize=12, fontweight='bold', color='gray', alpha=0.9, ha='center')
#plt.plot(wo,(lo+0.9E39)/1.1,'r',label='TARDIS')
#plt.plot(wo3FOE_CO21,(lo3FOE_CO21+0.9E39)/1.1,'b',label='TARDIS')
plt.xlim(3000,10500)
plt.ylim(-0.06,2.18)
plt.xlabel(r'Rest-frame wavelength ($\AA$)',fontsize=15)
plt.ylabel('Relative flux',fontsize=15)
plt.tick_params(axis='y', labelsize=14, labeltop='off')
plt.tick_params(axis='x', labelsize=14, labelbottom='off')
#plt.text(9700, 1.7E39, r"Day 2", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 1.65, r"Day 11", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 1.05, r"Day 15", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.57, r"Day 17", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.text(9700, 0.2, r"Day 27", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
plt.legend(fontsize=14)
plt.savefig('SN2020oi_vs_SN2004aw.pdf')

#load TARDIS results
def plot_tardis():
    import matplotlib as mpl

    sns.set_context("talk",font_scale=2.5)
    wsim_day11 = np.genfromtxt('SN2020oi_sim_spec_Day11.txt', usecols=0, skip_header=0)
    fsim_day11 = np.genfromtxt('SN2020oi_sim_spec_Day11.txt', usecols=1, skip_header=0)

    wsim_day15 = np.genfromtxt('SN2020oi_sim_spec_Day15.txt', usecols=0, skip_header=0)
    fsim_day15 = np.genfromtxt('SN2020oi_sim_spec_Day15.txt', usecols=1, skip_header=0)

    wsim_day17 = np.genfromtxt('SN2020oi_sim_spec_Day17.txt', usecols=0, skip_header=0)
    fsim_day17 = np.genfromtxt('SN2020oi_sim_spec_Day17.txt', usecols=1, skip_header=0)

    wsim_day19 = np.genfromtxt('SN2020oi_sim_spec_Day19.txt', usecols=0, skip_header=0)
    fsim_day19 = np.genfromtxt('SN2020oi_sim_spec_Day19.txt', usecols=1, skip_header=0)

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


    plt.figure(figsize=(20,22))
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
    plt.plot(w11/1.005,(l11 + 1.15E39),'k',lw=2, zorder=100)
    plt.plot(w15/1.005,(l15 + 0.65E39),'k',lw=2, zorder=100)
    plt.plot(w17/1.005,(l17 + 0.25E39),'k',lw=2, zorder=100)
    plt.plot(w19/1.005,(l19),'k',lw=2, zorder=100)
    #plot TARDIS
    plt.plot(wsim_day11/1.005,(fsim_day11 + 1.15E39) ,'#00916E',lw=3)
    plt.plot(wsim_day15/1.005,(fsim_day15 +0.65E39),'#00916E',lw=3)
    plt.plot(wsim_day17/1.005,(fsim_day17 + 0.25E39),'#00916E',lw=3)
    plt.plot(wsim_day19/1.005,(fsim_day19),'#00916E',lw=3)



    plt.plot(w114I_day8/1.001,(l114I_day8*3E-15 + 1.15E39) ,c='#A165E2',label='1994I', alpha=0.8)
    plt.plot(w114I_day13/1.001,(l114I_day13*6 +0.65E39),c='#A165E2',alpha=0.8)
    plt.plot(w114I_day15/1.001,(l114I_day15*2E-15 + 0.3E39),c='#A165E2', alpha=0.8)
    #plot 2004aw
    #plt.plot(w04aw_day15/1.023,(l04aw_day15s +0.65E39)/1E39,'b',label='2004aw')
    #plt.plot(w04aw_day17/1.023,(l04aw_day17s*1E-15 + 0.3E39)/1E39,'b')
    #plt.plot(w04aw_day27/1.023,(l04aw_day27s*2E-15) ,'b')
    #plot lines
#    plt.plot([3700, 3700], [1.75*1E39, 1.85*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(3700, 1.95*1E39, r'Ca II', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
#    plt.plot([4250, 4250], [1.85*1E39, 1.95*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(4250, 2.05*1E39, r'Mg II, Fe II', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
#    plt.plot([4800, 4800], [1.75*1E39, 1.85*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(4800, 1.95*1E39, r'Fe II', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
#    plt.plot([6097, 6097], [1.05*1E39, 1.15*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(6097, 1.25*1E39, r'Si II', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
#    plt.plot([6320, 6320], [1.8*1E39, 1.9*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(6320, 2.0*1E39, r'C II', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
#    plt.plot([7533, 7533], [1.65*1E39, 1.65*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(7400, 1.85*1E39, r'O I', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
#    plt.plot([8030, 8030], [1.65*1E39, 1.75*1E39], '-', lw=2, color='gray', alpha=0.8)
#    plt.text(8030, 1.85*1E39, r'Ca II, O I', fontsize=18, fontweight='bold', color='gray', alpha=0.9, ha='center')
    #plt.plot(wo,(lo+0.9E39)/1.1,'r',label='TARDIS')
    #plt.plot(wo3FOE_CO21,(lo3FOE_CO21+0.9E39)/1.1,'b',label='TARDIS')
    plt.xlim(3000,10500)
    plt.ylim(-0.06*1E39,2.2*1E39)
    #plt.xlabel(r'Rest-frame wavelength (\AA)',fontsize=15)
    #plt.ylabel('Relative flux',fontsize=15)
    plt.tick_params(axis='y',labeltop='off')
    plt.tick_params(axis='x', labelbottom='off')
    #plt.text(9700, 1.7E39, r"Day 2", horizontalalignment='center', fontsize=15, color='black', alpha=0.9, fontweight='bold')
#    plt.text(9700, 1.55*1E39, r"Day 11", horizontalalignment='center', fontsize=26, color='gray')
#    plt.text(9700, 0.93*1E39, r"Day 15", horizontalalignment='center', fontsize=26, color='gray')
#    plt.text(9700, 0.50*1E39, r"Day 17", horizontalalignment='center', fontsize=26, color='gray')
#    plt.text(9700, 0.2*1E39, r"Day 27", horizontalalignment='center', fontsize=26, color='gray')
    plt.gca().yaxis.set_ticklabels([])
#    plt.legend(fontsize=14)
    plt.xlabel(r"Rest-Frame Wavelength (\AA)")
    plt.ylabel(r"Normalized $F_{\lambda}$ + Offset")
#    plt.savefig('/Users/alexgagliano/Documents/Research/2020oi/img/SN2020oi_TARDIS_1994I.png',dpi=200, bbox_inches='tight')
plot_tardis()
