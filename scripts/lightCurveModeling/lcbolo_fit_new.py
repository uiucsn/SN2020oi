#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:29:10 2020

@author: lucaizzo
"""

import os
os.chdir('/Users/alexgagliano/Documents/Research/2020oi/scripts')
import numpy as np
from scipy import special
from matplotlib import pyplot as plt
import pandas as pd

t0 = 58854.2
Lbol = pd.read_csv("/Users/alexgagliano/Documents/Research/2020oi/scripts/superbol/SBO_input/superbol_output_2020oi/logL_bb_2020oi_AUBgVriz.txt", delimiter='\t', header=None, names=['MJD', 'logL', 'logLerr'])
day = Lbol['MJD'] - t0
lum = 10**Lbol['logL']
peak = 58864.866

np.where(lum == np.nanmax(lum))
day[68]

np.nanmax(lum)
Lbol,peak = 3.62e+42 erg/s
tpeak = 10.67 days
plt.axvline(10.66)
plt.plot(day, lum)

lumerr = 2.303*lum*Lbol['logLerr']
#some constants
tauNi = 8.77
tauCo = 111.3

tauNis = 8.77*86400
tauCos = 111.3*86400

epsNi = 3.9E10
epsCo = 6.78E9

MSun = 2E33

#def variables/functions
def y(tau):
    return tau/(2*tauNi)

def s(tau):
    return tau * (tauCo - tauNi) / (2 * tauNi * tauCo)

def A(t, tau):
    return np.exp(-(y(tau)**2)) * (np.sqrt(np.pi) * y(tau) * special.erfi(t/tau - y(tau)) + np.sqrt(np.pi) * y(tau) * special.erfi(y(tau)) + np.exp((t/tau - y(tau))**2) - np.exp(y(tau)**2))

def B(t, tau):
    return np.exp(-(s(tau) - y(tau))**2) * (np.sqrt(np.pi) * (s(tau) - y(tau)) * (special.erfi(s(tau) - y(tau)) - special.erfi(s(tau) + t/tau - y(tau))) + np.exp((s(tau) + t/tau - y(tau))**2) - np.exp((s(tau) - y(tau))**2))

def L(t, tau, MNi):
    return ((MNi*MSun) * np.exp(-(t/tau)**2) * ((epsNi - epsCo) * A(t, tau) +  B(t, tau)))

#try to fit
from scipy.optimize import curve_fit
R_stat = []
for i in np.arange(0, 5, 0.1):
    popt, pcov = curve_fit(L, day+i, lum, bounds=(0, [20, 1.5]))
    R = np.sum((L(day+i, *popt) - lum)**2)/len(lum)
    R_stat.append(R)
    R_best = np.min(R_stat)
    t0 = np.arange(0, 5, 0.1)[np.where(R_stat == R_best)[0][0]]
    popt, pcov = curve_fit(L, day+t0, lum, bounds=(0, [20, 1.5]))

#plot
xx = np.arange(0, 50, 1)
fig, ax1 = plt.subplots(figsize=(9,5))
plt.plot(xx+t0, L(xx+t0, *popt), label=r'fit: t$_d$=%5.3f, M$_{Ni}$=%5.3f' % tuple(popt))
plt.scatter(day+t0, lum, color='black')
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.set_ylabel(r'Luminosity $(erg/s)$', fontsize=15)
ax1.set_xlabel(r'Days from explosion - MJD %5.2f' % np.round(day[0] - t0, 2), fontsize=15)
plt.legend()
plt.show()

#estimate ejected mass and kin energy - parameters
kopt = 0.07#cm2/g
beta = 13.8
c = 3E10#cm/s
#velocity from Si II
#vSi = 1150000000#cm/s
vSi = 12744e5#cm/s

dvSi = 100000000
#ejected mass
Mej = vSi * ((popt[0]*86400)**2) * (beta*c/(2*kopt))#g
dMej = np.sqrt((dvSi/vSi)**2 + (pcov[0][0]/popt[0])**2) * Mej

Ekin = (3/10) * Mej * vSi**2#erg
dEkin = np.sqrt((dMej/Mej)**2 + 2*(dvSi/vSi)**2) * Ekin

print()
print('Final results')
print()
print(r'diffusion time t$_d$ (days) : '+str(np.round(popt[0], 2))+r' $\pm$ '+str(np.round(np.sqrt(pcov[0][0]),2))+'\n')
print()
print(r'Nickel mass M$_{Ni}$ (MSun) : '+str(np.round(popt[1], 2))+r' $\pm$ '+str(np.round(np.sqrt(pcov[1][1]),2))+'\n')
print()
print(r't$_0$ (MJD) : '+str(np.round(day[0] - t0, 2))+'\n')
print()
print(r'Ejected mass M$_{tot}$ (MSun) : '+str(np.round(Mej/MSun, 3))+r' $\pm$ '+str(np.round(dMej/MSun, 3))+'\n')
print()
print(r'Kinetic energy (FOE) : '+str(np.round(Ekin/1E51, 2))+r' $\pm$ '+str(np.round(dEkin/1E51, 3))+'\n')
print()

 Lbol['MJD'].values[0] - 0.6


#Lbol['MJD'].values[0]-1.97

58854.03+0.14-0.16

#Final results
#diffusion time t$_d$ (days) : 10.12 $\pm$ 0.18
#Nickel mass M$_{Ni}$ (MSun) : 0.18 $\pm$ 0.0
#t$_0$ (MJD) : -1.8
#Ejected mass M$_{tot}$ (MSun) : 1.44 $\pm$ 0.113
#Kinetic energy (FOE) : 1.4 $\pm$ 0.191

#let's use now the Khasami & Kasen 2019 formulation
#we will use the equation 41 given in that paper, which provides the relation between
#the peak time and the Nickel mass

#def L_KK(tp, beta):


#however, we must determine the peak time - for this we will use the heating function
#with contributes from Nickel and Cobalt decay, defined here

def L_heat(t, K):
    return (K * MSun) * ((epsNi - epsCo)*np.exp(-t/tauNis) + epsCo*np.exp(-t/tauCos))

#try to fit
from scipy.optimize import curve_fit
R_stat = []
days = day*86400
for i in np.arange(0, 15, 0.1)*86400:
    popt, pcov = curve_fit(L_heat, days+i, lum)
    R = np.sum((L_heat(days+i, *popt) - lum)**2)/len(lum)
    R_stat.append(R)
    R_best = np.min(R_stat)
    time_array = np.arange(0, 15, 0.1)*86400
    t0 = time_array[np.where(R_stat == R_best)[0][0]]
    popt, pcov = curve_fit(L_heat, days+t0, lum)

#plot
xx = np.arange(0, 50, 1)*86400
fig, ax1 = plt.subplots(figsize=(9,5))
plt.plot(xx+t0, L_heat(xx+t0, *popt), label=r'fit: M$_{Ni}$=%5.3f' % tuple(popt))
#plt.plot(xx +t0, L_heat(xx, 0.1))
plt.scatter(days+t0, lum, color='black')
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.set_ylabel(r'Luminosity $(erg/s)$', fontsize=15)
ax1.set_xlabel(r'Days from explosion - MJD %5.2f' % np.round(days[0] - t0, 2), fontsize=15)
plt.legend()
plt.show()
