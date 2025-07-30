#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:22:59 2023

@author: pp423
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import tikzplotlib

from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

eps = 0.5
N = 500
lam = 40
omega = 6
t = 300
run_no = 100

with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/relu_gamp_disc_eps{}_N{}_lam{}_om{}_t{}_{}runs.txt".format(eps, N, lam, omega, t, run_no), "r") as filehandle:
    filecontents = filehandle.readlines()

delta_domain = json.loads(filecontents[0])
iid_mse_array = json.loads(filecontents[1])
iid_mse_array_std = json.loads(filecontents[2])
sc_mse_array = json.loads(filecontents[3])
sc_mse_array_std = json.loads(filecontents[4])

with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/noiseless_relu_disc_potential_eps{}.txt".format(eps), "r") as filehandle:
    filecontents = filehandle.readlines()

pot_delta_array = json.loads(filecontents[0])
mmse_bound = json.loads(filecontents[1])
iid_bound = json.loads(filecontents[2])

with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/se_relu_eps{}_t{}_lam{}_om{}.txt".format(eps, t, lam, omega), "r") as filehandle:
    filecontents = filehandle.readlines()

delta_array = json.loads(filecontents[0])
iid_mse_pred_array = json.loads(filecontents[1])
sc_delta_array = json.loads(filecontents[2])
sc_mse_pred_array = json.loads(filecontents[3])

with open("/home/pp423/Documents/PhD/3rd Year/Text File Data/relu_scgamp_eps{}_dct_lam{}_om{}_N{}_t{}_{}runs.txt".format(eps, 200, 20, N, t, run_no), "r") as filehandle:
    filecontents = filehandle.readlines()

dct_delta_domain = json.loads(filecontents[0])
dct_sc_mse_array = json.loads(filecontents[1])
dct_sc_mse_array_std = json.loads(filecontents[2])

with open("/home/pp423/Documents/PhD/3rd Year/Text File Data/relu_scse_eps{}_lam{}_om{}_t{}.txt".format(eps, 200, 20, 500), "r") as filehandle:
    filecontents = filehandle.readlines()

se_delta_domain = json.loads(filecontents[0])
sc_se200_mse_array = json.loads(filecontents[1])

#MSE PLOTS
plt.figure()
plt.plot(pot_delta_array, mmse_bound, label = "Global Minimizer",color='orange')
plt.plot(pot_delta_array, iid_bound, label = "Largest Stat. Pt.",color='green')
plt.errorbar(delta_domain, iid_mse_array, yerr=iid_mse_array_std, label ="iid GAMP", fmt='o', color='dodgerblue',ecolor='lightblue', elinewidth=3, capsize=0, linestyle='none')
plt.plot(delta_array, iid_mse_pred_array, label="iid SE", color='dodgerblue', linestyle='dashed')
plt.errorbar(delta_domain, sc_mse_array, yerr=sc_mse_array_std, label =r"SC-GAMP, ($\omega=6, \Lambda=40$)", fmt='o', color='maroon',ecolor='mistyrose', elinewidth=3, capsize=0, linestyle='none')
plt.plot(sc_delta_array, sc_mse_pred_array, label=r"SC SE, ($\omega=6, \Lambda=40$)", color='maroon', linestyle='dashed')
plt.errorbar(dct_delta_domain, dct_sc_mse_array, yerr=dct_sc_mse_array_std, label =r"SC-GAMP, ($\omega=20, \Lambda=200$)", fmt='o', color='hotpink',ecolor='lightpink', elinewidth=3, capsize=0, linestyle='none')
plt.plot(se_delta_domain, sc_se200_mse_array, label=r"SC SE, ($\omega=20, \Lambda=200$)", color='hotpink', linestyle='dashed')


"""#MSE PLOTS
plt.figure()
plt.plot(pot_delta_array, mmse_bound, label = "Global Minimizer",color='orange')
plt.plot(pot_delta_array, iid_bound, label = "Largest Stat. Pt.",color='green')
plt.errorbar(delta_domain, sc_mse_array, yerr=sc_mse_array_std, label ="SC-GAMP", fmt='o', color='red',ecolor='mistyrose', elinewidth=3, capsize=0, linestyle='none')
plt.plot(sc_delta_array, sc_mse_pred_array, label="SC SE", color='red', linestyle='dashed')
plt.errorbar(delta_domain, iid_mse_array, yerr=iid_mse_array_std, label ="i.i.d. GAMP", fmt='o', color='blue',ecolor='lightblue', elinewidth=3, capsize=0, linestyle='none')
plt.plot(delta_array, iid_mse_pred_array, label="i.i.d. SE", color='blue', linestyle='dashed')"""

plt.xlabel(r'Sampling ratio $\delta=n/p$')
plt.ylabel('Mean Squared Error')
plt.ylim((-0.0269,0.9766))
plt.legend(loc="lower left")
plt.grid(alpha=0.4, linestyle="--")
plt.rc('axes', labelsize=16) #fontsize of the x and y labels
plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
plt.rc('ytick', labelsize=16) #fontsize of the y tick labels
plt.legend(fontsize=12)
#plt.title(r"ReLU, $\epsilon=${}, $N=${}, $\Lambda={}$, $\omega={}$, averaged over {} runs, Discrete".format(eps, N, lam, omega, run_no))
tikzplotlib.save('relu_plot.tex')