#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:22:59 2023

@author: pp423
"""

import matplotlib.pyplot as plt
import json

N = 500
lam = 40
omega = 6
prob = 0.6
run_no = 100
t = 500

with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/pr_gamp_non0_mean_N{}_lam{}_om{}_prob{}_{}runs.txt".format(N, lam, omega, prob, run_no), "r") as filehandle:
    filecontents = filehandle.readlines()

delta_domain = json.loads(filecontents[0])
iid_mse_array = json.loads(filecontents[1])
iid_mse_array_std = json.loads(filecontents[2])
sc_mse_array = json.loads(filecontents[3])
sc_mse_array_std = json.loads(filecontents[4])

with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/pr_pot_non0_mean_prob{}.txt".format(prob), "r") as filehandle:
    filecontents = filehandle.readlines()

pot_delta_array = json.loads(filecontents[0])
mmse_bound = json.loads(filecontents[1])
iid_bound = json.loads(filecontents[2])

with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/se_pr_non0_mean_prob{}_lam{}_om{}_t{}.txt".format(prob, lam, omega, t), "r") as filehandle:
    filecontents = filehandle.readlines()

sc_delta_array = json.loads(filecontents[0])
sc_mse_pred_array = json.loads(filecontents[1])
delta_array = json.loads(filecontents[2])
iid_mse_pred_array = json.loads(filecontents[3])

#MSE PLOTS
plt.figure()
plt.plot(pot_delta_array, mmse_bound, label = "Global Minimizer",color='orange')
plt.plot(pot_delta_array, iid_bound, label = "Largest Stat. Pt.",color='green')
plt.errorbar(delta_domain, iid_mse_array, yerr=iid_mse_array_std, label =r"iid GAMP", fmt='o', color='dodgerblue',ecolor='lightblue', elinewidth=3, capsize=0, linestyle='none')
plt.plot(delta_array, iid_mse_pred_array, label=r"iid SE", color='dodgerblue', linestyle='dashed')
plt.errorbar(delta_domain, sc_mse_array, yerr=sc_mse_array_std, label =r"SC-GAMP, $\omega=6, \Lambda=40$", fmt='o', color='maroon',ecolor='mistyrose', elinewidth=3, capsize=0, linestyle='none')
plt.plot(sc_delta_array, sc_mse_pred_array, label=r"SC SE, $\omega=6, \Lambda=40$", color='maroon', linestyle='dashed')


plt.ylim((-0.021,1.09766))
plt.xlabel(r'$\delta=m/n$')
plt.ylabel('Mean Squared Error')
plt.legend(loc="upper right")
plt.grid(alpha=0.4, linestyle="--")
plt.rc('axes', labelsize=16) #fontsize of the x and y labels
plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
plt.rc('ytick', labelsize=16) #fontsize of the y tick labels
plt.legend(fontsize=12)
#plt.title(r"ReLU, $\epsilon=${}, $N=${}, $\Lambda={}$, $\omega={}$, averaged over {} runs, Discrete".format(eps, N, lam, omega, run_no))
