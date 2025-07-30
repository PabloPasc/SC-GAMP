#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:22:59 2023

@author: pp423
"""
import numpy as np
import matplotlib.pyplot as plt
import json

eps = 1
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
sc_delta_array = json.loads(filecontents[3])
sc_mse_pred_array = json.loads(filecontents[4])

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
plt.plot(pot_delta_array, iid_bound, label = "Largest Minimizer",color='green')
plt.errorbar(delta_domain, iid_mse_array, yerr=iid_mse_array_std, label ="GAMP, iid Gaussian", fmt='o', color='blue',ecolor='lightblue', elinewidth=3, capsize=0, linestyle='none')
plt.plot(delta_array, iid_mse_pred_array, label="SE, iid Gaussian", color='blue', linestyle='dashed')
plt.errorbar(delta_domain, sc_mse_array, yerr=sc_mse_array_std, label =r"GAMP, SC Gaussian, $\Lambda=40, \omega=6$", fmt='o', color='red',ecolor='mistyrose', elinewidth=3, capsize=0, linestyle='none')
plt.plot(sc_delta_array, sc_mse_pred_array, label=r"SE, SC Gaussian, $\Lambda=40, \omega=6$", color='red', linestyle='dashed')
plt.errorbar(dct_delta_domain, dct_sc_mse_array, yerr=dct_sc_mse_array_std, label =r"GAMP, SC Gaussian, $\Lambda=200, \omega=20$", fmt='o', color='purple',ecolor='thistle', elinewidth=3, capsize=0, linestyle='none')
plt.plot(se_delta_domain, sc_se200_mse_array, label=r"SE, SC Gaussian, $\Lambda=200, \omega=20$", color='purple', linestyle='dashed')

plt.xlabel(r'$\delta=m/n$')
plt.ylabel('Mean Squared Error')
plt.legend(loc="upper right")
plt.grid(alpha=0.4, linestyle="--")
#plt.title(r"ReLU, $\epsilon=${}, $N=${}, $\Lambda={}$, $\omega={}$, averaged over {} runs, Discrete".format(eps, N, lam, omega, run_no))
