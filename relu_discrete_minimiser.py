#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:32:28 2022

@author: pp423
"""
from scipy.integrate import quad
import numpy as np
import scipy.stats, scipy.signal
import math
import matplotlib.pyplot as plt
from numba import jit
import numba as nb
from NumbaQuadpack import quadpack_sig, dqags #Integration using numba (not scipy)
import json
#import produce_data

@jit(nopython=True)
def cdf(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0


@jit(nopython=True)
def expectation_p_g(p_r, tau_p, g_r):
    """if tau_p < 1e-20:
        tau_p = 1e-20"""
    y = max(p_r + np.sqrt(tau_p)*g_r, 0)
    if y > 0:
        res = 1
    elif y == 0:    
        p_sqrt_taup = p_r/np.sqrt(tau_p)
        pdf_p_taup = norm_pdf(p_sqrt_taup)
        cdf_negp_taup = cdf(-p_sqrt_taup)
        num = -p_sqrt_taup*pdf_p_taup*cdf_negp_taup + pdf_p_taup**2
        denomin = (cdf_negp_taup)**2
        if denomin == 0:
            print(p_r, tau_p, g_r, y, pdf_p_taup, cdf_negp_taup)
            print("num", num, "denomin", denomin)
            """if denomin < 1e-20:
                denomin += 1e-15"""
        res = num/denomin
    return res



@jit(nopython=True)
def integral_1(g_r, p_r, tau_p):
    #print("g, p, tau_p", g_r, p_r, tau_p)
    integrand = norm_pdf(g_r)*(expectation_p_g(p_r, tau_p, g_r))
    return integrand

#@jit(nopython=True)
def integral_2(p_r, tau_p, exp_zr2):
    var_p = exp_zr2 - tau_p
    integrand = norm_pdf(p_r, scale=np.sqrt(var_p))*(quad(integral_1, -7,7, args=(p_r, tau_p))[0])
    return integrand



@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

@nb.cfunc(quadpack_sig)   
def diff_ent_int(y, data_):
    data2 = nb.carray(data_, (2,)) #data2 = [s_sqrt, eps, a]
    f_y = (data2[1]/2)*norm_pdf(y, data2[2]*data2[0], 1) + (data2[1]/2)*norm_pdf(y, -data2[2]*data2[0], 1) + (1-data2[1])*norm_pdf(y)
    ent = f_y * np.log(f_y)
    return ent
diff_ent_intptr = diff_ent_int.address

@jit(nopython=True)
def mutual_inf_exact(s_sqrt, eps, a):
    data = np.array([s_sqrt, eps, a], np.float64)
    if a*s_sqrt < 25:
        integ = dqags(diff_ent_intptr, -30,30, data=data)[0]
    elif eps < 1:
        print("Flag - truncation of bounds")
        integ = dqags(diff_ent_intptr, -10,10, data=data)[0] + dqags(diff_ent_intptr, -a*s_sqrt - 5, -a*s_sqrt + 5, data=data)[0] + \
        dqags(diff_ent_intptr, a*s_sqrt - 5, a*s_sqrt + 5, data=data)[0]
    elif eps == 1:
        integ = dqags(diff_ent_intptr, -a*s_sqrt - 5, -a*s_sqrt + 5, data=data)[0] + \
        dqags(diff_ent_intptr, a*s_sqrt - 5, a*s_sqrt + 5, data=data)[0]
    mutu = -integ - 0.5*np.log(2*np.pi*np.exp(1))
    #print(integ, 0.5*np.log(2*np.pi*np.exp(1)))
    return mutu
"""
@jit(nopython=True)
def expec_var_pr(exp_x2, x, delta):
    exp_z2 = exp_x2/delta
    data1 = np.array([(x/delta), exp_z2], np.float64)
    var = dqags(integral2ptr, -30,30, data=data1)[0]
    outp = (x/delta)*(1-var)
    return outp #quad(integral_2, -30, 30, args=((x/delta), exp_z2))[0]

@nb.cfunc(quadpack_sig)
def scalar_pot_integrand(z, data_):
    data2 = nb.carray(data_, (2,)) #data2 = [exp_x2, delta]
    integrand = -(1/z)*((data2[1]/z)*expec_var_pr(data2[0], z, data2[1]))
    return integrand
scalar_pot_intptr = scalar_pot_integrand.address

@jit(nopython=True)
def scalar_pot_cs(x, delta, eps, a, exp_x2=1):
    data = np.array([exp_x2, delta], np.float64)
    #print((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta)))
    #u_s = delta*((delta/x)*expec_var_pr(exp_x2, x, delta) - 1) + delta*(quad(scalar_pot_integrand, 0.00001, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    #ALT: Get rid of ln 0
    u_s = delta*((delta/x)*expec_var_pr(exp_x2, x, delta) - 1) + delta*np.log(x)+delta*(dqags(scalar_pot_intptr, 0.001, x, data=data)[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    
    return u_s

@jit(nopython=True)    
def diff_ent_int(y, s_sqrt, eps, a):
    f_y = (1-eps)*norm_pdf(y) + (eps/2)*norm_pdf(y, a*s_sqrt, 1) + (eps/2)*norm_pdf(y, -a*s_sqrt, 1)
    ent = f_y * np.log(f_y)
    return ent

def mutual_inf_exact(s_sqrt, eps, a):
    if a*s_sqrt < 25:
        integ = quad(diff_ent_int, -30, 30, args=(s_sqrt,eps, a))[0]
    elif eps < 1:
        print("Flag - truncation of bounds")
        integ = quad(diff_ent_int, -10, 10, args =(s_sqrt, eps, a))[0] + quad(diff_ent_int, -a*s_sqrt - 5, -a*s_sqrt + 5, args =(s_sqrt, eps, a))[0] + \
        quad(diff_ent_int, a*s_sqrt - 5, a*s_sqrt + 5, args =(s_sqrt, eps, a))[0]
    elif eps == 1:
        integ = quad(diff_ent_int, -a*s_sqrt - 5, -a*s_sqrt + 5, args =(s_sqrt, eps, a))[0] + \
        quad(diff_ent_int, a*s_sqrt - 5, a*s_sqrt + 5, args =(s_sqrt, eps, a))[0]
    mutu = -integ - 0.5*np.log(2*np.pi*np.exp(1))
    #print(integ, 0.5*np.log(2*np.pi*np.exp(1)))
    return mutu
"""
def expec_var_relu(exp_x2, x, delta):
    exp_z2 = exp_x2/delta
    expec = quad(integral_2, -30, 30, args=((x/delta), exp_z2))[0]
    var = (x/delta)*(1-expec)
    return var

def scalar_pot_integrand(z, exp_x2, delta):
    integrand = -(1/z)*((delta/z)*expec_var_relu(exp_x2, z, delta))
    return integrand

def scalar_pot_relu(x, delta, eps, a, exp_x2=1):
    #print((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta)))
    #u_s = delta*((delta/x)*expec_var_pr(exp_x2, x, delta) - 1) + delta*(quad(scalar_pot_integrand, 0.00001, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    #ALT: Get rid of ln 0
    u_s = delta*((delta/x)*expec_var_relu(exp_x2, x, delta) - 1) + delta*np.log(x)+delta*(quad(scalar_pot_integrand, 0.0005, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_relu(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    
    return u_s

"""
eps = 0.5
y_domain = np.arange(0,100,0.1)
x_domain = []
for y in y_domain:
    x_domain.append(mmse_new(np.sqrt(1-y), eps, 1/np.sqrt(eps)))
plt.figure()
plt.plot(y_domain, x_domain)"""

"""
#Check g is strictly increasing
dom = np.arange(0.01,1,0.001)
delta = 4.2
g_x = []
for x in dom:
    print(x)
    g_x.append(1000 - (delta/x)*(1-(delta/x)*expec_var_pr(1, x, delta)))
plt.figure()    
plt.plot(dom, g_x)
plt.xlabel(r'$x$')
plt.ylabel(r'$g(x)$')
plt.title(r"$g(x)$ for $\delta$={}".format(delta))


eps = 1
y_domain = []
delta = 0.7
x_domain = np.arange(0.001,1.04,0.04)
#sigma_2 = 0.01
#sigma_2_inv = sigma_2**(-1)

for x in x_domain:
    #summand = sigma_2 + (x/delta)/(1 - (delta/x)*expec_var_pr(1, x, delta))
    print(x)
    #y = sigma_2_inv - summand**(-1)
    y = scalar_pot_pr(x, delta, eps, 1/np.sqrt(eps))
    y_domain.append(y)
plt.figure()    
plt.plot(x_domain, y_domain)"""

def relu_bound_disc(delta, x_domain, eps, a):
    
    #start_time = time.time()
    
    scalar_pot_arr = np.zeros(len(x_domain))
    for i in range(len(x_domain)):
        scalar_pot_arr[i] = scalar_pot_relu(x_domain[i], delta, eps, a)
        print(i)
    scalar_pot_arr_ext = np.concatenate([scalar_pot_arr, [100]])
    print(scalar_pot_arr)
    #plt.plot(x_domain, scalar_pot_arr)
    sp = scipy.signal.argrelmin(scalar_pot_arr_ext)[0]
    print(sp)
    sp_pot=[]
    final_value = scalar_pot_arr[-1]
    start_value = scalar_pot_arr[0]
    
    if sp.size != 0:
        largest = x_domain[sp[-1]]
        for it in sp:
            sp_pot.append(scalar_pot_arr[it])
        if np.min(sp_pot) < start_value:
            if np.min(sp_pot)<final_value:
                minimizer = x_domain[sp[np.argmin(sp_pot)]]
            else:
                minimizer = x_domain[-1]
        else:
            if final_value < start_value:
                minimizer = x_domain[-1]
            else:
                minimizer = 0
    else:
        if final_value < start_value:
            minimizer = x_domain[-1]
        else:
            minimizer = 0
        largest = minimizer
    
    
    mse_bound = minimizer#*((lam + omega)/lam)
    iid_bound = largest#*((lam + omega)/lam)
    #print("Minimiser of Potential Method: --- %s seconds ---" % (time.time() - start_time))
    return mse_bound, iid_bound, scalar_pot_arr

eps = 0.5
delta_domain = [0.4,0.8]
delta_domain_new = [0.35,0.65]#[0.2,0.3,0.5,0.6,0.7]
mse_bound_array = []
iid_bound_array = []


mse_bound, iid_bound, _ = relu_bound_disc(0.35, np.linspace(0.001,0.8,num=81),eps,1/np.sqrt(eps))
mse_bound_array.append(mse_bound)
iid_bound_array.append(iid_bound)

mse_bound, iid_bound, _ = relu_bound_disc(0.65, np.linspace(0.001,0.5,num=51),eps,1/np.sqrt(eps))
mse_bound_array.append(mse_bound)
iid_bound_array.append(iid_bound)




#delta_domain_new = [2.02, 2.04, 2.06, 2.08, 2.1, 2.12, 2.14, 2.16, 2.18, 2.42, 2.44, 2.46, 2.48, 2.5, 2.52, 2.54, 2.56, 2.58, 2.8]
for delta in delta_domain_new:
    print("Delta: ", delta)
    mse_bound, iid_bound, _ = relu_bound_disc(delta, np.linspace(0.001,0.995,num=101),eps,1/np.sqrt(eps))
    mse_bound_array.append(mse_bound)
    iid_bound_array.append(iid_bound)
"""    
delta_domain_ext = np.concatenate([delta_domain_ext, delta_domain_new])
delta_domain_ext = np.sort(delta_domain_ext)
mse_bound_array1 = np.sort(mse_bound_array)[::-1]
iid_bound_array1 = np.sort(iid_bound_array)[::-1]"""


"""with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/noiseless_relu_disc_potential_eps{}.txt".format(eps), "r") as filehandle:
    filecontents = filehandle.readlines()
delta_domain_ext = json.loads(filecontents[0])
mse_bound_array = json.loads(filecontents[1])
iid_bound_array = json.loads(filecontents[2])"""




plt.figure()
plt.xlabel(r"$\delta$")
plt.ylabel("Mean Squared Error")
plt.title(r"Global and Largest Minimizer of Potential Function, ReLU, Discrete Prior, $\epsilon=${}".format(eps))
plt.plot(delta_domain_ext, mse_bound_array, label="MMSE bound")
plt.plot(delta_domain_ext, iid_bound_array, label="iid bound")

"""with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/noiseless_relu_disc_potential_eps{}.txt".format(eps), "w") as output:
    output.write(str(delta_domain_ext)+'\n')
    output.write(str(mse_bound_array1)+'\n')
    output.write(str(iid_bound_array1)+'\n')"""