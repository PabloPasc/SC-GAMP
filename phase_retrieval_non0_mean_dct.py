#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:27:02 2021

@author: pp423
"""
from scipy.integrate import quad
import numpy as np
#import scipy.stats, scipy.signal
import matplotlib.pyplot as plt
from numba import jit
#import numba as nb
#from NumbaQuadpack import quadpack_sig, dqags #Integration using numba (not scipy)
import produce_data, produce_data_dct

@jit(nopython=True)
def cond_exp_s_non0_mean(arg, s_sqrt, a, prob):
    #arg = np.float128(arg)
    denomin = prob*norm_pdf(arg - a*s_sqrt) + (1-prob)*norm_pdf(arg + a*s_sqrt)
    frac = (1/denomin)*(prob*a*norm_pdf(arg - a*s_sqrt) - (1-prob)*a*norm_pdf(arg + a*s_sqrt))
    #frac = np.float64(frac)
    return frac

@jit(nopython=True)
def mmse_integrand_non0_mean(y, s_sqrt, a, prob):
    f_y = prob*norm_pdf(y, a*s_sqrt, 1) + (1-prob)*norm_pdf(y, -a*s_sqrt, 1)
    integ = f_y*(cond_exp_s_non0_mean(y, s_sqrt, a, prob)**2)
    return integ


#Only true for exp_x2 = 1!!!!
def mmse_new_non0_mean(s_sqrt, a, prob):
    if a*s_sqrt < 25:
        integral = quad(mmse_integrand_non0_mean, -30, 30, args=(s_sqrt, a, prob))[0]
        mmse = a**2 - integral
    else:
        integral = quad(mmse_integrand_non0_mean, -a*s_sqrt - 5, -a*s_sqrt + 5, args =(s_sqrt, a, prob))[0] + \
        quad(mmse_integrand_non0_mean, a*s_sqrt - 5, a*s_sqrt + 5, args =(s_sqrt, a, prob))[0]
        mmse = a**2 - integral
    return mmse

@jit(nopython=True)
def expectation_p_g(p_r, tau_p, g_r):
    y = (p_r + np.sqrt(tau_p)*g_r)**2
    res = y*(1 - np.tanh((p_r*np.sqrt(y))/tau_p)**2)
    return res

@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

@jit(nopython=True)
def integral_1(g_r, p_r, tau_p):
    integrand = norm_pdf(g_r)*(expectation_p_g(p_r, tau_p, g_r))
    return integrand

#@jit(nopython=True)
def integral_2(p_r, tau_p, exp_zr2):
    var_p = exp_zr2 - tau_p
    integrand = norm_pdf(p_r, scale=np.sqrt(var_p))*(quad(integral_1, -20,20, args=(p_r, tau_p))[0])
    return integrand

#print(quad(integral_2, -30,30, args=(0.2, 1))[0])


@jit(nopython=True)
def bayes_eta_non0_mean(arg, sqrt_tau, a, prob):
    #print("arg = {}, tau = {}".format(arg, sqrt_tau))
    
    if sqrt_tau < 0.001:
        output = bayes_eta_non0_mean(arg, 0.001, a, prob)
    
    pdf_1 = norm_pdf((arg - a)/sqrt_tau)
    pdf_2 = norm_pdf((arg + a)/sqrt_tau)
    #arg = np.float128(arg)
    #tau = np.float128(tau) #Making one variable a 128-bit float is enough


    denomin = prob*pdf_1 + (1-prob)*pdf_2
    output = (1/denomin)*(prob*a*pdf_1 - (1-prob)*a*pdf_2)
    #output = np.float64(output)
    #print("output = {}".format(output))
    return output


#We create a function for the derivative of a Standard Normal
@jit(nopython=True)
def deriv_norm(x):
    output = - x * norm_pdf(x)
    return output

@jit(nopython=True)
def deriv_bayes_eta_non0_mean(arg, sqrt_tau, a, prob):
    
    if sqrt_tau < 0.001:
        output = deriv_bayes_eta_non0_mean(arg, 0.001, a, prob)
    
    pdf_1 = norm_pdf((arg - a)/sqrt_tau)
    pdf_2 = norm_pdf((arg + a)/sqrt_tau)
    
    num = 4*((a/sqrt_tau)**2)*prob*(1-prob)*pdf_1*pdf_2
    denomin = (prob*pdf_1 + (1-prob)*pdf_2)**2
    output = num/denomin
    #output = np.float64(output)
    
    return output

#print(bayes_eta(np.ones(5)*0.5, 1e-5, 0.5, 1/np.sqrt(0.5)))



#for r in range(R):
    #tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse((1/tau_q),))

#------IID GAUSSIAN DESIGN MATRIX---------

#STATE EVOLUTION
#@jit(nopython=True)
def se_iid_pr_exact(delta, a, prob, it):
    
    exp_x2 = (a**2)
    var_x = exp_x2 - (a*(2*prob-1))**2
    exp_z2 = exp_x2/delta
    
    tau_p_array = []
    tau_p_prev = 0
    
    #ALTERNATIVE INITIALISATION
    #tau_q = 1000
    #tau_p = (1/delta)*mmse_new(np.sqrt(1/tau_q),eps, a)
    
    #t=0:
    tau_p = (1/delta)*0.99*var_x
    
    tau_p_array.append(tau_p)
    
    tau_p_inv = (tau_p)**(-1)

    #Compute expectation of variance term first
    #data1 = np.array([tau_p, exp_z2], np.float64)
    #expec = dqags(integral2ptr, -30,30, data=data1)[0]
    expec = quad(integral_2, -30,30, args=(tau_p, exp_z2))[0]
    
    print(expec)
    scaled_expec = (tau_p_inv)*(1-(tau_p_inv*expec))
    #print(scaled_expec)
    
    tau_q = 1/(scaled_expec)
    
    print(tau_p)
    print(tau_q)
    
    for t in range(it-1):
        print(t)
        
        print(np.sqrt(1/tau_q))
        #print(np.sqrt(1/tau_q))
        tau_p = (1/delta)*mmse_new_non0_mean(np.sqrt(1/tau_q), a, prob)
        
        if tau_p == 0:
            break
        
        tau_p_inv = (tau_p)**(-1)
        print(tau_p_inv)
             
        expec = quad(integral_2, -30,30, args=(tau_p, exp_z2))[0]
        #data1 = np.array([tau_p, exp_z2], np.float64)
        #expec = dqags(integral2ptr, -30,30, data=data1)[0]
        
        if expec == 0:
            break
        print(expec)
        scaled_expec = (tau_p_inv)*(1-(tau_p_inv*expec))
        #print(scaled_expec)

        tau_q = 1/(scaled_expec)
        print(tau_q)
        if tau_q < 1e-6:
            break                  
        tau_p_array.append(0 + tau_p)
        
        print(((tau_p - tau_p_prev)**2)/tau_p)
        
        if ((tau_p - tau_p_prev)**2)/tau_p < 1e-9:
            break
        
        
        tau_p_prev = 0 + tau_p
    
    #Compute MSE Prediction
    mse_pred = delta*tau_p
    
    #Normalized Correlation Squared using x_hat
    nc_pred = (1 - mse_pred)
    
    #Normalized Correlation Squared wrt q
    #nc_pred = (exp_x2/tau_q)/(1 + (exp_x2/tau_q))
    
    return tau_p_array, mse_pred, nc_pred
"""
se_iid_array = []

prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))
print(se_iid_pr_exact(0.94, a, prob, 300))


delta_domain_small = np.linspace(0.1, 1.1, num=41)
for delta in delta_domain_small:
    se_iid_array.append(se_iid_pr_exact(delta, a, prob, 300)[1])

plt.plot(delta_domain_small, se_iid_array)
"""



@jit(nopython=True)
def g_out_pr(p, y, tau_p):
    #Lower-bound tau_p to avoid Div. by Zero error
    if np.abs(tau_p)<1e-9:
        tau_p = 1e-9
    #p = np.float128(p)
    sqrt_y = np.sqrt(y)
    output = (1/tau_p)*(sqrt_y*np.tanh((p*sqrt_y)/tau_p) - p)
    #output = np.float64(output)
    return output

@jit(nopython=True)
def neg_deriv_g_out_pr(p, y, tau_p):
    
    #Lower-bound tau_p to avoid Div. by Zero error
    if np.abs(tau_p)<1e-9:
        tau_p = 1e-9
    #tau_p = np.float128(tau_p)
    sqrt_y = np.sqrt(y)
    tau_p_inv = 1/tau_p
    output = tau_p_inv*(1 - y*tau_p_inv*(1-(np.tanh(p*sqrt_y*tau_p_inv)**2)))
    #output = np.float64(output)
    return output

#@jit(nopython=True)
def pr_gamp_iid_non0_mean_dct(m, n, y, t, a, prob, x_0, rand_seed, Ax=None, Ay=None):
    delta = m/n
    # Initialise
    p = np.zeros(m)
    tau_p_prev = 1000
    
    if (Ax is None) or (Ay is None):
        Ax, Ay = produce_data_dct.sub_dct_iid(m, n, rand_seed)
    
    exp_x2 = (a**2)
    var_x = exp_x2 - (a*(2*prob-1))**2
    exp_z2 = exp_x2/(delta)
    exp_x = a*(2*prob - 1)
    
    tau_p = var_x/delta*0.99
    
    
    g_in_q = np.ones(n)*exp_x
    
    p = Ax(g_in_q)
    
    
    #Set Initial tau_q to be large, positive to avoid nan error in 1st iteration (tau_p can give negative value for p very close to 0)
    tau_q = 1/np.average(neg_deriv_g_out_pr(p, y, tau_p))#1/((1/tau_p) - exp_z2/(tau_p**2) + (1/tau_p**2)*np.average(y*(np.tanh(p*(y**0.5)/tau_p)**2)))
    print(tau_q)
    # Calculate q
    q = g_in_q + tau_q*Ay(g_out_pr(p, y, tau_p))
    #print(tau_p, tau_p_prev)
    #print(tau_q)
    error_norm_array = []
    norm_correl_array = []
    
    for it in range(t):
        
        #Estimate of x
        x_hat = g_in_q
        print(x_hat)
        
        #Phase-retrieval error might be calculated differently
        mse_x_pos = (1/n)*(np.linalg.norm(x_hat - x_0)**2)
        mse_x_neg = (1/n)*(np.linalg.norm(-x_hat - x_0)**2)
        #print(mse_x_neg, mse_x_pos)
        mse = min(mse_x_pos, mse_x_neg)
        
        #ALT Measure of Performance: Normalised Correlation
        norm_correl = (np.dot(x_hat, x_0)/(np.linalg.norm(x_hat)*np.linalg.norm(x_0)))**2
        
        if np.isnan(mse) or np.isnan(norm_correl):
            break
        
        error_norm_array.append(mse)
        norm_correl_array.append(norm_correl)
        
        
        g_in_q = bayes_eta_non0_mean(q, np.sqrt(tau_q), a, prob)
        print("g_in_q", g_in_q)
        
        # Calculate tau_p - exact
        #tau_p = (1/delta)*mmse_new_non0_mean(np.sqrt(1/tau_q), a, prob)
        tau_p = (tau_q/m)*np.sum(deriv_bayes_eta_non0_mean(q, np.sqrt(tau_q), a, prob))
        print("Tau_p", tau_p)
        #print(tau_p, tau_p_prev)
        if tau_p<=0 and tau_p> -1e-8:
            break
        
        
        if np.isnan(tau_p):
            #print(norm_correl_array)
            break
        
        # Calculate p
        p = Ax(g_in_q)- tau_p*g_out_pr(p, y, tau_p_prev)
        #print("p", p )
        
        """expec = quad(integral_2, -30,30, args=(tau_p, exp_z2))[0]
        tau_p_inv = 1/tau_p
        #print(expec)
        scaled_expec = (tau_p_inv)*(1-(tau_p_inv*expec))
        #print(scaled_expec)

        tau_q = 1/(scaled_expec)
        
        """
        # Calculate tau_q
        tau_q = 1/np.average(neg_deriv_g_out_pr(p, y, tau_p))#1/((1/tau_p) - exp_z2/(tau_p**2) + (1/tau_p**2)*np.average(y*(np.tanh(p*(y**0.5)/tau_p)**2)))#
        print("Tau_q at t={}: ".format(it), tau_q)
        
        if tau_q<=1e-16 and tau_q > -1 or np.isnan(tau_q):
            break
        # Calculate q
        q = g_in_q + tau_q*Ay(g_out_pr(p, y, tau_p))
        
        #print(q)
        
        
        #print((tau_p - tau_p_prev)**2/tau_p**2)
        #Stopping criterion - Relative norm tolerance
        if (tau_p - tau_p_prev)**2/tau_p**2 < 1e-12 and it >= 1:
            break
        
        tau_p_prev = 0 + tau_p
    
    return error_norm_array, norm_correl_array, x_hat

"""
error_array = []
delta = 0.5
n = 20000
m = int(delta*n)
print("Delta: ", delta)
mse_runs = []
nc_runs = []
it = 0
run_no = 20
prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))

A_iid = produce_data.create_A_iid(m,n)
A_iid_T = np.transpose(A_iid)

x_0 = produce_data.create_x_0_non0_mean(n, a, prob)
print(np.average(x_0))
y = (np.dot(A_iid, x_0))**2


#error_norm_array, x_hat= cs_gamp_iid_new(A_iid, y, 300, eps, 1/np.sqrt(eps), x_0, exp_x=0)
error_norm_array, norm_correl_array, x_hat = pr_gamp_iid_non0_mean(A_iid, A_iid_T, y, 300, a, prob, x_0)
print(error_norm_array[-1])

plt.figure()
#tau_p_array, mse_pred, nc_pred = se_iid_pr_exact(delta, a, prob, 300)
#plt.plot([delta*i for i in tau_p_array], label="iid SE")
plt.plot(error_norm_array, label="iid GAMP")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.title("Phase Retrieval MSE Evolution, p={}, delta={}".format(prob, delta))





rand_seed = np.random.randint(0,3000)


prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))
n = 200000
x_0 = produce_data.create_x_0_non0_mean(n, a, prob)
delta = 1.1
m = int(delta*n)

Ax, Ay = produce_data_dct.sub_dct_iid(m, n, rand_seed)

y = Ax(x_0)**2

err, norm, x_hat = pr_gamp_iid_non0_mean_dct(m, n, y, 300, a, prob, x_0, rand_seed)
print(err,norm,x_hat)"""

#------SPATIALLY COUPLED DESIGN MATRIX-----------
    
#STATE EVOLUTION
#@jit(nopython=True)
def se_sc_pr_exact(W, delta_hat, a, prob, it):

    R, C = len(W), len(W[0])

    exp_x2 = (a**2)
    var_x = exp_x2 - (a*(2*prob-1))**2

    tau_p = np.ones(R)
    #tau_q = np.zeros(C)
    tau_q_inv = np.zeros(C)
    
    exp_zr2 = (exp_x2/delta_hat)*np.sum(W[:,:],axis=1)
    
    tau_p = np.zeros(R)
    scaled_expec = np.zeros(R)
    mmse_vec = np.zeros(C)
    
    tau_p_prev = np.zeros(R)
    
    tau_p_array = []
    mmse_vec_array = []
    mse_pred_array = []
    nc_pred_array = []
    
    #t=0:
    mmse_vec = np.ones(C)*0.99*var_x
    mmse_vec_array.append(np.zeros(C)+mmse_vec)
    
    for r in range(R):
        tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse_vec)

    tau_p_inv = (tau_p)**(-1)

    #Compute expectation of variance term first, only depends on r, fixed for any c
    for r in range(R):
        #print(r)
        expec = quad(integral_2, -30,30, args=(tau_p[r], exp_zr2[r]))[0]
        #data1 = np.array([tau_p[r], exp_zr2[r]], np.float64)
        #expec = dqags(integral2ptr, -30,30, data=data1)[0]
        
        scaled_expec[r] = (tau_p_inv[r])*(1-(tau_p_inv[r]*expec))

    for c in range(C):
        tau_q_inv[c] = (np.dot(W[:,c], scaled_expec))
    
    #print(tau_p)
    #print(tau_q)
    
    for t in range(it-1):
        print(t)
        
        for c in range(C):
            #print(np.sqrt(1/tau_q[c]))
            mmse_vec[c]= mmse_new_non0_mean(np.sqrt(tau_q_inv[c]), a, prob)
            #print(mmse_vec)
        
        for r in range(R):
            tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse_vec)
            if tau_p[r]<=0 and tau_p[r]> -1e-12:
                tau_p[r]+= 1e-10
        
        print(tau_p)
        tau_p_inv = (tau_p)**(-1)
        
        #Compute expectation of variance term first, only depends on r, fixed for any c
        for r in range(R):
            #print(r)
            expec = quad(integral_2, -30,30, args=(tau_p[r], exp_zr2[r]))[0]
            #data1 = np.array([tau_p[r], exp_zr2[r]], np.float64)
            #expec = dqags(integral2ptr, -30,30, data=data1)[0]
            
            scaled_expec[r] = (tau_p_inv[r])*(1-(tau_p_inv[r]*expec))

        for c in range(C):
            tau_q_inv[c] = (np.dot(W[:,c], scaled_expec))
        
        #print(tau_p)
        
        tau_p_array.append(np.zeros(R) + tau_p)
        
        print("Convergence:",np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2))
        
        if (np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2)) < 1e-12:
                break
        
        
        tau_p_prev = np.zeros(R) + tau_p
    
        #Compute MSE Prediction
        mse_pred = (1/C)*np.sum(mmse_vec)
        mmse_vec_array.append(np.zeros(C)+mmse_vec)
        mse_pred_array.append(mse_pred)
    
        #Normalized Correlation Squared using x_hat
        nc_pred_array.append(1 - mse_pred)
    
    return tau_p_array, mmse_vec_array, mse_pred_array, nc_pred_array

"""delta_domain_sc = np.linspace(0.1, 1.1, num=21)
prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))
lam=40
omega=6
W = produce_data.create_W(lam, omega)

se_sc_array = []

for delta in delta_domain_sc:
    delta_hat = delta*lam/(lam + omega - 1)
    se_sc_array.append(se_sc_pr_exact(W, delta_hat, a, prob, 300)[1])"""

"""prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))
delta = 0.7
lam=40
omega=6
W = produce_data.create_W(lam, omega)
delta_hat = delta*lam/(lam + omega - 1)
tau_p_array, mmse_vec_array, _, _ = se_sc_pr_exact(W, delta_hat, a, prob, 300)

#WAVE PROPAGATION BEHAVIOUR OF SPATIAL COUPLING
plt.figure()
plt.xlabel(r"Column block index $c$")
plt.ylabel(r"Mean Squared Error")
#plt.title(r"Discrete Prior, Spatially Coupled, $\Lambda=${}, $\omega=${}, $\epsilon=${}, $\delta=${}".format(lam, omega, eps, delta))
l = int(len(mmse_vec_array)/5 + 1)
for i in range(l):
    plt.plot(mmse_vec_array[int(5*i)], label="t={}".format(5*i))
plt.plot(mmse_vec_array[-1], label="t={}".format(len(mmse_vec_array)-1))
plt.legend(bbox_to_anchor=(0.95, 0.85), framealpha=1, fontsize=12)
plt.grid(alpha=0.4, linestyle="--")
plt.rc('axes', labelsize=12) #fontsize of the x and y labels
plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
"""

#@jit(nopython=True)
def pr_gamp_sc_non0_mean_dct(W, m, n, y, x_0, a, prob, t, rand_seed, Ab=None, Az=None):
    R, C = len(W), len(W[0])
    M = int(m/R)#
    N = int(n/C)
    delta_hat = M/N
    
    exp_x2 = (a**2)
    var_x = exp_x2 - (a*(2*prob-1))**2
    exp_x = a*(2*prob - 1)
    
    #Define vectors
    tau_p = np.zeros(R)
    tau_p_prev = np.zeros(R)
    tau_q = np.zeros(C)
    deriv_average = np.zeros(C)
    denoised_q = np.zeros(n)
    neg_dgout_av = np.zeros(R)
    g_out_tau_p = np.zeros(m)
    g_out_p = np.zeros(m)
    tau_q_A_gout = np.zeros(n)
    
    z_tau_p = np.zeros(m)
    
    if (Ab is None) or (Az is None):
        Ab, Az = produce_data_dct.sc_dct(W, m, n, rand_seed)
    
    # Initialisation
    #t=0:
    mmse_vec = np.ones(C)*0.99*var_x
    
    for r in range(R):
        tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse_vec)

    
    exp_zr2 = (exp_x2/delta_hat)*np.sum(W[:,:],axis=1)

    g_in_q = np.ones(n)*exp_x
    
    p = Ab(g_in_q)

    tau_p_prev = np.zeros(R) + tau_p    
    
    for r in range(R):
        neg_dgout_av[r]=np.average(neg_deriv_g_out_pr(p[M*r:(M*r+M)],y[M*r:(M*r+M)], tau_p[r]))
    
    # Calculate tau_q
    for c in range(C):
        tau_q[c] = 1/(np.dot(W[:,c], neg_dgout_av))
    print(tau_q)
    tau_q = np.ones(C)*100
    # Calculate q
    #print(len(p[M*r:(M*r+M)]), len(y[M*r:(M*r+M)]), (tau_p[r]))
    for r in range(R):
        g_out_p[M*r: (M*r+M)] = g_out_pr(p[M*r:(M*r+M)],y[M*r:(M*r+M)], tau_p[r])
    
    #print(c, n, len(np.dot(A_T, g_out_p)))
    for c in range(C):
        tau_q_A_gout[N*c: (N*c + N)] = tau_q[c]*Az(g_out_p)[N*c:(N*c+N)]
        
    q = g_in_q + tau_q_A_gout
    #print(tau_p, tau_p_prev)
    #print(tau_q)
    error_norm_array = []
    norm_correl_array = []
    
    
    for it in range(t):
        
        for c in range(C):
            deriv_average[c] = np.maximum(tau_q[c]*np.average(deriv_bayes_eta_non0_mean(q[N*c:(N*c + N)], np.sqrt(tau_q[c]), a, prob)), 1e-6)
            denoised_q[N*c:(N*c+N)] = bayes_eta_non0_mean(q[N*c:(N*c + N)], np.sqrt(tau_q[c]), a, prob)
        
        print(deriv_average)
        for r in range(R):
            tau_p_new= (1/delta_hat)*np.dot(W[r,:], deriv_average)
            if tau_p_new < 1e-16 and tau_p_new > -1e-2:
                tau_p_new = 1e-10
            tau_p[r] = tau_p_new
            
            
        for r in range(R):
            g_out_tau_p[M*r:(M*r+M)] = tau_p[r]*g_out_pr(p[M*r:(M*r+M)],y[M*r:(M*r+M)], tau_p_prev[r])
            
        p = Ab(denoised_q) - g_out_tau_p
        
        print("Tau_p at t={}: ".format(it), tau_p)
        for r in range(R):
            #print(r, tau_p)
            neg_dgout_av[r]=np.average(neg_deriv_g_out_pr(p[M*r:(M*r+M)],y[M*r:(M*r+M)], tau_p[r]))
    
        # Calculate tau_q
        for c in range(C):
            tau_q_new = 1/(np.dot(W[:,c], neg_dgout_av))
            if tau_q_new > 0:
                tau_q[c] = tau_q_new#1/((1/tau_p) - exp_z2/(tau_p**2) + (1/tau_p**2)*np.average(y*(np.tanh(p*(y**0.5)/tau_p)**2)))#
        
        print("Tau_q at t={}: ".format(it), tau_q)
        
        if (np.amax(tau_q)<=1e-8 and np.amin(tau_q) > -1) or np.isnan(tau_q.any()):
            break
        
        for r in range(R):
            g_out_p[M*r:(M*r+M)] = g_out_pr(p[M*r:(M*r+M)],y[M*r:(M*r+M)], tau_p[r])
            
        for c in range(C):
            tau_q_A_gout[N*c: (N*c + N)] = tau_q[c]*Az(g_out_p)[N*c:(N*c+N)]
        
        # Calculate q
        q = denoised_q + tau_q_A_gout
        
        print(np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2))
        #print((tau_p - tau_p_prev)**2/tau_p**2)
        #Stopping criterion - Relative norm tolerance
        if (np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2)) < 1e-6 or np.linalg.norm(tau_p_prev, 2)<1e-12:
                break
        
        tau_p_prev = np.zeros(R) + tau_p
        
        x_hat = denoised_q
        
        #Phase-retrieval error might be calculated differently
        mse_x_pos = (1/n)*(np.linalg.norm(x_hat - x_0)**2)
        mse_x_neg = (1/n)*(np.linalg.norm(-x_hat - x_0)**2)
        #print(mse_x_neg, mse_x_pos)
        mse = min(mse_x_pos, mse_x_neg)
        
        #ALT Measure of Performance: Normalised Correlation
        norm_correl = (np.dot(x_hat, x_0)/(np.linalg.norm(x_hat)*np.linalg.norm(x_0)))**2
        print(mse, norm_correl)
        if np.isnan(mse) or np.isnan(norm_correl):
            break
        
        error_norm_array.append(mse)
        norm_correl_array.append(norm_correl)
        
    return error_norm_array, norm_correl_array, x_hat


error_array = []


mse_runs = []
nc_runs = []
it = 0
run_no = 1
prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))

rand_seed = np.random.randint(0,1000)

lam=320
omega= 30
W = produce_data.create_W(lam, omega)
delta_hat = 0.65
delta = delta_hat*(lam + omega - 1)/lam
print("Delta: ", delta)
N = 1000
n = int(N*lam)
M = int(N*delta_hat)
m = int(delta*n)
x_0 = produce_data.create_x_0_non0_mean(n, a, prob)
Ab, Az = produce_data_dct.sc_dct(W, m, n, rand_seed)
y = Ab(x_0)**2

error_norm_array, nc_array, x_hat = pr_gamp_sc_non0_mean_dct(W, m, n, y, x_0, a, prob, 300, rand_seed)
print("Final Error: ", error_norm_array[-1])
"""
plt.figure()
tau_p_array, mse_pred_array, nc_pred_array = se_sc_pr_exact(W, delta_hat, a, prob, 300)
plt.plot(mse_pred_array, label="SC SE")
plt.plot(error_norm_array, label="SC GAMP")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.title("Phase Retrieval MSE Evolution, p={}, delta={}".format(prob, delta))






prob = 0.45
a = np.sqrt(1/(1-(2*prob-1)**2))
lam=40
omega=6
W = produce_data.create_W(lam, omega)
delta = 0.7
delta_hat = delta*lam/(lam + omega - 1)

N = 500
n = 20000
m = int(delta*n)
x_0 = produce_data.create_x_0_non0_mean(n, a, prob)
A = produce_data.create_A_sc(W, N, delta_hat)
A_T = np.transpose(A)
y = (np.dot(A, x_0))**2

error_norm_array, nc_array, x_hat = pr_gamp_sc_non0_mean(W, A, A_T, y, x_0, a, prob, 300)
plt.figure()
plt.plot(error_norm_array)"""
