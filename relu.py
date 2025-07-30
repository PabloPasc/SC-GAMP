#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:27:02 2021

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
import produce_data

@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

@jit(nopython=True)
def cond_exp_s(arg, s_sqrt, eps, a):
    #arg = np.float128(arg)
    denomin = (eps/2)*norm_pdf(arg - a*s_sqrt) + (eps/2)*norm_pdf(arg + a*s_sqrt) + \
        (1 - eps)*norm_pdf(arg)
    frac = (1/denomin)*((eps/2)*a*norm_pdf(arg - a*s_sqrt) - (eps/2)*a*norm_pdf(arg + a*s_sqrt))
    #frac = np.float64(frac)
    return frac

@jit(nopython=True)
def mmse_integrand(y, s_sqrt, eps, a):
    f_y = (eps/2)*norm_pdf(y, a*s_sqrt, 1) + (eps/2)*norm_pdf(y, -a*s_sqrt, 1) + (1-eps)*norm_pdf(y)
    integ = f_y*(cond_exp_s(y, s_sqrt,eps, a)**2)
    return integ

#Only true for exp_x2 = 1!!!!
def mmse_new(s_sqrt, eps, a):
    if a*s_sqrt < 25:
        integral = quad(mmse_integrand, -30, 30, args=(s_sqrt,eps, a))[0]
        mmse = 1 - integral
    elif eps < 1:
        integral = quad(mmse_integrand, -20, 20, args =(s_sqrt, eps, a))[0] + quad(mmse_integrand, -a*s_sqrt - 5, -a*s_sqrt + 5, args =(s_sqrt, eps, a))[0] + \
        quad(mmse_integrand, a*s_sqrt - 5, a*s_sqrt + 5, args =(s_sqrt, eps, a))[0]
        mmse = 1 - integral
    else:
        integral = quad(mmse_integrand, -a*s_sqrt - 5, -a*s_sqrt + 5, args =(s_sqrt, eps, a))[0] + \
        quad(mmse_integrand, a*s_sqrt - 5, a*s_sqrt + 5, args =(s_sqrt, eps, a))[0]
        mmse = 1 - integral
    return mmse


@nb.cfunc(quadpack_sig)
def mmse_integ(y, data_):
    data2 = nb.carray(data_, (2,)) #data2 = [s_sqrt, eps, a]
    f_y = (data2[1]/2)*norm_pdf(y, data2[2]*data2[0], 1) + (data2[1]/2)*norm_pdf(y, -data2[2]*data2[0], 1) + (1-data2[1])*norm_pdf(y)
    integ = f_y*(cond_exp_s(y, data2[0],data2[1], data2[2])**2)
    return integ
mmse_integptr = mmse_integ.address


#Only true for exp_x2 = 1!!!!
@jit(nopython=True)
def mmse_numba(s_sqrt, eps, a):
    data = np.array([s_sqrt, eps, a], np.float64)
    if a*s_sqrt < 25:
        integral = dqags(mmse_integptr, -30,30, data=data)[0]
        mmse = 1 - integral
    elif eps < 1:
        integral = dqags(mmse_integptr, -20,20, data=data)[0] + dqags(mmse_integptr, -a*s_sqrt - 5, -a*s_sqrt + 5, data=data)[0] + \
        dqags(mmse_integptr, a*s_sqrt - 5, a*s_sqrt + 5, data=data)[0]
        mmse = 1 - integral
    else:
        integral = dqags(mmse_integptr, -a*s_sqrt - 5, -a*s_sqrt + 5, data=data)[0] + \
        dqags(mmse_integptr, a*s_sqrt - 5, a*s_sqrt + 5, data=data)[0]
        mmse = 1 - integral
    return mmse
    
@jit(nopython=True)
def cdf_scalar(x):
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

@jit(nopython=True)
def cdf_vector(x):
    output = np.zeros(len(x))
    for i in range(len(x)):
        output[i]=(1.0 + math.erf(x[i] / np.sqrt(2.0))) / 2.0
    return output

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
        cdf_negp_taup = cdf_scalar(-p_sqrt_taup)
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
    integrand = norm_pdf(p_r, scale=np.sqrt(var_p))*(quad(integral_1, -5,5, args=(p_r, tau_p))[0])
    return integrand



"""
#IMPLEMENTATION WITH NUMBA QUAD PACK
@nb.cfunc(quadpack_sig)
def integ_1(g_r, data_):
    data = nb.carray(data_, (2,))
    integrand = norm_pdf(g_r)*(expectation_p_g(data[0], data[1], g_r))#data[0]=p_r, data[1]=tau_p
    return integrand
integral1ptr = integ_1.address

@nb.cfunc(quadpack_sig)
def integ_2(p_r, data_):
    data1 = nb.carray(data_, (2,))
    var_p = data1[1] - data1[0] #data1[0]=tau_p, data1[1]=exp_zr2
    data = np.array([p_r, data1[0]], np.float64)
    integrand = norm_pdf(p_r, scale=np.sqrt(var_p))*(dqags(integral1ptr, -40,40, data=data)[0])
    return integrand
integral2ptr = integ_2.address"""


@jit(nopython=True)
def bayes_eta(arg, tau, eps, a):
    #arg = np.float128(arg)
    #tau = np.float128(tau) #Making one variable a 128-bit float is enough
    if tau < 0.001:
        output = bayes_eta(arg, 0.001, eps, a)
    else:
        denomin = (eps/2)*norm_pdf((arg + a)/tau) + (eps/2)*norm_pdf((arg - a)/tau) + \
        (1 - eps)*norm_pdf(arg/tau)
        output = (1/denomin)*((eps/2)*a*norm_pdf((arg - a)/tau) - (eps/2)*a*norm_pdf((arg + a)/tau))
    #output = np.float64(output)
    
    return output

#We create a function for the derivative of a Standard Normal
@jit(nopython=True)
def deriv_norm(x):
    output = - x * norm_pdf(x)
    return output

@jit(nopython=True)
def deriv_bayes_eta(arg, tau, eps, a):

    if tau < 0.001:
        #output = np.ones(len(arg))*1e-6
        output = deriv_bayes_eta(arg, 0.001, eps, a)
    else:
        u = (eps/2)*a*norm_pdf((arg - a)/tau) - (eps/2)*a*norm_pdf((arg + a)/tau)
        v = (eps/2)*norm_pdf((arg + a)/tau) + (eps/2)*norm_pdf((arg - a)/tau) + \
        (1 - eps)*norm_pdf(arg/tau)
        if v.all()==0:
            v = np.ones(len(v))*1e-9
        
        du = (1/tau)*((eps/2)*a*deriv_norm((arg - a)/tau) - (eps/2)*a*deriv_norm((arg + a)/tau))
        dv = (1/tau)*((eps/2)*deriv_norm((arg + a)/tau) + (eps/2)*deriv_norm((arg - a)/tau) + (1 - eps)*deriv_norm(arg/tau))
        
        output = (du*v - u*dv)/(v**2)
        
    """if output.all()==0:
            output = np.ones(len(output))*1e-9
    if np.isnan(output).any():
        print(output)
        for i in np.where(np.isnan(output)):
            output[i] = 1e-10"""
        
    #output = np.float64(output)
    
    return output

#print(deriv_bayes_eta(np.ones(5)*0.5, 1e-5, 0.5, 1/np.sqrt(0.5)))


"""
def expectation_p_g(p_r, tau_p, g_r):
    y = (p_r + np.sqrt(tau_p)*g_r)**2
    res = y*(1 - np.tanh((p_r*np.sqrt(y))/tau_p)**2)
    return res
    
def integral_1(g_r, p_r, tau_p):
    integrand = scipy.stats.norm.pdf(g_r)*(expectation_p_g(p_r, tau_p, g_r))
    return integrand

def integral_2(p_r, tau_p, exp_zr2):
    var_p = exp_zr2 - tau_p
    integrand = scipy.stats.norm.pdf(p_r, loc=0, scale=np.sqrt(var_p))*(quad(integral_1, -30,30, args=(p_r, tau_p))[0])
    return integrand"""


#for r in range(R):
    #tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse((1/tau_q),))

#------IID GAUSSIAN DESIGN MATRIX---------

#STATE EVOLUTION
#@jit(nopython=True)
def se_iid_relu_exact(delta, eps, a, it):
    
    exp_x2 = eps*(a**2)
    exp_z2 = exp_x2/delta
    
    tau_p_array = []
    tau_p_prev = 0
    
    #ALTERNATIVE INITIALISATION
    #tau_q = 1000
    #tau_p = (1/delta)*mmse_new(np.sqrt(1/tau_q),eps, a)
    
    #t=0:
    tau_p = (1/delta)*0.99
    
    tau_p_array.append(tau_p)
    
    tau_p_inv = (tau_p)**(-1)

    #Compute expectation of variance term first
    #data1 = np.array([tau_p, exp_z2], np.float64)
    #expec = dqags(integral2ptr, -30,30, data=data1)[0]
    expec = quad(integral_2, -60,60, args=(tau_p, exp_z2))[0]
    
    print(expec)
    scaled_expec = (tau_p_inv)*(expec)
    #print(scaled_expec)
    
    tau_q_inv = (scaled_expec)
    
    print(tau_p)
    #print(tau_q)
    
    for t in range(it-1):
        print(t)
        
        print(np.sqrt(tau_q_inv))
        #print(np.sqrt(1/tau_q))
        tau_p = (1/delta)*mmse_new(np.sqrt(tau_q_inv),eps, a)
        #print("Tau_p  ", tau_p)
        tau_p_array.append(0 + tau_p)
        if tau_p < 1e-3:
            break
        
        tau_p_inv = (tau_p)**(-1)
        print(tau_p_inv)
        
        #var_p = exp_z2 - tau_p
        expec = quad(integral_2, -60,60, args=(tau_p, exp_z2))[0] #+ 1e-9
        #data1 = np.array([tau_p, exp_z2], np.float64)
        #expec = dqags(integral2ptr, -30,30, data=data1)[0]
        
        if expec == 0:
            break
        print(expec)
        scaled_expec = (tau_p_inv)*(expec)
        #print(scaled_expec)

        tau_q_inv = (scaled_expec)
        print(tau_q_inv)
        if tau_q_inv > 1e9:
            break                  
        
        
        print(((tau_p - tau_p_prev)**2)/tau_p)
        
        if ((tau_p - tau_p_prev)**2)/tau_p < 1e-9:
            break
        
        
        tau_p_prev = 0 + tau_p
    
    #Compute MSE Prediction
    mse_pred = delta*tau_p_array[-1]
    
    #Normalized Correlation Squared using x_hat
    nc_pred = (1 - mse_pred)
    
    #Normalized Correlation Squared wrt q
    #nc_pred = (exp_x2/tau_q)/(1 + (exp_x2/tau_q))
    
    return tau_p_array, mse_pred, nc_pred

def se_iid_relu_exact_tau0(delta, eps, a, it, tau_0=None):
    
    exp_x2 = eps*(a**2)
    exp_z2 = exp_x2/delta
    
    tau_p_array = []
    tau_p_prev = 0
    
    #ALTERNATIVE INITIALISATION
    #tau_q = 1000
    #tau_p = (1/delta)*mmse_new(np.sqrt(1/tau_q),eps, a)
    
    #t=0:
    if tau_0 is None:
        tau_0 = (1/delta)*0.99
        
    tau_p = tau_0
    
    tau_p_array.append(tau_p)
    
    tau_p_inv = (tau_p)**(-1)

    #Compute expectation of variance term first
    #data1 = np.array([tau_p, exp_z2], np.float64)
    #expec = dqags(integral2ptr, -30,30, data=data1)[0]
    expec = quad(integral_2, -60,60, args=(tau_p, exp_z2))[0]
    
    print(expec)
    scaled_expec = (tau_p_inv)*(expec)
    #print(scaled_expec)
    
    tau_q_inv = (scaled_expec)
    
    print(tau_p)
    #print(tau_q)
    
    for t in range(it-1):
        print(t)
        
        print(np.sqrt(tau_q_inv))
        #print(np.sqrt(1/tau_q))
        tau_p = (1/delta)*mmse_new(np.sqrt(tau_q_inv),eps, a)
        #print("Tau_p  ", tau_p)
        tau_p_array.append(0 + tau_p)
        if tau_p < 1e-3:
            break
        
        tau_p_inv = (tau_p)**(-1)
        print(tau_p_inv)
        
        #var_p = exp_z2 - tau_p
        expec = quad(integral_2, -60,60, args=(tau_p, exp_z2))[0] #+ 1e-9
        #data1 = np.array([tau_p, exp_z2], np.float64)
        #expec = dqags(integral2ptr, -30,30, data=data1)[0]
        
        if expec == 0:
            break
        print(expec)
        scaled_expec = (tau_p_inv)*(expec)
        #print(scaled_expec)

        tau_q_inv = (scaled_expec)
        print(tau_q_inv)
        if tau_q_inv > 1e9:
            break                  
        
        
        print(((tau_p - tau_p_prev)**2)/tau_p)
        
        if ((tau_p - tau_p_prev)**2)/tau_p < 1e-9:
            break
        
        
        tau_p_prev = 0 + tau_p
    
    #Compute MSE Prediction
    mse_pred = delta*tau_p_array[-1]
    
    #Normalized Correlation Squared using x_hat
    nc_pred = (1 - mse_pred)
    
    #Normalized Correlation Squared wrt q
    #nc_pred = (exp_x2/tau_q)/(1 + (exp_x2/tau_q))
    
    return tau_p_array, mse_pred, nc_pred


#print(se_iid_relu_exact(0.45, 0.5, 1/np.sqrt(0.5), 100))
"""
nc_pred_array = []
mse_pred_array = []
delta_array = np.linspace(0.1,1,num=91)
#delta_array_ext = np.sort(np.concatenate([delta_array, [2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.5, 2.51]]))
eps = 1
for delta in delta_array:
    print("Delta: ", delta)
    tau_p_array, mse_pred, nc_pred = se_iid_relu_exact(delta, eps, 1/np.sqrt(eps), 300)
    nc_pred_array.append(nc_pred)
    mse_pred_array.append(mse_pred)

plt.figure()
plt.plot(delta_array, mse_pred_array, marker="o", label="iid State Evolution")
plt.xlabel(r"$\delta$")
plt.ylabel("Normalized Correlation Squared")
plt.title(r"iid Gaussian State Evolution, ReLU, Discrete Prior, $\epsilon=${}".format(eps))
"""
@jit(nopython=True)
def g_out_relu_y0(p, y, tau_p):
    #Lower-bound tau_p to avoid Div. by Zero error
    """if np.abs(tau_p)<1e-12:
        tau_p = 1e-12"""
    #p = np.float128(p)
    p_sqrt_taup = p/np.sqrt(tau_p)
    pdf_p_taup = norm_pdf(p_sqrt_taup)
    cdf_negp_taup = cdf_vector(-p_sqrt_taup)
    
    denomin = np.sqrt(tau_p)*cdf_negp_taup
    output = - pdf_p_taup/denomin
    #output = np.float64(output)
    return output

def g_out_relu_ypos(p, y, tau_p):
    #Lower-bound tau_p to avoid Div. by Zero error
    """if np.abs(tau_p)<1e-12:
        tau_p = 1e-12"""
    #p = np.float128(p)
    output = (y-p)/tau_p
    #output = np.float64(output)
    return output

@jit(nopython=True)
def neg_deriv_g_out_relu_y0(p, y, tau_p):
    
    #Lower-bound tau_p to avoid Div. by Zero error
    """if np.abs(tau_p)<1e-12:
        tau_p = 1e-12"""
    #tau_p = np.float128(tau_p)
    p_sqrt_taup = p/np.sqrt(tau_p)
    pdf_p_taup = norm_pdf(p_sqrt_taup)
    cdf_negp_taup = cdf_vector(-p_sqrt_taup)
    num = p_sqrt_taup*pdf_p_taup*(-cdf_negp_taup) + pdf_p_taup**2
    denomin = tau_p*(cdf_negp_taup)**2
    output = num/denomin
    return output

@jit(nopython=True)
def neg_deriv_g_out_relu_ypos(p, y, tau_p):
    
    #Lower-bound tau_p to avoid Div. by Zero error
    """if np.abs(tau_p)<1e-12:
        tau_p = 1e-12"""
    #tau_p = np.float128(tau_p)
    output = (1/tau_p)*np.ones(len(p))

    return output

"""def cs_gamp_iid_new(A, y, t, eps, a, x_0, exp_x=0):
    error_norm_array = []
    m, n = len(A), len(A[0])
    A_T = np.transpose(A)
    delta = m/n
    
    exp_x2 = eps*(a**2)
    
    tau_p = (1/delta)*exp_x2

    tau_p_prev = 0 + tau_p
    
    x_hat = np.ones(n)*exp_x
    
    p = np.dot(A, x_hat)
    
    tau_q = (np.average(neg_deriv_g_out_cs(p, y, tau_p)))**(-1)
    
    q = x_hat + tau_q*np.dot(A_T, g_out_cs(p, y, tau_p))
    
    for iteration in range(t):
       
        tau_p = (tau_q/m)*np.sum(deriv_bayes_eta(q, np.sqrt(tau_q), eps, a))
       
        x_hat = bayes_eta(q, np.sqrt(tau_q), eps, a)
       
        p = np.dot(A, x_hat) - tau_p*g_out_cs(p, y, tau_p_prev)
        
        tau_q = (np.average(neg_deriv_g_out_cs(p, y, tau_p)))**(-1)
        print(tau_q)
        if tau_q == 0:
            break
        q = x_hat + tau_q*np.dot(A_T, g_out_cs(p, y, tau_p))
       
        tau_p_prev = 0 + tau_p
       
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
    
    
    return error_norm_array, x_hat"""

#1-bit Compressed Sensing does not require Spectral Initialization
def relu_gamp_iid(A, A_T, y, t, eps, a, x_0, exp_x=0):
    m, n = len(A), len(A[0])
    delta = m/n
    # Initialise
    p = np.zeros(m)
    
    exp_x2 = eps*(a**2)
    exp_z2 = exp_x2/(delta)
    
    
    #Find positions of 0s in y
    non_zero_seeds = np.array(np.where(y)[0])
    zero_seeds = np.array(np.where(y==0)[0])
    
    #tau_p = (1/delta)*0.99
    
    #expec = quad(integral_2, -30,30, args=(tau_p, exp_z2))[0]
    #tau_p_inv = 1/tau_p
    #print(expec)
    #scaled_expec = (tau_p_inv)*(1-(tau_p_inv*expec))
    #print(scaled_expec)
     
    tau_p = (1/delta)*exp_x2
    
    tau_p_prev = 0 + tau_p
    
    g_in_q = np.ones(n)*exp_x
    
    p = np.dot(A, g_in_q)
    
    neg_deriv_array = np.zeros(m)
    neg_deriv_array[non_zero_seeds] = neg_deriv_g_out_relu_ypos(p[non_zero_seeds], y[non_zero_seeds], tau_p)
    neg_deriv_array[zero_seeds]= neg_deriv_g_out_relu_y0(p[zero_seeds], y[zero_seeds], tau_p)
    tau_q = 1/np.average(neg_deriv_array)#1/((1/tau_p) - exp_z2/(tau_p**2) + (1/tau_p**2)*np.average(y*(np.tanh(p*(y**0.5)/tau_p)**2)))
    print(tau_q)
    # Calculate q
    g_out_array = np.zeros(m)
    g_out_array[non_zero_seeds]=g_out_relu_ypos(p[non_zero_seeds], y[non_zero_seeds], tau_p)
    g_out_array[zero_seeds]=g_out_relu_y0(p[zero_seeds], y[zero_seeds], tau_p)
    q = g_in_q + tau_q*np.dot(A_T, g_out_array)
    #print(tau_p, tau_p_prev)
    #print(tau_q)
    error_norm_array = []
    norm_correl_array = []
    
    for it in range(t):
        
        #Estimate of x
        x_hat = bayes_eta(q, np.sqrt(tau_q), eps, a)
        print(x_hat)
        
        #Phase-retrieval error might be calculated differently
        mse_x_pos = (1/n)*(np.linalg.norm(x_hat - x_0)**2)
        mse_x_neg = (1/n)*(np.linalg.norm(-x_hat - x_0)**2)
        #print(mse_x_neg, mse_x_pos)
        mse = min(mse_x_pos, mse_x_neg)
        
        #ALT Measure of Performance: Normalised Correlation
        norm_correl = (np.dot(x_hat, x_0)/(np.linalg.norm(x_hat)*np.linalg.norm(x_0)))**2
        print("MSE: ", mse)
        if np.isnan(mse) or np.isnan(norm_correl):
            break
        
        error_norm_array.append(mse)
        norm_correl_array.append(norm_correl)
        
        
        g_in_q = bayes_eta(q, np.sqrt(tau_q), eps, a)
        # Calculate tau_p - exact
        #tau_p = (1/delta)*mmse_new(np.sqrt(1/tau_q),eps, a)
        print(q, np.sqrt(tau_q))
        print(deriv_bayes_eta(q, np.sqrt(tau_q), eps, a))
        tau_p = (tau_q/m)*np.sum(deriv_bayes_eta(q, np.sqrt(tau_q), eps, a))
        print("Tau_p", tau_p)
        #print(tau_p, tau_p_prev)
        if tau_p<=0 and tau_p> -1e-8:
            break
        
        
        """if tau_p < 1e-3*exp_z2 or np.isnan(tau_p):
            #print(norm_correl_array)
            break"""
        
        # Calculate p
        p = np.dot(A, g_in_q)- tau_p*g_out_array
        #print("p", p )
        """
        expec = quad(integral_2, -30,30, args=(tau_p, exp_z2))[0]
        tau_p_inv = 1/tau_p
        #print(expec)
        scaled_expec = (tau_p_inv)*(1-(tau_p_inv*expec))
        #print(scaled_expec)

        tau_q = 1/(scaled_expec)"""
        
        
        # Calculate tau_q
        neg_deriv_array[non_zero_seeds] = neg_deriv_g_out_relu_ypos(p[non_zero_seeds], y[non_zero_seeds], tau_p)
        neg_deriv_array[zero_seeds]= neg_deriv_g_out_relu_y0(p[zero_seeds], y[zero_seeds], tau_p)
        tau_q = 1/np.average( neg_deriv_array)#1/((1/tau_p) - exp_z2/(tau_p**2) + (1/tau_p**2)*np.average(y*(np.tanh(p*(y**0.5)/tau_p)**2)))#
        print("Tau_q at t={}: ".format(it), tau_q)
        
        if tau_q<=1e-9 and tau_q > -1 or np.isnan(tau_q):
            break
        
        
        # Calculate q
        g_out_array[non_zero_seeds]=g_out_relu_ypos(p[non_zero_seeds], y[non_zero_seeds], tau_p)
        g_out_array[zero_seeds]=g_out_relu_y0(p[zero_seeds], y[zero_seeds], tau_p)
        q = g_in_q + tau_q*np.dot(A_T, g_out_array)
        
        #print(q)
        
        
        #print((tau_p - tau_p_prev)**2/tau_p**2)
        #Stopping criterion - Relative norm tolerance
        if (tau_p - tau_p_prev)**2/tau_p**2 < 1e-12 and it >= 1:
            break
        
        tau_p_prev = 0 + tau_p
    
    return error_norm_array, norm_correl_array, x_hat
"""
delta = 0.65
n = 20000
m = int(delta*n)
print("Delta: ", delta)
mse_runs = []
nc_runs = []
it = 0
run_no = 1
eps = 1
while it < run_no:
    #print(it)
    A_iid = produce_data.create_A_iid(m,n)
    A_iid_T = np.transpose(A_iid)

    x_0 = produce_data.create_x_0(eps, n, "Discrete")*(1/np.sqrt(eps))
    y = np.maximum(np.dot(A_iid, x_0),0)
    
    
    #error_norm_array, x_hat= cs_gamp_iid_new(A_iid, y, 300, eps, 1/np.sqrt(eps), x_0, exp_x=0)
    error_norm_array, norm_correl_array, x_hat = relu_gamp_iid(A_iid, A_iid_T, y, 300, eps, 1/np.sqrt(eps), x_0)
    it += 1
"""

#------SPATIALLY COUPLED DESIGN MATRIX-----------
    
#STATE EVOLUTION
def se_sc_relu_exact(W, delta_hat, eps, a, it):

    R, C = len(W), len(W[0])

    exp_x2 = eps*(a**2)
    tau_p = np.ones(R)
    #tau_q = np.zeros(C)
    tau_q_inv = np.zeros(C)
    
    exp_zr2 = (exp_x2/delta_hat)*np.sum(W,axis=1)
    
    tau_p = np.zeros(R)
    scaled_expec = np.zeros(R)
    mmse_vec = np.zeros(C)
    
    tau_p_prev = np.zeros(R)
    
    tau_p_array = []
    
    #t=0:
    mmse_vec = np.ones(C)*0.99
    
    for r in range(R):
        tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse_vec)

    tau_p_inv = (tau_p)**(-1)

    #Compute expectation of variance term first, only depends on r, fixed for any c
    for r in range(R):
        #print(r)
        var_p = exp_zr2[r] - tau_p[r]
        if var_p == 0:
            expec = quad(integral_1, -5,5, args=(0, tau_p[r]))[0]
        else:
            expec = quad(integral_2, -5*var_p,5*var_p, args=(tau_p[r], exp_zr2[r]))[0]
        #data1 = np.array([tau_p[r], exp_zr2[r]], np.float64)
        #expec = dqags(integral2ptr, -30,30, data=data1)[0]
        scaled_expec[r] = (tau_p_inv[r])*(expec)

    for c in range(C):
        tau_q_inv[c] = (np.dot(W[:,c], scaled_expec))
    
    tau_p_array.append(np.zeros(R) + tau_p)
    #print(tau_p)
    #print(tau_q)
    
    for t in range(it-1):
        print(t)
        print(1/tau_q_inv)
        
        for c in range(C):
            #print(np.sqrt(1/tau_q[c]))
            mmse_vec[c]= mmse_new(np.sqrt(tau_q_inv[c]),eps, a)
            
            #print(mmse_vec)
        #print("MMSE Vec: ", mmse_vec)
        for r in range(R):
            tau_p[r] = (1/delta_hat)*np.dot(W[r,:], mmse_vec)
            #if tau_p[r] > tau_p_array[-1][r]:
            #   tau_p[r]=np.copy(tau_p_array[-1][r])
                
            if tau_p[r]<=0 and tau_p[r]>-1e-3:
                tau_p[r]= 1e-20
        
        #print(tau_p)
        
        tau_p_array.append(np.zeros(R) + tau_p)
        
        if np.max(tau_p) < 1e-2:
            break
        
        tau_p_inv = (tau_p)**(-1)
        
        #Compute expectation of variance term first, only depends on r, fixed for any c
        for r in range(R):
            #print(tau_p[r], exp_zr2[r])
            var_p = exp_zr2[r] - tau_p[r]
            if var_p == 0:
                expec = quad(integral_1, -5,5, args=(0, tau_p[r]))[0]
            else:
                expec = quad(integral_2, -5*var_p,5*var_p, args=(tau_p[r], exp_zr2[r]))[0]
            #data1 = np.array([tau_p[r], exp_zr2[r]], np.float64)
            #expec = dqags(integral2ptr, -30,30, data=data1)[0]
            scaled_expec[r] = (tau_p_inv[r])*(expec)

        for c in range(C):
            
            tau_q_inv_c_new = (np.dot(W[:,c], scaled_expec))
            #TRICK: to avoid edge coordinates increasing back up once they have fallen
            if tau_q_inv_c_new > tau_q_inv[c]:
                tau_q_inv[c]=np.copy(tau_q_inv_c_new)
        
        #print(tau_p)
        
        print("Convergence:",np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2))
        
        if (np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2)) < 1e-8:
                break
        
        
        tau_p_prev = np.zeros(R) + tau_p
    
    #Compute MSE Prediction
    mse_pred = (1/C)*np.sum(mmse_vec)
    
    #Normalized Correlation Squared using x_hat
    nc_pred = 1 - mse_pred
    
    return tau_p_array, mse_pred, nc_pred
"""
#Generate omega-lambda base matrix
omega = 6
lam = 40

C = lam
R = lam + omega - 1


W = produce_data.create_W(lam, omega)

nc_pred_array = []
mse_pred_array = []
delta_array = [0.6]#np.linspace(0.8,1,num=5)
#delta_array_zoom = [1.36,1.38, 1.4, 1.42]
#delta_array_plus = np.sort(np.concatenate([delta_array_ext, delta_array_zoom]))
#mse_pred_array = np.sort(mse_pred_array)[::-1]


eps = 1
for delta in delta_array:
    print("Delta: ", delta)
    delta_hat = delta*lam/(lam + omega - 1)
    tau_p_array, mse_pred, nc_pred = se_sc_relu_exact(W, delta_hat, eps, 1/np.sqrt(eps), 300)
    nc_pred_array.append(nc_pred)
    mse_pred_array.append(mse_pred)

plt.figure()
plt.plot(delta_array, mse_pred_array, marker="o", linestyle="-")
plt.xlabel(r"$\delta$")
plt.ylabel("Normalized Correlation Squared")
plt.title(r"Spatially Coupled Gaussian State Evolution, ReLU, Discrete Prior, $\epsilon=${}".format(eps))

#WAVE PROPAGATION BEHAVIOUR OF SPATIAL COUPLING
plt.figure()
plt.xlabel(r"Index $r$")
plt.ylabel(r"$\tau_r^p(t)$")
plt.title(r"Discrete Prior, Spatially Coupled, $\Lambda=${}, $\omega=${}, $\epsilon=${}, $\delta=${}".format(lam, omega, eps, delta))
l = int(len(tau_p_array)/10 + 1)
for i in range(l):
    plt.plot(tau_p_array[int(10*i)], label="t={}".format(10*i+1), alpha=(i+1)/l, color="red")
plt.legend()"""


def relu_gamp_sc(W, A, A_T, y, x_0, eps, a, t, exp_x=0):
    m, n = len(A), len(A[0])
    R, C = len(W), len(W[0])
    M = int(m/R)#
    N = int(n/C)
    delta_hat = M/N
    
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
    x_hat = np.zeros(n)

    #Find positions of 0s in y, set them into length-R array  
    non_zero_seeds= [[] for i in range(R)]
    zero_seeds = [[] for i in range(R)]
    ov = np.array(np.where(y)[0])
    ov0 = np.array(np.where(y==0)[0])
    for r in range(R):
        non_zero_seeds[r] = ov[(ov>=M*r) & (ov<M*r+M)]
        zero_seeds[r] = ov0[(ov0>=M*r) & (ov0<M*r+M)]
    
    
    exp_x2 = eps*(a**2)
    exp_zr2 = (exp_x2/delta_hat)*np.sum(W[:,:],axis=1)
    
       
    tau_p = exp_zr2
    
    tau_p_prev = 0 + tau_p
    
    x_hat_0 = np.ones(n)*exp_x
    
    p = np.dot(A, x_hat_0)
    
    
    neg_deriv_array = np.zeros(m)
    g_out_array = np.zeros(m)

    
    for r in range(R):
        neg_deriv_array[non_zero_seeds[r]]=neg_deriv_g_out_relu_ypos(p[non_zero_seeds[r]], y[non_zero_seeds[r]], tau_p[r])
        neg_deriv_array[zero_seeds[r]]=neg_deriv_g_out_relu_y0(p[zero_seeds[r]], y[zero_seeds[r]], tau_p[r])
        neg_dgout_av[r]=np.average(neg_deriv_array[M*r:(M*r+M)])
    
    # Calculate tau_q
    for c in range(C):
        tau_q[c] = 1/(np.dot(W[:,c], neg_dgout_av))
    #print(tau_q)
    # Calculate q
    #print(len(p[M*r:(M*r+M)]), len(y[M*r:(M*r+M)]), (tau_p[r]))
    for r in range(R):
        g_out_array[non_zero_seeds[r]]=g_out_relu_ypos(p[non_zero_seeds[r]], y[non_zero_seeds[r]], tau_p[r])
        g_out_array[zero_seeds[r]]=g_out_relu_y0(p[zero_seeds[r]], y[zero_seeds[r]], tau_p[r])

    
    #print(c, n, len(np.dot(A_T, g_out_p)))
    for c in range(C):
        tau_q_A_gout[N*c: (N*c + N)] = tau_q[c]*np.dot(A_T, g_out_array)[N*c:(N*c+N)]
        
    q = x_hat_0 + tau_q_A_gout
    #print(tau_p, tau_p_prev)
    #print(tau_q)
    error_norm_array = []
    norm_correl_array = []
    x_hat_array = []
    
    
    
    for it in range(t):
        
        for c in range(C):
            deriv_average[c] = tau_q[c]*np.average(deriv_bayes_eta(q[N*c:(N*c + N)], np.sqrt(tau_q[c]), eps, a))
            denoised_q[N*c:(N*c+N)] = bayes_eta(q[N*c:(N*c + N)], np.sqrt(tau_q[c]), eps, a)
        
        for r in range(R):
            tau_p_r_new= max((1/delta_hat)*np.dot(W[r,:], deriv_average), 1e-10)#(1/delta_hat)*np.dot(W[r,:], deriv_average)
            if np.isnan(tau_p_r_new) == False:
                tau_p[r]=np.copy(tau_p_r_new)
            
        for r in range(R):
            g_out_tau_p[M*r:(M*r+M)] = tau_p[r]*g_out_array[M*r:(M*r+M)]
            
            
        p = np.dot(A, denoised_q) - g_out_tau_p
        
        print("Tau_p at t={}: ".format(it), tau_p)
        for r in range(R):
            #print(r, tau_p)
            neg_deriv_array[non_zero_seeds[r]]=neg_deriv_g_out_relu_ypos(p[non_zero_seeds[r]], y[non_zero_seeds[r]], tau_p[r])
            neg_deriv_array[zero_seeds[r]]=neg_deriv_g_out_relu_y0(p[zero_seeds[r]], y[zero_seeds[r]], tau_p[r])
            neg_dgout_av[r]=np.average(neg_deriv_array[M*r:(M*r+M)])
    
        # Calculate tau_q
        for c in range(C):
            
            tau_q_c_new = 1/(np.dot(W[:,c], neg_dgout_av))
            #TRICK: to avoid edge coordinates increasing back up once they have fallen
            if tau_q_c_new < tau_q[c]:
                tau_q[c]=np.copy(tau_q_c_new)
                
            if tau_q[c] < 1e-10:
                tau_q[c] = 1e-10
            
            #tau_q[c] = 1/(np.dot(W[:,c], neg_dgout_av))#1/((1/tau_p) - exp_z2/(tau_p**2) + (1/tau_p**2)*np.average(y*(np.tanh(p*(y**0.5)/tau_p)**2)))#
        print("Tau_q at t={}: ".format(it), tau_q)
        
        if tau_q.any()<=1e-3 and tau_q.any() > -1 or np.isnan(tau_q.any()):
            print("Break 1")
            break
        
        for r in range(R):
            g_out_array[non_zero_seeds[r]]=g_out_relu_ypos(p[non_zero_seeds[r]], y[non_zero_seeds[r]], tau_p[r])
            g_out_array[zero_seeds[r]]=g_out_relu_y0(p[zero_seeds[r]], y[zero_seeds[r]], tau_p[r])

        for c in range(C):
            tau_q_A_gout[N*c: (N*c + N)] = tau_q[c]*np.dot(A_T, g_out_array)[N*c:(N*c+N)]
        
        # Calculate q
        q = denoised_q + tau_q_A_gout
        
        stop_crit = np.linalg.norm(tau_p - tau_p_prev, 2)/np.linalg.norm(tau_p, 2)
        print(stop_crit)
        #print((tau_p - tau_p_prev)**2/tau_p**2)
        #Stopping criterion - Relative norm tolerance
        if stop_crit < 1e-6 or np.linalg.norm(tau_p_prev, 2)<1e-12:
            print("Break 2")
            #print(tau_p, tau_p_prev,np.linalg.norm(tau_p - tau_p_prev, 2),np.linalg.norm(tau_p, 2) )
            break
        
        tau_p_prev = np.zeros(R) + tau_p
        x_hat_array.append(0 + x_hat)
        x_hat = denoised_q
        
        #Phase-retrieval error might be calculated differently
        mse_x_pos = (1/n)*(np.linalg.norm(x_hat - x_0)**2)
        mse_x_neg = (1/n)*(np.linalg.norm(-x_hat - x_0)**2)
        #print(mse_x_neg, mse_x_pos)
        mse = min(mse_x_pos, mse_x_neg)
        
        #ALT Measure of Performance: Normalised Correlation
        norm_correl = (np.dot(x_hat, x_0)/(np.linalg.norm(x_hat)*np.linalg.norm(x_0)))**2
        print(mse, norm_correl, x_hat)
        if np.isnan(mse) or np.isnan(norm_correl):
            print(x_hat_array[-2:])
            if np.any(np.isnan(x_hat)):
                #print(1)
                x_hat = x_hat_array[-1]
                if np.any(np.isnan(x_hat)):
                    #print(2)
                    x_hat = x_hat_array[-2] 
            break
                
        error_norm_array.append(mse)
        norm_correl_array.append(norm_correl)
        
    return error_norm_array, norm_correl_array, x_hat

"""
#Generate omega-lambda base matrix
omega = 6
lam = 40

C = lam
R = lam + omega - 1


W = produce_data.create_W(lam, omega)

eps_domain = np.arange(0.1,1.1,0.1)
delta = 0.9
mse_array = []
for eps in eps_domain:
    delta_hat = delta*lam/(lam + omega - 1)
    print("Delta: ", delta)
    _, mse_pred = se_iid_pr_exact(delta, eps, 1/np.sqrt(eps),300)
    mse_array.append(mse_pred)

#plt.figure()
plt.plot(eps_domain, mse_array)
plt.xlabel(r'$\epsilon$')
plt.ylabel('Predicted MSE')
run_no = 0
error = []
#Discrete Prior, eps = 0.3, delta = 0.5, varying sigma^2
eps = 1
delta = 0.5
lam = 40
omega = 6
delta_hat = delta*lam/(lam + omega - 1)
print("Delta: ", delta)
N = 500
t = 300
n = N*lam
while run_no < 1:
    W = produce_data.create_W(lam, omega)
    A_sc = produce_data.create_A_sc(W, N, delta_hat)
    
    x_0 = produce_data.create_x_0(eps, n, "Discrete")*(1/np.sqrt(eps))
    print(x_0)
    y = np.maximum(np.dot(A_sc, x_0),0) #+ np.random.normal(0, 0.0001, size=int(delta*n))
    A_sc_T = np.transpose(A_sc)
    
    error_norm_array, norm_correl_array, x_hat = relu_gamp_sc(W, A_sc, A_sc_T, y, x_0, eps, 1/np.sqrt(eps), t)
    error.append(error_norm_array)
    run_no += 1
    #plt.figure()
    #plt.plot(error_norm_array)
    #plt.grid(alpha=0.4, linestyle="--")
    
print(error)"""
"""

run_no = 30
eps = 0.1
t = 50
delta_domain = np.arange(0.5,1.55,0.1)
delta_domain_zoom = np.arange(0.75, 0.9, 0.02)
delta_domain_large = np.arange(0.2,1.55,0.02)
mse_array = []
nc_array =[]
se_mse_array = []
se_nc_array = []
mse_array_std = []
nc_array_std = []
for delta in delta_domain:
    n = 10000
    m = int(delta*n)
    print("Delta: ", delta)
    mse_runs = []
    nc_runs = []
    it = 0
    while it < run_no:
        print(it)
        A_iid = produce_data.create_A_iid(m,n)
        A_iid_T = np.transpose(A_iid)
    
        x_0 = produce_data.create_x_0(eps, n, "Discrete")*(1/np.sqrt(eps))
        y = np.dot(A_iid, x_0)**2
        
        
        error_norm_array, norm_correl_array, x_hat = pr_gamp_iid(A_iid, A_iid_T, y, t, eps, 1/np.sqrt(eps), x_0)
        it += 1
        
        print(norm_correl_array[-1])
        nc_runs.append(norm_correl_array[-1])
        mse_runs.append(error_norm_array[-1])
    mse_array.append(np.average(mse_runs))
    mse_array_std.append(np.std(mse_runs))
    nc_array.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    
for delta in delta_domain_large:
    _, iid_se_mse, iid_se_nc = se_iid_pr_exact(delta, eps, 1/np.sqrt(eps), 500)
    se_mse_array.append(iid_se_mse)
    se_nc_array.append(iid_se_nc)
    

#delta_domain = np.concatenate((delta_domain, delta_domain_zoom), axis=None)

plt.figure()
plt.errorbar(delta_domain, nc_array, yerr=nc_array_std, label ="GAMP, iid Gaussian", fmt='o', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.plot(delta_domain_large, se_nc_array, label="State Evolution, iid Gaussian", color='red')
plt.xlabel(r'$\delta$')
plt.ylabel('Normalized Squared Correlation')
plt.legend(loc="lower right")
plt.title(r"$\epsilon=${}, $n=${}, averaged over {} runs, Discrete".format(eps, n, run_no))

plt.figure()
plt.errorbar(delta_domain, mse_array, yerr=mse_array_std, label ="GAMP, iid Gaussian", fmt='o', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.plot(delta_domain_large, se_mse_array, label="State Evolution, iid Gaussian", color='red')
plt.xlabel(r'$\delta$')
plt.ylabel('Mean Squared Error')
plt.legend(loc="upper right")
plt.title(r"$\epsilon=${}, $n=${}, averaged over {} runs, Discrete".format(eps, n, run_no))
#error_norm_array, norm_correl_array, x_hat = pr_gamp_iid(A_iid, A_iid_T, y, t, eps, 1/np.sqrt(eps), x_0)

#plt.plot(error_norm_array)
#iid_se = se_iid_pr_exact(1.25, eps, 1/np.sqrt(eps), t)[-1]"""
"""
with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/noiseless_pr_iid_n{}_eps{}_t{}_{}runs.txt".format(n, eps, t, run_no), "w") as output:
    output.write(str(delta_domain_large.tolist())+'\n')
    output.write(str(se_mse_array)+'\n')
    output.write(str(se_nc_array)+'\n')
    output.write(str(delta_domain.tolist())+'\n')
    output.write(str(mse_array)+'\n')
    output.write(str(mse_array_std)+'\n')
    output.write(str(nc_array)+'\n')
    output.write(str(nc_array_std)+'\n')"""

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
def expec_var_cs(exp_x2, x, delta):
    exp_z2 = exp_x2/delta
    expec = quad(integral_2, -30, 30, args=((x/delta), exp_z2))[0]
    var = (x/delta)*(1-expec)
    return var

def scalar_pot_integrand(z, exp_x2, delta):
    integrand = -(1/z)*((delta/z)*expec_var_cs(exp_x2, z, delta))
    return integrand

def scalar_pot_cs(x, delta, eps, a, exp_x2=1):
    #print((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta)))
    #u_s = delta*((delta/x)*expec_var_pr(exp_x2, x, delta) - 1) + delta*(quad(scalar_pot_integrand, 0.00001, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    #ALT: Get rid of ln 0
    u_s = delta*((delta/x)*expec_var_cs(exp_x2, x, delta) - 1) + delta*np.log(x)+delta*(quad(scalar_pot_integrand, 0.0005, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_cs(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    
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

def cs_bound_disc(delta, x_domain, eps, a):
    
    #start_time = time.time()
    
    scalar_pot_arr = np.zeros(len(x_domain))
    for i in range(len(x_domain)):
        scalar_pot_arr[i] = scalar_pot_cs(x_domain[i], delta, eps, a)
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

#mse_bound, iid_bound, scalar_pot_arr = cs_bound_disc(1.3, np.linspace(0.001,0.99,num=99),1,1)
