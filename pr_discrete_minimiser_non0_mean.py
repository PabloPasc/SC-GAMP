import numpy as np
import scipy.stats, scipy.optimize, scipy.signal
from scipy.integrate import quad
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

@jit(nopython=True)
def expectation_p_g(p_r, tau_p, g_r):
    y = (p_r + np.sqrt(tau_p)*g_r)**2
    res = y*(1 - np.tanh((p_r*np.sqrt(y))/tau_p)**2)
    return res

@jit(nopython=True)
def integral_1(g_r, p_r, tau_p):
    integrand = norm_pdf(g_r)*(expectation_p_g(p_r, tau_p, g_r))
    return integrand

#@jit(nopython=True)
def integral_2(p_r, tau_p, exp_zr2):
    var_p = exp_zr2 - tau_p
    integrand = norm_pdf(p_r, scale=np.sqrt(var_p))*(quad(integral_1, -30,30, args=(p_r, tau_p))[0])
    return integrand

@jit(nopython=True)    
def diff_ent_int(y, s_sqrt, a, prob):
    f_y = prob*norm_pdf(y, a*s_sqrt, 1) + (1-prob)*norm_pdf(y, -a*s_sqrt, 1)
    ent = f_y * np.log(f_y)
    return ent

def mutual_inf_exact(s_sqrt, a, prob):
    if a*s_sqrt < 25:
        integ = quad(diff_ent_int, -30, 30, args=(s_sqrt, a, prob))[0]
    else:
        integ = quad(diff_ent_int, -a*s_sqrt - 5, -a*s_sqrt + 5, args =(s_sqrt, a, prob))[0] + \
        quad(diff_ent_int, a*s_sqrt - 5, a*s_sqrt + 5, args =(s_sqrt, a, prob))[0]
    mutu = -integ - 0.5*np.log(2*np.pi*np.exp(1))
    #print(integ, 0.5*np.log(2*np.pi*np.exp(1)))
    return mutu

def expec_var_pr(exp_x2, x, delta):
    exp_z2 = exp_x2/delta
    return quad(integral_2, -40, 40, args=((x/delta), exp_z2))[0]

def scalar_pot_integrand(z, exp_x2, delta):
    integrand = -(1/z)*((delta/z)*expec_var_pr(exp_x2, z, delta))
    return integrand

def scalar_pot_pr(x, delta, a, prob):
    exp_x2 = a**2
    #print((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta)))
    #u_s = delta*((delta/x)*expec_var_pr(exp_x2, x, delta) - 1) + delta*(quad(scalar_pot_integrand, 0.00001, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    #ALT: Get rid of ln 0
    u_s = delta*((delta/x)*expec_var_pr(exp_x2, x, delta) - 1) + delta*np.log(x)+delta*(quad(scalar_pot_integrand, 0.005, x, args=(exp_x2, delta))[0]) + 2*mutual_inf_exact(np.sqrt((delta/x)*(1-(delta/x)*expec_var_pr(exp_x2, x, delta))), a, prob) #- 2*mutual_inf_exact(np.sqrt(1), eps, a)
    
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
dom = np.arange(0.01,1,0.001)
delta = 1.2
g_x = []
for x in dom:
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

def pr_bound_disc(delta, x_domain, a, prob):
    
    #start_time = time.time()
    
    scalar_pot_arr = np.zeros(len(x_domain))
    for i in range(len(x_domain)):
        scalar_pot_arr[i] = scalar_pot_pr(x_domain[i], delta, a, prob)
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

prob = 0.6
a = np.sqrt(1/(1-(2*prob-1)**2))
x_domain_new = np.linspace(0.005,(a**2-5e-3), num=200)
"""print(pr_bound_disc(0.8, x_domain, a, prob))
eps = 0.5"""

glob_min_array = []
larg_min_array = []
pot_array_array = []
delta_domain = [0.27]#[0.26, 0.27, 0.28]#, 0.4, 0.6, 0.8]#np.linspace(0.21,0.29, num=9)
delta_domain_new = np.linspace(0.3,1.2,num=37)
#delta_domain_ext = np.concatenate([delta_domain_new, delta_domain])

for delta in delta_domain:
    print("Delta: ", delta)
    glob_min, larg_min, pot_array = pr_bound_disc(delta, x_domain, a, prob)
    print(glob_min, larg_min)
    glob_min_array.append(glob_min)
    larg_min_array.append(larg_min)
    pot_array_array.append(pot_array)

plt.figure()
#plt.plot(x_domain, pot_array_array[0], label="delta = 0.26")
plt.plot(x_domain, pot_array_array[1], label="delta = 0.27")
plt.plot(x_domain, pot_array_array[3], label="delta = 0.268")
#print(scalar_pot_pr(0.99995,0.42, eps, 1/np.sqrt(eps)))
  
#plt.figure()
#plt.plot(delta_domain_ext, glob_min_array, label = "Global Minimizer",color='red', marker="x")
#plt.plot(delta_domain_ext, larg_min_array, label = "Largest Minimizer",color='green',marker="x")



with open("/home/pp423/Documents/PhD/2nd Year/Text file Data/noiseless_pr_potential_delta0.27.txt", "w") as output:
    output.write(str(delta_domain)+'\n')
    output.write(str(glob_min_array[1])+'\n')
    output.write(str(larg_min_array[1])+'\n')
    output.write(str(pot_array.tolist()))

