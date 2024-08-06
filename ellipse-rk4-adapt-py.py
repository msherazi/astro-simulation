"""
Solves the 1-body gravitational problem for an elliptical orbit using the
fourth-order Runge-Kutta method with adaptive step size. 
Units such that G=1, distance in AU, time in years. Outputs the absolute 
fractional error for the total energy.
"""

import numpy as np, matplotlib.pyplot as plt, a401, timeit
import scipy as sp 


def f_1body(t, w):
    """
    The function for the evolution of the 1-body problem, eqs 1.6.24-1.6.27.
    Units of AU, yr, and mass such that G=1. 
    """
    
    k = -(2.0*np.pi)**2
    r3 = (w[0]**2 + w[1]**2)**1.5
    f = np.zeros(4)
    f[0] = w[2]
    f[1] = w[3]
    f[2] = k*w[0]/r3
    f[3] = k*w[1]/r3
    
    return f

start = timeit.default_timer()
t_0 = 0.0 
t_f = 1.0e2   
g_acc = 1.0e-14  
dt0 = 1.0e-6  
e = 0.99 

w0 = np.zeros(4)
w0[0] = 1.0 - e
w0[3] = 2.0*np.pi*np.sqrt((1.0+e)/(1.0-e))
E_m0 = (-4.0*np.pi**2/np.sqrt(w0[0]**2 + w0[1]**2)) + \
           0.5*(w0[2]**2 + w0[3]**2) 


t_span = np.array([t_0, t_f]) 
sol = sp.integrate.solve_ivp(f_1body, t_span, w0, rtol = g_acc, atol = g_acc)
w = sol.y[: , -1]

GM = 4.0*np.pi**2
r = np.sqrt(w[0]**2 + w[1]**2)
v2 = w[2]**2 + w[3]**2
E_m = - GM/r + 0.5*v2
err_E = abs((E_m - E_m0)/E_m0)
err_E = abs((E_m - E_m0)/E_m0)    
a_final = 1.0/((2.0/r) - (v2/GM))  
e_final = np.sqrt(1.0 - ((w[0]*w[3]-w[1]*w[2])**2)/(GM*a_final))
err_e = abs((e_final-e)/e)

stop = timeit.default_timer()
run_time = stop - start

print(g_acc, err_E, err_e, run_time)