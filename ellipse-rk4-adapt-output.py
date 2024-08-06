import numpy as np, matplotlib.pyplot as plt, timeit, a401

start = timeit.default_timer()
t_0 = 0.0  
t_f = 1.0e2  
g_acc = 1.0e-14  
dt0 = 1.0e-6  
e = 0.99  

n_out_tot = 1001  
t_out = np.linspace(t_0, t_f, n_out_tot)  


w0 = np.zeros(4)
w0[0] = 1.0 - e
w0[3] = 2.0*np.pi*np.sqrt((1.0+e)/(1.0-e))
E_m0 = (-4.0*np.pi**2/np.sqrt(w0[0]**2 + w0[1]**2)) + 0.5*(w0[2]**2 + w0[3]**2) 

w_out, n_step_out  = a401.evolve_rk4_adapt_output(dt0, t_0, t_f, w0, g_acc, t_out, n_out_tot, a401.f_1body)

x = w_out[:, 0]
y = w_out[:, 1]
v_x = w_out[:, 2]
v_y = w_out[:, 3]
GM = 4.0*np.pi**2
r = np.sqrt(x**2 + y**2)
v2 = v_x**2 + v_y**2
E_m = - GM/r + 0.5*v2
err_E = abs((E_m - E_m0)/E_m0)
a_final = 1.0/((2.0/r) - (v2/GM))  
e_final = np.sqrt(1.0 - ((x*v_y-y*v_x)**2)/(GM*a_final))  
err_e = abs((e_final-e)/e)

stop = timeit.default_timer()
run_time = stop - start


output = np.column_stack((t_out, err_E, err_e, n_step_out))
np.savetxt('out.rk4-adaptive', output)

print(g_acc, err_E, err_e, run_time)

