import numpy as np, matplotlib.pyplot as plt, timeit

output = np.loadtxt("out.rk4-adaptive", float)
t0, enerr, err_e0, nsteps = output.T
n_tot = t0.size
t = t0[1:n_tot-1]
errEnergy = enerr[1:n_tot-1]
nstep = nsteps[1:n_tot-1]

g_acc = 1.0e-14
quantity = g_acc*nstep


plt.figure()
plt.plot(t, errEnergy, ls='-', marker='o', color='r', label= "Energy error")


ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("time")

plt.savefig("rk4error.jpg")