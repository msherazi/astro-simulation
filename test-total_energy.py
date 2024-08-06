import numpy as np, a401

N = 3
m = np.array([1.0, 1.3, 1.7])   # masses
x = np.array([0.32, -0.80, 0.62])  # x-coordinates
y = np.array([-0.74, -0.12, 0.33])  # x-coordinates
v_x = np.array([0.42, 0.91, -0.57])  # v_x
v_y = np.array([-0.18, 0.24, 0.61])  # v_x


w = np.hstack((x, y, v_x, v_y))
K, U = a401.total_energy(w, m)

print(K, U,"  = total K and U from function call")

K2 = 0.5 * (m[0] * (v_x[0]**2 + v_y[0]**2) + m[1] * (v_x[1]**2 + v_y[1]**2) + m[2] * (v_x[2]**2 + v_y[2]**2))

print(np.abs((K-K2)/K2),"  = absolute fractional error for K")


x1 = x[1] - x[0]
y1 = y[1] - y[0]

x2 = x[2] - x[0]
y2 = y[2] - y[0]

x3 = x[2] - x[1]
y3 = y[2] - y[1]

r1 = np.sqrt(x1**2 + y1**2)
r2 = np.sqrt(x2**2 + y2**2)
r3 = np.sqrt(x3**2 + y3**2)

U2 = -1 * ((m[0] * m[1])/r1 + (m[0] * m[2])/r2 + (m[1] * m[2])/r3)

print(U2)

print(np.abs((U-U2)/U2),"  = absolute fractional error for U")