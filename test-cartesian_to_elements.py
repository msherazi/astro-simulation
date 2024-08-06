import numpy as np , a401

m = np.array ([1.0 , 0.2 , 0.03 , 0.12 , 0.45])
a = np.array ([0.1 , 1.4 , 3.7 , 2.5 , 0.8])
e = np.array ([0.2 , 0.6 , 0.25 , 0.88 , 0.99])
theta0 = np.array([0.7 , -0.3 , 1.2 , 2.9 , -2.0])
sense = np.ones(5 , int )
sense[2] = -1
sense [4] = -1

# pericenter 

theta = theta0

x1, y1, v_x1, v_y1 = a401.elements_to_cartesian(m, a, e, theta0, theta, sense)

a2, e2, theta02, theta2, sense2 = a401.cartesian_to_elements(m, x1, y1, v_x1, v_y1)

err_a = np.max(np.abs((a2 - a)/a))
err_e = np.max(np.abs((e2 - e)/e))
err_theta0 = np.max(np.abs((theta02 - theta0)/theta0))
err_theta = np.max(np.abs((theta2 - theta)/theta))
err_sense = np.max(np.abs((sense2 - sense)/sense))

z = np.array ([err_a, err_e, err_theta0, err_theta, err_sense])
print('pericenter')
print(z)

# apocenter

theta = theta0 + np.pi

x1, y1, v_x1, v_y1 = a401.elements_to_cartesian(m, a, e, theta0, theta, sense)

a2, e2, theta02, theta2, sense2 = a401.cartesian_to_elements(m, x1, y1, v_x1, v_y1)

err_a = np.max(np.abs((a2 - a)/a))
err_e = np.max( np.abs((e2 - e)/e))
err_theta0 = np.max(np.abs((theta02 - theta0)/theta0))

for i in range(0, 5):
    if(theta2[i] < 0):
        theta2[i] = theta2[i] + 2.0 * np.pi
        err_theta = np.max(np.abs((theta2 - theta)/theta))
        err_sense = np.max(np.abs((sense2 - sense)/sense))

z = np.array([err_a,err_e, err_theta0, err_theta, err_sense])
print('apocenter')
print(z)

# circle

theta = np.array([0.4, 0.2, -0.7, 0.5, 1.4])
e = np.zeros(5)

x1, y1, v_x1, v_y1 = a401.elements_to_cartesian(m, a, e, theta0, theta, sense)
a2, e2, theta02, theta2, sense2 = a401.cartesian_to_elements(m, x1, y1, v_x1, v_y1)

print(e)

err_a = np.max(np.abs((a2 - a)/a))
err_theta = np.max(np.abs((theta2 - theta)/theta))
err_sense = np.max (np.abs((sense2 - sense)/sense))

z = np.array([err_a , err_theta, err_sense])
print(z)
