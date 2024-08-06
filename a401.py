import numpy as np, scipy.optimize as opt
from numba import njit

def f_Kepler(E, e, M):
    """
    eq 1.3.1, function for Kepler's equation
    """
    return E - e*np.sin(E) - M

def find_E_bisection(e, M, E_1, E_2, acc):
    """
    Use bisection to find the root E for Kepler's equation, given bracketing 
    values E_1 and E_2 and the desired accuracy acc. Eccentricity e and 
    mean anomaly M are also input. 
    """
    E_low = E_1
    f_low = f_Kepler(E_low, e, M)
    E_high = E_2
    E_mid = 0.5*(E_high + E_low)
    f_mid = f_Kepler(E_mid, e, M)
    while np.abs(f_mid) > acc:
        if (f_low*f_mid) > 0:
            E_low = E_mid
            f_low = f_mid
        else:
            E_high = E_mid
        E_mid = 0.5*(E_high + E_low)
        f_mid = f_Kepler(E_mid, e, M)
    return E_mid

def full_orbit_Kepler_sol(e, n_M, acc):
    """
    Find the solution of Kepler's equation for a full orbit with semimajor
    axis equal to 1. 

    Input:
    e = eccentricity
    n_M = number of points along the orbit (evenly spaced times)
    acc = accuracy of solution (allowed absolute deviation from zero)

    Output:
    M_arr = array of mean anomaly M
    E = array of eccentric anomaly E
    r = array of radial coordinate r
    theta = array of angular coordinate theta
    """

    M_arr = np.linspace(0.0, 2.0*np.pi, n_M)
    E = np.zeros(n_M)
    r = np.zeros(n_M)
    theta = np.zeros(n_M)

    E[0] = 0.0
    r[0] = 1.0 - e
    theta[0] = 0.0

    E[n_M-1] = 2.0*np.pi
    r[n_M-1] = 1.0 - e
    theta[n_M-1] = 2.0*np.pi

    E_1 = 0.0
    E_2 = 2.0*np.pi

    for i in range(1, n_M-1):
        M = M_arr[i]
        E[i] = find_E_bisection(e, M, E_1, E_2, acc)
        
        r[i] = 1.0 - e*np.cos(E[i])   
        cE = np.cos(E[i])
        theta[i] = np.arccos((cE-e)/(1.0-e*cE))  
        
        if E[i] > np.pi:
            theta[i] = 2.0*np.pi - theta[i]
   
    return M_arr, E, r, theta

@njit
def f_1body(w):
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

@njit
def rk2_step(w, dt, f):
    """
    Take a second-order Runge-Kutta step, eqs 1.6.43-1.6.45, given the
    initial vector w, time step dt, and function f. 
    """

    k1 = dt*f(w)
    k2 = dt*f(w+0.5*k1)
    w += k2

    return w

@njit
def rk4_step(w, dt, f):
    """
    Take a fourth-order Runge-Kutta step, eqs 1.6.46-1.6.50, given the
    initial vector w, time step dt, and function f. 
    """

    k1 = dt*f(w)
    k2 = dt*f(w+0.5*k1)
    k3 = dt*f(w+0.5*k2)
    k4 = dt*f(w+k3)    
    w += (k1+2.0*k2+2.0*k3+k4)/6.0

    return w

@njit
def evolve_rk_fixed(dt,t_tot, w0, f, meth):
    """
    Evolve a system whose evolution is described by function f for a time 
    t_tot using the RK2 or RK4 method and a fixed time step dt. 
    The initial vector of positions and velocities is w0. 
    Returns the final vector w. 
    """
    
    n = int(t_tot/dt) 
    w = np.copy(w0)
    for j in range(n):
        if (meth == "RK2"):
            w = rk2_step(w, dt, f)
        else:
            w = rk4_step(w, dt, f)
            
    return w            

@njit
def rk4_step_adapt(w, h, g_acc, f):
    """
    Take an adaptive RK4 step of size 2*h (two steps of size h for error
    estimation), given the initial vector w, the target accuracy g_acc for
    the step, and the function f. 
    Output: w2 = double-step update of the vector w  (eq 1.7.28)
            er = error estimate  (eq 1.7.29)
            h2 = 2*h
            alpha = step size adjustment parameter  (eq 1.7.35)
    """

    h2 = 2.0*h
    k11 = h2*f(w)                                 
    k21 = h2*f(w+0.5*k11)                         
    k31 = h2*f(w+0.5*k21)                         
    k41 = h2*f(w+k31)                             
    w1 = w + (k11+2.0*k21+2.0*k31+k41)/6.0        
    k12a = 0.5*k11                                
    k22a = h*f(w+0.5*k12a)                        
    k32a = h*f(w+0.5*k22a)                        
    k42a = h*f(w+k32a)                            
    w2a = w + (k12a+2.0*k22a+2.0*k32a+k42a)/6.0   
    k12b = h*f(w2a)                               
    k22b = h*f(w2a+0.5*k12b)                      
    k32b = h*f(w2a+0.5*k22b)                      
    k42b = h*f(w2a+k32b)                         
    w2 = w2a + (k12b+2.0*k22b+2.0*k32b+k42b)/6.0  
    er = (w2 - w1)/15.0                           
    er0 = g_acc*(np.abs(w) + np.abs(k11))         
    alpha0 = (er0/np.abs(er))**0.2                
    alpha = np.amin(alpha0)
            
    return w2, er, h2, alpha

@njit
def evolve_rk4_adapt(dt0, t_0, t_f, w0, g_acc, f):
    """
    Evolve a system whose evolution is described by function f from time 
    t_0 to time t_f (a slight overshooting of t_f is allowed) using the RK4 
    method and an adaptive time step. 
    The initial vector of positions and velocities is w0. 
    Returns the final vector w and the actual time t when the simulation ends.
    """

    t = t_0
    h = dt0
    w = np.copy(w0)

    while t < t_f:
        alpha = 0.5
        while alpha < 1.0:
            w2, er, h2, alpha = rk4_step_adapt(w, h, g_acc, f)
            if alpha > 10.0:
                alpha = 10.0
            if alpha < 0.1:
                alpha = 0.1
            h = alpha*h
        w = w2 + er                                      
        t += h2
            
    return w, t

@njit
def mod_mid(n, h, w0, f):
    """
    Step across an interval H = n*h using the modified midpoint method.
    Input: n = number of steps
           h = step length
           w0 = vector of initial positions and velocities
           f = function of time derivatives of w
    Returns: vector of final positions and velocities
    Note: n must be at least 2. 
    """
    
    zmm = np.copy(w0) 
    zm = zmm + h*f(zmm)
    
    for m in range(1,n):
        zmm, zm = zm, zmm+2.0*h*f(zm)
    return 0.5*(zm + zmm + h*f(zm))  

@njit
def bs_step(H, k_max, g_acc, w0, f):
    """
    Take a Bulirsch-Stoer step of length H, starting at w0, for function f.
    k_max + 1 = max number of modified midpoint method calls
    g_acc = target absolute fractional accuracy (for the least accurate
            component of w0)
    Returns: success = 1 if the target accuracy is reached for k <= k_max; 
                       0 otherwise
             w = vector of positions and velocities at the end of the B-S step
             k = final index for modified midpoint call, plus 1
    """

    NN = w0.size  
    T1 = np.zeros((1, NN))
    T1[0] = mod_mid(2, 0.5*H, w0, f)  
    frac_err = 1.0e6
    k = 1
    while (k <= k_max) and (frac_err > g_acc):
        T2 = np.zeros((k+1, NN))
        n = 2*(k+1)
        h = H/n
        T2[0] = mod_mid(n, h, w0, f)
        for j in range(k):
           
            T2[j+1] = T2[j] + (T2[j] - T1[j])/(((k+1)/(k-j))**2 - 1)
        frac_err = np.max(np.abs(T2[k] - T2[k-1])/np.fmax(np.abs(T2[k]), \
                                                          np.abs(T2[k-1])))
        w = T2[k]
        T1 = T2
        k += 1

    if frac_err <= g_acc:
        success = 1
    else:
        success = 0
        
    return success, w, k

def evolve_bs(H0, t_0, t_f, w0, g_acc, k_max, k_c1, f):
    """
    Evolve a system whose evolution is described by function f from time 
    t_0 to time t_f (a slight overshooting of t_f is allowed) using the BS 
    method and an adaptive time step. 
    H0 = initial trial step size
    w0 = the initial vector of positions and velocities
    g_acc = target fractional accuracy per step
    k_max + 1 = max number of steps in modified midpoint call
    k_c1 = k_c + 1, where k_c is the critical value of k such that we
           increase the step size if the modified midpoint method reaches
           the target fractional error for k < k_c

    Returns the final vector w and the actual time t when the simulation ends.
    """

    H = H0
    t = t_0
    w = np.copy(w0)
    
    while t < t_f:
        success = 0
        while success == 0:
            success, w_prov, k1 = bs_step(H, k_max, g_acc, w, f)
            if success == 0:
                H = 0.2*H  # decrease B-S step size if B-S step fails
        t = t + H  # update t following successful B-S step
        w = w_prov  # update w following successful B-S step
        if k1 < k_c1:
            H = 3.0*H  # increase step size since k didn't get too big in
                       # B-S step
    
    return w, t 

def evolve_rk4_adapt_output(dt0, t_0, t_f, w0, g_acc, t_out, n_out_tot, f):
    """
    Evolve a system whose evolution is described by function f from time 
    t_0 to time t_f using the RK4 method and an adaptive time step. 
    The initial vector of positions and velocities is w0 and the target
    absolute fractional error per step is g_acc. 
    t_out = an array (length n_out_tot) of times at which output is
    to be produced; assume that t_out[0] = t_0 and t_out[n_out_tot-1] = t_f

    Returns the vector w and the cumulative number of steps taken 
    for each time in t_out. 
    """

    t = t_0
    h = dt0
    w = np.copy(w0)
    w_out = np.zeros([n_out_tot, w0.size])
    w_out[0] = w0
    n_step_out = np.zeros(n_out_tot, int)
    n_step_out[0] = 0
    
    n_out = 1
    n_step = 0    

    while t < t_f:
        alpha = 0.5
        h_trunc = 0.5*(t_out[n_out] - t)
        if h_trunc <= h:
            h = h_trunc
            record = 1  
        else:
            record = 0
        while alpha < 1.0:
            w2, er, h2, alpha = rk4_step_adapt(w, h, g_acc, f)
            if alpha > 10.0:
                alpha = 10.0
            if alpha < 0.1:
                alpha = 0.1
            if alpha < 1.0:
                record = 0
            h = alpha*h
        w = w2 + er                                    
        t += h2
        n_step = n_step + 1
        if record == 1:
            w_out[n_out] = w    
            n_step_out[n_out] = n_step
            n_out = n_out + 1

    return w_out, n_step_out

def evolve_bs_output(H0, t_0, t_f, w0, g_acc, k_max, k_c1, t_out, n_out_tot, f):
    """
    Evolve a system whose evolution is described by function f from time 
    t_0 to time t_f using the BS method and an adaptive time step. 
    H0 = initial trial step size
    w0 = the initial vector of positions and velocities
    g_acc = target fractional accuracy per step
    k_max + 1 = max number of steps in modified midpoint call
    k_c1 = k_c + 1, where k_c is the critical value of k such that we
           increase the step size if the modified midpoint method reaches
           the target fractional error for k < k_c
    t_out = an array (length n_out_tot) of times at which output is
    to be produced; assume that t_out[0] = t_0 and t_out[n_out_tot-1] = t_f

    Returns the vector w and and the cumulative number of steps taken 
    for each time in t_out.
    """

    H = H0
    t = t_0
    w = np.copy(w0)
    w_out = np.zeros([n_out_tot, w0.size])
    w_out[0] = w0
    n_step_out = np.zeros(n_out_tot, int)
    n_step_out[0] = 0
    
    n_out = 1
    n_step = 0    

    while t < t_f:
        success = 0
        H_trunc = t_out[n_out] - t
        if H_trunc <= H:
            H_prev = H
            H = H_trunc
            record = 1  
        else:
            record = 0
        while success == 0:
            success, w_prov, k1 = bs_step(H, k_max, g_acc, w, f)
            if success == 0:
                H = 0.2*H  
                record = 0
        t = t + H  
        w = w_prov  
        n_step = n_step + 1
        if k1 < k_c1:
            H = 3.0*H  

        if record == 1:
            w_out[n_out] = w    
            n_step_out[n_out] = n_step
            n_out = n_out + 1
            H = H_prev  
            
    return w_out, n_step_out

@njit
def f_Nbody_plane(w, m):
    """
    Evaluates the evolution function for the gravitational N-body problem
    with all motion in the x-y plane. 
    Input:
    w = the array of positions and velocities for N bodies:
    w = (x_0, ..., x_(N-1), y_0, ..., y_(N-1), v_x0, ..., v_x(N-1), v_y0, .., 
         v_y(N-1))
    m = the array of masses (in units with G=1) for the N bodies

    Output: 
    f = the array of time derivatives of the elements of w.  
    """

    N = np.int(w.size/4)
    A = np.zeros((N, N, 2))
        
    for j in range(N-1):
        for k in range(j+1, N):
            x_kj = w[k] - w[j]
            y_kj = w[N+k] - w[N+j]
            r_kj3 = (x_kj**2 + y_kj**2)**1.5
            qx = x_kj/r_kj3
            qy = y_kj/r_kj3
            A[j, k, 0] = m[k]*qx
            A[j, k, 1] = m[k]*qy
            A[k, j, 0] = - m[j]*qx
            A[k, j, 1] = - m[j]*qy
    
    f = np.zeros(4*N)
    for j in range(N):
        f[j] = w[2*N+j]
        f[N+j] = w[3*N+j]
        f[2*N+j] = np.sum(A[j, :, 0])
        f[3*N+j] = np.sum(A[j, :, 1])
        
    return f

@njit
def init_zero(x, y, v_x, v_y, m):
    """
    Assigns values for the position and velocity of the body with index
    j=0 so as to enforce the condition that the position and velocity of the
    CM are both zero. 
    m is the array of masses. 
    x, y, v_x, v_y are arrays with length N (number of bodies). Their
    values for index j=1 through N-1 are the position and velocity components
    of the bodies. On input, the values for index j=0 are meaningless. On
    output, they contain the desired initial values for the j=0 body.
    """

    x[0] = -(np.sum(x[1:]*m[1:]))/m[0]
    y[0] = -(np.sum(y[1:]*m[1:]))/m[0]
    v_x[0] = -(np.sum(v_x[1:]*m[1:]))/m[0]
    v_y[0] = -(np.sum(v_y[1:]*m[1:]))/m[0]

    return x, y, v_x, v_y

@njit
def r_and_v_CM(w, m):
    """
    Find the position and velocity of the center of mass
    """

    MM = np.sum(m)
    N = np.int(w.size/4)
    x = w[0:N]
    y = w[N : 2*N]
    v_x = w[2*N : 3*N]
    v_y = w[3*N : 4*N]
    
    x_cm = np.sum(x*m)/MM
    y_cm = np.sum(y*m)/MM
    vx_cm = np.sum(v_x*m)/MM
    vy_cm = np.sum(v_y*m)/MM
    
    return x_cm, y_cm, vx_cm, vy_cm

@njit
def elements_to_cartesian(m, a, e, theta0, theta, sense):
    """
    Convert orbital elements to Cartesian position and velocity components
    for all bodies except that with j=0 (the "primary")
    Input: all are arrays; the j=0 element is only meaningful for m
    m = mass  (units such that G=1)
    a = semimajor axis 
    e = eccentricity
    theta0 = polar angle of pericenter
    theta = polar angle of body
    sense = +1 for prograde or -1 for retrograde
    Output arrays: x, y, v_x, v_y
    """

    M = m[0] + m
    rat = m[0]/M
    a_r = a/rat
    f1 = a_r*(1.0-e**2)
    f2 = 1.0 + e*np.cos(theta - theta0)
    ct = np.cos(theta)
    st = np.sin(theta)
    r = f1 / f2
    x_r = r*ct
    y_r = r*st
    theta_dot = sense * np.sqrt(M*f1)/(r**2)
    r_dot = r*theta_dot*e*np.sin(theta-theta0)/f2
    v_x_r = r_dot*ct - r*theta_dot*st
    v_y_r = r_dot*st + r*theta_dot*ct

    x = x_r*rat
    y = y_r*rat
    v_x = v_x_r*rat
    v_y = v_y_r*rat
    
    return x, y, v_x, v_y

@njit
def cartesian_to_elements(m, x, y, v_x, v_y):
    """
    Convert Cartesian position and velocity components to orbital elements
    for all bodies except that with j=0 (the "primary")
    Input: all are arrays; the j=0 element is only meaningful for m
    m = mass  (units such that G=1)
    Output arrays: 
    a = semimajor axis 
    e = eccentricity
    theta0 = polar angle of pericenter
    theta = polar angle of body
    sense = +1 for prograde or -1 for retrograde
    If e = 0, then theta0 is undefined; returns zero in this case.
    """

  
    
    M = m[0] + m
    rat = m[0]/M
    x_r = x/rat    
    y_r = y/rat
    vx_r = v_x/rat
    vy_r = v_y/rat

    r_r = np.sqrt(x_r**2 + y_r**2)
    cos_theta = x_r/r_r
    sin_theta = y_r/r_r
    v2_r = vx_r**2 + vy_r**2
    a_r = 1.0/((2.0/r_r) - (v2_r/M)) 
    a = a_r*rat    
    theta_dot = (vy_r*cos_theta - vx_r*sin_theta)/r_r

    N = m.size
    theta = np.zeros(N)
    sense = np.zeros(N)
    e = np.zeros(N)
    theta0 = np.zeros(N)

    for i in range(N):
        theta[i] = np.arctan2(sin_theta[i], cos_theta[i])
        sense[i] = int(np.sign(theta_dot[i]))
        u = 1.0 - (r_r[i]**4)*(theta_dot[i]**2)/(M[i]*a_r[i])
        if (u <= 0.0):
            e[i] = 0.0
        else:
            e[i] = np.sqrt(u)  
        if (e[i] > 0.0):
            r_dot = vx_r[i]*cos_theta[i] + vy_r[i]*sin_theta[i]
            ctt0 = ((a_r[i]/r_r[i])*(1.0-e[i]**2) - 1.0)/e[i]
            stt0 = r_dot*(1.0 + e[i]*ctt0)/(e[i]*r_r[i]*theta_dot[i])
                  
            theta0[i] = theta[i] - np.arctan2(stt0, ctt0)
            if (theta0[i] < (-1.0*np.pi)):
                theta0[i] = theta0[i] + 2.0*np.pi
            if (theta0[i] > np.pi):
                theta0[i] = theta0[i] - 2.0*np.pi    
    
    return a, e, theta0, theta, sense

def total_energy (w, m):
    """
    Find the total kinetic energy K and the total potential energy U
    given the array m of masses and the array w of position and velocity
    components .
    """
    
    n = int(w.size/4)

    x = w[0 : n]
    y = w[n: 2*n]

    v_x = w[2*n : 3*n]
    v_y = w[3*n : 4*n]

    K = 0.5 * np.dot(m, (v_x**2 + v_y**2))

    U = 0.0

    for i in range (0, n - 1):
        for j in range (i+1 , n):
            x1 = w[j] - w[i]
            y1 = w[n + j] - w[n + i]

            r = (x1**2 + y1**2)**(0.5)

            U = U + (m[i] * m[j])/r

    U = -1 * U

    return K, U


    
