# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def f(z,t):
    alpha = 2
    beta = 0.5
    delta = 0.5
    gamma = 2
    fxd = alpha*z[0]-beta*z[0]*z[1]
    fyd = delta*z[0]*z[1] - gamma*z[1]
    return np.array([fxd, fyd], dtype = float)

def RK4(f, z, t, h):
    k1 = h*f(z, t)
    k2 = h*f(z+k1/2, t+h/2)
    k3 = h*f(z+k2/2, t+h/2)
    k4 = h*f(z+k3,t+h)
    return (k1 + 2*k2 + 2*k3 + k4)/6

tmax = 10
n = 10000
T = np.linspace(0,tmax,n)

Z = np.zeros((n,2))
Z[0] = np.array([2,2], dtype = float)

h = tmax/n
for k in range(1,n):
    Z[k] = Z[k-1] + RK4(f,Z[k-1],k*h,h)


plt.figure()
plt.subplot(211)
plt.plot(T,Z[:,0])
plt.ylabel("x(t)")
plt.subplot(212)
plt.plot(T,Z[:,1])
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()
