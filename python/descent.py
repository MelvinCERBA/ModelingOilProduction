# -*- coding: utf-8 -*-
from functions import least_square, sigmoide, grad_least_square, grad_sigmoide
from data_processing_Sig import Data_processing
import numpy as np
import matplotlib.pyplot as plt

def descent_sigmoide(data, init_args, t_start=0, dt=10**-6, eps = 0.1): # ne fonctionne qu'avec un dt assez petit
    Smax_init, tmid_init, tau_init = init_args
    theta = np.array([Smax_init, tmid_init, tau_init])
    

    F = [least_square(data, t_start, dt, sigmoide, theta)]

    n = 0
    grad = grad_least_square(data, t_start, dt, sigmoide, grad_sigmoide, theta)


    while np.linalg.norm(grad)>eps and n<10000:
        grad = grad_least_square(data, t_start, dt, sigmoide, grad_sigmoide, theta)
        theta = theta - dt*grad
        F.append(least_square(data, t_start, dt, sigmoide, theta))
        n += 1
        #print(theta, np.linalg.norm(grad), F[-1])
        if n%100==0:
            print(theta, np.linalg.norm(grad), F[-1])

    return theta, F

def descentRamijo_sigmoide(data, init_args, t_start=0, dt=10**-6, eps = 0.1, alpha_max= 1, reb = 0.5, omega = 0.5, Niter = 200): # ne fonctionne qu'avec un dt assez petit
    Smax_init, tmid_init, tau_init = init_args
    theta = np.array([Smax_init, tmid_init, tau_init])
    
    Ls = least_square(data, t_start, dt, sigmoide, theta)
    F = [Ls]
    F_light = [Ls] # same as F, but not appended in the second loop
    grad = grad_least_square(data, t_start, dt, sigmoide, grad_sigmoide, theta)
    deltaF = [] # variation of the criterion 

    n = 1
    while np.linalg.norm(grad)>eps and n<Niter:
        d = -grad
        alpha = alpha_max
        theta = theta - dt*grad
        
        # array of the values 
        Ls = least_square(data, t_start, dt, sigmoide, theta)
        F_light.append(Ls)
        F.append(Ls)
        deltaF.append(F[-2]-F[-1])
        
        prodScal = np.dot(grad,d)
        
        while deltaF[-1] < -alpha*omega*prodScal:
            alpha = alpha*reb
            theta_new = theta + alpha*d # addition car d est la direction de descente
            F.append(least_square(data, t_start, dt, sigmoide, theta_new))
            deltaF.append(F[-2]-F[-1])
            
        n += 1
        print(theta, np.linalg.norm(grad), F[-1])
        if n%100==0:
            print(theta, np.linalg.norm(grad), F[-1])

    return theta, F_light

def opti_sigmoide(location, init_args, optiFunc):
    data = Data_processing(location).get_data()
    theta, F = optiFunc(data, init_args)

    X, Y = [k for k in range(0,len(data))],[data]

    plt.figure()

    plt.subplot(211)
    plt.plot(F)

    # plots the data
    plt.subplot(212)
    plt.scatter(X,Y,marker="+", color="red")
    
    #plots the sigmoide
    t = np.linspace(X[0],X[-1],1000)
    plt.plot(t,sigmoide(t-X[0], theta))

    plt.show()
    

    
""" 
        MAIN
"""

#test_courbe_Hubbert("oil_france",init_args_france)

""" Optimization using the sigmoid """
init_args_france_Sigmoide = (75000,10,1)


opti_sigmoide("FRA",init_args_france_Sigmoide, descentRamijo_sigmoide)

