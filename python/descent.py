# -*- coding: utf-8 -*-
from functions import least_square, sigmoide, grad_least_square, grad_sigmoide, noised_sigmoide
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

def descentRamijo_sigmoide(data, init_args, t_start=0, eps = 0.001, alpha_max= 10**-6, reb = 0.1, omega = 0.5, Niter = 5000): # ne fonctionne qu'avec un dt assez petit
    
    # initial guess for the parameter to optimize...
    Smax_init, tmid_init, tau_init  = init_args
    theta                           = np.array([Smax_init, tmid_init, tau_init])
    
    # Corresponding value for the criterion...
    Ls      = least_square(data, t_start, alpha_max, sigmoide, theta)
    
    # Initilization of the vector storing the criterion values...
    F       = [Ls]

    # Corresponding value for the gradient... 
    grad            = grad_least_square(data, t_start, alpha_max, sigmoide, grad_sigmoide, theta)
    norm_grad_init  = np.linalg.norm(grad)
    deltaF          = [] # variation of the criterion 


    niter = 1
    print(np.linalg.norm(grad))
    
    while np.linalg.norm(grad) > eps*norm_grad_init and niter < Niter:
        grad    = grad_least_square(data, t_start, alpha_max, sigmoide, grad_sigmoide, theta) 
        
        # Descent direction...
        d       = -grad
        
        # initial step-size to be tested...
        alpha   = alpha_max

        # initial update to be tested...
        theta   = theta + alpha*d
        
        # array of the values 
        Ls = least_square(data, t_start, alpha, sigmoide, theta)
        F.append(Ls)
        
        # Criterion difference for the tested update...
        deltaF  = F[-2] - F[-1]
        
        # scalar product involved in the Arùijo rule...
        prodScal = np.dot(grad,d)
        
        # line-search based on the Armijo inquality [REF]
        while deltaF < -alpha*omega*prodScal:
            # Armijo inequality does not hold => We need to reduce the step-size...
            alpha = alpha*reb
            # New update to be tested...
            theta_new = theta + alpha*d # addition car d est la direction de descente
            # New crietrion difference...
            deltaF  = F[-2] - least_square(data, t_start, alpha, sigmoide, theta_new)
            
        niter += 1
        print(theta, np.linalg.norm(grad), F[-1])
        if niter%100==0:
            print(theta, np.linalg.norm(grad), F[-1])

    return theta, F

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
    
def opti_sigmoide_gen(init_args, optiFunc):
    data = noised_sigmoide(0,Qmax=100, ts=30, tau=6, t_start=0, t_end=60)
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
init_args_france_Sigmoide = (90,25,5)


# opti_sigmoide("FRA",init_args_france_Sigmoide, descentRamijo_sigmoide) # ça ne fonctionne pas selon les params de départ
opti_sigmoide_gen(init_args_france_Sigmoide, descentRamijo_sigmoide)
