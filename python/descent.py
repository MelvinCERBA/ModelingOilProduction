# -*- coding: utf-8 -*-
from functions import least_square, sigmoide, grad_least_square, grad_sigmoide
from data_processing_Sig import Data_processing
import numpy as np
import matplotlib.pyplot as plt

def descent_sigmoide(data, init_args, t_start=0, dt=10**-8, eps = 0.1): # ne fonctionne qu'avec un dt assez petit
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

def test_courbe_sigmoide(location, init_args):
    data = Data_processing(location).get_data()
    theta, F = descent_sigmoide(data, init_args)

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


test_courbe_sigmoide("FRA",init_args_france_Sigmoide)

