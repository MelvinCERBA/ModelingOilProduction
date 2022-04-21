# -*- coding: utf-8 -*-
from functions import least_square, sigmoide, grad_least_square, grad_sigmoide, noised_sigmoide, scale_matrix
from data_processing_Sig import Data_processing
import numpy as np
import matplotlib.pyplot as plt





def descent(data, init_args, t_start=0, dt=10**-6, eps = 0.1, Niter= 10000): # ne fonctionne qu'avec un dt assez petit
# =============================================================================
#     Basic gradient descent applied to the sigmoide
#
# input:
#   data        : vector of the data to be modelized
#   init_args   : initial guess for the parameters to optimize
#   t_start     : first value of t 
#   dt          : size of the step
#   eps         : scale used to decide when the gradient is small enough
#   Niter       : maximum number of iterations
# 
# output:
#   theta       : optimized parameters
#   F           : successive values of the criterion
# =============================================================================

    # initial guess for the parameter to optimize...
    Smax_init, tmid_init, tau_init      = init_args
    theta                               = np.array([Smax_init, tmid_init, tau_init])
    
    # Corresponding value for the criterion...
    F       = [least_square(data, t_start, dt, sigmoide, theta)]

    # Corresponding value of the gradient...
    grad    = grad_least_square(data, t_start, dt, sigmoide, grad_sigmoide, theta)

    niter       = 0
    while np.linalg.norm(grad) > eps and niter < Niter:
        
        # update of the gradient
        grad        = grad_least_square(data, t_start, dt, sigmoide, grad_sigmoide, theta)
        
        # update of the parameters
        theta       = theta - dt*grad
        
        # new value of the criterion
        F.append(least_square(data, t_start, dt, sigmoide, theta))
        
        niter   += 1
        
        #print(theta, np.linalg.norm(grad), F[-1])
        if niter%100 == 0:
            print(theta, np.linalg.norm(grad), F[-1])

    return theta, F



def descentArmijo(data, init_args, t_start=0, eps = 0.001, Niter = 5000, alpha_max= 10**-6, reb = 0.1, omega = 0.5): # ne fonctionne qu'avec un dt assez petit
# =============================================================================
#     Gradient descent with Armijo rule applied to the sigmoide
#
# input:
#   data        : vector of the data to be modelized
#   init_args   : initial guess for the parameters to optimize
#   t_start     : first value of t 
#   eps         : scale used to decide when the gradient is small enough
#   Niter       : maximum number of iterations
#   alpha_max   : initial size of the step
#   reb         : rate at wich the size of the step is reduced
#   omega       : scale used to decide when deltaF is small enough
# 
# output:
#   theta       : optimized parameters
#   F           : successive values of the criterion
# =============================================================================    


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
        
        # Descent scale and direction...
        d       = -grad
        
        # initial step-size to be tested...
        alpha   = alpha_max

        # initial update to be tested...
        theta   = theta + alpha*d
        
        # array of the values 
        Ls = least_square(data, t_start, alpha, sigmoide, theta)
        F.append(Ls)
        
        # Criterion difference for the tested update...
        deltaF      = F[-2] - F[-1]
        
        # scalar product involved in the Arùijo rule...
        prodScal    = np.dot(grad,d)
        
        # line-search based on the Armijo inquality [REF]
        while deltaF < -alpha*omega*prodScal:
            # Armijo inequality does not hold => We need to reduce the step-size...
            alpha       = alpha*reb
            # New update to be tested...
            theta_new = theta + alpha*d # addition car d est la direction de descente
            # New crietrion difference...
            deltaF      = F[-2] - least_square(data, t_start, alpha, sigmoide, theta_new)
            
        niter += 1
        print(theta, np.linalg.norm(grad), F[-1])
        if niter%100==0:
            print(theta, np.linalg.norm(grad), F[-1])

    return theta, F



def descentScaled(data, init_args, t_start=0, eps = 0.001, Niter = 100, alpha_max= 1, reb = 0.1, omega = 0.5): # ne fonctionne qu'avec un dt assez petit
# =============================================================================
#     Gradient descent with Armijo rule and scaling of the descent direction, applied to the sigmoide
#
# input:
#   data        : vector of the data to be modelized
#   init_args   : initial guess for the parameters to optimize
#   t_start     : first value of t 
#   eps         : scale used to decide when the gradient is small enough
#   Niter       : maximum number of iterations
#   alpha_max   : initial size of the step
#   reb         : rate at wich the size of the step is reduced
#   omega       : scale used to decide when deltaF is small enough
# 
# output:
#   theta       : optimized parameters
#   F           : successive values of the criterion
# =============================================================================    

    # initial guess for the parameter to optimize...
    Smax_init, tmid_init, tau_init  = init_args
    theta                           = np.array([Smax_init, tmid_init, tau_init])
    
    # Corresponding value for the criterion...
    Ls      = least_square(data, t_start, alpha_max, sigmoide, theta)
    # print("ls = ", Ls)
    
    # Initilization of the vector storing the criterion values...
    F       = [Ls]
    

    # Corresponding value for the gradient... 
    grad            = grad_least_square(data, t_start, alpha_max, sigmoide, grad_sigmoide, theta)
    norm_grad_init  = np.linalg.norm(grad)
    deltaF          = [] # variation of the criterion 


    niter = 1
    #print(np.linalg.norm(grad))
    
    while np.linalg.norm(grad) > eps*norm_grad_init and niter < Niter and F[-1] > 1: # added last condition not to loop 100 times when testing generated data with no noise and perfect args
        grad        = grad_least_square(data, t_start, alpha_max, sigmoide, grad_sigmoide, theta) 
        
        # Descent scale...
        scaler      = scale_matrix(t_start, len(data)-1, grad_sigmoide, theta)
        
        # Descent direction...
        d           = -np.matmul(scaler,grad.transpose())
        # print(scaler, -grad, d)
        
        # initial step-size to be tested...
        alpha       = alpha_max

        # initial update to be tested...
        theta       = theta + alpha*d
        
        # initial value of the criterion... 
        Ls          = least_square(data, t_start, alpha, sigmoide, theta)
        F.append(Ls)
        
        # Criterion difference for the tested update...
        deltaF      = F[-2] - F[-1]
        
        # scalar product involved in the Armijo rule...
        prodScal    = np.dot(grad,d)
        
        # line-search based on the Armijo inquality [REF]
        while (deltaF < -alpha*omega*prodScal):
            # Armijo inequality does not hold => We need to reduce the step-size...
            alpha       = alpha*reb
            # New update to be tested...
            theta_new   = theta + alpha*d # addition car d est la direction de descente
            # New criterion difference...
            try:
                deltaF      = F[-2] - least_square(data, t_start, alpha, sigmoide, theta_new)
            except TypeError:
                continue
        niter += 1
        # print(theta, np.linalg.norm(grad), F[-1])
        if niter%100==0:
            print(theta, np.linalg.norm(grad), F[-1])
        
    return theta, F


