# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoide, noised_sigmoide, grad_sigmoide, least_square,grad_least_square
from mpl_toolkits.axes_grid.axislines import SubplotZero
from descent import descent, descentArmijo, descentScaled
from data_processing_Sig import Data_processing
import time

def plot_sigmoide(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t       = np.arange(t_start, t_end, 1)
    sig     = sigmoide(t, (Qmax, ts, tau))

    plt.figure()
    plt.plot(t,sig)
    plt.show()

def plot_noised_sigmoide(noise,Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t           = np.arange(t_start, t_end, 1)
    noised_sig  = noised_sigmoide(noise,Qmax=100, ts=30, tau=6, t_start=0, t_end=60)

    plt.figure()
    plt.plot(t,noised_sig)
    plt.show()

def test_least_square(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    t       = np.linspace(t_start, t_end, 10000)
    sig     = sigmoide(t, (Qmax, ts, tau))

    ls      = least_square(sig, t_start, t[1]-t[0], sigmoide, (Qmax, ts, tau))

    print("==test least square==")
    print("\t-least_square = ", ls)
    print("test result : ")
    return ls == 0

def test_grad_sigmoide(delta, Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Compare the gradient of the sigmoide function to a finite differences gradient made with the sigmoide function

    """
    diff_max = None

    T                   = np.arange(t_start, t_end, 1)
    
    LQmax, Lts, Ltau    = [],[],[]
    
    Mgrad               = np.zeros((t_end-t_start,3))
    
    for k,t in enumerate(T):
        dQdQmax     = (sigmoide(t, (Qmax+delta, ts, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdts       = (sigmoide(t, (Qmax, ts+delta, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdtau      = (sigmoide(t, (Qmax, ts, tau+delta))-sigmoide(t, (Qmax, ts, tau)))/delta
        

        grad_sig    = grad_sigmoide(t, (Qmax, ts, tau))
        
        LQmax.append(dQdQmax)
        Lts.append(dQdts)
        Ltau.append(dQdtau)
        Mgrad[k] = grad_sig

        diff = abs(grad_sig - np.array([dQdQmax, dQdts, dQdtau]))

        if diff_max is None or np.linalg.norm(diff_max)<np.linalg.norm(diff):
            diff_max = diff
            
    plt.figure()
    plt.plot(T,LQmax)
    plt.plot(T,Mgrad[:,0])
    plt.show()
    
    plt.figure()
    plt.plot(T,Lts, label = "diff fini")
    plt.plot(T,Mgrad[:,1])
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(T,Ltau, label = "diff fini")
    plt.plot(T,Mgrad[:,2])
    plt.legend()
    plt.show()
    

    return diff_max

def test_grad_least_square(delta, Qmax=100, ts=10, tau=6, t_start=0, t_end=50):
    """
    Compare the gradient of the least square function to a finite differences gradient made with the least square function

    """

    t = np.arange(t_start, t_end, 1)
    data = sigmoide(t,(Qmax+10,ts,tau))

    least_square_origine = least_square(data, t_start, 1, sigmoide, (Qmax, ts, tau))

    dJdQmax     = (least_square(data, t_start, 1, sigmoide,(Qmax+delta/2, ts, tau))-
                           least_square(data, t_start, 1, sigmoide,(Qmax-delta/2, ts, tau)))/delta

    dJdts       = (least_square(data, t_start, 1,sigmoide, (Qmax, ts+delta, tau))-least_square_origine)/delta

    dJdtau      = (least_square(data, t_start, 1,sigmoide,(Qmax, ts, tau+delta))-least_square_origine)/delta

    grad_sig = grad_least_square(data, t_start, 1,sigmoide, grad_sigmoide,(Qmax, ts, tau))

    return grad_sig , np.array([dJdQmax, dJdts, dJdtau]),np.array([dJdQmax, dJdts, dJdtau])/grad_sig


def plot_isocurve_Qmax_fixed(percentage, Qmax=1, ts_init=50, tau_init=6, t_start=0, t_end=200):

    # Sigmoide
    t = np.arange(t_start, t_end, 1)
    sig = sigmoide(t, (Qmax, ts_init, tau_init))

    # window of value
    n = 10**2
    L_ts = np.linspace((1-percentage)*ts_init,(1+percentage)*ts_init,n)
    L_tau = np.linspace(np.clip((1-percentage)*tau_init,0.1,(1-percentage)*tau_init),(1+percentage)*tau_init,n)

    # Matrix of potential
    M = np.zeros((n,n))

    for i,ts in enumerate(L_ts):
        for j,tau in enumerate(L_tau):
            M[i,j] = least_square(sig, t_start, 1, sigmoide, (Qmax, ts, tau))

    # Number of isocurve
    N_isocurve = least_square

    potentials = [M[k*(n//2)//N_isocurve +n//2-1,n//2] for k in range(N_isocurve)]

    np.sort([M[n//2,n//2]] + potentials)

    Isocurve_i = [[] for _ in range(N_isocurve)]

    Isocurve_j = [[] for _ in range(N_isocurve)]

    # Seeking for isocurve
    for i in range(n):
        for j in range(n):
            for k,potential in enumerate(potentials[1:]):
                if abs(M[i,j]-potential)<(potentials[k+1]-potentials[k])*2*N_isocurve/n:
                    Isocurve_i[k].append(L_tau[i])  # y coordinate of the point of the isocurve
                    Isocurve_j[k].append(L_ts[j])   # x coordinate


    plt.figure()
    # plt.pcolormesh(L_ts,L_tau,M)
    for k in range(N_isocurve):
        plt.scatter(Isocurve_j[k],Isocurve_i[k],marker=".", label=f'isocourbe {k+1}')
    
    # Plots the center "+" 
    plt.plot(L_ts[n//2],L_tau[n//2],"+")
    
    # Titles the axes
    ax = plt.axes()
    ax.set_ylabel(r'$\tau$')
    ax.set_xlabel(r'$t_*$')
    
    # plt.legend()
    plt.show()

def plot_isocurve_ts_fixed(percentage, Smax_init=1, ts=50, tau_init=6, t_start=0, t_end=200):

    # Sigmoide
    t = np.arange(t_start, t_end, 1)
    sig = sigmoide(t, (Smax_init, ts, tau_init))

    # window of value
    n = 2*10**3
    L_Smax = np.linspace((1-percentage)*Smax_init,(1+percentage)*Smax_init,n)
    L_tau = np.linspace(np.clip((1-percentage)*tau_init,0.1,(1-percentage)*tau_init),(1+percentage)*tau_init,n)

    # Matrix of potential
    M = np.zeros((n,n))

    for i,Smax in enumerate(L_Smax):
        for j,tau in enumerate(L_tau):
            M[i,j] = least_square(sig, t_start, 1, sigmoide, (Smax, ts, tau))

    # Number of isocurve
    N_isocurve = 4

    potentials = [M[k*(n//2)//N_isocurve +n//2-1,n//2] for k in range(N_isocurve)]

    np.sort([M[n//2,n//2]] + potentials)

    Isocurve_i = [[] for _ in range(N_isocurve)]
    Isocurve_j = [[] for _ in range(N_isocurve)]

    # Seeking for isocurve
    for i in range(n):
        for j in range(n):
            for k,potential in enumerate(potentials[1:]):
                if abs(M[i,j]-potential)<(potentials[k+1]-potentials[k])*2*N_isocurve/n:
                    Isocurve_i[k].append(L_tau[i])
                    Isocurve_j[k].append(L_Smax[j])


    plt.figure()
    # plt.pcolormesh(L_ts,L_tau,M)
    for k in range(N_isocurve):
        plt.scatter(Isocurve_j[k],Isocurve_i[k],marker="_", label=f'isocourbe {k+1}')
    
    # Plots the center "+" 
    plt.plot(L_Smax[n//2],L_tau[n//2],"+")
    
    # Titles the axes
    ax = plt.axes()
    ax.set_ylabel(r'$\tau$')
    ax.set_xlabel(r'$S_{max}$')
    
    plt.legend()
    plt.show()

def plot_isocurve_tau_fixed(percentage, Smax_init=1, ts_init=50, tau=6, t_start=0, t_end=200):

    # Sigmoide
    t = np.arange(t_start, t_end, 1)
    sig = sigmoide(t, (Smax_init, ts_init, tau))

    # window of value
    n = 2*10**3
    L_Smax = np.linspace((1-percentage)*Smax_init,(1+percentage)*Smax_init,n)
    L_ts = np.linspace((1-percentage)*ts_init,(1+percentage)*ts_init,n)

    # Matrix of potential
    M = np.zeros((n,n))

    for i,Smax in enumerate(L_Smax):
        for j,ts in enumerate(L_ts):
            M[i,j] = least_square(sig, t_start, 1, sigmoide, (Smax, ts, tau))

    # Number of isocurve
    N_isocurve = 4

    potentials = [M[k*(n//2)//N_isocurve +n//2-1,n//2] for k in range(N_isocurve)]

    np.sort([M[n//2,n//2]] + potentials)

    Isocurve_i = [[] for _ in range(N_isocurve)]

    Isocurve_j = [[] for _ in range(N_isocurve)]

    # Seeking for isocurve
    for i in range(n):
        for j in range(n):
            for k,potential in enumerate(potentials[1:]):
                if abs(M[i,j]-potential)<(potentials[k+1]-potentials[k])*2*N_isocurve/n:
                    Isocurve_i[k].append(L_ts[i])
                    Isocurve_j[k].append(L_Smax[j])


    plt.figure()
    # plt.pcolormesh(L_ts,L_tau,M)
    for k in range(N_isocurve):
        plt.scatter(Isocurve_j[k],Isocurve_i[k],marker="_", label=f'isocourbe {k+1}')
    
    # Plots the center "+" 
    plt.plot(L_Smax[n//2],L_ts[n//2],"+")
    
    # Titles the axes
    ax = plt.axes()
    ax.set_ylabel(r'$t_*$')
    ax.set_xlabel(r'$S_{max}$')
    
    plt.legend()
    plt.show()


def opti_Country(location, init_args, optiFunc = descentScaled):
# =============================================================================
#     Optimization of the parameters of the sigmoide to match the data of a specific country
#input:
#    location        : abreviation of the country (ex: FRA)
#    init_args       : initial guess of the parameters to optimize (should not be too far from the solution)
#    optiFunc        : algorithm to be used for the optimization
#         
#output:
#    plot of F               : shows the evolution of the criterion during the optimization
#    plot of the sigmoid     : plots the sigmoide corresponding to the optimized parameters on top of the data
# =============================================================================
    
    # data of the selected country...
    data        = Data_processing(location).get_data()
    
    # optimized parameters and values of the criterion during the optimization... 
    theta, F    = optiFunc(data, init_args)

    # plots the criterion's values and the optimized sigmoide on top of the data
    plot_F_Data_Sigmoide(F, data, theta)
    
    
    
    
def opti_generatedData(noise, init_args, optiFunc = descentScaled):
# =============================================================================
#     Optimization of the parameters of the sigmoide to match generated data
#input:
#    noise           : level of noise to apply to the data
#    init_args       : initial guess of the parameters to optimize (should not be too far from the solution)
#    optiFunc        : algorithm to be used for the optimization
#         
#output:
#    plots F               : shows the evolution of the criterion during the optimization
#    plots the sigmoid     : plots the sigmoide corresponding to the optimized parameters on top of the data
# =============================================================================
    
    # data with the desired amount of noise...
    data        = noised_sigmoide(noise,Qmax=100, ts=30, tau=6, t_start=0, t_end=60)
    
    # optimized parameters and values of the criterion during the optimization ...
    theta, F    = optiFunc(data, init_args)

    # plots the criterion's values and the optimized sigmoide on top of the data
    plot_F_Data_Sigmoide(F, data, theta)




def plot_F_Data_Sigmoide(F, data, theta):
    
    # years from start (X) and corresponding cumulated production (Y)
    X, Y = [k for k in range(0,len(data))],[data]

    plt.figure()
    
    # plots the successive values of the criterion...
    plt.subplot(211)
    plt.plot(F)

    # plots the data...
    plt.subplot(212)
    plt.scatter(X,Y,marker="+", color="red")
    
    #plots the optimized sigmoide...
    t = np.linspace(X[0],X[-1],1000)
    plt.plot(t,sigmoide(t-X[0], theta))

    plt.show()
    
def testPerf_NoisedData(perfect_args, noise_steps = 3, noise_dt = 10, argsDelta_steps = 3, argsDelta_dt = 0.1, optiFunc = descentScaled):
# =============================================================================
#     monitors the optimization on generated data for different values of noise and for different initial params
# input:
#   perfect_args        : args of the function used to generate data
#   noise_steps         : number of values of noise to be tested
#   noise_dt            : amount of noise to add at each step
#   argsDelta_steps     : number of sets of args to be tested
#   argsDelta_dt        : percentage the init_args to add at each step
#   optiFunc            : descent algorithm to be used for optimization
#
#output:
#   R^3 matrix    : time needed and criterion obtained for each level of noise and set of args tested 
# 
# =============================================================================
    
    # perfect parameters : those of the sigmoide used to generate data
    Qmax_perfect, ts_perfect, tau_perfect   = perfect_args
    perfect_args                            = np.array(perfect_args)

    # all levels of noise to be tested...
    noise_levels    = [ k * noise_dt for k in range( 0, noise_steps+1, 1) ]
    
    # all values of args to be tested...
    args_values     = [ [(1+(k * argsDelta_dt))] * perfect_args for k in range( 0, argsDelta_steps+1, 1)]
    
    # for each level of noise (columns), for each value of args (lines), 
        # we will save here : 1) the criterion obtained 2) the time taken to complete the optomization...
    results         = np.zeros([ len(noise_levels), len(args_values), 2])
    
    
    
    for i in range ( 0, len(noise_levels), 1):
        
        # data to be tested...
        data = noised_sigmoide(noise_levels[i], Qmax_perfect, ts_perfect, tau_perfect)
        
        for j in range ( 0, len(args_values), 1):
            
            # value of the args to be tested...
            init_args           = args_values[j]
            
            # descent algorithm ...
            start               = time.time()
            theta, F            = descentScaled(data, init_args)     
            end                 = time.time()
            
            # saving the last value of the criterion and the time taken by the descent (seconds)...
            results[i,j]        = [F[-1], end-start]
    
    results = np.array(results)
    for i in range(0,noise_steps,1):
        delta       = str(1+(k * argsDelta_dt)
        x = results[:,i]
            
        fig, ax1 = plt.subplots()
            
        ax2 = ax1.twinx()
        ax1.plot(x, y1, label='Criterion')
        ax2.plot(x, y2, 'r', label='Time(seconds)')
            
        ax1.set_xlabel('Noise')
        ax1.set_ylabel('Criterion (least square)')
        ax2.set_ylabel('Time (s)')
        
        ax1.legend()
        ax2.legend()
        plt.title(r'{}'.format(name))
        plt.show()
            
    return results

# =============================================================================
#                                    Testing
# =============================================================================

# inital guess of the parameters of the optimized sigmoide for France's data :
init_args_france = (70000,15,5)
# inital guess of the parameters of the optimized sigmoide for data generated with zero noise :
init_args_gen = (90,25,5)

# opti_Country('FRA', init_args_france)
# opti_generatedData(10, init_args_gen)

testPerf_NoisedData(init_args_gen)

#plot_isocurve_Qmax_fixed(0.9)
#plot_isocurve_ts_fixed(0.9)
# plot_isocurve_tau_fixed(0.9)
# test_least_square()