# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoide, noised_sigmoide, grad_sigmoide, least_square,grad_least_square
from mpl_toolkits.axes_grid.axislines import SubplotZero

def plot_sigmoide(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t = np.arange(t_start, t_end, 1)
    sig = sigmoide(t, (Qmax, ts, tau))

    plt.figure()
    plt.plot(t,sig)
    plt.show()

def plot_noised_sigmoide(noise,Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t = np.arange(t_start, t_end, 1)
    noised_sig = noised_sigmoide(noise,Qmax=100, ts=30, tau=6, t_start=0, t_end=60)

    plt.figure()
    plt.plot(t,noised_sig)
    plt.show()

def test_least_square(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    t = np.linspace(t_start, t_end, 10000)
    sig = sigmoide(t, (Qmax, ts, tau))

    ls = least_square(sig, t_start, t[1]-t[0], sigmoide, (Qmax, ts, tau))

    print("==test least square==")
    print("\t-least_square = ", ls)
    print("test result : ")
    return ls == 0

def test_grad_sigmoide(delta, Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Compare the gradient of the sigmoide function to a finite differences gradient made with the sigmoide function

    """
    diff_max = None

    T = np.arange(t_start, t_end, 1)
    
    LQmax, Lts, Ltau = [],[],[]
    
    Mgrad = np.zeros((t_end-t_start,3))
    
    for k,t in enumerate(T):
        dQdQmax = (sigmoide(t, (Qmax+delta, ts, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdts = (sigmoide(t, (Qmax, ts+delta, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdtau = (sigmoide(t, (Qmax, ts, tau+delta))-sigmoide(t, (Qmax, ts, tau)))/delta

        grad_sig = grad_sigmoide(t, (Qmax, ts, tau))
        
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

    dJdQmax = (least_square(data, t_start, 1, sigmoide,(Qmax+delta/2, ts, tau))-
               least_square(data, t_start, 1, sigmoide,(Qmax-delta/2, ts, tau)))/delta

    dJdts = (least_square(data, t_start, 1,sigmoide, (Qmax, ts+delta, tau))-least_square_origine)/delta

    dJdtau = (least_square(data, t_start, 1,sigmoide,(Qmax, ts, tau+delta))-least_square_origine)/delta

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

#plot_isocurve_Qmax_fixed(0.9)
#plot_isocurve_ts_fixed(0.9)
# plot_isocurve_tau_fixed(0.9)
# test_least_square()