# -*- coding: utf-8 -*-
import numpy as np

"""
List of function used by the algorithm
"""

def sigmoide(t, args):
    """
    Sigmoide function

    Q(t;Qmax,ts, tau) = Qmax / (1+ exp((-t-ts)/tau))

    Parameters
    ----------
    t: int or float
        Time.
    args : List
        Arguments of the function.

    Returns
    -------
    Q(t;args).

    """
    Qmax, ts, tau = args
    return Qmax/(1+np.e**(-(t-ts)/tau))

def grad_sigmoide(t, args):
    """
    Sigmoide function

    dQ(t;Qmax,ts, tau)/dQmax = 1 / (1+ exp((-t-ts)/tau))

    dQ(t;Qmax,ts, tau)/dt* = - (1 / tau) * Qmax / (1+ exp(-(t-ts)/tau))^2

    dQ(t;Qmax,ts, tau)/dtau = (t-ts)/tau * Qmax / (1+ exp(-(t-ts)/tau))

    Parameters
    ----------
    t: int or float
        Time.
    args : List
        Arguments of the function.

    Returns
    -------
    grad Q(t;args).

    """
    Qmax, ts, tau = args

    dQdQmax = 1/(1+np.e**(-(t-ts)/tau))

    dQdts = (-1/tau) * Qmax/(1+np.e**(-(t-ts)/tau))**2

    dQdtau = ((t-ts)/tau) * Qmax/(1+np.e**(-(t-ts)/tau))**2

    return np.array([dQdQmax, dQdts, dQdtau])

def least_square(data,t_start, Dt, func, args):
    """
    J(arg) = Dt || func(args) - data ||^2

    Parameters
    ----------
    Dt : float
        Time step.
    func : Function
        Function of modelisation.
    args : List
        Arguments of the function.

    Returns
    -------
    J(args).
    """
    J = 0
    for n, d_n in enumerate(data):
        J += (func(t_start + n*Dt,args)-d_n)**2

    return Dt*J

def noised_sigmoide(noise, Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    t = np.arange(t_start, t_end, 1)
    sig = sigmoide(t, (Qmax, ts, tau))

    noised_sig = [sig[0]]

    for k in range(len(sig)-1):
        rd = np.random.normal(noised_sig[k] + sig[k+1]-sig[k] ,noise*(sig[k+1]-sig[k])/5)

        clip_rd = np.clip(rd,noised_sig[k],np.inf)
        noised_sig.append(clip_rd)

    return noised_sig