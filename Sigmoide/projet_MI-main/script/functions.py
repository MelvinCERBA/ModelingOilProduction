# -*- coding: utf-8 -*-

import numpy as np

# Hubbert curve

def Hubbert_curve(t, args):
    a, b, tau = args
    r1 = a*b/tau
    return r1 * np.e**(-t/tau) / (1+b*np.e**(-t/tau))**2

def gradient_Hubbert_curve(t, args):
    a, b, tau = args

    c_a = a*np.e**(-t/tau)/ (tau*(1+b*np.e**(-t/tau)))
    c_b = a*np.e**(-t/tau) * (1-b*np.e**(-t/tau))/ (tau*(1+b*np.e**(-t/tau))**3)
    c_tau = -a*b*np.e**(t/tau)*(b*(tau+t)+(tau-t)*np.e**(t/tau))/(tau*tau*tau*(b+np.e**(t/tau))**3)
    return np.array([c_a, c_b, c_tau])

"""
    Q function (hubbert curve's integral)
-> sert à tracer Q dans courbe_sigmöide
                  
input:
    tstart
    tstop
    dt
    theta (Qmax, tmid, tau)
    
output:
    Q(t; theta)
"""

def Q(t, args):  # tested 
    Qmax, tmid, tau = args
    exp = np.e**(-(t-tmid)/tau) 
    res = Qmax*1/(1+exp)
    return res

"""
    logistic function (hubbert curve's integral)
-> returns the the values of Q(tn) for n in [[0, N-1]]
                  
input:
    tstart
    tstop
    dt
    theta (Qmax, tmid, tau)
    
output:
    {Qn}0->N-1
"""

def Logistic(start, stop, dt, theta): # not tested
    L=[]
    for i in range (start, stop, dt):
        L+=[Q(i, theta)]
    return L

