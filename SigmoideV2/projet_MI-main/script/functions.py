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
-> used to plot Q in courbe_sigmöide
                  
input:
    t
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
    gradQ function (gradient of hubbert curve's integral
                  
input:
    tstart
    tstop
    dt
    theta (Qmax, tmid, tau)
    
output:
    grad(Q(t; theta))
"""

def gradQ(t, args):
    Qmax, tmid, tau = args
    
    r = -(t-tmid)/tau
    
    d_Qmax = 1/(1+np.e**(r))
    d_tmid = (-Qmax/tau)*np.e**(r)/(1+np.e**(r)**2)
    d_tau = (-Qmax/(tau**2))*(np.e**(r)*(t-tmid))/((1+np.e**(r))**2)
    
    return np.array([d_Qmax, d_tmid, d_tau])

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

