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

