# -*- coding: utf-8 -*-

import numpy as np

# Hubbert curve

def Hubbert_curve(t, args):
    a, b, tau = args
    r1 = a*b/tau
    return r1 * np.e**(-t/tau) / (1+b*np.e**(-t/tau))**2
