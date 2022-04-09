# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoide, grad_sigmoide, least_square

def plot_sigmoide(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t = np.arange(t_start, t_end, 10000)
    sig = sigmoide(t, (Qmax, ts, tau))

    plt.figure()
    plt.plot(t,sig)
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

    T = np.arange(t_start, t_end, 10000)
    for t in T:
        dQdQmax = (sigmoide(t, (Qmax+delta, ts, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdts = (sigmoide(t, (Qmax, ts+delta, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdtau = (sigmoide(t, (Qmax, ts, tau+delta))-sigmoide(t, (Qmax, ts, tau)))/delta

        grad_sig = grad_sigmoide(t, (Qmax, ts, tau))

        diff = abs(grad_sig - np.array([dQdQmax, dQdts, dQdtau]))

        if diff_max is None or np.linalg.norm(diff_max)<np.linalg.norm(diff):
            diff_max = diff

    return diff_max