# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib.transforms import BlendedGenericTransform
from funcs import Hubbert_curve,Q

# Define initial parameters
global tau, a, b, N, tmin, tmax, t, Qmax, tmid
tau = 7
a = 90000
b = 30

Qmax = a/(4*tau)
tmid = tau*np.log(b)
    
N = 10000
tmin = 0
tmax = 50
t = np.linspace(tmin, tmax, N)


def plotHubbertAnnote():
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    # function to be plotted
    H = Hubbert_curve
    
    plt.close()
    fig = plt.figure("Courbe de Hubbert",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    plt.xticks([tmid],[r'$\tau ln(b)$'])
    plt.yticks([Qmax],[r'$\frac{a}{4 \tau }$'], rotation = 180)
    
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
    
    
    plt.xlabel("x")
    plt.ylabel(r"$H(x)$")
        
    # Plots the curve
    line, = plt.plot(t, H(t-tmin, (a, b, tau)), lw=2)
    
    # defines the window
    ax.set_ylim([0, 4000])
    ax.set_xlim([tmin, tmax])
    
    # plots the doted lines
    plt.plot(np.linspace(tmin, tmid, N),[Qmax]*N,'--', color = "red")
    plt.plot([tmid]*N,np.linspace(0, Qmax, N),'--', color = "red")
    
    plt.show()
    

def plotSigmoideAnnote():
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    # function to be plotted
    S = Q
    
    plt.close()
    fig = plt.figure("Courbe de Hubbert",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
    
    
    plt.xlabel("x")
    plt.ylabel(r"$S(x)$")
        
    # Plots the curve
    line, = plt.plot(t, S(t-tmin, (Qmax, tmid, tau)), lw=2)
    
    # defines the window
    ax.set_ylim([0, 4000])
    ax.set_xlim([tmin, tmax])
    
    # plots the doted lines
    plt.xticks([tmid],[r'$t_{*}$'])
    plt.plot([tmid]*N,np.linspace(0, Qmax/2, N),'--', color = "red")
    
    plt.yticks([Qmax],[r'$Q_{max}$'])
    plt.plot(t,[Qmax]*N,'--', color = "red")
    
    plt.yticks([Qmax/2],[r'$\frac{Q_{max}}{2}$'])
    plt.plot(np.linspace(0, tmid, N),[Qmax/2]*N,'--', color = "red")
    
    plt.plot(t, delta(t-tmin))
    
    plt.show()

def delta(x): # tangente à la sigmoide au point d'inflexion
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    return (Hubbert_curve(tmid, (a,b,tau))*(x-tmid))+(Qmax/2) # tangente de f en a = f'(a)*(x-a)+f(a) 

#plotHubbertAnnote()
plotSigmoideAnnote()
print(Hubbert_curve(tmid, (a,b,tau)))