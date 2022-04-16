# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from funcs import Hubbert_curve,Q

# Define initial parameters
global tau, a, b, N, tmin, tmax, t, Qmax, tmid
tau = 7
a = 90000
b = 30
Hmax = a/(4*tau)

Qmax = a
tmid = tau*np.log(b)
    
N = 10000
tmin = 0
tmax = 50
t = np.linspace(tmin, tmax, N)

def plotData(nameFile, hideScale=False):
    
    # closes any previous plot
    plt.close()
    
    X, Y = [],[]
    with open("../data/{}.csv".format(nameFile),"r") as file:
        for line in file.readlines()[1:]:
            X.append(int(line.split(";")[5]))
            Y.append(float(line.split(";")[6]))
            
    tmin = X[0]
    tmax = X[-1]
    
    # Create the figure and the line that we will manipulate
    fig = plt.figure("Courbe de Hubbert",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
    
    # turns off axis numbers if desired
    if hideScale:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    # plots data
    plt.scatter(X,Y,color = "red", marker = "+")
    
    # Defines window
    ax.set_ylim([0, max(Y)*1.5])
    ax.set_xlim([tmin, tmax])
    ax.set_xlabel('Temps')
    ax.set_ylabel('Production')
    return
            

def plotHubbert(hideScale=True):
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    # function to be plotted
    H = Hubbert_curve
    
    # closes any previous plot
    plt.close()
    
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
    
    # turns off axis numbers if desired
    if hideScale:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # titles the axis
    plt.xlabel("x")
    plt.ylabel(r"$H(x)$")
        
    # Plots the curve
    line, = plt.plot(t, H(t-tmin, (a, b, tau)), lw=2)
    
    # defines the window
    ax.set_ylim([0, 4000])
    ax.set_xlim([tmin, tmax])
    
    plt.show()

def plotHubbertAnnote(hideScale=False):
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    # function to be plotted
    H = Hubbert_curve
    
    # closes any previous plot
    plt.close()
    
    plt.close()
    fig = plt.figure("Courbe de Hubbert",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    plt.xticks([tmid],[r'$\tau ln(b)$'])
    plt.yticks([Hmax],[r'$\frac{a}{4 \tau }$'], fontsize='xx-large') # fontsize ne change rien au latex
    ax.tick_params(axis='both', which='minor', labelsize='large') # Ne change rien
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
    
    # titles the axis
    plt.xlabel("t")
    plt.ylabel(r"$Hubbert_{a,b,\tau}(t)$")
    
    # turns off axis numbers if desired
    if hideScale:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
    # Plots the curve
    line, = plt.plot(t, H(t-tmin, (a, b, tau)), lw=2)
    
    # defines the window
    ax.set_ylim([0, 4000])
    ax.set_xlim([tmin, tmax])
    
    # plots the doted lines
    plt.plot(np.linspace(tmin, tmid, N),[Hmax]*N,'--', color = "red")
    plt.plot([tmid]*N,np.linspace(0, Hmax, N),'--', color = "red")
    
    plt.show()
    
def plotHubbertAndData(hideScale=True):
    f = Hubbert_curve

    # closes any previous plot
    plt.close()
    
    # Data
    nameFile = "oil_france"
    
    X, Y = [],[]
    with open("../data/{}.csv".format(nameFile),"r") as file:
        for line in file.readlines()[1:]:
            X.append(int(line.split(";")[5]))
            Y.append(float(line.split(";")[6]))
    
    # Define initial parameters
    init_tau = 7
    init_a = 4*max(Y)*init_tau
    init_b = np.e**((X[Y.index(max(Y))]-X[0])/init_tau)
    
    n = 1000
    tmin = X[0]
    tmax = X[-1]
    t = np.linspace(tmin, tmax, n)
    
    # Create the figure and the line that we will manipulate
    fig = plt.figure("Courbe de Hubbert",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
            
    # turns off axis numbers if desired
    if hideScale:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    # plots data
    plt.scatter(X,Y,color = "red", marker = "+")
    # plots Hubbert Curve
    line, = plt.plot(t, f(t-tmin, (init_a, init_b, init_tau)), lw=2)
    
    # Defines window
    ax.set_ylim([0, max(Y)*1.5])
    ax.set_xlim([tmin, tmax])
    ax.set_ylabel('Production')
    ax.set_xlabel('Temps')
    
def plotSigmoide(hideScale=True):
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    S = Q
    
    # closes any previous plot
    plt.close()
    
    fig = plt.figure("Sigmoide",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
    
    # turns off axis numbers if desired
    if hideScale:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    # titles the axis
    plt.xlabel("x")
    plt.ylabel(r"$S(x)$")
        
    # Plots the curve
    line, = plt.plot(t, S(t-tmin, (Qmax, tmid, tau)), lw=2)
    
    # defines the window
    ax.set_ylim([0, Qmax*1.2])
    ax.set_xlim([tmin, tmax])
    
    plt.show()

def plotSigmoideAnnote(hideScale=False):
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    # function to be plotted
    S = Q
    
    # closes any previous plot
    plt.close()
    
    fig = plt.figure("Sigmoide",figsize = (5,4))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    for direction in ["right", "top"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(False)
    for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
    
    # titles the axis
    plt.xlabel("t")
    plt.ylabel(r"$Sigmoide_{S_{max},t_*,\tau}(t)$")
        
    # Plots the curve
    line, = plt.plot(t, S(t-tmin, (Qmax, tmid, tau)), lw=2)
    
    # defines the window
    ax.set_ylim([0, Qmax*1.2])
    ax.set_xlim([tmin, tmax])
    
    # plots the doted lines
    plt.xticks([tmid],[r'$t_{*}$'], fontsize=16)
    plt.plot([tmid]*N,np.linspace(0, Qmax/2, N),'--', color = "red")
    
    plt.yticks([Qmax, Qmax/2],[r'$S_{max}$', r'$\frac{S_{max}}{2}$'], fontsize=16)
    plt.plot(t,[Qmax]*N,'--', color = "red")
    plt.plot(np.linspace(0, tmid, N),[Qmax/2]*N,'--', color = "red")

    
    # turns off axis numbers if desired
    if hideScale:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    plt.annotate(r'$\Delta = \frac{t_{*} S_{max}}{\tau} (x-t_{*})+\frac{S_{max}}{2}$', (tmid+3,Qmax/2), fontsize=12)
    
    # plots to curve's tangent at tmid
    plt.plot(t, delta(t-tmin))
    
    plt.show()

def delta(x): # tangente à la sigmoide au point d'inflexion
    global tau, a, b, N, tmin, tmax, t, Qmax, tmid
    return Hubbert_curve(tmid,(a,b,tau))*(x-tmid)+(Qmax/2) # tangente de f en a = f'(a)*(x-a)+f(a) 

#plotHubbert()
#plotSigmoide()

plotHubbertAnnote()
# plotSigmoideAnnote()

#plotData('oil_france', True)
#plotHubbertAndData()
