# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoide
from mpl_toolkits.axes_grid.axislines import SubplotZero
from descent import descentScaled


def model(data, optiFunc=descentScaled):
# =============================================================================
#     Takes any length of data, applies the descent to it and returns the optimized sigmoide
# input:
#     data        : data to be modelized
#     optuFunc    : optimization function to be used 
# output:
#     theta       : parameters of the optimal sigmoide
#     F           : successive values of the criterion
# =============================================================================
    Smax_init   = max(data)
    ts_init     = len(data)//2
    tau_init    = 6.5 # value of tau used when testing the descent on france's data. It's a bit arbitrary, but it seems to work
    
    init_args   = ( Smax_init, ts_init, tau_init)
    
    theta, F    = optiFunc(data, init_args)
    
    # we visualize the optimized sigmoide and corresponding data...
    #plot_ModelAndData(data, theta, F)
    
    return theta, F

def plot_ModelAndData(data, theta, F, plot = True, save = False, filename = ""):
# =============================================================================
#     Plots the data and the "full" sigmoide
# input:
#     data        : data to be modelized
#     optuFunc    : optimization function to be used 
#     plot        : boolean, whether we want to plot the model or not
#     save        : boolean, plot is saved if true
#     filename    : name under wich the plot should be saved
# output:
#     plot of the model       : parameters of the optimal sigmoide
#     plot of F               : successive values of the criterion
#     png file of both plots  : if savePlot true, plots are saved under filename.png
# =============================================================================

    # final args returned by the optimization function
    Smax, ts, tau = theta
    
    # years from start (X) and corresponding cumulated production (Y)
    X, Y = [k for k in range(0,len(data))],[data]


# =============================================================================
#     fig, axes       = plt.subplots(2,1)
# 
#     # first subplot (first row, first column)
#     plt.sca(axes[0])
#     plt.plot(F)
#     
#     # second subplot (first row, first column)
#     plt.sca(axes[1])
#     
#     # putting ticks on ts and Smax
#     plt.xticks([theta[1]],[r'peak in year {}'.format(ts)])
#     plt.yticks([theta[0]],[r'estimated total of {} L'.format(Smax)])
#     
#     # Defining the window ...
#     plt.xlim([ 0, 2*ts])    
#     plt.ylim([ 0, 1.2*Smax])
#     
#     # plots the doted lines corresponding to ts and Smax
#     N       = 100 # number of dots
#     plt.plot(np.linspace(0, 2*ts, N),[Smax]*N,'--', color = "orange")
#     plt.plot([ts]*N,np.linspace(0, Smax/2, N),'--', color = "orange")
# 
#     #plots the data
#     plt.scatter(X,Y,marker="+", color="red")
#     
#     #plots the optimized sigmoide...
#     t = np.linspace(X[0],2*theta[1],1000)
#     plt.plot(t,sigmoide(t-X[0], theta))
# =============================================================================


    plt.figure()
    
    # plots the successive values of the criterion...
    ax1             = plt.subplot(211)
    plt.plot(F, label = "")

    # plots the data...
    ax2             = plt.subplot(212)
    
    # Definign the window ...
    ax2.set_xlim([ 0, 2*ts])    
    ax2.set_ylim([ 0, 1.2*Smax])   
    
    # titling the axes
    ax1.set_xlabel("Itérations")
    ax1.set_ylabel("Critère")
    
    ax2.set_xlabel("Temps (années)")
    ax2.set_ylabel("Production (L)")
    
    # displays the estimated time of the peak and final cumulated production
    ax2.set_xticks(ticks = [theta[1]])
    ax2.set_xticklabels(labels= [r'année {}'.format(round(ts))])
    
    ax2.set_yticks(ticks = [theta[0]])
    ax2.set_yticklabels(labels= [r'{} L'.format(round(Smax))])
    
    # plots the doted lines corresponding to ts and Smax
    N       = 100 # number of dots
    ax2.plot(np.linspace(0, 2*ts, N),[Smax]*N,'--', color = "orange")
    ax2.plot([ts]*N,np.linspace(0, Smax/2, N),'--', color = "orange")
    
    #plots the data
    plt.scatter(X,Y,marker="+", color="red", label ="Données")
    
    #plots the optimized sigmoide...
    t = np.linspace(X[0],2*theta[1],1000)
    plt.plot(t,sigmoide(t-X[0], theta), label = "Modèle")
    
    plt.legend()
    if save:
        plt.savefig("../graphes/anticipation/{}.png".format(filename + str(theta)
                                                            .replace(", ", "_")
                                                            .replace("[", "_")
                                                            .replace("]", "")))

    if plot:
        plt.show()
    else:
        plt.close()
    
    return

