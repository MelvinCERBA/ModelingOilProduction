# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoide, noised_sigmoide, grad_sigmoide, least_square,grad_least_square, hubbert, sig_toHub
from mpl_toolkits.axes_grid.axislines import SubplotZero
from descent import descent, descentArmijo, descentScaled
from data_processing_Sig import Data_processing
from data_processing_Hub import Data_processing_Hub
import time as time
from anticipation import model, plot_ModelAndData
from datetime import datetime


# ======================== Testing values =====================================
# inital guess of the parameters of the optimized sigmoide for France's data :
init_args_france    = ( 70000, 15, 5)
# inital guess of the parameters of the optimized sigmoide for data generated with zero noise :
init_args_gen       = ( 90, 25, 5)
# actual parameters of the optimized sigmoide for generated data with zero noise :
perfect_args_gen    = ( 100, 30, 6)
# =============================================================================




def plot_sigmoide(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t       = np.arange(t_start, t_end, 1)
    sig     = sigmoide(t, (Qmax, ts, tau))

    plt.figure()
    plt.plot(t,sig)
    plt.show()



def plot_noised_sigmoide(noise,Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Joli plot à faire avec t*, Qmax et un segment de la tangeante en t*

    """
    t           = np.arange(t_start, t_end, 1)
    noised_sig  = noised_sigmoide(noise,Qmax, ts, tau, t_start, t_end)

    plt.figure()
    plt.scatter(t,noised_sig)
    plt.show()
# ========================== test ===============================
# plot_noised_sigmoide(10,100,30,6,0,60)
# =============================================================================



def test_least_square(Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    t       = np.linspace(t_start, t_end, 10000)
    sig     = sigmoide(t, (Qmax, ts, tau))

    ls      = least_square(sig, t_start, t[1]-t[0], sigmoide, (Qmax, ts, tau))

    print("==test least square==")
    print("\t-least_square = ", ls)
    print("test result : ")
    return ls == 0



def test_grad_sigmoide(delta, Qmax=100, ts=30, tau=6, t_start=0, t_end=60):
    """
    Compare the gradient of the sigmoide function to a finite differences gradient made with the sigmoide function

    """
    diff_max = None

    T                   = np.arange(t_start, t_end, 1)
    
    LQmax, Lts, Ltau    = [],[],[]
    
    Mgrad               = np.zeros((t_end-t_start,3))
    
    for k,t in enumerate(T):
        dQdQmax     = (sigmoide(t, (Qmax+delta, ts, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdts       = (sigmoide(t, (Qmax, ts+delta, tau))-sigmoide(t, (Qmax, ts, tau)))/delta

        dQdtau      = (sigmoide(t, (Qmax, ts, tau+delta))-sigmoide(t, (Qmax, ts, tau)))/delta
        

        grad_sig    = grad_sigmoide(t, (Qmax, ts, tau))
        
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

    dJdQmax     = (least_square(data, t_start, 1, sigmoide,(Qmax+delta/2, ts, tau))-
                           least_square(data, t_start, 1, sigmoide,(Qmax-delta/2, ts, tau)))/delta

    dJdts       = (least_square(data, t_start, 1,sigmoide, (Qmax, ts+delta, tau))-least_square_origine)/delta

    dJdtau      = (least_square(data, t_start, 1,sigmoide,(Qmax, ts, tau+delta))-least_square_origine)/delta

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

# =============================================================================
# plot_isocurve_Qmax_fixed(0.9)
# plot_isocurve_ts_fixed(0.9)
# plot_isocurve_tau_fixed(0.9)
# =============================================================================
    

    
def opti_generatedData(noise, perfect_args, delta_args=1.1, tend=60, optiFunc = descentScaled):
# =============================================================================
#     Optimization of the parameters of the sigmoide to match generated data
#input:
#    noise           : level of noise to apply to the data
#    perfect_args    : parameters used to generate data
#    delta_args      : proportion of the perfect args to be used as the initial parameters in the optimization process
#    optiFunc        : algorithm to be used for the optimization
#         
#output:
#    plots F               : shows the evolution of the criterion during the optimization
#    plots the sigmoid     : plots the sigmoide corresponding to the optimized parameters on top of the 
#    chrono                : time taken by the optimization
#    crit                  : last value of the criterion
# =============================================================================
    
    # perfect parameters : those of the sigmoide used to generate data
    Qmax_perfect, ts_perfect, tau_perfect   = perfect_args
    perfect_args                            = np.array(perfect_args)
    
    # data with the desired amount of noise...
    data        = noised_sigmoide(noise, Qmax=Qmax_perfect, ts=ts_perfect, tau=tau_perfect, t_start=0, t_end=tend)
    
    # optimized parameters and values of the criterion during the optimization ...
    start       = time.time()
    theta, F    = optiFunc(data, delta_args*perfect_args)
    end         = time.time()
    
    chrono      = end-start
    crit        = F[-1]
    # plots the criterion's values and the optimized sigmoide on top of the data
    plot_F_Data_Sigmoide(F, data, theta)

    return chrono, crit
# =============================================================================
# chrono, crit                = opti_generatedData(2, perfect_args_gen, delta_args=1, tend = 45)
# print("chrono = ", chrono, "crit = ", crit)
# =============================================================================


    
def testPerf_NoisedData(perfect_args, noise_steps = 8, noise_dt = 25, argsDelta_steps = 3, argsDelta_dt = 0.1, optiFunc = descentScaled, average_of = 1, Niter = 100):
# =============================================================================
#     monitors the optimization on generated data for different values of noise and for different initial params
# input:
#   perfect_args        : args of the function used to generate data
#   noise_steps         : number of values of noise to be tested
#   noise_dt            : amount of noise to add at each step
#   argsDelta_steps     : number of sets of args to be tested
#   argsDelta_dt        : percentage the init_args to add at each step
#   optiFunc            : descent algorithm to be used for optimization
#   average_of          : number of time we want to test the same values of noise and deltaArgs
#   Niter               : max number of iteration to be completed by the descent algorithm
#output:
#   criterion_results    : criterion obtained for each level of noise and set of args tested 
#   time_results         : time needed to complete the optimization for each level of noise and set of args tested
# =============================================================================
    
    # perfect parameters : those of the sigmoide used to generate data
    Qmax_perfect, ts_perfect, tau_perfect   = perfect_args
    perfect_args                            = np.array(perfect_args)

    # all levels of noise to be tested...
    noise_levels    = [ k * noise_dt for k in range( 0, noise_steps+1, 1) ]
    
    # all values of args to be tested...
    args_values     = [ [(1+(k * argsDelta_dt))] * perfect_args for k in range( 0, argsDelta_steps+1, 1)]
    
    # for each level of noise (columns), for each value of args (lines), 
        # we will save here : 1) the criterion obtained 2) the time taken to complete the optomization...
    criterion_results       = np.zeros([ len(noise_levels), len(args_values)])
    time_results            = np.zeros([ len(noise_levels), len(args_values)])
    
    # counts the number of iteration of the descent algorithm
    iter_results            = np.zeros([ len(noise_levels), len(args_values)])
    
    # we want the average value of the criterion and execution time, because it can vary a lot for the same values of noise and argsDelta
    for a in range(average_of):
        
        # Testing each level of noise...
        for i in range ( 0, len(noise_levels), 1):
            
            # data to be tested...
            data = noised_sigmoide(noise_levels[i], Qmax=Qmax_perfect, ts=ts_perfect, tau=tau_perfect)
            
            # Testing each value of argsDelta ...
            for j in range ( 0, len(args_values), 1):
                
                # value of the args to be tested...
                init_args           = args_values[j]
                
                # descent algorithm ...
                start               = time.time()
                # print("noise = ", noise_levels[i], "  args_values = ", args_values[j])
                theta, F            = descentScaled(data, init_args, Niter = Niter)     
                end                 = time.time()
                
                # saving the last value of the criterion and the time taken by the descent (seconds)...
                criterion_results[i,j]   += F[-1]
                time_results[i,j]        += end-start
                
                #
                iter_results[i,j]        += len(F)
                # print("critère = ", F[-1], "  temps d'execution = ", end-start)
    
    # print("crits = ",criterion_results)
    # print("times = ",time_results)
    
    # calculating the average
    for i in range ( 0, len(noise_levels), 1):
        for j in range ( 0, len(args_values), 1):
            criterion_results[i,j]      = criterion_results[i,j] / average_of
            # time_results[i,j]           = time_results[i,j] / average_of
            iter_results[i,j]           = iter_results[i,j] / average_of
    
    # max criterion and max time are used to standardize the scale of the plots...
    max_criterion       = max([ max(c) for c in criterion_results])
    # max_time            = max([ max(t) for t in time_results])
    
    plt.close()
    
    # width of the bars
    w               = max(noise_levels)/(2*2*noise_steps)
    for i in range(0,argsDelta_steps+1,1):
        # variation of the perfect args we are testing (percentage)... 
        delta       = round(i * argsDelta_dt * 100)
        # print("i= ", i, "delta= ", delta)
        
        # for a fixed value of init_args, variation of the final criterion 
                # and execution time depending on the level of noise...
        # times        = time_results[:,i]
        criterions   = criterion_results[:,i]
        iters        = iter_results[:,i]
            
        fig, ax1 = plt.subplots()
            
        # creating two arrays with a slight offset to plot bars separatly...
        bar1    = noise_levels
        bar2    = [x+w for x in bar1]

        # plotting the time and criterion values for the different levels of noise...
        ax2     = ax1.twinx()
        ax1.bar(bar1, criterions, label='Critère', width = w)
        ax2.bar(bar2, iters, color='r', label='Itérations', width = w)
        # ax2.bar(bar2, times, color='r', label='Temps (s)', width = w)
        
        # Definign the window ...
        # ax1.set_ylim([ 0, 1.2*max_criterion])  # results vary a lot, making the smaller ones invisible
        ax1.set_yscale('log')                  # wich is why we will be using a logarithmic scale
        ax2.set_ylim([ 0, 1.2*Niter])
        # ax2.set_ylim([ 0, 1.2*max_time])
        
        # legending the graph...
        ax1.set_xlabel('Bruit')
        ax1.set_ylabel('Critère (moindre carrés)')
        ax2.set_ylabel('Itérations')
        # ax2.set_ylabel('Temps (secondes)')
        
        ax1.legend(loc=(0,0.9))
        ax2.legend(loc=(0.75,0.9))
        plt.title(r'Arguments intiaux = {}% des arguments parfaits'.format(str(100+delta)))
        
        plt.savefig("../graphes/performances/noise_0-{}_args{}perc.png".format( noise_levels[-1], str(100+delta)))
        plt.show()
            
    return time_results, criterion_results
# ==================== test testPerf_NoisedData() =============================
# testPerf_NoisedData(perfect_args_gen, noise_steps = 5, noise_dt = 20, argsDelta_steps = 6, argsDelta_dt = 0.1, optiFunc = descentScaled, average_of = 100)
# =============================================================================



def plot_F_Data_Sigmoide(F, data, theta):
    
    # years from start (X) and corresponding cumulated production (Y)
    X, Y = [k for k in range(0,len(data))],[data]

    plt.figure()
    
    
    # plots the successive values of the criterion...
    ax1             = plt.subplot(211)
    plt.plot(F)

    # plots the data...
    ax2             = plt.subplot(212)
    
    # Definign the window ...
    ax2.set_xlim([ 0, 2*theta[1]])     # 2*ts
    ax2.set_ylim([ 0, 1.2*theta[0]])   # 1.2*Smax
    
    plt.scatter(X,Y,marker="+", color="red")
    
    #plots the optimized sigmoide...
    t = np.linspace(X[0],2*theta[1],1000)
    plt.plot(t,sigmoide(t-X[0], theta))
    

    plt.show()

def opti_Country(location, optiFunc = descentScaled, savePlot = False, anticipation = 0):
# =============================================================================
#     Optimization of the parameters of the sigmoide to match the data of a specific country
#input:
#    location        : abreviation of the country (ex: FRA)
#    optiFunc        : algorithm to be used for the optimization
#    anticipation    : number of years to predict   
#output:
#    location                : name of the country
#    plot of F               : shows the evolution of the criterion during the optimization
#    plot of the sigmoid     : plots the sigmoide corresponding to the optimized parameters on top of the data
#    chrono                  : time taken by the optimization
#    crit                    : last value of the criterion
#    init_args               : init_args passed as input. Used to visualize whether our initial guess was close or not
# =============================================================================
    
    # data of the selected country...
    data                = Data_processing(location).get_data()
    
    # initial guess of the country's optimal parameters :
    Smax_init   = max(data)
    ts_init     = len(data)//2
    tau_init    = 5 # value of tau used when testing the descent on france's data. It's a bit arbitrary, but it seems to work
    
    init_args   = (Smax_init, ts_init, tau_init)
    
    # data to be used for optimization ( = data - the years we want the model to anticipate)
    shortData       = [data[k] for k in range(len(data) - anticipation)]
    
    # optimized parameters and values of the criterion during the optimization ...
    start       = time.time()
    theta, F    = optiFunc(shortData, init_args)
    end         = time.time()
    
    chrono      = end-start
    crit        = F[-1]

    if savePlot:
        save_CountryPlot(location, F, data, theta, init_args, anticipation = anticipation)
    else:
        # plots the criterion's values and the optimized sigmoide on top of the data
        plot_F_Data_Sigmoide(F, data, theta)
    
    return location, chrono, crit, theta, F, init_args
# ==================== Test ==================================================
# name, chrono, crit, theta, F, init_args        = opti_Country('FRA', init_args_france, savePlot=True)
# print("chrono = ", chrono, "crit = ", crit)
# =============================================================================
    
def save_CountryPlot(country, F, data, theta, init_args, anticipation = 0):
    
    # non-cumulated data 
    DP                  = Data_processing_Hub(country)
    dataHub             = DP.get_data() 
    startYear           = DP.get_Tstart()
    
    # first year to be anticipated
    pivotYear           = len(data) - anticipation
    
    # years from start to last data used by the optimization
    years               = [startYear + k for k in range( 0, pivotYear , 1)]
    
    # years the model is trying to anticipate
    yearsAnticipated    = [ years[-1] + k for k in range( 1, anticipation + 1, 1)]


    # sets of data to be used to plot in different colors the data that's been used or not
    usedDataSig         = [ data[k] for k in range( 0, pivotYear , 1)]
    unusedDataSig       = [ data[k] for k in range( pivotYear, len(data) , 1)]
    
    usedDataHub         = [ dataHub[k] for k in range( 0, pivotYear , 1)]
    unusedDataHub       = [ dataHub[k] for k in range( pivotYear, len(dataHub) , 1)]
    
    
    plt.figure()
    
# ======== plots the successive values of the criterion... ====================
#     plt.subplot(211)
#     plt.plot(F)
#     plt.xlabel("Itérations")
#     plt.ylabel("Critère")
# =============================================================================

    # plots the cumulated data...
    plt.subplot(211)
    
    
    # titles the axes...
    # plt.xlabel("Temps (années)") 
    plt.ylabel("Prodcution totale (L)")
    
    # previews the future according to the optimized model
    if anticipation != 0:
        # plots both used and unused data
        plt.scatter(years, usedDataSig, marker="+", color="red", label = "Données utilisées")
        plt.scatter(yearsAnticipated, unusedDataSig, marker="+", color="orange", label = "Données inutilisées")
        
        # plots the optimized sigmoide and its previsions in different colors...
        t1 = np.linspace( years[0], years[-1], 1000)
        t2 = np.linspace( yearsAnticipated[0], yearsAnticipated[-1], 1000)
        
        # optimized sigmoide...
        plt.plot( t1, sigmoide(t1-years[0], theta), label = "Sigmoïde optimisée")
        
        # previsions of the model...
        plt.plot( t2, sigmoide(t2-years[0], theta), label = "Prévisions")
        
        
    else:
        # plots the data
        plt.scatter(years, data, marker="+", color="red", label = "Données de l'OCDE")
        
        # plots the optimized sigmoide...
        t = np.linspace(years[0],years[-1],1000)
        plt.plot( t, sigmoide(t-years[0], theta), label = "Courbe sigmoïde optimisée")
    
    
    
    # titles and legend the plot
    plt.title(country + f" : prévisions sur {anticipation} ans")
    plt.legend()
    
    
# ========== plots the sigmoide corresponding to the initial guess ============
#     plt.plot( t, sigmoide(t-X[0], init_args), '--', color='black' )
# =============================================================================
    
    # plots the non cumulated data...
    plt.subplot(212)
    # plt.scatter(years, usedDataHub ,marker="+", color="red", label = "Données utilisées")
    # plt.scatter(yearsAnticipated, unusedDataHub, marker="+", color="orange", label = "Données inutilisée")
    
    # titles the axes...
    plt.xlabel("Temps (années)")
    plt.ylabel("Prodcution (L)")
    
    #plots the optimized sigmoide...
    # previews the future according to the optimized model
    if anticipation != 0:
        # plots both used and unused data
        plt.scatter(years, usedDataHub, marker="+", color="red", label = "Données utilisées")
        plt.scatter(yearsAnticipated, unusedDataHub, marker="+", color="orange", label = "Données inutilisées")
        
        # plots the optimized hubbert curve and its previsions in different colors...
        t1 = np.linspace( years[0], years[-1], 1000)
        t2 = np.linspace( yearsAnticipated[0], yearsAnticipated[-1], 1000)
        
        # optimized hubbert curve...
        plt.plot( t1, hubbert(t1-years[0], theta), label = "Hubbert optimisée")
        
        # previsions of the model...
        plt.plot( t2, hubbert(t2-years[0], theta), label = "Prévisions")
        
        
    else:
        # plots the data
        plt.scatter(years, dataHub, marker="+", color="red", label = "Données de l'OCDE")
        
        # plots the optimized sigmoide...
        t = np.linspace(years[0],years[-1],1000)
        plt.plot( t, hubbert(t-years[0], theta), label = "Courbe de Hubbert optimisée")
        

    plt.legend()
    
    if anticipation != 0:
        plt.savefig("../graphes/Pays/previsions/prev{}_{}.png".format( anticipation, country + str(datetime.date(datetime.now()))))
    else:
        plt.savefig("../graphes/Pays/{}.png".format(country + str(datetime.date(datetime.now()))))
        
    plt.close()
# ==================== Test save_countryPlot() ================================
# name, chrono, crit, theta, F, init_args        = opti_Country('FRA', init_args_france, savePlot=True, anticipation = 20)
# print("chrono = ", chrono, "crit = ", crit)
# =============================================================================

# ==================== Test save_countryPlot() anticipations ==================
for a in range(5,21,5):
    print("a = ", a)
    opti_Country('DNK', savePlot=True, anticipation = a)
# =============================================================================


def test_OCDE(save=True, anticipation = 0):
# =============================================================================
#     Optimization on each country of the OCDE
#input:
#    save            : whether we want to save the resulting plots or not
#    anticipation    : number of years to predict   
#output:
#    .png file    : plots of both the sigmoide and hubbert curve corresponding to the optimized parameters, on top of the data
# =============================================================================    
    plt.close()
    
    results = [[]]
    countries = []
    
    # we get every country's acronym from the file...
    with open("../data/Crude_oil_production.csv","r") as fileCSV:
            for line in fileCSV.readlines()[1:]:
                line_split      = line.split(";")
                country_name    = line_split[0]
                if country_name in countries: # one country is present on multiple lines ( one for each year )
                    continue
                countries.append(line_split[0])
                
    # for each country, we try the optimization process and save its results            
    for country in countries:
        print(country, f" : prévisions sur {anticipation} ans...")
        results += [opti_Country(country, optiFunc = descentScaled, savePlot=save, anticipation = anticipation)]
        
    return results
# ==================== Test test_OCDE() =======================================
# test_OCDE(save = True, anticipation = 20) 
# =============================================================================    

# ==================== Test tes_OCDE() previsions =============================
# for a in range(5,31,5):
#     test_OCDE(save = True, anticipation = a) 
# =============================================================================

def test_Model(data, optiFunc = descentScaled, plot = True, save = False, filename = "", anticipation = 0):
# =============================================================================
#     Takes data and executes the optimization on the chosen portion, then evaluates the final model on the complete data
#input:
#       data        : data to be modelized
#       percent     : percentage of the data to be used for optimization (in ]0,1])
#       optiFunc    : optimization function to be used
#       plot        : boolean, whether we want to plot the model or not
#       save        : boolean, plot is saved if true
#       filename    : name under wich the plot should be saved
#output:
#       theta       : parameters of the optimal sigmoide
#       fitness     : estimate of the quality of the model (last value of the criterion returned by the descent)
# =============================================================================

    # reduced data
    new_data    = [data[i] for i in range( 0, len(data) - anticipation, 1)]
    print("new_data length = ", len(new_data),
          "/n data length = ", len(data))
    
    # optimized parameters and values of the criterion during the optimization ...
    start       = time.time()
    theta, F    = model(new_data, optiFunc)
    end         = time.time()
    
    # performance of the optimization : time of execution and criterion of the final sigmoide
    chrono      = end-start
    crit        = least_square(data, t_start = 0, Dt = 1, func = sigmoide, args = theta) # we compare the model to the complete data (dt doesn't matter as long as it is the same for every model we compare)

    # plots and/or saves the results, according to the parameters
    save_Model(filename, data, theta, anticipation)
    
    return theta, F, chrono, crit
# ========================= Test =============================================
# test_Model(data = noised_sigmoide(0, 100, 30, 6), percent = 0.6, optiFunc = descentScaled, plot = True, save = True, filename ="test")
# =============================================================================

def save_Model(name, data, theta, anticipation = 0):
    
    # non-cumulated data 
    dataHub             = sig_toHub(data)
    startYear           = 0
    
    # first year to be anticipated
    pivotYear           = len(data) - anticipation
    
    # years from start to last data used by the optimization
    years               = [startYear + k for k in range( 0, pivotYear , 1)]
    
    # years the model is trying to anticipate
    yearsAnticipated    = [ years[-1] + k for k in range( 1, anticipation + 1, 1)]


    # sets of data to be used to plot in different colors the data that's been used or not
    usedDataSig         = [ data[k] for k in range( 0, pivotYear , 1)]
    unusedDataSig       = [ data[k] for k in range( pivotYear, len(data) , 1)]
    
    usedDataHub         = [ dataHub[k] for k in range( 0, pivotYear , 1)]
    unusedDataHub       = [ dataHub[k] for k in range( pivotYear, len(dataHub) , 1)]
    
    
    plt.figure()
    
# ======== plots the successive values of the criterion... ====================
#     plt.subplot(211)
#     plt.plot(F)
#     plt.xlabel("Itérations")
#     plt.ylabel("Critère")
# =============================================================================

    # plots the cumulated data...
    plt.subplot(211)
    
    
    # titles the axes...
    # plt.xlabel("Temps (années)") 
    plt.ylabel("Prodcution totale (L)")
    
    # previews the future according to the optimized model
    if anticipation != 0:
        # plots both used and unused data
        plt.scatter(years, usedDataSig, marker="+", color="red", label = "Données utilisées")
        plt.scatter(yearsAnticipated, unusedDataSig, marker="+", color="orange", label = "Données inutilisées")
        
        # plots the optimized sigmoide and its previsions in different colors...
        t1 = np.linspace( years[0], years[-1], 1000)
        t2 = np.linspace( yearsAnticipated[0], yearsAnticipated[-1], 1000)
        
        # optimized sigmoide...
        plt.plot( t1, sigmoide(t1-years[0], theta), label = "Sigmoïde optimisée")
        
        # previsions of the model...
        plt.plot( t2, sigmoide(t2-years[0], theta), label = "Prévisions")
        
        
    else:
        # plots the data
        plt.scatter(years, data, marker="+", color="red", label = "Données utilisées")
        
        # plots the optimized sigmoide...
        t = np.linspace(years[0],years[-1],1000)
        plt.plot( t, sigmoide(t-years[0], theta), label = "Courbe sigmoïde optimisée")
    
    
    
    # titles and legend the plot
    plt.title( f" Prévisions sur {anticipation} ans")
    plt.legend()
    
    
# ========== plots the sigmoide corresponding to the initial guess ============
#     plt.plot( t, sigmoide(t-X[0], init_args), '--', color='black' )
# =============================================================================
    
    # plots the non cumulated data...
    plt.subplot(212)
    # plt.scatter(years, usedDataHub ,marker="+", color="red", label = "Données utilisées")
    # plt.scatter(yearsAnticipated, unusedDataHub, marker="+", color="orange", label = "Données inutilisée")
    
    # titles the axes...
    plt.xlabel("Temps (années)")
    plt.ylabel("Prodcution (L)")
    
    #plots the optimized sigmoide...
    # previews the future according to the optimized model
    if anticipation != 0:
        # plots both used and unused data
        plt.scatter(years, usedDataHub, marker="+", color="red", label = "Données utilisées")
        plt.scatter(yearsAnticipated, unusedDataHub, marker="+", color="orange", label = "Données inutilisées")
        
        # plots the optimized hubbert curve and its previsions in different colors...
        t1 = np.linspace( years[0], years[-1], 1000)
        t2 = np.linspace( yearsAnticipated[0], yearsAnticipated[-1], 1000)
        
        # optimized hubbert curve...
        plt.plot( t1, hubbert(t1-years[0], theta), label = "Hubbert optimisée")
        
        # previsions of the model...
        plt.plot( t2, hubbert(t2-years[0], theta), label = "Prévisions")
        
        
    else:
        # plots the data
        plt.scatter(years, dataHub, marker="+", color="red", label = "Données utilisées")
        
        # plots the optimized sigmoide...
        t = np.linspace(years[0],years[-1],1000)
        plt.plot( t, hubbert(t-years[0], theta), label = "Courbe de Hubbert optimisée")
        

    plt.legend()
    
    if anticipation != 0:
        plt.savefig("../graphes/anticipation/model{}_{}.png".format( anticipation, name + str(datetime.date(datetime.now()))))
    else:
        plt.savefig("../graphes/{}.png".format(name + str(datetime.date(datetime.now()))))
        
    # plt.close()

# ================== test save_Model() ========================================
# for a in range( 5, 51, 5):
#     save_Model("", noised_sigmoide(0) , perfect_args_gen , anticipation = a)
# =============================================================================

# =================== test noise variable =====================================
# for n in range( 0, 4, 1):
#     save_Model(f"noise{n}", noised_sigmoide(n) , perfect_args_gen , anticipation = 40)
# =============================================================================

def test_Model_onNoisedData(perfect_args, noise_steps = 3, noise_dt = 10, anticipation_steps = 3, anticipation_dt = 1, optiFunc = descentScaled):
# =============================================================================
#     monitors the optimization on generated data for different values of noise and for different initial params
# input:
#   perfect_args        : args of the function used to generate data
#   noise_steps         : number of values of noise to be tested
#   noise_dt            : amount of noise to add at each step
#   anticipation_steps  : number of anticipation values to be tested
#   anticipation_dt     : number of years of data to remove at each step
#   optiFunc            : descent algorithm to be used for optimization
#
#output:
#   criterion_results    : criterion obtained for each level of noise and set of args tested 
#   time_results         : time needed to complete the optimization for each level of noise and set of args tested
# =============================================================================
    
    # perfect parameters : those of the sigmoide used to generate data
    Qmax_perfect, ts_perfect, tau_perfect   = perfect_args
    perfect_args                            = np.array(perfect_args)

    # all levels of noise to be tested...
    noise_levels    = [ k * noise_dt for k in range( 0, noise_steps+1, 1) ]
    
    # all values of args to be tested...
    anticipation_values     = [ k * anticipation_dt for k in range( 0, anticipation_steps+1, 1)]
    
    # for each level of noise (columns), for each value of args (lines), 
        # we will save here : 1) the criterion obtained 2) the time taken to complete the optomization...
    criterion_results       = np.zeros([ len(noise_levels), len(anticipation_values)])
    time_results            = np.zeros([ len(noise_levels), len(anticipation_values)])
    
    
    
    for i in range ( 0, len(noise_levels), 1):
        
        # data to be tested...
        data = noised_sigmoide(noise_levels[i], Qmax=Qmax_perfect, ts=ts_perfect, tau=tau_perfect)
        
        for j, p in enumerate(anticipation_values):
            print("p =", p)
            # descent algorithm on the selectad amount of data...
            theta, F, chrono, crit            = test_Model(data, plot = False, save = True, 
                                                           filename= "noise{}_prev{}".format( noise_levels[i], p), anticipation = p)     
            
            # saving the last value of the criterion and the time taken by the descent (seconds)...
            criterion_results[i,j]   = crit
            time_results[i,j]        = chrono
    
    # print("crits = ",criterion_results)
    # print("times = ",time_results)
    
    # max criterion and max time are used to standardize the scale of the plots...
    max_criterion       = max([ max(c) for c in criterion_results])
    max_time            = max([ max(t) for t in time_results])
    
    # width of the bars
    w               = max(noise_levels)/(2*2*noise_steps)
    
    plt.close()
    for i in range(0,anticipation_steps+1,1):
        # variation of the perfect args we are testing (percentage)... 
        delta       = round(i * anticipation_dt * 100)
        print("i= ", i, "delta= ", delta)
        
        # for a fixed value of init_args, variation of the final criterion 
                # and execution time depending on the level of noise...
        times        = time_results[:,i]
        criterions   = criterion_results[:,i]
            
        fig, ax1 = plt.subplots()
            
        # creating two arrays with a slight offset to plot bars separatly...
        bar1    = noise_levels
        bar2    = [x+w for x in bar1]

        # plotting the time and criterion values for the different levels of noise...
        ax2     = ax1.twinx()
        ax1.bar(bar1, criterions, label='Critère', width = w)
        ax2.bar(bar2, times, color='orange', label='Temps (s)', width = w)
        
        # Definign the window ...
        ax1.set_ylim([ 0, 1.2*max_criterion])
        ax2.set_ylim([ 0, 1.2*max_time])
        
        # legending the graph...
        ax1.set_xlabel('Bruit')
        ax1.set_ylabel('Critère (moindre carrés)')
        ax2.set_ylabel('Temps (secondes)')
        
        ax1.legend(loc=(0,0.9))
        ax2.legend(loc=(0.75,0.9))
        plt.title(r'Optimisation sur {}% des données'.format(str(100-delta)))
        
        # Size of the plot's window 
        fig.set_figheight(8)
        fig.set_figwidth(10)

        plt.savefig("../graphes/performances/perfs_ModelOnGeneratedData_DataQ_{}.png".format(100-delta))
        plt.show()
            
    return time_results, criterion_results
# ==================== Test testPerfs_Model_onNoisedData() ====================
# test_Model_onNoisedData(perfect_args_gen, noise_steps = 10, noise_dt = 0.1, anticipation_steps = 3, anticipation_dt= 10)
# =============================================================================















