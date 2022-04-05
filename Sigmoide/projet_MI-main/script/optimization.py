# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from data_processing import Data_Processing
from functions import Hubbert_curve, gradient_Hubbert_curve

init_args_france = (10000,6,8.4)

def optimiser_Hubbert(nameFile, init_args, dt=10**-6, eps = 0.1):
    DP = Data_Processing(nameFile, Hubbert_curve, gradient_Hubbert_curve)

    a_init, b_init, tau_init = init_args
    theta = np.array([a_init, b_init, tau_init])

    F = [DP.fitness(theta, dt)]

    n = 0
    grad = DP.gradient_fitness(theta)


    while np.linalg.norm(grad)>eps and n<1000000: #n = 10'000'000 = 30mn
        grad = DP.gradient_fitness(theta)
        theta = theta - dt*grad
        F.append(DP.fitness(theta, dt))
        n += 1
        if n%100==0:
            print(theta, np.linalg.norm(grad), F[-1])

    return theta, F


def optimiser_Hubbert_france():
    return optimiser_Hubbert("oil_france", init_args_france)

def test_courbe_Hubbert(nameFile, init_args):
    theta, F = optimiser_Hubbert(nameFile, init_args)

    X, Y = [],[]
    with open("../data/{}.csv".format(nameFile),"r") as file:
        for line in file.readlines()[1:]:
            X.append(int(line.split(";")[5]))
            Y.append(float(line.split(";")[6]))

    plt.figure()

    plt.subplot(211)
    plt.plot(F)

    plt.subplot(212)
    plt.scatter(X,Y,marker="+", color="red")
    t = np.linspace(X[0],X[-1],1000)
    plt.plot(t,Hubbert_curve(t-X[0], theta))

    plt.show()

    

test_courbe_Hubbert("oil_france",init_args_france)
#test_courbe_Hubbert_Acc("oil_france",init_args_france)