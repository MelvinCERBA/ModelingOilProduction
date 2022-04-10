# -*- coding: utf-8 -*-

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import csv

"""
            Data_Generator CLASS
used to create data from the model with given parameters

"""

class Data_Generator:
    
    """
                init function
    creates the data 
    
    input:
        func -> either the hubbert function or the sigmoid 
        args -> theta, the arguments of the chosen function
        length -> desired length of the simulated data set
        bruit -> level of noise desired (= sigma parameter of the gaussian curve)
    """

    def __init__(self, func, grad_func, args, length, bruit):
        self.func = func
        self.grad_func = grad_func
    
        X, Y = [1],[func(1, args)]
        for i in range(0,length,1):
            X.append(X[-1]+1)
            mu, sigma = 0, bruit # mu = 0 pour que les perturbations négatives et positives soient equiprobables
            bruitGauss = rd.gauss(mu,sigma) 
            Y.append(func(X[-1]+bruitGauss, args))
    
        self.data = np.zeros((len(X),2))
        self.data[:,0] = X
        self.data[:,0] = self.data[:,0] - X[0]
        self.data[:,1] = Y
       
        self.n = len(X)
        
    """
                visualize function
    plots the generated data 
    
    """
            
    def visualize(self):
        plt.figure()
        plt.scatter(self.data[:,0],self.data[:,1],marker="+", color="red")
    
        plt.show()
        
        
    """
                save function
    saves the generated data to a csv file (so it can be used by data_processing)
    
    """
    def save(self): 
        with open('../data/generated_data.csv', 'wb', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([0,0,0])
        return
   

