# -*- coding: utf-8 -*-

import numpy as np

class Data_Processing:

    def __init__(self, nameFile, func, grad_func):
        self.nameFile = nameFile
        self.func = func
        self.grad_func = grad_func

        X, Y = [],[]
        with open("../data/{}.csv".format(nameFile),"r") as f:
            for line in f.readlines()[1:]:
                X.append(int(line.split(";")[5]))
                Y.append(float(line.split(";")[6]))

        self.data = np.zeros((len(X),2))
        self.data[:,0] = X
        self.data[:,0] = self.data[:,0] - X[0]
        self.data[:,1] = Y
   
        self.n = len(X)

    def get_data(self):
        return self.data

    def fitness(self, args, dt): #fonction qui évalue la proximité de la courbe aux données
        score = 0

        for x, y in self.data:
            score += dt*(self.func(x,args)-y)**2
        return score/self.n

    def gradient_fitness(self, args): # gradient de la fonction fitness
        grad_theta = np.zeros(3)

        for x, y in self.data:
            grad_theta += 2*self.grad_func(x, args)*(self.func(x, args)-y) # grad_func est ici le gradient de la courbe d'hubert

        return grad_theta/self.n
    
    def gradient_fitness_scaled(self, args):
        pass
    
    """
        cumulative function
    
    input:
        tn
    
    output:
        sum of the petrol extracted from the t0 to tn
    """
    
    def cumulative(self, tn):
        total=0
        for x, y in self.data:
            if x>tn:
                break
            total+=y
        return total
    
    """
        criterion function (evaluates the distance between the data and the model)
    
    input:
        theta (args)
        tstart
        tstop
        dt
    
    output:
        average squared distance between the data and the model
    """

    def criterion(self, args, dt):
        score = 0

        for x, y in self.data:
            score += dt*(self.func(x,args)-self.cumulative(x))**2
        return score/self.n
        
    def gradient_criterion(self, args): # gradient de la fonction criterion
        grad_theta = np.zeros(3)

        for x, y in self.data:
            grad_theta += 2*self.grad_func(x, args)*(self.func(x, args)-self.cumulative(x)) # grad_func est ici le gradient de la courbe d'hubert

        return grad_theta/self.n
        
