# -*- coding: utf-8 -*-

import numpy as np

class Data_Processing:

    def __init__(self, nameFile, func, grad_func):
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

    def fitness(self, args):
        score = 0

        for x, y in self.data:
            score += (self.func(x,args)-y)**2
        return score/self.n

    def gradient_fitness(self, args):
        grad_theta = np.zeros(3)

        for x, y in self.data:
            grad_theta += 2*self.grad_func(x, args)*(self.func(x, args)-y)

        return grad_theta/self.n
