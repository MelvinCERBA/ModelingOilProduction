# -*- coding: utf-8 -*-

import numpy as np

from functions import Hubbert_curve

class Data_Processing:

    def __init__(self, nameFile, func):
        self.func = func


        X, Y = [],[]
        with open("../data/{}.csv".format(nameFile),"r") as f:
            for line in f.readlines()[1:]:
                X.append(int(line.split(";")[5]))
                Y.append(float(line.split(";")[6]))

        self.data = np.zeros((len(X),2))
        self.data[:,0] = X
        self.data[:,0] = self.data[:,0] - X[0]
        self.data[:,1] = Y

    def get_data(self):
        return self.data

    def fitness(self, *args):
        score = 0

        for x, y in self.data:
            score += (self.func(x,args)-y)**2

        return score


DP = Data_Processing("oil_france",Hubbert_curve)
data = DP.get_data()
score = DP.fitness(93922.052,6.4,7)