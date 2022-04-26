# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
"""
Data
"""

class Data_processing_Hub:

    t_start: int  # year
    t_end: int  # year
    oil_production: list  # crude oil production data
    location: str  # Location

    def __init__(self, location):
        self.location = location

        self.oil_production = []

        T = []

        # Find the country and fill the data
        with open("../data/Crude_oil_production.csv","r") as fileCSV:
            for line in fileCSV.readlines()[1:]:
                line_split = line.split(";")
                key = line_split[0]
                if key == location:
                    if line_split[6] != '': # if data is present, we add it
                        T.append(int(line_split[5]))
                        self.oil_production.append(float(line_split[6]))
                        
                    else:                   # else, we add 0 production (empty data cause the algortihm to dysfunction )
                        T.append(int(line_split[5]))
                        self.oil_production.append(float(0))

        if len(T) == 0:
            raise Exception("Location not found")

        self.t_start = T[0]
        self.t_end = T[-1]

    def get_data(self):
        return self.oil_production
    
    def get_Tstart(self):
        return self.t_start

    def plot(self):
        plt.figure()
        plt.scatter(range(self.t_start,self.t_end+1),self.oil_production)
        plt.show()


