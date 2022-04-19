# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
"""
Data
"""

class Data_processing:

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
                    if line_split[6] != '':
                        T.append(int(line_split[5]))
                        self.oil_production.append(float(line_split[6]))

        if len(T) == 0:
            raise Exception("Location not found")

        self.t_start = T[0]
        self.t_end = T[-1]

        for k in range(len(self.oil_production[1:])):
            self.oil_production[k+1] += self.oil_production[k]

    def get_data(self):
        return self.oil_production

    def plot(self):
        plt.figure()
        plt.scatter(range(self.t_start,self.t_end+1),self.oil_production)
        plt.show()


