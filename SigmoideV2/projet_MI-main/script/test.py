# -*- coding: utf-8 -*-
from data_processing import Data_Processing
from data_generator import Data_Generator
from functions import Q, gradQ, Hubbert_curve, gradient_Hubbert_curve

"""
TESTS data_proccessing
"""

nameFile="oil_france"
DP = Data_Processing(nameFile, Q, gradQ)

def testCumulative(tn):
    return DP.cumulative(tn)

print(testCumulative(5))


"""
TESTS data_generator
"""
args_Sigmoide = [75000, 17, 6.5]
args_Hubbert = [7, 93922, 6.41]

bruit = 1
length = 100
    
# Pour la sigmoide
DG = Data_Generator(Q, gradQ, args_Sigmoide, length, bruit)
DG.visualize()

# Pour la courbe de Hubbert
DG = Data_Generator(Hubbert_curve, gradient_Hubbert_curve, args_Hubbert, length, bruit)
DG.visualize()

DG.save
