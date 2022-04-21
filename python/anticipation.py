# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoide, noised_sigmoide, grad_sigmoide, least_square,grad_least_square
from mpl_toolkits.axes_grid.axislines import SubplotZero
from descent import descent, descentArmijo, descentScaled
from data_processing_Sig import Data_processing
import time as time


def model(data, optiFunc=descentScaled):
# =============================================================================
#     Takes incomplete data, applies the descent to it and returns the optimized sigmoide
# input:
#     data        : data to be modelized
# output:
#     theta       : parameters of the optimal sigmoide
#     fitness     : estimate of the quality of the model (last value of the criterion returned by the descent)
# =============================================================================
    pass
