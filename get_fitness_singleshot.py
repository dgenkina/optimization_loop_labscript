# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:03:23 2019

@author: dng5
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import lyse
from fancy_cost_functions import *
import pandas as pd


try:
    lyse.path
except NameError:
    import sys
    path = sys.argv[1]

series = lyse.data(lyse.path)
param1 = series['param1']
param2 = series['param2']


run_instance = lyse.Run(lyse.path)
print(lyse.path)

sigma0 = (3/2/np.pi) * 780e-9**2


# atom_number = np.nansum(naive_OD*(5.6e-6/6.0)**2/sigma0)
params = np.array([param1, param2])
fitness = rosenbrock(params)


# run_instance.save_result('average_counts', atom_avg)
run_instance.save_result('fitness', fitness)

print('Saved fitness value of ' + str(fitness))