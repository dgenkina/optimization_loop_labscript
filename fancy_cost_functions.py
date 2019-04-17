from __future__ import division, print_function
import numpy as np
import h5py
import matplotlib.pyplot as plt
import lyse
#import gradient_descent_functions as optimization_protocol
import pandas as pd
#from optimization_classes import *


camera = 'imagesFlea3'

def ackley( x, a=20, b=0.2, c=2*np.pi ):
    """
    implements ackley function
    """
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = x.shape[0]
    s1 = np.sum( x**2, axis=0 )
    s2 = np.sum( np.cos( c * x ),  axis=0)
    return - a*np.exp( -b*np.sqrt( s1 / n )) - np.exp( s2 / n ) + a + np.exp(1)

ackley._bounds = [-5.0, 5.0]

def sphere( x):
    return np.dot(x,x)

def rosenbrock( x ):  
    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
        # a sum of squares, so LevMar (scipy.optimize.leastsq) is pretty good
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    
    return (np.sum( (1 - x0) **2, axis=0 )
        + 100 * np.sum( (x1 - x0**2) **2, axis=0 ))

rosenbrock._bounds   = [-2.4, 2.4]  # wikipedia

def noise(funct, sigma):
    """
    returns a function that wraps funct and adds noise given by sigma
    """

    def wrapped(*args, **kwargs):
        ans = funct(*args, **kwargs)

        if np.isscalar(ans):
            return ans + np.random.normal(0, sigma, size=1)[0]
        else:
            return ans + np.random.normal(0, sigma, size=ans.shape)
    
    wrapped._bounds = funct._bounds
    
    return wrapped


def get_cost(average_number):
    cost = - np.exp(-(float(average_number)-200.0)**2.0/1000.0)
    return cost

def get_cost_Himmelbaum(average_number,dummy1,dummy2):
    cost = - np.exp(-(float(average_number)-200.0)**2.0/10000.0)+(dummy1**2.0+dummy2-11.0)**2.0+(dummy1+dummy2**2.0-7)**2.0
    return cost



try:
    lyse.path
except NameError:
    import sys
    path = sys.argv[1]

series = lyse.data(lyse.path)
x = series['x']
y = series['y']
z = series['z']

# with h5py.File(lyse.path) as dummy_h5_label:
#     x = dummy_h5_label['globals'].attrs['x']
#     y = dummy_h5_label['globals'].attrs['y']
#     z = dummy_h5_label['globals'].attrs['z']

run_instance = lyse.Run(lyse.path)
print(lyse.path)

sigma0 = (3/2/np.pi) * 780e-9**2


# atom_number = np.nansum(naive_OD*(5.6e-6/6.0)**2/sigma0)
params = np.array([x, y, z])
cost = ackley(params)


# run_instance.save_result('average_counts', atom_avg)
run_instance.save_result('Cost', cost)

print('Saved Cost value of ' + str(cost))






    