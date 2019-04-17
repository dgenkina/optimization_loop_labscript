from __future__ import division
import numpy as np
import labscript_utils.h5_lock
import h5py
import matplotlib.pyplot as plt
from lyse import *
import pandas as pd


from optimization_classes_v2 import *

experiment_name = 'testrun_nothing'
globals_filepath_list = [r'C:\labscript_suite\userlib\labscriptlib\Assembly_room\testrun.h5']
param_name_list = ['dummy1','dummy2']
param_min_max_list = [(-5.0,5.0),(-5.0,5.0)]


storage = routine_storage

if not hasattr(storage, 'optimization_protocol'):
    storage.optimization_protocol = DifferentialEvolution(experiment_name, globals_filepath_list, param_name_list,param_min_max_list, loops = 100, popsize = 20)

optimization_protocol = storage.optimization_protocol


df = data()
df = df.sort_values('run time')
run_number = df['run number']
print 'Runnumber of last shot'
print  run_number[-1]

param_dict = {}
for param_name in param_name_list:
    param_dict[param_name] = np.array(df[param_name].tolist())

labscript_name=df['labscript'][-1]
filepath = df['filepath'][-1]
sequence_index = df['sequence_index']
sequence_index = np.array(sequence_index.tolist())
cost_series = df['fancy_cost_functions','Cost']
cost = np.array(cost_series.tolist())
# print 'cost'
# print cost
# _fig_ = plt.figure(frameon=True)

# ax1= plt.subplot(111)
# line1 = ax1.plot(cost)


if labscript_name == experiment_name + '.py':
    optimization_protocol.run_first_batch()

elif run_number[-1] == optimization_protocol.batch_size-1:
    param_array = np.zeros((optimization_protocol.batch_size,0))
    for param_name in param_name_list:
        param_array = np.append(param_array,param_dict[param_name][-optimization_protocol.batch_size:].reshape(optimization_protocol.batch_size,1),axis=1)
 
    cost_list = cost[-optimization_protocol.batch_size:]
    print 'cost_list'
    print cost_list
    optimization_protocol.run_next_batch(cost_list,param_array)

    _fig_ = plt.figure(frameon=True)

    ax1= plt.subplot(111)
    line1 = ax1.plot(optimization_protocol.batch_number_list,optimization_protocol.best_cost_list)


    
