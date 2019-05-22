from __future__ import division
import os
import numpy as np
import labscript_utils.h5_lock
import h5py
import matplotlib.pyplot as plt
from lyse import *
import pandas as pd
import runmanager.remote
from labscript_utils.labconfig import LabConfig
labconfig = LabConfig()
from optimization_classes import *

save_file_path = os.path.join(labconfig.get('DEFAULT', 'experiment_shot_storage'), 'Optimization_testRoutine_8.h5')

params = ['rfEvapStart_amp', 'rfEvapStart_freq', 'rfEvapMid_amp', 'rfEvapEnd_amp', 'rfEvapEnd_freq']
param_ranges = [[0.0,1.0],[10.0,40.0], [0.0, 1.0], [0.0, 1.0], [1.0, 20.0]]
optimizer = DownhillSimplex(save_file_path, params, param_ranges, initial_point = [0.5623, 22, .78, .45, 4] )


df = data()
try:
    singleshot_filename = 'single_shot_optimization'
    fitness = df[singleshot_filename,'fitness'].tolist()
    trial_id = df['trial_id'].tolist()
    fitnesses = pd.DataFrame({'fitness':fitness,'trial_id': trial_id})
except(KeyError):
    print(''''No fitnesses found. If this is not the first run of the 
          optimization routine, check that you have given the right
          name for the singleshot routine to use and that it is functioning 
          properly. Setting fitnesses dictionary to empty.''')
    fitnesses = pd.DataFrame({})

new_trials, fitness_best = optimizer.get_trials(fitnesses)
print('New trials:')
print(new_trials)


for ind in new_trials.index.values.tolist(): 
    globals = new_trials.loc[ind].to_dict()
    runmanager.remote.set_globals(globals)
    runmanager.remote.engage()

try:
    _figMulti_ = plt.figure()
    #plt.plot(trial_id, fitness, 'bo')
    plt.plot(fitness_best, 'bo')
except:
    pass