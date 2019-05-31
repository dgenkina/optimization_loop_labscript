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

save_file_path = os.path.join(labconfig.get('DEFAULT', 'experiment_shot_storage'), 'Optimization_testRoutine_4.h5')

params = ['param1', 'param2']
param_ranges = [[-1.0,2.0],[-1.0,2.0]]
optimizer = DownhillSimplex(save_file_path, params, param_ranges, side = 1.5)


df = data()
try:
    singleshot_filename = 'get_fitness_singleshot'
    fitness = df[singleshot_filename,'fitness'].tolist()
    trial_id = df['trial_id'].tolist()
    fitnesses = pd.DataFrame({'fitness':fitness,'trial_id': trial_id})
except(KeyError):
    print(''''No fitnesses found. If this is not the first run of the 
          optimization routine, check that you have given the right
          name for the singleshot routine to use and that it is functioning 
          properly. Setting fitnesses dictionary to empty.''')
    fitnesses = pd.DataFrame({})


new_trials = optimizer.get_trials(fitnesses)
print('New trials:')
print(new_trials)


for ind in new_trials.index.values.tolist(): 
    globals = new_trials.loc[ind].to_dict()
    runmanager.remote.set_globals(globals)
    runmanager.remote.engage()

try:
    _figMulti_ = plt.figure()
    plt.plot(trial_id, fitness, 'bo')
except:
    pass