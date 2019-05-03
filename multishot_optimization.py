from __future__ import division
import os
import numpy as np
import labscript_utils.h5_lock
import h5py
import matplotlib.pyplot as plt
from lyse import *
import pandas as pd

from labscript_utils.labconfig import LabConfig
labconfig = LabConfig()
from optimization_classes_v2 import *

save_file_path = os.path.join(labconfig.get('DEFAULT', 'experiment_shot_storage'), 'Optimization_testRoutine_1.h5')

from analysislib.my_optimization_globals import alpha

optimizer = GradientDescent('test.h5', params=['x', 'y', 'z'], alpha=alpha)

new_shots = df[df['fitness'] == np.nan]
for shot in new_shots:
    df[]

# compute fitnesses
new_trials = optimizer.get_trials({shot['trial_id']: shot['fitnesses'] for shot in newly_analysed_shots})

for params in new_trials:
    for name, value in params.items():
        runmanager.remote.set_value(name, value)
    runmanager.remote.engage()