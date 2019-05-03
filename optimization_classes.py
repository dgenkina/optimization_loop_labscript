import numpy as np
import pandas as pd
import os
import h5py

class OptimizationRoutine(object):
    def __init__(self, save_file, params=[], param_ranges=[], **kwargs):
        # params is a list of parameter names 
        # param_ranges is a list of lists, a list of tuples, or an array of min, max values for each param
        self.save_file = save_file
        self.temp_file = os.path.join(os.path.dirname(save_file),'optimization_temp')
        self.trials = pd.DataFrame()
        if os.path.exists(save_file):
            self.restore_state(save_file)
        else:
            self.initialize(save_file, params,param_ranges, **kwargs)

    def initialize(self, save_file, *args, **kwargs):
        # Make initial shots and save h5 file
        pass

    def restore_state(self, save_file):
        with h5py.File(save_file) as f:
            self.trials = pd.read_hdf(f,'trials')
        # open h5 file, load state
        pass


    def save_state(self, save_file):
        # Add/update the h5 file with new state
        self.trials.to_hdf(self.temp_file, key='trials')
        os.rename(self.temp_file,self.save_file)
        pass

    def update_fitnesses(self, fitnesses):
        # Append new fitnesses to the fitnesses dataset
        for trial_id, fitness in fitnesses.items():
            if self.trials[trial_id]['fitness'] == np.nan:
                self.trials[trial_id]['fitness'] = fitness

    def get_trials(self, fitnesses, resend_pending=False):
        pass


class GradientDescent(OptimizationRoutine):
    def __init__(self, save_file, params, param_ranges, alpha=1.0, epsilon = 0.01):
        OptimizationRoutine.__init__(self, save_file, params, alpha=alpha, epsilon = epsilon)


    def initialize(self, save_file, params, param_ranges, alpha=alpha, epsilon = epsilon):
        # Define the columns of your dataframe
        # Must have all parameter names and 'fitness'
        columns = params + ['fitness','alpha','set','set_size']
        self.trials = pd.DataFrame(columns)

        #Find the initial point randomly
        num_params = self.param_name_list.size
        initial_point = np.random.rand(num_params)*(np.array(param_ranges)[:,1]-np.array(param_ranges)[:,0]) + np.array(param_ranges)[:,0] #random values in the parameter ranges
        epsilon_matrix = np.append([np.zeros(num_params)],epsilon*(np.array(param_ranges)[:,1]-np.array(param_ranges)[:,0])*np.diag(np.ones(num_params)),axis=0)
        initial_points = epsilon_matrix + initial_point
        initial_points = self.to_real(initial_points)
        for i in range(2):
            trial = {}
            for param in params:
                trial[param] = 


        OptimizationRoutine.save_state(self, save_file)
        return initial_trails

    def restore_state(self, save_file):
        OptimizationRoutine.restore_state(self, save_file)
        # other state restoration here

    def get_trials(self, fitnesses, resend_pending=False):
        self.update_fitnesses(fitnesses)
        trials = your_code_here()
        self.save_state()
        return trials


