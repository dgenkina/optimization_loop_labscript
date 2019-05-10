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
        self.params = params
        self.param_ranges = np.array(param_ranges)
        if os.path.exists(save_file):
            self.restore_state(save_file)
        else:
            self.initialize(save_file, self.params,self.param_ranges, **kwargs)

    def initialize(self, save_file, *args, **kwargs):
        # Make dataframe and save h5 file
        pass

    def restore_state(self, save_file):
        self.trials = pd.read_hdf(save_file,'trials')
        with h5py.File(save_file) as f:
            params = f.attrs['params']
            if np.all(params==self.params):
                pass
            else:
                print('Cannot change parameters mid optimization')
                print('Falling back to previous parameters:')
                self.params = params
                print(self.params)
        # open h5 file, load state
        pass


    def save_state(self, save_file):
        # Add/update the h5 file with new state
        self.trials.to_hdf(self.temp_file, key='trials')
        with h5py.File(self.temp_file) as f:
            f.attrs['params'] = self.params
        try:
            os.remove(self.save_file)
        except:
            pass
        os.rename(self.temp_file,self.save_file)
        pass

    def update_fitnesses(self, fitnesses):
        #input a fitnesses dataframe with columns 'trial_id' and 'fitness'}
        # Append new fitnesses to the fitnesses dataset
        for ind, fit_dict in fitnesses.iterrows():
            new_fitness = pd.isnull(self.trials['fitness'])
            right_point = self.trials['trial_id']==fit_dict['trial_id']
            conditions = np.logical_and(new_fitness, right_point)
            self.trials.loc[conditions,'fitness'] = fit_dict['fitness']

        return

    def get_trials(self, fitnesses, resend_pending=False):
        pass
    
    def check_bounds_and_clamp(self,param_name,param_value):
        param_ind = self.params.index(param_name)
        if param_value < self.param_ranges[param_ind,0]:
            print(param_name + ' value of ' + str(param_value) + ' below defined lower bound ' + str(self.param_ranges[param_ind,0]))
            clamped_param = self.param_ranges[param_ind,0]
            print('Setting ' + param_name + ' to minimum valueof '+str(clamped_param))
        elif param_value > self.param_ranges[param_ind,1]:
            print(param_name + ' value of ' + str(param_value) + ' above defined lower bound ' + str(self.param_ranges[param_ind,1]))
            clamped_param = self.param_ranges[param_ind,1]
            print('Setting ' + param_name + ' to maximum value of '+str(clamped_param))
        else:
            clamped_param = param_value
        return clamped_param


class GradientDescent(OptimizationRoutine):
    def __init__(self, save_file, params, param_ranges, alpha=1.0, epsilon = 0.01):
        OptimizationRoutine.__init__(self, save_file, params, param_ranges, alpha=alpha, epsilon = epsilon)
        self.alpha = alpha
        self.epsilon = epsilon


    def initialize(self, save_file, params, param_ranges, alpha=1.0, epsilon = 0.01):
        # Define the columns of your dataframe
        # Must have all parameter names and 'trial_id','fitness'
        columns = params + ['trial_id','fitness','alpha','set','num_in_set','param_ranges']
        self.trials = pd.DataFrame(columns = columns)
        OptimizationRoutine.save_state(self, save_file)
        return

    def restore_state(self, save_file):
        OptimizationRoutine.restore_state(self, save_file)
        # other state restoration here
        return

    def get_trials(self, fitnesses, resend_pending=False):
        #TODO: make resend pending option to resend trials that don't have
        # fitnesses, for the case when re-starting an optimization that was
        # stopped abruptly
        OptimizationRoutine.update_fitnesses(self, fitnesses) 
        
        num_params = np.array(self.params).size
        param_diff = self.param_ranges[:,1]-self.param_ranges[:,0]
        param_min = self.param_ranges[:,0]
        cols_to_output = np.append(self.params,'trial_id')
        
        if self.trials.shape[0] == 0:
            #Find the initial point randomly
            initial_point = np.random.rand(num_params)*param_diff + param_min  #random values in the parameter ranges
            
            #make trials of the initial point plus points for derivatives in each direction
            for i in range(num_params+1):
                trial = {}
                for ind, param in enumerate(self.params):
                    trial[param] = self.check_bounds_and_clamp(param,initial_point[ind])
                if i>0:
                    param_delta = self.params[i-1]
                    delta_val = trial[param_delta] + self.epsilon*param_diff[i-1]
                    delta_val = self.check_bounds_and_clamp(param_delta, delta_val)
                    trial[param_delta] = delta_val
                trial['trial_id'] = i
                trial['fitness'] = np.nan
                trial['alpha'] = self.alpha
                trial['set'] = 1
                trial['num_in_set'] = i
                trial['param_ranges'] = self.param_ranges
                self.trials = self.trials.append(trial, ignore_index=True)
                
            new_trials = self.trials[cols_to_output]
            
        else:   
            set_size = num_params + 1
            max_id = np.max(self.trials['trial_id'])
            finished_trials = self.trials.loc[pd.notnull(self.trials['fitness'])]
            last_set_num = np.max(self.trials['set'])
            last_trials = finished_trials.loc[finished_trials['set']==last_set_num]
            current_set = last_set_num + 1

            #calculate new shots if the last set is done
            if np.all(np.isin(np.arange(set_size), last_trials['num_in_set'])):
                
                #get gradients in each direction
                last_center = last_trials.loc[last_trials['num_in_set']==0, self.params].to_numpy()[0]
                dx = last_trials.loc[last_trials['num_in_set']>0,self.params].to_numpy() - last_center
                dy = last_trials.loc[last_trials['num_in_set']>0,'fitness'].to_numpy() - last_trials.loc[last_trials['num_in_set']==0, 'fitness'].to_numpy()
                gradient = np.sum(dx/dy, axis=0)

                
                #get new point center
                new_center = last_center - self.alpha*gradient 

                #make trials of the point center plus points for derivatives in each direction
                for i in range(set_size):
                    trial = {}
                    for ind, param in enumerate(self.params):
                        trial[param] = self.check_bounds_and_clamp(param,new_center[ind])
                    if i>0:
                        param_delta = self.params[i-1]
                        delta_val = trial[param_delta] + self.epsilon*param_diff[i-1]
                        delta_val = self.check_bounds_and_clamp(param_delta, delta_val)
                        trial[param_delta] = delta_val
                    trial['trial_id'] = max_id + i + 1
                    trial['fitness'] = np.nan
                    trial['alpha'] = self.alpha
                    trial['set'] = current_set
                    trial['num_in_set'] = i
                    trial['param_ranges'] = self.param_ranges
                    self.trials = self.trials.append(trial, ignore_index=True)
            new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]
            
        OptimizationRoutine.save_state(self, self.save_file)
        return new_trials

    def check_bounds_and_clamp(self,param_name,param_value):
        return OptimizationRoutine.check_bounds_and_clamp(self,param_name,param_value)


"""
Runner = GenericDescent()
Runner.params = [] # things to remember
# all other settings done here too (trials, threshold, ....)

def function(parent, state, info):
    # state = parent.get_state())
    
    # info = parent.get_last_evaulatoins()
    
    ....
    parent.request_evaulation(parms)
    parent.request_evaulation(parms1)
    parent.request_evaulation(parms2)
    parent.request_evaulation(parms3)
    
   return state

Runner.func = function

"""


class GenericDescent(OptimizationRoutine):
    def __init__(self, save_file, params, param_ranges, alpha=1.0, epsilon = 0.01):
        OptimizationRoutine.__init__(self, save_file, params, param_ranges, alpha=alpha, epsilon = epsilon)
        self.alpha = alpha
        self.epsilon = epsilon
        self.func = None


    def initialize(self, save_file, params, param_ranges, alpha=1.0, epsilon = 0.01):
        # Define the columns of your dataframe
        # Must have all parameter names and 'trial_id','fitness'
        columns = params + ['trial_id','fitness','alpha','set','num_in_set','param_ranges']
        self.trials = pd.DataFrame(columns = columns)
        OptimizationRoutine.save_state(self, save_file)
        return

    def restore_state(self, save_file):
        OptimizationRoutine.restore_state(self, save_file)
        # other state restoration here
        return

    def get_trials(self, fitnesses, resend_pending=False):
        #TODO: make resend pending option to resend trials that don't have
        # fitnesses, for the case when re-starting an optimization that was
        # stopped abruptly
        OptimizationRoutine.update_fitnesses(self, fitnesses) 
        
        num_params = np.array(self.params).size
        param_diff = self.param_ranges[:,1]-self.param_ranges[:,0]
        param_min = self.param_ranges[:,0]
        cols_to_output = np.append(self.params,'trial_id')
        
        self.func(self)
        
        if self.trials.shape[0] == 0:
            #Find the initial point randomly
            initial_point = np.random.rand(num_params)*param_diff + param_min  #random values in the parameter ranges
            
            #make trials of the initial point plus points for derivatives in each direction
            for i in range(num_params+1):
                trial = {}
                for ind, param in enumerate(self.params):
                    trial[param] = self.check_bounds_and_clamp(param,initial_point[ind])
                if i>0:
                    param_delta = self.params[i-1]
                    delta_val = trial[param_delta] + self.epsilon*param_diff[i-1]
                    delta_val = self.check_bounds_and_clamp(param_delta, delta_val)
                    trial[param_delta] = delta_val
                trial['trial_id'] = i
                trial['fitness'] = np.nan
                trial['alpha'] = self.alpha
                trial['set'] = 1
                trial['num_in_set'] = i
                trial['param_ranges'] = self.param_ranges
                self.trials = self.trials.append(trial, ignore_index=True)
                
            new_trials = self.trials[cols_to_output]
            
            
        else:   
            set_size = num_params + 1
            max_id = np.max(self.trials['trial_id'])
            finished_trials = self.trials.loc[pd.notnull(self.trials['fitness'])]
            last_set_num = np.max(self.trials['set'])
            last_trials = finished_trials.loc[finished_trials['set']==last_set_num]
            current_set = last_set_num + 1

            #calculate new shots if the last set is done
            if np.all(np.isin(np.arange(set_size), last_trials['num_in_set'])):
                
                #get gradients in each direction
                last_center = last_trials.loc[last_trials['num_in_set']==0, self.params].to_numpy()[0]
                dx = last_trials.loc[last_trials['num_in_set']>0,self.params].to_numpy() - last_center
                dy = last_trials.loc[last_trials['num_in_set']>0,'fitness'].to_numpy() - last_trials.loc[last_trials['num_in_set']==0, 'fitness'].to_numpy()
                gradient = np.sum(dx/dy, axis=0)

                
                #get new point center
                new_center = last_center - self.alpha*gradient 

                #make trials of the point center plus points for derivatives in each direction
                for i in range(set_size):
                    trial = {}
                    for ind, param in enumerate(self.params):
                        trial[param] = self.check_bounds_and_clamp(param,new_center[ind])
                    if i>0:
                        param_delta = self.params[i-1]
                        delta_val = trial[param_delta] + self.epsilon*param_diff[i-1]
                        delta_val = self.check_bounds_and_clamp(param_delta, delta_val)
                        trial[param_delta] = delta_val
                    trial['trial_id'] = max_id + i + 1
                    trial['fitness'] = np.nan
                    trial['alpha'] = self.alpha
                    trial['set'] = current_set
                    trial['num_in_set'] = i
                    trial['param_ranges'] = self.param_ranges
                    self.trials = self.trials.append(trial, ignore_index=True)
            new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]
            
        OptimizationRoutine.save_state(self, self.save_file)
        return new_trials

    def check_bounds_and_clamp(self,param_name,param_value):
        return OptimizationRoutine.check_bounds_and_clamp(self,param_name,param_value)

