
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
        dtype_list = [float]*len(params) + [int, float, float, int, int, float]
        self.trials = pd.DataFrame(columns = columns)
        for ind, column in enumerate(columns): self.trials[column] = self.trials[column].astype(dtype_list[ind])
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

class DownhillSimplex(OptimizationRoutine):
    def __init__(self, save_file, params, param_ranges, side = 1, initial_point = []):
        OptimizationRoutine.__init__(self, save_file, params, param_ranges)
        self.side = side
        self.initial_point = initial_point


    def initialize(self, save_file, params, param_ranges, side = 1, initial_point = []):
        # Define the columns of your dataframe
        # Must have all parameter names and 'trial_id','fitness'
        #===================================================================================
        columns = params + ['trial_id','fitness','set','num_in_set','param_ranges']
        #===================================================================================
        self.trials = pd.DataFrame(columns = columns)
        self.save_dict = {}
        OptimizationRoutine.save_state(self, save_file)
        return

    def restore_state(self, save_file):
        OptimizationRoutine.restore_state(self, save_file)
        self.save_dict = {}
        with h5py.File(save_file) as f:
            names = f.attrs['dict_keys']
            for ind, name in enumerate(names):
                self.save_dict[name] = f.attrs[name] 
        # other state restoration here
        return
    
    def save_state(self, save_file):
        # Add/update the h5 file with new state
        self.trials.to_hdf(self.temp_file, key='trials')
        with h5py.File(self.temp_file) as f:
            f.attrs['params'] = self.params
            names = list(self.save_dict.keys())
            f.attrs['dict_keys'] = names
            for ind, name in enumerate(names):
                f.attrs[name] = self.save_dict[name]
        try:
            os.remove(self.save_file)
        except:
            pass
        
        
        os.rename(self.temp_file,self.save_file)
        pass

    def get_trials(self, fitnesses, resend_pending=False):
        #TODO: make resend pending option to resend trials that don't have
        # fitnesses, for the case when re-starting an optimization that was
        # stopped abruptly
        OptimizationRoutine.update_fitnesses(self, fitnesses) 
        
        num_params = np.array(self.params).size
        param_diff = self.param_ranges[:,1]-self.param_ranges[:,0]
        param_min = self.param_ranges[:,0]
        cols_to_output = np.append(self.params,'trial_id')
        
        #=============================================================================
        #if this is the first set of shots to be run
        if self.trials.shape[0] == 0:
            print('Making first trials')
            # Makes an original array of random starting values for each of the parameters
            if self.initial_point == []:
                initial_point = np.random.rand(num_params)*param_diff + param_min
            else:
                initial_point = self.initial_point
                
            # Creates initial arrays for the coordinates
            x_vertices = np.zeros((num_params + 1, num_params))
            
            # Adds the initial point to the array to create the other vertex values
            x_vertices[0] = initial_point
            
            set_size = num_params + 1
            # Makes simplex in the determined number of dimensions with a specific side length
            for value in range(1, num_params + 1):
                    x_vertices[value] = initial_point
                    x_vertices[value, value - 1] = initial_point[value - 1] + self.side
           
            # Make trials of the initial points 
            for i in range(x_vertices.shape[0]):
                trial = {}
                for ind, param in enumerate(self.params):
                    trial[param] = self.check_bounds_and_clamp(param, x_vertices[i,ind])
                trial['trial_id'] = i
                trial['fitness'] = np.nan
                trial['set'] = 0
                trial['num_in_set'] = i
                trial['param_ranges'] = self.param_ranges
                self.trials = self.trials.append(trial, ignore_index=True)
            tag = 'initial points'
            self.save_dict['tag'] = tag
            self.save_dict['set_size'] = set_size
            self.save_dict['x vertices'] = x_vertices

            fitness_best = []
            self.save_dict['fitness_best'] = fitness_best
            print ('fitness best type before saving')
            print(type(fitness_best))
            #self.save_dict['fitness_best'] = fitness_best
            # Tells the experiment to run the new trials  
            new_trials = self.trials[cols_to_output]
            # Creates a list to plot the best y_values
            
        else:   
            print('Not first trial')
            set_size = self.save_dict['set_size']
            fitness_best = self.save_dict['fitness_best']
            fitness_best = list(fitness_best)
            
            max_id = np.max(self.trials['trial_id'])
            finished_trials = self.trials.loc[pd.notnull(self.trials['fitness'])]
            last_set_num = np.max(self.trials['set'])
            last_trials = finished_trials.loc[finished_trials['set']==last_set_num]
            current_set = last_set_num + 1
            
            
            #calculate new shots if the last set is done
            #if previous shots have been ran and are done
            if np.all(np.isin(np.arange(set_size), last_trials['num_in_set'])):  
                print('Previous set completed')
                tag = self.save_dict['tag']
                #fitness_best = self.save_dict['fitness_best']
                # If this is the first set after we created the initial array of x_vertices:
                if last_set_num == 0 or tag == 'shrunk':
                    print('If second run or shrunk')
                    # Gets back my x_vertices array
                    x_vertices = self.save_dict['x vertices']
                    
                    # Retrieves the fitness of the function at each point
                    y_vertices = np.array(last_trials['fitness'].tolist())
         
                    # Indexes the fitness to locate the best and worst vertices in the arrays
                    index_best = np.argmin(y_vertices)
                    index_worst = np.argmax(y_vertices)
                    
                    # Calculates move distance for the new test point
                    move_distance = (-(num_params + 1) * x_vertices[index_worst])/num_params
                    fitness_best.append(y_vertices[index_best])
                    print (fitness_best)
                    
                    # Finds a new point by reflecting the simplex and takes the function value there
                    x_new = x_vertices[index_worst] + 2 * move_distance
                    self.save_dict['x_new'] = x_new
                    
                    
                    # Establishes a new set size because it'll only be taking one point at a time
                    set_size = 1  
                    
                    trial = {}
                    for ind, param in enumerate(self.params):
                        trial[param] = self.check_bounds_and_clamp(param, x_new[ind])       
                    trial['trial_id'] = max_id + 1
                    trial['fitness'] = np.nan
                    trial['set'] = current_set
                    trial['num_in_set'] = 0
                    trial['param_ranges'] = self.param_ranges
                    self.trials = self.trials.append(trial, ignore_index=True)
                    print(trial)
                    tag = 'reflected'
                    self.save_dict['tag'] = tag
                    self.save_dict['y_vertices'] = y_vertices
                    self.save_dict['set_size'] = set_size
                    #self.save_dict['fitness_best'] = fitness_best
                    
                    new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]
                
                else:
                    # If this is not the second time running the experiment
                    print("Not second time running or shrunk")
                    # Brings back my arrays of vertices and other stuff from the saved file
                    x_vertices = self.save_dict['x vertices']
                    y_vertices = np.array(self.save_dict['y_vertices'])
                    x_new = self.save_dict['x_new']
                    tag = self.save_dict['tag']

                    # Locates the vertices with the best and worst fitnesses
                    index_best = np.argmin(y_vertices)

                    index_worst = np.argmax(y_vertices)
                    
                    # Add the best fitness value to a list to plot
                    fitness_best.append(y_vertices[index_best])
                    
                    #Calculates move distance for the new test point
                    move_distance = (-(num_params + 1) * x_vertices[index_worst])/num_params
                    
                    # Gets the fitness of my last point and saves it as y_new
                    y_new = last_trials['fitness'].tolist()[0]

                    if y_new <= y_vertices[index_best]:
                        print("Reflected point is best, accepting and expanding")
                        
                        # If the new point is the best one, exchange it for the worst point
                        x_vertices[index_worst] = x_new
                        y_vertices[index_worst] = y_new
                            
                        # It's good, so let's expand the reflection by moving the new point out more
                        x_new = x_vertices[index_worst] + move_distance
                        
                        set_size = 1  
                    
                        trial = {}
                        for ind, param in enumerate(self.params):
                            trial[param] = self.check_bounds_and_clamp(param, x_new[ind])       
                        trial['trial_id'] = max_id + 1
                        trial['fitness'] = np.nan
                        trial['set'] = current_set
                        trial['num_in_set'] = 0
                        trial['param_ranges'] = self.param_ranges
                        self.trials = self.trials.append(trial, ignore_index=True)
                        
                        tag = 'expanded'
                        self.save_dict['tag'] = tag
                        self.save_dict['y_vertices'] = y_vertices
                        self.save_dict['x vertices'] = x_vertices
                        self.save_dict['x_new'] = x_new
                        self.save_dict['set_size'] = set_size
                        self.save_dict['y_new'] = y_new
                        #self.save_dict['fitness_best'] = fitness_best
                            
                        new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]
                        
                    elif y_new <= y_vertices[index_worst]:
                        # Accept expansion or reflection if this point is not best but not worst
                        print("Reflected point is accepted")
                        x_vertices[index_worst] = x_new
                        y_vertices[index_worst] = y_new
                        
                        index_best = np.argmin(y_vertices)
                        index_worst = np.argmax(y_vertices)
                        
                        # Calculates move distance for the new test point
                        move_distance = (-(num_params + 1) * x_vertices[index_worst])/num_params
                        
                        # Adds the best fitness value to a list that I can plot
                        fitness_best.append(y_vertices[index_best])
            
                        # Finds a new point by reflecting the simplex and takes the function value there
                        x_new = x_vertices[index_worst] + 2 * move_distance                        
                        
                        # Establishes a new set size because it'll only be taking one point at a time
                        set_size = 1  
                        
                        trial = {}
                        for ind, param in enumerate(self.params):
                            trial[param] = self.check_bounds_and_clamp(param, x_new[ind])       
                        trial['trial_id'] = max_id + 1
                        trial['fitness'] = np.nan
                        trial['set'] = current_set
                        trial['num_in_set'] = 0
                        trial['param_ranges'] = self.param_ranges
                        self.trials = self.trials.append(trial, ignore_index=True)
                        tag = 'reflected'
                        
                        self.save_dict['tag'] = tag
                        self.save_dict['y_vertices'] = y_vertices
                        self.save_dict['x vertices'] = x_vertices
                        self.save_dict['x_new'] = x_new
                        self.save_dict['set_size'] = set_size
                        #self.save_dict['fitness_best'] = fitness_best
                        
                        new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]

                        
                    else:
                        print("The reflected point is the worst point")
                        if tag == 'contracted' and y_new >= y_vertices[index_worst]:
                            print("The simplex has already been contracted and the new point is still the worst, shrinking")
                        # If the simplex has already been contracted and the new point is still the worst
                            for value in range(len(x_vertices)):
                                if value != index_best:
                                    x_vertices[value] = (x_vertices[value] - x_vertices[index_best]) 

                            set_size = num_params  
                            
                            for i in range(x_vertices.shape[0]):
                                trial = {}
                                for ind, param in enumerate(self.params):
                                    trial[param] = self.check_bounds_and_clamp(param, x_vertices[i, ind])       
                                trial['trial_id'] = max_id + i + 1
                                trial['fitness'] = np.nan
                                trial['set'] = current_set
                                trial['num_in_set'] = i
                                trial['param_ranges'] = self.param_ranges
                                self.trials = self.trials.append(trial, ignore_index=True)
                        
                            tag = 'shrunk'
                            self.save_dict['tag'] = tag
                            self.save_dict['y_vertices'] = y_vertices
                            self.save_dict['x vertices'] = x_vertices
                            self.save_dict['x_new'] = x_new
                            self.save_dict['set_size'] = set_size
                            #self.save_dict['fitness_best'] = fitness_best
                            new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]    
                            
                        else:    
                            print("Performing contraction")
                            # Contract the simplex by bringing that new point back in a little
                            x_new = x_vertices[index_worst] + .5 * move_distance
                                
                            set_size = 1  
                            

                            trial = {}
                            for ind, param in enumerate(self.params):
                                trial[param] = self.check_bounds_and_clamp(param, x_new[ind])       
                            trial['trial_id'] = max_id + 1
                            trial['fitness'] = np.nan
                            trial['set'] = current_set
                            trial['num_in_set'] = 0
                            trial['param_ranges'] = self.param_ranges
                            self.trials = self.trials.append(trial, ignore_index=True)
                               
                            tag = 'contracted'
                            self.save_dict['tag'] = tag
                            self.save_dict['y_vertices'] = y_vertices
                            self.save_dict['x vertices'] = x_vertices
                            self.save_dict['x_new'] = x_new
                            self.save_dict['set_size'] = set_size    
                            #self.save_dict['fitness_best'] = fitness_best
                            new_trials = self.trials.loc[self.trials['set']==current_set,cols_to_output]
            
            else:
                new_trials = pd.DataFrame({})
                print("Awaiting more trials")
        
        self.save_dict['fitness_best'] = fitness_best              
        self.save_state(self.save_file)
        return new_trials, fitness_best

    def check_bounds_and_clamp(self,param_name,param_value):
        return OptimizationRoutine.check_bounds_and_clamp(self,param_name,param_value)
