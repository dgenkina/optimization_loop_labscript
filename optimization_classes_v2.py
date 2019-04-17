import runmanager as rm
import numpy as np
from functools import wraps
from datetime import date, datetime
import os
import sys
from labscript_utils.labconfig import LabConfig
import labscript_utils.h5_lock
import h5py

class OptimizationProcess():
	def __init__(self,experiment_name,globals_filepath_list,param_name_list,param_min_max_list,initial_param_vals=None,use_current_param_vals=True):
		self.param_name_list=np.array(param_name_list)
		self.param_min_max_list=np.array(param_min_max_list)
		self.initial_param_vals=initial_param_vals
		self.save_Dict = {}
		self.experiment_name = experiment_name
		self.globals_filepath_list = globals_filepath_list
		self.params_ran = np.zeros((0,self.param_name_list.size))
		self.costs_ran = np.array([])
		self.params_running = np.zeros((0,self.param_name_list.size))
		self.make_globals_file()
		self.sequence_index_list =[]
		self.batch_number_list = []
		self.best_cost_list = []

	def make_globals_file(self,**kwargs):
		folder, file = os.path.split(self.globals_filepath_list[-1])
		new_globals_filepath = os.path.join(folder,'multishot_optimization.h5')
		with h5py.File(new_globals_filepath, 'w') as new_globals_file:
			new_globals_file.create_group('globals')

			try:
				main_group = new_globals_file['globals'].create_group('optimization_params')
			except(ValueError):
				main_group = new_globals_file['globals']['optimization_params']

			try:
				unitsgroup = main_group.create_group('units')
			except(ValueError):
				unitsgroup = main_group['units']

			try:
				expansiongroup = main_group.create_group('expansion')
			except(ValueError):
				expansiongroup = main_group['expansion']
				
			main_group.attrs['Stop'] = repr(False)
			unitsgroup.attrs['Stop'] = ''
			expansiongroup.attrs['Stop'] = ''

			for key, value in kwargs.items():
				main_group.attrs[key] = repr(value)
				unitsgroup.attrs[key] = ''
				expansiongroup.attrs[key] = ''

		self.globals_filepath_list.append(new_globals_filepath)

		return


	def check_for_iterables_and_evaluate(self,seq_globals):

	    evaled_globals, _, expansions = rm.evaluate_globals(seq_globals)

	    # Make sure there are no iterables in the globals
	    for group in evaled_globals.values():
	        for k, v in group.items():
	            if hasattr(v, '__iter__'):
	                print('Deleting iterable {}.'.format(k))
	                group[k] = np.nan

	    # make sure all expansions are empty
	    for k, v in expansions.items():
	        expansions[k] = u''

	    for group in seq_globals.values():
	        for k, v in group.items():
	            if v[-1]:  # if there is an expansion
	                print('Deleting expansion for iterable {}.'.format(k))
	                group[k] = v[:-1] + (u'',)

	    return seq_globals, evaled_globals, expansions


	def create_folder(self):
	    sequence_index = 0
	    exp_config = LabConfig()
	    shot_storage = exp_config.get('paths', 'experiment_shot_storage')

	    folder = rm.generate_output_folder(self.experiment_name, shot_storage,
	                                       date.today().strftime('%Y\%m\%d'),
	                                       sequence_index)
	    folder = os.path.join(os.path.dirname(folder),
	                          '%04d_test'%sequence_index)

	    # Make sure shots go to a unique directory
	    while True:
	        try:
	            os.makedirs(folder)
	            break
	        except:
	            sequence_index += 1
	            folder = rm.generate_output_folder(self.experiment_name, shot_storage,
	                                               date.today().strftime('%Y\%m\%d'),
	                                               sequence_index)
	            folder = os.path.join(os.path.dirname(folder),
	                                  '%04d_test'%sequence_index)

	    print('Created {} folder.'.format(folder))
	    return folder, sequence_index


	def run_new_batch_of_shots(self, param_array):
	    param_array = np.array(param_array)
	    groups = rm.get_all_groups(self.globals_filepath_list)

	    # Check to see if a "my class name" such as Differential Evolution exists.  Create if not
	    # within group populate default variables if they don't exist.  Otherwise keep and use the values
	    # in h5 file for optimization parameters, ... including terminate.  So make a default loops = -1 for "dont stop ever"
	    # and a "StopNow" boolean

	    # for k, v in groups.iteritems():
	    #     print(k, v)
	    seq_globals = rm.get_globals(groups)
	    # for k, v in seq_globals['Calibrations'].iteritems():
	    #     print(k, v)
	    seq_globals, evaled_globals, expansions = self.check_for_iterables_and_evaluate(seq_globals)


	    # TODO keep those as the true globals and make copies to pass to the different outer products.
	    # sg1 = seq_globals.copy()
	    # eg1 = evaled_globals.copy()
	    # exp1 = expansions.copy()

	    # --------------------- set values

	    # only set the booleans assuming the other values are sensible
	    shots=[]
	    for ind_shot in np.arange(param_array.shape[0]):
	        for ind_param,param_name in enumerate(self.param_name_list):
	            evaled_globals['main'][param_name] = param_array[ind_shot,ind_param]

	    # # and add them to outer products
	    # if param_values.size>1:
	    #     print 'adding inner products'
	    #     expansions[param_name] = u'inner'
	    #     seq_globals['main'][param_name] = \
	    #         seq_globals['main'][param_name][:-1] + (u'inner',)


	        shot = rm.expand_globals(seq_globals, evaled_globals)[0]  # dict
	        shots.append(shot)
	    print 'Shots fired:'
	    print shots

	    folder, sequence_index = self.create_folder()
	    self.current_folder = folder
	    try:
	    	self.sequence_index_list.append(sequence_index)
	    except(AttributeError):
	    	self.sequence_index_list = []
	    	self.sequence_index_list.append(sequence_index)


	    seq_id = rm.generate_sequence_id(self.experiment_name, '%Y_%m_%d')
	    # print(seq_id)

	    filenames = rm.make_run_files(folder, seq_globals, shots,
	                                  seq_id, sequence_index, '', shuffle=True)
	    # print(filenames.next())

	    print('Submitting {} shots.'.format(len(shots)))

	    exp_config = LabConfig()
	    script_storage = exp_config.get('paths', 'labscriptlib')
	    script_name = script_storage +'\\' +  self.experiment_name + '.py'

	    for fid in filenames:
	        return_code, stdout, stderr = rm.compile_labscript(script_name, fid)
	        print(return_code, stdout, stderr)
	        response = rm.submit_to_blacs(fid)
	        print(response)
	    return 

	def read_global(self,global_name):
		groups = rm.get_all_groups(self.globals_filepath_list)
		seq_globals = rm.get_globals(groups)
		found_group = False
		for group_name in groups:
			try:
				global_value = eval(seq_globals[group_name][global_name][0])
				found_group = True
			except KeyError:
				pass
		if not found_group:
				print 'Didn\'t find a group for variable ' + global_name
				print ''
		return global_value


	def read_globals(self):
		param_values = np.zeros(self.param_name_list.size)
		groups = rm.get_all_groups(self.globals_filepath_list)
		seq_globals = rm.get_globals(groups)

		for ind,param_name in enumerate(self.param_name_list):
			found_group = False
			for group_name in groups:
				try:
					param_values[ind] = eval(seq_globals[group_name][param_name][0])
					found_group = True
				except KeyError:
					pass
			if not found_group:
				print 'Didn\'t find a group for variable ' + param_name
				print ''
		return param_values

	

	def run_first_batch(self):
		initial_points = self.get_first_batch()
		print 'Running first batch of shots'
	#	print initial_points
		self.run_new_batch_of_shots(initial_points)
		self.params_running = initial_points
		print 'params running from first batch:'
		print self.params_running

		return 

	def get_first_batch(self):
		if use_current_param_vals:
			initial_points = self.read_globals()
		elif self.initial_param_vals is None:
			raise NotImplementedError('The specific optimization protocol subclass must overwrite this function!')
		else:
			initial_points = self.initial_param_vals
		return initial_points

	def run_next_batch(self,cost_list,ran_param_array):
		self.read_in_costs(cost_list,ran_param_array)
		try:
			self.best_cost_list.append(np.min(cost_list))
		except(AttributeError):
			self.best_cost_list = []
			self.best_cost_list.append(np.min(cost_list))
		try:
			batch_num = self.batch_number_list[-1] +1
			self.batch_number_list.append(batch_num)
		except(AttributeError):
			self.batch_number_list = []
			self.batch_number_list.append(1)


		if self.continue_condition():
			next_points = self.get_next_batch()
			self.run_new_batch_of_shots(next_points)
			self.params_running = next_points
		else:
			print 'Continue_condition evaluated to false'
		return

	def read_in_costs(self,cost_list,ran_param_array):
		self.params_ran = np.append(self.params_ran,self.params_running, axis = 0)
		cost_list_ordered = np.zeros(cost_list.size)
		for ind, params in enumerate(ran_param_array):
			true_ind = np.where((self.params_ran==params).all(axis=1))[0]
			cost_list_ordered[true_ind] = cost_list[ind]
		self.costs_ran = np.append(self.costs_ran,cost_list_ordered)
		self.params_running = np.zeros((0,self.param_name_list.size))


		# Save h5 file in same folder as shot files for things in this batch
		filename = self.experiment_name + 'multishot_optimization.h5'
		with h5py.File(os.path.join(self.current_folder,filename),'a') as batch_file:
			batch_file['param_name_list'] = self.param_name_list
			for key, value in self.save_Dict.iteritems():
				batch_file[key] = value 

		return

	def get_next_batch(self):
		raise NotImplementedError('The specific optimization protocol subclass must overwrite this function!')
		return next_points

	def continue_condition(self):
		stop = self.read_global('Stop')
		if stop:
			# make h5 file with meta optimization params in the same place as the shot files
			filename = self.experiment_name + 'multishot_optimization' + str(self.sequence_index_list[0])+'-'+str(self.sequence_index_list[-1])+'.h5' 
			folder, subfolder = os.path.split(self.current_folder)
			with h5py.File(os.path.join(folder,filename),'a') as batch_file:
				batch_file['param_name_list'] = self.param_name_list
				for key, value in self.save_Dict.iteritems(): 
					batch_file[key] = value

			# put best parameters into output globals file for convenience
			with h5py.File(self.globals_filepath_list[-1], 'r+') as globals_file:
				output_group = globals_file['globals'].create_group('optimized_params')
				unitsgroup = output_group.create_group('units')
				expansiongroup = output_group.create_group('expansion')
				best_ind = np.argmax(self.cost_list)
				best_params = self.params_ran[best_ind]

				for ind, parm_name in enumerate(self.param_name_list):
					output_group.attrs[parm_name] = repr(best_params[ind])
					unitsgroup.attrs[parm_name] = ''
					expansiongroup.attrs[parm_name] = ''

		return not stop

	#
	# Some functions to use to convert between true parameter values and 'nice' (normalized to be between 0 and 1) values
	# and back to 'real' 	
	def to_nice(self,real_param_value_list):
		real_param_value_list=np.array(real_param_value_list)
		param_min_max_list = self.param_min_max_list
		if len(real_param_value_list.shape)==1:
			nice_param_value_list = (real_param_value_list-param_min_max_list[:,0])/(param_min_max_list[:,1]-param_min_max_list[:,0])
		elif len(real_param_value_list.shape)==2:
			nice_param_value_list = np.zeros(real_param_value_list.shape)
			for ind1 in np.arange(real_param_value_list.shape[0]):
				nice_param_value_list[ind1] = (real_param_value_list[ind1]-param_min_max_list[:,0])/(param_min_max_list[:,1]-param_min_max_list[:,0])
		else:
			print 'Not a 1-d array of parameter values or 2-d array of differnt shots of parameter values'
			print 'Could not convert real parameters to nice'
		return nice_param_value_list

	def to_real(self,nice_param_value_list):
		nice_param_value_list=np.array(nice_param_value_list)
		param_min_max_list = self.param_min_max_list
		if len(nice_param_value_list.shape)==1:
			real_param_value_list = (param_min_max_list[:,1]-param_min_max_list[:,0])*nice_param_value_list+param_min_max_list[:,0]
		elif len(nice_param_value_list.shape)==2:
			real_param_value_list = np.zeros(nice_param_value_list.shape)
			for ind1 in np.arange(nice_param_value_list.shape[0]):
				real_param_value_list[ind1] = (param_min_max_list[:,1]-param_min_max_list[:,0])*nice_param_value_list[ind1]+param_min_max_list[:,0]
		else:
			print 'Not a 1-d array of parameter values or 2-d array of differnt shots of parameter values'
			print 'Could not convert nice paramets to real'
		return real_param_value_list


	#
	# Check that the parameter values recommended by whatever algorithm are within the bounds specified at the beginning
	#
	def check_bounds_and_clamp(self,real_param_value_list):
		real_param_value_list=np.array(real_param_value_list)
		if len(real_param_value_list.shape)==1:
			for ind,param in enumerate(real_param_value_list):
				if param <= self.param_min_max_list[ind,0]:
					print self.param_name_list[ind] + ' value of ' + str(param) + ' below defined lower bound ' + str(self.param_min_max_list[ind,0])
					real_param_value_list[ind] = self.param_min_max_list[ind,0]
					print 'Setting ' + self.param_name_list[ind] + ' to minimum value'
				if param >= self.param_min_max_list[ind,1]:
					print self.param_name_list[ind] + ' value of ' + str(param) +' above defined upper bound ' + str(self.param_min_max_list[ind,1])
					real_param_value_list[ind] = self.param_min_max_list[ind,1]
					print 'Setting ' + self.param_name_list[ind] + ' to maximum value'
		elif len(real_param_value_list.shape)==2:
			for ind1 in np.arange(real_param_value_list.shape[0]):
				for ind2,param in enumerate(real_param_value_list[ind1]):
					if param <= self.param_min_max_list[ind2,0]:
						print self.param_name_list[ind2] + ' value of ' + str(param) + ' below defined lower bound ' + str(self.param_min_max_list[ind2,0])
						real_param_value_list[ind1,ind2] = self.param_min_max_list[ind2,0]
						print 'Setting ' + self.param_name_list[ind2] + ' to minimum value'
					if param >= self.param_min_max_list[ind2,1]:
						print self.param_name_list[ind2] + ' value of ' + str(param) +' above defined upper bound ' + str(self.param_min_max_list[ind2,1])
						real_param_value_list[ind1,ind2] = self.param_min_max_list[ind2,1]
						print 'Setting ' + self.param_name_list[ind2] + ' to maximum value'
		else:
			print 'Not a 1-d array of parameter values or 2-d array of differnt shots of paarameter values'
			print 'Could not check bounds of parameters'
		return real_param_value_list





class DifferentialEvolution(OptimizationProcess):

	def __init__(self, experiment_name, globals_filepath_list,param_name_list, bounds, mut=0.8, crossp=0.7, popsize=20, loops=100,samples=1,resample=5,elite=0.5,use_current_param_vals=True):
		OptimizationProcess.__init__(self, experiment_name, globals_filepath_list, param_name_list, bounds, use_current_param_vals=use_current_param_vals)
		self.dimensions = self.param_name_list.size
		#self.nonoise = nonoise
		self.DE_parms_dict = {}
		self.DE_parms_dict['mut'] = mut
		self.DE_parms_dict['crossp'] = crossp
		self.popsize = popsize
		self.DE_parms_dict['loops'] = loops
		self.DE_parms_dict['samples'] = samples
		self.DE_parms_dict['resample'] = resample
		elite = int(np.ceil(popsize*elite))
		elite = np.max([elite,4])
		self.DE_parms_dict['elite'] = elite
		self.bounds = bounds
		min_b, max_b = np.asarray(bounds).transpose()
		self.min_b = min_b
		self.max_b = max_b 
		self.diff = np.fabs(min_b - max_b)
		self.make_globals_file(**self.DE_parms_dict)
	    
		pop = []
		for j in range(popsize-1):
			agent = {'sequence': np.random.rand(self.dimensions),'fitness': 0.0,'n_sample': 0, 'sample': samples}

			pop.append(agent)
		if use_current_param_vals:
			print 'Reading in parameter values from globals file:'
			initial_param_vals = self.read_globals()
			print self.param_name_list
			print initial_param_vals
			agent = {'sequence': (initial_param_vals - self.min_b)/self.diff,'fitness': 0.0,'n_sample': 0, 'sample': samples}
			pop.append(agent)
		else:
			agent = {'sequence': np.random.rand(self.dimensions),'fitness': 0.0,'n_sample': 0, 'sample': samples}
			pop.append(agent)

		self.pop = pop

		self.next_batch = []

		self.best_array = [ ]
		self.fitness_array = [ ]
		self.true_fitness_array = [ ]
		self.batch_number = 0
		self.submitted = []
		self.save_Dict = {}

	def get_first_batch(self):

		self.submitted = []
		initial_points = []
		for agent in self.pop:

			# Submit this agent a number of times requsted
			for j in range(agent['sample']):
				self.submitted += [agent]
				initial_points.append(self.min_b + agent['sequence'] * self.diff)      
			agent['sample'] = 0
			agent['n_sample'] = 0
		self.batch_size = len(self.submitted)
		self.batch_number += 1

		return np.array(initial_points)

	def read_in_costs(self,cost_list,ran_param_array):

		#Read in cost values for the previous batch
		for j, agent in enumerate(self.submitted):
			true_ind = np.where((ran_param_array==self.min_b + agent['sequence'] * self.diff).all(axis=1))
			print 'true_ind:'
			print true_ind
			print 'ran_param_array:'
			print ran_param_array
			print 'searched for param vals:'
			print self.min_b + agent['sequence'] * self.diff

			fitness = np.average(cost_list[true_ind])
			agent['n_sample'] += 1
			agent['fitness'] = (fitness)# + agent['fitness']* (agent['n_sample']-1) ) / agent['n_sample']

		last_params = [self.min_b + agent['sequence'] * self.diff for agent in self.submitted]
		print 'last_params:'
		print last_params
		print 'last_fitness:'

		last_fitness = [agent['fitness'] for agent in self.submitted]
		print last_fitness	
		self.save_Dict = {'params':last_params, 'cost':last_fitness}

		#
		# Information for logging
		#

		fitness = [agent['fitness'] for agent in self.pop]

		best_idx = np.argmin(fitness)
		best = self.min_b + self.pop[best_idx]['sequence'] * self.diff

		self.best_array.append(best)
		self.fitness_array.append(fitness[best_idx])

		# Save h5 file in same folder as shot files for things in this batch
		filename = self.experiment_name + '_multishot_optimization.h5' 
		with h5py.File(os.path.join(self.current_folder,filename),'a') as batch_file:
			batch_file['param_name_list'] = self.param_name_list
			for key, value in self.save_Dict.iteritems():
				batch_file[key] = value 


		return

	def get_next_batch(self):
		# Update population (use answers to get new population)
		#
		for j, next_agent in enumerate(self.next_batch):
			agent = self.pop[j]

			next_fitness = next_agent['fitness']
			fitness = agent['fitness']
			  
			if next_fitness < fitness:
			    self.pop[j] = next_agent

		#
		# Check if the Differential evolution parameters have changed from runmanager via the globals file
		#

		for key, value in self.DE_parms_dict.items():
			val = self.read_global(key)
			self.DE_parms_dict[key] = val

		#
		# Get next batch of shots ready.
		#

		# Sort
		sort_func = lambda x: x['fitness']
		self.pop.sort(key=sort_func)        

		# New Test Points
		self.next_batch = []
		for j in range(self.popsize):
			idxs = [idx for idx in range(self.popsize) if idx != j]
			idxs = idxs[0:self.DE_parms_dict['elite']]

			s = np.random.choice(idxs, 3, replace = False)
			s = [self.pop[select]['sequence'] for select in s]

			mutant = np.clip(s[0] + self.DE_parms_dict['mut'] * (s[1] - s[2]), 0, 1)

			cross_points = np.random.rand(self.dimensions) < self.DE_parms_dict['crossp']
			if not np.any(cross_points):
				cross_points[np.random.randint(0, self.dimensions)] = True
			trial = np.where(cross_points, mutant, self.pop[j]['sequence'])

			agent = {'sequence': trial,'fitness': 0.0,'n_sample': 0, 'sample': self.DE_parms_dict['samples']}

			self.next_batch.append(agent)

		# resample
		for agent in self.pop:
			if self.DE_parms_dict['resample']*np.random.rand() < 1:
				agent['sample'] = self.DE_parms_dict['samples']


		self.submitted = []
		next_points = []
		for agent in self.pop + self.next_batch:

			# Submit this agent a number of times requsted
			for j in range(agent['sample']):
				self.submitted += [agent]
				next_points.append(self.min_b + agent['sequence'] * self.diff)      
			agent['sample'] = 0
			agent['n_sample'] = 0
		self.batch_size = len(self.submitted)
		self.batch_number += 1





		return np.array(next_points)
		#
		# TODO: in the "group" for this file, make new variables containing the optimized parameters? Or HA! a tuple (has to be a tuple) containing names
		# and one containg best values. And have a overwrite boolean option that can actuyaly update the the best parameters in the variables file
		# And/or create a totally new groups file with the new values
		#


	def continue_condition(self):
		stop = False
		stop_bool = self.read_global('Stop')

		if stop_bool:
			stop = True
			print 'Stopping - \'Stop\' global changed to True'

		elif self.batch_number >= self.DE_parms_dict['loops']:
			stop = True
			print 'All loops ran!'

		if stop:
			
			# print 'best value array:'
			# print self.best_array
			# print 'best fitness array:'
			# print self.fitness_array
			self.save_Dict = {'best_value_array':self.best_array, 'best_cost_array': self.fitness_array}

			# make h5 file with meta optimization params in the same place as the shot files
			folder, subfolder = os.path.split(self.current_folder)
			filename = self.experiment_name + '_multishot_optimization_' + str(self.sequence_index_list[0])+'-'+str(self.sequence_index_list[-1])+'.h5' 		
			with h5py.File(os.path.join(folder,filename),'a') as batch_file:
				batch_file['param_name_list'] = self.param_name_list
				for key, value in self.save_Dict.iteritems(): 
					batch_file[key] = value

			with h5py.File(self.globals_filepath_list[-1], 'r+') as globals_file:
				output_group = globals_file['globals'].create_group('optimized_params')
				unitsgroup = output_group.create_group('units')
				expansiongroup = output_group.create_group('expansion')
				best_ind = np.argmin(self.fitness_array)
				best_params = self.best_array[best_ind]

				for ind, parm_name in enumerate(self.param_name_list):
					output_group.attrs[parm_name] = repr(best_params[ind])
					unitsgroup.attrs[parm_name] = ''
					expansiongroup.attrs[parm_name] = ''
		return not stop


class GradientDescent(OptimizationProcess):

	def __init__(self,param_name_list,param_min_max_list,initial_param_vals=None):
		self.param_name_list=np.array(param_name_list)
		self.batch_size=self.param_name_list.size + 1
		self.param_min_max_list=np.array(param_min_max_list)
		self.initial_param_vals=initial_param_vals
		self.epsilon = 1.0e-3
		self.alpha = 10.0e-3

	# def to_nice(real_param_value_list):
	# 	nice_param_value_list = (real_param_value_list-param_min_max_list[:,0])/(param_min_max_list[:,1]-param_min_max_list[:,0])
	# 	return nice_param_value_list

	# def to_real(nice_param_value_list):
	# 	real_param_value_list = (param_min_max_list[:,1]-param_min_max_list[:,0])*nice_param_value_list+param_min_max_list[:,0]
	# 	return real_param_value_list


	def run_first_batch(self):
		num_params = self.param_name_list.size
		initial_point = np.random.rand(num_params)
		epsilon_matrix = np.append([np.zeros(num_params)],self.epsilon*np.diag(np.ones(num_params)),axis=0)
		initial_points = epsilon_matrix + initial_point
		initial_points = self.to_real(initial_points)
		print 'Initial points array:'
		print initial_points
		return initial_points


	def run_next_batch(self,real_params_list,cost_list):
		real_params_list=np.array(real_params_list)
		cost_list = np.array(cost_list)
		num_params = real_params_list.shape[1]
		num_shots = real_params_list.shape[0]
		last_center_location = (np.sum(real_params_list,axis=0)-self.epsilon)/num_shots
		nice_params_list = self.to_nice(real_params_list)
		gradient_list = (cost_list[1:]-cost_list[0])*(np.sum(nice_params_list[1:]-nice_params_list[0],axis=1))
		nice_last_center_location = self.to_nice(last_center_location)
		nice_new_center_location = nice_last_center_location + self.alpha*np.dot(nice_params_list[1:]-nice_last_center_location,gradient_list)/self.epsilon
		epsilon_matrix = np.append([np.zeros(num_params)],self.epsilon*np.diag(np.ones(num_params)),axis=0)
		nice_new_param_list = epsilon_matrix + nice_new_center_location 		
		real_new_param_list = self.to_real(nice_new_param_list)
		real_new_param_list = self.check_bounds_and_clamp(real_new_param_list)
		return real_new_param_list

	def continue_condition(self,real_params_list,cost_list):
		cost_list = np.array(cost_list)
		condition_met_list = np.zeros(cost_list.size,dtype=bool)
		for ind,cost in enumerate(cost_list):
			if np.abs(cost+1.0) < 1e-3:
				condition_met_list[ind]=True
				print 'Optimum found at ' + str(real_params_list[ind])
		return not any(condition_met_list)


