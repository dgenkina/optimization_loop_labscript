# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:13:55 2019

@author: dng5
"""
import runmanager as rm
import numpy as np
from functools import wraps
from datetime import date, datetime
import os
import sys
from labscript_utils.labconfig import LabConfig
import labscript_utils.h5_lock
import h5py

    
    
def check_for_iterables_and_evaluate(seq_globals):
    
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
    
    
def create_folder(experiment_name):
    sequence_index = 0
    exp_config = LabConfig()
    shot_storage = exp_config.get('paths', 'experiment_shot_storage')
    
    folder = rm.generate_output_folder(experiment_name, shot_storage,
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
            folder = rm.generate_output_folder(experiment_name, shot_storage,
                                               date.today().strftime('%Y\%m\%d'),
                                               sequence_index)
            folder = os.path.join(os.path.dirname(folder),
                                  '%04d_test'%sequence_index)
    
    print('Created {} folder.'.format(folder))
    return folder, sequence_index
        
        
def run_new_batch_of_shots(param_df,globals_filepath_list,experiment_name):
    param_name_list = param_df.columns.tolist()
    param_array = param_df.to_numpy()
    groups = rm.get_all_groups(globals_filepath_list)
    
    seq_globals = rm.get_globals(groups)
    seq_globals, evaled_globals, expansions = check_for_iterables_and_evaluate(seq_globals)
    
    shots=[]
    for ind_shot in np.arange(param_array.shape[0]):
        shot = rm.expand_globals(seq_globals, evaled_globals)[0]  # dict
        for ind_param,param_name in enumerate(param_name_list):
            shot[param_name] = param_array[ind_shot,ind_param]
    
    
        
        shots.append(shot)
    print('Shots fired:')
    print(shots)
    
    folder, sequence_index = create_folder(experiment_name)    
    seq_id = rm.generate_sequence_id(experiment_name, '%Y_%m_%d')
    
    filenames = rm.make_run_files(folder, seq_globals, shots,
                                  seq_id, sequence_index, '', shuffle=True)
    
    print('Submitting {} shots.'.format(len(shots)))
    
    exp_config = LabConfig()
    script_storage = exp_config.get('paths', 'labscriptlib')
    script_name = script_storage +'\\' +  experiment_name + '.py'
    
    for fid in filenames:
        return_code, stdout, stderr = rm.compile_labscript(script_name, fid)
        print(return_code, stdout, stderr)
        response = rm.submit_to_blacs(fid)
        print(response)
    return 