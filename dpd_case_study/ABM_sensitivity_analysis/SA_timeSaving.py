
#%%
# run the SA on SA_set (100 time periods randomly chosen)
# - scenario 1: system is stable: low demand/low supply time periods (weekday 8am-16pm)
# - scenario 2: system is under pressure: high demand/medium supply time periods (weekend midnight to 8am)


import pickle
import sys
#sys.path.append('./Model/GA')
sys.path.append('../Model/Framework/')
sys.stdout.flush()

import traceback 

import os
from os import getpid
import time
from functools import partial
import ModelFramework
import AgentFramework
import Env

import geopandas as gpd

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import warnings

from datetime import datetime, timedelta
import datetime as dt
#import datetime as dt
#import importlib
#importlib.reload(ModelFramework)
#importlib.reload(AgentFramework)

import random 

import io
from contextlib import redirect_stdout

#%%

def run_single_ABM(time_saving_percent, ABM_env):
    # NB: the time periods is comprised in ABM_env
    agents_per_precinct = 4
    
    print('------')
    print('----> ',agents_per_precinct, ' agents per precinct')

    ABM_STEP_TIME = 1
    #ABM_NUM_STEPS = 10 # <---- REMOVE HERE
    
    ###### POPULATE list_num_agents_scas #######
    k= agents_per_precinct
    # Init the configuration with zero agent in each patrol beat
    list_num_agents_scas = np.zeros((len(ABM_env.patrol_beats),), dtype=int)
    list_num_agents_scas

    precincts = gpd.read_file('../data/DPD_Precincts/dpd_precincts.shp')
    precincts['name'] = precincts['name'].astype(int).astype(str)
    # remove the last row (duplicate of precinct 7)
    precincts = precincts[:-1]

    for precinct in precincts['name']:
        #print('------> precinct', precinct)
        # get scas for precinct
        precinct_scas = [sca for sca in ABM_env.patrol_beats if sca.precinct == precinct]

        # CHOOSE k HOTTEST SCAS PER PRECINCT (not random)
        # list of all number of crimes 
        list_num_crimes = [len(sca.historical_crimes) for sca in ABM_env.patrol_beats if sca.precinct == precinct]
        # chosen top k highest number of crimes
        selection_list_num_crimes = sorted(list_num_crimes, reverse=True)[:k]

        # find the scas with those chosen number of crimes, to a max of k (in case of duplicates)
        chosen_scas = [sca for sca in precinct_scas if len(sca.historical_crimes) in selection_list_num_crimes][:k]
        
        
        for sca in chosen_scas:
            list_index = ABM_env.patrol_beats.index(sca)
            #print("list_index", list_index)
            list_num_agents_scas[list_index] = 1
    ########################################

    # create configuration dictionnarry to feed in the ABM
    pairs = zip(ABM_env.patrol_beats, list_num_agents_scas)
    # Create a dictionary from zip object
    configuration = dict(pairs)
    print(configuration)

    ## Initialiaze model
    print('Initialising ABM...')
    trap = io.StringIO()
    with redirect_stdout(trap):
        warnings.filterwarnings('ignore')
        model = ModelFramework.Model(ABM_env, configuration, 'Ph', time_saving_percent=time_saving_percent)
        warnings.filterwarnings('default')
        
    #print('end_datetime', model.end_datetime)


    ## Run model
    print('Running ABM...')
    trap = io.StringIO()
    with redirect_stdout(trap):
        try: 
            _, _ = model.run_model(ABM_STEP_TIME)
        except:
            raise ValueError('@@@ shift = {}, num_agents = {}, strat = {}'.format(ABM_env.start_datetime, agents_per_precinct, idle_strat))
            
             

    ## Evaluate model  
    #(num_failed, avg_dispatch_time, avg_travel_time, avg_response_time) = model.evaluate_model()
    print('Evaluating ABM...')
    df_metrics, _ = model.evaluate_model()
    #series_metrics = pd.Series(list_metrics)

    return df_metrics#series_metrics 




def run_ABMs_on_one_shift(shift): 
    print('-------------------------------------------')
    print("I'm process", getpid())
    print('Running experiment on shift {}'.format(shift))

    print('Importing ABM environment...')
    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]
    print('... env imported.')

    # Import environment 
    print(os.getcwd())
    os.chdir('/nobackup/mednche/GA-Detroit/Validation/')
    print(os.getcwd())

    try:
        warnings.filterwarnings('ignore')

        historical_crimes = pd.read_csv("../data/Crimes_edited_preprocessed.csv")
        historical_crimes['Patrol_beat'] = historical_crimes['Patrol_beat'].apply(str)
        #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
        historical_crimes.Date_Time = pd.to_datetime(historical_crimes.Date_Time)
        # get the incidents one year prior to start of time period (or anything desired: could be 2 years)
        historical_crimes_year = historical_crimes[(historical_crimes['Date_Time'] >= ABM_START_DATETIME- dt.timedelta(days = 365)) & 
                                        (historical_crimes['Date_Time'] < ABM_START_DATETIME)]

        cfs_incidents = pd.read_csv("../data/incidents.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        # get all incidents within the time interval of the shift
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]

                                        
        ABM_env = Env.Environment('../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_crimes_year)
        warnings.filterwarnings('default')
    except:
        raise ValueError('@@@ Env import problem @@@ for shift = {} and dir = {}'.format(shift, os.getcwd()))
            

    
    # FOR SA:
    # run a single ABM on the shift for all values of list_time_saving_percent (10, 20 and 30)
    list_time_saving_percent = [10,20,30,40,50,60]
    list_of_dfs = map(partial(run_single_ABM, ABM_env = ABM_env), list_time_saving_percent)
    
    dict_for_shift = dict(zip(list_time_saving_percent,list_of_dfs))
    
    #print("Process {} finised".format(getpid()))
    return dict_for_shift
  
    

#%%
def main():
    start_time = time.time()

    # SCENARIO 1
    with open('SA_set.pkl', 'rb') as f:
        list_shifts = pickle.load(f)
    """ with open('validation_set_scenario12.pkl', 'rb') as f:
        list_shifts_scenario2 = pickle.load(f)
    
    # place both sets back to back (200 shifts in total)
    list_shifts = list_shifts_scenario1+ list_shifts_scenario2 """

    print('{} processes available'.format(cpu_count()))
    
    #list_shifts = list_shifts[0:2] # <--- REMOVE HERE

    #list_of_shift_dicts = run_ABMs_on_one_shift(list_shifts[5])

    pool = Pool(cpu_count())
    list_of_shift_dicts = pool.map(run_ABMs_on_one_shift, list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    print(os.getcwd())
    with open('results_SA_timeSaving.pkl', 'wb') as f:
        pickle.dump(list_of_shift_dicts, f)


    list_time_saving_percent = [10,20,30,40,50,60]
    dict_timeSaving = {}
    for time_saving_percent in list_time_saving_percent:
        list_dfs_for_num = [shift_dict[time_saving_percent] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_timeSaving[time_saving_percent] = pd.concat(list_dfs_for_num, ignore_index=True)
 
    
    # Save as dictionnary
    with open('dict_SA_timeSaving.pkl', 'wb') as f:
        pickle.dump(dict_timeSaving, f)
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == '__main__':
    main()
