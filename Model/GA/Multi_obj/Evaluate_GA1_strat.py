
#%%
import pickle
import sys
#sys.path.append('./Model/GA')
sys.path.append('../Model/Framework/')
sys.stdout.flush()

import datetime as dt
import traceback 

import os
from os import getpid
import time
from functools import partial
import ModelFramework_no_MESA
import AgentFramework2
import Env

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import warnings

from datetime import datetime, timedelta
import datetime
#import datetime as dt
#import importlib
#importlib.reload(ModelFramework_no_MESA)
#importlib.reload(AgentFramework2)

import random 

import io
from contextlib import redirect_stdout

#%%

def run_single_ABM(individual, ABM_env):

    agents_in_beats = individual # All but last element in tuple
    #idle_strat = individual[-1] # last element in tuple

    ABM_STEP_TIME = 1

    # sum of all agents in stations
    num_agents = sum(agents_in_beats)

    # create configuration dict
    pairs = zip(ABM_env.patrol_beats, agents_in_beats)
    # Create a dictionary from zip object
    configuration = dict(pairs)


    ## Initialiaze model
    #print('Initialising ABM for ind {} and shift {}-{} ...'.format(individual, ABM_env.start_datetime, ABM_env.end_datetime))
    trap = io.StringIO()
    with redirect_stdout(trap):
        warnings.filterwarnings('ignore')
        model = ModelFramework_no_MESA.Model(ABM_env, configuration, 'Ph')
        warnings.filterwarnings('default')
        

    ## Run model 
    try: 
        trap = io.StringIO()
        with redirect_stdout(trap):
            _, _ = model.run_model(ABM_STEP_TIME)
    except:
        raise ValueError('@@@ cant run model for shift = {}, num_agents = {}, strat = {}'.format(ABM_env.start_datetime, agents_in_beats))
            
             

    ## Evaluate model  
    #(num_failed, avg_dispatch_time, avg_travel_time, avg_response_time) = model.evaluate_model()
    #print('Evaluating ABM for ind {} and shift {}-{} ...'.format(individual, ABM_env.start_datetime, ABM_env.end_datetime))
    trap = io.StringIO()
    with redirect_stdout(trap):
        df_metrics, _ = model.evaluate_model()
    #series_metrics = pd.Series(list_metrics)
    return df_metrics#series_metrics 




def run_ABMs_on_one_shift(shift, individual, patrol_beats_df = None): 
    #population = tuple(population)

    print('SHIFT: {}'.format(shift))

    #print('Importing ABM environment for shift {}...'.format(shift))
    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]

    # Import environement 
    #os.chdir('/nobackup/mednche/GA-Detroit/Baseline_experiment/')

    try:
        #### CHANGE THIS HERE TO IMPROVE: RUN IN PARALLEL FOR THE 131 PATROL BEATS INSIDE ENV SAVES LOAD OF TIME!
        warnings.filterwarnings('ignore')

        historical_crimes = pd.read_csv("../../Data/Crimes_edited_preprocessed.csv")
        historical_crimes['Patrol_beat'] = historical_crimes['Patrol_beat'].apply(str)
        #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
        historical_crimes.Date_Time = pd.to_datetime(historical_crimes.Date_Time)
        # get the incidents one year prior to start of time period (or anything desired: could be 2 years)
        historical_crimes_year = historical_crimes[(historical_crimes['Date_Time'] >= ABM_START_DATETIME- dt.timedelta(days = 365)) & 
                                        (historical_crimes['Date_Time'] < ABM_START_DATETIME)]

        cfs_incidents = pd.read_csv("../../Data/Incidents_new_preprocessed.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        # get all incidents within the time interval of the shift
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]

                                        
        
        trap = io.StringIO()
        with redirect_stdout(trap):
            ABM_env = Env.Environment('./../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = historical_crimes_year, patrol_beats_df=patrol_beats_df)
        #ABM_env = Env.Environment('./../../', ABM_START_DATETIME, ABM_END_DATETIME)
        warnings.filterwarnings('default')
    except:
        raise ValueError('@@@ Env import problem @@@ for shift = {}'.format(shift))
           
    try:
        df_metrics = run_single_ABM( individual, ABM_env = ABM_env)
    except:
        raise ValueError('@@@ failed to run ABM on an individual of pop')
    
    
   
    #print("Process {} finised".format(getpid()))
    return df_metrics




#%%
def main():

    ## THIS MAIN() IS ONLY READ WHEN RUNNING FROM TERMINAL EVALUATE_GA1_STRAT, NOT FROM WITHIN THE GA
    start_time = time.time()

    #print(os.getcwd())
    # Get list of shifts

    with open('./testing_set_scenario1.pkl', 'rb') as f:
        list_shifts = pickle.load(f)

    print('{} processes available'.format(cpu_count()))
    
    #list_shifts = list_shifts[0:2] # <--- REMOVE HERE

    #list_of_shift_dicts = run_ABMs_on_one_shift(list_shifts[5])
    print(sys.argv)
    individual = sys.argv[0]

    pool = Pool(cpu_count())
    list_of_shift_dfs = pool.map(partial(run_ABMs_on_one_shift, individual = individual), list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    print(os.getcwd())
    with open('list_of_shift_dfs.pkl', 'wb') as f:
        pickle.dump(list_of_shift_dfs, f)

    """with Pool(processes=cpu_count()) as pool:
        result_list = pool.map(run_ABMs_on_one_shift_wrapped, list_shifts)
        print("--- %s seconds ---" % (time.time() - start_time))
        """
    
    #result_list= run_ABMs_on_one_shift(list_shifts[0])

    print("--- %s seconds ---" % (time.time() - start_time))
    #print(type(result_list[0]))

    #dff = pd.DataFrame(result_list)
    #print(dff)

    #incidents['Street_index'], incidents['Patrol_beat'] = [item[0] for item in result_list], [item[1] for item in result_list] 
        
if __name__ == '__main__':
    main()
#
# %%

# %%
