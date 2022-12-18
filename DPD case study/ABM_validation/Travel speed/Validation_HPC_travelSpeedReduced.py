
#%%
# run the validation on two types scenarios (marked by 100 time periods each): 
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
import ModelFramework_no_MESA
import AgentFramework2
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
#importlib.reload(ModelFramework_no_MESA)
#importlib.reload(AgentFramework2)

import random 

import io
from contextlib import redirect_stdout

def getCrimesForScenario(crimes, scenario_num):
    with open('./training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        training_set_scenario = pickle.load(f)

    crimes['Patrol_beat'] = crimes['Patrol_beat'].apply(str)
    #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
    crimes.Date_Time = pd.to_datetime(crimes.Date_Time)
 
    historical_crimes_scenario = pd.DataFrame()
    for shift in training_set_scenario:
        historical_crimes_scenario = historical_crimes_scenario.append(crimes[(crimes['Date_Time'] >= shift[0]) & 
                                    (crimes['Date_Time'] < shift[1])])


    return historical_crimes_scenario


#%%
def getTargettedConfig(k, ABM_env) :
    # Init the configuration with zero agent in each patrol beat
    list_num_agents_scas = np.zeros((len(ABM_env.patrol_beats),), dtype=int)

    # get scas for precinct
    #precinct_scas = [sca for sca in ABM_env.patrol_beats if sca.precinct == precinct]

    # CHOOSE k HOTTEST SCAS PER PRECINCT (not random)
    # list of all number of crimes 
    list_num_crimes = [len(sca.historical_crimes) for sca in ABM_env.patrol_beats]
    # chosen top k highest number of crimes
    selection_list_num_crimes = sorted(list_num_crimes, reverse=True)[:k]
    # find the scas with those chisen number of crimes, to a max of k (in case of duplicates)
    chosen_scas = [sca for sca in ABM_env.patrol_beats if len(sca.historical_crimes) in selection_list_num_crimes][:k]

    for sca in chosen_scas:
        list_index = ABM_env.patrol_beats.index(sca)
        #print("list_index", list_index)
        list_num_agents_scas[list_index] = 1
    ########################################

    # create config dict
    pairs = zip(ABM_env.patrol_beats, list_num_agents_scas)
    # Create a dictionary from zip object
    configuration = dict(pairs)
    return configuration


def run_single_ABM(num_agents, ABM_env):
    """ RUN 1 ABM for aconfiguration with a given number agents (e.g. 10 agents) and an ABM_env that has the param value in it"""
    # NB: the time periods is comprised in ABM_env
    print('------')
    print('----> ',num_agents, ' agents')

    ABM_STEP_TIME = 1
    ###### POPULATE list_num_agents_scas #######
    configuration = getTargettedConfig(num_agents, ABM_env)

    ## Initialiaze model
    print('Initialising ABM...')
    trap = io.StringIO()
    with redirect_stdout(trap):
        warnings.filterwarnings('ignore')
        model = ModelFramework_no_MESA.Model(ABM_env, configuration, 'Ph')
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

    df_metrics.insert(0, "num_agents", num_agents)

    #avg_response_time = np.mean(df_metrics['Dispatch_time'] + df_metrics['Travel_time'])
    #real_avg_response_time = np.mean(df_metrics['Real_dispatch_time'] + df_metrics['Real_travel_time'])

    return df_metrics #series_metrics 


def run_ABMs_on_one_shift(shift, historical_crimes_scenario): 
    print('-------------------------------------------')
    print("I'm process", getpid())
    print('Running experiment on shift {}'.format(shift))

    print('Importing ABM environment...')
    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]
    print('... env imported.')

    # Import environment 
    """print(os.getcwd())
    os.chdir('/nobackup/mednche/GA-Detroit/Validation/')
    print(os.getcwd())"""


    try:
        warnings.filterwarnings('ignore')

        # GET INCIDENTS THAT WILL TAKE PLACE DURING THIS SHIFT
        cfs_incidents = pd.read_csv("../Data/Incidents_new_preprocessed.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        # get all incidents within the time interval of the shift
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]
                                        
        # setup the env with the num_streets param value
        ABM_env = Env.Environment('../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_crimes_scenario = historical_crimes_scenario, travel_speed_percent=-30)
        
        warnings.filterwarnings('default')

    except:
        raise ValueError('@@@ Env import problem @@@ for shift = {} and dir = {}'.format(shift, os.getcwd()))

    
    # run ABMs for this env for all values of num_agents    
    num_agents = [10,20,30,40,50,60]
    list_of_dfs = map(partial(run_single_ABM, ABM_env = ABM_env), num_agents)

    dict_for_shift = dict(zip(num_agents,list_of_dfs))
    
    #print("Process {} finised".format(getpid()))
    return dict_for_shift

#%%
def main():
    start_time = time.time()

    historical_crimes = pd.read_csv("../Data/Crimes_edited_preprocessed.csv")


    # SCENARIO 1
    with open('./testing_set_scenario1.pkl', 'rb') as f:
        list_shifts = pickle.load(f)

    print('{} processes available'.format(cpu_count()))
    
    # GET HISTORICAL CFS AND CRIMES FOR SCENARIO 1
    #historical_cfs = pd.read_csv("../Data/Incidents_new_preprocessed.csv")
    #historical_cfs_scenario = getIncidentsForScenario(historical_cfs, 1)
    historical_crimes_scenario = getCrimesForScenario(historical_crimes, 1)


    pool = Pool(cpu_count())
    list_of_shift_dicts = pool.map(partial(run_ABMs_on_one_shift, historical_crimes_scenario=historical_crimes_scenario), list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()


    print(os.getcwd())
    with open('results_validation_travelSpeedReduced_scenario1.pkl', 'wb') as f:
        pickle.dump(list_of_shift_dicts, f)



    list_num_agents = [10,20,30,40,50,60]  
    dict_pair = {}
    for num_agents in list_num_agents:
        list_dfs_for_pair = [shift_dict[num_agents] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_pair[num_agents] = pd.concat(list_dfs_for_pair, ignore_index=True)
 
    # Save as dictionnary
    with open('dict_pair_validation_travelSpeedReduced_scenario1.pkl', 'wb') as f:
        pickle.dump(dict_pair, f)







    # SCENARIO 2
    with open('./testing_set_scenario2.pkl', 'rb') as f:
        list_shifts = pickle.load(f)


    print('{} processes available'.format(cpu_count()))
    
    #list_shifts = list_shifts[0:2] # <--- REMOVE HERE

    historical_crimes_scenario = getCrimesForScenario(historical_crimes, 2)
    

    pool = Pool(cpu_count())
    list_of_shift_dicts = pool.map(partial(run_ABMs_on_one_shift, historical_crimes_scenario=historical_crimes_scenario), list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    print(os.getcwd())
    with open('results_validation_travelSpeedReduced_scenario2.pkl', 'wb') as f:
        pickle.dump(list_of_shift_dicts, f)

    list_num_agents = [10,20,30,40,50,60]  
    dict_pair = {}
    for num_agents in list_num_agents:
        list_dfs_for_pair = [shift_dict[num_agents] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_pair[num_agents] = pd.concat(list_dfs_for_pair, ignore_index=True)
 
    
    # Save as dictionnary
    with open('dict_pair_validation_travelSpeedReduced_scenario2.pkl', 'wb') as f:
        pickle.dump(dict_pair, f)

    



    """list_num_hot_streets = [-30, -20, -10, 0, 10, 20, 30]
    dict_num_hot_streets = {}
    for num_hot_streets in list_num_hot_streets:
        list_df_for_travelSpeed = [shift_dict[num_hot_streets] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_num_hot_streets[num_hot_streets] = pd.concat(list_df_for_travelSpeed, ignore_index=True)
        
        #cat(list_dfs_for_pair, ignore_index=True)
    """
    
    # Save as dictionnary
    """with open('dict_SA_hotStreets.pkl', 'wb') as f:
        pickle.dump(dict_num_hot_streets, f)"""
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == '__main__':
    main()
#
# %%

# %%

