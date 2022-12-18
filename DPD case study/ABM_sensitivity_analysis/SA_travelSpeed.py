
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

def getIncidentsForScenario(incidents, scenario_num):
    with open('./training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        training_set_scenario = pickle.load(f)

    incidents['Patrol_beat'] = incidents['Patrol_beat'].apply(str)
    #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
    incidents.Date_Time = pd.to_datetime(incidents.Date_Time)
 
    historical_incidents_scenario = pd.DataFrame()
    for shift in training_set_scenario:
        historical_incidents_scenario = historical_incidents_scenario.append(incidents[(incidents['Date_Time'] >= shift[0]) & 
                                    (incidents['Date_Time'] < shift[1])])


    return historical_incidents_scenario



#%%
def getRandomConfig(k, ABM_env) :
   
    # Init the configuration with zero agent in each patrol beat
    list_num_agents_scas = np.zeros((len(ABM_env.patrol_beats),), dtype=int)

    # randomly select k scas within precinct
    #random.seed(222)
    chosen_scas = random.sample(ABM_env.patrol_beats, k)

    for sca in chosen_scas:
        list_index = ABM_env.patrol_beats.index(sca)
        #print("list_index", list_index)
        list_num_agents_scas[list_index] = 1

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
    configuration = getRandomConfig(num_agents, ABM_env)

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
    df_metrics, sum_deterrence = model.evaluate_model()
    #series_metrics = pd.Series(list_metrics)

    avg_response_time = np.mean(df_metrics['Dispatch_time'] + df_metrics['Travel_time'])

    ## Get the percentage of responses that were failed
    fail_threshold = 15 # <--- CHANGE HERE
    # number of failed responses
    num_failed= len(df_metrics[df_metrics['Dispatch_time'] + df_metrics['Travel_time'] > fail_threshold])
    percent_failed=(num_failed/len(df_metrics))*100

    print('>>> result: ', [avg_response_time, percent_failed, sum_deterrence])

    return [avg_response_time, percent_failed, sum_deterrence]#series_metrics 


def run_ABMs_on_one_shift(shift): 
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

    

            
    def run_ABMs_for_value(travel_speed_percent) :
        try:
            warnings.filterwarnings('ignore')

            # Get the CFS and crime data for the chosen time period

            

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
            ABM_env = Env.Environment('../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, travel_speed_percent=travel_speed_percent)
            
            # run ABMs for this env for all values of num_agents
            num_agents = [10,20,30,40,50,60]
            list_results_for_num_agents = map(partial(run_single_ABM, ABM_env = ABM_env), num_agents)
            
            df_for_num_agents = pd.DataFrame(list_results_for_num_agents, columns =['avg_response_time', 'percent_failed', 'deterrence'])
            df_for_num_agents.insert(0, "num_agents", num_agents)
            df_for_num_agents.insert(0, "travel_speed_percent", np.repeat(travel_speed_percent, len(num_agents)))

        except:
            raise ValueError('@@@ Env import problem @@@ for shift = {} and dir = {}'.format(shift, os.getcwd()))

        return df_for_num_agents

    # FOR SA:
    # run a single ABM on the shift for all values of list_num_hot_streets (2, 5 and 10)
    
    list_travel_speed_percent = [-30, -20, -10, 0, 10, 20, 30]

    list_of_dfs_for_value = map(partial(run_ABMs_for_value), list_travel_speed_percent)

    warnings.filterwarnings('default')
    
    df_for_shift = pd.concat(list_of_dfs_for_value, ignore_index=True)

    #dict_for_shift = dict(zip(list_travel_speed_percent,list_of_dfs))
    
    #print("Process {} finised".format(getpid()))
    return df_for_shift

#%%
def main():
    start_time = time.time()

    # SCENARIO 1
    with open('./testing_set_scenario1.pkl', 'rb') as f:
        list_shifts = pickle.load(f)

    print('{} processes available'.format(cpu_count()))
    
    # GET HISTORICAL CFS AND CRIMES FOR SCENARIO 1
    #historical_cfs = pd.read_csv("../Data/Incidents_new_preprocessed.csv")
    #historical_cfs_scenario = getIncidentsForScenario(historical_cfs, 1)
    #historical_crimes = pd.read_csv("../Data/Crimes_edited_preprocessed.csv")
    #historical_crimes_scenario = getIncidentsForScenario(historical_crimes, 1)
    

    pool = Pool(cpu_count())
    list_of_shift_dfs = pool.map(run_ABMs_on_one_shift, list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    df = pd.concat(list_of_shift_dfs, ignore_index=True)

    print(os.getcwd())
    with open('results_SA_travelSpeed_scenario1.pkl', 'wb') as f:
        pickle.dump(df, f)




    # SCENARIO 2
    with open('./testing_set_scenario2.pkl', 'rb') as f:
        list_shifts = pickle.load(f)


    print('{} processes available'.format(cpu_count()))
    
    #list_shifts = list_shifts[0:2] # <--- REMOVE HERE

    #list_of_shift_dicts = run_ABMs_on_one_shift(list_shifts[5])

    pool = Pool(cpu_count())
    list_of_shift_dfs = pool.map(run_ABMs_on_one_shift, list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    df = pd.concat(list_of_shift_dfs, ignore_index=True)

    print(os.getcwd())
    with open('results_SA_travelSpeed_scenario2.pkl', 'wb') as f:
        pickle.dump(df, f)



    """list_travel_speed_percent = [-30, -20, -10, 0, 10, 20, 30]
    dict_travel_speed_percent = {}
    for travel_speed_percent in list_travel_speed_percent:
        list_df_for_travelSpeed = [shift_dict[travel_speed_percent] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_travel_speed_percent[travel_speed_percent] = pd.concat(list_df_for_travelSpeed, ignore_index=True)
        
        #cat(list_dfs_for_pair, ignore_index=True)
    """
    
    # Save as dictionnary
    """with open('dict_SA_travelSpeed.pkl', 'wb') as f:
        pickle.dump(dict_travel_speed_percent, f)"""
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == '__main__':
    main()
#
# %%

# %%

