
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
        
    
def getTargettedConfig(k, ABM_env, historical_data_type = 'CFS') :
    # Init the configuration with zero agent in each patrol beat
    list_num_agents_scas = np.zeros((len(ABM_env.patrol_beats),), dtype=int)

    # CHOOSE k HOTTEST SCAS in force
    if historical_data_type == 'CFS' :
        list_num_crimes = [len(sca.historical_cfs) for sca in ABM_env.patrol_beats]
    else :
        # get the list of num of crimes 
        list_num_crimes = [len(sca.historical_crimes) for sca in ABM_env.patrol_beats]
        
        
    # create a dictionnary {beat : num_crimes} combining both lists
    dict_crime_in_beats = dict(zip(ABM_env.patrol_beats, list_num_crimes))
    # get k pairs (beat, num_crimes)
    ordered_list_beats = sorted(dict_crime_in_beats.items(), key=lambda item: item[1], reverse=True)[:k]
    # get only the beat part of that pair
    chosen_scas = [pair[0] for pair in ordered_list_beats]

    for sca in chosen_scas:
        list_index = ABM_env.patrol_beats.index(sca)
        list_num_agents_scas[list_index] = 1

    # create config dict
    pairs = zip(ABM_env.patrol_beats, list_num_agents_scas)
    # Create a dictionary from zip object
    configuration = dict(pairs)
    return configuration

def run_ABM_for_config(configuration, ABM_env) :

        ABM_STEP_TIME = 1
        print(configuration)

        ## Initialiaze model
        print('Initialising ABM...')
        trap = io.StringIO()
        with redirect_stdout(trap):
            warnings.filterwarnings('ignore')
            model = ModelFramework_no_MESA.Model(ABM_env, configuration)
            warnings.filterwarnings('default')
            
        #print('end_datetime', model.end_datetime)


        ## Run model
        print('Running ABM...')
        trap = io.StringIO()
        with redirect_stdout(trap):
            try: 
                _, _ = model.run_model(ABM_STEP_TIME)
            except:
                raise ValueError('@@@ shift = {}, num_agents = {}, strat = {}'.format(ABM_env.start_datetime, num_agents))
                
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

        avg_time_patrolling = np.mean([agent.steps_patrolling for agent in model.agents_schedule.agents])

        print('>>> result: ', [avg_response_time, percent_failed, sum_deterrence, avg_time_patrolling])

        return [avg_response_time, percent_failed, sum_deterrence, avg_time_patrolling]#series_metrics 
    


def run_ABMs_for_num_agents(num_agents, ABM_env):
    # NB: the time periods is comprised in ABM_env
    print('------')
    print('----> ',num_agents, ' agents per precinct')

 
    # Get targetted configuration 
    targetted_config = getTargettedConfig(num_agents, ABM_env)
    # Get random configuration
    random_config = getRandomConfig(num_agents, ABM_env)

    # Run 2 ABMs (one per confid)
    list_results_for_config = map(partial(run_ABM_for_config, ABM_env = ABM_env), [targetted_config, random_config])
    
    df_for_num_agents = pd.DataFrame(list_results_for_config, columns =['avg_response_time', 'percent_failed', 'deterrence', 'avg_time_patrolling'])
    df_for_num_agents.insert(0, "configuration", ["T", 'R'])
    df_for_num_agents.insert(0, "num_agents", np.repeat(num_agents, 2))

    
    return df_for_num_agents
    
    




def run_ABMs_on_one_shift(shift, historical_cfs_scenario, historical_crimes_scenario): 
    print('-------------------------------------------')
    #print("I'm process", getpid())
    print('Running experiment on shift {}'.format(shift))

    print('Importing ABM environment...')

    # choose shift
    #shift = random.choice(testing_set_scenario)
   
    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]
    

    # Import environment 
    #print(os.getcwd())
    #os.chdir('/nobackup/mednche/GA-Detroit/Benchmark_simple_configs/')
    #print(os.getcwd())

    try:
        warnings.filterwarnings('ignore')

        # GET CFS INCIDENT THAT WILL TAKE PLACE DURING THIS SHIFT
        cfs_incidents = pd.read_csv("../Data/Incidents_new_preprocessed.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        # get all incidents within the time interval of the shift
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]

                                        
        ABM_env = Env.Environment('../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario = historical_cfs_scenario, historical_crimes_scenario = historical_crimes_scenario)
        warnings.filterwarnings('default')
        print('... env imported.')
    except:
        raise ValueError('@@@ Env import problem @@@ for shift = {} and dir = {}'.format(shift, os.getcwd()))
            


    num_agents = [10,20,30,40,50,60]  
    # run a single ABM on the shift for all values of num_agents
    list_of_dfs_for_num_agents = map(partial(run_ABMs_for_num_agents, ABM_env = ABM_env), num_agents)
    
    df_for_shift = pd.concat(list_of_dfs_for_num_agents, ignore_index=True)

    """dict_for_shift = dict(zip(num_agents, list_of_num_agents_dicts))
    
    print('dict_for_shift', dict_for_shift)
    #print("Process {} finised".format(getpid()))"""
    return df_for_shift


#%%
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



def main():
    start_time = time.time()

    # SCENARIO 1
    print('SCENARIO 1')
    with open('./testing_set_scenario1.pkl', 'rb') as f:
        list_shifts = pickle.load(f)
    """ with open('validation_set_scenario12.pkl', 'rb') as f:
        list_shifts_scenario2 = pickle.load(f)
    
    # place both sets back to back (200 shifts in total)
    list_shifts = list_shifts_scenario1+ list_shifts_scenario2 """

    print('{} processes available'.format(cpu_count()))
    
    #list_shifts = list_shifts[0:2] # <--- REMOVE HERE

    #list_of_shift_dicts = run_ABMs_on_one_shift(list_shifts[5])

    # get the historical crimes/CFS to use to target deployement to hot beats


    historical_cfs = pd.read_csv("../Data/Incidents_new_preprocessed.csv")
    historical_cfs_scenario = getIncidentsForScenario(historical_cfs, 1)
    historical_crimes = pd.read_csv("../Data/Crimes_edited_preprocessed.csv")
    historical_crimes_scenario = getIncidentsForScenario(historical_crimes, 1)
    


    pool = Pool(cpu_count())
    list_of_shift_dfs= pool.map(partial(run_ABMs_on_one_shift, historical_cfs_scenario = historical_cfs_scenario, historical_crimes_scenario = historical_crimes_scenario), list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    df = pd.concat(list_of_shift_dfs, ignore_index=True)

    #print(os.getcwd())
    with open('results_benchmark_scenario1.pkl', 'wb') as f:
        pickle.dump(df, f)


    """
    list_num_agents = [2,3,4,5,6,7] 
    dict_pair = {}
    for num_agents in list_num_agents:
        list_dfs_for_pair = [shift_dict[num_agents] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_pair[num_agents] = pd.concat(list_dfs_for_pair, ignore_index=True)
 
    
    # Save as dictionnary
    with open('dict_pair_benchmark_scenario1.pkl', 'wb') as f:
        pickle.dump(dict_pair, f)
    """
        

    ###############################################
    ### SCENARIO 2
    print('SCENARIO 2')
    with open('./testing_set_scenario2.pkl', 'rb') as f:
        list_shifts = pickle.load(f)

    print('{} processes available'.format(cpu_count()))
    
    #list_shifts = list_shifts[0:2] # <--- REMOVE HERE

    #list_of_shift_dicts = run_ABMs_on_one_shift(list_shifts[5])

    historical_cfs = pd.read_csv("../Data/Incidents_new_preprocessed.csv")
    historical_cfs_scenario = getIncidentsForScenario(historical_cfs, 2)
    historical_crimes = pd.read_csv("../Data/Crimes_edited_preprocessed.csv")
    historical_crimes_scenario = getIncidentsForScenario(historical_crimes, 2)
    


    pool = Pool(cpu_count())
    list_of_shift_dfs = pool.map(partial(run_ABMs_on_one_shift, historical_cfs_scenario = historical_cfs_scenario, historical_crimes_scenario = historical_crimes_scenario), list_shifts)
    print('Closing Pool...')
    pool.close()
    print('Joining Pool...')
    pool.join()

    df = pd.concat(list_of_shift_dfs, ignore_index=True)


    #print(os.getcwd())
    with open('results_benchmark_scenario2.pkl', 'wb') as f:
        pickle.dump(df, f)

    #print(type(result_list[0]))

    #dff = pd.DataFrame(result_list)
    #print(dff)

    #incidents['Street_index'], incidents['Patrol_beat'] = [item[0] for item in result_list], [item[1] for item in result_list] 
    
    """list_num_agents = [2,3,4,5,6,7]  
    dict_pair = {}
    for num_agents in list_num_agents:
        list_dfs_for_pair = [shift_dict[num_agents] for shift_dict in list_of_shift_dicts ] # columns = incidents.columns
        dict_pair[num_agents] = pd.concat(list_dfs_for_pair, ignore_index=True)
    
    
    # Save as dictionnary
    with open('dict_pair_benchmark_scenario2.pkl', 'wb') as f:
        pickle.dump(dict_pair, f)
    """


    print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == '__main__':
    main()
#
# %%

# %%
