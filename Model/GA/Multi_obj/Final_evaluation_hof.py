"""This script needs to be run on the HPC and takes almost 15.4 hours to run the 84 unique strategies
It is to be run after the GA_multiObj_multiprocessing.py script, once the GA has searched the space and converged.
And it is to be in the same folder.
Its purpose is to:
1- combine together the final hof and final pop and remove duplicates
2- evaluate all these individuals on the testing_set (100 shifts) so they can be compared like for like.
3- Select the non-dominated solutions (= the new real hof)

After that, I use the jupyter notebook called 'Visualise optima multi obj' to plot the Pareto frontier."""


import pickle
import sys
#sys.path.append('./Model/GA')
sys.path.append('../../Model/Framework/')

import traceback 
import os

from functools import partial

import os.path

import ModelFramework_no_MESA
import AgentFramework2
import Env

import numpy as np
import pandas as pd

import warnings

from datetime import datetime, timedelta
import datetime

from multiprocessing import Pool, cpu_count
import datetime as dt
#import importlib

import random 

import io
from contextlib import redirect_stdout

MAX_NUM_AGENTS = 60

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
        df_metrics, sum_deterrence = model.evaluate_model()
    #series_metrics = pd.Series(list_metrics)
    return [df_metrics, sum_deterrence]#series_metrics 





def run_ABMs_on_one_shift(shift, population, historical_crimes_scenario): 
    """ Function to be run on each hof strategy identified during training
    to evaluate its performance on the shift (step 2 of the GA)"""
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

        

        """historical_crimes = pd.read_csv("../../Data/Crimes_edited_preprocessed.csv")
        historical_crimes['Patrol_beat'] = historical_crimes['Patrol_beat'].apply(str)
        #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
        historical_crimes.Date_Time = pd.to_datetime(historical_crimes.Date_Time)
        # get the incidents one year prior to start of time period (or anything desired: could be 2 years)
        historical_crimes_year = historical_crimes[(historical_crimes['Date_Time'] >= ABM_START_DATETIME- dt.timedelta(days = 365)) & 
                                        (historical_crimes['Date_Time'] < ABM_START_DATETIME)]"""

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
            ABM_env = Env.Environment('./../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = historical_crimes_scenario)
        #ABM_env = Env.Environment('./../../', ABM_START_DATETIME, ABM_END_DATETIME)
        warnings.filterwarnings('default')
    except:
        raise ValueError('@@@ Env import problem @@@ for shift = {}'.format(shift))
           
    try:
        with Pool(processes=cpu_count()) as pool:
            # multiprocessing: run an ABM for this shift for each ind in population
            list_of_results = pool.map(partial(run_single_ABM, ABM_env = ABM_env), population) 

        #df_metrics = run_single_ABM( individual, ABM_env = ABM_env)
    except:
        raise ValueError('@@@ failed to run ABM on an individual of pop')
    
    dict_for_shift = dict(zip(population,list_of_results))
    #print('dict_for_shift:{}'.format(len(dict_for_shift)))
    #print(dict_for_shift.keys())
   
    #print("Process {} finised".format(getpid()))
    return dict_for_shift


def getIncidentsForScenario(incidents, scenario_num):
    with open('../training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        training_set_scenario = pickle.load(f)

    incidents['Patrol_beat'] = incidents['Patrol_beat'].apply(str)
    #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
    incidents.Date_Time = pd.to_datetime(incidents.Date_Time)
 
    historical_incidents_scenario = pd.DataFrame()
    for shift in training_set_scenario:
        historical_incidents_scenario = historical_incidents_scenario.append(incidents[(incidents['Date_Time'] >= shift[0]) & 
                                    (incidents['Date_Time'] < shift[1])])


    return historical_incidents_scenario


def evaluateHOFScenario(scenario_num) :

    global MAX_NUM_AGENTS
    
    ##################################################################
    ### Import population and hof
    ##################################################################
    # Find for which gen we last saved a pop
    last_gen = 0
    for gen in range (5, 100):
        if os.path.isfile('population_gen_{}_scenario{}'.format(gen, scenario_num)):
            last_gen = gen
                

    # Loading the last pop
    print('Loading last pop for gen:', last_gen)
    with open('population_gen_{}_scenario{}'.format(last_gen, scenario_num), 'rb') as f:
        last_pop = pickle.load(f)

    # Loading last hof   
    with open('logbook_file_scenario{}'.format(scenario_num), 'rb') as f:
        logbook = pickle.load(f)
        hof = logbook[-1]['hof']

    # Print current fitnesses of last hof
    print('Loaded previous hof: ')
    for ind in hof:
        print(sum(ind), ind.fitness)


    # Combine all individuals from last pop and hof
    pop = last_pop + [hofer for hofer in hof] #
    #pop = [hofer for hofer in hof][0:5]
    print('Num ind to evaluate in HOF', len(pop))



    pop_list = [tuple(ind) for ind in pop]

    # Save a dict to return to object from lists later
    dict_list_to_object = dict(zip(pop_list, pop))

    # Select all unique individuals in pop to evaluate that have a num_agents is within range 0-60
    pop_unique_list = list(set([ind for ind in pop_list if sum(ind) <= MAX_NUM_AGENTS and sum(ind) > 0 ]))
    print('Num unique ind,', len(pop_unique_list))
    #print(pop_unique_list)


    ##################################################################
    ### EVALUATE best hof strats on testing set (100 shifts)
    ##################################################################

    # Evaluate all individuals in the hof on 100 shifts of testing set
    with open('./../testing_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)
        #list_shifts  = list_shifts[0:2]# <--- REMOVE HERE

    print('Evaluating individuals')

    # GET HISTORICAL CRIMES FOR SCENARIO
    #historical_cfs = pd.read_csv("../../Data/Incidents_new_preprocessed.csv")
    #historical_cfs_scenario = getIncidentsForScenario(historical_cfs, 1)
    historical_crimes = pd.read_csv("../../Data/Crimes_edited_preprocessed.csv")
    historical_crimes_scenario = getIncidentsForScenario(historical_crimes, scenario_num)

    #results= [pool.map(partial(run_ABMs_on_one_shift_eval, individual = ind), list_shifts) for ind in pop_unique_list]
    # evaluate all ind in pop (multiprocessing as could be > 230 ind) for each of the 100 shifts
    results = [run_ABMs_on_one_shift(shift, pop_unique_list, historical_crimes_scenario) for shift in list_shifts]


    #list_of_list_shift_dfs = [map(partial(run_ABMs_on_one_shift, individual = ind), list_shifts) for ind in pop_unique_list]
    #print('results', [dict_for_shift for dict_for_shift in results])


    # Revert to object Individuals to be able to update their fitness values
    pop_unique_ind = [dict_list_to_object[item] for item in pop_unique_list]
    print('pop_unique_ind', len(pop_unique_ind))
    
    print('Extracting fitnesses')
    # Calculate fitnesses of each hofer
    #for ind, results in zip(pop_unique_ind, results): 
    for ind in pop_unique_ind:
        print('----------------------------------------------------')
        #print('>>>>>> ind:', ind)
        #print([dict_for_shift[tuple(ind)] for dict_for_shift in results])
        # list of sum deterrence (one value per shift for that individual) [1,20, 30.7]
        list_sum_deterrence = [dict_for_shift[tuple(ind)][1] for dict_for_shift in results]
        # get the total sum deterrence for that individual across all shifts
        total_sum_deterrence = sum(list_sum_deterrence)
        print('total_sum_deterrence', total_sum_deterrence)

        # list of dfs (one value per shift for that individual) [df, df, df]
        list_dfs = [dict_for_shift[tuple(ind)][0] for dict_for_shift in results]
        #print('list_dfs', list_dfs[0:2])
        # Concatenate all df_metrics across shifts for that ind
        df_ind= pd.concat(list_dfs)

        ## Get the percentage of responses that were failed
        fail_threshold = 15 # <--- CHANGE HERE
        # number of failed responses
        num_failed= len(df_ind[df_ind['Dispatch_time'] + df_ind['Travel_time'] > fail_threshold])
        percent_failed=(num_failed/len(df_ind))*100

        # Get the average response time 
        avg_response_time = np.mean(df_ind['Dispatch_time'] + df_ind['Travel_time'])
        # Get the total number of agents
        total_num_agents = sum(ind[:-1])
        
        

        # Save value in a list of size: len(pop)
        #list_fitnesses.append((avg_response_time, total_num_agents))
        ind.fitness.values = (avg_response_time, total_num_agents, percent_failed, total_sum_deterrence)
        print(len(ind), ind.fitness.values)

        

       
    # Update hof by selecting only nondominated solutions
    #new_hof = tools.sortNondominated(pop_unique_ind, k=len(pop_unique_ind), first_front_only=True)[0]
    #print('New non dominated front: ',new_hof)

    print('Saving evaluated HOF')
    with open('final_HOF_scenario{}'.format(scenario_num), 'wb') as f:
        pickle.dump(pop_unique_ind, f)

    # This is because we need to drill down to individual incidents when making an average of average or an average percentage
    """print('Saving dict of values too')
    with open('final_HOF_dict_scenario{}'.format(scenario_num), 'wb') as f:
        pickle.dump(dict(zip(pop_unique_list, results)), f)"""

    
    
def main():
    
    
    #from deap import tools
    import time
    from deap import base
    from deap import creator
    

    creator.create("Fitness", base.Fitness, weights=(-1.0,-1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    start_time = time.time()

    print('Evaluating HOF SCENARIO 1...')
    evaluateHOFScenario(scenario_num= 1)
    print('Evaluating HOF SCENARIO 2...')
    evaluateHOFScenario(scenario_num= 2)
    
    end_time = time.time()

    print("Running time step 2: {} mins".format((end_time-start_time)/60))

if __name__ == "__main__":
    main()