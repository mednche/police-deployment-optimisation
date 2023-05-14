#%%

from deap import base
from deap import creator
from deap import tools
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import os
from os import getpid
import warnings
import matplotlib.pyplot as plt
import io
import datetime as dt
import random
from contextlib import redirect_stdout
import time
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import random


import sys
sys.stdout.flush()
# This is where the ABM files are stored
sys.path.append('../../ABM/')

import ModelFramework
import Env

sys.path.append('../')
from Evaluate_GA1_strat import run_ABMs_on_one_shift as run_ABMs_on_one_shift_eval

# CHANGE VALUES BELOW
MAX_NUM_AGENTS = 60 # maximum num agents in the force
NUM_PATROL_BEATS = 131 # number of patrol beats in the force
RSS = 2 # number of randomly sampled time periods to present to the individuals at each generation for evaluation

ABM_STEP_TIME = 1 # Changing this value not recommended (code is not optimised yet for other values)

#%%
def run_single_ABM(individual, ABM_env):
    """ Function to run a single ABM for a single individual.
    Input: 
    - the individual
    - the ABM environment (which was loaded once before running every single ABM for this individual)
    Returns: 
    - df: a dataframe of dispatch and travel time for each incident.
    """
    global ABM_STEP_TIME
    
    agents_in_beats = individual # All but last element in tuple

    # Get the sum of all agents in the beat
    num_agents = sum(agents_in_beats)

    # Create configuration dict
    pairs = zip(ABM_env.patrol_beats, agents_in_beats)
    # Create a dictionary from zip object
    configuration = dict(pairs)


    ## Initialize ABM
    trap = io.StringIO()
    with redirect_stdout(trap):
        warnings.filterwarnings('ignore')
        model = ModelFramework.Model(ABM_env, configuration, 'Ph')
        warnings.filterwarnings('default')
        
    ## Run ABM 
    try: 
        trap = io.StringIO()
        with redirect_stdout(trap):
            _, _ = model.run_model(ABM_STEP_TIME)
    except:
        raise ValueError('❌ Error running the ABM for shift = {}, num_agents = {}, strat = {}'.format(ABM_env.start_datetime, agents_in_beats))
            
    ## Evaluate model  
    trap = io.StringIO()
    with redirect_stdout(trap):
        df_metrics, sum_deterrence = model.evaluate_model()
    return [df_metrics, sum_deterrence]


def getIncidentsForScenario(incidents, scenario_num):
    """ This function returns a subset of incidents (historical CFS incidents or crimes) provided that took place on all time periods of a scenario_num
    i.e. low-demand time periods or high-demand time periods
    Inputs:
    - incidents: the dataframe containing all incidents or crimes (depending on which dataset is provided)
    - scenario_num: 1 for low demand and 2 for high demand.
    Returns:
    - historical_incidents_scenario: a dataframe of CFS incidents (or crimes) that occured on time periods of the chosen scenario (Detroit: year 2017)
    """
    with open('./../../../data/historical_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        historical_set_scenario = pickle.load(f)

    incidents['Patrol_beat'] = incidents['Patrol_beat'].apply(str)
    incidents.Date_Time = pd.to_datetime(incidents.Date_Time)
 
    historical_incidents_scenario = pd.DataFrame()
    for shift in historical_set_scenario:
        historical_incidents_scenario = historical_incidents_scenario.append(incidents[(incidents['Date_Time'] >= shift[0]) & 
                                    (incidents['Date_Time'] < shift[1])])

    return historical_incidents_scenario


def run_ABMs_on_one_shift(shift, population, toolbox, historical_crimes_scenario): 
    """ This function to be run at the end of the training on each individual in the last population
    to evaluate its performance on the shift.
    Note: unlike the single objective GA, we here make use of historical crimes (historical_crimes_scenario) for the multi-objective (deterrence on patrol).
    """

    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]

    try:
        warnings.filterwarnings('ignore')

        # Get the incidents for the time period (shift)
        cfs_incidents = pd.read_csv("./../../../data/incidents.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]
      
        trap = io.StringIO()
        with redirect_stdout(trap):
            # Initialise the Env with incidents for the shift
            ABM_env = Env.Environment('./../../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario = None, historical_crimes_scenario = historical_crimes_scenario)
        warnings.filterwarnings('default')
    except:
        raise ValueError('❌ Env import problem for shift = {}'.format(shift))
           
    try:
        # Run ABM for each individual in pop on that Env (shift)
        multi_results  = toolbox.map(partial(run_single_ABM, ABM_env = ABM_env), population)
    except:
        raise ValueError('❌ Failed to run ABM on an individual of pop')
    
    
    dict_for_shift = dict(zip(population,multi_results))

    return dict_for_shift




def evalMultiObj(population, scenario_num, historical_crimes_scenario):
    global RSS, MAX_NUM_AGENTS
    """ This function evaluates the fitness of all individuals on a set of chosen time periods for this generation.
    It calls run_ABMs_on_one_shift() for each shift in the training_set for scenario_num (100 shifts per scenario).
    Inputs: 
    - population: list of individuals (DEAP object)
    - scenario_num: 1 is low_demand, 2 is high demand
    - historical_crimes_scenario: a dataset of historical crimes used in the initialisation of the ABM environment (Env). This is specific to the multi-objective GA
    Returns:
    - list_fitnesses: a list of tupples representing the fitness values, one for each individual (e.g. [(3.5, 30, 5, 536), (2.5, 55, 0.2, 651) ...])
    The fitness of an indivual represents the tuple (avg_response_time, total_num_agents, percent_failed, total_sum_deterrence) 
    calculated over the 100 shifts of the training set, combining all the df_metrics for those shifts.
    """
    
    with open('./../../../data/training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)
    
    # WE ONLY EVALUATE ON RSS NUMBER OF RANDOMLY CHOSEN SHIFT(S) AT EACH GENERATION (e.g. RSS=2)
    indices = np.random.choice(len(list_shifts), RSS, replace=False)
    list_shifts = np.array(list_shifts)[indices]
    
    # Convert to tuple for dict key later
    population = [tuple(ind) for ind in population]
    print('Num_ind_to_evaluate', len(population))

    pop_unique_ind = set(population)
    print('Num unique ind,', len(pop_unique_ind))

    list_dict_for_shift = [run_ABMs_on_one_shift(shift,pop_unique_ind,toolbox, historical_crimes_scenario) for shift in list_shifts]
    print('-------->>>> Finished evaluating {} ABMs for each ind in population of {}<<<<<-------- '.format(len(list_shifts), len(pop_unique_ind)))
    list_fitnesses = []

    # For each ind in population
    # The dictionnary is essential to not have to evaluate duplicate individuals
    for ind in population:
        # list of sum deterrence (one value per shift for that individual) [1, 20, 30.7]
        list_sum_deterrence = [dict_for_shift[ind][1] for dict_for_shift in list_dict_for_shift]
        # list of dfs (one value per shift for that individual) [df, df, df]
        list_dfs = [dict_for_shift[ind][0] for dict_for_shift in list_dict_for_shift]
        # Concatenate all df_metrics across shifts for that ind
        df_ind= pd.concat(list_dfs)
        
        # Get the total number of agents
        total_num_agents = sum(ind)

        # Penalty for individuals outside the range of num of agents
        if total_num_agents > MAX_NUM_AGENTS or total_num_agents == 0 :
            avg_response_time = 1000
            percent_failed = 100
            total_sum_deterrence = 0
        
        else :
            # Get the average response time 
            avg_response_time = np.mean(df_ind['Dispatch_time'] + df_ind['Travel_time'])
            
            ## Get the percentage of responses that were failed
            fail_threshold = 15 # <--- CHANGE HERE
            # number of failed responses
            num_failed= len(df_ind[df_ind['Dispatch_time'] + df_ind['Travel_time'] > fail_threshold])
            percent_failed=(num_failed/len(df_ind))*100
            
            # get the total sum deterrence for that individual across all shifts
            total_sum_deterrence = sum(list_sum_deterrence)
        
        # Save value in a list of size: len(pop)
        list_fitnesses.append((avg_response_time, total_num_agents, percent_failed, total_sum_deterrence))


    return list_fitnesses


# Variation algortihm
def modifiedVarAnd(population, toolbox, lambda_, cxpb, mutpb):
    """This function applies a variation algorithm to individuals in the population.
    Inputs:
    - population: the parent/breeding pool (typically a subset of population mu)
    - toolbox: DEAP toolbox pre-initialised
    - lambda_: number of offspring to generate in the process
    - cxpb: crossover probability
    - mutpb: mutation probability
    Returns: 
    - offsrping: a list of individuals created through the variation process

    Note: SAME AS SINGLE-OBJECTIVE GA
    """
    
    offspring = []

    # create the necessary number of offspring
    for i in range(lambda_):

        # Apply crossover
        op1_choice = random.random()
        if op1_choice < cxpb:        
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            #print('Crossover between {} and {}'.format(ind1, ind2))
            ind, _ = toolbox.mate(ind1, ind2)
            del ind.fitness.values
        # No crossover
        else: 
            ind = random.choice(population)

        # Apply mutation
        op2_choice = random.random()
        if op2_choice < mutpb:  # Apply mutation
            ind = toolbox.clone(ind)
            print('Mutating ind')
            ind, = toolbox.mutate(ind)
            del ind.fitness.values

        offspring.append(ind)

    return offspring


# Evolution algorithm
def modifiedEaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, indpb, ngen, last_gen, logbook,
                    stats, scenario_num, halloffame=None, verbose=__debug__):
    """ This function applies the main evolutionary algorithm on the population.
        Inputs:
        - population: listof individuals (DEAP object)
        - toolbox: DEAP toolbox pre-initialised
        - mu: size of the population (number of individuals)
        - lambda_: number of offspring to generate in the process
        - cxpb: crossover probability
        - mutpb: mutation probability
        - ngen: number of generations to run the algorithm
        - last_gen: last population saved from previous runs of this GA (to continue learning from there)
        - logbook: DEAP logbook object to save key statistics of the learning along the way
        Returns:
        - population: the population at the end of the learning
        - logbook: the logbook at the end of the learning
    
    Notes: I decided to:
    - Use a breeding pool selection (parent selection) on top of the survivor selection.
    This is to accelerate the convergence of population towards optimum. Otherwise too slow
    - Not let the population age and stay in the next gen. All replaced by new offspring 
    (some of which are clones of parents since no crossover or mutation occured) """

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Chose 12 so that tournamentDCD receives a population of size 12 + 40 =52 which is a multiple of 4 
    breeding_pool_size = 12 # <- CHANGE HERE TO TUNE GA, APPLY MORE PRESSURE (typically int(mu/2) )
    
    #=====================================================
    # Evaluate the individuals with an 'invalid' fitness 
    # (i.e. calculate a fitness for individuals that don't yet have one because they have undergone variation/are new)
    # Note: at gen 0 this basically means evaluating all ind in pop
    #=====================================================
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # Only if there are invalid ind (new ones)
    if invalid_ind :
        fitnesses = toolbox.evaluate(invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            print(sum(ind), fit)
            ind.fitness.values = fit
    #=====================================================


    #=====================================================
    #       Assign crowding distance to individuals
    #=====================================================
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    population = toolbox.selectSurvivors(population, mu) # NSGA-II
    #=====================================================

    # Log HOF
    if halloffame is not None:
        halloffame.update(population)

    # Write first record in logbook
    population_without_na = [ind for ind in population if not np.isnan(ind.fitness.values[0])]
    print('population: ', len(population), 'population_without_na: ', len(population_without_na))
    record = stats.compile(population_without_na) if stats is not None else {}
    logbook.record(gen=last_gen, nevals=len(population), hof = [hofer for hofer in halloffame], time = dt.datetime.now(), **record)
    if verbose:
        print(logbook.stream)


    # Save the population (list of individuals) in a pickle file so we can continue from there.
    with open('population_gen_{}_scenario{}'.format(last_gen, scenario_num), 'wb') as f:
        pickle.dump(population, f)

    #====================================================
    #====================================================
    ###         Begin the generational process        ###
    #====================================================
    #====================================================
    for gen in range(last_gen+1, last_gen + ngen+1):
        sys.stdout.flush()
        print('------------------------')
        print('--------------')
        print('GENERATION:', gen)
        print('--------------')
        original_stdout = sys.stdout
        with open('logbook', 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # Append new line at the end of the logbook
            print("{}\n".format(logbook.stream), file=f)
            sys.stdout = original_stdout # Reset the standard output to its original value

        
        #=====================================
        #            Parent selection        #
        #=====================================
        # Select breeding_pool_size parents from pop of size mu
        # This is so that there are fewer individuals in the parent pools
        # In single obj, we use tournament which uses replacement so this attempts to apply equivalent pressure
        print('Selecting best parents for breeding...')
        parent_pool =  toolbox.selectParents(population,breeding_pool_size)
        print('...', len(parent_pool))

        #=====================================
        #          Vary the population     #
        #=====================================
        offspring = modifiedVarAnd(parent_pool, toolbox, lambda_, cxpb, indpb)
        print('Offspring created:', len(offspring))

        #===================================================
        #           Evaluate offspring   
        # Evaluate the individuals with an invalid fitness 
        # (i.e. those that we have never seen before)
        #===================================================
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        try:
            fitnesses = toolbox.evaluate(invalid_ind)
        except:
            raise ValueError("❌ Error evaluating fitnesses of invalid_ind : {}".format(invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            print(sum(ind), fit)
            ind.fitness.values = fit
        #=====================================

        
        ######   Update HOF with the generated individuals #####
        if halloffame is not None:
            halloffame.update(offspring)

        print('Current HOF size:', len([hofer for hofer in halloffame]))
        
       

        #============================================================================================
        #          Survivor selection        #
        # Select mu individuals for next gen pop 
        # from parents + offspring pops (mu + lambda) selection 
        # (alternatives selection: elitism, round-robin tournament, (mu + lambda) selection, (mu, lambda))
        #============================================================================================
        
        print('Selecting best inds for next gen:')
        population[:] = toolbox.selectSurvivors(population + offspring, mu)

        #============================================================================================
        
        # Update the statistics with the new population
        population_without_na = [ind for ind in population if not np.isnan(ind.fitness.values[0])]
        print('population: ', len(population), 'population_without_na: ', len(population_without_na))
        record = stats.compile(population_without_na) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), hof = [hofer for hofer in halloffame], time = dt.datetime.now(), **record)
        if verbose: 
            print(logbook.stream)

        # Save the population in a pickle file so we can continue from there.
        with open('population_gen_{}_scenario{}'.format(gen,scenario_num), 'wb') as f:
            pickle.dump(population, f)

        with open('logbook_file_scenario{}'.format(scenario_num), 'wb') as f:
            pickle.dump(logbook, f)

        

    return halloffame, population, logbook


#####################################################################


def create_toolbox():
    """ This function initialise the DEAP toolbox.
    """
    global MAX_NUM_AGENTS, NUM_PATROL_BEATS
    
    scenario_num = sys.argv[1]

    # GET HISTORICAL CRIMES FOR SCENARIO
    historical_crimes = pd.read_csv("../../../data/crimes.csv")
    historical_crimes_scenario = getIncidentsForScenario(historical_crimes, scenario_num)


    def gen_random_parameters():
        """ This function initialise an individual by randomly creating an array of NUM_PATROL_BEATS 
        with MAX_NUM_AGENTS random values as 1 and the rest as 0. 
        Note: same as single-objective GA
        """
        global MAX_NUM_AGENTS, NUM_PATROL_BEATS

        list_num_agents_scas = np.zeros((NUM_PATROL_BEATS,), dtype=int)

        # get a random number of beats to staff
        k = random.randint(1, MAX_NUM_AGENTS)
        #print('staffing {} beats'.format(k))
        # randomly choose k indexed of beats to staff without replacement
        staffed_indexes=random.sample(range(NUM_PATROL_BEATS), k)
        # replace
        for index in staffed_indexes:
            list_num_agents_scas[index] = 1

        return tuple(list_num_agents_scas)

    # Define fitness with weights and direction of objectives (negative weight to minimise)
    creator.create("Fitness", base.Fitness, weights=(-1.0,-1.0,-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    ## TOOLBOX
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attribute_generator", gen_random_parameters)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                        toolbox.attribute_generator) 

    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
    
    # Tool initializers
    toolbox.register("evaluate", evalMultiObj, scenario_num=scenario_num, historical_crimes_scenario=historical_crimes_scenario) # CHANGE HERE FOR MULTI
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1) # independent probability of each attribute to be mutated 
    toolbox.register("selectSurvivors", tools.selNSGA2)
    toolbox.register("selectParents", tools.selTournamentDCD)


    return toolbox


def main():

    # get the scenario number for which to run the GA.
    scenario_num = sys.argv[1]
    print('Scenario_num', scenario_num)
    
    # CHANGE KEY GA VALUES HERE
    NGEN = 60
    mu = 40 #must be a multiple of 4 for selTournamentDCD

    # Process Pool of 4 workers
    from multiprocessing import Pool

    pool = Pool(processes=min(cpu_count(), mu)) # max of 40 on University of Leeds HPC 
    toolbox.register("map", pool.map)
    

    #===================================================
    #        Import or randomly create population       #
    #===================================================
    
    last_gen = 0
    import os.path
    for gen in range (20, 100):
        if os.path.isfile('population_gen_{}_scenario{}'.format(gen, scenario_num)):
            last_gen = gen
            

    # If there is a previous population to carry on from, open the file
    if last_gen != 0:
        with open('population_gen_{}_scenario{}'.format(last_gen, scenario_num), 'rb') as f:
            print('Loading previous pop for gen:', last_gen)
            pop = pickle.load(f)
            
            
        # Get logbook for hof
        
        with open('logbook_file_scenario{}'.format(scenario_num), 'rb') as f:
            logbook = pickle.load(f)
            hof = logbook[-1]['hof']
            print('Loading previous hof: ')
        
            
        # But if the number of individuals is different, create a new pop randomly
        if len(pop) != mu :
            print('Creating new pop, because mu was different')
            pop = toolbox.population(n=mu)
            hof = tools.ParetoFront() # List of all non-dominated solutions

    # otherwise create a new one randomly
    else:
        print('Creating new pop, because no previous pop was available')
        pop = toolbox.population(n=mu)
        hof = tools.ParetoFront() # List of all non-dominated solutions

        # Initialise the logbook
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "respTime", "deterrence", "num_vehicles"
        logbook.chapters["respTime"].header = "min", "avg", "median", "max"
        logbook.chapters["deterrence"].header = "min", "avg", "median", "max"
        logbook.chapters["num_vehicles"].header = "min", "avg", "max"


    #======================================================
    

    l = len(pop[0])
    CXPB = 0.9 # [0.6-0.8] or [0.7-0.9]
    print('Pc = ', CXPB)
    
    MUTPB = 1/mu # [1/l, 1/mu]
    print('MUTPB = ', MUTPB)
    lambda_ = mu
    print('lambda_: ',lambda_)


    # Initialise stats for logbook
    #In order of ind fitness: avg_response_time, total_num_agents, percent_failed, total_sum_deterrence
    stats_respTime = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_deterrence= tools.Statistics(key=lambda ind: ind.fitness.values[3])
    stats_size = tools.Statistics(key=lambda ind: sum(ind))
    mstats = tools.MultiStatistics(respTime=stats_respTime, deterrence=stats_deterrence, num_vehicles=stats_size)

    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("avg", np.mean)
    mstats.register("median", np.median)
    mstats.register("std", np.std)

    #==============================================
    #                TRAIN THE GA
    #==============================================
    start_time = time.time()
    
    hof, population, logbook2 = modifiedEaMuCommaLambda(pop, toolbox, mu, lambda_, CXPB, MUTPB, NGEN, last_gen, logbook, mstats, scenario_num, hof, verbose = True)
    
    # Note: the population is saved after each generation (see above)
    
    end_time = time.time()

    with open('logbook_file_scenario{}'.format(scenario_num), 'ab') as f:
        pickle.dump(logbook2, f)

    print("Running time step 1: {} mins".format((end_time-start_time)/60))
    #================================================

    
        
# %%

toolbox = create_toolbox()

if __name__ == "__main__":
    main()
