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

import sys
sys.stdout.flush()
# This is where the ABM files are stored
sys.path.append('../../ABM/')

import ModelFramework
import Env

sys.path.append('../')
sys.stdout.flush()

# CHANGE VALUES BELOW
MAX_NUM_AGENTS = 60 # maximum num agents in the force
NUM_PATROL_BEATS = 131 # number of patrol beats in the force
RSS = 1 # number of randomly sampled time periods to present to the individuals at each generation for evaluation

ABM_STEP_TIME = 1 # Changing this value not recommended (code is not optimised yet for other values)

#%%

def run_ABMs_on_one_shift_eval(shift, individual, patrol_beats_df = None): 
    """ Function to be run at the end of the training on each individual in the last population
    to evaluate its performance on the shift.
    Note: no historical crimes or incidents given to the environment upon initialisation of the ABM in single objective.
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
            ABM_env = Env.Environment('./../../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario = None, historical_crimes_scenario = None, patrol_beats_df=patrol_beats_df)
        warnings.filterwarnings('default')
    
    except:
        raise ValueError('❌ Env import problem for shift = {}'.format(shift))
           
    try:
        # Run ABM for that individual on that Env (shift)
        df_metrics = run_single_ABM(individual, ABM_env = ABM_env)
    except:
        raise ValueError('❌ Failed to run ABM on an individual of pop')
    
    return df_metrics




def run_single_ABM(individual, ABM_env):
    """ Function to run a single ABM for a single individual.
    Input: 
    - the individual
    - the ABM environment (which was loaded once before running every single ABM for this individual)
    Returns: 
    - df: a dataframe of dispatch and travel time for each incident.
    """    
    global ABM_STEP_TIME


    agents_in_beats = individual

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
            
             
    ## Evaluate ABM  
    trap = io.StringIO()
    with redirect_stdout(trap):
        df_metrics, _ = model.evaluate_model()
   
    return df_metrics



def run_ABMs_on_one_shift(shift, population, toolbox): 
    """ This function runs an ABM for the provided shift for all individuals in the population.
    Note: no historical crimes or incidents given to the environement upon initialisation of the ABM in single objective.
   
    Inputs:
    - shift: the time period range for the shift (a list of 2 datetimes)
    - population: the list of individuals in the population (DEAP object)
    - toolbox: DEAP toolbox already initialised 
    Returns:
    - dict_for_shift: form 
        {
            ind1 : [df_metrics_shift1, df_metrics_shift2, ..., df_metrics_shift100],
            ind2 : [df_metrics_shift1, df_metrics_shift2, ..., df_metrics_shift100]
            .
            .
            .
            indn : [df_metrics_shift1, df_metrics_shift2, ..., df_metrics_shift100]
        }
        where df_metrics_shift is the dataframe of dispatch time and travel time for each incident produced when 
        evaluating the ABM.
    """

    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]

   
    try:
        warnings.filterwarnings('ignore')
        
        # Get all incidents within the time period (shift)
        cfs_incidents = pd.read_csv("./../../../data/incidents.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]


        # Initialise the Env with incidents for that shift. DO this once for all the individuals saves time.                     
        ABM_env = Env.Environment('./../../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario = None, historical_crimes_scenario = None)
 
        warnings.filterwarnings('default')
    except:
        raise ValueError('❌ Error importing Env for shift = {}'.format(shift))
           
    try:
        # Run an ABM for each individual in the population for that shift using the same initialised Env
        list_of_dfs = toolbox.map(partial(run_single_ABM, ABM_env = ABM_env), population)
    except:
        raise ValueError('❌ Error running ABM on an individual of pop')
    
    
    dict_for_shift = dict(zip(population,list_of_dfs))
  
    return dict_for_shift




def evalSingleObj(population, scenario_num):
    global MAX_NUM_AGENTS, RSS
    """ This function evaluates the fitness of all individuals on a set of chosen time periods for this generation.
    It calls run_ABM_on_shift() for each shift in the training_set for scenario_num (100 shifts per scenario).
    Inputs: 
    - population: list of individuals (DEAP object)
    - scenario_num: 1 is low_demand, 2 is high demand
    Returns:
    - list_fitnesses: a list of fitness values, one for each individual (e.g. [2.3, 5.25, 8.10, ...])
    The fitness of an indivual represents  the average response time over the 100 shifts of the training set, combining all the df_metrics for those shifts.
    """
    
    with open('./../../../data/training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)
    
    # WE ONLY EVALUATE ON RSS NUMBER OF RANDOMLY CHOSEN SHIFT(S) AT EACH GENERATION (e.g. RSS=1)
    indices = np.random.choice(len(list_shifts), RSS, replace=False)
    list_shifts = np.array(list_shifts)[indices]
   
    # Convert to tuple for dict key later
    population = [tuple(ind) for ind in population]

    pop_unique_ind = set(population)
    print('Unique individuals in pop: {}'.format(len(pop_unique_ind)))

    # Only select the feasible individual to run the ABM on
    pop_feasible_unique_ind = [ind for ind in pop_unique_ind if sum(ind) <= MAX_NUM_AGENTS ]
    print('Number of unique feasible ind,', len(pop_feasible_unique_ind), ' of ', len(population))

    # Only run ABM on pop_feasible_unique_ind as it takes CPU time!
    list_dict_for_shift = [run_ABMs_on_one_shift(shift,pop_feasible_unique_ind,toolbox) for shift in list_shifts]
    print('-------->>>> Finished evaluating {} ABMs for each ind in population of {} inds<<<<<-------- '.format(len(list_shifts), len(pop_unique_ind)))
    
    list_fitnesses = []
    # The dictionnary is key to avoid having to re-evaluate duplicate/clone individuals
    for ind in population:
        # PENALTY HANDLING: assign bad fitness value! (1000)
        if sum(ind) > MAX_NUM_AGENTS :
            avg_response_time = 1000

        else :
            list_dfs = [dict_for_shift[ind] for dict_for_shift in list_dict_for_shift]
            # Concatenate all df_metrics across shifts for that ind
            df_ind= pd.concat(list_dfs)
            # Get the average response time 
            avg_response_time = np.mean(df_ind['Dispatch_time'] + df_ind['Travel_time'])
        
        # Save fitness value in a list of fitnesses (size = len(pop))
        list_fitnesses.append((avg_response_time,))

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
        # No crossover, select an individual at random in the population
        else: 
            ind = random.choice(population)

        # Apply mutation
        op2_choice = random.random()
        if op2_choice < mutpb: 
            ind = toolbox.clone(ind)
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
        offspring.append(ind)

    return offspring


# Evolution algorithm
def modifiedEaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, last_gen, logbook,
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
    """

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an 'invalid' fitness 
    # (i.e. calculate a fitness for individuals that don't yet have one because they have undergone variation/are new)
    # Note: at gen 0 this basically means evaluating all ind in pop
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    try:
        fitnesses = toolbox.evaluate(invalid_ind)
    except:
        raise ValueError("❌ Error evaluating fitnesses for invalid_ind : {}".format(invalid_ind))
    
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Write first record in logbook if this is gen 0 (but not if picking up from existing last_gen)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=last_gen, nevals=len(population), time = dt.datetime.now(), **record)
    if verbose:
        print(logbook.stream)


    # Save the population (list of individuals) in a pickle file so we can continue from there.
    with open('population_gen_{}_scenario{}'.format(last_gen, scenario_num), 'wb') as f:
        pickle.dump(population, f)


    # Begin the generational process
    for gen in range(last_gen+1, last_gen + ngen + 1):
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
        
    
        # Parent selection BEFORE BREEDING
        # selecting mu parents from pop of size mu (with replacement).
        # NB: selectParents uses DEAP selTournament agorithm which relies on selRandom which has replacement
        parent_pool = toolbox.selectParents(population, mu)

        # Turn individuals in parent_pool into a list instead of a DEAP object to be printed
        #parent_pool_list = [tuple(ind) for ind in parent_pool]
        #print('Unique parents: {}'.format(len(set(parent_pool_list))))

        # Vary the population
        offspring = modifiedVarAnd(parent_pool, toolbox, lambda_, cxpb, mutpb)
        print('Offspring after variations:', len(offspring))

        # Evaluate the individuals with an 'invalid' fitness (those that we have never seen before)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print('New ind to evaluate', len(invalid_ind))
        try:
            fitnesses = toolbox.evaluate(invalid_ind)
        except:
            raise ValueError("❌ Error evaluating fitnesses of invalid_ind : {}".format(invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
          
        # Survivor selection
        # Select mu individuals for next gen pop from all lambda offspring
        # Simple replacement instead of selection using age component only, not fitness based 
        # (alternatives: elitism, round-robin tournament, (mu + lambda) selection, (mu, lambda) selection)
        # toolbox.selectSurvivers(offspring, mu)
        print('Selecting ALL offspring for next gen (no selection):')
        population[:] = offspring 
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), time = dt.datetime.now(), **record)
        if verbose: 
            print(logbook.stream)
        print('------------------------')

        # Save the population (list of individuals) in a pickle file so we can continue from there at a later run.
        with open('population_gen_{}_scenario{}'.format(gen, scenario_num), 'wb') as f:
            pickle.dump(population, f)

        with open('logbook_file_scenario{}'.format(scenario_num), 'wb') as f:
            pickle.dump(logbook, f)

    return population, logbook

#####################################################################


def create_toolbox():
    """ This function initialise the DEAP toolbox.
    """
    global MAX_NUM_AGENTS, NUM_PATROL_BEATS

    scenario_num = sys.argv[1]

    # Attribute generator (for random attribute values in individuals) 
    def gen_random_parameters():
        """ This function initialise an individual by randomly creating an array of NUM_PATROL_BEATS 
        with MAX_NUM_AGENTS random values as 1 and the rest as 0. 
        """
        global MAX_NUM_AGENTS, NUM_PATROL_BEATS
        list_num_agents_scas = np.zeros((NUM_PATROL_BEATS,), dtype=int)

        # get a random number of beats to staff
        k = random.randint(1, MAX_NUM_AGENTS)
        # randomly choose k indexed of beats to staff without replacement
        staffed_indexes=random.sample(range(NUM_PATROL_BEATS), k)
        # replace values at indexes with 1s
        for index in staffed_indexes:
            list_num_agents_scas[index] = 1

        return tuple(list_num_agents_scas)


    # Define fitness with weights and direction of objective (negative weight to minimise)
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    ## TOOLBOX
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attribute_generator", gen_random_parameters)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute_generator) 
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
    
    # Tool initializers
    toolbox.register("evaluate", evalSingleObj, scenario_num=scenario_num)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.1) # independent probability of each attribute to be mutated
    toolbox.register("selectParents", tools.selTournament, tournsize = 3)
    toolbox.register("selectSurvivers", tools.selBest)

    return toolbox


#%%
def main():
    # get the scenario number for which to run the GA.
    scenario_num = sys.argv[1]
    print('Scenario_num', scenario_num)

    # CHANGE KEY GA VALUES HERE
    NGEN = 40 # Number of generations
    mu = 40 # size of the population

    # Process Pool of 4 workers
    from multiprocessing import Pool

    pool = Pool(processes=min(cpu_count(), mu)) # max of 40 on University of Leeds HPC 
    toolbox.register("map", pool.map) 
    
    #======================================================
    #        Import or randomly create population          #
    #======================================================

    last_gen = 0
    import os.path
    for gen in range (5, 100):
        if os.path.isfile('population_gen_{}_scenario{}'.format(gen, scenario_num)):
            last_gen = gen

    # If there is a previous population to carry on from, open the file
    if last_gen != 0 :
        with open('population_gen_{}_scenario{}'.format(last_gen, scenario_num), 'rb') as f:
            print('Loading previous pop for gen:', last_gen)
            pop = pickle.load(f)
            # But if the number of individuals is different, create a new pop randomly
            if len(pop) != mu :
                print('Creating new pop, because mu was different')
                pop = toolbox.population(n=mu)
        
        # Get the existing logbook
        with open('logbook_file_scenario{}'.format(scenario_num), 'rb') as f:
            logbook = pickle.load(f)
            print(logbook)

    # otherwise create a new one randomly
    else:
        print('Creating new pop, because no previous pop was available')
        pop = toolbox.population(n=mu)

        # Initialise the logbook
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "fitness", "num_vehicles"
        logbook.chapters["fitness"].header = "min", "max", "avg", 'median', "std"
        logbook.chapters["num_vehicles"].header = "min", "avg", "max"

    #======================================================
    

    l = len(pop[0])
    CXPB = 0.9 # [0.6-0.8] or [0.7-0.9]
    print('Pc = ', CXPB)
    
    MUTPB = 1/mu # [1/l, 1/mu] => [1/131 1/40]
    print('Pm = ', MUTPB)

    lambda_ = mu
    print('lambda_: ',lambda_)

    # Initialise stats for logbook
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=lambda ind: sum(ind))
    mstats = tools.MultiStatistics(fitness=stats_fit, num_vehicles=stats_size)

    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("avg", np.mean)
    mstats.register("median", np.median)
    mstats.register("std", np.std)

    #=============================================================================================
    #   STEP 1: TRAIN THE GA
    #=============================================================================================
    print('>>>>>>>  STEP 1 - TRAIN THE GA <<<<<<<')
    start_time = time.time()
    
    population, logbook2 = modifiedEaMuCommaLambda(pop, toolbox, mu, lambda_, CXPB, MUTPB, NGEN, last_gen, logbook, mstats, scenario_num, verbose = True)
    
    with open('logbook_file_scenario{}'.format(scenario_num), 'wb') as f:
        pickle.dump(logbook2, f)
        
    end_time = time.time()

    print("Running time step 1: {} mins".format((end_time-start_time)/60))
    #======================================================




    #================================================================================
    #   STEP 2: EVALUATE best individuals in final pop on testing set (100 shifts)
    #================================================================================
    print('>>>>>>>  STEP 2 - EVALUATE THE INDS IN FINAL POP<<<<<<<')

    start_time = time.time()
    warnings.filterwarnings('ignore')

    # Get final unique individuals in the population
    population = [tuple(ind) for ind in population]
    pop_unique_ind = set(population)
    print('Unique individuals in final pop: {}'.format(len(pop_unique_ind)))
    # remove those with more than max number of agents
    pop_unique_ind = [ind for ind in pop_unique_ind if sum(ind) <= MAX_NUM_AGENTS]

    # Evaluate all unique individuals on 100 shifts of testing set
    
    with open('./../../../data/testing_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)

    # Open patrol beat boundaries once and pass on to all processes
    patrol_beats_df = gpd.read_file('{}data/patrol_beats/patrol_beats.shp'.format('./../../../'))
    patrol_beats_df.rename(columns={"centroid_n": "centroid_node"}, inplace=True)


    list_of_list_shift_dfs = [pool.map(partial(run_ABMs_on_one_shift_eval, individual = ind, patrol_beats_df=patrol_beats_df), list_shifts) for ind in pop_unique_ind]
    

    pool.close() 
    #======================================================
    

    ## Find the best individual
    min_fit, best_ind = None, None
    dict_final_pop = {}
    for ind, list_dfs in zip(pop_unique_ind, list_of_list_shift_dfs): 
        df_ind= pd.concat(list_dfs)
        # Get the average response time 
        avg_response_time = np.mean(df_ind['Dispatch_time'] + df_ind['Travel_time'])
        fit = avg_response_time
        #print(sum(ind), fit)

        dict_final_pop[ind] = fit

        # If this is the first individual we are evaluating
        if not min_fit:
            min_fit = fit
            best_ind = ind

            # save the evaluation results for best ind to display results
            with open('list_of_shift_dfs_scenario{}.pkl'.format(scenario_num), 'wb') as f:
                pickle.dump(list_dfs, f)
        
        else:

            if fit < min_fit :
                best_ind = ind
                min_fit = fit

                # save the evaluation results for best ind to display results
                with open('list_of_shift_dfs_scenario{}.pkl'.format(scenario_num), 'wb') as f:
                    pickle.dump(list_dfs, f)
   
    print('All time best ind on testing set: {}, {}'.format(best_ind, min_fit))
    with open('best_ind_scenario{}'.format(scenario_num), 'wb') as f:
        pickle.dump(best_ind, f)

    with open('dict_final_pop_scenario{}.pkl'.format(scenario_num), 'wb') as f:
        pickle.dump(dict_final_pop, f)

    end_time = time.time()
    
    print("Running time step 2: {} mins".format((end_time-start_time)/60))
        

# %%

toolbox = create_toolbox()


if __name__ == "__main__":
    main()

# %%
