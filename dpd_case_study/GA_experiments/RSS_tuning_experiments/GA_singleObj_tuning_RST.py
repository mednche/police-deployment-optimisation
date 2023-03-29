#%%
from deap import base
from deap import creator
from deap import tools
#from deap import algorithms
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import os
from os import getpid

import warnings

import matplotlib.pyplot as plt

import sys
sys.stdout.flush()

import io
import datetime as dt
import random
from contextlib import redirect_stdout
import time

from multiprocessing import Pool, cpu_count
from functools import partial

import pickle

#import sys
# This is where the model files are stored
#sys.path.insert(0, "/Users/natachachenevoy/Documents/GitHub/ABM-Detroit-Police-Dispatch/Model")
sys.path.append('../../../Model/Framework/')

sys.stdout.flush()

import ModelFramework

import multiprocessing
import Env

sys.path.append('../../')
#from Evaluate_GA1_strat import run_ABMs_on_one_shift as run_ABMs_on_one_shift_eval

#ABM_START_DATETIME = dt.datetime(2019,1, 20, 16)
#ABM_END_DATETIME = dt.datetime(2019,1, 20, 21)
# ABM_NUM_STEPS = 60 # optional.Use for gif making but not for GA

# TODO: change here to desired range of num agents per precinct
##MIN_STATION_CAPACITY = 2
#MAX_STATION_CAPACITY = 7

MAX_NUM_AGENTS = 77
NUM_PATROL_BEATS = 131

# get the precincts for Detroit
precincts = gpd.read_file('../../../data/DPD_Precincts/dpd_precincts.shp')
precincts['name'] = precincts['name'].astype(int).astype(str)
# remove the last row (duplicate of precinct 7)
precincts = precincts[:-1]

# get the scas for Detroit
scas = gpd.read_file('../../../data/DPD_Scout_Car_Areas-shp/DPD_SCAs_preprocessed.shp')
scas['bool'] = 0

#%%

def run_ABMs_on_one_shift_eval(shift, individual): 
    """ Function to be run on each hof strategy identified during training
    to evaluate its performance on the shift (step 2 of the GA)"""
    #population = tuple(population)

    #print('SHIFT: {}'.format(shift))

    #print('Importing ABM environment for shift {}...'.format(shift))
    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]

    # Import environement 
    #os.chdir('/nobackup/mednche/GA-Detroit/Baseline_experiment/')

    try:
        #### CHANGE THIS HERE TO IMPROVE: RUN IN PARALLEL FOR THE 131 PATROL BEATS INSIDE ENV SAVES LOAD OF TIME!
        warnings.filterwarnings('ignore')

        """
        #
        historical_crimes = pd.read_csv("../../data/Crimes_edited_preprocessed.csv")
        historical_crimes['Patrol_beat'] = historical_crimes['Patrol_beat'].apply(str)
        #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
        historical_crimes.Date_Time = pd.to_datetime(historical_crimes.Date_Time)
        # get the incidents one year prior to start of time period (or anything desired: could be 2 years)
        historical_crimes_year = historical_crimes[(historical_crimes['Date_Time'] >= ABM_START_DATETIME- dt.timedelta(days = 365)) & 
                                        (historical_crimes['Date_Time'] < ABM_START_DATETIME)]"""

        cfs_incidents = pd.read_csv("../../../data/incidents.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        # get all incidents within the time interval of the shift
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]

                                        
        
        trap = io.StringIO()
        with redirect_stdout(trap):
            ABM_env = Env.Environment('./../../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = None)
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




    """ def gen_random_parameters():

    # TODO: it has to be completely random!
    global MIN_STATION_CAPACITY, MAX_STATION_CAPACITY, precincts, scas

    # random number of agents
    #k = random.randint(MIN_STATION_CAPACITY,MAX_STATION_CAPACITY) # return a number between 2 and 7 (both included)

    patrol_beats = scas.copy()

    # Init the configuration with zero agent in each patrol beat
    #list_num_agents_scas = np.zeros((len(patrol_beats),), dtype=int)
    #print(list_num_agents_scas)

    for precinct in precincts['name']:
        # get scas for precinct
        # random number of agents to sample from precinct
        k = random.randint(MIN_STATION_CAPACITY,MAX_STATION_CAPACITY)
        chosen_scas = patrol_beats[patrol_beats.precinct == precinct].sample(k)
        #print(chosen_scas.index)
        for i in chosen_scas.index :
            #print(i)
            patrol_beats.loc[i, 'bool'] = 1
        

    return tuple(patrol_beats['bool'].tolist())
     """




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
        model = ModelFramework.Model(ABM_env, configuration, 'Ph')
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



def run_ABMs_on_one_shift(shift, population, toolbox): 
    """ run one ABM (for one time period) for each individual in the population """
    #population = tuple(population)

    #print('SHIFT: {}'.format(shift))

    #print('Importing ABM environment for shift {}...'.format(shift))
    ABM_START_DATETIME = shift[0]
    ABM_END_DATETIME = shift[1]

    # Import environement 
    #os.chdir('/nobackup/mednche/GA-Detroit/Baseline_experiment/')

    try:
        #### CHANGE THIS HERE TO IMPROVE: RUN IN PARALLEL FOR THE 131 PATROL BEATS INSIDE ENV SAVES LOAD OF TIME!
        warnings.filterwarnings('ignore')

        """historical_crimes = pd.read_csv("../../data/Crimes_edited_preprocessed.csv")
        historical_crimes['Patrol_beat'] = historical_crimes['Patrol_beat'].apply(str)
        #historical_crimes['Precinct'] = historical_crimes['Precinct'].apply(str)
        historical_crimes.Date_Time = pd.to_datetime(historical_crimes.Date_Time)
        # get the incidents one year prior to start of time period (or anything desired: could be 2 years)
        historical_crimes_year = historical_crimes[(historical_crimes['Date_Time'] >= ABM_START_DATETIME- dt.timedelta(days = 365)) & 
                                        (historical_crimes['Date_Time'] < ABM_START_DATETIME)]"""

        cfs_incidents = pd.read_csv("../../../data/incidents.csv")
        cfs_incidents.Date_Time = pd.to_datetime(cfs_incidents.Date_Time)
        cfs_incidents.Date_Time = cfs_incidents.Date_Time.dt.tz_localize(None)
        cfs_incidents['Patrol_beat'] = cfs_incidents['Patrol_beat'].apply(str)
        cfs_incidents['Precinct'] = cfs_incidents['Precinct'].apply(str)
        # get all incidents within the time interval of the shift
        cfs_incidents_shift= cfs_incidents[(cfs_incidents['Date_Time'] >= ABM_START_DATETIME) & 
                                        (cfs_incidents['Date_Time'] < ABM_END_DATETIME)]

                                        
        #trap = io.StringIO()
        #with redirect_stdout(trap):
        ABM_env = Env.Environment('./../../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = None)
 
        warnings.filterwarnings('default')
    except:
        raise ValueError('@@@ Env import problem @@@ for shift = {}'.format(shift))
           
    try:
        list_of_dfs = toolbox.map(partial(run_single_ABM, ABM_env = ABM_env), population)
    except:
        raise ValueError('@@@ failed to run ABM on an individual of pop')
    
    
    dict_for_shift = dict(zip(population,list_of_dfs))
    #print('dict_for_shift:{}'.format(len(dict_for_shift)))
    #print(dict_for_shift.keys())
   
    #print("Process {} finised".format(getpid()))
    return dict_for_shift




def evalSingleObj(population, test, training_sample_size, scenario_num,):
    global MAX_NUM_AGENTS
    """ Evaluates the fitness (training or test fitness if test==True) of all individuals in a population
    Function calls run_ABM_on_shift() for each shift in list_shifts (training_set/test_set).
    It returns the list of indivuals' fitnesses
    """
    #os.chdir('/nobackup/mednche/GA-Detroit/GA1_experiment/')
    if test == True:
        print('Calculating test fitness on 20 shifts')
        with open('./../../testing_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
            list_shifts = pickle.load(f)
            # WE ONLY EVALUATE ON 2 SHIFTS AT EACH GENERATION
            indices = np.random.choice(len(list_shifts), 20, replace=False) # CHANGE HERE TO A HIGHER NUMBER
            list_shifts = np.array(list_shifts)[indices]
    
    else :
        with open('./../../training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
            list_shifts = pickle.load(f)
        
        # WE ONLY EVALUATE ON 2 SHIFTS AT EACH GENERATION
        indices = np.random.choice(len(list_shifts), training_sample_size, replace=False)
        list_shifts = np.array(list_shifts)[indices]
        #list_shifts = np.random.choice(list_shifts, 2, replace=False)
        #list_shifts = list_shifts[0:2] #<---- remove here
        

    # Convert to tuple for dict key later
    population = [tuple(ind) for ind in population]
    #print('Num_ind_to_evaluate', len(population))
    pop_unique_ind = set(population)
    #print('Num unique ind,', len(pop_unique_ind))
    # only select the feasible individual to run the ABM on!
    pop_feasible_unique_ind = [ind for ind in pop_unique_ind if sum(ind) <= MAX_NUM_AGENTS ]
    print('Num unique feasible ind,', len(pop_feasible_unique_ind), ' of ', len(population))

    ## ONLY RUN ABM ON pop_feasible_unique_ind as it takes CPU time!
    list_dict_for_shift = [run_ABMs_on_one_shift(shift,pop_feasible_unique_ind,toolbox) for shift in list_shifts]
    print('-------->>>> Finished evaluating {} ABMs for each ind in population of {} inds<<<<<-------- '.format(len(list_shifts), len(pop_unique_ind)))
    
    list_fitnesses = []
    # For each ind in population
    # The dictionnary is essential to not have to evaluate duplicate individuals
    for ind in population:
        # PENALTY HANDLING: assign bad fitness value! (1000)
        if sum(ind) > MAX_NUM_AGENTS :
            #print('>>>>> PENALTY HANDLING!', sum(ind))
            avg_response_time = 1000

        else :
            list_dfs = [dict_for_shift[ind] for dict_for_shift in list_dict_for_shift]
            #print('list_dfs:', list_dfs)
            # Concatenate all df_metrics across shifts for that ind
            df_ind= pd.concat(list_dfs)
            #print('df_ind shaoe:', df_ind.shape)
            # Get the average response time 
            avg_response_time = np.mean(df_ind['Dispatch_time'] + df_ind['Travel_time'])
        
        # Save value in a list of size: len(pop)
        list_fitnesses.append((avg_response_time,))

    return list_fitnesses


# Variation algorithm
def modifiedVarAnd(population, toolbox, lambda_, cxpb, mutpb, indpb):
    """Modified varOr to look like varAnd, i.e. probas pc and pm are independant
    (but keep the Lambda)
    Population will be the breeding pool (a sample of population mu)
    but can be the entire population mu if needed. 
    Regarless of the size of population in input, it will create lambda offsprings
    NB: Some individuals may be created in the process that have more than the max number of agents per precinct due to crossovers"""
    #print('lambda_: ',lambda_)
    offspring = []
    # create the necessary number of offspring
    for i in range(lambda_):
        #print('Creating offspring {}?'.format(i))
        op1_choice = random.random()
        op2_choice = random.random()
        #print('Random throw: {}'.format(op_choice))

        if op1_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            #print('Crossover between {} and {}'.format(ind1, ind2))
            ind, _ = toolbox.mate(ind1, ind2)
            #print(' Resulting offsping number {}: {}'.format(i, ind1))

            if op2_choice < mutpb:  # Apply mutation
                ind = toolbox.clone(ind1)
                #print('Mutating offspring too')
                ind, = toolbox.mutate(ind)
                #print('Resulting offspring number {}: {}'.format(i, ind))

            del ind.fitness.values
            offspring.append(ind)

        else:                    # Apply reproduction if no crossover
            #print('No crossover')
            ind = random.choice(population)
            #print('Resulting offspring number {}: {}'.format(i, ind))

            if op2_choice < mutpb:  # Apply mutation
                ind = toolbox.clone(ind)
                #print('Mutating ind')
                ind, = toolbox.mutate(ind)
                #print('Resulting offspring number {}: {}'.format(i, ind))
                del ind.fitness.values

            offspring.append(ind)
    
    return offspring


# Evolution algorithm
def modifiedEaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, indpb, ngen, last_gen, logbook,
                    stats, scenario_num, training_sample_size, run_num, halloffame=None, verbose=__debug__):

    assert lambda_ >= mu, "lambda must be greater or equal to mu."


    # Evaluate the individuals with an invalid fitness (the ones that have been modified recently)
    # At gen 0 it basically means evaluating all ind in pop
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    try:
        training_fitnesses = toolbox.evaluate(invalid_ind, test=False, training_sample_size=training_sample_size) # evaluate on training set     
    except:
        raise ValueError("@@@ Can't evaluate fitnesses of invalid_ind : {}".format(invalid_ind))
    
    for ind, fit in zip(invalid_ind, training_fitnesses):
        ind.fitness.values = fit
        
    # get best ind:
    pop_of_single_best_ind = toolbox.selectBest(population, 1)
    print('pop_of_single_best_ind:', pop_of_single_best_ind)
    test_fitness = toolbox.evaluate(pop_of_single_best_ind, test=True, training_sample_size=None) # even though best_ind is on their own, they need to be a list
        
    # Write first record in logbook
    record = stats.compile(population) if stats is not None else {}
    logbook.record(
            run_num = run_num,
            gen= 0, 
            #nevals=len(population), 
            time = dt.datetime.now(), 
            training_fitness = pop_of_single_best_ind[0].fitness.values[0], 
            test_fitness = test_fitness[0][0],
            **record
        )
    if verbose:
        print(logbook.stream)


    # Save the population (list of individuals) in a pickle file so we can continue from there.
    with open('population_gen_{}_scenario{}_RSS{}_run{}'.format(last_gen, scenario_num, training_sample_size, run_num), 'wb') as f:
        pickle.dump(population, f)


    # Begin the generational process
    for gen in range(last_gen+1, last_gen + ngen + 1):
        sys.stdout.flush()
        print('------------------------')
        print('--------------')
        print('GENERATION:', gen)
        print('--------------')
        original_stdout = sys.stdout
        """with open('logbook', 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # Append new line at the end of the logbook
            print("{}\n".format(logbook.stream), file=f)
            sys.stdout = original_stdout # Reset the standard output to its original value"""
        
        #print(population)

        # Parent selection
        # THIS IS THE LINE I HAVE CHANGED, NOW DOING PARENT SELECTION BEFORE BREEDING
        # selecting lambda_ parents from pop of size mu.
        # This addition can be removed though.
        # NB: selTournament relies on selRandom which has replacement
        #print('Selecting best parents for breeding:')
        # If parent selection then I need to lambda_ = mu and remove the selection on offspring
        parent_pool = toolbox.selectParents(population, mu)
        #print(parent_pool)

        # Vary the population
        offspring = modifiedVarAnd(parent_pool, toolbox, lambda_, cxpb, mutpb, indpb)
        print('Offspring after variations:', len(offspring))

        # Evaluate the individuals with an invalid fitness (those that we have never seen before)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print('New ind to evaluate', len(invalid_ind))
        
        try:
            training_fitnesses = toolbox.evaluate(invalid_ind, test=False, training_sample_size=training_sample_size)
        
        except:
            raise ValueError("@@@ Can't evaluate fitnesses of invalid_ind : {}".format(invalid_ind))
        
        for ind, fit in zip(invalid_ind, training_fitnesses):
            ind.fitness.values = fit

        # Survivor selection
        # Select mu individuals for next gen pop from all lambda offspring
        print('Selecting ALL offspring for next gen (no selection):')
        population[:] = offspring # Simple replacement instead of selection using age component only, not fitness based (alternatives: elitism, round-robin tournament, (mu + lambda) selection, (mu, lambda) selection)
        # toolbox.selectSurvivers(offspring, mu)

        ## TEST FITNESS FOR 1 BEST IND ON UNSEEN DATA AFTER SELECTION OF NEW POPULATION ##
        try:
            # get best ind:
            pop_of_single_best_ind = toolbox.selectBest(population, 1)
            print('pop_of_single_best_ind:', pop_of_single_best_ind)
            test_fitness = toolbox.evaluate(pop_of_single_best_ind, test=True, training_sample_size=None) # even though best_ind is on their own, they need to be a list
        
        except:
            raise ValueError("@@@ Can't evaluate test fitnesses of population : {}".format(population))
        


        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(
            run_num = run_num,
            gen= gen, 
            #nevals=len(population), 
            time = dt.datetime.now(), 
            training_fitness = pop_of_single_best_ind[0].fitness.values[0], 
            test_fitness = test_fitness[0][0],
            **record
            )
        if verbose: 
            print(logbook.stream)
        print('------------------------')

        # Save the logbook in a pickle file so we can continue from there.
        with open('logbook_file_scenario{}_RSS{}_run{}'.format(scenario_num, training_sample_size, run_num), 'wb') as f:
            pickle.dump(logbook, f)

        # Save the population (list of individuals) in a pickle file so we can continue from there.
        with open('population_gen_{}_scenario{}_RSS{}_run{}'.format(gen, scenario_num, training_sample_size, run_num), 'wb') as f:
            pickle.dump(population, f)

    return population, logbook


#####################################################################


def create_toolbox():
    global MAX_NUM_AGENTS, NUM_PATROL_BEATS

    scenario_num = sys.argv[1]

    # Attribute generator (for random attribute values in individuals) 
    def gen_random_parameters():
        """Create an array of NUM_PATROL_BEATS with MAX_NUM_AGENTS random values as 1 and the rest as 0. 
        This prevent the starting population from having too many agents above the constraints 
        (these would be penalised and discarded, ultimately reducing the diversity of the population quite early on!"""
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


    creator.create("Fitness", base.Fitness, weights=(-1.0,)) # TO minimise objective
    creator.create("Individual", list, fitness=creator.Fitness, test_fitness=creator.Fitness)

    ## TOOLBOX
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attribute_generator", gen_random_parameters)
    #toolbox.register("attr_bool", random.randint, 0, 1)

    # Structure initializers
    # NB: initRepeat if same creation function for n = 131 genes (parameters)
    # But here it is initIterate for just once because the different genes are related to each other (max 77 of them will have a 1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute_generator) 
    #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, NUM_PATROL_BEATS)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

    # the defined evaluation function
    toolbox.register("evaluate", evalSingleObj, scenario_num=scenario_num) # CHANGE HERE FOR MULTI
    # constraint handling with penalty function
    #toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1000)) # can also add optional distance function away from constraint value to punish even more
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.1) # independent probability of each attribute to be mutated 0.1
    #toolbox.register("select", tools.selNSGA2) # for multi-objectives - ref_points=ref_points?
    toolbox.register("selectParents", tools.selTournament, tournsize = 3) # forsingle obj
    toolbox.register("selectSurvivers", tools.selBest)
    toolbox.register("selectBest", tools.selBest)

    return toolbox



#%%
# if __name__ == "__main__":
def main():
    # get the scenario number for which to run the GA.
    scenario_num = sys.argv[1]
    print('>> scenario_num: ', scenario_num)

    training_sample_size = int(sys.argv[2]) # convert string to int
    print('>> training_sample_size: ', training_sample_size)

    NGEN = 20
    mu = 40

    # Process Pool of 40 workers
    from multiprocessing import Pool

    pool = Pool(processes=min(cpu_count(), mu)) #max of 40 on HPC <---------------
    toolbox.register("map", pool.map) #<---------------""" 
    
    # these 2 are unused no as I use the best ind training and test fitness
    #stats_training_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    #stats_test_fit = tools.Statistics(key=lambda ind: ind.test_fitness.values)
    
    stats_size = tools.Statistics(key=lambda ind: sum(ind))
    #mstats = tools.MultiStatistics(training_fitness=stats_training_fit, test_fitness=stats_test_fit, num_vehicles=stats_size)
    mstats = tools.MultiStatistics(num_vehicles=stats_size)

    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("avg", np.mean)
    #mstats.register("median", np.median)
    #mstats.register("std", np.std)

    #print('POPULATION SIZE', len(pop))
    #print('>> mstats', mstats.compile(pop))


    # # Define the statistics to use in logbook
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("min", np.min, axis=0)
    # stats.register("max", np.max, axis=0)
    # stats.register("avg", np.mean, axis=0)
    # stats.register("std", np.std, axis=0)

    
    # For genealogy tree
    #history = tools.History()
    # Decorate the variation operators
    #toolbox.decorate("mate", history.decorator)
    #toolbox.decorate("mutate", history.decorator)
    
    ######################################################
    ###### Import or randomly create population ##########
    ######################################################
    
    
    """import os.path
    last_gen = 0
    for gen in range (5, 100):
        if os.path.isfile('population_gen_{}_scenario{}_RSS{}'.format(gen, scenario_num, training_sample_size)):
            last_gen = gen

    # If there is a previous population to carry on from, open the file
    if last_gen != 0 :
        with open('population_gen_{}_scenario{}_RSS{}'.format(last_gen, scenario_num, training_sample_size), 'rb') as f:
            print('Loading previous pop for gen:', last_gen)
            pop = pickle.load(f)
            # But if the number of individuals is different, create a new pop randomly
            if len(pop) != mu :
                print('Creating new pop, because mu was different')
                pop = toolbox.population(n=mu)
        
        # Get the existing logbook
        with open('logbook_file_scenario{}_RSS{}'.format(scenario_num, training_sample_size), 'rb') as f:
            logbook = pickle.load(f)
            print(logbook)
            #print('LAST GEN OF THE PREVIOUS LOGBOOK: ', logbook[-1]['gen'])"""


    # Initialise the logbook
    logbook = tools.Logbook()
    logbook.header = "run_num", "gen", "training_fitness", "test_fitness", "num_vehicles"
    #logbook.chapters["training_fitness"].header = "min"
    #logbook.chapters["test_fitness"].header = "min"
    logbook.chapters["num_vehicles"].header = "min", "avg", "max"
    #pop = [tuple(ind) for ind in pop]

    start_time = time.time()

    # run 1, 2, 3, 4 
    for run_num in range(1, 5) :

        print('RUN NUMER >>', run_num)
        last_gen = 0

        print('Creating new pop')
        pop = toolbox.population(n=mu)

        l = len(pop[0])
        CXPB = 0.9 # [0.6-0.8] or [0.7-0.9]
        print('Pc = ', CXPB)
        MUTPB = 1/mu
        print('MUTPB = ', MUTPB)
        INDPB = 0.1 # [1/l, 1/mu]
        print('Pm = ', MUTPB)
        lambda_ = mu#int(mu + mu/2)# so that fewer individuals to evaluate with ABMs
        print('lambda_: ',lambda_)


        ##################################################################
        ### TRAIN GA FOR RUN 
        ##################################################################
        #hof, population, logbook = modifiedEaMuCommaLambda(pop, toolbox, mu, lambda_, CXPB, MUTPB, NGEN, mstats, hof, verbose = True)
        population, logbook = modifiedEaMuCommaLambda(pop, toolbox, mu, lambda_, CXPB, MUTPB, INDPB, NGEN, last_gen, logbook, mstats, scenario_num, training_sample_size, run_num, verbose = True)



    # AFTER ALL RUNS HAVE FINISHED: SAVE THE FULL LOGBOOK
    with open('full_logbook_file_scenario{}_RSS{}'.format(scenario_num, training_sample_size), 'wb') as f:
        pickle.dump(logbook, f)
        
    end_time = time.time()

    print("Running time: {} mins".format((end_time-start_time)/60))


    ##################################################################
    ### STEP 2: EVALUATE final individuals on test set (100 shifts)
    ##################################################################
    """print('>>>>>>>  STEP 2 - EVALUATE THE FINAL INDS <<<<<<<')

    start_time = time.time()
    warnings.filterwarnings('ignore')

    # Get final unique individuals in the population
    population = [tuple(ind) for ind in population]
    pop_unique_ind = set(population)
    print('Unique individuals in final pop: {}'.format(len(pop_unique_ind)))

    # Evaluate all unique individuals on 100 shifts of testing set
    
    with open('./../../testing_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)

    list_of_list_shift_dfs = [pool.map(partial(run_ABMs_on_one_shift_eval, individual = ind), list_shifts) for ind in pop_unique_ind]
    
    pool.close() #<---------------
    
    ## Find the best individual
    min_fit, best_ind = None, None
    for ind, list_dfs in zip(pop_unique_ind, list_of_list_shift_dfs): 
        df_ind= pd.concat(list_dfs)
        # Get the average response time 
        avg_response_time = np.mean(df_ind['Dispatch_time'] + df_ind['Travel_time'])
        fit = avg_response_time
        #print(sum(ind), fit)

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

    """

    """ with open('logbook_file_scenario{}'.format(scenario_num), 'wb') as f:
        pickle.dump(logbook, f) """

    pool.close()
    #end_time = time.time()
    
    #print("Running time step 2: {} mins".format((end_time-start_time)/60))
        
        
    # # Plot genealogy tree
    # plt.figure(figsize=(5,5))
    # graph = nx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()     # Make the graph top-down
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    # nx.draw(graph, node_color=colors)
    # plt.title('Genealogy tree')
    # plt.show()
    # plt.savefig("Genealogy_tree.png")
# %%

toolbox = create_toolbox()


if __name__ == "__main__":
    main()

# %%

# %%

# %%
