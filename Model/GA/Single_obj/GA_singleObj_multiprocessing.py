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
sys.path.append('../../Model/Framework/')

sys.stdout.flush()

import ModelFramework_no_MESA

import multiprocessing
import Env

sys.path.append('../')
#from Evaluate_GA1_strat import run_ABMs_on_one_shift as run_ABMs_on_one_shift_eval

#ABM_START_DATETIME = dt.datetime(2019,1, 20, 16)
#ABM_END_DATETIME = dt.datetime(2019,1, 20, 21)
# ABM_NUM_STEPS = 60 # optional.Use for gif making but not for GA

# TODO: change here to desired range of num agents per precinct
MIN_STATION_CAPACITY = 2
MAX_STATION_CAPACITY = 7

MAX_NUM_AGENTS = 60
NUM_PATROL_BEATS = 131
RSS = 1

# get the precincts for Detroit
precincts = gpd.read_file('../../Data/DPD_Precincts/dpd_precincts.shp')
precincts['name'] = precincts['name'].astype(int).astype(str)
# remove the last row (duplicate of precinct 7)
precincts = precincts[:-1]

# get the scas for Detroit
scas = gpd.read_file('../../Data/DPD_Scout_Car_Areas-shp/DPD_SCAs_preprocessed.shp')
scas['bool'] = 0

#%%

def run_ABMs_on_one_shift_eval(shift, individual, patrol_beats_df = None): 
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
        historical_crimes = pd.read_csv("../../Data/Crimes_edited_preprocessed.csv")
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
            ABM_env = Env.Environment('./../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = None, patrol_beats_df=patrol_beats_df)
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



def run_ABMs_on_one_shift(shift, population, toolbox): 
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

                                        
        #trap = io.StringIO()
        #with redirect_stdout(trap):
        ABM_env = Env.Environment('./../../', cfs_incidents_shift, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = None)
 
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




def evalSingleObj(population, scenario_num):
    global MAX_NUM_AGENTS, RSS
    """ Function calls run_ABM_on_shift() for each shift in the training_set.
    It returns the average response time over the 100 shifts
    """
    #os.chdir('/nobackup/mednche/GA-Detroit/GA1_experiment/')
    
    with open('./../training_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)
    
    # WE ONLY EVALUATE ON RSS=1 SHIFT AT EACH GENERATION - RANDOMLY CHOSEN
    indices = np.random.choice(len(list_shifts), RSS, replace=False)
    list_shifts = np.array(list_shifts)[indices]
    #list_shifts = np.random.choice(list_shifts, 2, replace=False)
    #list_shifts = list_shifts[0:2] #<---- remove here
    

    # Convert to tuple for dict key later
    population = [tuple(ind) for ind in population]
    #print('Num_ind_to_evaluate', len(population))

    pop_unique_ind = set(population)
    print('Unique individuals in pop: {}'.format(len(pop_unique_ind)))

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

# Variation algortihm
def modifiedVarAnd(population, toolbox, lambda_, cxpb, mutpb):
    """Modified varOr to look like varAnd, i.e. probas pc and pm are independant
    (but keep the Lambda)
    Population will be the breeding pool (a sample of population mu)
    but can be the entire population mu if needed. 
    Regarless of the size of population in input, it will create lambda offsprings
    """
    #print('lambda_: ',lambda_)
    offspring = []
    #print('mutpb: ', mutpb)
    #print('cxpb: ', cxpb)
    # create the necessary number of offspring
    for i in range(lambda_):

        # Apply crossover
        op1_choice = random.random()
        #print('op1_choice: ', op1_choice)
        if op1_choice < cxpb:        
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            #print('Crossover between {} and {}'.format(ind1, ind2))
            ind, _ = toolbox.mate(ind1, ind2)
            del ind.fitness.values
            #print(' Resulting offsping number {}: {}'.format(i, ind1))
        # No crossover
        else: 
            ind = random.choice(population)

        # Apply mutation
        op2_choice = random.random()
        #print('op2_choice: ', op2_choice)
        if op2_choice < mutpb:  # Apply mutation
            print('mutating ind...')
            ind = toolbox.clone(ind)
            #print('Mutating ind')
            ind, = toolbox.mutate(ind)
            #print('Resulting offspring number {}: {}'.format(i, ind))
            del ind.fitness.values
        offspring.append(ind)

    return offspring


# Evolution algorithm
def modifiedEaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, last_gen, logbook,
                    stats, scenario_num, halloffame=None, verbose=__debug__):

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    #print(population)

    # Evaluate the individuals with an invalid fitness (the ones that have been modified recently)
    # At gen 0 it basically means evaluating all ind in pop
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    try:
        fitnesses = toolbox.evaluate(invalid_ind)
        #fitnesses = map(toolbox.evaluate, invalid_ind)

        
    except:
        raise ValueError("@@@ Can't evaluate fitnesses of invalid_ind : {}".format(invalid_ind))
    
    for ind, fit in zip(invalid_ind, fitnesses):
        #print(sum(ind), fit)
        ind.fitness.values = fit
        #history_index += 1
        #genealogy_history_fit[history_index] = fit
    

    # Write first record in logbook if this is gen 0 (no if picking up from existing gen)
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
        
        #print(population)

        # Parent selection
        # THIS IS THE LINE I HAVE CHANGED, NOW DOING PARENT SELECTION BEFORE BREEDING
        # selecting lambda_ parents from pop of size mu.
        # This addition can be removed though.
        # NB: selTournament relies on selRandom which has replacement
        #print('Selecting best parents for breeding:')
        # If parent selection then I need to lambda_ = mu and remove the selection on offspring
        parent_pool = toolbox.selectParents(population, mu)

        parent_pool_list = [tuple(ind) for ind in parent_pool]
        print('Unique parents: {}'.format(len(set(parent_pool_list))))


        # Vary the population
        offspring = modifiedVarAnd(parent_pool, toolbox, lambda_, cxpb, mutpb)
        print('Offspring after variations:', len(offspring))


        # Evaluate the individuals with an invalid fitness (those that we have never seen before)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print('New ind to evaluate', len(invalid_ind))
        try:
            #fitnesses = map(toolbox.evaluate, invalid_ind)
            fitnesses = toolbox.evaluate(invalid_ind)
        except:
            raise ValueError("@@@ Can't evaluate fitnesses of invalid_ind : {}".format(invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            #print(sum(ind), fit)
            ind.fitness.values = fit
            #history_index += 1
            #genealogy_history_fit[history_index] = fit


        # Survivor selection
        # Select mu individuals for next gen pop from all lambda offspring
        print('Selecting ALL offspring for next gen (no selection):')
        population[:] = offspring # Simple replacement instead of selection using age component only, not fitness based (alternatives: elitism, round-robin tournament, (mu + lambda) selection, (mu, lambda) selection)
        # toolbox.selectSurvivers(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), time = dt.datetime.now(), **record)
        if verbose: 
            print(logbook.stream)
        print('------------------------')

        # Save the population (list of individuals) in a pickle file so we can continue from there.
        with open('population_gen_{}_scenario{}'.format(gen, scenario_num), 'wb') as f:
            pickle.dump(population, f)

        with open('logbook_file_scenario{}'.format(scenario_num), 'wb') as f:
            pickle.dump(logbook, f)

    return population, logbook


#####################################################################


def create_toolbox():
    global MAX_NUM_AGENTS, NUM_PATROL_BEATS

    scenario_num = sys.argv[1]
    print('scenario_num in toolbox', scenario_num)


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
    creator.create("Individual", list, fitness=creator.Fitness)

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
    
    """ def feasible(population):
        global MAX_NUM_AGENTS
        #Feasibility function for the individual. Returns True if feasible False
        #otherwise. Used in DeltaPenalty
        
        print('>>> individual', individual)
        if sum(individual) < MAX_NUM_AGENTS:
            return True
        return False """

    # the defined evaluation function
    toolbox.register("evaluate", evalSingleObj, scenario_num=scenario_num) # CHANGE HERE FOR MULTI
    # constraint handling with penalty function
    #toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1000)) # can also add optional distance function away from constraint value to punish even more
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.1) # independent probability of each attribute to be mutated 0.1
    #toolbox.register("select", tools.selNSGA2) # for multi-objectives - ref_points=ref_points?
    toolbox.register("selectParents", tools.selTournament, tournsize = 3) # forsingle obj
    toolbox.register("selectSurvivers", tools.selBest)

    return toolbox



#%%
# if __name__ == "__main__":
def main():
    # get the scenario number for which to run the GA.
    scenario_num = sys.argv[1]
    print('scenario_num', scenario_num)

    NGEN = 40
    mu = 40

    # Process Pool of 4 workers
    from multiprocessing import Pool

    pool = Pool(processes=min(cpu_count(), mu)) #max of 40 on HPC <---------------
    toolbox.register("map", pool.map) #<---------------""" 
    
    # For genealogy tree
    #history = tools.History()
    # Decorate the variation operators
    #toolbox.decorate("mate", history.decorator)
    #toolbox.decorate("mutate", history.decorator)
    
    ######################################################
    ###### Import or randomly create population ##########
    ######################################################
    
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
            #print('LAST GEN OF THE PREVIOUS LOGBOOK: ', logbook[-1]['gen'])



    # otherwise create a new one randomly
    else:
        print('Creating new pop, because no previous pop was available')
        pop = toolbox.population(n=mu)

        # Initialise the logbook
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "fitness", "num_vehicles"
        logbook.chapters["fitness"].header = "min", "max", "avg", 'median', "std"
        logbook.chapters["num_vehicles"].header = "min", "avg", "max"
        #pop = [tuple(ind) for ind in pop]
    ######################################################


    #history.update(pop)
    

    l = len(pop[0])
    CXPB = 0.9 # [0.6-0.8] or [0.7-0.9]
    print('Pc = ', CXPB)
    #MUTPB = 1/mu
    
    MUTPB = 1/mu # [1/l, 1/mu] 1/131 1/40
    #INDPB = 0.1 
    #print('INDPB = ', INDPB)
    print('Pm = ', MUTPB)
    lambda_ = mu#int(mu + mu/2)
    print('lambda_: ',lambda_)

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=lambda ind: sum(ind))
    mstats = tools.MultiStatistics(fitness=stats_fit, num_vehicles=stats_size)

    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("avg", np.mean)
    mstats.register("median", np.median)
    mstats.register("std", np.std)

    #print('POPULATION SIZE', len(pop))
    #print('>> mstats', mstats.compile(pop))


    # # Define the statistics to use in logbook
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("min", np.min, axis=0)
    # stats.register("max", np.max, axis=0)
    # stats.register("avg", np.mean, axis=0)
    # stats.register("std", np.std, axis=0)

    ##################################################################
    ### STEP 1: SEARCH PARAMETER SPACE FOR 5 best hof strats on various shifts each gen
    ##################################################################
    print('>>>>>>>  STEP 1 - TRAIN THE GA <<<<<<<')
    start_time = time.time()
    
    #hof, population, logbook = modifiedEaMuCommaLambda(pop, toolbox, mu, lambda_, CXPB, MUTPB, NGEN, mstats, hof, verbose = True)
    population, logbook2 = modifiedEaMuCommaLambda(pop, toolbox, mu, lambda_, CXPB, MUTPB, NGEN, last_gen, logbook, mstats, scenario_num, verbose = True)
    
    with open('logbook_file_scenario{}'.format(scenario_num), 'wb') as f:
        pickle.dump(logbook2, f)
        
    end_time = time.time()

    print("Running time step 1: {} mins".format((end_time-start_time)/60))





    ##################################################################
    ### STEP 2: EVALUATE best individuals in final pop on testing set (100 shifts)
    ##################################################################
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
    
    with open('../testing_set_scenario{}.pkl'.format(scenario_num), 'rb') as f:
        list_shifts = pickle.load(f)

    # Open patrol beat boundaries once and pass on to all processes
    patrol_beats_df = gpd.read_file('{}Data/DPD_Scout_Car_Areas-shp/DPD_SCAs_preprocessed.shp'.format('./../../'))
    patrol_beats_df.rename(columns={"centroid_n": "centroid_node"}, inplace=True)


    list_of_list_shift_dfs = [pool.map(partial(run_ABMs_on_one_shift_eval, individual = ind, patrol_beats_df=patrol_beats_df), list_shifts) for ind in pop_unique_ind]
    

    pool.close() #<---------------

    

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

# %%

# %%
