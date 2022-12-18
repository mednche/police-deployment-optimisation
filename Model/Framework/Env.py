#%%

import IncidentFramework
import BeatFramework

#from deap import algorithms
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import os

import matplotlib.pyplot as plt

import sys
import io
import datetime as dt
import random
from contextlib import redirect_stdout
import time
#import sys
# This is where the model files are stored
#sys.path.insert(0, "/Users/natachachenevoy/Documents/GitHub/ABM-Detroit-Police-Dispatch/Model")

sys.path.append('../Framework')

import GraphFramework

import multiprocessing
from multiprocessing import cpu_count

#%%

class Environment():
    """ 
    Attributes:
    - G: NetworkX graph
    - G_proj: UTM NetworkX graph
    - gdf_nodes: geo dataframe
    - gdf_nodes_proj: geo dataframe UTM
    - patrol beats: geodataframe
    - incidents: list of priority 1 incidents
    
    """

    def __init__(self, path_to_root, cfs_incidents, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario=None, historical_crimes_scenario = None, num_hot_streets = 5, travel_speed_percent=None, patrol_beats_df=None):
        """historical_crimes is the dataframe of historical crimes to be used for patrolling behaviour
        It is optional parameter in initialisation. If None: random density_hist_inc chosenfor each edge 
        (for routing on patrol and for initialising streets_to_patrol and patrol_route)
        - initialising edge attribute density_hist_inc
        - initialising patrol beat attribue streets_to_patrol and patrol_route

        historical_cfs_scenario is not used in the GA. But when deciding where to target deploy k agents to hottest beats, I'm calling this attribute outside the model prior to running it (for Benchmark and ABM Experiments)
        
        For SA_hotStreets, we provide different values for the input parameter num_hot_streets (2,5,10). Default is 5"""

        self.start_datetime = ABM_START_DATETIME
        self.end_datetime = ABM_END_DATETIME

        self.historical_crimes_scenario= historical_crimes_scenario


        ####################        INIT THE ENVIRONEMNT (graph)      #######################
        G=nx.read_gpickle("{}G.gpickle".format(path_to_root))
        G_proj=nx.read_gpickle("{}G_proj.gpickle".format(path_to_root))
        self.graph = GraphFramework.Graph(G, G_proj)

        # Initialise gdf_edges column 'density_hist_inc' with zeros
        self.graph.initAttributeDensityHistInc()

        # Modify travel_time_mins on edges based on travel speed alteration (for SA_travelSpeed only)
        if travel_speed_percent :
            self.graph.gdf_edges.travel_time_mins = ((self.graph.gdf_edges['length']/1000) / ((self.graph.gdf_edges.speed_kph+ (travel_speed_percent*self.graph.gdf_edges.speed_kph)/100)/60))
            



        #### FILTER THE HISTORICAL CRIME DATASET ##
        # if the user provided a crime dataset, get the historical crimes that match the time period (shift) being simulated
        if not (self.historical_crimes_scenario is None) :
            """ historical_crimes_shift = historical_crimes[((historical_crimes.Date_Time.dt.weekday == self.start_datetime.weekday()) & 
                        (historical_crimes.Date_Time.dt.time >= self.start_datetime.time())) |
                        ((historical_crimes.Date_Time.dt.weekday == self.end_datetime.weekday()) & 
                        (historical_crimes.Date_Time.dt.time < self.end_datetime.time()))]

            print('Found {} historical crime in the shift period'.format(len(historical_crimes_shift)))
            
            # If too few incidents were found, try for the whole year (non specific shifts)
            ## CHANGE HERE: ARBITRARY VALUE
            if len(historical_crimes_shift) < 500 :
                print('NO HISTORICAL CRIMES OCCURED FOR THAT SHIFT IN DATASET')
                historical_crimes_shift = self.historical_crimes
                print('...using the entire dataset instead') """

       

            #######  COMPLETE density_hist_inc COLUMN IN GDF_EDGES   ######
            ## FOR DETROIT ONLY:
            # ALLOCATE density_hist_inc TO ALL EDGES AROUND EACH NODE WITH INCIDENTS (for routing)
            self.graph.calculateDensityIncEdges(historical_crimes_scenario) ## THIS IS FOR DETROIT ONLY
            # OTHERWISE incidents have a 'edge_index' column from pre-processing
            # calculateDensityIncEdges2(self.incidents)

            #print('{} edges in graph with a density > 0'.format(len(self.graph.gdf_edges[self.graph.gdf_edges.density_hist_inc > 0])))


        # if user did not provide any historical crime dataset    
        else :
            print('user did not provide any historical crime dataset')
            #historical_crimes_scenario = []
        
        ## Add column with num_inc_attributed_edge_desc
        self.graph.add_density_hist_inc_desc_attribute_edges()

        # UPDATE G to incorporate the new attributes for routing decisions!
        self.graph.udpate_G()
       
        ####################################################################
        

        ########################        INIT THE PATROL BEATS      ###########################
        # Get the boundaries of patrol beats
        if patrol_beats_df is None : 
            patrol_beats_df = gpd.read_file('{}Data/DPD_Scout_Car_Areas-shp/DPD_SCAs_preprocessed.shp'.format(path_to_root))
            patrol_beats_df.rename(columns={"centroid_n": "centroid_node"}, inplace=True)

        ### REMOVE HERE!!!!!
        #patrol_beats_df = patrol_beats_df[patrol_beats_df.name == '712']

        self.patrol_beats = []
        for _, beat in patrol_beats_df.iterrows() :
            self.patrol_beats.append(BeatFramework.Beat(self, beat['name'], beat['precinct'], beat['geometry'], 
                                                            beat['centroid_node'], historical_cfs_scenario, historical_crimes_scenario, num_hot_streets))

 
            """  # get historical incidents for that patrol beat
            df_beat = self.getIncidentsBeat(beat, historical_crimes)

            # If no incidents were found, try for the whole year (non specific shifts)
            if len(df_beat) == 0:
                print('NO INCIDENTS OCCURED IN PATROL BEAT {} IN THE PAST YEAR'.format(beat['name']))
                df_beat = self.getIncidentsBeatNonSpecific(beat, historical_crimes)
                print('...and {} incidents found on other shifts'.format(len(df_beat)))

            self.patrol_beats.append(BeatFramework.Beat(self.graph, beat['name'], beat['precinct'], beat['geometry'], 
                                                        beat['centroid_node'], df_beat))
            """

        ############################################################################
        


        ######################      INIT THE CFS INCIDENTS FOR THE TIME PERIOD   ###########################

        # Instanciate incidents and add to list
        # FIFO 
        self.incidents = []
        for index, inc in cfs_incidents.iterrows():
            
            try :
                self.incidents.append(IncidentFramework.Incident(index, inc['Node'],  
                                                            inc['Precinct'], inc['Patrol_beat'], 
                                                            inc['Date_Time'],
                                                            inc['Time On Scene'], inc['Dispatch Time'], 
                                                            inc['Travel Time']))
            except :
                print(index, inc)

        ########################################################################

        #self.dict_hot_streets_in_patrol_beats = self.get_hot_streets_dict()


    """ def getIncidentsBeat(self, patrol_beat, historical_crimes) :
        df_beat = pd.DataFrame()
        # Get the same shift on that day last week and up to the end of the historical_crimes df
        # recomended: 52 weeks prior (the last year basically). 
        
        #calculate number of weeks in the historical_crimes

        for i in range(1,num_weeks):
            start_dt_historic = self.start_datetime - dt.timedelta(days = i*7)
            end_dt_historic = self.end_datetime - dt.timedelta(days = i*7)

            subset_data_shift = historical_crimes.loc[(historical_crimes.Patrol_beat == patrol_beat['name']) & 
                                                    (historical_crimes.Date_Time < end_dt_historic) & 
                                                    (historical_crimes.Date_Time > start_dt_historic)]
            df_beat = df_beat.append(subset_data_shift)
        #print(start_dt_historic, end_dt_historic)
        return df_beat



    def getIncidentsBeatNonSpecific(self, patrol_beat, historical_crimes) :
        df_beat = pd.DataFrame()
        # Get any shift in the 52 weeks prior: the last year basically. 
        
        start_date_historic = (self.start_datetime - dt.timedelta(weeks = 52)).date()
        end_date_historic = (self.start_datetime - dt.timedelta(days = 1)).date()
        print('Trying interval', start_date_historic, end_date_historic)
        df_beat = historical_crimes.loc[(historical_crimes.Patrol_beat == patrol_beat['name']) & 
                                                (historical_crimes.Date_Time.dt.date < end_date_historic) & 
                                                (historical_crimes.Date_Time.dt.date > start_date_historic)]

        
        #print(start_dt_historic, end_dt_historic)
        return df_beat """



    def get_k_hottest_beats_in_precinct(self, precinct, dictionnary, k):
        
        # Select all patrol beats in precinct
        beats = [beat for beat in self.patrol_beats if beat.precinct == precinct]
        
        # Measure the hotness of each patrol beat by summing the risk on each hot edge
        
        def calculate_beat_hotness(row, dictionnary):
            """Adds a column to the patrol beats dataframe with hotness
            Hotness is calculated as the sum of the 'risk' variable in each hot street
            NB: dictionnary is dict_hot_streets_in_patrol_beats"""
            beat = row['name']

            hotness = historical_crimes[historical_crimes.Patrol_beat == beat ]['risk'].sum()
            return hotness
        beats = beats.assign(hotness= beats.apply(calculate_beat_hotness, dictionnary = dictionnary,axis=1))

        # get top k hottest patrol beats
        k_hotest_beats = beats.sort_values(by=['hotness'], ascending = False)[0:k]
        return k_hotest_beats['name'].tolist()

    """ def get_k_random_beats_in_precinct(self, precinct, k):
        # Get list patrol beats in precinct
        patrol_beats = self.patrol_beats[self.patrol_beats['precinct'] == precinct]
        # Random selection of k (k=num_agents) beats WITHOUT replacement for comparison
        random_beats = patrol_beats.sample(n = k, replace = False) 
        
        return random_beats['name'].tolist() """

# %%

# %%
