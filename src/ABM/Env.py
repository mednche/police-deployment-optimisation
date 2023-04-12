#%%

import IncidentFramework
import BeatFramework

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

import GraphFramework

import multiprocessing
from multiprocessing import cpu_count

#%%

class Environment():
    """ This entity represents the environment of the model.
        
    Attributes:
    - G: NetworkX graph (the road network)
    - G_proj: UTM NetworkX graph
    - gdf_nodes: geo dataframe
    - gdf_nodes_proj: geo dataframe UTM
    - patrol beats: geodataframe
    - incidents: dataframe of priority 1 incidents that took place during the time period
 
    """

    def __init__(self, path_to_root, cfs_incidents, ABM_START_DATETIME, ABM_END_DATETIME, historical_cfs_scenario = None, historical_crimes_scenario = None, num_hot_streets = 5, travel_speed_percent=None, patrol_beats_df=None):
        """
        Inputs:
        - path_to_root: path to repo root directory
        - ABM_START_DATETIME, ABM_END_DATETIME: datetimes representing the start and end of the simulated period
        - cfs_incidents: dataframe of priority 1 incidents that took place during the simulated period
        - historical_crimes_scenario (optional): the historical crimes that took place on similar time periods in the past (used for routing on patrol and for initialising 'streets_to_patrol' and 'patrol_route')
        - num_hot_streets: number of hot streets to patrol in a beat (as part of a patrol route).

        Note: If 'historical_crimes_scenario' is not provided: random 'density_hist_inc' chosen for each edge
        - initialising edge attribute 'density_hist_inc'
        - initialising patrol beat attribue 'streets_to_patrol' and 'patrol_route'

        Note: part of the GRAPH INIT section of this code is specific to situations where incidents are provided in a dataset at the **node** level and not the **edge** level. 
        A conversion step is thus necessary (via calculateDensityIncEdges).
        """

        self.start_datetime = ABM_START_DATETIME
        self.end_datetime = ABM_END_DATETIME

        self.historical_crimes_scenario= historical_crimes_scenario

        # =============================================================
        #                   INIT THE GRAPH (ROAD NETWORK)               
        # =============================================================
       
        G=nx.read_gpickle("{}data/G.gpickle".format(path_to_root))
        G_proj=nx.read_gpickle("{}data/G_proj.gpickle".format(path_to_root))
        self.graph = GraphFramework.Graph(G, G_proj)

        # Initialise gdf_edges column 'density_hist_inc' with zeros
        self.graph.initAttributeDensityHistInc()

        # Modify travel_time_mins on edges based on travel speed alteration (for SA_travelSpeed only)
        if travel_speed_percent :
            self.graph.gdf_edges.travel_time_mins = ((self.graph.gdf_edges['length']/1000) / ((self.graph.gdf_edges.speed_kph+ (travel_speed_percent*self.graph.gdf_edges.speed_kph)/100)/60))
            

        #### FILTER THE HISTORICAL CRIME DATASET ##
        # if the user provided a crime dataset, get the historical crimes that match the time period (shift) being simulated
        if not (self.historical_crimes_scenario is None) :

            #######  COMPLETE density_hist_inc COLUMN IN GDF_EDGES   ######
            # ALLOCATE density_hist_inc TO ALL EDGES AROUND EACH NODE WITH INCIDENTS (for routing)
            self.graph.calculateDensityIncEdges(historical_crimes_scenario) ## THIS IS FOR DETROIT ONLY AS CRIMES ARE SPATIALLY PERTURBED
     

        # if user did not provide any historical crime dataset    
        else :
            print('User did not provide any historical crime dataset. Deterrence score will be 0.')
        
        ## Add column with num_inc_attributed_edge_desc
        self.graph.add_density_hist_inc_desc_attribute_edges()

        # UPDATE G to incorporate the new attributes for routing decisions!
        self.graph.udpate_G()
       
        # =============================================================

        


        # =============================================================
        #                   INIT THE PATROL BEATS      
        # =============================================================
        # Get the boundaries of patrol beats
        if patrol_beats_df is None : 
            patrol_beats_df = gpd.read_file('{}data/patrol_beats/patrol_beats.shp'.format(path_to_root))
            patrol_beats_df.rename(columns={"centroid_n": "centroid_node"}, inplace=True)

      
        self.patrol_beats = []
        for _, beat in patrol_beats_df.iterrows() :
            self.patrol_beats.append(BeatFramework.Beat(self, beat['name'], beat['precinct'], beat['geometry'], 
                                                            beat['centroid_node'], historical_cfs_scenario, historical_crimes_scenario, num_hot_streets))

        # =============================================================        




        # =============================================================
        #          INIT THE CFS INCIDENTS FOR THE TIME PERIOD   
        # =============================================================
        
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

        # =============================================================

