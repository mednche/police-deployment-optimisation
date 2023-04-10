#!/usr/bin/env python
# coding: utf-8

## Pre-process incidents and crimes datasets before running ABM.
# To be run each time you re-generate the graph G in 'Data pre-processing' notebook

# This script needs to be run from terminal for multiprocessing to work. 
# Takes 2 to 4 hours on Uni of Leeds's HPC (40 nodes)

# The preprocessing involves finding the nearest node on the graph to the location of each incident/crime. 

import networkx as nx
import osmnx as ox
import geopandas as gpd
import datetime as dt
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from shapely.geometry import Point

def find_nearest_node(inc, patrol_beats, G ):
    """ Function finds and returns the nearest node WITHIN PATROL BEAT to an incident on the graph"""
    # Find the patrol beat for the incident
    beat_num = inc.Patrol_beat
    # Get the geometry of the patrol beat
    beat_poly = patrol_beats[patrol_beats['name'] == beat_num].geometry.iloc[0]
    
    # Truncate G by polygon
    G_beat = ox.truncate.truncate_graph_polygon(G, beat_poly, truncate_by_edge=False)

    node = ox.get_nearest_node(G_beat, (inc.lat, inc.lon))

    return node

def main():

    from functools import partial
    from multiprocessing import Pool, cpu_count

    start_time = time.time() 

    # Import patrol_beats
    patrol_beats = gpd.read_file('../data/patrol_beats/patrol_beats.shp')
        
    #Import graph
    G=nx.read_gpickle("../data/G.gpickle")

    # ===========================================================================
    ##                                 1. Incidents
    # ===========================================================================

    # Import CFS incidents
    incidents = pd.read_csv("../data/incidents.csv")
   
    if len(incidents): 

        # Convert to a list for multiprocessing each incident
        list_incidents = [incident for index, incident in incidents.iterrows()]
        print('Converted df to list of {} incidents'.format(len(list_incidents)))

        with Pool(processes=cpu_count()) as pool:
            result_list = pool.map(partial(find_nearest_node, patrol_beats=patrol_beats, G = G), list_incidents)
        
        # Get the result of the map() as a list
        incidents['Node'] = list(result_list)
        print('Finished multiprocessing incidents')

        ## SAVE AS CSV
        incidents.to_csv('"../data/incidents.csv"', index=False)

    # ===========================================================================



    # ===========================================================================
    ##                                 2. Crimes
    # ===========================================================================

    # Import crimes
    crimes = pd.read_csv("../data/crimes.csv")
   
    if len(crimes): 
        # Convert to a list for multiprocessing each incident
        list_crimes = [incident for index, incident in incidents.iterrows()]
        print('Converted df to list of {} crimes'.format(len(list_crimes)))

        with Pool(processes=cpu_count()) as pool:
            result_list = pool.map(partial(find_nearest_node, patrol_beats=patrol_beats, G = G), list_crimes)
        
        # Get the result of the map() as a list
        crimes['Node'] = list(result_list)
        print('Finished multiprocessing crimes')

        ## SAVE AS CSV
        crimes.to_csv('"../data/crimes.csv"', index=False)

    # ===========================================================================


    print("--- %s seconds ---" % (time.time() - start_time))

        
if __name__ == '__main__':
    main()

# %%
