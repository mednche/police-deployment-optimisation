
import AgentFramework
import IncidentFramework
import GraphFramework
import SchedulerFramework

import datetime as dt
import pandas as pd
import numpy as np
import os
import imageio

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib as mpl

import random
import networkx as nx
import osmnx as ox
import geopandas as gpd
import math

class Beat():
    """  
        A patrol beat.
    """
    def __init__(self, env, name, precinct,  geometry, centroid_node, historical_cfs_scenario, historical_crimes_scenario, num_hot_streets):

        """
            Inputs:
            - env: the environment entity
            - name (string): name of the patrol beat(e.g. '107')
            - precinct (string): name of the precinct the patrol beat belongs to (e.g '1')
            - geometry (polygon): list of linestring objects spatially representing the shape of the beat
            - centroid_node (integer): the osmid of the node closest to the spatial centroid of the beat
            - historical_cfs_scenario: a dataset of historical cfs across the force for the simulated demand scenario
            - historical_crimes_scenario: a dataset of historical crimes across the force for the simulated demand scenario
            - num_hot_streets (integer): the number of streets to patrol in the beat as part of the patrol route

            Attributes:
            - name (string): name of the patrol beat(e.g. '107')
            - precinct (string): name of the precinct the patrol beat belongs to (e.g '1')
            - geometry (polygon): list of linestring objects spatially representing the shape of the beat
            - centroid_node (integer): the osmid of the node closest to the spatial centroid of the beat
            - graph: the graph entity (extracted from env) for easy access
            - historical_cfs: the historical incidents that took place in the beat for the simulated demand scenario
            - historical_crimes: the historical crimes that took place in the beat for the simulated demand scenario
            - streets_to_patrol (list of edge osmids): the list of edge osmids in the beat that need to be patrolled
            - patrol_route (list of node osmids): ordered list of node osmids in the beat to be visited by a patrolling agent
        """

        self.name = name
        self.precinct = precinct
        self.geometry = geometry
        self.centroid_node = centroid_node
        self.env = env
        self.graph = env.graph

        # =================================================================
        # NOTE: I used this for getting targetted configurations (towards hsitorical crimes/incidents)
        # =================================================================
        
        self.historical_crimes = None
        # filter the historical crimes for THAT particular beat
        if isinstance(historical_crimes_scenario, pd.DataFrame):
            self.historical_crimes = self.getIncidentsBeat(historical_crimes_scenario)
        
        self.historical_cfs = None
        # filter the historical incidents for THAT particular beat
        if isinstance(historical_cfs_scenario, pd.DataFrame):
            self.historical_cfs = self.getIncidentsBeat(historical_cfs_scenario)

        # =================================================================

        # get the streets to patrol based on incidents
        self.streets_to_patrol = self.getHotStreets(num_hot_streets)

        # create patrol routes between the chosen streets_to_patrol
        self.patrol_route = self.createPatrolRoute()


 
    def getIncidentsBeat(self, historical_crimes) :
        """
            This function returns a subset of historical_crimes that took place in this beat.
        """

        # Get the historical crimes that took place in the beat
        df_beat = historical_crimes[historical_crimes.Patrol_beat == self.name]

        return df_beat


    def createPatrolRoute(self):
        """ This function calculates a route that visits all streets_to_patrol. 
        This route will be used by the agents when patrolling.
        Important: the route begins at the centroid node of the patrol beat. 
        This means that agents will have to pass via the centroid every time they finish a round and start a new one.
        
        Returns: 
        - route: a list of node osmids to be visisted by patrolling agents
        """

        # Initialise remaining_hot_streets as all the streets_to_patrol 
        remaining_hot_streets = self.streets_to_patrol.copy()
        
        # Node is the node from where to calculate distances to remaining_hot_streets
        # Starting node is the centroid node of the patrol beat
        node = self.centroid_node  
        route = [node]

        # While there are still remaining hot streets to visit:
        while len(remaining_hot_streets) :

            # Find the index of the closest remaining street
            index = self.graph.findClosestHotStreetIndex(node, remaining_hot_streets)

            # Find best route to the u node of closest edge (weighting by density_hist_inc_desc for maximum deterrence)
            new_route_segment = self.graph.findBestRoute(start_node = node, target_node = remaining_hot_streets.loc[index].u, 
                                                                                weight = 'density_hist_inc_desc')
          
            # Add node v at the end of the route
            v_node = remaining_hot_streets.loc[index].v
            if v_node not in route :
                new_route_segment.append(v_node)
                
            # Add this new segment of route to the patrol route
            route+= new_route_segment[1:]

            # Update remaining streets (remove the one just added to the route)
            remaining_hot_streets.drop(index, inplace = True, errors = 'ignore')

            # Update the node from where to calculate distances to remaining_hot_streets
            # Take the last node in the route as new starting point
            node = route[-1] 

        return route
       


    def getRandomStreets(self, number_streets):
        """ This function randomly selects number_streets streets in the beat to be part of the patrol route.
        This is used when no historical CFS incident or crime dataset was provided by the user.
        Important: the random_state is fixed by a random seed to avoid stochasticity in the ABM. 
        In that sense, the streets are chosen arbitrarily not randomly.
        Returns:
        - a list of edge osmids of size number_streets. 
        """
        
        streets_within_patrol_beat = self.graph.get_streets_within_beat(self.geometry )
        streets_within_patrol_beat_utm = ox.projection.project_gdf(streets_within_patrol_beat)
        random_streets_utm_df = streets_within_patrol_beat_utm.sample(number_streets, random_state=11) # <---RANDOM SEED FIXED TO AVOID STOCHASTICITY IN THE ABM
        
        return random_streets_utm_df[['u', 'v', 'geometry']]
        


    def getHotStreets(self, number_streets):
        """ This function selects number_streets streets in the beat to be part of the patrol route 
        based on their density_hist_inc value. 
        To identify streets with incidents, this function uses the get_streets_with_incident_in_beat method from the graph entity. This
        relies on the edge value for density_hist_inc that was created in the initialisation of the env entity. This value is 0 if no
        historical dataset was provided.
        Returns:
        - a list of edge osmids The list is of size number_streets, or smaller if there aren't that many streets with incidents in the beat 
        """

        ## SELECT ALL EDGES WITH INCIDENTS IN THE PATROL BEAT 
        # (if user did not provide a historical crime dataset, or if no historical crime in specific beat:
        # this will return an empty list)
        streets_with_incidents_in_beat = self.graph.get_streets_with_incident_in_beat(self.name)

        # If any street had incidents
        if len(streets_with_incidents_in_beat) > 0:

            # ==================================================================================================
            #        REMOVE DUPLICATE STREETS (two way roads ) USING OSMID BEFORE IDENTIFYING HOT STREETS       #
            # ==================================================================================================
            
            def osmidIsInt(row) :
                return type(row.osmid) == int

            # Get the rows that are int
            subset_streets_osmid_int = streets_with_incidents_in_beat[streets_with_incidents_in_beat.apply(osmidIsInt, axis=1)]
            
            # Get the rows that are list
            subset_streets_osmid_list = streets_with_incidents_in_beat[~streets_with_incidents_in_beat.index.isin(subset_streets_osmid_int.index)]
            
            # Remove duplicated rows (based on index) for the ones where osmid is a list
            subset_streets_osmid_list = subset_streets_osmid_list[~subset_streets_osmid_list.index.duplicated()]

            # Remove duplicate from rows that are int
            subset_streets_osmid_int.drop_duplicates(subset=['osmid'], inplace=True)

            # Merge back two subset dfs together
            new_streets_with_incidents_in_beat = pd.concat([subset_streets_osmid_int, subset_streets_osmid_list])

            # ==================================================================================================
            
            # Select num_streets hottest streets amongst those with incidents
            if len(new_streets_with_incidents_in_beat) > number_streets:
                print('{} streets with incidents found in beat {}'.format(len(new_streets_with_incidents_in_beat), self.name))
                # For simplicity, only take the num_streets hottest of each patrol beat if more than num_streets (e.g. > 5)
                hot_streets_df = new_streets_with_incidents_in_beat.sort_values(by='density_hist_inc',ascending=False).head(number_streets) 
                
                
            else :
                print('Only found {} streets with incidents'.format(len(new_streets_with_incidents_in_beat)))
                hot_streets_df = new_streets_with_incidents_in_beat
                
            # Convert to UTM so we can calculate the distance to each node with shapely
            hot_streets_utm_df = ox.projection.project_gdf(hot_streets_df)[['u', 'v', 'geometry']]
            

        else :
            # If no historical crime occured in the patrol beat: get random streets
            print('NO STREETS WITH HISTORICAL CRIME DETECTED IN BEAT - {} streets chosen arbitrarily'.format(number_streets))
            hot_streets_utm_df = self.getRandomStreets(number_streets)

        return hot_streets_utm_df

        
    
    
    