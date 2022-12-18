
import AgentFramework2
import IncidentFramework
import GraphFramework
import SchedulerFramework
import imp
imp.reload(GraphFramework)

import datetime as dt
import pandas as pd
import numpy as np
import os
import imageio
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib as mpl

import random
import networkx as nx
import osmnx as ox
import geopandas as gpd
#from mesa import Model   
import math

class Beat():

    def __init__(self, env, name, precinct,  geometry, centroid_node, historical_cfs_scenario, historical_crimes_scenario, num_hot_streets):

        self.name = name
        self.precinct = precinct
        self.geometry = geometry
        self.centroid_node = centroid_node

        
        self.env = env
        self.graph = env.graph


        """  # if user did not provide any historical crime dataset
        if len(historical_crimes_shift) == 0:
            print('no historical crimes: getting random streets to patrol')
            self.incidents = []
            # choose random streets to patrol
            self.streets_to_patrol = self.getRandomStreets(5) """
            
        
        # get historical crime incidents for that particular patrol beat (only used for validation: emulate DPD's targetted deployement)
        if isinstance(historical_crimes_scenario, pd.DataFrame):
            self.historical_crimes = self.getIncidentsBeat(historical_crimes_scenario)
        else :
            self.historical_crimes: None


        if isinstance(historical_cfs_scenario, pd.DataFrame):
            self.historical_cfs = self.getIncidentsBeat(historical_cfs_scenario)
        else :
            self.historical_cfs: None

        
        # get the streets to patrol based on incidents
        self.streets_to_patrol = self.getHotStreets(num_hot_streets)
        
        
    
        
            


        # create patrol routes between the choen streets_to_patrol
        self.patrol_route = self.createPatrolRoute()


 
    def getIncidentsBeat(self, historical_incidents) :

        # Get the same shift on that day last week and up to the end of the historical_crimes df
        # recomended: 52 weeks prior (the last year basically). 
        
        #get the historical crimes for the same shift (same time period and same day of the week)
        df_beat = historical_incidents[historical_incidents.Patrol_beat == self.name]

        """  # if no incidents in the beat for the specific shift, use the whole dataset entered by the user
        if len(df_beat) == 0 :
            print('NO INCIDENTS IN BEAT FOR THE TIME PERIOD')
            df_beat = self.env.historical_crimes[self.env.historical_crimes.Patrol_beat == self.name]
            print('... {} incidents found in entire dataset for beat {}'.format(len(df_beat), self.name))
        """

      
        
        return df_beat





    def createPatrolRoute(self):
        """ Function calculates a route that visits all streets_to_patrol. 
        This route will be used by the agents when patrolling.
        The route BEGINS at the centroid node of the patrol beat. 
        That means that agents will have to pass via the centroid every time they finish a round and stat a new one
        """

        remaining_hot_streets = self.streets_to_patrol.copy()
        # Node is the node from where to calculate distances to remaining_hot_streets
        # Starting node is the centroid node of the patrol beat
        node = self.centroid_node  
        route = [node]
        # While there are still remaining hot street to visit:
        while len(remaining_hot_streets) :

            #print('starting node for calculating distance:', node)
            #print('remaining_hot_streets:', len(remaining_hot_streets))
            ####### Find the closest remaining street #########
            index = self.graph.findClosestHotStreetIndex(node, remaining_hot_streets)
            #print('index', index)


            ###### Append route to u (or v) ######
            # route to the u node of closest edge (weighting by density_hist_inc_desc)
            new_route_segment = self.graph.findBestRoute(start_node = node, target_node = remaining_hot_streets.loc[index].u, 
                                                                                weight = 'density_hist_inc_desc')
            #print('new_route_segment', new_route_segment)

            # add node v to the end of the route
            v_node = remaining_hot_streets.loc[index].v
            if v_node not in route :
                new_route_segment.append(v_node)
                #self.target = v_node
                #print('new_route_segment with v', new_route_segment)
            #print('New target for agent {} is node {}'.format(self.unique_id, v_node))
            
            route+= new_route_segment[1:]
            #print('new route: ', route)


            ###### Update remaining streets (remove the one just added to the route)  ######
            remaining_hot_streets.drop(index, inplace = True, errors = 'ignore')

            ####### Update the node from where to calculate distances to remaining_hot_streets ######
            node = route[-1] # take the last node in the route as new strating point
            #print('new node to start routing from', node)

        
        
       

        return route
        #print('FINAL ROUTE UPDATED for agent 28: ', self.route)
        #print('FINAL TARGET UPDATED for agent 28: ', self.target)




    def getRandomStreets(self, number_streets):
        streets_within_patrol_beat = self.graph.get_streets_within_beat(self.geometry )
        streets_within_patrol_beat_utm = ox.projection.project_gdf(streets_within_patrol_beat)
        random_streets_utm_df = streets_within_patrol_beat_utm.sample(number_streets, random_state=11)
        #print(random_streets_utm_df)
        return random_streets_utm_df[['u', 'v', 'geometry']]
        # <------- CHANGE RANDOM STATE HERE (CURRENTLY RANDOM SEED FIXED AND 5 STREETS)


    def getHotStreets(self, number_streets):
        
        ## SELECT ALL EDGES WITH INCIDENTS IN THE PATROL BEAT 
        # (if user did not provide a historical crime dataset, or if no historical crime in specific beat:
        # this will return an empty list)
        streets_with_incidents_in_beat = self.graph.get_streets_with_incident_in_beat(self.name)


        # If any street had incidents
        if len(streets_with_incidents_in_beat) > 0:


            ######## REMOVE DUPLICATE STREETS (OSMID, two way roads) BEFORE IDENTIFYING HOT STREETS #######
            def osmidIsInt(row) :
                return type(row.osmid) == int

            # Get the rows that are int
            subset_streets_osmid_int = streets_with_incidents_in_beat[streets_with_incidents_in_beat.apply(osmidIsInt, axis=1)]
            #print('>>> int subset: ', len(subset_streets_osmid_int))
            
            # Get the rows that are list
            subset_streets_osmid_list = streets_with_incidents_in_beat[~streets_with_incidents_in_beat.index.isin(subset_streets_osmid_int.index)]
            #print('>>> list subset: ', len(subset_streets_osmid_list))
            # Remove duplicated rows (based on index) for the ones where osmid is a list
            subset_streets_osmid_list = subset_streets_osmid_list[~subset_streets_osmid_list.index.duplicated()]


            # remove duplicate from rows that are int
            subset_streets_osmid_int.drop_duplicates(subset=['osmid'], inplace=True)
            #print('>>> int subset: ', len(subset_streets_osmid_int))

            # merge back two subset dfs together
            new_streets_with_incidents_in_beat = pd.concat([subset_streets_osmid_int, subset_streets_osmid_list])
            #print('>>> total: ', len(new_streets_with_incidents_in_beat))
            
            
            
            
            if len(new_streets_with_incidents_in_beat) > number_streets:
                print('{} streets with incidents found in beat {}'.format(len(new_streets_with_incidents_in_beat), self.name))
                # For simplicity, only take the 5 hottest of each patrol beat if more than 5
                hot_streets_df = new_streets_with_incidents_in_beat.sort_values(by='density_hist_inc',ascending=False).head(number_streets) 
                
                
            else :
                print('only found {} streets with incidents'.format(len(new_streets_with_incidents_in_beat)))
                hot_streets_df = new_streets_with_incidents_in_beat
                

            # Need to convert to UTM so that I can calulate the distance to each node with shapely
            hot_streets_utm_df = ox.projection.project_gdf(hot_streets_df)[['u', 'v', 'geometry']]
            

        else :
            # if no historical crime occur in the patrol beat: get random streets
            #print('NO STREETS WITH HISTORICAL CRIME DETECTED IN BEAT - streets chosen arbitrarily')
            hot_streets_utm_df = self.getRandomStreets(number_streets)


        # convert keys to int instead of string ('02'--> 2)
        #dict_hot_streets_in_patrol_beats = {int(k):v for k,v in dict_hot_streets_in_patrol_beats.items()}
        return hot_streets_utm_df

        
    
    
    