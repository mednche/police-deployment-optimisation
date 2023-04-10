
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:55:08 2019

@author: mednche

This file contains the class Model which conrresponds to the main ABM.

"""
import DispatcherFramework
import AgentFramework
import IncidentFramework
import GraphFramework
import SchedulerFramework

from PIL import Image

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

class Model():
    """The model is composed of a dispatcher, some agents and the environment """
    
    def __init__(self, env, configuration, idle_strategy = 'Ph', time_saving_percent = 20):
        """
        Inputs:
        - env contains the following entities:
            - The environement (env.graph -> Graph object)
            - The patrol beats (env.patrol_beats -> Beat objects)
            - The CFS incidents for the whole shift period (env.incidents) -> Incident objects
        - configuration is a list (with a binary value for each patrol beat)
        - idle strategy is 'Ph' by default for all agents (patrolling hot streets, as opposed to random patrol)
        - time_saving_percent is the percentage of the travel time of the currently dispatched agent that can be saved by dispatching another agent on the fly (used in experiments)
        """

        self.env = env
        # List of all incident objects for the shift
        self.incidents = env.incidents
        ## RESET INCIDENTS
        [incident.resetIncidents() for incident in self.incidents]
        

        self.start_datetime = env.start_datetime
        self.end_datetime = env.end_datetime

        # These will be filled when running the model (with run_model() method)
        self.step_time = None # IN MINUTES PUT THIS IN THE BASESCHEDULER
        self.time = None # this is necessary for later logging the time incidents are sorted by agents
        self.num_steps = None

        self.graph = env.graph # the graph is an object with attribute G

        # The scheduler with which to activate the agent 
        # step 1: dispatcher distribute event needs agents to be randomised every turn
        # step 2: no need to randomise every step as step fo each agent independant from others
        self.agents_schedule = SchedulerFramework.BaseScheduler(self) # Scheduler

        #### INITIALISE THE AGENTS ###
        # start with first agent
        agent_ID = 1

        # for each pair of {beat_object : num_agents}
        for beat_object, num_agents in configuration.items() :

            if (num_agents > 1) :
                raise ValueError('A patrol beat should not have more than 1 agent in it!')

            # If there is a unit assigned to this patrol beat, add an agent on its cendroid node
            if (num_agents == 1) :
                a = AgentFramework.Agent(agent_ID, self, idle_strategy, beat_object.centroid_node, beat_object.precinct, beat_object)
                self.agents_schedule.add(a) # Scheduler
            
                agent_ID +=1



        #### INITIALISE THE DISPATCHER ###
        self.dispatcher = DispatcherFramework.Dispatcher(self, time_saving_percent= time_saving_percent)

        

    def scheduled_step(self, date_time, next_date_time):
        """Advance the model by one step. The step represent the time interval between date_time and next_date_time."""

        # Get list of new incidents between date_time and next_date_time
        interval_incidents = [incident for incident in self.incidents if ((incident.call_datetime >= date_time) and (incident.call_datetime < next_date_time))]
        #print("{} new incidents".format(incidents_df.shape[0]))

        # Add these incidents to the dispatcher's queue of unresolved incidents
        self.dispatcher.addIncidentsToQueue(interval_incidents)

        # (1) Dispatcher step: distributing the unattended incidents in the queue to avail agents in precinct
        #print('*********  DISPATCHER STEP *******')
        self.dispatcher.step()

        # (2) Agent step with basic random activation 
        #print('*********  AGENTS STEP *******')
        self.agents_schedule.step()

        # (3) Incidents step: increase dispatch time and travel time by one model step
        #print('*********  INCIDENTS STEP *******')
        for incident in self.incidents:
            incident.step(self)
            

    def run_model(self,step_time, num_steps = None):
        """ This function runs the ABM
            Inputs: 
            - step_time: in minutes, the real time elapsing during one model step
            - num_steps: if provided, run for the number of steps, otherwise run for the whole period duration
            
        """
        self.step_time = step_time

        if num_steps:
            print('running for {} steps'.format(num_steps))
            self.num_steps = num_steps
        else:
            print('running from {} to {}'.format(self.start_datetime, self.end_datetime))
            self.num_steps = (self.end_datetime - self.start_datetime).total_seconds()/(self.step_time*60) # difference is always in seconds by default
        


        #### FOR ANIMATED GIF ONLY (REMOVE THIS FOR GA) ###
        agents_pos_all_steps = []
        inc_pos_all_steps = []
        

        # The model looks back on data from past step_time (1 min)
        # date_time is initialised as the start_datetime
        date_time = self.start_datetime
        step_num = 0
        while (step_num < self.num_steps) : #(date_time <= self.end_datetime) :
            ##### CHANGE HERE: REMOVE THE STEP (which is in minutes)!
            print('-'*15, 'STEP: {}'.format(step_num), '-'*15)
            
            # Add 1 time step (i.e. 1 min)
            next_date_time = date_time + dt.timedelta(minutes=self.step_time)
            self.time = next_date_time # this is necessary for later logging the time incidents are sorted by ahents
            
            
            # Model makes one step (for dispatcher and agents)
            self.scheduled_step(date_time, next_date_time)


            # update date_time
            date_time = next_date_time
            step_num+=1


            #### FOR ANIMATED GIF ONLY (REMOVE THIS FOR GA) ###
            # Agents' locations
            agents_pos = [agent.pos for agent in self.agents_schedule.agents]
            agents_pos_all_steps.append(agents_pos)

            # locations of unresolved incidents
            inc_pos = [inc.node for inc in self.incidents if inc.status > 0 and inc.status < 4]
            inc_pos_all_steps.append(inc_pos)

        return agents_pos_all_steps, inc_pos_all_steps
    


    def make_gif(self, agents_pos_all_steps, inc_pos_all_steps, ABM_NUM_STEPS, path_to_root):
        end_simulation_dt = self.start_datetime + dt.timedelta(minutes = ABM_NUM_STEPS)
        date_times = pd.date_range(self.start_datetime,(end_simulation_dt - pd.DateOffset(minutes=1)) , freq = 'T')

        print(os.getcwd())

        # get precinct polygons
        precincts = gpd.read_file('{}data/precincts/precincts.shp'.format(path_to_root))

        # get patrol beats polygons
        patrol_beats_df = gpd.read_file('{}data/patrol_beats/patrol_beats.shp'.format(path_to_root))

        # get all station nodes
        stations_pos = pd.read_csv('{}data/stations.csv'.format(path_to_root))['Node'].tolist()

        # combine all hot streets from all patrol beats into one geopandas dataframe
        hot_streets_global = pd.concat(
            [patrol_beat.streets_to_patrol for patrol_beat in self.env.patrol_beats])

        
        
        ########################################################################
        def visualise(step, road_network, gdf_nodes, precincts, patrol_beats, agents_pos, inc_pos, stations_pos,
             date_time, hot_streets) :
            """
                Plot agents, incidents and hot streets on the road network
            """

            print(step)
            def fig2array(fig):

                """
                @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
                @param fig a matplotlib figure
                @return a numpy 3D array of RGB values
                """
                # draw the renderer
                fig.canvas.draw()
                
                # Get the RGB buffer from the figure
                w,h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                
                # Reshape
                buf.shape = (h, w,3)

                return buf


            # Make a custom cmap for precincts so that we have 11 pastel colours (no duplicate colour on map)
            cmap1 = mpl.cm.get_cmap('Pastel1')
            cmap2 = mpl.cm.get_cmap('Pastel2')
            list_colors = cmap1.colors + cmap2.colors
            cmap = mpl.colors.ListedColormap(list_colors)

            fig, ax = plt.subplots(1, figsize=(30, 30))
            

            ## Plot the precincts in colour
            precincts.plot(ax=ax, column='name', cmap = cmap, zorder = -2)
            ## Plot the patrol beats outlines
            patrol_beats.boundary.plot(ax=ax, edgecolor="black", zorder = 0)
            patrol_beats.apply(lambda x: ax.annotate(s=int(x['name']), xy=x.geometry.centroid.coords[0], ha='center', weight='bold', fontsize=9),axis=1);


            road_network.plot(ax=ax, alpha=1, zorder = 1, legend=True, edgecolor='grey', linewidth=0.5)
            hot_streets.plot(alpha=1, ax=ax, linewidth=3,edgecolor='red', zorder = 2)
            #hot_streets.apply(lambda x: ax.annotate(s=str(x.name), xy=x.geometry.centroid.coords[0], ha='center', fontfamily = 'sans-serif', weight='bold', fontsize=18),axis=1);

            
            ## NODES
            X_agents, Y_agents, X_stations, Y_stations, X_inc, Y_inc = [], [], [], [], [], []
            sizes, colours = [], []
            for element in gdf_nodes.iterrows():
                if element[0] in stations_pos:
                    x = element[1].x
                    y = element[1].y
                    nc = 'green'
                    ns = 15
                    X_stations.append(x)
                    Y_stations.append(y)
                    sizes.append(ns)
                    colours.append(nc)
                    
                
                    
                elif element[0] in agents_pos:
                    x = element[1].x
                    y = element[1].y
                    nc = 'blue'
                    ns = 15
                    X_agents.append(x)
                    Y_agents.append(y)
                    sizes.append(ns)
                    colours.append(nc)

                elif element[0] in inc_pos:
                    x = element[1].x
                    y = element[1].y
                    nc = 'red'
                    ns = 15
                    X_inc.append(x)
                    Y_inc.append(y)
                    sizes.append(ns)
                    colours.append(nc)
                    
                else:
                    nc = 'grey'
                    ns = 0
                    sizes.append(ns)
                    colours.append(nc)
                    
            # plot incidents as image 
            image_inc = OffsetImage(plt.imread('{}Images/incident.png'.format(path_to_root)), zoom = 0.09)
            ax.scatter(X_inc, Y_inc, s = 0.0001, zorder=1)
            for x, y in zip(X_inc, Y_inc):
                ab = AnnotationBbox(image_inc, (x, y), frameon=False)
                ab.set_zorder(3)
                ax.add_artist(ab)
            
            # plot agents as image of a car
            image_car = OffsetImage(plt.imread('{}Images/car.png'.format(path_to_root)), zoom = 0.15)
            ax.scatter(X_agents, Y_agents, s = 0.0001, zorder=2)
            for x, y in zip(X_agents, Y_agents):
                ab = AnnotationBbox(image_car, (x, y), frameon=False)
                ab.set_zorder(4)
                ax.add_artist(ab)

          
            # title of the image frame with datetime
            ax.set_title("{} {}".format(date_time.date(), date_time.time()))
            
            ax.axis('off') 

            ### Convert figure to image
            image = fig2array (fig)
            
            im = Image.fromarray(image)
            
            left=300
            top = 400
            right = 1900
            bottom = 1650
            b=(left,top,right,bottom)
        
            cropped_im=im.crop(box=b)

            plt.close()

            return cropped_im
            

        ########################################################################

        print('Num steps in gif: {}'.format(len(date_times)))
        
        # Convert them back to latlong
        hot_streets_latlon = hot_streets_global.to_crs("EPSG:4326")

        # DATE RANGE for title of image
        images = [visualise(step, self.graph.gdf_edges, self.graph.gdf_nodes, precincts, patrol_beats_df, agents_pos_all_steps[step], 
                            inc_pos_all_steps[step], stations_pos,
                        date_times[step], hot_streets_latlon) 
                for step in range(len(date_times))]
        
        print(len(images))

        
        # create a directory if does not already exist
        directory = "./ABM_animation_gif/{}_agents".format(len(self.agents_schedule.agents))
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save gif in said directory
        imageio.mimsave(directory+'/{}_steps.gif'.format(len(date_times)), images, duration = .5)
       


    def evaluate_model(self, fail_threshold = 15):
        """ This function calculates and displays a set of metrics to evaluate 
        the performance of the simulated configuration. 
        Returns: 
            - df: a dataframe with real versus simulated dispatch time and travel time for each incident in the time period
            - sum_deterrence: the total deterrence score across agents throughout the simulation
        """
        
        # Note: incident has to be sorted to be evaluated
        print('Statuses of incidents', [inc.status for inc in self.incidents])
        
        print("Time period: {} - {}".format(self.start_datetime, self.start_datetime + dt.timedelta(minutes = self.num_steps*self.step_time)))
        
        print("Total number of incidents for period: {}".format(len(self.incidents)))
        
        ## Get the incidents (that have occured) for evaluation 
        all_inc = [inc for inc in self.incidents if inc.status > 0]

    
        sorted_inc = [inc for inc in self.incidents if inc.status >=3]
        print("Number of sorted incidents: {}/{}".format(len(sorted_inc), len(all_inc)))
        

        dispatch_times = [inc.dispatch_time for inc in all_inc]
        travel_times = [inc.travel_time if inc.status >= 2 else None for inc in all_inc ]

        avg_travel_time = np.mean([inc.travel_time for inc in all_inc if inc.status >= 2  ])
        print("Average travel time per incident: {} minutes".format(round(avg_travel_time)))
    
        ## Distance travelled (in meters)
        #travel_dists = [agent.distance for agent in self.schedule.agents]
        #avg_travel_dist = np.mean(travel_dists)
        #print("Average travel distance per vehicle: {} meters".format(int(avg_travel_dist)))
        
        ## Risk prevented (for now I won't calculate that. The added value of P is in being near responses.)
        #sums_risk = [agent.sum_risk for agent in self.schedule.agents]
        #avg_total_sum_risk_per_agent = np.mean(sums_risk)
        #print("Average risk prevented per agent during shift: {}".format(round(avg_total_sum_risk_per_agent,2)))
        
        #list_metrics = [len(failed_sorted_inc), dispatch_times, travel_times]

        indexes = [inc.id for inc in all_inc]
        #lats, lons = [inc.lat for inc in sorted_inc], [inc.lon for inc in sorted_inc]
        precincts = [inc.precinct for inc in all_inc]
        patrol_beats = [inc.patrol_beat for inc in all_inc]

        call_datetimes = [inc.call_datetime for inc in all_inc]
        real_dispatch_times = [inc.real_dispatch_time for inc in all_inc]
        real_travel_times = [inc.real_travel_time for inc in all_inc]
        #real_response_times = [x+y for x, y in zip(real_dispatch_times, real_travel_times)]

        data_tuples = list(zip(indexes, precincts, patrol_beats, call_datetimes, 
                               real_dispatch_times, real_travel_times, 
                               dispatch_times, travel_times))
        
        df = pd.DataFrame(data_tuples, columns=['Index', 'Precinct',  'Patrol_beat', 'Date_Time', 'Real_dispatch_time', 
                                                        'Real_travel_time', 'Dispatch_time', 'Travel_time'])
        


        ## Average dispatch time per incident
        #print('len(dispatch_times)', len(dispatch_times))
        avg_dispatch_time = df['Dispatch_time'].mean()
        print("Average dispatch time per incident: {} minutes".format(round(avg_dispatch_time)))
        
        ## Average travel time per incident
        #print('len(travel_times)', len(travel_times))
        #avg_travel_time = df[~df['Travel_time'].isna() ['Travel_time'].mean()
        #print("Average travel time per incident: {} minutes".format(round(avg_travel_time)))
    
        ## Average response time per incident (replace NAN in travel time by zero otherwise response time will be NA)
        response_times = df['Dispatch_time'] + df['Travel_time'].fillna(0)
        #print('len(response_times)', len(response_times))
        avg_response_time = response_times.mean()
        print("Average response time per incident: {} minutes".format(round(avg_response_time)))

        ## Get the number of failed responses
        failed_sorted_inc= response_times[response_times> fail_threshold]
        print("Number of failed responses: {}".format(len(failed_sorted_inc)))

        # sum the deterrence attribute of all agents
        #print('list of deterrence', [agent.deterrence for agent in self.agents_schedule.agents])
        sum_deterrence =  round(sum([agent.deterrence for agent in self.agents_schedule.agents]))
        print('sum_deterrence', sum_deterrence)
        #outputData.to_csv('Simulation_Output_Detroit.csv', index = False)

        return df, sum_deterrence #list_metrics[:num_objectives]
    

    def validate_model(self):
        """ This function is ran to validate the model. Get the average of all real and simulated differences between real and simulated dispatch times across incidents """
        sorted_inc = [inc for inc in self.incidents if inc.sorted == 1]
        evaluable_inc = [inc for inc in sorted_inc if ((not np.isnan(inc.real_travel_time)) and (not np.isnan(inc.real_dispatch_time)))]
        
        # REAL
        real_dispatch_times = [inc.real_dispatch_time for inc in evaluable_inc]
        real_travel_times = [inc.real_travel_time for inc in evaluable_inc]
        real_response_times = [x+y for x, y in zip(real_dispatch_times, real_travel_times)]
        
        # SIMULATED
        dispatch_times = [inc.dispatch_time.seconds / 60 for inc in evaluable_inc]
        travel_times = [inc.travel_time.seconds / 60 for inc in evaluable_inc]
        response_times = [x+y for x, y in zip(dispatch_times, travel_times)]
        
        
        # DIFFERENCE BETWEEN REAL AND SIMULATED RESPONSE TIME
        diff_res_time = np.subtract(real_response_times, response_times)
        avg_diff_res_time = np.mean(np.round(diff_res_time))
        return avg_diff_res_time
        
        

    def evaluate_model_vs_reality(self):
        """ Unused. This function compares real and simulated dispatch time and driving 
        time. This is to evaluate the ABM against reality.
        It generates a CSV file to later use for visualisation."""
        
        print("------------- EVALUATION AGAINST REAL TIMES --------------------")
        
        ## Get the sorted incidents for evaluation
        sorted_inc = [inc for inc in self.incidents if inc.status == 4]
        print("Number of sorted incidents: {}".format(len(sorted_inc)))
        
        ## Compare simulated reponse time with real (if evaluable incident)
        #'response time' column is the sum of Intake, Dispatch and Travel times
        #evaluable_inc = [inc for inc in sorted_inc if inc.evaluable == 1]
        evaluable_inc = [inc for inc in sorted_inc if ((not np.isnan(inc.real_travel_time)) and (not np.isnan(inc.real_dispatch_time)))]
        print("Number of sorted incidents that have a travel time: {}".format(len(evaluable_inc)))
        
        # REAL
        real_dispatch_times = [inc.real_dispatch_time for inc in evaluable_inc]
        real_travel_times = [inc.real_travel_time for inc in evaluable_inc]
        real_response_times = [x+y for x, y in zip(real_dispatch_times, real_travel_times)]
        
        # SIMULATED
        dispatch_times = [inc.dispatch_time.seconds / 60 for inc in evaluable_inc]
        travel_times = [inc.travel_time.seconds / 60 for inc in evaluable_inc]
        response_times = [x+y for x, y in zip(dispatch_times, travel_times)]
        
        
        # DIFFERENCE BETWEEN REAL AND SIMULATED RESPONSE TIME
        diff_res_time = np.subtract(real_response_times, response_times)
        avg_diff_res_time = np.mean(np.round(diff_res_time))
        if avg_diff_res_time < 0:
            print("On average, real response time was {} minutes faster".format(abs(round(avg_diff_res_time))))
        else:
            print("On average, real response time was {} minutes slower".format(round(avg_diff_res_time)))
        
        diff_dis_time = np.subtract(real_dispatch_times, dispatch_times)
        avg_diff_dis_time = np.mean(np.round(diff_dis_time))
        if avg_diff_dis_time < 0:
            print("On average, real dispatch time was {} minutes faster".format(abs(round(avg_diff_dis_time))))
        else:
            print("On average, real dispatch time was {} minutes slower".format(round(avg_diff_dis_time)))


        diff_trav_time = np.subtract(real_travel_times, travel_times)
        avg_diff_trav_time = np.mean(np.round(diff_trav_time))
        if avg_diff_trav_time < 0:
            print("On average, real travel time was {} minutes faster".format(abs(round(avg_diff_trav_time))))
        else:
            print("On average, real travel time was {} minutes slower".format(round(avg_diff_trav_time)))
        
        
        ### generate csv output data for analysis and viz
        indexes = [inc.id for inc in evaluable_inc]
        lats, lons = [inc.lat for inc in evaluable_inc], [inc.lon for inc in evaluable_inc]
        nhoods, precincts = [inc.nhood for inc in evaluable_inc], [inc.precinct for inc in evaluable_inc]
        call_datetimes = [inc.call_datetime for inc in evaluable_inc]
        
        data_tuples = list(zip(indexes, lats, lons, nhoods, precincts, call_datetimes, 
                               real_dispatch_times, real_travel_times, 
                               dispatch_times, travel_times))
        outputData = pd.DataFrame(data_tuples, columns=['Index', 'Lat','Lon', 'Nhood', 'Precinct',  'Date_Time', 'Real_dispatch_time', 
                                                        'Real_travel_time', 'Dispatch_time', 'Travel_time'])
        
    
        outputData.to_csv('Simulation_Output_Detroit.csv', index = False)



