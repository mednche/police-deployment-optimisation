# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:55:08 2019

@author: mednche

This file contains the class Model which conrresponds to the main ABM.

"""

from Framework import AgentFramework3, IncidentFramework, GraphFramework
import imp
imp.reload(GraphFramework)

from mesa import Model
from mesa.time import RandomActivation

import datetime as dt
import pandas as pd
import numpy as np


import networkx as nx
import osmnx as ox



class Model(Model):
    """The model with some number of agents."""
    
    def __init__(self, num_agents, num_iter, step, G, PS_nodes):
        
        
        self.num_agents = num_agents
        self.num_iter = num_iter
        self.incidents = []
        self.schedule = RandomActivation(self) # for MESA scheduler

        self.graph = GraphFramework.Graph(G) # the graph is an object with attribute G
        
        #self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.graph.G)
        self.stations = PS_nodes
        
        self.step = step # IN MINUTES
        
        # Create agents
        for i in range(self.num_agents):
            a = AgentFramework3.Agent(i, self)
            self.schedule.add(a) # MESA scheduler
            
        self.avail_agents = [agent for agent in self.schedule.agents if agent.idle == 1]
            
    def scheduled_step(self):
        """Advance the model by one step."""
        self.schedule.step()

        
            
    def import_day_incidents(self, date, incidents):
        """ A function that selects the a subset of all incidents that occured 
        on specific 'date'.
        * Inputs:
            - date as datetime
            - all incidents as pandas dataframe
        * Returns: dataframe with only incidents for the day (date)"""
        
        # convert to timestamp
        incidents['End_Call_Time']= pd.to_datetime(incidents['End_Call_Time'])
        
        # select incidents for that 'date'
        day_data = incidents[incidents['End_Call_Time'].dt.date == date]
        
        return day_data
    

    def update_avail_agents(self):
         self.avail_agents = [agent for agent in self.schedule.agents if agent.idle == 1]

    
    
    def get_unsorted_incidents(self):
        return [incident for incident in self.incidents if incident.sorted == 0]
    
    
    def find_nearest_avail_agent(self, incident):
        """Find nearest available agent for a given incident
        (they should be either in station or on their way back).
        NB: This function is called at every model step for all non-sorted incidents
        * Input: 
            - an object 'incident'
            - list of available agents (ojects)
        * Returns: 
            - chosen agent (if a new available agent is found to be closer)
            - None (if there is already a closer agent targetting this incident)
        """
    
        #by default the first agent in the list is chosen for the task
        # NB: the first agent is different every step because we use MESA scheduler
        chosen_agent = self.avail_agents[0]  
        min_route_len = nx.shortest_path_length(self.graph.G, chosen_agent.pos, incident.loc, weight='drive_time')
        #print('min_route:{}'.format(min_route_len))
        
        # iterate through the other agents
        for agent in self.avail_agents[1:]:
            route_len = nx.shortest_path_length(self.graph.G, agent.pos, incident.loc, weight='drive_time')
            #print(route_len)
            if route_len < min_route_len:
                min_route_len = route_len
                chosen_agent = agent
        
        print('Candidate agent: {} for incident {}'.format(chosen_agent.unique_id, incident.id))
        
        # if the incident was already targeted by an agent (non available)
        if incident.agent :
            print('Agent {} was already on this job'.format(incident.agent.unique_id))
            print('Comparing both routes...')
            
            # calculate shortest path
            current_agent_len = nx.shortest_path_length(self.graph.G, incident.agent.pos, incident.loc, 
                                                        weight='drive_time')
            
            if min_route_len < current_agent_len :
                print('New agent {} is closer! CHOSEN AGENT'.format(chosen_agent.unique_id))
                
                # Dismiss previous agent
                self.dismiss_previous_agent(incident)
                
            else:
                print('Current agent is closer')
                chosen_agent = None
        
        return chosen_agent
                
      
              
    def distribute_incidents(self):
        """Distribute the incidents to available agents.
        NB: If an incident was already the target of agent A but agent B becomes 
        available closer to the incident, agent B replaces agent A on this duty."""
        
        # Get list of available agents
        self.update_avail_agents()
        
        print('{} available agents: {}'.format(len(self.avail_agents), 
              [agent.unique_id for agent in self.avail_agents]))
        
        # Get list of unsorted incidents
        incidents = self.get_unsorted_incidents()
        print('{} unsorted incidents: {}'.format(len(incidents), 
                  [inc.id for inc in incidents]))
        
        # for all unsorted incidents
        for incident in incidents:
            
            # If there are NO MORE available agents
            if not self.avail_agents :
                print('No more available agents')
                break
            
            # If some available agents left
            else:
                
                # find the nearest available agent 
                chosen_agent = self.find_nearest_avail_agent(incident)
                
                # Send chosen agent on duty
                # NB: chosen_agent can be None if no available agent was closer 
                # than existing agent. This saves having to dispatch the old agent again.
                if chosen_agent:
                    chosen_agent.dispatch_to_incident(incident)
                    # remove the agent from the list of available agents
                    self.avail_agents.remove(chosen_agent)
                    
        
        print('No more unsorted incident')
        
    def dismiss_previous_agent(self, incident):
        """ This function replaces the agent in charge of sorting an incident 
        by another agent. This function is called when a new agent is found 
        closer to a non-sorted incident and replaces the agent already on the case.
        Dismissing an agent consists in:
        - Marking them as available
        - Sending them back to its station until further notice
        
        * Inputs:
            - object 'incident'
        """
        
        # get previous agent targeting incident
        previous_agent = incident.agent

        # mark previous agent as available
        # but won't be available for this step because already listed available agents
        previous_agent.idle = 1 
        
        # Add the dismissed agent to the list of available agents for this step
        # Otherwise it would wait until the next step to be available
        self.avail_agents.append(previous_agent)

        print("DISMISSING old agent {}".format(previous_agent.unique_id))

            
    def run_model(self, date, incidents):
        """ This function runs the ABM
        * Inputs:
            - datetime 'date'
            - dataframe of all incidents
            """
        
        # get daily incidents
        day_data = self.import_day_incidents(date, incidents)
        
        # get start time (midnight)
        date_time = dt.datetime.combine(date, dt.time(0, 0))
        
        # get last date_time of the day
        end_date_time = dt.datetime.combine(date, dt.time(23, 55))
        
        agents_pos_all_steps = []
        targets_pos_all_steps = []
        inc_pos_all_steps = []
        
        step_num = 0
        while (date_time < end_date_time) and (step_num < self.num_iter) :
            
            print('-'*15, 'STEP: {}'.format(step_num), '-'*15)
                 
            # Add 1 time step
            next_date_time = date_time + dt.timedelta(minutes=self.step)
            self.time = next_date_time
            print(self.time)
            
            
            # Get list of new incidents
            incidents_df = day_data[(day_data.End_Call_Time >= date_time) & (day_data.End_Call_Time < next_date_time)]
            print("{} new incidents".format(incidents_df.shape[0]))
    
            # Instanciate incidents and add to list
            # FIFO 
            for index, inc in incidents_df.iterrows():
                self.incidents.append(IncidentFramework.Incident(index, self, inc.lat, inc.lon, inc.Neighborhood,
                                                                 inc['City Council Districts'], inc.End_Call_Time,
                                                                 inc.Priority, inc.Category, inc.Testing_set,
                                                                 inc['Dispatch Time'], inc['Travel Time'],
                                                                 inc['Time On Scene']))
                
                    

            # Distribute incidents to available agents
            self.distribute_incidents()
            
            # Save number of undistributed incidents for printing at the end.
            # Cannot be negative so 0 otherwise
            #num_undistr_inc = len(list_unsorted_incidents)
            #print('{} new incidents not distributed'.format(num_undistr_inc))

            # Model makes one step
            print(" >>>>>>> Moving agents <<<<<<<")
            self.scheduled_step()


            # Agents' locations
            agents_pos = [agent.pos for agent in self.schedule.agents]
            agents_pos_all_steps.append(agents_pos)
            
            # current targets' locations
            targets_pos = [agent.target for agent in self.schedule.agents]
            targets_pos_all_steps.append(targets_pos)
            
            # locations of unsorted incidents
            inc_pos = [inc.loc for inc in self.incidents if inc.sorted == 0]
            inc_pos_all_steps.append(inc_pos)
        
            # Visualise step snapshot
            # visualise(self, agents_pos, targets_pos, inc_pos)
        
        
            # update date_time
            date_time = next_date_time
            step_num+=1
        
        return agents_pos_all_steps, inc_pos_all_steps, targets_pos_all_steps
    
    
    
    
    def visualise(self, agents_pos, targets_pos, inc_pos):
        """ Function to visualise the position of the agents as a snapshot for each step"""

        nc = ['g' if node in self.stations else
          'b' if node in agents_pos else 
          'r' if node in targets_pos else
          '#FFD700' if node in inc_pos else
          'grey' for node in self.graph.G.nodes()]
        
        ns = [15 if node in agents_pos or
              node in inc_pos or
              node in targets_pos or
              node in self.stations else
              1 for node in self.graph.G.nodes()]
                  
        fig, ax = ox.plot_graph(self.graph.G, node_color=nc, node_zorder=3, node_size=ns, fig_height = 10, fig_width = 10)
    
    
    
    
    def evaluate_model(self):
        """ This function calculates and display a set of metrics to evaluate 
        the performace of the simulated strategy"""
        
        # Note: incident has to be sorted to be evaluated
        
        print("Time period: {} 00:00 - {}".format(self.time.date(), self.time.time()))
        
        print("Total number of incidents: {}".format(len(self.incidents)))
        
        ## Get the sorted incidents for evaluation
        sorted_inc = [inc for inc in self.incidents if inc.sorted == 1]
        print("Number of sorted incidents: {}".format(len(sorted_inc)))
        
        ## Average response time per incident
        response_times = [(inc.travel_time.seconds / 60) for inc in sorted_inc]
        avg_response_time = np.mean(np.round(response_times))
        print("Average response time per incident: {} minutes".format(round(avg_response_time)))

        ## Distance travelled (in meters)
        travel_dists = [agent.distance for agent in self.schedule.agents]
        avg_travel_dist = np.mean(travel_dists)
        print("Average travel distance per vehicle: {} meters".format(int(avg_travel_dist)))

        # =============================================================================
        ### Total fuel cost (in dollars)
        #def calculateCost(distance, MilesPerGallon, pricePerGallon):
        #   KmPerGallon = 1.609344 * MilesPerGallon
        #   return (distance / 100) * (1/KmPerGallon) * pricePerGallon
        #cost=0
        #for distance in travel_dists:
        #    cost+= calculateCost(distance, 19, 2.7) 
        #print("Total cost: ${}".format(int(cost)))
        # =============================================================================
        

    def evaluate_model_vs_reality(self):
        """ This function compares real and simulated dispatch time and driving 
        time. This is to evaluate the ABM against reality.
        It generates a CSV file to later use for visualisation."""
        
        print("------------- EVALUATION AGAINST REAL TIMES --------------------")
        
        ## Get the sorted incidents for evaluation
        sorted_inc = [inc for inc in self.incidents if inc.sorted == 1]
        print("Number of sorted incidents: {}".format(len(sorted_inc)))
        
        ## Compare simulated reponse time with real (if evaluable incident)
        evaluable_inc = [inc for inc in sorted_inc if inc.evaluable == 1]
        print("Number of sorted incidents that are evaluable: {}".format(len(evaluable_inc)))
        
        # REAL
        real_dispatch_times = [inc.real_dispatch_time for inc in evaluable_inc]
        print("Real Dispatch Time: {}".format(real_dispatch_times))
        real_travel_times = [inc.real_travel_time for inc in evaluable_inc]
        print("Real Travel Time: {}".format(real_travel_times))
        real_response_times = [x+y for x, y in zip(real_dispatch_times, real_travel_times)]
        
        # SIMULATED
        dispatch_times = [inc.dispatch_time.seconds / 60 for inc in evaluable_inc]
        print("Dispatch Time: {}".format(dispatch_times))
        travel_times = [inc.travel_time.seconds / 60 for inc in evaluable_inc]
        print("Travel Time: {}".format(travel_times))
        response_times = [x+y for x, y in zip(dispatch_times, travel_times)]
        
        print(real_response_times)
        print(response_times)
        
        # DIFFERENCE BETWEEN REAL AND SIMULATED RESPONSE TIME
        diff_res_time = np.subtract(real_response_times, response_times)
        avg_diff_res_time = np.mean(np.round(diff_res_time))
        print("On avergae, real response time was {} minutes slower".format(round(avg_diff_res_time)))
        
        
        ### generate csv output data for analysis and viz
        indexes = [inc.id for inc in evaluable_inc]
        lats, lons = [inc.lat for inc in evaluable_inc], [inc.lon for inc in evaluable_inc]
        nhoods, districts = [inc.nhood for inc in evaluable_inc], [inc.district for inc in evaluable_inc]
        end_call_times = [inc.end_call_time for inc in evaluable_inc]
        
        data_tuples = list(zip(indexes, lats, lons, nhoods, districts, end_call_times, 
                               real_dispatch_times, real_travel_times, 
                               dispatch_times, travel_times))
        outputData = pd.DataFrame(data_tuples, columns=['Index', 'Lat','Lon', 'Nhood', 'District',  'End_call_time', 'Real_dispatch_time', 
                                                        'Real_travel_time', 'Dispatch_time', 'Travel_time'])
        
    
        outputData.to_csv('Simulation_Output_Detroit.csv', index = False)