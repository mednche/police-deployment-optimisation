
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:55:08 2019

@author: mednche

This file contains the class Dispatcher which acts like a command and control room

"""

import AgentFramework
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

class Dispatcher():

    def __init__(self, model, time_saving_percent = 20):
        """ time_saving_percent: for Sensitivity Analaysis: test impact on inc travel times when the dispatcher
        only redispatch on the fly if new agent travel time saves time_saving_percent compared to existing one
        (10%, 20%, 30%) Default is 20%
        """
        # The queue of incidents filled in as incidents come in
        self.incident_queue = [] # rename to queue

        self.model = model

        
        # List of available agents
        self.avail_agents = [agent for agent in self.model.agents_schedule.agents if agent.status == 'patrol']

        self.time_saving_percent = time_saving_percent

        """ self.schedule_avail_agents = SchedulerFramework.RandomActivation(self.model)
        
        for agent in self.model.schedule.agents:
            if agent.status == 'idle':
                self.schedule_avail_agents.add(agent) """



    def AddIncidentsToQueue(self, incident_interval):
        """ Add the list of new incidents to the queue of unattended incidents
        Input: list of incident objects that for the model time step"""
        self.incident_queue += incident_interval

        for incident in incident_interval:
            # add 1 to the status of the incident, i.e. mark as 1: occured but unalloated
            incident.updateStatus()


    def removeIncidentFromQueue(self, incident):
        self.incident_queue.remove(incident)
        print('incident {} removed from queue'.format(incident.id))
        print('new incident queue: ', [inc.id for inc in self.incident_queue])


    def update_avail_agents(self):
        """ Function adds newly available model agents to the list of available agents
        """
        #print('>>> updating available agents')
        for agent in self.model.agents_schedule.agents:
            if (agent.status == 'patrol' and (agent not in self.avail_agents)):
                self.avail_agents.append(agent)

        #random.shuffle(self.avail_agents) # already randomly activated agents
        print('avail_agents: ', [agent.unique_id for agent in self.avail_agents])



    def addAgentToListAvailAgents(self, agent):
        """ Function adds one agent agent to the list of available agents. 
        Then reshuffle it so new agent is not always placed at the end
        This function is called when reallocation on the fly occurs and an agent is dismissed, 
        becoming available for subsequent incidents
        """

        print('Adding agent {} to list available agents'.format(agent.unique_id))
        self.avail_agents.append(agent)

        #random.shuffle(self.avail_agents) # already randomly activated agents
        print('avail_agents: ', [agent.unique_id for agent in self.avail_agents])


    def dismissPreviousAgent(self, incident):
        """ This function replaces the agent currently dispatched to the incident 
        by another agent. This function is called when a new agent is found 
        closer to a non-sorted incident and replaces the agent already on the case.
        Dismissing an agent consists in:
        
        * Inputs:
            - object 'incident'
        """
        
        # get previous agent targeting incident
        previous_agent = incident.agent

        # mark previous agent as available
        # but won't be available for this step because already listed available agents
        previous_agent.updateStatus('patrol')

        print('Dismissing agent {} '.format(previous_agent.unique_id))
       
        # Add the dismissed agent to the list of available agents for this step
        # Otherwise it would wait until the next step to be available
        self.addAgentToListAvailAgents(previous_agent)

        #print("DISMISSING old agent {}".format(previous_agent.unique_id))


    def find_nearest_avail_agent(self, incident, list_agents_in_precinct):
        """Find nearest available agent for a given incident
        NB: This function is called at every model step for all unattended incidents in the queue
        * Input: 
            - an object 'incident'
            - the precinct in which the incident took place
            - list of available agents (ojects)
        * Returns: 
            - chosen agent (if a new available agent is found to be closer)
            - None (if there is already a closer agent targetting this incident)
        """
        
        #by default the first agent in the list is chosen for the task
        # NB: the first agent is different every step because we use a scheduler
        """chosen_agent = list_agents_in_precinct[0]  
        shortest_route = self.model.graph.findShortestRoute(chosen_agent.pos, incident.node, weight='travel_time_mins')
        # get the estimated drive time for that route
        drive_time_list = ox.utils_graph.get_route_edge_attributes(self.model.graph.G, shortest_route, 
                                                    attribute = 'travel_time_mins', minimize_key='travel_time_mins')
        min_drive_time = sum(drive_time_list)
        #nx.shortest_path_length(self.graph.G, chosen_agent.pos, incident.loc, weight='travel_time_mins')
        #print('min_route:{}'.format(min_route_len))
        # iterate through the other agents
        for agent in self.schedule_avail_agents.agent_buffer(shuffled=True):
            
            
            if drive_time < min_drive_time:
                #print('agent {} closer than default candidate agent {}'. format(agent.unique_id, chosen_agent.unique_id))
                min_drive_time = drive_time
                #print('Drive time: {}'.format(min_drive_time))
                chosen_agent = agent"""

        # Find out the shortest route to incident from all avail agents in precinct
        list_shortest_routes = [self.model.graph.findBestRoute(agent.pos, incident.node, weight='travel_time_mins') for agent in list_agents_in_precinct]
        # Calculated estimated drive time for the route
        list_drive_times = [self.model.graph.getDriveTime(shortest_route) for shortest_route in list_shortest_routes]
    
        #list_drive_times = [self.model.graph.getRouteAndDriveTime(agent.pos, incident.node) for agent in list_agents_in_precinct]
        # Get the min drive time
        min_drive_time = min(list_drive_times)
        # get the index of the min drive time
        index = list_drive_times.index(min_drive_time)
        # Get the agent corresponding to the min drive time
        chosen_agent = list_agents_in_precinct[index]
        route_chosen_agent = list_shortest_routes[index]
        #print("list_drive_times: ", list_drive_times)
        #print("min_drive_time: ", min_drive_time)
        #print("chosen_agent: ", chosen_agent.unique_id)


        #print('Candidate agent: {} for incident {}'.format(chosen_agent.unique_id, incident.id))
        
        # if the incident was already targeted by an agent (non available)
        if incident.agent :
            print('Agent {} was already on this job'.format(incident.agent.unique_id))
            print('Comparing both routes...')
            
            #get current agent's route
            current_agent_shortest_route = incident.agent.route
            # get the estimated drive time for that route
            current_agent_drive_time = self.model.graph.getDriveTime(current_agent_shortest_route)
            
            #print("Current agent's drive time: {}".format(current_agent_drive_time))
            # Only reallocate if new agent's drive time is saving more than time_saving_percent of exisiting drive time
            if min_drive_time < ((100-self.time_saving_percent)/100)*current_agent_drive_time :
                print('New agent {} is closer by {} mins! CHOSEN AGENT'.format(chosen_agent.unique_id, current_agent_drive_time - min_drive_time))
                
                # Dismiss previous agent
                self.dismissPreviousAgent(incident)
                
            else:
                print('Sticking with current agent')
                chosen_agent = None
        
        
        return [chosen_agent, route_chosen_agent]
                
      
              
    def distribute_incidents(self):
        """Distribute unattended incidents in the queue to available agents.
        NB: If an incident was already the target of agent A but agent B becomes 
        available closer to the incident, agent B replaces agent A on this duty."""
        
        
        
        # print('{} available agents: {}'.format(len(self.avail_agents), 
        #       [agent.unique_id for agent in self.avail_agents]))
        
        # Get list of unattended incidents in the queue
        print('{} unattended incidents in the queue: {}'.format(len(self.incident_queue), 
                  [inc.id for inc in self.incident_queue]))
        
        # for all unattended incidents in the queue
        for incident in self.incident_queue:
            
            
            # find out which precinct the incident is in:
            inc_precinct = incident.precinct
            
            print('Incident {} took place in precinct: {}'.format(incident.id, inc_precinct))
            
            # get the available agents in precinct
            list_agents_in_precinct = [agent for agent in self.avail_agents if agent.precinct == inc_precinct]
            #print('Available agents in this precinct: {}'.format([agent.unique_id for agent in list_agents_in_precinct]))
            
            # If there are NO MORE available agents
            #if not list_agents_in_precinct :
            #print('No more available agents in precinct {}'.format(inc_precinct))
                
            # If some available agents left
            if list_agents_in_precinct :
                print('some agents are available in precinct')
                
                # find the nearest available agent 
                chosen_agent, route_chosen_agent = self.find_nearest_avail_agent(incident, list_agents_in_precinct)
                #print('chosen_agent', chosen_agent)
                #print('route_chosen_agent', route_chosen_agent)
                # Send chosen agent on duty
                # NB: chosen_agent can be None if no available agent was closer 
                # than existing agent. This saves having to dispatch the old agent again.
                if chosen_agent:
                    #print('1. chosen agent {}: route: {} '.format(chosen_agent.unique_id, chosen_agent.route))
                    chosen_agent.dispatchToIncident(incident, route_chosen_agent)
 
                    # if the incident has not been assigned yet (first time)
                    if (incident.status == 1):
                        # update the status of the incident to 2: allocated but unattended
                        incident.markAsAssigned(self.model.time)
                    
                    # remove the agent from the list of available agents
                    self.avail_agents.remove(chosen_agent)
                    
                    #print([agent.unique_id for agent in self.avail_agents])
                    
                    
        
        #print('No more unsorted incident')
        
    


    def step(self) :
        """The method makes the dispatcher perform one step: 
        (1) update the list avail agents with newly available ones
        (2) distribute the unattended incidents in the queue"""

        if (len(self.incident_queue) > 0) :
            # Get list of available agents. No need otherwise as no incident to distribute
            print('======> updating avail agents')
            self.update_avail_agents()   

            print('======> distributing incidents')
            # Distribute incidents to available agents
            self.distribute_incidents()
        else :
            print('NO INCIDENTS TO DISTRIBUTE!')


        