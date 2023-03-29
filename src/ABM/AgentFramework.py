# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:27:53 2019

@author: mednche


Improvement:
    - agents should not become idle autiomatically when they rech their target. 
    For instance if their target was a call, once they reach it, they should start sorting it before turing idle.
"""
#from mesa import Agent

import random
from shapely.geometry import Point
import osmnx as ox
import geopandas as gpd
import numpy as np
from numpy import *
import networkx as nx
import datetime as dt
import sys

class Agent():
    """An agent"""
    
    def __init__(self, unique_id, model, strategy, starting_node, precinct, patrol_beat):
        
        """ patrol_beat is an object"""
        
        self.model = model
        self.unique_id = unique_id
        
        #self.avail = 1 # available to respond to incidents

        # place on strating node (in network)
        self.pos = starting_node
        
        self.idle_strategy = strategy # (S = stationary, R = return to station, P for patrol, W = walk randomly)

        self.status = 'patrol'
        self.route = []
        self.total_route_sim = []

        self.time_at_scene = 0
        self.incident = None
        
        self.distance = 0
        self.time_on_node = 0 # time spent on a node waiting to cross a long edge
        
        self.precinct = precinct
        self.assignPatrolBeat(patrol_beat)

        #self.hot_streets_utm = None # used for calculating closest hot edge to current location when patrolling
        #self.streets_to_patrol = self.patrol_beat.streets_to_patrol
        

        self.deterrence = 0 # sum of num_hist_inc (number of incident prevented by patrolling the edge)
        self.steps_patrolling = 0
        

        print('Agent {} starting at node {} in precinct {} and patrol beat {}'.format(
            self.unique_id, self.pos, self.precinct, self.patrol_beat.name))
        

    def updateTargetIncident(self, incident):
        self.incident = incident
        print('agent {} now assigned to incident {}'.format(self.unique_id, self.incident.id))

        
    def assignPatrolBeat(self, patrol_beat) :
        """ patrol_beat is an object"""
        self.patrol_beat = patrol_beat
        

    def updateStatus(self, status):
        """This method changes the status of the agent (idle, dispatched, at_scene)
        * Input: status (patrol, dispatched, at_scene)
        * Updates the agent's attribute status """
        
        print('STATUS UPDATED FOR AGENT {} - Time {} - Status {} -> {}'.format(self.unique_id, self.model.time, self.status, status))
        self.status = status

        # also reset the route
        self.route = []




    def getRouteToStartPatrolRoute(self):
        patrol_route_first_node = self.patrol_beat.patrol_route[0]

        route_to_start_patrol_route = self.model.graph.findBestRoute(start_node = self.pos, 
                                target_node = patrol_route_first_node, weight = 'prob_num_inc_desc')

        # Always remove the first node (current position) from route
        #print('route to start patrol route: ', route_to_start_patrol_route[1:])
        return route_to_start_patrol_route[1:]


    
    def P(self):
        """This function is used for both Pr and Ph the same way"""
        #print('>'*3, 'agent {} patrolling... in beat {}'.format(self.unique_id, self.patrol_beat), '>'*3)
        

        # If agent just became idle OR if they have finished the first round of patrol (all streets visited)
        # If they have arrived at the last node of their patrolling round: route is [last_node]
        #if (self.route == []) or (len(self.route) == 1):
        if (len(self.route) <= 1):
            #print('ROUTE:', self.route)
            #print('>>>>>> route is now empty, need to create a new route through the hot streets<<<<<<')

            ## Re-init the agent route 
            self.route = self.getRouteToStartPatrolRoute() + self.patrol_beat.patrol_route[1:]
            #print('patrol route: ', self.patrol_beat.patrol_route)
            #print('TOTAL new route for agent:',self.route)
            
            self.move()
            #print('moved agent after creating route!')
        
        # If agent is still doing their patrol round of the hot streets
        else:
            try: 
                self.move()
                # increase time patrolling by 1 time step
                self.steps_patrolling += 1

            except:
                print('>>>>> ERROR <<<<<')
                print('agent ID: ', self.unique_id)
                print('pos: ', self.pos)
                print('route', self.route)
                #print('target', self.target)
                sys.exit(1)
            #print('moved agent along their existing route!')

        
    def changeIdleStrategy(self, strat):
        """This method modifies the agent attribute idle_strategy.
        * Input: one of the following options: 'S', 'Pr', 'Ph', 'R'
        *  """
        self.idle_strategy = strat
        print("New idle strategy is {}".format(strat))
        
        
        
    def idleStrategy(self):
        """This method tell the agent which idle strategy to follow:
        * Input:
        *  """
        #print("Agent {} engaged in idle behaviour {}".format(self.unique_id, self.idle_strategy))
        
        # if stationnary
        if self.idle_strategy ==  'S':
            pass

        # if patrolling patrol_beat they were assigned to at initialisation
        elif self.idle_strategy ==  'Ph':

            #print('Agent {} will patrol HOT streets in patrol beat {}'.format(self.unique_id, self.patrol_beat.name))
            #print('Scanning for streets to patrol in beat...')

            self.P() # DO patrolling strategy

        # if patrolling random patrol beat every time idle, not the one allocated at start
        else:
            #print('Current target: {}'.format(self.target))
            
            # if beginning the patrol, scan for nearby edges
            if self.route == []:
                #print('Agent {} will patrol RANDOM streets in patrol beat {}'.format(self.unique_id, self.patrol_beat))
            
                # Get a random patrol beat
                patrol_beat = random.shuffle(self.model.env.patrol_beats)
                print('random patrol beat selected: ', patrol_beat.name)

                # Assign that new patrol beat to the agent
                self.assignPatrolBeat(patrol_beat)

                #print(self.streets_to_patrol)
                
              
            # Patrol these streets
            self.P() 
        
    
    
    def dispatchToIncident(self, incident, route):
        """This method dispatches the agent to attend an incident
        * Input: incident (object)
        * Call various other methods to
        - update the incident attribute of the agent
        - update the route of the agent
        - change their availability and idling status
        - udpate the incident's attributes: agent and dispatch time"""
        

        # update the agent's incident attribute
        self.updateTargetIncident(incident)

        # change status to travel, route is now reset to []
        self.updateStatus('travel')
       
        # update the agent's route
        self.route = route
        print('4. chosen agent {}: route: {} '.format(self.unique_id, self.route))
    
        # note the new agent for this incident
        incident.agent = self

    
        

            
    def move(self):
        """This method moves the agent toward their current target.
        If the time required to travel an edge is longer than the time 
        left within current model step, remember the time they had left to do 
        a fraction of that edge with attribute self.time_on_node """
       
        #print('>'*5, ' MOVING AGENT {}'.format(self.unique_id), '>'*5)
        
        #print('time_on_node: {}'.format(self.time_on_node))
        # get next node on the route that can be reached under ONE minute
        
        i=0 # i is the number of nodes on their route they travel through on this step
        #print('len(self.route): ', len(self.route))
        while (i < len(self.route)-1):
            #print('i: ', i)
            # time needed to travel along the edge (between node i and i+1)
            edge_drive_time = self.model.graph.get_driving_time_edge(self.route[i], self.route[i+1])
            #print("edge drive time: "+ str(edge_drive_time))
            
            # time it will be when reaching next node
            # total travelling time if make it to the next node
            # (excluding time spent on previous steps waiting)
            # NB: either time = 0  or time_on_node = 0, never both >0
            time_travelled_at_next_node = self.time_travelled_on_step + edge_drive_time - self.time_on_node
            #print("time at next node would be: "+ str(time_at_next_node))
            
            # If can't reach the next node within the step time,
            # even with time spent on this node at previous steps
            #if (time_at_next_node - self.time_on_node > self.model.step_time):
            if (time_travelled_at_next_node > self.model.step_time):
                # store the bit of time left to travel on that edge to continue at the next step
                # that bit of time is added to the time_on_node attribute
                self.time_on_node = self.time_on_node + (self.model.step_time - self.time_travelled_on_step)
                # agent should not be allowed any more time for other actions in this step 
                # (remaining time alloweance will be spent waiting on this node)
                self.time_travelled_on_step = 0 
                #print('CANNOT MOVE TO NEXT NODE!')
                #print('Will wait on this node at this step for: ' +str(self.time_on_node))
                break # need to break the while loop here
                
            else:
                #print('Traveled to next node successfully')
                # update time travelled on THIS step so far
                self.time_travelled_on_step = time_travelled_at_next_node
                
                # reset time_on_node
                self.time_on_node = 0
                
                # Add edge length to the total distance covered by this agent
                #self.distance += self.model.graph.get_length_edge(self.route[i], self.route[i+1])
            
                # next node ID to consider
                i += 1

        # If agent is currently patrolling, sum up all risk of edges travelled
        if self.status == 'patrol' :
            # get the sum of deterrence on all the route that the agent has moved along in this time step
            self.deterrence = self.model.graph.getDeterrenceForRoute(self.route[:i+1])
            print('deterrence = ',self.deterrence)
            
            """ list_nodes = self.route[:i+1]
            gdf_edges = self.model.graph.gdf_edges
            # loop through pairs of nodes
            for u, v in zip(list_nodes, list_nodes[1:]):
                # find the corresponding edge
                edge_risk = gdf_edges.loc[(gdf_edges.u == u) & (gdf_edges.v == v), 'risk'].iloc[0]
                self.sum_risk += edge_risk """
            
        # FOR VIEWING TRAIL (VALIDATION)
        # Save the nodes agent has visited this step
        self.total_route_sim += self.route[:i]
        #print('total_route_sim for agent {}'.format(self.unique_id), self.total_route_sim)

        # Remove all visited nodes from route
        self.route = self.route[i:]      
        #print('route', self.route)
        
        
        
        # move the agent to the end node
        node_id = self.route[0]
        self.model.graph.move_agent(self, node_id)
        #print('new position: {}'.format(self.pos))
      
          
    def stayAtScene(self):
        # the first time the agent reaches the scene after travelling,
        # they may have used some of their time allowance (self.time_travelled_on_step)
        # so need to remove that from the time spent on node at the beginning of the agent being at the scene

        
        # while agent has not spent long enough at the scene
        time_allowed_on_node_for_step = (self.model.step_time - self.time_travelled_on_step)

        if self.time_on_node + time_allowed_on_node_for_step < self.incident.resolution_time :
            
            self.time_on_node += time_allowed_on_node_for_step
            
            print('Agent {} has been at the scene for {} of {} mins'.format(self.unique_id, self.time_on_node, self.incident.resolution_time))
        
        
        # when agent has spent long enough at the scene
        else:
            # update how much time the agent used of their step time to finish staying at the scene
            # the remainder of its time will be resuming patrolling
            self.time_travelled_on_step = self.incident.resolution_time  -  self.time_on_node
            # mark incident as resolved
            self.incident.markAsResolved()
            # agent has now no current incident
            self.incident = None
            # agent status becomes idle
            self.updateStatus('patrol')
            # reset time at scene as 0 for next incident
            self.time_on_node = 0 
                
                
                   
                    

    def step(self):
        """The method activate the agents for one step. The action(s) they do is based on their status. 
        The order of the actions is important so that as an agent's status changes, they can 
        start to perform the next action in this time step!"""
        
        # This is the time they have travelled so far on this step (adding the time travel to each nodes along the way)
        # this will be used for the succession of 3 actions within one step
        self.time_travelled_on_step = 0

        #print('agent: ', self.unique_id)
        # if agent is not idle (NB: agents don't have targets when waiting)
        # And if they are not currently sorting an incident (spending time at the scene)
        # Idle agents don't need to move
        if self.status == 'travel':
            print('Agent {} travelling to incident with {} mins of allowance'.format(self.unique_id, self.model.step_time - self.time_travelled_on_step))
            
            #print("Agent {} currently at node {} - going to node: {}".format(self.unique_id, self.pos, self.target))
            
            #print('route before moving at this step:', self.route)

            self.move()

            #print('route after moving at this step:', self.route)
                
            # if they just reached target node (incident node)
            if self.pos == self.incident.node:
                #print('agent {} reached the scene of incident {}'.format(self.unique_id, self.incident.id))
                
                """# find which incident it is the agent is sorting
                for inc in self.model.incidents:
                    if inc.agent == self:
                        sorted_incident = inc"""
                        
                #print('!'*10, ' AGENT {} ARRIVED AT THE SCENE OF INCIDENT {}'.format(self.unique_id, sorted_incident.id), '!'*10)
                #print('Agent needed at scene for {} mins'.format(sorted_incident.real_time_scene))
                # mark agent as being at the scene now
                # so the incident doesn't belong to the unsorted list now
                self.updateStatus('at_scene')
                
                # update incident travel time
                # NB: add 1 min to prevent travel time from being 0 (when model step time is 1 min or so)
                self.incident.markAsBeingTended(self.model.time+dt.timedelta(minutes = 1))
                # tell dispatcher to remove incident from the queue
                self.model.dispatcher.removeIncidentFromQueue(self.incident)
                
                
                
        # If agent is AT THE SCENE:        
        if self.status == 'at_scene':
            #print('Agent {} at the scene with {} mins of allowance'.format(self.unique_id, self.model.step_time - self.time_travelled_on_step))
            self.stayAtScene()
                
     
        # If agent is IDLE:        
        if self.status == 'patrol':
            #print('AGENT IDLE: ENGAGING IN IDLE STRAT')
            #print('Agent {} patrolling with {} mins of allowance'.format(self.unique_id, self.model.step_time - self.time_travelled_on_step))
            self.idleStrategy()
            
                    
