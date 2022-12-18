# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:27:53 2019

@author: mednche
"""
from mesa import Agent

import random


class Agent(Agent):
    """An agent"""
    
    def __init__(self, unique_id, model):
        self.driving_time = 0
        self.model = model
        self.unique_id = unique_id
        self.route = []
        #self.avail = 1 # available to respond to incidents
        self.idle = 1 # no need to run move() function (@station, random walk or stationary)
        self.idle_strategy = 'S' #(S = stationary, R = return to station, W = walk randomly)
        self.at_scene = 0
        #self.at_station = 1
        self.target = None
        self.distance = 0
        
        
        #############      Place agent on graph    ##############
        # Choose one police station of the model for the agent
        station_node = random.choice(self.model.stations)
        # update the agent's attribute 'station'
        self.assign_station(station_node)
        # place on station node on network
        self.model.graph.place_agent(self, station_node)

        ######################################################
            
    def assign_station(self, station_node) :
        self.station = station_node
        print('Agent {} assigned to police station {}'.format(self.unique_id, self.station))
    
    def update_route(self, target):
        """This method calculate shortest path to a new target node and update 
        the agent's route with the calculated path.
        - Input: a new target node
        - Updates agent's attribute 'route'."""
        
        # new target
        self.target = target
        
        # new route
        self.route= self.model.graph.calculate_shortest_path(self.pos, self.target, 'drive_time')
        
        
        
    def change_availability(self, avail):
        """This method changes the availability of the agent:
        * Input: avail (boolean)
        * Updates the agent's attribute 'avail' """
        
        self.avail = avail
        
        
        # if becoming unavailable
        if avail == 0:
            print('Agent {} no longer available'.format(self.unique_id))
        
        # if becoming available
        else:
            print('Agent {} now available'.format(self.unique_id))
    
    def change_idle(self, idle):
        """This method changes the idle status of the agent:
        * Input: idle (boolean)
        * Updates the agent's attribute 'idle' """
        
        self.idle = idle

    
    def idleStrategy(self):
        """This method :
        * Input:
        *  """
        print("Agent {} IDLE (idle status: {})".format(self.unique_id, self.idle))
        
        # if stationnary
        if self.idle_strategy ==  'S':
            pass
        # if return to station
        elif self.idle_strategy ==  'R':
            # route back to station unless they are chosen for a new task
            self.update_route(self.station)
            pass
        # if random walk
        else:
            pass
    
    
    def dispatch_to_incident(self, incident):
        """This method dispatches the agent to attend an incident
        * Input: incident (object)
        * Call various other methods to
        - update the route of the agent
        - change their availability and idling status
        - udpate the incident's attributes: agent and dispatch time"""
        
        # update the route
        print("Incident {} given to agent {}".format(incident.id, self.unique_id))
        self.update_route(incident.loc)
        # mark as unavailable
        #self.change_availability(0)
        # mark as no longer idle
        self.idle = 0  
        # note the new agent for this incident
        incident.agent = self
        
        # note the dispatch time (time it took to find an available agent)
        incident.dispatch_time = self.model.time - incident.end_call_time
        
        

            
    def move(self):
        """This method moves the agent toward their current target"""
       
        print('>'*2, ' MOVING AGENT {}'.format(self.unique_id), '>'*2)
        
        # get next node on the route that can be reached under ONE minute
        #time = 0
        i=0
        while (self.driving_time < 1) and (i < len(self.route)-1):
            
            # time needed to travel the edge (between node A and B)
            edge_drive_time = self.model.graph.get_driving_time_edge(self.route[i], self.route[i+1])
            
            # total time travelling on route for this step
            self.driving_time += edge_drive_time.iloc[0]
            # update total distance travelling on route for this step
            self.distance += self.model.graph.get_length_edge(self.route[i], self.route[i+1])
            
                
            i += 1
        
        # Remove all visited nodes from route
        if i>0:
            self.route = self.route[i:]
            self.driving_time=0
        else:
            self.driving_time += 1
        
        # move the agent to the end node
        node_id = self.route[0]
        self.model.graph.move_agent(self, node_id)
        
      
          
    def step(self):
        """The method makes the agent perform one model step consisting of a 
        combination of the following actions:
            - Move agent (if they are going to an incident or on their way back)
            NB: idle agents don't have a target (they are either at station or 
            stationary or random walking) so they don't need to move
            - Wait at traffic light
            - Sort incident (if arrived at the scene) and change their availibilty
            - Spend time at the scene
            - etc"""
           
        print("Agent {} currently at node {} - going to node: {}".format(self.unique_id, self.pos, self.target))
        
        # if agent is not idle (NB: agents don't have targets when waiting)
        # And if they are not currently sorting an incident (spending time at the scene)
        # Idle agents don't need to move
        # NB: agents move and start sorting the incident in the same step!
        if not self.idle and not self.at_scene:
            
            self.move()
                
            # if they just reached target (incident)
            if self.target == self.pos:
                
                # find which incident it is the agent is sorting
                for inc in self.model.incidents:
                    if inc.agent == self:
                        sorted_incident = inc
                        
                print('!'*10, ' AGENT {} ARRIVED AT THE SCENE OF INCIDENT {}'.format(self.unique_id, sorted_incident.id), '!'*10)
                # mark agent as being at the scene now
                # so the incident doesn't belong to the unsorted list now
                self.at_scene = 1
                
                # mark incident as sorted
                sorted_incident.markAsSorted(self.model.time)
                
                
            
                
        # If agent is IDLE:        
        elif self.at_scene == 1:
            
            
            ## CHANGE HERE SO THAT DO THIS STEP UNTIL TIME AT SCENE OVER

            # when agent has spent lonmg enough at the scene
            condition = bool(random.getrandbits(1)) ## CHANGE HERE LATER: so far it is random condition
            if condition:
                self.at_scene = 0
            
                # Become idle (won't commence idle strategy until next step)
                self.idle = 1 ## CHANGE HERE AS THEY WILL HAVE REMAIN STILL FOR A CERTAIN TIME BEFORE BECOMING IDLE AGAIN
                
                print('AGENT {} sorted incident - agent now idle (at_scene status: {} ,idle status: {})'.format(
                        self.unique_id, 
                        self.at_scene,
                        self.idle))
                
            # If still sorting the incident print
            ### NB: REMOVE THIS ONCE RUNNING
            else:
                print('Agent {} at the scene'.format(self.unique_id))
            
        # If agent is IDLE:        
        else:
            self.idleStrategy()
            
                    
