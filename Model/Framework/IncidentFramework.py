# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:13:58 2019

@author: mednche
"""
import osmnx as ox

class Incident():
    """An Incident"""
    
    def __init__(self, unique_id, node, precinct, patrol_beat, call_datetime, 
                 real_time_scene, real_dispatch_time = None, real_travel_time= None):
        
       
        
        # STATIC
        self.id = unique_id
        self.node = node # NB: node was found in pre-processing
        self.call_datetime = call_datetime
        self.precinct = precinct
        self.patrol_beat = patrol_beat
        self.real_dispatch_time = real_dispatch_time
        self.real_travel_time = real_travel_time
        self.resolution_time = round(real_time_scene) # round because my agents can never spend less than 1 minute (time step)
    
        # DYNAMIC
        # Simulated response time
        #self.resolved = False
        self.status = 0 # initialise the status as not yet occured (because inicdents are initialised prior to running the model)
        # status values are 0: not yet occured, 1: unallocated, 2: allocated unattended, 3: being tended and 4: sorted
        self.agent = None
        #self.dispatch_time = None
        #self.travel_time = None

        # These times will automatically increase at each step of the model depending on the agent status 
        self.dispatch_time = 1 # for status 1 (always 1 to start with)
        self.travel_time = 1 # for status 2 (always 1 to start with)
        
        # For evaluation against real response time
        #self.evaluable = evaluable
        
    """ def changeIncidentNode(self, node):
        print('Incident node set to: {}'.format(node))
        self.node = node """

    def resetIncidents(self):
        """this function reset the dynamic attributes of the incidents so they can 
        be used for a different configuration without needing to re-run the Env.py (phase 1 of initialisation)"""
        self.status = 0
        self.agent = None
        self.dispatch_time = 1
        self.travel_time = 1


    def updateStatus(self):
        """ incident status goes from 0 (init) to 4 (resolved) incrementially """
        self.status += 1
        print('incident {} status is now {}'.format(self.id, self.status))

    def markAsAssigned(self, time):
        """ An agent has been dispatched"""

        self.updateStatus()
        
        # Update travel time
        #self.updateDispatchTime(time)
        print('dispatch_time:', self.dispatch_time)
        #print('dispatch_time from dispatcher:', self.dispatch_time)

    def markAsBeingTended(self, time):
        """ An agent has reached the scene"""

        self.updateStatus()
        
        # Update travel time
        #self.updateTravelTime(time)
        #print('travel_time from agent:', self.travel_time)
        print('travel_time:', self.travel_time)

  
    def markAsResolved(self):
        """ An agent stayed at the scene for the duration of resolution_time"""
        
        self.updateStatus()
        


    def updateDispatchTime(self, time):
        """dispatch is the the time between time at the call datetime and model current time"""
        
        td = time - self.call_datetime
        self.dispatch_time = td #td.total_seconds() / 60 #(td.seconds//60)%60
        

    def updateTravelTime(self, time):
        """travel time is the the time between time at dispatch and model current time"""
        self.travel_time = time - (self.call_datetime + self.dispatch_time)
        
        
        
         
    def step (self, model):
        # if incident has occured but is still unallocated
        if self.status == 1 :
            print('INCIDENT: ', self.id)
            self.dispatch_time += model.step_time
            print('dispatch_time:', self.dispatch_time)
            #print('dispatch_time from agent:', self.dispatch_time)
        # if incident is allocated but is still unattended
        elif self.status == 2 :
            print('INCIDENT: ', self.id)
            self.travel_time += model.step_time
            print('travel_time:', self.travel_time)
            #print('travel_time from agent:', self.travel_time)
       
