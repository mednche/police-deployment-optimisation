# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:05:20 2019

@author: mednche

Class object NetworkGrid:
    - attributes are: G (the graph istself)
    - a series of methods to edit the graph or calculate shortest path
    
"""

import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np

class Graph:
    """ Network Grid where each node contains zero or more agents. 
    Attributes: 
        - G: NetworkX graph
        - G_proj: UTM NetworkX graph
        - gdf_nodes: geo dataframe
        - gdf_edges: geo dataframe
        - gdf_nodes_proj: geo dataframe UTM,
        - gdf_edges_proj: geo dataframe UTM"""

    def __init__(self, G, G_proj):
        
        self.G = G
        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.G)
        ## DO THIS IN PREPROCESS
        #self.gdf_edges['patrol_beat'] = self.gdf_edges['patrol_beat'].apply(str)
        
        
        # UTM projected graph
        self.G_proj = G_proj
        self.gdf_nodes_proj, self.gdf_edges_proj = ox.graph_to_gdfs(self.G_proj)


# =============================================================================
#         for node_id in self.G.nodes:
#             G.nodes[node_id]['agent'] = list()
# =============================================================================

            
    def place_agent(self, agent, node_id):
        """ Place a agent in a node. """

        #self._place_agent(agent, node_id)
        agent.pos = node_id


# =============================================================================
#     def get_neighbors(self, node_id, include_center=False):
#         """ Get all adjacent nodes """
# 
#         neighbors = list(self.G.neighbors(node_id))
#         if include_center:
#             neighbors.append(node_id)
# 
#         return neighbors
# =============================================================================


    def move_agent(self, agent, node_id):
        """ Move an agent from its current node to a new node. """

        #self._remove_agent(agent, agent.pos)
        #self._place_agent(agent, node_id)
        agent.pos = node_id
    
    
    def findNodeWithPath(self, start_node, target, nodes_list, weight, target_is_problem = False):
        """Functions checks there is a path between nodes in input list and target
        - Input: list of nodes, target_is_problem means we are cannot reach target because it is on a one way road. 
        We have to follow a slighlty different process for that
        - Output: (node_with_shortest_path and corresponding shortest route from node to target  """
        # In case it cannot find a path
        node_with_shortest_path = None
        shortest_route_node_to_target = None

        #print('nodes_list:', nodes_list)
        if target_is_problem == False :
            # Look for node(s) with a path to target
            nodes_with_path = []
            for node in nodes_list :
                    if nx.has_path(self.G, node, target):
                        nodes_with_path.append(node)
                        #print('Found a path to target from node {}'.format(nodes_with_path))

               
            
            # If any path was found at all:
            if len(nodes_with_path) > 0 :
                # Calculate the route to target from first node_with_path in list
                shortest_route_node_to_target = self.calculate_shortest_path(nodes_with_path[0], target, weight)
                node_with_shortest_path = nodes_with_path[0]
                
                # If more than one neigbour had a path to target: the route from node to target will be the shortest one
                if len(nodes_with_path) > 1:
                    print('More than one node to target')
                    len_min_route = len(shortest_route_node_to_target)

                    # Compare with all other nodes except the first (already done)
                    for node_with_path in nodes_with_path[1:] :
                        route_node_to_target = self.calculate_shortest_path(node_with_path, target, weight)

                        # If route is shorter, update min length and shortest_route
                        if len(route_node_to_target) < len_min_route:
                            shortest_route_node_to_target = route_node_to_target
                            len_min_route = len(shortest_route_node_to_target)
                            node_with_shortest_path = node_with_path
        else:
            print('WARNING STARTING NODE IS THE PROBLEM!')
            # Look for node(s) with a path to target
            nodes_with_path = []
            for node in nodes_list :
                # Find a path to the closest node from where a path to target exist
                if nx.has_path(self.G, start_node, node):
                    nodes_with_path.append(node)
                    print('Found a path to target from node {}'.format(nodes_with_path))
            
            # If any path was found at all:
            if len(nodes_with_path) > 0 :
                # Calculate the route to target from first node_with_path in list
                shortest_route_node_to_target = self.calculate_shortest_path(start_node, nodes_with_path[0], weight)
                node_with_shortest_path = nodes_with_path[0]
                
                # If more than one neigbour had a path to target: the route from node to target will be the shortest one
                if len(nodes_with_path) > 1:
                    print('More than one node to target')
                    len_min_route = len(shortest_route_node_to_target)

                    # Compare with all other nodes except the first (already done)
                    for node_with_path in nodes_with_path[1:] :
                        route_node_to_target = self.calculate_shortest_path(start_node, node_with_path, weight)

                        # If route is shorter, update min length and shortest_route
                        if len(route_node_to_target) < len_min_route:
                            shortest_route_node_to_target = route_node_to_target
                            len_min_route = len(shortest_route_node_to_target)
                            node_with_shortest_path = node_with_path

        #print('Node_with_shortest_path found: ', node_with_shortest_path)
        #print('Shortest_route_node_to_target: ', shortest_route_node_to_target) 

        return (node_with_shortest_path, shortest_route_node_to_target)


    def findClosestHotStreetIndex(self, node, remaining_hot_streets):
        """ node can be agent.pos or the u or v node of an edge when calculating the total route in advance"""

        #print('Searching for closest target')
        
        # Need to convert current location to UTM
        current_location_utm = self.gdf_nodes_proj.loc[node].geometry
        # index of closest edge
        index= remaining_hot_streets.distance(current_location_utm).idxmin()
        #print('closest street: ', index)

        return index

    def getDriveTime(self, route):
        """ function that gets the drive time (sum of edge drive time on all edges) for a given route"""
        return sum(ox.utils_graph.get_route_edge_attributes(self.G, route, 
                                                    attribute = 'travel_time_mins', minimize_key='travel_time_mins'))


    """def getRouteAndDriveTime(self, start_node, end_node):
        #Function called by the dispater that calculates the route first and then returns the corresponding drive time
        shortest_route = self.findBestRoute(start_node, end_node, weight='travel_time_mins')
        drive_time = self.getDriveTime(shortest_route)
        
        return drive_time"""



    def getDeterrenceForRoute(self, route):
        deterrence = sum(ox.utils_graph.get_route_edge_attributes(self.G, route, 
                                                    attribute = 'density_hist_inc_desc', minimize_key='density_hist_inc_desc'))

        
        return deterrence


    def findBestRoute(self, start_node, target_node, weight = 'travel_time_mins'):
        """This method calculate shortest path to a new target node and returns it
        - Input: a new target node, the incident object if the function is to find a path to incident.
        If incident is None (default), we are looking for a path to the hot node.
        - Checks whether there is a route (due to one way roads) to the target. 
        If there is no route, create one by creating a new reverse edge.
        - Returns the found shorttest route
        Weight method for shortest path calculation is travel_time_mins, unless otherwise stated.
        It is stated otherwise as 'risk' when patrolling hot edges
        NB: I have to keep the complex code for case 1 and 2 in case there is no possible reverse route 
        (oneway road in one direction near agent pos and oneway road in another near target or the opposite. """

        origin_of_problem = 0

        # Try to find a path from the start_node (current pos of agent or else)
        nodes_list = [start_node]
        node_with_shortest_route, shortest_route = self.findNodeWithPath(start_node, target_node, nodes_list, weight = weight)
        
        
        
        # Memorize the past nodes so we don't have to try them again
        memo_nodes_list = nodes_list
        
        #######################################################
        # While there is no path between starting_node, (or its neibhours, or neighbours' neighbours etc) and target
        # Case 1: the issue is with the start_node
        #######################################################
        counter=0
        while (node_with_shortest_route is None) and (counter < 10) :
            origin_of_problem = 1
            print("@@@@@@@ No path was found between nodes {} and target {}".format(nodes_list, target_node))

            # get new neighbours and replace previous nodes_list
            new_nodes_list = []
            for node in nodes_list:
                new_nodes_list = new_nodes_list + [node for node in np.unique(list(nx.all_neighbors(self.G, node))) if node not in memo_nodes_list]
            
            # update new nodes_list
            nodes_list = new_nodes_list
            # Memorize the past nodes so we don't have to try them again
            memo_nodes_list += new_nodes_list

            # Look for a path between this new list of nodes and target
            node_with_shortest_route, shortest_route = self.findNodeWithPath(start_node, target_node, nodes_list, weight = weight)
            counter +=1
        

        #######################################################
        # Case 2 : the issue was with the location of the target.
        # It changes the location of the incident so not ideal but need to do it sometimes
        #######################################################
        counter=0
        nodes_list = [target_node]
        # Memorize the past nodes so we don't have to try them again
        memo_nodes_list = nodes_list
        while (node_with_shortest_route is None) and (counter < 10):
            origin_of_problem = 2
            print('----------> The issue seems to be with the target')
            print('WARNING: CHANGING LOCATION OF TARGET TO BE ABLE TO REACH IT!')
            
            # get new neighbours and replace previous nodes_list
            new_nodes_list = []
            for node in nodes_list:
                new_nodes_list = new_nodes_list + [node for node in np.unique(list(nx.all_neighbors(self.G, node))) if node not in memo_nodes_list]
            
            # update new nodes_list
            nodes_list = new_nodes_list
            # Memorize the past nodes so we don't have to try them again
            memo_nodes_list += new_nodes_list

            # Look for a path between this new list of nodes and target
            node_with_shortest_route, shortest_route = self.findNodeWithPath(start_node, target_node, nodes_list, weight = weight, target_is_problem= True)
            
            counter +=1
        ########################################################################################
        



        #######################################################
        # Case 3 : the issue was with both agent pos and loc of the target.
        #######################################################
        if (node_with_shortest_route is None) :
            origin_of_problem = 3
            # Take the reverse route 
            reverse_route = self.calculate_shortest_path(target_node, start_node, weight)
            #print(reverse_route)

            self.add_necessary_edges_at_route_ends(reverse_route)
            
            #  Look for a path now that the edges have been added
            node_with_shortest_route, shortest_route = self.findNodeWithPath(start_node, target_node, [start_node], weight)
            #print('node_with_shortest_route: {}, shortest_route: {}'.format(node_with_shortest_route, shortest_route))


        # If there was no problem
        # (path found between agent posprint and target)
        if origin_of_problem == 0 or origin_of_problem == 3:
            total_route_found = shortest_route
        
        # If the problem was the position of the agent
        # (path found is not from current position of the agent)
        elif origin_of_problem == 1 :
            print('Authorizing driving on one way road from agent pos!')
            # Find a route between agent.pos and that node
            # Get the reverse_route: from node to agent.pos
            reverse_route = self.calculate_shortest_path(node_with_shortest_route, start_node, weight)
            #print('Reverse route: node -> pos', reverse_route)
            # Reverse the reverse_route to find a route from agent.pos to node
            route_pos_to_node = reverse_route.copy()
            route_pos_to_node.reverse() # reverse on the spot the list
            #print('Route to node: pos -> node', route_pos_to_node)

            ## Add new edges to the geodataframe (for route_pos_to_node)
            for u, v in zip(route_pos_to_node, route_pos_to_node[1:]):
                # create a reverse edge between nodes
               
                new_row = self.gdf_edges.loc[(self.gdf_edges.u == v) & (self.gdf_edges.v == u)]

                old_u = new_row.u.copy()
                old_v = new_row.v.copy()

                new_row.loc[:, 'v'] = old_u
                new_row.loc[:, 'u'] = old_v

                #append row to the dataframe
                self.gdf_edges = self.gdf_edges.append(new_row, ignore_index=True)
                
                #print('Added an edge between {} and {}'.format(u, v))
            
            # update graph with new edges added
            self.G = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
        

            # Append both routes together to make a route between agent.pos and target
            total_route_found = route_pos_to_node + shortest_route[1:] # start from one to not repeat the same noe twice
        

        # If the problem was the target 
        # (path found is not to all the way to the target)
        else:
            print('Autorizing driving on one way road to the target!')
            # Find a route between that node and target
            # First, get the reverse_route: from target to node
            reverse_route = self.calculate_shortest_path(target_node, node_with_shortest_route, weight)
            
            # Then reverse the reverse_route to find a route from agent.pos to node
            route_node_to_target = reverse_route.copy()
            route_node_to_target.reverse() # reverse on the spot the list
            #print('Route to node: pos -> node', route_pos_to_node)

            ## Add new edges to the geodataframe (for route_pos_to_node)
            for u, v in zip(route_node_to_target, route_node_to_target[1:]):
                # create a reverse edge between nodes
                new_row = self.gdf_edges.loc[(self.gdf_edges.u == v) & (self.gdf_edges.v == u)]

                old_u = new_row.u.copy()
                old_v = new_row.v.copy()

                new_row.loc[:, 'v'] = old_u
                new_row.loc[:, 'u'] = old_v

                #append row to the dataframe
                self.gdf_edges = self.gdf_edges.append(new_row, ignore_index=True)
                
                #print('Added an edge between {} and {}'.format(u, v))
            
            # update graph with new edges added
            self.G = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
        

            # Append both routes together to make a route between agent.pos and target
            total_route_found = shortest_route + route_node_to_target[1:] # start from one to not repeat the same one twice
        
    

        #print('total_route_found', total_route_found)
        return total_route_found
    


    
    def calculate_shortest_path(self, pos, target, weight='travel_time_mins'):
        return nx.shortest_path(self.G, pos, target, weight)
    
    
    
    def get_driving_time_edge(self, a, b):
        """ Looks up the driving time for a particular edge (between a and b) in the G directly (using networkx)
        - Input: a and b are two consecutive nodes on the graph
        - Returns: travel_time value in minutes for that edge """
        
        # I should use an index to look this one up! Much much faster
        
        """return self.gdf_edges[(self.gdf_edges.u == a) & 
                              (self.gdf_edges.v == b)].travel_time_mins"""
        # note: if multiple parrallel road betweeb node a and b, I select the first one 
        # (not ideal, I should check which one has the lowest travel time but speed of running the code matters more)
        return self.G.get_edge_data(a, b)[0]['travel_time_mins']
    
    def add_necessary_edges_at_route_ends(self, route) :
        # Start from the beginning of route

        for u, v in zip(route, route[1:]):
            
            if self.G.get_edge_data(a, b)[0]['oneway'] == True:
                print('oneway between {} and {}'.format(u, v))

                # create a reverse edge between nodes
                new_row = self.gdf_edges.loc[(self.gdf_edges.u == u) & (self.gdf_edges.v == v)]

                old_u = new_row.u.copy()
                old_v = new_row.v.copy()

                new_row.loc[:, 'v'] = old_u
                new_row.loc[:, 'u'] = old_v

                #print(new_row)
        
                #append row to gdf_edges
                self.gdf_edges = self.gdf_edges.append(new_row, ignore_index=True)
            else:
                # Break as soon as route stops being one way
                break
        
        # Start from the end of route
        rev_route = route[::-1] # Reverse the current route
        for u, v in zip(rev_route, rev_route[1:]):

            if self.G.get_edge_data(u, v)[0]['oneway'] == True:
                
                print('oneway between {} and {}'.format(v, u))
                # create a reverse edge between nodes
                new_row = self.gdf_edges.loc[(self.gdf_edges.u == v) & (self.gdf_edges.v == u)]

                old_u = new_row.u.copy()
                old_v = new_row.v.copy()

                new_row.loc[:, 'v'] = old_u
                new_row.loc[:, 'u'] = old_v

                #print(new_row)

                #append row to gdf_edges
                self.gdf_edges = self.gdf_edges.append(new_row, ignore_index=True)
            else:
                # Break as soon as route stops being one way
                break

        self.G = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
        
            

                              
    def get_length_edge(self, a, b):
        """ Looks up the length of a particular edge (between a and b) in G directely (using networkx)
        - Input: a and b are two consecutive nodes on the graph
        - Returns: length value for that edge"""
        return self.G.get_edge_data(a, b)[0]['length']
        
    """def get_edge_between_nodes(self, node1, node2):
        return self.G.get_edge_data(node1, node2)"""


    def get_edges_node(self, node) :
        """Function get the edges that surround that node in the entire network"""
        edges_df = self.gdf_edges[(self.gdf_edges.u == node) | (self.gdf_edges.v == node)]
        #print(edges_df)
        return edges_df


    


    def calculateDensityIncEdges2(self, incidents):
        """For normal case where incidents have an 'edge_index' column indicating which edge they occured on"""

        def densitySingleEdge(row, incidents):
            num_inc = incidents[incidents.street_index == row.name] # row.name is the index
            return num_inc/row['length']

        self.gdf_edges['density_hist_inc'] = self.gdf_edges.apply(densitySingleEdge, incidents= incidents, axis=1)



    def calculateDensityIncEdges(self, incidents):
        """ DETROIT ONLY
        This function updates the column 'density_hist_inc' in gdf_edges in place. 
        The column is the number of incidents that occured on the edge, based on incidents on its two nodes u and v
        
       """

        def calculateNumIncSingleEdge(edge, node, num_inc_node):
            """Function that calculates the density_hist_inc for an edge based on the proportion of the total length of all 
            neighbouring edges the edge holds. 
            """
            edge_length_prop = edge['length']/self.gdf_nodes.loc[node]['total_length_adj_edges']
            prob_density_hist_inc = edge_length_prop*num_inc_node
            return prob_density_hist_inc


        # count num incidents per node (only for nodes with incidents)
        count_inc_series = incidents['Node'].value_counts()
        count_inc_series = count_inc_series.rename('density_hist_inc')
        #print('count_inc_series: ', count_inc_series)
        

        for node, value in count_inc_series.items():
            
            #print('Node: ', node, ' Value:', value)
            
            # get JUST edges for that nodes
            edges_df = self.get_edges_node(node) 
            #print('edges_df', edges_df)
            
            # calculate a new value of density_hist_inc
            new_density_hist_inc = edges_df.apply(calculateNumIncSingleEdge, node = node, num_inc_node= value, axis=1)
            #print(new_density_hist_inc)
            # add this value to existing value in gdf_edges
            edges_df['density_hist_inc'] = edges_df['density_hist_inc'] + new_density_hist_inc

            # Update some edges with their new density_hist_inc value (might change again later for another node processed)
            self.gdf_edges.update(edges_df['density_hist_inc'])
            

  

       
        
        


    def get_streets_with_incident_in_beat(self, beat_num) :

        """ Select the streets with incidents in the patrol beat using the column created in pre-processing
        'patrol_beat' for each edge in gdf_edges"""

        # Get all streets within the patrol beat where density_hist_inc > 0 (those with incidents!)
        streets_with_inc_in_beat = self.gdf_edges[(self.gdf_edges['patrol_beat'] == beat_num) & 
                                                    (self.gdf_edges['density_hist_inc'] > 0)]
        #print(len(streets_with_inc_in_beat), ' streets with incidents in beat ', beat_num)

        return streets_with_inc_in_beat



    def add_density_hist_inc_desc_attribute_edges(self):
        """ create a new column that transform density_hist_inc into (max_num_inc - density_hist_inc) so it can be used
        as a value to minimize when routing using osmnx. 
        Function adds a new column to gdf_edges in place: nothing returned"""
        
        max_num_inc = self.gdf_edges.loc[:,'density_hist_inc'].max()
        self.gdf_edges.loc[:,'density_hist_inc_desc'] = max_num_inc - self.gdf_edges.loc[:,'density_hist_inc']

    def get_streets_within_beat(self, patrol_beat_poly):
        return self.gdf_edges[self.gdf_edges.intersects(patrol_beat_poly)]

    def initAttributeDensityHistInc(self):
        self.gdf_edges.loc[:,'density_hist_inc'] = np.repeat(0, len(self.gdf_edges))

    

    def udpate_G(self):

        self.G = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)


# =============================================================================
#     def _place_agent(self, agent, node_id):
#         """ Place the agent at the correct node. """
# 
#         #self.G.node[node_id]['agent'].append(agent)
# 
#     def _remove_agent(self, agent, node_id):
#         """ Remove an agent from a node. """
# 
#         #self.G.node[node_id]['agent'].remove(agent)
# =============================================================================
