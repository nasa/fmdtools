# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:30:59 2025

@author: cwang29

a_star()          ← ENTRY POINT
  └─► nx_graph_gen()      – builds a weighted NetworkX graph
        └─► get_edge_weights()  – bulk-fills per-edge weights
              └─► get_edge_weight()  – fills weights from one source node
                    ├─► get_fuel_cost()
                    └─► get_grid_cost()
        └─► adj_list_gen()      – neighbor map via recursive_neighbor_gen()
  └─► nx.astar_path()     – finds shortest (risk-aware) path

"""
# risk aware path planning algorithm
"""
figure out: all features, states must be tuples. change in methods.
"""
from fmdtools.define.object.coords import Coords, CoordsParam
import networkx as nx
import math


class DroneFlightGridParam(CoordsParam):
    x_size: int = 48
    y_size: int = 48
    blocksize: float = 2.5
    
    """
    weighing suboptimality of each region
    """
    disallowed_cost: float = 10.0
    occupied_cost: float = 20.0
    restricted_cost: float = 1000.0
    
    """
    start, end points
    """
    point_start: tuple = (10.0, 10.0)
    point_end: tuple = (100.0, 100.0)
    
    """
    every grid point assigned 
    """
    state_grid_costs: tuple = (float, 0.0)
    state_fuel_costs: tuple = (dict, None)
    state_total_costs: tuple = (dict, None)
    state_edge_weights: tuple = (dict, None)
    
class DroneFlightGrid(Coords):
    container_p = DroneFlightGridParam
    
    def init_properties(self, **kwargs):
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                
                self.fuel_costs[i, j]   = {}
                self.total_costs[i, j]  = {}
                self.edge_weights[i, j] = {}
    def get_edge_weights(self, env_coords, fuel_rate, disallowed_cost, occupied_cost, restricted_cost):
        """
        calls get_edge_weight to assign all edge weights into array
        """
        self.get_grid_costs(env_coords, disallowed_cost, occupied_cost, restricted_cost)
        self.get_fuel_costs(fuel_rate)
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                self.get_edge_weight(env_coords, j, i, fuel_rate, disallowed_cost, occupied_cost, restricted_cost)
                
    def get_edge_weight(self, env_coords, curr_j, curr_i, fuel_rate, disallowed_cost, occupied_cost, restricted_cost):
        """
        gets total cost, edge weight <- total cost/world coord dist. edge weight to be used in A*
        Optimization [for future]: only assign relevant closer ones?
        """
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                fuel_cost = self.fuel_costs[curr_i, curr_j][(j, i)]
                grid_cost = self.grid_costs[i, j]
                total_cost = fuel_cost + grid_cost
                self.total_costs[curr_i,curr_j][(j, i)] = total_cost
                x = j * self.p.blocksize + self.p.blocksize/2
                y = i * self.p.blocksize + self.p.blocksize/2
                curr_x = curr_j * self.p.blocksize + self.p.blocksize / 2
                curr_y = curr_i * self.p.blocksize + self.p.blocksize / 2
                dist = math.hypot(curr_x - x, curr_y - y)
                edge_weight = total_cost / dist if dist != 0 else float('inf')
                self.edge_weights[curr_i, curr_j][(j, i)] = edge_weight
                
    def get_fuel_costs(self, fuel_rate):
        """
        calls get_fuel_cost in order to assign all fuel costs into array. 
        Optimization [for future]: only assign relevant closer ones?
        """
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                self.get_fuel_cost(j, i, fuel_rate)
                
    def get_fuel_cost(self, curr_j, curr_i, fuel_rate):
        """
        calculate cost due to distance traveled per timestep.
        """
        curr_x = curr_j * self.p.blocksize + self.p.blocksize / 2
        curr_y = curr_i * self.p.blocksize + self.p.blocksize / 2
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                x = j * self.p.blocksize + self.p.blocksize/2
                y = i * self.p.blocksize + self.p.blocksize/2
                fuel_cost = fuel_rate * math.hypot(curr_x - x, curr_y - y)
                self.fuel_costs[curr_i, curr_j][(j, i)] = fuel_cost
                
    def get_grid_costs(self, env_coords, disallowed_cost, occupied_cost, restricted_cost):
        """
        Calculate the GRID COST somewhere in the finer (than environment) drone traversal grid.
        Grid cost + traversal cost = total cost per timestep. defines A* grid weights.
        Heuristic = dist(place, goal.)
        maybe there will be more or less time per timestep? dependent on distance traveled. this seems like a problem.
        env_coords: hurricanecoords grid.
        disallowed_cost, occupied_cost, restricted_cost: how unsavory it is to fly above those areas.
        x, y: grid position.
        env_i, env_j: HurricaneCoords grid indices.
        i
        IMPLEMENT GAUSSIAN LATER!!! DO weighted average: 
            current tile: 0.4
            adj. tiles: 0.1
            diagonal tiles: 0.05
        """
        
        
        """
        review [i][j] vs [i, j]
        """
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                x = j * self.p.blocksize + self.p.blocksize/2
                y = i * self.p.blocksize + self.p.blocksize/2
                env_i = int(x // env_coords.p.blocksize)
                env_j = int(y // env_coords.p.blocksize)
                occupied = env_coords.features["occupied"][env_i, env_j]
                disallowed = env_coords.features["disallowed"][env_i, env_j]
                restricted = env_coords.features["restricted"][env_i, env_j]
                goal_x = env_coords.p.point_end[0]
                goal_y = env_coords.p.point_end[1]
                dx, dy = goal_x - x, goal_y - y
                dist = math.hypot(dx, dy)
                grid_cost = dist * (disallowed_cost * disallowed + occupied_cost * occupied + restricted_cost * restricted)
                self.grid_costs[i, j] = grid_cost
                
    def adj_list_gen(self, max_distance):
        """
        Takes in maximum any-direction grid jump length. 
        returns dictionary adjacency list. keys: vertices. Values: tuple of adjacent tuples.
        """
        adj_dict = {}
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                vertex = (j, i)
                visited = []
                x = j * self.p.blocksize + self.p.blocksize / 2
                y = i * self.p.blocksize + self.p.blocksize / 2
                adj_list = self.recursive_neighbor_gen(x, y, max_distance, visited)
                adj_dict[vertex] = list(adj_list)
        return adj_dict
    
    def recursive_neighbor_gen(self, x, y, dist_remaining, visited = None):
        """
        takes in coordinates, returns tuple of world coordinate tuples
        """
        if visited is None:
            visited = set()
        if (x, y) in visited or dist_remaining < 0:
            return set()
        visited.add((x, y))
        neighbors = {(x, y)}
        for x2, y2 in self.get_neighbors(x, y):
            neighbors |= self.recursive_neighbor_gen(x2, y2, dist_remaining - 1, visited)
        return neighbors
        
    def nx_graph_gen(self, env_coords, max_distance, disallowed_cost, occupied_cost, restricted_cost, fuel_rate):
        """
        Uses path-generation methods in order to create NetworkX graph
        NetworkX: open source graph gen library.
        """
        flight_grid = nx.DiGraph()
        self.get_edge_weights(env_coords, fuel_rate = fuel_rate, disallowed_cost = disallowed_cost, occupied_cost = occupied_cost, restricted_cost = restricted_cost)
        adj_dict = self.adj_list_gen(max_distance = max_distance)
        for vertex, neighbors in adj_dict.items():
            flight_grid.add_node(vertex)
            for neighbor in neighbors:
                weight = self.edge_weights[vertex[1], vertex[0]][neighbor]
                flight_grid.add_edge(vertex, neighbor, weight = weight)
        return flight_grid
        
    def a_star(self, env_coords, start, goal, max_distance = 3, disallowed_cost = 10.0, occupied_cost = 20.0, restricted_cost = 1000.0, fuel_rate = 2.0):
        """
        returns a tuple of tuples which the aircraft should fly through, beginning at the start and finishing at the end.
        Ex.: ((0, 0), (2, 2), (2, 5), (4, 6), (7, 6), (9, 7), (10, 7), (10, 10))
        """
        G = self.nx_graph_gen(env_coords, max_distance = max_distance, disallowed_cost = disallowed_cost, occupied_cost = occupied_cost, restricted_cost = restricted_cost, fuel_rate = fuel_rate)
        heuristic = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        path = nx.astar_path(G, start, goal, heuristic = heuristic, weight = "weight")
        return tuple(path)