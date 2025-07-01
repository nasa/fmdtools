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
    fuel cost per distance
    """
    fuel_rate: float = 2.0
    """
    maximum distance traveled per timestep
    """
    max_distance: int = 3
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
    state_edge_weights: tuple = (dict, None)
    
class DroneFlightGrid(Coords):
    container_p = DroneFlightGridParam

    def init_properties(self, **kwargs):
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                self.fuel_costs[i, j] = {}
                self.edge_weights[i, j] = {}

    def get_edge_weights(self, env_coords, fuel_rate, disallowed_cost,
                     occupied_cost, restricted_cost, max_distance):
        """
        Only initialize edge weights for reachable neighbors,
        defined by both x and y distance not being greater than max_dist..
        """
        self.get_grid_costs(env_coords, disallowed_cost, occupied_cost, restricted_cost)
    
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                curr_x = j * self.p.blocksize + self.p.blocksize / 2
                curr_y = i * self.p.blocksize + self.p.blocksize / 2
                neighbors = self.recursive_neighbor_gen(j, i, max_distance)
                for (j2, i2) in neighbors:
                    x2 = j2 * self.p.blocksize + self.p.blocksize / 2
                    y2 = i2 * self.p.blocksize + self.p.blocksize / 2
                    dist = math.hypot(curr_x - x2, curr_y - y2)
                    fuel_cost = fuel_rate * dist
                    grid_cost = self.grid_costs[i2, j2]
                    total_cost = fuel_cost + grid_cost
                    self.fuel_costs[i, j][(j2, i2)] = fuel_cost
                    self.edge_weights[i, j][(j2, i2)] = total_cost
       
    def get_grid_costs(self, env_coords, disallowed_cost, occupied_cost, 
                       restricted_cost):
        """
        Calculate the GRID COST somewhere in the finer (than environment) 
        drone traversal grid.
        IMPLEMENT GAUSSIAN LATER!!! DO weighted average: 
            current tile: 0.4
            adj. tiles: 0.1
            diagonal tiles: 0.05
        """
        print("Occupied feature type:", type(env_coords.features["occupied"]))
        print("Sample element:", env_coords.features["occupied"])
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
                grid_cost = dist * (disallowed_cost * disallowed +
                                    occupied_cost * occupied +
                                    restricted_cost * restricted)
                self.grid_costs[i, j] = grid_cost
                
    def adj_list_gen(self, max_distance):
        """
        Takes in maximum any-direction grid jump length. 
        returns dictionary adjacency list. keys: vertices. 
        Values: tuple of adjacent tuples.
        """
        adj_dict = {}
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                vertex = (j, i)
                visited = set()
                adj_list = self.recursive_neighbor_gen(j, i, max_distance, visited)
                adj_dict[vertex] = list(adj_list)
        return adj_dict
    
    def recursive_neighbor_gen(self, j, i, dist_remaining, visited = None):
        """
        takes in indices, returns tuple of index tuples
        """
        if visited is None:
            visited = set()
        if (j, i) in visited or dist_remaining < 0:
            return set()
        visited.add((j, i))
        neighbors = {(j, i)}
        x = j * self.p.blocksize + self.p.blocksize / 2
        y = i * self.p.blocksize + self.p.blocksize / 2
        for x2, y2 in self.get_neighbors(x, y):
            j2, i2 = self.to_index(x2, y2)
            neighbors |= self.recursive_neighbor_gen(j2, i2, 
                                                     dist_remaining - 1,
                                                     visited)
        return neighbors
        
    def nx_graph_gen(self, env_coords, max_distance, disallowed_cost,
                     occupied_cost, restricted_cost, fuel_rate):
        """
        Uses path-generation methods in order to create NetworkX graph
        NetworkX: open source graph gen library.
        """
        flight_grid = nx.DiGraph()
        self.get_edge_weights(env_coords, fuel_rate = fuel_rate,
                              disallowed_cost = disallowed_cost,
                              occupied_cost = occupied_cost,
                              restricted_cost = restricted_cost,
                              max_distance = max_distance)
        adj_dict = self.adj_list_gen(max_distance = max_distance)
        for vertex, neighbors in adj_dict.items():
            flight_grid.add_node(vertex)
            for neighbor in neighbors:
                weight = self.edge_weights[vertex[1], vertex[0]][neighbor]
                flight_grid.add_edge(vertex, neighbor, weight = weight)
        return flight_grid
        
    def a_star(self, env_coords, start, goal, max_distance, disallowed_cost, 
               occupied_cost, restricted_cost, fuel_rate):
        """
        returns a tuple of tuples which the aircraft should fly through, 
        beginning at the start and finishing at the end.
        Ex.: ((0, 0), (2, 2), (2, 5), (4, 6), (7, 6), (9, 7), (10, 7), (10, 10))
        this is in grid indices, not world coordinates!
        """
        G = self.nx_graph_gen(env_coords, max_distance = max_distance,
                              disallowed_cost = disallowed_cost,
                              occupied_cost = occupied_cost,
                              restricted_cost = restricted_cost,
                              fuel_rate = fuel_rate)
        heuristic = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        path = nx.astar_path(G, start, goal, heuristic = heuristic, weight = "weight")
        return tuple(path)
    
    def a_star_worldcoords(self, env_coords, start_xy, goal_xy, max_distance = 3,
              disallowed_cost = 10.0, occupied_cost = 20.0,
              restricted_cost = 1000.0, fuel_rate = 2.0):
        """
        Borrows a_star functionality, but turns grid indices into world coordinates.
        This method should be called from the outside.
        """
        start_ij = self.to_index(*start_xy)
        goal_ij = self.to_index(*goal_xy)
        path_ij = self.a_star(env_coords, start_ij, goal_ij,
                              max_distance = max_distance,
                              disallowed_cost = disallowed_cost,
                              occupied_cost = occupied_cost,
                              restricted_cost = restricted_cost,
                              fuel_rate=fuel_rate)
        path_xy = tuple(tuple(self.grid[j, i]) for (j, i) in path_ij)
        return path_xy