# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:30:59 2025

@author: cwang29

a_star_worldcoords()
  └─► a_star()
        └─► nx_graph_gen()
              └─► get_edge_weights()
                    └─► get_grid_costs()
                    └─► neighbor_gen()
        └─► nx.astar_path()
"""
from fmdtools.define.object.coords import Coords, CoordsParam
import networkx as nx
import math

class DroneFlightGridParam(CoordsParam):
    x_size: int = 120
    y_size: int = 120
    blocksize: float = 1.0
    
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

    def get_edge_weights(self, fuel_rate,
                         disallowed_cost, occupied_cost, restricted_cost,
                         max_distance):
        self.get_grid_costs(disallowed_cost,
                            occupied_cost, restricted_cost)
        for row in range(self.p.y_size):
            for col in range(self.p.x_size):
                cx = col*self.p.blocksize + self.p.blocksize/2
                cy = row*self.p.blocksize + self.p.blocksize/2
                neighbours = self.neighbor_gen(col, row, max_distance)
                for (ncol, nrow) in neighbours:
                    if ncol == col and nrow == row:
                        continue
                    nx = ncol*self.p.blocksize + self.p.blocksize/2
                    ny = nrow*self.p.blocksize + self.p.blocksize/2
                    dist       = math.hypot(nx - cx, ny - cy)
                    fuel_cost  = fuel_rate * dist
                    env_cost   = self.get_properties(nrow, ncol)['grid_costs']
                    total_cost = fuel_cost + dist*env_cost
                    
                    fc = self.get_properties(row, col)['fuel_costs']
                    fc[(ncol, nrow)] = fuel_cost
                    self.set(row, col, 'fuel_costs', fc)

                    ew = self.get_properties(row, col)['edge_weights']
                    ew[(ncol, nrow)] = total_cost
                    self.set(row, col, 'edge_weights', ew)

    def get_grid_costs(self, disallowed_cost, occupied_cost, restricted_cost):
        max_offset = int(round(4 / self.p.blocksize))
        avg_range = range(-max_offset, max_offset + 1)
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                total = 0.0
                for di in avg_range:
                    for dj in avg_range:
                        ci, cj = i + di, j + dj
                        if 0 <= ci < self.p.y_size and 0 <= cj < self.p.x_size:
                            x = cj * self.p.blocksize + self.p.blocksize / 2
                            y = ci * self.p.blocksize + self.p.blocksize / 2
                            disallowed = self.env_coords.get(x, y, 'disallowed', outside=True)
                            occupied = self.env_coords.get(x, y, 'occupied', outside=True)
                            restricted = self.env_coords.get(x, y, 'restricted', outside=True)
                            weight = 0.04
                            total += weight * (
                                disallowed_cost * disallowed +
                                occupied_cost   * occupied +
                                restricted_cost * restricted
                            )
                self.set(i, j, 'grid_costs', total)

    def neighbor_gen(self, j, i, max_distance, visited=None):
        if visited is None:
            visited = set()

        neighbors = set()
        for dj in range(-max_distance, max_distance + 1):
            for di in range(-max_distance, max_distance + 1):
                nj, ni = j + dj, i + di
                if 0 <= nj < self.p.x_size and 0 <= ni < self.p.y_size:
                    dist = math.hypot(dj, di)
                    if dist <= max_distance:
                        neighbors.add((nj, ni))
        return neighbors
    
    def nx_graph_gen(self, max_distance, disallowed_cost,
                     occupied_cost, restricted_cost, fuel_rate):
        flight_grid = nx.DiGraph()
        self.get_edge_weights(fuel_rate,
                              disallowed_cost, occupied_cost,
                              restricted_cost, max_distance)
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                v = (j, i)
                ew = self.get_properties(i, j)['edge_weights']
                for u, w in ew.items():
                    flight_grid.add_edge(v, u, weight=w)
        return flight_grid
        
    def a_star(self, start, goal, max_distance, disallowed_cost, 
               occupied_cost, restricted_cost, fuel_rate):
        # this is in grid indices, not world coordinates!
        G = self.nx_graph_gen(max_distance = max_distance,
                              disallowed_cost = disallowed_cost,
                              occupied_cost = occupied_cost,
                              restricted_cost = restricted_cost,
                              fuel_rate = fuel_rate)
        heuristic = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        if start not in G:
            print(f"Start node {start} not in graph")
        if goal not in G:
            print(f"Goal node {goal} not in graph")
        if not nx.has_path(G, start, goal):
            print(f"No path from {start} to {goal}")
        path = nx.astar_path(G, start, goal, heuristic = heuristic, weight = "weight")
        return tuple(path)
    
    def a_star_worldcoords(self, start_xy, goal_xy, max_distance,
              disallowed_cost, occupied_cost,
              restricted_cost, fuel_rate):
        start_ij = self.to_index(*start_xy)
        goal_ij = self.to_index(*goal_xy)
        path_ij = self.a_star(start_ij, goal_ij,
                              max_distance = max_distance,
                              disallowed_cost = disallowed_cost,
                              occupied_cost = occupied_cost,
                              restricted_cost = restricted_cost,
                              fuel_rate=fuel_rate)
        path_xy = tuple(tuple(self.grid[j, i]) for (j, i) in path_ij)
        return path_xy