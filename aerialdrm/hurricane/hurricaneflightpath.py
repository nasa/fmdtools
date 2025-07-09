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
from aerialdrm.hurricane.hurricaneenvironment import HurricaneCoords, HurricaneCoordsParam
import networkx as nx
import math

class DroneFlightGridParam(CoordsParam):
    x_size: int = 96
    y_size: int = 96
    blocksize: float = 1.25
    
    """
    fuel cost per distance
    """
    fuel_rate: float = 2.0
    """
    maximum distance traveled per timestep
    """
    max_distance: int = 5
    """
    weighing suboptimality of each region
    """
    disallowed_cost: float = 10.0
    occupied_cost: float = 20.0
    restricted_cost: float = 1000.0
    
    """
    start, end points
    
    TUNABLE PARAMETERS FOR PRESENTATION, AVERAGING VS GAUSSIAN THING, FUEL COSTS IS WAY TOO LOW (SHOW WHAT HAPPENS WHEN STUFF CHNAGE)
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
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                total = 0.0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ci, cj = i + di, j + dj
                        if 0 <= ci < self.p.y_size and 0 <= cj < self.p.x_size:
                            x = cj * self.p.blocksize + self.p.blocksize / 2
                            y = ci * self.p.blocksize + self.p.blocksize / 2
                            col = int(x // self.env_coords.p.blocksize)
                            row = int(y // self.env_coords.p.blocksize)
                            disallowed = self.env_coords.get(x, y, 'disallowed', outside=True)
                            occupied = self.env_coords.get(x, y, 'occupied', outside=True)
                            restricted = self.env_coords.get(x, y, 'restricted', outside=True)
                            # print(
                            #     f"Grid ({j}, {i}) sampling Env ({col}, {row}) at offset ({dj}, {di}) → "
                            #     f"disallowed={props['disallowed']}, "
                            #     f"occupied={props['occupied']}, "
                            #     f"restricted={props['restricted']}"
                            #     )
                            weight = (
                                0.4 if (di == 0 and dj == 0)
                                else 0.1 if (di == 0 or dj == 0)
                                else 0.05
                            )
                            total = weight * (
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
        """
        Uses path-generation methods in order to create NetworkX graph
        NetworkX: open source graph gen library.
        """
        flight_grid = nx.DiGraph()
        self.get_edge_weights(fuel_rate,
                              disallowed_cost, occupied_cost,
                              restricted_cost, max_distance)
        
        
        print("=== DEBUG: cell costs and edge-weights ===")
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                gc = self.get_properties(i, j)["grid_costs"]
                if gc != 0:
                    print(f"Cell {(j,i)} flat grid_cost = {gc:.1f}")
                fw = self.get_properties(i, j)["fuel_costs"]
                ew = self.get_properties(i, j)["edge_weights"]
                    # sort by total cost so we see the cheapest jumps first
                for nbr, total in sorted(ew.items(), key=lambda x: x[1]):
                    fuel = fw[nbr]
                    # recompute dist so we can print grid_component cleanly
                    dx = (nbr[0] - j)*self.p.blocksize
                    dy = (nbr[1] - i)*self.p.blocksize
                    dist = math.hypot(dx, dy)
                    grid_comp = total - fuel
                    print(
                        f"  {(j,i)}→{nbr}: dist={dist:.2f}, "
                        f"fuel={fuel:.1f}, grid={grid_comp:.1f}, total={total:.1f}"
                    )
        print("=== end debug ===")
      
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                v = (j, i)
                ew = self.get_properties(i, j)['edge_weights']
                for u, w in ew.items():
                    flight_grid.add_edge(v, u, weight=w)
        return flight_grid
        
    def a_star(self, start, goal, max_distance, disallowed_cost, 
               occupied_cost, restricted_cost, fuel_rate):
        """
        returns a tuple of tuples which the aircraft should fly through, 
        beginning at the start and finishing at the end.
        Ex.: ((0, 0), (2, 2), (2, 5), (4, 6), (7, 6), (9, 7), (10, 7), (10, 10))
        this is in grid indices, not world coordinates!
        """
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
    
    def a_star_worldcoords(self, start_xy, goal_xy, max_distance = 3,
              disallowed_cost = 10.0, occupied_cost = 20.0,
              restricted_cost = 1000.0, fuel_rate = 2.0):
        """
        Borrows a_star functionality, but turns grid indices into world coordinates.
        This method should be called from the outside.
        """
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