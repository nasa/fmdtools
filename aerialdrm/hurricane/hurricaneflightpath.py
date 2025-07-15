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
    """
    Parameters for configuring the DroneFlightGrid.
    
    Class Variables
    ---------------
    x_size : int
        Number of rows in the x-dimension
    y_size : int
        Number of rows in the y-dimension
    blocksize : float
        Coordinate resolution
    max_cost : float
        Maximum cost before algorithm termination
    
    Fields
    ---------------
    state_grid_costs : tuple
        Holds environmental cost at each FlightGrid point
    state_fuel_costs : tuple
        Holds fuel costs from every single-timestep accessible FlightGrid point
        in a dictionary entry; each FlightGrid point has its own dictionary
    state_edge_weights : tuple
        Holds total costs from every single-timestep accessible FlightGrid point
        in a dictionary entry; each FlightGrid point has its own dictionary
    """
    x_size: int = 120
    y_size: int = 120
    blocksize: float = 1.0
    
    max_cost = 1000000.0
    
    state_grid_costs: tuple = (float, 0.0)
    state_fuel_costs: tuple = (dict, None)
    state_edge_weights: tuple = (dict, None)
    
class DroneFlightGrid(Coords):
    """
    Grid representation for drone path planning using A* search.

    Examples
    --------
    >>> from aerialdrm.hurricane.hurricaneflightpath import DroneFlightGrid, DroneFlightGridParam # FIX DOCTEST TO ACCOUNT FOR NEW: OBSTACLE LOGIC, env =/= env_coords
    >>> from aerialdrm.hurricane.hurricaneenvironment import HurricaneCoords, HurricaneCoordsParam
    >>> env_param = HurricaneCoordsParam(x_size=120, y_size=120, blocksize=10.0)
    >>> env = HurricaneCoords(p=env_param)
    >>> param = DroneFlightGridParam(x_size=120, y_size=120, blocksize=1.0)
    >>> grid = DroneFlightGrid(env, p=param)
    >>> start = (0, 0)
    >>> goal = (40, 50)
    >>> path = grid.a_star(start, goal,
    ...                    max_distance=1,
    ...                    disallowed_cost=5.0,
    ...                    occupied_cost=2.0,
    ...                    restricted_cost=100.0,
    ...                    fuel_rate=5.0)
    >>> isinstance(path, tuple)
    True
    >>> path[0] == start and path[-1] == goal
    True
    >>> bool(all(isinstance(p, tuple) for p in path))
    True
    >>> bool(all(len(p) == 2 for p in path))
    True

    """
    container_p = DroneFlightGridParam

    def init_properties(self, env, **kwargs):
        self.env_coords = env.c
        self.env = env
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                self.fuel_costs[i, j] = {}
                self.edge_weights[i, j] = {}

    def get_edge_weights(self, fuel_rate,
                         disallowed_cost, occupied_cost, restricted_cost,
                         max_distance, obstacle):
        """
        Assign edge weights between all accessible nodes in flight grid,
        accounting for environmental and fuel costs.
        """
        self.get_grid_costs(disallowed_cost,
                            occupied_cost, restricted_cost, obstacle)
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

    def get_grid_costs(self, disallowed_cost, occupied_cost, restricted_cost, obstacle):
        """
        Assign suboptimality of all environment regions 
        into correspondent FlightGrid areas.
        """
        if obstacle:
            uav_geom = self.env.ga.geoms()['uav']
            coarse_block = self.env.p.blocksize
            fine_block = self.p.blocksize
            threat_cells = []

            for pt in uav_geom.grid:
                if uav_geom.at(pt, 'safety'):
                    threat_cells.append(pt)

            for (cx, cy) in threat_cells:
                min_x = cx - coarse_block / 2
                max_x = cx + coarse_block / 2
                min_y = cy - coarse_block / 2
                max_y = cy + coarse_block / 2

                i_start = max(0, int(min_y // fine_block))
                i_end   = min(self.p.y_size, int(max_y // fine_block) + 1)
                j_start = max(0, int(min_x // fine_block))
                j_end   = min(self.p.x_size, int(max_x // fine_block) + 1)

                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        self.set(i, j, 'grid_costs', restricted_cost)
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
                            disallowed = self.env_coords.get(x, y, 'disallowed',
                                                             outside=True)
                            occupied = self.env_coords.get(x, y, 'occupied',
                                                           outside=True)
                            restricted = self.env_coords.get(x, y, 'restricted',
                                                             outside=True)
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
                     occupied_cost, restricted_cost, fuel_rate, obstacle):
        flight_grid = nx.DiGraph()
        self.get_edge_weights(fuel_rate,
                              disallowed_cost, occupied_cost,
                              restricted_cost, max_distance, obstacle)
        for i in range(self.p.y_size):
            for j in range(self.p.x_size):
                v = (j, i)
                ew = self.get_properties(i, j)['edge_weights']
                for u, w in ew.items():
                    flight_grid.add_edge(v, u, weight=w)
        return flight_grid
        
    def a_star(self, start, goal, max_distance, disallowed_cost, occupied_cost,
               restricted_cost, fuel_rate, obstacle):
        """
        Find the optimal A* path between two grid indices at start and goal.
        
        Parameters
        ----------
        start : tuple
            Grid index (j, i) of the starting point.
        goal : tuple
            Grid index (j, i) of the destination point.
        max_distance : int 
            Maximum grid distance jumped in each direction during each A* 
            node-node step. 
        disallowed_cost : float
            Cost adder for flying in zones where landing is disallowed.
        occupied_cost : float
            Cost adder for flying in zones which are human-occupied.
        restricted_cost : float
            (Arbitrarily high) Cost adder for flying in restricted zones.
        fuel_rate : float
            Cost multiplier due to fuel consumption.
            
        Returns
        -------
        path : tuple(tuple(int, int))
            DroneFlightGrid A*-generated path.
        """
        G = self.nx_graph_gen(max_distance, disallowed_cost, occupied_cost,
                              restricted_cost, fuel_rate, obstacle)
        heuristic = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        if start not in G:
            print(f"Start node {start} not in graph")
        if goal not in G:
            print(f"Goal node {goal} not in graph")
        if not nx.has_path(G, start, goal):
            print(f"No path from {start} to {goal}")
            return (start, start)
        path = tuple(nx.astar_path(G, start, goal, heuristic = heuristic,
                                   weight = "weight"))
        cost = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
        if cost > self.p.max_cost:
            G_test = self.nx_graph_gen(max_distance, disallowed_cost, occupied_cost, restricted_cost, 0)
            cost_test = nx.astar_path_length(G_test, start, goal, heuristic = heuristic, weight = "weight")
            if cost_test < self.p.max_cost:
                """detect fuel infeasibility as opposed to environmental"""
                # PLAN to land at the reachable spot with the LOWEST cost to goal + cost to reach associated reachable spot.
            else:
                path = (start, start)
        return path
    
    def a_star_worldcoords(self, start_xy, goal_xy, max_distance, disallowed_cost, 
               occupied_cost, restricted_cost, fuel_rate, obstacle):
        """
        Converts start_xy & goal_xy world coordinates into DroneFlightGid indices, 
        takes a_star functionality.

        Parameters
        ----------
        start_xy : tuple
            x/y coordinates of start location
        goal_xy : tuple
            x/y coordinates of goal location
        *args : 
            Additional arguments to DroneFlightGrid.a_star
        """
        start_ij = self.to_index(*start_xy)
        goal_ij = self.to_index(*goal_xy)
        path_ij = self.a_star(start_ij, goal_ij, max_distance, disallowed_cost, 
                   occupied_cost, restricted_cost, fuel_rate, obstacle)
        path_xy = tuple(tuple(self.grid[j, i]) for (j, i) in path_ij)
        return path_xy
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
"""
remove passing in the environments cororsd and just the evnrioment so i can also the threats from it. and then i change the confirttional to say if its within some distance of the tthreat we replan but if its within 0 just keep the standstill logci. also updating the restricted flight spaces based off of the point array thing. 
"""