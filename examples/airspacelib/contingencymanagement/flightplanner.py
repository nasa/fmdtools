# -*- coding: utf-8 -*-
"""
Constructs for flight planning.

@author: cwang29

Called externally via:
a_star_worldcoords()
  └─► a_star()
        └─► nx_graph_gen()
              └─► get_edge_weights()
                    └─► get_grid_costs()
                    └─► neighbor_gen()
        └─► nx.astar_path()

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
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
    max_cost: float = 1000000.0
    
class DroneFlightGrid(Coords):
    """
    Grid representation for drone path planning using A* search.

    Examples
    --------
    >>> from examples.airspacelib.contingencymanagement.flightplanner import DroneFlightGrid, DroneFlightGridParam
    >>> from examples.airspacelib.contingencymanagement.environment import ContingencyEnvironment
    >>> env = ContingencyEnvironment()
    >>> param = DroneFlightGridParam(x_size=120, y_size=120, blocksize=1.0)
    >>> grid = DroneFlightGrid(env, p=param)
    >>> start = (10.0, 10.0)
    >>> goal = (100.0, 100.0)
    >>> path = grid.a_star_worldcoords(start_xy=start, goal_xy=goal,
    ...     max_distance=1,
    ...     disallowed_cost=5.0,
    ...     occupied_cost=2.0,
    ...     restricted_cost=100.0,
    ...     fuel_rate=1.0,
    ...     obstacle=False)
    >>> isinstance(path, tuple)
    True
    >>> all([isinstance(p, tuple) and len(p) == 2 for p in path])
    True
    >>> path[0] == start or len(path[0]) == 2
    True
    >>> path[-1] == goal or len(path[-1]) == 2
    True
    """

    __slots__ = ('env', 'env_coords')
    container_p = DroneFlightGridParam
    state_grid_costs: tuple = (float, 0.0)
    state_fuel_costs: tuple = (dict, None)
    state_edge_weights: tuple = (dict, None)

    def init_properties(self, env={}, **kwargs):
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
        Assign edge weights between all accessible nodes in flight grid.

        Accounts for environmental and fuel costs.

        Parameters
        ----------
        fuel_rate : float
            Fuel rate determining cost of fuel use
        disallowed_cost : float
            Cost for flying over disallowed areas
        restricted_cost : float
            Cost for flying through restreicted areas
        max_distance : float
            Maximum distance possible
        obstacle: bool
            Whether or not there is an obstacle.
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
        Assign suboptimality of all environment regions into FlightGrid areas.

        Parameters
        ----------
        disallowed_cost : float
            Cost for flying over disallowed areas
        restricted_cost : float
            Cost for flying through restreicted areas
        max_distance : float
            Maximum distance possible
        obstacle: bool
            Whether or not there is an obstacle.
        """
        unsafe_points = set()
        if obstacle:
            try:
                uav_geom = self.env.ga.geoms()['uav']
            except (AttributeError, KeyError):
                print("No such UAV geometry architecture exists")
            uav_geom = self.env.ga.geoms()['uav']
            coarse_block = self.env.c.p.blocksize
            fine_block = self.p.blocksize
            coarse_unsafe_pts = [pt for pt in self.env_coords.pts if uav_geom.at(pt, 'safety')]
            for (cx, cy) in coarse_unsafe_pts:
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
                        unsafe_points.add((i, j))

        max_offset = int(self.env_coords.p.blocksize / self.p.blocksize / 3 + 1)
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
                            weight = (2 * max_offset + 1) ** -2
                            total += weight * (
                                disallowed_cost * disallowed +
                                occupied_cost   * occupied +
                                restricted_cost * restricted
                            )
                if (i, j) in unsafe_points:
                    total += restricted_cost
                self.set(i, j, 'grid_costs', total)

    def neighbor_gen(self, j, i, max_distance):
        """Generate neighbor points to j,i within given max_distance."""
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
        """Generate a networkx graph of edge weights to run a* on."""
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
        obstacle: boolean
            Whether or not there exists an aerial UAV threat.

        Returns
        -------
        path : tuple(tuple(int, int))
            DroneFlightGrid A*-generated path.
        """
        G = self.nx_graph_gen(max_distance, disallowed_cost, occupied_cost,
                              restricted_cost, fuel_rate, obstacle)
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1]) * fuel_rate

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
            G_test = self.nx_graph_gen(max_distance, disallowed_cost,
                                       occupied_cost, restricted_cost,
                                       0, obstacle)
            cost_test = nx.astar_path_length(G_test, start, goal,
                                             heuristic=lambda a,b: 0.0,
                                             weight = "weight")
            if cost_test < self.p.max_cost:
                end = self.find_new_goal(start, max_distance, disallowed_cost,
                                         occupied_cost, restricted_cost,
                                         fuel_rate, obstacle)
                end = (end[0] +
                       self.env_coords.p.blocksize//(2 * self.p.blocksize),
                       end[1] +
                       self.env_coords.p.blocksize // (2 * self.p.blocksize))
                path =  self.a_star(start, end, max_distance, disallowed_cost,
                                    occupied_cost, restricted_cost,
                                    fuel_rate, obstacle)
            else:
                path = (start, start)
        return path

    def find_new_goal(self, start_idx, max_distance,
                      disallowed_cost, occupied_cost,
                      restricted_cost, fuel_rate, obstacle):
        """
        Identify the nearest reachable grid cell.

        Grid cell must satisfy prioritized suitability criteria under
        cost-constrained pathfinding framework.

        Returns
        -------
        (j, i) tuple grid index of new target location
        """
        G = self.nx_graph_gen(max_distance, disallowed_cost,
                              occupied_cost, restricted_cost,
                              fuel_rate, obstacle)
        j0, i0 = start_idx
        psize = self.env_coords.p.blocksize
        fsize = self.p.blocksize
        x0 = j0 * fsize + fsize / 2
        y0 = i0 * fsize + fsize / 2

        closest = self.env_coords.find_closest(x0, y0, "suitable")
        if closest is not None:
            j = int(closest[0] // fsize) - int(psize // (2 * fsize))
            i = int(closest[1] // fsize) - int(psize // (2 * fsize))
            candidate = (j, i)
            if (candidate in G
                    and self.is_feasible_path(G, start_idx, candidate,
                                              fuel_rate)):
                return candidate

        suitable_coords = self.env_coords.suitable
        if suitable_coords is not None and len(suitable_coords) > 0:
            sorted_coords = sorted(
                [((x - x0)**2 + (y - y0)**2, x, y) for x, y in suitable_coords]
            )
            for _, x, y in sorted_coords:
                j = int(x // fsize) - int(psize // (2 * fsize))
                i = int(y // fsize) - int(psize // (2 * fsize))
                candidate = (j, i)
                if candidate in G and self.is_feasible_path(G, start_idx,
                                                            candidate,
                                                            fuel_rate):
                    return candidate

        found = [None, None]
        coarse_start = (round(x0 / psize), round(y0 / psize))
        shell_radius = 1
        max_shell = max(self.env_coords.p.x_size, self.env_coords.p.y_size)

        while None in found and shell_radius < max_shell:
            for dj in range(-shell_radius, shell_radius + 1):
                for di in range(-shell_radius, shell_radius + 1):
                    if abs(dj) != shell_radius and abs(di) != shell_radius:
                        continue
                    cj = coarse_start[0] + dj
                    ci = coarse_start[1] + di
                    if not (0 <= cj < self.env_coords.p.x_size
                            and 0 <= ci < self.env_coords.p.y_size):
                        continue
                    x = cj * psize + psize / 2
                    y = ci * psize + psize / 2
                    disallowed = self.env_coords.get(x, y, 'disallowed',
                                                     outside=True)
                    occupied = self.env_coords.get(x, y, 'occupied',
                                                   outside=True)
                    restricted = self.env_coords.get(x, y, 'restricted',
                                                     outside=True)
                    j = int((x - fsize / 2) / fsize)
                    i = int((y - fsize / 2) / fsize)
                    candidate = (j, i)
                    if candidate not in G:
                        continue
                    if not self.is_feasible_path(G, start_idx, candidate,
                                                 fuel_rate):
                        continue
                    if (disallowed and not occupied and not restricted
                            and found[0] is None):
                        found[0] = candidate
                    elif (occupied and not disallowed and not restricted and
                          found[1] is None):
                        found[1] = candidate
            shell_radius += 1

        for fallback in found:
            if fallback is not None:
                return fallback
        return start_idx

    def is_feasible_path(self, G, start, goal, fuel_rate):
        """
        Determine whether a path exists between start/goal within max_cost.

        Parameters
        ----------
        start : tuple
            i/j grid indices of start location
        goal : tuple
            i/j grid indices of goal location
        """
        try:
            def heuristic(a, b):
                return math.hypot(a[0] - b[0], a[1] - b[1]) * fuel_rate
            cost = nx.astar_path_length(G, start, goal, heuristic=heuristic,
                                        weight="weight")
            return cost <= self.p.max_cost
        except nx.NetworkXNoPath:
            return False

    def a_star_worldcoords(self, start_xy, goal_xy, max_distance, disallowed_cost, 
               occupied_cost, restricted_cost, fuel_rate, obstacle):
        """
        Convert start_xy & goal_xy and run a*.

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