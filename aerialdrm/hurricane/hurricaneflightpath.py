# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:30:59 2025

@author: cwang29
"""
"""
decided to move new classes into a new file
"""
from fmdtools.define.object.coords import Coords, CoordsParam
from aerialdrm.base.aircraft.state import AircraftPosition
import maths
"""importing math for heuristic difference calculation"""
import numpy as np
"""importing numpy for zeroes initialization, init_properties"""

"""TODO: 
    assign costs to suitabilities of areas below. 
        Suitable: 0.
        Disallowed: 10.
        Occupied: 20.
        Restricted: 1000.
    create new grid on top of existing code for the drone to fly in. 
        class DroneFlightGridParam(CoordsParam):
        class DroneFlightGrid(Coords):

"""
class DroneFlightGridParam(CoordsParam):
    x_size: int = 48
    y_size: int = 48
    blocksize: float = 2.5
    feature_grid_cost: float = 0.0
    feature_fuel_cost: float = 0.0
    feature_total_cost: float = 0.0
    feature_heuristic: float = 0.0
    feature_edge_weight: float = 0.0 # total cost normalized wrt dist
    
"""GET NEIGHBORS"""
class DroneFlightGrid(Coords):
    container_p = DroneFlightGridParam
    
    def init_properties(self, **kwargs):
        shape = self.param.y_size, self.param.x_size
        self.set("grid_cost", np.zeroes(shape))
        self.set("fuel_cost", np.zeroes(shape))
        self.set("total_cost", np.zeroes(shape))
        self.set("edge_weight", np.zeroes(shape))
        self.set("heuristic", np.zeroes(shape))
        
    def get_edge_weights(self, env_coords, curr_x, curr_y, fuel_rate = 2.0, disallowed_cost = 10.0, occupied_cost = 20.0, restricted_cost = 1000.0, dist_cost = 2.0):
        """
        gets total cost, edge weight <- total cost/dist. edge weight to be used in A*
        """
        self.get_fuel_cost(curr_x, curr_y, fuel_rate)
        self.get_grid_cost(env_coords, disallowed_cost, occupied_cost, restricted_cost)
        for i in range(self.param.y_size):
            for j in range(self.param.x_size):
                total_cost = self.features["fuel_cost"][i][j] + self.features["grid_cost"][i][j]
                self.features["total_cost"][i][j] = total_cost
                x = j * self.param.blocksize + self.param.blocksize/2
                y = i * self.param.blocksize + self.param.blocksize/2
                dist = math.hypot(curr_x - x, curr_y - y)
                edge_weight = total_cost / dist
                self.features["edge_weight"][i][j] = edge_weight
            
    def get_fuel_cost(self, curr_x, curr_y, fuel_rate = 2.0):
        """
        calculate cost due to distance traveled per timestep.
        """
        for i in range(self.param.y_size):
            for j in range(self.param.x_size):
                x = j * self.param.blocksize + self.param.blocksize/2
                y = i * self.param.blocksize + self.param.blocksize/2
                self.features["fuel_cost"][i, j] = fuel_rate * math.hypot(curr_x - x, curr_y - y)
    def get_grid_cost(self, env_coords, disallowed_cost = 10.0, occupied_cost = 20.0, restricted_cost = 1000.0):
        """
        Calculate the GRID COST somewhere in the finer (than environment) drone traversal grid.
        Grid cost + traversal cost = total cost per timestep. defines A* grid weights.
        Heuristic = dist(place, goal.)
        maybe there will be more or less time per timestep? dependent on distance traveled. this seems like a problem.
        env_coords: hurricanecoords grid.
        disallowed_cost, occupied_cost, restricted_cost: how unsavory it is to fly above those areas.
        x, y: grid position.
        env_i, env_j: HurricaneCoords grid indices.
        implement gaussian later        
        """
        
        get_neighbors
        for i in range(self.param.y_size):
            for j in range(self.param.x_size):
                x = j * self.param.blocksize + self.param.blocksize/2
                y = i * self.param.blocksize + self.param.blocksize/2
                env_i = int(x // env_coords.param.blocksize)
                env_j = int(y // env_coords.param.blocksize)
                occupied = env_coords.features["occupied"][env_i, env_j]
                disallowed = env_coords.features["disallowed"][env_i, env_j]
                restricted = env_coords.features["restricted"][env_i, env_j]
                suitable = env_coords.features["suitable"][env_i, env_j]
                grid_cost = disallowed_cost * disallowed + occupied_cost * occupied + restricted_cost * restricted
                self.features["grid_cost"][i, j] = grid_cost
                goal_x = env_coords.param.point_end[0]
                goal_y = env_coords.param.point_end[1]
                dx, dy = goal_x - x, goal_y - y
                heuristic = math.hypot(dx, dy)
                self.features["heuristic"][i, j] = heuristic
                
    def a_star(self, start, goal):