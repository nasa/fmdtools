# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:03:39 2023

@author: dhulse
"""
import numpy as np

class Map(object):
    x_size = 100
    y_size = 100
    blocksize = 10
    features = []
    states = []
    properties = {}

    def __init__(self):
        self.grid = np.array([[(i, j) for j in range(0, self.y_size, self.blocksize)]
                               for i in range(0, self.x_size, self.blocksize)])
        self.pts = self.grid.reshape(int(self.grid.size/2), 2)
        
        self.properties = [p for p in dir(self) if "_feature_" in p]
        self.states = [p for p in dir(self) if "_state_" in p]
        self.properties = {p[9:]: getattr(self, p)
                           for p in dir(self) if "_feature_" in p}
        self.properties.update({p[7:]: getattr(self, p)
                                for p in dir(self) if "_state_" in p})
        for propname, prop in self.properties.items():
            prop_type, prop_default = prop
            proparray = np.full((self.blocksize, self.blocksize), prop[1], dtype=prop[0])
            setattr(self, propname, proparray)
            # if prop in self.features:
            #    proparray.flags.writeable = False
            #    setattr(self, propname, proparray)

    def find_all(self, name, value=True, comparator = np.equal):
        prop = getattr(self, name)
        where = np.where(comparator(prop, value))
        pts_with_condition = [(p, where[1][i]) for i, p in enumerate(where[0])]
        return [self.grid[tuple(p)] for p in pts_with_condition]

    def to_gridpoint(self, *args):
        """Finds the grid point closest to the given x/y values"""
        return tuple(round(arg/self.blocksize)*self.blocksize for arg in args)

    def get(self, x, y, prop):
        proparray = getattr(self, prop)
        x_i, y_i = self.to_gridpoint(x, y)
        return proparray[x_i, y_i]
    
    def set(self, x, y, prop, value):
        proparray = getattr(self, prop)
        x_i, y_i = self.to_gridpoint(x, y)
        proparray[x_i, y_i] = value
        

class SpecialMap(Map):
    _feature_safe = (bool, True)
    _feature_allowed = (bool, False)
    _feature_occupied = (bool, False)
    _feature_height = (float, 0.0)
    _state_wind = (float, 1.0)
    _point_start = (0, 0)
    _point_end = (90, 90)
    _collections = ["safe"]
        



b = Map()
c = SpecialMap()
c.find_all("safe")