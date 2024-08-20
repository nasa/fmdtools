#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Coords` class for representing coordinate systems.

Has classes:

- :class:`CoordsParam`, which is used to define :class:`Coords` attributes.
- :class:`Coords`, which is used to define coordinate systems.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.rand import Rand
from fmdtools.define.base import is_iter
from fmdtools.define.object.base import BaseObject
from fmdtools.analyze.common import setup_plot, consolidate_legend, clear_prev_figure
from fmdtools.analyze.common import prep_animation_title, add_title_xylabs

import numpy as np
from typing import ClassVar

from matplotlib import pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.colors import to_rgba, ListedColormap, TABLEAU_COLORS
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import art3d


class CoordsParam(Parameter):
    """
    Defines the underlying arrays that make up the Coords object.

    Modifiers may be added to add additional properties (e.g., features, states,
    collections, points) to the coordinates. Additionally has class fields which may be
    overwritten.

    Class Variables
    ---------------
    x_size : int
        Number of rows in the x-dimension
    y_size : int
        Number of rows in the y-dimension
    blocksize : float
        Coordinate resolution
    gapwidth : float
        Width between coordinate cells (if any). Blocksize is inclusive of this width.

    Other Modifiers
    ---------------
    feature_featurename : tuple
        Tuple (datatype, defaultvalue) defining immutable grid features to instantiate
        as arrays.
    state_statename : tuple
        Tuple (datatype, defaultvalue) defining mutable grid features to instantiate
        as arrays.
    collect_collectionname : tuple
        Tuple (propertyname, value, comparator) defining a collection of points to
        instantiate as a list, where the tuple is arguments to Coords.find_all_prop
    point_pointname: tuple
        Tuple (x, y) referring to a point in the grid with a given name.

    Examples
    --------
    Defining the following classes will define a grid with a, v features, an
    h state, a point "start", and a "high_v" collection:

    >>> class ExampleCoordsParam(CoordsParam):
    ...     feature_a: tuple = (bool, False)
    ...     feature_v: tuple = (float, 1.0)
    ...     state_st: tuple = (float, 0.0)
    ...     point_start: tuple = (0.0, 0.0)
    ...     collect_high_v: tuple = ("v", 5.0, np.greater)
    >>> ex = ExampleCoordsParam()
    """

    x_size: ClassVar[int] = 10
    y_size: ClassVar[int] = 10
    blocksize: ClassVar[float] = 10.0
    gapwidth: ClassVar[float] = 0.0


class Coords(BaseObject):
    """
    Class for generating, accessing, and setting gridworld properties.

    Creates arrays, points, and lists of points which may correspond to desired
    modeling properties.

    ...

    Class Variables/Modifiers
    ---------------
    init_p: CoordsParam
        Parameter controlling default grid matrix (see CoordsParam), along with other
        properties of interest. Sets the .p container.
    init_r: Rand
        Random number generator. sets the .r container.
    init_properties: method
        Method that initializes the (non-default) properties of the Coords.

    Examples
    --------
    >>> class ExampleCoords(Coords):
    ...    container_p = ExampleCoordsParam
    ...    def init_properties(self, *args, **kwargs):
    ...        self.set_pts([[0.0, 0.0], [10.0, 0.0]], "v", 10.0)

    Instantiating a class with (see ExampleCoordsParam):

    - immutable arrays a and v,
    - mutable array st,
    - point start at (0.0), and
    - collection high made up of all points where v > 10.0.

    As shown, features are normal numpy arrays set to readonly:

    >>> ex = ExampleCoords()
    >>> type(ex.a)
    <class 'numpy.ndarray'>
    >>> np.size(ex.a)
    100
    >>> ex.a[0, 0] = True
    Traceback (most recent call last):
      ...
    ValueError: assignment destination is read-only

    The main difference with states is that they can be set, e.g.,:

    >>> ex.st[0, 0] = 100.0
    >>> ex.st[0, 0]
    100.0

    Collections are lists of points that map to immutable properties. In ExampleCoords,
    all points where v > 5.0 should be a part of high_v, as shown:

    >>> ex.high_v
    array([[ 0.,  0.],
           [10.,  0.]])

    Additionally, defined points (e.g., start) should be accessible via their names:

    >>> ex.start
    (0.0, 0.0)

    Note that these histories are tracked:

    >>> h = ex.create_hist([0, 1, 2])
    >>> h.keys()
    dict_keys(['r.probdens', 'st'])
    >>> h.st[0]
    array([[100.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]])
    """

    __slots__ = ("p", "r", "grid", "pts", "points", "collections", "features", "states",
                 "properties", "_args", "_kwargs")
    container_p = CoordsParam
    container_r = Rand
    roledicts = ['points', 'collections', 'features', 'states']
    immutable_roles = BaseObject.immutable_roles + ['points', 'collections', 'features']
    default_track = ["r", "states"]

    def __init__(self, *args, track='default', **kwargs):
        """Initialize class with properties in init_properties."""
        self._args = args
        self._kwargs = kwargs
        super().__init__(*args, track=[], **kwargs)
        self.init_grids(*args, **kwargs)
        self.init_properties(*args, **kwargs)
        self.build()
        self.init_track(track)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Coords

    def check_role(self, roletype, rolename):
        """Check that the rolename for coords is 'c'."""
        if roletype != 'coords':
            raise Exception("Invalid roletype for coords: " + roletype)
        if rolename != 'c':
            raise Exception("Invalid container name for Coords: "+rolename)

    def init_grids(self, *args, **kwargs):
        """Prepare class with defined features."""
        self.grid = np.array([[(i, j) for j in range(0, self.p.y_size)]
                             for i in range(0, self.p.x_size)]) * self.p.blocksize
        self.pts = self.grid.reshape(int(self.grid.size/2), 2)

        self.init_role_dict("point", set_attr=True)
        self.init_role_dict("collect", "ions")
        self.init_role_dict("feature")
        self.init_role_dict("state")
        self.properties = {**self.features, **self.states}
        for propname, prop in self.properties.items():
            prop_type, prop_default = prop
            proparray = np.full((self.p.x_size, self.p.y_size),
                                prop[1], dtype=prop[0])
            setattr(self, propname, proparray)

    def init_properties(self, *args, **kwargs):
        """Initialize arrays with non-default values."""
        return 0

    def build(self):
        """Set features as immutable."""
        for propname, prop in self.properties.items():
            if propname in self.features:
                proparray = getattr(self, propname)
                proparray.flags.writeable = False
        for cname, collection in self.collections.items():
            if collection[0] in self.features:
                setattr(self, cname, self.find_all_prop(*collection))
            else:
                raise Exception("Invalid collection: " + cname +
                                " collections may only map to (immutable) features")

    def find_all_prop(self, name, value=True, comparator=np.equal):
        """
        Find all points in array satisfying statement defined by value and comparator.

        Parameters
        ----------
        name : str
            Name of the underlying property (state or feature).
        value : bool/float/str/etc, optional
            Value to pass to comparator. The default is True.
        comparator : function, optional
            Function to use to compare the value with the array.
            (e.g. np.equal, np.greater, np.less...). The default is np.equal.

        Returns
        -------
        all: np.array
            List of points where the comparator method returns true.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.find_all_prop("v", 10.0, np.equal)
        array([[ 0.,  0.],
               [10.,  0.]])
        """
        prop = getattr(self, name)
        where = np.where(comparator(prop, value))
        pts_with_condition = [(p, where[1][i]) for i, p in enumerate(where[0])]
        return np.array([self.grid[tuple(p)] for p in pts_with_condition])

    def find_all(self, *points_colls, in_points_colls=True, **prop_kwargs):
        """
        Find all points in array satisfying multiple statements.

        Parameters
        ----------
        *points_colls: str
            Name(s) of points or collections defining the set of points to check.
            If not provided, assumes all points.
        in_points_colls: bool
            Whether the properties are to be searched in the given set of
            points/collections (True) or outside the given set of points/collections
            (False). The default is True
        **prop_kwargs : kwargs
            keyword arguments corresponding to properties, values and comparators, e.g.:
            statename=(True, np.equal)

        Returns
        -------
        all: np.array
            List of points where the comparator methods returns true.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.find_all(v=(10.0, np.equal))
        array([[ 0.,  0.],
               [10.,  0.]])
        >>> ex.st[0, 0] = 5.0
        >>> ex.find_all(v=(10.0, np.equal), st=(0.0, np.greater))
        array([[0., 0.]])
        >>> ex.find_all("high_v", v=(10.0, np.equal))
        array([[ 0.,  0.],
               [10.,  0.]])
        >>> ex.find_all("high_v", v=(10.0, np.less))
        array([], dtype=float64)
        >>> ex.st[2,2] = 1.0
        >>> ex.find_all("high_v", in_points_colls=False, st=(0.0, np.greater))
        array([[20., 20.]])
        """
        # check if in points or collections
        if points_colls:
            pts = []
            for name in points_colls:
                prop = getattr(self, name)
                if name in self.points:
                    pts.append(prop)
                elif name in self.collections:
                    pts.extend(prop)
                else:
                    raise Exception(name+"not in points or collections")
            true_array = np.full((self.p.x_size, self.p.y_size), not in_points_colls)
            for pt in pts:
                true_array[self.to_index(*pt)] = in_points_colls
        else:
            true_array = np.full((self.p.x_size, self.p.y_size), True)

        # check properties
        for name, (value, comparator) in prop_kwargs.items():
            prop = getattr(self, name)
            true_array *= comparator(prop, value)
        where = np.where(true_array)
        pts_with_condition = [(p, where[1][i]) for i, p in enumerate(where[0])]
        return np.array([self.grid[tuple(p)] for p in pts_with_condition])

    def to_index(self, *args):
        """
        Find the index of the array corresponding to the given x/y values.

        Parameters
        ----------
        *args : number, number,...
            x-y values corresponding to the scalar location of the point in the grid.

        Returns
        -------
        gridpoint: tuple
            x-y integer values corresponding to the corresponding array index.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.to_index(10, 20)
        (1, 2)
        >>> ex.to_index(54.234, 23.41)
        (5, 2)
        """
        return tuple(round(arg/self.p.blocksize) for arg in args)

    def to_gridpoint(self, *args):
        """
        Find the closest point in the grid corresponding to the given x/y values.

        Parameters
        ----------
        *args : number, number
            x-y value corresponding to the (unrounded) scalar location in the grid

        Returns
        -------
        pt : np.array
            x-y location corresponding to the (rounded) scalar location in the grid

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.to_gridpoint(3.5, 4)
        array([0., 0.])
        >>> ex.to_gridpoint(14.0, 12.0)
        array([10., 10.])
        """
        return self.grid[self.to_index(*args)]

    def get_properties(self, x, y):
        """
        Return a dictionary of all properties (features/states) at an x-y location.

        Parameters
        ----------
        x : number
            x-location to get from.
        y : number
            y-location to get the properties from

        Returns
        -------
        properties : dict
            Dictionary of property values at the given point.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.get_properties(0, 0)
        {'a': False, 'v': 10.0, 'st': 0.0}
        """
        properties = {}
        for prop in self.properties:
            properties[prop] = self.get(x, y, prop)
        return properties

    def get(self, x, y, prop, outside="error"):
        """
        Get the value of the property at the given scalar x/y values.

        Parameters
        ----------
        x : number
            Scalar x location
        y : number
            Scalar y location
        prop : str
            Name of the property to get.
        outside : value
            Value to provide if not in range. Default is 'error', which throws an error

        Returns
        -------
        value: int/bool/float/...
            Value to get from that point

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.get(10.0, 0.0, "v")
        10.0
        >>> ex.get(12.0, 4.9, "v")
        10.0
        >>> ex.get(50.0, 50.0, "v")
        1.0
        """
        if not self.in_range(x, y):
            if outside == "error":
                raise Exception("Outside bounds of grid: "+str(x)+','+str(y))
            else:
                return outside
        proparray = getattr(self, prop)
        x_i, y_i = self.to_index(x, y)
        return proparray[x_i, y_i]

    def assign_from(self, hist, t, *properties):
        """
        Assign properties of the coords to a value from the history.

        Useful for plotting progression of states over time.

        Parameters
        ----------
        hist : History
            History for the the coords object.
        t : int
            Time-step to get from the history.
        *properties : str
            Properties to assign from the history. If none provided, all are assigned.
        """
        if not properties:
            properties = [k for k in self.properties if k in hist.keys()]
            if not properties:
                raise Exception("No properties: "+str(properties))
        for prop in properties:
            prop_array = getattr(self, prop)
            prop_array[:] = hist.get(prop)[t]

    def set(self, x, y, prop, value):
        """
        Set the value of the property at the given scalar x/y values.

        Parameters
        ----------
        x : number
            Scalar x location
        y : number
            Scalar y location
        prop : str
            Name of the property to get.
        value: int/bool/float/...
            Value to set at that point

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.set(15.0, 12.0, "st", 100.0)
        >>> ex.get(15.0, 12.0, "st")
        100.0
        """
        proparray = getattr(self, prop)
        x_i, y_i = self.to_index(x, y)
        proparray[x_i, y_i] = value

    def set_range(self, prop, value, xmin=0, xmax='max', ymin=0, ymax='max',
                  inclusive=True):
        """
        Set ranges of the grid property to a given value.

        Parameters
        ----------
        prop : str
            Name of the range
        value : value
            Value to set
        xmin : number, optional
            minimum x-value. The default is 0.
        xmax : number, optional
            maximum x-value. The default is 'max'.
        ymin : number, optional
            minimum y-value. The default is 0.
        ymax : number, optional
            maximum y-value. The default is 'max'.
        inclusive : bool, optional
            whether to include the end of the range. The default is False.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.set_range("st", 100.0, 20, 40, 20, 40)
        >>> ex.st
        array([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0., 100., 100., 100.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0., 100., 100., 100.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0., 100., 100., 100.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]])
        >>> ex.set_range("st", 10.0)
        >>> ex.st
        array([[10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]])
        >>> ex.set_range("st", 0.0)
        >>> ex.set_range("st", 20.0, 20, 40, 20, 40, inclusive=False)
        >>> ex.st
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0., 20., 20.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0., 20., 20.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        """
        x_min_ind, y_min_ind = self.to_index(xmin, ymin)
        if xmax == 'max':
            x_max_ind = None
        else:
            x_max_ind, _ = self.to_index(xmax, 0.0)
            if inclusive and x_max_ind < self.p.x_size:
                x_max_ind += 1
        if ymax == 'max':
            y_max_ind = None
        else:
            _, y_max_ind = self.to_index(0.0, ymax)
            if inclusive and y_max_ind < self.p.y_size:
                y_max_ind += 1
        proparray = getattr(self, prop)
        proparray[x_min_ind:x_max_ind, y_min_ind:y_max_ind] = value

    def set_pts(self, pts, prop, value):
        """
        Set the property to a given value over a list of provided points.

        Parameters
        ----------
        pts : list
            List of points (e.g., [(1.0, 2.0), (3.0, 4.0)]) to set.
        prop : str
            Name of the property to set.
        value : int/bool/float or list...
            Value to set the points to. Can also pass a list.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.set_pts([(50,50), [80,80]], "st", -20.0)
        >>> ex.get(50, 50, "st")
        -20.0
        >>> ex.get(80, 80, "st")
        -20.0

        or:

        >>> ex.set_pts([(50,50), [80,80]], "st", [-10.0, -5.0])
        >>> ex.get(50, 50, "st")
        -10.0
        >>> ex.get(80, 80, "st")
        -5.0
        """
        if is_iter(value):
            if len(value) == len(pts):
                for i, pt in enumerate(pts):
                    self.set(*pt, prop, value[i])
            else:
                raise Exception("Value " + value
                                + "doesn't match length of pts: " + pts)
        else:
            for pt in pts:
                self.set(*pt, prop, value)

    def get_neighbors(self, x, y, direction="all"):
        """
        Get the points neighboring x and y.

        Parameters
        ----------
        x : float
            x-position
        y : float
            y-position
        direction: str/list
            Direction(s) to get neighbors from. Default is "all".

        Returns
        -------
        neighbors : list
            List of points next to the given grid point.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.get_neighbors(0, 0)
        [array([10.,  0.]), array([ 0., 10.])]
        >>> ex.get_neighbors(10, 10)
        [array([ 0., 10.]), array([20., 10.]), array([10., 20.]), array([10.,  0.])]
        >>> ex.get_neighbors(10, 10, direction="left")
        [array([ 0., 10.])]
        >>> ex.get_neighbors(10, 10, direction="up-down")
        [array([10., 20.]), array([10.,  0.])]
        >>> ex.get_neighbors(10, 10, direction=["up","down"])
        [array([10., 20.]), array([10.,  0.])]
        """
        ind = self.to_index(x, y)
        neighbor_list = []
        if 'left' in direction or direction == 'all':
            neighbor_list.append((ind[0]-1, ind[1]))
        if 'right' in direction or direction == 'all':
            neighbor_list.append((ind[0]+1, ind[1]))
        if 'up' in direction or direction == 'all':
            neighbor_list.append((ind[0], ind[1]+1))
        if 'down' in direction or direction == 'all':
            neighbor_list.append((ind[0], ind[1]-1))

        neighbors = []
        for i, n_point in enumerate(neighbor_list):
            if not (n_point[0] < 0 or
                    n_point[1] < 0 or
                    n_point[0] >= self.p.x_size or n_point[1] >= self.p.y_size):
                neighbors.append(self.grid[n_point[0], n_point[1]])
        return neighbors

    def find_closest(self, x, y, prop, include_pt=True, value=True,
                     comparator=np.equal):
        """
        Find the closest point in the grid satisfying a given property.

        Parameters
        ----------
        x : number
            x-position to check from.
        y : number
            y-position to check from.
        prop : str
            Property or collection of the grid to check.
        include_pt : bool, optional
            Whether to include the containing grid point. The default is True.
        value : bool/int/str, optional
            Value to compare against. The default is None, which returns the points
            that have the value True.
        comparator : method, optional
            Comparator function to use (e.g., np.equal, np.greater...).
            The default is np.equal.

        Returns
        -------
        pt: np.array
            x-y position of the closest point satisfying the property.

        Examples
        --------
        Can be used with default options to check collections, e.g.,:

        >>> ex = ExampleCoords()
        >>> ex.find_closest(20, 0, "high_v")
        array([10.,  0.])

        Alternatively can be used to search for the closest with a given property value,
        e.g.,:

        >>> ex.set(0, 0, "st", 1.0)
        >>> ex.find_closest(20, 0, "st", value=1.0, comparator=np.equal)
        array([0., 0.])
        """
        if prop in self.properties:
            pts = self.find_all_prop(prop, value, comparator)
        elif prop in self.collections or prop == 'pts':
            pts = getattr(self, prop)
        else:
            raise Exception(prop+" not in .properties or .collections")

        p_rounded = self.to_gridpoint(x, y)

        if p_rounded.tolist() in pts.tolist():
            return p_rounded
        else:
            if not include_pt:
                pts = np.array([p for p in pts if all(p != p_rounded)])
            dists = np.sqrt(np.sum((np.array([x, y])-pts)**2, 1))
            closest_ind = np.argmin(dists)
            xy = pts[closest_ind]
            return xy

    def in_range(self, x, y):
        """
        Check to see if the x-y point is in the range of the coordinate system.

        Parameters
        ----------
        x : number
            x-position to check from.
        y : number
            y-position to check from.

        Returns
        -------
        in: bool
            Whether the point is in the range of the grid
        """
        return (0.0 <= x <= self.p.blocksize * self.p.x_size and
                0.0 <= y <= self.p.blocksize * self.p.y_size)

    def in_area(self, x, y, coll):
        """
        Check to see if the point x, y is in a given collection or at a point.

        Parameters
        ----------
        x : number
            x-position to check from.
        y : number
            y-position to check from.
        coll: str
            Property or collection of the grid to check.

        Returns
        -------
        in: bool
            Whether the point is in the collection

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.in_area(0.4, 0.2, 'start')
        True
        >>> ex.in_area(10, 10, 'start')
        False
        """
        pts = getattr(self, coll)
        try:
            pt = self.to_gridpoint(x, y)
        except IndexError:
            return False
        if coll in self.points:
            return np.all(pt == pts)
        elif coll in self.collections:
            return pt in pts
        else:
            raise Exception("coll "+coll+" not a point or collection")

    def set_rand_pts(self, prop, value, number, pts=None, replace=False):
        """
        Ses a given number of points for a property to random value.

        Parameters
        ----------
        prop : str
            Property to set
        value : int/float/str/etc
            Value to set the points to
        number : int
            Number of points to set
        pts : list, optional
            List of points to select from.
            The default is None (which selects from all points).
        replace : bool, optional
            Whether to select with replacement. The default is False.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.set_rand_pts("st", 40, 5)
        >>> len(ex.find_all_prop("st", 40))
        5
        """
        if pts is None:
            pts = self.pts
        else:
            pts = pts
        set_pts = self.r.rng.choice(pts, number, replace=replace)
        self.set_pts(set_pts, prop, value)

    def set_prop_dist(self, prop, dist, *args, **kwargs):
        """
        Randomizes a property according to a given distribution.

        Parameters
        ----------
        prop : str
            Property to set
        dist : str
            Name of distribution to call from the rng.
            (see documentation for numpy.random)
        *args : tuple, optional
            Arguments to the distribution method (e.g., (min, max)). The default is ().
        **kwargs : kwargs, optional
            Keyword arguments to the distribution method. The default is {}.
        """
        p = getattr(self, prop)
        meth = getattr(self.r.rng, dist)
        new_p = meth(*args, size=p.shape, **kwargs)
        setattr(self, prop, new_p)

    def return_mutables(self):
        """Check if grid properties have changed (used in propagation)."""
        return tuple([*(tuple(map(tuple, getattr(self, state)))
                        for state in self.states)])

    def copy(self):
        """
        Copy the Coords object.

        Examples
        --------
        >>> ex = ExampleCoords()
        >>> ex.set(0, 0, "st", 25.0)
        >>> cop = ex.copy()
        >>> cop.get(0, 0, "st")
        25.0
        >>> np.all(ex.st == cop.st)
        True
        >>> id(ex.st) == id(cop.st)
        False
        """
        cop = self.__class__(*self._args, **self._kwargs)
        for state in self.states:
            setattr(cop, state, np.copy(getattr(self, state)))
        return cop

    def get_all_possible_track(self):
        """Extend BaseObject to include states in tracking."""
        return BaseObject.get_all_possible_track(self) + [*self.states]

    def get_collection(self, prop):
        """
        Get the points for a given collection.

        Parameters
        ----------
        prop : str
            Name of the collection.

        Returns
        -------
        coll: np.ndarray
            Array of points in the collection
        """
        if prop in self.collections or prop == 'pts':
            return getattr(self, prop)
        elif prop in self.points:
            return np.array([getattr(self, prop)])
        else:
            raise Exception("Not a point or collection")

    def show_property(self, prop, xlabel="x", ylabel="y", title='', proplab="prop",
                      as_bool=False, color='green', cmap='Greens', ec='black',
                      legend_kwargs={}, fig=None, ax=None, figsize=(5, 5), **kwargs):
        """
        Plot a given property 'prop' as a colormesh on an x-y grid.

        See matplotlib.pyplot.pcolormesh.

        Parameters
        ----------
        prop : str
            Name of the property to plot.
        xlabel : str, optional
            Label for x-axis. The default is "x".
        ylabel : str, optional
            Label for y-axis. The default is "y".
        title : str, optional
            Title for the plot. The default is ''.
        proplab : str, optional
            Label for the property. The default is "prop", which uses the name of the
            property provided.
        as_bool : bool, optional
            Whether to interpret the property as a boolean where >0.0 returns as True
            and <=0.0 returns as False.
        color : str, optional
            Color to use if the property is boolean. Default is 'green'.
        cmap : str, optional
            Colormap to use if property is continuous. Default is 'Greens'.
        ec : str, optional
            Default edge color. Default is 'black'. If 'face', no edges are drawn.
        **kwargs : kwargs
            Keyword arguments to matplotlib.pyplot.pcolormesh (e.g., cmap, edgecolors)

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        # create mesh
        p = getattr(self, prop)
        x = np.linspace(0., self.p.blocksize*(self.p.x_size-1), self.p.x_size)
        y = np.linspace(0., self.p.blocksize*(self.p.y_size-1), self.p.y_size)
        X, Y = np.meshgrid(x, y)
        # refine property for plotting
        if as_bool:
            p = p > 0.0
        if p.dtype == 'bool':
            cmap = ListedColormap([color])
            default_kwargs = dict(ec=ec, cmap=cmap)
            p = np.ma.array(p, mask=~p).swapaxes(0, 1)
            vmin, vmax = 0, 1
        else:
            default_kwargs = dict(ec=ec, cmap=cmap)
            p = p.swapaxes(0, 1)
            vmin = p.min()
            vmax = p.max()
            if vmin == vmax:
                vmax = vmin + 1.0
        kwargs = {**kwargs, **default_kwargs}

        im = ax.pcolormesh(X, Y, p, vmin=vmin, vmax=vmax, **kwargs)

        add_title_xylabs(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        if proplab == "prop":
            proplab = prop

        if p.dtype == 'bool':
            # if boolean, create a legend
            import matplotlib.patches as mpatches
            patch = mpatches.Patch(color=color, label=proplab)
            consolidate_legend(ax, add_handles=[patch], **legend_kwargs,
                               title="Properties")
        else:
            # if float, create a colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(proplab, rotation=270)
        return fig, ax

    def show_property_z(self, prop, z="prop", z_res=10, collections={},
                        xlabel="x", ylabel="y", zlabel="prop",
                        proplab="prop", cmap="Greens",
                        fig=None, ax=None, figsize=(4, 5), **kwargs):
        """
        Plot a given properties 'prop' and 'z' as a voxels on an x-y-z grid.

        See mpl_toolkits.mplot3d.axes3d.Axes3D.voxels.

        Parameters
        ----------
        prop : str
            Name of the property to represent a color.
        z : str, optional
            Name of the property to plot as z. The default is "prop", which uses same
            property as prop.
        z_res : int, optional
            Resolution to plot z at. The default is 10.
        collections:  dict, optional
            Collections to plot and their respective kwargs for show_collection.
            The default is {}.
        xlabel : str, optional
            Label for x-axis. The default is "x".
        ylabel : str, optional
            Label for y-axis. The default is "y".
        zlabel : str, optional
            Label for the z-axis. The default is "prop", which uses the name of the
            property.
        proplab : str, optional
            Label for the property. The default is "prop", which uses the name of the
            property provided.
        cmap : str, optional
            Name of the matplotlib colormap to use for colors. The default is "Greens".
        fig : matplotlib.figure, optional
            Existing Figure. The default is None.
        ax : matplotlib.axis, optional
            Existing axis. The default is None.
        figsize : tuple, optional
            Size for the figure. Default is (4,4)
        **kwargs : kwargs
            Kwargs to pass to Axes3D.voxels

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        default_kwargs = dict(edgecolor='k')
        kwargs = {**kwargs, **default_kwargs}

        fig, ax = setup_plot(fig=fig, ax=ax, z=True, figsize=figsize)

        c_array = getattr(self, prop)
        if z == "prop":
            z_array = c_array
        elif not z:
            z_array = c_array * 0
        else:
            z_array = getattr(self, z)

        dims = z_array.shape
        X, Y, Z = np.indices((dims[0]+1, dims[1]+1, z_res+1))
        z_shape = Z[:-1, :-1, :-1].swapaxes(0, 2).swapaxes(1, 2)

        max_z = 1 * z_array.max()
        min_z = 1 * z_array.min()
        norm_z_array = z_res * (1*z_array - min_z)/(max_z - min_z + 0.00000001)
        round_z_array = np.digitize(norm_z_array, [i for i in range(z_res)])
        shape = z_shape < round_z_array
        shape = shape.swapaxes(0, 1).swapaxes(1, 2)
        X_scale = X * self.p.blocksize - self.p.blocksize/2
        Y_scale = Y * self.p.blocksize - self.p.blocksize/2
        Z_scale = Z * (max_z - min_z) / z_res + min_z

        color_shape = np.array([c_array for i in range(z_res)])
        norm = plt.Normalize(color_shape.min(), color_shape.max())
        cmap = colormaps[cmap]
        colors = cmap(norm(color_shape)).swapaxes(0, 1).swapaxes(1, 2)

        for i, (prop, coll_kwargs) in enumerate(collections.items()):
            coll_colors = cm.rainbow(np.linspace(0, 1, len(collections)))
            coll = self.get_collection(prop)
            if 'color' not in coll_kwargs:
                coll_color = colormaps['rainbow'](coll_colors[i])
            else:
                coll_color = to_rgba(coll_kwargs['color'])

            if "text_z_offset" not in coll_kwargs:
                coll_kwargs['text_z_offset'] = (max_z - min_z) / z_res
            for pt in coll:
                index = self.to_index(*pt)
                inds = np.where(shape[index])
                if any(inds[0]):
                    z_index = inds[0][-1]
                else:
                    z_index = 0
                colors[index[0], index[1], z_index] = coll_color

        ax.voxels(X_scale, Y_scale, Z_scale, shape, facecolors=colors, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        return fig, ax

    def show_collection(self, prop, fig=None, ax=None, label=True, z="",
                        xlabel='x', ylabel='y', title='',
                        legend_args=False, text_z_offset=0.0, figsize=(4, 4), **kwargs):
        """
        Show a collection on the grid as square patches.

        Parameters
        ----------
        prop : str
            Name of the collection
        fig : matplotlib.figure, optional
            Existing Figure. The default is None.
        ax : matplotlib.axis, optional
            Existing axis. The default is None.
        label : str/bool, optional
            Label for the collection. The default is True, which shows the collection
            name. If False, no label is provided. If a string, the string is used as
            the label.
        z: str
            Argument to plot as third dimension on 3d plot. Default is '', which
            returns a 2d plot. If a number is provided, the plot will be 3d with
            the height at that constant z-value.
        xlabel : str, optional
            Label for x-axis. The default is "x".
        ylabel : str, optional
            Label for y-axis. The default is "y".
        zlabel : str, optional
            Label for the z-axis. The default is "prop", which uses the name of the
            property.
        legend_args : dict/False
            Specifies arguments to legend. Default is False, which shows no legend.
        text_z_offset : float
            Offset for text. Default is 0.0
        figsize : tuple
            Size for the figure. Default is (4,4)
        **kwargs : kwargs
            Kwargs to matplotlib.patches.Rectangle

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        offset = self.p.blocksize/2
        if not ax:
            fig, ax = setup_plot(z=z, figsize=figsize)
            if isinstance(z, str) and z:
                ax.set_zlim(getattr(self, z).min(), getattr(self, z).max())
            ax.set_xlim(-offset, self.p.x_size*self.p.blocksize+offset)
            ax.set_ylim(-offset, self.p.y_size*self.p.blocksize+offset)
        else:
            fig, ax = setup_plot(fig=fig, ax=ax, z=z, figsize=figsize)

        coll = self.get_collection(prop)
        for i, pt in enumerate(coll):
            corner = pt - np.array([offset, offset])
            rect = Rectangle(corner, self.p.blocksize, self.p.blocksize,
                             label=prop, **kwargs)
            ax.add_patch(rect)
            if isinstance(z, str) and z:
                z_h = self.get(pt[0], pt[1], z)
                art3d.patch_2d_to_3d(rect, z=z_h)
            elif type(z) in [float, int]:
                z_h = z
                art3d.patch_2d_to_3d(rect, z=z_h)
            else:
                z_h = None
            if label:
                if isinstance(label, str):
                    lab = rect.get_label()
                else:
                    lab = label
                if not z_h == None:
                    ax.text(pt[0], pt[1], z_h+text_z_offset, lab,
                            horizontalalignment="center", verticalalignment="center")
                else:
                    ax.text(pt[0], pt[1], lab,
                            horizontalalignment="center", verticalalignment="center")
        if not legend_args == False:
            if legend_args == True:
                legend_args = {}
            consolidate_legend(ax, **legend_args)
        add_title_xylabs(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return fig, ax

    def _show_properties(self, properties, fig, ax, pallette, **kwargs):
        """Show multiple properties on a plot. Helper function to .show()."""
        for i, (prop, prop_kwargs) in enumerate(properties.items()):
            kwar = {**kwargs,
                    'color': pallette[i],
                    'xlabel': '', 'ylabel': '', 'title': '',
                    **prop_kwargs}
            fig, ax = self.show_property(prop, fig=fig, ax=ax, **kwar)

    def _show_collections(self, collections, fig, ax, pallette, c_offset=0):
        """Show multiple collections on a plot. Helper function to .show()."""
        for i, (coll, coll_kwargs) in enumerate(collections.items()):
            kwar = {'color': pallette[i+c_offset],
                    'xlabel': '', 'ylabel': '', 'title': '',
                    'legend_args': True,
                    **coll_kwargs}
            self.show_collection(coll, fig=fig, ax=ax, **kwar)

    def show(self, properties={}, collections={}, coll_overlay=True, fig=None, ax=None,
             figsize=(5, 5), xlabel='x', ylabel='y', title='',
             pallette=[*TABLEAU_COLORS.keys()], **kwargs):
        """
        Plot a property and set of collections on the grid.

        Parameters
        ----------
        properties : dict
            Properties to plot and their arguments, e.g. {'prop1': {'color': 'green'}}
        collections : dict, optional
            Collections to plot and their respective kwargs for show_collection.
            The default is {}.
        coll_overlay : bool, optional
            If True, show collections in front of properties. If False, show properties
            in front of collections. Default is True.
        xlabel : str
            x-axis label.
        ylabel : str
            y-axis label.
        title : str
            title for the plot.
        pallete : list
            List of colors (in order) to cycle through for each plot.
        **kwargs : kwargs
            overall kwargs to show_property.

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        c_offset = len(properties)
        if coll_overlay:
            self._show_properties(properties, fig, ax, pallette, **kwargs)
            self._show_collections(collections, fig, ax, pallette, c_offset=c_offset)
        else:
            self._show_collections(collections, fig, ax, pallette, c_offset=c_offset)
            self._show_properties(properties, fig, ax, pallette, **kwargs)

        add_title_xylabs(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return fig, ax

    def show_from(self, t, history={}, properties={}, clear_fig=False, **kwargs):
        """
        Run Coords.show() at a particular time in the history.

        Parameters
        ----------
        t : int
            Time index to show the Coords object at.
        hist : History
            History to show the Coords object at.
        clear_fig : bool
            Whether to clear the figure beforehand. Default is False.
        **kwargs : kwargs
            kwargs for self.show

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        kwargs = prep_animation_title(t, **kwargs)
        if clear_fig:
            kwargs = clear_prev_figure(**kwargs)
        props = [p for p in properties if p in history]
        self.assign_from(history, t, *props)
        return self.show(properties=properties, **kwargs)

    def animate(self, hist, times='all', clear_fig=True, **kwargs):
        """
        Animate the coords over a history using show_from.

        Parameters
        ----------
        hist : History
            History of coords.
        times : list/'all'
            Times to animate over.
        **kwargs : kwargs
            Arguments to self.show.

        Returns
        -------
        ani : animation.Funcanimation
            Object with animation.
        """
        return hist.animate(self.show_from, times=times, clear_fig=clear_fig, **kwargs)

    def show_z(self, prop, z="prop", collections={}, legend_args=False, voxels=True,
               **kwargs):
        """
        Plot a property and set of collections in a discretized version of the grid.

        Parameters
        ----------
        prop : str
            Property to plot.
        z : str, optional
            Property to use as the height. The default is "prop".
        collections : dict, optional
            Collections to plot and their respective kwargs for show_collection.
            The default is {}.
        legend_args : dict/False
            Specifies arguments to legend. Default is False, which shows no legend.
        voxels : bool
            Whether or not to plot the grid as voxels. Default is True.
        **kwargs : kwargs
            kwargs to show_property3d.

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        if z == "prop":
            z = prop
        elif z == '':
            z = 0.0
        if voxels:
            fig, ax = self.show_property_z(prop, z=z, collections=collections, **kwargs)
        else:
            fig, ax = self.show_collection("pts", z=z, legend_args=legend_args,
                                           label=False, **kwargs)
        for coll in collections:
            self.show_collection(coll, fig=fig, ax=ax, legend_args=legend_args,
                                 **collections[coll], z=z)
        return fig, ax


class ExampleCoordsParam(CoordsParam):
    """Example of a Coords param for use in documentation/testing."""

    feature_a: tuple = (bool, False)
    feature_v: tuple = (float, 1.0)
    state_st: tuple = (float, 0.0)
    point_start: tuple = (0.0, 0.0)
    collect_high_v: tuple = ("v", 5.0, np.greater)


class ExampleCoords(Coords):
    """Example of Coords class for use in documentation and testing."""

    container_p = ExampleCoordsParam

    def init_properties(self, *args, **kwargs):
        """Initialize points where v=10.0."""
        self.set_pts([[0.0, 0.0], [10.0, 0.0]], "v", 10.0)


if __name__ == "__main__":
    ex = ExampleCoords()
    import doctest
    doctest.testmod(verbose=True)

    ex = ExampleCoords()
    ex.show_property("v", cmap="Greys")
    ex.show_collection("high_v")
    ex.show({"st": {}}, collections={"high_v": {"alpha": 0.5, "color": "red"}})
    ex.show_property_z("st", z="v",
                       collections={"high_v": {"alpha": 0.5, "color": "red"}})

    ex.show_property("v", cmap="Greys")
    ex.show_property_z("v")
    ex.show_property_z("st", z="v")
    ex.show_collection("high_v")
    ex.show_collection("high_v", z="v")
    ex.show_z("st", z="v",
            collections={"pts": {"color": "blue"},
                         "high_v": {"alpha": 0.5, "color": "red"}},
            legend_args=True)
