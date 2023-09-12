# -*- coding: utf-8 -*-
"""
Classes for creating environments.
"""
import numpy as np
import copy
from typing import ClassVar
from fmdtools.define.parameter import Parameter
from fmdtools.define.rand import Rand
from fmdtools.define.common import is_iter, get_obj_track, init_obj_attr
from fmdtools.analyze.result import History
from fmdtools.define.flow import CommsFlow



class GridParam(Parameter):
    x_size: ClassVar[int] = 10
    y_size: ClassVar[int] = 10
    blocksize: ClassVar[float] = 10.0
    gapwidth: ClassVar[float] = 0.0


class Grid(object):
    """
    Class for generating, accessing, and setting gridworld properties. Creates arrays,
    points, and lists of points which may correspond to desired modelling properties.

    Class Variables/Modifiers
    ---------------
    init_p: Parameter
        Parameter controlling default grid matrix (see GridParam), along with other
        properties of interest. Sets the .p role.
    init_r: Rand
        Random number generator. sets the .r role.
    _feature_featurename : tuple
        Tuple (datatype, defaultvalue) defining immutable grid features to instantiate
        as arrays.
    _state_statename : tuple
        Tuple (datatype, defaultvalue) defining mutable grid features to instantiate
        as arrays.
    _collect_collectionname : tuple
        Tuple (propertyname, value, comparator) defining a collection of points to
        instantiate as a list, where the tuple is arguments to Grid.find_all
    _point_pointname: tuple
        Tuple (x, y) referring to a point in the grid with a given name.
    init_properties: method
        Method that initializes the (non-default) properties of the Grid.

    e.g., defining the following class:

    >>> class ExampleGrid(Grid):
    ...    _feature_a = (bool, False)
    ...    _feature_v = (float, 1.0)
    ...    _state_h = (float, 0.0)
    ...    _point_start = (0.0, 0.0)
    ...    _collect_high_v = ("v", 5.0, np.greater)
    ...    def init_properties(self, *args, **kwargs):
    ...        self.set_pts([[0.0, 0.0], [10.0, 0.0]], "v", 10.0)

    Instantiating this class a class with:
        - immutable arrays a and v,
        - mutable array h,
        - point start at (0.0), and
        - collection high made up of all points where v > 10.0

    As shown, features are normal numpy arrays set to readonly:

    >>> ex = ExampleGrid()
    >>> type(ex.a)
    <class 'numpy.ndarray'>
    >>> np.size(ex.a)
    100
    >>> ex.a[0, 0] = True
    Traceback (most recent call last):
      ...
    ValueError: assignment destination is read-only

    The main difference with states is that they can be set, e.g.:

    >>> ex.h[0, 0] = 100.0
    >>> ex.h[0, 0]
    100.0

    Collections are lists of points that map to immutable properties. In ExampleGrid,
    all points where v > 5.0 should be a part of high_v, as shown:

    >>> ex.high_v
    array([[ 0.,  0.],
           [10.,  0.]])

    Additionally, defined points (e.g., start) should be accessible via their names:

    >>> ex.start
    (0.0, 0.0)
    """
    __slots__ = ("p", "r", "grid", "pts", "points", "collections", "features", "states",
                 "properties", "_args", "_kwargs", "default_track", )
    _init_p = GridParam
    _init_r = Rand

    def __init__(self, *args, **kwargs):
        """Initializes class with properties in init_properties"""
        self._args = args
        self._kwargs = kwargs
        self.init_grids(*args, **kwargs)
        self.init_properties(*args, **kwargs)
        self.build()
        if not hasattr(self, 'default_track'):
            self.default_track = self.states


    def init_grids(self, *args, **kwargs):
        """Prepares class with defined features."""
        self.p = self._init_p(**kwargs.get('p', {}))
        self.r = self._init_r(**kwargs.get('r', {}))
        self.grid = np.array([[(i, j) for j in range(0, self.p.y_size)]
                             for i in range(0, self.p.x_size)]) * self.p.blocksize
        self.pts = self.grid.reshape(int(self.grid.size/2), 2)

        self.points = {p[7:]: getattr(self, p) for p in dir(self) if "_point_" in p}
        for pt_name, pt in self.points.items():
            setattr(self, pt_name, pt)

        self.collections = {p[9:]: getattr(self, p)
                            for p in dir(self) if "_collect_" in p}

        self.features = [p[9:] for p in dir(self) if "_feature_" in p]
        self.properties = {p[9:]: getattr(self, p)
                           for p in dir(self) if "_feature_" in p}

        self.states = [p[7:] for p in dir(self) if "_state_" in p]
        self.properties.update({p[7:]: getattr(self, p)
                               for p in dir(self) if "_state_" in p})
        for propname, prop in self.properties.items():
            prop_type, prop_default = prop
            proparray = np.full((self.p.x_size, self.p.y_size),
                                prop[1], dtype=prop[0])
            setattr(self, propname, proparray)

    def init_properties(self, *args, **kwargs):
        """Method used to initialize arrays with non-default values."""
        return 0

    def build(self):
        """Sets features as immutable."""
        for propname, prop in self.properties.items():
            if propname in self.features:
                proparray = getattr(self, propname)
                proparray.flags.writeable = False
        for cname, collection in self.collections.items():
            if collection[0] in self.features:
                setattr(self, cname, self.find_all(*collection))
            else:
                raise Exception("Invalid collection: " + cname +
                                " collections may only map to (immutable) features")

    def find_all(self, name, value=True, comparator=np.equal):
        """
        Finds all points in one of the arrays that satisfies the statement defined by
        the value and comparator.

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

        e.g.,

        >>> ex = ExampleGrid()
        >>> ex.find_all("v", 10.0, np.equal)
        array([[ 0.,  0.],
               [10.,  0.]])
        """
        prop = getattr(self, name)
        where = np.where(comparator(prop, value))
        pts_with_condition = [(p, where[1][i]) for i, p in enumerate(where[0])]
        return np.array([self.grid[tuple(p)] for p in pts_with_condition])

    def to_index(self, *args):
        """
        Finds the index of the array corresponding to the given x/y values.

        Parameters
        ----------
        *args : number, number,...
            x-y values corresponding to the scalar location of the point in the grid.

        Returns
        -------
        gridpoint: tuple
            x-y integer values corresponding to the corresponding array index.

        e.g.,

        >>> ex = ExampleGrid()
        >>> ex.to_index(10, 20)
        (1, 2)
        >>> ex.to_index(54.234, 23.41)
        (5, 2)
        """
        return tuple(round(arg/self.p.blocksize) for arg in args)

    def to_gridpoint(self, *args):
        """
        Finds the closest point in the grid corresponding to the given x/y values.

        Parameters
        ----------
        *args : number, number
            x-y value corresponding to the (unrounded) scalar location in the grid

        Returns
        -------
        pt : np.array
            x-y location corresponding to the (rounded) scalar location in the grid

        e.g.,

        >>> ex = ExampleGrid()
        >>> ex.to_gridpoint(3.5, 4)
        array([0., 0.])
        >>> ex.to_gridpoint(14.0, 12.0)
        array([10., 10.])
        """
        return self.grid[self.to_index(*args)]

    def get_properties(self, x, y):
        """
        Returns a dictionary of all properties (features and states) at the given
        x-y location.

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

        e.g.,

        >>> ex = ExampleGrid()
        >>> ex.get_properties(0, 0)
        {'a': False, 'v': 10.0, 'h': 0.0}
        """
        properties = {}
        for prop in self.properties:
            properties[prop] = self.get(x, y, prop)
        return properties

    def get(self, x, y, prop, outside = "error"):
        """
        Gets the value of the property at the given scalar x/y values

        Parameters
        ----------
        x : number
            Scalar x location
        y : number
            Scalar y location
        prop : str
            Name of the property to get.
        outside : value
            Value to provide if not in range. Default i 'error', which throws an error

        Returns
        -------
        value: int/bool/float/...
            Value to get from that point

        e.g.,
        >>> ex = ExampleGrid()
        >>> ex.get(10.0, 0.0, "v")
        10.0
        >>> ex.get(12.0, 4.9, "v")
        10.0
        >>> ex.get(50.0, 50.0, "v")
        1.0
        """
        if not self.in_range(x, y):
            if outside == "error":
                raise Exception("Outside bounds of grid: "+str(x, y))
            else:
                return outside
        proparray = getattr(self, prop)
        x_i, y_i = self.to_index(x, y)
        return proparray[x_i, y_i]

    def set(self, x, y, prop, value):
        """
        Sets the value of the property at the given scalar x/y values.

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

        e.g.,
        >>> ex = ExampleGrid()
        >>> ex.set(15.0, 12.0, "h", 100.0)
        >>> ex.get(15.0, 12.0, "h")
        100.0
        """
        proparray = getattr(self, prop)
        x_i, y_i = self.to_index(x, y)
        proparray[x_i, y_i] = value

    def set_pts(self, pts, prop, value):
        """
        Sets the property to a given value over a list of provided points.

        Parameters
        ----------
        pts : list
            List of points (e.g. [(1.0, 2.0), (3.0, 4.0)]) to set.
        prop : str
            Name of the property to set.
        value : int/bool/float or list...
            Value to set the points to. Can also pass a list.

        e.g.,
        >>> ex = ExampleGrid()
        >>> ex.set_pts([(50,50), [80,80]], "h", -20.0)
        >>> ex.get(50, 50, "h")
        -20.0
        >>> ex.get(80, 80, "h")
        -20.0

        or,
        >>> ex.set_pts([(50,50), [80,80]], "h", [-10.0, -5.0])
        >>> ex.get(50, 50, "h")
        -10.0
        >>> ex.get(80, 80, "h")
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

    def find_closest(self, x, y, prop, include_pt=True, value=None, comparator=None):
        """
        Finds the closest point in the grid satisfying a given property.

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
            Value to compare against. The default is None, which solely finds the
            closest in the array.
        comparator : TYPE, optional
            Comparator function to use (e.g., np.equal, np.greater...).
            The default is None.

        Returns
        -------
        pt: np.array
            x-y position of the closest point satisfying the property.

        Can be used with default options to check collections, e,g.:

        >>> ex = ExampleGrid()
        >>> ex.find_closest(20, 0, "high_v")
        array([10.,  0.])

        Alternatively can be used to search for the closest with a given property value,
        e.g.,:

        >>> ex.set(0, 0, "h", 1.0)
        >>> ex.find_closest(20, 0, "h", value=1.0, comparator=np.equal)
        array([0., 0.])
        """
        if prop in self.properties:
            pts = self.find_all(prop, value, comparator)
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
        Checks to 

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
        Checks to see if the point x, y is in a given collection or at a point.

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
        """
        pts = getattr(self, coll)
        if coll in self.points:
            pt = getattr(self, coll)
            return np.all(pt == pts)
        elif coll in self.collections:
            pt = self.to_gridpoint(x, y)
            return pt in pts
        else:
            raise Exception("coll "+coll+" not a point or collection")

    def set_rand_pts(self, prop, value, number, pts=None, replace=False):
        """
        Sets a given number of points for a property to random value.

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

        e.g.,
        >>> ex = ExampleGrid()
        >>> ex.set_rand_pts("h", 40, 5)
        >>> len(ex.find_all("h", 40))
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
        """Used in propagation to check if grid properties have changed."""
        return tuple([*(getattr(self, state) for state in self.states)])

    def copy(self):
        """Copies the grid.
        e.g.,
        >>> ex = ExampleGrid()
        >>> ex.set(0, 0, "h", 25.0)
        >>> cop = ex.copy()
        >>> cop.get(0, 0, "h")
        25.0
        >>> np.all(ex.h == cop.h)
        True
        >>> id(ex.h) == id(cop.h)
        False
        """
        cop = self.__class__(*self._args, **self._kwargs)
        for state in self.states:
            setattr(cop, state, np.copy(getattr(self, state)))
        return cop

    def create_hist(self, timerange, track):
        """
        Creates a history of states for the grid

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'.
            The default is None.

        Returns
        -------
        hist : History
            History of fields specified in track.

        e.g.,
        >>> ex = ExampleGrid()
        >>> h = ex.create_hist([0, 1, 2], "all")
        >>> h.keys()
        dict_keys(['h'])
        >>> h.h[0]
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """
        track = get_obj_track(self, track, all_possible = self.states)
        h = History()
        for att in track:
            val = getattr(self, att)
            h.init_att(att, val, timerange, track, dtype=np.ndarray)
        return h

    def get_collection(self, prop):
        """
        Gets the points for a given collection.

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

    def return_states(self):
        """Returns the mutable states of a grid."""
        states = dict.fromkeys(self.states)
        for state in states:
            states[state] = copy.copy(getattr(self, state))
        return states


class ExampleGrid(Grid):
    """Example of Grid class for use in documentation and testing. Must match
    docstrings for Grid."""
    _feature_a = (bool, False)
    _feature_v = (float, 1.0)
    _state_h = (float, 0.0)
    _point_start = (0.0, 0.0)
    _collect_high_v = ("v", 5.0, np.greater)

    def init_properties(self, *args, **kwargs):
        self.set_pts([[0.0, 0.0], [10.0, 0.0]], "v", 10.0)


class Environment(CommsFlow):
    """
    Class for representing environments (in development).
    
    Environments are CommsFlows in order to readily enable perception as well as
    sending and recieving of information. In addition to having normal flow properties,
    they also contain the roles:

    Roles
    ---------------
    g: Grid
        Representation of gridworld properties
    r: Rand
        Representaiton of random variables/rng
    f: Form
        (in development): Representaion of shapes/forms
    """
    slots = ["g", "_args_g", "r", "_args_r"]
    _init_g = Grid
    _init_r = Rand
    default_track = ('s', 'i', 'g')

    def __init__(self, name, glob=[], p={}, s={}, g={}, r={}):
        super().__init__(name, glob=glob, p=p, s=s)
        init_obj_attr(self, r=r, g=g)
        self.update_seed()

    def return_mutables(self):
        return (*super().return_mutables(),
                self.r.return_mutables(),
                self.g.return_mutables())

    def copy(self, glob=[], p={}, s={}):
        cop = super().copy(glob=glob, p=p, s=s)
        cop.r.assign(self.r)
        cop.g = self.g.copy()
        return cop

    def status(self):
        stat = super().status()
        stat["g"] = self.g.return_states()
        return stat

    def reset(self):
        super().reset()
        self.r.reset()
        self.g = self._init_g(**self._args_g)

    def update_seed(self, seed=[]):
        if not seed:
            seed = self.r.seed
        self.g.r.update_seed(seed)

    def return_probdens(self):
        return self.r.return_probdens() * self.g.r.return_probdens()

class ExampleEnvironment(Environment):
    _init_g = ExampleGrid


if __name__ == "__main__":

    import doctest
    doctest.testmod(verbose=True)
    e = ExampleEnvironment("env")
    d = e.copy()
