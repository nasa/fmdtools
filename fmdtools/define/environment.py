# -*- coding: utf-8 -*-
"""
Classes for creating environments.
"""
import numpy as np
from typing import ClassVar
from fmdtools.define.parameter import Parameter
from fmdtools.define.rand import Rand
from fmdtools.define.common import is_iter


class GridParam(Parameter):
    x_size: ClassVar[int] = 10
    y_size: ClassVar[int] = 10
    blocksize: ClassVar[float] = 10.0


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
    [array([0., 0.]), array([10.,  0.])]

    Additionally, defined points (e.g., start) should be accessible via their names:

    >>> ex.start
    (0.0, 0.0)
    """
    __slots__ = ("p", "r", "grid", "pts")
    _init_p = GridParam
    _init_r = Rand
    points = {}
    collections = {}
    features = []
    states = []
    properties = {}

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

    def __init__(self, *args, **kwargs):
        """Initializes class with properties in init_properties"""
        self._args = args
        self._kwargs = kwargs
        self.init_grids(*args, **kwargs)
        self.init_properties(*args, **kwargs)
        self.build()

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
        all: list
            List of points where the comparator method returns true.

        e.g.,

        >>> ex = ExampleGrid()
        >>> ex.find_all("v", 10.0, np.equal)
        [array([0., 0.]), array([10.,  0.])]
        """
        prop = getattr(self, name)
        where = np.where(comparator(prop, value))
        pts_with_condition = [(p, where[1][i]) for i, p in enumerate(where[0])]
        return [self.grid[tuple(p)] for p in pts_with_condition]

    def to_gridpoint(self, *args):
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
        >>> ex.to_gridpoint(10, 20)
        (1, 2)
        >>> ex.to_gridpoint(54.234, 23.41)
        (5, 2)
        """
        return tuple(round(arg/self.p.blocksize) for arg in args)

    def get(self, x, y, prop):
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
        proparray = getattr(self, prop)
        x_i, y_i = self.to_gridpoint(x, y)
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
        x_i, y_i = self.to_gridpoint(x, y)
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

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
