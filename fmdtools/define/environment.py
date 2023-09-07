# -*- coding: utf-8 -*-
"""
Classes for creating environments.
"""
import numpy as np
from typing import ClassVar
from fmdtools.define.parameter import Parameter
from fmdtools.define.rand import Rand
from fmdtools.define.common import is_iter
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch, Rectangle


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
                 "properties", "_args", "_kwargs")
    _init_p = GridParam
    _init_r = Rand

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
        elif prop in self.collections:
            pts = getattr(self, prop)
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

    def show_property(self, prop, xlab="x", ylab="y", proplab="prop", dim="2d",
                      **kwargs):
        """
        Plots a given property 'prop' as a colormesh on an x-y grid.
        See matplotlib.pyplot.pcolormesh.

        Parameters
        ----------
        prop : str
            Name of the property to plot.
        xlab : str, optional
            Label for x-axis. The default is "x".
        ylab : str, optional
            Label for y-axis. The default is "y".
        proplab : str, optional
            Label for the property. The default is "prop", which uses the name of the
            property provided.
        **kwargs : kwargs
            Keyword arguments to matplotlib.pyplot.pcolormesh (e.g., cmap, edgecolors)

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        default_kwargs = dict(edgecolors='black', cmap="Greens")
        kwargs = {**kwargs, **default_kwargs}

        if dim == "2d":
            fig, ax = plt.subplots(1)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        p = getattr(self, prop)
        # im = ax.matshow(p, **kwargs)
        offset = self.p.blocksize/2
        x = np.linspace(0., self.p.blocksize*(self.p.x_size-1), self.p.x_size)
        y = np.linspace(0., self.p.blocksize*(self.p.y_size-1), self.p.y_size)
        X, Y = np.meshgrid(x, y)
        if dim == "2d":
            im = ax.pcolormesh(X, Y, p.swapaxes(0, 1), **kwargs)
        else:
            im = ax.plot_surface(X, Y, p.swapaxes(0, 1), **kwargs)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

        if dim == "2d":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            if proplab == "prop":
                proplab = prop
            cbar.set_label(proplab, rotation=270)
        return fig, ax

    def show_collection(self, prop, fig=None, ax=None, label=True, dim="2d", **kwargs):
        """
        Shows a collection on the grid as square patches.

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
            fig, ax = plt.subplots(1)
            ax.set_xlim(-offset, self.p.x_size*self.p.blocksize+offset)
            ax.set_ylim(-offset, self.p.y_size*self.p.blocksize+offset)
        coll = getattr(self, prop)
        for pt in coll:
            corner = pt - np.array([offset, offset])
            rect = Rectangle(corner, self.p.blocksize, self.p.blocksize,
                             label=prop, **kwargs)
            ax.add_patch(rect)
            if label:
                if type(label) != str:
                    lab = rect.get_label()
                else:
                    lab = label
                ax.text(pt[0], pt[1], lab,
                        horizontalalignment="center", verticalalignment="center")
        return fig, ax

    def show(self, prop, collections={}, dim="2d", **kwargs):
        """
        Plots a property and set of collections on the grid.

        Parameters
        ----------
        prop : str
            Property to plot.
        collections : dict, optional
            Collections to plot and their respective kwargs for show_collection.
            The default is {}.
        **kwargs : kwargs
            kwargs to show_property.

        Returns
        -------
        fig : mpl.figure
            Plotted figure object
        ax : mpl.axis
            Ploted axis object.
        """
        fig, ax = self.show_property(prop, dim=dim, **kwargs)
        for coll in collections:
            self.show_collection(coll, fig=fig, ax=ax, dim=dim, **collections[coll])
        return fig, ax

def as_voxels(array, res=10):
    from matplotlib import colormaps
    cmap = colormaps['viridis']
    
    dims = array.shape
    X, Y, Z = np.indices((dims[0], dims[1], res))
    z_shape = Z.swapaxes(0, 2).swapaxes(1, 2)
    shape = z_shape < array
    shape = shape.swapaxes(0, 1).swapaxes(1, 2)
    
    norm= plt.Normalize(z_shape.min(), z_shape.max())
    colors = cmap(norm(z_shape)).swapaxes(0, 1).swapaxes(1, 2) # .swapaxes(0, 1).swapaxes(1, 2)
    #colors[shape] = cmap(Z)
    #colors = np.array([z_shape for i in range(res)])
    #shape = np.swapaxes(X < array, 0, 2)
    #shape = np.swapaxes(shape, 0, 1)
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(shape, edgecolor='k', facecolors = colors) #, facecolors=colors, edgecolor='k')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.set_zlim(0, 100)
    
    
    

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
    ex = ExampleGrid()
    ex.show_property("v", cmap="Greys")
    ex.show_collection("high_v")
    ex.show("h", collections={"high_v":{"alpha":0.5, "color":"red"}})
    as_voxels(ex.v)
    
    ex.show_property("v", cmap="Greys", dim="3d")
    import doctest
    doctest.testmod(verbose=True)
