# -*- coding: utf-8 -*-
"""
Now:
    Static Geoms, with properties tied to parameters and states representing
    allocations.
Future:
    Dynamic Geoms, with properties tied to states
"""
from fmdtools.define.base import BaseObject
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist
from fmdtools.analyze.common import setup_plot, consolidate_legend

from shapely import LineString, Point, Polygon
from shapely.ops import nearest_points
from typing import ClassVar
from recordclass import astuple, asdict
import numpy as np


class Geom(BaseObject):
    """
    Base class for defining geometries.

    Geometry objects are essentially wrappers around various shapely classes used to
    convey the spacial properties of a model.

    Roles
    -----
    s : State
        State defining mutable geom properties (e.g., allocation, color, etc)
    p : Param
        Parameter defining immutable geom characteristics (e.g., shapely inputs, buffer)
    """

    container_s = State
    container_p = Parameter
    default_track = ['s']
    all_possible = ['s']

    def __init__(self, *args, s={}, p={}, **kwargs):
        super().__init__(*args, s=s, p=p)
        self.shape = self.shapely_class(*self.p.as_args())
        self.init_dict("buffer")
        for b, dist in self.buffers.items():
            setattr(self, b, self.shape.buffer(dist))

    def at(self, pt, buffername='shape'):
        """
        Determine whether the point x, y is within the buffer 'buffername'.

        Parameters
        ----------
        *pt : tuple
            Locations of x, y, z coordinates.
        buffername : str
            Name of buffer property

        Returns
        -------
        at : bool
            Whether x,y is within the buffer.
        """
        buffer = getattr(self, buffername)
        return buffer.covers(Point(*pt))

    def all_at(self, *pt):
        """
        Find all geom attributes (shape, buffers, etc.) containing the point.

        Parameters
        ----------
        *pt : x,y,z
            Locations of x, y, z coordinates.

        Returns
        -------
        all_at : list
            Buffer/shape containing the pt.

        Examples
        --------
        >>> exp = ExPoint()
        >>> exp.all_at(1.0, 1.0)
        ['shape', 'on']
        >>> exp.all_at(1.0, 0.1)
        ['on']
        >>> exp.all_at(0.0, 0.0)
        []
        """
        buffernames = ['shape', *self.buffers]
        all_at = []
        for bname in buffernames:
            if self.at(pt, bname):
                all_at.append(bname)
        return all_at

    def copy(self, *args, **kwargs):
        """Copy the Geom with given *args and **kwargs."""
        cop = self.__class__(*args, **kwargs)
        cop.s.assign(self.s)
        return cop

    def reset(self):
        """Reset the Geom to initial state."""
        self.s = self.container_s(**self._args_s)

    def return_mutables(self):
        return astuple(self.s)

    def create_hist(self, timerange, track):
        track = self.get_track(track, all_possible=self.all_possible)
        h = History()
        for att in track:
            att_track = get_sub_include(att, track)
            val = getattr(self, att)
            h[att] = val.create_hist(timerange, att_track)
        return h

    def vect_to_shape(self, pt, buffername='shape'):
        """
        Gets the vector (x, y) to a given shape.

        Parameters
        ----------
        pt : tuple/list
            Point to get vector from
        buffername : str
            Name of shape/buffer. Default is 'shape'.

        Examples
        --------
        >>> e = ExPoint()
        >>> e.vect_to_shape((0,0))
        array([[1.],
               [1.]])
        >>> e.vect_to_shape((2,0))
        array([[-1.],
               [ 1.]])
        >>> e.vect_to_shape((1,1))
        array([[0.],
               [0.]])
        """
        buffer = getattr(self, buffername)
        geom_pt = Point(pt)
        geom_c, close_pt = nearest_points(geom_pt, buffer)
        vect_to_shape = np.array(close_pt.xy) - np.array(geom_pt.xy)
        return vect_to_shape

    def vect_at_shape(self, pt, buffername='shape', dist_forward=0.1):
        """
        Get the vector (x, y) at a given shape (e.g., direction of a line at pt).

        Parameters
        ----------
        pt : tuple/lost
            Point closest to shape
        buffername : str
            Name of shape/buffer. Default is 'shape'.
        dist_forward : float
            Distance forward along line segment to project. Give a negative number to
            reverse directions

        Examples
        --------
        >>> e=ExLine(p={'xys':((0,0),(1,1), (1,0))})
        >>> e.vect_at_shape((0,0))
        array([[0.07071068],
               [0.07071068]])
        >>> e.vect_at_shape((1.5,0.5))
        array([[ 0. ],
               [-0.1]])
        """
        buffer = getattr(self, buffername)
        geom_pt = Point(pt)
        geom_c, close_pt = nearest_points(geom_pt, buffer)
        line_dist = buffer.line_locate_point(geom_pt)
        next_pt = buffer.line_interpolate_point(line_dist + dist_forward)
        vect_at_shape = np.array(next_pt.xy) - np.array(close_pt.xy)
        return vect_at_shape

    def show(self, shapes={'all': {}}, fig=None, ax=None, figsize=(4, 4), z=False,
             geomlabel='', **kwargs):
        """
        Show a Geom (shape and buffers) as lines on a plot.

        Parameters
        ----------
        geom : Geom
            Geom object.
        shapes : dict, optional
            Aspects of the Geom to plot and their corresponding plot kwargs.
            The default is {'all': {}}.
        fig : matplotlib.figure, optional
            Existing Figure. The default is None.
        ax : matplotlib.axis, optional
            Existing axis. The default is None.
        figsize : tuple, optional
            Size for figure (if instantiating). The default is (4, 4).
        z : bool/number, optional
            If plotting on a 3d axis, set z to a number which will be the z-level.
            The default is False.
        geomlabel : str, optional
            Overall label for the geom (if desired). The default is ''.
        **kwargs : kwargs
            overall kwargs for plt.plot for all shapes.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        if not ax:
            fig, ax = setup_plot(z=z, figsize=figsize)
        if 'all' in shapes:
            shapes = {'shape': {}, **{v: {} for v in self.buffers}}
        if type(z) in (int, float):
            plot_kwargs = {'zs': z, 'zdir': 'z', **kwargs}
        else:
            plot_kwargs = kwargs
        for shape, shape_kwargs in shapes.items():
            if geomlabel:
                shape_label = geomlabel + "." + shape
            else:
                shape_label = shape
            local_kwargs = {**plot_kwargs, 'label': shape_label, **shape_kwargs}
            shap = getattr(self, shape)
            if isinstance(shap, Point):
                ax.scatter(shap.x, shap.y, **local_kwargs)
            elif isinstance(shap, LineString):
                linecoords = np.array([*shap.coords])
                ax.plot(linecoords[:, 0], linecoords[:, 1], **local_kwargs)
            elif isinstance(shap, Polygon):
                ax.plot(*shap.exterior.xy, **local_kwargs)
        ax.axis('equal')
        consolidate_legend(ax, **kwargs)
        return fig, ax


class PointParam(Parameter):
    """
    Parameter defining points. May be extended with buffers.

    Fields
    ------
    x : float
        x-location of point.
    y : float
        y-location of point.

    Examples
    --------
    a point with center at 1.0, 1.0 and radius of 1.0 defining whether something
    is on the point would be defined by the parameter:

    >>> class ExPointParam(PointParam):
    ...    x: float = 1.0
    ...    y: float = 1.0
    ...    buffer_on: float = 1.0

    >>> expa = ExPointParam()
    >>> expa.as_args()
    ([1.0, 1.0],)
    """

    x: ClassVar[float] = 0.0
    y: ClassVar[float] = 0.0

    def as_args(self):
        """Create arguments for shapely Point class based on fields."""
        return ([self[i] for i in ['x', 'y', 'z'] if i in dir(self)], )


class GeomPoint(Geom):
    """
    Point geometry representing x-y coordinate and buffers.

    Defined by parameter (x, y and buffer(s)) as well as states (properties of
    of point).

    Examples
    --------
    >>> class ExPoint(GeomPoint):
    ...    container_p = ExPointParam
    ...    container_s = ExGeomState
    >>> exp = ExPoint()
    >>> exp.at((1.0, 1.0), "on")
    True
    >>> exp.at((0.0, 0.0), "on")
    False
    >>> exp.at((1.0, 0.1), "on")
    True

    Additionally, note the underlying shapely attribute at .shape:

    >>> type(exp.shape)
    <class 'shapely.geometry.point.Point'>

    as well as the buffer (on):

    >>> type(exp.on)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    container_p = PointParam
    shapely_class = Point


class ExGeomState(State):
    """
    Example Geom state for testing.

    Has 'occupied' property defining whether something is at the geom or not.
    """

    occupied: bool = False


class ExPointParam(PointParam):
    """
    Example point parameter for testing.

    Has a default center at (1.0, 1.0) and radius defining "on" of 1.0.
    """

    x: float = 1.0
    y: float = 1.0
    buffer_on: float = 1.0


class ExPoint(GeomPoint):
    """Example point for testing."""

    container_p = ExPointParam
    container_s = ExGeomState


class LineParam(Parameter):
    """
    Parameter defining lines. May be extended with buffers.

    Fields
    ------
    xys : tuple
        tuple of points ((x1, y1), ...) defining shapely.LineString.

    Examples
    --------
    A point with ends at (0.0, 0.0) and (1.0, 1.0) and radius of 1.0 defining
    whether something is on the line would be defined by the parameter:

    >>> class ExLineParam(LineParam):
    ...     xys: tuple = ((0.0, 0.0), (1.0, 1.0))
    ...     buffer_on: float = 1.0
    >>> exlp = ExLineParam()
    >>> exlp.as_args()
    (((0.0, 0.0), (1.0, 1.0)),)
    """

    xys: ClassVar[tuple] = ()

    def as_args(self):
        """Create arguments for shapely LineString class based on fields."""
        return (self.xys, )


class ExLineParam(LineParam):
    """Example parameter defining a line with a given buffer 'on'."""

    xys: tuple = ((0.0, 0.0), (1.0, 1.0))
    buffer_on: float = 1.0


class GeomLine(Geom):
    """Point geometry representing a line and possible buffers.

    Defined by parameter (xys and buffer(s)) as well as states (properties of
    of point).

    Examples
    --------
    >>> class ExLine(GeomLine):
    ...    container_p = ExLineParam
    ...    container_s = ExGeomState
    >>> exp = ExLine()
    >>> exp.at((1.0, 1.0), "on")
    True
    >>> exp.at((0.0, 0.0), "on")
    True
    >>> exp.at((2.0, 2.0), "on")
    False

    Additionally, note the underlying shapely attribute at .shape ::

    >>> type(exp.shape)
    <class 'shapely.geometry.linestring.LineString'>

    As well as the buffer (on)::

    >>> type(exp.on)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    container_p = LineParam
    shapely_class = LineString


class ExLine(GeomLine):
    """Example GeomLine to use in testing."""

    container_p = ExLineParam
    container_s = ExGeomState


class PolyParam(Parameter):
    """
    Parameter defining polygons. May be extended with buffers.

    Fields
    ------
    shell : tuple
        tuple of points ((x1, y1), ...) defining outer shell.
    holes : tuple
        tuple of points defining holes.

    Examples
    --------
    The following PolyParam defines a hollow right triangle:

    >>> class ExPolyParam(PolyParam):
    ...     shell: tuple = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    ...     holes: tuple = (((0.3, 0.2), (0.6, 0.2), (0.6, 0.5)), )
    >>> expp = ExPolyParam()
    >>> expp.as_args()
    (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), (((0.3, 0.2), (0.6, 0.2), (0.6, 0.5)),))
    """

    shell: ClassVar[tuple] = ()
    holes: ClassVar[tuple] = ()

    def as_args(self):
        """Create arguments for shapely Polygon class based on fields."""
        return self.shell, self.holes


class ExPolyParam(PolyParam):
    """Example polygon parameter defining a basic triangle for use in testing."""

    shell: tuple = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    holes: tuple = (((0.3, 0.2), (0.6, 0.2), (0.6, 0.5)), )


class GeomPoly(Geom):
    """
    Polygon geometry defining shape and possible buffers.

    Defined by PolyParam, which is used to instantiate the Polygon class. Also may
    contain a state for the given status (e.g., occupied, red/blue, etc.).

    Examples
    --------
    >>> class ExPoly(GeomPoly):
    ...    container_p = ExPolyParam
    ...    container_s = ExGeomState
    >>> egp = ExPoly()
    >>> egp.at((0.1, 0.05))
    True
    >>> egp.at((0.4, 0.3))
    False

    Additionally, note the underlying shapely attribute at .shape:

    >>> type(egp.shape)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    container_p = PolyParam
    shapely_class = Polygon


class ExPoly(GeomPoly):
    """Example Polygon for use in testing."""

    container_p = ExPolyParam
    container_s = ExGeomState


class GeomArch(BaseObject):
    """
    Agglomeration of multiple geoms/shapes.

    Architecture is defined using add_shape method in user-defined init_shapes method.

    Examples
    --------
    for an architecture with the geoms already defined:
    >>> class ExGeomArch(GeomArch):
    ...    def init_geoms(self):
    ...        self.add_geom("ex_point", ExPoint)
    ...        self.add_geom("ex_line", ExLine)
    ...        self.add_geom("ex_poly", ExPoly)

    This can then be used in containing classes (e.g., environments) that need multiple
    geoms. We can then access the individual geoms in the geoms dict, e.g.:

    >>> ega = ExGeomArch()
    >>> ega.geoms['ex_point'].s
    ExGeomState(occupied=False)
    """

    container_p = Parameter
    default_track = ['geoms']
    all_possible = ['geoms']

    def __init__(self, *args, p={}, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.points = []
        self.lines = []
        self.polys = []
        self.geoms = {}
        super().__init__(p=p)
        self.init_geoms(**kwargs)

    def check_role(self, rolename):
        if rolename != 'ga':
            raise Exception("Invalid rolename for GeomArch: "+rolename)

    def init_geoms(self, **kwargs):
        """Use this placeholder method to define custom architectures."""
        a = 1

    def add_geom(self, name, gclass, *args, **kwargs):
        """
        Add/instantiate an individual geom to the overall architecture.

        Parameters
        ----------
        name : str
            Name of the geom object to instantiate.
        gclass : Geom
            Class defining the geom.
        *args : args
            args defining the object for gclass.
        **kwargs : kwargs
            kwargs defining the object for gclass.
        """
        setattr(self, name, gclass(*args, **kwargs))
        if issubclass(gclass, GeomPoint):
            self.points.append(name)
        elif issubclass(gclass, GeomLine):
            self.lines.append(name)
        elif issubclass(gclass, GeomPoly):
            self.polys.append(name)
        elif not issubclass(gclass, Geom):
            raise Exception(name + " gclass " + str(gclass) + " not a Geom")
        self.geoms[name] = getattr(self, name)

    def copy(self):
        """Copy geoms in the architecture (mirrors current states)."""
        cop = self.__class__()
        for geom in self.geoms:
            cop.geoms[geom].s.assign(self.geoms[geom].s)
        return cop

    def reset(self):
        for geom in self.geoms:
            self.geoms[geom].reset()

    def all_at(self, *pt):
        """
        Find all geoms (and buffers) a given is at.

        Parameters
        ----------
        *pt : x,y
            x, y, z location to check.

        Returns
        -------
        all_at : dict
            Names of geoms where the point is at (and their properties)

        Examples
        --------
        >>> exga = ExGeomArch()
        >>> exga.all_at(1.0, 1.0)
        {'ex_point': ['shape', 'on'], 'ex_line': ['shape', 'on'], 'ex_poly': ['shape']}
        >>> exga.all_at(0.0, 0.0)
        {'ex_line': ['shape', 'on'], 'ex_poly': ['shape']}
        >>> exga.all_at(0.4, 0.3)
        {'ex_point': ['on'], 'ex_line': ['on']}
        """
        all_at = {}
        for geomname, geom in self.geoms.items():
            at_geom = geom.all_at(*pt)
            if at_geom:
                all_at[geomname] = at_geom
        return all_at

    def create_hist(self, timerange, track):
        """
        Create history for the architecture.

        Examples
        --------
        >>> ega = ExGeomArch()
        >>> h = ega.create_hist([0.0], 'default')
        >>> h.flatten()
        geoms.ex_point.s.occupied:      array(1)
        geoms.ex_line.s.occupied:       array(1)
        geoms.ex_poly.s.occupied:       array(1)
        """
        track = self.get_track(track, all_possible=self.all_possible)
        hist = History()
        init_indicator_hist(self, hist, timerange, track)
        geoms_track = get_sub_include('geoms', track)
        if geoms_track:
            hist['geoms'] = History()
            for geomname, geom in self.geoms.items():
                sh = geom.create_hist(timerange,
                                      get_sub_include(geomname, geoms_track))
                if sh:
                    hist.geoms[geomname] = sh
        return hist

    def return_states(self):
        states = {}
        for geomname, geom in self.geoms.items():
            states[geomname] = asdict(geom.s)
        return states

    def return_mutables(self):
        """
        Return all mutables (geom states).

        Examples
        --------
        >>> ega = ExGeomArch()
        >>> ega.return_mutables()
        (False, False, False)
        """
        mutes = []
        for geom in self.geoms.values():
            mutes.extend(geom.return_mutables())
        return tuple(mutes)

    def show(self, geoms={'all': {}}, fig=None, ax=None, figsize=(4, 4), z=False,
             **kwargs):
        """
        Show the shapes of a GeomArch all on one plot.

        Parameters
        ----------
        geomarch : GeomArch
            Geometric architecture to plot.
        geoms : dict, optional
            Individual shapes to plot and their corresponding kwargs.
            The default is {'all': {}}.
        fig : matplotlib.figure, optional
            Existing Figure. The default is None.
        ax : matplotlib.axis, optional
            Existing axis. The default is None.
        figsize : tuple, optional
            Size for figure (if instantiating). The default is (4, 4).
        z : bool/number, optional
            If plotting on a 3d axis, set z to a number which will be the z-level.
            The default is False.
        **kwargs : kwargs
            Overall kwargs to show.geom for all geoms.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        if not ax:
            fig, ax = setup_plot(z=z, figsize=figsize)
        if 'all' in geoms:
            geoms = {g: {'shapes': 'all'} for g in self.geoms}

        for geomname, geom_kwargs in geoms.items():
            local_kwargs = {**kwargs, 'geomlabel': geomname, **geom_kwargs}
            fig, ax = self.geoms[geomname].show(ax=ax, fig=fig, z=z, **local_kwargs)
        return fig, ax


class ExGeomArch(GeomArch):
    """Example Geometric Architecture for testing etc."""

    def init_geoms(self):
        """Initialize example geoms."""
        self.add_geom("ex_point", ExPoint)
        self.add_geom("ex_line", ExLine)
        self.add_geom("ex_poly", ExPoly)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    exp = ExPoint()
    ega = ExGeomArch()
