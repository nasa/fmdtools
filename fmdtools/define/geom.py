# -*- coding: utf-8 -*-
"""
Now:
    Static Geoms, with properties tied to parameters and states representing
    allocations.
Future:
    Dynamic Geoms, with properties tied to states
"""
from fmdtools.define.common import init_obj_attr, init_obj_dict
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State

from shapely import LineString, Point, Polygon
from typing import ClassVar


class Geom(object):
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
    _init_s = State
    _init_p = Parameter

    def __init__(self, *args, s={}, p={}, **kwargs):
        init_obj_attr(self, s=s, p=p)
        self.shape = self.shapely_class(*self.p.as_args())
        init_obj_dict(self, "buffer")
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

        e.g.,::
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
        self.s = self._init_s(**self._args_s)


class PointParam(Parameter):
    """
    Parameter defining points. May be extended with buffers.

    Fields
    ------
    x : float
        x-location of point.
    y : float
        y-location of point.

    e.g., a point with center at 1.0, 1.0 and radius of 1.0 definining whether something
    is on the point would be defined by the parameter::
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
        return ([self[i] for i in ['x', 'y', 'z'] if i in self.__fields__], )


class GeomPoint(Geom):
    """
    Point geometry representing x-y coordinate and buffers.

    Defined by parameter (x, y and buffer(s)) as well as states (properties of
    of point). e.g.::
    >>> class ExPoint(GeomPoint):
    ...    _init_p = ExPointParam
    ...    _init_s = ExGeomState
    >>> exp = ExPoint()
    >>> exp.at((1.0, 1.0), "on")
    True
    >>> exp.at((0.0, 0.0), "on")
    False
    >>> exp.at((1.0, 0.1), "on")
    True

    Additionally, note the underlying shapely attribute at .shape::
    >>> type(exp.shape)
    <class 'shapely.geometry.point.Point'>

    as well as the buffer (on)::
    >>> type(exp.on)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    _init_p = PointParam
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

    _init_p = ExPointParam
    _init_s = ExGeomState


class LineParam(Parameter):
    """
    Parameter defining lines. May be extended with buffers.

    Fields
    ------
    xys : tuple
        tuple of points ((x1, y1), ...) defining shapely.LineString.

    e.g., a point with ends at (0.0, 0.0) and (1.0, 1.0) and radius of 1.0 defining
    whether something is on the line would be defined by the parameter::
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
    of point). e.g.::
    >>> class ExLine(GeomLine):
    ...    _init_p = ExLineParam
    ...    _init_s = ExGeomState
    >>> exp = ExLine()
    >>> exp.at((1.0, 1.0), "on")
    True
    >>> exp.at((0.0, 0.0), "on")
    True
    >>> exp.at((2.0, 2.0), "on")
    False

    Additionally, note the underlying shapely attribute at .shape::
    >>> type(exp.shape)
    <class 'shapely.geometry.linestring.LineString'>

    as well as the buffer (on)::
    >>> type(exp.on)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    _init_p = LineParam
    shapely_class = LineString


class ExLine(GeomLine):
    """Example GeomLine to use in testing."""

    _init_p = ExLineParam
    _init_s = ExGeomState


class PolyParam(Parameter):
    """
    Parameter defining polygons. May be extended with buffers.

    Fields
    ------
    shell : tuple
        tuple of points ((x1, y1), ...) defining outer shell.
    holes : tuple
        tuple of points defining holes.

    e.g., the following PolyParam defines a hollow right triangle::
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
    contain a state for the given status (e.g., occupied, red/blue, etc.). e.g.::
    >>> class ExPoly(GeomPoly):
    ...    _init_p = ExPolyParam
    ...    _init_s = ExGeomState
    >>> egp = ExPoly()
    >>> egp.at((0.1, 0.05))
    True
    >>> egp.at((0.4, 0.3))
    False

    Additionally, note the underlying shapely attribute at .shape::
    >>> type(egp.shape)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    _init_p = PolyParam
    shapely_class = Polygon


class ExPoly(GeomPoly):
    """Example Polygon for use in testing."""

    _init_p = ExPolyParam
    _init_s = ExGeomState


class GeomArchitecture(object):
    """
    Agglomeration of multiple geoms/shapes.

    Architecture is defined using add_shape method in user-defined init_shapes method.
    e.g., for an architecture with the geoms already defined::
    >>> class ExGeomArch(GeomArchitecture):
    ...    def init_shapes(self):
    ...        self.add_geom("ex_point", ExPoint)
    ...        self.add_geom("ex_line", ExLine)
    ...        self.add_geom("ex_poly", ExPoly)

    This can then be used in containing classes (e.g., environments) that need multiple
    geoms. We can then access the individual geoms in the shapes dict, e.g.:...
    >>> ega = ExGeomArch()
    >>> ega.shapes['ex_point'].s
    ExGeomState(occupied=False)
    """

    _init_p = Parameter

    def __init__(self, *args, p={}, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.points = []
        self.lines = []
        self.polys = []
        self.shapes = {}
        init_obj_attr(self, p=p)
        self.init_shapes(**kwargs)

    def init_shapes(self, **kwargs):
        """Use this placeholder method to define custom shape architectures."""
        a = 1

    def add_geom(self, name, gclass, *args, **kwargs):
        """
        Add/instantiate an individual shape to the overall architecture.

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
        self.shapes[name] = getattr(self, name)

    def copy(self):
        """Copy shapes in the architecture (mirrors current states)."""
        cop = self.__class__()
        for shape in self.shapes:
            cop.shapes[shape].s.assign(self.shapes[shape].s)
        return cop

    def reset(self):
        for shape in self.shapes:
            self.shapes[shape].reset()

    def all_at(self, *pt):
        """
        Find all shapes (and buffers) a given is at.

        Parameters
        ----------
        *pt : x,y
            x, y, z location to check.

        Returns
        -------
        all_at : dict
            Names of shapes where the point is at (and their properties)

        e.g.,::
        >>> exga = ExGeomArch()
        >>> exga.all_at(1.0, 1.0)
        {'ex_point': ['shape', 'on'], 'ex_line': ['shape', 'on'], 'ex_poly': ['shape']}
        >>> exga.all_at(0.0, 0.0)
        {'ex_line': ['shape', 'on'], 'ex_poly': ['shape']}
        >>> exga.all_at(0.4, 0.3)
        {'ex_point': ['on'], 'ex_line': ['on']}
        """
        all_at = {}
        for shapename, shape in self.shapes.items():
            at_shape = shape.all_at(*pt)
            if at_shape:
                all_at[shapename] = at_shape
        return all_at


class ExGeomArch(GeomArchitecture):
    """Example Geometric Architecture for testing etc."""

    def init_shapes(self):
        """Initialize example shapes."""
        self.add_geom("ex_point", ExPoint)
        self.add_geom("ex_line", ExLine)
        self.add_geom("ex_poly", ExPoly)
    

        

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    exp = ExPoint()
    ega = ExGeomArch()
