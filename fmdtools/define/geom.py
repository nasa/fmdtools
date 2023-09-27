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

class Geom(object):
    """
    Base class for defining geometry.

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
        self.shape = self.shapely_class(self.p.as_args())
        init_obj_dict(self, "buffer")
        for b, dist in self.buffers.items():
            setattr(self, b, self.shape.buffer(dist))

    def at(self, buffername, *pt):
        """
        Determine whether the point x, y is within the buffer 'buffername'.

        Parameters
        ----------
        buffername : str
            Name of buffer property
        *pt : numbers
            Locations of x, y, z coordinates.

        Returns
        -------
        at : bool
            Whether x,y is within the buffer.
        """
        buffer = getattr(self, buffername)
        return buffer.covers(Point(pt))


class PointParam(Parameter):
    """
    Parameter defining points. May be extended with buffers.

    Fields
    ------
    x : float
        x-location of point.
    y : float
        y-location of point.
    """

    x: float = 0.0
    y: float = 0.0

    def as_args(self):
        """Create arguments for shapely Point class based on fields."""
        return [self[i] for i in ['x', 'y', 'z'] if i in self.__fields__]


class GeomPoint(Geom):
    _init_p = PointParam
    shapely_class = Point


# [0.0, 0.0], [0.0, 1.0] example
class LineParam(Parameter):
    xys: tuple = ()

    def as_args(self):
        return self.xys


class GeomLine(Geom):
    _init_p = LineParam
    shapely_class = LineString


class PolyParam(Parameter):
    shell: tuple = ()
    holes: tuple = ()

    def as_args(self):
        return self.shell, self.holes


class GeomPoly(Geom):
    _init_p = PolyParam
    shapely_class = Polygon


class GeomArchitecture(object):
    _init_p = Parameter

    def __init__(self, *args, p={}, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.points = {}
        self.lines = {}
        self.polys = {}
        init_obj_attr(self, p=p)
        self.init_shapes(**kwargs)

    def init_shapes(self, **kwargs):
        a = 1

    def add_shape(self, name, sclass, *args, **kwargs):
        setattr(self, name, sclass(*args, **kwargs))
        if issubclass(sclass, GeomPoint):
            self.points[name] = getattr(self, name)
        elif isinstance(sclass, GeomLine):
            self.lines[name] = getattr(self, name)
        elif isinstance(sclass, GeomPoly):
            self.polys[name] = getattr(self, name)
