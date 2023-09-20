# -*- coding: utf-8 -*-
"""
Now:
    Static Geoms, with properties tied to parameters and states representing
    allocations.
Future:
    Dynamic Geoms, with properties tied to states
"""
from shapely import LineString, Point, Polygon, GeometryCollection
import numpy as np
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.common import init_obj_attr, get_true_fields
from fmdtools.define.environment import init_obj_dict
from matplotlib import pyplot as plt


def sin_func(x, amp, period):
    return amp * np.sin(x * 2 *np.pi / period)

xy = [[x, sin_func(x, 1,1)] for x in np.arange(0,100, 1)]

ls_xy = LineString(xy)


class Geom(object):
    _init_s = State

    def __init__(self, *args, s={}, p={}, **kwargs):
        init_obj_attr(self, s=s, p=p)
        self.shape = self.shapely_class(self.p.as_args())
        init_obj_dict(self, "buffer")
        for b, dist in self.buffers.items():
            setattr(self, b, self.shape.buffer(dist))

    def at(self, buffername, *pt):
        buffer = getattr(self, buffername)
        return buffer.covers(Point(pt))


class PointParam(Parameter):
    x: float = 0.0
    y: float = 0.0

    def as_args(self):
        return [self[i] for i in ['x', 'y', 'z'] if i in self.__fields__]


class GeomPoint(Geom):
    _init_p = PointParam
    shapely_class = Point

class StartParam(PointParam):
    buffer_on = 1.0
    buffer_near = 2.0

class Start(GeomPoint):
    _init_p = StartParam


# [0.0, 0.0], [0.0, 1.0] example
class LineParam(Parameter):
    xys: tuple = ()

    def as_args(self):
        return self.xys


class GeomLine(Geom):
    _init_p = LineParam
    shapely_class = LineString

class PathParam(LineParam):
    buffer_on: float = 0.1
    buffer_poor: float = 0.2
    buffer_near: float = 0.3

class PathLine(GeomLine):
    _init_p = PathParam

# [0.0, 0.0], [0.0, 1.0], [1.0, 0.0] example
class PolyParam(Parameter):
    shell: tuple = ()
    holes: tuple = ()

    def as_args(self):
        return self.shell, self.holes


class GeomPoly(Geom):
    _init_p = PolyParam



a = Start()

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



class LineParam(Parameter):
    amp: float = 1.0
    period: float = 2 * np.pi
    x_min: float = 0.0
    x_max: float = 10.0
    x_res: float = 0.1


class LineGeomArch(GeomArchitecture):
    _init_p = LineParam

    def init_shapes(self, **kwargs):
        ls = tuple([[x, sin_func(x, self.p.amp, self.p.period)]
              for x in np.arange(self.p.x_min, self.p.x_max, self.p.x_res)])
        self.add_shape('line', PathLine, p={'xys': ls})
        self.add_shape('start', Start, p={'x': ls[0][0], 'y': ls[0][1]})
        self.add_shape('end', Start, p={'x': ls[-1][0], 'y': ls[-1][1]})

lf = LineGeomArch()

plt.plot(*lf.line.near.exterior.xy)
plt.plot(*lf.line.on.exterior.xy)


plt.plot(np.array([*lf.line.shape.coords])[:,0],
         np.array([*lf.line.shape.coords])[:,1])
plt.scatter(lf.start.shape.x, lf.start.shape.y)
plt.scatter(lf.end.shape.x, lf.end.shape.y)