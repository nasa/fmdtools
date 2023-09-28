# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from fmdtools.define.geom import PointParam, GeomPoint, LineParam, GeomLine
from fmdtools.define.geom import GeomArch
from fmdtools.define.parameter import Parameter
from matplotlib import pyplot as plt


def sin_func(x, amp, period):
    return amp * np.sin(x * 2 * np.pi / period)


#ls_xy = LineString(xy)

class StartParam(PointParam):
    x: float = 0.0
    y: float = 0.0
    buffer_on = 1.0
    buffer_near = 2.0

class Start(GeomPoint):
    _init_p = StartParam

sp = Start()


class PathParam(LineParam):
    xys: tuple = tuple([[x, sin_func(x, 1,1)] for x in np.arange(0,100, 1)])
    buffer_on: float = 0.1
    buffer_poor: float = 0.2
    buffer_near: float = 0.3


class PathLine(GeomLine):
    _init_p = PathParam


path = PathLine()


class LineParam(Parameter):
    amp: float = 1.0
    period: float = 2 * np.pi
    x_min: float = 0.0
    x_max: float = 10.0
    x_res: float = 0.1


class LineGeomArch(GeomArch):
    _init_p = LineParam

    def init_geoms(self, **kwargs):
        ls = tuple([[x, sin_func(x, self.p.amp, self.p.period)]
              for x in np.arange(self.p.x_min, self.p.x_max, self.p.x_res)])
        self.add_geom('line', PathLine, p={'xys': ls})
        self.add_geom('start', Start, p={'x': ls[0][0], 'y': ls[0][1]})
        self.add_geom('end', Start, p={'x': ls[-1][0], 'y': ls[-1][1]})


if __name__ == "__main__":

    lf = LineGeomArch()
    
    plt.plot(*lf.geoms['line'].near.exterior.xy)
    plt.plot(*lf.geoms['line'].on.exterior.xy)
    
    
    plt.plot(np.array([*lf.geoms['line'].shape.coords])[:,0],
             np.array([*lf.geoms['line'].shape.coords])[:,1])
    plt.scatter(lf.geoms['start'].shape.x, lf.geoms['start'].shape.y)
    plt.scatter(lf.geoms['end'].shape.x, lf.geoms['end'].shape.y)
