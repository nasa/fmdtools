# -*- coding: utf-8 -*-
"""

"""

import numpy as np

from fmdtools.define.geom import PointParam, GeomPoint, LineParam, GeomLine
from fmdtools.define.geom import GeomArchitecture
from fmdtools.define.parameter import Parameter
from matplotlib import pyplot as plt


def sin_func(x, amp, period):
    return amp * np.sin(x * 2 * np.pi / period)

xy = [[x, sin_func(x, 1,1)] for x in np.arange(0,100, 1)]

#ls_xy = LineString(xy)




class StartParam(PointParam):
    buffer_on = 1.0
    buffer_near = 2.0

class Start(GeomPoint):
    _init_p = StartParam





class PathParam(LineParam):
    buffer_on: float = 0.1
    buffer_poor: float = 0.2
    buffer_near: float = 0.3

class PathLine(GeomLine):
    _init_p = PathParam

# [0.0, 0.0], [0.0, 1.0], [1.0, 0.0] example


a = Start()


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
