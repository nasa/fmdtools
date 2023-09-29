# -*- coding: utf-8 -*-
"""

"""
import numpy as np


from fmdtools.define.geom import GeomArch
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from matplotlib import pyplot as plt





path = PathLine()





if __name__ == "__main__":

    lf = GroundGeomArch(p={'linetype': 'turn', 'x_max': 30.0})
    lf = GroundGeomArch()
    
    plt.plot(*lf.geoms['line'].near.exterior.xy)
    plt.plot(*lf.geoms['line'].on.exterior.xy)
    
    
    plt.plot(np.array([*lf.geoms['line'].shape.coords])[:,0],
             np.array([*lf.geoms['line'].shape.coords])[:,1])
    plt.scatter(lf.geoms['start'].shape.x, lf.geoms['start'].shape.y)
    plt.scatter(lf.geoms['end'].shape.x, lf.geoms['end'].shape.y)
    
    from fmdtools.analyze import show
    fig, ax = show.geom(lf.geoms['line'], shapes = {'all'}, geomlabel='line')
    fig, ax = show.geom(lf.geoms['start'], shapes = {'all'}, ax=ax, fig=fig, geomlabel='start')
    fig, ax = show.geom(lf.geoms['end'], shapes = {'all'}, ax=ax, fig=fig, geomlabel='end')
    
    fig, ax = show.geomarch(lf)
    
