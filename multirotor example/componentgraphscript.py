# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 13:37:00 2020

@author: hulsed
"""
import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
import matplotlib.pyplot as plt

from quad_mdl import *

mdl = Quadrotor()

g=mdl.return_componentgraph('AffectDOF')
rp.show_bipartite(g)

g1 = mdl.return_stategraph(gtype='component')

endresults, resgraph, mdlhist = fp.run_one_fault(mdl, 'DistEE', 'short', time=5, staged=True, gtype='component')
rp.show_bipartite(resgraph, faultscen='DistEE short', time=5, showfaultlabels=False)