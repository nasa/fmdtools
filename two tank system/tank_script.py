# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:08:05 2020

@author: Daniel Hulse
"""

import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
from tank_model import Tank



mdl = Tank()

endresults, resgraph, mdlhist = fp.run_nominal(mdl)

rp.plot_mdlhistvals(mdlhist)
rp.show_graph(resgraph)