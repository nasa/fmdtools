# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:36:41 2020

@author: Daniel Hulse
"""

import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp

from disease_model import *

dm1 = DiseaseModel()

rp.show_graph(dm1.graph)

endresults, resgraph, mdlhist = fp.run_nominal(dm1)

rp.plot_mdlhist(mdlhist)