# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:32:18 2020

@author: Daniel Hulse
"""

import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quad_mdl import *
import time

#scenlist=fp.listinitfaults(graph, mdl.times)
mdl = Quadrotor()

app = SampleApproach(mdl)
#endclasses, mdlhists = fp.run_approach(mdl, app)

#endclasses, mdlhists = fp.run_approach_parallel(mdl, app)

#NOTE: multiprocessing, pathos, joblib don't seem to work for this. I get pickle errors every time
# 

# Ray may be able to solve these problems? - it seems to be tailored to a similar application
#   - see: https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
#   - and: https://ray.readthedocs.io/en/latest/walkthrough.html

# If not, try dispy? http://dispy.sourceforge.net/