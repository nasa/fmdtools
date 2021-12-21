# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:31:47 2020

@author: hulsed
"""

import sys
sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.faultsim.networks as networks
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_mdl import *
import time

#scenlist=fp.listinitfaults(graph, mdl.times)
mdl = Drone()

pos = rd.graph.set_pos(mdl, gtype='bipartite')

bridgingNodes, fig, ax = networks.find_bridging_nodes(mdl, plot='on', gtype='normal')

plt.figure()
highdegreeNodes, fig, ax = networks.find_high_degree_nodes(mdl, plot='on', gtype='normal')

app = SampleApproach(mdl, faults='single-component')

