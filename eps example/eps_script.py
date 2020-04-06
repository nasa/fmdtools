# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:46:24 2020

@author: hulsed
"""
import sys
sys.path.append('../')
from eps import EPS
import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd

mdl= EPS()
rd.graph.show(mdl.bipartite, gtype='bipartite')

endresults,resgraph, mdlhists = propagate.one_fault(mdl, 'EE_to_ME', 'toohigh_torque')
rd.graph.show(resgraph)


endclasses, mdlhists = propagate.single_faults(mdl)

reshists, diffs, summary = rd.process.hists(mdlhists)

sumtable = rd.tabulate.summary(summary)


degtimemap = rd.process.avgdegtimeheatmap(reshists)

rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=degtimemap)
rd.graph.show(resgraph,heatmap=degtimemap)

endclasses, mdlhists = propagate.single_faults(mdl)