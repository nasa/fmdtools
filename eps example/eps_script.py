# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:46:24 2020

@author: hulsed
"""
import sys
sys.path.append('../')
from eps import EPS
import fmdtools.faultprop as fp
import fmdtools.resultproc as rp

mdl= EPS()
rp.show_bipartite(mdl.bipartite)

endresults,resgraph, mdlhists = fp.run_one_fault(mdl, 'EE_to_ME', 'toohigh_torque')
rp.show_graph(resgraph)


endclasses, mdlhists = fp.run_list(mdl)

reshists, diffs, summary = rp.compare_hists(mdlhists)

sumtable = rp.make_summarytable(summary)


degtimemap = rp.make_avgdegtimeheatmap(reshists)

rp.show_bipartite(mdl.bipartite,heatmap=degtimemap)
rp.show_graph(resgraph,heatmap=degtimemap)