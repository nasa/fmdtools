# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:26:40 2023

@author: dhulse
"""
import os
import unittest
from examples.pump.ex_pump import Pump
from fmdtools.analyze.graph import Graph, ModelGraph, ModelFxnGraph, ModelFlowGraph, ModelTypeGraph
from fmdtools.analyze.plot import suite_for_plots
from fmdtools.sim import propagate
import unittest

class ModelGraphTests(unittest.TestCase):
    def setUp(self):
        self.mdl = Pump()
    def test_modelgraph_plot(self):
        a= ModelGraph(self.mdl)
        a.draw()
    def test_fxngraph_plot(self):
        a = ModelFxnGraph(self.mdl)
        #a.set_edge_labels(title='label', subtext='flows')
        a.draw()
    def test_flowgraph_plot(self):
        a = ModelFlowGraph(self.mdl)
        #a.set_edge_labels(title='label', subtext='functions')
        a.draw()
    def test_typegraph_plot(self):
        a = ModelTypeGraph(self.mdl)
        a.draw()
    def test_fault_plot(self):
        endresults, mdlhist=propagate.one_fault(self.mdl, 'move_water', 'short', time=10, 
                                        desired_result=['graph','endclass','endfaults'])
        endresults.graph.set_node_styles(degraded={}, faulty={})
        endresults.graph.draw(title="Should show Faults (color edges) as well as degradations (orange color)")

#def test_move_nodes(self):
#   p = endresults.graph.move_nodes()
    
if __name__ == '__main__':
    
    
    #runner = unittest.TextTestRunner()
    #runner.run(suite_for_plots(ModelGraphTests, plottests=False))
    #runner.run(suite_for_plots(ModelGraphTests, plottests=True))
    
    mdl = Pump()
    endresults, mdlhist=propagate.one_fault(mdl, 'move_water', 'short', time=10, 
                                    desired_result=['graph','endclass','endfaults'],
                                    track='all')
    #endresults.graph.set_node_styles(degraded={}, faulty={})
    #endresults.graph.set_node_labels(title='id', subtext='faults')
    #endresults.graph.draw()
    
    a= ModelGraph(mdl)
    #a.draw_from(10, mdlhist)
    #a.draw_from(50, mdlhist)
    
    an = a.animate_from(mdlhist)
    from IPython.display import HTML
    HTML(an.to_jshtml())
    
    #p = endresults.graph.move_nodes()