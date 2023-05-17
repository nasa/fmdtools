# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:26:40 2023

@author: dhulse
"""
import os
import unittest
from examples.pump.ex_pump import Pump
from examples.rover.rover_model import Rover
from fmdtools.analyze.graph import Graph, ModelGraph, ModelFxnGraph, ModelFlowGraph, ModelTypeGraph
from fmdtools.analyze.plot import suite_for_plots
from fmdtools.sim import propagate
import unittest

class ModelGraphTests(unittest.TestCase):
    def setUp(self):
        self.mdl = Pump()
        self.rvr = Rover()
    def test_modelgraph_plot(self):
        a= ModelGraph(self.mdl)
        a.draw()
        a.set_exec_order(self.mdl)
        a.draw()
        
        b=ModelGraph(self.rvr)
        b.set_exec_order(self.rvr, next_edges={"edge_color":"red"})
        b.draw(title="Should show Order, timestep, and dynamic properties of ModelGraph with red arrows for next")
        
    def test_fxngraph_plot(self):
        a = ModelFxnGraph(self.mdl)
        #a.set_edge_labels(title='label', subtext='flows')
        a.draw()
        a.set_exec_order(self.mdl)
        a.draw()
        
        b=ModelFxnGraph(self.rvr)
        b.set_exec_order(self.rvr)
        b.draw()
    def test_flowgraph_plot(self):
        a = ModelFlowGraph(self.mdl)
        #a.set_edge_labels(title='label', subtext='functions')
        a.draw()
        a.set_exec_order(self.mdl)
        a.draw()
        
        b=ModelFlowGraph(self.rvr)
        b.set_exec_order(self.rvr)
        b.draw(title="Should show Order, timestep, and dynamic properties of FlowGraph")
    def test_typegraph_plot(self):
        a = ModelTypeGraph(self.mdl)
        a.draw(title="Should show the Pump model Containing functions, which in turn contain Signal, Water, Electricity Flows")
    def test_fault_plot(self):
        endresults, mdlhist=propagate.one_fault(self.mdl, 'move_water', 'short', time=10, 
                                        desired_result=['graph','endclass','endfaults'])
        endresults.graph.set_node_styles(degraded={}, faulty={})
        endresults.graph.draw(title="Should show Faults (color edges) as well as degradations (orange color)")
        
        endresults.graph.set_node_styles(degraded={'node_color':'green'}, faulty={'node_size': 1500, 'edgecolors':'purple'})
        endresults.graph.draw(title="Should be identical but faulty nodes are large and have purple edges while degradations are green")
    def test_result_from_plot(self):
        endresults, mdlhist=propagate.one_fault(self.mdl, 'move_water', 'short', time=10, track='all', 
                                        desired_result=['graph','endclass','endfaults'])
        mg = ModelGraph(self.mdl)
        mg.draw_from(11, mdlhist)
        
#def test_move_nodes(self):
#   p = endresults.graph.move_nodes()
    
if __name__ == '__main__':
    
    
    
    runner = unittest.TextTestRunner()
    runner.run(suite_for_plots(ModelGraphTests, plottests=False))
    runner.run(suite_for_plots(ModelGraphTests, plottests=True))
    
    mdl = Pump()
    endresults, mdlhist=propagate.one_fault(mdl, 'move_water', 'short', time=10, 
                                    desired_result=['graph','endclass','endfaults'],
                                    track='all')
    
    
    
    #p = endresults.graph.move_nodes()
    #endresults.graph.set_node_styles(degraded={}, faulty={})
    #endresults.graph.set_node_labels(title='id', subtext='faults')
    #endresults.graph.draw()
    
    a= ModelTypeGraph(mdl)
    a.draw_pyvis()
    #a.draw_from(10, mdlhist)
    #a.draw_from(50, mdlhist)
    
    #an = a.animate_from(mdlhist)
    #from IPython.display import HTML
    #HTML(an.to_jshtml())
    
    #p = endresults.graph.move_nodes()