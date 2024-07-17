# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:45:11 2024

@author: dhulse
"""
from examples.asg_demo.demo_model import Human, HazardModel
from fmdtools.analyze.graph.model import ModelGraph
mdl = HazardModel()
fmg = ModelGraph(mdl, with_methods=True)
fmg.set_edge_labels(title="role")
fmg.draw_graphviz()

from fmdtools.define.architecture.base import ArchitectureGraph
from examples.pump.ex_pump import Pump, ImportEE

rg = ArchitectureGraph(Pump(), flow_edges=True)
rg.draw()

from fmdtools.define.object.base import ObjectGraph
rg2 = ObjectGraph(Pump().fxns['import_ee'], get_source=True)
rg2.set_node_labels(subtext='code', subtext_style=dict(horizontalalignment='left'))
rg2.draw()
rg2.draw_graphviz()

rg3 = ArchitectureGraph(Human(), flow_edges=True, with_root=True)
rg3.draw()
mdl = Pump()

from examples.multiflow_demo.multiflow_demo import ExModel

fmg2 = ModelGraph(ExModel())
fmg2.draw_graphviz()



# looking at hierarchy
