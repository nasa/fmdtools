# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:04:16 2020

@author: hulsed
"""

import sys
sys.path.append('../')
sys.path.append('Users\hulsed\Downloads\netgraph-master')
from eps import EPS
import fmdtools.resultdisp as rd
import netgraph
import matplotlib.pyplot as plt

mdl= EPS()

#netgraph.draw(mdl.bipartite,node_labels={n:n for n in mdl.bipartite.nodes},node_label_font_size=10)

#netgraph.draw_node_labels({n:n for n in mdl.bipartite.nodes},node_positions=nx.spring_layout(mdl.bipartite))
#plot_instance = netgraph.InteractiveGraph(mdl.bipartite,node_size=10, node_color='gray',node_edge_width=0, node_positions=nx.spring_layout(mdl.bipartite), node_labels={n:n for n in mdl.bipartite.nodes},node_label_font_size=8)

pos = rd.graph.set_pos(mdl.graph)

#plot_instance.node_positions

rd.graph.show(mdl.graph, gtype='normal', pos=pos)

plt.figure()
g=mdl.graph
edgeflows={}
for edge in g.edges:
    flows=list(g.get_edge_data(edge[0],edge[1]).keys())
    edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
netgraph.draw(g,node_size=20*1,node_shape='s',node_color='g', node_edge_width=0, node_positions=pos, edge_labels=edgeflows, edge_label_font_size=8, node_labels={n:n for n in g.nodes},node_label_font_size=8)

#pos = rd.graph.set_pos(mdl.bipartite, gtype ='bipartite')
#rd.graph.show(mdl.bipartite, gtype='bipartite', pos=pos)
