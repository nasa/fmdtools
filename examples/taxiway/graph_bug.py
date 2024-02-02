# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:04:17 2023

@author: dhulse
"""
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

g = nx.Graph()
g.add_nodes_from((1,2,3))
g.add_edges_from(((1,2),(2,3)))
pos = nx.spring_layout(g)
b = nx.draw_networkx_edges(g, pos, [(1,2)], label="edge1", style = 'dashed',
                           arrows=True, arrowstyle='-|>')
b[0].set_label("edge1")
# lin = mlines.Line2D([], [], color='black', linestyle='dashed', label='edge1')

c = nx.draw_networkx_edges(g, pos, [(2,3)], label="edge2", style = 'solid')
nx.draw_networkx_nodes(g, pos, label="nodes")
plt.legend()
#ax = plt.gca()
#handles, labels = ax.get_legend_handles_labels()
#plt.legend(handles=[lin]+handles)