# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:06:57 2024

@author: dhulse
"""
from fmdtools.analyze.graph.model import ModelGraph


class BlockGraph(ModelGraph):
    """Blockgraph represents the definition of a Block."""

    def __init__(self, mdl, with_methods=True, **kwargs):
        ModelGraph.__init__(self, mdl, with_methods=with_methods, **kwargs)

    def set_edge_labels(self, title='edgetype', title2='', subtext='role',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)

    def set_node_labels(self, title='shortname', title2='classname', **node_label_styles):
        super().set_node_labels(title=title, title2=title2, **node_label_styles)