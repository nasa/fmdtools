#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines label arguments for graph plotting.

Has classes:

- :class:`LabelStyle`: Holds kwargs for nx.draw_networkx_labels to be applied to labels
- :class:`EdgeLabelStyle`: Controls edge labels to ensure they do not rotate
- :class:`Labels`: Defines a set of labels to be drawn using draw_networkx_labels.

And Functions:
- :func:`label_for_entry`: Gets the label from an nx.graph for a given entry.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import networkx as nx
from recordclass import dataobject, asdict


class LabelStyle(dataobject):
    """Holds kwargs for nx.draw_networkx_labels to be applied as a style for labels."""

    font_size: int = 12
    font_color: str = "k"
    font_weight: str = "normal"
    alpha: float = 1.0
    horizontalalignment: str = "center"
    verticalalignment: str = "center"
    clip_on: bool = False
    bbox: dict = dict(alpha=0)

    def kwargs(self):
        """Return kwargs for nx.draw_networkx_labels."""
        return asdict(self)

    def gv_align(self, text):
        if self.horizontalalignment == "left":
            text = "\\l".join(text.split("\n")) + "\\l"
        elif self.horizontalalignment == "right":
            text = "\\r".join(text.split("\n")) + "\\r"
        return text.replace("    ", "....")


class EdgeLabelStyle(LabelStyle):
    """Holds kwargs for nx.draw_networkx_edge_labels."""

    rotate: bool = False


def make_shortname(name):
    return name.split('.')[-1] 


def make_lastname(name):
    return name.split("_")[-1]


def shorten_name(name, rem_ind=0):
    return ".".join(name.split('.')[rem_ind:])


def label_for_entry(g, iterator, entryname):
    """
    Create the label dictionary for a given entry value of interest.

    Parameters
    ----------
    g : nx.graph
        Networkx graph structure to create labels for
    iterator : nx.graph.nodes/edges
        Property to iterate over (e.g., nodes or edges)
    entryname : str
        Property to get from the graph attributes. Options are:

        - 'id' : The full name of the node/edge

        - 'shortname' :The short name of the node/edge (after .xx)

        - 'last' : The last part (after all "_" characters) of the name of the node/edge

        - 'nodetype'/'edgetype' : The type property of the node or edge.

        - 'faults_and_indicators' : Fault and indicator properties from the node/edge

        - <str> : Any other property corresponding to the key in the node/edge dict

    Returns
    -------
    entryvals : dict
        Dictionary of values to show for the given entry
    """
    if entryname == "id":
        entryvals = {n: n for n in iterator}
    elif entryname == 'shortname':
        entryvals = {n: make_shortname(n) for n in iterator}
    elif entryname == "last":
        entryvals = {n: make_lastname(n) for n in iterator}
    elif 'type' in entryname:
        entryvals = {n: '<'+v[entryname]+'>' for n, v in iterator.items()}
    elif entryname == 'faults_and_indicators':
        modes = nx.get_node_attributes(g, 'm')
        faults = {n: [*m['faults'], 'sub_faults'] if m['sub_faults']
                  else [*m['faults']] for n, m in modes.items()}
        indicators = nx.get_node_attributes(g, 'indicators')
        all_entries = [*faults, *indicators]
        entryvals = {n: faults.get(n, [])+indicators.get(n, []) for n in all_entries}
    elif 'Edge' in iterator.__class__.__name__:
        entryvals = nx.get_edge_attributes(g, entryname)
    elif 'Node' in iterator.__class__.__name__:
        entryvals = nx.get_node_attributes(g, entryname)
    else:
        entryvals = {}
    return entryvals


class Labels(dataobject, mapping=True):
    """
    Define a set of labels to be drawn using draw_networkx_labels.

    Labels have three distinct parts:

    - title (upper text for the node/edge)

    - title2 (if provided, uppder text for the node/edge after a colon)

    - subtext (lower text of the node/edge)

    Title and subtext may both be given different LabelStyles.
    """

    title: dict = {}
    title_style: LabelStyle = LabelStyle()
    subtext: dict = {}
    subtext_style: LabelStyle = LabelStyle()

    def from_iterator(g, iterator, LabStyle,
                      title='shortname', title2='', subtext='', **node_label_styles):
        """
        Construct the labels from an interator (nodes or edges).

        Parameters
        ----------
        g : nx.graph
            Networkx graph structure to create labels for
        iterator : nx.graph.nodes/edges
            Property to iterate over (e.g., nodes or edges)
        LabStyle : class
            Class to use for label styles (e.g., LabelStyle or EdgeStyle)
        title : str, optional
            entry for title text. (See :func:`label_for_entry` for options).
            The default is 'id'.
        title2 : str, optional
            entry for title text after the colon. (See :func:`label_for_entry` for
            options). The default is ''.
        subtext : str, optional
            entry for the subtext. (See :func:`label_for_entry` for options).
            The default is ''.
        **node_label_styles : dict
            LabStyle arguments to overwrite.

        Returns
        -------
        labs : Labels
            Labels corresponding to the given inputs
        """
        labs = Labels()
        for entry in ['title', 'title2', 'subtext']:
            entryval = vars()[entry]
            evals = label_for_entry(g, iterator, entryval)

            if evals:
                if entry == 'title':
                    labs.title = evals
                elif entry == 'title2':
                    all1 = {k: v for k, v in labs.title.items()}
                    all2 = {k: v for k, v in evals.items()}
                    alls = [*all1, *all2]
                    labs.title = {k: all1.get(k, '')+': '+all2.get(k, '') for k in alls}
                elif entry == 'subtext':
                    labs.subtext = evals

        node_labels = labs.iter_groups()
        for entry in node_labels:
            if len(labs) > 1:
                if entry == 'title':
                    verticalalignment = 'bottom'
                elif entry == 'subtext':
                    verticalalignment = 'top'
            else:
                verticalalignment = 'center'
            if entry == 'title' and 'Node' in iterator.__class__.__name__:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            def_style = dict(verticalalignment=verticalalignment,
                             font_weight=font_weight,
                             **node_label_styles.get(entry+'_style', {}))
            labs[entry+'_style'] = LabStyle(**def_style)
        return labs

    def iter_groups(self):
        """Return groups to iterate through when calling nx.draw_labels."""
        return [n for n in ['title', 'subtext'] if getattr(self, n)]

    def draw_nx_edges(self, g, pos, ax=None):
        """Draw edge labels for a given graph."""
        for level in self.iter_groups():
            nx.draw_networkx_edge_labels(g, pos, self[level],
                                         **self[level+'_style'].kwargs(), ax=ax)

    def draw_nx_nodes(self, g, pos, ax=None):
        """Draw node labels for a given graph."""
        for level in self.iter_groups():
            nx.draw_networkx_labels(g, pos, self[level],
                                    **self[level+'_style'].kwargs(), ax=ax)

    def make_gv_label(self, node):
        """Make the label for graphviz for a given node."""
        label = ""
        if node in self.title:
            label += self.title_style.gv_align(self.title[node])
        if node in self.subtext:
            label += self.subtext_style.gv_align('\n'+str(self.subtext[node]))
        if ('<' in label or '>' in label):
            label = "\\" + label
        return label


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
