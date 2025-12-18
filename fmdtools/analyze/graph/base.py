#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides representations and visualizations of a model graph structure.

Main user-facing individual graphing classes:

- :class:`Graph`: Base graph class.
- :class:`GraphInteractor`: Class for moving graph nodes.

Private Methods:

- :func:`sff_one_trial`: Calculates one trial of the sff model
- :func:`data_average`: Averages each column in data
- :func:`data_error`: Calculates error for each column in data
- :func:`get_label_groups`: Creates groups of nodes/edges in terms of discrete values
  for the given tags.
- :func:`get_group_kwarg`: Get style kwargs for a given group.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.analyze.common import setup_plot
from fmdtools.analyze.graph.style import edge_style_factory, node_style_factory
from fmdtools.analyze.graph.style import to_legend_label, gv_import_check
from fmdtools.analyze.graph.style import nx_plot_ending, gv_plot_ending
from fmdtools.analyze.graph.label import Labels, EdgeLabelStyle, LabelStyle

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import get_backend

plt.rcParams['pdf.fonttype'] = 42


class Graph(object):
    """
    Base class for graphs which can be extended to represent various objects.

    Essentially provides a convenience interface for networkx to enable quick
    visualization and analysis of model properties.

    Attributes
    ----------
    g : nx.Graph
        Networkx graph to run analyses on.
    pos : dict
        Dict of node positions set using set_pos.
    edge_styles : dict
        Dict of styles for each edge set using set_edge_styles.
    node_groups : dict
        Dict of node groups by tag with key (tag1, tag2) and value [node1, node2].
    node_styles : dict
        Dict of styles for each node set using set_node_styles.
    edge_groups : dict
        Dict of edge groups by tag with key (tag1, tag2) and value [edge1, edge2].
    edge_labels : Labels
        Labels for each edge.
    node_labels : Labels
        Labels for each node.

    Parameters
    ----------
    g: networkx.Graph
        Graph to analyze.

    Examples
    --------
    >>> from fmdtools.analyze.graph.style import mod_prefix
    >>> loc = mod_prefix()
    >>> graph = Graph(ex_nxgraph)
    >>> graph.set_pos(auto='kamada_kawai')
    >>> fig, ax = graph.draw(saveas=loc+'nx_funcdecomp.svg')
    >>> graph.set_edge_labels(title="name")
    >>> dot = graph.draw_graphviz(disp=False, saveas=loc+'gv_funcdecomp.svg')
    """

    def __init__(self, g, check_info=True):
        if isinstance(g, nx.Graph):
            self.g = g
        else:
            raise Exception(str(g) + " not a networkx Graph object.")
        if check_info:
            self.check_type_info()
        


    def check_type_info(self):
        """Check that nodes and edges have type data."""
        for n, v in self.g.nodes(data=True):
            if 'nodetype' not in v:
                raise Exception("nodetype not defined for node: "+n)
        for end0, end1, v in self.g.edges(data=True):
            if 'edgetype' not in v:
                raise Exception('edgetype not defined for edge: '+end0+', '+end1)

    def set_pos(self, auto=True, overwrite=True, **pos):
        """
        Set graph positions to given positions, (automatically or manually).

        Parameters
        ----------
        auto : str, optional
            Whether to auto-layout the node position. The default is True. If a string
            is provided, calls method_layout, where method is the string provided
        overwrite : bool, optional
            Whether to overwrite the existing pos. Default is True.
        **pos : nodename=(x,y)
            Positions of nodes to set. Otherwise updates to the auto-layout or (0.5,0.5)
        """
        if not getattr(self, 'pos', False):
            self.pos = {}
        if overwrite or not self.pos:
            if auto:
                if isinstance(auto, str):
                    auto_method = getattr(nx, auto+'_layout')
                    self.pos = auto_method(nx.MultiGraph(self.g))
                else:
                    try:
                        self.pos = nx.planar_layout(nx.MultiGraph(self.g))
                    except nx.NetworkXException:
                        self.pos = nx.spring_layout(nx.MultiGraph(self.g))
            else:
                self.pos = {n: self.pos.get(n, (0.5, 0.5)) for n in self.g.nodes}
            self.pos.update(pos)

    def set_edge_styles(self, edgetype={}, **edge_styles):
        """
        Set self.edge_styles and self.edge_groups given the provided edge styles.

        Parameters
        ----------
        edgetype : dict, optional
            kwargs to EdgeStyle for the given node type (e.g., containment, etc).
        **edge_styles : dict, optional
            Dictionary of tags, labels, and styles for the edges that overwrite the
            default. Has structure {tag:{label:kwargs}}, where kwargs are the keyword
            arguments to nx.draw_networkx_edges. The default is {"label":{}}.
        """
        self.edge_style_labels = ['edgetype', *edge_styles]
        self.edge_groups = get_label_groups(self.g.edges, *self.edge_style_labels)
        self.edge_styles = {}
        for edge_group in self.edge_groups:
            styles = {k: v for i, (k, v) in enumerate(edge_styles.items())
                      if edge_group[i+1]}
            group_kwar = get_group_kwarg(styles.pop('group', {}), edge_group)
            kwar = {**group_kwar, **edgetype.get(edge_group[0], {})}
            self.edge_styles[edge_group] = edge_style_factory(edge_group[0],
                                                              styles=styles,
                                                              **kwar)

    def set_node_styles(self, nodetype={}, **node_styles):
        """
        Set self.node_styles and self.edge_groups given the provided node styles.

        Parameters
        ----------
        nodetype : dict, optional
            kwargs to NodeStyle for the given node type (e.g., Block, Flow, etc).
        **node_styles : dict, optional
            Dictionary of tags, labels, and style kwargs for the nodes that overwrite
            the default. Has structure {tag:{label:kwargs}}, where kwargs are the
            keyword arguments to nx.draw_networkx_nodes. The default is {"label":{}}.
        """
        self.node_style_labels = ['nodetype', *node_styles]
        self.node_groups = get_label_groups(self.g.nodes(), *self.node_style_labels)
        self.node_styles = {}
        for node_group in self.node_groups:
            styles = {k: v for i, (k, v) in enumerate(node_styles.items())
                      if node_group[i+1]}
            group_kwar = get_group_kwarg(styles.pop('group', {}), node_group)
            kwar = {**group_kwar, **nodetype.get(node_group[0], {})}
            self.node_styles[node_group] = node_style_factory(node_group[0],
                                                              styles=styles,
                                                              **kwar)

    def set_edge_labels(self, title='edgetype', title2='', subtext='states',
                        **edge_label_styles):
        """
        Create labels using Labels.from_iterator for the edges in the graph.

        Parameters
        ----------
        title : str, optional
            property to get for title text. The default is 'id'.
        title2 : str, optional
            property to get for title text after the colon. The default is ''.
        subtext : str, optional
            property to get for the subtext. The default is 'states'.
        **edge_label_styles : dict
            LabelStyle arguments to overwrite.
        """
        self.edge_labels = Labels.from_iterator(self.g, self.g.edges, EdgeLabelStyle,
                                                title=title, title2=title2,
                                                subtext=subtext, **edge_label_styles)

    def set_node_labels(self, title='shortname', title2='', subtext='',
                        **node_label_styles):
        """
        Create labels using Labels.from_iterator for the nodes in the graph.

        Parameters
        ----------
        title : str, optional
            Property to get for title text. The default is 'id'.
        title2 : str, optional
            Property to get for title text after the colon. The default is ''.
        subtext : str, optional
            property to get for the subtext. The default is ''.
        node_label_styles :  dict
            LabelStyle arguments to overwrite.
        """
        self.node_labels = Labels.from_iterator(self.g, self.g.nodes, LabelStyle,
                                                title=title, title2=title2,
                                                subtext=subtext, **node_label_styles)

    def add_node_groups(self, **node_groups):
        """
        Create arbitrary groups of nodes to displayed with different styles.

        Parameters
        ----------
        **node_groups : iterable
            nodes in groups. see example.

        Examples
        --------
        >>> graph = Graph(ex_nxgraph)
        >>> graph.add_node_groups(group1=('function_a', 'function_b'), group2=('function_c',))
        >>> graph.set_node_styles(group={'group1': {'nx_node_color':'green'}, 'group2': {'nx_node_color':'red'}})
        >>> fig, ax = graph.draw()

        would show two different groups of nodes, one with green nodes, and the other
        with red nodes
        """
        group_attrs = {}
        for node_group, nodes in node_groups.items():
            group_attrs.update({n: node_group for n in nodes})
        group_attrs.update({n: '' for n in self.g.nodes if n not in group_attrs})
        nx.set_node_attributes(self.g, group_attrs, 'group')

    def set_heatmap(self, heatmap, cmap=plt.cm.coolwarm, default_color_val=0.0,
                    vmin=None, vmax=None):
        """
        Set the association and plotting of a heatmap on a graph.

        Parameters
        ----------
        heatmap : dict/result
            dict/result with keys corresponding to the nodes and values in the range
            of a heatmap (0-1)
        cmap : mpl.Colormap, optional
            Colormap to use for the heatmap. The default is plt.cm.coolwarm.
        default_color_val : float, optional
            Value to use if a node is not in the heatmap dict. The default is 0.0.
        vmin : float
            Minimum value for the heatmap. Default is None, which sets it to the minimum
            value of the heatmap.
        vmax : float
            Maximum value for the heatmap. Default is None, which sets it to the maximum
            value of the heatmap.

        Examples
        --------
        The below should draw function a red, function b blue, function c pink, and all
        else grey:
        >>> graph = Graph(ex_nxgraph)
        >>> graph.set_heatmap({'function_a': 1.0, 'function_b': 0.0, 'function_c': 0.75}, default_color_val=0.5)
        >>> fig, ax = graph.draw()
        """
        allc = [default_color_val,
                *[v for n, v in heatmap.items() if n in self.g.nodes()]]
        if not vmin:
            vmin = np.min(allc)
        if not vmax:
            vmax = np.max(allc)
        self.set_node_styles()
        for label, nodes in self.node_groups.items():
            nodes_colors = [heatmap[node] if node in heatmap else default_color_val
                            for node in nodes]
            self.node_styles[label].nx_node_color = nodes_colors
            self.node_styles[label].nx_cmap = cmap
            self.node_styles[label].nx_vmin = vmin
            self.node_styles[label].nx_vmax = vmax

    def set_properties(self, **kwargs):
        """Set properties using kwargs where there is a given set_kwarg command."""
        for to_set in ['pos', 'edge_styles', 'edge_labels',
                       'node_styles', 'node_labels']:
            if to_set in kwargs or not hasattr(self, to_set):
                set_func = getattr(self, 'set_'+to_set)
                set_func(**kwargs.pop(to_set, {}))
        return kwargs

    def draw(self, figsize=(12, 10), title="", fig=False, ax=False, withlegend=True,
             legend_bbox=(1, 0.5), legend_loc="center left", legend_labelspacing=2,
             legend_borderpad=1, saveas='', **kwargs):
        """
        Draw a graph with given styles corresponding to the node/edge properties.

        Parameters
        ----------
        figsize : tuple, optional
            Size for the figure (plt.figure arg). The default is (12,10).
        title : str, optional
            Title for the plot. The default is "".
        fig : bool, optional
            matplotlib figure to project on (if provided). The default is False.
        ax : bool, optional
            matplotlib axis to plot on (if provided). The default is False.
        withlegend : bool, optional
            Whether to include a legend. The default is True.
        legend_bbox : tuple, optional
            bbox to anchor the legend to. The default is (1,0.5), which places legend
            on the right.
        legend_loc : str, optional
            loc argument for plt.legend. The default is "center left".
        legend_labelspacing : float, optional
            labelspacing argument for plt.legend. the default is 2.
        legend_borderpad : str, optional
            borderpad argument for plt.legend. the default is 1.
        saveas : str, optional
            file to save as (if provided).
        **kwargs : kwargs
            Arguments for various supporting functions:
            (set_pos, set_edge_styles, set_edge_labels, set_node_styles,
            set_node_labels, etc)

        Returns
        -------
        fig : matplotlib figure
            matplotlib figure to draw
        ax : matplotlib axis
            Ax in the figure
        """
        fig, ax = setup_plot(figsize=figsize, fig=fig, ax=ax)
        self.set_properties(**kwargs)
        # draw edges
        edge_handles = []
        for label, edges in self.edge_groups.items():
            legend_label = to_legend_label(label, self.edge_style_labels)
            style = self.edge_styles[label]
            style.draw_nx(self.g, self.pos, edges, label=legend_label, ax=ax)
            # edge handles: used to fix edge legend bug in matplotlib/networkx
            edge_handles.append(style.nx_legend_line(legend_label))
        # draw edge labels
        self.edge_labels.draw_nx_edges(self.g, self.pos, ax=ax)
        # draw nodes
        for label, nodes in self.node_groups.items():
            legend_label = to_legend_label(label, self.node_style_labels)
            self.node_styles[label].draw_nx(self.g, self.pos, nodes,
                                            label=legend_label, ax=ax)
        # draw node labels
        self.node_labels.draw_nx_nodes(self.g, self.pos, ax=ax)
        nx_plot_ending(fig, ax, title, withlegend, saveas=saveas,
                       labelspacing=legend_labelspacing,
                       borderpad=legend_borderpad, bbox_to_anchor=legend_bbox,
                       loc=legend_loc, add_handles=edge_handles)
        return fig, ax

    def draw_drawio(self, saveas='', **kwargs):
        """
        Generate DrawIO diagram using existing graph structure and positions.

        Parameters
        ----------
        saveas : str, optional
            File path to save the DrawIO XML. If empty, returns XML content.
        **kwargs : dict
            Additional arguments for graph styling and positioning.

        Returns
        -------
        xml_content : str
            DrawIO XML content.

        Examples
        --------
        >>> graph = Graph(ex_nxgraph)
        >>> xml_content = graph.draw_drawio()  # Get XML content
        >>> xml_content = graph.draw_drawio("graph.drawio")  # Save to file
        """
        if not hasattr(self, 'pos') or not self.pos:
            # Set default positions if none exist
            self.set_pos(auto='spring')

        # Improve spacing by scaling positions more aggressively
        self._improve_drawio_spacing()

        xml_content = self._create_drawio_xml()

        if saveas:
            with open(saveas, 'w') as f:
                f.write(xml_content)
        return xml_content

    def _improve_drawio_spacing(self):
        """Improve spacing between nodes for better DrawIO visualization."""
        if not hasattr(self, 'pos') or not self.pos:
            return

        # Find the range of positions - positions should always be (x, y) tuples
        x_coords = [pos[0] for pos in self.pos.values()]
        y_coords = [pos[1] for pos in self.pos.values()]

        if not x_coords or not y_coords:
            return

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Calculate scaling factors to ensure minimum spacing
        min_spacing = 500  # Minimum pixels between node centers
        current_x_range = x_max - x_min if x_max != x_min else 1
        current_y_range = y_max - y_min if y_max != y_min else 1

        # Scale to ensure minimum spacing
        x_scale = max(1, min_spacing / current_x_range)
        y_scale = max(1, min_spacing / current_y_range)

        # Apply improved scaling to all positions
        for node, pos in self.pos.items():
            # Scale and center the positions with more aggressive spacing
            new_x = (pos[0] - x_min) * x_scale * 2.5 + 200  # Multiply by 2.5 for more spacing
            new_y = (pos[1] - y_min) * y_scale * 2.5 + 200
            self.pos[node] = (new_x, new_y)

    def _create_drawio_xml(self):
        """Create basic DrawIO XML from graph data."""
        # Calculate canvas size based on node positions
        if hasattr(self, 'pos') and self.pos:
            x_coords = [pos[0] for pos in self.pos.values()]
            y_coords = [pos[1] for pos in self.pos.values()]
            if x_coords and y_coords:
                canvas_width = max(x_coords) + 200  # Add padding
                canvas_height = max(y_coords) + 200
            else:
                canvas_width, canvas_height = 1000, 800
        else:
            canvas_width, canvas_height = 1000, 800

        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<mxfile host="fmdtools" version="1.0">',
            '  <diagram id="graph" name="Graph">',
            f'    <mxGraphModel dx="{canvas_width}" dy="{canvas_height}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{canvas_width + 50}" pageHeight="{canvas_height + 50}" math="0" shadow="0">',
            '      <root>',
            '        <mxCell id="0"/>',
            '        <mxCell id="1" parent="0"/>'
        ]

        # Add nodes with proper positioning
        for node, pos in self.pos.items():
            # Use the improved positions directly
            x = float(pos[0])
            y = float(pos[1])

            node_data = self.g.nodes[node]
            nodetype = node_data.get('nodetype', 'default')

            # Infer nodetype from node name if it's 'default'
            if nodetype == 'default':
                if '.flows.' in node:
                    nodetype = 'flow'
                elif '.fxns.' in node:
                    nodetype = 'function'
                elif '.actions.' in node:
                    nodetype = 'action'
                elif '.components.' in node:
                    nodetype = 'component'

            style = self._get_node_style(nodetype)

            xml_parts.append(
                f'        <mxCell id="{node}" value="{node}" style="{style}" vertex="1" parent="1">'
            )
            xml_parts.append(
                f'          <mxGeometry x="{x}" y="{y}" width="80" height="40" as="geometry"/>'
            )
            xml_parts.append('        </mxCell>')

        # Add edges
        for edge in self.g.edges():
            source, target = edge
            edge_data = self.g.edges[edge]
            edgetype = edge_data.get('edgetype', 'default')
            style = self._get_edge_style(edgetype)

            xml_parts.append(
                f'        <mxCell id="edge_{source}_{target}" style="{style}" edge="1" parent="1" source="{source}" target="{target}">'
            )
            xml_parts.append('          <mxGeometry relative="1" as="geometry"/>')
            xml_parts.append('        </mxCell>')

        xml_parts.extend([
            '      </root>',
            '    </mxGraphModel>',
            '  </diagram>',
            '</mxfile>'
        ])

        return '\n'.join(xml_parts)

    def _get_node_style(self, nodetype):
        """
        Get DrawIO style for node type using the existing style system.

        Examples
        --------
        >>> graph = Graph(nx.DiGraph())
        >>> style = graph._get_node_style('Function')
        >>> 'fillColor=#cce5ff' in style
        True
        >>> 'strokeColor=#6699cc' in style
        True

        >>> style = graph._get_node_style('Flow')
        >>> 'ellipse' in style
        True
        >>> 'fillColor=#90EE90' in style
        True

        >>> style = graph._get_node_style('Environment')
        >>> 'fillColor=#ffccff' in style
        True
        """
        try:
            # Use the existing style system
            style_obj = node_style_factory(nodetype)
            drawio_props = style_obj.drawio_kwargs()

            # Convert style properties to DrawIO XML format
            style_parts = []

            # Shape
            if 'shape' in drawio_props:
                if drawio_props['shape'] == 'ellipse':
                    style_parts.append('ellipse')
                elif drawio_props['shape'] == 'rhombus':
                    style_parts.append('rhombus')
                elif drawio_props['shape'] == 'triangle':
                    style_parts.append('triangle')
                elif drawio_props['shape'] == 'hexagon':
                    style_parts.append('shape=hexagon;perimeter=hexagonPerimeter2')
                else:
                    style_parts.append('rounded=0')
            else:
                style_parts.append('rounded=0')

            # Basic properties
            style_parts.extend(['whiteSpace=wrap', 'html=1'])

            # Colors
            if 'fillcolor' in drawio_props:
                style_parts.append(f"fillColor={drawio_props['fillcolor']}")
            if 'strokecolor' in drawio_props:
                style_parts.append(f"strokeColor={drawio_props['strokecolor']}")

            # Font properties
            if 'fontsize' in drawio_props:
                style_parts.append(f"fontSize={drawio_props['fontsize']}")
            if 'fontstyle' in drawio_props:
                style_parts.append(f"fontStyle={drawio_props['fontstyle']}")

            return ';'.join(style_parts)

        except Exception:
            # Fallback to default style
            return 'rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;'

    def _get_edge_style(self, edgetype):
        """
        Get DrawIO style for edge type using the existing style system.

        Examples
        --------
        >>> graph = Graph(nx.DiGraph())
        >>> style = graph._get_edge_style('connection')
        >>> 'strokeWidth=1' in style
        True

        >>> style = graph._get_edge_style('activation')
        >>> 'dashed=1' in style
        True
        """
        try:
            # Use the existing style system
            style_obj = edge_style_factory(edgetype)
            drawio_props = style_obj.drawio_kwargs()

            # Convert style properties to DrawIO XML format
            style_parts = []

            # Stroke properties
            if 'strokewidth' in drawio_props:
                style_parts.append(f"strokeWidth={drawio_props['strokewidth']}")
            if 'strokecolor' in drawio_props:
                style_parts.append(f"strokeColor={drawio_props['strokecolor']}")

            # Arrow properties
            if 'startarrow' in drawio_props:
                style_parts.append(f"startArrow={drawio_props['startarrow']}")
            if 'endarrow' in drawio_props:
                style_parts.append(f"endArrow={drawio_props['endarrow']}")

            # Line style
            if 'dashed' in drawio_props and drawio_props['dashed']:
                style_parts.append('dashed=1')

            return ';'.join(style_parts)

        except Exception:
            # Fallback to default style
            return 'strokeWidth=1;strokeColor=#cccccc;endArrow=classic;'

    def move_nodes(self, **kwargs):
        """
        Set the position of nodes for plots in analyze.graph using a graphical tool.

        Note: make sure matplotlib is set to plot in an external window
        (e.g., using '%matplotlib qt)

        Parameters
        ----------
        **kwargs : kwargs
            keyword arguments for graph.draw

        Returns
        -------
        p : GraphIterator
            Graph Iterator (in analyze.Graph)
        """
        plt.ion()
        p = GraphInteractor(self, **kwargs)
        if 'inline' in get_backend():
            print("Cannot place nodes in inline version of plot. Use '%matplotlib qt'" +
                  " (or '%matplotlib osx') to open in external window")
        return p

    def draw_graphviz(self, disp=True, saveas='', **kwargs):
        """
        Draw the graph using pygraphviz for publication-quality figures.

        Note that the style may not match one-to-one with the defined none/edge styles.

        Parameters
        ----------
        disp : bool
            Whether to display the plot. The default is True.
        saveas : str
            File to save the plot as. The default is '' (which doesn't save the plot').
        **kwargs : kwargs
            Arguments for various supporting functions:
            (set_pos, set_edge_styles, set_edge_labels, set_node_styles,
            set_node_labels, etc)
            Can also provide kwargs for Digraph() initialization.

        Returns
        -------
        dot : PyGraphviz DiGraph
            Graph object corresponding to the figure.
        """
        kwargs = self.set_properties(**kwargs)
        Digraph, Graph = gv_import_check()
        dot = Digraph(graph_attr=kwargs)

        for group, nodes in self.node_groups.items():
            gv_kwargs = self.node_styles[group].gv_kwargs()
            for node in nodes:
                dot.node(node, label=self.node_labels.make_gv_label(node), **gv_kwargs)

        for group, edges in self.edge_groups.items():
            gv_kwargs = self.edge_styles[group].gv_kwargs()
            for edge in edges:
                dot.edge(edge[0], edge[1], label=self.edge_labels.make_gv_label(edge),
                         **gv_kwargs)
        gv_plot_ending(dot, disp=disp, saveas=saveas)
        return dot

    def calc_aspl(self):
        """
        Compute average shortest path length of the graph.

        Returns
        -------
        aspl: float
            Average shortest path length
        """
        return nx.average_shortest_path_length(self.g.to_undirected())

    def calc_modularity(self):
        """
        Compute network modularity of the graph.

        Returns
        -------
        modularity : Modularity
        """
        from networkx.algorithms.community import greedy_modularity_communities
        from networkx.algorithms.community.quality import modularity
        communities = list(greedy_modularity_communities(self.g))
        return modularity(self.g, communities)

    def find_bridging_nodes(self):
        """
        Determine bridging nodes in a graph representation of model mdl.

        Returns
        -------
        bridgingNodes : list of bridging nodes
        """
        from networkx.algorithms.community import greedy_modularity_communities
        g = self.g
        communitiesRaw = list(greedy_modularity_communities(g))
        communities = [list(x) for x in communitiesRaw]
        numCommunities = len(communities)
        nodes = list(g.nodes)
        numNodes = len(nodes)
        bridgingNodes = list()
        nodeEdges = list()
        for i in range(0, numNodes):
            nodeEdges.append(list(g.edges(nodes[i])))
            lenNodeEdges = len(nodeEdges[i])
            for j in range(numCommunities):
                if nodes[i] in communities[j]:
                    communityIdx = j
            for j in range(lenNodeEdges):
                nodeEdgePair = list(nodeEdges[i][j])
                if nodeEdgePair[1] in communities[communityIdx]:
                    pass
                else:
                    bridgingNodes.append(nodes[i])
        bridgingNodes = sorted(list(set(bridgingNodes)))
        return bridgingNodes

    def plot_bridging_nodes(self, title='bridging nodes',
                            node_kwargs={'nx_node_color': 'red', 'gv_fillcolor': 'red'},
                            **kwargs):
        """
        Plot bridging nodes using self.draw().

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is 'bridging nodes'.
        node_kwargs : dict, optional
            Non-default fields for NodeStyle
        **kwargs : kwargs
            kwargs for Graph.draw

        Returns
        -------
        fig : matplotlib figure
            Figure
        """
        bridgingnodes = self.find_bridging_nodes()
        self.add_node_groups(bridging_nodes=bridgingnodes)
        self.set_node_styles(group={'bridging_nodes': node_kwargs})
        fig = self.draw(title=title, **kwargs)
        return fig

    def find_high_degree_nodes(self, p=90):
        """
        Determine highest degree nodes, up to percentile p, in graph.

        Parameters
        ----------
        p : int (optional)
            percentile of degrees to return, between 0 and 100

        Returns
        -------
        highDegreeNodes : list of high degree nodes in format (node,degree)
        """
        g = self.g
        d = list(g.degree())

        def take_second(elem):
            return elem[1]
        sortedNodes = sorted(d, key=take_second, reverse=True)
        sortedDegrees = [x[1] for x in sortedNodes]
        sortedDegreesSet = set(sortedDegrees)
        sortedDegreesUnique = list(sortedDegreesSet)
        sortedDegreesUniqueArray = np.array(sortedDegreesUnique)
        topPercentileDegree = np.percentile(sortedDegreesUniqueArray, p)
        numNodes = len(sortedNodes)
        highDegreeNodes = [sortedNodes[0]]
        for i in range(1, numNodes):
            if sortedNodes[i][1] < topPercentileDegree:
                pass
            else:
                highDegreeNodes.append(sortedNodes[i])
        return highDegreeNodes

    def plot_high_degree_nodes(self, p=90, title='',
                               node_kwargs={'nx_node_color': 'red', 'gv_fillcolor': 'red'},
                               **kwargs):
        """
        Plot high-degree nodes using self.draw().

        Parameters
        ----------
        p : int (optional)
            percentile of degrees to return, between 0 and 100
        title : str, optional
            Title for the plot. The default is 'High Degree Nodes'.
        node_kwargs : dict : kwargs to overwrite the default NodeStyle
        **kwargs : kwargs
            kwargs for Graph.draw

        Returns
        -------
        fig : matplotlib figure
            Figure
        """
        if not title:
            title = 'High Degree Nodes ('+str(p)+'th Percentile)'
        hdnodes = self.find_high_degree_nodes()
        self.add_node_groups(high_degree_nodes=[h[0] for h in hdnodes])
        self.set_node_styles(group={'high_degree_nodes': node_kwargs})
        fig = self.draw(title=title, **kwargs)
        return fig

    def calc_robustness_coefficient(self, trials=100, seed=False):
        """
        Compute robustness coefficient of graph representation of model mdl.

        Parameters
        ----------
        trials : int
            number of times to run robustness coefficient algorithm
            (result is averaged over all trials)
        seed : int
            optional seed to instantiate test with

        Returns
        -------
        RC : robustness coefficient
        """
        g = self.g.to_undirected()
        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        trialsRC = list()
        for itr in range(trials):
            tmp = g.copy()
            N = float(len(tmp))
            largestCC = max(nx.connected_components(tmp), key=len)
            s = [float(len(largestCC))]
            rs = rng.choice(range(int(s[0])), int(s[0]), replace=False)
            nodes = list(g)
            for i in range(int(s[0])-1):
                tmp.remove_node(nodes[rs[i]])
                largestCC = max(nx.connected_components(tmp), key=len)
                s.append(float(len(largestCC)))
            trialsRC.append((200*sum(s)-100*s[0])/N/N)
        RC = sum(trialsRC)/len(trialsRC)
        return RC

    def plot_degree_dist(self):
        """
        Plot degree distribution of graph representation of model mdl.

        Returns
        -------
        fig : matplotlib figure
            plot of distribution
        """
        import math
        g = self.g.to_undirected()
        degrees = [g.degree(n) for n in g.nodes()]
        degreesSet = set(degrees)
        degreesUnique = list(degreesSet)
        freq = [degrees.count(n) for n in degreesUnique]
        maxFreq = max(freq)
        freqint = list(range(0, maxFreq+1))
        degreeint = list(range(min(degrees), math.ceil(max(degrees))+1))

        degreesSet = set(degrees)
        degreesUnique = list(degreesSet)
        numDegreesUnique = len(degreesUnique)

        fig = plt.figure()
        plt.hist(degrees, bins=np.arange(numDegreesUnique)-0.5)
        plt.xticks(degreeint)
        plt.yticks(freqint)
        plt.title('Degree distribution')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()
        return fig

    def sff_model(self, endtime=5, pi=.1, pr=.1,
                  num_trials=100, start_node='random', error_bar_option='off'):
        """
        Susceptible-fix-fail model.

        Parameters
        ----------
        endtime: int
            simulation end time
        pi : float
            infection (failure spread) rate
        pr : float
            recovery (fix) rate
        num_trials : int
            number of times to run the epidemic model, default is 100
        error_bar_option : str
            option for plotting error bars (first to third quartile), default is off
        start_node : str
            start node to use in the trial. default is 'random'

        Returns
        -------
        fig: plot of susc, fail, and fix nodes over time
        """
        g = self.g.to_undirected()
        if start_node == 'random':
            nodes = list(g.nodes)
            start_node_selected = nodes[np.random.randint(0, len(nodes))]
        else:
            start_node_selected = start_node
        num_susc_all_trials = []
        num_fail_all_trials = []
        num_fix_all_trials = []
        for trials in range(0, num_trials):
            s_f_f = sff_one_trial(start_node_selected, g, endtime=endtime, pi=pi, pr=pr)
            num_susc_trial, num_fail_trial, num_fix_trial = s_f_f
            num_susc_all_trials.append(num_susc_trial)
            num_fail_all_trials.append(num_fail_trial)
            num_fix_all_trials.append(num_fix_trial)
        num_susc_average = data_average(num_susc_all_trials)
        num_fail_average = data_average(num_fail_all_trials)
        num_fix_average = data_average(num_fix_all_trials)

        fig = plt.figure()
        time_list = range(0, endtime+1)
        if error_bar_option == 'on':
            num_susc_lower_error, num_susc_upper_error = data_error(num_susc_all_trials,
                                                                    num_susc_average)
            num_fail_lower_error, num_fail_upper_error = data_error(num_fail_all_trials,
                                                                    num_fail_average)
            num_fix_lower_error, num_fix_upper_error = data_error(num_fix_all_trials,
                                                                  num_fix_average)
            num_susc_asymmetric_error = np.abs([num_susc_lower_error, num_susc_upper_error])
            num_fail_asymmetric_error = np.abs([num_fail_lower_error, num_fail_upper_error])
            num_fix_asymmetric_error = np.abs([num_fix_lower_error, num_fix_upper_error])
            plt.errorbar(time_list, num_susc_average,
                         yerr=num_susc_asymmetric_error, fmt='-o', label='Susceptible')
            plt.errorbar(time_list, num_fail_average,
                         yerr=num_fail_asymmetric_error, fmt='-o', label='Failed')
            plt.errorbar(time_list, num_fix_average,
                         yerr=num_fix_asymmetric_error, fmt='-o', label='Fixed')
        else:
            plt.plot(time_list, num_susc_average, label='Susceptible')
            plt.plot(time_list, num_fail_average, label='Failed')
            plt.plot(time_list, num_fix_average, label='Fixed')
        plt.legend()
        plt.title('SFF model')
        plt.xlabel('Time steps')
        plt.ylabel('Number of nodes')
        plt.show()
        return fig

    def calc_node_betweenness(self):
        """Calculate betweenness for nodes."""
        return nx.betweenness_centrality(self.g)

    def calc_node_closeness(self):
        """Calculate closeness dict for nodes."""
        return nx.closeness_centrality(self.g)

    def calc_node_degree(self):
        """Calculate degree dict for nodes."""
        return dict(self.g.degree())

    def calc_node_eigenvector(self):
        """Calculate eigenvector centrality for nodes."""
        # Handle eigenvector centrality for potentially disconnected graphs
        g_check = self.g if self.g.is_directed() else self.g.to_undirected()
        is_connected = (nx.is_strongly_connected(g_check) if self.g.is_directed()
                       else nx.is_connected(g_check))
        if not is_connected:
            centrality = {node: 0.0 for node in self.g.nodes()}
        else:
            try:
                centrality = nx.eigenvector_centrality(self.g, max_iter=100)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                try:
                    centrality = nx.eigenvector_centrality_numpy(self.g)
                except (nx.AmbiguousSolution, nx.NetworkXError):
                    centrality = {node: 0.0 for node in self.g.nodes()}
        return centrality

    def get_node_percentiles(self, metric='betweenness', quartiles = [0, 25, 50, 75]):
        """
        Get the quartiles of a node for a given metric.

        Parameters
        ----------
        metric : str or callable, optional
            If str: Predefined centrality (calls calc_strname). Options:
            - 'betweenness': Betweenness centrality (via NetworkX)
            - 'closeness': Closeness centrality (via NetworkX)
            - 'eigenvector': Eigenvector centrality (via NetworkX)
            - 'degree': Degree centrality
            If callable: Function that accepts self.g and returns dict of {node: value}.
            Default is 'betweenness'.
        quartiles : list, optional
            Percentile boundaries for grouping nodes. Default is [0, 25, 50, 75].
            Must have length >= 2.

        Returns
        -------
        groups : dict
            Dict of nodes in each quartile.

        Examples
        --------
        >>> graph = Graph(ex_nxgraph)
        >>> graph.get_node_percentiles("degree")
        {0: ['external_signals', 'external_energy_in', 'external_material_in', 'external_material_out', 'external_energy_out'], 25: [], 50: ['control_signal', 'internal_energy'], 75: ['function_a', 'function_b', 'function_c']}
        """
        num_groups = len(quartiles)
        if num_groups < 2:
            raise ValueError("quartiles must have at least 2 values")

        # Calculate centrality based on metric
        if callable(metric):
            centrality = metric(self.g)
        elif hasattr(self, "calc_node_"+metric):
            centrality = getattr(self, "calc_node_"+metric)()
        else:
            raise Exception("Invalid option for metric: "+metric)

        # Get centrality values
        values = np.array(list(centrality.values()))
        if len(values) == 0:
            raise ValueError("Graph has no nodes to compute centrality for")

        # Check if all values are zero
        if np.all(values == 0):
            # Use uniform grouping for zero centrality
            nodes_list = list(centrality.keys())
            group_size = max(1, len(nodes_list) // num_groups)
            groups = {}
            for i, group_name in enumerate(quartiles):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < num_groups - 1 else len(nodes_list)
                groups[group_name] = nodes_list[start_idx:end_idx]
        else:
            # Create node groups by centrality quartiles
            percentile_values = np.percentile(values, quartiles)
            groups = {name: [] for name in quartiles}

            for node, val in centrality.items():
                # Find which quartile this node belongs to
                for i in range(len(percentile_values) - 1):
                    if val <= percentile_values[i + 1]:
                        groups[quartiles[i]].append(node)
                        break
                else:
                    # Handle nodes above highest percentile
                    groups[quartiles[-1]].append(node)
        return groups


    def plot_node_percentiles(self, metric='betweenness', title='',
                              colormap=plt.cm.coolwarm,
                              quartiles = [0, 25, 50, 75], quartile_names={},
                              quartile_styles={}, **kwargs):
        """
        Plot nodes colored by a centrality or custom metric.

        Visualizes the graph with nodes grouped by their metric scores into
        quartiles. Supports both predefined centrality metrics and custom
        metric functions.

        Parameters
        ----------
        title : str, optional
            Plot title. If empty, generates title from metric name.
        colormap : matplotlib colormap, optional
            Matplotlib colormap object for coloring nodes. plt.cm.coolwarm.
        quartile_names : dict, optional
            Names for quartile labels. Default is {}, which is just the numbers.
        quartile_styles : dict, optional
            Custom styles for each quartile group. Keys are quartile indices (0, 1, 2, 3),
            values are style dicts with 'nx_node_color' and/or 'gv_fillcolor'.
            Example: {0: {'nx_node_color': 'red'}, 3: {'nx_node_color': 'blue'}}
        **kwargs : dict
            Additional keyword arguments passed to Graph.draw().

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.

        Examples
        --------
        >>> # Using predefined metric
        >>> graph = Graph(ex_nxgraph)
        >>> graph.set_pos(auto='spring')
        >>> fig = graph.plot_node_percentiles(metric='betweenness')
        >>>
        >>> # Using custom metric function
        >>> import networkx as nx
        >>> custom_metric = lambda g: nx.pagerank(g)
        >>> fig = graph.plot_node_percentiles(metric=custom_metric, title='PageRank')
        >>>
        >>> # Using custom quartiles and styles
        >>> quartile_styles = {"0": {'nx_node_color': 'orange'},
        ...                    "30": {'nx_node_color': 'green'}}
        >>> fig = graph.plot_node_percentiles(metric='degree', quartiles=[0, 30, 60, 90],
        ...                                   quartile_styles=quartile_styles)
        """
        groups = self.get_node_percentiles(metric=metric, quartiles=quartiles)
        groups = {quartile_names.get(g, str(g)): v for g, v in groups.items()}
        self.add_node_groups(**groups)

        # Generate colors from cmap
        colors = [colormap(i/100) for i in quartiles]
        # Build style dictionary
        style_dict = {}
        for i, group_name in enumerate(groups):
            default_style = {'nx_node_color': np.array([colors[i]]),
                             'gv_fillcolor': f'#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}'}
            style = {**default_style, **quartile_styles.get(group_name, {})}
            style_dict[group_name] = style

        self.set_node_styles(group=style_dict)

        # Generate title if not provided
        if not title:
            if callable(metric):
                metric = metric.__name__
            title = "Percentiles of Node "+metric

        return self.draw(title=title, **kwargs)

    def summary(self):
        """
        Generate a summary dictionary of key graph metrics.

        Provides a quick overview of graph structure and properties useful for
        model documentation, comparison, and resilience analysis.

        Returns
        -------
        dict
            Summary statistics including:
            - num_nodes : int
                Number of nodes in the graph
            - num_edges : int
                Number of edges in the graph
            - density : float
                Graph density (ratio of actual to possible edges)
            - is_connected : bool
                Whether the graph is connected (for directed graphs, this checks
                weak connectivity)
            - num_components : int
                Number of connected components (weakly connected for directed graphs)
            - avg_degree : float
                Average node degree
            - aspl : float or None
                Average shortest path length (only if graph is connected, None otherwise)
            - modularity : float
                Network modularity score (measures strength of division into communities)

        Examples
        --------
        >>> graph = Graph(ex_nxgraph)
        >>> summary = graph.summary()
        >>> summary['num_nodes']
        10
        >>> summary['num_edges']
        12
        """
        # Convert to undirected for connectivity checks
        g_undirected = self.g.to_undirected()
        is_connected = nx.is_connected(g_undirected)

        num_nodes = self.g.number_of_nodes()
        if num_nodes == 0:
            raise ValueError("Cannot compute summary for empty graph")

        summary_dict = {
            'num_nodes': num_nodes,
            'num_edges': self.g.number_of_edges(),
            'density': nx.density(self.g),
            'is_connected': is_connected,
            'num_components': nx.number_connected_components(g_undirected),
            'avg_degree': sum(dict(self.g.degree()).values()) / num_nodes,
        }

        # Only compute ASPL if graph is connected
        if is_connected:
            summary_dict['aspl'] = self.calc_aspl()
        else:
            summary_dict['aspl'] = None

        summary_dict['modularity'] = self.calc_modularity()

        return summary_dict

    def compare_with(self, other_graph):
        """
        Compare this graph with another graph to identify structural differences.

        Useful for comparing different model variants, design alternatives, or
        tracking how model structure evolves across versions.

        Parameters
        ----------
        other_graph : Graph
            Another Graph object to compare with.

        Returns
        -------
        dict
            Comparison results including:
            - nodes_added : set
                Nodes present in other_graph but not in this graph
            - nodes_removed : set
                Nodes present in this graph but not in other_graph
            - edges_added : set
                Edges present in other_graph but not in this graph
            - edges_removed : set
                Edges present in this graph but not in other_graph
            - structure_similarity : float
                Jaccard similarity coefficient for overall structure (0-1)
            - summary_this : dict
                Summary statistics for this graph
            - summary_other : dict
                Summary statistics for the other graph

        Examples
        --------
        >>> from examples.pump.ex_pump import Pump
        >>> from examples.rover.rover_model import Rover
        >>> from fmdtools.define.architecture.function import FunctionArchitectureGraph
        >>> pump_graph = FunctionArchitectureGraph(Pump())
        >>> rover_graph = FunctionArchitectureGraph(Rover())
        >>> comparison = pump_graph.compare_with(rover_graph)
        >>> len(comparison['nodes_added']) > 0
        True
        """
        nodes_this = set(self.g.nodes())
        nodes_other = set(other_graph.g.nodes())
        edges_this = set(self.g.edges())
        edges_other = set(other_graph.g.edges())

        nodes_added = nodes_other - nodes_this
        nodes_removed = nodes_this - nodes_other
        edges_added = edges_other - edges_this
        edges_removed = edges_this - edges_other

        # Calculate structure similarity using Jaccard coefficient
        all_nodes = nodes_this | nodes_other
        common_nodes = nodes_this & nodes_other
        all_edges = edges_this | edges_other
        common_edges = edges_this & edges_other

        if len(all_nodes) == 0 and len(all_edges) == 0:
            structure_similarity = 1.0
        else:
            # Weight nodes and edges equally
            node_sim = len(common_nodes) / len(all_nodes) if len(all_nodes) > 0 else 1.0
            edge_sim = len(common_edges) / len(all_edges) if len(all_edges) > 0 else 1.0
            structure_similarity = (node_sim + edge_sim) / 2

        return {
            'nodes_added': nodes_added,
            'nodes_removed': nodes_removed,
            'edges_added': edges_added,
            'edges_removed': edges_removed,
            'structure_similarity': structure_similarity,
            'summary_this': self.summary(),
            'summary_other': other_graph.summary()
        }


def sff_one_trial(start_node_selected, g, endtime=5, pi=.1, pr=.1):
    """
    Calculate one trial of the sff model.

    Parameters
    ----------
    start_node_selected : str
        node to start the trial from
    g : networkx graph
        graph to run the trial over
    endtime: int
        simulation end time
    pi : float
        infection (failure spread) rate
    pr : float
        recovery (fix) rate
    """
    rng = np.random.default_rng()
    nodes = list(g.nodes)
    num_nodes = len(nodes)
    time = 0
    susc_nodes = nodes
    susc_nodes.remove(start_node_selected)
    fail_nodes = [start_node_selected]
    fix_nodes = []
    num_susc = [num_nodes-1]
    num_fail = [1]
    num_fix = [0]
    while time < endtime:
        time = time + 1
        new_exposed_nodes = []
        for i in range(0, len(fail_nodes)):
            n = list(g.neighbors(fail_nodes[i]))
            new_exposed_nodes.extend(n)
        ri_list = [rng.random() for iter in range(len(new_exposed_nodes))]
        new_fail_nodes = []
        for i in range(0, len(new_exposed_nodes)):
            if new_exposed_nodes[i] in fix_nodes:
                pass
            else:
                if ri_list[i] <= pi:
                    new_fail_nodes.append(new_exposed_nodes[i])
        new_fail_nodes_set = set(new_fail_nodes)
        new_fail_nodes = list(new_fail_nodes_set)
        for i in range(0, len(new_fail_nodes)):
            if new_fail_nodes[i] in fail_nodes:
                pass
            else:
                susc_nodes.remove(new_fail_nodes[i])
        fail_nodes.extend(new_fail_nodes)
        fail_nodes_set = set(fail_nodes)
        fail_nodes = list(fail_nodes_set)
        rf_list = [rng.random() for iter in range(len(fail_nodes))]
        new_fix_nodes = []
        for i in range(0, len(fail_nodes)):
            if rf_list[i] <= pr:
                new_fix_nodes.append(fail_nodes[i])
        fix_nodes.extend(new_fix_nodes)
        fail_nodes = list(set(fail_nodes)-set(new_fix_nodes))
        num_susc.append(len(susc_nodes))
        num_fail.append(len(fail_nodes))
        num_fix.append(len(fix_nodes))
    return num_susc, num_fail, num_fix


def data_average(data):
    """Average each column in data."""
    list_average = []
    for i in range(0, len(data[0])):
        list_average.append(sum(x[i] for x in data)/len(data))
    return list_average


def data_error(data, average):
    """
    Calculate error for each column in data.

    Parameters
    ----------
    data : list
        List of lists from sff_model
    average : list
        Average of data generated from sff_model over time

    Returns
    -------
    lower_error : float
        Lower bound of error
    upper_error : float
        Upper bound of error
    """
    q1 = []
    q3 = []
    for i in range(0, len(data[0])):
        current_array = np.array([float(x[i]) for x in data])
        q1.append(np.percentile(current_array, 25))
        q3.append(np.percentile(current_array, 75))
    lower_error = [avg - low for avg, low in zip(average, q1)]
    upper_error = [hi - avg for avg, hi in zip(average, q3)]
    return lower_error, upper_error


def get_label_groups(iterator, *tags):
    """
    Create groups of nodes/edges in terms of discrete values for the given tags.

    Parameters
    ----------
    iterator : iterable
        e.g., nx.graph.nodes(), nx.graph.edges()
    *tags : list
        Tags to find in the graph object (e.g., `label`, `status`, etc.)

    Returns
    -------
    label_groups : dict
        Dict of groups of nodes/edges with given tag values. With structure::
        {(tagval1, tagval2...):[list_of_nodes]}
    """
    try:
        labels = {k: tuple(vals[tag] for tag in tags) for k, vals in iterator.items()}
    except KeyError as e:
        unable = {k: tuple(tag for tag in tags if tag not in vals)
                  for k, vals in iterator.items()}
        unable = {k: v for k, v in unable.items() if v}
        raise Exception("The following keys lack the following tags: " +
                        str(unable)) from e
    label_groups = {}
    for key, label in labels.items():
        if label in label_groups:
            label_groups[label].append(key)
        else:
            label_groups[label] = [key]
    return label_groups


def get_group_kwarg(group_dict, group_membership):
    """Get the kwargs related to group membership for a node/edge style."""
    this_group = [g for g in group_membership if g in group_dict]
    if not this_group:
        return {}
    elif len(this_group) == 1:
        return group_dict[this_group[0]]
    else:
        raise Exception("Cannot belong to more than one group at once.")


class GraphInteractor:
    """
    A simple interactive graph for consistent node placement, etc.

    Used in set_pos to set node positions.
    """

    showverts = True
    epsilon = 0.2  # max pixel distance to count as a vertex hit

    def __init__(self, g_obj, **kwargs):
        """
        Initialize the interactive graph.

        Parameters
        ----------
        g_obj : Graph
            Graph object to plot interactively
        kwargs : dict
            kwargs for Graph.draw
        """
        self.t = 0
        gridspec_kw = {'height_ratios': [1, 10]}
        self.fig, (self.bax, self.ax) = plt.subplots(2, gridspec_kw=gridspec_kw)
        self.kwargs = kwargs
        self.g_obj = g_obj
        self.g_obj.set_pos()
        self.refresh_plot()
        self._clicked_node = None
        bnext = Button(self.bax, 'Print positions')
        bnext.on_clicked(self.print_pos)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def get_closest_point(self, event):
        """Find the closest node to the given click to see if it should be move."""
        pt_x = np.array([x[0] for x in self.g_obj.pos.values()])
        pt_y = np.array([x[1] for x in self.g_obj.pos.values()])
        pt_names = [*self.g_obj.pos.keys()]

        dists = np.hypot(pt_x - event.xdata, pt_y - event.ydata)
        closest_pt = pt_names[dists.argmin()]
        if dists.min() >= self.epsilon:
            closest_pt = None
        return closest_pt

    def on_button_press(self, event):
        """Determine what to do when a button is pressed."""
        if event.inaxes is None:
            return
        if event.inaxes == self.bax:
            self.print_pos()
            return
        if event.button != 1:
            return
        self._clicked_node = self.get_closest_point(event)

    def on_button_release(self, event):
        """Determine what to do when the mouse is released."""
        if event.button != 1:
            return
        self._clicked_node = None
        self.ax.clear()
        self.refresh_plot()

    def on_mouse_move(self, event):
        """Change the node position when the user drags it."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if self._clicked_node:
            self.g_obj.pos[self._clicked_node] = [x, y]

    def refresh_plot(self):
        """Refresh the plot with the new positions."""
        self.g_obj.pos = {pt: np.round(loc, 2) for pt, loc in self.g_obj.pos.items()}
        self.g_obj.draw(fig=self.fig, ax=self.ax, withlegend=False, **self.kwargs)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis('on')

        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.set_aspect('equal')
        self.ax.grid(True, which='both')
        self.ax.set_title('Drag nodes to change their positions')
        self.t += 1
        plt.pause(0.001)

    def print_pos(self):
        """Print the current node positions in the graph from the console."""
        print({k: [float(v[0]), float(v[1])] for k, v in self.g_obj.pos.items()})


"""Example graph for testing. Matches example from FRDL spec."""
ex_nxgraph = nx.DiGraph()
ex_nxgraph.add_nodes_from(["function_a", "function_b", "function_c"],
                          nodetype="function")
ex_nxgraph.add_nodes_from(["external_signals", "control_signal", "external_energy_in",
                           "internal_energy", "external_material_in",
                           "external_material_out", "external_energy_out"],
                          nodetype="flow")
ex_nxgraph.add_edges_from([("function_a", "external_signals"),
                           ("function_a", "control_signal"),
                           ("function_b", "control_signal"),
                           ("function_b", "external_energy_in"),
                           ("function_b", "internal_energy"),
                           ("function_c", "internal_energy"),
                           ("function_c", "external_material_in"),
                           ("function_c", "external_material_out"),
                           ("function_c", "external_energy_out")], edgetype="flow")

ex_nxgraph.add_edge("function_a", "function_b", edgetype="activation",
                    name="new_control_signal")
ex_nxgraph.add_edge("function_b", "function_c", edgetype="activation",
                    name="change_in_energy_usage")
ex_nxgraph.add_edge("function_c", "function_b", edgetype="activation",
                    name="change_in_energy_potential")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
