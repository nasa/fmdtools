#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines style arguments for plotting graphs.

Has Classes:

- :class:`EdgeStyle`: Holds kwargs for nx.draw_networkx_edges to be applied to edges
- :class:`FlowEdgeStyle`: Edge style for flow aggregation.
- :class:`ActivationEdgeStyle`: Edge style for activation/sequence.
- :class:`ContainmentEdgeStyle`: Edge style for containment.
- :class:`ConnectionEdgeStyle`: Edge style for (weak) connections.
- :class:`NodeStyle`: Holds kwargs for nx.draw_networkx_nodes to be applied to nodes
- :class:`BlockNodeStyle`: Node style for :term:`Block`
- :class:`ArchitectureNodeStyle`: Node style for :term:`Architecture`
- :class:`FlowNodeStyle`: Node style for :term:`Flow
- :class:`MultiFlowNodeStyle`: Node style for multiflows.
- :class:`CommsFlowNodeStyle`: Node style for communications flows.
- :class:`ContainerNodeStyle`: Node style for containers.
- :func:`to_legend_label`: Creates a legend label string for the group corresponding to
  style_labels

And functions:

- :func:`edge_style_factory`: Factory for constructing edge styles.
- :func:`node_style_factory`: Factory for constructing node styles.
- :func:`gv_import_check`: Checks if graphviz is installed on the system before plotting
- :func:`save_dot`: Helper function for saving graphviz dot objects.

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

from fmdtools.analyze.common import setup_plot, consolidate_legend

from recordclass import dataobject, asdict
from typing import ClassVar

import networkx as nx
from IPython.display import display, SVG
from matplotlib.colors import Colormap
import matplotlib.lines as mlines


def gv_import_check():
    """Check if graphviz is installed on the system before plotting."""
    try:
        from graphviz import Digraph, Graph
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)
        raise Exception("GraphViz not installed. Please see:",
                        "\n https://pypi.org/project/graphviz/",
                        "\n https://www.graphviz.org/download/")
    return Digraph, Graph


def nx_plot_ending(fig, ax, title='', withlegend=True, saveas='', **leg_kwargs):
    """Add additional options for nx plot."""
    if title:
        ax.set_title(title)
    ax.axis("off")
    if withlegend:
        consolidate_legend(ax, **leg_kwargs)
    if saveas:
        fig.savefig(saveas, bbox_inches='tight')


def gv_plot_ending(dot, disp=True, saveas=''):
    """Add additional options for gv plots."""
    if disp:
        display(SVG(dot._repr_image_svg_xml()))
    save_dot(dot, saveas)


def mod_prefix():
    """Fix for doctest saving--changes save location."""
    import os
    wd = os.getcwd()
    if 'graph' in wd:
        return '../../../docs-source/figures/frdl/'
    else:
        return 'docs-source/figures/frdl/'


class BaseStyle(dataobject, copy_default=True):
    """Base class to define node/edge styles."""

    def __init__(self, *args, styles={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_styles(**styles)

    def set_styles(self, **styles):
        """Modify the style based on a modifier."""
        style_kwargs = {}
        for tagname, tagstyles in styles.items():
            style_kwargs.update(self.modifiers.get(tagname, {}))
            style_kwargs.update(tagstyles.get(tagname, {}))
            for k, v in style_kwargs.items():
                setattr(self, k, v)

    def gv_kwargs(self):
        """Return keyword arguments for dot.edge."""
        return {k[3:]: v for k, v in asdict(self).items() if k.startswith('gv_')}

    def nx_kwargs(self):
        """Return kwargs to nx.draw_networkx_nodes."""
        return {k[3:]: v for k, v in asdict(self).items() if k.startswith('nx_')}


class EdgeStyle(BaseStyle):
    """
    Define style to use to represent an edge.

    Holds kwargs for nx.draw_networkx_edges in nx_arg and for graphviz in gv_arg.

    Parameters
    ----------
    nx_edge_color : str
        Color to represent edges with. Default is 'black'.
    nx_style : str
        Linestyle to represent edges with. Default is 'solid'.
    nx_arrows : bool
        Whether to include arrows. Default is False.
    nx_arrowstyle : bool
        Arrow stle to use. Default is '-|>'.
    nx_arrowsize : int
        Size of arrows to use. Default is 15.
    gv_arrowhead : str
        Graphviz arrowhead.
    gv_color : str
        Graphviz color.
    gv_style : str
        Graphviz line style.
    modifiers : dict
        Modifiers to previous parameters to apply based on particular styles.
    """

    nx_edge_color: ClassVar[str] = 'black'
    nx_style: ClassVar[str] = 'solid'
    nx_arrows: ClassVar[bool] = False
    nx_arrowstyle: ClassVar[str] = '-|>'
    nx_arrowsize: ClassVar[str] = 15
    gv_arrowhead: ClassVar[str] = 'open'
    gv_color: ClassVar[str] = 'black'
    gv_style: ClassVar[str] = 'solid'
    modifiers = dict(degraded=dict(nx_edge_color='orange', gv_color='orange'),
                     active=dict(nx_edge_color='green', gv_color='green'))

    def nx_kwargs(self):
        """Return the style-defined arguments for nx.draw_networkx_edges.v."""
        return {k[3:]: v for k, v in asdict(self).items()
                if (k.startswith('nx_') and
                    not (not self.nx_arrows and k in ('nx_arrowstyle', 'nx_arrowsize')))}

    def nx_legend_line(self, legend_label=''):
        """Return mlines.Line2D patch for legend."""
        if not legend_label:
            legend_label = self.__class__.__name__
        return mlines.Line2D([], [], color=self.nx_edge_color, linestyle=self.nx_style,
                             label=legend_label)

    def show_nx(self, fig=None, ax=None, figsize=(1, 1), withlegend=False, saveas=''):
        """Show how the edge will look in networkx."""
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        d = nx.DiGraph()
        d.add_nodes_from([1, 2])
        d.add_edge(1, 2)
        pos = {1: (0, 0), 2: (1, 0)}
        nx.draw_networkx_edges(d, pos=pos, **self.nx_kwargs())
        lin = self.nx_legend_line(self.__class__.__name__)
        nx_plot_ending(fig, ax, title=self.__class__.__name__, withlegend=withlegend,
                       add_handles=[lin], saveas=saveas)
        return fig, ax

    def show_gv(self, disp=True, saveas=''):
        """Show how the edge will look in graphviz."""
        Digraph, Graph = gv_import_check()
        dot = Digraph()
        dot.edge('0', '1', label=self.__class__.__name__, **self.gv_kwargs())
        gv_plot_ending(dot, disp=disp, saveas=saveas)
        return dot

    def draw_nx(self, g, pos, edges, label='', ax=''):
        """Draw the edges of a graph with networkx."""
        nx.draw_networkx_edges(g, pos, edges, label=label, ax=ax, **self.nx_kwargs())


def save_dot(dot, saveas=''):
    """Save a graphviz diagram."""
    if saveas:
        filecomponents = saveas.split('.')
        dot.render('.'.join(filecomponents[:-1]), format=filecomponents[-1], view=False)


class FlowEdgeStyle(EdgeStyle):
    """
    EdgeStyle representing flows sharing.

    Examples
    --------
    >>> FlowEdgeStyle(nx_arrows=False).nx_kwargs()
    {'edge_color': 'black', 'style': 'solid', 'arrows': False}
    >>> FlowEdgeStyle(nx_arrows=True).nx_kwargs()
    {'edge_color': 'black', 'style': 'solid', 'arrows': True, 'arrowstyle': '-|>', 'arrowsize': 15}
    """

    nx_edge_color: str = 'black'
    nx_style: str = 'solid'
    nx_arrows: bool = True
    nx_arrowstyle: str = '-|>'
    nx_arrowsize: int = 15
    gv_arrowhead: str = 'none'
    gv_color: str = 'black'
    gv_style: str = 'solid'
    gv_arrowtail: str = 'ediamond'
    gv_dir: str = 'both'


class ActivationEdgeStyle(EdgeStyle):
    """EdgeStyle representing activation/conditions."""

    nx_edge_color: str = 'black'
    nx_style: str = 'dashed'
    nx_arrows: bool = True
    nx_arrowstyle: str = '->'
    nx_arrowsize: int = 30
    gv_arrowhead: str = 'open'
    gv_color: str = 'black'
    gv_style: str = 'dashed'


class ContainmentEdgeStyle(EdgeStyle):
    """EdgeStyle representing containment."""

    nx_edge_color: str = 'black'
    nx_style: str = 'solid'
    nx_arrows: bool = True
    nx_arrowstyle: str = '-|>'
    nx_arrowsize: int = 15
    gv_arrowhead: str = 'none'
    gv_arrowtail: str = 'diamond'
    gv_color: str = 'black'
    gv_style: str = 'solid'
    gv_dir: str = 'both'


class ConnectionEdgeStyle(EdgeStyle):
    """EdgeStyle representing activation/conditions."""

    nx_edge_color: str = 'grey'
    nx_style: str = 'dashed'
    nx_arrows: bool = False
    nx_arrowstyle: str = ''
    nx_arrowsize: int = 30
    gv_arrowhead: str = 'none'
    gv_color: str = 'black'
    gv_style: str = 'dashed'


class InheritanceEdgeStyle(EdgeStyle):
    """EdgeStyle representing inheritance."""

    nx_edge_color: str = 'grey'
    nx_style: str = 'solid'
    nx_arrows: bool = True
    nx_arrowstyle: str = '-|>'
    nx_arrowsize: int = 15
    gv_arrowhead: str = 'empty'
    gv_color: str = 'black'
    gv_style: str = 'solid'


def edge_style_factory(style_tag, styles={}, **kwargs):
    """
    Get the appropriate EdgeStyle for networkx plotting.

    Parameters
    ----------
    style_tag : str
        Tag defining the type of edge (e.g. 'flow', 'activation', etc.)
    styles : dict
        edge_styles based on style membership
    **kwargs : kwargs
        Additional keyword arguments for the EdgeStyle

    Examples
    --------
    >>> loc = mod_prefix()
    >>> fs = edge_style_factory('flow')
    >>> fig, ax = fs.show_nx(saveas=loc+'nx_flowconnection.svg')
    >>> sv = fs.show_gv(disp=False, saveas=loc+'gv_flowconnection.svg')

    >>> a_s = edge_style_factory('activation')
    >>> fig, ax = a_s.show_nx(saveas=loc+'nx_activation.svg')
    >>> sv = a_s.show_gv(disp=False, saveas=loc+'gv_activation.svg')

    >>> c_s = edge_style_factory('containment')
    >>> fig, ax = c_s.show_nx(saveas=loc+'nx_containment.svg')
    >>> sv = c_s.show_gv(disp=False, saveas=loc+'gv_containment.svg')

    >>> cs = edge_style_factory('connection')
    >>> fig, ax = cs.show_nx(saveas=loc+'nx_connection.svg')
    >>> sv = cs.show_gv(disp=False, saveas=loc+'gv_connection.svg')
    """
    if style_tag in ['flow', 'flows', 'Flow', 'functions', 'aggregation']:
        style_class = FlowEdgeStyle
    elif style_tag == 'activation':
        style_class = ActivationEdgeStyle
    elif style_tag in ['containment', 'Containment']:
        style_class = ContainmentEdgeStyle
    elif style_tag == 'connection':
        style_class = ConnectionEdgeStyle
    elif style_tag == 'inheritance':
        style_class = InheritanceEdgeStyle
    else:
        raise Exception("Invalid edge style: "+str(style_tag))
    return style_class(styles=styles, **kwargs)


class NodeStyle(BaseStyle):
    """
    Define style to use to represent an edge.

    Holds kwargs for nx.draw_networkx_nodes in nx_arg and for graphviz in gv_arg.

    Parameters
    ----------
    nx_node_shape : str
        Node shape to networkx. Extended by subclasses.
    nx_lindwidths : int
        Width of node edge in networkx. Extended by subclasses.
    nx_node_color : str
        Node color to networkx.
    nx_node_size : int
        Node size in networkx.
    nx_edgecolors : str
        Edge color to networkx.
    nx_cmap : str
        Colormap to use in networkx.
    gv_shape : str
        Node shape to graphviz. Extended by subclasses.
    gv_penwidth : str
        Width of node edge in graphviz. Extended by subclasses.
    gv_style : str
        Style of node in graphviz. Default is 'filled'.
    gv_fillcolor : str
        Fill color in graphviz. Default is 'lightgrey'.
    gv_color : str
        Edge color in graphviz. Default is 'grey'.
    modifiers : dict
        Modifiers to previous parameters to apply based on particular styles.
    """

    nx_node_shape: ClassVar[str] = 'o'
    nx_linewidths: ClassVar[int] = 0
    nx_node_color: str = "lightgrey"
    nx_node_size: int = 500
    nx_edgecolors: str = 'grey'
    nx_cmap: Colormap = None
    nx_vmin: float = None
    nx_vmax: float = None
    gv_shape: ClassVar[str] = 'ellipse'
    gv_penwidth: ClassVar[str] = '0'
    gv_style: ClassVar[str] = 'filled'
    gv_fillcolor: str = 'lightgrey'
    gv_color: str = 'grey'
    modifiers = dict(active=dict(nx_node_color='green', gv_fillcolor='green'),
                     degraded=dict(nx_node_color='orange', gv_fillcolor='orange'),
                     faulty=dict(nx_edgecolors='red', gv_color='red'),
                     high_degree_nodes=dict(nx_node_color='red', gv_fillcolor='red'),
                     static=dict(nx_node_color='cyan', gv_fillcolor='cyan'),
                     dynamic=dict(nx_edgecolors='teal', gv_color='teal'))

    def show_nx(self, fig=None, ax=None, figsize=(1, 1), withlegend=False, saveas=''):
        """Show how the node will look in networkx."""
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        d = nx.DiGraph()
        d.add_node(0)
        nx.draw_networkx_nodes(d, pos={0: (0, 0)}, label=self.__class__.__name__,
                               **self.nx_kwargs())
        ax.set_title(self.__class__.__name__)
        nx_plot_ending(fig, ax, title=self.__class__.__name__, withlegend=withlegend,
                       saveas=saveas)
        return fig, ax

    def show_gv(self, disp=True, saveas=''):
        """Show how the edge will look in graphviz."""
        Digraph, Graph = gv_import_check()
        dot = Digraph()
        dot.node('0', label=self.__class__.__name__, **self.gv_kwargs())
        if disp:
            display(SVG(dot._repr_image_svg_xml()))
        gv_plot_ending(dot, disp=disp, saveas=saveas)
        return dot

    def draw_nx(self, g, pos, nodes, label='', ax=None):
        """Draw the nodes using networkx."""
        nx.draw_networkx_nodes(g, pos, nodes, **self.nx_kwargs(), label=label, ax=ax)


class BlockNodeStyle(NodeStyle):
    """Style representing Functions."""

    nx_node_shape: str = 's'
    nx_linewidths: int = 2
    gv_shape: ClassVar[str] = 'rectangle'
    gv_penwidth: str = '2'


class FunctionNodeStyle(BlockNodeStyle):
    """Style representing Functions."""

    gv_style: str = 'filled'
    gv_shape: str = 'rectangle'


class ActionNodeStyle(BlockNodeStyle):
    """Style representing Actions."""

    gv_style: str = 'rounded, filled'
    gv_shape: str = 'rectangle'


class ComponentNodeStyle(NodeStyle):
    """Style representing Components."""

    gv_style: str = 'filled'
    gv_shape: str = 'trapezium'


class ArchitectureNodeStyle(NodeStyle):
    """Style representing Actions."""

    nx_node_shape: str = '^'
    gv_style: str = 'filled'
    gv_shape: str = 'triangle'


class FlowNodeStyle(NodeStyle):
    """Style representing Flow objects."""

    nx_node_shape: str = 'o'
    nx_linewidths: int = 0
    gv_style: str = 'filled'
    gv_shape: str = 'ellipse'
    gv_penwidth: str = '0'


class MultiFlowNodeStyle(NodeStyle):
    """Style representing MultiFlow objects."""

    nx_node_shape: str = 'p'
    nx_linewidths: int = 0
    gv_style: str = 'filled'
    gv_shape: str = 'pentagon'
    gv_penwidth: str = '0'


class CommsFlowNodeStyle(NodeStyle):
    """Style representing CommsFlow objects."""

    nx_node_shape: str = '8'
    nx_linewidths: int = 0
    gv_style: str = 'filled'
    gv_shape: str = 'octagon'
    gv_penwidth: str = '0'


class ContainerNodeStyle(NodeStyle):
    """Style representing containers."""

    nx_node_shape: str = 's'
    nx_linewidths: int = 0
    gv_style: str = 'filled'
    gv_shape: str = 'tab'
    gv_penwidth: str = '1'


class MethodNodeStyle(NodeStyle):

    nx_node_shape: str = 'd'
    nx_linewidths: int = 0
    gv_style: str = 'filled'
    gv_shape: str = 'component'
    gv_penwidth: str = '1'


class OtherNodeStyle(NodeStyle):
    """Style representing other properties."""
    nx_node_shape: str = "P"
    nx_linewidths: int = 0
    gv_style: str = 'filled'
    gv_shape: str = 'Msquare'
    gv_penwidth: str = '1'


def node_style_factory(style_tag, styles={}, **kwargs):
    """
    Get the keywords for networkx plotting.

    Parameters
    ----------
    styles : dict
        edge_styles/node_styles
    label : tuple
        tuple of tag values to create the keywords for

    Examples
    --------
    >>> loc = mod_prefix()
    >>> fs = node_style_factory('flow')
    >>> fig, ax = fs.show_nx(saveas=loc+'nx_flow.svg')
    >>> sv = fs.show_gv(disp=False, saveas=loc+'gv_flow.svg')

    >>> ms = node_style_factory('multiflow')
    >>> fig, ax = ms.show_nx(saveas=loc+'nx_multiflow.svg')
    >>> sv = ms.show_gv(disp=False, saveas=loc+'gv_multiflow.svg')

    >>> cs = node_style_factory('commsflow')
    >>> fig, ax = cs.show_nx(saveas=loc+'nx_commsflow.svg')
    >>> sv = cs.show_gv(disp=False, saveas=loc+'gv_commsflow.svg')

    >>> fs = node_style_factory('function')
    >>> fig, ax = fs.show_nx(saveas=loc+'nx_function.svg')
    >>> sv = fs.show_gv(disp=False, saveas=loc+'gv_function.svg')

    >>> a_s = node_style_factory('action')
    >>> fig, ax = a_s.show_nx(saveas=loc+'nx_action.svg')
    >>> sv = a_s.show_gv(disp=False, saveas=loc+'gv_action.svg')

    >>> cs = node_style_factory('component')
    >>> fig, ax = cs.show_nx(saveas=loc+'nx_component.svg')
    >>> sv = cs.show_gv(disp=False, saveas=loc+'gv_component.svg')

    >>> cs = node_style_factory('container')
    >>> fig, ax = cs.show_nx(saveas=loc+'nx_container.svg')
    >>> sv = cs.show_gv(disp=False, saveas=loc+'gv_container.svg')

    >>> a_s = node_style_factory('architecture')
    >>> fig, ax = a_s.show_nx(saveas=loc+'nx_architecture.svg')
    >>> sv = a_s.show_gv(disp=False, saveas=loc+'gv_architecture.svg')
    """
    if style_tag in ['flow', 'Flow']:
        node_style = FlowNodeStyle
    elif style_tag in ['multiflow', 'MultiFlow']:
        node_style = MultiFlowNodeStyle
    elif style_tag in ['commsflow', 'CommsFlow', 'environment', 'Environment']:
        node_style = CommsFlowNodeStyle
    elif 'Architecture' in style_tag or 'architecture' in style_tag:
        node_style = ArchitectureNodeStyle
    elif style_tag in ['block', 'Block']:
        node_style = BlockNodeStyle
    elif style_tag in ['function', 'Function']:
        node_style = FunctionNodeStyle
    elif style_tag in ['action', 'Action']:
        node_style = ActionNodeStyle
    elif style_tag in ['component', 'Component']:
        node_style = ComponentNodeStyle
    elif style_tag in ['container', 'state', 'Container', 'State']:
        node_style = ContainerNodeStyle
    elif style_tag in ['condition', 'Condition', 'method']:
        node_style = MethodNodeStyle
    elif style_tag in ['dict', 'flexible']:
        node_style = OtherNodeStyle
    elif style_tag in ['class', 'Class']:
        node_style = OtherNodeStyle
    else:
        raise Exception("Invalid node style: "+str(style_tag))
    return node_style(styles=styles, **kwargs)


def to_legend_label(group_label, style_labels):
    """
    Create a legend label string for the group corresponding to style_labels.

    Parameters
    ----------
    group_label : tuple
        tuple defining the group
    style_labels : list
        properties the tuple is meant to describe

    Returns
    -------
    legend_label : str
        String labeling the group

    Examples
    --------
    >>> to_legend_label(('Function', True, False), ['label', 'static', 'dynamic'])
    'Function, static'
    >>> to_legend_label(('Function', False, True), ['label', 'static', 'dynamic'])
    'Function, dynamic'
    """
    if len(group_label) != len(style_labels):
        raise Exception("Mismatch between groups " + str(group_label) +
                        " and styles "+str(style_labels))
    legend_label = ""
    for i, entry in enumerate(group_label):
        if entry is True:
            legend_label += style_labels[i] + ', '
        elif entry is not False:
            legend_label += entry + ", "
    if legend_label:
        legend_label = legend_label[:len(legend_label)-2]
    return legend_label


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
