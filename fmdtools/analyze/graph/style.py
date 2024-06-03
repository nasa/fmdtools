"""
Defines style arguments for plotting graphs.

Shared Method Parameters:

- :data:`edge_modifiers`: Default modified appearance for edges in model network graphs.
- :data:`default_node_kwargs`: Default appearance for nodes in model network graphs.

Has Classes:

- :class:`EdgeStyle`: Holds kwargs for nx.draw_networkx_edges to be applied to edges
- :class:`NodeStyle`: Holds kwargs for nx.draw_networkx_nodes to be applied to nodes
- :class:`LabelStyle`: Holds kwargs for nx.draw_networkx_labels to be applied to labels
- :class:`EdgeLabelStyle`: Controls edge labels to ensure they do not rotate
- :class:`Labels`: Defines a set of labels to be drawn using draw_networkx_labels.
- :class:`GraphInteractor`: Used to set nodes in set_pos when creating interactive graph
- :func:`label_for_entry`: Gets the label from an nx.graph for a given entry.
- :func:`to_legend_label`: Creates a legend label string for the group corresponding to
  style_labels
  - :func:`gv_import_check`: Checks if graphviz is installed on the system before plotting
"""

import networkx as nx
from IPython.display import display, SVG
from matplotlib.colors import Colormap
from recordclass import dataobject, asdict
from fmdtools.analyze.common import setup_plot, consolidate_legend
import matplotlib.lines as mlines
from typing import ClassVar


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


class BaseStyle(dataobject, copy_default=True):
    """Base class to define node/edge styles."""

    def __init__(self, *args, group={}, styles={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_styles(**group)
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

    def show_nx(self):
        """Show how the edge will look in networkx."""
        fig, ax = setup_plot()
        d = nx.DiGraph()
        d.add_nodes_from([1, 2])
        d.add_edge(1, 2)
        pos = {1: (0, 0), 2: (1, 0)}
        nx.draw_networkx_edges(d, pos=pos, **self.nx_kwargs())
        lin = self.nx_legend_line(self.__class__.__name__)
        consolidate_legend(ax, add_handles=[lin])
        ax.set_title(self.__class__.__name__)

    def show_gv(self):
        """Show how the edge will look in graphviz."""
        Digraph, Graph = gv_import_check()
        dot = Digraph()
        dot.edge('0', '1', label=self.__class__.__name__, **self.gv_kwargs())
        display(SVG(dot._repr_image_svg_xml()))


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


def edge_style_factory(style_tag, group={}, styles={}, **kwargs):
    """
    Get the appropriate EdgeStyle for networkx plotting.

    Parameters
    ----------
    styles : dict
        edge_styles/node_styles
    """
    if style_tag == 'flow':
        style_class = FlowEdgeStyle
    elif style_tag == 'activation':
        style_class = ActivationEdgeStyle
    elif style_tag == 'containment':
        style_class = ContainmentEdgeStyle
    elif style_tag == 'connection':
        style_class = ConnectionEdgeStyle
    else:
        raise Exception("Invalid edge style: "+str(style_tag))
    return style_class(styles=styles, group=group, **kwargs)


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
    gv_shape: ClassVar[str] = 'ellipse'
    gv_penwidth: ClassVar[str] = '0'
    gv_style: str = 'filled'
    gv_fillcolor: str = 'lightgrey'
    gv_color: str = 'grey'
    modifiers = dict(active=dict(nx_node_color='green', gv_fillcolor='green'),
                     degraded=dict(nx_node_color='orange', gv_fillcolor='orange'),
                     faulty=dict(nx_edgecolors='red', gv_color='red'),
                     high_degree_nodes=dict(nx_node_color='red', gv_fillcolor='red'),
                     static=dict(nx_node_color='cyan', gv_fillcolor='cyan'),
                     dynamic=dict(nx_edgecolors='teal', gv_color='teal'))

    def show_nx(self):
        """Show how the node will look in networkx."""
        fig, ax = setup_plot()
        d = nx.DiGraph()
        d.add_node(0)
        nx.draw_networkx_nodes(d, pos={0: (0, 0)}, label=self.__class__.__name__,
                               **self.nx_kwargs())
        ax.set_title(self.__class__.__name__)
        consolidate_legend(ax)

    def show_gv(self):
        """Show how the edge will look in graphviz."""
        Digraph, Graph = gv_import_check()
        dot = Digraph()
        dot.node('0', label=self.__class__.__name__, **self.gv_kwargs())
        display(SVG(dot._repr_image_svg_xml()))
        dot = Digraph()


class BlockNodeStyle(NodeStyle):
    """Style representing Functions."""

    nx_node_shape: str = 's'
    nx_linewidths: int = 2
    gv_shape: str = 'rectangle'
    gv_penwidth: str = '2'


class ArchitectureNodeStyle(NodeStyle):
    """Style representing Actions."""

    nx_node_shape: str = '^'
    gv_shape: str = 'triangle'


class FlowNodeStyle(NodeStyle):
    """Style representing Actions."""

    nx_node_shape: str = 'o'
    nx_linewidths: int = 0
    gv_shape: str = 'ellipse'
    gv_penwidth: str = '0'


class MultiFlowNodeStyle(NodeStyle):
    """Style representing Actions."""

    nx_node_shape: str = 'p'
    nx_linewidths: int = 0
    gv_shape: str = 'pentagon'
    gv_penwidth: str = '0'


class CommsFlowNodeStyle(NodeStyle):
    """Style representing Actions."""

    nx_node_shape: str = '8'
    nx_linewidths: int = 0
    gv_shape: str = 'octagon'
    gv_penwidth: str = '0'


class ContainerNodeStyle(NodeStyle):
    """Style representing containers."""

    nx_node_shape: str = 'd'
    nx_linewidths: int = 0
    gv_shape: str = 'diamond'
    gv_penwidth: str = '0'


def node_style_factory(style_tag, group={}, styles={}, **kwargs):
    """
    Get the keywords for networkx plotting.

    Parameters
    ----------
    styles : dict
        edge_styles/node_styles
    label : tuple
        tuple of tag values to create the keywords for
    """
    if style_tag in ['flow', 'Flow']:
        node_style = FlowNodeStyle
    elif style_tag in ['multiflow', 'MultiFlow']:
        node_style = MultiFlowNodeStyle
    elif style_tag in ['commsflow', 'CommsFlow']:
        node_style = CommsFlowNodeStyle
    elif style_tag == 'architecture':
        node_style = ArchitectureNodeStyle
    elif style_tag in ['block', 'Function', 'Action']:
        node_style = BlockNodeStyle
    elif style_tag in ['container', 'state']:
        node_style = ContainerNodeStyle
    else:
        raise Exception("Invalid node style: "+str(style_tag))
    return node_style(styles=styles, group=group, **kwargs)


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


class EdgeLabelStyle(LabelStyle):
    """Holds kwargs for nx.draw_networkx_edge_labels."""

    rotate: bool = False


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

        - 'id' : The name of the node/edge

        - 'last' : The last part (after all "_" characters) of the name of the node/edge

        - 'label' : The label property of the node/edge (usually indicates type)

        - 'faults_and_indicators' : Fault and indicator properties from the node/edge

        - <str> : Any other property corresponding to the key in the node/edge dict

    Returns
    -------
    entryvals : dict
        Dictionary of values to show for the given entry
    """
    if entryname == "id":
        entryvals = {n: n for n in iterator}
    elif entryname == "last":
        entryvals = {n: n.split("_")[-1] for n in iterator}
    elif entryname == 'label':
        entryvals = {n: '<'+v['label']+'>' for n, v in iterator.items()}
    elif entryname == 'faults_and_indicators':
        faults = nx.get_node_attributes(g, 'faults')
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
                      title='id', title2='', subtext='', **node_label_styles):
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
                    labs.title = {k: v+': '+evals.get(k, '')
                                  for k, v in labs.title.items()}
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
                             **node_label_styles.get(entry, {}))
            labs[entry+'_style'] = LabStyle(**def_style)
        return labs

    def iter_groups(self):
        return [n for n in ['title', 'subtext'] if getattr(self, n)]

    def styles(self):
        return {k: self[k+'_style'] for k in self.iter_groups()}

    def group_styles(self):
        return {k: (self[k], self[k+'_style']) for k in self.iter_groups()}


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    FlowEdgeStyle().show_nx()
    FlowEdgeStyle().show_gv()

    ActivationEdgeStyle().show_nx()
    ActivationEdgeStyle().show_gv()

    ContainmentEdgeStyle().show_nx()
    ContainmentEdgeStyle().show_gv()

    ConnectionEdgeStyle().show_nx()
    ConnectionEdgeStyle().show_gv()

    BlockNodeStyle().show_nx()
    BlockNodeStyle().show_gv()

    MultiFlowNodeStyle().show_nx()
    MultiFlowNodeStyle().show_gv()

    CommsFlowNodeStyle().show_nx()
    CommsFlowNodeStyle().show_gv()
