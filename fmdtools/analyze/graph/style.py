"""
Defines style arguments for plotting graphs.

Shared Method Parameters:

- :data:`default_edge_kwargs`: Default appearance for edges in model network graphs.
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
"""

import networkx as nx
from matplotlib.colors import Colormap
from recordclass import dataobject, asdict


default_edge_kwargs = {'sends': dict(edge_color='grey', style='dashed'),
                       'contains': dict(arrows=True),
                       'condition': dict(arrows=True, arrowstyle='->', arrowsize=30),
                       'next': dict(arrows=True, arrowstyle='->',
                                    arrowsize=30, style='dashed')}


class EdgeStyle(dataobject, copy_default=True):
    """Hold kwargs for nx.draw_networkx_edges to apply as a style for multiple edges."""

    edge_color: str = 'black'
    style: str = 'solid'
    arrows: bool = False
    arrowstyle: str = '-|>'
    arrowsize: int = 15

    def from_styles(styles, label):
        """
        Get the keywords for networkx plotting.

        Parameters
        ----------
        styles : dict
            edge_styles/node_styles
        label : tuple
            tuple of tag values to create the keywords for
        """
        style_kwargs = {}
        for i, tagstyles in enumerate(styles.values()):
            style_kwargs.update(default_edge_kwargs.get(label[i], {}))
            style_kwargs.update(tagstyles.get(label[i], {}))
        return EdgeStyle(**style_kwargs)

    def kwargs(self):
        return {k: v for k, v in asdict(self).items()
                if not (not self.arrows and k in ('arrowstyle', 'arrowsize'))}

    def line_kwargs(self):
        return {'color': self.edge_color, 'linestyle': self.style}

    def as_gv_kwargs(self):
        """
        Transate elements of the style (arrow, color, style) into kwargs for graphviz.

        Returns
        -------
        gv : dict
            kwargs for graphviz
        """
        gv_arrowstyles = {'-|>': 'open',
                          '': 'none',
                          '->': 'normal'}
        gv = {'color': self.edge_color,
              'style': self.style}
        if self.arrows:
            gv['arrowhead'] = gv_arrowstyles.get(self.arrowstyle, 'none')
        else:
            gv['arrowhead'] = 'none'
        return gv


default_node_kwargs = {'Model': dict(node_shape='^'),
                       'Block': dict(node_shape='s', linewidths=2),
                       'Function': dict(node_shape='s', linewidths=2),
                       'Action': dict(node_shape='s', linewidths=2),
                       'Flow': dict(node_shape='o'),
                       'MultiFlow': dict(node_shape='h'),
                       'CommsFlow': dict(node_shape='8'),
                       'State': dict(node_shape='d'),
                       'active': dict(node_color='green'),
                       'degraded': dict(node_color='orange'),
                       'faulty': dict(edgecolors='red'),
                       'high_degree_nodes': dict(node_color='red'),
                       'static': dict(node_color='cyan'),
                       'dynamic': dict(edgecolors='teal')}


class NodeStyle(dataobject, copy_default=True):
    """Hold kwargs for nx.draw_networkx_nodes to apply as a style for multiple nodes."""

    node_color: str = "lightgrey"
    node_size: int = 500
    node_shape: str = 'o'
    edgecolors: str = 'grey'
    linewidths: int = 0
    cmap: Colormap = None

    def from_styles(styles, label):
        """
        Get the keywords for networkx plotting.

        Parameters
        ----------
        styles : dict
            edge_styles/node_styles
        label : tuple
            tuple of tag values to create the keywords for
        """
        style_kwargs = {}
        for i, (style, tagstyles) in enumerate(styles.items()):
            if isinstance(label[i], bool) and label[i]:
                style_kwargs.update(default_node_kwargs.get(style, {}))
                style_kwargs.update(**tagstyles)
            else:
                style_kwargs.update(default_node_kwargs.get(label[i], {}))
                style_kwargs.update(tagstyles.get(label[i], {}))
        return NodeStyle(**style_kwargs)

    def kwargs(self):
        return asdict(self)

    def as_gv_kwargs(self):
        """
        Transate elements of the style (shape, color, width) into kwargs for graphviz.

        Returns
        -------
        gv : dict
            kwargs for graphviz
        """
        gv_shapes = {'^': 'triangle',
                     's': 'box',
                     'o': 'ellipse',
                     'h': 'hexagon',
                     '8': 'octagon',
                     'd': 'diamond'}
        gv = dict(fillcolor=self.node_color,
                  color=self.edgecolors,
                  shape=gv_shapes.get(self.node_shape, 'ellipse'),
                  penwidth=str(self.linewidths))
        return gv


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
        return asdict(self)


class EdgeLabelStyle(LabelStyle):
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
    """
    legend_label = ""
    for i, entry in enumerate(group_label):
        if entry is True:
            legend_label += style_labels[i] + ', '
        elif entry is not False:
            legend_label += entry + ", "
    if legend_label:
        legend_label = legend_label[:len(legend_label)-2]
    return legend_label
