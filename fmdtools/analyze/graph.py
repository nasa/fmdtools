"""
Description: Gives graph-level visualizations of the model using installed renderers.

Public user-facing methods:
    - :func:`set_pos`:                      Set graph node positions manually
    - :func:`show`:                         Plots a single graph object g. Has options for heatmaps/overlays and
                                            matplotlib/graphviz/pyvis renderers.
    - :func:`exec_order`:                   Displays the propagation order and type (dynamic/static) in the model. Works
                                            with matplotlib/graphviz renderers.
    - :func:`history`:                      Displays plots of the graph over time given a dict history of graph objects.
                                            Works with matplotlib/graphviz renderers.
    - :func:`result_from`:                  Plots a representation of the model graph at a specific time in the results
                                            history. Works with matplotlib/graphviz renderers.
    - :func:`results_from`:                 Plots a set of representations of the model graph at given times in the
                                            results history. Works with matplotlib/graphviz renderers.
    - :func:`animation_from`:               Creates an animation of the model graph using results at given times in the
                                            results history.  Works with matplotlib renderers.
Private class:
    - :class:`GraphInteractor`:             Used to set nodes in set_pos
"""

# File Name: analyze/graph.py
# Contributors: Daniel Hulse, Sequoia Andrade, Hannah Walsh, Johan Louwers
# Created: November 2019
# Refactored: MAY 2023
# Added major interfaces: July 2021
# Adopted object-oriented architecture and merged in networks.py: April 2023


import networkx as nx
import numpy as np
import copy
import matplotlib.animation
import matplotlib.pyplot as plt
from numpy.random import random
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from matplotlib import get_backend
from matplotlib.colors import Colormap
from recordclass import dataobject, asdict
from .result import Result, History


plt.rcParams['pdf.fonttype'] = 42

default_edge_kwargs={'sends':       dict(edge_color='grey', style='dashed'),
                     'contains':    dict(arrows=True),
                     'condition':   dict(arrows=True, arrowstyle='->', arrowsize=30),
                     'next':        dict(arrows=True, arrowstyle='->', arrowsize=30, style='dashed')}


class EdgeStyle(dataobject):
    """
    Holds kwargs for nx.draw_networkx_edges to be applied as a style for multiple edges
    """
    edge_color: str = 'black'
    style:      str = 'solid'
    arrows:     bool = False
    arrowstyle: str = '-|>'
    arrowsize:  int = 15

    def from_styles(styles, label):
        """
        Gets the keywords for networkx plotting

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
        return asdict(self)

    def as_gv_kwargs(self):
        """
        Transates elements of the style (arrow, color, style) into kwargs for graphviz

        Returns
        -------
        gv : dict
            kwargs for graphviz
        """
        gv_arrowstyles = {'-|>': 'open',
                          '':   'none',
                          '->': 'normal'}
        gv = {'color':      self.edge_color,
              'style':      self.style}
        if self.arrows:
            gv['arrowhead'] = gv_arrowstyles.get(self.arrowstyle, 'none')
        else:
            gv['arrowhead'] = 'none'
        return gv
        

default_node_kwargs={'Model':       dict(node_shape='^'),
                     'Block':       dict(node_shape='s', linewidths=2),
                     'FxnBlock':    dict(node_shape='s', linewidths=2),
                     'Action':      dict(node_shape='s', linewidths=2),
                     'Flow':        dict(node_shape='o'),
                     'MultiFlow':   dict(node_shape='h'),
                     'CommsFlow':   dict(node_shape='8'),
                     'State':       dict(node_shape='d'),
                     'active':      dict(node_color='green'),
                     'degraded':    dict(node_color='orange'),
                     'faulty':      dict(edgecolors='red'),
                     'high_degree_nodes': dict(node_color='red'),  # TODO this looks to be double. Needs checking
                     'high_degree_nodes': dict(node_color='red'),
                     'static':      dict(node_color='cyan'),
                     'dynamic':     dict(edgecolors='teal')}  


class NodeStyle(dataobject):
    """
    Holds kwargs for nx.draw_networkx_nodes to be applied as a style for multiple nodes
    """
    node_color: str = "lightgrey"
    node_size:  int = 500
    node_shape: str = 'o'
    edgecolors: str = 'grey'
    linewidths: int = 0
    cmap:       Colormap = None

    def from_styles(styles, label):
        """
        Gets the keywords for networkx plotting

        Parameters
        ----------
        styles : dict
            edge_styles/node_styles
        label : tuple
            tuple of tag values to create the keywords for
        """
        style_kwargs = {}
        for i, (style, tagstyles) in enumerate(styles.items()):
            if type(label[i]) == bool and label[i]:
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
    Transates elements of the style (shape, color, width) into kwargs for graphviz

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
    """
    Holds kwargs for nx.draw_networkx_labels to be applied as a style for multiple labels
    """
    font_size:              int = 12
    font_color:             str = 'k'
    font_weight:            str = 'normal'
    alpha:                  float = 1.0
    horizontalalignment:    str = 'center'
    verticalalignment:      str = 'center'
    clip_on:                bool = False
    bbox:                   dict = dict(alpha=0)

    def kwargs(self):
        return asdict(self)


class EdgeLabelStyle(LabelStyle):
    rotate:                 bool = False
    

class Labels(dataobject, mapping=True):
    """
    Defines a set of labels to be drawn using draw_networkx_labels. Labels have
    three distinct parts: 
        - title (upper text for the node/edge)
        - title2 (if provided, uppder text for the node/edge after a colon)
        - subtext (lower text of the node/edge)
        
    title and subtext may both be given different LabelStyles.
    """
    title:          dict = {}
    title_style:    LabelStyle = LabelStyle()
    subtext:        dict = {}
    subtext_style:  LabelStyle = LabelStyle()

    def from_iterator(g, iterator, LabStyle, title='id', title2='', subtext='', **node_label_styles):
        """
        Condstructs the labels from an interator (nodes or edges)

        Parameters
        ----------
        g : nx.graph
        iterator : nx.graph.nodes/edges
        LabStyle : class
            Class to use for label styles (e.g. LabelStyle or EdgeStyle)
        title : str, optional
            property to get for title text. The default is 'id'.
        title2 : str, optional
            property to get for title text after the colon. The default is ''.
        subtext : str, optional
            property to get for the subtext. The default is ''.
        **node_label_styles : TYPE
            :abStyle arguments to overwrite.

        Returns
        -------
        labs : Labels
            Labels corresponding to the given inputs
        """
        is_edge = 'Edge' in iterator.__class__.__name__
        is_node = 'Node' in iterator.__class__.__name__
        labs = Labels()
        for entry in ['title', 'title2', 'subtext']:
            entryval = vars()[entry]
            if entryval == "id":
                evals = {n: n for n in iterator}
            elif entryval == "last":
                evals = {n: n.split("_")[-1] for n in iterator}
            elif entryval == 'label':
                evals = {n: '<'+v['label']+'>' for n, v in iterator.items()}
            elif is_edge:
                evals = nx.get_edge_attributes(g, entryval)
            elif is_node:
                evals = nx.get_node_attributes(g, entryval)
            else:
                evals = {}
            if evals:
                if entry == 'title':
                    labs.title = evals
                elif entry == 'title2':
                    labs.title = {k: v+': '+evals.get(k, '') for k, v in labs.title.items()}
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
            if entry == 'title' and is_node:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            def_style = dict(verticalalignment=verticalalignment,
                             font_weight=font_weight, **node_label_styles.get(entry, {}))
            labs[entry+'_style'] = LabStyle(**def_style)
        return labs

    def iter_groups(self):
        return [n for n in ['title', 'subtext'] if getattr(self, n)]

    def styles(self):
        return {k: self[k+'_style'] for k in self.iter_groups()}

    def group_styles(self):
        return {k: (self[k], self[k+'_style']) for k in self.iter_groups()}


def get_style_kwargs(styles, label, default_kwargs={}, style_class=EdgeStyle):
    """
    Gets the keywords for networkx plotting

    Parameters
    ----------
    styles : dict
        edge_styles/node_styles
    label : tuple
        tuple of tag values to create the keywords for
    styletype : "node"/"edge", optional
        Whether the kwargs are for a node or edge. The default is "edge".

    Returns
    -------
    style_kwargs : dict
        Keyword arguments for nx.draw_networkx_nodes and nx.draw_networkx_edges
    """
    style_kwargs = {}
    for i, tagstyles in enumerate(styles.values()):
        style_kwargs.update(tagstyles[label[i]])
    return style_class(**style_kwargs)


def get_label_groups(iterator, *tags):
    """
    Creates groups of nodes/edges in terms of discrete values for the given tags.

    Parameters
    ----------
    iterator : iterable
        e.g. nx.graph.nodes(), nx.graph.edges()
    *tags : list
        Tags to find in the graph object (e.g., `label`, `status`, etc.)

    Returns
    -------
    label_groups : dict
        Dict of groups of nodes/edges with given tag values. With structure:
        {(tagval1, tagval2...):[list_of_nodes]}
    """
    try:
        labels = {k: tuple(vals[tag] for tag in tags) for k, vals in iterator.items()}
    except KeyError as e:
        unable = {k: tuple(tag for tag in tags if tag not in vals) for k, vals in iterator.items()}
        raise Exception("The following keys lack the following tags: "+str(unable)) from e
    label_groups = {}
    for key, label in labels.items():
        if label in label_groups:
            label_groups[label].append(key)
        else:
            label_groups[label] = [key]
    return label_groups


def to_legend_label(group_label, style_labels):
    """
    Creates a legend label string for the group corresponding to style_labels

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
        if entry == True:
            legend_label+=style_labels[i]+', '
        elif entry != False:
            legend_label+=entry+", "
    if legend_label:
        legend_label = legend_label[:len(legend_label)-2]
    return legend_label


class Graph(object):
    def __init__(self, obj, get_states=True, **kwargs):
        """
        Creates a Graph.
        
        Parameters
        ----------
        obj: object
            must either be a networkx graph (or be a verion of Graph corresponding to the object)
        get_states: bool
            whether to get states for the graph
        **kwargs:
            keyword arguments for self.nx_from_obj
        """
        if isinstance(obj, nx.Graph):
            self.g = obj
        elif hasattr(self, 'nx_from_obj'):
            self.g = self.nx_from_obj(obj, get_states=get_states **kwargs)

    def set_pos(self, auto=True, **pos):
        """
        Sets graph positions to given positions, (automatically or manually)

        Parameters
        ----------
        auto : str, optional
            Whether to auto-layout the node position. The default is True. 
        **pos : nodename=(x,y)
            Positions of nodes to set. Otherwise updates to the auto-layout or (0.5,0.5)
        """
        if not getattr(self, 'pos', False):
            self.pos = {}
        if auto:
            try:
                self.pos = nx.planar_layout(nx.MultiGraph(self.g))
            except:
                self.pos = nx.spring_layout(nx.MultiGraph(self.g))
        else:
            self.pos = {n: self.pos.get(n, (0.5, 0.5)) for n in self.g.nodes}
        self.pos.update(pos)

    def set_edge_styles(self, **edge_styles):
        """
        Sets self.edge_styles and self.edge_groups given the provided edge styles.

        Parameters
        ----------
        **edge_styles : dict, optional
            Dictionary of tags, labels, and styles for the edges that overwrite the default. 
            Has structure {tag:{label:kwargs}}, where kwargs are the keyword arguments to 
            nx.draw_networkx_edges. The default is {"label":{}}.
        """
        self.edge_styles = {}
        if "label" not in edge_styles:
            edge_styles["label"] = {}
        self.edge_groups = get_label_groups(self.g.edges(), *edge_styles)
        for edge_group in self.edge_groups:
            self.edge_styles[edge_group] = EdgeStyle.from_styles(edge_styles, edge_group)
        self.edge_style_labels = [*edge_styles.keys()]

    def set_node_styles(self, **node_styles):
        """
        Sets self.node_styles and self.edge_groups given the provided node styles.

        Parameters
        ----------
        **node_styles : dict, optional
            Dictionary of tags, labels, and style kwargs for the nodes that overwrite the default. 
            Has structure {tag:{label:kwargs}}, where kwargs are the keyword arguments to 
            nx.draw_networkx_nodes. The default is {"label":{}}.
        """
        self.node_styles = {}
        if "label" not in node_styles:
            node_styles['label'] = {}
        self.node_groups = get_label_groups(self.g.nodes(), *node_styles)
        for node_group in self.node_groups:
            self.node_styles[node_group] = NodeStyle.from_styles(node_styles, node_group)
        self.node_style_labels = [*node_styles.keys()]
    def set_edge_labels(self, title='label', title2='', subtext='states', **edge_label_styles):
        """
        Creates labels using Labels.from_iterator for the edges in the graph

        Parameters
        ----------
        title
        title2
        subtext
        edge_label_styles

        Returns
        -------

        """
        self.edge_labels = Labels.from_iterator(self.g, self.g.edges, EdgeLabelStyle, title=title, title2=title2, subtext=subtext, **edge_label_styles)

    def set_node_labels(self, title='id', title2='', subtext='', **node_label_styles):
        """
        Creates labels using Labels.from_iterator for the nodes in the graph

        Parameters
        ----------
        title
        title2
        subtext
        node_label_styles

        Returns
        -------

        """
        self.node_labels = Labels.from_iterator(self.g, self.g.nodes, LabelStyle, title=title, title2=title2, subtext=subtext, **node_label_styles)

    def add_node_groups(self, **node_groups):
        """
        Creates arbitrary groups of nodes which may be then be displayed with different styles

        Parameters
        ----------
        **node_groups : iterable
            
        e.g. 
        graph.add_node_groups(group1=('node1', 'node2'), group2=('node3'))
        graph.set_node_styles(group={'group1':{'color':'green'}, 'group2':{'color':'red'}})
        graph.draw()
        
        would show two different groups of nodes, one with green nodes, and the other with red nodes
        """
        group_attrs = {}
        for node_group, nodes in node_groups.items():
            group_attrs.update({n: node_group for n in nodes})
        group_attrs.update({n: '' for n in self.g.nodes if n not in group_attrs})
        nx.set_node_attributes(self.g, group_attrs, 'group')

    def set_resgraph(self, other=False):
        """
        Standard results processing for results graphs (show faults and degradations)

        Parameters
        ----------
        other : Graph, optional
            Graph to compare with (for degradations). The default is False.
        """
        if not other:
            other = self
        self.set_degraded(other)
        self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='faults')

    def set_degraded(self, other):
        """
        Sets 'degraded' state in underlying networkx graph based on difference between
        states with another Graph object

        Parameters
        ----------
        other : Graph
            (assumed nominal) Graph to compare to
        """
        g = self.g 
        nomg = other.g
        for node in g.nodes:   
            g.nodes[node]['degraded'] = g.nodes[node]['states'] != nomg.nodes[node]['states']
            g.nodes[node]['faulty'] = any(g.nodes[node].get('faults', []))

    def set_heatmap(self, heatmap, cmap=plt.cm.coolwarm, default_color_val=0.0):
        """
        Enables the association and plotting of a heatmap on a graph.
        
        e.g. graph.set_heatmap({'node_1':1.0, 'node_2': 0.0, 'node_3':0.5})
        graph.draw()
        Should draw node_1 the bluest, node_2 the reddest, and node_3 in between.

        Parameters
        ----------
        heatmap : dict/result
            dict/result with keys corresponding to the nodes and values in the range 
            of a heatmap (0-1)
        cmap : mpl.Colormap, optional
            Colormap to use for the heatmap. The default is plt.cm.coolwarm.
        default_color_val : float, optional
            Value to use if a node is not in the heatmap dict. The default is 0.0.
        """
        self.set_node_styles()
        for label, nodes in self.node_groups.items():
            nodes_colors = [heatmap[node] if node in heatmap else default_color_val for node in nodes]
            self.node_styles[label].node_color = nodes_colors
            self.node_styles[label].cmap = cmap

    def draw(self, figsize=(12, 10), title="", fig=False, ax=False, withlegend=True,
             legend_bbox=(1, 0.5), legend_loc="center left", legend_labelspacing=2,
             legend_borderpad=1, **kwargs):
        """
        Draws a networkx graph g with given styles corresponding to the node/edge properties.
    
        Parameters
        ----------
        figsize : tuple, optional
            Size for the figure (plt.figure arg). The default is (12,10).
        title : str, optional
            Title for the plot. The default is "".
        fig : bool, optional
            TODO : Need documentation update
        ax : bool, optional
            TODO : need documentation update
        withlegend : bool, optional
            Whether to include a legend. The default is True.
        legend_bbox : tuple, optional
            bbox to anchor the legend to. The default is (1,0.5) (places legend on the right)
        legend_loc : str, optional
            loc argument for plt.legend. the default is "center left"
        legend_labelspacing : float, optional
            labelspacing argument for plt.legend. the default is "2
        legend_borderpad : str, optional
            borderpad argument for plt.legend. the default is 1
        **kwargs : kwargs
            Arguments for various supporting functions:
                (set_pos, set_edge_styles, set_edge_labels, set_node_styles, set_node_labels, etc)
    
        Returns
        -------
        fig : matplotlib figure
            matplotlib figure to draw
        ax : matplotlib axis
            Ax in the figure
        """
        if not fig:
            fig = plt.figure(figsize=figsize)
        if not ax:
            ax = plt.gca()
        for to_set in ['pos', 'edge_styles', 'edge_labels', 'node_styles', 'node_labels']:
            if to_set in kwargs or not hasattr(self, to_set):
                set_func = getattr(self, 'set_'+to_set)
                set_func(**kwargs.get(to_set, {}))
        
        for label, edges in self.edge_groups.items():
            legend_label = to_legend_label(label, self.edge_style_labels)
            nx.draw_networkx_edges(self.g, self.pos, edges, **self.edge_styles[label].kwargs(), label=legend_label, ax=ax)
        
        for level in self.edge_labels.iter_groups():
            nx.draw_networkx_edge_labels(self.g, self.pos, self.edge_labels[level], **self.edge_labels[level+'_style'].kwargs(), ax=ax)
        
        for label, nodes in self.node_groups.items():
            legend_label = to_legend_label(label, self.node_style_labels)
            nx.draw_networkx_nodes(self.g, self.pos, nodes, **self.node_styles[label].kwargs(), label=legend_label, ax=ax)
        
        for level in self.node_labels.iter_groups():
            nx.draw_networkx_labels(self.g, self.pos, self.node_labels[level], **self.node_labels[level+'_style'].kwargs(), ax=ax)
        
        if withlegend:
            legend = plt.legend(labelspacing=legend_labelspacing, borderpad=legend_borderpad, 
                                bbox_to_anchor=legend_bbox, loc=legend_loc)
        plt.axis('off')
        
        if title:
            plt.title(title)
        return fig, ax
    def move_nodes(self, **kwargs):
        """
        Sets the position of nodes for plots in analyze.graph using a graphical tool.
        Note: make sure matplotlib is set to plot in an external window (e.g using '%matplotlib qt)
    
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
            print("Cannot place nodes in inline version of plot. Use '%matplotlib qt' (or '%matplotlib osx') to open in external window")
        return p

    def draw_from(self, time, history=History(), **kwargs):
        """
        Draws the graph with degraded/fault data at a given time.

        Parameters
        ----------
        time : int
            Time to draw the graph (in the history)
        history : History, optional
            History with nominal and faulty history. The default is History().
        **kwargs : **kwargs
            arguments for Graph.draw

        Returns
        -------
        fig : matplotlib figure
            matplotlib figure to draw
        ax : matplotlib axis
            Ax in the figure
        """
        faulty = history.get_faulty_hist(*self.g.nodes, withtotal=False, withtime=False).get_slice(time)
        fault_nodes = {n: bool(faulty.get(n, 0)) for n in self.g.nodes}
        nx.set_node_attributes(self.g, fault_nodes, 'faulty')
        
        faults = Result(history.get_faults_hist(*self.g.nodes).get_slice(time))
        faults_nodes = {n: [k for k in faults.get(n)] if faults.get(n)
                        else [] for n in self.g.nodes}
        nx.set_node_attributes(self.g, faults_nodes, 'faults')
        
        degraded = history.get_degraded_hist(*self.g.nodes, withtotal=False, withtime=False).get_slice(time)
        deg_nodes = {n: not bool(degraded.get(n, 1)) for n in self.g.nodes}
        nx.set_node_attributes(self.g, deg_nodes, 'degraded')
        
        # nx.set_node_attributes(self.g, state_nodes, 'states')
        self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='faults')
        kwargs['title'] = kwargs.get('title', '')+' t='+str(time)
        if 'fig' in kwargs:
            kwargs['fig'].clf()
        return self.draw(**kwargs)

    def animate_from(self, history, times='all', figsize=(6, 4), **kwargs):
        """
        Successively animates a plot using Graph.draw_from

        Parameters
        ----------
        history : History
            History with faulty and nominal states
        times : list, optional
            List of times to animate over. The default is 'all'.
        figsize : tuple, optional
            Size for the figure. The default is (6,4).
        **kwargs : kwargs

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            Animation object with the given frames
        """
        from functools import partial
        if times == 'all':
            t_inds = [i for i in range(len(history.faulty.time))]
        else:
            t_inds = times
         
        fig = plt.figure(figsize=figsize)
        
        ani = matplotlib.animation.FuncAnimation(fig, partial(self.draw_from, history=history, fig=fig, withlegend=False, **kwargs), frames=t_inds)
        return ani

    def draw_graphviz(self, filename='', filetype='png', **kwargs):
        """
        Draws the graph using pygraphviz for publication-quality figures.
        
        Note that the style may not match one-to-one with the defined none/edge styles.

        Parameters
        ----------
        filename : str, optional
            Name to save the figure to (if saving the figure). The default is ''.
        filetype : str, optional
            Type of file to safe. The default is 'png'.
        **kwargs : kwargs
            kwargs to 

        Returns
        -------
        dot : PyGraphviz DiGraph
            Graph object corresponding to the figure.
        """
        from IPython.display import display, SVG
        Digraph, Graph = gv_import_check()
        dot = Digraph(graph_attr=kwargs)

        for group, nodes in self.node_groups.items():
            gv_kwargs = self.node_styles[group].as_gv_kwargs()
            for node in nodes:
                label = ""
                if node in self.node_labels.title:
                    label+=self.node_labels.title[node]
                if node in self.node_labels.subtext:
                    label+='\n'+str(self.node_labels.subtext[node])
                
                dot.node(node, style="filled", label=label, **gv_kwargs)
            
        for group, edges in self.edge_groups.items():
            gv_kwargs = self.edge_styles[group].as_gv_kwargs()
            for edge in edges:
                label = ""
                if edge in self.edge_labels.title:
                    label+=self.edge_labels.title[edge]
                if edge in self.edge_labels.subtext:
                    label+='\n'+self.edge_labels.subtext[edge]
                
                dot.edge(edge[0], edge[1], label=label, **gv_kwargs)
        
        if filename:
            dot.render(filename=filename, format=filetype)
        else:
            display(SVG(dot._repr_image_svg_xml()))
        
        return dot

    def draw_pyvis(self, filename="graph", width=1000, filt=True, physics=False, notebook=False):
        """
        Method for plotting graphs with pyvis. Produces interactive HTML!

        Parameters
        ----------
        filename : str, optional
            File to save the html to. The default is "typegraph.html".
        width : int, optional
            Width of the frame in px. The default is 1000.
        filt : Dict/Bool, optional
            Whether to display sliders. The default is True.
        physics : Bool, optional
            Whether to use physics during node placement. The default is False.
        notebook : Bool, optional
            todo : requires documentation update.
        Returns
        -------
        n : pyvis object
            pyvis object of the drawn graph
        """
        from pyvis.network import Network
        width = str(width)+"px"

        if isinstance(self, ModelTypeGraph): 
            n = Network(directed=True, layout='hierarchical', width=width, notebook=notebook)
        else:
            n = Network(width=width, notebook=notebook)
        g = self.g.copy()
        nx.set_node_attributes(g, {g: g for g in g.nodes}, name='label')
        
        for nd in g.nodes():  # fixes JSON serializability needed for pyvis
            for attr in g.nodes[nd]:
                if type(g.nodes[nd][attr]) in (set, dict):
                    g.nodes[nd][attr] = str(g.nodes[nd][attr])
        
        n.from_nx(g)
        n.toggle_physics(physics)
        if filt:
            n.show_buttons(filter_=filt)
        n.show(filename+".html")
        return n

    def calc_aspl(self):
        """
        Computes average shortest path length of

        Returns
        -------
        aspl: float
            Average shortest path length
        """
        return nx.average_shortest_path_length(self.g)

    def calc_modularity(self):
        """
        Computes network modularity of the graph.
            
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
        Determines bridging nodes in a graph representation of model mdl. 
        
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

    def plot_bridging_nodes(self, title='bridging nodes', node_kwargs={'node_color': 'red'}, **kwargs):
        """
        Plots bridging nodes using self.draw()

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is 'bridging nodes'.
        node_kwargs : TODO : need documentation update.
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
        Determines highest degree nodes, up to percentile p, in graph representation of model mdl.
        
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

    def plot_high_degree_nodes(self, p=90, title='', node_kwargs={'node_color': 'red'}, **kwargs):
        """
        Plots high-degree nodes using self.draw()

        Parameters
        ----------
        p : int (optional)
            percentile of degrees to return, between 0 and 100
        title : str, optional
            Title for the plot. The default is 'High Degree Nodes'.
        node_kwargs : todo : needs documentation update
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
        Computes robustness coefficient of graph representation of model mdl.
        
        Parameters
        ----------
        trials : int 
            number of times to run robustness coefficient algorithm (result is averaged over all trials)
        seed : int
            optional seed to instantiate test with
        
        Returns
        -------
        RC : robustness coefficient
        """
        g = self.g
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
        Plots degree distribution of graph representation of model mdl.
        
        Returns
        -------
        fig : matplotlib figure
            plot of distribution
        """
        import math
        g = self.g
        degrees = [g.degree(n) for n in g.nodes()]
        degreesSet = set(degrees)
        degreesUnique = list(degreesSet)
        freq = [degrees.count(n) for n in degreesUnique]
        maxFreq = max(freq)
        freqint = list(range(0, maxFreq+1))
        degreeint = list(range(min(degrees), math.ceil(max(degrees))+1))
        degreesSet = set(degrees)  # TODO degreeSet looks to be not used, consider removing - JLO
        degreesUnique = list(degrees)
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

    def sff_model(self, endtime=5, pi=.1, pr=.1, num_trials=100, start_node='random', error_bar_option='off'):
        """
        susc-fix-fail model.
        
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
        g = self.g
        if start_node == 'random':
            nodes = list(g.nodes)
            start_node_selected = nodes[random.randint(0, len(nodes))]
        else:
            start_node_selected = start_node
        num_susc_all_trials = []
        num_fail_all_trials = []
        num_fix_all_trials = []
        for trials in range(0, num_trials):
            num_susc_trial, num_fail_trial, num_fix_trial = sff_one_trial(start_node_selected, g, endtime=endtime, pi=pi, pr=pr)
            num_susc_all_trials.append(num_susc_trial)
            num_fail_all_trials.append(num_fail_trial)
            num_fix_all_trials.append(num_fix_trial)
        num_susc_average = data_average(num_susc_all_trials)
        num_fail_average = data_average(num_fail_all_trials)
        num_fix_average = data_average(num_fix_all_trials)
       
        fig = plt.figure()
        time_list = range(0, endtime+1)
        if error_bar_option == 'on':
            num_susc_lower_error, num_susc_upper_error = data_error(num_susc_all_trials, num_susc_average)
            num_fail_lower_error, num_fail_upper_error = data_error(num_fail_all_trials, num_fail_average)
            num_fix_lower_error, num_fix_upper_error = data_error(num_fix_all_trials, num_fix_average)
            num_susc_asymmetric_error = [num_susc_lower_error, num_susc_upper_error]
            num_fail_asymmetric_error = [num_fail_lower_error, num_fail_upper_error]
            num_fix_asymmetric_error = [num_fix_lower_error, num_fix_upper_error]
            plt.errorbar(time_list, num_susc_average, yerr=num_susc_asymmetric_error, fmt='-o', label='Susceptible')
            plt.errorbar(time_list, num_fail_average, yerr=num_fail_asymmetric_error, fmt='-o', label='Failed')
            plt.errorbar(time_list, num_fix_average, yerr=num_fix_asymmetric_error, fmt='-o', label='Fixed')
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


def sff_one_trial(start_node_selected, g, endtime=5, pi=.1, pr=.1):
    """
    Calculates one trial of the sff model
    
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
    """Averages each column in data"""
    list_average = []
    for i in range(0,len(data[0])):
        list_average.append(sum(x[i] for x in data)/len(data))
    return list_average

def data_error(data, average):
    """
    Calculates error for each column in data

    Parameters
    ----------
    data
    average

    Returns
    -------

    """
    q1 = []
    q3 = []
    for i in range(0, len(data[0])):
        current_array = np.array([float(x[i]) for x in data])
        q1.append(np.percentile(current_array, 25))
        q3.append(np.percentile(current_array, 75))
    lower_error = [x - y for x, y in zip(average, q1)]
    upper_error = [x - y for x, y in zip(q3, average)]
    return lower_error, upper_error


def gv_import_check():
    """
    Checks if graphviz is installed on the system before plotting.
    Returns
    -------

    """
    try:
        from graphviz import Digraph, Graph
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)
        raise Exception("GraphViz not installed. Please see:\n https://pypi.org/project/graphviz/ \n https://www.graphviz.org/download/")
    return Digraph, Graph


class GraphInteractor:
    """A simple interactive graph for consistent node placement, etc--used in set_pos to set node positions"""
    showverts = True
    epsilon = 0.2  # max pixel distance to count as a vertex hit

    def __init__(self, g_obj, **kwargs):
        """

        Parameters
        ----------
        g_obj
        kwargs
        """
        self.t = 0
        self.fig, (self.bax, self.ax) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 10]})
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
        """
        Finds the closest node to the given click to see if it should be moved
        Parameters
        ----------
        event

        Returns
        -------

        """
        pt_x = np.array([x[0] for x in self.g_obj.pos.values()])
        pt_y = np.array([x[1] for x in self.g_obj.pos.values()])
        pt_names = [*self.g_obj.pos.keys()]

        dists = np.hypot(pt_x - event.xdata, pt_y - event.ydata)
        closest_pt = pt_names[dists.argmin()]
        if dists.min() >= self.epsilon:
            closest_pt = None
        return closest_pt

    def on_button_press(self, event):
        """
        Determines what to do when a button is pressed

        Parameters
        ----------
        event

        Returns
        -------
        Returns nothing (intended)

        """
        """"""
        if event.inaxes is None:
            return
        if event.inaxes == self.bax:
            self.print_pos()
            return
        if event.button != 1:
            return
        self._clicked_node = self.get_closest_point(event)

    def on_button_release(self, event):
        """
        Determines what to do when the mouse is released

        Parameters
        ----------
        event

        Returns
        -------
        Returns nothing (intended)
        """
        if event.button != 1:
            return
        self._clicked_node = None
        self.ax.clear()
        self.refresh_plot()

    def on_mouse_move(self, event):
        """
        Changes the node position when the user drags it
        Parameters
        ----------
        event

        Returns
        -------
        Returns nothing (intended)
        """
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
        """
        Refreshes the plot with the new positions.

        Returns
        -------

        """
        self.g_obj.pos = {pt: np.round(loc, 2) for pt, loc in self.g_obj.pos.items()}
        self.g_obj.draw(fig=self.fig, ax=self.ax, withlegend=False, **self.kwargs)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        limits = plt.axis('on')  # TODO : Looks like limits is not used and might be removed from the code.
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.set_aspect('equal')
        self.ax.grid(True, which='both')
        self.ax.set_title('Drag nodes to change their positions')
        self.t+=1
        plt.pause(0.001)

    def print_pos(self):
        """
        TODO : Needs documentation
        Returns
        -------

        """
        print({k: list(v) for k, v in self.g_obj.pos.items()})
    

# INDIVIDUAL GRAPH VARIANTS
# MODELS 
class ModelGraph(Graph):
    """
    Creates a Graph of Model functions and flow for display, where both functions
    and flows are nodes.
    
    If withstates option is used on instantiation, a `states` dict is associated
    with the edges/nodes which can then be used to visualize function/flow attributes.
    """
    def __init__(self, mdl, withstates=True, **kwargs):
        """
        Generates the ModelGraph corresponding to a given Model

        Parameters
        ----------
        mdl : define.Model
            fmdtools model to represent graphically
        withstates : bool, optional
            Whether to copy states to the node/edge 'states' property. The default is True.
        **kwargs : kwargs
            (placeholder for kwargs)
        """
        self.g = self.nx_from_obj(mdl)
        if withstates:
            self.set_nx_states(mdl)

    def nx_from_obj(self, mdl):
        """
        TODO Needs documentation
        Parameters
        ----------
        mdl

        Returns
        -------

        """
        g = mdl.graph.copy()
        labels = {fname: f.get_typename() for fname, f in mdl.fxns.items()}
        labels.update({fname: f.get_typename() for fname, f in mdl.flows.items()})
        nx.set_node_attributes(g, labels, name='label')
        nx.set_edge_attributes(g, 'contains', name='label')
        return g

    def set_nx_states(self, mdl):
        """
        TODO : need documentation
        Parameters
        ----------
        mdl

        Returns
        -------

        """
        self.set_flow_nodestates(mdl)
        self.set_fxn_nodestates(mdl)

    def set_fxn_nodestates(self, mdl):
        """
        TODO : requires documentation
        Parameters
        ----------
        mdl

        Returns
        -------

        """
        fxnfaults, fxnstates = {}, {}
        for fxnname, fxn in mdl.fxns.items():
            fxnstates[fxnname] = asdict(mdl.fxns[fxnname].s)
            fxnfaults[fxnname] = [*mdl.fxns[fxnname].m.faults]
        nx.set_node_attributes(self.g, fxnstates, 'states')
        nx.set_node_attributes(self.g, fxnfaults, 'faults')

    def set_flow_nodestates(self, mdl):
        """
        TODO : requires documentation

        Parameters
        ----------
        mdl

        Returns
        -------

        """
        flowstates = {}
        for flowname, flow in mdl.flows.items():
            flowstates[flowname] = asdict(flow.s)
        nx.set_node_attributes(self.g, flowstates, 'states') 

    def get_multi_edges(self, mdl, subedges):
        """
        Used by subclasses to attach functions/flows (subedges arg) to edges

        Parameters
        ----------
        mdl
        subedges

        Returns
        -------

        """
        flows = {}
        multgraph = nx.projected_graph(mdl.graph, subedges, multigraph=True)
        g = nx.projected_graph(mdl.graph, subedges)
        for edge in g.edges:
            midedges = list(multgraph.subgraph(edge).edges)
            flows[edge] = [midedge[2] for midedge in midedges]
        return flows

    def set_exec_order(self, mdl, static={}, dynamic={}, next_edges={}, label_order=True, label_tstep=True):
        """
        Enables the plotting of ModelGraph execution order.

        Parameters
        ----------
        mdl : Model
            Model to plot the execution order of.
        static : dict/False, optional
            kwargs to overwrite the default style for functions/flows in the static execution step.
            If False, static functions are not differentiated. The default is {}.
        dynamic : dict/False, optional
            kwargs to overwrite the default style for functions/flows in the dynamic execution step.
            If False, dynamic functions are not differentiated. The default is {}.
        next_edges : dict
            kwargs to overwrite the default style for edges indicating the flow order.
            If False, these edges are not added. the default is {}.
        label_order : bool, optional
            Whether to label execution order (with a number on each node). The default is True.
        label_tstep : bool, optional
            Whether to label each timestep (with a number in the subtitle). The default is True.
        """
        node_style_kwargs = {}
        if not static == False:
            staticnodes = list(mdl.staticfxns) + list(set([n for node in mdl.staticfxns for n in mdl.graph.neighbors(node)]))
            nx.set_node_attributes(self.g, {n: n in staticnodes for n in self.g.nodes()}, name='static')
            node_style_kwargs['static'] = static
        if not dynamic == False:
            dynamicnodes = list(mdl.dynamicfxns) 
            orders = {n: str(i) for i, n in enumerate(dynamicnodes)}
            nx.set_node_attributes(self.g, {n: n in orders for n in self.g.nodes()}, name='dynamic')
            node_style_kwargs['dynamic'] = dynamic
        
        if not next_edges == False:
            self.g.add_edges_from([(dynamicnodes[n], dynamicnodes[n+1]) for n in range(len(dynamicnodes)-1) 
                                   if (dynamicnodes[n] in self.g.nodes and dynamicnodes[n+1] in self.g.nodes)], 
                                  label='next')
            self.set_edge_styles(label={'next': next_edges})
        
        if label_order:
            orders.update({n: "" for n in self.g.nodes() if n not in orders})
            nx.set_node_attributes(self.g, orders, name='order')
            title2 = 'order'
        else:
            title2 = ''
        
        if label_tstep:
            tsteps = {n: str(mdl.fxns[n].t.dt) if n in mdl.fxns else "" for n in self.g.nodes}
            nx.set_node_attributes(self.g, tsteps, name='tstep')
            subtext = 'tstep'
        else:
            subtext = ''

        self.set_node_styles(**node_style_kwargs)
        self.set_node_labels(title='id', title2=title2, subtext=subtext)
        
    def draw_graphviz(self, layout="twopi", overlap='voronoi', **kwargs):
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)


class ModelFlowGraph(ModelGraph):
    """
    Creates a Graph of model flows for display, where flows are
    set as nodes and connections (via functions) are edges
    """
    def nx_from_obj(self, mdl):
        g = nx.projected_graph(mdl.graph, mdl.flows)
        labels = {fname: f.get_typename() for fname, f in mdl.flows.items()}
        nx.set_node_attributes(g, labels, name='label')
        fxns = self.get_multi_edges(mdl, mdl.flows)
        edgelabels = {e: str(fl) for e, fl in fxns.items()}
        nx.set_edge_attributes(g, edgelabels, name='functions')
        nx.set_edge_attributes(g, {e: "functions" for e in g.edges()}, name='label')
        return g

    def set_nx_states(self, mdl):
        self.set_flow_nodestates(mdl)

    def set_edge_labels(self, title='label', title2='', subtext='functions', **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext, **edge_label_styles)


class ModelCompGraph(ModelGraph):
    """
    Creates a graph of model functions, and flows, with component containment
    relationships shown for functions.
    """
    def nx_from_obj(self, mdl):
        graph = super().nx_from_obj(mdl)
        for fxnname, fxn in mdl.fxns.items():
            if {**fxn.components, **fxn.actions}: 
                graph.add_nodes_from({**fxn.components, **fxn.actions}, bipartite=1, label="Block")
                graph.add_edges_from([(fxnname, comp) for comp in {**fxn.components, **fxn.actions}])
        return graph

    def set_nx_states(self, mdl):
        self.set_flowgraph_states(mdl)
        self.set_compgraph_blockstates(mdl)

    def set_compgraph_blockstates(self, mdl):
        compfaults, compstates, comptypes, fxnstates, fxnfaults = {}, {}, {}, {}
        for fxnname, fxn in mdl.fxns.items():
            fxnstates[fxnname] = asdict(mdl.fxns[fxnname].s)
            fxnfaults[fxnname] = copy.copy(mdl.fxns[fxnname].m.faults)
            for mode in fxnfaults[fxnname].copy():
                for compname, comp in {**fxn.actions, **fxn.components}.items():
                    compstates[compname] = {}
                    comptypes[compname] = True
                    if mode in comp.faultfaults:
                        compfaults[compname] = compfaults.get(compname, set())
                        compfaults[compname].update([mode])
                        fxnfaults[fxnname].remove(mode)
                        fxnfaults[fxnname].update(['comp_fault'])
        nx.set_node_attributes(self.g, fxnstates, 'states')
        nx.set_node_attributes(self.g, fxnfaults, 'faults')
        nx.set_node_attributes(self.g, compstates, 'states')
        nx.set_node_attributes(self.g, compfaults, 'faults') 
        nx.set_node_attributes(self.g, comptypes, 'iscomponent')


class ModelFxnGraph(ModelGraph):
    """ Returns a graph representation of the functions of the model, where
    functions are nodes and flows are edges"""
    def nx_from_obj(self, mdl):
        g = nx.projected_graph(mdl.graph, mdl.fxns)
        labels = {fname: f.get_typename() for fname, f in mdl.fxns.items()}
        nx.set_node_attributes(g, labels, name='label')
        flows = self.get_multi_edges(mdl, mdl.fxns)
        edgelabels = {e: str(fl) for e, fl in flows.items()}
        nx.set_edge_attributes(g, edgelabels, name='flows')
        nx.set_edge_attributes(g, {e: "flows" for e in g.edges()}, name='label')
        return g

    def set_nx_states(self, mdl):
        self.set_flow_edgestates(mdl)
        self.set_fxn_nodestates(mdl)

    def set_flow_edgestates(self, mdl):
        edgevals = {}
        flows = self.get_multi_edges(mdl, mdl.fxns)
        for edge, flows in flows.items():
            flowdict = {}
            for flow in flows: 
                flowdict[flow] = asdict(mdl.flows[flow].s)
            edgevals[edge] = flowdict
        nx.set_edge_attributes(self.g, edgevals) 

    def set_degraded(self, other):
        super().set_degraded(other)
        g = self.g
        nomg = other.g
        for edge in g.edges:
            degraded = False
            for flow in list(g.edges[edge].keys()):            
                if g.edges[edge][flow] != nomg.edges[edge][flow]:
                    degraded = True
            g.edges[edge]['degraded'] = degraded

    def set_edge_labels(self, title='label', title2='', subtext='flows', **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext, **edge_label_styles)


class ModelTypeGraph(ModelGraph):
    """
    Creates a graph representation of model Classes, showing the containment relationship
    between function classes and flow classes in the model.
    """
    def nx_from_obj(self, mdl, withflows = True, **kwargs):
        """
        Returns a graph with the type containment relationships of the different model constructs.

        Parameters
        ----------
        mdl : TODO Needs documentation update.

        withflows : bool, optional
            Whether to include flows. The default is True.

        Returns
        -------
        g : nx.DiGraph
            networkx directed graph of the type relationships
        """
        g = nx.DiGraph()
        modelname = type(mdl).__name__
        g.add_node(modelname, level=1, label="Model")
        g.add_nodes_from(mdl.fxnclasses(), level=2, label="FxnBlock")
        function_connections = [(modelname, fname) for fname in mdl.fxnclasses()]
        g.add_edges_from(function_connections, label="contains")
        if withflows:
            g.add_nodes_from(mdl.flowtypes(), level=3, label="Flow")
            fxnclass_flowtype = mdl.flowtypes_for_fxnclasses()
            flow_edges = [(fxn, flow) for fxn, flows in fxnclass_flowtype.items() for flow in flows]
            g.add_edges_from(flow_edges, label="contains")
        return g

    def set_nx_states(self, mdl):
        graph = self.g
        flowstates = {}
        for flowtype in mdl.flowtypes():
            flowstates[flowtype] = {flow: asdict(mdl.flows[flow].s) 
                                    for flow in mdl.flows_of_type(flowtype)}
        nx.set_node_attributes(graph, flowstates, 'states')
        fxnstates, fxnfaults = {}, {}
        for fxnclass in mdl.fxnclasses(): 
            fxnstates[fxnclass] = {fxn: asdict(mdl.fxns[fxn].s) 
                                   for fxn in mdl.fxns_of_class(fxnclass)}
            fxnfaults[fxnclass] = {fxn: copy.copy(mdl.fxns[fxn].m.faults) 
                                  for fxn in mdl.fxns_of_class(fxnclass)}
        nx.set_node_attributes(graph, fxnstates, 'states')
        nx.set_node_attributes(graph, fxnfaults, 'faults')

    def set_degraded(self, nomg):
        g = self.g
        rg = self.g.copy()
        for node in g.nodes:
            if g.nodes[node]['level'] == 2:
                faulty = any({fxn for fxn, m in g.nodes[node]['faults'].items() if m not in [['nom'], []]})
                rg.nodes[node]['faulty'] = faulty
            if g.nodes[node]['level'] >= 2:
                degraded = g.nodes[node]['states'] != nomg.nodes[node]['states']
                rg.nodes[node]['degraded'] = degraded
        self.g = rg

    def set_pos(self, auto=True, **pos):
        if auto:
            self.pos = nx.multipartite_layout(self.g, 'level')
        super().set_pos(auto=False, **pos) 

    def draw_graphviz(self, layout="dot", ranksep='2.0', **kwargs):
        return super().draw_graphviz(layout=layout, ranksep=ranksep, **kwargs)

    def set_exec_order(self, *args, **kwargs):
        raise Exception("Cannot specify exec_order for ModelTypeGraph")


# FLOW/MULTIFLOW/COMMSFLOW
class MultiFlowGraph(Graph):
    def __init__(self, flow, include_glob=False,
                               send_connections={"closest": "base"},
                               connections_as_tags=True,
                               include_states=False,
                               get_states=True):
        """
        Creates a networkx graph corresponding to the MultiFlow.
    
        Parameters
        ----------
        include_glob : bool, optional
            Whether to include the base flow (if used). The default is False.
        send_connections : dict/list, optional
            Tags/edges to create as send/recieve connections between local views of the 
            flow without explicit containment relationships.
            
            With structure {in_tag : out_tag}. The default is {}.
            Or structure [(in_node : out_node)]
        include_states:
            whether to include states in the graph
        get_states:
            whether to attach state information as node attributes
    
        Returns
        -------
        g : nx.DiGraph
            Networkx graph corresponding to the MultiFlow
        """
        g = nx.DiGraph()
        if include_glob:
            add_g_nested(g, flow, flow.name, include_states=include_states, get_states=get_states)
        else:
            for loc in flow.locals:
                local_flow = getattr(flow, loc)
                add_g_nested(g, local_flow, loc, include_states=include_states, get_states=get_states)
        if type(send_connections) == dict:
            send_iter = send_connections.items();
            connections_as_tags = True
        elif type(send_connections) == list:
            send_iter = send_connections;
            connections_as_tags = False
        
        for in_tag, out_tag in send_iter:
            for in_node in g.nodes:
                if node_is_tagged(connections_as_tags, in_tag, in_node):
                    for out_node in g.nodes:
                        if (node_is_tagged(connections_as_tags, out_tag, out_node)
                            and not((in_node, out_node) in g.edges) and in_node != out_node):
                            g.add_edge(in_node, out_node, label="sends")
        self.g = g

    def set_resgraph(self, other=False):
        """
        Standard results processing for results graphs (show faults and degradations)

        Parameters
        ----------
        other : Graph, optional
            Graph to compare with (for degradations). The default is False.
        """
        if other:
            self.set_degraded(other)
            self.set_node_styles(degraded={}, faulty={})
        else:
            self.set_degraded(self)
            self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='faults')

    def draw_graphviz(self, layout="neato", overlap='false', **kwargs):
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)


class CommsFlowGraph(MultiFlowGraph):
    def __init__(self, flow, include_glob=False, ports_only=False, get_states=True):
        """
        Creates a graph representation of the CommsFlow (assuming no additional locals)
    
        Parameters
        ----------
        include_glob : bool, optional
            Whether to include the base (root) node. The default is False.
        ports_only : bool, optional
            Whether to only include the explicit port connections betwen flows. The default is False
        with_internal: bool, optional
            Whether to include the internal aspect of the commsflow in the commsflow.
    
        Returns
        -------
        g : networkx.DiGraph
            Graph of the commsflow connections.
        """
        send_connections = []
        for f in flow.fxns:
            int_flow = getattr(flow, f)
            int_ports = int_flow.locals
            out_flow = getattr(flow, f+"_out")
            out_ports = out_flow.locals
            send_connections.append((f, f+"_out"))
            for port in int_ports:
                portname = f+"_"+port
                if port in out_ports:
                    send_connections.append((portname, f+"_out_"+port))
                else:
                    send_connections.append((portname, f+"_out"))
            for f2 in flow.fxns:
                f2_int = getattr(flow, f2)
                if f2 in out_ports:
                    for port in out_ports:
                        portname = f+"_out: "+port
                        if port in f2_int.locals:
                            send_connections.append((portname, f2+": "+port))
                        elif port == f2:
                            send_connections.append((portname, f2))
                else:
                    if f in f2_int.locals:
                        send_connections.append((f+"_out", f2+"_"+f))
                    elif not ports_only:
                        send_connections.append((f+"_out", f2))
                        
        super().__init__(flow, include_glob=include_glob, 
                         send_connections=send_connections, 
                         get_states=get_states)


def node_is_tagged(connections_as_tags, tag, node):
    return (connections_as_tags and (tag in node or (tag == "base" and not("_" in node)))) or tag == node


def add_g_nested(g, multiflow, base_name, include_states=False, get_states=False):
    """
    Helper function for MultiFlow.create_multigraph. Iterates recursively
    through multigraph locals to construct the containment tree.

    Parameters
    ----------
    g : networkx.graph
        Existing graph
    multiflow : MultiFlow
        Multiflow Structure
    base_name : str
        Name at the current level of recursion
    include_states : bool, optional
        Whether to include state attributes in the plot. The default is False.
    get_states : bool, optional
        Whether to attach states as attributes to the graph. The default is False
    """
    if not get_states:
        kwargs = {}
    else:
        kwargs = {"states": multiflow.return_states()}
    g.add_node(base_name, label=multiflow.get_typename(), **kwargs)
    if include_states:
        for state in multiflow.s.__fields__:
            if get_states:
                kwargs = {"states": getattr(multiflow.s, state)}
            g.add_node(base_name+"_"+state, label="State", **kwargs)
            g.add_edge(base_name, base_name+"_"+state, label="contains")
    for loc in multiflow.locals:
        local_flow = getattr(multiflow, loc)
        local_name = base_name+"_"+loc
        if get_states:
            kwargs = {"states": local_flow.return_states()}
        
        g.add_node(local_name, label=local_flow.get_typename(), **kwargs)
        g.add_edge(base_name, local_name, label="contains")
        if local_flow.locals:
            add_g_nested(g, local_flow, local_name)
        if include_states:
            for state in local_flow.s.__fields__:
                if get_states:
                    kwargs = {"states": getattr(multiflow.s, state)}
                g.add_node(local_name+"_"+state, label="State", **kwargs)
                g.add_edge(local_name, local_name+"_"+state, label="contains")

# ASG


class ASGGraph(Graph):
    """
    Shows a visual representation of the internal Action Sequence Graph of 
    the Function Block, with:
        - Sequence as edges
        - Flows as (circular) Nodes
        - Actions as (square) Nodes
    """
    def __init__(self, asg, withstates=True):
        self.g = nx.compose(asg.flow_graph, asg.action_graph) 
        self.set_nx_labels(asg)
        if withstates:
            self.set_nx_states(asg)

    def set_nx_labels(self, asg):
        """
        TODO : requires documentation
        Parameters
        ----------
        asg

        Returns
        -------

        """
        for n in self.g.nodes():
            if n in asg.action_graph.nodes():
                self.g.nodes[n]['label'] = 'Action'
            elif n in asg.flow_graph.nodes():
                self.g.nodes[n]['label'] = 'Flow'
        for e in self.g.edges():
            if e in asg.action_graph.edges():
                self.g.edges[e]['label'] = 'condition'
            elif e in asg.flow_graph.edges():
                self.g.edges[e]['label'] = 'contains'

    def set_nx_states(self, asg):
        """
        TODO : requires documentation

        Parameters
        ----------
        asg

        Returns
        -------

        """
        for g in self.g.nodes():
            self.g.nodes[g]['active'] = g in asg.active_actions
        states = {}
        faults = {}
        for aname, action in asg.actions.items():
            states[aname] = asdict(action.s)
            faults[aname] = [*action.m.faults]
        for fname, flow in asg.flows.items():
            states[fname] = asdict(flow.s)
        nx.set_node_attributes(self.g, states, 'states')
        nx.set_node_attributes(self.g, faults, 'faults')

    def set_edge_labels(self, title='label', title2='', subtext='name', **edge_label_styles):
        """
        TODO : requires documentation
        Parameters
        ----------
        title
        title2
        subtext
        edge_label_styles

        Returns
        -------

        """
        super().set_edge_labels(title=title, title2=title2, subtext=subtext, **edge_label_styles)

    def set_node_styles(self, active={}, **node_styles):
        """
        TODO : requires documentation

        Parameters
        ----------
        active
        node_styles

        Returns
        -------

        """
        super().set_node_styles(active=active, **node_styles)

    def draw_graphviz(self, layout="twopi", overlap='voronoi', **kwargs):
        """
        TODO : requires documentation

        Parameters
        ----------
        layout
        overlap
        kwargs

        Returns
        -------

        """
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)


class ASGActGraph(ASGGraph):
    """
    Variant of ASGGraph where only the sequence between actions is shown.
    """
    def __init__(self, asg, withstates=True):
        self.g = asg.action_graph.copy()
        self.set_nx_labels(asg)
        if withstates:
            self.set_nx_states(asg)


class ASGFlowGraph(ASGGraph):
    """
    Variant of ASGGraph where only the flow relationships between actions is shown.
    """
    def __init__(self, asg, withstates=True):
        self.g = asg.flow_graph.copy()
        self.set_nx_labels(asg)
        if withstates:
            self.set_nx_states(asg)


def graph_factory(obj, **kwargs):
    """
    Creates the default Graph for a given object. Used in fmdtools.sim.get_result

    Parameters
    ----------
    obj : object
        object corresponding to a specific graph type
    **kwargs : kwargs
        Keyword arguments for the Graph class

    Returns
    -------
    graph : Graph
        Graph of the appropriate (default) class
    """
    from fmdtools.define.model import Model 
    from fmdtools.define.flow import CommsFlow, MultiFlow
    from fmdtools.define.block import ASG
    
    if isinstance(obj, Model):
        return ModelGraph(obj, **kwargs)
    elif isinstance(obj, CommsFlow):
        return CommsFlowGraph(obj, **kwargs)
    elif isinstance(obj, MultiFlow):
        return MultiFlowGraph(obj, **kwargs)
    elif isinstance(obj, ASG):
        return ASGGraph(obj, **kwargs)
    else:
        raise Exception("No default graph for class "+obj.__class__.__name__)
