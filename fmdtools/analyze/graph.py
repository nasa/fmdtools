"""
Description: Gives graph-level visualizations of the model using installed renderers.

Public user-facing methods:
    - :func:`set_pos`:                      Set graph node positions manually
    - :func:`show`:                         Plots a single graph object g. Has options for heatmaps/overlays and matplotlib/graphviz/pyvis renderers.
    - :func:`exec_order`:                   Displays the propagation order and type (dynamic/static) in the model. Works with matplotlib/graphviz renderers.
    - :func:`history`:                      Displays plots of the graph over time given a dict history of graph objects.  Works with matplotlib/graphviz renderers.
    - :func:`result_from`:                  Plots a representation of the model graph at a specific time in the results history. Works with matplotlib/graphviz renderers.
    - :func:`results_from`:                 Plots a set of representations of the model graph at given times in the results history. Works with matplotlib/graphviz renderers.
    - :func:`animation_from`:               Creates an animation of the model graph using results at given times in the results history.  Works with matplotlib renderers.
Private class:
    - :class:`GraphInteractor`:             Used to set nodes in set_pos
"""
#File Name: analyze/graph.py
#Contributors: Daniel Hulse and Sequoia Andrade
#Created: November 2019
#Refactored: April 2020
#Added major interfaces: July 2021


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
import matplotlib.animation
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from matplotlib import get_backend
from recordclass import dataobject, asdict


default_edge_kwargs={'sends':       dict(edge_color='grey', style='dashed'),
                     'contains':    dict(arrows=True)}
class EdgeStyle(dataobject):
    edge_color: str = 'black'
    style:      str = 'solid'
    arrows:     bool = False
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
            style_kwargs.update(tagstyles.get(label[i],{}))
        return EdgeStyle(**style_kwargs)
    def kwargs(self):
        return asdict(self)

default_node_kwargs={'Model':       dict(node_shape='^'),
                     'Block':       dict(node_shape='s'),
                     'FxnBlock':    dict(node_shape='s'),
                     'Flow':        dict(node_shape='o'),
                     'MultiFlow':   dict(node_shape='h'),
                     'CommsFlow':   dict(node_shape='8'),
                     'State':       dict(node_shape='d'),
                     'degraded':    dict(node_color='orange'),
                     'faulty':      dict(node_color='red')}  

class NodeStyle(dataobject):
    node_color: str="lightgrey"
    node_size:  int=500
    node_shape: str='o'
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
            if type(label[i])==bool and label[i]: 
                style_kwargs.update(default_node_kwargs.get(style, {}))
            else:   
                style_kwargs.update(default_node_kwargs.get(label[i], {}))
            style_kwargs.update(tagstyles.get(label[i],{}))
        return NodeStyle(**style_kwargs)
    def kwargs(self):
        return asdict(self)

class LabelStyle(dataobject):
    font_size:              int=12 
    font_color:             str='k'
    font_weight:            str='normal'
    alpha:                  float=1.0
    horizontalalignment:    str='center'
    verticalalignment:      str='center'
    clip_on:                bool=False
    def kwargs(self):
        return asdict(self)
class EdgeLabelStyle(LabelStyle):
    rotate:                 bool=False
    
class Labels(dataobject, mapping=True):
    title:          dict={}
    title_style:    LabelStyle=LabelStyle()
    subtext:        dict={}
    subtext_style:  LabelStyle=LabelStyle()
    def from_iterator(g, iterator, LabStyle, title='id', title2='', subtext='', **node_label_styles):
        is_edge = 'Edge' in iterator.__class__.__name__
        is_node = 'Node' in iterator.__class__.__name__
        labs = Labels()
        for entry in ['title', 'title2', 'subtext']:
            entryval = vars()[entry]
            if entryval=="id":      evals={n:n for n in iterator}
            elif entryval=="last":  evals={n:n.split("_")[-1] for n in iterator}
            elif entryval=='label': evals={n:'<'+v['label']+'>' for n, v in iterator.items()}
            elif is_edge: evals=nx.get_edge_attributes(g, entryval)
            elif is_node: evals=nx.get_node_attributes(g, entryval)
            else:         evals={}
            if evals:
                if entry=='title':      labs.title=evals
                elif entry=='title2':   labs.title={k: v+': '+evals.get(k,'') for k,v in title.items()}
                elif entry=='subtext':  labs.subtext=evals
        
        node_labels = labs.iter_groups()
        for entry in node_labels:
            if len(labs)>1:
                if entry=='title':          verticalalignment='bottom'
                elif entry=='subtext':      verticalalignment='top'
            else:                           verticalalignment='center'
            if entry=='title' and is_node:  font_weight ='bold'
            else:                           font_weight='normal'
            def_style = dict(verticalalignment=verticalalignment,
                             font_weight=font_weight, **node_label_styles.get(entry,{}))
            labs[entry+'_style'] = LabStyle(**def_style)
        return labs
    def iter_groups(self):
        return [n for n in ['title', 'subtext'] if getattr(self, n)]
    def styles(self):
        return {k:self[k+'_style'] for k in self.iter_groups()}     
    def group_styles(self):
        return {k:(self[k], self[k+'_style']) for k in self.iter_groups()}

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
        labels = {k:tuple(vals[tag] for tag in tags) for k,vals in iterator.items()}
    except KeyError as e:
        unable = {k: tuple(tag for tag in tags if tag not in vals) for k, vals in iterator.items()}
        raise Exception("The following keys lack the following tags: "+str(unable)) from e
    label_groups = {}
    for key, label in labels.items():
        if label in label_groups:   label_groups[label].append(key)
        else:                       label_groups[label]=[key]
    return label_groups

class Graph(object):
    def __init__(self, obj, get_states=True, **kwargs):
        if isinstance(obj, nx.Graph):       self.g=obj
        elif hasattr(self, 'nx_from_obj'):  self.g=self.nx_from_obj(obj, get_states=get_states **kwargs)
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
        if not getattr(self, 'pos', False): self.pos={}
        if auto:
            try:                    self.pos=nx.planar_layout(nx.MultiGraph(self.g))
            except:                 self.pos=nx.spring_layout(nx.MultiGraph(self.g))
        else:                       self.pos = {n:self.pos.get(n, (0.5,0.5)) for n in self.g.nodes}
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
        self.edge_styles={}
        if "label" not in edge_styles: edge_styles["label"]={}
        self.edge_groups = get_label_groups(self.g.edges(), *edge_styles)
        for edge_group in self.edge_groups:
            self.edge_styles[edge_group]=EdgeStyle.from_styles(edge_styles, edge_group)
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
        self.node_styles={}
        if "label" not in node_styles: node_styles['label']={}
        self.node_groups = get_label_groups(self.g.nodes(), *node_styles)
        for node_group in self.node_groups:
            self.node_styles[node_group]=NodeStyle.from_styles(node_styles, node_group)
    def set_edge_labels(self, title='label', title2='', subtext='', **edge_label_styles):
        self.edge_labels = Labels.from_iterator(self.g, self.g.edges, EdgeLabelStyle, title=title, title2=title2, subtext=subtext, **edge_label_styles)
    def set_node_labels(self, title='id', title2='', subtext='', **node_label_styles):
        self.node_labels = Labels.from_iterator(self.g, self.g.nodes, LabelStyle, title=title, title2=title2, subtext=subtext, **node_label_styles)
    def add_node_groups(self, **node_groups):
        group_attrs={}
        for node_group, nodes in node_groups.items():
            group_attrs.update({n:node_group for n in nodes})
        group_attrs.update({n:'' for n in self.g.nodes if n not in group_attrs})
        nx.set_node_attributes(self.g, group_attrs, 'group')
    def draw(self, figsize=(12,10), withlegend=True, title="", **kwargs):
        """
        Draws a networkx graph g with given styles corresponding to the node/edge properties.
    
        Parameters
        ----------
        figsize : tuple, optional
            Size for the figure (plt.figure arg). The default is (12,10).
        pos : dict, optional
            dictionary for node positions. The default is {}.

        nodelabels : str, optional
            How to label the nodes. If "id", gives the full node name. If "last" just
            gives the last part of the node name. The default is "id".

        withlegend : bool, optional
            Whether to include a legend. The default is True.
        title : str, optional
            Title for the plot. The default is "".
        **kwargs : kwargs
            Arguments for various supporting functions (set_pos, set_edge_groups, etc)
    
        Returns
        -------
        fig : matplotlib figure
            Figure object that the graph is plotted on.
        """
        fig = plt.figure(figsize=figsize)
        for to_set in ['pos', 'edge_styles', 'edge_labels', 'node_styles', 'node_labels']:
            if to_set in kwargs or not hasattr(self, to_set):
                set_func = getattr(self, 'set_'+to_set)
                set_func(**kwargs.get(to_set, {}))
        
        for label, edges in self.edge_groups.items():
            nx.draw_networkx_edges(self.g, self.pos, edges, **self.edge_styles[label].kwargs(), label=label)
        
        for level in self.edge_labels.iter_groups():
            nx.draw_networkx_edge_labels(self.g, self.pos, self.edge_labels[level], **self.edge_labels[level+'_style'].kwargs())
        
        for label, nodes in self.node_groups.items():
            nx.draw_networkx_nodes(self.g, self.pos, nodes, **self.node_styles[label].kwargs(), label=label)
        
        for level in self.node_labels.iter_groups():
            nx.draw_networkx_labels(self.g, self.pos, self.node_labels[level], **self.node_labels[level+'_style'].kwargs())
        
        if withlegend:
            legend = plt.legend(labelspacing=2, borderpad=1)
        
        if title: plt.title(title)
        return fig
    def move_nodes(self, g, gtype='fxnflowgraph', **kwargs):
        """
        Sets the position of nodes for plots in analyze.graph using a graphical tool.
        Note: make sure matplotlib is set to plot in an external window (e.g using '%matplotlib qt)
    
        Parameters
        ----------
        g : networkx graph or model or function
            fxngraph or fxnflowgraph graph of the model of interest
        gtype : 'fxngraph' or 'fxnflowgraph', optional
            Type of graph to plot. The default is 'fxngraph'.
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
    def set_degraded(self, other):
        g = self.g 
        nomg = other.g
        for node in g.nodes:   
            g.nodes[node]['degraded'] = g.nodes[node]['states']!=nomg.nodes[node]['states']

class GraphInteractor: 
    """A simple interactive graph for consistent node placement, etc--used in set_pos to set node positions"""
    showverts = True
    epsilon = 0.2  # max pixel distance to count as a vertex hit
    def __init__(self, g_obj, **kwargs):
        self.t=0
        self.fig, (self.bax, self.ax) = plt.subplots(2, gridspec_kw={'height_ratios': [1,10]})
        self.kwargs=kwargs
        self.g_obj=g_obj
        self.g_obj.set_pos()
        self.refresh_plot()
        self._clicked_node=None
        bnext = Button(self.bax, 'Print positions')
        bnext.on_clicked(self.print_pos)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
    def get_closest_point(self, event):
        """Finds the closest node to the given click to see if it should be moved"""
        pt_x = np.array([x[0] for x in self.pos.values()])
        pt_y = np.array([x[1] for x in self.pos.values()])
        pt_names =[*self.pos.keys()]

        dists = np.hypot(pt_x - event.xdata, pt_y - event.ydata)
        closest_pt = pt_names[dists.argmin()]
        if dists.min()>= self.epsilon:
            closest_pt = None
        return closest_pt
    def on_button_press(self, event):
        """Determines what to do when a button is pressed"""
        if event.inaxes is None:
            return
        if event.inaxes==self.bax:
            self.print_pos()
            return
        if event.button != 1:
            return
        self._clicked_node = self.get_closest_point(event)
    def on_button_release(self, event):
        """Determines what to do when the mouse is released"""
        if event.button != 1:
            return
        self._clicked_node = None
        self.ax.clear()
        self.refresh_plot()
    def on_mouse_move(self, event):
        """Changes the node position when the user drags it"""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if self._clicked_node: self.g.pos[self._clicked_node]=[x,y]
    def refresh_plot(self):
        """Refreshes the plot with the new positions."""
        self.g_obj.pos = {pt:np.round(loc,2) for pt, loc in self.g.pos.items()}
        self.g_obj.show(fig=self.fig, **self.kwargs)
        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        limits=plt.axis('on')
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.set_aspect('equal')
        self.ax.grid(True, which='both')
        self.ax.set_title('Drag nodes to change their positions')
        self.t+=1
        plt.pause(0.0001)
        plt.show()
    def print_pos(self):
        print({k:list(v) for k,v in self.pos.items()})


