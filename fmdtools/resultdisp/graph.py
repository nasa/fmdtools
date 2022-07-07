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
#File Name: resultdisp/graph.py
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

class GraphInteractor:
    """A simple interactive graph for consistent node placement, etc--used in set_pos to set node positions"""
    showverts = True
    epsilon = 0.2  # max pixel distance to count as a vertex hit
    def __init__(self, g, gtype='bipartite', pos=[], **kwargs):
        self.t=0
        self.fig, (self.bax, self.ax) = plt.subplots(2, gridspec_kw={'height_ratios': [1,10]})
        self.g=g
        self.gtype=gtype
        if type(pos)==dict and len(pos)<len(g.nodes):
            pos.update({f:[0.5,0.5] for f in g.nodes if f not in pos})
        pos=get_pos_robust(g, gtype,pos)
        self.pos=pos
        self.kwargs=kwargs
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
        if self._clicked_node: self.pos[self._clicked_node]=[x,y]
    def refresh_plot(self):
        """Refreshes the plot with the new positions."""
        self.pos = {pt:np.round(loc,2) for pt, loc in self.pos.items()}
        show(self.g, self.gtype, pos=self.pos, fig=self.fig, **self.kwargs)
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
        print(self.pos)


def set_pos(g, gtype='bipartite', **kwargs):
    """
    Sets the position of nodes for plots in resultdisp.graph using a graphical tool.
    Note: make sure matplotlib is set to plot in an external window (e.g using '%matplotlib qt)

    Parameters
    ----------
    g : networkx graph or model or function
        normal or bipartite graph of the model of interest
    gtype : 'normal' or 'bipartite', optional
        Type of graph to plot. The default is 'normal'.
    **kwargs : kwargs
        keyword arguments for graph.show_matplotlib

    Returns
    -------
    p : GraphIterator
        Graph Iterator (in resultdisp.Graph)
    """
    if getattr(g,'type', '')=='model':
        mdl=g
        g, pos = get_graph_pos(mdl,kwargs.get('pos',{}), gtype)
    elif getattr(g,'type', '')=='function':
        fxn=g
        g,gtype, kwargs['pos'], kwargs['seqgraph'], kwargs['arrows'] = get_asg_pos(fxn, kwargs.get('pos',{}),gtype, kwargs.get('arrows', False))
    plt.ion()
    p = GraphInteractor(g, gtype, **kwargs)
    if 'inline' in get_backend():
        print("Cannot place nodes in inline version of plot. Use '%matplotlib qt' (or '%matplotlib osx') to open in external window")
    return p

def show(g, gtype='bipartite', renderer = 'matplotlib', filename="", **kwargs):
    """
    Plots a single graph object g.

    Parameters
    ----------
    g : networkx graph or model or function
        The multigraph to plot
    gtype : str (optional)
        Type of graph input to show. Default is 'bipartite.'
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'component'   (for graph/model input): plots functions, flows, and componenets as nodes
        - 'typegraph'   (for graph/model input): plots the class structure of the model, functions, and flows
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    renderer : 'matplotlib' or 'graphviz' or 'pyvis'
        Renderer to use with the drawing. Renderer must be installed. Default is 'matplotlib'
    filename : string, optional
        the filename for the output. The default is '' (in which a file is not saved except in pyvis).
    **kwargs : dictionary
        keyword arguments for the individual methods. See the documentation for
            graph.show_graphviz
            graph.show_maplotlib
            graph.show_pyvis
        for more information on these arguments
    """
    if renderer=='graphviz':
        dot = show_graphviz(g, gtype, filename=filename,  **kwargs)
        return dot
    elif renderer == 'matplotlib':
        fig, ax= show_matplotlib(g, gtype=gtype, filename=filename, **kwargs)
        return fig, ax
    elif renderer == 'pyvis':
        n = show_pyvis(g, gtype=gtype, filename=filename, **kwargs)
        return n
    else: raise Exception("Invalid renderer: "+renderer)

def show_matplotlib(g, gtype='bipartite', filename='', filetype='png', pos=[], scale=1, faultscen=[], time=[], figsize=(6,4), showfaultlabels=True, highlight=[], colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, seqgraph={},seqlabels=False, arrows=False, fig=[]):
    """
    Plots a single graph object g using matplotlib

    Parameters
    ----------
    g : networkx graph or model or function
        The multigraph to plot
    gtype : str (optional)
        Type of graph input to show. Default is 'bipartite.'
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'component'   (for graph/model input): plots functions, flows, and componenets as nodes
        - 'typegraph'   (for graph/model input): plots the class structure of the model, functions, and flows
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    filename : string
        Name to give the saved file, if saved. Default is '' (not saving the file)
    filetype : string
        Type of file to save the figure as (if saving)
    pos : dict
        Positions for nodes
    scale: float
        Changes sizes of nodes in bipartite graph
    faultscen : str, optional
        Name of the fault scenario (for the title). The default is [].
    time : float, optional
        Time of fault injection. The default is [].
    showfaultlabels : bool, optional
        Whether or not to label the faults on the functions. The default is True.
    highlight : list, optional
        Functions/flows to highlight using [faulty functions, degraded functions, degraded flows] labelling scheme.
        Used for custom overlays. Default is []
    colors : list, optional
        List of colors to use for nominal, degraded, and faulty functions/flows.
        Default is: ['lightgray','orange', 'red']
    heatmap : dict, optional
        A heatmap dictionary to overlay on the plot. The default is {}.
    cmap : mpl colormap
        Colormap to use for heatmap visualizations
    arrows : bool, optional
        Whether to display arrows on normal plots (for 'actions' ASGs--default is False)
    seqgraph: networkx graph, optional
        Directed graph to overlay on graph views (for 'flows'/'combined' ASGs--default is {})
    seqlabels: bool
        Whether to show directed edge labels on overlaid seqgraph
    fig : mpl figure
        Current matplotlib figure to plot on
    Returns
    -------
    fig, ax : matplotlib figure/axis
        Matplotlib figure object of the drawn graph
    """
    if getattr(g,'type', '')=='model':
        mdl=g
        g, pos = get_graph_pos(mdl,pos, gtype)
    elif getattr(g,'type', '')=='function':
        fxn=g
        g,gtype, pos, seqgraph, arrows = get_asg_pos(fxn,pos, gtype, arrows)
    elif isinstance(g, nx.classes.graph.Graph): a=1
    else: raise Exception("Invalid object type: "+str(type(g))+" use a model, function, or networkx graph instead")
    pos=get_pos_robust(g, gtype,pos)
    if not fig: fig = plt.figure(figsize=figsize)
    if gtype=='normal':
        edgeflows=dict()
        nodesize=scale*2000
        font_size=scale*12
        for edge in g.edges:
            flows=list(g.get_edge_data(edge[0],edge[1]).keys())
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows if flow not in ['name', 'arrow'])
        if heatmap:
            colors=[]
            for node in g.nodes():
                colors = colors +[heatmap.get(node,0.0)]
                nx.draw_networkx_edges(g,pos, width=2)
            nx.draw_networkx_nodes(g,pos,node_size=nodesize, node_shape='s', node_color=colors, cmap=cmap, alpha=0.7)
            nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeflows, font_size=font_size, font_weight='bold', rotate=False)
            labels={node:node for node in g.nodes}
            nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, font_weight='bold')
        elif highlight:
            faultnodes, degradednodes, faultlabels = highlight_to_labels(highlight, showfaultlabels)
            faultedges = highlight[2]
            faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge])]) for edge in faultedges}
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos,colors=colors, show=False, arrows=arrows)
        else:
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            if not list(g.nodes(data='status'))[0][1]: faultedges = {}; faultflows = {}
            else:
                faultedges = [edge for edge in g.edges if any([g.edges[edge][flow].get('status','nom')=='Degraded' for flow in g.edges[edge]])]
                faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos,colors=colors, show=False, arrows=arrows)
    elif gtype in ['bipartite', 'component']:
        labels={node:node for node in g.nodes}
        functions = [f for f, val in g.nodes.items() if val['bipartite']==0]
        flows = [f for f, val in g.nodes.items() if val['bipartite']==1]
        nodesize=scale*700
        font_size=scale*6
        if heatmap:
            #nx.draw(g, pos, node_size=nodesize,node_color = 'k', alpha=0.3)
            functioncolors = []; flowcolors = []
            for node in functions:
                functioncolors = functioncolors + [heatmap.get(node, 0.0)]
            for node in flows:
                flowcolors = flowcolors + [heatmap.get(node, 0.0)]
            nx.draw_networkx_edges(g, pos)
            nx.draw_networkx_nodes(g, pos, nodelist=functions,  node_color=functioncolors, cmap=cmap, alpha=0.6, node_size=nodesize, node_shape='s')
            nx.draw_networkx_nodes(g, pos, nodelist=flows,  node_color=flowcolors, cmap=cmap, alpha=0.6, node_size=nodesize)
            nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, font_weight='bold')
            if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
        elif highlight:
            faultnodes, degradednodes, faultlabels = highlight_to_labels(highlight, showfaultlabels)
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, functions = functions, flows=flows, show=False, seqgraph=seqgraph, seqlabels=seqlabels)
        else:                                      #plots graph with status information
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, functions = functions, flows=flows, show=False, seqgraph=seqgraph, seqlabels=seqlabels)
    elif gtype == 'typegraph':
        if heatmap or highlight: raise Exception("Invalid option for typegraph--not implemented")
        if "mdl" in locals():
            nx.draw(g, pos=pos, with_labels=True, node_size=scale*700, font_size=scale*8, font_weight='bold', node_color=colors[0])
        else:
            #faultnodes = list({o.__class__.__name__ for f,o in mdl.fxns.items() if o.any_faults()})
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            fig_axis =plot_bipgraph(g,labels, faultnodes, degradednodes, faultlabels, faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, show=False, seqgraph=seqgraph)
    if filename:fig.savefig(filename=filename, format=filetype, bbox_inches = 'tight', pad_inches = 0)
    return fig, fig.axes[0]
def highlight_to_labels(highlight, showfaultlabels):
    """Creates labels dictionary given the highlight list"""
    faultnodes = highlight[0]
    degradednodes = highlight[1]
    if showfaultlabels and type(faultnodes)==dict: 
        faultlabels=faultnodes
    elif showfaultlabels:
        faultlabels= faultlabels = {f:str(i) for i,f in enumerate(faultnodes)}
    else:               faultlabels={}
    return faultnodes, degradednodes, faultlabels
    
def show_graphviz(g, gtype='bipartite', faultscen=[], time=[],filename='',filetype='png', showfaultlabels=True, highlight=[], colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm,arrows=False, seqgraph={}, seqlabels=False, **kwargs):
    """
    Translates an existing nx graph to a graphviz graph. Saves the graph output and dot file.
    Called from show() by passing in graphviz=True and filename

    Parameters
    ----------
    g : networkx graph or model or function
        The multigraph to plot
    gtype : str (optional)
        Type of graph input to show. Default is 'bipartite.'
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'component'   (for graph/model input): plots functions, flows, and componenets as nodes
        - 'typegraph'   (for graph/model input): plots the class structure of the model, functions, and flows
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    filename : string, optional
        the filename for the rendered output (if any). The default is '' (in which the file is not saved).
    filetype : string
        Type of file to save the figure as (if saving)
    faultscen : str, optional
        Name of the fault scenario (for the title). The default is [].
    time : float, optional
        Time of fault injection. The default is [].
    showfaultlabels : bool, optional
        Whether or not to label the faults on the functions. The default is True.
    highlight : list, optional
        Functions/flows to highlight using [faulty functions, degraded functions, degraded flows] labelling scheme.
        Used for custom overlays. Default is []
    colors : list, optional
        List of colors to use for nominal, degraded, and faulty functions/flows.
        Default is: ['lightgray','orange', 'red']
    heatmap : dict, optional
        A heatmap dictionary to overlay on the plot. The default is {}.
    cmap : mpl colormap
        Colormap to use for heatmap visualization
    arrows : bool, optional
        Whether to display arrows on normal plots (for 'actions' ASGs--default is False)
    seqgraph: networkx graph, optional
        Directed graph to overlay on graph views (for 'flows'/'combined' ASGs--default is {})
    seqlabels: bool
        Whether to show directed edge labels on overlaid seqgraph
    **kwargs : dictionary
        dictionary of graphviz attributes used to customize the output.
        this includes layout, overlap, node padding, node separation, font, fontsize, etc.
        see http://www.graphviz.org/doc/info/attrs.html for all options

    Returns
    -------
    dot: a graphviz object

    """
    from IPython.display import display, SVG
    Digraph, Graph = gv_import_check()
    #setting up default layouts for graph types
    if gtype in  ['bipartite', 'component']:
        kwargs["layout"] = kwargs.get("layout", "twopi")
        kwargs["overlap"] = kwargs.get("overlap", "voronoi")
    elif gtype == 'typegraph':
        kwargs["pad"] = kwargs.get("pad", "0.5")
        kwargs["ranksep"] = kwargs.get("ranksep", "2")
    if kwargs.pop('pos',False):     print('invalid option: pos')
    if kwargs.pop('scale', False):  print('invalid option: scale')

    if getattr(g,'type', '')=='model':
        mdl=g
        g, pos = get_graph_pos(mdl,kwargs.pop('pos',[]), gtype)
    elif getattr(g,'type', '')=='function':
        fxn=g
        g,gtype, pos, seqgraph, arrows = get_asg_pos(fxn,kwargs.pop('pos',[]), gtype, arrows)
        a=1
    elif isinstance(g, nx.classes.graph.Graph):
        a=1
    else: raise Exception("Invalid object type: "+str(type(g))+" use a model, function, or networkx graph instead")
    #bipartite
    if gtype in ['bipartite', 'component']:
        functions = [f for f, val in g.nodes.items() if val['bipartite']==0]
        flows = [f for f, val in g.nodes.items() if val['bipartite']==1]
        edges = g.edges
        #handles faults
        labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
        #handles heatmap and highlight
        if highlight != []:
            faultnodes, degradednodes, faultlabels = highlight_to_labels(highlight, showfaultlabels)
        faultlabels_form = {node:'\n\n '+str(fault) for node,fault in faultlabels.items() if fault!={'nom'}}
        colors_dict = gv_colors(g, gtype, colors, heatmap, cmap, faultnodes, degradednodes, functions=functions, flows=flows)
        if seqgraph:    dot = Digraph(comment="model network", graph_attr=kwargs)
        else:           dot = Graph(comment="model network", graph_attr=kwargs)
        dot = plot_gv_bipartite(g, faultnodes, degradednodes, faultlabels_form, faultscen, time, showfaultlabels, colors_dict, functions, flows, edges, dot, seqgraph, seqlabels)
    #typegraph
    elif gtype == 'typegraph':
        dot = Digraph(comment="model type graph network", graph_attr=kwargs)
        for node in g.nodes:
            dot.node(node,style="filled")
        for edge in g.edges:
            dot.edge(edge[0], edge[1])
    #normal graph
    elif gtype == 'normal':
        #handles faults
        edgeflows=dict()
        for edge in g.edges:
            flows=list(g.get_edge_data(edge[0],edge[1]).keys())
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows if flow not in ['name', 'arrow'])
        if highlight != []:
            faultnodes, degradednodes, faultlabels = highlight_to_labels(highlight, showfaultlabels)
            faultedges = highlight[2]
            faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge])]) for edge in faultedges}
        else:
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            if not list(g.nodes(data='status'))[0][1]: faultedges = {}; faultflows = {}
            else:
                faultedges = [edge for edge in g.edges if any([g.edges[edge][flow].get('status','nom')=='Degraded' for flow in g.edges[edge]])]
                faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
        #handles heatmap and highlight
        faultlabels_form = {node:'\n\n '+str(fault) for node,fault in faultlabels.items() if fault!={'nom'}}
        colors_dict = gv_colors(g, gtype, colors=colors, heatmap=heatmap, cmap=cmap, faultnodes=faultnodes, degradednodes=degradednodes, faultedges=faultflows, edgeflows=edgeflows)
        if arrows:  dot = Digraph(comment="model network", graph_attr=kwargs)
        else:       dot = Graph(comment="model network", graph_attr=kwargs)
        dot = plot_gv_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels_form, faultedges, faultscen, time, showfaultlabels, colors_dict, dot)

    #rendering
    dot.attr(outputorder = "edgesfirst")
    if filename:    dot.render(filename = filename+gtype, format = filetype)
    else:           display(SVG(dot._repr_image_svg_xml()))
    return dot

def show_pyvis(g, gtype='typegraph', filename="typegraph", width=1000, filt=True, physics=False, notebook=False):
    """
    Method for plotting graphs with pyvis. Produces interactive HTML!

    Parameters
    ----------
    g : networkx graph or model
        Graph to plot or fmdtools model (which will be used to get the graph)
    gtype : 'hierarchical'/'bipartite'/'component', optional
        Type of model graph to plot The default is 'hierarchical'.
    filename : str, optional
        File to save the html to. The default is "typegraph.html".
    width : int, optional
        Width of the frame in px. The default is 1000.
    filt : Dict/Bool, optional
        Whether to display sliders. The default is True.
    physics : Bool, optional
        Whether to use physics during node placement. The default is False.
    Returns
    -------
    n : pyvis object
        pyvis object of the drawn graph
    """
    from pyvis.network import Network
    if type(g) not in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        mdl=g
        g, pos = get_graph_pos(mdl, [], gtype)
    width = str(width)+"px"

    if gtype=='typegraph':   n = Network(directed=True, layout='hierarchical', width=width, notebook=notebook)
    elif gtype in ["component", "bipartite"]: n = Network(width=width, notebook=notebook)
    else:   raise Exception("Not a valid graph type")
    n.from_nx(g)
    n.toggle_physics(physics)
    if filt: n.show_buttons(filter_=filt)
    n.show(filename+".html")
    return n


def exec_order(mdl, renderer='matplotlib', gtype='bipartite', colors=['lightgray', 'cyan','teal'], show_dyn_order=True, show_dyn_arrows=False, show_dyn_tstep=True, title="Execution Order", legend=True,  **kwargs):
    """
    Displays the execution order/types of the model, where the functions and flows in the
    static step are highlighted and the functions in the dynamic step are listed (with corresponding order)

    Parameters
    ----------
    mdl : fmdtools Model
        Model of the system to visualize.
    renderer : 'matplotlib' or 'graphviz'
        Renderer to use for the graph
    gtype : str, optional
        Representation of the model to use. The default is 'bipartite'.
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    colors : list, optional
        Colors to use for unexecuted functions, static propagation steps, and dynamic functions.
        The default is ['lightgray', 'cyan','teal'].
    show_dyn_order : bool, optional
        Whether to label the execution order for dynamic functions. The default is True.
    show_dyn_tstep : bool, optional
        Whether to label local timesteps of dynamic functions. The default is True.
    show_dyn_arrows:
        Whether to place arrows to denote the sequence between functions. The default is False.
    title : str, optional
        Title for the plot. The default is "Function Execution Order".
    legend : bool, optional
        Whether to show a legend. The default is True.
    **kwargs : see arguments for the respective renderers
    Returns
    -------
    tuple of form (figure, axis)

    """
    if show_dyn_order and show_dyn_tstep:   dyn_highlight = {fxn:str(i)+",dt="+str(mdl.fxns[fxn].dt) if mdl.fxns[fxn].dt!=mdl.tstep else str(i) for i,fxn in enumerate(mdl.dynamicfxns)}
    elif show_dyn_tstep:                    dyn_highlight = {fxn:"dt="+str(mdl.fxns[fxn].dt) if mdl.fxns[fxn].dt!=mdl.tstep else '' for i,fxn in enumerate(mdl.dynamicfxns)}
    elif show_dyn_order:                    dyn_highlight = {fxn:str(i) for i,fxn in enumerate(mdl.dynamicfxns)}
    else:                                   dyn_highlight = list(mdl.dynamicfxns)
    showfaultlabels = (show_dyn_order or show_dyn_tstep)
    if gtype =='normal': fig_axis = show(mdl, renderer=renderer, gtype=gtype, highlight=[dyn_highlight, mdl.staticfxns,  mdl.graph.edges(mdl.staticfxns)], colors=colors, showfaultlabels= showfaultlabels, **kwargs)
    elif gtype=='bipartite':
        staticnodes = list(mdl.staticfxns) + list(set([n for node in mdl.staticfxns for n in mdl.bipartite.neighbors(node)]))
        dynamicnodes = list(mdl.dynamicfxns) #+ list(set().union(*[nx.node_connected_component(mdl.bipartite, node) for node in mdl.dynamicfxns]))
        if show_dyn_arrows:
            seqgraph = nx.DiGraph([(dynamicnodes[n], dynamicnodes[n+1]) for n in range(len(dynamicnodes)-1)])
        else: seqgraph=[]
        fig_axis = show(mdl, renderer=renderer, gtype=gtype, highlight=[dyn_highlight, staticnodes], colors=colors, showfaultlabels= showfaultlabels, seqgraph=seqgraph, **kwargs)
    elif gtype=='actions':
        fig_axis = show(mdl, renderer=renderer, gtype=gtype, highlight=[mdl.actions, [],  []], colors=colors, showfaultlabels= showfaultlabels, arrows=show_dyn_arrows, **kwargs)
    elif gtype in ['flows', 'combined']:
        fig_axis = show(mdl, renderer=renderer, gtype=gtype, highlight=[mdl.actions, [],  []], colors=colors, showfaultlabels= showfaultlabels,  **kwargs)

    if legend:
        if renderer=='graphviz': gv_execute_order_legend(colors)
        else:
            legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0], label='No Execution'),
                               Patch(facecolor=colors[2], edgecolor=colors[2], label='Dynamic Step'),
                               Patch(facecolor=colors[1], edgecolor=colors[1], label='Static Step')]

            fig_axis[1].legend(handles = legend_elements, ncol=3, bbox_to_anchor = (1.0,-0.05))
    if title:
        if renderer=='graphviz':    print('title not implemented in graphviz renderer')
        else:                       fig_axis[1].set_title(title)
    return fig_axis

def history(ghist, **kwargs):
    """
    Displays plots of the graph over time given a dict history of graph objects

    Parameters
    ----------
    ghist : dict
        A dictionary of the history of the graph over time with structure:
       {time: graphobject}, where
           - time is the time where the snapshot of the graph was recorded
           - graphobject is the snapshot of the graph at that time
    **kwargs : kwargs
        keyword arguments for graph.show()
    Returns
    ----------
    figobjs : dict
        Set of graph objects from graph.show() for the given renderer
    """
    figobjs={}
    for time, graph in ghist.items():
        figobjs[time] = show(graph,**kwargs)
    return figobjs

def result_from(mdl, reshist, time, renderer='matplotlib', gtype='bipartite', **kwargs):
    """
    Plots a representation of the model graph at a specific time in the results history.

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from process.hists() or process.typehist() for the typegraph option)
    time : float
        The time in the history to plot the graph at.
    renderer : 'matplotlib' or 'graphviz'
        Renderer to use to plot the graph. Default is 'matplotlib'
    gtype : str (optional)
        Type of graph input to show. Default is 'bipartite.'
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'component'   (for graph/model input): plots functions, flows, and componenets as nodes
        - 'typegraph'   (for graph/model input): plots the class structure of the model, functions, and flows
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    MATPLOTLIB OPTIONS:
    ----------
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    """
    from IPython.display import display, SVG
    [[t_ind,],] = np.where(reshist['time']==time)
    if getattr(mdl,'type', '')=='model':
        g, pos = get_graph_pos(mdl,kwargs.pop('pos',[]), gtype)
    elif getattr(mdl,'type', '')=='function':
        g,_, pos, kwargs['seqgraph'], kwargs['arrows'] = get_asg_pos(mdl,kwargs.pop('pos',[]), gtype, kwargs.get('arrows',False))
    else: raise Exception("Invalid object type: "+str(type(mdl))+" use a model or function instead")
    if renderer=='matplotlib':
        fig  = plt.figure(figsize=kwargs.pop('figsize', (6,4)))
        if gtype=='bipartite':      update_bipplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='typegraph':    update_typegraphplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='normal':       update_graphplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='actions':      update_actplot(t_ind,mdl, reshist, g, pos, **kwargs)
        elif gtype in ['flows', 'combined']: update_flowgraphplot( t_ind,mdl, reshist, g, pos, **kwargs)
        else:           raise Exception("Graph type "+gtype+" not a valid option")
        return fig, plt.gca()
    elif renderer=='graphviz':
        if gtype=='bipartite':                  dot = update_gv_bipplot(t_ind, reshist, g, **kwargs)
        elif gtype=='normal':                   dot = update_gv_graphplot(t_ind, reshist, g, **kwargs)
        elif gtype=='actions':                  dot =update_gv_actplot(t_ind,mdl, reshist, g, pos, **kwargs)
        elif gtype in ['flows', 'combined']:    dot = update_gv_flowgraphplot(t_ind,mdl, reshist, g, pos, **kwargs)
        else:           raise Exception("Graph type "+gtype+" not a valid option for graphviz renderer")
        a=1
        display(SVG(dot._repr_image_svg_xml()))
        return dot
    else: raise Exception("Invalid renderer: "+renderer)

def results_from(mdl, reshist, times, renderer='matplotlib', gtype='bipartite', **kwargs):
    """
    Plots a set of representations of the model graph at given times in the results history.

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from process.hists() or process.typehist() for the typegraph option)
    times : list or 'all'
        The times in the history to plot the graph at. If 'all', plots them all
    renderer : 'matplotlib' or 'graphviz' or
        Renderer to use to plot the graph. Default is 'matplotlib'
    gtype : str (optional)
        Type of graph input to show. Default is 'bipartite.'
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'component'   (for graph/model input): plots functions, flows, and componenets as nodes
        - 'typegraph'   (for graph/model input): plots the class structure of the model, functions, and flows
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    MATPLOTLIB OPTIONS:
    ----------
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    Returns
    ----------
    frames : Dict
        Dictionary of mpl figures keyed at each time {time:fig}
    """
    if getattr(mdl,'type', '')=='model':    g, pos = get_graph_pos(mdl,kwargs.pop('pos',[]), gtype)
    elif getattr(mdl,'type', '')=='function':
        g,_, pos, kwargs['seqgraph'], kwargs['arrows'] = get_asg_pos(mdl,kwargs.pop('pos',[]), gtype, kwargs.get('arrows',False))
    else: raise Exception("Invalid object type: "+str(type(mdl))+" use a model or function instead")
    if times=='all':    t_inds= [i for i in range(0,len(reshist['time']))]
    else:               t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    frames = {}
    if renderer == 'matplotlib':
        for t_ind in t_inds:
            fig = plt.figure(figsize=kwargs.get('figsize', (6,4)))
            if gtype=='bipartite':      update_bipplot(t_ind, reshist, g, pos, show=False, **kwargs)
            elif gtype=='typegraph':    update_typegraphplot(t_ind, reshist, g, pos, show=False, **kwargs)
            elif gtype=='normal':       update_graphplot(t_ind, reshist, g, pos, show=False, **kwargs)
            elif gtype=='actions':      update_actplot(t_ind,mdl, reshist, g, pos, **kwargs)
            elif gtype in ['flows', 'combined']: update_flowgraphplot(t_ind,mdl, reshist, g, pos, **kwargs)
            else:           raise Exception("Graph type "+gtype+" not a valid option")
            frames[t_ind] = fig
    elif renderer == 'graphviz':
        for t_ind in t_inds:
            if gtype=='bipartite':                  dot = update_gv_bipplot(t_ind, reshist, g, **kwargs)
            elif gtype=='normal':                   dot = update_gv_graphplot(t_ind, reshist, g, **kwargs)
            elif gtype=='actions':                  dot =update_gv_actplot(t_ind,mdl, reshist, g, pos, **kwargs)
            elif gtype in ['flows', 'combined']:    dot = update_gv_flowgraphplot(t_ind,mdl, reshist, g, pos, **kwargs)
            else:           raise Exception("Graph type "+gtype+" not a valid option for graphviz renderer")
            frames[t_ind] = dot
    return frames

def animation_from(mdl, reshist, times='all', faultscen=[], gtype='bipartite',figsize=(6,4), showfaultlabels=True, scale=1, show=False, pos=[], colors=['lightgray','orange', 'red'], **kwargs):
    """
    Creates an animation of the model graph using results at given times in the results history.
    To view, use %matplotlib qt from spyder or %matplotlib notebook from jupyter
    To save (or do anything useful)h, make sure ffmpeg is installed  https://www.wikihow.com/Install-FFmpeg-on-Windows

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from process.hists() or process.typehist() for the typegraph option)
    times : list or 'all'
        The times in the history to plot the graph at. If 'all', plots them all
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str (optional)
        Type of graph input to show. Default is 'bipartite.'
        - 'normal'      (for graph/model input): plots functions as nodes and flows as edges
        - 'bipartite'   (for graph/model input): plots functions and flows as nodes
        - 'component'   (for graph/model input): plots functions, flows, and componenets as nodes
        - 'typegraph'   (for graph/model input): plots the class structure of the model, functions, and flows
        - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
        - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
        - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        NOTE: Not all gtypes and options are supported by all renderers. See show_<renderer> for more details
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    show : bool, optional
        Whether to show the plot at the end (may be redundant). The default is True.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    """
    if getattr(mdl,'type', '')=='model':      g, pos = get_graph_pos(mdl,kwargs.pop('pos',[]), gtype)
    elif getattr(mdl,'type', '')=='function': g,_, pos, seqgraph, arrows = get_asg_pos(mdl,kwargs.pop('pos',[]), gtype, kwargs.get('arrows',False))
    else: raise Exception("Invalid object type: "+str(type(mdl))+" use a model or function instead")
    if times=='all':    t_inds= [i for i in range(0,len(reshist['time']))]
    else:   t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]

    if gtype=='bipartite':  update_plot = update_bipplot
    elif gtype=='normal':   update_plot = update_graphplot
    elif gtype=='typegraph':update_plot = update_typegraphplot
    fig = plt.figure(figsize=figsize)
    if gtype in ['bipartite', 'normal', 'typegraph']:
        ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=t_inds, fargs=(reshist, g, pos, faultscen, showfaultlabels, scale, False, colors))
    elif gtype=='actions':
        ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=t_inds, fargs=(mdl,reshist, g, pos, faultscen, showfaultlabels, scale, False, colors, arrows))
    elif gtype=='flows':
        ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=t_inds, fargs=(mdl,reshist, g, pos, faultscen, showfaultlabels, scale, False, colors))
    elif gtype=='combined':
        ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=t_inds, fargs=(mdl,reshist, g, pos, faultscen, showfaultlabels, scale, False, colors, seqgraph, kwargs.get('seqlabels', True)))
    if show: plt.show()
    return ani

###HELPER FUNCTIONS
#############################
def get_graph_pos(mdl, pos, gtype):
    """Helper function for getting the right graph/positions from a model"""
    if gtype=='normal':
        g = mdl.graph.copy()
        if not pos: pos=mdl.graph_pos
    elif gtype=='bipartite':
        g = mdl.bipartite.copy()
        if not pos: pos=mdl.bipartite_pos
    elif gtype=='typegraph':
        g=mdl.return_typegraph()
    elif gtype=='component':
        g = mdl.return_stategraph('component')
    else: raise Exception("Graph type "+gtype+" not valid")
    pos=get_pos_robust(g, gtype,pos)
    return g,pos
def get_pos_robust(g, gtype='bipartite', pos={}):
    """Tries to get the best positions for the graph"""
    if not pos:
        if gtype=='typegraph': pos=nx.multipartite_layout(g, 'level')
        else:
            try: pos=nx.planar_layout(g)
            except: pos=nx.spectral_layout(g)
    return pos
def get_asg_pos(fxn, pos, gtype, arrows):
    """Helper function for getting the right graph/positions from a function."""
    if not pos: pos=fxn.asg_pos
    if gtype=='actions':
        gtype='normal'
        g= fxn.action_graph; seqgraph={}; arrows=True
        if not pos: pos = getattr(fxn, 'action_graph_pos', {})
    elif gtype=='flows':
        gtype='bipartite'
        g= fxn.flow_graph; seqgraph={}
        if not pos: pos = getattr(fxn, 'flow_graph_pos', {})
    elif gtype=='combined':
        gtype='bipartite'
        g= fxn.flow_graph; seqgraph=fxn.action_graph
        if not pos: pos = getattr(fxn, 'flow_graph_pos', {})
    else: raise Exception("Graph type "+gtype+" not valid")
    pos = get_pos_robust(g, gtype, pos)
    return g,gtype, pos, seqgraph, arrows
def get_graph_annotations(g, gtype='bipartite'):
    """Helper method that returns labels/lists degraded nodes for the plot annotations"""
    labels={node:node for node in g.nodes}
    statuses=dict(g.nodes(data='status', default='Nominal'))
    faultnodes=[node for node,status in statuses.items() if status=='Faulty' or 'Faulty' in status]
    degradednodes=[node for node,status in statuses.items() if status=='Degraded' or 'Degraded' in status]
    faults=dict(g.nodes(data='modes', default={'nom'}))
    if gtype=='typegraph':
        faultlabels = {fclass:set(fxns.keys()) for fclass, fxns in g.nodes(data='modes') if fxns and set([mode for modes in fxns.values() for mode in modes if mode!='nom'])}
    else: faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
    return labels, faultnodes, degradednodes, faults, faultlabels
def get_plotlabels(g, reshist, t_ind):
    """
    Assigns labels to a graph g from reshist at time t so that it can be plotted

    Parameters
    ----------
    g : networkx graph
        The graph to get labels for
    reshist : dict
        The dict of results history over time (from process.hists() or process.typehist() for the typegraph option)
    t_ind : float
        The time in reshist to update the graph at

    Returns
    -------
    labels : dict
        labels for the graph.
    faultfxns : dict
        functions with faults in them
    degfxns : dict
        functions that are degraded
    degflows : dict
        flows that are degraded
    faultlabels : dict
        names of each fault
    faultedges : dict
        edges with faults in them
    faultedgeflows : dict
        names of flows that are degraded on each edge
    edgelabels : dict
        labels of each edge
    """
    labels={node:node for node in g.nodes}
    functions = reshist['functions'].keys()

    faultfxns = []
    degfxns = []
    degflows = []
    faultlabels = {}
    edgelabels=dict()
    for edge in g.edges:
        flows=list(g.get_edge_data(edge[0],edge[1]).keys())
        edgelabels[edge[0],edge[1]]=''.join(flow for flow in flows)
    for function in functions:
        if reshist['functions'][function]['numfaults'][t_ind]:
            faultfxns+=[function]
            if type(reshist['functions'][function]['faults']) == dict:
                faultlabels[function] = {fault for fault, occ in reshist['functions'][function]['faults'].items() if occ[t_ind]}
            else: faultlabels[function] = reshist['functions'][function]['faults'][t_ind]
        if not reshist['functions'][function]['status'][t_ind]:
            degfxns+=[function]
    flows = reshist['flows'].keys()
    for flow in flows:
        if not reshist['flows'][flow][t_ind]==1:
            degflows+=[flow]
    faultedges = [edge for edge in g.edges if any([reshist['flows'][flow][t_ind]==0 for flow in g.edges[edge].keys()])]
    faultedgeflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if reshist['flows'][flow][t_ind]==0)]) for edge in faultedges}
    return labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgelabels

def get_asg_plotlabels(g, fxn, reshist, t_ind):
    """
    Assigns labels to the ASG graph g in the given fxn from reshist at time t so that it can be plotted

    Parameters
    ----------
    g : networkx graph
        The graph to get labels for
    fxn : FxnBlock
        Corresponding function block for the graph g
    reshist : dict
        The dict of results history over time (from process.hists() or process.typehist() for the typegraph option)
    t_ind : float
        The time in reshist to update the graph at

    Returns
    -------
    labels : dict
        labels for the graph.
    faultfxns : dict
        functions with faults in them
    degfxns : dict
        functions that are degraded
    degflows : dict
        flows that are degraded
    faultlabels : dict
        names of each fault
    faultedges : dict
        edges with faults in them
    faultedgeflows : dict
        names of flows that are degraded on each edge
    edgelabels : dict
        labels of each edge
    """
    labels={node:node for node in g.nodes}
    fxnname=fxn.name
    rhist = reshist['functions'][fxnname]
    actions = fxn.actions

    faultfxns = []
    degfxns = []
    degflows = []
    faultlabels = {}
    edgelabels=dict()
    for edge in g.edges:
        edgelabels[edge[0],edge[1]]=g.get_edge_data(edge[0],edge[1]).get('name','')
    for action in actions:
        if rhist[action]['numfaults'][t_ind]:
            faultfxns+=[action]
            if type(rhist[action]['faults']) == dict:
                faultlabels[action] = {fault for fault, occ in rhist[action]['faults'].items() if occ[t_ind]}
            else: faultlabels[action] = rhist['faults'][t_ind]
        if not rhist['status'][t_ind]:
            degfxns+=[action]
    flows = [flow for flow in {**fxn.flows, **fxn.internal_flows} if flow in g]
    for flow in flows:
        if flow in rhist and any([v[t_ind]!=1 for v in rhist[flow].values()]):
            degflows+=[flow]
        elif flow in reshist['flows'] and not reshist['flows'][flow][t_ind]==1:
            degflows+=[flow]
    faultedges = [] #[edge for edge in g.edges if any([reshist['flows'][flow][t_ind]==0 for flow in g.edges[edge].keys()])]
    faultedgeflows = {} #{edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if reshist['flows'][flow][t_ind]==0)]) for edge in faultedges}
    return labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgelabels
###MATPLOTLIB HELPER FUNCTIONS
#############################
def plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True, colors=['lightgray','orange', 'red'], title=[], show_edgelabels=True, arrows=False, **kwargs):
    """ Plots a standard graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*2000
    font_size=scale*12
    pos=get_pos_robust(g, 'normal',pos)
    nx.draw_networkx(g,pos,node_size=nodesize,font_size=font_size, node_shape='s',edge_color='gray', node_color=colors[0], width=3, font_weight='bold')
    if show_edgelabels: nx.draw_networkx_edge_labels(g,pos,font_size=font_size, edge_labels=edgeflows, rotate=False)
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_shape='s',node_color = colors[2], node_size = nodesize*1.2)
    nx.draw_networkx_nodes(g, pos, nodelist=degfxns,node_shape='s', node_color = colors[1], node_size = nodesize)
    nx.draw_networkx_edges(g,pos,edgelist=faultedges, edge_color=colors[1], arrows=arrows)

    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=faultedgeflows,font_size=font_size, font_color=colors[1], rotate=False)
    plt.axis('off')
    return plt.gcf(), plt.gca()

def plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True, colors=['lightgray','orange', 'red'], title=[],functions=[], flows=[], seqgraph={}, seqlabels=True, **kwargs):
    """ Plots a bipartite graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*700
    font_size=scale*8
    pos=get_pos_robust(g,'bipartite',pos)
    if functions and flows:
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_nodes(g, pos, nodelist = functions, node_shape='s', node_size=nodesize, node_color = colors[0])
        nx.draw_networkx_nodes(g, pos, nodelist = flows, node_size=nodesize, node_color = colors[0])
        degfxns = [node for node in degnodes if node in functions]
        degflows = [node for node in degnodes if node in flows]
        square_faultfxns = [f for f in faultfxns if f in functions]
        circle_faultfxns = [f for f in faultfxns if f in flows]
        nx.draw_networkx_nodes(g, pos, nodelist=square_faultfxns, node_shape='s', node_color = colors[2], node_size=nodesize*1.2)
        nx.draw_networkx_nodes(g, pos, nodelist=circle_faultfxns, node_color = colors[2], node_size=nodesize*1.2)
        nx.draw_networkx_nodes(g, pos, nodelist=degfxns, node_shape='s', node_color = colors[1], node_size=nodesize)

        nx.draw_networkx_nodes(g, pos, nodelist=degflows,node_color = colors[1], node_size=nodesize)
        nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size,font_weight='bold')

    elif functions or flows:
        raise Exception("Invalid option--either provide list of functions and flows, or neither")
    else:
        nx.draw(g, pos, labels=labels,font_size=font_size, node_size=nodesize, node_color = colors[0], font_weight='bold')
        nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_color = colors[2], node_size=nodesize*1.2)
        nx.draw_networkx_nodes(g, pos, nodelist=degnodes,node_color = colors[1], node_size=nodesize)
    if seqgraph:
        nx.draw_networkx_edges(seqgraph, pos, arrows=True, arrowsize=nodesize/20)
        if seqlabels:
            edge_labels = {(in_node, out_node): label for in_node, out_node, label in seqgraph.edges(data='name') if label}
            nx.draw_networkx_edge_labels(seqgraph, pos, edge_labels=edge_labels, font_size=font_size, font_color='k', rotate=False)

    if showfaultlabels:
        faultlabels_form = {node:'\n\n '+str(fault) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
    plt.axis('off')
    return plt.gcf(), plt.gca()
def update_bipplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], **kwargs):
    """Updates a bipartite graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors, functions = reshist['functions'].keys(), flows=reshist['flows'].keys(), **kwargs)
def update_graphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], **kwargs):
    """Updates a normal graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, pos, show, colors=colors, **kwargs)
def update_typegraphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], **kwargs):
    """Updates a typegraph-stype plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors, **kwargs)
def update_actplot(t_ind,fxn, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], arrows=True, **kwargs):
    """Updates an action graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_asg_plotlabels(g, fxn, reshist, t_ind)
    plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, pos, show, colors=colors, arrows=True, **kwargs)
def update_flowgraphplot(t_ind,fxn, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], seqgraph={}, seqlabels=True, **kwargs):
    """Updates an ASG plot with flows at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_asg_plotlabels(g, fxn, reshist, t_ind)
    degnodes = degfxns + degflows
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors, functions = fxn.actions, flows = [f for f in {**fxn.flows, **fxn.internal_flows} if f in g], seqgraph=seqgraph, seqlabels=seqlabels, **kwargs)
###GRAPHVIZ HELPER FUNCTIONS
############################
def gv_import_check():
    """Checks if graphviz is installed on the system before plotting."""
    try:
        from graphviz import Digraph, Graph
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)
        raise Exception("GraphViz not installed. Please see:\n https://pypi.org/project/graphviz/ \n https://www.graphviz.org/download/")
    return Digraph, Graph
def plot_gv_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultscen, time, showfaultlabels, colors_dict, dot, **kwargs):
    """ Plots a normal graph representation using the graphviz toolkit. Used in other functions"""
    for node in g.nodes:
        node_label = node
        if node in faultlabels and showfaultlabels == True:
            node_label += " \\n "
            node_label += faultlabels[node]
        dot.node(node,label=node_label, style="filled", fillcolor=colors_dict[node], shape='box')
    for edge in edgeflows:
        edge_label = edgeflows[edge]
        if edge in faultflows:
            if (faultflows[edge].strip(" ")) != edgeflows[edge]:
                edge_label  += " \\n "
                edge_label += faultflows[edge]
        dot.edge(edge[0], edge[1], label=edge_label, color=colors_dict[edge], labelangle="180",arrowhead="normal", arrowsize='2')
    return dot
def plot_gv_bipartite(g, faultnodes, degradednodes, faultlabels, faultscen, time, showfaultlabels, colors_dict, functions, flows, edges, dot, seqgraph={}, seqlabels=False,**kwargs):
    """ Plots a bipartite graph representation using the graphviz toolkit. Used in other functions"""
    shapes = {f:'ellipse' for f in flows}
    shapes.update({ f1:'box' for f1 in functions})

    for node in functions+flows:
        node_label = node
        if node in faultlabels and showfaultlabels == True:
            node_label += " \\n "
            node_label += faultlabels[node]
        dot.node(node,label=node_label, style="filled", fillcolor=colors_dict[node], shape=shapes[node])
    for edge in edges:
        dot.edge(edge[0], edge[1], arrowhead='none')
    if seqgraph:
        for edge in seqgraph.edges():
            if seqlabels:   dot.edge(edge[0], edge[1], label=seqgraph.get_edge_data(edge[0], edge[1])['name'], labelangle="180",arrowhead="normal", arrowsize='2')
            else:           dot.edge(edge[0], edge[1], arrowhead="normal", arrowsize='2')
    return dot

def gv_execute_order_legend(colors):
    """Provides legend for model execution order in the graphviz toolkit"""
    from graphviz import Graph
    from IPython.display import display, SVG
    legend = Graph(name='legend')
    legend.attr(sep="+0")
    legend.node("No Execution", label="No Execution", style="filled", fillcolor=colors[0], shape='box')
    legend.node("Dynamic Step", label="Dynamic Step", style="filled", fillcolor=colors[2], shape='box')
    legend.node("Static Step", label="Static Step", style="filled", fillcolor=colors[1], shape='box')
    legend.attr(rank='source')
    display(SVG(legend._repr_image_svg_xml()))
    return

def update_gv_bipplot(t_ind, reshist, g, faultscen=[], showfaultlabels=True, colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, **kwargs):
    """graphviz helper: updates a bipartite graph plot at a given timestep t_ind given the result history reshist"""
    Digraph, Graph = gv_import_check()
    kwargs["layout"] = kwargs.get("layout", "twopi")
    kwargs["overlap"] = kwargs.get("overlap", "voronoi")
    time = reshist['time'][t_ind]
    functions = list(reshist['functions'].keys()); flows=list(reshist['flows'].keys())
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    colors_dict = gv_colors(g, 'bipartite', colors, heatmap,cmap, faultfxns, degnodes, faultedges=faultedges, edgeflows=edgeflows, functions=functions, flows=flows)
    faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
    dot = Graph(comment="model network", graph_attr=kwargs)
    dot = plot_gv_bipartite(g, faultfxns, degnodes, faultlabels_form, faultscen, time, showfaultlabels, colors_dict, functions, flows, g.edges, dot)
    return dot
def update_gv_graphplot(t_ind, reshist, g, faultscen=[], showfaultlabels=True, colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, arrows=False, **kwargs):
    """graphviz helper: Updates a normal graph plot at a given timestep t_ind given the result history reshist"""
    Digraph, Graph = gv_import_check()
    kwargs["pad"] = kwargs.get("pad", "0.5")
    kwargs["ranksep"] = kwargs.get("ranksep", "2")
    time = reshist['time'][t_ind]
    functions = list(reshist['functions'].keys()); flows=list(reshist['flows'].keys())
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    colors_dict = gv_colors(g, 'normal', colors, heatmap,cmap, faultfxns, degfxns, faultedges=faultedges, edgeflows=edgeflows, functions=functions, flows=flows)
    faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
    dot = Graph(comment="model network", graph_attr=kwargs)
    dot = plot_gv_normgraph(g, edgeflows, faultfxns, degfxns, degflows, faultlabels_form, faultedges, faultscen, time, showfaultlabels, colors_dict, dot)
    return dot

def update_gv_actplot(t_ind,fxn, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, arrows=True, seqgraph={}, **kwargs):
    """Updates an action graph plot at a given timestep t_ind given the result history reshist"""
    Digraph, Graph = gv_import_check()
    kwargs["pad"] = kwargs.get("pad", "0.5")
    kwargs["ranksep"] = kwargs.get("ranksep", "2")
    time = reshist['time'][t_ind]
    functions = [*fxn.actions]; flows = [f for f in {**fxn.flows, **fxn.internal_flows}]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_asg_plotlabels(g, fxn, reshist, t_ind)
    colors_dict = gv_colors(g, 'normal', colors, heatmap,cmap, faultfxns, degfxns, faultedges=faultedges, edgeflows=edgeflows, functions=functions, flows=flows)
    faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
    if arrows:  dot = Digraph(comment="model network", graph_attr=kwargs)
    else:       dot = Graph(comment="model network", graph_attr=kwargs)
    dot = plot_gv_normgraph(g, edgeflows, faultfxns, degfxns, degflows, faultlabels_form, faultedges, faultscen, time, showfaultlabels, colors_dict, dot, **kwargs)
    return dot
def update_gv_flowgraphplot(t_ind,fxn, reshist, g, faultscen=[], showfaultlabels=True, colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, arrows=[],seqgraph={}, seqlabels=False, **kwargs):
    """Updates an ASG plot with flows at a given timestep t_ind given the result history reshist"""
    Digraph, Graph = gv_import_check()
    kwargs["pad"] = kwargs.get("pad", "0.5")
    kwargs["ranksep"] = kwargs.get("ranksep", "2")
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_asg_plotlabels(g, fxn, reshist, t_ind)
    functions =  [*fxn.actions]; flows = [f for f in {**fxn.flows, **fxn.internal_flows}]
    degnodes = degfxns + degflows
    colors_dict = gv_colors(g, 'bipartite', colors, heatmap,cmap, faultfxns, degnodes, faultedges=faultedges, edgeflows=edgeflows, functions=functions, flows=flows)
    faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
    if seqgraph:    dot = Digraph(comment="model network", graph_attr=kwargs)
    else:           dot = Graph(comment="model network", graph_attr=kwargs)
    dot = plot_gv_bipartite(g, faultfxns, degnodes, faultlabels_form, faultscen, time, showfaultlabels, colors_dict, functions, flows, g.edges, dot, seqgraph=seqgraph, seqlabels=seqlabels)
    return dot

def gv_colors(g, gtype, colors, heatmap, cmap, faultnodes, degradednodes, faultedges=[], edgeflows={}, functions=[], flows=[], highlight=[]):
    """
    creates dictonary of node/edge colors for a graphviz plot

    Parameters
    ----------
    g : nx graph object or model
        The multigraph to plot
    gtype : string, optional
        Type of graph input to show
        values are 'normal', 'bipartite', or 'typegraph'.
    colors : list, optional
        List of colors to use for nominal, degraded, and faulty functions/flows.
        Default is: ['lightgray','orange', 'red']
    heatmap : dict, optional
        A heatmap dictionary to overlay on the plot. The default is {}.
    cmap : mpl colormap
        Colormap to use for heatmap visualization
    faultnodes : list
        list of the nodes with faults
    degradednodes : list
        list of the nodes with degraded functionality
    faultedges : list
        list of edges(flows) that have faults. Only used for 'normal' graph. The default is [].
    edgeflows : dictionary
        dictionary of edges (n1,n2) and edge/flow names. The default is {}.
    functions : list, optional
        list of function nodes. Only used for 'bipartite' graph. The default is [].
    flows : list, optional
        list of flow nodes. Only used for 'bipartite' graph. The default is [].

    Returns
    -------
    colors_dict : dictionary
        dictionary withe keys as nodes/edges and values colors.

    """
    if gtype == 'normal':
        if heatmap == {}: #or highlight != []:
                colors_dict = {fn: colors[2] for fn in faultnodes}
                colors_dict.update({dn: colors[1] for dn in degradednodes})
                colors_dict.update({f: colors[0] for f in g.nodes if f not in degradednodes and f not in faultnodes})
                colors_dict.update({fe: colors[1] for fe in faultedges})
                colors_dict.update({ne: colors[0] for ne in edgeflows if ne not in faultedges})
        elif heatmap != {}:
            colors_dict = {}
            colors_val_dict = {}
            for node in g.nodes:
                colors_val_dict[node] = heatmap.get(node, 0.0)
            for node in edgeflows:
                 colors_dict[node] = "black"
            Arange = [colors_val_dict[node] for node in colors_val_dict if node not in edgeflows]
            node_labels = [node for node in colors_val_dict if node not in edgeflows]
            norm = matplotlib.colors.Normalize(vmin = min(Arange), vmax = max(Arange))
            m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            mm = m.to_rgba(Arange)
            mm = [matplotlib.colors.to_hex(mm_i) for mm_i in mm]
            for i in range(len(mm)):
                colors_dict[node_labels[i]] = mm[i]
    elif gtype in ['bipartite', 'component']:
        if heatmap == {}:
            colors_dict = {fn: colors[2] for fn in faultnodes}
            colors_dict.update({dn: colors[1] for dn in degradednodes})
            colors_dict.update({f: colors[0] for f in functions+flows if f not in degradednodes and f not in faultnodes})
        else:
            colors_dict = {}
            colors_val_dict = {}
            for node in functions+flows:
                colors_val_dict[node] = heatmap.get(node, 0.0)
            Arange = [colors_val_dict[node] for node in colors_val_dict]
            node_labels = [node for node in colors_val_dict]
            norm = matplotlib.colors.Normalize(vmin = min(Arange), vmax = max(Arange))
            m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            mm = m.to_rgba(Arange)
            mm = [matplotlib.colors.to_hex(mm_i) for mm_i in mm]
            for i in range(len(mm)):
                colors_dict[node_labels[i]] = mm[i]
    return colors_dict
