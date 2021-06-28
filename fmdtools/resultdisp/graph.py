"""
File Name: resultdisp/graph.py
Author: Daniel Hulse
Created: November 2019 (Refactored April 2020)

Description: Gives graph-level visualizations of the model.

Public user-facing methods:
    - set_pos:                      Set graph node positions manually
    - show:                         Plots a single graph object g.
    - history:                      Displays plots of the graph over time given a dict history of graph objects
    - result_from:                  Plots a representation of the model graph at a specific time in the results history.
    - results_from:                 Plots a set of representations of the model graph at given times in the results history.
    - animation_from:               Creates an animation of the model graph using results at given times in the results history.
Private/helper methods: 
    - update_bipplot:               updates a bipartite graph plot at a given timestep t_ind given the result history reshist
    - update_graphplot:             updates a graph plot at a given timestep t_ind given the result history reshist
    - plot_normgraph:               Plots a standard graph. 
    - plot_bipgraph:                Plots a bipartite graph. 
    - get_plotlabels:               Assigns labels to a graph g from reshist at time t so that it can be plotted
"""


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patches import Patch
import netgraph

def set_pos(g, gtype='normal',scale=1,node_color='lightgray', label_size=7, initpos={}):
    """
    Provides graphical interface to set graph node positions. If model is provided, it will also set the positions in the model object. 
    
    To work, this method must be opened in an external window, so change the IPython before use usings %matplotlib qt' (or '%matplotlib osx')

    Parameters
    ----------
    g : networkx graph or model
        normal or bipartite graph of the model of interest
    gtype : 'normal' or 'bipartite', optional
        Type of graph to plot. The default is 'normal'.
    scale : float, optional
        scale for the node sizes. The default is 1.
    node_color : str, optional
        color to use for the nodes. The default is 'lightgray'.
    label_size : float, optional
        size to use for the labels. The default is 8.
    initpos : dict, optional
        dict of initial positions for the labels (e.g. from nx.spring_layout). The default is {}.

    Returns
    -------
    pos: dict
        dict of node positions for use in graph plotting functions
    """
    set_mdl=False
    if type(g) not in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        mdl=g
        set_mdl=True
        if gtype=='normal':         g=mdl.graph
        elif gtype=='bipartite':    g=mdl.bipartite
        elif gtype=='typegraph':    mdl.return_typegraph()
    plt.ion()
    fig = plt.figure()
    if gtype=='normal':
        if not initpos: initpos = nx.shell_layout(g)
        edgeflows={}
        for edge in g.edges:
            flows=list(g.get_edge_data(edge[0],edge[1]).keys())
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
        plot_instance = netgraph.InteractiveGraph(g,node_size=20*scale,node_shape='s',node_color=node_color, node_edge_width=0, node_positions=initpos, edge_labels=edgeflows, edge_label_font_size=label_size, node_labels={n:n for n in g.nodes},node_label_fontdict={'size':label_size, 'fontweight':'bold'})
    elif gtype=='bipartite':
        if not initpos: initpos = nx.spring_layout(g)
        plot_instance = netgraph.InteractiveGraph(g,node_size=7*scale,node_color=node_color, node_edge_width=0, node_positions=initpos, node_labels={n:n for n in g.nodes},node_label_fontdict={'size':label_size, 'fontweight':'bold'})
    elif gtype=='typegraph':
        plot_instance = netgraph.InteractiveGraph(g,node_size=7*scale,node_color=node_color, node_edge_width=0, node_layout='dot', node_labels={n:n for n in g.nodes},node_label_fontdict={'size':label_size, 'fontweight':'bold'})
    plt.title("Click and drag to place nodes.")
    plt.xlabel("Close window to continue...")
    plt.show(block=False)
    t=0
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
        t+=0.1
    if t< 0.2:
        print("Cannot place nodes in inline version of plot. Use '%matplotlib qt' (or '%matplotlib osx') to open in external window")
    pos = {node:list(loc) for node,loc in plot_instance.node_positions.items()}
    if set_mdl:
        if gtype=='normal':         mdl.graph_pos = pos
        elif gtype=='bipartite':    mdl.bipartite_pos = pos  
    return pos

def show_pyvis(g, gtype='hierarchical', filename="typegraph.html", width=1000, filt=True, physics=False):
    """
    Method for plotting graphs with pyvis. Produces interactive HTML!

    Parameters
    ----------
    g : networkx graph
        Graph to plot.
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
    """
    from pyvis.network import Network
    width = str(width)+"px"
    
    if gtype=='hierarchical':   n = Network(directed=True, layout='hierarchical', width=width)
    elif gtype in ["component", "bipartite"]: n = Network(width=width)
    else:   raise Exception("Not a valid graph type")     
    n.from_nx(g)
    n.toggle_physics(physics)
    if filt: n.show_buttons(filter_=filt)
    n.show(filename)

def show(g, gtype='normal', pos=[], scale=1, faultscen=[], time=[], showfaultlabels=True, retfig=False, highlight=[], colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm):
    """
    Plots a single graph object g.

    Parameters
    ----------
    g : networkx graph or model
        The multigraph to plot
    gtype : 'normal' or 'bipartite'
        Type of graph input to show--normal (multgraph) or bipartite
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
    retfig : bool, optional
        Whether to return the figure. The default is False.
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
    """
    if type(g) not in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        mdl=g
        if gtype=='normal':         
            g=mdl.graph
            if not pos: pos=mdl.graph_pos
        elif gtype=='bipartite':    
            g=mdl.bipartite 
            if not pos: pos=mdl.bipartite_pos
        elif gtype=='component':
            g = mdl.return_stategraph('component')
            if not pos: pos=nx.spring_layout(g)
    plt.figure()
    fig_axis = 0
    if gtype=='normal':
        edgeflows=dict()
        if not pos: pos=nx.shell_layout(g)
        nodesize=scale*2000
        font_size=scale*12
        for edge in g.edges:
            flows=list(g.get_edge_data(edge[0],edge[1]).keys())
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
        if heatmap:
            colors=[]
            for node in g.nodes():
                colors = colors +[heatmap.get(node,0.0)]
                nx.draw_networkx_edges(g,pos, width=2)
            nx.draw_networkx_nodes(g,pos,node_size=nodesize, node_shape='s', node_color=colors, cmap=cmap, alpha=0.7)
            nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeflows, font_size=font_size, font_weight='bold')
            labels={node:node for node in g.nodes} 
            nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, font_weight='bold')
        elif highlight:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
            faultedges = highlight[2]
            if showfaultlabels: faultlabels = {f:[str(i)] for i,f in enumerate(faultnodes)}
            else:               faultlabels = {}
            faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge])]) for edge in faultedges}
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos, retfig=retfig,colors=colors)
        else:
            statuses=dict(g.nodes(data='status', default='Nominal'))
            faultnodes=[node for node,status in statuses.items() if status=='Faulty']
            degradednodes=[node for node,status in statuses.items() if status=='Degraded']
            faults=dict(g.nodes(data='modes', default={'nom'}))
            faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
            if not list(g.nodes(data='status'))[0][1]: faultedges = {}; faultflows = {}
            else:
                faultedges = [edge for edge in g.edges if any([g.edges[edge][flow].get('status','nom')=='Degraded' for flow in g.edges[edge]])]
                faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=pos, retfig=retfig,colors=colors)
    elif gtype in ['bipartite', 'component']:
        labels={node:node for node in g.nodes}
        functions = [f for f, val in g.nodes.items() if val['bipartite']==0]
        flows = [f for f, val in g.nodes.items() if val['bipartite']==1]
        if not pos: pos=nx.spring_layout(g)
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
            nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, node_size=nodesize, font_weight='bold')
            if faultscen:
                plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
            plt.show()
        elif highlight:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
            if showfaultlabels: 
                faultlabels = {f:[str(i)] for i,f in enumerate(faultnodes)}
            else:               faultlabels={}
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=True, scale=scale, pos=pos, retfig=retfig,colors=colors, functions = functions, flows=flows)
        else:                                      #plots graph with status information 
            statuses=dict(g.nodes(data='status', default='Nominal'))
            faultnodes=[node for node,status in statuses.items() if status=='Faulty']
            degradednodes=[node for node,status in statuses.items() if status=='Degraded']
            faults=dict(g.nodes(data='modes', default={'nom'}))
            faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=True, scale=scale, pos=pos, retfig=retfig,colors=colors, functions = functions, flows=flows)
    elif gtype == 'typegraph':
        if not pos: pos = netgraph.get_sugiyama_layout(list(classgraph.edges), nodes=classgraph.nodes)
        nx.draw(g, pos=pos, with_labels=True, node_size=scale*700, font_size=scale*8, font_weight='bold', node_color=colors[0])
    if retfig: 
        if fig_axis: return fig_axis
        else: return plt.gcf(), plt.gca()
        
def exec_order(mdl, gtype='bipartite', pos=[], scale=1, colors=['lightgray', 'cyan','teal'], show_dyn_order=True, retfig=True, title="Function Execution Order", legend=True):
    """
    Displays the execution order/types of the model, where the functions and flows in the
    static step are highlighted and the functions in the dynamic step are listed (with corresponding order)

    Parameters
    ----------
    mdl : fmdtools Model
        Model of the system to visualize.
    gtype : 'normal'/'bipartite', optional
        Representation of the model to use. The default is 'bipartite'.
    pos : dict optional
        Dictionary of positions for the model. The default is [].
    scale : float, optional
        Scale factor for the node sizes. The default is 1.
    colors : list, optional
        Colors to use for unexecuted functions, static propagation steps, and dynamic functions. 
        The default is ['lightgray', 'cyan','teal'].
    show_dyn_order : bool, optional
        Whether to label the execution order for dynamic functions. The default is True.
    retfig : bool, optional
        Whether to retun the figure and axis objects. The default is True.
    title : str, optional
        Title for the plot. The default is "Function Execution Order".
    legend : bool, optional
        Whether to show a legend. The default is True.

    Returns
    -------
    tuple of form (figure, axis) (if retfig is true)

    """
    if gtype =='normal': fig_axis = show(mdl, gtype=gtype, pos=pos, highlight=[mdl.dynamicfxns, mdl.staticfxns,  mdl.graph.edges(mdl.staticfxns)], scale=scale, colors=colors, retfig=True, showfaultlabels= show_dyn_order)
    else:
        staticnodes = list(mdl.staticfxns) + list(set([n for node in mdl.staticfxns for n in mdl.bipartite.neighbors(node)]))
        dynamicnodes = list(mdl.dynamicfxns) #+ list(set().union(*[nx.node_connected_component(mdl.bipartite, node) for node in mdl.dynamicfxns]))
        fig_axis = show(mdl, gtype=gtype, pos=pos, highlight=[dynamicnodes, staticnodes], scale=scale, colors=colors, retfig=True, showfaultlabels= show_dyn_order)
    
    if legend:
        legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0], label='No Execution'),
                           Patch(facecolor=colors[2], edgecolor=colors[2], label='Dynamic Step'),
                           Patch(facecolor=colors[1], edgecolor=colors[1], label='Static Step')]
        
        fig_axis[1].legend(handles = legend_elements, ncol=3, bbox_to_anchor = (1.0,-0.05))
    if title: fig_axis[1].set_title(title)
    if retfig: return fig_axis
    
def history(ghist, gtype='normal', pos=[], scale=1, faultscen=[],showfaultlabels=True, colors=['lightgray','orange', 'red']):
    """
    Displays plots of the graph over time given a dict history of graph objects

    Parameters
    ----------
    ghist : dict
        A dictionary of the history of the graph over time with structure:
       {time: graphobject}, where
           - time is the time where the snapshot of the graph was recorded
           - graphobject is the snapshot of the graph at that time
    gtype : 'normal' or 'bipartite'
        Type of graph input to show--normal (multgraph) or bipartite
    pos : dict
        Positions for nodes
    scale: float
        Changes sizes of nodes in bipartite graph
    faultscen : str, optional
        Name of the fault scenario (for the title). The default is [].
    showfaultlabels : bool, optional
        Whether or not to label the faults on the functions. The default is True.
    """
    for time, graph in ghist.items():
        show(graph,gtype=gtype,pos=pos, scale=scale, faultscen=faultscen, time=time, showfaultlabels=showfaultlabels, colors=colors)

def result_from(mdl, reshist, time, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, pos=[], retfig=False, colors=['lightgray','orange', 'red']):
    """
    Plots a representation of the model graph at a specific time in the results history.

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from compare_hists())
    time : float
        The time in the history to plot the graph at.
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    retfig:, bool, optional
        whether to return the figure and axis objects of the plot. The default is False.
    """
    if gtype=='normal' and not pos:         pos=mdl.graph_pos
    elif gtype=='bipartite' and not pos:    pos=mdl.bipartite_pos
    [[t_ind,],] = np.where(reshist['time']==time)
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
        degnodes = degfxns + degflows
        fig_axis = plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos=pos, retfig=retfig, colors=colors, functions = mdl.fxns.keys(), flows=mdl.flows.keys())
    elif gtype=='normal':
        g = mdl.graph.copy()
        labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
        fig_axis= plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, retfig=retfig, colors=colors)
    if retfig: return fig_axis

def results_from(mdl, reshist, times, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, pos=[],colors=['lightgray','orange', 'red']):
    """
    Plots a set of representations of the model graph at given times in the results history.

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from compare_hists())
    times : list or 'all'
        The times in the history to plot the graph at. If 'all', plots them all
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    """
    if gtype=='normal' and not pos:         pos=mdl.graph_pos
    elif gtype=='bipartite' and not pos:    pos=mdl.bipartite_pos
    if times=='all':
        t_inds= [i for i in range(0,len(reshist['time']))]
    else:
        t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        pos=nx.spring_layout(g)
        for t_ind in t_inds:
            update_bipplot(t_ind, reshist, g, pos, faultscen=faultscen, showfaultlabels=showfaultlabels, scale=scale, colors=colors)
    elif gtype=='normal':
        g = mdl.graph.copy()
        pos=nx.shell_layout(g)
        for t_ind in t_inds:
            update_graphplot(t_ind, reshist, g, pos, faultscen=faultscen, showfaultlabels=showfaultlabels, scale=scale, colors=colors)

def animation_from(mdl, reshist, times, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, show=False, pos=[], colors=['lightgray','orange', 'red']):
    """
    Creates an animation of the model graph using results at given times in the results history.
    To view, use %matplotlib qt from spyder or %matplotlib notebook from jupyter
    To save (or do anything useful)h, make sure ffmpeg is installed  https://www.wikihow.com/Install-FFmpeg-on-Windows

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from compare_hists())
    times : list or 'all'
        The times in the history to plot the graph at. If 'all', plots them all
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    show : bool, optional
        Whether to show the plot at the end (may be redundant). The default is True.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    """
    if gtype=='normal' and not pos:         pos=mdl.graph_pos
    elif gtype=='bipartite' and not pos:    pos=mdl.bipartite_pos
    if times=='all':
        t_inds= [i for i in range(0,len(reshist['time']))]
    else:
        t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        if not pos: pos=nx.spring_layout(g)
        fig, ax = plt.subplots(figsize=(6,4))
        ani = matplotlib.animation.FuncAnimation(fig, update_bipplot, frames=t_inds, fargs=(reshist, g, pos, faultscen, showfaultlabels, scale, False, colors))
        if show: plt.show()
    elif gtype=='normal':
        g = mdl.graph.copy()
        if not pos: pos=nx.shell_layout(g)
        fig, ax = plt.subplots(figsize=(6,4))
        ani = matplotlib.animation.FuncAnimation(fig, update_graphplot, frames=t_inds, fargs=(reshist, g, pos, faultscen, showfaultlabels, scale, False, colors))
        if show: plt.show()
    return ani
def update_bipplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red']):
    """Updates a bipartite graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors, functions = reshist['functions'].keys(), flows=reshist['flows'].keys())
def update_graphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red']):
    """Updates a normal graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, pos, show, colors=colors)

def plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True, retfig=False, colors=['lightgray','orange', 'red'], title=[]):
    """ Plots a standard graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*2000
    font_size=scale*12
    if not pos: pos=nx.shell_layout(g)
    nx.draw_networkx(g,pos,node_size=nodesize,font_size=font_size, node_shape='s',edge_color='gray', node_color=colors[0], width=3, font_weight='bold')
    nx.draw_networkx_edge_labels(g,pos,font_size=font_size, edge_labels=edgeflows)
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_shape='s',node_color = colors[2],width=3,font_size=font_size, font_weight='bold', node_size = nodesize*1.2)
    nx.draw_networkx_nodes(g, pos, nodelist=degfxns,node_shape='s', node_color = colors[1],width=3,font_size=font_size, font_weight='bold', node_size = nodesize)
    nx.draw_networkx_edges(g,pos,edgelist=faultedges, edge_color=colors[1],font_size=font_size, width=2)
        
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=faultedgeflows,font_size=font_size, font_color=colors[1])
    plt.axis('off')
    if retfig:
        return plt.gcf(), plt.gca()
    elif show: plt.show()

def plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True, retfig=False, colors=['lightgray','orange', 'red'], title=[],functions=[], flows=[]):
    """ Plots a bipartite graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*700
    font_size=scale*8
    if not pos: pos=nx.spring_layout(g)
    if functions and flows:
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_nodes(g, pos, nodelist = functions, node_shape='s', labels=labels,font_size=font_size, node_size=nodesize, node_color = colors[0], font_weight='bold')
        nx.draw_networkx_nodes(g, pos, nodelist = flows, labels=labels,font_size=font_size, node_size=nodesize, node_color = colors[0], font_weight='bold')
        degfxns = [node for node in degnodes if node in functions]
        degflows = [node for node in degnodes if node in flows]
        nx.draw_networkx_nodes(g, pos, nodelist=faultfxns, node_shape='s', node_color = colors[2],labels=labels, node_size=nodesize*1.2, font_weight='bold')
        nx.draw_networkx_nodes(g, pos, nodelist=degfxns, node_shape='s', node_color = colors[1],labels=labels, node_size=nodesize, font_weight='bold')
        nx.draw_networkx_nodes(g, pos, nodelist=degflows,node_color = colors[1],labels=labels, node_size=nodesize, font_weight='bold')
        nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, node_size=nodesize, font_weight='bold')

    elif functions or flows:
        raise Exception("Invalid option--either provide list of functions and flows, or neither")
    else:
        nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_color = colors[2], node_size=nodesize*1.2, font_weight='bold')
        nx.draw(g, pos, labels=labels,font_size=font_size, node_size=nodesize, node_color = colors[0], font_weight='bold')
        nx.draw_networkx_nodes(g, pos, nodelist=degnodes,node_color = colors[1], node_size=nodesize, font_weight='bold')
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
    plt.axis('off')
    if retfig:
        return plt.gcf(), plt.gca()
    elif show: plt.show()
def get_plotlabels(g, reshist, t_ind):
    """
    Assigns labels to a graph g from reshist at time t so that it can be plotted

    Parameters
    ----------
    g : networkx graph
        The graph to get labels for
    reshist : dict
        The dict of results history over time (e.g. from compare_mdlhist)
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

def plot_norm_netgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True, retfig=False, colors=['lightgray','orange', 'red']):
    """ Experimental method for plotting with netgraph instead of networkx"""
    nodesize=scale*20
    font_size=scale*12
    if not pos: pos=nx.shell_layout(g)
    netgraph.draw(g,pos,node_size=nodesize,font_size=font_size, node_shape='s', node_color=colors[0], width=3, font_weight='bold')
    netgraph.draw_edge_labels(list(edgeflows.keys()), edgeflows, pos,edge_label_font_size=font_size)
    netgraph.draw_nodes({n:pos[n] for n in degfxns}, node_labels=degfxns, node_shape='s', node_color = colors[1],width=3,font_size=font_size, font_weight='bold', node_size = nodesize)
    netgraph.draw_nodes({n:pos[n] for n in faultfxns}, node_labels=faultfxns, node_shape='s', node_color = colors[2],width=3,font_size=font_size, font_weight='bold', node_size = nodesize)
    netgraph.draw_edges(faultedges,pos, edge_color=colors[1],font_size=font_size, width=2)
    netgraph.draw_node_labels({p:p for p in pos}, pos)
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        netgraph.draw_node_labels(faultlabels_form, pos, font_size=font_size, font_color='k')
        netgraph.draw_edge_labels(list(faultedgeflows.keys()), faultedgeflows, pos, font_size=font_size, font_color=colors[1])
    if faultscen:
        plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    if retfig:
        return plt.gcf(), plt.gca()
    elif show: plt.show()






