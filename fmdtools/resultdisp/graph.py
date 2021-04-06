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
import netgraph

def set_pos(g, gtype='normal',scale=1,node_color='lightgray', label_size=8, initpos={}):
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
    if not type(g)==nx.classes.graph.Graph:
        mdl=g
        set_mdl=True
        if gtype=='normal':         g=mdl.graph
        elif gtype=='bipartite':    g=mdl.bipartite    
    plt.ion()
    fig = plt.figure()
    if gtype=='normal':
        if not initpos: initpos = nx.shell_layout(g)
        edgeflows={}
        for edge in g.edges:
            flows=list(g.get_edge_data(edge[0],edge[1]).keys())
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
        plot_instance = netgraph.InteractiveGraph(g,node_size=20*scale,node_shape='s',node_color=node_color, node_edge_width=0, node_positions=initpos, edge_labels=edgeflows, edge_label_font_size=label_size, node_labels={n:n for n in g.nodes},node_label_font_size=label_size)
    elif gtype=='bipartite':
        if not initpos: initpos = nx.spring_layout(g)
        plot_instance = netgraph.InteractiveGraph(g,node_size=7*scale,node_color=node_color, node_edge_width=0, node_positions=initpos, node_labels={n:n for n in g.nodes},node_label_font_size=label_size)
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

def show(g, gtype='normal', pos=[], scale=1, faultscen=[], time=[], showfaultlabels=True, heatmap={}, retfig=False, colors=['lightgray','orange', 'red'],cmap=plt.cm.coolwarm):
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
    heatmap : dict, optional
        A heatmap dictionary to overlay on the plot. The default is {}.
    colors : list, optional
        List of colors to use for nominal, degraded, and faulty functions/flows
    cmap : mpl colormap
        Colormap to use for heatmap visualizations
    """
    if not type(g)==nx.classes.graph.Graph:
        mdl=g
        if gtype=='normal':         
            g=mdl.graph
            if not pos: pos=mdl.graph_pos
        elif gtype=='bipartite':    
            g=mdl.bipartite 
            if not pos: pos=mdl.bipartite_pos
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
        elif not list(g.nodes(data='status'))[0][1]:    
            nx.draw_networkx(g,pos,node_size=nodesize,node_shape='s', node_color=colors[0], width=3,font_size=font_size, font_weight='bold')
            nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeflows,font_size=font_size)
        else:
            statuses=dict(g.nodes(data='status', default='Nominal'))
            faultnodes=[node for node,status in statuses.items() if status=='Faulty']
            degradednodes=[node for node,status in statuses.items() if status=='Degraded']
            faultedges = [edge for edge in g.edges if any([g.edges[edge][flow]['status']=='Degraded' for flow in g.edges[edge]])]
            faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
            faults=dict(g.nodes(data='modes', default={'nom'}))
            faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=pos, retfig=retfig,colors=colors)
    elif gtype == 'bipartite' or 'component':
        labels={node:node for node in g.nodes}
        if not pos: pos=nx.spring_layout(g)
        nodesize=scale*700
        font_size=scale*6
        if heatmap:
            #nx.draw(g, pos, node_size=nodesize,node_color = 'k', alpha=0.3)
            colors = []
            for node in labels.keys():
                colors = colors + [heatmap.get(node, 0.0)]
            nx.draw(g, pos, node_color=colors, cmap=cmap, alpha=0.6, node_size=nodesize)
            nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, node_size=nodesize, font_weight='bold')
            if faultscen:
                plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
            plt.show()
        elif not list(g.nodes(data='status'))[0][1]: #just plots graph if no status information 
            nx.draw(g, pos, labels=labels,font_size=font_size, node_size=nodesize,node_color = colors[0], font_weight='bold')
        else:                                      #plots graph with status information 
            statuses=dict(g.nodes(data='status', default='Nominal'))
            faultnodes=[node for node,status in statuses.items() if status=='Faulty']
            degradednodes=[node for node,status in statuses.items() if status=='Degraded']
            faults=dict(g.nodes(data='modes', default={'nom'}))
            faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=True, scale=scale, pos=pos, retfig=retfig,colors=colors)
    if retfig: 
        if fig_axis: return fig_axis
        else: return plt.gcf(), plt.gca()
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
        
        fig_axis = plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos=pos, retfig=retfig, colors=colors)
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
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors)
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
    nx.draw_networkx_nodes(g, pos, nodelist=degfxns,node_shape='s', node_color = colors[1],width=3,font_size=font_size, font_weight='bold', node_size = nodesize)
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_shape='s',node_color = colors[2],width=3,font_size=font_size, font_weight='bold', node_size = nodesize)
    nx.draw_networkx_edges(g,pos,edgelist=faultedges, edge_color=colors[1],font_size=font_size, width=2)
        
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=faultedgeflows,font_size=font_size, font_color=colors[1])
    if retfig:
        return plt.gcf(), plt.gca()
    elif show: plt.show()

def plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True, retfig=False, colors=['lightgray','orange', 'red'], title=[]):
    """ Plots a bipartite graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*700
    font_size=scale*6
    if not pos: pos=nx.spring_layout(g)
    
    nx.draw(g, pos, labels=labels,font_size=font_size, node_size=nodesize, node_color = colors[0], font_weight='bold')
    nx.draw_networkx_nodes(g, pos, nodelist=degnodes,node_color = colors[1], node_size=nodesize, font_weight='bold')
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_color = colors[2], node_size=nodesize, font_weight='bold')
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
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







