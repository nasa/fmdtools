"""
Description: Gives graph-level visualizations of the model using installed renderers.

Public user-facing methods:
    - :func:`set_pos`:              Set graph node positions manually (uses netgraph)
    - :func:`show`:                         Plots a single graph object g. Has options for heatmaps/overlays and matplotlib/graphviz/netgraph/pyvis renderers.
    - :func:`exec_order`:                   Displays the propagation order and type (dynamic/static) in the model. Works with matplotlib/graphviz/netgraph renderers.
    - :func:`history`:                      Displays plots of the graph over time given a dict history of graph objects.  Works with matplotlib/graphviz/netgraph renderers.
    - :func:`result_from`:                  Plots a representation of the model graph at a specific time in the results history. Works with matplotlib/graphviz/netgraph renderers.
    - :func:`results_from`:                 Plots a set of representations of the model graph at given times in the results history. Works with matplotlib/graphviz/netgraph renderers.
    - :func:`animation_from`:               Creates an animation of the model graph using results at given times in the results history.  Works with matplotlib/netgraph renderers.
"""
#File Name: resultdisp/graph.py
#Contributors: Daniel Hulse and Sequoia Andrade
#Created: November 2019 
#Refactored: April 2020
#Added major interfaces: July 2021


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patches import Patch
import netgraph

def set_pos(g, gtype='bipartite',scale=1,node_color='lightgray', label_size=7, initpos={}, figsize=(6,4)):
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
    figsize : tuple, optional
        size of matplotlib frame. Default is (6,4)

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
        elif gtype=='typegraph':    g=mdl.return_typegraph()
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

def show(g, gtype='bipartite', renderer = 'matplotlib', filename="", **kwargs):
    """
    Plots a single graph object g.

    Parameters
    ----------
    g : networkx graph or model
        The multigraph to plot
    gtype : 'normal' or 'bipartite'
        Type of graph input to show--normal (multgraph) or bipartite
    renderer : 'matplotlib' or 'graphviz' or 'pyvis' or 'netgraph'
        Renderer to use with the drawing. Renderer must be installed. Default is 'matplotlib'
    filename : string, optional
        the filename for the output. The default is '' (in which a file is not saved except in pyvis).
    **kwargs : dictionary
        keyword arguments for the individual methods. See the documentation for 
            graph.show_graphviz
            graph.show_maplotlib
            graph.show_pyvis
            graph.show_netgraph
        for more information on these arguments
    """
    if renderer=='graphviz':
        dot = show_graphviz(g, gtype, filename=filename,  **kwargs)
        return dot
    elif renderer == 'matplotlib':
        fig, ax = show_matplotlib(g, gtype=gtype, filename=filename, **kwargs)
        return fig, ax
    elif renderer =='netgraph':
        fig, ax, gra = show_netgraph(g, gtype=gtype, filename=filename, **kwargs)
        return fig, ax, gra
    elif renderer == 'pyvis':
        n = show_pyvis(g, gtype=gtype, filename=filename, **kwargs)
        return n
    else: raise Exception("Invalid renderer: "+renderer)

def show_matplotlib(g, gtype='bipartite', filename='', filetype='png', pos=[], scale=1, faultscen=[], time=[], figsize=(6,4), showfaultlabels=True, highlight=[], colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm):
    """
    Plots a single graph object g using matplotlib

    Parameters
    ----------
    g : networkx graph or model
        The multigraph to plot
    gtype : 'normal' or 'bipartite'
        Type of graph input to show--normal (multgraph) or bipartite
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

    Returns
    -------
    fig, ax : matplotlib figure/axis
        Matplotlib figure object of the drawn graph
    """
    if type(g) not in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        mdl=g
        g, pos = get_graph_pos(mdl,pos, gtype)
    fig = plt.figure(figsize=figsize)
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
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos,colors=colors, show=False)
        else:
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            if not list(g.nodes(data='status'))[0][1]: faultedges = {}; faultflows = {}
            else:
                faultedges = [edge for edge in g.edges if any([g.edges[edge][flow].get('status','nom')=='Degraded' for flow in g.edges[edge]])]
                faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
            fig_axis = plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos,colors=colors, show=False)
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
            nx.draw_networkx_labels(g, pos, labels=labels,font_size=font_size, font_weight='bold')
            if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
        elif highlight:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
            if showfaultlabels: faultlabels = {f:[str(i)] for i,f in enumerate(faultnodes)}
            else:               faultlabels={}
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, functions = functions, flows=flows, show=False)
        else:                                      #plots graph with status information 
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            fig_axis = plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, functions = functions, flows=flows, show=False)
    elif gtype == 'typegraph':
        if not pos: pos = netgraph.get_sugiyama_layout(list(g.edges), nodes=g.nodes)
        if heatmap or highlight: raise Exception("Invalid option for typegraph--not implemented")
        if "mdl" in locals():
            nx.draw(g, pos=pos, with_labels=True, node_size=scale*700, font_size=scale*8, font_weight='bold', node_color=colors[0])
        else:
            #faultnodes = list({o.__class__.__name__ for f,o in mdl.fxns.items() if o.any_faults()})
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            fig_axis =plot_bipgraph(g,labels, faultnodes, degradednodes, faultlabels, faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, show=False)
    if filename:fig.savefig(filename=filename, format=filetype, bbox_inches = 'tight', pad_inches = 0)
    return fig, fig.axes[0]
def show_graphviz(g, gtype='bipartite', faultscen=[], time=[],filename='',filetype='png', showfaultlabels=True, highlight=[], colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, **kwargs):
    """
    Translates an existing nx graph to a graphviz graph. Saves the graph output and dot file.
    Called from show() by passing in graphviz=True and filename
    
    Parameters
    ----------
    g : nx graph object or model
        The multigraph to plot
    gtype : string, optional
        Type of graph input to show
        values are 'normal', 'bipartite', or 'typegraph'. The default is 'normal'.
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
        
    if type(g) not in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        mdl=g
        g, pos = get_graph_pos(mdl, kwargs.get('pos', []), gtype)
    #bipartite
    if gtype in ['bipartite', 'component']:
        functions = [f for f, val in g.nodes.items() if val['bipartite']==0]
        flows = [f for f, val in g.nodes.items() if val['bipartite']==1]
        edges = g.edges
        #handles faults
        labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
        faultlabels_form = {node:'\n\n '+str(fault) for node,fault in faultlabels.items() if fault!={'nom'}}
        #handles heatmap and highlight
        if highlight != []:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
        colors_dict = gv_colors(g, gtype, colors, heatmap, cmap, faultnodes, degradednodes, functions=functions, flows=flows)
        dot = Graph(comment="model network", graph_attr=kwargs)
        dot = plot_gv_bipartite(g, faultnodes, degradednodes, faultlabels_form, faultscen, time, showfaultlabels, colors_dict, functions, flows, edges, dot)
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
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
        if highlight != []:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
            faultedges = highlight[2]
            if showfaultlabels: faultlabels = {f:[str(i)] for i,f in enumerate(faultnodes)}
            else:               faultlabels = {}
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
        dot = Graph(comment="model network", graph_attr=kwargs)
        dot = plot_gv_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels_form, faultedges, faultscen, time, showfaultlabels, colors_dict, dot)
            
    #rendering
    dot.attr(outputorder = "edgesfirst")
    if filename:    dot.render(filename = filename+gtype, format = filetype)
    else:           display(SVG(dot._repr_image_svg_xml()))
    return dot
def show_netgraph(g, gtype='bipartite', filename='', filetype='png', pos=[], scale=1, faultscen=[], time=[], figsize=(6,4), showfaultlabels=True, highlight=[], colors=['lightgray','orange', 'red'], **kwargs):
    """
    Plots a single graph object g using netgraph

    Parameters
    ----------
    g : networkx graph or model
        The multigraph to plot
    gtype : 'normal' or 'bipartite'
        Type of graph input to show--normal (multgraph) or bipartite
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
    Returns
    -------
    fig, ax : matplotlib figure/axis
        Matplotlib figure object of the drawn graph
    gra : netgraph Graph
        Netgraph object which can be further manipulated
    """
    from netgraph import Graph
    if kwargs.get('heatmap',False):     raise Exception("Heatmap option not implemented in netgraph renderer")
    if type(g) not in [nx.classes.graph.Graph, nx.classes.digraph.DiGraph]:
        mdl=g
        g, pos = get_graph_pos(mdl,pos, gtype)
    fig = plt.figure(figsize=figsize)
    if gtype=='normal':
        edgeflows=dict()
        if not pos: pos=nx.shell_layout(g)
        for edge in g.edges:
            flows=list(g.get_edge_data(edge[0],edge[1]).keys())
            edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
        if highlight:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
            faultedges = highlight[2]
            if showfaultlabels: faultlabels = {f:[str(i)] for i,f in enumerate(faultnodes)}
            else:               faultlabels = {}
            faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge])]) for edge in faultedges}
            fig_axis = plot_norm_netgraph(g, g.nodes(), faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos, colors=colors, show=False, **kwargs)
        else:
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            if not list(g.nodes(data='status'))[0][1]: faultedges = {}; faultflows = {}
            else:
                faultedges = [edge for edge in g.edges if any([g.edges[edge][flow].get('status','nom')=='Degraded' for flow in g.edges[edge]])]
                faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
            fig_axis = plot_norm_netgraph(g, labels, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=scale, pos=pos, colors=colors, show=False, **kwargs)
    elif gtype in ['bipartite', 'component']:
        labels={node:node for node in g.nodes}
        functions = [f for f, val in g.nodes.items() if val['bipartite']==0]
        flows = [f for f, val in g.nodes.items() if val['bipartite']==1]
        if not pos: pos=nx.spring_layout(g)
        if highlight:
            faultnodes = highlight[0]
            degradednodes = highlight[1]
            if showfaultlabels: 
                faultlabels = {f:[str(i)] for i,f in enumerate(faultnodes)}
            else:               faultlabels={}
            fig_axis = plot_bip_netgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, functions = functions, flows=flows, show=False, **kwargs)
        else:                                      #plots graph with status information 
            labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
            fig_axis = plot_bip_netgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, functions = functions, flows=flows, show=False, **kwargs)
    elif gtype == 'typegraph':
        if not pos: pos = netgraph.get_sugiyama_layout(list(g.edges), nodes=g.nodes)
        if kwargs.get('heatmap', False): raise Exception("Invalid option for typegraph--not implemented")
        labels, faultnodes, degradednodes, faults, faultlabels = get_graph_annotations(g, gtype)
        fig_axis =plot_bip_netgraph(g,labels, faultnodes, degradednodes, faultlabels, faultscen, time, showfaultlabels=showfaultlabels, scale=scale, pos=pos, colors=colors, **kwargs)
    if filename:fig.savefig(filename=filename, format=filetype, bbox_inches = 'tight', pad_inches = 0)
    return fig, fig.axes[0], fig_axis[2]

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
        

def exec_order(mdl, renderer='matplotlib', gtype='bipartite', colors=['lightgray', 'cyan','teal'], show_dyn_order=True, title="Function Execution Order", legend=True, **kwargs):
    """
    Displays the execution order/types of the model, where the functions and flows in the
    static step are highlighted and the functions in the dynamic step are listed (with corresponding order)

    Parameters
    ----------
    mdl : fmdtools Model
        Model of the system to visualize.
    renderer : 'matplotlib' or 'graphviz'
        Renderer to use for the graph
    gtype : 'normal'/'bipartite', optional
        Representation of the model to use. The default is 'bipartite'.
    colors : list, optional
        Colors to use for unexecuted functions, static propagation steps, and dynamic functions. 
        The default is ['lightgray', 'cyan','teal'].
    show_dyn_order : bool, optional
        Whether to label the execution order for dynamic functions. The default is True.
    title : str, optional
        Title for the plot. The default is "Function Execution Order".
    legend : bool, optional
        Whether to show a legend. The default is True.
    **kwargs : see arguments for the respective renderers
    Returns
    -------
    tuple of form (figure, axis) 

    """

    if gtype =='normal': fig_axis = show(mdl, renderer=renderer, gtype=gtype, highlight=[mdl.dynamicfxns, mdl.staticfxns,  mdl.graph.edges(mdl.staticfxns)], colors=colors, showfaultlabels= show_dyn_order, **kwargs)
    else:
        staticnodes = list(mdl.staticfxns) + list(set([n for node in mdl.staticfxns for n in mdl.bipartite.neighbors(node)]))
        dynamicnodes = list(mdl.dynamicfxns) #+ list(set().union(*[nx.node_connected_component(mdl.bipartite, node) for node in mdl.dynamicfxns]))
        fig_axis = show(mdl, renderer=renderer, gtype=gtype, highlight=[dynamicnodes, staticnodes], colors=colors, showfaultlabels= show_dyn_order, **kwargs)
    
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
    renderer : 'matplotlib' or 'graphviz' or 'netgraph'
        Renderer to use to plot the graph. Default is 'matplotlib'
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
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
    g, pos = get_graph_pos(mdl, kwargs.get('pos', []), gtype)
    if renderer=='matplotlib':
        fig  = plt.figure(figsize=kwargs.pop('figsize', (6,4)))
        if gtype=='bipartite':      update_bipplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='typegraph':    update_typegraphplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='normal':       update_graphplot(t_ind, reshist, g, pos, **kwargs)
        else:           raise Exception("Graph type "+gtype+" not a valid option")
        return fig
    elif renderer=='netgraph':
        fig = plt.figure(figsize=kwargs.pop('figsize', (6,4)))
        if gtype=='bipartite':      fig, ax, gra = update_net_bipplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='typegraph':    fig, ax, gra = update_net_typegraphplot(t_ind, reshist, g, pos, **kwargs)
        elif gtype=='normal':       fig, ax, gra = update_net_graphplot(t_ind, reshist, g, pos, **kwargs)
        else:                       raise Exception("Graph type "+gtype+" not a valid option")
        return fig, gra
    elif renderer=='graphviz':
        if gtype=='bipartite': dot = update_gv_bipplot(t_ind, reshist, g, **kwargs)
        elif gtype=='normal':   dot = update_gv_graphplot(t_ind, reshist, g, **kwargs)
        else:           raise Exception("Graph type "+gtype+" not a valid option for graphviz renderer")
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
    renderer : 'matplotlib' or 'graphviz' or 'netgraph'
        Renderer to use to plot the graph. Default is 'matplotlib'
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
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
    g, pos = get_graph_pos(mdl, kwargs.get('pos', []), gtype)
    if times=='all':    t_inds= [i for i in range(0,len(reshist['time']))]
    else:               t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    frames = {}
    if renderer == 'matplotlib':
        for t_ind in t_inds:
            fig = plt.figure(figsize=kwargs.get('figsize', (6,4)))
            if gtype=='bipartite':      update_bipplot(t_ind, reshist, g, pos, show=False, **kwargs)
            elif gtype=='typegraph':    update_typegraphplot(t_ind, reshist, g, pos, show=False, **kwargs)
            elif gtype=='normal':       update_graphplot(t_ind, reshist, g, pos, show=False, **kwargs)
            else:           raise Exception("Graph type "+gtype+" not a valid option")
            frames[t_ind] = fig
    elif renderer == 'netgraph':
        for ind in t_inds:
            fig = plt.figure(figsize=kwargs.get('figsize', (6,4)))
            if gtype=='bipartite':      update_net_bipplot(t_ind, reshist, g, pos, **kwargs)
            elif gtype=='typegraph':    update_net_typegraphplot(t_ind, reshist, g, pos, **kwargs)
            elif gtype=='normal':       update_net_graphplot(t_ind, reshist, g, pos, **kwargs)
            else:           raise Exception("Graph type "+gtype+" not a valid option")
            frames[t_ind] = fig
    elif renderer == 'graphviz':
        for t_ind in t_inds:
            if gtype=='bipartite': dot = update_gv_bipplot(t_ind, reshist, g, **kwargs)
            elif gtype=='normal':   dot = update_gv_graphplot(t_ind, reshist, g, **kwargs)
            else:           raise Exception("Graph type "+gtype+" not a valid option for graphviz renderer")
            frames[t_ind] = dot
    return frames

def animation_from(mdl, reshist, times='all', faultscen=[], gtype='bipartite',figsize=(6,4), showfaultlabels=True, scale=1, show=False, pos=[], colors=['lightgray','orange', 'red'], renderer='matplotlib'):
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
    g, pos = get_graph_pos(mdl, pos, gtype)
    if times=='all':    t_inds= [i for i in range(0,len(reshist['time']))]
    else:   t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    if renderer=='matplotlib':
        if gtype=='bipartite':  update_plot = update_bipplot
        elif gtype=='normal':   update_plot = update_graphplot
        elif gtype=='typegraph':update_plot = update_typegraphplot
    elif renderer=='netgraph':
        if gtype=='bipartite':  update_plot = update_net_bipplot
        elif gtype=='normal':   update_plot = update_net_graphplot
        elif gtype=='typegraph':update_plot = update_net_typegraphplot
    
    fig = plt.figure(figsize=figsize)
    ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=t_inds, fargs=(reshist, g, pos, faultscen, showfaultlabels, scale, False, colors))
    if show: plt.show()
    return ani

###HELPER FUNCTIONS
#############################
def get_graph_pos(mdl, pos, gtype):
    """Helper function for getting the right graph/positions from a model"""
    if gtype=='normal': 
        g = mdl.graph.copy()
        if not pos:
            if mdl.graph_pos:   pos=mdl.graph_pos
            else:               pos=nx.shell_layout(g)
    elif gtype=='bipartite':
        g = mdl.bipartite.copy()
        if not pos:
            if mdl.bipartite_pos:   pos=mdl.bipartite_pos
            else:                   pos=nx.spring_layout(g)
    elif gtype=='typegraph':
        g=mdl.return_typegraph()
        if not pos: pos = netgraph.get_sugiyama_layout(list(g.edges), nodes=g.nodes)
    elif gtype=='component':
        g = mdl.return_stategraph('component')
        if not pos: pos=nx.spring_layout(g)
    else: raise Exception("Graph type "+gtype+" not valid")
    return g,pos
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

###MATPLOTLIB HELPER FUNCTIONS
#############################
def plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True, colors=['lightgray','orange', 'red'], title=[], show_edgelabels=True):
    """ Plots a standard graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*2000
    font_size=scale*12
    if not pos: pos=nx.shell_layout(g)
    nx.draw_networkx(g,pos,node_size=nodesize,font_size=font_size, node_shape='s',edge_color='gray', node_color=colors[0], width=3, font_weight='bold')
    if show_edgelabels: nx.draw_networkx_edge_labels(g,pos,font_size=font_size, edge_labels=edgeflows)
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_shape='s',node_color = colors[2], node_size = nodesize*1.2)
    nx.draw_networkx_nodes(g, pos, nodelist=degfxns,node_shape='s', node_color = colors[1], node_size = nodesize)
    nx.draw_networkx_edges(g,pos,edgelist=faultedges, edge_color=colors[1])
        
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=font_size, font_color='k')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=faultedgeflows,font_size=font_size, font_color=colors[1])
    plt.axis('off')
    return plt.gcf(), plt.gca()

def plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True, colors=['lightgray','orange', 'red'], title=[],functions=[], flows=[]):
    """ Plots a bipartite graph. Used in other functions"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    nodesize=scale*700
    font_size=scale*8
    if not pos: pos=nx.spring_layout(g)
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
def plot_gv_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultscen, time, showfaultlabels, colors_dict, dot):
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
        dot.edge(edge[0], edge[1], label=edge_label, color=colors_dict[edge], labelangle="180")
    return dot
def plot_gv_bipartite(g, faultnodes, degradednodes, faultlabels, faultscen, time, showfaultlabels, colors_dict, functions, flows, edges, dot):
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
        dot.edge(edge[0], edge[1])
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
    display(SVG(legend._repr_svg_()))
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
def update_gv_graphplot(t_ind, reshist, g, faultscen=[], showfaultlabels=True, colors=['lightgray','orange', 'red'], heatmap={}, cmap=plt.cm.coolwarm, **kwargs):
    """graphviz helpwer: Updates a normal graph plot at a given timestep t_ind given the result history reshist"""
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

###NETGRAPH HELPER FUNCTIONS
#############################
def plot_norm_netgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True, colors=['lightgray','orange', 'red'], title=[], show_edgelabels=True, **kwargs):
    """ Experimental method for plotting with netgraph instead of networkx"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    if not pos: pos=nx.shell_layout(g)
    from netgraph import Graph
    node_shape = {}; node_color ={}; node_edge_color={}
    for n in g.nodes:
        node_edge_color[n]=colors[0]
        node_shape[n]='s'
        if n in degfxns:  
            node_color[n]=colors[1]
            if n in faultfxns: node_edge_color[n] = colors[2]
        elif n in faultfxns: node_color[n]=colors[2]
        else:               node_color[n]=colors[0]
        if showfaultlabels and faultlabels.get(n,False):
            labels[n]=labels[n]+' \n'+' '.join([f for f in faultlabels[n] if f!='nom'])
            
    edge_color = {}
    for e in g.edges:
        edge_color[e] = colors[0]
        if e in faultedges: edge_color[e] = colors[1]
    if showfaultlabels and any(faultedgeflows): 
        edgelabels = faultedgeflows
        edge_label_fontdict={'size':scale*4, 'color':'red'}
    elif show_edgelabels: 
        edgelabels=edgeflows
        edge_label_fontdict={'size':scale*4, 'color':'black'}
    else:               
        edgelabels={}
        edge_label_fontdict={'size':scale*4, 'color':'black'}
    gra = Graph(g, node_layout=pos, edge_color=edge_color, edge_size=scale, edge_labels=edgelabels,edge_label_font_size=scale*2, edge_zorder=1,edge_label_fontdict=edge_label_fontdict,\
                node_label_fontdict={'size':scale*8, 'fontweight':'bold'}, node_size=scale*20, node_edge_width=scale*2,\
                node_shape = node_shape, node_color = node_color, node_edge_color = node_edge_color, node_labels=labels,  node_zorder=2)
    return plt.gcf(), plt.gca(), gra
def plot_bip_netgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True, colors=['lightgray','orange', 'red'], title=[],functions=[], flows=[], **kwargs):
    """ Experimental method for plotting with netgraph instead of networkx"""
    if faultscen:   plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    elif title:     plt.title(title)
    if not pos: pos=nx.spring_layout(g)
    if type(g)==nx.classes.digraph.DiGraph: arrows = True
    else:                                   arrows = False
    from netgraph import Graph
    node_shape = {}; node_color ={}; node_edge_color={}
    for n in g.nodes:
        if n in functions: node_shape[n]='s'
        else:               node_shape[n]='o'
        node_edge_color[n]=colors[0]
        if n in degnodes:  
            node_color[n]=colors[1]
            if n in faultfxns: node_edge_color[n] = colors[2]
            else:               node_edge_color[n] = colors[1]
        elif n in faultfxns: node_color[n]=colors[2]
        else:               node_color[n]=colors[0]
        if showfaultlabels and faultlabels.get(n,False):
            labels[n]=labels[n]+' \n'+''.join([f for f in faultlabels[n] if f!='nom'])
    gra = Graph(g, node_layout=pos, node_label_fontdict={'size':scale*8, 'fontweight':'bold'}, node_size=scale*10, node_edge_width=scale,\
                node_shape = node_shape, node_color = node_color, node_edge_color = node_edge_color, node_labels=labels,  node_zorder=2, arrows=arrows)
    return plt.gcf(), plt.gca(), gra
def update_net_graphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], **kwargs):
    """Updates a normal graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    fig, ax, gra = plot_norm_netgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, pos, show, colors=colors, **kwargs)
    return fig, ax, gra
def update_net_bipplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], **kwargs):
    """Updates a bipartite graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    fig, ax, gra = plot_bip_netgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors, functions = reshist['functions'].keys(), flows=reshist['flows'].keys(), **kwargs)
    return fig, ax, gra
def update_net_typegraphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True, colors=['lightgray','orange', 'red'], **kwargs):
    """Updates a typegraph-stype plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    fig, ax, gra = plot_bip_netgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show, colors=colors, **kwargs)
    return fig, ax, gra