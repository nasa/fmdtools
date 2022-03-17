# -*- coding: utf-8 -*-
"""
Description: Methods for high-level network simulation and analysis.

Main Methods:
    - :meth:`calc_aspl()`:              Computes average shortest path length of graph representation of model mdl.
    - :meth:`calc_modularity()`:        Computes graph modularity given a graph representation of model mdl.
    - :meth:`find_bridging_nodes()`:    Determines bridging nodes in a graph representation of model mdl.
    - :meth:`find_high_degree_nodes()`: Determines highest degree nodes, up to percentile p, in graph representation of model mdl.
    - :meth:`calc_robustness_coefficient()`:    Computes robustness coefficient of graph representation of model mdl.
    - :meth:`sff_model()`:                      Susceptible-Fixed-Failed Model Simulation
    - :meth:`degree_dist()`:            Plots degree distribution of graph representation of model mdl.
"""
#File name: networks.py
#Author: Hannah Walsh
#Created: April 2020

import numpy as np
import networkx as nx
import random
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import math
from fmdtools.resultdisp.graph import plot_normgraph, plot_bipgraph

# Network Metric Quantification
def calc_aspl(mdl, gtype='parameter'):
    """
        Computes average shortest path length of graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
            graph to run the analysis for. (will get from model if provided)
        gtype : str
            type of graph representation of the model to show. default is 'bipartite'
        
        Returns
        -------
        ASPL : average shortest path length
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    ASPL = nx.average_shortest_path_length(g)
    return ASPL
def calc_modularity(mdl, gtype='parameter'):
    """
        Computes graph modularity given a graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
            graph to run the analysis for. (will get from model if provided)
        gtype : str
            type of graph representation of the model to show. default is 'bipartite'
        
        Returns
        -------
        modularity : Modularity
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    communities = list(greedy_modularity_communities(g))
    m = modularity(g,communities)
    return m
def find_bridging_nodes(mdl,plot='off', gtype = 'parameter', pos={}, scale=1):
    """
        Determines bridging nodes in a graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
            graph to run the analysis for. (will get from model if provided)
        plot : str (optional)
            plots graph with high degree nodes visualized if set to 'on'
        gtype : str
            type of graph representation of the model to show. default is 'bipartite'
        pos : dict  (optional)
            dict of node positions in the model (if desired)
        scale : int (optional)
            scale for the plot. Default is 1. 
        
        Returns
        -------
        bridgingNodes : list of bridging nodes
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    communitiesRaw = list(greedy_modularity_communities(g))
    communities = [list(x) for x in communitiesRaw]
    numCommunities = len(communities)
    nodes = list(g.nodes)
    numNodes = len(nodes)
    bridgingNodes = list()
    nodeEdges = list()
    for i in range(0,numNodes):
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
    if plot == 'on':
        if gtype=='normal':
            fig, ax= plot_normgraph(g,{},bridgingNodes,{},{},{},{},{},{},{},False,{}, pos=pos, scale=scale, colors=['lightgray','yellow', 'yellow'],show=False, title='Bridging Nodes')
        else:
            fig, ax = plot_bipgraph(g,{n:n for n in g.nodes()},{},bridgingNodes,{},{},showfaultlabels=False, pos=pos, scale=scale, colors=['lightgray','yellow', 'yellow'],show=False,  title='Bridging Nodes')
        plt.show()
        return bridgingNodes, fig, ax
    else:
        return bridgingNodes
def find_high_degree_nodes(mdl,p=90,plot='off', gtype='bipartite', pos={}, scale=1):
    """
        Determines highest degree nodes, up to percentile p, in graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
            graph to run the analysis for. (will get from model if provided)
        p : int (optional)
            percentile of degrees to return, between 0 and 100
        plot : str (optional)
            plots graph with high degree nodes visualized if set to 'on'
        gtype : str
            type of graph representation of the model to show. default is 'bipartite'
        pos : dict  (optional)
            dict of node positions in the model (if desired)
        scale : int (optional)
            scale for the plot. Default is 1. 
        
        Returns
        -------
        highDegreeNodes : list of high degree nodes in format (node,degree)
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    d = list(g.degree())
    def take_second(elem):
        return elem[1]
    sortedNodes = sorted(d, key=take_second, reverse=True)
    sortedDegrees = [x[1] for x in sortedNodes]
    sortedDegreesSet = set(sortedDegrees)
    sortedDegreesUnique = list(sortedDegreesSet)
    sortedDegreesUniqueArray = np.array(sortedDegreesUnique)
    topPercentileDegree = np.percentile(sortedDegreesUniqueArray,p)
    numNodes = len(sortedNodes)
    highDegreeNodes = [sortedNodes[0]]
    for i in range(1,numNodes):
        if sortedNodes[i][1] < topPercentileDegree:
            pass
        else:
            highDegreeNodes.append(sortedNodes[i])
    if plot == 'on':
        if gtype=='normal':
            fig, ax= plot_normgraph(g,{},[h for h,i in highDegreeNodes],{},{},{},{},{},{},{},False,{}, pos=pos, scale=scale, colors=['lightgray','red', 'red'],show=False, title='High Degree Nodes ('+str(p)+'th Percentile)')
        else:
            fig, ax = plot_bipgraph(g,{n:n for n in g.nodes()}, {},[h for h,i in highDegreeNodes],{},{},showfaultlabels=False, pos=pos, scale=scale, colors=['lightgray','red', 'red'], show=False,  title='High Degree Nodes ('+str(p)+'th Percentile)')
        plt.show()
        return highDegreeNodes, fig, ax
    else:
        return highDegreeNodes
def calc_robustness_coefficient(mdl,trials=100, gtype='bipartite'):
    """
        Computes robustness coefficient of graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
            graph to calculate robustness coefficent for. (will get from model if provided)
        trials : int 
            number of times to run robustness coefficient algorithm (result is averaged over all trials)
        gtype : str
            type of graph representation of the model to show. default is 'bipartite'
        
        Returns
        -------
        RC : robustness coefficient
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    trialsRC = list()
    for itr in range(trials):
        tmp = g.copy()
        N = float(len(tmp))
        largestCC = max(nx.connected_components(tmp), key=len)
        s = [float(len(largestCC))]
        rs = random.sample(range(int(s[0])),int(s[0]))
        nodes = list(g)
        for i in range(int(s[0])-1):
            tmp.remove_node(nodes[rs[i]])
            largestCC = max(nx.connected_components(tmp), key=len)
            s.append(float(len(largestCC)))
        trialsRC.append((200*sum(s)-100*s[0])/N/N)
    RC = sum(trialsRC)/len(trialsRC)
    return RC
def degree_dist(mdl, gtype='bipartite'):
    """
        Plots degree distribution of graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
            graph to calculated degree distribution for. (will get from model if provided)
        gtype : str
            type of graph representation of the model to show. default is 'bipartite'
        Returns
        -------
        fig : matplotlib figure
            plot of distribution
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    degrees = [g.degree(n) for n in g.nodes()]
    degreesSet = set(degrees)
    degreesUnique = list(degreesSet)
    freq = [degrees.count(n) for n in degreesUnique]
    maxFreq = max(freq)
    freqint = list(range(0,maxFreq+1))
    degreeint = list(range(min(degrees),math.ceil(max(degrees))+1))
    degreesSet = set(degrees)
    degreesUnique = list(degrees)
    numDegreesUnique = len(degreesUnique)
    fig = plt.figure()
    plt.hist(degrees,bins=np.arange(numDegreesUnique)-0.5)
    plt.xticks(degreeint)
    plt.yticks(freqint)
    plt.title('Degree distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()
    return fig

def sff_model(mdl,gtype='parameter',endtime=5,pi=.1,pr=.1,num_trials=100,start_node='random',error_bar_option='off'):
    """
    susc-fix-fail model.
    
    Parameters
    ----------
    mdl : model or graph
        graph to run trials over (will get from model if provided)
    endtime: int
        simulation end time
    pi : float
        infection (failure spread) rate
    pf : float
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
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:									g = get_graph(mdl, gtype)
    if start_node == 'random':
        nodes = list(g.nodes)
        start_node_selected= nodes[random.randint(0,len(nodes))]
    else: start_node_selected=start_node
    num_susc_all_trials = []
    num_fail_all_trials = []
    num_fix_all_trials = []
    for trials in range(0,num_trials):
   		num_susc_trial, num_fail_trial, num_fix_trial = sff_one_trial(start_node_selected,g,endtime=endtime,pi=pi,pr=pr)
   		num_susc_all_trials.append(num_susc_trial)
   		num_fail_all_trials.append(num_fail_trial)
   		num_fix_all_trials.append(num_fix_trial)
    num_susc_average = data_average(num_susc_all_trials)
    num_fail_average = data_average(num_fail_all_trials)
    num_fix_average = data_average(num_fix_all_trials)
   
    fig = plt.figure()
    time_list = range(0,endtime+1)
    if error_bar_option == 'on':
   		num_susc_lower_error, num_susc_upper_error = data_error(num_susc_all_trials,num_susc_average)
   		num_fail_lower_error, num_fail_upper_error = data_error(num_fail_all_trials,num_fail_average)
   		num_fix_lower_error, num_fix_upper_error = data_error(num_fix_all_trials,num_fix_average)
   		num_susc_asymmetric_error = [num_susc_lower_error, num_susc_upper_error]
   		num_fail_asymmetric_error = [num_fail_lower_error, num_fail_upper_error]
   		num_fix_asymmetric_error = [num_fix_lower_error, num_fix_upper_error]
   		plt.errorbar(time_list,num_susc_average,yerr=num_susc_asymmetric_error,fmt='-o',label='Susceptible')
   		plt.errorbar(time_list,num_fail_average,yerr=num_fail_asymmetric_error,fmt='-o',label='Failed')
   		plt.errorbar(time_list,num_fix_average,yerr=num_fix_asymmetric_error,fmt='-o',label='Fixed')
    else:
   		plt.plot(time_list,num_susc_average,label='Susceptible')
   		plt.plot(time_list,num_fail_average,label='Failed')
   		plt.plot(time_list,num_fix_average,label='Fixed')
    plt.legend()
    plt.title('SFF model')
    plt.xlabel('Time steps')
    plt.ylabel('Number of nodes')
    plt.show()
    return fig

def sff_one_trial(start_node_selected,g,endtime=5,pi=.1,pr=.1):
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
    pf : float
        recovery (fix) rate
    """
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
        for i in range(0,len(fail_nodes)):
            n = list(g.neighbors(fail_nodes[i])) 
            new_exposed_nodes.extend(n)
        ri_list = [random.random() for iter in range(len(new_exposed_nodes))]
        new_fail_nodes = []
        for i in range(0,len(new_exposed_nodes)):
            if new_exposed_nodes[i] in fix_nodes: pass
            else:
                if ri_list[i] <= pi:
                    new_fail_nodes.append(new_exposed_nodes[i])
        new_fail_nodes_set = set(new_fail_nodes)
        new_fail_nodes = list(new_fail_nodes_set)
        for i in range(0,len(new_fail_nodes)):
            if new_fail_nodes[i] in fail_nodes: pass
            else: susc_nodes.remove(new_fail_nodes[i])
        fail_nodes.extend(new_fail_nodes)
        fail_nodes_set = set(fail_nodes)
        fail_nodes = list(fail_nodes_set)
        rf_list = [random.random() for iter in range(len(fail_nodes))]
        new_fix_nodes = []
        for i in range(0,len(fail_nodes)):
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

def data_error(data,average):
    """Calculates error for each column in data"""
    q1 = []
    q3 = []
    for i in range(0,len(data[0])):
        current_array = np.array([float(x[i]) for x in data])
        q1.append(np.percentile(current_array,25))
        q3.append(np.percentile(current_array,75))
    lower_error = [x - y for x, y in zip(average,q1)]
    upper_error = [x - y for x, y in zip(q3,average)]
    return lower_error, upper_error

def get_graph(mdl, gtype):
    "gets the appropriate graph of type gtype from mdl"
    if gtype == 'normal':       g = mdl.graph
    elif gtype == 'bipartite':  g = mdl.bipartite
    elif gtype == 'parameter':  g = mdl.return_paramgraph()
    elif gtype == 'component':  g = mdl.return_stategraph(gtype='component')
    return g
