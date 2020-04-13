# -*- coding: utf-8 -*-
"""
File name: networks.py
Author: Hannah Walsh
Created: April 2020


"""

import numpy as np
import networkx as nx
import random
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import math


# Network Metric Quantification
def calc_aspl(mdl, gtype='parameter'):
    """
        Computes average shortest path length of graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
        
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
        
        Returns
        -------
        modularity : Modularity
        """
    if type(mdl)==nx.classes.graph.Graph:   g = mdl
    else:                                   g = get_graph(mdl, gtype)
    communities = list(greedy_modularity_communities(g))
    m = modularity(g,communities)
    return m
def find_bridging_nodes(mdl,plot='off', gtype = 'parameter'):
    """
        Determines bridging nodes in a graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
        
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
        plt.figure()
        color_map = []
        for node in g:
            if node in bridgingNodes:
                color_map.append('yellow')
            else:
                color_map.append('gray')
        nx.draw_networkx(g,node_color=color_map,with_labels=True)
        plt.title('Bridging Nodes')
        plt.show()
    return bridgingNodes
def find_high_degree_nodes(mdl,p=.1,plot='off', gtype='bipartite'):
    """
        Determines highest degree nodes, up to percentile p, in graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
        p : percentile of degrees to return, between 0 and 1
        plot : plots graph with high degree nodes visualized if set to 'on'
        
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
    numDegrees = len(sortedDegreesUnique)
    topPercentileDegree = sortedDegreesUnique[int(round(numDegrees*p))-1]
    numNodes = len(sortedNodes)
    highestDegree = sortedNodes[0][1]
    highDegreeNodes = [sortedNodes[0]]
    for i in range(1,numNodes):
        if sortedNodes[i][1] < topPercentileDegree:
            pass
        else:
            highDegreeNodes.append(sortedNodes[i])
    if plot == 'on':
        plt.figure()
        color_map = []
        for node in g:
            if node in [x[0] for x in highDegreeNodes]:
                color_map.append('red')
            else:
                color_map.append('gray')
        nx.draw_networkx(g,node_color=color_map,with_labels=True)
        plt.title('High Degree Nodes')
        plt.show()
    return highDegreeNodes
def calc_robustness_coefficient(mdl,trials=100, gtype='bipartite'):
    """
        Computes robustness coefficient of graph representation of model mdl.
        
        Parameters
        ----------
        mdl : model or graph
        trials : number of times to run robustness coefficient algorithm (result is averaged over all trials)
        
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
        
        Returns
        -------
        
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
    plt.figure()
    plt.hist(degrees,bins=np.arange(numDegreesUnique)-0.5)
    plt.xticks(degreeint)
    plt.yticks(freqint)
    plt.title('Degree distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def get_graph(mdl, gtype):
    "gets the appropriate graph of type gtype from mdl"
    if gtype == 'normal':       g = mdl.graph
    elif gtype == 'bipartite':  g=mdl.bipartite
    elif gtype == 'parameter':  g = mdl.return_paramgraph()
    elif gtype == 'component':  g = mdl.return_stategraph(gtype='component')
    return g
