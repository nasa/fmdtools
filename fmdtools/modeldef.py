# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""
import numpy as np
import operator
import itertools
import networkx as nx
from scipy.stats import binom

# MAJOR CLASSES
class Block(object):
    """ 
    Superclass for FxnBlock and Component subclasses. Has functions for model setup, querying state, reseting the model
    
    Attributes
    ----------
    timely : bool
        Whether or not the block state depends on time (or just inputs and outputs)
    failrate : float
        Failure rate for the block
    time : float
        internal time of the function
    faults : set
        faults currently present in the block. If the function is nominal, set is {'nom'}
    faultmodes : dict
        faults possible to inject in the block and their properties. Has structure:
            - faultname :
                - dist : (float of % failures due to this fualt)
                - oppvect : (list of relative probabilities of the fault occuring in each phase)
                - rcost : cost of repairing the fault
    """
    def __init__(self, states={}, timely=True):
        """
        Instance superclass. Called by FxnBlock and Component classes.

        Parameters
        ----------
        states : dict, optional
            Internal states (variables, essentially) of the block. The default is {}.
        timely : bool, optional
            Whether or not the function is dependent on time (or just inputs/outputs). The default is True.
        """
        self.timely=timely
        self._states=states.keys()
        self._initstates=states.copy()
        self.failrate = getattr(self, 'failrate', 1.0)
        for state in states.keys():
            setattr(self, state,states[state])
        self.faults=set(['nom'])
        if timely: self.time=0.0
    def add_he_rate(self,gtp,EPCs={'na':[1,0]}):
        """
        Calculates self.failrate based on a human error probability model.

        Parameters
        ----------
        gtp : float
            Generic Task Probability. (from HEART)
        EPCs : Dict or list
            Error producing conditions (and respective factors) for a given task (from HEART). Used in format:
            Dict {'name':[EPC factor, Effect proportion]} or list [[EPC factor, Effect proportion],[[EPC factor, Effect proportion]]]
        """
        if type(EPCs)==dict:    EPC_f = np.prod([((epc-1)*x+1) for _, [epc,x] in EPCs.items()])
        elif type(EPCs)==list:  EPC_f = np.prod([((epc-1)*x+1) for [epc,x] in EPCs])
        self.failrate = gtp*EPC_f
    def assoc_modes(self, modes, name='', probtype='rate', units='hr'):
        """
        Associates modes with the block when called in the function or component.

        Parameters
        ----------
        modes : dict
            Dictionary of modes with structure {faultname:[dist,oppvect, rcost]}
        name : str, optional
            (for components only) Name of the component. The default is ''.
        """
        if not getattr(self, 'faultmodes', []): 
            if name: self.faultmodes=dict()
            else:    self.faultmodes=dict.fromkeys(modes)
        for mode in modes:
            self.faultmodes[name+mode]=dict.fromkeys(('dist', 'oppvect', 'rcost', 'probtype', 'units'))
            self.faultmodes[name+mode]['probtype'] = probtype
            self.faultmodes[name+mode]['units'] = units
            if type(modes) == set: # minimum information - here the modes are only a set of labels
                self.faultmodes[name+mode]['dist'] =     1.0/len(modes)
                self.faultmodes[name+mode]['oppvect'] =  [1.0]
                self.faultmodes[name+mode]['rcost'] =    0.0
            elif type(modes[mode]) == float: # dict of modes: dist, where dist is the distribution (or individual rate/probability)
                self.faultmodes[name+mode]['dist'] =     modes[mode]
                self.faultmodes[name+mode]['oppvect'] =  [1.0]
                self.faultmodes[name+mode]['rcost'] =    0.0
            elif len(modes[mode]) == 3:   # three-arg mode definition: dist, oppvect, repair costs
                self.faultmodes[name+mode]['dist'] =     modes[mode][0]
                self.faultmodes[name+mode]['oppvect'] =  modes[mode][1]
                self.faultmodes[name+mode]['rcost'] =    modes[mode][2]
            elif len(modes[mode]) == 2:  # dist, repair costs
                self.faultmodes[name+mode]['dist'] =     modes[mode][0]
                self.faultmodes[name+mode]['oppvect'] =  [1.0]
                self.faultmodes[name+mode]['rcost'] =    modes[mode][1]
            elif len(modes[mode]) == 1:  # dist only
                self.faultmodes[name+mode]['dist'] =     modes[mode][0]
                self.faultmodes[name+mode]['oppvect'] =  [1.0]
                self.faultmodes[name+mode]['rcost'] =    0.0
            else:
                raise Exception("Invalid mode definition")                            
    def has_fault(self,fault): 
        """Check if the block has fault (a str)"""
        return self.faults.intersection(set([fault]))
    def has_faults(self,faults): 
        """Check if the block has any in the list of faults"""
        return self.faults.intersection(set(faults))
    def add_fault(self,fault): 
        """Adds fault (a str) to the block"""
        self.faults.update([fault])
    def add_faults(self,faults): 
        """Adds list of faults to the block"""
        self.faults.update(faults)
    def replace_fault(self, fault_to_replace,fault_to_add): 
        """Replaces fault_to_replace with fault_to_add in the set of faults"""
        self.faults.add(fault_to_add)
        self.faults.remove(fault_to_replace)
    def reset(self):            #reset requires flows to be cleared first
        """ Resets the block to the initial state with no faults. Used (only for components) when resetting the model"""
        self.faults.clear()
        self.faults.add('nom')
        for state in self._initstates.keys():
            setattr(self, state,self._initstates[state])
        self.time=0
    def return_states(self):
        """
        Returns states of the block at the current state. Used (iteratively) to record states over time.

        Returns
        -------
        states : dict
            States (variables) of the block
        faults : set
            Faults present in the block
        """
        states={}
        for state in self._states:
            states[state]=getattr(self,state)
        return states, self.faults.copy()

#Function superclass 
class FxnBlock(Block):
    """
    Superclass for functions.
    
    Attributes
    ----------
    type : str
        labels the function as a function (may not be necessary) Default is 'function'
    flows : dict
        flows associated with the function. structured {flow:{value:XX}}
    components : dict
        component instantiations of the function (if any)
    timers : set
        names of timers to be used in the function (if any)
    tstep : float
        timestep of the model in the function (added in model definition)
    """
    def __init__(self,flownames,flows, states={}, components={},timers={}, timely=True):
        """
        Intances the function superclass with the relevant parameters.

        Parameters
        ----------
        flownames : list
            Names of flows (in order) to use in the function 
        flows :list
            Flow objects to (in order correspoinding to flownames) associate with the function
        states : dict, optional
            Internal states to associate with the function. The default is {}.
        components : dict, optional
            Component objects to associate with the function. The default is {}.
        timers : set, optional
            Set of names of timers to use in the function. The default is {}.
        timely : bool, optional
            Whether or not the function depends on time (or just input/output). The default is True.
        """
        self.type = 'function'
        self.flows=self.make_flowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow,self.flows[flow])
        self.components=components
        if not getattr(self, 'faultmodes', []): self.faultmodes={}
        if self.components: self.compfaultmodes= dict()
        for cname in components:
            self.faultmodes.update(components[cname].faultmodes)
            self.compfaultmodes.update({modename:cname for modename in components[cname].faultmodes})
        self.timers = timers
        for timername in timers:
            setattr(self, timername, Timer(timername))
        super().__init__(states, timely)
    def make_flowdict(self,flownames,flows):
        """
        Puts a list of flows with a list of flow names in a dictionary.

        Parameters
        ----------
        flownames : list
            names of flows corresponding to flows
        flows : list
            flows

        Returns
        -------
        flowdict : dict
            dict of flows indexed by flownames
        """
        flowdict={}
        for ind, flowname in enumerate(flownames):
            flowdict[flowname]=flows[ind]
        return flowdict
    def condfaults(self,time):
        """ Placeholder for function condfaults methods """
        return 0
    def behavior(self,time):
        """ Placeholder for function behavior methods """
        return 0        
    def reset(self):            
        """
        Resets the internal states and faults of the function to the intial state. Used when reseting the model. Requires associated flows to be cleared first.
        """
        self.faults.clear()
        self.faults.add('nom')
        for state in self._initstates.keys():
            setattr(self, state,self._initstates[state])
        for name, component in self.components.items():
            component.reset()
        for timername in self.timers:
            getattr(self, timername).reset()
        if hasattr(self, 'time'): self.time=0.0
        if hasattr(self, 'tstep'): self.tstep=self.tstep
        self.updatefxn(faults=['nom'], time=0)
    def copy(self, newflows, *attr):
        """
        Creates a copy of the function object with newflows and arbitrary parameters associated with the copy. Used when copying the model.

        Parameters
        ----------
        newflows : list
            list of new flow objects to be associated with the copy of the function
        *attr : any
            arbitrary parameters to add (if funciton takes in more than flows e.g. design variables)

        Returns
        -------
        copy : FxnBlock
            Copy of the given function with new flows
        """
        copy = self.__class__(newflows, *attr)  # Is this adequate? Wouldn't this give it new components?
        copy.faults = self.faults.copy()
        for state in self._initstates.keys():
            setattr(copy, state, self._initstates[state])
        if hasattr(self, 'time'): copy.time=self.time
        if hasattr(self, 'tstep'): copy.tstep=self.tstep
        return copy
    def updatefxn(self,faults=['nom'], time=0):
        """
        Updates the state of the function at a given time and injects faults.

        Parameters
        ----------
        faults : list, optional
            Faults to inject in the function. The default is ['nom'].
        time : float, optional
            Model time. The default is 0.
        """
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults(time)           #conditional faults and behavior are then run
        if self.components:     # propogate faults from function level to component level
            for fault in self.faults:
                if fault in self.compfaultmodes:
                    self.components[self.compfaultmodes[fault]].add_fault(fault)
        self.behavior(time)
        if self.components:     # propogate faults from component level to function level
            for compname, comp in self.components.items():
                self.faults.update(comp.faults) 
        self.time=time
        return

#Future ActGraph class
#Development in progress
#class ActGraph(Block):
#    def __init__(self):
#        super().__init__()
#        self.graph = nx.DiGraph()
#    def add_actions(self,actions):
#        self.actionmethods = {act:val[0] for act,val in actions.items()}
#        self.actionmodes =   {act:val[1] for act,val in actions.items()}
#        self.actiontimes =  {act:val[2] for act,val in actions.items()}
#        self.actions = list(actions.keys())
#        self.graph.add_nodes_from(self.actions)
#    def connect_actions(self,actionpairs, root='na'):
#        self.actionpairs = actionpairs
#        self.graph.add_edges_from(self.actionpairs)
#        if root == 'na': root = self.actions[0]
#
#    def add_actionmodes(self,modes):
#        return 0 
#    def updatefxn(self,faults=['nom'], time=0):
#         self.time=time
 ###     
        
class Component(Block):
    """
    Superclass for components (most attributes and methods inherited from Block superclass)
    """
    def __init__(self,name, states={}, timely=True):
        """
        Inherit the component class

        Parameters
        ----------
        name : str
            Unique name ID for the component
        states : dict, optional
            States to use in the component. The default is {}.
        timely : bool, optional
            Whether the component depends on time or just input/output behavior. The default is True.
        """
        self.type = 'component'
        self.name = name
        super().__init__(states, timely)
    def behavior(self,time):
        """ Placeholder for component behavior methods """
        return 0

class Flow(object):
    """
    Superclass for flows. Instanced by Model.add_flow but can also be used as a flow superclass if flow attributes are not easily definable as a dict.
    """
    def __init__(self, attributes, name):
        """
        Instances the flow with given attributes.

        Parameters
        ----------
        attributes : dict
            attributes and their values to be associated with the flow
        name : str
            name of the flow
        """
        self.type='flow'
        self.flow=name
        self._initattributes=attributes.copy()
        self._attributes=attributes.keys()
        for attribute in self._attributes:
            setattr(self, attribute, attributes[attribute])
    def reset(self):
        """ Resets the flow to the initial state"""
        for attribute in self._initattributes:
            setattr(self, attribute, self._initattributes[attribute])
    def status(self):
        """
        Returns a dict with the current states of the flow.
        """
        attributes={}
        for attribute in self._attributes:
            attributes[attribute]=getattr(self,attribute)
        return attributes
    def copy(self):
        """
        Returns a copy of the flow object (used when copying the model)
        """
        attributes={}
        for attribute in self._attributes:
            attributes[attribute]=getattr(self,attribute)
        if self.__class__==Flow:
            copy = self.__class__(attributes, self.flow)
        else:
            copy = self.__class__()
            for attribute in self._attributes:
                setattr(copy, attribute, getattr(self,attribute))
        return copy

#Model superclass    
class Model(object):
    """
    Model superclass used to construct the model, return representations of the model, and copy and reset the model when run.
    
    Attributes
    ----------
    type : str
        labels the model as a model (may not be necessary)
    flows : dict
        dictionary of flows objects in the model indexed by name
    fxns : dict
        dictionary of functions in the model indexed by name
    params : dict
        dictionary of (optional) parameters for a given instantiation of a model
    timelyfxns : set
        set of functions that are timely (depend on time, not just input/output)
    bipartite : networkx graph
        bipartite graph view of the functions and flows
    graph : networkx graph
        multigraph view of functions and flows
    """
    def __init__(self, params={},modelparams={}):
        """
        Instantiates internal model attributes with predetermined:
            - params (design variables of he model), and
            - modelparams (dictionary of phases, times, and timestep to run the model with)
        """
        self.type='model'
        self.flows={}
        self.fxns={}
        self.params=params
        
        # model defaults to static representation if no timerange
        self.phases=modelparams.get('phases',{'na':[1]})
        self.times=modelparams.get('times',[1])
        self.tstep = modelparams.get('tstep', 1.0)
        self.units = modelparams.get('units', 'hr')
        
        self.timelyfxns=set()
        self._fxnflows=[]
        self._fxninput={}
    def add_flow(self,flowname, flowtype, flowdict):
        """
        Adds a flow with given attributes to the model.

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the model
        flowtype : str
            Type of flow (e.g. EE, ME, etc)
        flowdict : dict or Flow
            Dictionary of flow attributes e.g. {'value':XX}, or the Flow object
        """
        if type(flowdict) == dict:
            self.flows[flowname]=Flow(flowdict, flowtype)
        elif isinstance(flowdict, Flow):
            self.flows[flowname] = flowdict
        else: raise Exception('Invalid flow. Must be dict or flow')
    def add_fxn(self,name,classobj, flownames, *args):
        """
        Instantiates a given function in the model.

        Parameters
        ----------
        name : str
            Name to give the function.
        classobj : Class
            Class to instantiate the function as.
        flownames : list
            List of flows to associate with the function.
        *args : arbitrary
            Other parameters to send to the __init__ method of the function class
        """
        flows=self.get_flows(flownames)
        if args: 
            self.fxns[name]=classobj(flows,args)
            self._fxninput[name]={'flows': flownames, 'args': args}
        else: 
            self.fxns[name]=classobj(flows)
            self._fxninput[name]={'flows': flownames, 'args': []}
        for flowname in flownames:
            self._fxnflows.append((name, flowname))
        if self.fxns[name].timely: self.timelyfxns.update([name])
        self.fxns[name].tstep=self.tstep
    def get_flows(self,flownames):
        """ Returns a list of the model flow objects """
        return [self.flows[flowname] for flowname in flownames]
    def construct_graph(self):
        """
        Creates and returns a graph representation of the model

        Returns
        -------
        graph : networkx graph
            multgraph representation of the model functions and flows
        """
        self.bipartite=nx.Graph()
        self.bipartite.add_nodes_from(self.fxns, bipartite=0)
        self.bipartite.add_nodes_from(self.flows, bipartite=1)
        self.bipartite.add_edges_from(self._fxnflows)
        self.multgraph = nx.projected_graph(self.bipartite, self.fxns,multigraph=True)
        self.graph = nx.projected_graph(self.bipartite, self.fxns)
        attrs={}
        #do we still need to do this for the objects? maybe not--I don't think we use the info anymore
        for edge in self.graph.edges:
            midedges=list(self.multgraph.subgraph(edge).edges)
            flows= [midedge[2] for midedge in midedges]
            flowdict={}
            for flow in flows:
                flowdict[flow]=self.flows[flow]
            attrs[edge]=flowdict
        nx.set_edge_attributes(self.graph, attrs)
        
        nx.set_node_attributes(self.graph, self.fxns, 'obj')
        #self.graph=nx.DiGraph()
        #self.graph.add_nodes_from(self.fxn)
        #self.graph=
        return self.graph
    def return_componentgraph(self, fxnname):
        """
        Returns a graph representation of the components associated with a given funciton

        Parameters
        ----------
        fxnname : str
            Name of the function (e.g. in mdl.fxns)

        Returns
        -------
        g : networkx graph
            Bipartite graph representation of the function with components.
        """
        g = nx.Graph()
        g.add_nodes_from([fxnname], bipartite=0)
        g.add_nodes_from(self.fxns[fxnname].components, bipartite=1)
        g.add_edges_from([(fxnname, component) for component in self.fxns[fxnname].components])        
        return g
    def return_stategraph(self, gtype='normal'):
        """
        Returns a graph representation of the current state of the model.

        Parameters
        ----------
        gtype : str, optional
            Type of graph to return (normal, bipartite, or component). The default is 'normal'.

        Returns
        -------
        graph : networkx graph
            Graph representation of the system with the modes and states added as attributes.
        """
        if gtype=='normal':
            graph=nx.projected_graph(self.bipartite, self.fxns)
        elif gtype=='bipartite':
            graph=self.bipartite.copy()
        elif gtype=='component':
            graph=self.bipartite.copy()
            for fxnname, fxn in self.fxns.items():
                graph.add_nodes_from(fxn.components, bipartite=1)
                graph.add_edges_from([(fxnname, component) for component in fxn.components])     
        edgevals, fxnmodes, fxnstates, flowstates, compmodes, compstates, comptypes ={}, {}, {}, {}, {}, {}, {}
        if gtype=='normal': #set edge values for normal graph
            for edge in graph.edges:
                midedges=list(self.multgraph.subgraph(edge).edges)
                flows= [midedge[2] for midedge in midedges]
                flowdict={}
                for flow in flows: 
                    flowdict[flow]=self.flows[flow].status()
                edgevals[edge]=flowdict
            nx.set_edge_attributes(graph, edgevals) 
        elif gtype=='bipartite' or gtype=='component': #set flow node values for bipartite graph
            for flowname, flow in self.flows.items():
                flowstates[flowname]=flow.status()
            nx.set_node_attributes(graph, flowstates, 'states')
        #set node values for functions
        for fxnname, fxn in self.fxns.items():
            fxnstates[fxnname], fxnmodes[fxnname] = fxn.return_states()
            if gtype=='normal': del graph.nodes[fxnname]['bipartite']
            if gtype=='component':
                for mode in fxnmodes[fxnname].copy():
                    for compname, comp in fxn.components.items():
                        compstates[compname]={}
                        comptypes[compname]=True
                        if mode in comp.faultmodes:
                            compmodes[compname]=compmodes.get(compname, set())
                            compmodes[compname].update([mode])
                            fxnmodes[fxnname].remove(mode)
                            fxnmodes[fxnname].update(['Comp_Fault'])
        nx.set_node_attributes(graph, fxnstates, 'states')
        nx.set_node_attributes(graph, fxnmodes, 'modes')
        if gtype=='component': 
            nx.set_node_attributes(graph,compstates, 'states')
            nx.set_node_attributes(graph, compmodes, 'modes') 
            nx.set_node_attributes(graph, comptypes, 'iscomponent')
        return graph
    def return_faultmodes(self):
        """
        Returns faultmodes present in the model

        Returns
        -------
        modes : dict
            Fault modes present in the model indexed by function name
        modeprops : dict
            Fault mode properties (defined in the function definition) with structure {fxn:mode:properties}
        """
        modes, modeprops = {}, {}
        for fxnname, fxn in self.fxns.items():
            ms = [m for m in fxn.faults.copy() if m!='nom']
            if ms: 
                modeprops[fxnname] = {}
                modes[fxnname] = ms
            for mode in ms:
                if mode!='nom': 
                    modeprops[fxnname][mode] = fxn.faultmodes[mode]
        return modes, modeprops
    def copy(self):
        """
        Copies the model at the current state.

        Returns
        -------
        copy : Model
            Copy of the curent model.
        """
        copy = self.__class__(params=getattr(self, 'params', {}))
        for flowname, flow in self.flows.items():
            copy.flows[flowname]=flow.copy()
        for fxnname, fxn in self.fxns.items():
            flownames=self._fxninput[fxnname]['flows']
            args=self._fxninput[fxnname]['args']
            flows = copy.get_flows(flownames)
            if args:    copy.fxns[fxnname]=fxn.copy(flows, args)
            else:       copy.fxns[fxnname]=fxn.copy(flows)
        _ = copy.construct_graph()
        return copy
    def reset(self):
        """Resets the model to the initial state (with no faults, etc)"""
        for flowname, flow in self.flows.items():
            flow.reset()
        for fxnname, fxn in self.fxns.items():
            fxn.reset()
    def find_classification(self,resgraph, endfaults, endflows, scen, mdlhists):
        """Placeholder for model find_classification methods (for running nominal models)"""
        return {'rate':scen['properties']['rate'], 'cost': 1, 'expected cost': 1}

class Timer():
    """class for model timers used in functions (e.g. for conditional faults) """
    def __init__(self, name, tstep=1.0):
        self.name=name
        self.time=0
    def t(self):
        """ Returns the time elapsed """
        return self.time
    def inc(self, tstep):
        """ Increments the time elapsed by tstep"""
        self.time+=tstep
    def reset(self):
        """ Resets the time to zero"""
        self.time=0

class SampleApproach():
    """
    Class for defining the sample approach to be used for a set of faults.
    
    Attributes
    ----------
    phases : dict
        phases defined in the model
    tstep : float
        timestep defined in the model
    fxnrates : dict
        overall failure rates for each function
    comprates : dict
        overall failure rates for each component
    jointmodes : list
        (if any) joint fault modes to be injected in the approach
    rates : dict
        rates of each mode (fxn, mode) in each phase, structured {fxnmode: {phase:rate}}
    sampletimes : dict
        faults to inject at each time in each phase, structured {phase:time:fnxmode}
    weights : dict
        weight to put on each time each fault was injected, structured {fxnmode:phase:time:weight}
    sampparams : dict
        parameters used to sample each mode
    scenlist : list
        list of fault scenarios (dicts of faults and properties) that fault propagation iterates through
    scenids : dict
        a list of scenario ids associated with a given fault in a given phase, structured {(fxnmode,phase):listofnames}
    """
    def __init__(self, mdl, faults='all', jointfaults={'faults':'None'}, sampparams={}, defaultsamp={'samp':'evenspacing','numpts':1}):
        """
        Initializes the sample approach for a given model

        Parameters
        ----------
        mdl : Model
            Model to sample.
        faults : str or list, optional
            List of faults (tuple (fxn, mode)) to inject in the model. The default is 'all'.
        jointfaults : dict, optional
            Defines how the approach considers joint faults. The default is {'faults':'None'}. Has structure:
                - faults : float    
                    # of joint faults to inject
                - jointfuncs :  bool 
                    determines whether more than one mode can be injected in a single function
                - pcond (optional) : float in range (0,1) 
                    conditional probabilities for joint faults. If not give, independence is assumed.
        sampparams : dict, optional
            Defines how specific modes in the model will be sampled over time. The default is {}. 
            Has structure: {(fxnmode,phase): sampparam}, where sampparam has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times, likeliest, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : quadpy quadrature
                    quadrature object if the quadrature option is selected.
        defaultsamp : TYPE, optional
            Defines how the model will be sampled over time by default. The default is {'samp':'evenspacing','numpts':1}. Has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times,likeliest, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : quadpy quadrature
                    quadrature object if the quadrature option is selected.
        """
        self.unit_factors = {'sec':1, 'min':60,'hr':360,'day':8640,'wk':604800,'month':2592000,'year':31556952}
        self.phases = mdl.phases
        self.tstep = mdl.tstep
        self.units = mdl.units
        self.init_modelist(mdl,faults, jointfaults)
        self.init_rates(mdl, jointfaults=jointfaults)
        self.create_sampletimes(sampparams, defaultsamp)
        self.create_scenarios()
    def init_modelist(self,mdl, faults, jointfaults={'faults':'None'}):
        """Initializes comprates, jointmodes internal list of modes"""
        self.comprates={}
        self._fxnmodes={}
        if faults=='all':
            self.fxnrates=dict.fromkeys(mdl.fxns)
            for fxnname, fxn in  mdl.fxns.items():
                for mode, params in fxn.faultmodes.items():
                    self._fxnmodes[fxnname, mode]=params
                self.fxnrates[fxnname]=fxn.failrate
                self.comprates[fxnname] = {compname:comp.failrate for compname, comp in fxn.components.items()}
        else:
            self.fxnrates=dict.fromkeys([fxnname for (fxnname, mode) in faults])
            for fxnname, mode in faults: 
                self._fxnmodes[fxnname, mode]=mdl.fxns[fxnname].faultmodes[mode]
                self.fxnrates[fxnname]=mdl.fxns[fxnname].failrate
                self.comprates[fxnname] = {compname:comp.failrate for compname, comp in mdl.fxns[fxnname].components}
        if type(jointfaults['faults'])==int:
            self.jointmodes=[]
            for numjoint in range(2, jointfaults['faults']+1):
                jointmodes = list(itertools.combinations(self._fxnmodes, numjoint))
                if not jointfaults.get('jointfuncs', False): 
                    jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                self.jointmodes = self.jointmodes + jointmodes
        elif type(jointfaults['faults'])==list: self.jointmodes = jointfaults['faults']
    def init_rates(self,mdl, jointfaults={'faults':'None'}):
        """ Initializes rates, rates_timeless"""
        self.rates=dict.fromkeys(self._fxnmodes)
        self.rates_timeless=dict.fromkeys(self._fxnmodes)
        for (fxnname, mode) in self._fxnmodes:
            self.rates[fxnname, mode]=dict.fromkeys(self.phases)
            self.rates_timeless[fxnname, mode]=dict.fromkeys(self.phases)
            overallrate = self.fxnrates[fxnname]
            if self.comprates[fxnname]:
                for compname, component in mdl.fxns[fxnname].components.items():
                    if mode in component.faultmodes:
                        overallrate=self.comprates[fxnname][compname]
            
            for ind, (phase, times) in enumerate(self.phases.items()):
                opp = self._fxnmodes[fxnname, mode]['oppvect'][ind]/sum(self._fxnmodes[fxnname, mode]['oppvect']) #forces to be a conditional probability
                dist = self._fxnmodes[fxnname, mode]['dist']
                if self._fxnmodes[fxnname, mode]['probtype']=='rate':      
                    dt = float(times[1]-times[0]) 
                    unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
                elif self._fxnmodes[fxnname, mode]['probtype']=='prob':    
                    dt = 1
                    unitfactor = 1
                self.rates[fxnname, mode][phase] = overallrate*opp*dist*dt*unitfactor #TODO: update with units
                self.rates_timeless[fxnname, mode][phase] = overallrate*opp*dist
                
        if getattr(self, 'jointmodes',False):
            for (j_ind, jointmode) in enumerate(self.jointmodes):
                self.rates.update({jointmode:dict.fromkeys(self.phases)})
                self.rates_timeless.update({jointmode:dict.fromkeys(self.phases)})
                for (phase, times) in self.phases.items():
                    rates=[self.rates[fmode][phase] for fmode in jointmode]
                    if not jointfaults.get('pcond', False): # if no input, assume independence
                        prob = np.prod(1-np.exp(-np.array(rates)))
                        self.rates[jointmode][phase] = -np.log(1.0-prob)
                    elif type(jointfaults['pcond'])==float:
                        self.rates[jointmode][phase] = jointfaults['pcond']*max(rates)
                    elif type(jointfaults['pcond'])==list:
                        self.rates[jointmode][phase] = jointfaults['pcond'][j_ind]*max(rates)  
                    self.rates_timeless[jointmode][phase] = self.rates[jointmode][phase]/(times[1]-times[0])          
    def create_sampletimes(self, params={}, default={'samp':'evenspacing','numpts':1}):
        """ Initializes weights and sampletimes """
        self.sampletimes=dict.fromkeys(self.phases.keys())
        self.weights={fxnmode:dict.fromkeys(rate) for fxnmode,rate in self.rates.items()}
        self.sampparams={}
        for phase, times in self.phases.items():
            possible_phasetimes = list(np.arange(times[0], times[1], self.tstep))
            for fxnmode in self.rates:
                param = params.get((fxnmode,phase), default)
                self.sampparams[fxnmode, phase] = param
                if param['samp']=='likeliest':
                    weights=[]
                    if self.rates[fxnmode][phase] == max(list(self.rates[fxnmode].values())):
                        phasetimes = [round(np.quantile(possible_phasetimes, 0.5)/self.tstep)*self.tstep]
                    else: phasetimes = []
                else: 
                    pts, weights = self.select_points(param, [pt for pt, t in enumerate(possible_phasetimes)])
                    phasetimes = [possible_phasetimes[pt] for pt in pts]
                self.add_phasetimes(fxnmode, phase, phasetimes, weights=weights)
    def select_points(self, param, possible_pts):
        """
        Selects points in the list possible_points according to a given sample strategy.

        Parameters
        ----------
        param : dict
            Sample parameter. Has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : quadpy quadrature
                    quadrature object if the quadrature option is selected.
        possible_pts : 
            list of possible points in time.

        Returns
        -------
        pts : list
            selected points
        weights : list
            weights for each point
        """
        weights=[]
        if param['samp']=='fullint': pts = possible_pts
        elif param['samp']=='evenspacing':
            if param['numpts']+2 > len(possible_pts): pts = possible_pts
            else: pts= [int(round(np.quantile(possible_pts, p/(param['numpts']+1)))) for p in range(param['numpts']+2)][1:-1]
        elif param['samp']=='quadrature':
            quantiles = param['quad'].points/2 +0.5
            if len(quantiles) > len(possible_pts): pts = possible_pts
            else: 
                pts= [int(round(np.quantile(possible_pts, q))) for q in quantiles]
                weights=param['quad'].weights/sum(param['quad'].weights)
        elif param['samp']=='randtimes':
            if param['numpts']>=len(possible_pts): pts = possible_pts
            else: pts= [possible_pts.pop(np.random.randint(len(possible_pts))) for i in range(min(param['numpts'], len(possible_pts)))]
        elif param['samp']=='symrandtimes':
            if param['numpts']>=len(possible_pts): pts = possible_pts
            else: 
                if len(possible_pts) %2 >0:  pts = [possible_pts.pop(int(np.floor(len(possible_pts)/2)))]
                else: pts = [] 
                possible_pts_halved = np.reshape(possible_pts, (2,int(len(possible_pts)/2)))
                possible_pts_halved[1] = np.flip(possible_pts_halved[1])
                possible_inds = [i for i in range(int(len(possible_pts)/2))]
                inds = [possible_inds.pop(np.random.randint(len(possible_inds))) for i in range(min(int(np.floor(param['numpts']/2)), len(possible_inds)))]
                pts= pts+ [possible_pts_halved[half][ind] for half in range(2) for ind in inds ]
                pts.sort()
        else: print("invalid option: ", param)
        if not any(weights): weights = [1/len(pts) for t in pts]
        if len(pts)!=len(set(pts)):
            raise Exception("Too many pts for quadrature at this discretization")
        return pts, weights
    def add_phasetimes(self, fxnmode, phase, phasetimes, weights=[]):
        """ Adds a set of times for a given mode to sampletimes"""
        if phasetimes:
            if not self.weights[fxnmode].get(phase): self.weights[fxnmode][phase] = {t: 1/len(phasetimes) for t in phasetimes}
            for (ind, time) in enumerate(phasetimes):
                if self.rates[fxnmode][phase]==0.0:
                    break
                if not self.sampletimes.get(phase): 
                    self.sampletimes[phase] = {time:[]}
                if self.sampletimes[phase].get(time): self.sampletimes[phase][time] = self.sampletimes[phase][time] + [fxnmode]
                else: self.sampletimes[phase][time] = [fxnmode]
                if any(weights): self.weights[fxnmode][phase][time] = weights[ind]
                else:       self.weights[fxnmode][phase][time] = 1/len(phasetimes)
    def create_nomscen(self, mdl):
        """ Creates a nominal scenario """
        nomscen={'faults':{},'properties':{}}
        for fxnname in mdl.fxns:
            nomscen['faults'][fxnname]='nom'
        nomscen['properties']['time']=0.0
        nomscen['properties']['type']='nominal'
        nomscen['properties']['name']='nominal'
        nomscen['properties']['weight']=1.0
        return nomscen
    def create_scenarios(self):
        """ Creates list of scenarios to be iterated over in fault injection. Added as scenlist and scenids """
        self.scenlist=[]
        self.times = []
        self.scenids = {}
        for phase, samples in self.sampletimes.items():
            if samples:
                for time, faultlist in samples.items():
                    self.times+=[time]
                    for fxnmode in faultlist:
                        if self.sampparams[fxnmode, phase]['samp']=='maxlike':    
                            rate = sum(self.rates[fxnmode].values())
                        else: 
                            rate = self.rates[fxnmode][phase] * self.weights[fxnmode][phase][time]
                        if type(fxnmode[0])==str:
                            name = fxnmode[0]+' '+fxnmode[1]+', t='+str(time)
                            scen={'faults':{fxnmode[0]:fxnmode[1]}, 'properties':{'type': 'single-fault', 'function': fxnmode[0],\
                                  'fault': fxnmode[1], 'rate': rate, 'time': time, 'name': name}}
                        else:
                            name = ' '.join([fm[0]+': '+fm[1]+',' for fm in fxnmode])+' t='+str(time)
                            faults = dict.fromkeys([fm[0] for fm in fxnmode])
                            for fault in faults:
                                faults[fault] = [fm[1] for fm in fxnmode if fm[0]==fault]
                            scen = {'faults':faults, 'properties':{'type': str(len(fxnmode))+'-joint-faults', 'functions':{fm[0] for fm in fxnmode}, \
                                    'modes':{fm[1] for fm in fxnmode}, 'rate': rate, 'time': time, 'name': name}}
                        self.scenlist=self.scenlist+[scen]
                        if self.scenids.get((fxnmode, phase)): self.scenids[fxnmode, phase] = self.scenids[fxnmode, phase] + [name]
                        else: self.scenids[fxnmode, phase] = [name]
        self.times.sort()
    def prune_scenarios(self,endclasses,samptype='piecewise', threshold=0.1, sampparam={'samp':'evenspacing','numpts':1}):
        """
        Finds the best sample approach to approximate the full integral (given the approach was the full integral).

        Parameters
        ----------
        endclasses : dict
            dict of results (cost, rate, expected cost) for the model run indexed by scenid 
        samptype : str ('piecewise' or 'bestpt'), optional
            Method to use. 
            If 'bestpt', finds the point in the interval that gives the average cost. 
            If 'piecewise', attempts to split the inverval into sub-intervals of continuity
            The default is 'piecewise'.
        threshold : float, optional
            If 'piecewise,' the threshold for detecting a discontinuity based on deviation from linearity. The default is 0.1.
        sampparam : float, optional
            If 'piecewise,' the sampparam sampparam to prune to. The default is {'samp':'evenspacing','numpts':1}, which would be a single point (optimal for linear).
        """
        newscenids = dict.fromkeys(self.scenids.keys())
        newsampletimes = {key:{} for key in self.sampletimes.keys()}
        newweights = {fault:dict.fromkeys(phasetimes) for fault, phasetimes in self.weights.items()}
        for modeinphase in self.scenids:
            costs= np.array([endclasses[scen]['cost'] for scen in self.scenids[modeinphase]])
            if samptype=='bestpt':
                errs = abs(np.mean(costs) - costs)
                mins = np.where(errs == errs.min())[0]
                pts=[mins[int(len(mins)/2)]]
                weights=[1]
            elif samptype=='piecewise':
                partlocs=[0, len(list(np.arange(self.phases[modeinphase[1]][0], self.phases[modeinphase[1]][1], self.tstep)))]
                reset=False
                for ind, cost in enumerate(costs[1:-1]): # find where fxn is no longer linear
                    if reset==True:
                        reset=False
                        continue
                    if abs(((cost-costs[ind]) - (costs[ind+2]-cost))/(costs[ind+2]-cost + 0.0001)) > threshold:  
                        partlocs = partlocs + [ind+2]
                        reset=True
                partlocs.sort()
                pts=[]
                weights=[]
                for (ind_part, partloc) in enumerate(partlocs[1:]): # add points in each section
                    partition = [i for i in range(partlocs[ind_part], partloc)]
                    part_pts, part_weights = self.select_points(sampparam, partition)
                    pts = pts + part_pts
                    overall_part_weight =  (partloc-partlocs[ind_part])/(partlocs[-1]-partlocs[0])
                    weights = weights + list(np.array(part_weights)*overall_part_weight)
                pts.sort()
            newscenids[modeinphase] =  [self.scenids[modeinphase][pt] for pt in pts]
            newscens = [scen for scen in self.scenlist if scen['properties']['name'] in newscenids[modeinphase]]
            newweights[modeinphase[0]][modeinphase[1]] = {scen['properties']['time']:weights[ind] for (ind, scen) in enumerate(newscens)}
            newscenids[modeinphase] =  [self.scenids[modeinphase][pt] for pt in pts]
            for newscen in newscens:
                if not newsampletimes[modeinphase[1]].get(newscen['properties']['time']):
                    newsampletimes[modeinphase[1]][newscen['properties']['time']] = [modeinphase[0]]
                else:
                    newsampletimes[modeinphase[1]][newscen['properties']['time']] = newsampletimes[modeinphase[1]][newscen['properties']['time']] + [modeinphase[0]]
        self.scenids = newscenids
        self.weights = newweights
        self.sampletimes = newsampletimes
        self.create_scenarios()
        self.sampparams={key:{'samp':'pruned '+samptype} for key in self.sampparams}
    def list_modes(self, joint=False):
        """ Returns a list of modes in the approach """
        if joint:
            return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()] + self.jointmodes
        else:
            return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()]
    def list_moderates(self):
        """ Returns the rates for each mode """
        return {(fxn, mode): sum(self.rates[fxn,mode].values()) for (fxn, mode) in self.rates.keys()}
        
    
def phases(times, names=[]):
    """ Creates named phases from a set of times defining the edges of hte intervals """
    if not names: names = range(len(times)-1)
    return {names[i]:[times[i], times[i+1]] for (i, _) in enumerate(times) if i < len(times)-1}

def m2to1(x):
    """
    Multiplies a list of numbers which may take on the values infinity or zero. In deciding if num is inf or zero, the earlier values take precedence

    Parameters
    ----------
    x : list 
        numbers to multiply

    Returns
    -------
    y : float
        result of multiplication
    """
    if np.size(x)>2:    x=[x[0], m2to1(x[1:])]
    if x[0]==np.inf:    y=np.inf
    elif x[1]==np.inf:
        if x[0]==0.0:   y=0.0
        else:           y=np.inf
    else:               y=x[0]*x[1]
    return y

def trunc(x):
    """truncates a value to 2 (useful if behavior unchanged by increases)"""
    if x>2.0:   y=2.0
    else:       y=x
    return y

def truncn(x, n):
    """truncates a value to n (useful if behavior unchanged by increases)"""
    if x>n: y=n
    else:   y=x
    return y

def union(probs):
    """ Calculates the union of a list of probabilities [p_1, p_2, ... p_n] p = p_1 U p_2 U ... U p_n """
    while len(probs)>1:
        if len(probs) % 2: 
            p, probs = probs[0], probs[1:]
            probs[0]=probs[0]+p -probs[0]*p
        probs = [probs[i-1]+probs[i]-probs[i-1]*probs[i] for i in range(1, len(probs), 2)]
    return probs[0]

def reseting_accumulate(vec):
    """ Accummulates vector for all positive output (e.g. if input =[1,1,1, 0, 1,1], output = [1,2,3,0,1,2])"""
    newvec = vec
    val=0
    for ind, i in enumerate(vec):
        if i > 0: val = i + val
        else:    val = 0
        newvec[ind] = val
    return newvec

def accumulate(vec):
    """ Accummulates vector (e.g. if input =[1,1,1, 0, 1,1], output = [1,2,3,3,4,5])"""
    return [sum(vec[:i+1]) for i in range(len(vec)) ]

    