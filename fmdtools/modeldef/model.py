# -*- coding: utf-8 -*-
"""
Description: A module for defining Models, which are aggregations of Functions and Flows. Has Classes:
    
- :class:`Model`:                       Superclass for defining simulation models.
- :class:`ModelParam`:                  Class for defining model simulation parameters.
- :func:`check_model_pickleability` :   Checks if a model is pickleable (and thus able to be parallelized)
"""
import numpy as np
from ordered_set import OrderedSet
from inspect import signature
import networkx as nx
import warnings
import sys
from recordclass import asdict

from .flow import Flow, init_flow
from .block import GenericFxn
from .common import check_pickleability, Parameter, get_var
import fmdtools.resultdisp.process as proc



class ModelParam(Parameter, readonly=True):
    """
    Class defining Model simulation parameters.
    
    Has fields:
        phases : tuple
            phases (('name', start, end)...) that the simulation progresses through
        times : tuple
            tuple of times to sample (if desired) (starttime, sampletime1, sampletime2,... endtime)
        dt : float
            timestep used in the simulation. default is 1.0
        units : str
            time-units. default is hours`
        use_end_condition : bool
            whether to use an end-condition method (defined by user-defined end_condition method) 
            or defined end time to end the simulation. Default is False
        seed : int
            seed used for the internal random number generator. Default is 42.
        use_local : bool
            Whether to use locally-defined timesteps in functions (if any). Default is True.
    """
    phases :            tuple = (('na', 0, 100),)
    times :             tuple = (0, 100)
    dt :                float = 1.0
    units :             str = "hr"
    units_set = ('sec', 'min', 'hr', 'day', 'wk', 'month', 'year')
    use_end_condition : bool = False
    seed :              int = 42
    use_local :         bool = True
    def __init__(self, *args, **kwargs):
        if ('times' in kwargs) and not('phases' in kwargs):
            kwargs['phases']=(("na", 0, kwargs['times'][-1]),)
        super().__init__(*args, **kwargs)

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
        dictionaries of (optional) parameters for a given instantiation of a model
    modelparams : ModelParam
        Simulation Parameters.
    valparams : 
        dictionary of parameters for defining what simulation constructs to record for find_classification
    bipartite : networkx graph
        bipartite graph view of the functions and flows
    graph : networkx graph
        multigraph view of functions and flows
    
    """
    def __init__(self, params={},modelparams=ModelParam(), valparams='all'):
        """
        Instantiates internal model attributes with predetermined:
        
        Parameters
        ----------
        params : dict 
            design variables of the model
        modelparams : ModelParam
            simulation parameters for the model.
        valparams dict or (`all`/`flows`/`fxns`)
            parameters to keep a history of in params needed for find_classification. default is 'all'
            dict option is of the form of mdlhist {fxns:{fxn1:{param1}}, flows:{flow1:{param1}}})
        """
        self.type='model'
        self.flows={}
        self.fxns={}
        self.params=params
        self.valparams = valparams
        self.modelparams=modelparams
        self.find_any_phase_overlap()
        self.set_rng()
        self.functionorder=OrderedSet() #set is ordered and executed in the order specified in the model
        self._fxnflows=[]
        self._fxninput={}
    def __repr__(self):
        fxnlist = ['- '+fxnname+':'+str(fxn.return_states())+' '+str(getattr(fxn,'active_actions',''))+'\n' for fxnname,fxn in self.fxns.items()]
        fxnlist = [fstr[:115]+'...\n'if len(fstr)>120 else fstr for fstr in fxnlist]
        if len(fxnlist)>15: fxnlist=fxnlist[:15]+["...("+str(len(fxnlist))+' total) \n']
        fxnstr = ''.join(fxnlist)
        flowlist = ['- '+flowname+':'+str(flow.status())+'\n' for flowname,flow in self.flows.items()]
        flowlist = [fstr[:115]+'...\n'if len(fstr)>120 else fstr for fstr in flowlist]
        if len(flowlist)>15:  flowlist=flowlist[:15]+["...("+str(len(flowlist))+' total) \n']
        flowstr = ''.join(flowlist)
        return self.__class__.__name__+' model at '+hex(id(self))+' \n'+'functions: \n'+fxnstr+'flows: \n'+flowstr
    def find_any_phase_overlap(self):
        phase_dict = {v[0]: [v[1], v[2]] for v in self.modelparams.phases}
        intervals = [*phase_dict.values()]
        int_low = np.sort([i[0] for i in intervals])
        int_high = np.sort([i[1] if len(i)==2 else i[0] for i in intervals])
        for i, il in enumerate(int_low):
            if i+1==len(int_low): break
            if int_low[i+1]<=int_high[i]:
                raise Exception("Global phases overlap (see mdlparams):"+str(self.modelparams.phases)+" Ensure the max of each phase < min of each other phase")
    def _update_model_seed(self, seed=[]):
        """ Updates/Initializes the model seed params (helper function--use update_seed instead)""" 
        if not seed:
            seed=np.random.SeedSequence.generate_state(np.random.SeedSequence(),1)[0] 
        kwargs = asdict(self.modelparams)
        kwargs['seed']=seed
        self.modelparams = ModelParam(**kwargs)
        self.set_rng()
    def set_rng(self):
        """Sets the Model internal rng self._rng from self.modelparams.seed."""
        self._rng = np.random.default_rng(self.modelparams.seed)
    def update_seed(self,seed=[]):
        """
        Updates model seed and the seed in all functions. 

        Parameters
        ----------
        seed : int, optional
            Seed to use. The default is [], which uplls from np.random.SeedSequence
        """
        self._update_model_seed(seed)
        for fxn in self.fxns:
            self.fxns[fxn].update_seed(self.modelparams.seed)
    def get_rand_states(self, auto_update_only=False):
        """Gets dictionary of random states throughout the model functions"""
        rand_states = {}
        for fxnname, fxn in self.fxns.items():
            if fxn.get_rand_states(auto_update_only=auto_update_only): 
                rand_states[fxnname]= fxn.get_rand_states(auto_update_only=auto_update_only)
        return rand_states
    def add_flows(self, flownames, flowdict={}, flowtype='generic', fclass=Flow, params={}):
        """
        Adds a set of flows with the same type and initial parameters

        Parameters
        ----------
        flownames : list
            Unique flow names to give the flows in the model
        flowdict : dict, Flow, set or empty set
            Dictionary of flow attributes e.g. {'value':XX}, or an already instantiated Flow object.
            If a set of attribute names is provided, each will be given a value of 1
            If an empty set is given, it will be represented w- {flowname: 1}
        flowtype : str, optional
            Denotes type for class (e.g. 'energy,' 'material,', 'signal')
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, flowdict, flowtype as input to __init__()
        params : dict, optional
            Parameter dictionary to instantiate the flow with
        """
        for flowname in flownames: self.add_flow(flowname, flowdict, flowtype, fclass, params)
    def add_flow(self,flowname, fclass=Flow, p={}, s={}, flowtype=''):
        """
        Adds a flow with given attributes to the model.

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, flowdict, flowtype as input to __init__()
            May alternatively provide already-instanced object.
        p : dict, optional
            Parameter dictionary to instantiate the flow with
        s : dict, optional
            State dictionary to overwrite Flow default state values with
        flowtype : str, optional
            Denotes type for class (e.g. 'energy,' 'material,', 'signal')
        """
        if not getattr(self, 'is_copy', False):
            self.flows[flowname] = init_flow(flowname,fclass, p=p, s=s, flowtype=flowtype)
    def add_fxn(self,name, flownames, fclass=GenericFxn, fparams='None', **fkwargs):
        """
        Instantiates a given function in the model.

        Parameters
        ----------
        name : str
            Name to give the function.
        flownames : list
            List of flows to associate with the function.
        fclass : Class
            Class to instantiate the function as.
        fparams : dict.
            Other parameters to send to the __init__ method of the function class
        fkwargs : dict
            Parameters to send to __init__ method of the FxnBlock superclass
        """
        if not getattr(self, 'is_copy', False):
            flows=self.get_flows(flownames)
            fkwargs = {**{'r':{"seed":self.modelparams.seed}}, **fkwargs}
            try:
                self.fxns[name] = fclass(name, flows=flows, params=fparams, **fkwargs)
            except TypeError as e:
                raise TypeError("Poorly specified class "+str(fclass)+" (or poor arguments) ") from e
            self._fxninput[name]={'name':name,'flows': flownames, 'fparams': fparams, 'kwargs': fkwargs}
            for flowname in flownames:
                self._fxnflows.append((name, flowname))
            self.functionorder.update([name])
            self.fxns[name].set_timestep(use_local=self.modelparams.use_local, global_tstep=self.modelparams.dt)
    def set_functionorder(self,functionorder):
        """Manually sets the order of functions to be executed (otherwise it will be executed based on the sequence of add_fxn calls)"""
        if not self.functionorder.difference(functionorder): self.functionorder=OrderedSet(functionorder)
        else:                                       raise Exception("Invalid list: "+str(functionorder)+" should have elements: "+str(self.functionorder))
    def get_flows(self,flownames):
        """ Returns a list of the model flow objects """
        return {flowname:self.flows[flowname] for flowname in flownames}
    def fxns_of_class(self, ftype):
        """Returns dict of functionname:functionobjects corresponding to the given class name ftype"""
        return {fxn:obj for fxn, obj in self.fxns.items() if obj.__class__.__name__==ftype}
    def fxnclasses(self):
        """Returns the set of class names used in the model"""
        return {obj.__class__.__name__ for fxn, obj in self.fxns.items()}
    def flowtypes(self):
        """Returns the set of flow types used in the model"""
        return {obj.type for fxn, obj in self.flows.items()}
    def flows_of_type(self, ftype):
        """Returns the set of flows for each flow type"""
        return {flow for flow, obj in self.flows.items() if obj.type==ftype}
    def flowtypes_for_fxnclasses(self):
        """Returns the flows required by each function class in the model (as a dict)"""
        class_relationship = dict()
        for fxn, obj in self.fxns.items():
            if class_relationship.get(obj.__class__.__name__,False):
                class_relationship[obj.__class__.__name__].update(obj.get_flowtypes())
            else: class_relationship[obj.__class__.__name__] = set(obj.get_flowtypes())
        return class_relationship
    def build_model(self, functionorder=[], graph_pos={}, bipartite_pos={}, require_connections=True):
        """
        Builds the model graph after the functions have been added.

        Parameters
        ----------
        functionorder : list, optional
            The order for the functions to be executed in. The default is [].
        graph_pos : dict, optional
            position of graph nodes. The default is {}.
        bipartite_pos : dict, optional
            position of bipartite graph nodes. The default is {}.
        """
        if not getattr(self, 'is_copy', False):
            if functionorder: self.set_functionorder(functionorder)
            self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() 
                                          if getattr(fxn, 'behavior', False) or getattr(fxn, 'static_behavior', False) 
                                          or (hasattr(fxn, 'a') and getattr(fxn.a, 'proptype','')=='static')])
            self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() 
                                           if getattr(fxn, 'dynamic_behavior', False) 
                                           or (hasattr(fxn, 'a') and getattr(fxn.a, 'proptype','')=='dynamic')])
            self.construct_graph(graph_pos, bipartite_pos, require_connections=require_connections)
            self.staticflows = [flow for flow in self.flows if any([ n in self.staticfxns for n in self.bipartite.neighbors(flow)])]
    def construct_graph(self, graph_pos={}, bipartite_pos={}, require_connections=True):
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
        
        dangling_nodes = [e for e in nx.isolates(self.bipartite)] # check to see that all functions/flows are connected
        if dangling_nodes and require_connections: raise Exception("Fxns/flows disconnected from model: "+str(dangling_nodes))
        
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
        self.graph_pos=graph_pos
        self.bipartite_pos=bipartite_pos
        return self.graph
    def return_typegraph(self, withflows = True):
        """
        Returns a graph with the type containment relationships of the different model constructs.

        Parameters
        ----------
        withflows : bool, optional
            Whether to include flows. The default is True.

        Returns
        -------
        g : nx.DiGraph
            networkx directed graph of the type relationships
        """
        g = nx.DiGraph()
        modelname = type(self).__name__
        g.add_node(modelname, level=1)
        g.add_nodes_from(self.fxnclasses(), level=2)
        function_connections = [(modelname, fname) for fname in self.fxnclasses()]
        g.add_edges_from(function_connections)
        if withflows:
            g.add_nodes_from(self.flowtypes(), level=3)
            fxnclass_flowtype = self.flowtypes_for_fxnclasses()
            flow_edges = [(fxn, flow) for fxn, flows in fxnclass_flowtype.items() for flow in flows]
            g.add_edges_from(flow_edges)
        return g
    def return_paramgraph(self):
        """ Returns a graph representation of the flows in the model, where flows are nodes and edges are 
        associations in functions """
        return nx.projected_graph(self.bipartite, self.flows)
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
    def return_stategraph(self, gtype='bipartite'):
        """
        Returns a graph representation of the current state of the model.

        Parameters
        ----------
        gtype : str/dict, optional
            Type of graph to return (normal, bipartite, component, or typegraph). The default is 'bipartite'.
            dict: for function/flowgraphs, a dict with {flow:**kwargs} will return the graph view corresponding 
            to that function/flow

        Returns
        -------
        graph : networkx graph
            Graph representation of the system with the modes and states added as attributes.
        """
        if  gtype==None: return None
        elif gtype=='normal':
            graph=nx.projected_graph(self.bipartite, self.fxns)
        elif gtype=='bipartite':
            graph=self.bipartite.copy()
        elif gtype=='component':
            graph=self.bipartite.copy()
            for fxnname, fxn in self.fxns.items():
                if {**fxn.components, **fxn.actions}: 
                    graph.add_nodes_from({**fxn.components, **fxn.actions}, bipartite=1)
                    graph.add_edges_from([(fxnname, comp) for comp in {**fxn.components, **fxn.actions}])
        elif gtype=='typegraph':
            graph=self.return_typegraph()
            
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
        elif gtype=='typegraph':
            for flowtype in self.flowtypes():
                flowstates[flowtype] = {flow:self.flows[flow].status() for flow in self.flows_of_type(flowtype)}
            nx.set_node_attributes(graph, flowstates, 'states')
        #set node values for functions
        if gtype=='typegraph':
            for fxnclass in self.fxnclasses(): 
                fxnstates[fxnclass] = {fxn:self.fxns[fxn].return_states()[0] for fxn in self.fxns_of_class(fxnclass)}
                fxnmodes[fxnclass] = {fxn:self.fxns[fxn].return_states()[1] for fxn in self.fxns_of_class(fxnclass)}
        else:
            for fxnname, fxn in self.fxns.items():
                fxnstates[fxnname], fxnmodes[fxnname] = fxn.return_states()
                if gtype=='normal': del graph.nodes[fxnname]['bipartite']
                if gtype=='component':
                    for mode in fxnmodes[fxnname].copy():
                        for compname, comp in {**fxn.actions, **fxn.components}.items():
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
    def calc_repaircost(self, additional_cost=0, default_cost=0, max_cost=np.inf):
        """
        Calculates the repair cost of the fault modes in the model based on given
        mode cost information for each function mode (in fxn.assoc_faultmodes).

        Parameters
        ----------
        additional_cost : int/float
            Additional cost to add if there are faults in the model. Default is 0.
        default_cost : int/float
            Cost to use for each fault mode if no fault cost information given 
            in assoc_faultmodes/ Default is 0.
        max_cost : int/float
            Maximum cost of repair (e.g. cost of replacement). Default is np.inf

        Returns
        -------
        repair_cost : float
            Cost of repairing the fault modes in the given model

        """
        repmodes, modeprops = self.return_faultmodes()
        modecost = sum([ c['rcost'] if c['rcost']>0.0 else default_cost for m in modeprops.values() for c in m.values()])
        repair_cost = np.min([modecost, max_cost])
        return repair_cost
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
            ms = [m for m in fxn.m.faults.copy() if m!='nom']
            if ms: 
                modeprops[fxnname] = {}
                modes[fxnname] = ms
            for mode in ms:
                if mode!='nom': 
                    modeprops[fxnname][mode] = fxn.m.faultmodes.get(mode)
                    if mode not in fxn.m.faultmodes: 
                        raise Exception("Mode "+mode+" not in m.faultmodes for fxn "+fxnname+" and may not be tracked.")
        return modes, modeprops
    def get_memory(self):
        """
        Returns the approximate memory usage of the model, along with a profile of fxn/flow memory usage.
        """
        mem_profile={}
        mem = 0
        mem_profile['params'] = sys.getsizeof(self.params)
        mem_profile['params'] += sys.getsizeof(self.modelparams)
        mem_profile['params'] += sys.getsizeof(self.valparams)
        for fxnname, fxn in self.fxns.items():
            mem_profile[fxnname]=fxn.get_memory()
        for flowname,flow in self.flows.items():
            mem_profile[flowname]=flow.get_memory()
        mem = np.sum([i for i in mem_profile.values()])
        return mem, mem_profile
    def copy(self):
        """
        Copies the model at the current state.

        Returns
        -------
        copy : Model
            Copy of the curent model.
        """
        copy = self.__new__(self.__class__)  # Is this adequate? Wouldn't this give it new components?
        copy.is_copy=True
        copy.__init__(params=getattr(self, 'params', {}),modelparams=getattr(self, 'modelparams', {}),valparams=getattr(self, 'valparams', {}))
        for flowname, flow in self.flows.items():
            copy.flows[flowname]=flow.copy()
            setattr(copy, flowname, copy.flows[flowname])
        for fxnname, fxn in self.fxns.items():
            flownames=self._fxninput[fxnname]['flows']
            fparams=self._fxninput[fxnname]['fparams']
            kwargs = self._fxninput[fxnname]['kwargs']
            flows = copy.get_flows(flownames)
            if fparams=='None':     
                copy.fxns[fxnname]=fxn.copy(flows, **kwargs)
            else:                   
                copy.fxns[fxnname]=fxn.copy(flows, fparams, **kwargs)
            copy.fxns[fxnname].set_timestep(use_local=self.modelparams.use_local, global_tstep=self.modelparams.dt)
            setattr(copy, fxnname, copy.fxns[fxnname])
        copy._fxninput=self._fxninput
        copy._fxnflows=self._fxnflows
        copy.is_copy=False
        copy.build_model(functionorder = self.functionorder, graph_pos=self.graph_pos, bipartite_pos=self.bipartite_pos)
        copy.is_copy=True
        return copy
    def reset(self):
        """Resets the model to the initial state (with no faults, etc)"""
        for flowname, flow in self.flows.items():
            flow.reset()
        for fxnname, fxn in self.fxns.items():
            fxn.reset()
        self._rng=np.random.default_rng(self.modelparams.seed)
    def find_sub_classifications(self, scen, mdlhists):
        """Enables the use of find_classification at the function level."""
        subclass={}
        for fxnname, fxn in self.fxns.items():
            if hasattr(fxn, "find_classification"):
                fhists = proc.create_fxnhist_view(mdlhists, fxnname)
                sc = fxn.find_classification(scen, fhists)
                if sc: subclass[fxnname] = sc
        return subclass
    def find_classification(self, scen, mdlhists):
        """Placeholder for model find_classification methods (for running nominal models)"""
        return {'rate':scen['properties'].get('rate', 0), 'cost': 1, 'expected cost': scen['properties'].get('rate',0)}
    def return_probdens(self):
        """Returns the probability desnity of the model distributions given a """
        probdens=1.0
        for fxn in self.fxns.values():
            probdens *= getattr(fxn, 'probdens', 1.0)
        return probdens
    def set_vars(self, *args, **kwargs):
        """
        Sets variables in the model to set values (useful for optimization, etc.)

        Parameters
        ----------
        varlist : list of lists/tuples
            List of variables to set, with possible structures:
                [['fxnname', 'att1'], ['fxnname2', 'comp1','att2'], ['flowname', 'att3']]
                ['fxnname.att1', 'fxnname.comp1.att2', 'flowname.att3']
        varvalues : list
            List of values corresponding to varlist
        kwargs : kwargs
            attribute-value pairs. If provided, must be passed using ** syntax:
            mdl.set_vars(**{'fxnname.varname':value})
        """
        if len(args)>0: 
            varlist=args[0]; varvalues=args[1]
            if type(varlist)==str:                      varlist = [varlist]
            if type(varvalues) in [str, float, int]:    varvalues= [varvalues]
            if len(varlist)!=len(varvalues): raise Exception("length of varlist and varvalues do not correspond: "+str(len(varlist))+", "+str(len(varvalues)))
        else: varlist=[]; varvalues=[]
        if kwargs: varlist = varlist+[*kwargs.keys()]; varvalues = varvalues + [*kwargs.values()]
        for i,var in enumerate(varlist):
            if var=='seed':  self.update_seed(seed=varvalues[i])
            else:
                if type(var)==str: var=var.split(".")             
                if var[0] in ['functions', 'fxns']: f=self.fxns[var[1]]; var=var[2:]
                elif var[0]=='flows':               f=self.flows[var[1]]; var=var[2:]
                elif var[0] in self.fxns:           f=self.fxns[var[0]]; var=var[1:]
                elif var[0] in self.flows:          f=self.flows[var[0]]; var=var[1:]             
                else: raise Exception(var[0]+" not a function, flow, or seed")
                f.set_var(var, varvalues[i])
    def get_vars(self, *variables, trunc_tuple=True):
        """
        Gets variable values in the model.

        Parameters
        ----------
        *variables : list/string
            Variables to get from the model. Can be specifid as: 
            a list ['fxnname2', 'comp1','att2'], or
            a str 'fxnname.comp1.att2'

        Returns
        -------
        variable_values: tuple 
            Values of variables. Passes (non-tuple) single value if only one variable.
        """
        if type(variables)==str:                      variables = [variables]
        variable_values = [None]*len(variables)
        for i, var in enumerate(variables):
            if type(var)==str: var=var.split(".")
            if var[0] in ['functions', 'fxns']: f=self.fxns[var[1]]; var=var[2:]
            elif var[0]=='flows':               f=self.flows[var[1]]; var=var[2:]
            elif var[0] in self.fxns:           f=self.fxns[var[0]]; var=var[1:]
            elif var[0] in self.flows:          f=self.flows[var[0]]; var=var[1:]
            else: raise Exception(var[0]+" not a function or flow")
            variable_values[i]=get_var(f, var)
        if len(variable_values)==1 and trunc_tuple: return variable_values[0]
        else:                                       return tuple(variable_values)

def check_model_pickleability(model):
    """ Checks to see which attributes of a model object will pickle, providing more detail about functions/flows"""
    unpickleable = check_pickleability(model)
    if 'flows' in unpickleable:
        print('FLOWS ')
        for flowname, flow in model.flows.items():
            print(flowname)
            check_pickleability(flow)
    if 'fxns' in unpickleable:
        print('FUNCTIONS ')
        for fxnname, fxn in model.fxns.items():
            print(fxnname)
            check_pickleability(fxn)