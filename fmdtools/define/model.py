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
import time
import copy

from .flow import Flow, init_flow
from .common import check_pickleability, get_var, set_var, init_obj_attr, get_obj_track, eq_units
from .parameter import Parameter, SimParam
from .rand import Rand
from .block import Simulable
from fmdtools.analyze.result import History, get_sub_include, init_hist_iter, init_indicator_hist

#Model superclass    
class Model(Simulable):
    __slots__ =['fxns', 'functionorder', '_fxnflows', '_fxninput', '_flowstates',
                'graph', 'staticfxns', 'dynamicfxns', 'staticflows'] #added in self.build())
    default_track=('fxns', 'flows', 'i')
    default_name='model'
    """
    Model superclass used to construct the model, return representations of the model, and copy and reset the model when run.
    
    Attributes
    ----------
    flows : dict
        dictionary of flows objects in the model indexed by name
    fxns : dict
        dictionary of functions in the model indexed by name
    params : dict
        dictionaries of (optional) parameters for a given instantiation of a model
    sp : ModelParam
        Simulation Parameters.
    track : 
        dictionary of parameters for defining what simulation constructs to record for find_classification
    graph : networkx graph
        fxnflowgraph graph view of the functions and flows (fxnflowgraph)
    graph : networkx graph
        multigraph view of functions and flows
    """
    def __init__(self, name='', p={}, sp={}, r={}, track=''):
        super().__init__(name=name, p=p, sp=sp, r=r, track=track)
        
        self.fxns=dict()
        self.functionorder=OrderedSet() #set is ordered and executed in the order specified in the model
        self._fxnflows=[]
        self._fxninput={}
        self._flowstates={}
    def __repr__(self):
        fxnlist = [fxn.__repr__() for fxn in self.fxns.values()]
        fxnlist = [fstr[:115]+'...'if len(fstr)>120 else fstr for fstr in fxnlist]
        if len(fxnlist)>15: fxnlist=fxnlist[:15]+["...("+str(len(fxnlist))+' total) \n']
        fxnstr = ''.join(fxnlist)
        flowlist = [flow.__repr__()+'\n' for flow in self.flows.values()]
        flowlist = [fstr[:115]+'...\n'if len(fstr)>120 else fstr for fstr in flowlist]
        if len(flowlist)>15:  flowlist=flowlist[:15]+["...("+str(len(flowlist))+' total) \n']
        flowstr = ''.join(flowlist)
        return self.__class__.__name__+' model at '+hex(id(self))+' \n'+'FUNCTIONS: \n'+fxnstr+'FLOWS: \n'+flowstr
    def get_typename(self):
        return "Model"
    def update_seed(self,seed=[]):
        """
        Updates model seed and the seed in all functions. 

        Parameters
        ----------
        seed : int, optional
            Seed to use. The default is [].
        """
        super().update_seed(seed)
        for fxn in self.fxns:
            self.fxns[fxn].update_seed(self.r.seed)
    def get_rand_states(self, auto_update_only=False):
        """Gets dictionary of random states throughout the model functions"""
        rand_states = {}
        for fxnname, fxn in self.fxns.items():
            if fxn.get_rand_states(auto_update_only=auto_update_only): 
                rand_states[fxnname]= fxn.get_rand_states(auto_update_only=auto_update_only)
        return rand_states
    def add_flows(self, flownames, fclass=Flow, p={}, s={}):
        """
        Adds a set of flows with the same type and initial parameters

        Parameters
        ----------
        flownames : list
            Unique flow names to give the flows in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, p, s as input to __init__()
            May alternatively provide already-instanced object.
        p : dict, optional
            Parameter dictionary to instantiate the flow with
        s : dict, optional
            State dictionary to overwrite Flow default state values with
        """
        for flowname in flownames: self.add_flow(flowname, fclass, p=p, s=s)
    def add_flow(self,flowname, fclass=Flow, p={}, s={}):
        """
        Adds a flow with given attributes to the model.

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, p, s as input to __init__()
            May alternatively provide already-instanced object.
        p : dict, optional
            Parameter dictionary to instantiate the flow with
        s : dict, optional
            State dictionary to overwrite Flow default state values with
        """
        if not getattr(self, 'is_copy', False):
            self.flows[flowname] = init_flow(flowname,fclass, p=p, s=s)
    def add_fxn(self,name, fclass, *flownames, fparams='None', **fkwargs):
        """
        Instantiates a given function in the model.

        Parameters
        ----------
        name : str
            Name to give the function.
        fclass : Class
            Class to instantiate the function as. If no class has been developed,
            the user can send the block.GenericFxn class.
        flownames : list
            List of flows to associate with the function.
        fparams : dict.
            Other parameters to send to the __init__ method of the function class
        fkwargs : dict
            Parameters to send to __init__ method of the FxnBlock superclass
        """
        if not getattr(self, 'is_copy', False):
            flows=self.get_flows(flownames)
            fkwargs = {**{'r':{"seed":self.r.seed}}, **{'t':{'dt': self.sp.dt}}, **fkwargs}
            try:
                self.fxns[name] = fclass(name, flows=flows, params=fparams, **fkwargs)
            except TypeError as e:
                raise TypeError("Poorly specified class "+str(fclass)+" (or poor arguments) ") from e
            self._fxninput[name]={'name':name,'flows': flownames, 'fparams': fparams, 'kwargs': fkwargs}
            for flowname in flownames:
                self._fxnflows.append((name, flowname))
            self.functionorder.update([name])
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
        return {obj.__class__.__name__ for f, obj in self.flows.items()}
    def flows_of_type(self, ftype):
        """Returns the set of flows for each flow type"""
        return {flow for flow, obj in self.flows.items() if obj.__class__.__name__==ftype}
    def flowtypes_for_fxnclasses(self):
        """Returns the flows required by each function class in the model (as a dict)"""
        class_relationship = dict()
        for fxn, obj in self.fxns.items():
            if class_relationship.get(obj.__class__.__name__,False):
                class_relationship[obj.__class__.__name__].update(obj.get_flowtypes())
            else: class_relationship[obj.__class__.__name__] = set(obj.get_flowtypes())
        return class_relationship
    def build(self, functionorder=[], require_connections=True):
        """
        Builds the model graph after the functions have been added.

        Parameters
        ----------
        functionorder : list, optional
            The order for the functions to be executed in. The default is [].
        """
        self.update_seed()
        if not getattr(self, 'is_copy', False):
            if functionorder: self.set_functionorder(functionorder)
            self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() 
                                          if getattr(fxn, 'behavior', False) or getattr(fxn, 'static_behavior', False) 
                                          or (hasattr(fxn, 'a') and getattr(fxn.a, 'proptype','')=='static')])
            self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() 
                                           if getattr(fxn, 'dynamic_behavior', False) 
                                           or (hasattr(fxn, 'a') and getattr(fxn.a, 'proptype','')=='dynamic')])
            self.construct_graph(require_connections=require_connections)
            self.staticflows = [flow for flow in self.flows if any([ n in self.staticfxns for n in self.graph.neighbors(flow)])]
    def construct_graph(self, require_connections=True):
        """
        Creates .graph nx.graph representation of the model
        """
        self.graph=nx.Graph()
        self.graph.add_nodes_from(self.fxns, bipartite=0)
        self.graph.add_nodes_from(self.flows, bipartite=1)
        self.graph.add_edges_from(self._fxnflows)
        
        dangling_nodes = [e for e in nx.isolates(self.graph)] # check to see that all functions/flows are connected
        if dangling_nodes and require_connections: raise Exception("Fxns/flows disconnected from model: "+str(dangling_nodes))
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
            ms, mps = fxn.return_faultmodes()
            if ms: 
                modeprops[fxnname] = mps
                modes[fxnname] = ms
        return modes, modeprops
    def get_memory(self):
        """
        Returns the approximate memory usage of the model, along with a profile of fxn/flow memory usage.
        """
        mem_profile={}
        mem = 0
        mem_profile['params'] = sys.getsizeof(self.p)
        mem_profile['params'] += sys.getsizeof(self.sp)
        mem_profile['params'] += sys.getsizeof(self.track)
        for fxnname, fxn in self.fxns.items():
            mem_profile[fxnname]=fxn.get_memory()
        for flowname,flow in self.flows.items():
            mem_profile[flowname]=flow.get_memory()
        mem = np.sum([i for i in mem_profile.values()])
        return mem, mem_profile
    def new_with_params(self, **kwargs):
        """
        Creates a new Model with the same parameters as the current model but
        with changes to params (p, sp, track, rand etc.)
        """
        p, sp, r, track = super().new_params(**kwargs)
        return self.__class__(p=p, sp=sp, r=r, track=track)
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
        copy.__init__(p=getattr(self, 'p', {}),sp=getattr(self, 'sp', {}),track=getattr(self, 'track', {}), r={'seed':self.r.seed})
        for flowname, flow in self.flows.items():
            copy.flows[flowname]=flow.copy()
        for fxnname, fxn in self.fxns.items():
            flownames=self._fxninput[fxnname]['flows']
            fparams=self._fxninput[fxnname]['fparams']
            kwargs = self._fxninput[fxnname]['kwargs']
            flows = copy.get_flows(flownames)
            if fparams=='None':     
                copy.fxns[fxnname]=fxn.copy(flows, **kwargs)
            else:                   
                copy.fxns[fxnname]=fxn.copy(flows, fparams, **kwargs)
        copy._fxninput=self._fxninput
        copy._fxnflows=self._fxnflows
        copy.is_copy=False
        copy.build(functionorder = self.functionorder)
        copy.is_copy=True
        if hasattr(self, 'h'): 
            hist = History()
            for k in self.h:
                for att in ['fxns', 'flows']:
                    if k.startswith(att):
                        fname = k.split('.')[1]
                        copy_f = getattr(copy, att)[fname]
                        hist[att+'.'+fname]=copy_f.h
                if k=='time' or k.startswith('i.'):
                    hist[k]=self.h[k].copy()
            copy.h = hist.flatten()
        return copy
    def reset(self):
        """Resets the model to the initial state (with no faults, etc)"""
        for flowname, flow in self.flows.items():
            flow.reset()
        for fxnname, fxn in self.fxns.items():
            fxn.reset()
        self.r.reset()
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
                set_var(f, var, varvalues[i])
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
    def create_hist(self, timerange, track):
        if not hasattr(self, 'h'):
            hist = History()
            track = get_obj_track(self, track, all_possible=Model.default_track)
            init_indicator_hist(self, hist, timerange, track)
            fxn_track = get_sub_include('fxns', track)
            if fxn_track:
                hist['fxns'] = History()
                for fxnname, fxn in self.fxns.items():
                    fh = fxn.create_hist(timerange, get_sub_include(fxnname, fxn_track))
                    if fh: hist.fxns[fxnname]=fh
            self.add_flow_hist(hist, timerange, track)
            #if len(hist)<len(track) and track!='all': #TODO: this warning should be valid for all hists
            #    raise Exception("History doesn't match tracking options (are names correct?): \n track="+str(track)+"\n hist= \n"+str(hist))
            self.h = hist.flatten()
        return self.h
    def propagate(self, time, fxnfaults={}, disturbances={}, run_stochastic=False):
        """
        Injects and propagates faults through the graph at one time-step

        Parameters
        ----------
        time : float
            The current timestep.
        fxnfaults : dict
            Faults to inject during this propagation step. With structure {'function':['fault1', 'fault2'...]}
        disturbances : 
            Variables to change during this propagation step. With structure {'function.var1':value}
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of random states over time.
        """
        #Step 0: Update model states with disturbances
        self.set_vars(**disturbances)
        
        #Step 1: Run Dynamic Propagation Methods in Order Specified and Inject Faults if Applicable
        for fxnname in self.dynamicfxns.union(fxnfaults.keys()):
            fxn=self.fxns[fxnname]
            faults = fxnfaults.get(fxnname, [])
            if type(faults)!=list: faults=[faults]
            fxn('dynamic', faults=faults, time=time, run_stochastic=run_stochastic)
            
        #Step 2: Run Static Propagation Methods
        try:
            self.prop_static(time, run_stochastic=run_stochastic)
        except Exception as e:
            raise Exception("Error in static propagation at time t="+str(time)) 
    def prop_static(self, time, run_stochastic=False):
        """
        Propagates behaviors through model graph (static propagation step)

        Parameters
        ----------
        time : float
            Current time-step.
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of random states over time.
        """
        #set up history of flows to see if any has changed
        activefxns=self.staticfxns.copy()
        nextfxns=set()
        if not self._flowstates: 
            self._flowstates=dict.fromkeys(self.staticflows)
            for flowname in self.staticflows:
                self._flowstates[flowname]=self.flows[flowname].return_mutables()
        n=0
        while activefxns:
            flows_to_check = {*self.staticflows}
            for fxnname in list(activefxns).copy():
                #Update functions with new values, check to see if new faults or states
                oldmutables = self.fxns[fxnname].return_mutables()
                self.fxns[fxnname]('static', time=time, run_stochastic=run_stochastic)
                if oldmutables!=self.fxns[fxnname].return_mutables(): 
                    nextfxns.update([fxnname])
                
                #Check to see what flows now have new values and add connected functions (done for each because of communications potential)
                for flowname in self.fxns[fxnname].flows:
                    if flowname in flows_to_check:
                        if self._flowstates[flowname]!=self.flows[flowname].return_mutables():
                            nextfxns.update(set([n for n in self.graph.neighbors(flowname) if n in self.staticfxns]))
                            flows_to_check.remove(flowname)
            # check remaining flows that have not been checked already
            for flowname in flows_to_check:
                if self._flowstates[flowname]!=self.flows[flowname].return_mutables():
                    nextfxns.update(set([n for n in self.graph.neighbors(flowname) if n in self.staticfxns]))
            # update flowstates
            for flowname in self.staticflows:
                self._flowstates[flowname]=self.flows[flowname].return_mutables()
            activefxns=nextfxns.copy()
            nextfxns.clear()
            n+=1
            if n>1000: #break if this is going for too long
                raise Exception("Undesired looping between functions in static propagation step",
                                "at t="+str(time)+", these functions remain active:"+str(activefxns))
    def get_scen_rate(self, fxnname, faultmode, time):
        return self.fxns[fxnname].get_scen_rate(faultmode, time)
        
def check_model_pickleability(model, try_pick=False):
    """ Checks to see which attributes of a model object will pickle, providing more detail about functions/flows"""
    print('FLOWS ')
    for flowname, flow in model.flows.items():
        print(flowname)
        check_pickleability(flow, try_pick=try_pick)
    print('FUNCTIONS ')
    for fxnname, fxn in model.fxns.items():
        print(fxnname)
        check_pickleability(fxn, try_pick=try_pick)
    time.sleep(0.2)
    print('MODEL')
    unpickleable = check_pickleability(model, try_pick=try_pick)


        