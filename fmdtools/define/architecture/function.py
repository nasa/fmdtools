# -*- coding: utf-8 -*-
"""
Description: A module for defining Models, which are aggregations of Functions and Flows.

Has Classes and Functions:

- :class:`Model`:                       Superclass for defining simulation models.

- :func:`check_model_pickleability` :   Checks if a model is pickleable (and thus able to be parallelized)
"""
import numpy as np
from ordered_set import OrderedSet
import networkx as nx
import sys
import time
import copy

from fmdtools.define.flow.base import Flow, init_flow
from fmdtools.define.base import check_pickleability, set_var, get_obj_track
from fmdtools.define.block.base import Simulable
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist

#Model superclass    
class FunctionArchitecture(Simulable):
    __slots__ =['fxns', 'functionorder', '_fxnflows', '_fxninput', '_flowstates',
                'graph', 'staticfxns', 'dynamicfxns', 'staticflows']
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

    def add_flows(self, flownames, fclass=Flow, **kwargs):
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
        kwargs: kwargs
            Dicts for non-default values to p, s, etc
        """
        for flowname in flownames:
            self.add_flow(flowname, fclass, **kwargs)

    def add_flow(self, flowname, fclass=Flow, **kwargs):
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
        kwargs: kwargs
            Dicts for non-default values to p, s, etc
        """
        if not getattr(self, 'is_copy', False):
            self.flows[flowname] = init_flow(flowname, fclass, **kwargs)

    def add_fxn(self, name, fclass, *flownames, args_f='None', **fkwargs):
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
        args_f : dict.
            Other parameters to send to the __init__ method of the function class
        fkwargs : dict
            Parameters to send to __init__ method of the Function superclass
        """
        if not getattr(self, 'is_copy', False):
            flows=self.get_flows(flownames)
            fkwargs = {**{'r':{"seed":self.r.seed}}, **{'t':{'dt': self.sp.dt}}, **fkwargs}
            try:
                self.fxns[name] = fclass(name, flows=flows, args_f=args_f, **fkwargs)
            except TypeError as e:
                raise TypeError("Poorly specified class "+str(fclass)+" (or poor arguments) ") from e
            self._fxninput[name]={'name':name, 'flows': flownames, 'args_f': args_f, 'kwargs': fkwargs}
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
    def build(self, functionorder=[], require_connections=True, update_seed=True):
        """
        Builds the model graph after the functions have been added.

        Parameters
        ----------
        functionorder : list, optional
            The order for the functions to be executed in. The default is [].
        """
        if not getattr(self, 'is_copy', False):
            if update_seed:
                self.update_seed()
            if functionorder:
                self.set_functionorder(functionorder)
            self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() 
                                          if fxn.is_static()])
            self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() 
                                           if fxn.is_dynamic()])
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
        modecost = sum([ c['cost'] if c['cost']>0.0 else default_cost for m in modeprops.values() for c in m.values()])
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

    def copy(self):
        """
        Copies the model at the current state.

        Returns
        -------
        copy : Model
            Copy of the curent model.
        """
        cop = self.__new__(self.__class__)  # Is this adequate? Wouldn't this give it new components?
        cop.is_copy = True
        cop.__init__(p=getattr(self, 'p', {}),
                     sp=getattr(self, 'sp', {}),
                     track=getattr(self, 'track', {}),
                     r = {'seed': self.r.seed})
        cop.r.assign(self.r)

        for flowname, flow in self.flows.items():
            cop.flows[flowname] = flow.copy()

        for fxnname, fxn in self.fxns.items():
            flownames = copy.deepcopy(self._fxninput[fxnname]['flows'])
            args_f = copy.deepcopy(self._fxninput[fxnname]['args_f'])
            kwargs = copy.deepcopy(self._fxninput[fxnname]['kwargs'])
            flows = cop.get_flows(flownames)
            if args_f == 'None':
                cop.fxns[fxnname] = fxn.copy(flows, **kwargs)
            else:
                cop.fxns[fxnname] = fxn.copy(flows, args_f, **kwargs)

        cop._fxninput = copy.deepcopy(self._fxninput)
        cop._fxnflows = copy.deepcopy(self._fxnflows)
        cop._flowstates = copy.deepcopy(self._flowstates)

        cop.is_copy = False
        cop.build(functionorder=copy.deepcopy(self.functionorder), update_seed=False)
        cop.is_copy = True
        if hasattr(self, 'h'):
            hist = History()
            for k in self.h:
                for att in ['fxns', 'flows']:
                    if k.startswith(att):
                        fname = k.split('.')[1]
                        copy_f = getattr(cop, att)[fname]
                        hist[att+'.'+fname] = copy_f.h
                if k == 'time' or k.startswith('i.'):
                    hist[k] = self.h[k].copy()
            cop.h = hist.flatten()
        return cop

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
            probdens *= fxn.return_probdens()
        return probdens
    def set_vars(self, *args, **kwargs):
        """
        Sets variables in the model to set values (useful for optimization, etc.)

        Parameters
        ----------
        varlist : list of lists/tuples
            List of variables to set, with possible structures:
                [['fxnname', 'att1'], ['fxnname2', 'comp1', 'att2'], ['flowname', 'att3']]
                ['fxnname.att1', 'fxnname.comp1.att2', 'flowname.att3']
        varvalues : list
            List of values corresponding to varlist
        kwargs : kwargs
            attribute-value pairs. If provided, must be passed using ** syntax:
            mdl.set_vars(**{'fxnname.varname':value})
        """
        if len(args) > 0: 
            varlist = args[0]
            varvalues = args[1]
            if type(varlist) == str:
                varlist = [varlist]
            if type(varvalues) in [str, float, int]:
                varvalues = [varvalues]
            if len(varlist) != len(varvalues):
                raise Exception("length of varlist and varvalues do not correspond: "
                                + str(len(varlist)) + ", "+str(len(varvalues)))
        else:
            varlist=[]
            varvalues=[]
        if kwargs:
            varlist = varlist + [*kwargs.keys()]
            varvalues = varvalues + [*kwargs.values()]
        for i, var in enumerate(varlist):
            if var == 'seed':
                self.update_seed(seed=varvalues[i])
            else:
                if type(var) == str:
                    var = var.split(".")
                if var[0] in ['functions', 'fxns']:
                    f = self.fxns[var[1]]
                    var = var[2:]
                elif var[0] == 'flows':
                    f = self.flows[var[1]]
                    var = var[2:]
                elif var[0] in self.fxns:
                    f = self.fxns[var[0]]
                    var = var[1:]
                elif var[0] in self.flows:
                    f = self.flows[var[0]]
                    var = var[1:]
                else:
                    raise Exception(var[0] + " not a function, flow, or seed")
                set_var(f, var, varvalues[i])

    def create_hist(self, timerange, track):
        if not hasattr(self, 'h'):
            hist = History()
            track = get_obj_track(self, track,
                                  all_possible=FunctionArchitecture.default_track)
            init_indicator_hist(self, hist, timerange, track)
            fxn_track = get_sub_include('fxns', track)
            if fxn_track:
                hist['fxns'] = History()
                for fxnname, fxn in self.fxns.items():
                    fh = fxn.create_hist(timerange, get_sub_include(fxnname, fxn_track))
                    if fh:
                        hist.fxns[fxnname] = fh
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
            fxn = self.fxns[fxnname]
            faults = fxnfaults.get(fxnname, [])
            if type(faults) != list:
                faults = [faults]
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
                        try:
                            if self._flowstates[flowname]!=self.flows[flowname].return_mutables():
                                nextfxns.update(set([n for n in self.graph.neighbors(flowname) if n in self.staticfxns]))
                                flows_to_check.remove(flowname)
                        except ValueError as e:
                            raise Exception("Invalid mutables in flow: "+flowname) from e
            # check remaining flows that have not been checked already
            for flowname in flows_to_check:
                if self._flowstates[flowname]!=self.flows[flowname].return_mutables():
                    nextfxns.update(set([n for n in self.graph.neighbors(flowname) if n in self.staticfxns]))
            # update flowstates
            for flowname in self.staticflows:
                self._flowstates[flowname]=self.flows[flowname].return_mutables()
            activefxns=nextfxns.copy()
            nextfxns.clear()
            n += 1
            if n > 1000: #break if this is going for too long
                raise Exception("Undesired looping between functions in static propagation step",
                                "at t=" + str(time) + ", these functions remain active:" + str(activefxns))

    def plot_dynamic_run_order(self, rotateticks=False, title="Dynamic Run Order"):
        """
        Plot the run order for the model during the dynamic propagation step.

        The x-direction is the order of each function executed and the y are the
        corresponding flows acted on by the given methods.

        Parameters
        ----------
        rotateticks : Bool, optional
            Whether to rotate the x-ticks (for bigger plots). The default is False.
        title : str, optional
            String to use for the title (if any). The default is "Dynamic Run Order".

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        from matplotlib import pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.ticker import AutoMinorLocator
        fxnorder = list(self.dynamicfxns)
        times = [i+0.5 for i in range(len(fxnorder))]
        fxntimes = {f: i for i, f in enumerate(fxnorder)}

        flowtimes = {f: [fxntimes[n] for n in self.graph.neighbors(
            f) if n in self.dynamicfxns] for f in self.flows}

        lengthorder = {k: v for k, v in
                       sorted(flowtimes.items(), key=lambda x: len(x[1]), reverse=True)
                       if len(v) > 0}
        starttimeorder = {k: v for k, v in sorted(lengthorder.items(),
                                                  key=lambda x: x[1][0], reverse=True)}
        endtimeorder = [k for k, v in sorted(starttimeorder.items(),
                                             key=lambda x: x[1][-1], reverse=True)]
        flowtimedict = {flow: i for i, flow in enumerate(endtimeorder)}

        fig, ax = plt.subplots()

        for flow in flowtimes:
            phaseboxes = [((t, flowtimedict[flow]-0.5),
                           (t, flowtimedict[flow]+0.5),
                           (t+1.0, flowtimedict[flow]+0.5),
                           (t+1.0, flowtimedict[flow]-0.5))
                          for t in flowtimes[flow]]
            bars = PolyCollection(phaseboxes)
            ax.add_collection(bars)

        flowtimes = [i+0.5 for i in range(len(self.flows))]
        ax.set_yticks(list(flowtimedict.values()))
        ax.set_yticklabels(list(flowtimedict.keys()))
        ax.set_ylim(-0.5, len(flowtimes)-0.5)
        ax.set_xticks(times)
        ax.set_xticklabels(fxnorder, rotation=90*rotateticks)
        ax.set_xlim(0, len(times))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which='minor',  linewidth=2)
        ax.tick_params(axis='x', bottom=False, top=False,
                       labelbottom=False, labeltop=True)
        if title:
            if rotateticks:
                fig.suptitle(title, fontweight='bold', y=1.15)
            else:
                fig.suptitle(title, fontweight='bold')
        return fig, ax


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


        
