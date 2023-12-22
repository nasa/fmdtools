# -*- coding: utf-8 -*-
"""
Description: A module for defining Functional Architectures.

Has Classes and Functions:

- :class:`FunctionArchitecture`:  Superclass for functional architectures.
"""
import numpy as np
from ordered_set import OrderedSet
import networkx as nx
import sys

from fmdtools.define.base import set_var
from fmdtools.define.architecture.base import Architecture
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History


class FunctionArchitecture(Architecture):
    """
    Class representing a functional architecture.

    Functional architectures enable the execution of multiple Function objects
    interacting with each other over time.

    Flexible Roles
    --------------
    flows : dict
        dictionary of flows objects in the model indexed by name
    fxns : dict
        dictionary of functions in the model indexed by name

    Special Attributes
    ------------------
    functionorder : OrderedSet
        Keeps track of function dynamic execution order.
    staticfxns : OrderedSet
        Keeps track of which functions run in static execution step.
    dynamicfxns : Orderedset
        Keeps track of which functions run in dynamic execution step.
    staticflows : list
        Flows to keep track of in static execution step.
    graph : networkx graph
        multigraph view of functions and flows

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> from fmdtools.define.flow.base import ExampleFlow
    >>> from fmdtools.define.container.parameter import ExampleParameter
    >>> class ExFxnArch(FunctionArchitecture):
    ...     container_p = ExampleParameter
    ...     def init_architecture(self, **kwargs):
    ...         self.add_flow("exf", ExampleFlow, s={'x': 0.0, 'y': 0.0})
    ...         self.add_fxn("ex_fxn", ExampleFunction, "exf", p=self.p)
    ...         self.add_fxn("ex_fxn2", ExampleFunction, "exf", p=self.p)

    >>> exfa = ExFxnArch(name="exfa")
    >>> exfa
    exfa ExFxnArch
    FUNCTIONS:
    ex_fxn ExampleFunction
    - ExampleState(x=1.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    ex_fxn2 ExampleFunction
    - ExampleState(x=1.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    FLOWS:
    exf ExampleFlow flow: ExampleState(x=0.0, y=0.0)

    This type of functional architecture only has dynamic functions:
    >>> exfa.dynamicfxns
    OrderedSet(['ex_fxn', 'ex_fxn2'])
    >>> exfa.staticfxns
    OrderedSet()

    This can in turn be simulated using FunctionArchitecture's built-in .propagate
    method. Note how the flow exf accumulates both ex_fxn and ex_fxn2 as reflected in
    their behavior methods:
    >>> exfa.propagate(1.0)
    >>> exfa
    exfa ExFxnArch
    FUNCTIONS:
    ex_fxn ExampleFunction
    - ExampleState(x=2.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    ex_fxn2 ExampleFunction
    - ExampleState(x=2.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    FLOWS:
    exf ExampleFlow flow: ExampleState(x=4.0, y=0.0)

    >>> exfa.propagate(2.0)
    >>> exfa
    exfa ExFxnArch
    FUNCTIONS:
    ex_fxn ExampleFunction
    - ExampleState(x=3.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    ex_fxn2 ExampleFunction
    - ExampleState(x=3.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    FLOWS:
    exf ExampleFlow flow: ExampleState(x=10.0, y=0.0)
    """

    __slots__ = ['fxns', 'functionorder', '_fxnflows', '_flowstates',
                 'graph', 'staticfxns', 'dynamicfxns', 'staticflows']
    default_track = ('fxns', 'flows', 'i')
    default_name = 'model'
    flexible_roles = ['flows', 'fxns']
    rolename = 'fa'

    def __init__(self, h={}, **kwargs):
        self.functionorder = OrderedSet()
        self._fxnflows = []
        self._flowstates = {}
        Architecture.__init__(self, h=h, **kwargs)

    def __repr__(self):
        fxnlist = ['\n' + fxn.__repr__() for fxn in self.fxns.values()]
        fxnlist = [fstr[:115] + '...' if len(fstr) > 120 else fstr for fstr in fxnlist]
        if len(fxnlist) > 15:
            fxnlist = fxnlist[:15]+["...("+str(len(fxnlist))+' total)\n']
        fxnstr = ''.join(fxnlist)
        flowlist = ['\n' + flow.__repr__() for flow in self.flows.values()]
        flowlist = [fstr[:115]+'...\n'if len(fstr) > 120 else fstr for fstr in flowlist]
        if len(flowlist) > 15:
            flowlist = flowlist[:15]+["...("+str(len(flowlist))+' total)\n']
        flowstr = ''.join(flowlist)
        repstr = (self.name + " " + self.__class__.__name__ +
                  '\n' + 'FUNCTIONS:' + fxnstr + '\nFLOWS:' + flowstr)
        return repstr

    def get_typename(self):
        return "Model"

    def inject_faults(self, faults):
        Architecture.inject_faults(self, 'fxns', faults)

    def add_fxn(self, name, fclass, *flownames, **fkwargs):
        """
        Instantiate a given function in the model.

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
        self.add_sim('fxns', name, fclass, *flownames, **fkwargs)
        for flowname in flownames:
            self._fxnflows.append((name, flowname))
        self.functionorder.update([name])

    def set_functionorder(self, functionorder):
        """
        Manually set the order of functions to be executed.

        (otherwise it will be executed based on the sequence of add_fxn calls)
        """
        if not self.functionorder.difference(functionorder):
            self.functionorder = OrderedSet(functionorder)
        else:
            raise Exception("Invalid list: "+str(functionorder) +
                            " should have elements: "+str(self.functionorder))

    def fxns_of_class(self, ftype):
        """Return dict of funcitons corresponding to the given class name ftype."""
        return {fxn: obj for fxn, obj in self.fxns.items()
                if obj.__class__.__name__ == ftype}

    def fxnclasses(self):
        """Return the set of class names used in the model."""
        return {obj.__class__.__name__ for fxn, obj in self.fxns.items()}

    def flowtypes_for_fxnclasses(self):
        """Return the flows required by each function class in the model (as a dict)."""
        class_relationship = dict()
        for fxn, obj in self.fxns.items():
            if class_relationship.get(obj.__class__.__name__, False):
                class_relationship[obj.__class__.__name__].update(obj.get_flowtypes())
            else:
                class_relationship[obj.__class__.__name__] = set(obj.get_flowtypes())
        return class_relationship

    def build(self, require_connections=True, **kwargs):
        """Build the model graph after the functions have been added."""
        super().build(**kwargs)
        self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items()
                                      if fxn.is_static()])
        self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items()
                                       if fxn.is_dynamic()])
        self.construct_graph(require_connections=require_connections)
        self.staticflows = [flow for flow in self.flows
                            if any([n in self.staticfxns
                                    for n in self.graph.neighbors(flow)])]

    def construct_graph(self, require_connections=True):
        """Create .graph nx.graph representation of the model."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.fxns, bipartite=0)
        self.graph.add_nodes_from(self.flows, bipartite=1)
        self.graph.add_edges_from(self._fxnflows)

        # check to see that all functions/flows are connected
        dangling_nodes = [e for e in nx.isolates(self.graph)]
        if dangling_nodes and require_connections:
            raise Exception("Fxns/flows disconnected from model: "+str(dangling_nodes))

    def calc_repaircost(self, additional_cost=0, default_cost=0, max_cost=np.inf):
        """
        Calculate the repair cost of the fault modes in the model.

        Uses given mode cost information for each function mode (in fxn.m).

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
        modecost = sum([c['cost'] if c['cost'] > 0.0 else default_cost
                        for m in modeprops.values() for c in m.values()])
        repair_cost = np.min([modecost, max_cost])
        return repair_cost

    def return_faultmodes(self):
        """
        Return faultmodes present in the model.

        Returns
        -------
        modes : dict
            Fault modes present in the model indexed by function name
        modeprops : dict
            Fault mode properties (defined in the function definition).
            Has structure {fxn:mode:properties}.
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
        Return the approximate memory usage of the model.

        Includes profile of fxn/flow memory usage.
        """
        mem_profile = {}
        mem = 0
        if hasattr(self, 'p'):
            mem_profile['params'] = sys.getsizeof(self.p)
        mem_profile['params'] += sys.getsizeof(self.sp)
        mem_profile['params'] += sys.getsizeof(self.track)
        for fxnname, fxn in self.fxns.items():
            mem_profile[fxnname], _ = fxn.get_memory()
        for flowname, flow in self.flows.items():
            mem_profile[flowname], _ = flow.get_memory()
        mem = np.sum([i for i in mem_profile.values()])
        return mem, mem_profile

    def reset(self):
        """Reset the model to the initial state (with no faults, etc)."""
        for flowname, flow in self.flows.items():
            flow.reset()
        for fxnname, fxn in self.fxns.items():
            fxn.reset()
        super().reset()

    def return_probdens(self):
        """Return the probability density of the model distributions."""
        probdens = 1.0
        for fxn in self.fxns.values():
            probdens *= fxn.return_probdens()
        return probdens

    def set_vars(self, *args, **kwargs):
        """
        Set variables in the model to set values (useful for optimization, etc.).

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
            varlist = []
            varvalues = []
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

    def propagate(self, time, fxnfaults={}, disturbances={}, run_stochastic=False):
        """
        Inject and propagates faults through the graph at one time-step.

        Parameters
        ----------
        time : float
            The current timestep.
        fxnfaults : dict
            Faults to inject during this propagation step.
            With structure {'function':['fault1', 'fault2'...]}
        disturbances : dict
            Variables to change during this propagation step.
            With structure {'function.var1':value}
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of
            random states over time.
        """
        # Step 0: Update model states with disturbances
        self.set_vars(**disturbances)

        # Step 1: Run Dynamic Propagation Methods in Order Specified
        # Inject Faults if Applicable
        for fxnname in self.dynamicfxns.union(fxnfaults.keys()):
            fxn = self.fxns[fxnname]
            faults = fxnfaults.get(fxnname, [])
            if type(faults) != list:
                faults = [faults]
            fxn('dynamic', faults=faults, time=time, run_stochastic=run_stochastic)

        # Step 2: Run Static Propagation Methods
        try:
            self.prop_static(time, run_stochastic=run_stochastic)
        except Exception as e:
            raise Exception("Error in static propagation at time t=" + str(time)) from e

    def prop_static(self, time, run_stochastic=False):
        """
        Propagate behaviors through model graph (static propagation step).

        Parameters
        ----------
        time : float
            Current time-step.
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of
            random states over time.
        """
        # set up history of flows to see if any has changed
        activefxns = self.staticfxns.copy()
        nextfxns = set()
        if not self._flowstates:
            self._flowstates = dict.fromkeys(self.staticflows)
            for flowname in self.staticflows:
                self._flowstates[flowname] = self.flows[flowname].return_mutables()
        n = 0
        while activefxns:
            flows_to_check = {*self.staticflows}
            for fxnname in list(activefxns).copy():
                # Update functions with new values, check to see if new faults or states
                oldmutables = self.fxns[fxnname].return_mutables()
                self.fxns[fxnname]('static', time=time, run_stochastic=run_stochastic)
                if oldmutables != self.fxns[fxnname].return_mutables():
                    nextfxns.update([fxnname])

                # Check what flows now have new values and add connected functions
                # (done for each because of communications potential)
                for flowname in self.fxns[fxnname].flows:
                    if flowname in flows_to_check:
                        try:
                            if self._flowstates[flowname] != self.flows[flowname].return_mutables():
                                nextfxns.update(set([n for n in self.graph.neighbors(flowname)
                                                     if n in self.staticfxns]))
                                flows_to_check.remove(flowname)
                        except ValueError as e:
                            raise Exception("Invalid mutables in flow: "
                                            + flowname) from e
            # check remaining flows that have not been checked already
            for flowname in flows_to_check:
                if self._flowstates[flowname] != self.flows[flowname].return_mutables():
                    nextfxns.update(set([n for n in self.graph.neighbors(flowname)
                                         if n in self.staticfxns]))
            # update flowstates
            for flowname in self.staticflows:
                self._flowstates[flowname] = self.flows[flowname].return_mutables()
            activefxns = nextfxns.copy()
            nextfxns.clear()
            n += 1
            if n > 1000:  # break if this is going for too long
                raise Exception("Undesired looping for functions in static propagation",
                                "at t=" + str(time) + ", these functions remain active:"
                                + str(activefxns))

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


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
