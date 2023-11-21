# -*- coding: utf-8 -*-
"""
Description: A module to define Functions, Components, Actions, and other classes with behaviors.
    
- :class:`Block`:       Superclass for Functions, Components, Actions, etc.
- :class:`FxnBlock`:    Class for defining model Functions
- :class:`Component`:   Class for defining Components (which have behaviors and live in a function)
- :class:`Action`:      Class for defining Actions (which have behaviors and live in a function, but have __call__ method for updating)
- :class:`CompArch`:    Class for defining Component Architectures, or sets of components to be contained by a FxnBlock
- :class:`ASG`:         Class for defining Action Sequence Graphs, or sets of actions with specific relationships.
- :class:`Simulable`:   Base class for object which simulate (blocks and models).
- :class:`GenericFxn`:  Generic function block. For use when the user has not yet defined a class for the given (to be implemented) 
                        function block. Acts as a placeholder that enables simulation.

- :func:`assoc_flows`:  Associates flows with the given object (Block, ASG, etc.) 
- :func:`inject_faults_internal`: Injects faults in the CompArch/ASG object obj.
"""
from typing import ClassVar
from decimal import Decimal
import sys
import itertools
import networkx as nx
import copy
import inspect
import warnings
from recordclass import dataobject, asdict, astuple

from fmdtools.define.state import State, ExampleState
from fmdtools.define.parameter import Parameter, SimParam, ExampleParameter
from fmdtools.define.rand import Rand
from fmdtools.define.common import get_true_fields, get_true_field, init_obj_attr
from fmdtools.define.common import get_obj_track, set_var, get_var
from fmdtools.define.time import Time
from fmdtools.define.mode import Mode, ExampleMode
from fmdtools.define.flow import init_flow, Flow
from fmdtools.analyze.result import Result
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist


def assoc_flows(obj, flows={}):
    """
    Associates flows with the given object (Block, ASG, etc.) Flows must be defined with the _init_ class variable
    pointing to the class to initialize (e.g., _init_flowname = FlowClass).
    
    Parameters
    ----------
    obj: object (Block, ASG)
        block requiring the flows
    
    flows : dict, optional
        If flows is provided AND it contains a flowname corresponding to the
        function's flowname, it will be used instead (so that it can act as a 
        connection to the rest of the model)
    """
    if hasattr(obj, 'flownames'):
        flows = {obj.flownames.get(fn, fn): flow for fn, flow in flows.items()}
    flows = flows.copy()
    for init_att in dir(obj):
        if init_att.startswith("_init_"):
            att = getattr(obj, init_att)
            attname = init_att[6:]
            if inspect.isclass(att) and issubclass(att, Flow) and not(attname in obj.flows):
                if attname in flows:
                    obj.flows[attname] = flows.pop(attname)
                else:
                    obj.flows[attname] = att(attname)
                if not isinstance(obj, dataobject):
                    setattr(obj, attname, obj.flows[attname])
    if flows:
        warnings.warn("these flows sent from model "+str([*flows.keys()])+" not added to class "+str(obj.__class__))

def inject_faults_internal(obj, faults):
    """
    Injects faults in the CompArch/ASG object obj.

    Parameters
    ----------
    obj : TYPE
        DESCRIPTION.
    faults : TYPE
        DESCRIPTION.
    """
    if isinstance(obj, ActArch):
        compdict = obj.actions
    elif isinstance(obj, CompArch):
        compdict = obj.components
    else:
        raise Exception("Invalid object type: "+type(obj)+" should be ASG or CompArch")

    for fault in faults:
        if fault in obj.faultmodes:
            comp = compdict[obj.faultmodes[fault]]
            comp.m.add_fault(fault[len(comp.name)+1:])


class Simulable(object):
    """
    Base class for object which simulate (blocks and models).
    
    Note that classes soley based on Simulable may not themselves be able to be simulated.
    """
    __slots__ = ('p', '_args_p', 'sp', '_args_sp', 'r', '_args_r', 'h', 'track', 'flows', 'name', 'is_copy')
    default_sp = {}
    default_track = ["all"]
    _init_p = Parameter
    _init_r = Rand
    _init_sp = SimParam

    def __init__(self, name='', p={}, sp={}, r={}, track={}):
        """
        Instantiates internal Simulable attributes with predetermined:
        
        Parameters
        ----------
        p : dict 
            Parameter values to set
        sp : dict
            Simulation parameter values to set
        r : dict
            Rand parameter values to set.
        track dict
            tracking dictionary
        """
        self.is_copy = False
        self.flows = dict()
        if not track:
            self.track = self.default_track
        else:
            self.track = track
        if not name:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name
        if not sp:
            sp = self.default_sp
        init_obj_attr(self, p=p, sp=sp, r=r)

    def add_flow_hist(self, hist, timerange, track):
        """
        Creates a history of flows for the Simulable and appends it to the History hist.

        Parameters
        ----------
        h : History
            History to append flow history to
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'. The default is None.
        """
        flow_track = get_sub_include('flows', track)
        if flow_track:
            hist['flows'] = History()
            for flowname, flow in self.flows.items():
                fh = flow.create_hist(timerange, get_sub_include(flowname, flow_track))
                if fh:
                    hist.flows[flowname] = fh

    def update_seed(self, seed=[]):
        """
        Updates seed and propogates update to contained actions/components.
        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        if seed:
            self.r.update_seed(seed)

    def find_classification(self, scen, mdlhists):
        """
        Placeholder for model find_classification methods (for running nominal models)

        Parameters
        ----------
        scen     : Scenario
            Scenario defining the model run.
        mdlhists : History
            History for the simulation(s)

        Returns
        -------
        endclass: Result
            Result dictionary with rate, cost, and expected cost values
        """
        return Result({'rate': scen.rate, 'cost': 1, 'expected cost': scen.rate})

    def new_params(self, p={}, sp={}, r={}, track={}):
        """
        Creates a copy of the defining parameters for use in a new Simulable
        Parameters
        ----------
        p     : dict
            Parameter args to update
        sp    : dict
            SimParam args to update
        r     : dict
            Rand args to update
        track : dict
            track kwargs to update.

        Returns
        -------
        p     : Param
            New Parameter
        sp    : SimParam
            New SimParam
        r     : dict
            Rand args
        track : dict
            track args
        """
        p = self.p.copy_with_vals(**p)
        sp = self.sp.copy_with_vals(**sp)
        if not r:
            r = {'seed': self.r.seed}
        if not track:
            track = copy.deepcopy(self.track)
        return p, sp, r, track

    def get_fxns(self):
        """
        Get fxns associated with the Simulable (self if FxnBlock, self.fxns if Model).

        Returns
        -------
        fxns: dict
            Dict with structure {fxnname: fxnobj}
        """
        if hasattr(self, 'fxns'):
            fxns = self.fxns
        else:
            fxns = {self.name: self}
        return fxns

    def get_vars(self, *variables, trunc_tuple=True):
        """
        Get variable values in the simulable.

        Parameters
        ----------
        *variables : list/string
            Variables to get from the model. Can be specified as a list
            ['fxnname2', 'comp1', 'att2'], or a str 'fxnname.comp1.att2'

        Returns
        -------
        variable_values: tuple
            Values of variables. Passes (non-tuple) single value if only one variable.
        """
        if type(variables) == str:
            variables = [variables]
        variable_values = [None]*len(variables)
        for i, var in enumerate(variables):
            if type(var) == str:
                var = var.split(".")
            if var[0] in ['functions', 'fxns']:
                f = self.get_fxns()[var[1]]
                var = var[2:]
            elif var[0] == 'flows':
                f = self.flows[var[1]]
                var = var[2:]
            elif var[0] in self.get_fxns():
                f = self.get_fxns()[var[0]]
                var = var[1:]
            elif var[0] in self.flows:
                f = self.flows[var[0]]
                var = var[1:]
            else:
                f = self
            variable_values[i] = get_var(f, var)
        if len(variable_values) == 1 and trunc_tuple:
            return variable_values[0]
        else:
            return tuple(variable_values)

    def get_scen_rate(self, fxnname, faultmode, time, phasemap={}, weight=1.0):
        """
        Get the scenario rate for the given single-fault scenario.

        Parameters
        ----------
        fxnname: str
            Name of the function with the fault
        faultmode: str
            Name of the fault mode
        time: int
            Time when the scenario is to occur
        phasemap : PhaseMap, optional
            Map of phases/modephases that define operations the mode will be injected
            during (and maps to the opportunity vector phases). The default is {}.
        weight : int, optional
            Scenario weight (e.g., if more than one scenario is sampled for the fault).
            The default is 1.

        Returns
        -------
        rate: float
            Rate of the scenario
        """
        fxn = self.get_fxns()[fxnname]
        fm = fxn.m.faultmodes.get(faultmode, False)
        if not fm:
            raise Exception("faultmode "+faultmode+" not in "+str(fxn.m.__class__))
        else:
            sim_time = self.sp.times[-1] - self.sp.times[0] + self.sp.dt
            rate = fm.calc_rate(time, phasemap=phasemap, sim_time=sim_time,
                                sim_units=self.sp.units, weight=weight)
        return rate

    def get_args(self, **kwargs):
        """
        Get the current arguments for a given Simulable stored in _at_args.

        Parameters
        ----------
        kwargs: dict
            Attributes to overwrite (and their values)

        Returns
        -------
        saved_args : dict
            Dictionary of saved arguments.
        """
        args = {}
        for at in dir(self):
            if at.startswith("_args"):
                role = at[6:]
                if role in kwargs:
                    args[role] = kwargs[role]
                else:
                    saved_arg = getattr(self, at, {})
                    if saved_arg:
                        args[role] = saved_arg
        return args


class Block(Simulable):
    __slots__ = ['s', '_args_s', 'm', '_args_m', 't', '_args_t']
    default_track = ['s', 'm', 'r', 't', 'i']
    _init_s = State
    _init_m = Mode
    _init_t = Time
    """ 
    Superclass for FxnBlock and Component subclasses. Has functions for model setup, querying state, reseting the model
    
    Attributes
    ----------
    p : Parameter
        Internal Parameter for the block. Instanced from _init_p
    s : State
        Internal State of the block. Instanced from _init_s.
    m : Mode
        Internal Mode for the block. Instanced from _init_m
    r : Rand
        Internal Rand for the block. Instanced from _init_r
    t : Time
        Internal Time for the block. Instanced from _init_t
    name : str
        Block name
    flows : dict
        Dictionary of flows included in the Block (if any are added via _init_flowname)
    is_copy : bool
        Marker for whether the object is a copy.
    """
    def __init__(self, name='', flows={}, s={}, p={}, m={}, r={}, t={}, sp={}, track=''):
        """
        Instance superclass. Called by FxnBlock and Component classes.

        Parameters
        ----------
        name : str
            Name for the Block instance.
        flows :dict
            Flow objects passed from the model level to use instead of instantiating locally.
        p : dict, optional
            Internal parameters to override from defaults. The default is {}.
        s : dict, optional
            Internal states to override from defaults. The default is {}.
        c : dict, optional
            Internal CompArch fields/arguments override from defaults. The default is {}.
            FxnBlock must have an _init_c property.
        a : dict, optional
            Internal ASG fields/arguments override from defaults. The default is {}.
            FxnBlock must have an _init_a property.
        r : dict, optional
            Internal Rand fields/arguments override from defaults. The default is {}.
        m : dict, optional
            Internal Mode fields/arguments override from defaults. The default is {}.
        t : dict, optional
            Internal Time fields/arguments to override from defaults. The defautl is {}
        """
        super().__init__(name=name, p=p, sp=sp, r=r, track=track)
        assoc_flows(self, flows=flows)
        init_obj_attr(self, s=s, m=m, t=t)
        self.update_seed()

    def new_with_params(self, s={}, m={}, t={}, **kwargs):
        """
        Creates a new Block with the same parameters as the current model but
        with changes to params (p, sp, track, rand etc.). For use when simulating
        individually.
        """
        p, sp, r, track = super().new_params(**kwargs)
        if not s:
            s = self._args_s
        if not m:
            m = self._args_m
        if not t:
            t = self._args_t
        return self.__class__(name=self.name, s=s, p=p, m=m, t=t, sp=sp, r=r, track=track)

    def get_typename(self):
        """
        Gets the name of the type (Block for Blocks)
        Returns
        -------
        typename: str
            Block
        """
        return "Block"
    
    def is_static(self):
        """Checks if Block has static execution step"""
        return (getattr(self, 'behavior', False) or 
                getattr(self, 'static_behavior', False) or
                (hasattr(self, 'aa') and getattr(self.aa, 'proptype','') == 'static'))
        
    def is_dynamic(self):
        """Checks if Block has dynamic execution step"""
        return (hasattr(self, 'dynamic_behavior') or
                (hasattr(self, 'aa') and getattr(self.aa, 'proptype', '') == 'dynamic'))

    def __repr__(self):
        """
        Provides a repl-friendly string showing the states of the Block
        
        Returns
        -------
        repr: str
            console string
        """
        if hasattr(self, 'name'):
            fxnstr = getattr(self, 'name', '')+' '+self.__class__.__name__+'\n'
            for at in ['s', 'm']:
                fxnstr = fxnstr+"- "+getattr(self, at).__repr__()+'\n'
            return fxnstr
        else:
            return 'New uninitialized '+self.__class__.__name__

    def get_rand_states(self, auto_update_only=False):
        """
        Gets dict of random states from block and associated actions/components

        Parameters
        ----------
        auto_update_only

        Returns
        -------

        """
        rand_states = self.r.get_rand_states(auto_update_only)
        if hasattr(self, 'ca'):
            rand_states.update(self.ca.get_rand_states(auto_update_only=auto_update_only))
        if hasattr(self, 'aa'):
            for actname, act in self.aa.actions.items():
                if act.get_rand_states(auto_update_only=auto_update_only): 
                    rand_states[actname] = act.get_rand_states(auto_update_only=auto_update_only)
        return rand_states

    def choose_rand_fault(self, faults, default='first', combinations=1):
        """
        Randomly chooses a fault or combination of faults to insert in fxn.m. 

        Parameters
        ----------
        faults : list
            list of fault modes to choose from
        default : str/list, optional
            Default fault to inject when model is run deterministically. 
            The default is 'first', which chooses the first in the list. 
            Can provide a mode as a str or a list of modes
        combinations : int, optional
            Number of combinations of faults to elaborate and select from. 
            The default is 1, which just chooses single fault modes.
        """
        if getattr(self.r, 'run_stochastic', True):
            faults = [list(x) for x in itertools.combinations(faults, combinations)]
            self.m.add_fault(*self.r.rng.choice(faults))
        elif default == 'first':
            self.m.add_fault(faults[0])
        elif type(default) == str:
            self.m.add_fault(default)
        else:
            self.m.add_fault(*default)

    def get_flowtypes(self):
        """
        Returns the names of the flow types in the model

        Returns
        -------

        """
        return {obj.__class__.__name__ for name, obj in self.flows.items()}

    def reset(self):            #
        """
        reset requires flows to be cleared first. Resets the block to the initial state with no faults. Used by default
        in derived objects when resetting the model. Requires associated flows to be cleared first.

        Returns
        -------

        """

        self.m.remove_any_faults()
        self.s = self._init_s(**self._args_s)
        self.r = self._init_r(**self._args_r)
        self.t.reset()
        for flow in self.flows.values():
            flow.reset()

    def copy(self, *args, **kwargs):
        """

        Parameters
        ----------
        args   : tuple
            New arguments to use to instantiate the block, (e.g., flows, p, s)
        kwargs :
            New kwargs to use to instantiate the block 

        Returns
        -------
        cop : Block
            copy of the exising block

        """
        cop = self.__new__(self.__class__)  # Is this adequate? Wouldn't this give it new components?
        cop.is_copy=True
        try:
            saved_kwargs = self.get_args(**kwargs)
            cop.__init__(self.name, *args, **saved_kwargs)
        except TypeError as e:
            raise Exception("Poor specification of "+str(self.__class__)) from e
        cop.m.mirror(self.m)
        cop.t = self.t.copy(**self._args_t)
        cop.s.assign(self.s)
        cop.r.assign(self.r)
        if hasattr(self, 'h'): 
            cop.h = self.h.copy()
        return cop

    def get_memory(self):
        """
        Gets the approximate memory usage of the block in bytes (not complete)

        Returns
        -------

        """
        mem = 0
        mem+=sys.getsizeof(self.m.opermodes)
        if hasattr(self, 'r'):
            mem+=sys.getsizeof(self.r)
        if hasattr(self, 'm'):
            for fm in self.m.faultmodes.values():
                mem+=sys.getsizeof(fm)
        if hasattr(self, 'mode_state_dict'):
            mem+=sys.getsizeof(self.mode_state_dict)
        if hasattr(self, 'timers'):
            for timer in self.timers:
                mem+=sys.getsizeof(timer)
        if hasattr(self, 'internal_flows'):
            for flowname, flow in self.internal_flows.items():
                mem+= flow.get_memory()
        if hasattr(self, 'ca'):
            for name, comp in self.ca.components.items():
                mem+=comp.get_memory()
        if hasattr(self, 'aa'):
            for name, comp in self.aa.actions.items():
                mem+=comp.get_memory()
        for state in asdict(self.s):
            mem+=2*sys.getsizeof(state)  # (*2 because both the initstate and the actual state should be counted)
        return mem

    def return_mutables(self):
        """
        Return all mutable values in the block.

        Used in static propagation steps to check if the block has changed.

        Returns
        -------
        states : tuple
            tuple of all states in the block
        """
        return (*astuple(self.s),
                *self.m.return_mutables(),
                *self.r.return_mutables(),
                *self.t.return_mutables())

    def return_probdens(self):
        """Gets the probability density associated with a Block and its components/actions (if any)"""
        state_pd = self.r.return_probdens()
        if hasattr(self, 'ca'): 
            for compname, comp in self.ca.components:
                state_pd*=comp.return_probdens()
        if hasattr(self, 'aa'):
            for actionname, action in self.aa.actions:
                state_pd*=action.return_probdens()
        return state_pd

    def create_hist(self, timerange, track='default'):
        """Initializes the function state history fxnhist of the model mdl over the time range timerange.
        A pointer to the history is then stored at self.h.
        
        Parameters
        ----------
        timerange : array
            Numpy array of times to initialize in the dictionary.
        track : 'all' or dict, 'none', optional
            Which model states to track over time, which can be given as 'all' or a 
            dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
            The default is 'all'.
        Returns
        -------
        fxnhist : dict
            A dictionary history of each recorded block property over the given timehist
        """
        if hasattr(self, 'h'):
            return self.h
        else:
            if isinstance(self, FxnBlock):
                all_track = FxnBlock.default_track+['flows']
            else:
                all_track = Block.default_track+['flows']
            track = get_obj_track(self, track, all_track)
            if track:
                hist = History()
                init_indicator_hist(self, hist, timerange, track)
                self.add_flow_hist(hist, timerange, track)
                other_tracks = [t for t in track if t not in ('i', 'flows')]
                for at in other_tracks:
                    at_track = get_sub_include(at, track)
                    attr = getattr(self, at, False)
                    if attr: 
                        at_h = attr.create_hist(timerange, at_track)
                        if at_h:
                            hist[at] = at_h
                
                self.h = hist.flatten()
                return self.h
            else:
                return History()

    def propagate(self, time, faults={}, disturbances={}, run_stochastic=False):
        """
        Injects and propagates faults through the graph at one time-step

        Parameters
        ----------
        time : float
            The current timestep.
        faults : dict
            Faults to inject during this propagation step.
            With structure {fname: ['fault1', 'fault2'...]}
        disturbances : dict
            Variables to change during this propagation step.
            With structure {'var1': value}
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of random states over time.
        """
        # Step 0: Update block states with disturbances
        for var, val in disturbances.items():
            set_var(self, var, val)
        faults = faults.get(self.name, [])

        # Step 1: Run Dynamic Propagation Methods in Order Specified and Inject Faults if Applicable
        if hasattr(self, 'dynamic_loading_before'):
            self.dynamic_loading_before(self, time)
        if self.is_dynamic():
            self("dynamic", time=time, faults=faults, run_stochastic=run_stochastic)
        if hasattr(self, 'dynamic_loading_after'):
            self.dynamic_loading_after(self, time)

        # Step 2: Run Static Propagation Methods
        active = True
        oldmutables = self.return_mutables()
        flows_mutables = {f: fl.return_mutables() for f, fl in self.flows.items()}
        while active:
            if self.is_static():
                self("static", time=time, faults=faults, run_stochastic=run_stochastic)

            if hasattr(self, 'static_loading'):
                self.static_loading(time)
            # Check to see what flows now have new values and add connected functions (done for each because of communications potential)
            active = False
            newmutables = self.return_mutables()
            if oldmutables != newmutables:
                active = True
                oldmutables = newmutables
            for flowname, fl in self.flows.items():
                newflowmutables = fl.return_mutables()
                if flows_mutables[flowname] != newflowmutables:
                    active = True
                    flows_mutables[flowname] = newflowmutables

# COMPONENT/COMPONENT ARCHITECTURES


class Component(Block):
    """
    Superclass for components (most attributes and methods inherited from Block superclass)
    """
    def behavior(self, time):
        """ Placeholder for component behavior methods. Enables one to include components
        without yet having a defined behavior for them."""
        return 0

    def get_typename(self):
        return "Component"


class CompArch(dataobject, mapping=True):
    """Container for holding component architectures."""

    archtype: ClassVar[str] = 'default'
    components: dict = dict()
    faultmodes: dict = dict()
    default_track = ('i', 'components')
    def make_components(self, CompClass, *args, **kwargs): # noqa
        """
        Adds components to the component architecture.

        Parameters
        ----------
        CompClass : Component
            Component to add
        *args : strs
            Names for the components to instantiate in the architecture
        **kwargs : kwargs
            keyword arguments to send CompClass, of form {'name':kwarg}.
            unless all have the same kwargs
        """
        if not self.components:
            self.components = dict()
        if not self.faultmodes:
            self.faultmodes = dict()

        for arg in args:
            if arg in kwargs:
                kwargs_comp = kwargs[arg]
            else:
                kwargs_comp = kwargs
            self.components[arg] = CompClass(arg, **kwargs_comp)
            self.faultmodes.update({self.components[arg].name + '_' + modename: arg
                                    for modename in self.components[arg].m.faultmodes})

    def copy_with_arg(self, **kwargs):
        cop = self.__class__(**kwargs)
        for compname, component in self.components.items():  # TODO: needs to cover all attributes, copy should a part of Block
            cop_comp = cop.components[compname]
            cop_comp.s = cop_comp._init_s(**asdict(component.s))
            cop_comp.m.mirror(component.m)
            cop_comp.t = component.t.copy()
            if hasattr(component, 'h'):
                cop_comp.h = component.h.copy()
        return cop

    def update_seed(self, seed):
        for comp in self.components.values():
            comp.update_seed(seed)

    def get_rand_states(self, auto_update_only=False):
        rand_states={}
        for compname, comp in self.components.items():
            if comp.get_rand_states(auto_update_only=auto_update_only): 
                rand_states[compname] = comp.get_rand_states(auto_update_only=auto_update_only)
        return rand_states

    def get_faults(self):
        return {comp.name+'_'+f for comp in self.components.values() for f in comp.m.faults}

    def reset(self):
        for name, component in self.components.items():
            component.reset()

    def get_true_field(self, fieldname, *args, **kwargs):
        return get_true_field(self, fieldname, *args, **kwargs)

    def get_true_fields(self, *args, **kwargs):
        return get_true_fields(self, *args, **kwargs)

    def create_hist(self, timerange, track):
        """
        Creates a history corresponding to CompArch attributes.

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'. The default is None.

        Returns
        -------
        h : History
            History corresponding to the CompArch
        """
        h = History()
        if track == 'default':
            track = self.default_track
        init_indicator_hist(self, h, timerange, track)
        
        components_track = get_sub_include('components', track)
        if components_track:
            hc = History()
            for c, comp in self.components.items():
                comp_track = get_sub_include(c, components_track)
                if comp_track: 
                    hc[c] = comp.create_hist(timerange, comp_track)
            h['components'] = hc
        return h

    def return_mutables(self):
        cm = []
        for c in self.components.values():
            cm.extend(c.return_mutables())
        return cm
    
# Actions/ASGs


class Action(Block):
    __slots__ = ('duration',)
    """
    Superclass for actions (most attributes and methods inherited from Block superclass)
    """
    def __init__(self, name, duration=0.0, **kwargs):
        self.duration=duration
        super().__init__(name, **kwargs)

    def __call__(self, time=0, run_stochastic=False, proptype='dynamic', dt=1.0):
        """
        Updates the behaviors, faults, times, etc of the action 

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool
            Whether to run the simulation using stochastic or deterministic behavior
        """
        if time > self.t.time:
            self.r.update_stochastic_states()
        if proptype == 'dynamic':
            if self.t.time < time:
                self.behavior(time)
                self.t.t_loc += dt
        else:
            self.behavior(time)
            self.t.t_loc += dt
        self.t.time = time
    
    def copy(self, *args, **kwargs):
        cop = super().copy(*args, **kwargs)
        cop.duration = self.duration
        return cop
    def behavior(self, time):
        """Placeholder behavior method for actions"""
        a = 0


class ActArch(object):
    """
    Constructs the Action Sequence Graph with the given parameters.
    
    Parameters
    ----------
    initial_action : str/list
        Initial action to set as active. Default is 'auto'
            - 'auto' finds the starting node of the graph and uses it
            - 'ActionName' sets the given action as the first active action
            - providing a list of actions will set them all to active (if multi-state rep is used)
    state_rep : 'finite-state'/'multi-state'
        How the states of the system are represented. Default is 'finite-state'
            - 'finite-state' means only one action in the system can be active at once (i.e., a finite state machine)
            - 'multi-state' means multiple actions can be performed at once
    max_action_prop : 'until_false'/'manual'/int
        How actions progress. Default is 'until_false'
            - 'until_false' means actions are simulated until all outgoing conditions are false
            - providing an integer places a limit on the number of actions that can be performed per timestep
    proptype : 'static'/'dynamic'/'manual'
        Which propagation step to execute the Action Sequence Graph in. Default is 'dynamic'
            - 'manual' means that the propagation is performed manually (defined in a behavior method)
    per_timestep : bool
        Defines whether the action sequence graph is reset to the initial state each time-step (True) or stays in the current action (False). Default is False
    """    
    initial_action = "auto"
    state_rep = "finite-state"
    max_action_prop = "until_false"
    proptype = 'dynamic'
    per_timestep = False
    default_track = ('actions', 'active_actions', 'i')

    def __init__(self, flows={}, is_copy=False):
        self.actions = {}
        self.flows = flows 
        self.conditions = {}
        self.action_graph = nx.DiGraph()
        self.flow_graph = nx.DiGraph()
        self.conditions = {}
        self.faultmodes = {}
        self.is_copy = is_copy
        self.active_actions = set()

    def build(self):
        if self.initial_action == 'auto':
            initial_action = [act for act, in_degree in self.action_graph.in_degree if in_degree == 0]
            if not initial_action:
                raise Exception("Cannot set initial action--no starting node")
        elif type(self.initial_action) == str:
            initial_action = [self.initial_action]
        self.set_active_actions(initial_action)
        if self.state_rep == 'finite-state' and len(self.active_actions) > 1:
            raise Exception("Cannot have more than one initial action with finite-state representation")

    def add_flow(self, flowname, fclass=Flow, p={}, s={}):
        """
        Adds a flow with given attributes to ASG. Used to enable a flexible
        internal flow architecture in the ASG.

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
        """
        if flowname not in self.flows:
            self.flows[flowname] = init_flow(flowname, fclass, p=p, s=s)

    def add_act(self, name, actclass, *flownames, duration=0.0, **params):
        """
        Associate an Action with the Function Block for use in the Action Sequence Graph

        Parameters
        ----------
        name : str
            Internal Name for the Action
        actclass : Action
            Action class to instantiate
        *flownames : flow
            Flows (optional) which connect the actions
        duration:
            Duration of the action. Default is 0.0
        **params : any
            parameters to instantiate the Action with. 
        """
        flows = {fl: self.flows[fl] for fl in flownames}
        action = actclass(name=name, flows={**flows}, **params)
        
        self.actions[name] = action
        self.actions[name].duration = duration
        
        self.action_graph.add_node(name)
        self.flow_graph.add_node(name, bipartite=0)
        for flow in flows:
            self.flow_graph.add_node(flow, bipartite=1)
            self.flow_graph.add_edge(name, flow)
            
        modes_to_add = {action.name+'_'+f: val for f, val in action.m.faultmodes.items()}
        fmode_intersect = set(modes_to_add).intersection(self.faultmodes)
        if any(fmode_intersect):
            raise Exception("Action "+name+" overwrites existing fault modes: "+str(fmode_intersect)+". Rename the faults")
        self.faultmodes.update({action.name+'_'+modename: name for modename in action.m.faultmodes})

    def cond_pass(self): # noqa
        return True

    def add_cond(self, start_action, end_action, name='auto', condition='pass'):
        """
        Associates a Condition with the Function Block for use in the Action Sequence Graph

        Parameters
        ----------
        start_action : str
            Action where the condition is checked
        end_action : str
            Action that the condition leads to.
        name : str
            Name for the condition. Defaults to numbered conditions if none are provided.
        condition : method
            Method in the class to use as a condition. Defaults to self.condition_pass if none are provided
        """
        if name == 'auto':
            name = str(len(self.conditions)+1)
        if condition == 'pass':
            condition = self.cond_pass
        self.conditions[name] = condition
        self.action_graph.add_edge(start_action, end_action, **{'name': name, name: 'name', 'arrow': True})

    def set_active_actions(self, actions):
        """Helper method for setting given action(s) as active"""
        if type(actions) == str:
            if actions in self.actions:
                actions = [actions]
            else:
                raise Exception("initial_action="+actions+" not in self.actions: "+str(self.actions))
        if type(actions) == list:
            self.active_actions = set(actions)
            if any(self.active_actions.difference(self.actions)):
                raise Exception("Initial actions not associated with model: "+str(self.active_actions.difference(self.actions)))
        else:
            raise Exception("Invalid option for initial_action")

    def __call__(self, time, run_stochastic, proptype, dt):
        """
        Propagates behaviors through the internal Action Sequence Graph

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool/str
            Whether to run the simulation using stochastic or deterministic behavior
        proptype : str
            Type of propagation step to update ('behavior', 'static_behavior', or 'dynamic_behavior')
        dt : float
            Timestep to propagate over.
        """
        if self.per_timestep:
            self.set_active_actions(self.initial_action)
            for action in self.active_actions:
                self.actions[action].t.t_loc = 0.0

        if proptype == self.proptype:
            active_actions = self.active_actions
            num_prop = 0
            while active_actions:
                new_active_actions = set(active_actions)
                for action in active_actions:
                    self.actions[action](time, run_stochastic, proptype=proptype, dt=dt)
                    action_cond_edges = self.action_graph.out_edges(action, data=True)
                    for act_in, act_out, atts in action_cond_edges:
                        try:
                            cond = self.conditions[atts['name']]()
                        except TypeError as e:
                            raise TypeError("Poorly specified condition "+str(atts['name'])+": ") from e
                        if cond and getattr(self.actions[action], 'duration', 0.0)+dt <= self.actions[action].t.t_loc:
                            self.actions[action].t.t_loc = 0.0
                            new_active_actions.add(act_out)
                            new_active_actions.discard(act_in)
                if len(new_active_actions) > 1 and self.state_rep == 'finite-state':
                    raise Exception("Multiple active actions in a finite-state representation: "+str(new_active_actions))
                num_prop += 1
                if type(self.proptype) == int and num_prop >= self.proptype:
                    break
                if new_active_actions == set(active_actions):
                    break
                else:
                    active_actions = new_active_actions
                if num_prop > 10000:
                    raise Exception("Undesired looping in Function ASG for: "+self.name)
            self.active_actions = active_actions

    def get_faults(self):
        return {act.name+"_"+f for act in self.actions.values() for f in act.m.faults}

    def update_seed(self, seed=[]):
        if seed:
            for act in self.actions.values():
                act.update_seed(seed)

    def copy(self, flows={}, **kwargs):
        newflows = {}
        for flowname, flow in self.flows.items():
            if flowname in flows:
                newflows[flowname] = flows[flowname]
            else:
                newflows[flowname] = self.flows[flowname].copy()

        cop = self.__class__(flows=newflows, is_copy=True, **kwargs)
        for flowname, flow in newflows.items():
            if flow.__hash__() != cop.flows[flowname].__hash__():
                raise Exception("Flow not associated w- lower level of ASG: " + flowname)
        
        for actname, action in self.actions.items(): 
            cop_act = cop.actions[actname]
            cop_act.duration = action.duration
            cop_act.s = action._init_s(**asdict(action.s))
            cop_act.m.mirror(action.m)
            cop_act.t = action.t.copy()
            if hasattr(action, 'h'):
                cop_act.h = action.h.copy()
            
        cop.active_actions = copy.deepcopy(self.active_actions)
        return cop

    def reset(self):
        for name, action in self.actions.items():
            action.reset()
        self.build()

    def create_hist(self, timerange, track):
        """
        Create a history corresponding to ASG attributes.

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'. The default is None.

        Returns
        -------
        h : History
            History corresponding to the ASG.
        """
        if track == 'default':
            track = self.default_track
        h = History()
        if 'i' in track or track == 'all':
            init_indicator_hist(self, h, timerange, track)
        actions_track = get_sub_include('actions', track)
        if actions_track:
            ha = History()
            for a, act in self.actions.items():
                act_track = get_sub_include(a, actions_track)
                if act_track: 
                    ha[a] = act.create_hist(timerange, act_track)
            h['actions'] = ha
        h.init_att('active_actions', self.active_actions, timerange=timerange, track=track)
        return h

    def return_mutables(self):
        am = []
        for a in self.actions.values():
            am.extend(a.return_mutables())
        for f in self.flows.values():
            am.extend(f.return_mutables())
        am.append(copy.copy(self.active_actions))
        return am
        

# Function superclass


class FxnBlock(Block):
    """
    Superclass for representing system functions.

    Additional roles
    ----------
    ca : CompArch
        Component Architecture fulfilling function.
    aa : ActArch
        Action Architecture performed by function.
    """

    __slots__ = ["ca", "_args_ca", "aa", "_args_aa", "args_f"]
    default_track = ["ca", "aa"]+Block.default_track

    def __init__(self, name='', flows={}, ca=dict(), aa=dict(), local=dict(),
                 args_f=dict(), **kwargs):
        """
        Instantiate the function superclass with the relevant parameters.

        Parameters
        ----------
        flows :dict
            Flow objects passed to use instead of instantiating locally.
        ca : dict, optional
            Internal CompArch fields/arguments. The default is {}.
        aa : dict, optional
            Internal ASG fields/arguments. The default is {}.
        args_f : dict, optional
            arguments to pass to custom __init__ function.
        """
        super().__init__(name=name, flows=flows.copy(), **kwargs)
        self.args_f = args_f

        for at in ['ca', 'aa']:  # NOTE: similar to init_obj_attr()
            at_arg = eval(at)
            at_init = getattr(self, '_init_'+at, False)
            if at_init:
                try:
                    if at == "aa":
                        setattr(self, at, at_init(flows=self.flows.copy(), **at_arg))
                    elif at == "ca":
                        setattr(self, at, at_init(**at_arg))
                except TypeError as e:
                    invalid_args = [a for a in at_arg if a not in at_init.__fields__]
                    if invalid_args:
                        argstr = ", Invalid args: "+', '.join(invalid_args)
                    else:
                        argstr = ''
                    raise TypeError("Poor specification for : " + str(at_init) +
                                    " with kwargs: " + str(at_arg) + argstr) from e
                setattr(self, '_args_' + at, at_arg)
                self.update_contained_modes(at)
            elif at_arg:
                raise Exception(at + " argument provided: " + str(at_arg) +
                                "without associating a CompArch/ActArch to _init_" + at)
        self.update_seed()

    def update_contained_modes(self, at):
        """
        Add contained faultmodes for the role at to the FxnBlock model.

        Parameters
        ----------
        at : str ('ca' or 'aa')
            Role to update (for CompArch or ActArch roles)
        """
        if at == 'ca':
            compacts = self.ca.components
        elif at == 'aa':
            compacts = self.aa.actions
        for ca in compacts.values():
            self.m.faultmodes.update({ca.name + "_" + f: vals
                                      for f, vals in ca.m.faultmodes.items()})


    def get_typename(self):
        return "FxnBlock"

    def return_faultmodes(self):
        """
        Get the fault modes present in the simulation (for propagate/model).

        Returns
        -------
        ms : list
            List of faults present.
        modeprops : dict
            Dict of corresponding fault mode properties.
        """
        ms = [m for m in self.m.faults.copy() if m != 'nom']
        modeprops = dict.fromkeys(ms)
        for mode in ms:
            modeprops[mode] = self.m.faultmodes.get(mode)
            if mode not in self.m.faultmodes:
                raise Exception("Mode " + mode + " not in m.faultmodes for fxn " +
                                self.__class__.__name__+" and may not be tracked.")
        return ms, modeprops

    def update_seed(self, seed=[]):
        """
        Update seed and propogates update to contained actions/components.

        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        super().update_seed(seed)

        if hasattr(self, 'ca'):
            self.ca.update_seed(self.r.seed)
        if hasattr(self, 'aa'):
            self.aa.update_seed(self.r.seed)

    def copy(self, *args, **kwargs):
        """
        Create a copy of the function object.

        Adds newflows and arbitrary parameters to be associated with the copy. Used when
        copying the model.

        Returns
        -------
        copy : FxnBlock
            Copy of the given function with new flows
        """
        cop = super().copy(*args, **kwargs)
        if hasattr(self, 'ca'):
            cop.ca = self.ca.copy_with_arg(**self._args_ca)
            cop.update_contained_modes('ca')
        if hasattr(self, 'aa'):
            cop.aa = self.aa.copy(flows=cop.flows.copy(), **self._args_aa)
            cop.update_contained_modes('aa')
        if hasattr(self, 'h'):
            if hasattr(self, 'ca'):
                for compname, comp in cop.ca.components.items():
                    ex_hist = cop.h.get("ca.components." + compname)
                    if ex_hist:
                        comp.h = ex_hist.copy()
                        for k, v in comp.h.items():
                            cop.h["ca.components." + compname + "." + k] = v
            if hasattr(self, 'aa'):
                #if "a.active_actions" in self.h.keys():
                #    cop.h["a.active_actions"] = self.h['a.active_actions'].copy()
                for actname, act in cop.aa.actions.items():
                    ex_hist = cop.h.get("aa.actions." + actname)
                    if ex_hist: 
                        act.h = ex_hist.copy()
                        for k, v in act.h.items():
                            cop.h["aa.actions." + actname + "." + k] = v
        return cop

    def return_mutables(self):
        bm = super().return_mutables()
        cm, am = (), ()
        if hasattr(self, 'ca'):
            cm = self.ca.return_mutables()
        if hasattr(self, 'aa'):
            am = self.aa.return_mutables()
        return *bm, *cm, *am

    def __call__(self, proptype, faults=[], time=0, run_stochastic=False):
        """
        Updates the state of the function at a given time and injects faults.

        Parameters
        ----------
        proptype : str
            Type of propagation step to update ('behavior', 'static_behavior', or 'dynamic_behavior')
        faults : list, optional
            Faults to inject in the function. The default is [].
        time : float, optional
            Model time. The default is 0.
        run_stochastic : book
            Whether to run the simulation using stochastic or deterministic behavior
        """
        self.r.run_stochastic = run_stochastic
        if faults:
            self.m.add_fault(*faults)  # if there is a fault, it is instantiated
        if hasattr(self, 'mode_state_dict') and any(faults):
            self.update_modestates()
        if hasattr(self, 'condfaults'):
            self.condfaults(time)    # conditional faults and behavior are then run
        if time > self.t.time:
            self.r.update_stochastic_states()
        if hasattr(self, 'ca'):
            inject_faults_internal(self.ca, faults)
        if hasattr(self, 'aa'):
            inject_faults_internal(self.aa, faults)
            try:
                self.aa(time, run_stochastic, proptype, self.t.dt)
            except TypeError as e:
                raise Exception("Poorly specified ActArch: "
                                + str(self.a.__class__)) from e

        if proptype == 'static' and hasattr(self, 'behavior'):
            self.behavior(time)     # generic behavioral methods are run at all steps
        if proptype == 'static' and hasattr(self, 'static_behavior'):
            self.static_behavior(time)
        elif proptype == 'dynamic' and hasattr(self, 'dynamic_behavior') and time > self.t.time:
            if self.t.run_times >= 1:
                for i in range(self.t.run_times):
                    self.dynamic_behavior(time)
            elif not Decimal(str(time)) % Decimal(str(self.t.dt)):
                self.dynamic_behavior(time)
        elif proptype == 'reset':
            if hasattr(self, 'static_behavior'):
                self.static_behavior(time)
            if hasattr(self, 'behavior'):
                self.behavior(time)
            if hasattr(self, 'dynamic_behavior'):
                self.dynamic_behavior(time)

        # propagate faults from action/component level to function level
        if hasattr(self, 'aa') and self.aa.actions:
            self.m.faults.difference_update(self.aa.faultmodes)
            self.m.faults.update(self.aa.get_faults())
        comps = getattr(self, 'ca', {'components': {}})['components']
        if comps:
            self.m.faults.difference_update(self.ca.faultmodes)
            self.m.faults.update(self.ca.get_faults())
        self.t.time = time
        if run_stochastic == 'track_pdf':
            self.r.probdens = self.r.return_probdens()
        if self.m.exclusive is True and len(self.m.faults) > 1:
            raise Exception("More than one fault present in " + self.name +
                            "\n at t= " + str(time) +
                            "\n faults: " + str(self.m.faults) +
                            "\n Is the mode representation nonexclusive?")
        return

    def reset(self):
        super().reset()
        if hasattr(self, 'ca'):
            self.ca.reset()
        if hasattr(self, 'aa'):
            self.aa.reset()
        self('reset', faults=[], time=0)


class ExampleFxnBlock(FxnBlock):
    """Example Function block for testing"""

    _init_p = ExampleParameter
    _init_s = ExampleState
    _init_m = ExampleMode

    def dynamic_behavior(self, time):
        """Increment x if nominal, else increment y."""
        if not self.m.any_faults():
            self.s.x += self.p.x
        else:
            self.s.y += self.p.y
        if time < 1.0:
            self.s.put(x=0.0, y=0.0)

    def find_classification(self, scen, hist):
        return {"xy": self.s.x + self.s.y}


class GenericFxn(FxnBlock):
    """Generic function block. For use when the user has not yet defined a class for the
    given (to be implemented) function block. Acts as a placeholder that enables simulation."""
    def __init__(self, name='', flows={}, args_f={}, **kwargs):
        super().__init__(name=name, flows=flows, **kwargs)
