# -*- coding: utf-8 -*-
"""
Description: A module to define Functions, Components, Actions, and other classes with behaviors.
    
- :class:`Block`:       Superclass for Functions, Components, Actions, etc.
- :class:`FxnBlock`:    Class for defining model Functions
- :class:`Component`:   Class for defining Components (which have behaviors and live in a function)
- :class:`Component`:   Class for defining Actions (which have behaviors and live in a function, but have updateact method)
- :class:`Timer`:       Class for counting/incrementing time
"""
import numpy as np
from decimal import Decimal
import sys
import itertools
import networkx as nx
import copy
import inspect
import warnings
from recordclass import dataobject, asdict

from .common import State, Parameter, Rand, get_true_fields,get_true_field, Timer
from .flow import init_flow, Flow

class Fault(dataobject, readonly=True, mapping=True):
    """
    Stores Default Attributes for for modes to use in Mode.faultmodes
    """
    dist:       float = 1.0
    oppvect:    dict = {"operating":1.0}
    rcost:      float = 0.0
    probtype:   str = 'rate'
    units:      str = 'hr'

class Mode(dataobject, readonly=False):
    """
    faults : set
        Set of faults present (or not) at any given time
    mode : str
        Name of the current mode. the default is 'nominal'
    opermodes : tuple
        Names of non-faulty operational modes.
    failrate : float
        Overall failure rate for the block. The default is 1.0.
    faultmodes : dict 
            Dictionary/Set of faultmodes, which can have the forms:
                - set {'fault1', 'fault2', 'fault3'} (just the respective faults)
                - dict {'fault1': faultattributes, 'fault2': faultattributes}, where faultattributes is:
                    - float: rate for the mode
                    - dict/set/str: opportunity vector for the mode specified as a dictionary/set/string
                    - tuple: (rate, oppvect, rcost) or (rate, rcost)
                        - a list of arguments where the float arguments are specified in the order rate, rcost (if provided) and
                            an oppvect opportunity vector is provided (anywhere) with the form:
                                -list: [float1, float2,...], a vector of relative likelihoods for each phase, or
                                -dict: {opermode:float1, opermode:float1}, a dict of relative likelihoods for each phase/mode
                                -set: {opermode, opermode,...}, a set of applicable phases (assumed equally likely).
                                the phases/modes to key by are defined in "key_phases_by"
                                -str: 'all'/'modename', either specifying all operational modes/phases or a single operational mode/phase
    probtype : str, optional
        Type of probability in the probability model, a per-time 'rate' or per-run 'prob'. 
        The default is 'rate'
    units : str, optional
        Type of units ('sec'/'min'/'hr'/'day') used for the rates. Default is 'hr' 
    exclusive : True/False
        Whether fault modes are exclusive of each other or not. Default is False (i.e. more than one can be present). 
    key_phases_by : 'self'/'none'/'global'/'fxnname'
        Phases to key the faultmodes by (using local, global, or an external function's modes'). Default is 'global'
    longnames : dict
        Longer names for the faults (if desired). {faultname: longname}
    """
    mode:               str = 'nominal'
    faults:             set = set()
    faultmodes:         dict = {}
    mode_state_dict:    dict={}
    faultparams = {}
    he_args = tuple()
    opermodes = ('nominal',)
    failrate = 1.0
    probtype = 'rate'
    units = 'hr'
    units_set = ('sec','min','hr','day')
    exclusive = False
    key_phases_by = 'global'
    longnames = {}
    def __init__(self, *args, s_kwargs={}, **kwargs):
        if self.he_args:
            kwargs['failrate']=self.add_he_rate(*self.he_args)
        args = get_true_fields(self, *args, **kwargs)
        super().__init__(*args)
        if not self.mode:            self.mode='nominal'
        if not self.faults:          self.faults=set()
        if not self.faultmodes:      self.faultmodes=dict()
        if not self.mode_state_dict: self.mode_state_dict=dict()
        
        if 's' in self.__fields__:
            self.s.set_atts(**s_kwargs)
        
        self.init_faultmodes()
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
        else: raise Exception("Invalid type for EPCs: "+str(type(EPCs)))
        return gtp*EPC_f
    def init_faultmodes(self):
        if self.key_phases_by=='self':  oppvect='all'
        else:                           oppvect=[1.0]
        default_kwargs={'dist':1.0/max(len(self.faultparams),1),
                        'probtype': self.probtype,
                        'units': self.units,
                        'oppvect': oppvect}
        for mode in self.faultparams:
            if type(self.faultparams) == tuple:   
                kwargs={**default_kwargs}
            elif type(self.faultparams[mode]) == float:   # dict of modes: dist, where dist is the distribution (or individual rate/probability)
                kwargs={**{**default_kwargs,'dist':self.faultparams[mode]}}
            elif type(self.faultparams[mode]) == dict:   
                kwargs = {**{**default_kwargs,'dist':self.faultparams[mode], **self.faultparams[mode]}}
            elif type(self.faultparams[mode]) == tuple: # provided list with oppvect, dist, rcost (rcost always after dist)
                if len(self.faultparams[mode])==2:
                    kwargs = {['dist', 'rcost'][i]:val for i,val in enumerate(self.faultparams[mode])}
                else:
                    kwargs = {['dist', 'oppvect', 'rcost'][i]:val for i,val in enumerate(self.faultparams[mode])}
                kwargs = {**{**default_kwargs,'dist':self.faultparams[mode], **kwargs}}
            else:   
                raise Exception("Invalid mode definition in "+str(self.__class__)+", "+mode+" modeparams values should be float/dict/tuple")
            
            if kwargs['oppvect']=='all': 
                kwargs['oppvect'] = {*self.opermodes}
            if type(kwargs['oppvect'])==set:            
                kwargs['oppvect'] = {o:1.0 for o in kwargs['oppvect']}
            self.faultmodes[mode] = Fault(**kwargs)
    def assoc_faultstates(self, franges = {}, mode_app = 'none', manual_modes={}, probtype='prob', units='hr', key_phases_by='global', seed=42):
        """
        Associates modes with given faultstates.

        Parameters
        ----------
        franges : dict, optional
            Dictionary of form {'state':{val1, val2...}) of ranges for each health state (if used to generate modes). The default is {}.
        mode_app : str
            type of modes to elaborate from the given health states.
        manual_modes : dict, optional
            Dictionary/Set of faultmodes with structure, which has the form:
                - dict {'fault1': [atts], 'fault2': atts}, where atts may be of form:
                    - states: {state1: val1, state2, val2}    
                    - [states, faultattributes], where faultattributes is the same as in assoc_modes
        probtype : str, optional
            Type of probability in the probability model, a per-time 'rate' or per-run 'prob'. 
            The default is 'rate'
        units : str, optional
            Type of units ('sec'/'min'/'hr'/'day') used for the rates. Default is 'hr' 
        """
        nom_fstates = {state: self.s.__defaults__[self.s.__fields__.index(state)] for state in franges}
        if mode_app=='none': a=0
        elif mode_app=='single-state':
            for state in franges:
                modes = {state+'_'+str(value):'synth' for value in franges[state]}
                modestates = {state+'_'+str(value): {state:value} for value in franges[state]}
                self.faultmodes.update(modes)
                self.mode_state_dict.update(modestates)
        elif mode_app =='all' or type(mode_app)==int:
            for state in franges: franges[state].add(nom_fstates[state])
            nomvals = tuple([*nom_fstates.values()])
            statecombos = [i for i in itertools.product(*franges.values()) if i!=nomvals]
            if type(mode_app)==int and len(statecombos)>0: 
                rng = np.random.default_rng(seed)
                sample = rng.choice([i for i,_ in enumerate(statecombos)], size=mode_app, replace=False)
                statecombos = [statecombos[i] for i in sample]
            self.faultmodes.update({'hmode_'+str(i):'synth' for i in range(len(statecombos))}) 
            self.mode_state_dict.update({'hmode_'+str(i): {list(franges)[j]:state for j, state in enumerate(statecombos[i])} for i in range(len(statecombos))})
        else: raise Exception("Invalid mode elaboration approach")

        for mode,atts in manual_modes.items():
            if type(atts)==list:
                self.mode_state_dict.update({mode:atts[0]})
                if not getattr(self, 'exclusive', False): print("Changing fault mode exclusivity to True")
                self.assoc_modes(faultmodes={mode:atts[1]}, initmode=getattr(self,'mode', 'nom'), probtype=probtype, proptype=probtype, exclusive=True, key_phases_by=key_phases_by)
            elif  type(atts)==dict:
                self.mode_state_dict.update({mode:atts})
                self.faultmodes.update({mode:'synth'})
    def set_mode(self, mode):
        """Sets a mode in the block
        
        Parameters
        ----------
        mode : str
            name of the mode to enter.
        """
        if self.exclusive:
            if self.any_faults():           raise Exception("Cannot set mode from fault state without removing faults.")
            elif  mode in self.faultmodes:  self.to_fault(mode)
            else:                           self.mode=mode
        else:                               self.mode = mode
    def in_mode(self,*modes):
        """Checks if the system is in a given operational mode
        
        Parameters
        ----------
        *modes : strs
            names of the mode to check
        """
        return self.mode in modes                    
    def has_fault(self,*faults): 
        """Check if the block has fault (a str)
        
        Parameters
        ----------
        *faults : strs
            names of the fault to check.
        """
        return any(self.faults.intersection(set(faults)))
    def no_fault(self,fault): 
        """Check if the block does not have fault (a str)
        
        Parameters
        ----------
        fault : str
            name of the fault to check.
        """
        return not(any(self.faults.intersection(set([fault]))))
    def any_faults(self):
        """check if the block has any fault modes"""
        return any(self.faults)
    def to_fault(self,fault): 
        """Moves from the current fault mode to a new fault mode
        
        Parameters
        ----------
        fault : str
            name of the fault mode to switch to
        """
        self.faults.clear()
        self.faults.add(fault)
        if self.exclusive: self.mode = fault
    def add_fault(self,*faults): 
        """Adds fault (a str) to the block
        
        Parameters
        ----------
        *fault : str(s)
            name(s) of the fault to add to the black
        """
        self.faults.update(faults)
        if self.exclusive: 
            if len(faults)>1:   raise Exception("Multiple fault modes added to function with exclusive fault representation")
            elif len(faults)==0 and self.mode in self.faultmodes: 
                raise Exception("In "+str(self.__class__)+"--no faults but mode is still in faultmode "+self.mode)
            elif faults: 
                self.mode=faults[0]
    def replace_fault(self, fault_to_replace,fault_to_add): 
        """Replaces fault_to_replace with fault_to_add in the set of faults
        
        Parameters
        ----------
        fault_to_replace : str
            name of the fault to replace
        fault_to_add : str
            name of the fault to add in its place
        """
        self.faults.add(fault_to_add)
        self.faults.remove(fault_to_replace)
        if self.exclusive: self.mode = fault_to_add
    def remove_fault(self, fault_to_remove, opermode=False, warnmessage=False):
        """Removes fault in the set of faults and returns to given operational mode
        
        Parameters
        ----------
        fault_to_replace : str
            name of the fault to remove
        opermode : str (optional)
            operational mode to return to when the fault mode is removed
        warnmessage : str/False
            Warning to give when performing operation. Default is False (no warning)
        """
        self.faults.discard(fault_to_remove)
        if opermode:    self.mode = opermode
        if self.exclusive and not(opermode):
            raise Exception("Unclear which operational mode to enter with fault removed")
        if warnmessage: self.warn(warnmessage,"Fault mode `"+fault_to_remove+"' removed.", stacklevel=3)
    def remove_any_faults(self, opermode=False, warnmessage=False):
        """Resets fault mode to nominal and returns to the given operational mode
        
        Parameters
        ----------
        opermode : str (optional)
            operational mode to return to when the fault mode is removed
        warnmessage : str/False
            Warning to give when performing operation. Default is False (no warning)
        """
        self.faults.clear()
        if opermode:    self.mode = opermode
        else:           self.mode = self.__defaults__[self.__fields__.index('mode')]
        if self.exclusive and not(self.mode):
            raise Exception("In "+str(self.__class__)+": Unclear which operational mode to enter with fault removed--no default or opermode specified")
        if warnmessage: self.warn(warnmessage, "All faults removed.")
    def mirror(self, mode_to_mirror):
        self.mode = mode_to_mirror.mode
        self.faults.clear()
        self.faults.update(mode_to_mirror.faults)
    def get_true_field(self, fieldname, *args, **kwargs):
        return get_true_field(self, fieldname, *args, **kwargs)
    def get_true_fields(self, *args, **kwargs):
        return get_true_fields(self, *args, **kwargs)

def assoc_flows(obj, flows={}):
    """
    Associates flows with with the given object (Block, ASG, etc.) 
    
    Flows must be defined with the _init_ class variable pointing to the class
    to initialize (e.g., _init_flowname = FlowClass).
    
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
    for init_att in dir(obj):
        if init_att.startswith("_init_"):
            att = getattr(obj, init_att)
            attname = init_att[6:]
            if inspect.isclass(att) and issubclass(att, Flow) and not(attname in obj.flows):
                if attname in flows:    obj.flows[attname]=flows.pop(attname)
                else:                   obj.flows[attname]=att(attname)
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
    if isinstance(obj, ASG):        compdict = obj.actions
    elif isinstance(obj, CompArch): compdict=obj.components
    else: raise Exception("Invalid object type: "+type(obj)+" should be ASG or CompArch") 

    for fault in faults:
        if fault in obj.faultmodes:
            comp = compdict[obj.faultmodes[fault]]
            comp.m.add_fault(fault[len(comp.name)+1:])


class Block():
    __slots__ = ['p', '_args_p', 's', '_args_s','m', '_args_m', 'r', '_args_r', '__dict__']
    _init_p = Parameter
    _init_s = State
    _init_m = Mode
    _init_r = Rand
    flows = dict()
    """ 
    Superclass for FxnBlock and Component subclasses. Has functions for model setup, querying state, reseting the model
    
    Attributes
    ----------
    s : State
        Internal State of the block. Instanced from _init_s.
    p : Parameter
        Internal Parameter for the block. Instanced from _init_p
    r : Rand
        Internal Rand for the block. Instanced from _init_r
    time : float
        internal time of the function
    """
    def __init__(self, name, flows={}, s={}, p={}, m={}, r={}):
        """
        Instance superclass. Called by FxnBlock and Component classes.

        Parameters
        ----------
        s : dict, optional
            Initial state values for the states s in the block. The default is {}.
        p : dict, optional
            Overriding parameter values for the parameters p in the block. The default is {}.
        """
        self.flows = dict()
        assoc_flows(self, flows=flows)
        for at in ['s','p','m','r']:
            at_arg = eval(at)
            if type(at_arg)!=dict: at_arg = asdict(at_arg)
            setattr(self, '_args_'+at, at_arg)
            init_at = getattr(self, '_init_'+at)
            setattr(self, at, init_at(**at_arg))
        #self._args_s = s
        #self._args_p = p
        #self._args_m = m
        #self._args_r = r
        #self.p=self._init_p(**p)
        #self.s=self._init_s(**s)
        #self.m=self._init_m(**m)
        #self.r=self._init_r(**r)
        self.update_seed()
        
        self.name=name
        self.time=0.0
    def __repr__(self):
        if hasattr(self,'name'):
            return getattr(self, 'name', '')+' '+self.__class__.__name__+' '+getattr(self,'type', '')+': '+str(self.return_states())
        else: return 'New uninitialized '+self.__class__.__name__
    def update_seed(self, seed=[]):
        """
        Updates seed and propogates update to contained actions/components.
        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        if seed: self.r.seed=seed
        
        if hasattr(self, 'c'): self.c.update_seed(self.seed)
        if hasattr(self, 'actions'):
            for act in self.actions.values():
                act.update_seed(self.seed)
    def get_rand_states(self, auto_update_only=False):
        """Gets dict of random states from block and associated actions/components"""
        rand_states = self.r.get_rand_states(auto_update_only)
        if hasattr(self, 'c'): rand_states.update(self.c.get_rand_states(auto_update_only=auto_update_only))
        if hasattr(self, 'actions'):
            for actname, act in self.actions.items():
                if act.get_rand_states(auto_update_only=auto_update_only): 
                    rand_states[actname] = act.get_rand_states(auto_update_only=auto_update_only)
        return rand_states
    def add_params(self, *params):
        """Adds given dictionary(s) of parameters to the function/block.
        e.g. self.add_params({'x':1,'y':1}) results in a block where:
            self.x = 1, self.y = 1
        """
        for param in params:
            for attr, val in param.items():
                setattr(self, attr, val)
    def set_timestep(self, use_local=True, local_tstep=None, global_tstep=1.0):
        """Sets the timestep of the function given the options use_local 
        (which selects whether it uses local_timestep or global_timestep)"""
        global_tstep=Decimal(str(global_tstep))
        if use_local:
            if local_tstep:             dt=Decimal(str(local_tstep)) 
            elif hasattr(self, 'dt'):   dt=Decimal(str(self.dt)) 
            else:                       dt=Decimal(str(global_tstep))
            if dt < global_tstep:
                if global_tstep%dt:
                    raise Exception("Local timestep: "+str(self.dt)+" doesn't line up with global timestep: "+str(global_tstep))
            else:
                if dt%global_tstep:
                    raise Exception("Local timestep: "+str(self.dt)+" doesn't line up with global timestep: "+str(global_tstep))
            self.run_times = int(global_tstep/dt)
        else:   dt=global_tstep; self.run_times=1
        self.dt = float(dt)
    def assoc_timers(self, *timers):
        """Associates timer objects with the given function/block"""
        if not getattr(self, 'timers', False): self.timers=set()
        self.timers.update(timers)
        for timername in timers:
            setattr(self, timername, Timer(timername))
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
        elif default=='first':      self.m.add_fault(faults[0])
        elif type(default)==str:    self.m.add_fault(default)
        else:                       self.m.add_fault(*default)
    def get_flowtypes(self):
        """Returns the names of the flow types in the model"""
        return {obj.type for name, obj in self.flows.items()}
    def reset(self):            #reset requires flows to be cleared first
        """ Resets the block to the initial state with no faults. Used by default in 
        derived objects when resetting the model. Requires associated flows to be cleared first."""
        self.m.remove_any_faults()
        self.s=self._init_s(**self._args_s)
        self.r=self._init_r(**self._args_r)
        if hasattr(self, 'time'): self.time=0.0
        if hasattr(self, 'dt'): self.dt=self.dt
        if hasattr(self, 'internal_flows'):
            for flowname, flow in self.internal_flows.items():
                flow.reset()
        if self.type=='function':
            if hasattr(self, 'c'): self.c.reset()
            for timername in self.timers:
                getattr(self, timername).reset()
            self.updatefxn('reset', faults=[], time=0)
    def get_memory(self):
        """ Gets the approximate memory usage of the block in bytes (not complete)"""
        mem=0
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
        if hasattr(self, 'components'):
            for name,comp in self.c.components.items():
                mem+=comp.get_memory()
        if hasattr(self, 'actions'):
            for name,comp in self.actions.items():
                mem+=comp.get_memory()
        for state in asdict(self.s):
            mem+=2*sys.getsizeof(state) # (*2 because both the initstate and the actual state should be counted)
        return mem
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
        if hasattr(self, 's'):
            for state, val in asdict(self.s).items():
                if type(val) in [set, dict]: val=copy.deepcopy(val)
                states[state]= val
        states['mode']=self.m.mode
        return states, self.m.faults.copy()
    def has_new_states(self, prev_states, prev_faults):
        states, faults = self.return_states()
        if prev_states!=states or prev_faults!=faults:  return True
        else:                                           return False
    def return_probdens(self):
        state_pd = self.r.return_probdens()
        if hasattr(self, 'components'): 
            for compname, comp in self.c.components:
                state_pd*=comp.return_probdens()
        if hasattr(self, 'actions'):
            for actionname, action in self.a.actions:
                state_pd*=action.return_probdens()
        return state_pd

## COMPONENT/COMPONENT ARCHITECTURES
class Component(Block):
    """
    Superclass for components (most attributes and methods inherited from Block superclass)
    """
    def __init__(self,name, s={}, p={}):
        """
        Inherit the component class

        Parameters
        ----------
        name : str
            Unique name ID for the component
        s : dict, optional
            States to use in the component. The default is {}.
        p : dict, optional
            Parameters to use in the component. The default is {}.
        """
        self.type = 'component'
        super().__init__(name, p=p, s=s)
    def behavior(self,time):
        """ Placeholder for component behavior methods. Enables one to include components
        without yet having a defined behavior for them."""
        return 0
class CompArch(dataobject, mapping=True):
    """Container for holding component architectures"""
    archtype:       str = 'default'
    components:     dict = dict()
    faultmodes:     dict = dict()
    def make_components(self, CompClass, *args, **kwargs):
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
        if self.components is None: self.components=dict()
        if self.faultmodes is None: self.faultmodes=dict()
        
        for arg in args:
            if arg in kwargs:   kwargs_comp = kwargs[arg]
            else:               kwargs_comp = kwargs
            self.components[arg]=CompClass(arg, **kwargs_comp)
            self.faultmodes.update({self.components[arg].name+'_'+modename:arg 
                               for modename in self.components[arg].m.faultmodes})
    def copy_with_arg(self, **kwargs):
        cop = self.__class__(**kwargs)
        for compname, component in self.components.items(): #TODO: needs to cover all attributes, copy should a part of Block 
            cop_comp = cop.components[compname]
            cop_comp.s = cop_comp._init_s(**asdict(component.s))
            cop_comp.m.mirror(component.m)
            cop_comp.time=component.time
        return cop
    def inject_fault_in_component(self, fault):
        if fault in self.faultmodes:
            component = self.components[self.faultmodes[fault]]
            component.m.add_fault(fault[len(component.name):]+1)
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
    
## Actions/ASGs
class Action(Block):
    """
    Superclass for actions (most attributes and methods inherited from Block superclass)
    """
    def __init__(self,name, flows={}, s={}, p={}):
        """
        Inherit the Block class

        Parameters
        ----------
        name : str
            Unique name ID for the action
        s : dict, optional
            States to use in the action. The default is {}.
        p : dict, optional
            Parameters to use in the component. The default is {}.
        """
        self.type = 'action'
        self.name = name
        self.t_loc=0.0 # local time find a place for this?
        super().__init__(name, flows=flows, p=p, s=s)
    def updateact(self, time=0, run_stochastic=False, proptype='dynamic', dt=1.0):
        """
        Updates the behaviors, faults, times, etc of the action 

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool
            Whether to run the simulation using stochastic or deterministic behavior
        """
        if time>self.time : self.r.update_stochastic_states()
        if proptype=='dynamic':
            if self.time<time:  self.behavior(time); self.t_loc+=dt
        else:                   self.behavior(time); self.t_loc+=dt
        self.time=time
    def behavior(self, time):
        """Placeholder behavior method for actions"""
        a=0
class ASG(dataobject, mapping=True):
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
    actions:            dict = {}
    action_graph:       nx.DiGraph = nx.DiGraph()
    flow_graph:         nx.Graph = nx.Graph()
    conditions:         dict = {}
    faultmodes:         dict = {}
    flows:              dict = {}
    active_actions:     set = {}
    pos:                dict = {}
    initial_action="auto" 
    state_rep="finite-state" 
    max_action_prop="until_false" 
    proptype='dynamic' 
    per_timestep=False
    def __init__(self, *args, flows={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions={} #TODO: remove restatement of defaults when fixed in recordclass
        self.action_graph= nx.DiGraph()
        self.flow_graph=nx.Graph()
        self.conditions={}
        self.faultmodes= {}
        self.flows = {}
        assoc_flows(self, flows=flows)
        self.active_actions = set()
    def build(self): #TODO: implement as post-__init__??
        if self.initial_action=='auto': 
            initial_action = [act for act, in_degree  in self.action_graph.in_degree if in_degree==0]
            if not initial_action: raise Exception("Cannot set initial action--no starting node")
        elif type(self.initial_action)==str: 
            initial_action=[self.initial_action]
        self.set_active_actions(initial_action)
        if self.state_rep=='finite-state' and len(self.active_actions)>1: 
            raise Exception("Cannot have more than one initial action with finite-state representation")
        if not self.pos: self.pos = nx.planar_layout(nx.compose(self.flow_graph, self.action_graph))
    def add_flow(self,flowname, fclass=Flow, p={}, s={}, flowtype=''):
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
        flowtype : str, optional
            Denotes type for class (e.g. 'energy,' 'material,', 'signal')
        """
        if not getattr(self, 'is_copy', False):
            self.flows[flowname] = init_flow(flowname,fclass, p=p, s=s, flowtype=flowtype) 
    def add_act(self, name, actclass, *flownames, duration=0.0, **params):
        """
        Associate an Action with the Function Block for use in the Action Sequence Graph

        Parameters
        ----------
        name : str
            Internal Name for the Action
        action : Action
            Action class to instantiate
        *flows : flow
            Flows (optional) which connect the actions
        **params : any
            parameters to instantiate the Action with. 
        """
        flows = {fl:self.flows[fl] for fl in flownames}
        action = actclass(name, flows={**flows}, **params)
        self.actions[name] = action
        self.actions[name].duration=duration
        self.action_graph.add_node(name)
        self.flow_graph.add_node(name, bipartite=0)
        for flow in flows:
            self.flow_graph.add_node(flow, bipartite=1)
            self.flow_graph.add_edge(name,flow)
            
        modes_to_add = {action.name+'_'+f:val for f,val in action.m.faultmodes.items()}
        fmode_intersect = set(modes_to_add).intersection(self.faultmodes)
        if any(fmode_intersect):
            raise Exception("Action "+name+" overwrites existing fault modes: "+str(fmode_intersect)+". Rename the faults")
        self.faultmodes.update({action.name+'_'+modename:name for modename in action.m.faultmodes})
    def cond_pass(self):
        return True
    def add_cond(self, start_action, end_action, name='auto',condition='pass'):
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
        if name=='auto': name = str(len(self.conditions)+1)
        if condition=='pass': condition = self.cond_pass
        self.conditions[name] = condition
        self.action_graph.add_edge(start_action, end_action, **{'name':name, name:'name', 'arrow':True})
    def set_active_actions(self, actions):
        """Helper method for setting given action(s) as active"""
        if type(actions)==str: 
            if actions in self.actions: actions = [actions]
            else: raise Exception("initial_action="+actions+" not in self.actions: "+str(self.actions))
        if type(actions)==list:
            self.active_actions = set(actions)
            if any(self.active_actions.difference(self.actions)): raise Exception("Initial actions not associated with model: "+str(self.active_actions.difference(self.actions)))
        else: raise Exception("Invalid option for initial_action")
    def show(self, gtype='combined', with_cond_labels=True, pos=[]):
        """
        Shows a visual representation of the internal Action Sequence Graph of the Function Block

        Parameters
        ----------
        gtype : 'combined'/'flows'/'actions'
            Gives a graphical representation of the ASG. Default is 'combined'
            - 'actions'     (for function input):    plots the sequence of actions in the function's Action Sequence Graph
            - 'flows'       (for function input):    plots the action/flow connections in the function's Action Sequence Graph
            - 'combined'    (for function input):    plots both the sequence of actions in the functions ASG and action/flow connections
        with_cond_labels: Bool
            Whether or not to label the conditions
        pos : dict
            Dictionary of node positions for actions/flows
        """
        import matplotlib.pyplot as plt; plt.rcParams['pdf.fonttype'] = 42 
        fig = plt.figure()
        if gtype=='combined':       graph = nx.compose(self.flow_graph, self.action_graph)
        elif gtype=='flows':        graph = self.flow_graph
        elif gtype=='actions':      graph = self.action_graph
        if not pos and self.pos: pos = self.pos
        else:                    pos=nx.planar_layout(nx.compose(self.flow_graph, self.action_graph))
        nx.draw(graph, pos=pos, with_labels=True, node_color='grey')
        nx.draw_networkx_nodes(self.action_graph, pos=pos, node_shape='s', node_color='skyblue')
        nx.draw_networkx_nodes(self.action_graph, nodelist=self.active_actions, pos=pos, node_shape='s', node_color='green')
        edge_labels = {(in_node, out_node): label for in_node, out_node, label in graph.edges(data='name') if label}
        if with_cond_labels: nx.draw_networkx_edge_labels(graph, pos, edge_labels)
        if gtype=='combined' or gtype=='conditions':
            nx.draw_networkx_edges(self.action_graph, pos,arrows=True, arrowsize=30, arrowstyle='->', node_shape='s', node_size=100)
        return fig
    def prop_internal(self, time, run_stochastic, proptype, dt):
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
        """
        if not self.per_timestep: 
            self.set_active_actions(self.initial_action)
            for action in self.active_actions: self.actions[action].t_loc=0.0
        if proptype==self.proptype:
            active_actions = self.active_actions
            num_prop = 0
            while active_actions:
                new_active_actions=set(active_actions)
                for action in active_actions:
                    self.actions[action].updateact(time, run_stochastic, proptype=proptype, )
                    action_cond_edges = self.action_graph.out_edges(action, data=True)
                    for act_in, act_out, atts in action_cond_edges:
                        try:
                            cond = self.conditions[atts['name']]()
                        except TypeError as e:
                            raise TypeError("Poorly specified condition "+str(atts['name'])+": ") from e
                        if  cond and getattr(self.actions[action], 'duration',0.0)+dt<=self.actions[action].t_loc:
                            self.actions[action].t_loc=0.0
                            new_active_actions.add(act_out)
                            new_active_actions.discard(act_in)
                if len(new_active_actions)>1 and self.state_rep=='finite-state':
                    raise Exception("Multiple active actions in a finite-state representation: "+str(new_active_actions))
                num_prop +=1 
                if type(self.proptype)==int and num_prop>=self.proptype:
                    break
                if new_active_actions==set(active_actions):
                    break
                else: active_actions=new_active_actions
                if num_prop>10000: raise Exception("Undesired looping in Function ASG for: "+self.name)
            self.active_actions = active_actions
    def get_faults(self):
        return {act.name+f for act in self.actions.values() for f in act.m.faults}


#Function superclass 
class FxnBlock(Block):
    slots = ["c", "_c_arg", "a", "_a_arg"]
    """
    Superclass for functions.
    
    Attributes (specific to FxnBlock--see Block glass for more)
    ----------
    type : str
        labels the function as a function (may not be necessary) Default is 'function'
    flows : dict
        flows associated with the function. structured {flow:{value:XX}}
    components : dict
        component instantiations of the function (if any)
    timers : set
        names of timers to be used in the function (if any)
    dt : float
        local timestep of the model in the function (overrides global timestep by default ('use_local':True in modelparameters))
    """
    def __init__(self,name, flows={}, params={}, p=dict(), s=dict(), c=dict(), a=dict(), r=dict(), m=dict(), timers=[],
                 local=dict(), comms=dict(), dt=1.0, seed=None):
        """
        Instantiates the function superclass with the relevant parameters.

        Parameters
        ----------
        flows :list
            Flow objects to associate with the function (from outside the FnxBlock)
        flownames : list/dict, optional
            Names of flows  to use in the function, if private flow names are needed (e.g. functions with in/out relationships).
            Either provided as a list (in the same order as the flows) of all flow names corresponding to those flows
            Or as a dict of form {External Flowname: Internal Flowname}
        p : dict, optional
            Internal parameters to override from defaults. The default is {}.
        s : dict, optional
            Internal states to override from defaults. The default is {}.
        c : dict, optional
            Internal CompArch fields/arguments override from defaults. The default is {}.
            FxnBlock must have an _init_c property.
        timers : set, optional
            Set of names of timers to use in the function. The default is {}.
        local : dict/list
            Views of MultiFlows to add instantiate local. May be of forms: 
                - {flowname:(localname, attrs)} (to only create local view of specific attributes)
                - {flowname:localname}          (to create view with all attributes)
                - [flowname1, flowname2...]     (to give overwrite the global flow with the local view of it)
        comms : dict/list
            Views of CommsFlows to add instantiate local. May be of forms: 
                - {flowname:(localname, attrs)} (to only create local view of specific attributes)
                - {flowname:localname}          (to create view with all attributes)
                - [flowname1, flowname2...]     (to give overwrite the global flow with the local view of it)
        dt : float
            Local timestep (if not inherited from model)
        seed : int/hash
            Random seed to use in model
        """
        self.type = 'function'
        self.assoc_timers(*timers)
        if dt: self.dt=dt
        
        super().__init__(name, flows=flows, p=p, s=s, r=r, m=m)
        for at in ['c', 'a']:
            at_arg = eval(at)
            at_init = getattr(self, '_init_'+at, False)
            if at_init:
                setattr(self, '_'+at+'_arg', at_arg)
                at_flows = dict()
                for flowname, flow in self.flows.items():
                    if hasattr(at_init, '_init_'+flowname): at_flows[flowname]=flow
                try:
                    if at_flows:    setattr(self, at,  at_init(flows=at_flows, **at_arg))
                    else:           setattr(self, at,  at_init(**at_arg))
                except TypeError as e:
                    invalid_args = [a for a in at_arg if a not in at_init.__fields__]
                    if invalid_args: argstr = ", Invalid args: "+', '.join(invalid_args)
                    else:            argstr=''
                    raise TypeError("Poor specification for : "+str(at_init)+" with kwargs: "+str(at_arg)+argstr) from e
                setattr(self, '_args_'+at,  at_arg)
                if at=='c':     compacts = self.c.components
                elif at=='a':   compacts = self.a.actions
                for ca in compacts.values():
                    self.m.faultmodes.update({ca.name+"_"+f:vals for f, vals in ca.m.faultmodes.items()})
            elif at_arg: 
                raise Exception(at+" argument provided: "+str(at_arg)+"without associating a CompArch/ASG to _init_"+at)
    def add_local_to_flowdict(self,flowdict, local, ftype):
        """
        Adds local flows to the flow dictionary during initialization

        Parameters
        ----------
        flowdict : dict
            Dictionary of flows {flowname:flow_object}
        local : dict/list
            Local flows to add. May be of forms: 
                - {flowname:(localname, attrs)} (to only create local view of specific attributes)
                - {flowname:localname}          (to create view with all attributes)
                - [flowname1, flowname2...]     (to give overwrite the global flow with the local view of it)
        ftype : str ('local'/'comms')
            Switches whether the flow added is to be a local or comms flow
        """
        for l in local:
            if ftype=='local':      gen_fl = flowdict[l].create_local
            elif ftype=='comms':    gen_fl = flowdict[l].create_comms
            if type(local)==dict and type(local[l]) in [list, tuple, set]:
                loc_flow = gen_fl(self.name, local[l][1])
                loc_name = local[l][0]
            elif type(local)==dict:
                loc_flow = gen_fl(self.name)
                loc_name = local[l]
            else:
                loc_flow = gen_fl(self.name)
                loc_name = l
            flowdict[loc_name]=loc_flow
    def copy(self, newflows, *attr, **kwargs):
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
        copy = self.__new__(self.__class__)  # Is this adequate? Wouldn't this give it new components?
        copy.is_copy=True
        try:
            copy.__init__(self.name, flows=newflows, **kwargs)
        except TypeError as e:
            raise Exception("Poor specification of "+str(self.__class__))  from e
        copy.m.mirror(self.m)
        if hasattr(self, 'mode_state_dict'):    copy.mode_state_dict = self.mode_state_dict
        # TODO: figure out copying for ASGs
        #for flowname, flow in self.internal_flows.items():
        #    copy.internal_flows[flowname] = flow.copy()
        #    setattr(copy, flowname, copy.internal_flows[flowname])
        #for action in self.actions: 
        #    copy.action.s = copy.action._init_s(**asdict(action.s))
        #    copy.actions[action].m.mirror(self.actions[action].m)
        #    copy.actions[action].time=self.actions[action].time
        setattr(copy, 'active_actions', getattr(self, 'active_actions', {}))
        if hasattr(self, 'c'): copy.c = self.c.copy_with_arg(**self._c_arg)
        for timername in self.timers:
            timer = getattr(self, timername)
            copytimer = getattr(copy, timername)
            copytimer.set_timer(timer.time, tstep=timer.tstep)
            copytimer.mode=timer.mode
        copy.s.assign(self.s)
        copy.r.assign(self.r)
        if hasattr(self, 'time'): copy.time=self.time
        if hasattr(self, 'dt'): copy.dt=self.dt
        if hasattr(self, 'run_times'): copy.run_times=self.run_times
        return copy
    def update_modestates(self):
        """Updates states of the model associated with a specific fault mode (see assoc_modes)."""
        num_update = 0
        for fault in self.faults:
            if fault in self.mode_state_dict:
                for state, value in self.mode_state_dict[fault].items():
                    setattr(self, state, value)
                num_update+=1
                if num_update > 1: raise Exception("Exclusive fault mode scenarios present at the same time")
    def updatefxn(self,proptype, faults=[], time=0, run_stochastic=False):
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
        self.r.run_stochastic=run_stochastic
        if faults: self.m.add_fault(*faults)  #if there is a fault, it is instantiated in the function
        if hasattr(self, 'mode_state_dict') and any(faults): self.update_modestates()
        if hasattr(self, 'condfaults'):    self.condfaults(time)    #conditional faults and behavior are then run
        if time>self.time: self.r.update_stochastic_states()
        if hasattr(self, 'c'):    
            inject_faults_internal(self.c, faults)
        if hasattr(self, 'a'): 
            inject_faults_internal(self.a, faults)
            try:
                self.a.prop_internal(time, run_stochastic, proptype, self.dt)
            except TypeError as e:
                raise Exception("Poorly specified ASG: "+str(self.a.__class__)) from e
        
        if proptype=='static' and hasattr(self,'behavior'):        self.behavior(time)     #generic behavioral methods are run at all steps
        if proptype=='static' and hasattr(self,'static_behavior'): self.static_behavior(time)
        elif proptype=='dynamic' and hasattr(self,'dynamic_behavior') and time > self.time: 
            if self.run_times>=1:
                for i in range(self.run_times):
                    self.dynamic_behavior(time)
            elif not Decimal(str(time))%Decimal(str(self.dt)):
                self.dynamic_behavior(time)
        elif proptype=='reset':                                                             
            if hasattr(self,'static_behavior'):  self.static_behavior(time)
            if hasattr(self,'dynamic_behavior'): self.dynamic_behavior(time)
        
        actions = getattr(self, 'a', {'actions':{}})['actions']
        if actions:     # propagate faults from component level to function level
            self.m.faults.difference_update(self.a.faultmodes)
            self.m.faults.update(self.a.get_faults())
        comps = getattr(self, 'c', {'components':{}})['components']
        if comps:
            self.m.faults.difference_update(self.c.faultmodes)
            self.m.faults.update(self.c.get_faults())
        self.time=time
        if run_stochastic=='track_pdf': self.probdens = self.r.return_probdens()
        if (self.m.exclusive==True and len(self.m.faults)>1): 
            raise Exception("More than one fault present in "+self.name+"\n at t= "+str(time)+"\n faults: "+str(self.m.faults)+"\n Is the mode representation nonexclusive?")
        return
class GenericFxn(FxnBlock):
    """Generic function block. For use when the user has not yet defined a class for the
    given (to be implemented) function block. Acts as a placeholder that enables simulation."""
    def __init__(self, name, flows):
        super().__init__(name, flows)






