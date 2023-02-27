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
from scipy import stats
from recordclass import dataobject, asdict

from .common import State, Parameter
from .flow import init_flow, Flow

class Fault(dataobject, readonly=True, mapping=True):
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
    mode:       str = 'nominal'
    faults:     set = set()
    opermodes = ('nominal',)
    failrate = 1.0
    faultparams = {}
    faultmodes = {}
    probtype = 'rate'
    units = 'hr'
    units_set = ('sec','min','hr','day')
    exclusive = False
    key_phases_by = 'global'
    longnames = {}
    def __init__(self, *args, **kwargs):
        kwargs = {'mode':'nominal', 'faults':set(), **kwargs}
        super().__init__(*args, **kwargs)
        self.init_faultmodes()
    def init_faultmodes(self):
        if self.key_phases_by=='self':   
            self.key_phases_by = self.name
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
            elif len(faults)==0: self.mode = 'nominal'
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
        if self.exclusive and not(opermode):
            raise Exception("Unclear which operational mode to enter with fault removed")
        if warnmessage: self.warn(warnmessage, "All faults removed.")
    def mirror(self, mode_to_mirror):
        self.mode = mode_to_mirror.mode
        self.faults.clear()
        self.faults.update(mode_to_mirror.faults)
    
    

class Block():
    __slots__ = ['p', '_args_p', 's', '_args_s', '__dict__']
    _init_p = Parameter
    _init_s = State
    _init_m = Mode
    """ 
    Superclass for FxnBlock and Component subclasses. Has functions for model setup, querying state, reseting the model
    
    Attributes
    ----------
    s : State
        Internal State of the block. Instanced from _init_s.
    p : Parameter
        Internal Parameter for the block. Instanced from _init_p
    failrate : float
        Failure rate for the block
    time : float
        internal time of the function
    faults : set
        faults currently present in the block. If the function is nominal, set is {}
    faultmodes : dict
        faults possible to inject in the block and their properties. Has structure:
            - faultname :
                - dist : (float of % failures due to this fualt)
                - oppvect : (list of relative probabilities of the fault occuring in each phase)
                - rcost : cost of repairing the fault
    opermodes : list
        possible modes for the block to enter
    rngs : dict
        dictionary of random number generators for random states
    seed : int
        seed sequence for internal random number generator
    mode : string
        current mode of block operation
    """
    def __init__(self, s={}, p={}, m={}):
        """
        Instance superclass. Called by FxnBlock and Component classes.

        Parameters
        ----------
        s : dict, optional
            Initial state values for the states s in the block. The default is {}.
        p : dict, optional
            Overriding parameter values for the parameters p in the block. The default is {}.
        """
        self._args_s = s
        self._args_p = p
        self.p=self._init_p(**p)
        self.s=self._init_s(**s)
        self.m=self._init_m(**m)
        
        # TODO : create class for modes (ideally with checking)
        self.localname=''
        self.update_seed()
        self.time=0.0
    def __repr__(self):
        if hasattr(self,'name'):
            return getattr(self, 'name', '')+' '+self.__class__.__name__+' '+getattr(self,'type', '')+': '+str(self.return_states())
        else: return 'New uninitialized '+self.__class__.__name__
    def update_seed(self, seed=[]):
        """
        Updates/initializes seeds for the random states in the block and its actions/components.
        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        self.rngs=getattr(self, 'rngs', {})
        self._rng_params=getattr(self, '_rng_params', {})
        if seed:    self.seed=seed
        elif not getattr(self, 'seed', []): 
            self.seed=np.random.SeedSequence.generate_state(np.random.SeedSequence(),1)[0]
        self.rng=np.random.default_rng(self.seed)
        for rng_name in self.rngs:
            seed = self.rng.integers(np.iinfo(np.int32).max)
            self.rngs[rng_name]=np.random.default_rng(seed)
            self._rng_params[rng_name]=(*self._rng_params[rng_name][:3],seed)
        if hasattr(self, 'components'):
            for comp in self.components.values():
                comp.update_seed(self.seed)
        if hasattr(self, 'actions'):
            for act in self.actions.values():
                act.update_seed(self.seed)
    def get_rand_states(self, auto_update_only=False):
        """Gets dict of random states from block and associated actions/components"""
        if auto_update_only: rand_states = {state:vals for state,vals in self._rng_params.items() if vals[1]}
        else: rand_states=self._rng_params
        
        if hasattr(self, 'components'):
            for compname, comp in self.components.items():
                if comp.get_rand_states(auto_update_only=auto_update_only): 
                    rand_states[compname] = comp.get_rand_states(auto_update_only=auto_update_only)
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
    def assoc_rand_states(self, *states):
        """
        Associates multiple random states with the model

        Parameters
        ----------
        *states : tuple
            can give any number of tuples for each of the states. 
            The tuple is of the form (name, default), where:
                name : str
                    name for the parameter to use in the model behavior.
                default : int/float/str/etc
                    Default value for the parameter 
        """
        if type(states[0])==tuple:
            for state in states:
                self.assoc_rand_state(*state)
        else: self.assoc_rand_state(*states)
    def assoc_rand_state(self,name,default, seed=None, auto_update=[]):
        """
        Associate a stochastic state with the Block. Enables the simulation of stochastic behavior over time.

        Parameters
        ----------
        name : str
            name for the parameter to use in the model behavior.
        default : int/float/str/etc
            Default value for the parameter for the parameter
        seed : int
            seed for the random state generator to use. Defaults to None.
        auto_update : list, optional
            If given, updates the state with the given numpy method at each time-step.
            List is made up of two arguments:
            - generator_method : str
                Name of the numpy random method to use. 
                see: https://numpy.org/doc/stable/reference/random/generator.html
            - generator_params : tuple
                Parameter inputs for the numpy generator function
        """
        if not auto_update: generator_method, generator_params= None,None
        else:               generator_method, generator_params = auto_update
        if not hasattr(self,'_states'): raise Exception("Call __init__ method for function first")
        self._states.append(name)
        self._initstates[name]=default
        setattr(self, name,default)
        if not seed: seed = self.rng.integers(np.iinfo(np.int32).max)
        if not hasattr(self,'rngs'):         self.rngs={name:np.random.default_rng(seed)} 
        else:                                 self.rngs[name]=np.random.default_rng(seed)
        if not hasattr(self,'_rng_params'):   self._rng_params={name:(default, generator_method, generator_params,seed)} 
        else:                                 self._rng_params[name]=(default, generator_method, generator_params,seed)
    def assoc_faultstates(self, fstates, mode_app='single-state', probtype='prob', units='hr'):
        """
        Adds health state attributes to the model (and a mode approach if desired). 
        
        Parameters
        ----------
        fstates : Dict
            Health states to incorporate in the model and their respective values. 
            e.g., {'state':[1,{0,2,-1}]}, {'state':{0,2,-1}}
        mode_app : str
            type of modes to elaborate from the given health states.
        """
        if not hasattr(self,'_states'): raise Exception("Call __init__ method for function first")
        franges = dict.fromkeys(fstates.keys())
        nom_fstates = {}
        for state in fstates:
            self._states.append(state)
            if type(fstates[state]) in [set, np.ndarray]:               
                nom_fstates[state] = 1.0
                franges[state]=set(fstates[state])
            elif  type(fstates[state])==list:           
                nom_fstates[state] = fstates[state][0]
                franges[state]=set(fstates[state][1]) 
            elif type(fstates[state]) in [float, int]:  
                nom_fstates[state] = fstates[state]
                franges[state]={}
            else: raise Exception("Invalid input option for health state")
            setattr(self, state, nom_fstates[state])
            self._initstates.update(nom_fstates)
        self.assoc_faultstate_modes(franges=franges, mode_app=mode_app, probtype=probtype, units=units)
    def assoc_faultstate_modes(self, franges = {}, mode_app = 'none', manual_modes={}, probtype='prob', units='hr', key_phases_by='global'):
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
        if not getattr(self, 'is_copy', False):
            if not getattr(self, 'faultmodes', []): self.faultmodes = dict()
            if not getattr(self, 'mode_state_dict', False): self.mode_state_dict = {}
            nom_fstates = {state: self._initstates[state] for state in franges}
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
                    sample = self.rng.choice([i for i,_ in enumerate(statecombos)], size=mode_app, replace=False)
                    statecombos = [statecombos[i] for i in sample]
                self.faultmodes.update({'hmode_'+str(i):'synth' for i in range(len(statecombos))}) 
                self.mode_state_dict.update({'hmode_'+str(i): {list(franges)[j]:state for j, state in enumerate(statecombos[i])} for i in range(len(statecombos))})
            else: raise Exception("Invalid mode elaboration approach")
            num_synth_modes = len(self.mode_state_dict)
            for mode,atts in manual_modes.items():
                if type(atts)==list:
                    self.mode_state_dict.update({mode:atts[0]})
                    if not getattr(self, 'exclusive', False): print("Changing fault mode exclusivity to True")
                    self.assoc_modes(faultmodes={mode:atts[1]}, initmode=getattr(self,'mode', 'nom'), probtype=probtype, proptype=probtype, exclusive=True, key_phases_by=key_phases_by)
                elif  type(atts)==dict:
                    self.mode_state_dict.update({mode:atts})
                    self.faultmodes.update({mode:'synth'})
                    num_synth_modes+=1
        if not hasattr(self,'key_phases_by'): self.key_phases_by=key_phases_by
        elif getattr(self, 'key_phases_by', '')!=key_phases_by: 
            print("Changing key_phases_by to "+key_phases_by)
            self.key_phases_by=key_phases_by
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
        self.failrate = gtp*EPC_f
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
        if getattr(self, 'run_stochastic', True):
            faults = [list(x) for x in itertools.combinations(faults, combinations)]
            self.m.add_fault(*self.rng.choice(faults))
        elif default=='first':      self.m.add_fault(faults[0])
        elif type(default)==str:    self.m.add_fault(default)
        else:                       self.m.add_fault(*default)
    def set_rand(self,statename,methodname, *args):
        """
        Update the given random state with a given method and arguments (if in run_stochastic mode)

        Parameters
        ----------
        statename : str
            name of the random state defined in assoc_rand_state(s)
        methodname : 
            str name of the numpy method to call in the rng
        *args : args
            arguments for the numpy method
        """
        if getattr(self, 'run_stochastic', True):
            self.set_rand_helper(statename, methodname, *args)
    def set_rand_helper(self,statename,methodname,*args):
        """
        Update the given random state with a given method and arguments (helper function - use set_rand instead)

        Parameters
        ----------
        statename : str
            name of the random state defined in assoc_rand_state(s)
        methodname : 
            str name of the numpy method to call in the rng
        *args : args
            arguments for the numpy method
        """
        gen_method = getattr(self.rngs[statename], methodname)
        newvalue = gen_method(*args)
        setattr(self, statename, newvalue)
        if self.run_stochastic == 'track_pdf':
            value_pds = get_pdf_for_rand(newvalue, methodname, args)
            self.pds.extend(value_pds)
    def to_default(self,*statenames):
        """ Resets (given or all by default) random states to their default values
        
        Parameters
        ----------
        *statenames : str, str, str...
            names of the random state defined in assoc_rand_state(s)
        """
        if not statenames: statenames=list(self._rng_params.keys())
        for statename in statenames: setattr(self, statename, self._rng_params[statename][0])
    def get_flowtypes(self):
        """Returns the names of the flow types in the model"""
        return {obj.type for name, obj in self.flows.items()}
    def update_stochastic_states(self):
        """Updates the defined stochastic states defined to auto-update (see assoc_randstates)."""
        if self.run_stochastic == 'track_pdf': self.pds=[]
        for statename, generator in self.rngs.items():
            if self._rng_params[statename][1]:
                self.set_rand_helper(statename, self._rng_params[statename][1], *self._rng_params[statename][2])
        
    def reset(self):            #reset requires flows to be cleared first
        """ Resets the block to the initial state with no faults. Used by default in 
        derived objects when resetting the model. Requires associated flows to be cleared first."""
        self.m.remove_any_faults()
        self.s=self._init_s(**self._args_s)
        for generator in self.rngs:
            self.rngs[generator]=np.random.default_rng(self._rng_params[generator][-1])
        self.rng = np.random.default_rng(self.seed)
        if hasattr(self, 'time'): self.time=0.0
        if hasattr(self, 'dt'): self.dt=self.dt
        if hasattr(self, 'internal_flows'):
            for flowname, flow in self.internal_flows.items():
                flow.reset()
        if self.type=='function':
            for name, component in self.components.items():
                component.reset()
            for timername in self.timers:
                getattr(self, timername).reset()
            self.updatefxn('reset', faults=[], time=0)
    def get_memory(self):
        """ Gets the approximate memory usage of the block in bytes (not complete)"""
        mem=0
        mem+=sys.getsizeof(self.opermodes)
        for rng in self.rngs.values():
            mem+=sys.getsizeof(rng)
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
            for name,comp in self.components.items():
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
        for state, val in asdict(self.s).items():
            if type(val) in [set, dict]: val=copy.deepcopy(val)
            states[state]= val
        return states, self.m.faults.copy()
    def has_new_states(self, prev_states, prev_faults):
        states, faults = self.return_states()
        if prev_states!=states or prev_faults!=faults:  return True
        else:                                           return False
    def return_probdens(self):
        state_pd = 1.0
        if hasattr(self, 'components'): 
            for compname, comp in self.components:
                state_pd*=comp.return_probdens()
        if hasattr(self, 'actions'):
            for actionname, action in self.actions:
                state_pd*=action.return_probdens()
        if hasattr(self, 'pds'): state_pd= np.prod(self.pds)
        else:                    state_pd= 1.0
        return state_pd
    def make_flowdict(self,flownames,flows):
        """
        Puts a list of flows with a list of flow names in a dictionary.

        Parameters
        ----------
        flownames : list or dict or empty
            names of flows corresponding to flows
            using {externalname: internalname}
        flows : list
            flows

        Returns
        -------
        flowdict : dict
            dict of flows indexed by flownames
        """
        flowdict = {}
        if not(flownames) or type(flownames)==dict:
            flowdict = {f.name:f for f in flows}
            if flownames:
                for externalname, internalname in flownames.items():
                    flowdict[internalname] = flowdict.pop(externalname)
        elif type(flownames)==list:
            if len(flownames)==len(flows):
                for ind, flowname in enumerate(flownames):
                    flowdict[flowname]=flows[ind]
            else:   raise Exception("flownames "+str(flownames)+"\n don't match flows "+str(flows)+"\n in: "+self.name)
        else:       raise Exception("Invalid flownames option in "+self.name)
        return flowdict

#Function superclass 
class FxnBlock(Block):
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
    def __init__(self,name, flows, flownames=[], p={}, s={}, components={},timers=[],
                 local={}, comms={}, dt=None, seed=None):
        """
        Instantiates the function superclass with the relevant parameters.

        Parameters
        ----------
        flows :list
            Flow objects to (in order correspoinding to flownames) associate with the function
        flownames : list/dict, optional
            Names of flows  to use in the function, if private flow names are needed (e.g. functions with in/out relationships).
            Either provided as a list (in the same order as the flows) of all flow names corresponding to those flows
            Or as a dict of form {External Flowname: Internal Flowname}
        p : dict, optional
            Internal parameters to override from defaults. The default is {}.
        s : dict, optional
            Internal states to override from defaults. The default is {}.
        components : dict, optional
            Component objects to associate with the function. The default is {}.
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
        self.name = name
        self.internal_flows=dict()
        self.flows=self.make_flowdict(flownames,flows)
        self.add_local_to_flowdict(self.flows, local, "local")
        self.add_local_to_flowdict(self.flows, comms, "comms")
        for flow in self.flows.keys():
            setattr(self, flow, self.flows[flow])
        self.components=components
        self.compfaultmodes= dict()
        self.exclusive = False
        for cname in components:
            self.faultmodes.update({components[cname].localname+f:vals for f, vals in components[cname].faultmodes.items()})
            self.compfaultmodes.update({components[cname].localname+modename:cname for modename in components[cname].faultmodes})
            setattr(self, cname, components[cname])
        self.assoc_timers(*timers)
        if dt: self.dt=dt
        self.actions={}; self.conditions={}; self.condition_edges={}; self.actfaultmodes = {}
        self.action_graph = nx.DiGraph(); self.flow_graph = nx.Graph()
        super().__init__(p=p, s=s)
    def __repr__(self):
        blockret = super().__repr__()
        if getattr(self, 'actions'): return blockret+', active: '+str(self.active_actions)
        else:                        return blockret
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
    def add_act(self, name, action, *flows, duration=0.0, **params):
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
        self.actions[name] = action(name,flows, **params)
        self.actions[name].duration=duration
        setattr(self, name, self.actions[name])
        self.action_graph.add_node(name)
        self.flow_graph.add_node(name, bipartite=0)
        for flow in flows:
            self.flow_graph.add_node(flow.name, bipartite=1)
            self.flow_graph.add_edge(name,flow.name)
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
        self.condition_edges[name] = (start_action, end_action)
        self.action_graph.add_edge(start_action, end_action, **{'name':name, name:'name', 'arrow':True})
    def build_ASG(self, initial_action="auto",state_rep="finite-state", max_action_prop="until_false", mode_rep="replace", asg_proptype='dynamic', per_timestep=False, asg_pos={}):
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
        mode_rep : 'replace'/'independent'
            How actions are used to represent modes. Default is 'replace.'
                - 'replace' uses the actions to represent the operational modes of the system (only compatible with 'exclusive' representation)
                - 'independent' keeps the actions and function-level mode seperate
        asg_proptype : 'static'/'dynamic'/'manual'
            Which propagation step to execute the Action Sequence Graph in. Default is 'dynamic'
                - 'manual' means that the propagation is performed manually (defined in a behavior method)
        per_timestep : bool
            Defines whether the action sequence graph is reset to the initial state each time-step (True) or stays in the current action (False). Default is False
        asg_pos : dict, optional
            Positions of the nodes of the action/flow graph {node: [x,y]}. Default is {}
        """
        if initial_action=='auto': initial_action = [act for act, in_degree  in self.action_graph.in_degree if in_degree==0]
        elif type(initial_action)==str: initial_action=[initial_action]
        self.set_active_actions(initial_action)
        self.set_atts(state_rep=state_rep, max_action_prop=max_action_prop, mode_rep=mode_rep, asg_proptype=asg_proptype,initial_action=initial_action, per_timestep=per_timestep)
        if self.state_rep=='finite-state' and len(initial_action)>1: raise Exception("Cannot have more than one initial action with finite-state representation")
        
        if self.mode_rep=='replace':
            if not self.exclusive:           raise Exception("Cannot use mode_rep='replace' option without an exclusive representation (set in assoc_modes)")
            elif not self.state_rep=='finite-state':    raise Exception("Cannot use mode_rep='replace' option without using state_rep=`finite-state`")
            elif self.opermodes:                        raise Exception("Cannot use mode_rep='replace' option simultaneously with defined operational modes in assoc_modes()")
            if len(self.m.faultmodes)>0:                raise Exception("Cannot use mode_rep='replace option while having Function-level fault modes (define at Action level)")
            else:
                self.opermodes = [*self.actions.keys()]
                self.mode=initial_action[0]
        elif self.mode_rep=='independent':
            if self.exclusive:               raise Exception("Cannot use mode_rep='independent option' without a non-exclusive fault mode representation (set in assoc_modes)")
        for aname, action in self.actions.items():
            modes_to_add = {action.localname+f:val for f,val in action.m.faultmodes.items()}
            self.m.faultmodes.update(modes_to_add)
            fmode_intersect = set(modes_to_add).intersection(self.actfaultmodes)
            if any(fmode_intersect):
                raise Exception("Action "+aname+" overwrites existing fault modes: "+str(fmode_intersect)+". Rename the faults (or use name option in assoc_modes)")
            self.actfaultmodes.update({action.localname+modename:aname for modename in action.m.faultmodes})
        self.asg_pos=asg_pos
        
    def set_active_actions(self, actions):
        """Helper method for setting given action(s) as active"""
        if type(actions)==str: 
            if actions in self.actions: actions = [actions]
            else: raise Exception("initial_action="+actions+" not in self.actions: "+str(self.actions))
        if type(actions)==list:
            self.active_actions = set(actions)
            if any(self.active_actions.difference(self.actions)): raise Exception("Initial actions not associated with model: "+str(self.active_actions.difference(self.actions)))
        else: raise Exception("Invalid option for initial_action")
    def show_ASG(self, gtype='combined', with_cond_labels=True, pos=[]):
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
        if gtype=='combined':      graph = nx.compose(self.flow_graph, self.action_graph)
        elif gtype=='flows':        graph = self.flow_graph
        elif gtype=='actions':   graph = self.action_graph
        if not pos: 
            if not self.asg_pos: pos=nx.planar_layout(graph)
            else: pos=self.asg_pos
        nx.draw(graph, pos=pos, with_labels=True, node_color='grey')
        nx.draw_networkx_nodes(self.action_graph, pos=pos, node_shape='s', node_color='skyblue')
        nx.draw_networkx_nodes(self.action_graph, nodelist=self.active_actions, pos=pos, node_shape='s', node_color='green')
        edge_labels = {(in_node, out_node): label for in_node, out_node, label in graph.edges(data='name') if label}
        if with_cond_labels: nx.draw_networkx_edge_labels(graph, pos, edge_labels)
        if gtype=='combined' or gtype=='conditions':
            nx.draw_networkx_edges(self.action_graph, pos,arrows=True, arrowsize=30, arrowstyle='->', node_shape='s', node_size=100)
        return plt.gcf()
    def add_flow(self,flowname, flowdict={}, flowtype='', fclass=Flow, params={}):
        """
        Adds a flow with given attributes to the Function Block

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the function
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
            Parameters to pass the flow. Default is {}
        """
        if not getattr(self, 'is_copy', False):
            self.internal_flows[flowname] = init_flow(flowname,flowdict, flowtype, fclass)
            setattr(self, flowname, self.internal_flows[flowname])
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
        copy = self.__new__(self.__class__)  # Is this adequate? Wouldn't this give it new components?
        copy.is_copy=True
        copy.__init__(self.name, newflows, *attr)
        copy.m.mirror(self.m)
        if hasattr(self, 'mode_state_dict'):    copy.mode_state_dict = self.mode_state_dict
        for flowname, flow in self.internal_flows.items():
            copy.internal_flows[flowname] = flow.copy()
            setattr(copy, flowname, copy.internal_flows[flowname])
        for action in self.actions: 
            copy.action.s = copy.action._init_s(**asdict(action.s))
            copy.actions[action].m.mirror(self.actions[action].m)
            copy.actions[action].time=self.actions[action].time
        for component in self.components: 
            copy.component.s = copy.component._init_s(**asdict(component.s))
            copy.components[component].m.mirror(self.components[component].m)
            copy.components[component].time=self.components[component].time
        setattr(copy, 'active_actions', getattr(self, 'active_actions', {}))
        for timername in self.timers:
            timer = getattr(self, timername)
            copytimer = getattr(copy, timername)
            copytimer.set_timer(timer.time, tstep=timer.tstep)
            copytimer.mode=timer.mode
        copy.s = copy._init_s(**asdict(self.s))
        for generator in self.rngs:
            copy.rngs[generator]=np.random.default_rng(self._rng_params[generator][-1])
            copy.rngs[generator].__setstate__(self.rngs[generator].__getstate__())
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
    def prop_internal(self, faults, time, run_stochastic, proptype):
        """
        Propagates behaviors through the internal Action Sequence Graph

        Parameters
        ----------
        faults : list, optional
            Faults to inject in the function. The default is [].
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool/str
            Whether to run the simulation using stochastic or deterministic behavior
        proptype : str
            Type of propagation step to update ('behavior', 'static_behavior', or 'dynamic_behavior')
        """
        active_actions = self.active_actions
        num_prop = 0
        while active_actions:
            new_active_actions=set(active_actions)
            for action in active_actions:
                self.actions[action].updateact(time, run_stochastic, proptype=proptype, dt=self.dt)
                action_cond_edges = self.action_graph.out_edges(action, data=True)
                for act_in, act_out, atts in action_cond_edges:
                    if self.conditions[atts['name']]() and getattr(self.actions[action], 'duration',0.0)+self.dt<=self.actions[action].t_loc:
                        self.actions[action].t_loc=0.0
                        new_active_actions.add(act_out)
                        new_active_actions.discard(act_in)
            if len(new_active_actions)>1 and self.state_rep=='finite-state':
                raise Exception("Multiple active actions in a finite-state representation: "+str(new_active_actions))
            num_prop +=1 
            if type(self.asg_proptype)==int and num_prop>=self.asg_proptype:
                break
            if new_active_actions==set(active_actions):
                break
            else: active_actions=new_active_actions
            if num_prop>10000: raise Exception("Undesired looping in Function ASG for: "+self.name)
        if self.mode_rep=='replace': self.mode=[*active_actions][0]
        self.active_actions = active_actions
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
        self.run_stochastic=run_stochastic
        self.m.add_fault(*faults)  #if there is a fault, it is instantiated in the function
        if hasattr(self, 'condfaults'):    self.condfaults(time)    #conditional faults and behavior are then run
        if hasattr(self, 'mode_state_dict') and any(faults): self.update_modestates()
        if time>self.time and run_stochastic: self.update_stochastic_states()
        comp_actions = {**self.components, **self.actions} 
        if getattr(self, 'per_timestep', False): 
            self.set_active_actions(self.initial_action)
            for action in self.active_actions: self.actions[action].t_loc=0.0
        if comp_actions:     # propogate faults from function level to component level
            for fault in self.faults:
                if fault in self.compfaultmodes:
                    component = self.components[self.compfaultmodes[fault]]
                    component.m.add_fault(fault[len(component.localname):])
                if fault in self.actfaultmodes:
                    action = self.actions[self.actfaultmodes[fault]]
                    action.m.add_fault(fault[len(action.localname):])
        if any(self.actions) and self.asg_proptype==proptype: self.prop_internal(faults, time, run_stochastic, proptype)
        if proptype=='static' and hasattr(self,'behavior'):        self.behavior(time)     #generic behavioral methods are run at all steps
        if proptype=='static' and hasattr(self,'static_behavior'):                          self.static_behavior(time)
        elif proptype=='dynamic' and hasattr(self,'dynamic_behavior') and time > self.time: 
            if self.run_times>=1:
                for i in range(self.run_times):
                    self.dynamic_behavior(time)
            elif not Decimal(str(time))%Decimal(str(self.dt)):
                self.dynamic_behavior(time)
        elif proptype=='reset':                                                             
            if hasattr(self,'static_behavior'):  self.static_behavior(time)
            if hasattr(self,'dynamic_behavior'): self.dynamic_behavior(time)
        if comp_actions:     # propogate faults from component level to function level
            self.faults.difference_update(self.compfaultmodes)
            self.faults.difference_update(self.actfaultmodes)
            for compname, comp in comp_actions.items():
                self.faults.update({comp.localname+f for f in comp.faults if f!='nom'}) 
        self.time=time
        if run_stochastic=='track_pdf' and self.rngs: self.probdens = self.return_probdens()
        if (self.m.exclusive==True and len(self.m.faults)>1): 
            raise Exception("More than one fault present in "+self.name+"\n at t= "+str(time)+"\n faults: "+str(self.m.faults)+"\n Is the mode representation nonexclusive?")
        return
class GenericFxn(FxnBlock):
    """Generic function block. For use when the user has not yet defined a class for the
    given (to be implemented) function block. Acts as a placeholder that enables simulation."""
    def __init__(self, name, flows):
        super().__init__(name, flows)
  
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
        self.name = name
        super().__init__(p=p, s=s)
    def behavior(self,time):
        """ Placeholder for component behavior methods. Enables one to include components
        without yet having a defined behavior for them."""
        return 0
class Action(Block):
    """
    Superclass for actions (most attributes and methods inherited from Block superclass)
    """
    def __init__(self,name, flows, flownames=[], s={}, p={}):
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
        self.flows=self.make_flowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow, self.flows[flow])
        self.t_loc=0.0 # local time find a place for this?
        super().__init__(p=p, s=s)
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
        self.run_stochastic=run_stochastic
        if time>self.time and run_stochastic: self.update_stochastic_states()
        if proptype=='dynamic':
            if self.time<time:  self.behavior(time); self.t_loc+=dt
        else:                   self.behavior(time); self.t_loc+=dt
        self.time=time
    def behavior(self, time):
        """Placeholder behavior method for actions"""
        a=0


class Timer():
    """class for model timers used in functions (e.g. for conditional faults) 
    Attributes
    ----------
    name : str
        timer name
    time : float
        internal timer clock time
    tstep : float
        time to increment at each time-step
    mode : str (standby/ticking/complete)
        the internal state of the timer
    """
    def __init__(self, name):
        """
        Initializes the Tymer

        Parameters
        ----------
        name : str
            Name for the timer
        """
        self.name=str(name)
        self.time=0.0
        self.tstep=-1.0
        self.mode='standby'
    def __repr__(self):
        return 'Timer '+self.name+': mode= '+self.mode+', time= '+str(self.time)
    def t(self):
        """ Returns the time elapsed """
        return self.time
    def inc(self, tstep=[]):
        """ Increments the time elapsed by tstep"""
        if self.time>=0.0:
            if tstep:   self.time+=tstep
            else:       self.time+=self.tstep
            self.mode='ticking'
        if self.time<=0: self.time=0.0; self.mode='complete'
    def reset(self):
        """ Resets the time to zero"""
        self.time=0.0
        self.mode='standby'
    def set_timer(self,time, tstep=-1.0, overwrite='always'):
        """ Sets timer to a given time
        
        Parameters
        ----------
        time : float
            set time to count down in the timer
        tstep : float (default -1.0)
            time to increment the timer at each time-step
        overwrite : str
            whether/how to overwrite the previous time
            'always' (default) sets the time to the given time
            'if_more' only overwrites the old time if the new time is greater
            'if_less' only overwrites the old time if the new time is less
            'never' doesn't overwrite an existing timer unless it has reached 0.0
            'increment' increments the previous time by the new time
        """
        if overwrite =='always':                        self.time=time
        elif overwrite=='if_more' and self.time<time:   self.time=time
        elif overwrite=='if_less' and self.time>time:   self.time=time
        elif overwrite=='never' and self.time==0.0:     self.time=time
        elif overwrite=='increment':                    self.time+=time
        self.tstep=tstep
        self.mode='set'
    def in_standby(self):
        """Whether the timer is in standby (time has not been set)"""
        return self.mode=='standby'
    def is_ticking(self):
        """Whether the timer is ticking (time is incrementing)"""
        return self.mode=='ticking'
    def is_complete(self):
        """Whether the timer is complete (after time is done incrementing)"""
        return self.mode=='complete'
    def is_set(self):
        """Whether the timer is set (before time increments)"""
        return self.mode=='set'


def get_pdf_for_rand(x, randname, args):
    """
    Gets the corresponding probability mass/density for  
    for random sample x from 'randname' function in numpy.
    
    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float/array of probability densities
    """
    if type(x) not in [np.ndarray, list]: x=[x]
    if randname=='integers':
        if len(args)==1:        pd= [1/args[0] for x in x]
        elif len(args)>=2:      pd= [1/(args[1]-args[0]) for x in x]
    elif randname=='random':    pd= [1 for x in x]
    elif randname=='bytes':     raise Exception("Not able to calculate pdf for bytes")
    elif randname=='choice':    
        if type(args[0])==int:  options = [*np.arange(args[0])]
        else:                   options = args[0]
        if len(args)==4:        p = args[3]
        else:                   p = [1/len(options) for i in options]
        pd= [p[options.index(i)]  for i in x]
    elif randname in ['shuffle', 'permutation']:
        pd= [1/np.math.factorial(len(args[0]))]
    elif randname=='permuted':
        if len(args)>1 and type(args[0])==np.ndarray:
            pd= [1/np.math.factorial(args[0].shape(args[1]))]
        else:
            pd= [1/np.math.factorial(len(args[0]))]
    else:
        pd = get_pdf_for_dist(x,randname,args)
    if type(pd)==list:              pd=np.array(pd)
    elif type(pd) != np.ndarray:    pd=np.array([pd]) 
    return pd

def get_scipy_pdf_helper(x, randname, args,pmf=False):
    """
    Gets probability mass/density for the outcome x from the distribution "randname" with arguments "args".
    Used as a helper function in determining stochastic model state probability

    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of scipy.stats probability distribution
    args : tuple
        Arguments to send to scipy.stats.randname.pdf
    pmf : Bool, optional
        Whether the distribution uses a probability mass function instead of a pdf. The default is False.

    Returns
    -------
    prob: float/array of probability densities

    """
    if randname=='dirichlet':
        a=1
    if pmf:     return getattr(stats, randname).pmf(x, *args)
    else:       return getattr(stats, randname).pdf(x, *args)
def get_pdf_for_dist(x, randname, args): # note: when python 3.10 releases, this should become match/case
    """
    Gets the corresponding probability mass/density (from scipy) for outcome x 
    for probability distributions with name 'randname' in numpy.
    
    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float/array of probability densities
    
    """
    if type(x) in [np.ndarray, list] and len(x)>1 and len(args)>0: args=args[:-1]
    
    same_funcs = ['beta', 'dirichlet', 'f', 'gamma', 'laplace', 'logistic', 'multivariate_normal', 'pareto', 'uniform', 'wald']
    same_funcs_pmf = ['multinomial', 'poisson', 'zipf']
    different_funcs_pmf = {'binomial':'binom', 'geometric':'geom', 'logseries':'logser',\
                           'multivariate_hypergeometric':'multivariate_hypergeom',\
                           'negative_binomial':'nbinom'}
    different_funcs = {'chisquare':'chi2', 'gumbel':'gumbel_r', 'noncentral_chisquare':'ncx2',\
                       'noncentral_f':'ncf', 'normal':'norm', 'power':'powerlaw', 'standard_cauchy':'cauchy',\
                       'standard_gamma':'gamma', 'standard_normal':'norm', 'weibull':'weibull_min'}
    if randname in same_funcs:            
        return get_scipy_pdf_helper(x,randname, args)
    elif randname in same_funcs_pmf:
        return get_scipy_pdf_helper(x, randname, args, pmf=True)
    elif randname in different_funcs:
        return get_scipy_pdf_helper(x,different_funcs[randname], args)
    elif randname in different_funcs_pmf:
        return get_scipy_pdf_helper(x,different_funcs_pmf[randname], args, pmf=True)       
    elif randname in ['exponential', 'rayleigh']:   
        if len(args)==0:            return getattr(stats, randname).pdf(x)
        elif len(args)==1:          return getattr(stats, randname).pdf(x, scale=args[0]) 
        elif len(args)==2:          return getattr(stats, randname).pdf(x, loc=args[1], scale=args[0]) 
        else: raise Exception("Too many arguments for "+randname+" distribution")
    elif randname=='hypergeometric': 
        n_pop = args[0]+args[1]
        n_good = args[0]
        n_sample = args[2]
        return stats.hypergeom.pmf(x,n_pop, n_good, n_sample)
    elif randname=='lognormal':
        s=args[1]
        scale=np.exp(args[0])
        return stats.lognormal.pdf(x, s, scale=scale) 
    elif randname=='standard_t': return stats.multivariate_t.pdf(x, df=args[0])
    elif randname=='triangular':
        left, mode, right = args[:3]
        loc = left
        scale = right-loc
        c = (mode-loc)/scale
        return stats.triang.pdf(x,c,loc,scale)
    elif randname=='vonmises':
        return stats.vonmises.pdf(x,args[1], args[0])
    else: raise Exception("Invalid randname distribution: "+randname+". Ensure that it is a part of numpy.random/scipy.stats")


