# -*- coding: utf-8 -*-
"""
Description: Module for helping define Modes (faulty and otherwise).

Has classes:
- :class:`Fault`: Class for defining fault parameters
- :Mode:`Mode`: Class for defining the mode property (and associated probability model) held in Blocks. 
"""
from recordclass import dataobject
import numpy as np
import itertools
from .common import get_true_fields, get_true_field
from fmdtools.faultsim.result import History, init_hist_iter

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
    Description: Class for defining the mode property (and associated probability model) held in Blocks. 
    
    Mode is meant to be inherited in order to define the specific faults related to a given Block.
    
    e.g., 
    class ExampleMode(Mode):
        faultmodes = {"high_heat", "low_heat"}
        mode = "start"
    Will create a Mode structure where m.mode = 'start' that can enter the given
    fault modes 'high_heat' and 'low_heat'.
    
    Mode has the following fields which can be modified to define the underlying 
    representation and probability model:
    -------------
        
    opermodes : tuple
        Names of non-faulty operational modes.
    failrate : float
        Overall failure rate for the block. The default is 1.0.
    faultparams : dict 
            Dictionary/Set of arguments defining faultmodes, which can have the forms:
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
    faults : set
        Set of faults present (or not) at any given time
    mode : str
        Name of the current mode. the default is 'nominal'
    he_args : tuple
        Arguments for add_he_rate defining a human error probability model.
        
    
    These properties can then be used in simulation
    ------------
    faults : set
        Set of faults present (or not) at any given time
    mode : str
        Name of the current mode. the default is 'nominal'
    mode_state_dict: dict
        Maps modes to states. Assigned by assoc_faultstates
        
    
    While these properties are used for determining scenario information 
    ------------
    faultmodes : dict 
            Dictionary of :class:`Fault` defining possible fault modes and their properties   
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
        """
        Initializes the self.faultmodes dictionary from the parameters of the Mode
        """
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
            tot_faults = len([f for s in franges.values() for f in s])
            for state in franges:
                modes = {state+'_'+str(value): Fault(probtype='prob', dist =1/tot_faults)  
                         for value in franges[state]}
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
            self.faultmodes.update({'hmode_'+str(i):Fault(probtype='prob', dist =1/len(statecombos)) 
                                    for i in range(len(statecombos))}) 
            self.mode_state_dict.update({'hmode_'+str(i): {list(franges)[j]:state for j, state in enumerate(statecombos[i])} for i in range(len(statecombos))})
        else: raise Exception("Invalid mode elaboration approach")

        for mode,atts in manual_modes.items():
            if type(atts)==list:
                self.mode_state_dict.update({mode:atts[0]})
                if not getattr(self, 'exclusive', False): print("Changing fault mode exclusivity to True")
                self.assoc_modes(faultmodes={mode:atts[1]}, initmode=getattr(self,'mode', 'nom'), probtype=probtype, proptype=probtype, exclusive=True, key_phases_by=key_phases_by)
            elif  type(atts)==dict:
                self.mode_state_dict.update({mode:atts})
                self.faultmodes.update({mode:Fault(probtype='prob', dist =1/len(manual_modes))})
    def update_modestates(self):
        """Updates states of the model associated with a specific fault mode (see assoc_modes)."""
        num_update = 0
        for fault in self.faults:
            if fault in self.mode_state_dict:
                for state, value in self.mode_state_dict[fault].items():
                    setattr(self, state, value)
                num_update+=1
                if num_update > 1: raise Exception("Exclusive fault mode scenarios present at the same time")
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
    def create_hist(self, timerange, track):
        h = History()
        if self.exclusive:   
            h.data = {faultmode: init_hist_iter(faultmode, False, timerange, track, dtype=bool)
                      for faultmode in self.faultmodes} 
        modelength = max([len(fm) for fm in self.faultmodes]+[len(m) for m in self.opermodes])
        str_size = '<U'+str(modelength)
        h['mode'] = init_hist_iter('mode', self.mode, timerange, track, str_size=str_size)
        return h