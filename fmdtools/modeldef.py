# -*- coding: utf-8 -*-
"""
Description: A module to define resilience models and simulations.

    - :class:`Common`:      Class defining common methods accessible by Function/Flow/Component Classes
    - :class:`FxnBlock`:    Class defining Model Functions and their attributes
    - :class:`Flow`:        Class defining Model Flows and their attributes
    - :class:`Component`:   Class defining Function Components and their attributes
    - :class:`SampleApproach`:  Class defining fault sampling approaches
    - :class:`NominalApproach`: Class defining parameter sampling approaches
"""
#File name: modeldef.py
#Author: Daniel Hulse
#Created: October 2019

import numpy as np
import itertools
import dill
import networkx as nx
import copy
import warnings
import sys
from decimal import Decimal
from ordered_set import OrderedSet
from operator import itemgetter
from collections.abc import Iterable
from collections import Hashable
from inspect import signature
import fmdtools.resultdisp.process as proc
# MAJOR CLASSES

class Common(object):
    def set_atts(self, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for 
        reducing length/adding clarity to assignment statements in __init__ methods
        (self.put is reccomended otherwise so that the iteration is on function/flow *states*)
        e.g., self.set_attr(maxpower=1, maxvoltage=1) is the same as saying
              self.maxpower=1; self.maxvoltage=1
        """
        for name, value in kwargs.items():
            setattr(self, name, value)
    def put(self,**kwargs):
        """Sets the given arguments to a given value. Mainly useful for 
        reducing length/adding clarity to assignment statements.
        e.g., self.EE.put(v=1, a=1) is the same as saying
              self.EE.v=1; self.EE.a=1
        """
        for name, value in kwargs.items():
            if name not in self._states: raise Exception(name+" not a property of "+self.name)
            setattr(self, name, value)
    def assign(self,obj,*states):
        """ Sets the same-named values of the current flow/function object to those of a given flow. 
        Further arguments specify which values.
        e.g. self.EE1.assign(EE2, 'v', 'a') is the same as saying
            self.EE1.a = self.EE2.a; self.EE1.v = self.EE2.v
        """
        if len(states)==0: states= obj._states
        for state in states:
            if state not in self._states: raise Exception(state+" not a property of "+self.name)
            setattr(self, state, getattr(obj,state))
    def get(self, *attnames, **kwargs):
        """Returns the given attribute names (strings). Mainly useful for reducing length
        of lines/adding clarity to assignment statements.
        e.g., x,y = self.Pos.get('x','y') is the same as
              x,y = self.Pos.x, self.Pos.y, or
              z = self.Pos.get('x','y') is the same as
              z = np.array([self.Pos.x, self.Pos.y])
        """
        if len(attnames)==1:    states = getattr(self,attnames[0])
        else:                   states = [getattr(self,name) for name in attnames]
        if not is_iter(states):                 return states
        elif len(states)==1:                    return states[0]
        elif kwargs.get('as_array', True):      return np.array(states)
        else:                                   return states
    def values(self):
        return self.gett(*self._states)
    def gett(self, *attnames):
        """Alternative to self.get that returns the given constructs as a tuple instead
        of as an array. Useful when a numpy array would translate the underlying data types
        poorly (e.g., np.array([1,'b'] would make 1 a string--using a tuple instead preserves
        the data type)"""
        states = self.get(*attnames,as_array=False)
        if not is_iter(states):                 return states
        elif len(states)==1:                    return states[0]
        else:                                   return tuple(states)
    def inc(self,**kwargs):
        """Increments the given arguments by a given value. Mainly useful for
        reducing length/adding clarity to increment statements.
        e.g., self.Pos.inc(x=1,y=1) is the same as
             self.Pos.x+=1; self.Pos.y+=1, or
             self.Pos.x = self.Pos.x + 1; self.Pos.y = self.Pos.y +1
             
        Can additionally be provided with a second value denoting a limit on the increments
        e.g. self.Pos.inc(x=(1,10)) will increment x by 1 until it reaches 10
        """
        for name, value in kwargs.items():
            if name not in self._states: raise Exception(name+" not a property of "+self.name)
            if type(value)==tuple:  
                current = getattr(self,name)
                sign = np.sign(value[0])
                newval = current + value[0]
                if sign*newval <= sign*value[1]:    setattr(self, name, newval)
                else:                               setattr(self,name,value[1])
            else:                   setattr(self, name, getattr(self,name)+ value)
    def limit(self,**kwargs):
        """Enforces limits on the value of a given property. Mainly useful for
        reducing length/adding clarity to increment statements.
        e.g., self.EE.limit(a=(0,100), v=(0,12)) is the same as
            self.EE.a = min(100, max(0,self.EE.a));
            self.EE.v = min(12, max(0,self.EE.v))
        """
        for name, value in kwargs.items():
            if name not in self._states: raise Exception(name+" not a property of "+self.name)
            setattr(self, name, min(value[1], max(value[0], getattr(self,name))))
    def mul(self,*states):
        """Returns the multiplication of given attributes of the model construct.
        e.g.,   a = self.mul('x','y','z') is the same as
                a = self.x*self.y*self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a = a * self.get(state)
        return a
    def div(self,*states):
        """Returns the division of given attributes of the model construct
        e.g.,   a = self.div('x','y','z') is the same as
                a = (self.x/self.y)/self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a = a / self.get(state)
        return a
    def add(self,*states):
        """Returns the addition of given attributes of the model construct
        e.g.,   a = self.add('x','y','z') is the same as
                a = self.x+self.y+self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a += self.get(state)
        return a
    def sub(self,*states):
        """Returns the addition of given attributes of the model construct
        e.g.,   a = self.div('x','y','z') is the same as
                a = (self.x-self.y)-self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a -= self.get(state)
        return a
    def same(self,values, *states):
        """Tests whether a given iterable values has the same value as each
        give state in the model construct.
        e.g.,   self.same([1,2],'a','b') is the same as
                all([1,2]==[self.a, self.b])"""
        test = values==self.get(*states)
        if is_iter(test):   return all(test)
        else:               return test
    def different(self,values, *states):
        """Tests whether a given iterable values has any different value the
        given states in the model construct.
        e.g.,   self.same([1,2],'a','b') is the same as
                any([1,2]!=[self.a, self.b])"""
        test = values!=self.get(*states)
        if is_iter(test):   return any(test)
        else:               return test
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
    def warn(self, *messages, stacklevel=2):
        """
        Prints warning message(s) when called.

        Parameters
        ----------
        *messages : str
            Strings to make up the message (will be joined by spaces)
        stacklevel : int
            Where the warning points to. The default is 2 (points to the place in the model)
        """
        warnings.warn(' '.join(messages), stacklevel=stacklevel)
    
class Block(Common):
    """ 
    Superclass for FxnBlock and Component subclasses. Has functions for model setup, querying state, reseting the model
    
    Attributes
    ----------
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
    opermodes : list
        possible modes for the block to enter
    rngs : dict
        dictionary of random number generators for random states
    seed : int
        seed sequence for internal random number generator
    mode : string
        current mode of block operation
    """
    def __init__(self, states={}):
        """
        Instance superclass. Called by FxnBlock and Component classes.

        Parameters
        ----------
        states : dict, optional
            Internal states (variables, essentially) of the block. The default is {}.
        """
        self._states=list(states.keys())
        self._initstates=states.copy()
        self.failrate = getattr(self, 'failrate', 1.0)
        self.localname=''
        for state in states.keys():
            setattr(self, state,states[state])
        self.faults=set(['nom'])
        self.opermodes= getattr(self, 'opermodes', [])
        self.faultmodes= getattr(self, 'faultmodes', {})
        self.rngs=getattr(self, 'rngs', {})
        self._rng_params=getattr(self, '_rng_params', {})
        if not getattr(self, 'seed', []): 
            self.seed=np.random.SeedSequence.generate_state(np.random.SeedSequence(),1)[0]
        self.rng=np.random.default_rng(self.seed)
        self.time=0.0
    def __repr__(self):
        if hasattr(self,'name'):
            return getattr(self, 'name', '')+' '+self.__class__.__name__+' '+getattr(self,'type', '')+': '+str(self.return_states())
        else: return 'New uninitialized '+self.__class__.__name__
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
                    if not getattr(self, 'exclusive_faultmodes', False): print("Changing fault mode exclusivity to True")
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
    def assoc_modes(self, faultmodes={}, opermodes=[],initmode='nom', name='', probtype='rate', units='hr', exclusive=False, key_phases_by='global', longnames={}):
        """
        Associates fault and operational modes with the block when called in the function or component.

        Parameters
        ----------
        faultmodes : dict, optional
            Dictionary/Set of faultmodes with structure, which can have the forms:
                - set {'fault1', 'fault2', 'fault3'} (just the respective faults)
                - dict {'fault1': faultattributes, 'fault2': faultattributes}, where faultattributes is:
                    - float: rate for the mode
                    - dict/set/str: opportunity vector for the mode specified as a dictionary/set/string
                    - list: [rate, oppvect, rcost]
                        - a list of arguments where the float arguments are specified in the order rate, rcost (if provided) and
                            an oppvect opportunity vector is provided (anywhere) with the form:
                                -list: [float1, float2,...], a vector of relative likelihoods for each phase, or
                                -dict: {opermode:float1, opermode:float1}, a dict of relative likelihoods for each phase/mode
                                -set: {opermode, opermode,...}, a set of applicable phases (assumed equally likely).
                                the phases/modes to key by are defined in "key_phases_by"
                                -str: 'all'/'modename', either specifying all operational modes/phases or a single operational mode/phase
        opermodes : list, optional
            List of operational modes
        initmode : str, optional
            Initial operational mode. Default is 'nom'
        name : str, optional
            (for components/actions only) Name of the component. The default is ''.
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
        if opermodes:
            self.opermodes = opermodes
            if initmode in self.opermodes:
                self._states.append('mode')
                self._initstates['mode'] = initmode
                self.mode = initmode
            else: raise Exception("Initial mode "+initmode+" not in defined modes for "+self.name)
        else: 
            self._states.append('mode')
            self._initstates['mode'] = initmode
            self.mode = initmode
        self.exclusive_faultmodes = exclusive
        self.localname = name
        if not getattr(self, 'is_copy', False): #saves time by using the same fault mode dictionary from previous
            if not getattr(self, 'faultmodes', []): 
                if name: self.faultmodes=dict()
                else:    self.faultmodes=dict.fromkeys(faultmodes)
            for mode in faultmodes:
                self.faultmodes[mode]=dict.fromkeys(('dist', 'oppvect', 'rcost', 'probtype', 'units'))
                self.faultmodes[mode]['probtype'] = probtype
                self.faultmodes[mode]['units'] = units
                if key_phases_by=='self': oppvect='all'
                else:                     oppvect=[1.0]
                dist=1.0/len(faultmodes); rcost=0.0
                if type(faultmodes) in {set,str}:     a=1 # minimum information - here the faultmodes are only a set of labels
                elif type(faultmodes[mode]) == float:   dist  =  faultmodes[mode] # dict of modes: dist, where dist is the distribution (or individual rate/probability)
                elif type(faultmodes[mode]) == dict:    oppvect = faultmodes[mode] # provided oppvect in dict form
                elif type(faultmodes[mode]) == list: # provided list with oppvect, dist, rcost (rcost always after dist)
                    oppvect_loc = [i for i,e in enumerate(faultmodes[mode]) if type(e) in [dict, set, list, str]]
                    if oppvect_loc: oppvect= faultmodes[mode].pop(oppvect_loc[0])
                    for i,e in enumerate(faultmodes[mode]):  
                        if i>=1:          rcost = e
                        else:             dist = e
                else:   raise Exception("Invalid mode definition")
                if 'all' in oppvect or oppvect=='all': 
                    if not opermodes:   oppvect = {'nom'}
                    else:               oppvect = {*opermodes}
                    if key_phases_by!='self': raise Exception("'all' option for oppvect only applies to key_phases_by='self'")
                elif type(oppvect)==str:                    oppvect={oppvect}
                if type(oppvect)==set:                      oppvect = {o:1.0 for o in oppvect}
                if key_phases_by =='none' and oppvect!=[1.0]: 
                    raise Exception("How should the opportunity vector be keyed? Provide 'key_phases_by' option.")                
                self.faultmodes[mode]['dist'] =     dist
                self.faultmodes[mode]['oppvect'] =  oppvect
                self.faultmodes[mode]['rcost'] =    rcost
                self.faultmodes[mode]['longname'] = longnames.get(mode,mode)
        if key_phases_by=='self':   self.key_phases_by = self.name
        else:                       self.key_phases_by = key_phases_by
    def choose_rand_fault(self, faults, default='first', combinations=1):
        """
        Randomly chooses a fault or combination of faults to insert in the function. 

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
            self.add_fault(*self.rng.choice(faults))
        elif default=='first':      self.add_fault(faults[0])
        elif type(default)==str:    self.add_fault(default)
        else:                       self.add_fault(*default)
    def set_rand(self,statename,methodname, *args):
        """
        Update the given random state with a given method and arguments

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
            gen_method = getattr(self.rngs[statename], methodname)
            setattr(self, statename, gen_method(*args))
    def to_default(self,*statenames):
        """ Resets (given or all by default) random states to their default values
        
        Parameters
        ----------
        *statenames : str, str, str...
            names of the random state defined in assoc_rand_state(s)
        """
        if not statenames: statenames=list(self._rng_params.keys())
        for statename in statenames: setattr(self, statename, self._rng_params[statename][0])
    def set_mode(self, mode):
        """Sets a mode in the block
        
        Parameters
        ----------
        mode : str
            name of the mode to enter.
        """
        if self.exclusive_faultmodes:
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
        return any(self.faults.difference({'nom'}))
    def to_fault(self,fault): 
        """Moves from the current fault mode to a new fault mode
        
        Parameters
        ----------
        fault : str
            name of the fault mode to switch to
        """
        self.faults.clear()
        self.faults.add(fault)
        if self.exclusive_faultmodes: self.mode = fault
    def add_fault(self,*faults): 
        """Adds fault (a str) to the block
        
        Parameters
        ----------
        *fault : str(s)
            name(s) of the fault to add to the black
        """
        self.faults.update(faults)
        if self.exclusive_faultmodes: 
            if len(faults)>1:   raise Exception("Multiple fault modes added to function with exclusive fault representation")
            elif len(faults)==1 and list(faults)[0]!='nom': self.mode =faults[0]
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
        if self.exclusive_faultmodes: self.mode = fault_to_add
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
        if len(self.faults) == 0: self.faults.add('nom')
        if opermode:    self.mode = opermode
        if self.exclusive_faultmodes and not(opermode):
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
        self.faults.add('nom')
        if opermode:    self.mode = opermode
        if self.exclusive_faultmodes and not(opermode):
            raise Exception("Unclear which operational mode to enter with fault removed")
        if warnmessage: self.warn(warnmessage, "All faults removed.")
    def get_flowtypes(self):
        """Returns the names of the flow types in the model"""
        return {obj.type for name, obj in self.flows.items()}
    def update_stochastic_states(self):
        """Updates the defined stochastic states defined to auto-update (see assoc_randstates)."""
        for statename, generator in self.rngs.items():
            if self._rng_params[statename][1]:
                gen_method = getattr(generator, self._rng_params[statename][1])
                setattr(self, statename, gen_method(*self._rng_params[statename][2]))
    def reset(self):            #reset requires flows to be cleared first
        """ Resets the block to the initial state with no faults. Used by default in 
        derived objects when resetting the model. Requires associated flows to be cleared first."""
        self.faults.clear()
        self.faults.add('nom')
        for state in self._initstates.keys():
            setattr(self, state,self._initstates[state])
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
            self.updatefxn('reset', faults=['nom'], time=0)
    def get_memory(self):
        """ Gets the approximate memory usage of the block in bytes (not complete)"""
        mem=0
        mem+=sys.getsizeof(self.opermodes)
        for rng in self.rngs.values():
            mem+=sys.getsizeof(rng)
        if hasattr(self, 'faultmodes'):
            for fm in self.faultmodes.values():
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
        for state in self._initstates.values():
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
        for state in self._states:
            states[state]=getattr(self,state)
        return states, self.faults.copy()
    def check_update_nominal_faults(self):
        if self.faults.difference({'nom'}): self.faults.difference_update({'nom'})
        elif len(self.faults)==0:           self.faults.update(['nom'])

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
    def __init__(self,name, flows, flownames=[], states={}, components={},timers=[], dt=None, seed=None):
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
        states : dict, optional
            Internal states to associate with the function. The default is {}.
        components : dict, optional
            Component objects to associate with the function. The default is {}.
        timers : set, optional
            Set of names of timers to use in the function. The default is {}.
        """
        self.type = 'function'
        self.name = name
        self.internal_flows=dict()
        self.flows=self.make_flowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow, self.flows[flow])
        self.components=components
        if not getattr(self, 'faultmodes', []): self.faultmodes={}
        self.compfaultmodes= dict()
        self.exclusive_faultmodes = False
        for cname in components:
            self.faultmodes.update({components[cname].localname+f:vals for f, vals in components[cname].faultmodes.items()})
            self.compfaultmodes.update({components[cname].localname+modename:cname for modename in components[cname].faultmodes})
            setattr(self, cname, components[cname])
        self.assoc_timers(*timers)
        if dt: self.dt=dt
        self.actions={}; self.conditions={}; self.condition_edges={}; self.actfaultmodes = {}
        self.action_graph = nx.DiGraph(); self.flow_graph = nx.Graph()
        super().__init__(states)
    def __repr__(self):
        blockret = super().__repr__()
        if getattr(self, 'actions'): return blockret+', active: '+str(self.active_actions)
        else:                        return blockret
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
            if not self.exclusive_faultmodes:           raise Exception("Cannot use mode_rep='replace' option without an exclusive_faultmodes representation (set in assoc_modes)")
            elif not self.state_rep=='finite-state':    raise Exception("Cannot use mode_rep='replace' option without using state_rep=`finite-state`")
            elif self.opermodes:                        raise Exception("Cannot use mode_rep='replace' option simultaneously with defined operational modes in assoc_modes()")
            if len(self.faultmodes)>0:                  raise Exception("Cannot use mode_rep='replace option while having Function-level fault modes (define at Action level)")
            else:
                self.opermodes = [*self.actions.keys()]
                self.mode=initial_action[0]
        elif self.mode_rep=='independent':
            if self.exclusive_faultmodes:               raise Exception("Cannot use mode_rep='independent option' without a non-exclusive fault mode representation (set in assoc_modes)")
        for aname, action in self.actions.items():
            modes_to_add = {action.localname+f:val for f,val in action.faultmodes.items()}
            self.faultmodes.update(modes_to_add)
            fmode_intersect = set(modes_to_add).intersection(self.actfaultmodes)
            if any(fmode_intersect):
                raise Exception("Action "+aname+" overwrites existing fault modes: "+str(fmode_intersect)+". Rename the faults (or use name option in assoc_modes)")
            self.actfaultmodes.update({action.localname+modename:aname for modename in action.faultmodes})
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
    def add_flow(self,flowname, flowdict={}, flowtype=''):
        """
        Adds a flow with given attributes to the Function Block

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the function
        flowattributes : dict, Flow, set or empty set
            Dictionary of flow attributes e.g. {'value':XX}, or the Flow object.
            If a set of attribute names is provided, each will be given a value of 1
            If an empty set is given, it will be represented w- {flowname: 1}
        """
        if not getattr(self, 'is_copy', False):
            if not flowtype: flowtype = flowname
            if not flowdict:                self.internal_flows[flowname]=Flow({flowname:1}, flowname, flowtype)
            elif type(flowdict) == set:     self.internal_flows[flowname]=Flow({f:1 for f in flowdict}, flowname, flowtype)
            elif type(flowdict) == dict:    self.internal_flows[flowname]=Flow(flowdict, flowname,flowtype)
            elif isinstance(flowdict, Flow):self.internal_flows[flowname] = flowdict
            else: raise Exception('Invalid flow. Must be dict or flow')
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
        copy.faults = self.faults.copy()
        if hasattr(self, 'faultmodes'):         copy.faultmodes = self.faultmodes
        if hasattr(self, 'mode_state_dict'):    copy.mode_state_dict = self.mode_state_dict
        for flowname, flow in self.internal_flows.items():
            copy.internal_flows[flowname] = flow.copy()
            setattr(copy, flowname, copy.internal_flows[flowname])
        for action in self.actions: 
            for state in copy.actions[action]._initstates.keys():
                setattr(copy.actions[action], state, getattr(self.actions[action], state))
            copy.actions[action].faults=self.actions[action].faults.copy()
        setattr(copy, 'active_actions', getattr(self, 'active_actions', {}))
        for timername in self.timers:
            timer = getattr(self, timername)
            copytimer = getattr(copy, timername)
            copytimer.set_timer(timer.time, tstep=timer.tstep)
            copytimer.mode=timer.mode
        for state in self._initstates.keys():
            setattr(copy, state, getattr(self, state))
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
            Faults to inject in the function. The default is ['nom'].
        time : float, optional
            Model time. The default is 0.
        run_stochastic : book
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
            Faults to inject in the function. The default is ['nom'].
        time : float, optional
            Model time. The default is 0.
        run_stochastic : book
            Whether to run the simulation using stochastic or deterministic behavior
        """
        self.run_stochastic=run_stochastic
        self.add_fault(*faults)  #if there is a fault, it is instantiated in the function
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
                    component.add_fault(fault[len(component.localname):])
                if fault in self.actfaultmodes:
                    action = self.actions[self.actfaultmodes[fault]]
                    action.add_fault(fault[len(action.localname):])
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
        self.check_update_nominal_faults()
        if self.exclusive_faultmodes==True and len(self.faults)>1: 
            raise Exception("More than one fault present in "+self.name+"\n at t= "+str(time)+"\n faults: "+str(self.faults)+"\n Is the mode representation nonexclusive?")
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
    def __init__(self,name, states={}):
        """
        Inherit the component class

        Parameters
        ----------
        name : str
            Unique name ID for the component
        states : dict, optional
            States to use in the component. The default is {}.
        """
        self.type = 'component'
        self.name = name
        super().__init__(states)
    def behavior(self,time):
        """ Placeholder for component behavior methods. Enables one to include components
        without yet having a defined behavior for them."""
        return 0
class Action(Block):
    """
    Superclass for actions (most attributes and methods inherited from Block superclass)
    """
    def __init__(self,name, flows, flownames=[], states={}):
        """
        Inherit the Block class

        Parameters
        ----------
        name : str
            Unique name ID for the action
        states : dict, optional
            States to use in the action. The default is {}.
        """
        self.type = 'action'
        self.name = name
        self.flows=self.make_flowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow, self.flows[flow])
        super().__init__({**states, 't_loc':0.0})
    def updateact(self, time=0, run_stochastic=False, proptype='dynamic', dt=1.0):
        """
        Updates the behaviors, faults, times, etc of the action 

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : book
            Whether to run the simulation using stochastic or deterministic behavior
        """
        self.run_stochastic=run_stochastic
        if time>self.time and run_stochastic: self.update_stochastic_states()
        if proptype=='dynamic':
            if self.time<time:  self.behavior(time); self.t_loc+=dt
        else:                   self.behavior(time); self.t_loc+=dt
        self.time=time
        self.check_update_nominal_faults()
    def behavior(self, time):
        """Placeholder behavior method for actions"""
        a=0

class Flow(Common):
    """
    Superclass for flows. Instanced by Model.add_flow but can also be used as a flow superclass if flow attributes are not easily definable as a dict.
    """
    def __init__(self, states, name, ftype='generic'):
        """
        Instances the flow with given states.

        Parameters
        ----------
        states : dict
            states and their values to be associated with the flow
        name : str
            name of the flow
        """
        self.type=ftype
        self.name=name
        self._initstates=states.copy()
        self._states=list(states.keys())
        for state in self._states:
            setattr(self, state, states[state])
    def __repr__(self):
        if hasattr(self,'name'):    
            return getattr(self, 'name')+' '+getattr(self, 'type')+' flow: '+str(self.status())
        else: return "Uninitialized Flow"
    def reset(self):
        """ Resets the flow to the initial state"""
        for state in self._initstates:
            setattr(self, state, self._initstates[state])
    def status(self):
        """
        Returns a dict with the current states of the flow.
        """
        states={}
        for state in self._states:
            states[state]=getattr(self,state)
        return states
    def get_memory(self):
        """
        Returns the approximate memory usage of the flow.
        """
        mem = 0
        for state in self._states:
            mem+=2*sys.getsizeof(getattr(self, state)) # (*2 to account for initstates)
        return mem
    def copy(self):
        """
        Returns a copy of the flow object (used when copying the model)
        """
        states={}
        for state in self._states:
            states[state]=getattr(self,state)
        if self.__class__==Flow:
            copy = self.__class__(states, self.name, self.type)
        else:
            copy = self.__class__()
            for state in self._states:
                setattr(copy, state, getattr(self,state))
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
    params,modelparams,valparams : dict
        dictionaries of (optional) parameters for a given instantiation of a model
    modelparams : dict
        dictionary of parameters for running a simulation. defines these parameters in the model:
            phases : dict
                phases {'name':[start, end]} that the simulation progresses through
            times : array
                array of times to sample (if desired) [starttime, sampletime1, sampletime2,... endtime]
            dt : float
                timestep used in the simulation. default is 1.0
            units : str
                time-units. default is hours
            use_end_condition : bool
                whether to use an end-condition method (defined by user-defined end_condition method) 
                or defined end time to end the simulation 
            seed : int
                seed used for the internal random number generator
    valparams : 
        dictionary of parameters for defining what simulation constructs to record for find_classification
    bipartite : networkx graph
        bipartite graph view of the functions and flows
    graph : networkx graph
        multigraph view of functions and flows
    
    """
    def __init__(self, params={},modelparams={}, valparams='all'):
        """
        Instantiates internal model attributes with predetermined:
        
        Parameters
        ----------
        params : dict 
            design variables of the model
        modelparams : dict 
            dictionary of: 
                       - global phases {'phase': [starttime, endtime]}
                       - times [starttime, ..., endtime] (middle time used for sampling), 
                       - timestep (float) to run the model with)
                       - seed (int) - if present, sets a seed to run the random number generators from
                       - use_end_condition (bool) - if True (default), uses end_condition() in the model to determine when the simulation ends.
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
        
        # model defaults to static representation if no timerange
        self.phases=modelparams.get('phases',{'na':[1]})
        self.find_any_phase_overlap()
        self.times=modelparams.get('times',[1])
        self.tstep = modelparams.get('tstep', 1.0)
        self.units = modelparams.get('units', 'hr')
        self.use_local = modelparams.get('use_local', True)
        self.use_end_condition = modelparams.get('use_end_condition', True)
        if modelparams.get('seed', False):  self.seed = modelparams['seed']
        else:
            self.seed=np.random.SeedSequence.generate_state(np.random.SeedSequence(),1)[0]                       
            modelparams['seed']=self.seed
        self._rng = np.random.default_rng(self.seed)
        
        self.functionorder=OrderedSet() #set is ordered and executed in the order specified in the model
        self._fxnflows=[]
        self._fxninput={}
    def __repr__(self):
        fxnstr = ''.join(['- '+fxnname+':'+str(fxn.return_states())+' '+str(getattr(fxn,'active_actions',''))+'\n' for fxnname,fxn in self.fxns.items()])
        flowstr = ''.join(['- '+flowname+':'+str(flow.status())+'\n' for flowname,flow in self.flows.items()])
        return self.__class__.__name__+' model at '+hex(id(self))+' \n'+'functions: \n'+fxnstr+'flows: \n'+flowstr
    def find_any_phase_overlap(self):
        intervals = [*self.phases.values()]
        int_low = np.sort([i[0] for i in intervals])
        int_high = np.sort([i[1] if len(i)==2 else i[0] for i in intervals])
        for i, il in enumerate(int_low):
            if i+1==len(int_low): break
            if int_low[i+1]<=int_high[i]:
                raise Exception("Global phases overlap (see mdlparams):"+str(self.phases)+" Ensure the max of each phase < min of each other phase")
    def add_flows(self, flownames, flowdict={}, flowtype='generic'):
        """
        Adds a set of flows with the same type and initial parameters

        Parameters
        ----------
        flowname : list
            Unique flow names to give the flows in the model
        flowattributes : dict, Flow, set or empty set
            Dictionary of flow attributes e.g. {'value':XX}, or the Flow object.
            If a set of attribute names is provided, each will be given a value of 1
            If an empty set is given, it will be represented w- {flowname: 1}
        """
        for flowname in flownames: self.add_flow(flowname, flowdict, flowtype)
    def add_flow(self,flowname, flowdict={}, flowtype=''):
        """
        Adds a flow with given attributes to the model.

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the model
        flowattributes : dict, Flow, set or empty set
            Dictionary of flow attributes e.g. {'value':XX}, or the Flow object.
            If a set of attribute names is provided, each will be given a value of 1
            If an empty set is given, it will be represented w- {flowname: 1}
        """
        if not getattr(self, 'is_copy', False):
            if not flowtype: flowtype = flowname
            if not flowdict:                self.flows[flowname]=Flow({flowname:1}, flowname, flowtype)
            elif type(flowdict) == set:     self.flows[flowname]=Flow({f:1 for f in flowdict}, flowname, flowtype)
            elif type(flowdict) == dict:    self.flows[flowname]=Flow(flowdict, flowname,flowtype)
            elif isinstance(flowdict, Flow):self.flows[flowname] = flowdict
            else: raise Exception('Invalid flow. Must be dict or flow')
    def add_fxn(self,name, flownames, fclass=GenericFxn, fparams='None'):
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
        fparams : arbitrary float, dict, list, etc.
            Other parameters to send to the __init__ method of the function class
        """
        
        if not getattr(self, 'is_copy', False):
            self.fxns[name]=fclass.__new__(fclass)
            self.fxns[name].seed=self._rng.integers(np.iinfo(np.int32).max)
            flows=self.get_flows(flownames)
            class_init_params = list(signature(fclass).parameters.keys())
            if 'name'!=class_init_params[0]:
                raise Exception('Invalid class specification for: '+str(fclass)+'. Make sure to include a name as the second argument of __init__.')
            if len(class_init_params)<2 or 'flows'!=class_init_params[1]:
                raise Exception('Invalid class specification for: '+str(fclass)+'. Make sure to include a name as the third argument of __init__.')
            if fparams=='None':
                if len(class_init_params)>2: raise Exception("fparams required by class "+str(fclass)+" __init__ method but not passed. Found in: "+name)
                self.fxns[name].__init__(name, flows)
                self._fxninput[name]={'name':name,'flows': flownames, 'fparams': 'None'}
            else: 
                if len(class_init_params)<=2: raise Exception("fparams given to class "+str(fclass)+" but __init__ has no params argument. Found in: "+name)
                self.fxns[name].__init__(name, flows,fparams)
                self._fxninput[name]={'name':name,'flows': flownames, 'fparams': fparams}
            for flowname in flownames:
                self._fxnflows.append((name, flowname))
            self.functionorder.update([name])
            self.fxns[name].set_timestep(use_local=self.use_local, global_tstep=self.tstep)
    def set_functionorder(self,functionorder):
        """Manually sets the order of functions to be executed (otherwise it will be executed based on the sequence of add_fxn calls)"""
        if not self.functionorder.difference(functionorder): self.functionorder=OrderedSet(functionorder)
        else:                                       raise Exception("Invalid list: "+str(functionorder)+" should have elements: "+str(self.functionorder))
    def get_flows(self,flownames):
        """ Returns a list of the model flow objects """
        return [self.flows[flowname] for flowname in flownames]
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
            self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() if getattr(fxn, 'behavior', False) or getattr(fxn, 'static_behavior', False) or getattr(fxn, 'asg_proptype','na')=='static'])
            self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() if getattr(fxn, 'dynamic_behavior', False) or getattr(fxn, 'asg_proptype','na')=='dynamic'])
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
        gtype : str, optional
            Type of graph to return (normal, bipartite, component, or typegraph). The default is 'bipartite'.

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
            ms = [m for m in fxn.faults.copy() if m!='nom']
            if ms: 
                modeprops[fxnname] = {}
                modes[fxnname] = ms
            for mode in ms:
                if mode!='nom': 
                    modeprops[fxnname][mode] = fxn.faultmodes.get(mode)
                    if mode not in fxn.faultmodes: warnings.warn("Mode "+mode+" not in faultmodes for fxn "+fxnname+" and may not be tracked.")
        return modes, modeprops
    def get_memory(self):
        """
        Returns the approximate memory usage of the model, along with a profile of fxn/flow memory usage.
        """
        mem_profile={}
        mem = 0
        mem_profile['params'] = sys.getsizeof(proc.flatten_hist(self.params))
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
        for fxnname, fxn in self.fxns.items():
            flownames=self._fxninput[fxnname]['flows']
            fparams=self._fxninput[fxnname]['fparams']
            flows = copy.get_flows(flownames)
            if fparams=='None':     copy.fxns[fxnname]=fxn.copy(flows)
            else:                   copy.fxns[fxnname]=fxn.copy(flows, fparams)
            copy.fxns[fxnname].set_timestep(use_local=self.use_local, global_tstep=self.tstep)
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
        self._rng=np.random.default_rng(self.seed)
    def find_classification(self, scen, mdlhists):
        """Placeholder for model find_classification methods (for running nominal models)"""
        return {'rate':scen['properties'].get('rate', 0), 'cost': 1, 'expected cost': scen['properties'].get('rate',0)}

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

class NominalApproach():
    """
    Class for defining sets of nominal simulations. To explain, a given system 
    may have a number of input situations (missions, terrain, etc) which the 
    user may want to simulate to ensure the system operates as desired. This 
    class (in conjunction with propagate.nominal_approach()) can be used to 
    perform these simulations.
    
    Attributes
    ----------
    scenarios : dict
        scenarios to inject based on the approach
    num_scenarios : int
        number of scenarios in the approach
    ranges : dict
        dict of the parameters defined in each method for the approach
    """
    def __init__(self):
        """Instantiates NominalApproach (simulation params are defined using methods)"""
        self.scenarios = {}
        self.num_scenarios = 0
        self.ranges = {}
    def add_seed_replicates(self, rangeid, seeds):
        """
        Generates an approach with different seeds to use for the model's internal stochastic behaviors

        Parameters
        ----------
        rangeid : str
            Name for the set of replicates
        seeds : int/list
            Number of seeds (if an int) or a list of seeds to use.
        """
        if type(seeds)==int: seeds = np.random.SeedSequence.generate_state(np.random.SeedSequence(),seeds)
        self.ranges[rangeid] = {'seeds':seeds, 'scenarios':[]}
        for i in range(len(seeds)):
            self.num_scenarios+=1
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]={'faults':{},'properties':{'type':'nominal','time':0.0, 'name':scenname, 'rangeid':rangeid,\
                                                                'modelparams':{'seed':seeds[i]}, 'prob':1/len(seeds)}}
            self.ranges[rangeid]['scenarios'].append(scenname)
    def add_param_replicates(self,paramfunc, rangeid, replicates, *args, ind_seeds=True, **kwargs):
        """
        Adds a set of repeated scenarios to the approach. For use in (external) random scenario generation.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the set of replicates
        replicates : int
            Number of replicates to use
        *args : any
            arguments to send to paramfunc
        ind_seeds : Bool/list
            Whether the models should be run with different seeds (rather than the same seed). Default is True
            When a list is provided, these seeds are are used. Must be of length replicates.
        **kwargs : any
            keyword arguments to send to paramfunc
        """
        if ind_seeds==True:         seeds = np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        elif type(ind_seeds)==list: 
            if len(ind_seeds)!=replicates: raise Exception("list ind_seeds must be of length replicates")
            else:                   seeds=ind_seeds
        else:                       seeds = [None for i in range(replicates)]
        self.ranges[rangeid] = {'fixedargs':args, 'inputranges':kwargs, 'scenarios':[], 'num_pts' : replicates}
        for i in range(replicates):
            self.num_scenarios+=1
            params = paramfunc(*args, **kwargs)
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]={'faults':{},\
                                      'properties':{'type':'nominal','time':0.0, 'name':scenname, 'rangeid':rangeid,\
                                                    'params':params,'inputparams':kwargs,'modelparams':{'seed':seeds[i]},\
                                                    'paramfunc':paramfunc, 'fixedargs':args, 'prob':1/replicates}}
            self.ranges[rangeid]['scenarios'].append(scenname)
    def get_param_scens(self, rangeid, *level_params):
        """
        Returns the scenarios of a range associated with given parameter ranges

        Parameters
        ----------
        rangeid : str
            Range id to check
        level_params : str (multiple)
            Level parameters iterate over

        Returns
        -------
        param_scens : dict
            The scenarios associated with each level of parameter (or joint parameters)
        """
        inputranges = {param:self.ranges[rangeid]['inputranges'][param] for param in level_params}
        partialspace= self.range_to_space(inputranges)
        partialspace = [tuple([a if isinstance(a, Hashable) else str(a) for a in p]) for p in partialspace]
        param_scens = {(p if len(p)>1 else p[0]):set() for p in partialspace}
        full_indices = list(self.ranges[rangeid]['inputranges'].keys())
        inds = [full_indices.index(param) for param in level_params]
        
        for xvals, scenarios in self.ranges[rangeid]['levels'].items():
            new_index = itemgetter(*inds)(xvals)
            if type(scenarios)==str: scenarios = [scenarios]
            param_scens[new_index].update(scenarios)
        return param_scens
    def range_to_space(self,inputranges):
        ranges = (np.arange(*arg) if type(arg)==tuple else tuple(arg) for k,arg in inputranges.items())
        space = [x for x in itertools.product(*ranges)]
        return space
    def add_param_ranges(self,paramfunc, rangeid, *args, replicates=1, seeds='shared',set_args={}, **kwargs):
        """
        Adds a set of scenarios to the approach.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the range being used. Default is 'nominal'
        *args: specifies values for positional args of paramfunc.
            May be given as a fixed float/int/dict/str defining a set value for positional arguments
        replicates : int
            Number of points to take over each range (for random parameters). Default is 1.
        seeds : str/list
            Options for seeding models/replicates: (Default is 'shared')
                - 'shared' creates random seeds and shares them between parameters and models
                - 'independent' creates separate random seeds for models and parameter generation
                - 'keep_model' uses the seed provided in the model for all of the model
            When a list is provided, these seeds are are used (and shared). Must be of length replicates.
        set_args : dict
            Dictionary of lists of values for each param e.g., {'param1':[value1, value2, value3]}
        **kwargs : specifies range for keyword args of paramfunc
            May be given as a fixed float/int/dict/str (k=value) defining a set value for the range (if not the default) or
            as a tuple k=(start, end, step) for the range, or
        """
        inputranges = {ind:rangespec for ind,rangespec in enumerate(args) if type(rangespec)==tuple}
        fixedkwargs = {k:v for k,v in kwargs.items() if not type(v)==tuple}
        inputranges = {k:v for k,v in kwargs.items() if type(v)==tuple}
        inputranges.update(set_args)
        fullspace = self.range_to_space(inputranges)
        inputnames = list(inputranges.keys())  
        
        if type(seeds)==list: 
            if len(seeds)!=replicates: raise Exception("list seeds must be of length replicates")
        else: seedstr=seeds;  seeds=np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        if seedstr=='shared':         mdlseeds=seeds
        elif seedstr=='independent':  mdlseeds=np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        elif seedstr=='keep_model':   mdlseeds= [None for i in range(replicates)]
        
        self.ranges[rangeid] = {'fixedargs':args, 'fixedkwargs':fixedkwargs, 'inputranges':inputranges, 'scenarios':[], 'num_pts' : len(fullspace), 'levels':{}, 'replicates':replicates}
        for xvals in fullspace:
            inputparams = {**{name:xvals[i] for i,name in enumerate(inputnames)}, **fixedkwargs}
            level_key = tuple([x if isinstance(x,Hashable) else str(x) for x in xvals])
            if replicates>1:    self.ranges[rangeid]['levels'][level_key]=[]
            for i in range(replicates):
                np.random.seed(seeds[i])
                self.num_scenarios+=1
                params = paramfunc(*args, **inputparams)
                scenname = rangeid+'_'+str(self.num_scenarios)
                self.scenarios[scenname]={'faults':{},\
                                          'properties':{'type':'nominal','time':0.0, 'name':scenname, 'rangeid':rangeid,\
                                                        'params':params,'inputparams':inputparams,'modelparams':{'seed':mdlseeds[i]},\
                                                        'paramfunc':paramfunc, 'fixedargs':args, 'fixedkwargs':fixedkwargs, 'prob':1/(len(fullspace)*replicates)}}
                self.ranges[rangeid]['scenarios'].append(scenname)
                if replicates>1:    self.ranges[rangeid]['levels'][level_key].append(scenname)
                else:               self.ranges[rangeid]['levels'][level_key]=scenname
    def update_factor_seeds(self, rangeid, inputparam, seeds='new'):
        """
        Changes/randomizes the seeds along a given factor in a range

        Parameters
        ----------
        rangeid : str
            Name of the range being updated
        inputparam : str
            Name of the parameter to vary the seeds over
        seeds : str/list, optional
            List of seeds to update to. The default is 'new', which picks them randomly
        """
        param_loc = [*self.ranges[rangeid]['inputranges'].keys()].index(inputparam)
        levels = [i for i in range(*self.ranges[rangeid]['inputranges'][inputparam])]
        if type(seeds) == list:
            if len(seeds)!=levels: raise Exception("Seeds (len: "+str(len(seeds))+") should math number of levels for "+inputparam+': '+str(len(levels)))
        elif seeds=="new":
            seeds = np.random.SeedSequence.generate_state(np.random.SeedSequence(), len(levels))
        else: raise Exception("Invalid option for seeds: "+str(seeds))
        for i, level in enumerate(levels):
            scens = [scen for lev, scen in self.ranges[rangeid]['levels'].items() if lev[param_loc]==level]
            for scen in scens:
                self.scenarios[scen]['properties']['modelparams']['seed'] = seeds[i]
            
    def change_params(self, rangeid='all', **kwargs):
        """
        Changes a given parameter accross all scenarios. Modifies 'params' (rather than regenerating params from the paramfunc).

        Parameters
        ----------
        rangeid : str
            Name of the range to modify. Optional. Defaults to "all"
        **kwargs : any
            Parameters to change stated as paramname=value or 
            as a dict paramname={'sub_param':value}, where 'sub_param' is the parameter of the dictionary with name paramname to update
        """
        for r in self.ranges:
            if rangeid=='all' or rangeid==r: 
                if not self.ranges.get('changes', False):   self.ranges[r]['changes'] = kwargs
                else:                                       self.ranges[r]['changes'].update(kwargs)
        for scenname, scen in self.scenarios.items():
            if rangeid=='all' or rangeid==scen['properties']['rangeid']:
                if not scen['properties'].get('changes', False):  scen['properties']['changes']=kwargs
                else:                                             scen['properties']['changes'].update(kwargs)
                for kwarg, kw_value in kwargs.items(): #updates 
                    if type(kw_value)==dict:    scen['properties']['params'][kwarg].update(kw_value)
                    else:                       scen['properties']['params'][kwarg]=kw_value
    def assoc_probs(self, rangeid, prob_weight=1.0, **inputpdfs):
        """
        Associates a probability model (assuming variable independence) with a 
        given previously-defined range of scenarios using given pdfs

        Parameters
        ----------
        rangeid : str
            Name of the range to apply the probability model to.
        prob_weight : float, optional
            Overall probability for the set of scenarios (to use if adding more ranges 
            or if the range does not cover the space of probability). The default is 1.0.
        **inputpdfs : key=(pdf, params)
            pdf to associate with the different variables of the model. 
            Where the pdf has form pdf(x, **kwargs) where x is the location and **kwargs is parameters
            (for example, scipy.stats.norm.pdf)
            and params is a dictionary of parameters (e.g., {'mu':1,'std':1}) to use '
            as the key/parameter inputs to the pdf
        """
        for scenname in self.ranges[rangeid]['scenarios']:
            inputparams = self.scenarios[scenname]['properties']['inputparams']
            inputprobs = [inpdf[0](inputparams[name], **inpdf[1]) for name, inpdf in inputpdfs.items()]
            self.scenarios[scenname]['properties']['prob'] = np.prod(inputprobs)
        totprobs = sum([self.scenarios[scenname]['properties']['prob'] for scenname in self.ranges[rangeid]['scenarios']])
        for scenname in self.ranges[rangeid]['scenarios']:
            self.scenarios[scenname]['properties']['prob'] = self.scenarios[scenname]['properties']['prob']*prob_weight/totprobs
    def add_rand_params(self, paramfunc, rangeid, *fixedargs, prob_weight=1.0, replicates=1000, seeds='shared', **randvars):
        """
        Adds a set of random scenarios to the approach.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the range being used. Default is 'nominal'
        prob_weight : float (0-1)
            Overall probability for the set of scenarios (to use if adding more ranges). Default is 1.0
        *fixedargs : any
            Fixed positional arguments in the parameter generator function. 
            Useful for discrete modes with different parameters.
        seeds : str/list
            Options for seeding models/replicates: (Default is 'shared')
                - 'shared' creates random seeds and shares them between parameters and models
                - 'independent' creates separate random seeds for models and parameter generation
                - 'keep_model' uses the seed provided in the model for all of the model
            When a list is provided, these seeds are are used (and shared). Must be of length replicates.
        **randvars : key=tuple
            Specification for each random input parameter, specified as 
            input = (randfunc, param1, param2...)
            where randfunc is the method producing random outputs (e.g. numpy.random.rand)
            and the successive parameters param1, param2, etc are inputs to the method
        """
        if type(seeds)==list: 
            if len(seeds)!=replicates: raise Exception("list seeds must be of length replicates")
        else: seedstr=seeds;  seeds=np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        if seedstr=='shared':         mdlseeds=seeds
        elif seedstr=='independent':  mdlseeds=np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        elif seedstr=='keep_model':   mdlseeds= [None for i in range(replicates)]
        
        self.ranges[rangeid] = {'fixedargs':fixedargs, 'randvars':randvars, 'scenarios':[], 'num_pts':replicates}
        for i in range(replicates):
            self.num_scenarios+=1
            np.random.seed(seeds[i])
            inputparams = {name: (ins() if callable(ins) else ins[0](*ins[1:])) for name, ins in randvars.items()}
            params = paramfunc(*fixedargs, **inputparams)
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]={'faults':{},\
                                      'properties':{'type':'nominal','time':0.0, 'name':scenname, 'rangeid':rangeid,\
                                                    'params':params,'inputparams':inputparams,'modelparams':{'seed':mdlseeds[i]},\
                                                    'paramfunc':paramfunc, 'fixedargs':fixedargs, 'prob':prob_weight/replicates}}
            self.ranges[rangeid]['scenarios'].append(scenname)
    def copy(self):
        """Copies the given sampleapproach. Used in nested scenario sampling."""
        newapp = NominalApproach()
        newapp.scenarios = copy.deepcopy(self.scenarios)
        newapp.ranges = copy.deepcopy(self.ranges)
        newapp.num_scenarios = self.num_scenarios
        return newapp
        

class SampleApproach():
    """
    Class for defining the sample approach to be used for a set of faults.
    
    Attributes
    ----------
    phases : dict
        phases given to sample the fault modes in
    globalphases : dict
        phases defined in the model
    modephases : dict
        Dictionary of modes associated with each state
    mode_phase_map : dict
        Mapping of modes to their corresponding phases
    tstep : float
        timestep defined in the model
    fxnrates : dict
        overall failure rates for each function
    comprates : dict
        overall failure rates for each component
    jointmodes : list
        (if any) joint fault modes to be injected in the approach
    rates/comprates/rates_timeless : dict
        rates of each mode (fxn, mode) in each model phase, structured {fxnmode: {phaseid:rate}}
    sampletimes : dict
        faults to inject at each time in each phase, structured {phaseid:time:fnxmode}
    weights : dict
        weight to put on each time each fault was injected, structured {fxnmode:phaseid:time:weight}
    sampparams : dict
        parameters used to sample each mode
    scenlist : list
        list of fault scenarios (dicts of faults and properties) that fault propagation iterates through
    scenids : dict
        a list of scenario ids associated with a given fault in a given phase, structured {(fxnmode,phaseid):listofnames}
    mode_phase_map : dict
        a dict of modes and their respective phases to inject with structure {fxnmode:{mode_phase_map:[starttime, endtime]}}
    units : str
        time-units to use in the approach probability model
    unit_factors : dict
        multiplication factors for converting some time units to others.
    """
    def __init__(self, mdl, faults='all', phases='global', modephases={},join_modephases=False, jointfaults={'faults':'None'}, 
                 sampparams={}, defaultsamp={'samp':'evenspacing','numpts':1}, reduce_to=False):
        """
        Initializes the sample approach for a given model

        Parameters
        ----------
        mdl : Model
            Model to sample.
        faults : str/list/tuple, optional
            - The default is 'all', which gets all fault modes from the model.
            - 'single-components' uses faults from a single component to represent faults from all components 
            - 'single-function' uses faults from a single function to represent faults from that type
            - passing the function name only includes modes from that function
            - List of faults of form [(fxn, mode)] to inject in the model.
            -Tuple arguments 
                - ('mode type', 'mode','notmode'), gets all modes with 'mode' as a string (e.g. "mech", "comms", "loss" faults). 'notmode' (if given) specifies strings to remove
                - ('mode types', ('mode1', 'mode2')), gets all modes with the listed strings (e.g. "mech", "comms", "loss" faults)
                - ('mode name', 'mode'), gets all modes with the exact name 'mode'
                - ('mode names', ('mode1', 'mode2')), gets all modes with the exact names defined in the tuple
                - ('function class', 'Classname'), which gets all modes from a function with class 'Classname'
                - ('function classes', ('Classname1', 'Classname2')), which gets all modes from a function with the names in the tuple
        phases: dict or 'global' or list
            Local phases in the model to sample. 
                Dict has structure: {'Function':{'phase':[starttime, endtime]}}
                List has structure: ['phase1', 'phase2'] where phases are phases in mdl.phases
            Defaults to 'global',here only the phases defined in mdl.phases are used.
            Phases and modephases can be gotten from process.modephases(mdlhist)
        modephases: dict
            Dictionary of modes associated with each phase. 
            For use when the opportunity vector is keyed to modes and each mode is 
            entered multiple times in a simulation, resulting in 
            multiple phases associated with that mode. Has structure:
                {'Function':{'mode':{'phase','phase1', 'phase2'...}}}
                Phases and modephases can be gotten from process.modephases(mdlhist)
        join_modephases: bool
            Whether to join phases with the same modes defined in modephases. Default is False
        jointfaults : dict, optional
            Defines how the approach considers joint faults. The default is {'faults':'None'}. Has structure:
                - faults : float    
                    # of joint faults to inject. 'all' specifies all faults at the same time
                - jointfuncs :  bool 
                    determines whether more than one mode can be injected in a single function
                - pcond (optional) : float in range (0,1) 
                    conditional probabilities for joint faults. If not give, independence is assumed.
                - inclusive (optional) : bool
                    specifies whether the fault set includes all joint faults up to the given level, or only the given level
                    (e.g., True with 'all' means SampleApproach includes every combination of joint fault modes while
                           False with 'all' means SampleApproach only includes the joint fault mode with all faults)
                - limit jointphases (optional) : int
                    Limits the number of jointphases to sample (by randomly sampling them instead). Necessary when the
                    number of faults is large
        sampparams : dict, optional
            Defines how specific modes in the model will be sampled over time. The default is {}. 
            Has structure: {(fxnmode,phase): sampparam}, where sampparam has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times, likeliest, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : dict
                    dict with structure {'nodes'[nodelist], 'weights':weightlist}
                    where the nodes in the nodelist range between -1 and 1
                    and the weights in the weightlist sum to 2.
        defaultsamp : TYPE, optional
            Defines how the model will be sampled over time by default. The default is {'samp':'evenspacing','numpts':1}. Has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times,likeliest, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : dict
                    dict with structure {'nodes'[nodelist], 'weights':weightlist}
                    where the nodes in the nodelist range between -1 and 1
                    and the weights in the weightlist sum to 2.
        reduce_to : int, optional
            Size of random sample to reduce the number of scenarios to (if any). Default is False.
        """
        self.unit_factors = {'sec':1, 'min':60,'hr':360,'day':8640,'wk':604800,'month':2592000,'year':31556952}
        if phases=='global':                self.globalphases = mdl.phases; self.phases = {}; self.modephases = modephases
        elif type(phases) in [list, set]:   self.globalphases = {ph:mdl.phases[ph] for ph in phases}; self.phases={}; self.modephases = modephases
        elif type(phases)==dict: 
            if   type(tuple(phases.values())[0])==dict:         self.globalphases = mdl.phases; self.phases = phases; self.modephases = modephases
            elif type(tuple(phases.values())[0][0]) in [int, float]:  self.globalphases = phases; self.phases ={}; self.modephases = modephases
            else:                                               self.globalphases = mdl.phases; self.phases = phases; self.modephases = modephases
        #elif type(phases)==set:    self.globalphases=mdl.phases; self.phases = {ph:mdl.phases[ph] for ph in phases}
        self.tstep = mdl.tstep
        self.units = mdl.units
        self.init_modelist(mdl,faults, jointfaults)
        self.init_rates(mdl, jointfaults=jointfaults, modephases=modephases, join_modephases=join_modephases)
        self.create_sampletimes(mdl, sampparams, defaultsamp)
        self.create_scenarios()
        if reduce_to: self.reduce_scens_to_samp(reduce_to)
    def init_modelist(self,mdl, faults, jointfaults={'faults':'None'}):
        """Initializes comprates, jointmodes internal list of modes"""
        self.comprates={}
        self._fxnmodes={}
        if faults=='all':
            self.fxnrates=dict.fromkeys(mdl.fxns)
            for fxnname, fxn in  mdl.fxns.items():
                for mode, params in fxn.faultmodes.items():
                    if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                    else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=fxn.failrate
                self.comprates[fxnname] = {compname:comp.failrate for compname, comp in fxn.components.items()}
        elif faults=='single-component':
            self.fxnrates=dict.fromkeys(mdl.fxns)
            for fxnname, fxn in  mdl.fxns.items():
                if getattr(fxn, 'components', {}):
                    firstcomp = list(fxn.components)[0]
                    for mode, params in fxn.faultmodes.items():
                        comp = fxn.compfaultmodes.get(mode, 'fxn')
                        if comp==firstcomp or comp=='fxn':
                            if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                            else:               self._fxnmodes[fxnname, mode] = params
                    self.fxnrates[fxnname]=fxn.failrate
                    self.comprates[fxnname] = {firstcomp: sum([comp.failrate for compname, comp in fxn.components.items()])}
                else:
                    for mode, params in fxn.faultmodes.items():
                        if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                        else:               self._fxnmodes[fxnname, mode] = params
                    self.fxnrates[fxnname]=fxn.failrate
                    self.comprates[fxnname] = {}
        elif faults=='single-function':
            fxnclasses = mdl.fxnclasses();
            fxns_for_class = {f:mdl.fxns_of_class(f) for f in fxnclasses} 
            fxns_to_use = {list(fxns)[0]: len(fxns) for f, fxns in fxns_for_class.items()}
            self.fxnrates=dict.fromkeys(fxns_to_use)
            for fxnname in fxns_to_use:
                fxn = mdl.fxns[fxnname]
                for mode, params in fxn.faultmodes.items():
                    if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                    else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=fxn.failrate * fxns_to_use[fxnname]
                self.comprates[fxnname] = {compname:comp.failrate for compname, comp in fxn.components.items()}
        else:
            if type(faults)==str:   faults = [(faults, mode) for mode in mdl.fxns[faults].faultmodes] #single-function modes
            elif type(faults)==tuple:
                if faults[0]=='mode name':          faults = [(fxnname, mode) for fxnname,fxn in mdl.fxns.items() for mode in fxn.faultmodes if mode==faults[1]]  
                elif faults[0]=='mode names':       faults = [(fxnname, mode) for f in faults[1] for fxnname,fxn in mdl.fxns.items() for mode in fxn.faultmodes if mode==f]  
                elif faults[0]=='mode type':        
                    faults = [(fxnname, mode) for fxnname,fxn in mdl.fxns.items() for mode in fxn.faultmodes if (faults[1] in mode and (len(faults)<3 or not faults[2] in mode))]
                elif faults[0]=='mode types':       
                    if type(faults[1])==str:    secondarg=(faults[1],)
                    else:                       secondarg=faults[1]
                    faults = [(fxnname, mode) for fxnname,fxn in mdl.fxns.items() for mode in fxn.faultmodes if any([f in mode for f in secondarg])]
                elif faults[0]=='function class':   faults = [(fxnname, mode) for fxnname,fxn in mdl.fxns_of_class(faults[1]).items() for mode in fxn.faultmodes]
                elif faults[0]=='function classes': faults = [(fxnname, mode) for f in faults[1] for fxnname,fxn in mdl.fxns_of_class(f).items() for mode in fxn.faultmodes]
                else: raise Exception("Invalid option in tuple argument: "+str(faults[0]))
            elif type(faults)==list: 
                if type(faults[0])!=tuple: raise Exception("Invalid list option: "+str(faults)+" , provide list of tuples") 
                faults=faults
            else: raise Exception("Invalid option for faults: "+str(faults)) 
            self.fxnrates=dict.fromkeys([fxnname for (fxnname, mode) in faults])
            for fxnname, mode in faults: 
                params = mdl.fxns[fxnname].faultmodes[mode]
                if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(faults),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=mdl.fxns[fxnname].failrate
                self.comprates[fxnname] = {compname:comp.failrate for compname, comp in mdl.fxns[fxnname].components.items()}
        if type(jointfaults['faults'])==int or jointfaults['faults']=='all':
            if jointfaults['faults']=='all': 
                if not jointfaults.get('jointfuncs', False): num_joint = len({i[0] for i in self._fxnmodes})
                else:                                        num_joint= len(self._fxnmodes)
            else:                                            num_joint=jointfaults['faults']
            self.jointmodes=[]
            inclusive = jointfaults.get('inclusive', True)
            if inclusive:
                for numjoint in range(2, num_joint+1):
                    jointmodes = list(itertools.combinations(self._fxnmodes, numjoint))
                    if not jointfaults.get('jointfuncs', False): 
                        jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                    self.jointmodes = self.jointmodes + jointmodes
            elif not inclusive:
                jointmodes = list(itertools.combinations(self._fxnmodes, num_joint))
                if not jointfaults.get('jointfuncs', False): 
                    jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                self.jointmodes=jointmodes
            else: raise Exception("Invalid option for jointfault['inclusive']")
        elif type(jointfaults['faults'])==list: self.jointmodes = jointfaults['faults']
        elif jointfaults['faults']!='None': raise Exception("Invalid jointfaults argument type: "+str(type(jointfaults['faults'])))
    def calc_intervaltime(self,times, tstep):
        return float(times[1]-times[0])+tstep
    def init_rates(self,mdl, jointfaults={'faults':'None'}, modephases={}, join_modephases=False):
        """ Initializes rates, rates_timeless"""
        self.rates=dict.fromkeys(self._fxnmodes)
        self.rates_timeless=dict.fromkeys(self._fxnmodes)
        self.mode_phase_map=dict.fromkeys(self._fxnmodes)
        
        for (fxnname, mode) in self._fxnmodes:
            self.rates[fxnname, mode]=dict(); self.rates_timeless[fxnname, mode]=dict(); self.mode_phase_map[fxnname, mode] = dict()
            overallrate = self.fxnrates[fxnname]
            dist = self._fxnmodes[fxnname, mode]['dist']
            if self.comprates[fxnname]:
                for compname, component in mdl.fxns[fxnname].components.items():
                    if mode in component.faultmodes:
                        overallrate=self.comprates[fxnname][compname]
            key_phases = mdl.fxns[fxnname].key_phases_by
            
            if modephases and join_modephases and (key_phases not in ['global', 'none']):
                if type (self._fxnmodes[fxnname, mode]['oppvect'])==list:
                    raise Exception("Poorly specified oppvect for fxn: "+fxnname+" mode: "+mode+"--provide a dict to use with modephases")
                oppvect = {**{phase:0 for phase in modephases[fxnname]}, **self._fxnmodes[fxnname, mode]['oppvect']}
                fxnphases = {m:[self.phases[fxnname][ph] for ph in m_phs] for m, m_phs in modephases[fxnname].items()}
            else:
                if key_phases=='global': fxnphases = self.globalphases
                elif key_phases=='none': fxnphases = {'operating':[mdl.times[0], mdl.times[-1]]} 
                else: fxnphases = self.phases.get(key_phases, self.globalphases)
                fxnphases = dict(sorted(fxnphases.items(), key = lambda item: item[1][0]))  
                if modephases and (key_phases not in ['global', 'none']):
                    modevect = self._fxnmodes[fxnname, mode]['oppvect']
                    oppvect = {phase:0 for phase in fxnphases}
                    oppvect.update({phase:modevect.get(mode, 0)/len(phases)  for mode,phases in modephases[key_phases].items() for phase in phases})
                else:
                    a=1
                    if type(self._fxnmodes[fxnname, mode]['oppvect'])==dict: 
                        oppvect = {phase:0 for phase in fxnphases}
                        oppvect.update(self._fxnmodes[fxnname, mode]['oppvect'])
                    else:
                        oppvect = self._fxnmodes[fxnname, mode]['oppvect']
                        if type(oppvect)==int or len(oppvect)==1: oppvect = {phase:1 for phase in fxnphases}
                        elif self.globalphases!=mdl.phases: oppvect = {phase:oppvect[i] for (i, phase) in enumerate(mdl.phases)}
                        elif len(oppvect)!=len(fxnphases): raise Exception("Invalid Opportunity vector: "+fxnname+". Invalid length.")
                        else: oppvect = {phase:oppvect[i] for (i, phase) in enumerate(fxnphases)}
            for phase, times in fxnphases.items():
                opp = oppvect[phase]/(sum(oppvect.values())+1e-100)
                
                if self._fxnmodes[fxnname, mode]['probtype']=='prob':   dt = mdl.tstep; unitfactor = 1
                elif type(times[0])==list:
                    dt = sum([self.calc_intervaltime(ts, mdl.tstep) for ts in times])
                    unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
                elif self._fxnmodes[fxnname, mode]['probtype']=='rate' and len(times)>1:      
                    dt = self.calc_intervaltime(times, mdl.tstep)
                    unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
                    times=[times]
                elif self._fxnmodes[fxnname, mode]['probtype']=='rate':  
                    dt = mdl.tstep
                    unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
                self.rates[fxnname, mode][key_phases, phase] = overallrate*opp*dist*dt*unitfactor #TODO: update with units
                self.rates_timeless[fxnname, mode][key_phases, phase] = overallrate*opp*dist
                self.mode_phase_map[fxnname, mode][key_phases, phase] = times
                
        if getattr(self, 'jointmodes',False):
            for (j_ind, jointmode) in enumerate(self.jointmodes):
                self.rates.update({jointmode:dict()})
                self.rates_timeless.update({jointmode:dict()})
                self.mode_phase_map.update({jointmode:dict()})
                jointphase_list = [self.mode_phase_map[mode] for mode in jointmode]
                jointphase_dict = {k:v for mode in jointmode for k,v in self.mode_phase_map[mode].items()}
                phasecombos = [i for i in itertools.product(*jointphase_list)]
                if 'limit jointphases' in jointfaults and jointfaults['limit jointphases']<len(phasecombos): 
                    rng = np.random.default_rng()
                    pc_inds = [i for i in range(len(phasecombos))]
                    pc_choices = rng.choice(pc_inds, jointfaults['limit jointphases'], replace=False)
                    phasecombos = [phasecombos[i] for i in pc_choices]
                for phase_combo in phasecombos:
                    intervals = [jointphase_dict[phase] for phase in phase_combo]
                    overlap, intervals_times = find_overlap_n(intervals)
                    if overlap: 
                        phaseid = tuple(set(phase_combo))
                        if len(phaseid) == 1: 
                            phaseid = phaseid[0]
                            rates=[self.rates[fmode][phaseid] for fmode in jointmode]
                        else:
                            rates = [self.rates[fmode][phase_combo[i]]* len(overlap)/intervals_times[i] for i,fmode in enumerate(jointmode)]
                        if not jointfaults.get('pcond', False): # if no input, assume independence
                            prob = np.prod(1-np.exp(-np.array(rates)))
                            self.rates[jointmode][phaseid] = -np.log(1.0-prob)
                        elif type(jointfaults['pcond']) in [float, int]:
                            self.rates[jointmode][phaseid] = jointfaults['pcond']*max(rates)
                        elif type(jointfaults['pcond'])==list:
                            self.rates[jointmode][phaseid] = jointfaults['pcond'][j_ind]*max(rates)
                        else: raise Exception("Invalid pcond argument in jointfaults: "+str(jointfaults['pcond']))
                        if len(overlap)>1:  
                            self.rates_timeless[jointmode][phaseid] = self.rates[jointmode][phaseid]/(len(overlap)*self.tstep)
                        else:
                            self.rates_timeless[jointmode][phaseid] = self.rates[jointmode][phaseid]
                        self.mode_phase_map[jointmode][phaseid] = overlap 
            if not jointfaults.get('inclusive', True): 
                for (fxnname, mode) in self._fxnmodes: 
                    self.rates.pop((fxnname,mode))
                    self.rates_timeless.pop((fxnname,mode))
                    self.mode_phase_map.pop((fxnname,mode))
    def create_sampletimes(self,mdl, params={}, default={'samp':'evenspacing','numpts':1}):
        """ Initializes weights and sampletimes """
        self.sampletimes={}
        self.weights={fxnmode:dict.fromkeys(rate) for fxnmode,rate in self.rates.items()}
        self.sampparams={}
        for fxnmode, ratedict in self.rates.items():
            for phaseid, rate in ratedict.items():
                if rate > 0.0:
                    times = self.mode_phase_map[fxnmode][phaseid]
                    param = params.get((fxnmode,phaseid), default)
                    self.sampparams[fxnmode, phaseid] = param
                    if type(times[0])!=list: times=[times]
                    possible_phasetimes=[]
                    for ts in times: 
                        if len(ts)==1:      possible_phasetimes = ts
                        elif len(ts)<2:     possible_phasetimes= ts
                        else:               possible_phasetimes = possible_phasetimes + list(np.arange(ts[0], ts[1]+self.tstep, self.tstep))
                    possible_phasetimes.sort()
                    possible_phasetimes=list(set(possible_phasetimes))
                    if len(possible_phasetimes)<=1: 
                        a=1
                        self.add_phasetimes(fxnmode, phaseid, possible_phasetimes)
                    else:
                        if param['samp']=='likeliest':
                            weights=[]
                            if self.rates[fxnmode][phaseid] == max(list(self.rates[fxnmode].values())):
                                phasetimes = [round(np.quantile(possible_phasetimes, 0.5)/self.tstep)*self.tstep]
                            else: phasetimes = []
                        else: 
                            pts, weights = self.select_points(param, [pt for pt, t in enumerate(possible_phasetimes)])
                            phasetimes = [possible_phasetimes[pt] for pt in pts]
                        self.add_phasetimes(fxnmode, phaseid, phasetimes, weights=weights)
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
                - 'quad' : dict
                    dict with structure {'nodes'[nodelist], 'weights':weightlist}
                    where the nodes in the nodelist range between -1 and 1
                    and the weights in the weightlist sum to 2.
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
            quantiles = param['quad']['nodes']/2 +0.5
            if len(quantiles) > len(possible_pts): pts = possible_pts
            else: 
                pts= [int(round(np.quantile(possible_pts, q))) for q in quantiles]
                weights=param['quad']['weights']/sum(param['quad']['weights'])
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
    def add_phasetimes(self, fxnmode, phaseid, phasetimes, weights=[]):
        """ Adds a set of times for a given mode to sampletimes"""
        if phasetimes:
            if not self.weights[fxnmode].get(phaseid): self.weights[fxnmode][phaseid] = {t: 1/len(phasetimes) for t in phasetimes}
            for (ind, time) in enumerate(phasetimes):
                if not self.sampletimes.get(phaseid): 
                    self.sampletimes[phaseid] = {time:[]}
                if self.sampletimes[phaseid].get(time): self.sampletimes[phaseid][time] = self.sampletimes[phaseid][time] + [(fxnmode)]
                else: self.sampletimes[phaseid][time] = [(fxnmode)]
                if any(weights): self.weights[fxnmode][phaseid][time] = weights[ind]
                else:       self.weights[fxnmode][phaseid][time] = 1/len(phasetimes)
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
        for phaseid, samples in self.sampletimes.items():
            if samples:
                for time, faultlist in samples.items():
                    self.times+=[time]
                    for fxnmode in faultlist:
                        if self.sampparams[fxnmode, phaseid]['samp']=='maxlike':    
                            rate = sum(self.rates[fxnmode].values())
                        else: 
                            rate = self.rates[fxnmode][phaseid] * self.weights[fxnmode][phaseid][time]
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
                        if self.scenids.get((fxnmode, phaseid)): self.scenids[fxnmode, phaseid] = self.scenids[fxnmode, phaseid] + [name]
                        else: self.scenids[fxnmode, phaseid] = [name]
        self.times = list(set(self.times))
        self.times.sort()
    def reduce_scens_to_samp(self, samp_size=100,seed=None):
        """Reduces the number of scenarios (in the scenlist) to a given sample size samp_size. Useful for
        choosing a random subset of an approach which would otherwise have a large number of scenarios.
        Note that many structures may not be preserved and some artefacts may be present."""
        if samp_size<len(self.scenlist):
            rng = np.random.default_rng(seed)
            self.scenlist = rng.choice(self.scenlist, samp_size, replace=False)
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
                if not self.phases or modeinphase[1][0]=='global': 
                    beginning, end = self.globalphases[modeinphase[1][1]]
                else: 
                    beginning, end = self.phases[modeinphase[1][0]][modeinphase[1][1]]
                partlocs=[0, len(list(np.arange(beginning,end, self.tstep)))]
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
        if joint and hasattr(self, 'jointmodes'):
            return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()] + self.jointmodes
        else:
            return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()]
    def list_moderates(self):
        """ Returns the rates for each mode """
        return {(fxn, mode): sum(self.rates[fxn,mode].values()) for (fxn, mode) in self.rates.keys()}
    def get_scenid_groups(self,group_by='phases', group_dict={}):
        """
        Returns a dict with different scenario ids grouped according to group_by. 
        group_by: str, with options:
        - 'none':           Returns {'scenid':'scenid'} for all scenarios
        - 'phase':          Returns {(fxnmode, fxnphase):{scenids}}--identical scenarios within a given phase are grouped 
        - 'fxnfault':       Returns {fxnmode:{scenids}} All identical scenarios (fxn, mode) are grouped
        - 'mode':           Returns {mode:{scenids}}. All scenarios with the same mode name are grouped
        - 'mode type':      Returns {modetype:scenids}. All scenarios with the same mode type (mode types must be given to the sampleapproach) are grouped
        - 'functions':      Returns {function:scenids}. All scenarios and modes from a given function are grouped.
        - 'times':          Returns {time:scenids}. All scenarios at a given time are grouped.
        - 'fxnclassfault':  Returns {(fxnclass, mode):scenids}. All scenarios (fxnclass, mode) from a given function class are grouped.
        - 'fxnclass':       Returns {fxnclass:scendis}. All scenarios from a given function class are grouped.
        For 'fxnclass', 'fxnclassfault', and 'modetype', a group_dict dictionary must be provided that groups the function/mode classes/types.
        -------------------
        Returns:
        - grouped_scens: dict
              A dictionary of the scenario ids associated with the given group {group:scenids}  
        """
        if group_by in ['fxnclass', 'fxnclassfault', 'modetype'] and not group_dict:
            raise Exception("group_dict must be provided to group by these")
        if group_by=='none':         grouped_scens =   {s:[s] for v in self.scenids.values() for s in v}
        elif group_by=='phase':      grouped_scens =   self.scenids
        elif group_by=='fxnfault':   
            grouped_scens = {m:set() for m in self.list_modes(True)}
            for modephase, ids in self.scenids.items(): grouped_scens[modephase[0]].update(ids)
        elif group_by=='mode':
            grouped_scens = {m[1]:set() for m in self.list_modes(True)}
            for modephase, ids in self.scenids.items(): grouped_scens[modephase[0][1]].update(ids)
        elif group_by=='functions':
            grouped_scens = {m[0]:set() for m in self.list_modes(True)}
            for modephase, ids in self.scenids.items(): grouped_scens[modephase[0][0]].update(ids)
        elif group_by=='times':
            grouped_scens = {float(t):set() for t in set(self.times)}
            for scen in self.scenlist: 
                time = float(scen['properties']['time'])
                grouped_scens[time].add(scen['properties']['name'])
        elif group_by=='fxnclass':
            fxn_groups = {sub_v:k for k,v in group_dict.items() for sub_v in v}
            grouped_scens= {fxn_groups[fxnmode[0]]:set() for fxnmode in self.list_modes(True)}
            grouped_scens['nominal']={'nominal'}
            for modephase, ids in self.scenids.items(): 
                fxn = modephase[0][0]
                group = fxn_groups[fxn]
                grouped_scens[group].update(ids)
        elif group_by=='fxnclassfault':
            fxn_groups = {sub_v:k for k,v in group_dict.items() for sub_v in v}
            grouped_scens= {(fxn_groups[fxnmode[0]], fxnmode[1]):set() for fxnmode in self.list_modes(True)}
            grouped_scens['nominal']={'nominal'}
            for modephase, ids in self.scenids.items(): 
                fxn, mode = modephase[0]
                group = fxn_groups[fxn]
                grouped_scens[group, mode].update(ids)
        elif group_by=='modetype':
            grouped_scens= {group:set() for group in group_dict}
            grouped_scens['ungrouped'] =set()
            for modephase, ids in self.scenids.items(): 
                mode = modephase[0][1]
                grouped=False
                for group in grouped_scens:
                    if group in mode: 
                        grouped_scens[group].update(ids)
                        grouped=True
                        break
                if not grouped: grouped_scens['ungrouped'].update(ids)    
        else: raise Exception("Invalid option for group_by: "+group_by)
        return grouped_scens
    def get_id_weights(self):
        """Returns a dictionary with weights for each scenario with structure {scenid:weight}"""
        id_weights ={}
        for scens, ids in self.scenids.items():
            num_phases = len([n for n,i in self.weights[scens[0]].items() if i])
            weights = np.array([*self.weights[scens[0]][scens[1]].values()])/num_phases
            id_weights.update({scenid:weights[i] for i,scenid in enumerate(ids)})
        return id_weights



def find_overlap_n(intervals):
    """Finds the overlap between given intervals.
    Used to sample joint fault modes with different (potentially overlapping) phases """
    try:
        joined_times={}
        intervals_times = []
        for i, interval in enumerate(intervals):
            if type(interval[0]) in [float, int]: interval=[interval]
            possible_times = set()
            possible_times.update(*[{*np.arange(i[0],i[-1]+1)} for i in interval])
            if i==0:    joined_times = possible_times
            else:       joined_times = joined_times.intersection(possible_times)
            intervals_times.append(len(possible_times))
        if not joined_times:    return [], intervals_times
        else:                   return [*np.sort([*joined_times])], intervals_times
    except IndexError:
        if all(intervals[0]==i for i in intervals): return intervals[0]
        else:                                       return 0
    

def phases(times, names=[]):
    """ Creates named phases from a set of times defining the edges of the intervals """
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

def trunc(x, n=2.0, truncif='greater'):
    """truncates a value to a given number (useful if behavior unchanged by increases)
    
    Parameters
    ----------
    x : float/int 
        number to truncate
    n : float/int (optional)
        number to truncate to if >= number
    truncif: 'greater'/'less'
        whether to truncate if greater or less than the given number
    """
    if truncif=='greater' and x>n:      y=n
    elif  truncif=='greater' and x<n:   y=n
    else:                               y=x
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

def is_iter(data):
    """ Checks whether a data type should be interpreted as an iterable or not and returned
    as a single value or tuple/array"""
    if isinstance(data, Iterable) and type(data)!=str:  return True
    else:                                               return False
"""Model checking"""
def check_pickleability(obj, verbose=True):
    """ Checks to see which attributes of an object will pickle (and thus parallelize)"""
    unpickleable = []
    for name, attribute in vars(obj).items():
        if not dill.pickles(attribute):
            unpickleable = unpickleable + [name]
    if verbose:
        if unpickleable: print("The following attributes will not pickle: "+str(unpickleable))
        else:           print("The object is pickleable")
    return unpickleable

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

