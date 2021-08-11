# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""
import numpy as np
import itertools
import dill
import pickle
import networkx as nx
from ordered_set import OrderedSet

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
    def get(self, *attnames):
        """Returns the given attribute names (strings). Mainly useful for reducing length
        of lines/adding clarity to assignment statements. 
        e.g., x,y = self.Pos.get('x','y') is the same as
              x,y = self.Pos.x, self.Pos.y
        """
        if len(attnames)==1:   return getattr(self,attnames[0])
        else:                   return tuple(getattr(self,name) for name in attnames)
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
        """Returns the multiplication of given states of the function/flow"""
        a= getattr(self,states[0])
        for state in states[1:]:
            a = a * getattr(self,state)
        return a
    def div(self,*states):
        """Returns the division of given states of the function/flow"""
        a= getattr(self,states[0])
        for state in states[1:]:
            a = a / getattr(self,state)
        return a
    def add(self,*states):
        """Returns the addition of given states of the function/flow"""
        a= 0.0
        for state in states:
            a += getattr(self,state)
        return a
    def sub(self,*states):
        """Returns the addition of given states of the function/flow"""
        a= getattr(self,states[0])
        for state in states[1:]:
            a -= getattr(self,state)
        return a
    
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
        for state in states.keys():
            setattr(self, state,states[state])
        self.faults=set(['nom'])
        self.opermodes= getattr(self, 'opermodes', [])
        self.faultmodes= getattr(self, 'faultmodes', {})
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
    def assoc_healthstates(self, hstates, mode_app='single-state', probtype='prob', units='hr'):
        """
        Adds health state attributes to the model (and a mode approach if desired). 
        
        Parameters
        ----------
        hstates : Dict
            Health states to incorporate in the model and their respective values. 
            e.g., {'state':[1,{0,2,-1}]}, {'state':{0,2,-1}}
        mode_app : str
            type of modes to elaborate from the given health states.
        """
        if not hasattr(self,'_states'): raise Exception("Call __init__ method for function first")
        hranges = dict.fromkeys(hstates.keys())
        nom_hstates = {}
        for state in hstates:
            self._states.append(state)
            if type(hstates[state]) in [set, np.ndarray]:               
                nom_hstates[state] = 1.0
                hranges[state]=set(hstates[state])
            elif  type(hstates[state])==list:           
                nom_hstates[state] = hstates[state][0]
                hranges[state]=set(hstates[state][1]) 
            elif type(hstates[state]) in [float, int]:  
                nom_hstates[state] = hstates[state]
                hranges[state]={}
            else: raise Exception("Invalid input option for health state")
            setattr(self, state, nom_hstates[state])
            self._initstates.update(nom_hstates)
        self.assoc_healthstate_modes(hranges=hranges, mode_app=mode_app, probtype=probtype, units=units)
    def assoc_healthstate_modes(self, hranges = {}, mode_app = 'none', manual_modes={}, probtype='prob', units='hr', key_phases_by='none'):
        """
        Associates modes with given healthstates.

        Parameters
        ----------
        hranges : dict, optional
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
            nom_hstates = {state: self._initstates[state] for state in hranges}
            if mode_app=='none': a=0
            elif mode_app=='single-state':
                for state in hranges:
                    modes = {state+'_'+str(value):'synth' for value in hranges[state]}
                    modestates = {state+'_'+str(value): {state:value} for value in hranges[state]}
                    self.faultmodes.update(modes)
                    self.mode_state_dict.update(modestates)
            elif mode_app =='all' or type(mode_app)==int:
                for state in hranges: hranges[state].add(nom_hstates[state])
                nomvals = tuple([*nom_hstates.values()])
                statecombos = [i for i in itertools.product(*hranges.values()) if i!=nomvals]
                if type(mode_app)==int: 
                    sample = np.random.choice([i for i,_ in enumerate(statecombos)], size=mode_app, replace=False)
                    statecombos = [statecombos[i] for i in sample]
                self.faultmodes.update({'hmode_'+str(i):'synth' for i in range(len(statecombos))}) 
                self.mode_state_dict.update({'hmode_'+str(i): {list(hranges)[j]:state for j, state in enumerate(statecombos[i])} for i in range(len(statecombos))})
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
        self.failrate = gtp*EPC_f
    def assoc_modes(self, faultmodes={}, opermodes=[],initmode='nom', name='', probtype='rate', units='hr', exclusive=False, key_phases_by='none', longnames={}):
        """
        Associates fault and operational modes with the block when called in the function or component.

        Parameters
        ----------
        faultmodes : dict, optional
            Dictionary/Set of faultmodes with structure, which can have the forms:
                - set {'fault1', 'fault2', 'fault3'} (just the respective faults)
                - dict {'fault1': faultattributes, 'fault2': faultattributes}, where faultattributes is:
                    - float: rate for the mode
                    - [float, float]: rate and repair cost for the mode
                    - float, oppvect, float]: rate, opportunity vector, and repair cost for the mode
                    opportunity vector can be specified as:
                        [float1, float2,...], a vector of relative likelihoods for each phase, or
                        {opermode:float1, opermode:float1}, a dict of relative likelihoods for each phase/mode
                        the phases/modes to key by are defined in "key_phases_by"
        opermodes : list, optional
            List of operational modes
        initmode : str, optional
            Initial operational mode. Default is 'nom'
        name : str, optional
            (for components only) Name of the component. The default is ''.
        probtype : str, optional
            Type of probability in the probability model, a per-time 'rate' or per-run 'prob'. 
            The default is 'rate'
        units : str, optional
            Type of units ('sec'/'min'/'hr'/'day') used for the rates. Default is 'hr' 
        exclusive : True/False
            Whether fault modes are exclusive of each other or not. Default is False (i.e. more than one can be present). 
        key_phases_by : 'self'/'none'/'global'/'fxnname'
            Phases to key the faultmodes by (using local, global, or an external function's modes'). Default is 'none'
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
        self.exclusive_faultmodes = exclusive
        if not getattr(self, 'is_copy', False): #saves time by using the same fault mode dictionary from previous
            if not getattr(self, 'faultmodes', []): 
                if name: self.faultmodes=dict()
                else:    self.faultmodes=dict.fromkeys(faultmodes)
            for mode in faultmodes:
                self.faultmodes[name+mode]=dict.fromkeys(('dist', 'oppvect', 'rcost', 'probtype', 'units'))
                self.faultmodes[name+mode]['probtype'] = probtype
                self.faultmodes[name+mode]['units'] = units
                if type(faultmodes) == set: # minimum information - here the faultmodes are only a set of labels
                    self.faultmodes[name+mode]['dist'] =     1.0/len(faultmodes)
                    self.faultmodes[name+mode]['oppvect'] =  [1.0]
                    self.faultmodes[name+mode]['rcost'] =    0.0
                elif type(faultmodes[mode]) == float: # dict of modes: dist, where dist is the distribution (or individual rate/probability)
                    self.faultmodes[name+mode]['dist'] =     faultmodes[mode]
                    self.faultmodes[name+mode]['oppvect'] =  [1.0]
                    self.faultmodes[name+mode]['rcost'] =    0.0
                elif len(faultmodes[mode]) == 3:   # three-arg mode definition: dist, oppvect, repair costs
                    self.faultmodes[name+mode]['dist'] =     faultmodes[mode][0]
                    self.faultmodes[name+mode]['oppvect'] =  faultmodes[mode][1]
                    self.faultmodes[name+mode]['rcost'] =    faultmodes[mode][2]
                    if key_phases_by =='none': raise Exception("How should the opportunity vector be keyed? Provide 'key_phases_by' option.")
                elif len(faultmodes[mode]) == 2:  # dist, repair costs
                    self.faultmodes[name+mode]['dist'] =     faultmodes[mode][0]
                    self.faultmodes[name+mode]['oppvect'] =  [1.0]
                    self.faultmodes[name+mode]['rcost'] =    faultmodes[mode][1]
                elif len(faultmodes[mode]) == 1:  # dist only
                    self.faultmodes[name+mode]['dist'] =     faultmodes[mode][0]
                    self.faultmodes[name+mode]['oppvect'] =  [1.0]
                    self.faultmodes[name+mode]['rcost'] =    0.0
                else:
                    raise Exception("Invalid mode definition")
                self.faultmodes[name+mode]['longname'] = longnames.get(mode,mode)
        if key_phases_by=='self':   self.key_phases_by = self.name
        else:                       self.key_phases_by = key_phases_by
    def set_mode(self, mode):
        """Sets a mode in the block"""
        if self.exclusive_faultmodes and self.any_faults():
            raise Exception("Cannot set mode from fault state without removing faults.")
        else:   self.mode = mode
    def in_mode(self,mode):
        "Checks if the system is in a given operational mode"
        return self.mode==mode 
    def in_modes(self, modes):
        "Checks if the block has one of the given list of operational modes"
        return self.mode in modes                      
    def has_fault(self,fault): 
        """Check if the block has fault (a str)"""
        return any(self.faults.intersection(set([fault])))
    def no_fault(self,fault): 
        """Check if the block has fault (a str)"""
        return not(any(self.faults.intersection(set([fault]))))
    def has_faults(self,faults): 
        """Check if the block has any in the list of faults"""
        return any(self.faults.intersection(set(faults)))
    def any_faults(self):
        """check if the block has any fault modes"""
        return any(self.faults.difference({'nom'}))
    def to_fault(self,fault): 
        """Moves from the current fault mode to a new fault mode"""
        self.faults.clear()
        self.faults.add(fault)
        if self.exclusive_faultmodes: self.mode = fault
    def add_fault(self,fault): 
        """Adds fault (a str) to the block"""
        self.faults.update([fault])
        if self.exclusive_faultmodes: self.mode = fault
    def add_faults(self,faults): 
        """Adds list of faults to the block"""
        self.faults.update(faults)
        if self.exclusive_faultmodes: 
            if len(faults)>1:   raise Exception("Multiple fault modes added to function with exclusive fault representation")
            elif len(faults)==1 and list(faults)[0]!='nom': self.mode =faults[0]
    def replace_fault(self, fault_to_replace,fault_to_add): 
        """Replaces fault_to_replace with fault_to_add in the set of faults"""
        self.faults.add(fault_to_add)
        self.faults.remove(fault_to_replace)
        if self.exclusive_faultmodes: self.mode = fault_to_add
    def remove_fault(self, fault_to_remove, opermode=False):
        """Removes fault in the set of faults and returns to given operational mode"""
        self.faults.discard(fault_to_remove)
        if len(self.faults) == 0: self.faults.add('nom')
        if opermode:    self.mode = opermode
        if self.exclusive_faultmodes and not(opermode):
            raise Exception("Unclear which operational mode to enter with fault removed")
    def remove_any_faults(self, opermode=False):
        """Resets fault mode to nominal and returns to the given operational mode"""
        self.faults.clear()
        self.faults.add('nom')
        if opermode:    self.mode = opermode
        if self.exclusive_faultmodes and not(opermode):
            raise Exception("Unclear which operational mode to enter with fault removed")
    def get_flowtypes(self):
        """Returns the names of the flow types in the model"""
        return {obj.type for name, obj in self.flows.items()}
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
        timestep of the model in the function (added/overridden by model definition)
    """
    def __init__(self,name, flows, flownames=[], states={}, components={},timers={}, tstep=1.0):
        """
        Intances the function superclass with the relevant parameters.

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
        self.flows=self.make_flowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow,self.flows[flow])
        self.components=components
        if not getattr(self, 'faultmodes', []): self.faultmodes={}
        if self.components: self.compfaultmodes= dict()
        self.exclusive_faultmodes = False
        for cname in components:
            self.faultmodes.update(components[cname].faultmodes)
            self.compfaultmodes.update({modename:cname for modename in components[cname].faultmodes})
        self.timers = timers
        for timername in timers:
            setattr(self, timername, Timer(timername))
        self.tstep=tstep
        super().__init__(states)
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
        self.updatefxn('reset', faults=['nom'], time=0)
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
        for state in self._initstates.keys():
            setattr(copy, state, getattr(self, state))
        if hasattr(self, 'time'): copy.time=self.time
        if hasattr(self, 'tstep'): copy.tstep=self.tstep
        return copy
    def update_modestates(self):
        num_update = 0
        for fault in self.faults:
            if fault in self.mode_state_dict:
                for state, value in self.mode_state_dict[fault].items():
                    setattr(self, state, value)
                num_update+=1
                if num_update > 1: raise Exception("Exclusive fault mode scenarios present at the same time")
    def updatefxn(self,proptype, faults=[], time=0):
        """
        Updates the state of the function at a given time and injects faults.

        Parameters
        ----------
        faults : list, optional
            Faults to inject in the function. The default is ['nom'].
        time : float, optional
            Model time. The default is 0.
        """
        self.add_faults(faults)  #if there is a fault, it is instantiated in the function
        if hasattr(self, 'condfaults'):    self.condfaults(time)    #conditional faults and behavior are then run
        if hasattr(self, 'mode_state_dict') and any(faults): self.update_modestates()
        if self.components:     # propogate faults from function level to component level
            for fault in self.faults:
                if fault in self.compfaultmodes:
                    self.components[self.compfaultmodes[fault]].add_fault(fault)
        if proptype=='static' and hasattr(self,'behavior'):        self.behavior(time)     #generic behavioral methods are run at all steps
        if proptype=='static' and hasattr(self,'static_behavior'):                          self.static_behavior(time)
        elif proptype=='dynamic' and hasattr(self,'dynamic_behavior') and time > self.time: self.dynamic_behavior(time)
        elif proptype=='reset':                                                             
            if hasattr(self,'static_behavior'):  self.static_behavior(time)
            if hasattr(self,'dynamic_behavior'): self.dynamic_behavior(time)
        if self.components:     # propogate faults from component level to function level
            for compname, comp in self.components.items():
                self.faults.update(comp.faults) 
        self.time=time
        if self.faults.difference({'nom'}): self.faults.difference_update({'nom'})
        elif len(self.faults)==0:           self.faults.update(['nom'])
        if self.exclusive_faultmodes==True and len(self.faults)>1: 
            raise Exception("More than one fault present in "+self.name+"\n at t= "+str(time)+"\n faults: "+str(self.faults)+"\n Is the mode representation nonexclusive?")
        return
class GenericFxn(FxnBlock):
    """Generic function block. For use when there is no Function Block defined"""
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
        """ Placeholder for component behavior methods """
        return 0

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
    params : dict
        dictionary of (optional) parameters for a given instantiation of a model
    bipartite : networkx graph
        bipartite graph view of the functions and flows
    graph : networkx graph
        multigraph view of functions and flows
    """
    def __init__(self, params={},modelparams={}, valparams='all'):
        """
        Instantiates internal model attributes with predetermined:
            - params (design variables of he model), and
            - modelparams (dictionary of 
                           global phases {'phase': [starttime, endtime]}
                           times [starttime, ..., endtime] (middle time used for sampling), 
                           timestep (float) to run the model with)
                           seed (int) - if present, sets a seed to run the  
            - valparams (`all`/`flows`/`fxns`/or dict of the form of mdlhist {fxns:{fxn1:{param1}}, flows:{flow1:{param1}}})
        """
        self.type='model'
        self.flows={}
        self.fxns={}
        self.params=params
        self.valparams = valparams
        self.modelparams=modelparams
        
        # model defaults to static representation if no timerange
        self.phases=modelparams.get('phases',{'na':[1]})
        self.times=modelparams.get('times',[1])
        self.tstep = modelparams.get('tstep', 1.0)
        self.units = modelparams.get('units', 'hr')
        if modelparams.get('seed', False):  
            np.random.seed(modelparams['seed'])
            self.seed=self.modelparams['seed']
        else:                               
            np.random.seed()
            self.seed = np.random.get_state()[1][0]
            modelparams['seed']=self.seed
        
        self.functionorder=OrderedSet() #set is ordered and executed in the order specified in the model
        self._fxnflows=[]
        self._fxninput={}
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
            flows=self.get_flows(flownames)
            if fparams=='None':
                self.fxns[name]=fclass(name, flows)
                self._fxninput[name]={'name':name,'flows': flownames, 'fparams': 'None'}
            else: 
                self.fxns[name]=fclass(name, flows,fparams)
                self._fxninput[name]={'name':name,'flows': flownames, 'fparams': fparams}
            for flowname in flownames:
                self._fxnflows.append((name, flowname))
            self.functionorder.update([name])
            self.fxns[name].tstep=self.tstep
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
        class_relationship = dict()
        for fxn, obj in self.fxns.items():
            if class_relationship.get(obj.__class__.__name__,False):
                class_relationship[obj.__class__.__name__].update(obj.get_flowtypes())
            else: class_relationship[obj.__class__.__name__] = set(obj.get_flowtypes())
        return class_relationship
    def build_model(self, functionorder=[], graph_pos={}, bipartite_pos={}):
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
            self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() if getattr(fxn, 'behavior', False) or getattr(fxn, 'static_behavior', False)])
            self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items() if getattr(fxn, 'dynamic_behavior', False)])
            self.construct_graph(graph_pos, bipartite_pos)
            self.staticflows = [flow for flow in self.flows if any([ n in self.staticfxns for n in self.bipartite.neighbors(flow)])]
    def construct_graph(self, graph_pos={}, bipartite_pos={}):
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
        if dangling_nodes: raise Exception("Fxns/flows disconnected from model: "+str(dangling_nodes))
        
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
                graph.add_nodes_from(fxn.components, bipartite=1)
                graph.add_edges_from([(fxnname, component) for component in fxn.components])
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
            copy.fxns[fxnname].tstep=self.tstep
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
    def find_classification(self, scen, mdlhists):
        """Placeholder for model find_classification methods (for running nominal models)"""
        return {'rate':scen['properties']['rate'], 'cost': 1, 'expected cost': scen['properties']['rate']}

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
        
class NominalApproach():
    """
    Class for defining sets of nominal simulations. To explain, a given system 
    may have a number of input situations (missions, terrain, etc) which the 
    user may want to simulate to ensure the system operates as desired. This 
    class (in conjunction with propagate.nominal_approach()) can be used to 
    perform these simulations.
    """
    def __init__(self):
        self.scenarios = {}
        self.num_scenarios = 0
        self.ranges = {}
    def add_param_ranges(self,paramfunc, rangeid, *fixedargs, **inputranges):
        """
        Adds a set of scenarios to the approach.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the range being used. Default is 'nominal'
        *fixedargs : any
            Fixed positional arguments in the parameter generator function. 
            Useful for discrete modes with different parameters.
        **inputranges : key=tuple
            Ranges for each input argument to be iterated over specified as input = (start, end, step)
            (note that end is not inclusive)
        """
        ranges = (np.arange(*arg) for k,arg in inputranges.items())
        fullspace = [x for x in itertools.product(*ranges)]
        inputnames = list(inputranges.keys())
        self.ranges[rangeid] = {'fixedargs':fixedargs, 'inputranges':inputranges, 'scenarios':[], 'num_pts' : len(fullspace)}
        for xvals in fullspace:
            self.num_scenarios+=1
            inputparams = {name:xvals[i] for i,name in enumerate(inputnames)}
            params = paramfunc(*fixedargs, **inputparams)
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]={'faults':{},\
                                      'properties':{'type':'nominal','time':0.0, 'name':scenname,\
                                                    'params':params,'inputparams':inputparams,\
                                                    'paramfunc':paramfunc, 'fixedargs':fixedargs, 'prob':1/len(fullspace)}}
            self.ranges[rangeid]['scenarios'].append(scenname)
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
    def add_rand_params(self, paramfunc, rangeid, *fixedargs, prob_weight=1.0, num_pts=1000, **randvars):
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
        **inputranges : key=tuple
            Specification for each random input parameter, specified as 
            input = (randfunc, param1, param2...)
            where randfunc is the method producing random outputs (e.g. numpy.random.rand)
            and the successive parameters param1, param2, etc are inputs to the method
        """
        self.ranges[rangeid] = {'fixedargs':fixedargs, 'randvars':randvars, 'scenarios':[], 'num_pts':num_pts}
        for i in range(num_pts):
            self.num_scenarios+=1
            inputparams = {name: (ins() if callable(ins) else ins[0](*ins[1:])) for name, ins in randvars.items()}
            params = paramfunc(*fixedargs, **inputparams)
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]={'faults':{},\
                                      'properties':{'type':'nominal','time':0.0, 'name':scenname,\
                                                    'params':params,'inputparams':inputparams,\
                                                    'paramfunc':paramfunc, 'fixedargs':fixedargs, 'prob':prob_weight/num_pts}}
            self.ranges[rangeid]['scenarios'].append(scenname)
            

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
    """
    def __init__(self, mdl, faults='all', phases='global', modephases={}, jointfaults={'faults':'None'}, sampparams={}, defaultsamp={'samp':'evenspacing','numpts':1}):
        """
        Initializes the sample approach for a given model

        Parameters
        ----------
        mdl : Model
            Model to sample.
        faults : str (all/single-component/function of interest) or list, optional
            List of faults (tuple (fxn, mode)) to inject in the model. The default is 'all'. 'single-components' uses faults from a single component to represent faults form all components while passing the function name only inludes modes from that function
        phases: dict or 'global'
            Local phases in the model to sample. Has structure:
                {'Function':{'phase':[starttime, endtime]}}
            Defaults to 'global',here only the phases defined in mdl.phases are used.
            Phases and modephases can be gotten from process.modephases(mdlhist)
        modephases: dict
            Dictionary of modes associated with each phase. 
            For use when the opportunity vector is keyed to modes and each mode is 
            entered multiple times in a simulation, resulting in 
            multiple phases associated with that mode. Has structure:
                {'Function':{'mode':{'phase','phase1', 'phase2'...}}}
                Phases and modephases can be gotten from process.modephases(mdlhist)
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
        if phases=='global':        self.globalphases = mdl.phases; self.phases = {}; self.modephases = modephases
        elif type(phases)==list:    self.globalphases = {ph:mdl.phases[ph] for ph in phases}; self.phases = {}; self.modephases = modephases
        elif type(phases)==dict:    
            if type(tuple(phases.values())[0][0]) in [int, float]:  self.globalphases = phases; self.phases ={}; self.modephases = modephases
            else:                                               self.globalphases = mdl.phases; self.phases = phases; self.modephases = modephases
        self.tstep = mdl.tstep
        self.units = mdl.units
        self.init_modelist(mdl,faults, jointfaults)
        self.init_rates(mdl, jointfaults=jointfaults, modephases=modephases)
        self.create_sampletimes(mdl, sampparams, defaultsamp)
        self.create_scenarios()
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
        else:
            if type(faults)==str:
                faults = [(faults, mode) for mode in mdl.fxns[faults].faultmodes]
            self.fxnrates=dict.fromkeys([fxnname for (fxnname, mode) in faults])
            for fxnname, mode in faults: 
                params = mdl.fxns[fxnname].faultmodes[mode]
                if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(faults),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=mdl.fxns[fxnname].failrate
                self.comprates[fxnname] = {compname:comp.failrate for compname, comp in mdl.fxns[fxnname].components.items()}
        if type(jointfaults['faults'])==int:
            self.jointmodes=[]
            for numjoint in range(2, jointfaults['faults']+1):
                jointmodes = list(itertools.combinations(self._fxnmodes, numjoint))
                if not jointfaults.get('jointfuncs', False): 
                    jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                self.jointmodes = self.jointmodes + jointmodes
        elif type(jointfaults['faults'])==list: self.jointmodes = jointfaults['faults']
    def init_rates(self,mdl, jointfaults={'faults':'None'}, modephases={}):
        """ Initializes rates, rates_timeless"""
        self.rates=dict.fromkeys(self._fxnmodes)
        self.rates_timeless=dict.fromkeys(self._fxnmodes)
        self.mode_phase_map=dict.fromkeys(self._fxnmodes)
        for (fxnname, mode) in self._fxnmodes:
            key_phases = mdl.fxns[fxnname].key_phases_by
            if key_phases=='global': fxnphases = self.globalphases
            elif key_phases=='none': fxnphases = {'operating':[mdl.times[0], mdl.times[-1]]} 
            else: fxnphases = self.phases[key_phases]
            fxnphases = dict(sorted(fxnphases.items(), key = lambda item: item[1][0]))
            self.rates[fxnname, mode]=dict(); self.rates_timeless[fxnname, mode]=dict(); self.mode_phase_map[fxnname, mode] = dict()
            overallrate = self.fxnrates[fxnname]
            if self.comprates[fxnname]:
                for compname, component in mdl.fxns[fxnname].components.items():
                    if mode in component.faultmodes:
                        overallrate=self.comprates[fxnname][compname]
            
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
                    elif len(oppvect)!=len(fxnphases): raise Exception("Invalid Opportunity vector: "+fxnname+". Invalid length.")
                    else: oppvect = {phase:oppvect[i] for (i, phase) in enumerate(fxnphases)}
            for phase, times in fxnphases.items():
                opp = oppvect[phase]/(sum(oppvect.values())+1e-100)
                dist = self._fxnmodes[fxnname, mode]['dist']
                if self._fxnmodes[fxnname, mode]['probtype']=='rate':      
                    dt = float(times[1]-times[0]) 
                    unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
                elif self._fxnmodes[fxnname, mode]['probtype']=='prob':    
                    dt = 1
                    unitfactor = 1
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
                for phase_combo in itertools.product(*jointphase_list):
                    intervals = [jointphase_dict[phase] for phase in phase_combo]
                    overlap = find_overlap_n(intervals)
                    if overlap: 
                        phaseid = tuple(set(phase_combo))
                        if len(phaseid) == 1: phaseid = phaseid[0]
                        rates = [self.rates[fmode][phase_combo[i]]* np.subtract(*overlap)/np.subtract(*self.mode_phase_map[fmode][phase_combo[i]]) for i,fmode in enumerate(jointmode)]
                        if not jointfaults.get('pcond', False): # if no input, assume independence
                            prob = np.prod(1-np.exp(-np.array(rates)))
                            self.rates[jointmode][phaseid] = -np.log(1.0-prob)
                        elif type(jointfaults['pcond'])==float:
                            self.rates[jointmode][phaseid] = jointfaults['pcond']*max(rates)
                        elif type(jointfaults['pcond'])==list:
                            self.rates[jointmode][phaseid] = jointfaults['pcond'][j_ind]*max(rates)  
                        self.rates_timeless[jointmode][phaseid] = self.rates[jointmode][phaseid]/(overlap[1]-overlap[0])
                        self.mode_phase_map[jointmode][phaseid] = overlap        
    def create_sampletimes(self,mdl, params={}, default={'samp':'evenspacing','numpts':1}):
        """ Initializes weights and sampletimes """
        self.sampletimes={}
        self.weights={fxnmode:dict.fromkeys(rate) for fxnmode,rate in self.rates.items()}
        self.sampparams={}
        for fxnmode, ratedict in self.rates.items():
            for phaseid, rate in ratedict.items():
                if rate > 0.0:
                    times = self.mode_phase_map[fxnmode][phaseid]
                    possible_phasetimes = list(np.arange(times[0], times[1], self.tstep))
                    param = params.get((fxnmode,phaseid), default)
                    self.sampparams[fxnmode, phaseid] = param
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


def find_overlap_n(intervals):
    upper_limits = [interval[1] for interval in intervals]
    lower_limits = [interval[0] for interval in intervals]
    if any(u < l for u in upper_limits for l in lower_limits): return []
    orderedintervals = np.sort(upper_limits+lower_limits)
    return [orderedintervals[len(intervals)-1],orderedintervals[len(intervals)]]

def find_overlap(interval1, interval2):
    """Finds the overlap between two intervals"""
    if interval1[1] < interval2[0] or interval1[0] > interval2[1]: return []
    else: 
        orderedintervals = np.sort(interval1 + interval2)
        return [orderedintervals[1], orderedintervals[2]]
    

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

"""Model checking"""
def check_pickleability(obj):
    """ Checks to see which attributes of an object will pickle (and thus parallelize)"""
    unpickleable = []
    for name, attribute in vars(obj).items():
        if not dill.pickles(attribute):
            unpickleable = unpickleable + [name]
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