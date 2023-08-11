# -*- coding: utf-8 -*-
"""
Description: Classes for defining scenarios to simulate. 

Classes:
    - :class:`Injection`:           Defines faults and disturbances to inject at a specific time
    - :class:`Scenario`:            Defines a generic scenario to simulation
    - :class:`SingleFaultScenario`: Defines the scenario of a single fault injected in a function
    - :class:`JointFaultScenario`:  Defines the scenario of multiple faults injected in a function at the same time
    - :class:`NominalScenario`:     Defines the scenario of a model having given parameters at the outset
    - :class:`ParamScenario`:       Defines the scenario of a model having parameters from a given paramfunc
    - :class:`Sequence`:     Creates an overall sequence of Injections from a given sequence of faults and disturbances
Functions::

"""
from recordclass import dataobject, asdict
from collections import UserDict
from typing import ClassVar


class Injection(dataobject, readonly=True, mapping=True):
    faults: dict={}
    disturbances: dict={}
    def get(self, entry, fallback):
        if not hasattr(self, entry): 
            return fallback
        else:
            return self[entry] 
    
    def update(self, inj):
        if hasattr(inj,'faults'):
            self.faults.update(inj['faults'])
        if hasattr(inj,'disturbances'):
            self.disturbances.update(inj['disturbances'])

class Sequence(UserDict):
    def __init__(self, faultseq={}, disturbances={}):
        times = {*faultseq, *disturbances}
        self.data = {t:Injection(faults=faultseq.get(t, {}), 
                            disturbances=disturbances.get(t, {})) 
                for t in times}
        
    def update_sequence(self, new_sequence):
        for i in new_sequence:
            if i not in self:             
                self[i]=new_sequence[i]
            else:
                self[i].update(new_sequence[i])

class BaseScenario(dataobject, readonly=True, mapping=True):
    sequence: dict = dict()
    times: tuple = ()
    prob: ClassVar[float] = 1.0
    rate: ClassVar[float] = 1.0
    name: ClassVar[str] = "nominal"
    time: ClassVar[float] = 0.0
    def copy_with(self, **kwargs):
        existing_kwargs = asdict(self)
        return self.__class__(**{**existing_kwargs, **kwargs})
    def get(self, entry, fallback):
        if not hasattr(self, entry): 
            return fallback
        else: 
            return self[entry]     

class Scenario(BaseScenario):
    rate: float = 1.0
    name: str = "nominal"
    time: float = 0.0

class SingleFaultScenario(BaseScenario):
    function: str=''
    fault: str=''
    rate: float = 1.0
    name: str = 'faulty'
    time: float = 0.0

class JointFaultScenario(BaseScenario):
    joint_faults: int=1
    functions: tuple=()
    modes: tuple=()
    rate: float = 1.0
    name: str = 'faulty'
    time: float = 0.0

class NominalScenario(BaseScenario, readonly=True):
    p: dict={}
    r: dict={}
    sp: dict={}
    prob: float=1.0
    inputparams: dict = {}
    rangeid: str=''
    name: str = 'nominal'

    def get_param(self, param, default="NA"):
        if param.startswith("p."):
            pval = self.p.get(param[2:], default)
        elif param.startswith("r."):
            pval = self.r.get(param[2:], default)
        elif param.startswith("sp."):
            pval = self.sp.get(param[3:], default)
        elif param.startswith("inputparams."):
            pval = self.inputparams.get(param[12:], default)
        elif param == 'prob':
            pval = self.prob
        else:
            pval = default
        return pval

    def get_params(self, *params, default="NA"):
        return [self.get_param(param, default=default) for param in params]


class ParamScenario(NominalScenario):
    paramfunc: callable 
    fixedargs: tuple = ()
    fixedkwargs: dict = {}
    
    def get_param(self, param, default="NA"):
        if param.startswith("fixedargs."):
            pval = self.p.get(param[10:], default)
        elif param.startswith("fixedkwargs."):
            pval = self.p.get(param[12:], default)
        else:
            pval = super().get_param(param, default=default)
        return pval




a = Scenario()

b = SingleFaultScenario()

seq = Sequence(faultseq={1:"fault"})


