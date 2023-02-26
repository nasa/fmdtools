# -*- coding: utf-8 -*-
"""
Description: A module to define base data structures for simulation. Contains:
    
- :class:`States`:      Class operations for states (inherited by FxnClass, Flow, etc)
"""
import numpy as np
from operator import attrgetter
import warnings
from recordclass import dataobject, asdict
from collections.abc import Iterable
import dill
import copy
import inspect

class States(object):
    def set_atts(self, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for 
        reducing length/adding clarity to assignment statements in __init__ methods
        (self.put is reccomended otherwise so that the iteration is on function/flow *states*)
        e.g., self.set_attr(maxpower=1, maxvoltage=1) is the same as saying
              self.maxpower=1; self.maxvoltage=1
        """
        for name, value in kwargs.items():
            setattr(self, name, value)
    def put(self, as_copy=True, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for 
        reducing length/adding clarity to assignment statements.
        e.g., self.EE.put(v=1, a=1) is the same as saying
              self.EE.v=1; self.EE.a=1
        
        as_copy: bool, set to True for dicts/sets to be copied rather than referenced
        """
        for name, value in kwargs.items():
            if name not in self._states: raise Exception(name+" not a property of "+self.name)
            if as_copy: value=copy.copy(value)
            setattr(self, name, value)
    def assign(self,obj,*states, as_copy=True, **statedict):
        """ Sets the same-named values of the current flow/function object to those of a given flow. 
        Further arguments specify which values.
        e.g. self.EE1.assign(EE2, 'v', 'a') is the same as saying
            self.EE1.a = self.EE2.a; self.EE1.v = self.EE2.v
        Can also be used to assign list values to a variable
        e.g. self.Pos.assign([1,2,3],'x','y','z')
        Can also provide dict in case value names don't match
        e.g. self.Pos_out.assign(self.Pos_in, x='dx',y='dy')
        as_copy: bool, set to True for dicts/sets to be copied rather than referenced
        """
        if type(obj)==list or isinstance(obj, np.ndarray):
            for i, state in enumerate(states):  
                if as_copy: val=copy.copy(obj[i])
                else:       val=obj[i]
                setattr(self, state, val)
        else:
            if not statedict:
                if len(states)==0:    statedict = {s:s for s in obj._states}
                else:                 statedict = {s:s for s in states}
            elif len(states)>0: raise Exception("Can only provide positional states or keyword states, not both")
            for set_state, get_state in statedict.items():
                if set_state not in self._states: raise Exception(set_state+" not a property of "+self.name)
                val = getattr(obj,get_state)
                if as_copy: val=copy.copy(val)
                setattr(self, set_state, val)
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
    def roundto(self, **kwargs):
        """
        Rounds the given arguments to a given resolution.
        e.g., self.Pos.roundto(x=0.1) will round Pos.x to the nearest 0.1.
        """
        for name, value in kwargs.items():
            current = getattr(self,name)
            setattr(self, name, round(current/value)*value)
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
        """Returns the subtraction of given attributes of the model construct
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
    def set_var(self,var, val):
        """
        Sets variable of the object to a given value

        Parameters
        ----------
        var : list/tuple of strings
            list of nested attributes
        val : attr
            attribute to set the value to

        Returns
        -------
        flowdict : dict
            dict of flows indexed by flownames
        """
        if type(var)==str: var=var.split(".")
        #if not attrgetter(".".join(var))(self): raise Exception("Attibute does not exist: "+str(var))
        
        if len(var)==1: setattr(self, var[0], val)
        else: 
            if getattr(self, var[0]): 
                subattr = getattr(self, var[0])
                if hasattr(subattr, 'set_var'): subattr.set_var(var[1:], val)
                elif type(subattr)==dict:  
                    if var[1] not in subattr:
                        subattr[eval(var[1])]=val
                    else:                       
                        subattr[var[1]]=val
                else: raise Exception("Model sub-attribute "+str(subattr)+" does not inherit from Common")
            else: raise Exception("Invalid variables :"+str(var))
    def get_var(self, var):
        """
        Gets the variable value of the object

        Parameters
        ----------
        var : str/list
            list specifying the attribute (or sub-attribute of the object
        Returns
        -------
        var_value: any
            value of the variable
        """
        if type(var)==str: var=var.split(".")
        return attrgetter(".".join(var))(self)
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

def is_iter(data):
    """ Checks whether a data type should be interpreted as an iterable or not and returned
    as a single value or tuple/array"""
    if isinstance(data, Iterable) and type(data)!=str:  return True
    else:                                               return False

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


class Parameter(dataobject, readonly=True):
    """
    The Parameter class defines model/function/flow values which are immutable,
    that is, the same from model instantiation through a simulation. Parameters 
    inherit from recordclass, giving them a low memory footprint, and use type
    hints and ranges to ensure parameter values are valid.
    
    e.g.,
    class Param(Parameter, readonly=True):
        x:          float = 30.0
        y:          float = 30.0
        x_lim = (0.0,100.0)
        y_set = (0.0,30.0,100.0)
    defines a parameter with float x and y fields with default values of 30 and
    x_lim minimum/maximum values for x and y_set possible values for y. Note that
    readonly=True should be set to ensure fields are not changed.
    
    This parameter can then be instantiated using:
        p = Param(x=1.0, y=0.0)
    """
    def __init__(self, *args, strict_immutability=True,**kwargs):
        """
        Initializes the parameter with given kwargs.

        Parameters
        ----------
        strict_immutability : bool
            Performs basic checks to ensure fields are immutable
        
        **kwargs : kwargs
            Fields to set to non-default values.
        """
        if not self.__doc__: 
            raise Exception("Please provide docstring")
            #self.__doc__=Parameter.__doc__
        for k, v in kwargs.items():
            self.check_lim(k,v)
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            raise Exception("Invalid args/kwargs: "+str(args)+" , "+str(kwargs))
        if strict_immutability: self.check_immutable()
        self.check_type()
        self.check_pickle()
    def check_lim(self, k, v):
        """
        Checks to ensure the value v for field k is within the defined limits
        self.k_lim or set constraints self.k_set

        Parameters
        ----------
        k : str
            Field to check
        v : mutable
            Value for the field to check

        Raises
        ------
        Exception
            Notification that the field is outside limits/set constraints.
        """
        var_lims = getattr(self, k+"_lim", False)
        if var_lims:
            if not(var_lims[0]<=v<=var_lims[1]):
                raise Exception("Variable "+k+" ("+str(v)+") outside of limits: "+str(var_lims))
        var_set = getattr(self, k+"_set", False)
        if var_set:
            if not(v in var_set):
                raise Exception("Variable "+k+" ("+str(v)+") outside of set constraints: "+str(var_set))
    def check_immutable(self):
        """
        (woefully incomplete) check to ensure defined field values are immutable.
        Checks if a known/common mutable or a known/common immutable, otherwise 
        gives a warning.

        Raises
        ------
        Exception
            Throws exception if a known mutable (e.g., dict, set, list, etc)
        """
        for f in self.__fields__:
            attr = getattr(self, f)
            attr_type = type(attr)
            if isinstance(attr, (list, set, dict)):
                raise Exception("Parameter "+f+" type "+str(attr_type)+" is mutable")
            elif not isinstance(attr, (int, float, tuple, str, Parameter, np.number)):
                warnings.warn("Parameter "+f+" type "+str(attr_type)+" may be mutable")
    def check_type(self):
        """
        Checks to ensure Parameter type-hints are being followed.

        Raises
        ------
        Exception
            Raises exception if a field is not the same as its defined type.
        """
        for typed_field in self.__annotations__:
            attr_type = type(getattr(self, typed_field))
            true_type = self.__annotations__.get(typed_field, False)
            if ((true_type and not attr_type==true_type) and
                str(true_type).split("'")[1] not in str(attr_type)): # weaker, but enables use of np.str, np.float, etc
                raise Exception(typed_field+" assigned incorrect type: "+str(attr_type)+" (should be "+str(true_type)+")")
    def copy_with_vals(self, **kwargs):
        """Creates a copy of itself with modified values given by kwargs"""
        return self.__class__(**{**asdict(self), **kwargs})
    def check_pickle(self):
        """Checks to make sure pickled object will get *args and **kwargs"""
        signature = str(inspect.signature(self.__init__))
        if not ('*args' in signature) and ('**kwargs' in signature):
            raise Exception("*args and **kwargs not in __init__()--will not pickle.")
    def get_true_field(self, fieldname, *args, **kwargs):
        """Gets the value that will be set to fieldname given *args and **kwargs"""
        if fieldname in kwargs:                         return kwargs[fieldname]
        field_ind = self.__fields__.index(fieldname)
        if args and len(args)>field_ind:                return args[field_ind]
        else:                                           return self.__defaults__[field_ind]

# def phases(times, names=[]):
#     """ Creates named phases from a set of times defining the edges of the intervals """
#     if not names: names = range(len(times)-1)
#     return {names[i]:[times[i], times[i+1]] for (i, _) in enumerate(times) if i < len(times)-1}
# def trunc(x, n=2.0, truncif='greater'):
#     """truncates a value to a given number (useful if behavior unchanged by increases)
    
#     Parameters
#     ----------
#     x : float/int 
#         number to truncate
#     n : float/int (optional)
#         number to truncate to if >= number
#     truncif: 'greater'/'less'
#         whether to truncate if greater or less than the given number
#     """
#     if truncif=='greater' and x>n:      y=n
#     elif  truncif=='greater' and x<n:   y=n
#     else:                               y=x
#     return y
# def union(probs):
#     """ Calculates the union of a list of probabilities [p_1, p_2, ... p_n] p = p_1 U p_2 U ... U p_n """
#     while len(probs)>1:
#         if len(probs) % 2: 
#             p, probs = probs[0], probs[1:]
#             probs[0]=probs[0]+p -probs[0]*p
#         probs = [probs[i-1]+probs[i]-probs[i-1]*probs[i] for i in range(1, len(probs), 2)]
#     return probs[0]










