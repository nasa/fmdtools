# -*- coding: utf-8 -*-
"""
Description: A module for defining States, which are (generic) containers for system attributes that change over time.
    
- :class:`State`: Superclass for Model States.
"""
from recordclass import dataobject, asdict
import numpy as np
from .common import get_true_fields, is_iter, get_dataobj_track
import copy
import warnings
from fmdtools.analyze.result import History

class State(dataobject, mapping=True):
    """
    Class for working with model states, which are variables in the model which 
    change over time. This class inherits from dataobject for low memory footprint
    and has a number of methods for making attribute assignment/copying easier.
    
    State is meant to be extended in the model definition object to add the corresponding
    field related to a simulation, e.g.,
    
    class Point(State):
        x : float=1.0
        y : float=1.0
    
    Creates a class point with fields x and y which are tagged as floats with default values of 1.0.
    
    Instancing State gives normal read/write access, e.g., one can do:
    
    p = Point()
    p.x
    > 1.0
    
    or 
    p = Point(x=10.0)
    p.x
    > 10.0
    """
    default_track='all'
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
            if name not in self.__fields__: raise Exception(name+" not a property of "+str(self.__class__))
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
        if type(obj) in [list, tuple] or isinstance(obj, np.ndarray):
            for i, state in enumerate(states):  
                if as_copy: val=copy.copy(obj[i])
                else:       val=obj[i]
                setattr(self, state, val)
        else:
            if not statedict:
                if len(states)==0:    statedict = {s:s for s in obj.__fields__}
                else:                 statedict = {s:s for s in states}
            elif len(states)>0: raise Exception("Can only provide positional states or keyword states, not both")
            for set_state, get_state in statedict.items():
                if set_state not in self.__fields__: raise Exception(set_state+" not a property of "+self.name)
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
        return self.gett(*self.__fields__)
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
            if name not in self.__fields__: raise Exception(name+" not a property of "+str(self.__class__))
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
            if name not in self.__fields__: raise Exception(name+" not a property of "+str(self.__class__))
            try:
                setattr(self, name, min(value[1], max(value[0], getattr(self,name))))
            except ValueError as e:
                raise Exception("Invalid state values for "+name+": "+str(getattr(self,name))) from e
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
    def create_hist(self, timerange=None, track=None, default_str_size='<U20'):
        """
        Creates a History corresponding to State

        Parameter
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'. The default is None.
                DESCRIPTION. The default is None.

        Returns
        -------
        hist : History
            History of fields specified in track.
        """
        track = get_dataobj_track(self, track)
        hist = History()
        for att in track:
            val = getattr(self, att)
            dtype = self.__annotations__[att]
            str_size=default_str_size
            if dtype==str:
                set_con = getattr(self, att+"_set", [])
                if set_con:
                    strlen = max([len(i) for i in set_con])
                    str_size="<U"+str(max(strlen))
            hist.init_att(att, val, timerange, track, dtype=dtype, str_size = str_size)
        return hist