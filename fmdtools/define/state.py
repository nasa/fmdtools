# -*- coding: utf-8 -*-
"""
Description: A module for defining States, which are (generic) containers for system
attributes that change over time.

- :class:`State`: Superclass for Model States.
"""
from recordclass import dataobject
import numpy as np
from .common import is_iter, get_dataobj_track
import copy
import warnings
from fmdtools.analyze.result import History


class State(dataobject, mapping=True):
    """
    Class for working with model states, which are variables in the model which
    change over time. This class inherits from dataobject for low memory footprint
    and has a number of methods for making attribute assignment/copying easier.

    State is meant to be extended in the model definition object to add the
    corresponding field related to a simulation, e.g.,

    >>> class ExamplePoint(State):
    ...     x : float=1.0
    ...     y : float=1.0

    Creates a class point with fields x and y which are tagged as floats with default
    values of 1.0.

    Instancing State gives normal read/write access, e.g., one can do:

    >>> p = ExamplePoint()
    >>> p.x
    1.0

    or:

    >>> p = ExamplePoint(x=10.0)
    >>> p.x
    10.0
    """
    default_track = 'all'

    def set_atts(self, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for
        reducing length/adding clarity to assignment statements in __init__ methods
        (self.put is reccomended otherwise so that the iteration is on function/flow
        *states*) e.g.,

        >>> p = ExamplePoint()
        >>> p.set_atts(x=2.0, y=2.0)
        >>> p.x
        2.0
        >>> p.y
        2.0
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

    def put(self, as_copy=True, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for
        reducing length/adding clarity to assignment statements. e.g.,

        >>> p = ExamplePoint()
        >>> p.put(x=2.0, y=2.0)
        >>> p.x
        2.0
        >>> p.y
        2.0

        as_copy: bool, set to True for dicts/sets to be copied rather than referenced
        """
        for name, value in kwargs.items():
            if name not in self.__fields__:
                raise Exception(name+" not a property of "+str(self.__class__))
            if as_copy:
                value = copy.copy(value)
            setattr(self, name, value)

    def assign(self, obj, *states, as_copy=True, **statedict):
        """ Sets the same-named values of the current flow/function object to those of a
        given flow.

        Further arguments specify which values.e.g.,

        >>> p1 = ExamplePoint(x=0.0, y=0.0)
        >>> p2 = ExamplePoint(x=10.0, y=20.0)
        >>> p1.assign(p2, 'x', 'y')
        >>> p1.x
        10.0
        >>> p1.y
        20.0

        Can also be used to assign list values to a variable, e.g.:

        >>> p1.assign([3.0,4.0], 'x', 'y')
        >>> p1.x
        3.0
        >>> p1.y
        4.0

        Can also provide kwargs in case value names don't match, e.g.:

        >>> p1.assign(p2, x='y', y='x')
        >>> p1.x
        20.0
        >>> p1.y
        10.0

        as_copy: bool,
            set to True for dicts/sets to be copied rather than referenced
        """
        if type(obj) in [list, tuple] or isinstance(obj, np.ndarray):
            for i, state in enumerate(states):
                if as_copy:
                    val = copy.copy(obj[i])
                else:
                    val = obj[i]
                setattr(self, state, val)
        else:
            if not statedict:
                if len(states) == 0:
                    statedict = {s: s for s in obj.__fields__}
                else:
                    statedict = {s: s for s in states}
            elif len(states) > 0:
                raise Exception(
                    "Can only provide positional states or keyword states, not both")
            for set_state, get_state in statedict.items():
                if set_state not in self.__fields__:
                    raise Exception(set_state+" not a property of "+self.name)
                val = getattr(obj, get_state)
                if as_copy:
                    val = copy.copy(val)
                setattr(self, set_state, val)

    def get(self, *attnames, **kwargs):
        """Returns the given attribute names (strings) as a numpy array. Mainly useful
        for reducing length of lines/adding clarity to assignment statements. e.g.,:

        >>> p = ExamplePoint(x=1.0, y=2.0)
        >>> p_arr = p.get("x", "y")
        >>> p_arr
        array([1., 2.])
        """
        if len(attnames) == 1:
            states = getattr(self, attnames[0])
        else:
            states = [getattr(self, name) for name in attnames]
        if not is_iter(states):
            return states
        elif len(states) == 1:
            return states[0]
        elif kwargs.get('as_array', True):
            return np.array(states)
        else:
            return states

    def values(self):
        return self.gett(*self.__fields__)

    def gett(self, *attnames):
        """Alternative to self.get that returns the given constructs as a tuple instead
        of as an array. Useful when a numpy array would translate the underlying data
        types poorly (e.g., np.array([1,'b'] would make 1 a string--using a tuple
        instead preserves the data type)"""
        states = self.get(*attnames, as_array=False)
        if not is_iter(states):
            return states
        elif len(states) == 1:
            return states[0]
        else:
            return tuple(states)

    def inc(self, **kwargs):
        """Increments the given arguments by a given value. Mainly useful for
        reducing length/adding clarity to increment statements, e.g.:

        >>> p = ExamplePoint(x=1.0, y=1.0)
        >>> p.inc(x=1, y=2)
        >>> p.x
        2.0
        >>> p.y
        3.0

        Can additionally be provided with a second value denoting a limit on the
        increments e.g.:
        >>> p = ExamplePoint(x=1.0, y=1.0)
        >>> p.inc(x=(3, 5.0))
        >>> p.x
        4.0
        >>> p.inc(x=(3, 5.0))
        >>> p.x
        5.0
        """
        for name, value in kwargs.items():
            if name not in self.__fields__:
                raise Exception(name+" not a property of "+str(self.__class__))
            if type(value) == tuple:
                current = getattr(self, name)
                sign = np.sign(value[0])
                newval = current + value[0]
                if sign*newval <= sign*value[1]:
                    setattr(self, name, newval)
                else:
                    setattr(self, name, value[1])
            else:
                setattr(self, name, getattr(self, name) + value)

    def roundto(self, **kwargs):
        """
        Rounds the given arguments to a given resolution. e.g.:

        >>> p = ExamplePoint(x=1.75850)
        >>> p.roundto(x=0.1)
        >>> p.x
        1.8
        """
        for name, value in kwargs.items():
            current = getattr(self, name)
            # np round used to avoid final rounding errors
            setattr(self, name, np.round(round(current/value)*value, 7))

    def limit(self, **kwargs):
        """Enforces limits on the value of a given property. Mainly useful for
        reducing length/adding clarity to increment statements. e.g.,:

        >>> p = ExamplePoint(x=200.0, y=-200.0)
        >>> p.limit(x=(0.0,100.0), y=(0.0, 100.0))
        >>> p.x
        100.0
        >>>
        0.0
        """
        for name, value in kwargs.items():
            if name not in self.__fields__:
                raise Exception(name+" not a property of "+str(self.__class__))
            try:
                setattr(self, name, min(value[1], max(value[0], getattr(self, name))))
            except ValueError as e:
                raise Exception("Invalid state values for "+name +
                                ": "+str(getattr(self, name))) from e

    def mul(self, *states):
        """Returns the multiplication of given attributes of the State. e.g.:

        >>> p = ExamplePoint(x=2.0, y=3.0)
        >>> p.mul("x","y")
        6.0
        """
        a = self.get(states[0])
        for state in states[1:]:
            a = a * self.get(state)
        return a

    def div(self, *states):
        """Returns the division of given attributes of the State, e.g.:

        >>> p = ExamplePoint(x=1.0, y=2.0)
        >>> p.div('x','y')
        0.5
        """
        a = self.get(states[0])
        for state in states[1:]:
            a = a / self.get(state)
        return a

    def add(self, *states):
        """Returns the addition of given attributes of the State, e.g.:

        >>> p = ExamplePoint(x=1.0, y=2.0)
        >>> p.add('x','y')
        3.0
        """
        a = self.get(states[0])
        for state in states[1:]:
            a += self.get(state)
        return a

    def sub(self, *states):
        """Returns the subtraction of given attributes of the State, e.g.:

        >>> p = ExamplePoint(x=1.0, y=2.0)
        >>> p.sub('x','y')
        -1.0
        """
        a = self.get(states[0])
        for state in states[1:]:
            a -= self.get(state)
        return a

    def same(self, values, *states):
        """Tests whether a given iterable values has the same value as each
        give state in the State, e.g.:

        >>> p = ExamplePoint(x=1.0, y=2.0)
        >>> p.same([1.0, 2.0], "x", "y")
        True
        >>> p.same([0.0, 2.0], "x", "y")
        False
        """
        test = values == self.get(*states)
        if is_iter(test):
            return all(test)
        else:
            return test

    def warn(self, *messages, stacklevel=2):
        """
        Prints warning message(s) when called.

        Parameters
        ----------
        *messages : str
            Strings to make up the message (will be joined by spaces)
        stacklevel : int
            Where the warning points to.
            The default is 2 (points to the place in the model)
        """
        warnings.warn(' '.join(messages), stacklevel=stacklevel)

    def create_hist(self, timerange=None, track=None, default_str_size='<U20'):
        """
        Creates a History corresponding to the State.

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'.
            The default is None.

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
            str_size = default_str_size
            if dtype == str:
                set_con = getattr(self, att+"_set", [])
                if set_con:
                    strlen = max([len(i) for i in set_con])
                    str_size = "<U"+str(max(strlen))
            hist.init_att(att, val, timerange, track, dtype=dtype, str_size=str_size)
        return hist


class ExamplePoint(State):
    """Example State class used for docstring tests"""
    x: float = 1.0
    y: float = 1.0
