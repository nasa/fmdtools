#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`State` class for representing variables that change over time.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.base import is_iter
from fmdtools.define.container.base import BaseContainer

import numpy as np
import warnings


class State(BaseContainer):
    """
    Class for working with model states.

    States which are variables in the model which change over time.

    State is meant to be extended in the model definition object to add the
    corresponding field related to a simulation, e.g.,

    >>> class ExampleState(State):
    ...     x : np.float64=1.0
    ...     y : np.float64=1.0

    Creates a class point with fields x and y which are tagged as floats with default
    values of 1.0.

    Instancing State gives normal read/write access, e.g., one can do:

    >>> p = ExampleState()
    >>> p
    ExampleState(x=1.0, y=1.0)
    >>> p.x
    np.float64(1.0)

    or:

    >>> p = ExampleState(x=10.0)
    >>> p.x
    np.float64(10.0)
    """

    rolename = 's'

    def __init__(self, *args, **kwargs):
        args = self.get_true_fields(*args, **kwargs)
        args, kwargs = self.set_arg_type(*args)
        super().__init__(*args)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return State

    def set_atts(self, **kwargs):
        """
        Set the given arguments to a given value.

        Mainly useful for reducing length/adding clarity to assignment statements in
        __init__ methods (self.put is recomended otherwise so that the iteration is on
        function/flow *states*) e.g.,

        >>> p = ExampleState()
        >>> p.set_atts(x=2.0, y=2.0)
        >>> p.x
        np.float64(2.0)
        >>> p.y
        np.float64(2.0)
        """
        for name, value in kwargs.items():
            self.set_field(name, value)

    def put(self, as_copy=True, **kwargs):
        """
        Set the given fields to a given value.

        Mainly useful for reducing length/adding clarity to assignment statements.

        Parameters
        ----------
        as_copy: bool
            set to True for dicts/sets to be copied rather than referenced
        **kwargs : values
            fields and values to set.

        Examples
        --------
        >>> p = ExampleState()
        >>> p.put(x=2.0, y=2.0)
        >>> p.x
        np.float64(2.0)
        >>> p.y
        np.float64(2.0)
        """
        for name, value in kwargs.items():
            self.set_field(name, value, as_copy=as_copy)

    def get(self, *attnames, **kwargs):
        """
        Return the given attribute names (strings) as a numpy array.

        Mainly useful for reducing length of lines/adding clarity to assignment
        statements. e.g.,:

        >>> p = ExampleState(x=1.0, y=2.0)
        >>> p_arr = p.get("x", "y")
        >>> p_arr
        array([1., 2.])
        >>> p.get("x")
        np.float64(1.0)
        >>> p.get("x", "y", as_array=False)
        [np.float64(1.0), np.float64(2.0)]
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
        """Return the values of the defined fields for the state."""
        return self.gett(*self.__fields__)

    def gett(self, *attnames):
        """
        Alternative to self.get that returns a tuple, not an array.

        Useful when a numpy array would translate the underlying data
        types poorly (e.g., np.array([1,'b'] would make 1 a string--using a tuple
        instead preserves the data type)).

        Examples
        --------
        >>> ExampleState().gett("x")
        np.float64(1.0)
        >>> ExampleState().gett("x", "y")
        (np.float64(1.0), np.float64(1.0))
        """
        states = self.get(*attnames, as_array=False)
        if not is_iter(states):
            return states
        elif len(states) == 1:
            return states[0]
        else:
            return tuple(states)

    def inc(self, **kwargs):
        """
        Increment the given arguments by a given value.

        Mainly useful for reducing length/adding clarity to increment statements, e.g.,:

        >>> p = ExampleState(x=1.0, y=1.0)
        >>> p.inc(x=1, y=2)
        >>> p.x
        np.float64(2.0)
        >>> p.y
        np.float64(3.0)

        Can additionally be provided with a second value denoting a limit on the
        increments e.g.,:

        >>> p = ExampleState(x=1.0, y=1.0)
        >>> p.inc(x=(3, 5.0))
        >>> p.x
        np.float64(4.0)
        >>> p.inc(x=(3, 5.0))
        >>> p.x
        np.float64(5.0)
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
                    setattr(self, name, current.__class__(value[1]))
            else:
                setattr(self, name, getattr(self, name) + value)

    def roundto(self, **kwargs):
        """
        Round the given arguments to a given resolution.

        Examples
        --------
        >>> p = ExampleState(x=1.75850)
        >>> p.roundto(x=0.1)
        >>> p.x
        np.float64(1.8)
        """
        for name, value in kwargs.items():
            current = getattr(self, name)
            # np round used to avoid final rounding errors
            setattr(self, name, np.round(round(current/value)*value, 7))

    def limit(self, **kwargs):
        """
        Enforce limits on the value of a given property.

        Mainly useful for reducing length/adding clarity to increment statements. e.g.,:

        >>> p = ExampleState(x=200.0, y=-200.0)
        >>> p.limit(x=(0.0,100.0), y=(0.0, 100.0))
        >>> p.x
        np.float64(100.0)
        >>>
        np.float64(0.0)
        """
        for name, value in kwargs.items():
            if name not in self.__fields__:
                raise Exception(name+" not a property of "+str(self.__class__))
            try:
                lim = np.min([value[1], np.max([value[0], getattr(self, name)])])
                setattr(self, name, lim)
            except ValueError as e:
                raise Exception("Invalid state values for "+name +
                                ": "+str(getattr(self, name))) from e

    def mul(self, *states):
        """
        Return the multiplication of given attributes of the State.

        Examples
        --------
        >>> p = ExampleState(x=2.0, y=3.0)
        >>> p.mul("x","y")
        np.float64(6.0)
        """
        a = self.get(states[0])
        for state in states[1:]:
            a = a * self.get(state)
        return a

    def div(self, *states):
        """
        Return the division of given attributes of the State.

        Examples
        --------
        >>> p = ExampleState(x=1.0, y=2.0)
        >>> p.div('x','y')
        np.float64(0.5)
        """
        a = self.get(states[0])
        for state in states[1:]:
            a = a / self.get(state)
        return a

    def add(self, *states):
        """
        Return the addition of given attributes of the State.

        Examples
        --------
        >>> p = ExampleState(x=1.0, y=2.0)
        >>> p.add('x','y')
        np.float64(3.0)
        """
        a = self.get(states[0])
        for state in states[1:]:
            a += self.get(state)
        return a

    def sub(self, *states):
        """
        Return the subtraction of given attributes of the State.

        Examples
        --------
        >>> p = ExampleState(x=1.0, y=2.0)
        >>> p.sub('x','y')
        np.float64(-1.0)
        """
        a = self.get(states[0])
        for state in states[1:]:
            a -= self.get(state)
        return a

    def same(self, *args, **kwargs):
        """
        Test whether a given iterable values has the same value as each in the state.

        Examples
        --------
        >>> p = ExampleState(x=1.0, y=2.0)
        >>> p.same([1.0, 2.0], "x", "y")
        True
        >>> p.same([0.0, 2.0], "x", "y")
        False
        >>> p.same([1.0], 'x')
        True
        >>> p.same(x=1.0, y=2.0)
        True
        >>> p.same(x=0.0, y=0.0)
        False
        """
        if args and kwargs:
            raise Exception("Cannot use args and kwargs at the same time.")
        if args:
            values = args[0]
            states = args[1:]
        elif kwargs:
            values = [*kwargs.values()]
            states = [*kwargs.keys()]
        if is_iter(values) and len(values) == 1:
            values = values[0]
        test = values == self.get(*states)
        if is_iter(test):
            return all(test)
        else:
            return bool(test)

    def warn(self, *messages, stacklevel=2):
        """
        Print warning message(s) when called.

        Parameters
        ----------
        *messages : str
            Strings to make up the message (will be joined by spaces)
        stacklevel : int
            Where the warning points to.
            The default is 2 (points to the place in the model)
        """
        warnings.warn(' '.join(messages), stacklevel=stacklevel)

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Extend init_hist_attr to use _set to get history size."""
        if self.__annotations__[att] == str:
            set_con = getattr(self, att+"_set", [])
            if set_con:
                strlen = max([len(i) for i in set_con])
                str_size = "<U"+str(max(strlen))

        BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)


class ExampleState(State):
    """Example State class used for docstring tests."""

    x: np.float64 = 1.0
    y: np.float64 = 1.0


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
