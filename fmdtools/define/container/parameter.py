#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Parameter` class to represent attributes that do not change.

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

from fmdtools.define.container.base import BaseContainer

import inspect
from recordclass import astuple
import warnings
import numpy as np


class Parameter(BaseContainer, readonly=True):
    """
    The Parameter class defines model/function/flow values which are immutable.

    That is, the same from model instantiation through a simulation. Parameters
    inherit from recordclass, giving them a low memory footprint, and use type
    hints and ranges to ensure parameter values are valid. e.g.,:

    Examples
    --------
    >>> class ExampleParameter(Parameter, readonly=True):
    ...    x: float = 1.0
    ...    y: float = 3.0
    ...    z: float = 0.0
    ...    x_lim = (0, 10)
    ...    y_set = (1.0, 2.0, 3.0, 4.0)

    defines a parameter with float x and y fields with default values of 30 and
    x_lim minimum/maximum values for x and y_set possible values for y. Note that
    readonly=True should be set to ensure fields are not changed.

    This parameter can then be instantiated using:

    >>> p = ExampleParameter(x=1.0, y=2.0)
    >>> p.x
    1.0
    >>> p.y
    2.0

    >>> p.copy()
    ExampleParameter(x=1.0, y=2.0, z=0.0)
    """

    rolename = "p"

    def __init__(self, *args, strict_immutability=True, check_type=True,
                 check_pickle=True, set_type=True, check_lim=True, **kwargs):
        """
        Initialize the parameter with given kwargs.

        Parameters
        ----------
        strict_immutability : bool
            Performs basic checks to ensure fields are immutable

        **kwargs : kwargs
            Fields to set to non-default values.
        """
        if not self.__doc__:
            raise Exception("Please provide docstring")
            # self.__doc__=Parameter.__doc__
        args = self.get_true_fields(*args, **kwargs)
        if set_type:
            args, kwargs = self.set_arg_type(*args, **kwargs)
        if args and isinstance(args[0], self.__class__):
            args = astuple(args[0])
        if check_lim:
            for i, k in enumerate(self.__fields__):
                self.check_lim(k, args[i])
        try:
            super().__init__(*args, **kwargs)
        except TypeError as e:
            raise Exception("Invalid args/kwargs: "+str(args)+" , " +
                            str(kwargs)+" in "+str(self.__class__)) from e
        if strict_immutability:
            self.check_immutable()

        if check_type:
            self.check_type()
        if check_pickle:
            self.check_pickle()

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Parameter

    def keys(self):
        return self.__fields__

    def check_lim(self, k, v):
        """
        Check to ensure the value v for field k is within the defined limits.

        Limits defined for field k defined in self.k_lim or set constraints self.k_set.

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
            if not (var_lims[0] <= v <= var_lims[1]):
                raise Exception("Variable "+k+" ("+str(v) +
                                ") outside of limits: "+str(var_lims))
        var_set = getattr(self, k+"_set", False)
        if var_set:
            if not (v in var_set):
                raise Exception("Variable "+k+" ("+str(v) +
                                ") outside of set constraints: "+str(var_set))

    def check_immutable(self):
        """
        Check if a known/common mutable or a known/common immutable.

        If known immutable, raise exception. If not known mutable, give a warning.

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
            elif isinstance(attr, np.ndarray):
                attr.flags.writeable = False
            elif not isinstance(attr, (int, float, tuple, str, Parameter, np.number)):
                warnings.warn("Parameter "+f+" type "+str(attr_type)+" may be mutable")

    def check_type(self):
        """
        Check to ensure Parameter type-hints are being followed.

        Raises
        ------
        Exception
            Raises exception if a field is not the same as its defined type.
        """
        for typed_field in self.__annotations__:
            attr_type = type(getattr(self, typed_field))
            true_type = self.__annotations__.get(typed_field, False)
            if ((true_type and not attr_type == true_type) and
                    str(true_type).split("'")[1] not in str(attr_type)):
                # weaker, but enables use of np.str, np.float, etc
                raise Exception(typed_field+" in "+str(self.__class__) +
                                " assigned incorrect type: " + str(attr_type) +
                                " (should be "+str(true_type)+")")

    def copy_with_vals(self, **kwargs):
        """Creates a copy of itself with modified values given by kwargs"""
        return self.__class__(**{**self.asdict(), **kwargs})

    def check_pickle(self):
        """Checks to make sure pickled object will get *args and **kwargs"""
        signature = str(inspect.signature(self.__init__))
        if not ('*args' in signature) and ('**kwargs' in signature):
            raise Exception("*args and **kwargs not in __init__()--will not pickle.")

    @classmethod
    def get_set_const(cls, field):
        if "." in field:
            field_split = field.split(".")
            true_field = field_split[0]
            subfield = ".".join(field_split[1:])
            subparam = cls.__annotations__[true_field]
            if isinstance(subparam, Parameter):
                return cls.__annotations__[true_field].get_set_const(subfield)
            else:
                return ()
        var_lims = getattr(cls, field+"_lim", False)
        if var_lims:
            return var_lims
        var_set = getattr(cls, field+"_set", False)
        if var_set:
            return set(var_set)
        return ()

    def copy(self):
        field_dict = self.get_field_dict(self)
        return self.__class__(**field_dict)

    def reset(self):
        """Do nothing since the parameter is immutable."""

    def return_mutables(self):
        return ()


class ExampleParameter(Parameter, readonly=True):
    """Example parameter for testing and documentation."""

    x: float = 1.0
    y: float = 3.0
    z: float = 0.0
    x_lim = (0, 10)
    y_set = (1.0, 2.0, 3.0, 4.0)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
