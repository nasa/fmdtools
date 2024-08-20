#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`BaseContainer` class which other containers inherit from.

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

from fmdtools.define.base import set_arg_as_type, remove_para
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History

from recordclass import dataobject, astuple, asdict
import copy
import pickle
import numpy as np
import sys


class BaseContainer(dataobject, mapping=True, iterable=True, copy_default=True):
    """
    Base container class.

    A container is a dataobject (from the recordclass library) that fulfills a specific
    role in a block.  This class inherits from dataobject for low memory footprint and
    has a number of methods for making attribute assignment/copying/tracking easier.
    """

    default_track = 'all'
    rolename = 'x'

    def get_typename(self):
        """Containers are typed as containers unless specified otherwise."""
        return "Container"

    def base_type(self):
        """Return fmdtools type of the model class."""
        return BaseContainer

    def check_role(self, roletype, rolename):
        """
        Check that the container will be given the correct name for its class.

        The correct container-names correspond to the role for the class embody, e.g.:
            State : s
            Rand : r
            Mode : m
            Parameter : p
            SimParam : sp
        """
        if roletype != 'container':
            raise Exception("Invalid roletype for container: " + roletype)
        if rolename != self.rolename:
            raise Exception("Invalid rolename "+rolename+" for "
                            + self.__class__.__name__ + ": "
                            + "should be " + self.rolename + " instead.")

    def get_track(self, track):
        """
        Get tracking params for a given dataobject (State, Mode, Rand, etc).

        Parameters
        ----------
        obj : dataobject
            State/Mode/Rand. Requires .default_track class variable.
        track : track
            str/tuple. Attributes to track.
            'all' tracks all fields
            'default' tracks fields defined in default_track for the dataobject
            'none' tracks none of the fields

        Returns
        -------
        track : tuple
            fields to track
        """
        if not track or track == 'default':
            track = self.default_track
        if track == 'all':
            track = self.__fields__
        elif track == 'none':
            track = ()
        elif isinstance(track, str):
            track = (track,)
        return track

    def get_true_fields(self, *args, force_kwargs=False, **kwargs):
        """
        Resolve the args to pass given certain defaults, *args and **kwargs.

        NOTE: must be used for pickling, since pickle passes arguments as *args and not
        **kwargs.
        """
        true_args = list([copy.copy(i) for i in self.__default_vals__])
        for i, n in enumerate(self.__fields__):
            if force_kwargs:
                true_args[i] = kwargs[n]
            if i < len(args):
                true_args[i] = args[i]
            elif n in kwargs:
                true_args[i] = kwargs[n]
        return true_args

    def get_true_field(self, fieldname, *args, **kwargs):
        """Get the value that will be set to fieldname given *args and **kwargs."""
        if fieldname in kwargs:
            return kwargs[fieldname]
        field_ind = self.__fields__.index(fieldname)
        if args and len(args) > field_ind:
            return args[field_ind]
        else:
            return copy.copy(self.__default_vals__[field_ind])

    def set_arg_type(self, *args, **kwargs):
        """
        Set Parameter field input to the predetermined field type.

        e.g., if the input to parameter is int for a float field, this converts it to a
        float in initialization.

        Parameters
        ----------
        obj : dataobject (or class)
            dataobject to get argument type for
        *args : *args
            args to dataobject
        **kwargs : **kwargs
            kwargs to dataobject

        Returns
        -------
        *new_args : tuple
            new args to dataobject (with proper type)
        **new_kwargs : dict
            new kwargs to dataobject (with proper type)
        """
        new_args = []
        new_kwargs = {}
        for i, typed_field in enumerate(self.__annotations__):
            true_type = self.__annotations__.get(typed_field, False)
            try:
                if i < len(args):
                    new_arg = set_arg_as_type(true_type, args[i])
                    new_args.append(new_arg)
                elif typed_field in kwargs:
                    new_arg = set_arg_as_type(true_type, kwargs[typed_field])
                    new_kwargs[typed_field] = new_arg

            except TypeError as e:
                try:
                    raise Exception("For field " + typed_field + " " + str(true_type) +
                                    ": unable to convert from " + str(new_arg) + " " +
                                    str(type(new_arg))) from e
                except UnboundLocalError as e1:
                    raise e
        return tuple(new_args), new_kwargs

    def get_field_dict(self, obj, *fields, **fielddict):
        """
        Get dict of values from object corresponding to Container fields.

        Parameters
        ----------
        obj : dataobject, list, tuple, or ndarray
            Object to get field dictionary from.
        *fields : str
            Names of corresponding fields (in self.__fields__)
        **fielddict
            Mapping of fields in obj corresponding to fields in self.

        Returns
        -------
        field_dict : dict
            Dictionary of fields and their values.

        Examples
        --------
        >>> ex = ExContainer(1.0, 2.0)
        >>> ex.get_field_dict([5.0])
        {'x': 5.0}

        >>> ex2 = ExContainer(3.0, 4.0)
        >>> ex.get_field_dict(ex2)
        {'x': 3.0, 'y': 4.0}

        >>> ex.get_field_dict({'x': 3.0, 'z': 40.0}, x='x', y='z')
        {'x': 3.0, 'y': 40.0}

        >>> ex.get_field_dict(ex)
        {'x': 1.0, 'y': 2.0}
        """
        if fielddict and fields:
            raise Exception("Provide positional states or keyword states, not both")
        if len(fields) == 0 and not fielddict:
            # if no states provided, assign all states
            fields = self.__fields__

        if type(obj) in [list, tuple] or isinstance(obj, np.ndarray):
            if fielddict:
                raise Exception("Only provided *args for lists/tuples, not **kwargs")
            fielddict = {state: obj[i]
                         for i, state in enumerate(fields) if i < len(obj)}
        elif isinstance(obj, dataobject):
            if not fielddict:
                # if states provided, only assign those states
                fielddict = {s: getattr(obj, s) for s in fields}
            else:
                # if kwarg states provided, assign keys to values
                fielddict = {k: getattr(obj, v) for k, v in fielddict.items()}
        elif isinstance(obj, dict):
            if not fielddict:
                fielddict = {s: obj[s] for s in fields if s in obj}
            else:
                fielddict = {k: obj[v] for k, v in fielddict.items() if v in obj}
        else:
            raise Exception("Invalid type to assign from: "+obj.__class__)
        return fielddict

    def assign(self, obj, *fields, as_copy=True, **fielddict):
        """
        Set the same-named values of the current object to those of another.

        Further arguments specify which values. e.g.,:

        >>> p1 = ExContainer(x=0.0, y=0.0)
        >>> p2 = ExContainer(x=10.0, y=20.0)
        >>> p1.assign(p2, 'x', 'y')
        >>> p1.x
        10.0
        >>> p1.y
        20.0

        Can also be used to assign list values to a variable, e.g.,:

        >>> p1.assign([3.0,4.0], 'x', 'y')
        >>> p1.x
        3.0
        >>> p1.y
        4.0

        Can also provide kwargs in case value names don't match, e.g.,:

        >>> p1.assign(p2, x='y', y='x')
        >>> p1.x
        20.0
        >>> p1.y
        10.0

        as_copy: bool,
            set to True for dicts/sets to be copied rather than referenced
        """
        fielddict = self.get_field_dict(obj, *fields, **fielddict)
        for field, value in fielddict.items():
            self.set_field(field, value, as_copy=as_copy)

    def set_field(self, fieldname, value, as_copy=True):
        """
        Set the field of the container to the given value.

        Parameters
        ----------
        fieldname : str
            Name of the field.
        value : value
            Value to set the field to.
        as_copy : bool, optional
            Whether to copy value. The default is True.

        Examples
        --------
        >>> ex_nest = ExNestContainer()
        >>> ex_inside = ExContainer(3.0, 4.0)
        >>> ex_nest.set_field('e1', ex_inside)
        >>> ex_nest
        ExNestContainer(e1=ExContainer(x=3.0, y=4.0), z=20.0)
        """
        if fieldname not in self.__fields__:
            raise Exception(fieldname+" not a property of "+self.name)
        if as_copy:
            value = copy.deepcopy(value)
        field = getattr(self, fieldname)
        if isinstance(field, BaseContainer):
            field.assign(value, as_copy=as_copy)
        else:
            setattr(self, fieldname, value)

    def reset(self):
        # TODO: major issue here is that multiple states initialized to different
        # defaults are not accomodated--there is only one "default"
        for field, default in self.__defaults__.items():
            self[field] = copy.copy(default)

    def to_default(self, *fieldnames):
        """
        Reset given fields to their default values.

        Examples
        --------
        >>> ex = ExContainer(3.0, 4.0)
        >>> ex.to_default()
        >>> ex
        ExContainer(x=1.0, y=2.0)

        >>> ex = ExContainer(4.0, 5.0)
        >>> ex.to_default('x')
        >>> ex
        ExContainer(x=1.0, y=5.0)
        """
        if not fieldnames:
            fieldnames = tuple(self.__defaults__)
        self.assign(self.__default_vals__, *fieldnames, as_copy=True)

    def copy(self):
        """
        Create an independent copy of the container with the same attributes.

        Returns
        -------
        cop : BaseContainer
            Copy of the container with the same attributes as self.

        Examples
        --------
        >>> ex = ExContainer(4.0, 5.0)
        >>> ex2 = ex.copy()
        >>> ex2
        ExContainer(x=4.0, y=5.0)

        >>> ex_nest = ExNestContainer(ex2, 40.0)
        >>> ex_nest.copy()
        ExNestContainer(e1=ExContainer(x=4.0, y=5.0), z=40.0)
        """
        cop = self.__class__()
        cop.assign(self, as_copy=True)
        return cop

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Initialize a field in the history."""
        if att in self.__fields__:
            val = getattr(self, att)
            dtype = self.__annotations__[att]
            # add history attribute
            if hasattr(val, 'create_hist'):
                subtrack = get_sub_include(att, track)
                hist[att] = val.create_hist(timerange=timerange, track=subtrack)
            else:
                hist.init_att(att, val, timerange, track,
                              dtype=dtype,str_size=str_size)

    def create_hist(self, timerange=None, track=None, default_str_size='<U20'):
        """
        Create a History corresponding to the State.

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

        Examples
        --------
        >>> nest_hist = ExNestContainer().create_hist()
        >>> nest_hist
        e1: 
        --x:                            array(1)
        --y:                            array(1)
        z:                              array(1)
        >>> nest_hist.e1.x
        [1.0]
        """
        track = self.get_track(track)
        hist = History()
        for att in track:
            self.init_hist_att(hist, att, timerange, track, str_size=default_str_size)
        return hist

    def get_memory(self):
        """Get approximate memory impact of dataobject and its fields."""
        mem_profile = dict()
        mem = sys.getsizeof(self)
        mem_profile['container'] = mem
        for fieldname in self.__fields__:
            field = getattr(self, fieldname)
            if hasattr(field, 'get_memory'):
                mem_profile['container'], _ = field.get_memory()
            else:
                mem_profile['container'] = sys.getsizeof(field)
            mem += mem_profile['container']
        return mem, mem_profile

    def return_mutables(self):
        """Return mutable aspects of the container."""
        return astuple(self)

    def asdict(self):
        """Return fields as a dictionary."""
        return asdict(self)

    def get_code(self, source):
        """Get the code defining the Container."""
        if self.__class__ == self.base_type():
            code = ''
        elif '\n\n    def' in source:
            code = source.split('\n\n    def')[0].split("'''")[-1].split('"""')[-1]
        else:
            code = source.split("'''")[-1].split('"""')[-1]
        code = "\n".join(code.split("\n    "))
        return remove_para(code)

def check_container_pick(container, *args, **kwargs):
    """
    Check that a given container class or object will pickle.

    Examples
    --------
    >>> ex = ExContainer()
    >>> check_container_pick(ex)
    True
    >>> check_container_pick(ExContainer, x=2.0)
    True
    >>> check_container_pick(ExContainer, 5.0, 40.0)
    True
    """
    if isinstance(container, BaseContainer):
        inputobj = container
    else:
        inputobj = container(*args, **kwargs)
    pickdata = pickle.dumps(inputobj)
    outputobj = pickle.loads(pickdata)
    same_values = [getattr(inputobj, field) == getattr(outputobj, field)
                   for field in container.__fields__]
    return all(same_values)


class ExContainer(BaseContainer):
    x: float = 1.0
    y: float = 2.0

class ExNestContainer(BaseContainer):
    e1: ExContainer = ExContainer()
    z: float = 20.0

# TODO: it seems possible to use the below property to reset containers.
# Need to look at bug report for this.
es = ExContainer(3.0, 4.0)
es.__defaults__['x'] = 2

# interestingly enough, __defaults__ does not change __default_vals__
es.__default_vals__

es1 = ExContainer()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
