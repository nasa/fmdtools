#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common methods used commonly in model definition constructs.

Includes functions:

- :func:`get_var`:Gets the variable value of the object
- :func:`set_var`:Sets variable of the object to a given value
- :func:`nest_dict`: Nest a dictionary by a certin number of levels.
- :func:`set_arg_as_type`: Change argument to given type.
- :func:`is_iter`: Checks whether a data type should be interpreted as an iterable
- :func:`is_numeric`: Check if a data type is numeric.
- :func:`is_bool` : check if a data type is boolean.
- :func:`is_numeric`: Helper function for Result Class, checks if a given value is
  numeric
- :func:`unpack_x`: Unpack an array x as a tuple argument.
- :func:`array_x`: Pack x into an array.
- :func:`eq_units`: Find conversion factor between rates and times.
- :func:`t_key`:Used to generate keys for a given (float) time that is queryable as an
  attribute of an object/dict
- :func:`round_float`: Round a float to a given precision.
- :func:`nan_to_x`: Helper function for Result Class, returns nan as zero if present,
  otherwise returns the number
 - :func:`gen_timerange`: Generates timerange from start/endtime
 - :func:`get_code_atrs`: Get code attributes defining a given object or method.
 - :func:`remove_para`: Remove paragraph newlines in a string.
 - :func:`get_obj_name`: Get the name of an object.
 - :func:`get_memory`: Get the memory an object takes.
 - :func:`get_inheritanc`: Find the bases classes an object inherits from.

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

from collections.abc import Iterable
from recordclass import dataobject
from ordered_set import OrderedSet
import numpy as np
import inspect
import sys


def get_var(obj, var):
    """
    Get the variable value of the object.

    Parameters
    ----------
    var : str/list
        list specifying the attribute (or sub-attribute of the object
    Returns
    -------
    var_value: any
        value of the variable
    """
    if isinstance(var, str):
        var_s = var.split(".")
    else:
        var_s = var
        var = ".".join(var_s)
    if len(var_s) == 1:
        k = var_s[0]
        if isinstance(obj, dict) or (hasattr(obj, 'keys') and hasattr(obj, 'values')):
            val = obj.get(k, None)
        elif type(obj) in {tuple, list} and k.isnumeric():
            val = obj[int(k)]
        else:
            val = getattr(obj, k)
        if hasattr(val, 'value'):
            return val.value
        else:
            return val
    else:
        if isinstance(obj, dict):
            if var_s[0] in obj:
                return get_var(obj[var_s[0]], var_s[1:])
            elif var in obj:
                return obj[var]
            else:
                raise Exception(var + "not in " + str(obj))
        elif (hasattr(obj, 'keys') and hasattr(obj, 'values')):
            if var_s[0] in obj.keys:
                return get_var(obj.get(var_s[0]), var_s[1:])
            elif var in obj.keys:
                return obj.get(var)
            else:
                raise Exception(var + "not in " + str(obj))
        else:
            return get_var(getattr(obj, var_s[0]), var_s[1:])


def set_var(obj, var, val):
    """
    Set variable of the object to a given value.

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
    if isinstance(var, str):
        var = var.split(".")

    if len(var) == 1:
        if isinstance(obj, dict):
            obj[var[0]] = val
        else:
            setattr(obj, var[0], val)
    else:
        if isinstance(obj, dict):
            set_var(obj[var[0]], var[1:], val)
        else:
            set_var(getattr(obj, var[0]), var[1:], val)


def nest_dict(dic, levels=float('inf'), separator=".", skip=0):
    """
    Nest a dictionary a certain number of levels by separator.

    Parameters
    ----------
    dict : dict
        Dictionary to nest. e.g. {'a.b': 1.0}
    levels : int, optional
        Levels to nest over. The default is float('inf').
    separator : str
        Seperator to nest by. The default is "."
    skip : str
        Levels to skip. The default is 0.

    Returns
    -------
    newhist : dict
        Nested dictionary. e.g. {'a': {'b': 1.0}}
    """
    newhist = dic.__class__()
    key_options = OrderedSet([".".join(h.split(separator)[:skip+1])
                              for h in dic.keys()])
    for key in key_options:
        if key in dic:
            newhist[key] = dic[key]
        else:
            subdict = {histkey[len(key)+1:]: val
                       for histkey, val in dic.items()
                       if histkey.startswith(key+separator)}
            subhist = dic.__class__(**subdict)
            lev = levels-1
            if lev > 0:
                newhist[key] = nest_dict(subhist, levels=lev, separator=separator)
            else:
                newhist[key] = subhist
    return newhist


def set_arg_as_type(true_type, new_arg):
    """
    Set a given argument as the type true_type.

    Parameters
    ----------
    true_type : class/type
        Class/type to set to
    new_arg : value
        Value to set as.

    Returns
    -------
    new_arg : value
        Value with correct type (if possible).
    """
    arg_type = type(new_arg)
    if arg_type != true_type:
        if arg_type == dict or issubclass(arg_type, dataobject):
            if true_type == tuple:
                new_arg = true_type(new_arg.values())
            else:
                new_arg = true_type(**new_arg)
        else:
            new_arg = true_type(new_arg)
    return new_arg


def is_iter(data):
    """
    Check whether a data type should be interpreted as an iterable or not.

    Returned as a single value or tuple/array.
    """
    if isinstance(data, Iterable) and not isinstance(data, str):
        return True
    else:
        return False


def is_numeric(val):
    """
    Check if a given value is a number.

    Examples
    --------
    >>> is_numeric(1.0)
    True
    >>> is_numeric("hi")
    False
    >>> is_numeric(np.array([1.0])[0])
    True
    >>> is_numeric(np.array(["hi"])[0])
    False
    """
    try:
        return np.issubdtype(np.array(val).dtype, np.number)
    except TypeError:
        return type(val) in [float, bool, int]


def is_bool(val):
    """
    Check if the value is a boolean.

    Examples
    --------
    >>> is_bool(True)
    True
    >>> is_bool(1.0)
    False
    >>> is_bool(np.array([True])[0])
    True
    >>> is_bool(np.array([1.0])[0])
    False
    """
    try:
        return val.dtype in ['bool']
    except AttributeError:
        return type(val) in [bool]


def is_known_immutable(val):
    """Check if value is known immutable."""
    return type(val) in [int, float, str, tuple, bool] or isinstance(val, np.number)


def is_known_mutable(val):
    """Check if value is a known mutable."""
    return type(val) in [dict, set]


def unpack_x(*x):
    """Unpack arrays/lists sent from libraries into tuples."""
    if len(x) == 1 and isinstance(x[0], Iterable):
        x = tuple(x[0])

    x_list = []
    for x_i in x:
        if isinstance(x_i, Iterable):
            x_list.append(unpack_x(x_i))
        else:
            x_list.append(x_i)
    x = tuple(x_list)
    return x


def array_x(*x):
    """Translate variable-length x into an array input."""
    if len(x) == 1 and isinstance(x[0], Iterable):
        x = np.array(x[0])
    else:
        x = np.array(x)
    return x


def eq_units(rateunit, timeunit):
    """
    Find conversion factor from rateunit (str) to timeunit (str).

    Options for units are: 'sec', 'min', 'hr', 'day', 'wk', 'month', and 'year'.
    """
    factors = {'sec': 1,
               'min': 60,
               'hr': 3600,
               'day': 86400,
               'wk': 604800,
               'month': 2629746,
               'year': 31556952}
    return factors[timeunit]/factors[rateunit]


def t_key(time):
    """
    Generate keys for a given (float) time in a queryable format.

    e.g. endresults.t10p0, the result at time t=10.0
    """
    return 't'+'p'.join(str(time).split('.'))


def round_float(number, res=1.0, min_r=7):
    """Round floats to a given resolution (avoiding fp errors)."""
    return np.round(round(number/res)*res, min_r)


def nan_to_x(metric, x=0.0):
    """
    Return nan as zero if present, otherwise return the number.

    Examples
    --------
    >>> nan_to_x(1.0)
    1.0
    >>> nan_to_x(np.nan, 10.0)
    10.0
    """
    if np.isnan(metric):
        return x
    else:
        return metric


def gen_timerange(start_time, end_time, dt=1.0, min_r=7):
    """Generate the times in a given interval given the timestep dt."""
    return np.round(np.arange(start_time, end_time + dt, dt), min_r)


def get_code_attrs(obj):
    """
    Get a dict of code attributes for a given object or method.

    Must be run from file other than where the code was originally written.

    Parameters
    ----------
    obj : Object/method
        Class to get code from.

    Returns
    -------
    code_attrs : dict
        Dict of "source", "code", and "docs" code attributes.
    """
    docs = inspect.getdoc(obj)
    if inspect.ismethod(obj):
        source = inspect.getsource(obj)
        code = source.split("'''")[-1].split('"""')[-1]
        code = "\n".join(code.split("\n        "))
        code = remove_para(code)
    else:
        if inspect.isclass(obj):
            objcl = obj
        else:
            objcl = obj.__class__
        source = inspect.getsource(objcl)
        if hasattr(obj, 'get_code'):
            code = obj.get_code(source)
        else:
            code = ""
        if str(obj.__doc__) not in source or 'fmdtools.' in objcl.__module__:
            docs = ""
    return {'source': source, 'code': code, 'docs': docs}


def remove_para(source):
    """Remove paragraph newlines in a string (e.g., of code)."""
    if source.startswith("\n"):
        return remove_para(source[1:])
    else:
        return source


def get_methods(obj):
    """Get methods from the given object."""
    methods = {at[0]: at[1] for at in inspect.getmembers(obj)
               if at[0] not in dir(obj.base_type()) and inspect.ismethod(at[1])}
    return methods


def get_obj_name(obj, role='', basename=''):
    """
    Get the name of an object.

    Parameters
    ----------
    obj : object
        Object to be graphed (BaseObject, BaseContainer, or other).
    role : str
        Role the object plays in the larger system. Determines the name of Containers.

    Returns
    -------
    name : str
        Name of the object.
    """
    if inspect.isclass(obj):
        return obj.__module__ + '.' + obj.__name__
    if hasattr(obj, 'get_full_name'):
        return obj.get_full_name()
    elif inspect.ismethod(obj):
        superobj = obj.__self__
        return get_obj_name(superobj, basename=basename, role=role) + '.' + obj.__name__
    else:
        if not basename or not role:
            raise Exception("No role (" + role + ") or basename (" + basename +
                            ") for object: " + str(obj))
        return basename + "." + role


def get_memory(role):
    """Get memory of an object."""
    if hasattr(role, 'get_memory'):
        mem, _ = role.get_memory()
    else:
        mem = sys.getsizeof(role)
    return mem


def get_inheritance(obj):
    """
    Get the base class(es) that the object inherits from.

    Parameters
    ----------
    obj : object
        Object to get base of.

    Returns
    -------
    classes : tuple
        Tuple of classes that are the base of the object.
    """
    if inspect.isclass(obj):
        return tuple([b for b in obj.__bases__
                      if b is not object and b is not dataobject])
    else:
        return (obj.__class__, )
