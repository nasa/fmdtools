#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common methods used commonly in model definition constructs.

Includes functions:

- :func:`get_var`:Gets the variable value of the object
- :func:`set_var`:Sets variable of the object to a given value
- :func:`is_iter`: Checks whether a data type should be interpreted as an iterable
- :func:`t_key`:Used to generate keys for a given (float) time that is queryable as an
  attribute of an object/dict

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
    # if not attrgetter(".".join(var))(self):
    #    raise Exception("does not exist: "+str(var))

    if len(var) == 1:
        if type(obj) == dict:
            obj[var[0]] = val
        else:
            setattr(obj, var[0], val)
    else:
        if type(obj) == dict:
            set_var(obj[var[0]], var[1:], val)
        else:
            set_var(getattr(obj, var[0]), var[1:], val)


def nest_dict(dic, levels=float('inf'), separator="."):
    """
    Nest a dictionary a certain number of levels by separator.

    Parameters
    ----------
    dict : dict
        Dictionary to nest. e.g. {'a.b': 1.0}
    levels : int, optional
        DESCRIPTION. The default is float('inf').
    separator : str
        Saparator to nest by. The default is "."

    Returns
    -------
    newhist : dict
        Nested dictionary. e.g. {'a': {'b': 1.0}}
    """
    newhist = dic.__class__()
    key_options = OrderedSet([h.split(separator)[0] for h in dic.keys()])
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
    if isinstance(data, Iterable) and type(data) != str:
        return True
    else:
        return False


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
    return np.round(round(number/res)*res, 7)


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
