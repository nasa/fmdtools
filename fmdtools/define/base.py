# -*- coding: utf-8 -*-
"""
Description: A module for methods used commonly in model definition constructs.

Functions contained in this module:

- :func:`get_var`:Gets the variable value of the object
- :func:`set_var`:Sets variable of the object to a given value
- :func:`is_iter`: Checks whether a data type should be interpreted as an iterable or
not.
- :func:`t_key`:Used to generate keys for a given (float) time that is queryable as
  an attribute of an object/dict

"""
from collections.abc import Iterable
from recordclass import dataobject


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
    if type(var) == str:
        var_s = var.split(".")
    else:
        var_s = var
        var = ".".join(var_s)
    if len(var_s) == 1:
        k = var_s[0]
        if type(obj) == dict or (hasattr(obj, 'keys') and hasattr(obj, 'values')):
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
        if type(obj) == dict:
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
    if type(var) == str:
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
    key_options = set([h.split(separator)[0] for h in dic.keys()])
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
    Find conversion factor for from rateunit (str) to timeunit (str).

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

    e.g. endresults.t10p0, the result at time t=10.0"""
    return 't'+'p'.join(str(time).split('.'))
