# -*- coding: utf-8 -*-
"""
Some common methods for analysis used by other modules.

Methods
-------
- :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap
- :func:`nan_to_x`: Helper function for Result Class, returns nan as zero if present,
  otherwise returns the number
- :func:`is_numeric`: Helper function for Result Class, checks if a given value is
  numeric
- :func:`join_key`: Helper function for Result Class
"""
import numpy as np


def get_sub_include(att, to_include):
    """Determines what attributes of att to include based on the provided
    dict/str/list/set to_include"""
    if type(to_include) in [list, set, tuple, str]:
        if att in to_include:
            new_to_include = 'default'
        elif type(to_include) == str and to_include == 'all':
            new_to_include = 'all'
        elif type(to_include) == str and to_include == 'default':
            new_to_include = 'default'
        else:
            new_to_include = False
    elif type(to_include) == dict and att in to_include:
        new_to_include = to_include[att]
    else:
        new_to_include = False
    return new_to_include


def to_include_keys(to_include):
    """Determine what dict keys to include from Result given nested to_include
    dictionary"""
    if type(to_include) == str:
        return [to_include]
    elif type(to_include) in [list, set, tuple]:
        return [to_i for to_i in to_include]
    elif type(to_include) == dict:
        keys = []
        for k, v in to_include.items():
            add = to_include_keys(v)
            keys.extend([k+'.'+v for v in add])
        return tuple(keys)


def is_numeric(val):
    """Checks if a given value is numeric"""
    try:
        return np.issubdtype(np.array(val).dtype, np.number)
    except:
        return type(val) in [float, bool, int]


def bootstrap_confidence_interval(data, method=np.mean, return_anyway=False, **kwargs):
    """
    Convenience wrapper for scipy.bootstrap.

    Parameters
    ----------
    data : list/array/etc
        Iterable with the data. May be float (for mean) or indicator (for proportion)
    method : method
        numpy method to give scipy.bootstrap.
    return_anyway: bool
        Gives a dummy interval of (stat, stat) if no . Used for plotting
    Returns
    ----------
    statistic, lower bound, upper bound
    """
    from scipy.stats import bootstrap
    if 'interval' in kwargs:
        kwargs['confidence_level'] = kwargs.pop('interval')*0.01
    if data.count(data[0]) != len(data):
        bs = bootstrap([data], np.mean, **kwargs)
        return method(data), bs.confidence_interval.low, bs.confidence_interval.high
    elif return_anyway:
        return method(data), method(data), method(data)
    else:
        raise Exception("All data are the same!")


def nan_to_x(metric, x=0.0):
    """returns nan as zero if present, otherwise returns the number"""
    if np.isnan(metric):
        return x
    else:
        return metric


def is_bool(val):
    try:
        return val.dtype in ['bool']
    except:
        return type(val) in [bool]


def join_key(k):
    if not isinstance(k, str):
        return '.'.join(k)
    else:
        return k