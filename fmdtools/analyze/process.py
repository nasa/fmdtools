"""
Description: Processes model results for visualization and analysis

Uses methods:
    - :func:`hists`:                    Processes a model histories for each scenario into results histories by comparing the states over time in each scenario with the states in the nominal scenario.
    - :func:`hist`:                     Compares model history with the nominal model history over time to make a history of degradation.
        - :func:`fxnhist`:              Compares the history of function states in mdlhist over time.
        - :func:`flowhist`:             Compares the history of flow states in mdlhist over time.
    - :func:`modephases`:               Identifies the phases of operation for the system based on a mdlhist with a history of its modes
    - :func:`graphflows`:               Extracts non-nominal flows by comparing the a results graph with a nominal results graph.
    - :func:`resultsgraph`:         Makes a dict history of results graphs given a dict history of the nominal and faulty graphs
    - :func:`resultsgraphs`:        Makes a dict history of results graphs given a dict history of the nominal and faulty graphs
    - :func:`state_probabilities`:  Calculates the probabilities of given end-state classifications given an endclasses dictionary
    - :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap. 
    - :func:`overall_diff`:         Calculates the difference between the nominal and fault scenarios for a set of nested endclasses
    - :func:`end_diff`:             Calculates the difference between the nominal and fault scenarios for a set of endclasses
    - :func:`percent`:              Calculates the percentage of a given indicator variable being True in endclasses
    - :func:`average`:              Calculates the average value of a given metric in endclasses
    - :func:`expected`:             Calculates the expected value of a given metric in endclasses using the rate variable in endclasses
    - :func:`rate`:                Calculates the rate of a given indicator variable being True in endclasses using the rate variable in endclasses
"""
#File Name: analyze/process.py
#Author: Daniel Hulse
#Created: November 2019 (Refactored April 2020)

import copy
import numpy as np
import pandas as pd
import os

from scipy.stats import bootstrap


def nan_to_x(metric, x=0.0):
    """returns nan as zero if present, otherwise returns the number"""
    if np.isnan(metric):    return x
    else:                   return metric
def expected(endclasses, metric):
    """Calculates the expected value of a given metric in endclasses using the rate variable in endclasses"""
    return sum([e[metric]*e['rate'] for k,e in endclasses.items() if not np.isnan(e[metric])])
def average(endclasses, metric, empty_as='nan'):
    """Calculates the average value of a given metric in endclasses"""
    ecs = [e[metric] for k,e in endclasses.items() if not np.isnan(e[metric])]
    if len(ecs)>0 or empty_as=='nan':   return np.mean(ecs)
    else:                               return empty_as
def percent(endclasses, metric):
    """Calculates the percentage of a given indicator variable being True in endclasses"""
    return sum([int(bool(e[metric])) for k,e in endclasses.items() if not np.isnan(e[metric])])/(len(endclasses)+1e-16)
def rate(endclasses, metric):
     """Calculates the rate of a given indicator variable being True in endclasses using the rate variable in endclasses"""
     return sum([int(bool(e[metric]))*e['rate'] for k,e in endclasses.items() if not np.isnan(e[metric])])
def end_diff(endclasses, metric, nan_as=np.nan, as_ind=False, no_diff=False):
    """
    Calculates the difference between the nominal and fault scenarios for a set of endclasses

    Parameters
    ----------
    endclasses : dict
        endclass dictionary for the set {scen:endclass}, where endclass is a dict of metrics
    metric : str
        metric to calculate the difference of in the endclasses
    nan_as : float, optional
        How do deal with nans in the difference. The default is np.nan.
    as_ind : bool, optional
        Whether to return the difference as an indicator (1,-1,0) or real value. The default is False.
    no_diff : bool, optional
        Option for not computing the difference (but still performing the processing here). The default is False.
    Returns
    -------
    difference : dict
        dictionary of differences over the set of scenarios
    """
    endclasses=endclasses.copy()
    nomendclass = endclasses.pop('nominal')
    if not no_diff: 
        if as_ind:  difference = {scen:bool(nan_to_x(ec[metric], nan_as))-bool(nan_to_x(nomendclass[metric], nan_as)) for scen, ec in endclasses.items()}
        else:       difference = {scen:nan_to_x(ec[metric], nan_as)-nan_to_x(nomendclass[metric], nan_as) for scen, ec in endclasses.items()}
    else:           
        difference = {scen:nan_to_x(ec[metric], nan_as) for scen, ec in endclasses.items()}
        if as_ind: difference = {scen:np.sign(metric) for scen,metric in difference.items()}
    return difference
def overall_diff(nested_endclasses, metric, nan_as=np.nan, as_ind=False, no_diff=False):
    """
    Calculates the difference between the nominal and fault scenarios over a set of endclasses

    Parameters
    ----------
    nested_endclasses : dict
        Nested dict of endclasses from propogate.nested
    metric : str
        metric to calculate the difference of in the endclasses
    nan_as : float, optional
        How do deal with nans in the difference. The default is np.nan.
    as_ind : bool, optional
        Whether to return the difference as an indicator (1,-1,0) or real value. The default is False.
    no_diff : bool, optional
        Option for not computing the difference (but still performing the processing here). The default is False.
    Returns
    -------
    differences : dict
        nested dictionary of differences over the set of fault scenarios nested in nominal scenarios 
    """
    return {scen:end_diff(endclass, metric, nan_as=nan_as, as_ind=as_ind, no_diff=no_diff) for scen, endclass in nested_endclasses.items()}
def bootstrap_confidence_interval(data, method=np.mean, return_anyway=False, **kwargs):
    """
    Convenience wrapper for scipy.bootstrap. 

    Parameters
    ----------
    data : list/array/etc
        Iterable with the data. May be float (for mean) or indicator (for proportion)
    method : 
        numpy method to give scipy.bootstrap.
    return_anyway: bool
        Gives a dummy interval of (stat, stat) if no . Used for plotting
    Returns
    ----------
    statistic, lower bound, upper bound
    """
    if 'interval' in kwargs: kwargs['confidence_level']=kwargs.pop('interval')*0.01
    if data.count(data[0])!=len(data):
        bs = bootstrap([data], np.mean, **kwargs)
        return method(data), bs.confidence_interval.low, bs.confidence_interval.high
    elif return_anyway: return method(data), method(data), method(data)
    else: raise Exception("All data are the same!")










