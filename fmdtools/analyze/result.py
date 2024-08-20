#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines how simulation results (histories) structured and processed.

Has classes:

- :class:`Result`: Class for defining simulation results

And functions:

- :func:`load`: Loads a given file to a Result/History
- :func:`load_folder`: Loads a given folder to a Result/History

Private Methods:

- :func:`file_check`: Check if files exists and whether to overwrite the file
- :func:`auto_filetype`: Helper function that automatically determines the filetype
  (npz, csv, or json) of a given filename
- :func:`create_indiv_filename`: Helper function that creates an individualized name for
  a file given the general filename and an individual id
- :func:`clean_resultdict_keys`: Helper function for recreating results dictionary keys
  (tuples) from a dictionary loaded from a file (where keys are strings)
  (used in csv/json results)
- :func:`get_dict_attr`: Gets attributes *attr from a given nested dict dict_in of class
  des_class
- :func:`fromdict`: Creates new history/result from given dictionary
- :func:`check_include_errors`: Helper function for Result Class, Cycles through
  `check_include_error`
- :func:`check_include_error`: Helper function to raise exceptions for errors
- :func:`get_sub_include`: Determines what attributes of att to include based on the
  provided dict/str/list/set to_include


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

from fmdtools.define.base import t_key, nest_dict
from fmdtools.analyze.common import to_include_keys, is_numeric, nan_to_x, is_bool
from fmdtools.analyze.common import bootstrap_confidence_interval, join_key
from fmdtools.analyze.common import get_sub_include, unpack_plot_values
from fmdtools.analyze.common import multiplot_legend_title, multiplot_helper
from fmdtools.analyze.common import set_empty_multiplots

import numpy as np
import pandas as pd
import sys
import os
from collections import UserDict


def file_check(filename, overwrite):
    """Check if files exists and whether to overwrite the file."""
    if os.path.exists(filename):
        if not overwrite:
            raise Exception("File already exists: "+filename)
        else:
            print("File already exists: "+filename+", writing anyway...")
            os.remove(filename)
    if "/" in filename:
        last_split_index = filename.rfind("/")
        foldername = filename[:last_split_index]
        if not os.path.exists(foldername):
            os.makedirs(foldername)


def auto_filetype(filename, filetype=""):
    """Automatically determines the filetype (pickle, csv, or json) of a filename."""
    if not filetype:
        if '.' not in filename:
            raise Exception("No file extension in: " + filename)
        if filename[-4:] == '.npz':
            filetype = "npz"
        elif filename[-4:] == '.csv':
            filetype = "csv"
        elif filename[-5:] == '.json':
            filetype = "json"
        else:
            raise Exception("Invalid File Type in: " + filename +
                            ", ensure extension is npz, csv, or json ")
    return filetype


def create_indiv_filename(filename, indiv_id, splitchar='_'):
    """Create filename name for a file given general filename and individual id."""
    filename_parts = filename.split(".")
    filename_parts.insert(1, '.')
    filename_parts.insert(1, splitchar+indiv_id)
    return "".join(filename_parts)


def clean_resultdict_keys(resultdict_dirty):
    """
    Clean the keys of a loaded dictionary.

    Helper function for recreating results dictionary keys (tuples) from a dictionary
    loaded from a file (where keys are strings) (used in csv/json results).

    Parameters
    ----------
    resultdict_dirty : dict
        Results dictionary where keys are strings

    Returns
    -------
    resultdict : dict
        Results dictionary where keys are tuples
    """
    resultdict = {}
    for key in resultdict_dirty:
        newkey = tuple(key.replace("'", "").replace("(", "").replace(")", "").split(", "))
        if any(['t=' in strs for strs in newkey]):
            joinfirst = [ind for ind, strs in enumerate(newkey) if 't=' in strs][0] + 1
        else:
            joinfirst = 0
        if joinfirst == 2:
            newkey = tuple([", ".join(newkey[:2])])+newkey[2:]
        elif joinfirst > 2:
            nomscen = newkey[:joinfirst-2]
            faultscen = tuple([", ".join(newkey[joinfirst-2:joinfirst])])
            vals = newkey[joinfirst:]
            newkey = nomscen+faultscen+vals
        resultdict[newkey[0]] = resultdict_dirty[key]
    return resultdict


def get_dict_attr(dict_in, des_class, *attr):
    """Get attributes *attr from a given nested dict dict_in of class des_class."""
    if len(attr) == 1:
        return dict_in[attr[0]]
    else:
        return get_dict_attr(des_class(dict_in[attr[0]]), *attr[1:])


def fromdict(resultclass, inputdict):
    """Create new history/result from given dictionary."""
    newhist = resultclass()
    for k, val in inputdict.items():
        if isinstance(val, dict):
            newhist[k] = resultclass.fromdict(val)
        else:
            newhist[k] = val
    return newhist


def check_include_errors(result, to_include):
    if type(to_include) is not str:
        for k in to_include:
            check_include_error(result, k)
    else:
        check_include_error(result, to_include)


def check_include_error(result, to_include):
    if to_include not in ('all', 'default') and to_include not in result:
        raise Exception("to_include key " + to_include +
                        " not in result keys: " + str(result.keys()))


class Result(UserDict):
    """
    Result is a special type of dictionary for simulation results.

    The goal of the Result class is to make it convenient to store, access, and load
    results form a model/simulation. As a dictionary, it supports dict-based item
    assignment (e.g. r['x']=10) but also enables convenient access via __getattr__,
    e.g.,:

    >>> r = Result()
    >>> r['x'] = 10
    >>> r
    x:                                    10

    It also can return a flattened version of its nested structure via Result.flatten(),
    e.g.,:

    >>> r = Result(y=Result(z=1))
    >>> r
    y: 
    --z:                                   1
    >>> r.keys()
    dict_keys(['y'])
    >>> rf = r.flatten()
    >>> rf
    y.z:                                   1
    >>> rf['y.z']
    1
    >>> rf.keys()
    dict_keys(['y.z'])

    It also enables saving and loading to files via r.save(), r.load(), and
    r.load_folder()
    """

    def __repr__(self, ind=0):
        str_rep = ""
        for k, val in self.items():
            if isinstance(val, np.ndarray) or isinstance(val, list):
                if type(k) is tuple:
                    k = str(k)
                val_rep = ind*"--"+k+": "
                if len(val_rep) > 40:
                    val_rep = val_rep[:20]
                form = '{:>'+str(40-len(val_rep))+'}'
                try:
                    lenstr = str(len(val))
                except TypeError:
                    lenstr = str(1)
                vv = form.format("array("+lenstr+")")
                str_rep = str_rep+val_rep+vv+'\n'
            elif isinstance(val, Result):
                res = val.__repr__(ind+1)
                if res:
                    val_rep = ind*"--"+k+": \n"+res
                    str_rep = str_rep+val_rep
            else:
                val_rep = ind*"--"+k+": "
                if len(val_rep) > 40:
                    val_rep = val_rep[:20]
                form = '{:>'+str(40-len(val_rep))+'}'
                vv = form.format(str(val))
                str_rep = str_rep+val_rep+vv+'\n'
        if str_rep.endswith('\n') and ind == 0:
            str_rep = str_rep[:-1]
        return str_rep

    def all(self):
        return tuple(self.data.values())

    def __eq__(self, other):
        """
        Check that the two values of the dictionary are equal.

        Enables the syntax result1 == result2, which returns True/False depending on if
        the keys/values are the same.

        Parameters
        ----------
        other : Result
            Result dictionary to compare against

        Returns
        -------
        equality : Bool
            Whether the results are equal
        """
        return all([all(v == other[k])
                    if isinstance(v, np.ndarray)
                    else v == other[k]
                    for k, v in self.data.items()])

    def __sub__(self, other):
        """
        Magic subtraction methods for Results.

        Used to enable uses such as:
            result1 - result2 = result3, where result3 is the difference between
            result1 and result2

            If the values are numeric (e.g., 1.5 and 1.0), the value returned will
            be the numeric difference (e.g., 0.5). Otherwise the value returned
            is a true/false value corresponding to whether or not they are the same
            (e.g. "yes", "no" -> False)

        Parameters
        ----------
        other : Result/History
            Result to subtract from the given result

        Returns
        -------
        ret : Result/History
            Result with values correspnding to the difference between the two.
        """
        ret = self.__class__()
        # creates a dict where the values are the mathematical difference if the
        # values are
        ret.data = {k: np.subtract(self[k], other[k], dtype=np.int32)
                    if is_bool(self[k])
                    else self[k]-other[k] if is_numeric(self[k])
                    else self[k] != other[k] for k in self.keys()}
        return ret

    def get_different(self, other):
        """
        Find the values of two results which are different.

        Parameters
        ----------
        other : Result
            Result to compare against

        Returns
        -------
        different : Result
            Result with entries corresponding to the difference between the two
            Results.
        """
        diff = self-other
        different = self.__class__()
        different.data = {k: v for k, v in diff.items() if v}
        return different

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def __reduce__(self):
        return type(self), (), None, None, iter(self.items())

    def __getattr__(self, argstr):
        try:
            args = argstr.split(".")
            return get_dict_attr(self.data, self.__class__, *args)
        except:
            try:
                return self.all_with(argstr)
            except:
                raise AttributeError("Not in dict: "+str(argstr))

    def __setattr__(self, key, val):
        if key == "data":
            UserDict.__setattr__(self, key, val)
        else:
            self.data[key] = val

    def get(self, *argstr,  **to_include):
        """
        Provide dict-like access to the history/result across a number of arguments.

        Parameters
        ----------
        *argstr : str
            keys to get directly (e.g. 'fxns.fxnname')
        **to_include : dict/str/
            to_include dict for arguments to get (e.g., {'fxns':{'fxnname'}})

        Returns
        -------
        Result/History
            Result/History with the attributes (or single att)
        """
        atts_to_get = argstr + to_include_keys(to_include)
        res = self.__class__()
        for at in atts_to_get:
            res[at] = self.__getattr__(at)
        if len(res) == 1 and at in res:
            return res[at]
        else:
            return res

    def all_with(self, attr):
        """Get all values with the attribute attr."""
        if attr in self:
            return self[attr]
        new = self.__class__()
        for k, v in self.items():
            if k.startswith(attr+'.'):
                new[k[len(attr)+1:]] = v
        if len(new) > 1:
            return new
        elif len(new) > 0:
            k = [*new.keys()][0]
            if k.endswith(attr):
                return new[k]
            else:
                return new
        else:
            raise Exception(attr+" not in Result keys: "+str(self.keys()))

    def fromdict(inputdict):
        return fromdict(Result, inputdict)

    def load(filename, filetype="", renest_dict=False, indiv=False):
        """Load as Result using :func:`load'."""
        inputdict = load(filename, filetype="", renest_dict=renest_dict,
                         indiv=indiv, Rclass=Result)
        return fromdict(Result, inputdict)

    def load_folder(folder, filetype, renest_dict=False):
        """Load as History using :func:`load_folder'."""
        files_toread = load_folder(folder, filetype)
        result = Result()
        for filename in files_toread:
            result.update(Result.load(folder+'/'+filename, filetype,
                          renest_dict=renest_dict, indiv=True))
        if renest_dict == False:
            result = result.flatten()
        return result

    def get_values(self, *values):
        """Get a dict with all values corresponding to the strings in *values."""
        h = self.__class__()
        flatself = self.flatten()
        k_vs = []
        for v in values:
            ks = [k for k in flatself.keys() if k.endswith(v)]
            if not ks:
                raise Exception("Value "+v+" not in Result keys.")
            k_vs.extend(ks)

        for k in k_vs:
            h[k] = flatself[k]
        return h

    def get_scens(self, *scens):
        """Get a dictlike with all scenarios corresponding to the strings in *scens."""
        h = self.__class__()
        k_s = [k for k in self.keys()
               for s in scens if k.startswith(s+".") or '.'+s+'.' in k]
        for k in k_s:
            h[k] = self[k]
        return h

    def get_comp_groups(self, *values, **groups):
        """
        Get comparison groups of *values (i.e., aspects of the model) in groups
        **groups (sets of scenarios with structure )

        Parameters
        ----------
        *values : str
            Values to get (e.g. `fxns.fxnname.s.val`)
        **groups : list
            Sets of scenarios to group (e.g. set_1=['scen1', 'scen2'...])

        Returns
        -------
        group_hist : History
            Single-level history with structure {group:{scenname.valuename}}
        """
        if not groups:
            groups = self.get_default_comp_groups()
        if 'time' not in values:
            values = values + ('time', )
        group_hist = self.__class__()
        for group, scens in groups.items():
            if scens == 'default':
                scens = {k.split('.')[0] for k in self.keys()}
            elif isinstance(scens, str):
                scens = [scens]
            k_vs = [k for k in self.keys() for scen in scens for v in values
                    if k.startswith(scen) and k.endswith(v) and '.t.' not in k]
            if len(k_vs) > 0 and (group not in group_hist):
                group_hist[group] = self.__class__()
            for k in k_vs:
                group_hist[group][k] = self[k]
        # Sort into comparison groups
        if not group_hist:
            raise Exception("Invalid comp_groups: " + str(groups) +
                            ", resulting grouped result is empty")
        return group_hist

    def get_default_comp_groups(self):
        """
        Gets a dict of nominal and faulty scenario keys from the Result

        Returns
        -------
        comp_groups : dict
            Dict with structure {'nominal': [list of nominal scenarios], 'faulty': [list of faulty scenarios]}.
            If no nominal or faulty, returns an empty dict {}.
        """
        nest = self.nest(1)
        nest2 = self.nest(2)
        if 'nominal' in nest.keys():
            comp_groups = {'nominal': 'nominal',
                           'faulty': [f for f in nest.keys() if f != 'nominal']}
        elif any(['nominal' in k for k in nest2.keys()]):
            comp_groups = {'nominal': [f for f in nest2.keys() if 'nominal' not in f],
                           'faulty': [f for f in nest2.keys() if 'nominal' not in f]}
        else:
            comp_groups = {'default': 'default'}
        return comp_groups

    def flatten(self, newhist=False, prevname="", to_include='all'):
        """
        Recursively creates a flattened result of the given nested model history

        Parameters
        ----------
        newhist : bool, default = False
        prevname : tuple, optional
            Current key of the flattened history (used when called recursively).
            The default is ().
        to_include : str/list/dict, optional
            What attributes to include in the dict. The default is 'all'. Can be of form
            - list e.g. ['att1', 'att2', 'att3'] to include the given attributes
            - dict e.g. fxnflowvals {'flow1':['att1', 'att2'],
                                     'fxn1':'all',
                                     'fxn2':['comp1':all, 'comp2':['att1']]}
            - str e.g. 'att1' for attribute 1 or 'all' for all attributes
        Returns
        -------
        newhist : dict
            Flattened model history of form: {(fxnflow, ..., attname):array(att)}
        """
        if newhist is False:
            newhist = self.__class__()

        check_include_errors(self, to_include)
        for att, val in self.items():
            if is_numeric(att):
                att = t_key(att)
            if prevname:
                newname = prevname+"."+att
            else:
                newname = att
            if isinstance(val, Result):
                new_to_include = get_sub_include(att, to_include)
                if new_to_include:
                    val.flatten(newhist, newname, new_to_include)
            elif to_include in ('all', 'default') or att in to_include:
                if len(newname) == 1:
                    newhist[newname[0]] = val
                else:
                    newhist[newname] = val
        return newhist

    def is_flat(self):
        """Checks if the history is flat."""
        for v in self.values():
            if isinstance(v, Result):
                return False
        return True

    def nest(self, levels=np.inf):
        """
        Re-nests a flattened result
        """
        return nest_dict(self, levels=levels)


    def get_memory(self):
        """
        Determines the memory usage of a given history and profiles.

        Returns
        -------
        mem_total : int
            Total memory usage of the history (in bytes)
        mem_profile : dict
            Memory usage of each construct of the model history (in bytes)
        """
        fhist = self.flatten()
        mem_total = 0
        mem_profile = dict.fromkeys(fhist.keys())
        for k, h in fhist.items():
            if (np.issubdtype(h.dtype, np.number) or
                   np.issubdtype(h.dtype, np.flexible) or
                   np.issubdtype(h.dtype, np.bool_)):
                mem = h.nbytes
            else:
                mem = 0
                for entry in h:
                    mem += sys.getsizeof(entry)
            mem_total += mem
            mem_profile[k] = mem
        return mem_total, mem_profile

    def save(self, filename, filetype="", overwrite=False, result_id=''):
        """
        Save a given result variable (endclasses or mdlhists) to a file filename.

        Files can be saved as npz, csv, or json.

        Parameters
        ----------
        filename : str
            File name for the file. Can be nested in a folder if desired.
        filetype : str, optional
            Optional specifier of file type (if not included in filename).
            The default is "".
        overwrite : bool, optional
            Whether to overwrite existing files with this name.
            The default is False.
        result_id : str, optional
            For individual results saving. Places an identifier for the result in the
            file. The default is ''.
        """
        import json
        import csv
        file_check(filename, overwrite)

        variable = self
        filetype = auto_filetype(filename, filetype)
        if filetype == 'npz':
            with open(filename, 'wb') as file_handle:
                if result_id:
                    res_to_save = Result({result_id: self})
                else:
                    res_to_save = self
                res_to_save = res_to_save.flatten()
                np.savez(filename, **res_to_save)
        elif filename[-4:] == '.csv':
            # add support for nested dict mdlhist using flatten_hist?
            variable = variable.flatten()
            with open(filename, 'w', newline='') as file_handle:
                writer = csv.writer(file_handle)
                if result_id:
                    writer.writerow([result_id])
                writer.writerow(variable.keys())
                if isinstance([*variable.values()][0], np.ndarray):
                    writer.writerows(zip(*variable.values()))
                else:
                    writer.writerow([*variable.values()])
        elif filename[-5:] == '.json':
            with open(filename, 'w', encoding='utf8') as file_handle:
                variable = variable.flatten()
                new_variable = {}
                for key in variable:
                    if isinstance(variable[key], np.ndarray):
                        new_variable[str(key)] = [var.item() for var in variable[key]]
                    else:
                        new_variable[str(key)] = variable[key]
                if result_id:
                    new_variable = {result_id: new_variable}
                strs = json.dumps(new_variable, indent=4, sort_keys=True,
                                  separators=(',', ': '), ensure_ascii=False)
                file_handle.write(str(strs))
        else:
            raise Exception("Invalid File Type")
        file_handle.close()

    def as_table(self):
        """Creates a table corresponding to the current dict structure"""
        flatdict = self.flatten()
        newdict = {join_key(k): v for k, v in flatdict.items()}
        return pd.DataFrame.from_dict(newdict)

    def create_simple_fmea(self, *metrics):
        """Makes a simple FMEA-stype table of the metrics in the endclasses
        of a list of fault scenarios run. If metrics not provided, returns all"""
        nested = {k: {**v.endclass} for k, v in self.nest(levels=1).items()}
        tab = pd.DataFrame.from_dict(nested).transpose()
        if not metrics:
            return tab
        else:
            return tab.loc[:, metrics]

    def get_expected(self, app=[], with_nominal=False, difference_from_nominal=False):
        """
        Take the expectation of numeric metrics in the result over given scenarios.

        Parameters
        ----------
        app : SampleApproach, optional
            Approach to use for weights (via rates). The default is [].
        with_nominal : bool, optional
            Whether to include the nominal scenario in the expectation.
            The default is False.
        difference_from_nominal : bool, optional
            Whether to calculated the difference of the expectation from nominal.
            The default is False.

        Returns
        -------
        expres : Result/History
            Result/History with values corresponding to the expectation of
            its quantities over the contained scenarios.
        """
        mh = self.nest(levels=1)

        nomhist = {k: v for k, v in mh.nominal.items() if is_numeric(v)}
        newhists = {k: hist for k, hist in mh.items()
                    if not ('nominal' in k and not (with_nominal))}
        if app:
            weights = [w.rate for w in app.scenarios()]
            if with_nominal:
                weights.append(1)
        else:
            weights = [1 for k in newhists]

        expres = self.__class__()
        for k in nomhist.keys():
            if difference_from_nominal:
                expres[k] = np.average([nomhist[k]-hist[k]
                                        for hist in newhists.values()],
                                       axis=0, weights=weights)
            else:
                expres[k] = np.average([hist[k] for hist in newhists.values()],
                                       axis=0, weights=weights)
        return expres

    def get_metric(self, value, metric=np.mean, args=(), axis=None):
        """
        Calculate a statistic of the value using a provided metric function.

        Parameters
        ----------
        value : str
            Value of the history to calculate the statistic over
        metric : func/'str', optional
            Function to process the history (e.g. np.mean, np.min...).
            The default is np.mean.
            May alternatively provide name of Result method (e.g., 'expected', 'rate')
        args : args
            Arguments for the metric function. Default is ().
        axis : None or 0 or 1
            Whether to take the metric over variables (0) or over time (1) or
            both (None). The default is None.
        """
        if isinstance(metric, str):
            method = getattr(self, metric)
            return method("."+value, *args)
        else:
            vals = self.get_values(value)
            return metric([*vals.values()], *args, axis=axis)

    def get_metric_ci(self, value, metric=np.mean, **kwargs):
        """
        Get the confidence interval for the given value over the set of scenarios.

        Parameters
        ----------
        value : str
            Value of the history to calculate the statistic over
        metric : func, optional
            Function to process the history (e.g., np.mean, np.min...).
            The default is np.mean
        **kwargs : kwargs
            kwargs to bootstrap_confidence_interval

        Returns
        -------
        statistic: number
            nominal statistic for the given metric
        lower bound : number
            lower bound of the statistic in the ci
        upper bound : number
            upper bound of the statistic in the ci
        """
        vals = self.get_values(value)
        ci = bootstrap_confidence_interval([*vals.values()], method=metric, **kwargs)
        return ci

    def get_metrics(self, *values, metric=np.mean, args=(), axis=None):
        """
        Calculate a statistic of the values using a provided metric function.

        Parameters
        ----------
        *values : strs
            Values of the history to calculate the statistic over
            (if none provided, creates metric of all)
        metric : func, optional
            Function to process the history (e.g. np.mean, np.min...).
            The default is np.mean.
        args : args, optional
            Arguments for the metric function. Default is ().
        axis : None or 0 or 1
            Whether to take the metric over variables (0) or over time (1)
            or both (None). The default is None.
        """
        if not values:
            values = self.keys()
        metrics = Result()
        for value in values:
            metrics[value] = self.get_metric(value, metric=metric, args=args, axis=axis)
        return metrics

    def total(self, metric):
        """
        Tabulates the total (non-weighted sum) of a metric over a number of runs.

        Parameters
        ----------
        metric: str
            metric to total

        Returns
        -------
        totalcost : Float
            The total metric of the scenarios.
        """
        return sum([e for e in self.get_values(metric).values()])

    def state_probabilities(self, prob_key='prob', class_key='classification'):
        """
        Tabulates the probabilities of different classifications in the result.

        Parameters
        ----------
        prob_key : str, optional
            string to use for rate/probability information. default is 'prob'
        class_key : str, optional
            string to use for the different classifications. default is 'classification'

        Returns
        -------
        probabilities : dict
            Dictionary of probabilities of different simulation classifications

        """
        classifications = self.get_values("." + class_key)
        class_len = len("." + class_key)
        probs = self.get_values("." + prob_key)
        probabilities = dict()
        for key, classif in classifications.items():
            prob = probs[key[:-class_len] + "." + prob_key]
            if classif in probabilities:
                probabilities[classif] += prob
            else:
                probabilities[classif] = prob
        return probabilities

    def expected(self, metric, prob_key='rate'):
        """Calculates the expected value of a given metric in endclasses using the rate
        variable in endclasses"""
        ecs = np.array([e for e in self.get_values(metric).values()
                        if not np.isnan(e)])
        weights = np.array([e for e in self.get_values(prob_key).values()
                            if not np.isnan(e)])
        return sum(ecs*weights)

    def average(self, metric, empty_as='nan'):
        """Calculates the average value of a given metric in endclasses"""
        ecs = [e for e in self.get_values(metric).values() if not np.isnan(e)]
        if len(ecs) > 0 or empty_as == 'nan':
            return np.mean(ecs)
        else:
            return empty_as

    def percent(self, metric):
        """Calculate the percentage of a given indicator variable being True."""
        return sum([int(bool(e)) for e in self.get_values(metric).values()
                    if not np.isnan(e)])/(len(self.get_values(metric))+1e-16)

    def rate(self, metric, prob_key='rate'):
        """Calculate the rate of a metric being True using the rate variable."""
        ecs = np.array([bool(e) for e in self.get_values(metric).values()
                        if not np.isnan(e)])
        weights = np.array([e for e in self.get_values(prob_key).values()
                            if not np.isnan(e)])
        return sum(ecs*weights)

    def end_diff(self, metric, nan_as=np.nan, as_ind=False, no_diff=False):
        """
        Calculate the difference between the nominal and fault scenarios.

        Parameters
        ----------
        metric : str
            metric to calculate the difference of in the endclasses
        nan_as : float, optional
            How do deal with nans in the difference. The default is np.nan.
        as_ind : bool, optional
            Whether to return the difference as an indicator (1,-1,0) or real value.
            The default is False.
        no_diff : bool, optional
            Option for not computing the difference
            (but still performing the processing here). The default is False.

        Returns
        -------
        difference : dict
            dictionary of differences over the set of scenarios
        """
        endclasses = self.copy()
        nomendclass = endclasses.pop('nominal')
        if not no_diff:
            if as_ind:
                difference = {scen: bool(nan_to_x(ec.endclass[metric], nan_as)) -
                              bool(nan_to_x(nomendclass[metric], nan_as))
                              for scen, ec in endclasses.items()}
            else:
                difference = {scen: nan_to_x(ec.endclass[metric], nan_as) -
                              nan_to_x(nomendclass[metric], nan_as)
                              for scen, ec in endclasses.items()}
        else:
            difference = {scen: nan_to_x(ec.endclass[metric], nan_as)
                          for scen, ec in endclasses.items()}
            if as_ind:
                difference = {scen: np.sign(metric)
                              for scen, metric in difference.items()}
        return difference

    def overall_diff(self, metric, nan_as=np.nan, as_ind=False, no_diff=False):
        """
        Calculate difference between the nominal and fault scenarios.

        Parameters
        ----------
        nested_endclasses : dict
            Nested dict of endclasses from propogate.nested
        metric : str
            metric to calculate the difference of in the endclasses
        nan_as : float, optional
            How do deal with nans in the difference. The default is np.nan.
        as_ind : bool, optional
            Whether to return the difference as an indicator (1,-1,0) or real value.
            The default is False.
        no_diff : bool, optional
            Option for not computing the difference
            (but still performing the processing here). The default is False.

        Returns
        -------
        differences : dict
            nested dictionary of differences over the set of fault scenarios nested in
            nominal scenarios
        """
        return {scen:
                endclass.end_diff(metric, nan_as=nan_as, as_ind=as_ind, no_diff=no_diff)
                for scen, endclass in self.items()}

    def plot_metric_dist(self, *values, cols=2, comp_groups={}, bins=10, metric_bins={},
                         legend_loc=-1, xlabels={}, ylabel='count', title='', titles={},
                         figsize='default',  v_padding=0.4, h_padding=0.05,
                         title_padding=0.1, legend_title=None, indiv_kwargs={},
                         fig=None, axs=None, **kwargs):
        """
        Plot histogram of given metric(s) over comparison groups of scenarios.

        Parameters
        ----------
        *values : str
            names of values to pull from the result (e.g., 'fxns.move_water.s.flowrate')
            Can also be specified as a dict (e.g. {'fxns':'move_water'}) to get all keys
            from a given fxn/flow/mode/etc.
        cols : int, optional
            columns to use in the figure. The default is 2.
        comp_groups : dict, optional
            Dictionary for comparison groups (if more than one).
            Has structure::
                {'group1': ('scen1', 'scen2'), 'group2': ('scen3', 'scen4')}.

            Default is {}, which compares nominal and faulty.
            If {'default': 'default'} is passed, all scenarios will be put in one group.
            If a legend is shown, group names are used as labels.
        bins : int
            Number of bins to use (for all plots). Default is None
        metric_bins : dict,
            Dictionary of number of bins to use for each metric.
            Has structure::
                {'metric':num}.

            Default is {}
        legend_loc : int, optional
            Specifies the plot to place the legend on, if runs are being compared.
            Default is -1 (the last plot)
            To remove the legend, give a value of False
        xlabels : dict, optional
            Label for the x-axes.
            Has structure::
                {'metric':'label'}

        ylabel : str, optional
            Label for the y-axes. Default is 'time'
        title : str, optional
            overall title for the plot. Default is ''
        indiv_kwargs : dict, optional
            dict of kwargs to differentiate the comparison groups.
            Has structure::
                {comp1: kwargs1, comp2: kwargs2}

            where kwargs is an individual dict of keyword arguments for the
            comparison group comp (or scenario, if not aggregated) which overrides
            the global kwargs (or default behavior).
        figsize : tuple (float,float)
            x-y size for the figure. The default is 'default', which dymanically gives
            3 for each column and 2 for each row
        v_padding : float
            vertical padding between subplots as a fraction of axis height.
        h_padding : float
            horizontal padding between subplots as a fraction of axis width.
        title_padding : float
            padding for title as a fraction of figure height.
        legend_title : str, optional
            title for the legend. Default is None.
        fig : matplotib figure
            Pre-existing figure (if any).
        axs : matplotlib axes
            Pre-existing axes (if any).
        **kwargs : kwargs
            keyword arguments to mpl.hist e.g. bins, etc.
        """
        # Sort into comparison groups
        plot_values = unpack_plot_values(values)
        fig, axs, cols, rows, titles = multiplot_helper(cols, *plot_values,
                                                        figsize=figsize,
                                                        titles=titles,
                                                        sharey=True, sharex=False,
                                                        fig=fig, axs=axs)
        groupmetrics = self.get_comp_groups(*plot_values, **comp_groups)
        num_bins = bins
        for i, plot_value in enumerate(plot_values):
            ax = axs[i]
            xlabel = xlabels.get(plot_value, plot_value)
            if isinstance(xlabel, str):
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(' '.join(xlabel))
            ax.grid(axis='y')
            fulldata = [i for endc in groupmetrics.values()
                        for i in [*endc.get_values(plot_value).values()]]
            bins = np.histogram(fulldata, metric_bins.get(plot_value, num_bins))[1]
            if not i % cols:
                ax.set_ylabel(ylabel)
            for group, endclasses in groupmetrics.items():
                local_kwargs = {**kwargs, **indiv_kwargs.get(group, {})}
                x = [*endclasses.get_values(plot_value).values()]
                ax.hist(x, bins, label=group, **local_kwargs)

        set_empty_multiplots(axs, len(plot_values), cols, xlab_ang=0, set_above=False)
        multiplot_legend_title(groupmetrics, axs, ax, legend_loc, title,
                               v_padding, h_padding, title_padding, legend_title)
        return fig, axs


def load(filename, filetype="", renest_dict=True, indiv=False, Rclass=Result):
    """
    Load a given (endclasses or mdlhists) results dictionary from a (npz/csv/json) file.

    e.g. a file saved using process.save_result or save_args in propagate functions.

    Parameters
    ----------
    filename : str
        Name of the file.
    filetype : str, optional
        Use to specify a filetype for the file (if not included in the filename).
        The default is "".
    renest_dict : bool, optional
        Whether to return . The default is True.
    indiv : bool, optional
        Whether the result is an individual file
        (e.g., in a folder of results for a given simulation).
        The default is False.
    Rclass : class
        Class to return (Result, History, or Dict)

    Returns
    -------
    result : Result/History
        Corresponding result/hist object with data loaded from the file.
    """
    if not os.path.exists(filename):
        raise Exception("File does not exist: "+filename)
    filetype = auto_filetype(filename, filetype)
    if filetype == 'npz':
        loaded = np.load(filename)
        resultdict = {k: v[()] for k, v in loaded.items()}
    elif filetype == 'csv':  # add support for nested dict mdlhist using flatten_hist?
        resultdict = load_csv(filename, indiv=indiv)
    elif filetype == 'json':
        resultdict = load_json(filename, indiv=indiv)
    else:
        raise Exception("Invalid File Type")
    if Rclass not in [dict, 'dict']:
        result = fromdict(Rclass, resultdict)
        if renest_dict == False:
            result = result.flatten()
    else:
        result = resultdict

    return result


def load_csv(filename, indiv=False):
    """Load csv files."""
    import pandas
    if indiv:
        resulttab = pandas.read_csv(filename, skiprows=1)
    else:
        resulttab = pandas.read_csv(filename)
    resultdict = resulttab.to_dict("list")
    resultdict = clean_resultdict_keys(resultdict)
    for key in resultdict:
        if (len(resultdict[key]) == 1 and
                (isinstance(resultdict[key], list) or
                 isinstance(resultdict[key], tuple))):
            resultdict[key] = resultdict[key][0]
        else:
            resultdict[key] = np.array(resultdict[key])
    if indiv:
        scenname = [*pandas.read_csv(filename, nrows=0).columns][0]
        resultdict = {scenname: resultdict}
    return resultdict


def load_json(filename, indiv=False):
    """Load json files."""
    import json
    with open(filename, 'r', encoding='utf8') as file_handle:
        loadeddict = json.load(file_handle)
        if indiv:
            key = [*loadeddict.keys()][0]
            loadeddict = loadeddict[key]
            loadeddict = {key+"."+innerkey: values for innerkey,
                          values in loadeddict.items()}
            resultdict = clean_resultdict_keys(loadeddict)
        else:
            resultdict = clean_resultdict_keys(loadeddict)
    file_handle.close()
    return resultdict


def load_folder(folder, filetype):
    """
    Loads endclass/mdlhist results from a given folder
    (e.g., that have been saved from multi-scenario propagate methods with 'indiv':True)

    Parameters
    ----------
    folder : str
        Name of the folder. Must be in the current directory
    filetype : str
        Type of files in the folder ('pickle', 'csv', or 'json')

    Returns
    -------
    files_to_read : list
        files to load for endclasses/mdlhists.
    """
    files = os.listdir(folder)
    files_toread = []
    for file in files:
        read_filetype = auto_filetype(file)
        if read_filetype == filetype:
            files_toread.append(file)
    return files_toread


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
