# -*- coding: utf-8 -*-
"""
Description: A module defining how simulation results (histories) structured and
processed. Has classes:

- :class:`Result`:  Class for defining result dictionaries
  (nested dictionaries of metric(s))
- :class:`History`: Class for defining simulation histories
  (nested dictionaries of arrays or lists)

And functions:

- :func:`load`: Loads a given file to a Result/History
- :func:`load_folder`: Loads a given folder to a Result/History

Private Methods:

- :func:`file_check`: Check if files exists and whether to overwrite the file
- :func:`auto_filetype`: Helper function that automatically determines the filetype
  (pickle, csv, or json) of a given filename
- :func:`create_indiv_filename`: Helper function that creates an individualized name for
  a file given the general filename and an individual id
- :func:`clean_resultdict_keys`: Helper function for recreating results dictionary keys
  (tuples) from a dictionary loaded from a file (where keys are strings)
  (used in csv/json results)
- :func:`get_dict_attr`: Gets attributes *attr from a given nested dict dict_in of class
  des_class
- :func:`fromdict`: Creates new history/result from given dictionary
- :func:`check_include_errors`: Helper function for Result Class, Cycles through
  `check_include_error`.
- :func:`check_include_error`: Helper function to raise exceptions for errors.
- :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap
- :func:`diff`: Helper function for finding inconsistent states between val1, val2, with
  the difftype option
- :func:`nan_to_x`: Helper function for Result Class, returns nan as zero if present,
  otherwise returns the number
- :func:`is_numeric`: Helper function for Result Class, checks if a given value is
  numeric
- :func:`join_key`: Helper function for Result Class
- :func:`is_known_immutable`: Helper function for History Class
- :func:`is_known_mutable`: Helper function for History Class
- :func:`to_include_keys`: Determine what dict keys to include from Result given nested
  to_include dictionary
- :func:`get_sub_include`: Determines what attributes of att to include based on the
  provided dict/str/list/set to_include
- :func:`init_indicator_hist`: Creates a history for an object with indicator methods
  (e.g., obj.indicate_XX)
- :func:`init_hist_iter`: Initializes the history for a given attribute att with value
  val. Enables the recursive definition of a history as a nested structure.
- :func:`init_dicthist`: Initializes histories for dictionary attributes (if any)
"""

import numpy as np
import pandas as pd
import copy
import sys
import os
from collections import UserDict
from fmdtools.define.common import get_var, t_key, get_obj_indicators, nest_dict


def file_check(filename, overwrite):
    """Check if files exists and whether to overwrite the file"""
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
    """Helper function that automatically determines the filetype (pickle, csv, or json)
    of a given filename"""
    if not filetype:
        if '.' not in filename:
            raise Exception("No file extension in: " + filename)
        if filename[-4:] == '.pkl':
            filetype = "pickle"
        elif filename[-4:] == '.csv':
            filetype = "csv"
        elif filename[-5:] == '.json':
            filetype = "json"
        else:
            raise Exception("Invalid File Type in: " + filename +
                            ", ensure extension is pkl, csv, or json ")
    return filetype


def create_indiv_filename(filename, indiv_id, splitchar='_'):
    """Helper file that creates an individualized name for a file given the general
    filename and an individual id"""
    filename_parts = filename.split(".")
    filename_parts.insert(1, '.')
    filename_parts.insert(1, splitchar+indiv_id)
    return "".join(filename_parts)


def clean_resultdict_keys(resultdict_dirty):
    """
    Helper function for recreating results dictionary keys (tuples) from a dictionary
    loaded from a file (where keys are
    strings) (used in csv/json results)

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
    """Gets attributes *attr from a given nested dict dict_in of class des_class"""
    if len(attr) == 1:
        return dict_in[attr[0]]
    else:
        return get_dict_attr(des_class(dict_in[attr[0]]), *attr[1:])


def fromdict(resultclass, inputdict):
    """Creates new history/result from given dictionary"""
    newhist = resultclass()
    for k, val in inputdict.items():
        if isinstance(val, dict):
            newhist[k] = resultclass.fromdict(val)
        else:
            newhist[k] = val
    return newhist


def check_include_errors(result, to_include):
    if type(to_include) != str:
        for k in to_include:
            check_include_error(result, k)
    else:
        check_include_error(result, to_include)


def check_include_error(result, to_include):
    if to_include not in ('all', 'default') and to_include not in result:
        raise Exception("to_include key " + to_include +
                        " not in result keys: " + str(result.keys()))

def is_numeric(val):
    """Checks if a given value is numeric"""
    try:
        return np.issubdtype(np.array(val).dtype, np.number)
    except:
        return type(val) in [float, bool, int]


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


class Result(UserDict):
    """
    Result is a special type of dictionary that makes it convenient to store, access,
    and load results form a model/simulation.

    As a dictionary, it supports dict-based item assignement (e.g. r['x']=10) but
    also enables convenient access via __getattr__, e.g.:
        r[x] = 10
        r.x
        > 10

    It also can return a flattened version of its nested sturcture via Result.flatten(),
    e.g.:
        r = Result(y=Result(z=1))
        rf = r.flatten()
        r[('y','z')]
        > 1

    It also enables saving and loading to files via r.save(), r.load(), and
    r.load_folder()
    """

    def __repr__(self, ind=0):
        str_rep = ""
        for k, val in self.items():
            if isinstance(val, np.ndarray) or isinstance(val, list):
                if type(k) == tuple:
                    k = str(k)
                val_rep = ind*"--"+k+": "
                if len(val_rep) > 40:
                    val_rep = val_rep[:20]
                form = '{:>'+str(40-len(val_rep))+'}'
                vv = form.format("array("+str(len(val))+")")
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
        return str_rep

    def all(self):
        return tuple(self.data.values())

    def __eq__(self, other):
        """
        Checks that the two values of the dictionary are equal. Enables the syntax
        result1 == result2, which returns True/False depending on if the keys/values
        are the same.

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
        Magic subtraction methods for Results. Used to enable uses such as:
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
        Finds the values of two results which are different.

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
        Provides dict-like access to the history/result across a number of arguments

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
        """Gets all values with the attribute attr"""
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

    def fromdict(inputdict):
        return fromdict(Result, inputdict)

    def load(filename, filetype="", renest_dict=False, indiv=False):
        """Loads as Result using :func:`load'"""
        inputdict = load(filename, filetype="", renest_dict=renest_dict,
                         indiv=indiv, Rclass=Result)
        return fromdict(Result, inputdict)

    def load_folder(folder, filetype, renest_dict=False):
        """Loads as History using :func:`load_folder'"""
        files_toread = load_folder(folder, filetype)
        result = Result()
        for filename in files_toread:
            result.update(Result.load(folder+'/'+filename, filetype,
                          renest_dict=renest_dict, indiv=True))
        if renest_dict == False:
            result = result.flatten()
        return result

    def get_values(self, *values):
        """Gets a dict with all values corresponding to the strings in *values"""
        h = self.__class__()
        flatself = self.flatten()
        k_vs = [k for k in flatself.keys() for v in values if k.endswith(v)]
        for k in k_vs:
            h[k] = flatself[k]
        return h

    def get_scens(self, *scens):
        """Gets a dictlike with all scenarios corresponding to the strings in *scens"""
        h = self.__class__()
        k_s = [k for k in self.keys()
               for s in scens if k.startswith(s+".") or '.'+s+'.' in k]
        for k in k_s:
            h[k] = self[k]
        return h

    def get_comp_groups(self, *values, **groups):
        """
        Gets comparison groups of *values (i.e., aspects of the model) in groups
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
            elif type(scens) == str:
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
            Dict with structure {'nominal': [list of nominal scenarios],
                                 'faulty': [list of faulty scenarios]}.
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
        newhist : Boolean, default = false
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
        Saves a given result variable (endclasses or mdlhists) to a file filename.
        Files can be saved as pkl, csv, or json.

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
        import dill
        import json
        import csv
        file_check(filename, overwrite)

        variable = self
        filetype = auto_filetype(filename, filetype)
        if filetype == 'pickle':
            with open(filename, 'wb') as file_handle:
                if result_id:
                    variable = {result_id: variable}
                dill.dump(variable, file_handle)
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
        """Makes a simple fmea-stype table of the metrics in the endclasses
        of a list of fault scenarios run. If metrics not provided, returns all"""
        nested = {k: {**v.endclass} for k, v in self.nest(levels=1).items()}
        tab = pd.DataFrame.from_dict(nested).transpose()
        if not metrics:
            return tab
        else:
            return tab.loc[:, metrics]

    def get_expected(self, app=[], with_nominal=False, difference_from_nominal=False):
        """
        Takes the expectation of numeric metrics in the result over given scenarios.

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
        """Calculates the percentage of a given indicator variable being True in
        endclasses"""
        return sum([int(bool(e)) for e in self.get_values(metric).values()
                    if not np.isnan(e)])/(len(self.get_values(metric))+1e-16)

    def rate(self, metric, prob_key='rate'):
        """Calculates the rate of a given indicator variable being True in endclasses
        using the rate variable in endclasses"""
        ecs = np.array([bool(e) for e in self.get_values(metric).values()
                        if not np.isnan(e)])
        weights = np.array([e for e in self.get_values(prob_key).values()
                            if not np.isnan(e)])
        return sum(ecs*weights)

    def end_diff(self, metric, nan_as=np.nan, as_ind=False, no_diff=False):
        """
        Calculates the difference between the nominal and fault scenarios for a set of
        endclasses

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
        Calculates the difference between the nominal and fault scenarios over a set of
        endclasses

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


def diff(val1, val2, difftype='bool'):
    """
    Helper function for finding inconsistent states between val1, val2, with the
    difftype option ('diff' (takes the difference), 'bool' (checks if the same),
                     and float (checks if under the provided tolerance))
    """
    if difftype == 'diff':
        return val1-val2
    elif difftype == 'bool':
        return val1 == val2
    elif type(difftype) == float:
        return abs(val1-val2) > difftype


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


def is_known_immutable(val):
    return type(val) in [int, float, str, tuple, bool] or isinstance(val, np.number)


def is_known_mutable(val):
    return type(val) in [dict, set]


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


def init_indicator_hist(obj, h, timerange, track):
    """
    Creates a history for an object with indicator methods (e.g., obj.indicate_XX)

    Parameters
    ----------
    obj : object
        Function/Flow/Model object with indicators
    h : History
        History of Function/Flow/Model object with indicators appended in h['i']
    timerange : iterable, optional
        Time-range to initialize the history over. The default is None.
    track : list/str/dict, optional
        argument specifying attributes for :func:`get_sub_include'. The default is None.

    Returns
    -------
    h : History
        History of states with structure {'XX':log} for indicator `obj.indicate_XX`
    """
    sub_track = get_sub_include('i', track)
    if sub_track:
        indicators = get_obj_indicators(obj)
        if indicators:
            h['i'] = History()
            for i, val in indicators.items():
                h['i'].init_att(i, val, timerange, sub_track, dtype=bool)


def init_hist_iter(att, val, timerange=None, track=None, dtype=None, str_size='<U20'):
    """
    Initializes the history for a given attribute att with value val. Enables
    the recursive definition of a history as a nested structure.

    If a timerange is given, the base fields are initializes as fixed-length numpy
    arrays corresponding to the data type of the field. Otherwise, an emty list
    is initialized.

    Parameters
    ----------
    att : str
        Name of the attribute.
    val : dict/field
        dict to be initialized as a History or field to be initialized as a list or
        numpy array
    timerange : iterable, optional
        Time-range to initialize the history over. The default is None.
    track : list/str/dict, optional
        argument specifying attributes for :func:`get_sub_include'. The default is None.
    dtype : str, optional
        Datatype to initialze the array as (if given). The default is None.
    str_size : str, optional
        Data type for strings. The default is '<U20'.

    Returns
    ---------
    Hist : History, List, or np.array
        Initialized history structure corresponding to the attribute
    """
    sub_track = get_sub_include(att, track)
    if sub_track and hasattr(val, 'create_hist'):
        return val.create_hist(timerange, sub_track)
    elif sub_track and isinstance(val, dict):
        return init_dicthist(val, timerange, sub_track)
    elif sub_track:
        hist = History()
        hist.init_att(att, val, timerange, track, dtype, str_size)


def init_dicthist(start_dict, timerange, track="all"):
    """
    Initializes histories for dictionary attributes (if any)

    Parameters
    ----------
    start_dict : dict
        Dictionary to initialize.
    timerange : iterable
        Timerange to initalize the hist over
    track : TYPE, optional
        DESCRIPTION. The default is "all".
    Returns
    -------
    Hist : History
        Initialized history structure corresponding to the attribute
    """
    hist = History()
    for att, val in start_dict.items():
        hist.init_att(att, val, timerange, track)
    return hist


class History(Result):
    """
    History is a special time of :class:'Result' specifically for keeping simulation
    histories (e.g., a log of states over time).

    It can be updated over time t using h.log(obj, t), where obj is an object with
    (nested) attributes that match the keys of the (nested) dictionary.
    """

    def init_att(self, att, val,
                 timerange=None, track=None, dtype=None, str_size='<U20'):
        sub_track = get_sub_include(att, track)
        if sub_track:
            if timerange is None:
                self[att] = [val]
            elif type(val) == str:
                self[att] = np.empty([len(timerange)], dtype=str_size)
            elif type(val) == dict:
                self[att] = init_dicthist(val, timerange, sub_track)
            elif type(val) == np.ndarray or dtype == np.ndarray:
                self[att] = np.array([val for i in timerange])
            elif dtype:
                self[att] = np.empty([len(timerange)], dtype=dtype)
            else:
                try:
                    self[att] = np.full(len(timerange), val)
                except:
                    self[att] = np.empty((len(timerange),), dtype=object)

    def fromdict(inputdict):
        return fromdict(History, inputdict)

    def load(filename, filetype="", renest_dict=False, indiv=False):
        """Loads file as History using :func:`load'"""
        inputdict = load(filename, filetype=filetype,
                         renest_dict=renest_dict, indiv=indiv, Rclass=History)
        return fromdict(History, inputdict)

    def load_folder(folder, filetype, renest_dict=False):
        """Loads folder as History using :func:`load_folder'"""
        files_toread = load_folder(folder, filetype)
        hist = History()
        for filename in files_toread:
            hist.update(History.load(folder+'/'+filename, filetype,
                                     renest_dict=renest_dict, indiv=True))
        if renest_dict == False:
            hist = hist.flatten()
        return hist


    def get_different(self, other):
        """
        Finds the values of two histories which are different.

        Parameters
        ----------
        other : History
            History to compare against

        Returns
        -------
        different : History
            History with entries corresponding to the difference between the two
            histories.
        """
        diff = self-other
        different = self.__class__()
        different.data = {k: v for k, v in diff.items() if any(v)}
        return different

    def copy(self):
        """Creates a new independent copy of the current history dict"""
        newhist = History()
        for k, v in self.items():
            if isinstance(v, History):
                newhist[k] = v.copy()
            else:
                newhist[k] = np.copy(v)
        return newhist

    def log(self, obj, t_ind, time=None):
        """
        Updates the history from obj at the time t_ind

        Parameters
        ----------
        obj : Model/Function/State...
            Object to log
        t_ind : int
            Time-index of the log.
        time : float
            Real time for the history (if initialized). Used at the top level of the
            history.
        """
        for att, hist in self.items():
            try:
                val = None
                if att == 'time' and time is not None:
                    val = time
                elif att.startswith('i.') or '.i.' in att:
                    split_att = att.split('.')
                    i_ind = split_att.index('i')
                    new_split_att = split_att[:i_ind] + ['indicate_'+split_att[-1]]
                    methname = '.'.join(new_split_att)
                    val = get_var(obj, methname)(time)
                elif 'faults' in att:
                    split_att = att.split('.')
                    faultind = split_att.index('faults')
                    modename = split_att[faultind+1]
                    fault_att = '.'.join(split_att[:faultind])
                    val = modename in get_var(obj, fault_att).faults
                else:
                    val = get_var(obj, att)
            except: 
                raise Exception("Unable to log att " + str(att) + " in " +
                                str(obj.__class__.__name__) + ', val=' + str(val))

            if type(hist) == History:
                hist.log(val, t_ind)
            else:
                if is_known_mutable(val):
                    val = copy.deepcopy(val)
                if type(hist) == list:
                    hist.append(val)
                elif isinstance(hist, np.ndarray):
                    try:
                        hist[t_ind] = val
                    except Exception as e:
                        obj_str = "Error logging obj "+obj.__class__.__name__+": "
                        if t_ind >= len(hist):
                            raise Exception(obj_str + "Time beyond range of model" +
                                            "history--check staged execution " +
                                            "and simulation time settings" +
                                            " (end condition, mdl.sp.times)") from e
                        elif not np.can_cast(type(val), type(hist[t_ind])):
                            raise Exception(obj_str + str(att)+" changed type: " +
                                            str(type(hist[t_ind])) + " to " +
                                            str(type(val)) + " at t_ind=" +
                                            str(t_ind)) from e
                        else:
                            raise Exception(obj_str + "Value too large to represent: "
                                            + att + "=" + str(val)) from e

    def cut(self, end_ind=None, start_ind=None, newcopy=False):
        """Cuts the history to a given index"""
        if newcopy:
            hist = self.copy()
        else:
            hist = self
        for name, att in hist.items():
            if isinstance(att, History):
                hist[name] = hist[name].cut(end_ind, start_ind, newcopy=False)
            else: 
                try:
                    if end_ind is None:
                        hist[name] = att[start_ind:]
                    elif start_ind is None:
                        hist[name] = att[:end_ind+1]
                    else:
                        hist[name] = att[start_ind:end_ind+1]
                except TypeError as e:
                    raise Exception("Invalid history for name " + name +
                                    " in history "+str(hist.data)) from e
        return hist

    def get_slice(self, t_ind=0):
        """
        Returns a dictionary of values from (flattenned) version of the history at t_ind
        """
        flathist = self.flatten()
        slice_dict = dict.fromkeys(flathist)
        for key, arr in flathist.items():
            slice_dict[key] = flathist[key][t_ind]
        return slice_dict

    def is_in(self, at):
        """ checks if at is in the dictionary"""
        return any([k for k in self.keys() if at in k])

    def get_fault_time(self, metric="earliest"):
        """
        Gets the time a fault is present in the system

        Parameters
        ----------
        metric : 'earliest','latest','total', optional
            Earliest, latest, or total time fault(s) are present.
            The default is "earliest".

        Returns
        -------
        int
            index in the history when the fault is present
        """
        flatdict = self.flatten()
        all_faults_hist = np.sum([v for k, v in flatdict.items() if 'faults' in k], 0)
        if metric == 'earliest':
            return np.where(all_faults_hist >= 1)
        elif metric == 'latest':
            return np.where(np.flip(all_faults_hist) >= 1)
        elif metric == 'total':
            return np.sum(all_faults_hist >= 1)

    def _prep_faulty(self):
        """Helper that creates a faulty history of states from the current history"""
        if self.is_in('faulty'):
            return self.faulty.flatten()
        else:
            return self.flatten()

    def _prep_nom_faulty(self, nomhist={}):
        """Helper that creates a nominal history of states from the current history"""
        if not nomhist:
            nomhist = self.nominal.flatten()
        else:
            nomhist = nomhist.flatten()
        return nomhist, self._prep_faulty()

    def get_degraded_hist(self, *attrs, nomhist={}, operator=np.prod, difftype='bool',
                          withtime=True, withtotal=True):
        """
        Gets history of times when the attributes *attrs deviate from their nominal
        values

        Parameters
        ----------
        *attrs : names of attributes
            Names to check (e.g., `flow_1`, `fxn_2`)
        nomhist : History, optional
            Nominal history to compare against
            (otherwise uses internal nomhist, if available)
        operator : function
            Method of combining multiple degraded values. The default is np.prod
        difftype : 'bool'/'diff'/float
            Way to calculate the difference:

                - for 'bool', it is calculated as an equality nom == faulty
                - for 'diff', it is calculated as a difference nom - faulty
                - if a float, is provided, it is calculated as nom - fault > diff
        threshold : 0.000001
            Threshold for degradation
        withtime : bool
            Whether to include time in the dict. Default is True.
        withtotal : bool
            Whether to include a total in the dict. Default is True.

        Returns
        -------
        deghist : History
            History of degraded attributes
        """
        if not attrs:
            attrs = self.keys()

        nomhist, faulthist = self._prep_nom_faulty(nomhist)
        deghist = History()
        for att in attrs:
            att_diff = [diff(nomhist[k], v, difftype)
                        for k, v in faulthist.items()
                        if att in k]
            if att_diff:
                deghist[att] = operator(att_diff, 0)
        if withtotal:
            deghist['total'] = len(deghist.values()) - np.sum([*deghist.values()], axis=0)
        if withtime:
            deghist['time'] = nomhist['time']
        return deghist

    def get_faults_hist(self, *attrs):
        """
        Gets fault names associated with the given attributes

        Parameters
        ----------
        *attrs : strs
            Names to find in the history.

        Returns
        -------
        faults_hist : History
            History of the attrs and their corresponding faults
        """
        faulthist = self._prep_faulty()
        faults_hist = History()
        if not attrs:
            attrs = self.keys()
        for att in attrs:
            faults_hist[att] = History({k.split('.')[-1]: v for k, v in faulthist.items()
                                        if ('.'+att+'.m.faults' in k) or
                                        (att+'.m.faults' in k and k.startswith(att))})
        return faults_hist

    def get_faulty_hist(self, *attrs, withtime=True, withtotal=True, operator=np.any):
        """
        Get the times when the attributes *attrs have faults present.

        Parameters
        ----------
        *attrs : names of attributes
            Names to check (e.g., `fxn_1`, `fxn_2`)
        withtime : bool
            Whether to include time in the dict. Default is True.
        withtotal : bool
            Whether to include a total in the dict. Default is True.
        operator : function
            Method of combining multiple degraded values. The default is np.any

        Returns
        -------
        has_faults_hist : History
            History of attrs being faulty/not faulty
        """
        faulthist = self._prep_faulty()
        faults_hist = self.get_faults_hist(*attrs)
        has_faults_hist = History()
        for att in attrs:
            if faults_hist[att]:
                has_faults_hist[att] = operator([*faults_hist[att].values()], 0)
        if withtotal and has_faults_hist:
            has_faults_hist['total'] = np.sum([*has_faults_hist.values()], axis=0)
        elif withtotal:
            has_faults_hist['total'] = 0 * faulthist['time']
        if withtime:
            has_faults_hist['time'] = faulthist['time']
        return has_faults_hist

    def get_summary(self, *attrs, operator=np.max):
        """
        Creates summary of the history based on a given metric

        Parameters
        ----------
        *attrs : names of attributes
            Names to check (e.g., `fxn_1`, `fxn_2`). If not provided, uses all.
        operator : aggregation function, optional
            Way to aggregate the time history (E.g., np.max, np.min, np.average, etc).
            The default is np.max.

        Returns
        -------
        summary : Result
            Corresponding summary metrics from this history
        """
        flathist = self.flatten()
        summary = Result()
        if not attrs:
            attrs = self.keys()
        for att in attrs:
            if att in self:
                summary[att] = operator(self[att])
        return summary

    def get_fault_degradation_summary(self, *attrs):
        """
        Creates a Result with values for the *attrs that are faulty/degraded

        Parameters
        ----------
        *attrs : str
            Attribute(s) to check.

        Returns
        -------
        Result
            Result dict with structure {'degraded':['degattrname'],
                                        'faulty':['faultyattrname']]}
        """
        faulty_hist = self.get_faulty_hist(*attrs, withtotal=False, withtime=False)
        faulty = [k for k, v in faulty_hist.items() if np.any(v)]
        deg_hist = self.get_degraded_hist(*attrs, withtotal=False, withtime=False)
        degraded = [k for k, v in deg_hist.items() if not np.all(v)]
        return Result(faulty=faulty, degraded=degraded)

    def get_metric(self, value, metric=np.mean, args=(), axis=None):
        """
        Calculates a statistic of the value using a provided metric function.

        Parameters
        ----------
        value : str
            Value of the history to calculate the statistic over
        metric : func, optional
            Function to process the history (e.g. np.mean, np.min...).
            The default is np.mean.
        args : args
            Arguments for the metric function. Default is ().
        axis : None or 0 or 1
            Whether to take the metric over variables (0) or over time (1) or
            both (None). The default is None.
        """
        vals = self.get_values(value)
        return metric([*vals.values()], *args, axis=axis)

    def get_metrics(self, *values, metric=np.mean, args=(), axis=None):
        """
        Calculates a statistic of the values using a provided metric function.

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


def load(filename, filetype="", renest_dict=True, indiv=False, Rclass=History):
    """
    Loads a given (endclasses or mdlhists) results dictionary from a (pickle/csv/json)
    file.
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
        (e.g. in a folder of results for a given simulation).
        The default is False.
    Rclass : class
        Class to return (Result, History, or Dict)

    Returns
    -------
    result : Result/History
        Corresponding result/hist object with data loaded from the file.
    """
    import dill
    import json
    import pandas
    if not os.path.exists(filename):
        raise Exception("File does not exist: "+filename)
    filetype = auto_filetype(filename, filetype)
    if filetype == 'pickle':
        with open(filename, 'rb') as file_handle:
            resultdict = dill.load(file_handle)
        file_handle.close()
    elif filetype == 'csv':  # add support for nested dict mdlhist using flatten_hist?
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
    elif filetype == 'json':
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
    else:
        raise Exception("Invalid File Type")
    if Rclass not in [dict, 'dict']:
        result = fromdict(Rclass, resultdict)
        if renest_dict == False:
            result = result.flatten()
    else:
        result = resultdict

    return result


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
