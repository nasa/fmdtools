# -*- coding: utf-8 -*-
"""
Description: A module defining how simulation histories are structured and
processed. Has classes:

- :class:`History`: Class for defining simulation histories
  (nested dictionaries of arrays or lists)

And functions/methods:

- :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap
- :func:`diff`: Helper function for finding inconsistent states between val1, val2, with
  the difftype option
- :func:`init_indicator_hist`: Creates a history for an object with indicator methods
  (e.g., obj.indicate_XX)
- :func:`init_hist_iter`: Initializes the history for a given attribute att with value
  val. Enables the recursive definition of a history as a nested structure.
- :func:`init_dicthist`: Initializes histories for dictionary attributes (if any)
"""
from fmdtools.analyze.result import Result, get_sub_include, load_folder, load, fromdict
from fmdtools.define.common import get_var, get_obj_indicators
import numpy as np


def is_known_immutable(val):
    """Check if value is known immutable."""
    return type(val) in [int, float, str, tuple, bool] or isinstance(val, np.number)


def is_known_mutable(val):
    """Check if value is a known mutable."""
    return type(val) in [dict, set]


def init_indicator_hist(obj, h, timerange, track):
    """
    Create a history for an object with indicator methods (e.g., obj.indicate_XX).

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
    Initialize the history for a given attribute att with value val.

    Enables the recursive definition of a history as a nested structure.

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
    -------
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
    Initialize histories for dictionary attributes (if any).

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


def diff(val1, val2, difftype='bool'):
    """
    Find inconsistent states between val1, val2.

    The difftype option ('diff' (takes the difference), 'bool' (checks if the same),
                         and float (checks if under the provided tolerance))
    """
    if difftype == 'diff':
        return val1-val2
    elif difftype == 'bool':
        return val1 == val2
    elif type(difftype) == float:
        return abs(val1-val2) > difftype


class History(Result):
    """
    Class for recording and analyzing simulation histories.

    Histories log states of the model over time.

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
        """Load file as History using :func:`load'."""
        inputdict = load(filename, filetype=filetype,
                         renest_dict=renest_dict, indiv=indiv, Rclass=History)
        return fromdict(History, inputdict)

    def load_folder(folder, filetype, renest_dict=False):
        """Load folder as History using :func:`load_folder'."""
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
        Find the values of two histories which are different.

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
        """Create a new independent copy of the current history dict."""
        newhist = History()
        for k, v in self.items():
            if isinstance(v, History):
                newhist[k] = v.copy()
            else:
                newhist[k] = np.copy(v)
        return newhist

    def log(self, obj, t_ind, time=None):
        """
        Update the history from obj at the time t_ind.

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
        """Cut the history to a given index."""
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
        Return a dictionary of values from (flattenned) version of the history at t_ind.
        """
        flathist = self.flatten()
        slice_dict = dict.fromkeys(flathist)
        for key, arr in flathist.items():
            slice_dict[key] = flathist[key][t_ind]
        return slice_dict

    def is_in(self, at):
        """Check if at is in the dictionary."""
        return any([k for k in self.keys() if at in k])

    def get_fault_time(self, metric="earliest"):
        """
        Get the time a fault is present in the system.

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
        Get history of times when the attributes *attrs deviate from nominal values.

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
        Get fault names associated with the given attributes.

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
        Create summary of the history based on a given metric.

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
        Create a Result with values for the *attrs that are faulty/degraded.

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
