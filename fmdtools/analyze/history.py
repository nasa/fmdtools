#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`History`: class and for structuring/analyzing simulation histories.

- :class:`History`: Class for defining simulation histories
  (nested dictionaries of arrays or lists)

And functions/methods:

- :func:`init_dicthist`: Initializes histories for dictionary attributes (if any)
- :func:`def prep_hists`: Prepare the history for plotting.

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

from fmdtools.define.base import get_var, is_known_mutable

from fmdtools.analyze.result import Result, load_folder, load, fromdict
from fmdtools.analyze.common import calc_metric_ci, get_sub_include
from fmdtools.analyze.common import unpack_plot_values, phase_overlay
from fmdtools.analyze.common import multiplot_legend_title, multiplot_helper
from fmdtools.analyze.common import plot_err_hist, setup_plot, set_empty_multiplots
from fmdtools.analyze.common import mark_times, consolidate_legend, add_title_xylabs
from fmdtools.analyze.common import prep_animation_title, clear_prev_figure
from fmdtools.analyze.common import diff

from matplotlib import animation
from functools import partial
import numpy as np
import copy


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


def prep_hists(simhists, plot_values, comp_groups, indiv_kwargs, time='time'):
    """Prepare hists for plotting."""
    # Process data - clip and flatten
    if "time" in simhists:
        simhists = History(nominal=simhists).flatten()
    else:
        simhists = simhists.flatten()

    plot_values = unpack_plot_values(plot_values)

    grouphists = simhists.get_comp_groups(*plot_values, time=time, **comp_groups)

    # Set up plots and iteration
    if 'nominal' in grouphists.keys() and len(grouphists) > 1:
        indiv_kwargs['nominal'] = indiv_kwargs.get(
            'nominal', {'color': 'blue', 'ls': '--'})
    else:
        indiv_kwargs.pop('nominal', '')

    if 'faulty' in grouphists.keys():
        indiv_kwargs['faulty'] = indiv_kwargs.get('faulty', {'color': 'red'})
    else:
        indiv_kwargs.pop('faulty', '')

    return simhists, plot_values, grouphists, indiv_kwargs


class History(Result):
    """
    Class for recording and analyzing simulation histories.

    Histories log states of the model over time.

    It can be updated over time t using h.log(obj, t), where obj is an object with
    (nested) attributes that match the keys of the (nested) dictionary.

    Examples
    --------
    # histories act the same as results, but with the values being arrays:

    >>> hist = History({"a": [1, 2, 3], "b": [4, 5, 6], "time": [0, 1, 2]})
    >>> hist
    a:                              array(3)
    b:                              array(3)
    time:                           array(3)

    history access is the same as a result:

    >>> hist.a
    [1, 2, 3]

    metrics can be gotten from histories over time:

    >>> hist = History({"a.a": [1, 2, 3], "b.a": [4, 5, 6], "time": [0, 1, 2]})
    >>> hist.get_metric("a", axis=0)
    array([2.5, 3.5, 4.5])

    or over all times:

    >>> hist.get_metric("a")
    3.5
    """

    def init_att(self, att, val,
                 timerange=None, track=None, dtype=None, str_size='<U20'):
        """Add key/hist array for an attribute over a given timerange."""
        sub_track = get_sub_include(att, track)
        if sub_track:
            if isinstance(val, dict) or (hasattr(val, 'keys') and hasattr(val, 'values')):
                self[att] = init_dicthist(val, timerange, sub_track)
            elif timerange is None:
                self[att] = [val]
            elif isinstance(val, str):
                self[att] = np.empty([len(timerange)], dtype=str_size)
            elif isinstance(val, np.ndarray) or dtype == np.ndarray:
                self[att] = np.array([val for i in timerange])
            elif dtype:
                self[att] = np.empty([len(timerange)], dtype=dtype)
            else:
                try:
                    self[att] = np.full(len(timerange), val)
                except ValueError:
                    self[att] = np.empty((len(timerange),), dtype=object)

    @classmethod
    def fromdict(cls, inputdict):
        """Create history from dictionary. Used in initialization."""
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
                    try:
                        val = get_var(obj, methname)(time)
                    except TypeError:
                        val = get_var(obj, methname)()
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

            if isinstance(hist, History):
                hist.log(val, t_ind)
            else:
                if is_known_mutable(val):
                    val = copy.deepcopy(val)
                if isinstance(hist, list):
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
                                            " (end condition, mdl.sp.end_time)") from e
                        elif not np.can_cast(type(val), type(hist[t_ind])):
                            raise Exception(obj_str + str(att)+" changed type: " +
                                            str(type(hist[t_ind])) + " to " +
                                            str(type(val)) + " at t_ind=" +
                                            str(t_ind)) from e
                        else:
                            raise Exception(obj_str + "Value too large to represent: "
                                            + att + "=" + str(val)) from e

    def cut(self, end_ind=None, start_ind=None, newcopy=False):
        """
        Cut the history to a given index.

        Parameters
        ----------
        end_ind : int, optional
            the index of the array that you want to cut the history upto.
            Default is None.
        start_ind: int, optional
            the index of the array that you want to start cutting the history from.
            The default is None, which starts from the 0th index.
        newcopy: bool, optional
            Tells whether to creat a new history variable with the information what was
            cut or cut the original history variable. Default is False.

        Examples
        --------
        >>> hist = History({'a':[2,3,4,5], 'b':[5,4,3,2,1,0], 'time': [0,1,2,3,4,5]})
        >>> cut_hist = hist.cut(3)
        >>> cut_hist
        a:                              array(4)
        b:                              array(4)
        time:                           array(4)

        >>> cut_hist.a
        [2, 3, 4, 5]
        >>> cut_hist.b
        [5, 4, 3, 2]
        >>> cut_hist.time
        [0, 1, 2, 3]
        """
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
        Return Result of values from (flattenned) version of the history at t_ind.

        Examples
        --------
        >>> h = History(a=[3,4,5], time=[1,2,3])
        >>> h.get_slice(0)
        a:                                     3
        time:                                  1
        >>> h.get_slice(2)
        a:                                     5
        time:                                  3
        """
        flathist = self.flatten()
        slice_dict = dict.fromkeys(flathist)
        for key, arr in flathist.items():
            slice_dict[key] = flathist[key][t_ind]
        return Result(slice_dict)

    def is_in(self, at):
        """Check if at is in the dictionary."""
        return any([k for k in self.keys() if at in k])

    def get_fault_time(self, metric="earliest"):
        """
        Get the time a fault is present in the system.

        Parameters
        ----------
        metric : 'earliest','latest','total', 'times', optional
            Earliest, latest, or total time fault(s) are present.
            The default is "earliest".

        Returns
        -------
        int
            index in the history when the fault is present

        Examples
        --------
        >>> History({'m.faults.fault1': [False, False, False]}).get_fault_time()
        nan
        >>> History({'m.faults.fault1': [False, False, True]}).get_fault_time()
        2
        """
        flatdict = self.flatten()
        all_faults_hist = np.sum([v for k, v in flatdict.items() if 'faults' in k], 0)
        if metric == 'total':
            return np.sum(all_faults_hist >= 1)
        else:
            times = np.where(all_faults_hist >= 1)[0]
            if times.size == 0:
                return np.NaN
            elif metric == 'times':
                return times
            elif metric == 'earliest':
                return times[0]
            elif metric == 'latest':
                return times[-1]

    def _prep_faulty(self):
        """Create a faulty history of states from the current history."""
        if self.is_in('faulty'):
            return self.faulty.flatten()
        else:
            return self.flatten()

    def _prep_nom_faulty(self, nomhist={}, align=True):
        """Create a nominal history of states from the current history."""
        if not nomhist:
            nomhist = self.nominal.flatten()
        else:
            nomhist = nomhist.flatten()
        faulthist = self._prep_faulty()
        if align:
            faulthist._align(nomhist)
        return nomhist, faulthist

    def _align(self, nomhist):
        """Align the timeranges of the current hist and an external hist."""
        nom_start = nomhist.time[0]
        fault_start = self.time[0]
        nom_start_ind, fault_start_ind = 0, 0
        if nom_start < fault_start:
            nom_start_ind = np.where(nomhist.time == fault_start)[0][0]
            fault_start_ind = 0
        elif fault_start < nom_start:
            nom_start_ind = 0
            fault_start_ind = np.where(self.time == nom_start)[0][0]

        nom_end = nomhist.time[-1]
        fault_end = self.time[-1]
        nom_end_ind = len(nomhist.time)
        fault_end_ind = len(self.time)
        if nom_end > fault_end:
            nom_end_ind = np.where(nomhist.time == fault_end)[0][0]
            fault_end_ind = len(self.time)
        elif fault_end > nom_end:
            nom_end_ind = len(nomhist.time)
            fault_end_ind = np.where(self.time == nom_end)[0][0]
        self.cut(fault_end_ind, fault_start_ind)
        nomhist.cut(nom_end_ind, nom_start_ind)
        if len(nomhist.time) != len(self.time):
            print(nomhist)

    def get_degraded_hist(self, *attrs, nomhist={}, operator=np.any, difftype='bool',
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
        withtime : bool
            Whether to include time in the dict. Default is True.
        withtotal : bool
            Whether to include a total in the dict. Default is True.

        Returns
        -------
        deghist : History
            History of degraded attributes

        Examples
        --------
        >>> h = History({'nominal.a': np.array([1,2,3]), 'nominal.time': [0,1,2], 'faulty.a': np.array([1,1,1]), 'faulty.time': [0,1,2]})
        >>> dh = h.get_degraded_hist("a")
        >>> dh['total']
        array([0, 1, 1])
        >>> dh['a']
        array([False,  True,  True])
        """
        if not attrs:
            attrs = self.keys()

        nomhist, faulthist = self._prep_nom_faulty(nomhist, align=True)
        deghist = History()
        for att in attrs:
            try:
                att_diff = [diff(nomhist[k], v, difftype)
                            for k, v in faulthist.items()
                            if att in k]
                if att_diff:
                    try:
                        deghist[att] = operator(att_diff, 0)
                    except ValueError:
                        deghist[att] = operator([operator(arr) for arr in att_diff], 0)
            except Exception as e:
                raise Exception("Unable to diff att " + att) from e

        if withtotal:
            deghist['total'] = np.sum([*deghist.values()], axis=0)
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

        Examples
        --------
        >>> h = History({'f1.m.faults': [False, False, True], 'f2.m.faults': [True, False, False], 'time': [0,1,2]})
        >>> fh = h.get_faulty_hist("f1", "f2")
        >>> fh
        f1:                             array(3)
        f2:                             array(3)
        total:                          array(3)
        time:                           array(3)
        >>> fh['total']
        array([1, 0, 1])
        >>> fh.f1
        array([False, False,  True])
        >>> fh.f2
        array([ True, False, False])
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
        degraded = [k for k, v in deg_hist.items() if np.any(v)]
        return Result(faulty=faulty, degraded=degraded)

    def plot_line(self, *plot_values, cols=2, aggregation='individual',
                  legend_loc=-1, xlabel='time', ylabels={}, max_ind='max', titles={},
                  title='', indiv_kwargs={}, time_slice=[], time_slice_label=None,
                  figsize='default', comp_groups={},
                  v_padding=None, h_padding=None, title_padding=0.0,
                  phases={}, phase_kwargs={}, legend_title=None,  **kwargs):
        """
        Plot history values over time aggregated over comp_groups.

        Parameters
        ----------
        *plot_values : strs
            names of values to pul (e.g., 'fxns.move_water.s.flowrate').
            Can also be specified as a dict (e.g., {'fxns': 'move_water'}) to get all
            from a given fxn/flow/mode/etc.
        cols : int, optional
            columns to use in the figure. The default is 2.
        aggregation : str, optional
            Way of aggregating the plot values (e.g., which plot_XX_line method to call)
            The default is 'individual'.
        comp_groups : dict, optional
            Dictionary for comparison groups (if more than one) with structure given
            by: {'group1': ('scen1', 'scen2'), 'group2':('scen3', 'scen4')}.
            Default is {}, which compares nominal and faulty.
            If {'default': 'default'} is passed, all scenarios will be put in one group.
            If a legend is shown, group names are used as labels.
        legend_loc : int, optional
            Specifies the plot to place the legend on, if compared. Default is
            -1 (the last plot). To remove the legend, give a value of False
        xlabel : str, optional
            Label for the x-axes. Default is 'time'
        ylabels : dict, optional
            Label for the y-axes.
            Has structure::
                {(fxnflowname, value): 'label'}
        max_ind : int, optional
            index (usually correlates to time) cutoff for the simulation. Default is
            'max', which uses the first simulation termination time.
        title : str, optional
            overall title for the plot. Default is ''
        indiv_kwargs : dict, optional
            Dict of kwargs to use to differentiate each comparison group.
            Has structure::
                {comp1: kwargs1, comp2: kwargs2}
            where kwargs is an individual dict of plt.plot arguments for the
            comparison group comp (or scenario, if not aggregated) which overrides
            the global kwargs (or default behavior). If no comparison groups are given,
            use 'default' for a single history or 'nominal'/'faulty' for a fault history
            e.g.,::
                kwargs = {'nominal': {color: 'green'}}

            would make the nominal color green. Default is {}.
        time_slice : int/list, optional
            overlays a bar or bars at the given index when the fault was injected
            (if any). Default is []
        time_slice_label : str, optional
            label to use for the time slice bars in the legend. Default is None.
        figsize : tuple (float,float)
            x-y size for the figure. The default is 'default', which dymanically gives 3
            for each column and 2 for each row.
        phases : dict, optional
            Provide to overlay phases on the individual function histories, where phases
            is a dict of PhaseMaps from analyze.phases.from_hist. Default is {}.
        phase_kwargs : dict
            kwargs to plot.phase_overlay.
        legend_title : str, optional
            title for the legend. Default is None
        **kwargs : kwargs
            Keyword arguments to aggregation plotting functions (plot_xx_line) as well
            ass multiplot_legend_title.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        simhists, plot_values, grouphists, indiv_kwargs = prep_hists(self,
                                                                     plot_values,
                                                                     comp_groups,
                                                                     indiv_kwargs)
        fig, axs, cols, rows, subplot_titles = multiplot_helper(cols, *plot_values,
                                                                figsize=figsize,
                                                                titles=titles)

        for i, value in enumerate(plot_values):
            ax = axs[i]
            ax.grid()
            if i >= (rows-1)*cols and xlabel:
                xlab = xlabel
            else:
                xlab = ' '
            ylab = ylabels.get(value, kwargs.get('ylabel', ''))
            for group, hists in grouphists.items():
                loc_kwargs = {**kwargs, 'label': group, 'xlabel': xlab, 'ylabel': ylab,
                              'title': subplot_titles[value],
                              **indiv_kwargs.get(group, {})}
                try:
                    if aggregation == 'individual':
                        hists.plot_individual_line(value, fig, ax, **loc_kwargs)
                    elif aggregation == 'mean_std':
                        hists.plot_mean_std_line(value, fig, ax, **loc_kwargs)
                    elif aggregation == 'mean_ci':
                        hists.plot_mean_ci_line(value, fig, ax, max_ind=max_ind,
                                                **loc_kwargs)
                    elif aggregation == 'mean_bound':
                        hists.plot_mean_bound_line(value, fig, ax, **loc_kwargs)
                    elif aggregation == 'percentile':
                        hists.plot_percentile_line(value, fig, ax, **loc_kwargs)
                    else:
                        raise Exception("Invalid aggregation option: "+aggregation)
                except Exception as e:
                    raise Exception("Error at plot_value " + str(value)
                                    + " and group: " + str(group)) from e
                if len(value) > 1 and value[1] in phases:
                    phase_overlay(ax, phases[value[1]], value[1])
            if isinstance(time_slice, int):
                ax.axvline(x=time_slice, color='k', label=time_slice_label)
            else:
                for ts in time_slice:
                    ax.axvline(x=ts, color='k', label=time_slice_label)
        set_empty_multiplots(axs, len(plot_values), cols, xlab_ang=0.0, grid=True)
        multiplot_legend_title(grouphists, axs, ax, legend_loc, title,
                               v_padding, h_padding, title_padding, legend_title)
        return fig, axs

    def plot_individual_line(self, value, fig=None, ax=None, figsize=(6, 4),
                             time='time', xlabel='', ylabel='', title='',
                             xlim=(), ylim=(), zlim=(), **kwargs):
        """Plot value in hist as individual lines."""
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        scens = [*self.nest(1).keys()]
        hist_to_plot = self.get_values(value)
        if 'color' not in kwargs:
            kwargs['color'] = ax._get_lines.get_next_color()
        for scen in scens:
            hist_to_plot = self.get(scen)
            h_value = hist_to_plot.get(value)
            times = hist_to_plot.get(time)
            ax.plot(times, h_value, **kwargs)
        if not xlim:
            min_ind = np.min([i[0] for i in self.get_values(time).values()])
            max_ind = np.max([i[-1] for i in self.get_values(time).values()])
            xlim = (min_ind, max_ind)
        add_title_xylabs(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                         xlim=xlim, ylim=ylim, zlim=zlim)
        return fig, ax

    def get_mean_std_errhist(self, value, time='time'):
        """
        Get aggregated err_hist of means surrounded by std deviation.

        Parameters
        ----------
        value : str
            Value to get mean and bounds of.

        Returns
        -------
        err_hist : History
            hist of line, low, high values. Has the form::
            {'time': times, 'stat': stat_values, 'low': low_values, 'high': high_values}

        Examples
        --------
        >>> hist = History({"a.a": [1, 2, 3], "b.a": [4, 5, 6], "time": [0, 1, 2]})
        >>> hist.get_mean_std_errhist("a").stat
        array([2.5, 3.5, 4.5])
        """
        hist = History()
        hist[time] = self.get_metric(time, axis=0)
        hist['stat'] = self.get_metric(value, np.mean, axis=0)
        std_dev = self.get_metric(value, np.std)
        hist['high'] = hist['stat']+std_dev/2
        hist['low'] = hist['stat']-std_dev/2
        return hist

    def plot_mean_std_line(self, value, fig=None, ax=None, figsize=(6, 4), time='time',
                           **kwargs):
        """Plot value in hist aggregated by mean and standard devation."""
        hist = self.get_mean_std_errhist(value, time=time)
        return plot_err_hist(hist, ax, fig, figsize, time=time, **kwargs)

    def get_mean_ci_errhist(self, value, ci=0.95, max_ind='max', time='time', **kwargs):
        """
        Get aggregated err_hist of means surrounded by confidence intervals.

        Parameters
        ----------
        value : str
            Value to get mean and bounds of.
        ci : float
            Fraction for confidence interval. Default is 0.95.
        max_ind : str/int
            Max index of time to clip to. Default is 'max'.

        Returns
        -------
        err_hist : History
            hist of line, low, high values. Has the form::
            {'time': times, 'stat': stat_values, 'low': low_values, 'high': high_values}

        Examples
        --------
        >>> hist = History({"a.a": [1, 2, 3], "b.a": [4, 5, 6], "time": [0, 1, 2]})
        >>> metric_ci = hist.get_mean_ci_errhist("a",rates=np.array([0,1]), metric=np.sum)
        >>> metric_ci.high
        array([4., 5., 6.])
        >>> metric_ci.low
        array([0., 0., 0.])
        >>> metric_ci.stat
        array([2. , 2.5, 3. ])
        """
        hist = History()
        hist[time] = self.get_metric(time, axis=0)
        if max_ind == 'max':
            max_ind = min([len(h) for h in self.values()])
        vals = np.array([*self.get_values(value).values()])[:, :max_ind]
        boot_stats = calc_metric_ci(vals, confidence_level=ci, axis=0, **kwargs)
        hist['stat'] = boot_stats[0]
        hist['low'] = boot_stats[1]
        hist['high'] = boot_stats[2]
        return hist

    def plot_mean_ci_line(self, value, fig=None, ax=None, figsize=(6, 4),
                          ci=0.95, max_ind='max', time='time', **kwargs):
        """Plot value in hist aggregated by bootstrap confidence interval for mean."""
        hist = self.get_mean_ci_errhist(value, ci, max_ind, time=time)
        return plot_err_hist(hist, ax, fig, figsize, time=time, **kwargs)

    def get_mean_bound_errhist(self, value, time='time'):
        """
        Get aggregated err_hist of means surrounded by bounds.

        Parameters
        ----------
        value : str
            Value to get mean and bounds of.

        Returns
        -------
        err_hist : History
            hist of line, low, high values. Has the form::
            {'time': times, 'stat': stat_values, 'low': low_values, 'high': high_values}

        Examples
        --------
        >>> hist = History({'a.b': [1], 'b.b': [2], 'c.b': [3], 'time': [0]})
        >>> errhist = hist.get_mean_bound_errhist("b")
        >>> errhist.stat
        array([2.])
        >>> errhist.high
        array([3])
        >>> errhist.low
        array([1])
        """
        hist = History()
        hist[time] = self.get_metric(time, axis=0)
        hist['stat'] = self.get_metric(value, np.mean, axis=0)
        hist['high'] = self.get_metric(value, np.max, axis=0)
        hist['low'] = self.get_metric(value, np.min, axis=0)
        return hist

    def plot_mean_bound_line(self, value, fig=None, ax=None, figsize=(6, 4),
                             time='time', **kwargs):
        """Plot the value in hist aggregated by the mean and variable bounds."""
        hist = self.get_mean_bound_errhist(value, time=time)
        return plot_err_hist(hist, ax, fig, figsize, time=time, **kwargs)

    def get_percentile_errhist(self, val, prange=50, time='time'):
        """
        Get aggregated err_hist of medians surrounded by percentile range prange.

        Parameters
        ----------
        val : str
            Value to get mean and percentiles of.
        prange : number
            Range of percentiles around the median to index.

        Returns
        -------
        err_hist : History
            hist of line, low, high values. Has the form::
            {'time': times, 'stat': stat_values, 'low': low_values, 'high': high_values}

        Examples
        --------
        >>> hist = History({"a.a": [1, 2, 3], "b.a": [4, 5, 6], "time": [0, 1, 2]})
        >>> hist.get_percentile_errhist("a").stat
        array([2.5, 3.5, 4.5])
        >>> hist.get_percentile_errhist("a").high
        array([3.25, 4.25, 5.25])
        >>> hist.get_percentile_errhist("a").low
        array([1.75, 2.75, 3.75])
        """
        hist = History()
        hist[time] = self.get_metric(time, axis=0)
        hist['stat'] = self.get_metric(val, np.median, axis=0)
        hist['low'] = self.get_metric(val, np.percentile, args=(50-prange/2,), axis=0)
        hist['high'] = self.get_metric(val, np.percentile, args=(50+prange/2,), axis=0)
        return hist

    def plot_percentile_line(self, value, fig=None, ax=None, figsize=(6, 4), prange=50,
                             with_bounds=True, time='time', **kwargs):
        """Plot the value in hist aggregated by percentiles."""
        hist = self.get_mean_bound_errhist(value, time=time)
        perc_hist = self.get_percentile_errhist(value, prange, time=time)
        if with_bounds:
            hist['med_low'] = perc_hist['low']
            hist['med_high'] = perc_hist['high']
        else:
            hist = perc_hist
        return plot_err_hist(hist, ax, fig, figsize, time=time, **kwargs)

    def plot_metric_dist(self, times, *plot_values, **kwargs):
        """
        Plot the distribution of values at defined time(s) over a number of scenarios.

        Parameters
        ----------
        times : list/int
            List of times (or single time) to key the model history from.
            If more than one time is provided, it takes the place of comp_groups.
        *plot_values : strs
            names of values to pull from the history (e.g., 'fxns.move_water.s.flow')
            Can also be specified as a dict (e.g., {'fxns':'move_water'}) to get all
            keys from a given fxn/flow/mode/etc.
        **kwargs : kwargs
            keyword arguments to Result.plot_metric_dist
        """
        flat_mdlhists = self.nest(levels=1)
        if type(times) in [int, float]:
            times = [times]
        if len(times) == 1 and kwargs.get('comp_groups', False):
            time_classes = Result({scen: Result(flat_hist.get_slice(times[0]))
                                   for scen, flat_hist in flat_mdlhists.items()})
            comp_groups = kwargs.pop('comp_groups')
        elif kwargs.get('comp_groups', False):
            raise Exception("Cannot compare times and comp_groups at the same time")
        else:
            time_classes = Result({str(t)+'_'+scen: Result(flat_hist.get_slice(t))
                                   for scen, flat_hist in flat_mdlhists.items()
                                   for t in times})
            comp_groups = {str(t): {str(t)+'_'+scen for scen in flat_mdlhists}
                           for t in times}
        res_to_plot = time_classes.flatten()
        fig, axs = res_to_plot.plot_metric_dist(*plot_values,
                                                comp_groups=comp_groups, **kwargs)
        return fig, axs

    def plot_metric_dist_from(self, time, plot_values=(), ax=False, **kwargs):
        """Alias for plot_metric_dist allowing animation."""
        kwargs = prep_animation_title(time, **kwargs)
        kwargs = clear_prev_figure(**kwargs)
        fig, axs = self.plot_metric_dist(time, *plot_values, **kwargs)
        return fig, axs

    def plot_trajectories(self, *plot_values,
                          comp_groups={}, indiv_kwargs={}, figsize=(4, 4), time='time',
                          time_groups=[], time_ticks=5.0, time_fontsize=8,
                          t_pretext="t=", xlim=(), ylim=(), zlim=(), legend=True,
                          xlabel='x', ylabel='y', zlabel='z', title='',
                          fig=None, ax=None, **kwargs):
        """
        Plot trajectories from the environment in 2d or 3d space.

        Parameters
        ----------
        *plot_values : str
            Plot values corresponding to the x/y/z values (e.g, 'position.s.x')
        comp_groups : dict, optional
            Dictionary for comparison groups (if more than one) with structure given by:
            ::
                {'group1': ('scen1', 'scen2'),
                 'group2':('scen3', 'scen4')}.

            Default is {}, which compares nominal and faulty.
            If {'default': 'default'} is passed, all scenarios will be put in one group.
            If a legend is shown, group names are used as labels.
        indiv_kwargs : dict, optional
            Dict of kwargs to use to differentiate each comparison group.
            Has structure::
                {comp1: kwargs1, comp2: kwargs2}

            where kwargs is an individual dict of plt.plot arguments for the
            comparison group comp (or scenario, if not aggregated) which overrides
            the global kwargs (or default behavior). If no comparison groups are given,
            use 'default' for a single history or 'nominal'/'faulty' for a fault history
            e.g.,::
                kwargs = {'nominal': {color: 'green'}}

            would make the nominal color green. Default is {}.
        figsize : tuple (float,float)
            x-y size for the figure. The default is 'default', which dymanically gives 3
            for each column and 2 for each row.
        time_groups : list, optional
            List of strings corresponding to groups (e.g., 'nominal') to label the time
            at each point in the trajectory. The default is [].
        time_ticks : float, optional
            Ticks for times (if used). The default is 5.0.
        time_fontsize : int, optional
            Fontsize for time-ticks. The default is 8.
        xlim : tuple, optional
            Limits on the x-axis. The default is ().
        ylim : tuple, optional
            Limits on the y-axis. The default is ().
        zlim : tuple, optional
            Limits on the z-axis. The default is ().
        legend : bool, optional
            Whether to show a legend. The default is True.
        title : str, optional
            Title to add. Default is '' (no title).
        fig : matplotlib.figure, optional
            Existing Figure. The default is None.
        ax : matplotlib.axis, optional
            Existing axis. The default is None.
         **kwargs : kwargs
            kwargs to ax.plot to use over all plots.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        simhists, plot_values, grouphists, indiv_kwargs = prep_hists(self,
                                                                     plot_values,
                                                                     comp_groups,
                                                                     indiv_kwargs,
                                                                     time=time)
        if len(plot_values) == 2:
            fig, ax = setup_plot(fig=fig, ax=ax, z=False, figsize=figsize)
        elif len(plot_values) == 3:
            fig, ax = setup_plot(fig=fig, ax=ax, z=True, figsize=figsize)
        else:
            raise Exception("Number of plot values must be 2 or 3, not "
                            + str(len(plot_values)))

        for group, hists in grouphists.items():
            mark_time = group in time_groups
            pass_kwargs = dict(label=group, fig=fig, ax=ax, time=time,
                               mark_time=mark_time, time_ticks=time_ticks,
                               time_fontsize=time_fontsize, t_pretext=t_pretext)
            local_kwargs = {**kwargs, **pass_kwargs, **indiv_kwargs.get(group, {})}
            if len(plot_values) == 2:
                hists.plot_trajectory(*plot_values, **local_kwargs)
            elif len(plot_values) == 3:
                hists.plot_trajectory3(*plot_values, **local_kwargs)
        if legend:
            consolidate_legend(ax, **kwargs)
        add_title_xylabs(ax, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title,
                         xlim=xlim, ylim=ylim, zlim=zlim)
        return fig, ax

    def plot_trajectory(self, xlab, ylab, fig=None, ax=None, figsize=(6, 4),
                        time="time", mark_time=False, time_ticks=1.0, time_fontsize=8,
                        t_pretext="t=", **kwargs):
        """
        Plot a single set of trajectories on an existing matplotlib axis.

        Used by History.plot_trajectories (the main interface).

        Parameters
        ----------
        ax : matplotlib axis
            Axis object to mark on
        fig : figure
            Matplotlib figure object
        xlab : str
            Name to use for the x-values.
        ylab : str
            Name to use for the y-values.
        mark_time : bool, optional
            Whether to mark the time of the trajectory at given ticks.
            The default is False.
        time_ticks : float, optional
            Time tick frequency. The default is 1.0.
        time_fontsize : int, optional
            Size of font for time ticks. The default is 8.
        **kwargs : kwargs
            kwargs to ax.plot
        """
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        xs = [*self.get_values(xlab).values()]
        ys = [*self.get_values(ylab).values()]
        times = [i for k, i in self.get_values(time).items() if "t." not in k]
        for i, x in enumerate(xs):
            ax.plot(x, ys[i], **kwargs)
            if mark_time:
                mark_times(ax, time_ticks, times[i], x, ys[i],
                           fontsize=time_fontsize, pretext=t_pretext)

    def plot_trajectory3(self, xlab, ylab, zlab, fig=None, ax=None, figsize=(6, 4),
                         time="time", mark_time=False, time_ticks=1.0, time_fontsize=8,
                         t_pretext="t=", **kwargs):
        """
        Plot a single set of trajectories on an existing matplotlib axis (3d).

        See History.plot_trajectory
        """
        xs = [*self.get_values(xlab).values()]
        ys = [*self.get_values(ylab).values()]
        zs = [*self.get_values(zlab).values()]
        times = [i for k, i in self.get_values(time).items() if "t." not in k]
        for i, x in enumerate(xs):
            ax.plot(x, ys[i], zs[i], **kwargs)
            if mark_time:
                mark_times(ax, time_ticks, times[i], x, ys[i], zs[i],
                           fontsize=time_fontsize, pretext=t_pretext)

    def plot_trajectories_from(self, t, plot_values=(), **kwargs):
        """
        Plot trajectories using History.plot_trajectories up to a given timestep.

        Parameters
        ----------
        t : int
            time index to plot trajectories from.
        plot_values : tuple, optional
            plot_values args for History.plot_trajectories. The default is ().
        **kwargs : kwargs
            Keyword arguments to History.plot_trajectories.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        kwargs = prep_animation_title(t, **kwargs)
        new_hist = self.cut(end_ind=t, newcopy=True)
        return new_hist.plot_trajectories(*plot_values, **kwargs)

    def animate(self, plot_func, times='all', figsize=(6, 4), z=False, **kwargs):
        """
        Create an animation of a plotting function over the history.

        Parameters
        ----------
        plot_func : method or str
            External update function for plot. If str, name of internal update method.
            (e.g., 'plot_trajectories_from').
        times : list or `all`, optional
            Times to animate over. The default is 'all'.
        figsize : tuple, optional
            Size of the figure. The default is (6, 4).
        z : int/Float/Bool, optional
            Whether to instantiate a z-value. The default is False.
        **kwargs : kwargs
            Keyword arguments.

        Returns
        -------
        ani : animation.Funcanimation
            Object with animation.
        """
        fig, ax = setup_plot(figsize=figsize, z=z)

        if times == 'all':
            max_time = np.min([len(h) for h in self.values()])
            t_inds = [i for i in range(max_time)]
        else:
            t_inds = times

        if isinstance(plot_func, str):
            p_func = getattr(self, plot_func)
            partial_draw = partial(p_func, fig=fig, ax=ax, **kwargs)
        else:
            partial_draw = partial(plot_func, history=self, fig=fig, ax=ax, **kwargs)

        ani = animation.FuncAnimation(fig, partial_draw, frames=t_inds)
        return ani


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
