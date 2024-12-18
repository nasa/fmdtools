#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for analyzing phases and time-based sampling schemes.

Has classes:

- :class:`PhaseMap`: A mapping of phases to times.

And functions:

- :func:`from_hist`: Creates dict of PhaseMaps based on mode progression in history.
- :func:`phaseplot`: Plots the progression of phases over time.
- :func:`samplemetric`: plots a metric for a single fault sampled by a SampleApproach
  over time with rates/
- :func:`samplemetrics`: plots a metric for a set of faults sampled by a SampleApproach
  over time with rates on separate plots
- :func:`find_interval_overlap`: Find overlap between given intervals.

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

from fmdtools.analyze.common import setup_plot
from fmdtools.define.base import gen_timerange

import numpy as np
from ordered_set import OrderedSet
import itertools

from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42


class PhaseMap(object):
    """
    Mapping of phases to times used to create scenario samples.

    Phases and modephases may be generated from Result.get_phases.

    Parameters
    ----------
    phases : dict
        Phases the mode will be injected during. Used to determine opportunity
        factor defined by the dict in fault.phases.
        Has structure {'phase1': [starttime, endtime]}. The default is {}.
        May also provide tuple with structure (('phase1', starttime, endtime))
    modephases : dict, optional
        Modes that the phases occur in. Used to determine opportunity vector defined
        by the dict in fault.phases (if .phases maps to modes of occurence an not
        phases).
        Has structure::
        {'on': {'on1', 'on2', 'on3'}}

        The default is {}.
    dt: float
        Timestep defining phases.
    """

    def __init__(self, phases, modephases={}, dt=1.0):
        if type(phases) == tuple:
            phases = {ph[0]: [ph[1], ph[2]] for ph in phases}
        self.phases = phases
        self.modephases = modephases
        self.dt = dt

    def __repr__(self):
        return 'PhaseMap(' + str(self.phases) + ', ' + str(self.modephases) + ')'

    def find_phase(self, time, dt=1.0):
        """
        Find the phase that a time occurs in.

        Parameters
        ----------
        time : float
            Occurence time.

        Returns
        -------
        phase : str
            Name of the phase time occurs in.
        """
        for phase, times in self.phases.items():
            if times[0] <= time < times[1]+dt:
                return phase
        raise Exception("time "+str(time)+" not in phases: "+str(self.phases))

    def find_modephase(self, phase):
        """
        Find the mode in modephases that a given phase occurs in.

        Parameters
        ----------
        phase : str
            Name of the phase (e.g., 'on1').

        Returns
        -------
        mode : str
            Name of the corresponding mode (e.g., 'on').

        Examples
        --------
        >>> pm = PhaseMap({}, {"on": {"on0", "on1", "on2"}})
        >>> pm.find_modephase("on1")
        'on'
        """
        for mode, mode_phases in self.modephases.items():
            if phase in mode_phases:
                return mode
        raise Exception("Phase "+phase+" not in modephases: "+str(self.modephases))

    def find_base_phase(self, time):
        """
        Find the phase or modephase (if provided) that the time occurs in.

        Parameters
        ----------
        time : float
            Time to check.

        Returns
        -------
        phase : str
            Phase or modephase the time occurs in.
        """
        phase = self.find_phase(time)
        if self.modephases:
            phase = self.find_modephase(phase)
        return phase

    def calc_samples_in_phases(self, *times):
        """
        Calculate the number of times the provided times show up in phases/modephases.

        Parameters
        ----------
        *times : float
            Times to check

        Returns
        -------
        phase_times : dict
            the number of time-steps in each phase

        Examples
        --------
        >>> pm = PhaseMap(phases={'on':[0, 3], 'off': [4, 5]})
        >>> pm.calc_samples_in_phases(1,2,3,4,5)
        {'on': 3, 'off': 2}
        >>> pm = PhaseMap({'on':[0, 3], 'off': [4, 5]}, {'oper': {'on', 'off'}})
        >>> pm.calc_samples_in_phases(1,2,3,4,5)
        {'oper': 5}
        """
        if self.modephases:
            phase_times = {ph: 0 for ph in self.modephases}
        else:
            phase_times = {ph: 0 for ph in self.phases}
        for time in times:
            phase = self.find_phase(time)
            if self.modephases:
                phase = self.find_modephase(phase)
            phase_times[phase] += 1
        return phase_times

    def calc_phase_time(self, phase):
        """
        Calculate the length of a phase.

        Parameters
        ----------
        phase : str
            phase to calculate.
        phases : dict
            dict of phases and time intervals.
        dt : float, optional
            Timestep length. The default is 1.0.

        Returns
        -------
        phase_time : float
            Time of the phase


        Examples
        --------
        >>> pm = PhaseMap({"on": [0, 4], "off": [5, 10]})
        >>> pm.calc_phase_time("on")
        5.0
        """
        phasetimes = self.phases[phase]
        phase_time = phasetimes[1] - phasetimes[0] + self.dt
        return phase_time

    def calc_modephase_time(self, modephase):
        """
        Calculate the amount of time in a mode, given that mode maps to multiple phases.

        Parameters
        ----------
        modephases : dict
            Dict mapping modes to phases

        Returns
        -------
        modephase_time : float
            Amount of time in the modephase

        Examples
        --------
        >>> pm = PhaseMap({"on1": [0, 1], "on2": [2, 3]}, {"on": {"on1", "on2"}})
        >>> pm.calc_modephase_time("on")
        4.0
        """
        modephase_time = sum([self.calc_phase_time(mode_phase)
                              for mode_phase in self.modephases[modephase]])
        return modephase_time

    def calc_scen_exposure_time(self, time):
        """
        Calculate the time for the phase/modephase at the given time.

        Parameters
        ----------
        time : float
            Time within the phase.

        Returns
        -------
        exposure_time : float
            Exposure time of the given phasemap.
        """
        phase = self.find_phase(time)
        if self.modephases:
            phase = self.find_modephase(phase)
            return self.calc_modephase_time(phase)
        else:
            return self.calc_phase_time(phase)

    def get_phase_times(self, phase):
        """
        Get the set of discrete times in the interval for a phase.

        Parameters
        ----------
        phase : str
            Name of a phase in phases or modephases.

        Returns
        -------
        all_times : list
            List of times corresponding to the phase

        Examples
        --------
        >>> pm = PhaseMap({"on1": [0, 1], "on2": [2, 3]}, {"on": {"on1", "on2"}})
        >>> pm.get_phase_times('on1')
        [0.0, 1.0]
        >>> pm.get_phase_times('on2')
        [2.0, 3.0]
        >>> pm.get_phase_times('on')
        [0.0, 1.0, 2.0, 3.0]
        """
        if phase in self.modephases:
            phases = self.modephases[phase]
            intervals = [self.phases[ph] for ph in phases]
        elif phase in self.phases:
            intervals = [self.phases[phase]]
        int_times = [gen_timerange(i[0], i[-1], self.dt) for i in intervals]
        all_times = list(set(np.concatenate(int_times)))
        all_times.sort()
        return all_times

    def get_sample_times(self, *phases_to_sample):
        """
        Get the times to sample for the given phases.

        Parameters
        ----------
        *phases_to_sample : str
            Phases to sample. If none are provided, the full set of phases or modephases
            is used.

        Returns
        -------
        sampletimes : dict
            dict of times to sample with structure::
            {'phase1': [t0, t1, t2], ...}

        Examples
        --------
        >>> pm = PhaseMap({"on1": [0, 4], "on2": [5, 6]}, {"on": {"on1", "on2"}})
        >>> pm.get_sample_times("on1")
        {'on1': [0.0, 1.0, 2.0, 3.0, 4.0]}
        >>> pm.get_sample_times("on")
        {'on': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        >>> pm.get_sample_times("on1", "on2")
        {'on1': [0.0, 1.0, 2.0, 3.0, 4.0], 'on2': [5.0, 6.0]}
        """
        if not phases_to_sample:
            if self.modephases:
                phases_to_sample = tuple(self.modephases)
            elif self.phases:
                phases_to_sample = tuple(self.phases)
        sampletimes = {}
        for phase in phases_to_sample:
            sampletimes[phase] = self.get_phase_times(phase)
        return sampletimes

    def plot(self, dt=1.0, phase_ticks='both', fig=None, ax=None):
        """Plot phasemap on existing axis."""
        fig, ax = setup_plot(fig=fig, ax=ax)
        modephases = self.modephases
        phases = self.phases

        if modephases:
            mode_nums = {ph: i for i, (k, v) in enumerate(modephases.items())
                         for ph in v}
            ylabels = list(modephases.keys())
        else:
            mode_nums = {ph: i for i, ph in enumerate(phases)}
            ylabels = list(mode_nums.keys())

        phaseboxes = [((v[0]-.5*dt, mode_nums[k]-.4),
                       (v[0]-.5*dt, mode_nums[k]+.4),
                       (v[1]+.5*dt, mode_nums[k]+.4),
                       (v[1]+.5*dt, mode_nums[k]-.4)) for k, v in phases.items()]
        color_options = list(mcolors.TABLEAU_COLORS.keys())[0:len(ylabels)]
        colors = [color_options[mode_nums[phase]] for phase in phases]
        bars = PolyCollection(phaseboxes, facecolors=colors)

        ax.add_collection(bars)
        ax.autoscale()

        ax.set_yticks(list(set(mode_nums.values())))
        ax.set_yticklabels(ylabels)

        times = [0]+[v[1] for k, v in phases.items()]
        if phase_ticks == 'both':
            xmin = list(ax.get_xticks())[0]
            xmax = list(ax.get_xticks())[-1]
            minor_ticks = [i for i in np.arange(xmin, xmax, dt)]
            ax.set_xticks(minor_ticks, minor=True)
            for t in times:
                ax.axvline(x=t, color='k')
        elif phase_ticks == 'phases':
            ax.set_xticks(times)
        ax.set_xlim(times[0], times[-1])
        plt.grid(which='major', axis='x')


def from_hist(hist, fxn_modephases='all'):
    """
    Identify the phases of operation for the system based on its modes.

    These phases and modephases are then be used to define a dict of PhaseMaps.

    Parameters
    ----------
    hist : History
        History of states with mode information in them
    fxn_modephases : list
        Functions to associate modephase information from
        (rather than just phase information)

    Returns
    -------
    phasemaps : dict
        Dictionary of distict phases that the system functions pass through,
        of the form: {'fxn': PhaseMap} where each phase is defined by its
        corresponding mode in the modelhist.
        Phases are numbered mode, mode1, mode2 for multiple modes and given a
        corresponding phasemap {mode: {mode, mode1, mode2}} mapping modes to
        phases for future sampling.

    Examples
    --------
    >>> from fmdtools.analyze.history import History
    >>> g1 = History({'a.m.mode': ['off', 'off', 'on'], 'time':[0,1,2]})
    >>> from_hist(g1)
    {'a': PhaseMap({'off': [0, 1], 'on': [2, 2]}, {'off': {'off'}, 'on': {'on'}})}
    >>> g2 = History({'b.m.mode': ['off', 'off', 'on'], 'c.m.mode':['up', 'down', 'left'], 'time':[0,1,2]})
    >>> from_hist(g2)
    {'b': PhaseMap({'off': [0, 1], 'on': [2, 2]}, {'off': {'off'}, 'on': {'on'}}), 'c': PhaseMap({'up': [0, 0], 'down': [1, 1], 'left': [2, 2]}, {'up': {'up'}, 'down': {'down'}, 'left': {'left'}})}
    """
    modephasemaps = {}
    times = hist['time']
    modehists = hist.get_values('m.mode')
    for k, modehist in modehists.items():
        if isinstance(k, str):
            k = k.split(".")
        fxn = k[k.index('m')-1]
        if len(modehist) != 0:
            modes = OrderedSet(modehist)
            modephases = dict.fromkeys(modes)
            phases_unsorted = dict()
            for mode in modes:
                modeinds = [ind for ind, m in enumerate(modehist) if m == mode]
                startind = modeinds[0]
                phasenum = 0
                phaseid = mode
                modephases[mode] = set()
                for i, ind in enumerate(modeinds):
                    if ind+1 not in modeinds:
                        phases_unsorted[phaseid] = [times[startind], times[ind]]
                        modephases[mode].add(phaseid)
                        if i != len(modeinds)-1:
                            startind = modeinds[i+1]
                            phasenum += 1
                            phaseid = mode+str(phasenum)
            phases = dict(sorted(phases_unsorted.items(),
                                 key=lambda item: item[1][0]))
            if fxn_modephases == 'all' or fxn in fxn_modephases:
                mph = modephases
            else:
                mph = {}
            modephasemaps[fxn] = PhaseMap(phases=phases, modephases=mph)
    return modephasemaps


def phaseplot(phasemaps, modephases=[], mdl=[], dt=1.0, singleplot=True,
              phase_ticks='both', figsize="default", v_padding=0.5, title_padding=-0.05,
              title="Progression of model through operational phases"):
    """
    Plot the phases of operation that the model progresses through.

    Parameters
    ----------
    phasemaps : dict or PhaseMap
        Dict of phasemaps that the functions of the model progresses through
        (e.g. from phases.from_hist).
    modephases : dict, optional
        dictionary that maps the phases to operational modes, if it is desired to track
        the progression through modes
    mdl : Model, optional
        model, if it is desired to additionally plot the phases of the model with the
        function phases
    singleplot : bool, optional
        Whether the functions' progressions through phases are plotted on the same plot
        or on different plots.
        The default is True.
    phase_ticks : 'std'/'phases'/'both', optional
        x-ticks to use (standard, at the edge of phases, or both). Default is 'both'
    figsize : tuple (float,float), optional
        x-y size for the figure. The default is 'default', which dymanically gives 2 for
        each row
    v_padding : float, optional
        vertical padding between subplots as a fraction of axis height
    title_padding : float
        padding for title as a fraction of figure height
    title : str, optional
        figure title. Default is "Progression of model through operational phases"

    Returns
    -------
    fig/figs : Figure or list of Figures
        Matplotlib figures to edit/use.
    """
    if mdl:
        phasemaps["Model"] = PhaseMap(mdl.phases)
        dt = mdl.tstep

    if isinstance(phasemaps, PhaseMap):
        phasemaps = {'': phasemaps}
    elif not isinstance(phasemaps, dict):
        raise Exception("Phasemaps not a dict or PhaseMap")

    if singleplot:
        num_plots = len(phasemaps)
        if figsize == 'default':
            figsize = (4, 2*num_plots)
        fig = plt.figure(figsize=figsize)
    else:
        if figsize == 'default':
            figsize = (4, 4)
        figs = []

    for i, (fxn, phasemap) in enumerate(phasemaps.items()):
        if singleplot:
            ax = plt.subplot(num_plots, 1, i+1, label=fxn)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        phasemap.plot(dt, phase_ticks, fig=fig, ax=ax)

        if singleplot:
            plt.title(fxn)
        else:
            plt.title(title)
            figs.append(fig)
    if singleplot:
        plt.suptitle(title, y=1.0+title_padding)
        plt.subplots_adjust(hspace=v_padding)
        return fig
    else:
        return figs


def samplemetric(faultsamp, endclasses, metric='cost', rad='rate', rad_scale=0.01,
                 label_rad="{:.2e}", line='stem', title="", ylims=None, **scen_kwargs):
    """
    Plot the sample metric and rate of a given fault over sample times.

    (note: not currently compatible with joint fault modes)

    Parameters
    ----------
    faultsamp : FaultSamp
        Fault sample defining the underlying samples to take with phasemap
    endclasses : Result
        A Result with the end classification of each fault (metrics, etc)
    metric : str, optional
        Metric to plot. The default is 'cost'
    rad : str, optional
        Metric to plot as a radius at each sample. Default is 'rate'.
    rad_scale : float, optional
        Scale factor for radius. Default is 0.01, which makes the max rad size 1/100 of
        the max metric value.
    label_rad : str, optional
        Format string for the radius (if any). Default is "{:.2e}".
    line : str ('stem' or 'line'), optional
        Whether to plot metrics as a stem or line plot
    title : str, optional
        Title for the plot
    ylims : tuple, optional
        y-limits for plot
    **scen_kwargs : kwargs
        Arguments to FaultSample.get_scens (e.g., modes etc to sample).

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot
    """
    scens = faultsamp.get_scens(**scen_kwargs)

    fig, axes = plt.subplots(2, 1, sharey=False, gridspec_kw={
                               'height_ratios': [3, 1]})
    # phase plot
    ax = faultsamp.phasemap.plot(ax=axes[1], fig=fig)

    # cost/metric plots
    costs = np.array([endclasses.get(scen).endclass[metric] for scen in scens])
    times = np.array([v.time for v in scens.values()])
    timesort = np.argsort(times)
    times = times[timesort]
    costs = costs[timesort]

    if line == 'line':
        axes[0].plot(times, costs, label=metric)
    elif line == 'stem':
        axes[0].stem(times, costs, label=metric, markerfmt=",")

    # rate/metric plot
    if rad:
        sizes = np.array([endclasses.get(scen).endclass[rad] for scen in scens])
        sizes = sizes[timesort]
        rad_scale *= np.max(abs(costs))/np.max(abs(sizes))
        axes[0].scatter(times, costs, s=rad_scale*sizes, label=rad, alpha=0.5)
        if label_rad:
            for i, t in enumerate(times):
                axes[0].text(times[i], costs[i], s=label_rad.format(sizes[i]))

    ts = (faultsamp.faultdomain.mdl.sp.start_time,
          faultsamp.faultdomain.mdl.sp.end_time)
    axes[0].set_xlim(ts[0], ts[-1])
    if ylims:
        axes[0].set_ylim(*ylims)

    axes[0].set_ylabel(metric)
    axes[0].grid()
    if title:
        axes[0].set_title(title)
    # plt.subplot_adjust()
    plt.tight_layout()
    return fig


def samplemetrics(app, endclasses, **kwargs):
    """
    Plot costs and rates of a set of faults injected over time according to the sample.

    Parameters
    ----------
    app : sampleapproach
        The sample approach used to run the list of faults
    endclasses : Result
        Results over the scenarios defined in app.
    **kwargs : kwargs
        kwargs to samplemetric

    Returns
    -------
    figs : dict
        dict of figures for each fault sample in the SampleApproach
    """
    figs = {}
    for faultsampname, faultsamp in app.faultsamples.items():
        figs[faultsampname] = samplemetric(faultsamp, endclasses,
                                           title=faultsampname, **kwargs)
    return figs


def find_interval_overlap(*intervals, dt=1.0):
    """
    Find the overlap between given intervals.

    Used to sample joint fault modes with different (potentially overlapping) phases.

    Examples
    --------
    >>> find_interval_overlap([0, 10], [4, 12])
    [4.0, 10.0]
    >>> find_interval_overlap([0, 3], [4, 12])
    []
    """
    try:
        joined_times = {}
        for i, interval in enumerate(intervals):
            possible_times = set()
            possible_times.update(*[{*gen_timerange(interval[0], interval[-1], dt)}
                                    for i in interval])
            if i == 0:
                joined_times = possible_times
            else:
                joined_times = joined_times.intersection(possible_times)
        if not joined_times:
            return []
        else:
            joined_times = [*np.sort([*joined_times])]
            return [joined_times[0], joined_times[-1]]
    except IndexError as e:
        if all(intervals[0] == i for i in intervals):
            return intervals[0]
        else:
            raise Exception("Invalid intervals: " + str(intervals)) from e


def join_phasemaps(*phasemaps):
    """
    Join multiple PhaseMaps into a single PhaseMap.

    Note that modephases are removed in this process.

    Parameters
    ----------
    *phasemaps : PhaseMap
        PhaseMaps with phases to join.

    Returns
    -------
    joint_phasemap : PhaseMap
        Phasemap keyed by tuples for joint phases

    Examples
    --------
    >>> a = PhaseMap({"a": [1, 3], "b": [4, 10]})
    >>> b = PhaseMap({"c": [2, 6], "d": [7, 9]})
    >>> join_phasemaps(a, b)
    PhaseMap({('a', 'c'): [2.0, 3.0], ('b', 'c'): [4.0, 6.0], ('b', 'd'): [7.0, 9.0]}, {})
    """
    joint_phases = {}
    all_combos = [*itertools.product(*[phasemap.phases for phasemap in phasemaps])]
    for combo in all_combos:
        phases = {c: phasemaps[i].phases[c] for i, c in enumerate(combo)}
        intervals = [i for i in phases.values()]
        joined_interval = find_interval_overlap(*intervals)
        if joined_interval:
            joint_phases[combo] = joined_interval
    return PhaseMap(joint_phases)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
