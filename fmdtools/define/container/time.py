#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Time` class for containing timers and time-related constructs.

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

from fmdtools.analyze.common import get_sub_include
from fmdtools.define.container.base import BaseContainer
from fmdtools.define.object.timer import Timer

from decimal import Decimal
import numpy as np


class Time(BaseContainer):
    """
    Class for defining all time-based aspects of a Block (e.g., time, timestep, timers).

    Attributes
    ----------
    time : float
        real time for the model
    dt : float
        timestep size
    t_ind : int
        index of the given time in the history.
    executed_static: bool
        Whether a sim's static behavior has executed yet this timestep (or not).
    executed_dynamic: bool
        Whether a sim's dynamic behavior has executed yet this timestep (or not).
    executing : bool
        Whether the timestep of the sim is executing or has finished executing.
        Gets set as true when time is updated and false when the timestep is incremented
    timers : dict
        dictionary of instantiated timers
    use_local : bool
        Whether to use the local timetep (vs global timestep)
    timernames: tuple
        Names of timers to instantiate.

    Examples
    --------
    Extending the time class gives one access to a dict of timers:

    >>> class ExtendedTime(Time):
    ...     timernames = ('t1', 't2')

    >>> t = ExtendedTime()
    >>> t.timers['t1']
    Timer t1: mode= standby, time= 0.0

    These timers can then be used:

    >>> t.timers['t1'].inc(1.0)
    >>> t.timers['t1']
    Timer t1: mode= ticking, time= 1.0

    Checking copy:

    >>> t2 = t.copy()
    >>> t2.timers
    {'t1': Timer t1: mode= ticking, time= 1.0, 't2': Timer t2: mode= standby, time= 0.0}

    Check that copied timers are independent:

    >>> t2.timers['t1'].__hash__() == t.timers['t1'].__hash__()
    False
    """

    rolename = "t"
    time: np.float64 = np.float64(-0.1)
    t_ind: np.int32 = np.int32(0)
    timers: dict = {}
    use_local: np.bool_ = True
    dt: np.float64 = np.float64(1.0)
    executed_static: np.bool_ = False
    executed_dynamic: np.bool_ = False
    executing: np.bool_ = False
    local_dt = np.float64(1.0)
    timernames = ()
    default_track = ('timers')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.timers:
            self.timers = {}
        for timername in self.timernames:
            self.timers[timername] = Timer(timername)
        self.set_timestep()

    def create_repr(self, fields=['time', "timers"], **kwargs):
        """Limit model repr to relevant time/timers fields."""
        return super().create_repr(fields=fields, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Time

    def has_executed(self):
        """Return whether the Simulabe has been called."""
        return self.executed_static or self.executed_dynamic

    def __getattr__(self, item):
        if item in self.timers:
            return self.timers[item]
        else:
            return super().__getattribute__(item)

    def return_mutables(self):
        return (*(t.time for t in self.timers.values()),
                self.time,
                self.t_ind,
                self.executed_dynamic,
                self.executed_static,
                self.executing)

    def update_time(self, time):
        """Update the current time from the overall simulation."""
        if time > self.time:
            self.assign(dict(time=time,
                             executed_static=False,
                             executed_dynamic=False,
                             executing=True))

    def set_timestep(self, **kwargs):
        """
        Set the timestep of the function given.

        If using the option use_local, local_timestep is used instead of
        global_timestep.
        """
        self.assign(kwargs)
        global_tstep = Decimal(str(self.dt))
        local_tstep = Decimal(self.local_dt)
        if self.use_local:
            dt = local_tstep
            if ((dt < global_tstep and global_tstep % dt)
                or (dt > global_tstep and dt % global_tstep)):
                raise Exception("Local timestep: " + str(dt) +
                                " doesn't line up with global timestep: " +
                                str(global_tstep))
        else:
            dt = global_tstep
        self.dt = float(dt)
        for timer in self.timers.values():
            timer.tstep = -self.dt

    def reset(self):
        """Reset time to the initial state."""
        self.assign(dict(time=-0.1, t_ind=0,
                         executed_static=False, executed_dynamic=False))
        for timer in self.timers.values():
            timer.reset()

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Add field 'att' to history. Accommodates time and timer tracking."""
        if att == 'timers':
            track_timers = get_sub_include('timers', track)
            for tname, timer in self.timers.items():
                sub_track = get_sub_include(tname, track_timers)
                if tname in sub_track or sub_track == 'default':
                    hist[tname] = timer.create_hist(timerange)
        else:
            BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)

    def get_sim_times(self, time):
        """
        Get the start and end time for the end of the timestep (if not provided).

        If the current time is -0.1, sets start_time at 0.0 for static propagation.
        Otherwise gets the time at the next timestep.

        Returns
        -------
        start_time : float
            Starting time for the timestep
        end_time : float
            Ending time for the timestep.
        """
        if self.time == -0.1:
            start_time = 0.0
        else:
            start_time = self.time
        if time is None:
            end_time = start_time + self.dt
        else:
            end_time = time
        return start_time, end_time


class ExtendedTime(Time):
    """Example extended time class for testing, etc."""

    timernames = ('t1', 't2')

t = ExtendedTime()
t.timers['t1']


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
