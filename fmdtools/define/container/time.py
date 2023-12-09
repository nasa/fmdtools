# -*- coding: utf-8 -*-
"""
Description: A module for defining time-based properties for use in blocks.

Has Classes:
- :class:`Timer`: Class defining timers
- :class:`Time`: Class containing all time-related Block constructs (e.g., timers).
"""
from decimal import Decimal
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History
from fmdtools.define.container.base import BaseContainer
from fmdtools.define.object.timer import Timer

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
        index of the given time
    t_loc : float
        local time (e.g., for actions with durations)
    run_times : int
        number of times to run the behavior if running at a different timestep than
        global
    timers : dict
        dictionary of instantiated timers
    use_local : bool
        Whether to use the local timetep (vs global timestep)
    timernames: tuple
        Names of timers to instantiate.
    """
    rolename = "t"
    time: float = -0.1
    t_ind: int = 0
    t_loc: float = 0.0
    timers: dict = {}
    use_local: bool = True
    dt: float = 1.0
    run_times: int = 1
    local_dt = 1.0
    timernames = ()
    default_track = ('timers')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.timers:
            self.timers = {}
        for timername in self.timernames:
            self.timers[timername] = Timer(timername)
        self.set_timestep()

    def __getattr__(self, item):
        if item in self.timers:
            return self.timers[item]
        else:
            return super().__getattribute__(item)

    def return_mutables(self):
        return (*(t.time for t in self.timers.values()),
                self.time,
                self.t_ind,
                self.t_loc,
                self.run_times)

    def set_timestep(self):
        """Sets the timestep of the function given the option use_local
        (which selects whether it uses local_timestep or global_timestep)"""
        global_tstep = Decimal(str(self.dt))
        local_tstep = Decimal(self.local_dt)
        if self.use_local:
            dt = local_tstep
            if dt < global_tstep:
                if global_tstep % dt:
                    raise Exception("Local timestep: " + str(dt) +
                                    " doesn't line up with global timestep: " +
                                    str(global_tstep))
            else:
                if dt % global_tstep:
                    raise Exception(
                        "Local timestep: " + str(dt) +
                        " doesn't line up with global timestep: " + str(global_tstep))
            self.run_times = int(global_tstep/dt)
        else:
            dt = global_tstep
            self.run_times = 1
        self.dt = float(dt)
        for timer in self.timers.values():
            timer.dt = -self.dt

    def reset(self):
        """Resets time to the initial state"""
        self.time = -0.1
        self.t_ind = 0
        self.t_loc = 0.0
        for timer in self.timers.values():
            timer.reset()

    def copy(self, *args, **t_args):
        """ Copies the timer"""
        cop = self.__class__(*args, **t_args)
        for timer in self.timers:
            cop.timers[timer] = self.timers[timer].copy()
        cop.run_times = self.run_times
        cop.time = self.time
        cop.t_ind = self.t_ind
        cop.t_loc = self.t_loc
        cop.dt = self.dt
        return cop

    def create_hist(self, timerange, track):
        """
        Creates a History corresponding to Time

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'.
            The default is None.

        Returns
        -------
        hist : History
            History of time/timer attribues specified in track.
        """
        hist = History()
        track = self.get_track(track)
        hist.init_att('time', self.time, timerange=timerange, track=track, dtype=float)
        if 'timers' in track:
            track_timers = get_sub_include('timers', track)
            for tname, timer in self.timers.items():
                sub_track = get_sub_include(tname, track_timers)
                hist[tname] = timer.create_hist(timerange, sub_track)
        return hist
