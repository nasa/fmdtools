# -*- coding: utf-8 -*-
"""
Description: A module for defining timers for use in Time containers.

Has Classes:
- :class:`Timer`: Class defining timers
"""
from fmdtools.analyze.history import History
from fmdtools.define.object.base import BaseObject


class Timer(BaseObject):
    """
    Class for model timers used in functions (e.g. for conditional faults).

    Attributes
    ----------
    name : str
        timer name
    time : float
        internal timer clock time
    tstep : float
        time to increment at each time-step
    mode : str (standby/ticking/complete)
        the internal state of the timer
    """

    default_track = ('time', 'mode')

    def __init__(self, name):
        """
        Initializes the Tymer

        Parameters
        ----------
        name : str
            Name for the timer
        """
        self.name = str(name)
        self.time = 0.0
        self.tstep = -1.0
        self.mode = 'standby'

    def __repr__(self):
        return 'Timer ' + self.name + ': mode= ' + self.mode + ', time= ' + str(self.time)

    def t(self):
        """ Returns the time elapsed """
        return self.time

    def inc(self, tstep=[]):
        """ Increments the time elapsed by tstep"""
        if self.time >= 0.0:
            if tstep:
                self.time += tstep
            else:
                self.time += self.tstep
            self.mode = 'ticking'
        if self.time <= 0:
            self.time = 0.0
            self.mode = 'complete'

    def reset(self):
        """ Resets the time to zero"""
        self.time = 0.0
        self.mode = 'standby'

    def set_timer(self, time, tstep=-1.0, overwrite='always'):
        """ Sets timer to a given time

        Parameters
        ----------
        time : float
            set time to count down in the timer
        tstep : float (default -1.0)
            time to increment the timer at each time-step
        overwrite : str
            whether/how to overwrite the previous time
            'always' (default) sets the time to the given time
            'if_more' only overwrites the old time if the new time is greater
            'if_less' only overwrites the old time if the new time is less
            'never' doesn't overwrite an existing timer unless it has reached 0.0
            'increment' increments the previous time by the new time
        """
        if overwrite == 'always':
            self.time = time
        elif overwrite == 'if_more' and self.time < time:
            self.time = time
        elif overwrite == 'if_less' and self.time > time:
            self.time = time
        elif overwrite == 'never' and self.time == 0.0:
            self.time = time
        elif overwrite == 'increment':
            self.time += time
        self.tstep = tstep
        self.mode = 'set'

    def in_standby(self):
        """Whether the timer is in standby (time has not been set)"""
        return self.mode == 'standby'

    def is_ticking(self):
        """Whether the timer is ticking (time is incrementing)"""
        return self.mode == 'ticking'

    def is_complete(self):
        """Whether the timer is complete (after time is done incrementing)"""
        return self.mode == 'complete'

    def is_set(self):
        """Whether the timer is set (before time increments)"""
        return self.mode == 'set'

    def copy(self):
        cop = self.__class__(self.name)
        cop.time = self.time
        cop.mode = self.mode
        cop.dt = self.dt
        return cop

    def create_hist(self, timerange, track):
        h = History()
        track = self.get_track(track, all_possible=('time', 'mode'))
        h.init_att('time', self.time, timerange=timerange, track=track, dtype=float)
        h.init_att('mode', self.mode, timerange=timerange, track=track, str_size='<U8')
        return h
