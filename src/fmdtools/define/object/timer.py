#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Timer` class for representing timers.

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

    Examples
    --------
    >>> t = Timer("test_timer")
    >>> t
    Timer test_timer: mode= standby, time= 0.0
    >>> t.set_timer(2)
    >>> t
    Timer test_timer: mode= set, time= 2
    >>> t.inc()
    >>> t
    Timer test_timer: mode= ticking, time= 1.0
    >>> t.inc()
    >>> t
    Timer test_timer: mode= complete, time= 0.0
    >>> t.reset()
    >>> t
    Timer test_timer: mode= standby, time= 0.0
    """

    default_track = ('time', 'mode')
    roletypes = []
    rolevars = ['time', 'mode']
    __slots__ = ('time', 'tstep', 'mode')

    def __init__(self, name=''):
        """
        Initialize the Timer.

        Parameters
        ----------
        name : str
            Name for the timer
        """
        BaseObject.__init__(self, name=name)
        self.time = 0.0
        self.tstep = -1.0
        self.mode = 'standby'

    def __repr__(self):
        return ('Timer ' + self.name + ': mode= '
                + self.mode + ', time= ' + str(self.time))

    def t(self):
        """Return the time elapsed."""
        return self.time

    def inc(self, tstep=[]):
        """Increment the time elapsed by tstep."""
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
        """Reset the time to zero."""
        self.time = 0.0
        self.mode = 'standby'

    def set_timer(self, time, tstep=None, overwrite='always'):
        """
        Set timer to a given time.

        Parameters
        ----------
        time : float
            set time to count down in the timer
        tstep : float (default -1.0)
            time to increment the timer at each time-step
        overwrite : str
            whether/how to overwrite the previous time.
            'always' (default) sets the time to the given time.
            'if_more' only overwrites the old time if the new time is greater.
            'if_less' only overwrites the old time if the new time is less.
            'never' doesn't overwrite an existing timer unless it has reached 0.0.
            'increment' increments the previous time by the new time.
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
        if tstep is not None:
            self.tstep = tstep
        self.mode = 'set'

    def indicate_standby(self):
        """Indicate if the timer is in standby (time not set)."""
        return self.mode == 'standby'

    def indicate_ticking(self):
        """Indictate if the timer is ticking (time is incrementing)."""
        return self.mode == 'ticking'

    def indicate_complete(self):
        """Indicate if the timer is complete (after time is done incrementing)."""
        return self.mode == 'complete'

    def indicate_set(self):
        """Indicate if the timer is set (before time increments)."""
        return self.mode == 'set'

    def copy(self):
        """Copy the Timer."""
        cop = self.__class__(self.name)
        cop.time = self.time
        cop.mode = self.mode
        cop.dt = self.dt
        return cop


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
